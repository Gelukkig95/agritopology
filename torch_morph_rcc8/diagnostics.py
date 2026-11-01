r"""
core/diagnostics.py
-------------------
Pure PyTorch GPU/CPU Morphological & Fuzzy RCC-8 Spatial Logic Operators.
Academic References:
  1. Randell, Cui, & Cohn (1992), KR '92, pp. 165-176 (RCC-8 JEPD axioms).
  2. Bloch (2005), Image and Vision Computing 23(2), 89-110 (Eq. 1-4, 15-16, 22, 35-36).
  3. Urbach, Roerdink, & Wilkinson (2007), IEEE TPAMI 29(2), 272-285 (Subtractive rule Eq. 8; Hu moment Eq. 42-48).
  4. Teeninga & Wilkinson (2026), DGMM LNCS 16296, 277-290 (Listing 1 GPU leaffix/rootfix & Sec. 4.4 Pebay moments).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RCC8_NAMES = ("DC", "EC", "PO", "TPP", "NTPP", "EQ", "TPPi", "NTPPi")


def morph_dilation(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Bloch (2005) Eq. (1) & (3): Morphological Dilation D_B(X) preserving float32/float64 precision."""
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)
    pad = kernel_size // 2
    padded = F.pad(t, (pad, pad, pad, pad), mode="constant", value=0.0)
    out = F.max_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)
    return out.reshape(orig_shape)


def morph_erosion(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    """Bloch (2005) Eq. (2) & (4): Dual Morphological Erosion E_B(X) preserving float32/float64 precision."""
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)
    pad = kernel_size // 2
    padded = F.pad(t, (pad, pad, pad, pad), mode="constant", value=border_value)
    out = -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1, padding=0)
    return out.reshape(orig_shape)


def kuratowski_closure(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Kuratowski closure (dilation) with unified 3x3 structuring element across Tensor and NumPy inputs."""
    if isinstance(mask, torch.Tensor):
        return morph_dilation(mask, kernel_size=3) > 0.5
    t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    return (morph_dilation(t, kernel_size=3) > 0.5).numpy()


def kuratowski_interior(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Kuratowski interior (erosion) with unified boundary-consistent zero padding."""
    if isinstance(mask, torch.Tensor):
        return morph_erosion(mask, kernel_size=3, border_value=0.0) > 0.5
    t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    return (morph_erosion(t, kernel_size=3, border_value=0.0) > 0.5).numpy()


def kuratowski_opening(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Idempotent 2x2 topological opening supporting PyTorch Tensors and NumPy arrays."""
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        t = mask if mask.is_floating_point() else mask.float()
    else:
        t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    orig_shape = t.shape
    while t.ndim < 4:
        t = t.unsqueeze(0)
    weight = torch.ones((1, 1, 2, 2), device=t.device, dtype=t.dtype)
    eroded = (F.conv2d(t, weight, padding=0) >= 3.5).to(t.dtype)
    opened = (F.conv_transpose2d(eroded, weight, stride=1) > 0.5).reshape(orig_shape)
    return opened if is_tensor else opened.cpu().numpy()


def classify_rcc8_grid_jepd(A: np.ndarray | torch.Tensor, B: np.ndarray | torch.Tensor) -> str:
    """Grid-topology JEPD RCC-8 classifier (Randell et al., 1992; Bloch, 2005, Eq. 35)."""
    t_a = A.bool() if isinstance(A, torch.Tensor) else torch.from_numpy(np.asarray(A, dtype=bool))
    t_b = B.bool() if isinstance(B, torch.Tensor) else torch.from_numpy(np.asarray(B, dtype=bool))
    if torch.equal(t_a, t_b):
        return "EQ"
    inter = t_a & t_b
    if not inter.any():
        touch = (kuratowski_closure(t_a) & t_b).any() or (t_a & kuratowski_closure(t_b)).any()
        return "EC" if bool(touch.item()) else "DC"
    if torch.equal(inter, t_a):
        return "NTPP" if torch.equal(t_a & kuratowski_interior(t_b), t_a) else "TPP"
    if torch.equal(inter, t_b):
        return "NTPPi" if torch.equal(t_b & kuratowski_interior(t_a), t_b) else "TPPi"
    return "PO"


@torch.no_grad()
def compute_rcc8_matrix(masks: torch.Tensor) -> torch.Tensor:
    """
    Vectorized GPU/CPU Pairwise RCC-8 JEPD Matrix for (N, H, W) binary masks.
    Codes: 0=DC, 1=EC, 2=PO, 3=TPP, 4=NTPP, 5=EQ, 6=TPPi, 7=NTPPi.
    """
    if masks.ndim != 3 or masks.shape[0] == 0:
        return torch.zeros((0, 0), dtype=torch.long, device=masks.device)
    N = masks.shape[0]
    m_bin = (masks > 0.5).float()
    cl_m = morph_dilation(m_bin, 3).reshape(N, -1)
    int_m = morph_erosion(m_bin, 3, border_value=0.0).reshape(N, -1)
    flat_m = m_bin.reshape(N, -1)

    area = flat_m.sum(dim=1, keepdim=True)
    inter = flat_m @ flat_m.T
    dil_inter = (cl_m @ flat_m.T) > 0
    sym_touch = dil_inter | dil_inter.T

    has_ov = inter > 0
    a_in_b = (inter == area) & (area > 0)
    b_in_a = (inter == area.T) & (area.T > 0)
    a_in_int_b = ((flat_m @ int_m.T) == area) & (area > 0)
    b_in_int_a = ((int_m @ flat_m.T) == area.T) & (area.T > 0)

    rcc = torch.zeros((N, N), dtype=torch.long, device=masks.device)
    rcc[~has_ov & sym_touch] = 1
    rcc[has_ov & ~a_in_b & ~b_in_a] = 2
    rcc[a_in_b & ~b_in_a & ~a_in_int_b] = 3
    rcc[a_in_b & ~b_in_a & a_in_int_b] = 4
    rcc[a_in_b & b_in_a] = 5
    rcc[b_in_a & ~a_in_b & ~b_in_int_a] = 6
    rcc[b_in_a & ~a_in_b & b_in_int_a] = 7
    return rcc


class DifferentiableFuzzyRCC8Loss(nn.Module):
    """
    100% Differentiable Fuzzy Spatial Relationship Operator (Bloch, 2005, Eq. 15, 16, 22, 36).
    Uses product t-norm t(a, b) = a * b and morphological max-pooling with exact analytical gradients.
    """
    def __init__(self, kernel_size: int = 3, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, mu: torch.Tensor, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        if mu.ndim == 3:
            mu = mu.unsqueeze(1)
        if nu.ndim == 3:
            nu = nu.unsqueeze(1)

        vol_mu = mu.sum(dim=(-2, -1))
        vol_nu = nu.sum(dim=(-2, -1))
        vol_inter = (mu * nu).sum(dim=(-2, -1))

        mu_int = vol_inter / (torch.minimum(vol_mu, vol_nu) + self.eps)
        mu_not_int = 1.0 - mu_int

        dil_mu = morph_dilation(mu, self.kernel_size)
        dil_nu = morph_dilation(nu, self.kernel_size)
        int_dmu_nu = (dil_mu * nu).sum(dim=(-2, -1)) / (torch.minimum(dil_mu.sum(dim=(-2, -1)), vol_nu) + self.eps)
        int_dnu_mu = (dil_nu * mu).sum(dim=(-2, -1)) / (torch.minimum(dil_nu.sum(dim=(-2, -1)), vol_mu) + self.eps)
        fuzzy_ec = (mu_not_int * int_dmu_nu * int_dnu_mu).mean()

        ero_nu = morph_erosion(nu, self.kernel_size, border_value=1.0)
        fuzzy_ntpp = ((mu * ero_nu).sum(dim=(-2, -1)) / (vol_mu + self.eps)).mean()

        return {
            "fuzzy_DC": mu_not_int.mean(),
            "fuzzy_EC": fuzzy_ec,
            "fuzzy_PO": mu_int.mean(),
            "fuzzy_NTPP": fuzzy_ntpp,
        }


def gpu_pebay_moment_merge(
    area: torch.Tensor,
    mean: torch.Tensor,
    m2: torch.Tensor,
    idx_targets: torch.Tensor,
    idx_sources: torch.Tensor,
) -> None:
    """Teeninga & Wilkinson (2026) Sec. 4.4 (p. 285): Numerically stable parallel Pebay moment reduction."""
    a0 = area[idx_targets]
    a1 = area[idx_sources]
    mu0 = mean[idx_targets]
    mu1 = mean[idx_sources]
    m2_0 = m2[idx_targets]
    m2_1 = m2[idx_sources]

    a_new = a0 + a1
    delta = mu1 - mu0
    safe_a = torch.clamp(a_new, min=1e-12)

    area[idx_targets] = a_new
    mean[idx_targets] = mu0 + (a1 * delta) / safe_a
    m2[idx_targets] = m2_0 + m2_1 + (a0 * a1 * (delta * delta)) / safe_a


def gpu_tree_leaffix_accumulate(edge_array_list: list[torch.Tensor], attributes: torch.Tensor) -> torch.Tensor:
    """Teeninga & Wilkinson (2026) Sec. 4.3 Listing 1 (p. 285): Parallel GPU leaffix attribute accumulation."""
    for edges in edge_array_list:
        idx_targets = edges[:, 0]
        idx_sources = edges[:, 1]
        attributes[idx_targets] += attributes[idx_sources]
    return attributes


def gpu_tree_rootfix_reconstruct(edge_array_list: list[torch.Tensor], deltas: torch.Tensor) -> torch.Tensor:
    """Teeninga & Wilkinson (2026) Sec. 4.3 Listing 1 & Urbach et al. (2007) Eq. (8): Parallel GPU rootfix reconstruction."""
    for edges in edge_array_list:
        idx_targets = edges[:, 1]
        idx_sources = edges[:, 0]
        deltas[idx_targets] += deltas[idx_sources]
    return deltas


class ConformalRiskCalibrator:
    """Finite-sample Conformal Risk Control calibrator."""
    def __init__(self, target_alpha: float = 0.02):
        self.alpha = target_alpha
        self.lambda_hat = 0.50

    def calibrate(self, scores: np.ndarray, is_disturbance: np.ndarray) -> float:
        N = len(scores)
        finite_sample_margin = 1.0 / np.sqrt(max(N, 1))
        for lam in np.linspace(0.01, 0.99, 99):
            fp_risk = np.sum((scores >= lam) & is_disturbance) / max(N, 1)
            if fp_risk + 0.35 * finite_sample_margin <= self.alpha:
                self.lambda_hat = float(lam)
                break
        return self.lambda_hat
