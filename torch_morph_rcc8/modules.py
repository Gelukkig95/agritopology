r"""
core/modules.py
---------------
Vectorized GPU Matrix-Multiplication RCC-8 External Connection (EC) Adjacency & Higra Token Extractor.
"""

import time
import numpy as np
import higra as hg
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rcc8_ec_matrix(token_masks: list | torch.Tensor, dilation_iter: int = 2) -> np.ndarray | torch.Tensor:
    r"""
    Vectorized PyTorch Tensor RCC8-EC Adjacency Matrix following Bloch (2005) Eq. (35).
    Verifies both disjoint interiors (X \cap Y = \emptyset) and dilated boundary contact.
    """
    is_tensor = isinstance(token_masks, torch.Tensor)
    if len(token_masks) == 0:
        return torch.zeros((0, 0)) if is_tensor else np.zeros((0, 0), dtype=np.float32)

    if is_tensor:
        masks_t = (token_masks > 0.5).float()
    else:
        masks_t = torch.from_numpy(np.stack([np.asarray(m, dtype=np.float32) > 0.5 for m in token_masks], axis=0)).float()

    k_size = 2 * dilation_iter + 1
    dilated = F.max_pool2d(masks_t.unsqueeze(1), kernel_size=k_size, stride=1, padding=dilation_iter).squeeze(1)

    flat_dil = dilated.flatten(1)
    flat_orig = masks_t.flatten(1)
    disjoint = (torch.matmul(flat_orig, flat_orig.T) == 0)
    touch = (torch.matmul(flat_dil, flat_orig.T) > 0) | (torch.matmul(flat_orig, flat_dil.T) > 0)
    ec_mat = (disjoint & touch).float()
    ec_mat.fill_diagonal_(0.0)
    return ec_mat if is_tensor else ec_mat.cpu().numpy()


def extract_higra_rcc8_tokens(scene: np.ndarray):
    """Extracts connected component tokens and their RCC8-EC adjacency matrix via Higra Max-Tree."""
    t0 = time.perf_counter()
    H, W = scene.shape
    quant = np.round(scene * 255.0).astype(np.float64) / 255.0
    bg_med = float(np.median(quant))

    graph = hg.get_4_adjacency_graph((H, W))
    tree, alt = hg.component_tree_max_tree(graph, quant)
    area = hg.attribute_area(tree)
    clean_pot = hg.reconstruct_leaf_data(tree, alt, area < 6).reshape(H, W)

    sup_mask = clean_pot > 0.20
    core_tier = clean_pot > 0.45

    lbl_sup, n_sup = ndimage.label(sup_mask)
    token_masks = []
    for sid in range(1, n_sup + 1):
        reg = (lbl_sup == sid)
        icore = reg & core_tier
        iec = reg & (~core_tier)
        if icore.sum() >= 10 and iec.sum() >= 10:
            token_masks.append(icore)
            l_ec, n_ec = ndimage.label(iec)
            for eid in range(1, n_ec + 1):
                if (l_ec == eid).sum() >= 8:
                    token_masks.append(l_ec == eid)
        elif reg.sum() >= 8:
            token_masks.append(reg)

    K = len(token_masks)
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    feats = np.zeros((K, 5), dtype=np.float32)
    for i, m in enumerate(token_masks):
        dyn = float(clean_pot[m].max() - bg_med)
        b_touch = min(1.0, float((m & border).sum()) / 12.0)
        ys, xs = np.where(m)
        if len(ys) > 5:
            eig = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs])))
            elong = min(1.0, float(np.sqrt(max(eig[-1], 1e-5) / max(eig[0], 1e-5))) / 10.0)
        else:
            elong = 0.1
        filled = ndimage.binary_fill_holes(m)
        fy, fx = np.where(filled)
        max_r = float(np.hypot(fy - fy.mean(), fx - fx.mean()).max())
        rad_fill = min(1.0, float(filled.sum() / max(np.pi * (max_r ** 2), 1.0)))
        hole_depth = max(0.0, float(clean_pot[m].mean() - clean_pot[filled].min()))
        feats[i] = [dyn, b_touch, elong, rad_fill, hole_depth]

    rcc8_ec = compute_rcc8_ec_matrix(token_masks, dilation_iter=2)
    ms = (time.perf_counter() - t0) * 1000.0
    return torch.from_numpy(feats), torch.from_numpy(rcc8_ec), token_masks, ms


class TopoRCC8Transformer(nn.Module):
    """Object-Centric Topological Token & RCC8 Relational Self-Attention Module."""
    def __init__(self, d_model: int = 16):
        super().__init__()
        self.learnable_tau = nn.Parameter(torch.full((5,), 0.30))
        self.encoder = nn.Linear(5, d_model)
        self.q_lin = nn.Linear(d_model, d_model, bias=False)
        self.k_lin = nn.Linear(d_model, d_model, bias=False)
        self.v_lin = nn.Linear(d_model, d_model, bias=False)
        self.rcc8_ec_gain = nn.Parameter(torch.tensor(3.2))
        self.rcc8_dc_mask = nn.Parameter(torch.tensor(-3.5))
        self.head = nn.Linear(d_model, 1)

    def forward(self, topo_feats: torch.Tensor, rcc8_ec_mat: torch.Tensor):
        predicates = torch.sigmoid(10.0 * (topo_feats - self.learnable_tau))
        h = F.silu(self.encoder(predicates))
        Q, K, V = self.q_lin(h), self.k_lin(h), self.v_lin(h)
        scores = torch.matmul(Q, K.T) / np.sqrt(h.shape[-1])

        eye = torch.eye(len(topo_feats), device=topo_feats.device)
        rel_bias = eye * 2.0 + rcc8_ec_mat * self.rcc8_ec_gain + (1.0 - eye - rcc8_ec_mat) * self.rcc8_dc_mask
        attn = F.softmax(scores + rel_bias, dim=-1)

        h_ctx = h + torch.matmul(attn, V)
        omega = torch.sigmoid(self.head(h_ctx)).squeeze(-1)
        return omega, attn
