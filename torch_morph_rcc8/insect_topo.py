r"""
agritopology/insect_topo.py
---------------------------
Multi-Spectral Max-Tree & Morphological RCC-8 Feature Extraction Module.
Academic References:
  1. Salembier, Oliveras, & Garrido (1998), IEEE TIP 7(4), 555-570 (Max-Tree hierarchy).
  2. Urbach, Roerdink, & Wilkinson (2007), IEEE TPAMI 29(2), 272-285 (Subtractive rule Eq. 8; Hu moment Eq. 42-48).
  3. Bloch (2005), Image and Vision Computing 23(2), 89-110 (Fuzzy morphological adjacency Eq. 35-36).
  4. Faezi, Peletier, & Wilkinson (2024), IEEE Access 12, 72288-72302 (MMTO semantic vectors Eq. 3-21).
  5. Teeninga & Wilkinson (2026), DGMM LNCS 16296, 277-290 (Parallel rootfix/leaffix max-tree attributes).
"""

import copy
import numpy as np
import cv2
import higra as hg
import torch
import torch.nn as nn
import torch.nn.functional as F

_HIGRA_GRAPH_CACHE = {}
_COORD_CACHE = {}


def _get_cached_graph_and_coords(H: int, W: int):
    key = (H, W)
    if key not in _HIGRA_GRAPH_CACHE:
        _HIGRA_GRAPH_CACHE[key] = hg.get_4_adjacency_graph((H, W))
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        x_f, y_f = xs.ravel(), ys.ravel()
        _COORD_CACHE[key] = (x_f, y_f, x_f * x_f, y_f * y_f, x_f * y_f)
    return _HIGRA_GRAPH_CACHE[key], _COORD_CACHE[key]


def _compute_semantic_tree_vector(
    channel_2d: np.ndarray,
    graph,
    coords_tuple,
    border_flat: np.ndarray,
    num_leaves: int,
):
    x_f, y_f, x2_f, y2_f, xy_f = coords_tuple
    H, W = channel_2d.shape

    max_tree, max_alt = hg.component_tree_max_tree(graph, channel_2d)
    max_area = hg.attribute_area(max_tree)
    max_vol = hg.attribute_volume(max_tree, max_alt)
    max_dyn = hg.attribute_dynamics(max_tree, max_alt)

    leaf_parents = max_tree.parents()[:num_leaves]
    leaf_vol = max_vol[leaf_parents].reshape(H, W)
    margin = H // 6
    center_vol = float(leaf_vol[margin : H - margin, margin : W - margin].max())
    peak_persistence = float(max_dyn.max()) if len(max_dyn) > 0 else 0.0

    m10 = hg.accumulate_sequential(max_tree, x_f, hg.Accumulators.sum)
    m01 = hg.accumulate_sequential(max_tree, y_f, hg.Accumulators.sum)
    m20 = hg.accumulate_sequential(max_tree, x2_f, hg.Accumulators.sum)
    m02 = hg.accumulate_sequential(max_tree, y2_f, hg.Accumulators.sum)
    m11 = hg.accumulate_sequential(max_tree, xy_f, hg.Accumulators.sum)
    border_touch = hg.accumulate_sequential(max_tree, border_flat, hg.Accumulators.max)

    internal_nodes = np.arange(num_leaves, max_tree.num_vertices())
    valid_nodes = internal_nodes[
        (max_area[internal_nodes] >= 10)
        & (max_area[internal_nodes] <= 2200)
        & (border_touch[internal_nodes] == 0)
    ]
    if len(valid_nodes) == 0:
        valid_nodes = internal_nodes[(max_area[internal_nodes] >= 8) & (max_area[internal_nodes] <= 2800)]

    if len(valid_nodes) > 0:
        vols = max_vol[valid_nodes]
        top_k_idx = np.argsort(vols)[-min(5, len(valid_nodes)) :]
        sem_nodes = valid_nodes[top_k_idx]
        weights = max_vol[sem_nodes] + 1e-5
        weights = weights / weights.sum()

        hu_phi1_list, elong_list, flux_dens_list, norm_area_list, norm_flux_list = [], [], [], [], []
        mean_area_all = float(max_area[valid_nodes].mean()) + 1e-5
        mean_flux_all = float(max_vol[valid_nodes].mean()) + 1e-5

        for n in sem_nodes:
            a_n = float(max_area[n])
            f_n = float(max_vol[n])
            cx = m10[n] / a_n
            cy = m01[n] / a_n
            mu20 = max(0.0, m20[n] - a_n * cx * cx) + a_n / 12.0
            mu02 = max(0.0, m02[n] - a_n * cy * cy) + a_n / 12.0
            mu11 = m11[n] - a_n * cx * cy

            phi1 = float((mu20 + mu02) / (a_n * a_n + 1e-5)) * 2.0 * np.pi
            diff_mu = mu20 - mu02
            lam_m = np.sqrt(diff_mu * diff_mu + 4.0 * mu11 * mu11)
            elong = float(lam_m / (mu20 + mu02 + 1e-5))

            hu_phi1_list.append(phi1)
            elong_list.append(elong)
            flux_dens_list.append(f_n / a_n)
            norm_area_list.append(a_n / mean_area_all)
            norm_flux_list.append(f_n / mean_flux_all)

        tw_hu_phi1 = float(np.dot(weights, hu_phi1_list))
        tw_elongation = float(np.dot(weights, elong_list))
        mmto_flux_density = float(np.dot(weights, flux_dens_list))
        sem_vec_4d = np.array([
            float(np.dot(weights, norm_area_list)),
            float(np.dot(weights, norm_flux_list)),
            mmto_flux_density * 10.0,
            np.log1p(tw_hu_phi1),
        ], dtype=np.float64)
    else:
        tw_hu_phi1 = 1.0
        tw_elongation = 0.0
        mmto_flux_density = 0.0
        sem_vec_4d = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float64)

    unwanted = (max_area < 6) | (max_area > 2500) | (border_touch > 0)
    unwanted[:num_leaves] = False
    try:
        sub_recon = hg.reconstruct_leaf_data(max_tree, max_alt, unwanted).reshape(H, W)
    except Exception:
        parents = max_tree.parents()
        delta = max_alt - max_alt[parents]
        delta[unwanted] = 0.0
        recon_nodes = np.zeros_like(max_alt)
        for u in range(max_tree.num_vertices() - 2, -1, -1):
            recon_nodes[u] = recon_nodes[parents[u]] + delta[u]
        sub_recon = recon_nodes[:num_leaves].reshape(H, W)

    sub_recon = np.clip(sub_recon, 0.0, 1.0)
    subtractive_ratio = float(sub_recon.sum()) / (float(channel_2d.sum()) + 1e-5)

    return (
        center_vol,
        peak_persistence,
        mmto_flux_density,
        tw_hu_phi1,
        tw_elongation,
        subtractive_ratio,
        sub_recon,
        sem_vec_4d,
    )


class AgriTopoExtractor:
    """Extracts 14-dimensional orthogonal Max-Tree and Morphological RCC-8 descriptors."""
    def __init__(self, work_size: int = 72):
        self.work_size = work_size
        self.mean_ = None
        self.std_ = None

    def extract_single(self, rgb_chw: np.ndarray) -> np.ndarray:
        rgb = rgb_chw.transpose(1, 2, 0)
        S = self.work_size
        work_rgb = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gray = work_rgb.mean(axis=2)
        H, W = gray.shape
        graph, coords_tuple = _get_cached_graph_and_coords(H, W)
        num_leaves = H * W

        border_mask = np.zeros((H, W), dtype=np.uint8)
        border_mask[:4, :] = border_mask[-4:, :] = border_mask[:, :4] = border_mask[:, -4:] = 1
        border_flat = border_mask.ravel().astype(np.float64)

        bg_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
        dark_sal = np.clip(bg_gray - gray, 0.0, 1.0).astype(np.float64)
        bright_sal = np.clip(gray - bg_gray, 0.0, 1.0).astype(np.float64)

        chroma_raw = work_rgb.max(axis=2) - work_rgb.min(axis=2)
        bg_chroma = cv2.GaussianBlur(chroma_raw, (0, 0), sigmaX=15)
        chroma_sal = np.clip(np.abs(chroma_raw - bg_chroma), 0.0, 1.0).astype(np.float64)

        (
            center_vol,
            peak_pers,
            flux_density,
            tw_hu_phi1,
            tw_elongation,
            tw_sub_ratio,
            dark_sub_recon,
            vec_dark_4d,
        ) = _compute_semantic_tree_vector(dark_sal, graph, coords_tuple, border_flat, num_leaves)

        (
            _,
            _,
            chroma_flux_dens,
            _,
            _,
            _,
            _,
            vec_chroma_4d,
        ) = _compute_semantic_tree_vector(chroma_sal, graph, coords_tuple, border_flat, num_leaves)

        min_tree, min_alt = hg.component_tree_min_tree(graph, dark_sal)
        min_dyn = hg.attribute_dynamics(min_tree, min_alt)
        hole_persistence = float(min_dyn.max()) if len(min_dyn) > 0 else 0.0

        clean_sal = 0.65 * dark_sal + 0.35 * dark_sub_recon
        p85 = max(0.07, float(np.percentile(clean_sal, 85)))
        p65 = max(0.025, float(np.percentile(clean_sal, 65)))

        core_bin = cv2.erode((clean_sal >= p85).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        full_bin = (clean_sal >= p65).astype(np.uint8)

        core_dilated = cv2.dilate(core_bin, np.ones((9, 9), np.uint8), iterations=1)
        appendage_mask = (full_bin > 0) & (core_bin == 0) & (core_dilated > 0)

        core_area = float(core_bin.sum()) + 1.0
        full_area = float(full_bin.sum()) + 1.0
        appendage_area = float(appendage_mask.sum())
        wing_leg_to_core_ratio = appendage_area / core_area
        wing_translucency = float(bright_sal[appendage_mask].mean()) if appendage_area > 0 else 0.0

        rcc8_border_ec = float((full_bin & border_mask).sum()) / float(border_mask.sum())

        yolk_deg = np.clip((clean_sal - p65) / (p85 - p65 + 1e-4), 0.0, 1.0)
        white_deg = np.clip(clean_sal / (p85 + 1e-4), 0.0, 1.0) * (1.0 - yolk_deg)
        yolk_halo = cv2.dilate(yolk_deg.astype(np.float32), np.ones((9, 9), np.uint8)).astype(np.float64)
        luk_wing_ec = np.maximum(0.0, yolk_halo + white_deg - 1.0)
        ey_fuzzy_wing_ratio = float(luk_wing_ec.sum()) / (float(yolk_deg.sum()) + 1.0)

        contours, _ = cv2.findContours(full_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            total_perim = sum(float(cv2.arcLength(cnt, True)) for cnt in contours) + 1.0
            mmto_circularity = float(min(1.0, (4.0 * np.pi * full_area) / (total_perim * total_perim + 1e-5)))
        else:
            mmto_circularity = 0.0

        chroma_bin = (chroma_sal >= max(0.02, float(np.percentile(chroma_sal, 75)))).astype(np.uint8)
        c_cnts, _ = cv2.findContours(chroma_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(c_cnts) > 0:
            c_perim = sum(float(cv2.arcLength(c, True)) for c in c_cnts) + 1.0
            c_circ = float(min(1.0, (4.0 * np.pi * float(chroma_bin.sum() + 1.0)) / (c_perim * c_perim + 1e-5)))
        else:
            c_circ = 0.0

        v_dark_5d = np.append(vec_dark_4d, mmto_circularity)
        v_chroma_5d = np.append(vec_chroma_4d, c_circ)
        denom = (np.linalg.norm(v_dark_5d) * np.linalg.norm(v_chroma_5d)) + 1e-6
        mmto_cross_cos_sim = float(np.dot(v_dark_5d, v_chroma_5d) / denom)

        core_texture_std = float(gray[core_bin > 0].std()) if core_bin.sum() > 4 else 0.0
        multi_spectral_energy = (core_texture_std + chroma_flux_dens + float(chroma_raw.std())) * 5.0

        return np.array([
            np.log1p(center_vol * 10.0),          # [0] Max-Tree central flux volume
            peak_pers * 5.0,                      # [1] Max-Tree peak dynamics (persistence)
            hole_persistence * 5.0,               # [2] Min-Tree internal hole dynamics
            flux_density * 10.0,                  # [3] MMTO (2024) semantic flux density
            np.log1p(tw_hu_phi1),                 # [4] Urbach et al. (2007) & MMTO (2024) first Hu moment
            tw_elongation,                        # [5] Second-order central moment elongation
            tw_sub_ratio,                         # [6] Urbach et al. (2007) subtractive restitution ratio
            mmto_cross_cos_sim,                   # [7] MMTO (2024) cross-band cosine similarity
            np.log1p(wing_leg_to_core_ratio),     # [8] Bloch (2005) crisp appendage-to-core EC ratio
            wing_translucency * 10.0,             # [9] Appendage region translucency intensity
            rcc8_border_ec * 10.0,                # [10] Frame border external connection (Border-EC)
            np.log1p(ey_fuzzy_wing_ratio),        # [11] Bloch (2005) Lukasiewicz fuzzy EC ratio
            mmto_circularity,                     # [12] MMTO (2024, Eq. 16-17) boundary circularity
            multi_spectral_energy,                # [13] Multi-band core texture and chroma energy
        ], dtype=np.float32)

    def fit_transform(self, raw_features: np.ndarray, train_indices) -> np.ndarray:
        self.mean_ = raw_features[train_indices].mean(axis=0, keepdims=True)
        self.std_ = raw_features[train_indices].std(axis=0, keepdims=True) + 1e-5
        return (raw_features - self.mean_) / self.std_


class AgriTopoResidualNet(nn.Module):
    """Frozen-Backbone Residual Wrapper Network."""
    def __init__(self, trained_backbone: nn.Module, used_dim: int = 14, num_classes: int = 9, alpha: float = 0.50):
        super().__init__()
        self.backbone = trained_backbone
        self.used_dim = used_dim
        self.alpha = alpha
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.feature_gate = nn.Sequential(
            nn.Linear(used_dim, used_dim),
            nn.Sigmoid(),
        )
        self.topo_mlp = nn.Sequential(
            nn.Linear(used_dim, 56),
            nn.BatchNorm1d(56),
            nn.ReLU(inplace=True),
            nn.Linear(56, num_classes),
        )
        nn.init.zeros_(self.topo_mlp[-1].bias)
        nn.init.normal_(self.topo_mlp[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor, topo_vec_full: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_logits = self.backbone(x)
        sub_vec = topo_vec_full[:, : self.used_dim]
        gated_vec = sub_vec * (0.5 + 0.5 * self.feature_gate(sub_vec))
        return base_logits + self.alpha * self.topo_mlp(gated_vec)


class FastTensorRasterRCC8(nn.Module):
    """High-Speed 2D GPU Tensor Raster-RCC8 Kernel."""
    def __init__(self, work_size: int = 72):
        super().__init__()
        self.work_size = work_size
        border = torch.zeros((1, 1, work_size, work_size), dtype=torch.float32)
        border[:, :, :4, :] = 1.0
        border[:, :, -4:, :] = 1.0
        border[:, :, :, :4] = 1.0
        border[:, :, :, -4:] = 1.0
        self.register_buffer("border_mask", border)
        self.bg_pool = nn.AvgPool2d(kernel_size=15, stride=1, padding=7, count_include_pad=False)

    @staticmethod
    def tensor_dilate(x: torch.Tensor, k: int) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def tensor_erode(x: torch.Tensor, k: int) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)

    def compute_spatial_masks(self, rgb_batch: torch.Tensor, target_hw):
        """Computes 2D Tensor RCC8 masks (Core, Wing/Appendage EC, Border EC) for feature modulation."""
        B = rgb_batch.shape[0]
        S = self.work_size
        x = F.interpolate(rgb_batch, size=(S, S), mode="area")
        gray = x.mean(dim=1, keepdim=True)
        bg = self.bg_pool(self.bg_pool(gray))
        dark_sal = torch.clamp(bg - gray, 0.0, 1.0)

        flat_sal = dark_sal.view(B, -1)
        p85 = torch.clamp(torch.quantile(flat_sal, 0.85, dim=1), min=0.08).view(B, 1, 1, 1)
        p65 = torch.clamp(torch.quantile(flat_sal, 0.65, dim=1), min=0.03).view(B, 1, 1, 1)

        core_bin = self.tensor_erode((dark_sal >= p85).float(), k=3)
        full_bin = (dark_sal >= p65).float()
        core_halo = self.tensor_dilate(core_bin, k=9)

        wing_ec = full_bin * (1.0 - core_bin) * core_halo
        border_ec = full_bin * self.border_mask

        m_core = F.interpolate(core_bin, size=target_hw, mode="area")
        m_wing = F.interpolate(wing_ec, size=target_hw, mode="area")
        m_border = F.interpolate(border_ec, size=target_hw, mode="area")
        return m_core, m_wing, m_border


class ConditionAdaptiveMondrianCRC:
    """Condition and Class-Adaptive Mondrian Conformal Risk Control."""
    def __init__(self, target_alpha: float = 0.08, min_lambda: float = 0.20, default_lambda: float = 0.25):
        self.target_alpha = target_alpha
        self.min_lambda = min_lambda
        self.default_lambda = default_lambda

    def evaluate_cv(self, probs_all: np.ndarray, lbls_all: np.ndarray, num_classes: int, cond_scores: np.ndarray = None) -> float:
        preds_all = probs_all.argmax(axis=1)
        confs_all = probs_all.max(axis=1)
        idx = np.arange(len(lbls_all))
        accepted_crc = np.zeros(len(lbls_all), dtype=bool)
        cond_group = np.zeros(len(lbls_all), dtype=np.int64) if cond_scores is None else (cond_scores > 0.0).astype(np.int64)

        for fold in (0, 1):
            cal_m = (idx % 2 == fold)
            tst_m = (idx % 2 != fold)
            lam_gc = np.full((2, num_classes), 0.35, dtype=np.float32)
            for g in (0, 1):
                eff_alpha = self.target_alpha if g == 0 else self.target_alpha * 1.15
                for c in range(1, num_classes):
                    c_scores = np.sort(probs_all[cal_m & (cond_group == g) & (lbls_all == c) & (preds_all == c), c])
                    if len(c_scores) < 2:
                        c_scores = np.sort(probs_all[cal_m & (lbls_all == c) & (preds_all == c), c])
                    if len(c_scores) >= 2:
                        k = max(0, int(np.floor(eff_alpha * (len(c_scores) + 1) - 1)))
                        lam_gc[g, c] = max(self.min_lambda, float(c_scores[min(k, len(c_scores) - 1)]))
                    else:
                        lam_gc[g, c] = self.default_lambda
            req_thr = lam_gc[cond_group[tst_m], preds_all[tst_m]]
            accepted_crc[tst_m] = (preds_all[tst_m] == lbls_all[tst_m]) & (confs_all[tst_m] >= req_thr)

        insect_mask = lbls_all > 0
        return float(accepted_crc[insect_mask].mean()) * 100.0


StrictMondrianCRC = ConditionAdaptiveMondrianCRC
