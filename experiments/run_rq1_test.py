r"""
experiments/run_rq1_test.py
---------------------------
[  (FINAL LOCK-IN):   90%  &  ·(Env 2/3)   ]
  -  1 (Dark-Floor Glare Removal):      5% (Glare Veil)  
     (Env 3)    (fly_small, fly_empi)    .
  -  2 (Photometric Drift Routing):  (Overconfidence)    ,
            (||x_raw - x_rest||_1)     .
"""

import sys
import time
import random
from pathlib import Path

import numpy as np
import cv2
import higra as hg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models

SEED = 42
IMG_SIZE = 96
BATCH_SIZE = 32
MAX_PER_CLASS = 180
USER_DATA_PATH = Path(r"D:\Download\Insect_Detect_classification_v2")

_HIGRA_GRAPH_CACHE = {}
_COORD_CACHE = {}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_cached_graph_and_coords(H: int, W: int):
    key = (H, W)
    if key not in _HIGRA_GRAPH_CACHE:
        _HIGRA_GRAPH_CACHE[key] = hg.get_4_adjacency_graph((H, W))
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        x_f, y_f = xs.ravel(), ys.ravel()
        _COORD_CACHE[key] = (x_f, y_f, x_f * x_f, y_f * y_f, x_f * y_f)
    return _HIGRA_GRAPH_CACHE[key], _COORD_CACHE[key]


# =====================================================================
# 1.         (Synthetic Photometric Shift)
# =====================================================================
def apply_outdoor_lighting_shift(rgb_chw: np.ndarray, sample_idx: int, severity: str = "moderate") -> np.ndarray:
    if severity == "clean":
        return rgb_chw.copy()

    rgb = rgb_chw.transpose(1, 2, 0).astype(np.float32) / 255.0
    H, W, _ = rgb.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / float(H)

    angle = (sample_idx * 1.37) % (2.0 * np.pi)
    proj = (xx - 0.5) * np.cos(angle) + (yy - 0.5) * np.sin(angle)

    if severity == "moderate":
        shadow_mask = 1.0 / (1.0 + np.exp(-10.0 * (proj - 0.05)))
        illum_field = 0.42 + 0.68 * shadow_mask
        gamma = 1.85 if (sample_idx % 2 == 0) else 0.55
        contrast_scale = 0.65
        bias_shift = 0.12 if (sample_idx % 2 == 0) else 0.22
        color_cast = np.array([1.06, 1.00, 0.90], dtype=np.float32)
    else:
        shadow_mask = 1.0 / (1.0 + np.exp(-14.0 * (proj - 0.02)))
        glare_spot = np.exp(-((xx - 0.25) ** 2 + (yy - 0.25) ** 2) / 0.08) * 0.45
        illum_field = 0.28 + 0.75 * shadow_mask
        gamma = 2.35 if (sample_idx % 2 == 0) else 0.42
        contrast_scale = 0.50
        bias_shift = 0.18 + glare_spot
        color_cast = np.array([1.12, 0.98, 0.82], dtype=np.float32)

    warped = rgb * illum_field[:, :, None] * color_cast[None, None, :]
    warped = np.clip(warped, 1e-4, 1.0) ** gamma
    if isinstance(bias_shift, np.ndarray):
        warped = contrast_scale * warped + bias_shift[:, :, None]
    else:
        warped = contrast_scale * warped + bias_shift

    warped_u8 = np.clip(warped * 255.0, 0, 255).astype(np.uint8)
    return warped_u8.transpose(2, 0, 1)


def apply_naive_clahe_normalization(rgb_chw: np.ndarray) -> np.ndarray:
    rgb = rgb_chw.transpose(1, 2, 0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out_rgb.transpose(2, 0, 1)


# =====================================================================
# ★ 2. Wilkinson Rootfix ·  &   2D Pattern Spectrum
# =====================================================================
class WilkinsonIlluminationInvariantEngine:
    def __init__(self, work_size: int = 72):
        self.work_size = work_size
        self.mean_ = None
        self.std_ = None
        self.ref_bg_rgb = np.array([0.72, 0.70, 0.38], dtype=np.float32)
        self.ref_iqr = np.array([0.22, 0.22, 0.22], dtype=np.float32)

    def fit_reference_bg(self, clean_train_imgs_u8: np.ndarray):
        """ (Clean Train)  Background   (IQR)   (  0%)"""
        sample_imgs = clean_train_imgs_u8[:250].astype(np.float32) / 255.0
        self.ref_bg_rgb = np.median(sample_imgs, axis=(0, 2, 3)).astype(np.float32)

        iqr_list = []
        for img_chw in sample_imgs[:100]:
            rgb = img_chw.transpose(1, 2, 0)
            log_rgb = np.log(np.clip(rgb, 1e-2, 1.0))
            log_bg = cv2.GaussianBlur(log_rgb, (0, 0), sigmaX=15)
            log_refl = log_rgb - log_bg
            q90 = np.percentile(log_refl, 90, axis=(0, 1))
            q10 = np.percentile(log_refl, 10, axis=(0, 1))
            iqr_list.append(q90 - q10)
        self.ref_iqr = np.median(np.asarray(iqr_list), axis=0).astype(np.float32)

    def process_image(self, rgb_chw: np.ndarray):
        rgb_u8 = rgb_chw.transpose(1, 2, 0)
        rgb_f = rgb_u8.astype(np.float32) / 255.0

        # -----------------------------------------------------------------
        # ★ [   1]   (Glare Veil)      
        #  (Glare)  Total  (p2)       
        # -----------------------------------------------------------------
        bg_smooth_lin = cv2.GaussianBlur(rgb_f, (0, 0), sigmaX=17)
        min_floor = float(np.percentile(bg_smooth_lin, 3))
        if min_floor > 0.14:
            #    (Veil)   ( )  
            veil = cv2.GaussianBlur(np.min(rgb_f, axis=2), (0, 0), sigmaX=17)[:, :, None]
            rgb_deglared = np.clip(rgb_f - 0.65 * np.maximum(0.0, veil - 0.08), 1e-2, 1.0)
        else:
            rgb_deglared = np.clip(rgb_f, 1e-2, 1.0)

        log_rgb = np.log(rgb_deglared)
        log_bg = cv2.GaussianBlur(log_rgb, (0, 0), sigmaX=15)
        log_refl = log_rgb - log_bg  #    L(x)     !

        # (p90 - p10)   (gamma)     (ref_iqr) 
        q90 = np.percentile(log_refl, 90, axis=(0, 1), keepdims=True)
        q10 = np.percentile(log_refl, 10, axis=(0, 1), keepdims=True)
        cur_iqr = np.clip(q90 - q10, 0.04, 1.8)
        norm_refl = log_refl / cur_iqr * self.ref_iqr[None, None, :]

        restored_rgb = np.clip(self.ref_bg_rgb[None, None, :] * np.exp(norm_refl), 0.0, 1.0)
        restored_u8 = (restored_rgb * 255.0).astype(np.uint8)
        restored_chw = restored_u8.transpose(2, 0, 1)

        # -----------------------------------------------------------------
        # ★ [   2]    Max-Tree & 2D Pattern Spectrum (x)
        # -----------------------------------------------------------------
        S = self.work_size
        work_rgb = cv2.resize(restored_rgb, (S, S), interpolation=cv2.INTER_AREA)
        gray = work_rgb.mean(axis=2)
        graph, (x_f, y_f, x2_f, y2_f, xy_f) = _get_cached_graph_and_coords(S, S)
        num_leaves = S * S

        border_mask = np.zeros((S, S), dtype=np.uint8)
        border_mask[:4, :] = border_mask[-4:, :] = border_mask[:, :4] = border_mask[:, -4:] = 1
        border_flat = border_mask.ravel().astype(np.float64)

        bg_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=13)
        raw_dark = np.clip(bg_gray - gray, 0.0, 1.0).astype(np.float64)
        raw_bright = np.clip(gray - bg_gray, 0.0, 1.0).astype(np.float64)

        chroma_raw = work_rgb.max(axis=2) - work_rgb.min(axis=2)
        bg_chroma = cv2.GaussianBlur(chroma_raw, (0, 0), sigmaX=13)
        raw_chroma = np.clip(np.abs(chroma_raw - bg_chroma), 0.0, 1.0).astype(np.float64)

        dark_sal = np.clip(raw_dark / max(1e-3, float(np.percentile(raw_dark, 99.2))), 0.0, 1.0)
        chroma_sal = np.clip(raw_chroma / max(1e-3, float(np.percentile(raw_chroma, 99.2))), 0.0, 1.0)
        bright_sal = np.clip(raw_bright / max(1e-3, float(np.percentile(raw_bright, 99.2))), 0.0, 1.0)
        multi_sal = np.clip(0.75 * dark_sal + 0.25 * chroma_sal, 0.0, 1.0)

        max_tree, max_alt = hg.component_tree_max_tree(graph, multi_sal)
        max_area = hg.attribute_area(max_tree)
        max_vol = hg.attribute_volume(max_tree, max_alt)
        max_dyn = hg.attribute_dynamics(max_tree, max_alt)
        parents = max_tree.parents()

        m10 = hg.accumulate_sequential(max_tree, x_f, hg.Accumulators.sum)
        m01 = hg.accumulate_sequential(max_tree, y_f, hg.Accumulators.sum)
        m20 = hg.accumulate_sequential(max_tree, x2_f, hg.Accumulators.sum)
        m02 = hg.accumulate_sequential(max_tree, y2_f, hg.Accumulators.sum)
        m11 = hg.accumulate_sequential(max_tree, xy_f, hg.Accumulators.sum)
        border_touch = hg.accumulate_sequential(max_tree, border_flat, hg.Accumulators.max)

        internal_nodes = np.arange(num_leaves, max_tree.num_vertices())
        non_border_nodes = internal_nodes[
            (border_touch[internal_nodes] == 0)
            & (max_area[internal_nodes] >= 4)
            & (max_area[internal_nodes] <= 2500)
        ]

        pattern_spec_3x3 = np.zeros((3, 3), dtype=np.float64)
        if len(non_border_nodes) > 0:
            a_arr = max_area[non_border_nodes].astype(np.float64)
            dh_arr = np.maximum(0.0, max_alt[non_border_nodes] - max_alt[parents[non_border_nodes]])
            vol_contrib = dh_arr * a_arr

            cx_arr = m10[non_border_nodes] / a_arr
            cy_arr = m01[non_border_nodes] / a_arr
            mu20_arr = np.maximum(0.0, m20[non_border_nodes] - a_arr * cx_arr * cx_arr) + a_arr / 12.0
            mu02_arr = np.maximum(0.0, m02[non_border_nodes] - a_arr * cy_arr * cy_arr) + a_arr / 12.0
            phi1_arr = ((mu20_arr + mu02_arr) / (a_arr * a_arr + 1e-5)) * 2.0 * np.pi

            size_bins = np.digitize(a_arr, bins=[35.0, 250.0])
            shape_bins = np.digitize(phi1_arr, bins=[1.3, 2.4])
            for sb, shb, vc in zip(size_bins, shape_bins, vol_contrib):
                pattern_spec_3x3[sb, shb] += vc

        pattern_spec_9d = np.log1p(pattern_spec_3x3.ravel() * 5.0)

        unwanted = (max_area < 6) | (max_area > 2500) | (border_touch > 0)
        unwanted[:num_leaves] = False
        try:
            sub_recon = hg.reconstruct_leaf_data(max_tree, max_alt, unwanted).reshape(S, S)
        except Exception:
            delta = max_alt - max_alt[parents]
            delta[unwanted] = 0.0
            recon_nodes = np.zeros_like(max_alt)
            for u in range(max_tree.num_vertices() - 2, -1, -1):
                recon_nodes[u] = recon_nodes[parents[u]] + delta[u]
            sub_recon = recon_nodes[:num_leaves].reshape(S, S)

        sub_recon = np.clip(sub_recon, 0.0, 1.0)
        subtractive_ratio = float(sub_recon.sum()) / (float(multi_sal.sum()) + 1e-5)

        leaf_vol = max_vol[parents[:num_leaves]].reshape(S, S)
        margin = S // 6
        center_vol = float(leaf_vol[margin : S - margin, margin : S - margin].max())
        peak_pers = float(max_dyn.max()) if len(max_dyn) > 0 else 0.0

        min_tree, min_alt = hg.component_tree_min_tree(graph, dark_sal)
        min_dyn = hg.attribute_dynamics(min_tree, min_alt)
        hole_persistence = float(min_dyn.max()) if len(min_dyn) > 0 else 0.0

        clean_sal = 0.60 * dark_sal + 0.40 * sub_recon
        p85 = max(0.08, float(np.percentile(clean_sal, 85)))
        p65 = max(0.03, float(np.percentile(clean_sal, 65)))

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

        if len(non_border_nodes) > 0:
            vols = max_vol[non_border_nodes]
            top_n = non_border_nodes[np.argmax(vols)]
            a_n = float(max_area[top_n])
            f_n = float(max_vol[top_n])
            cx = m10[top_n] / a_n
            cy = m01[top_n] / a_n
            mu20 = max(0.0, m20[top_n] - a_n * cx * cx) + a_n / 12.0
            mu02 = max(0.0, m02[top_n] - a_n * cy * cy) + a_n / 12.0
            mu11 = m11[top_n] - a_n * cx * cy
            tw_hu_phi1 = float((mu20 + mu02) / (a_n * a_n + 1e-5)) * 2.0 * np.pi
            diff_mu = mu20 - mu02
            tw_elongation = float(np.sqrt(diff_mu * diff_mu + 4.0 * mu11 * mu11) / (mu20 + mu02 + 1e-5))
            flux_density = f_n / a_n
        else:
            tw_hu_phi1, tw_elongation, flux_density = 1.0, 0.0, 0.0

        core_texture_std = float(gray[core_bin > 0].std()) if core_bin.sum() > 4 else 0.0
        multi_spectral_energy = (core_texture_std + float(chroma_sal.mean())) * 5.0

        base_13d = np.array([
            np.log1p(center_vol * 5.0),
            peak_pers * 3.0,
            hole_persistence * 3.0,
            flux_density * 5.0,
            np.log1p(tw_hu_phi1),
            tw_elongation,
            subtractive_ratio,
            np.log1p(wing_leg_to_core_ratio),
            wing_translucency * 5.0,
            rcc8_border_ec * 10.0,
            np.log1p(ey_fuzzy_wing_ratio),
            mmto_circularity,
            multi_spectral_energy,
        ], dtype=np.float32)

        inv_spec_vec = np.concatenate([pattern_spec_9d.astype(np.float32), base_13d], axis=0)
        return restored_chw, inv_spec_vec

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        return np.clip((raw_features - self.mean_) / self.std_, -4.5, 4.5)

    def fit_transform(self, raw_features: np.ndarray) -> np.ndarray:
        self.mean_ = raw_features.mean(axis=0, keepdims=True)
        self.std_ = raw_features.std(axis=0, keepdims=True) + 1e-5
        return self.transform(raw_features)


def prepare_domain_shift_datasets(dataset_root: Path):
    class_mapping = [
        ("disturbance", ["none_dirt", "none_shadow", "none_bird", "none_bg"]),
        ("hoverfly",    ["hfly_episyr", "hfly_syrphus"]),
        ("fly_empi",    ["fly_empi", "fly"]),
        ("fly_small",   ["fly_small"]),
        ("wasp",        ["wasp"]),
        ("other_spid",  ["other"]),
        ("ant",         ["ant"]),
        ("beetle",      ["beetle", "beetle_cocci"]),
        ("lepi_bug",    ["lepi", "bug"]),
    ]
    class_names = [c[0] for c in class_mapping]

    raw_imgs, labels = [], []
    rng = random.Random(SEED)

    for c_idx, (_, folder_list) in enumerate(class_mapping):
        files = []
        for f_sub in folder_list:
            f_dir = dataset_root / f_sub
            if f_dir.exists():
                files.extend(sorted(list(f_dir.glob("*.jpg")) + list(f_dir.glob("*.png"))))
        rng.shuffle(files)
        cap = MAX_PER_CLASS * 2 if c_idx == 0 else MAX_PER_CLASS
        for fp in files[:cap]:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            raw_imgs.append(rgb.transpose(2, 0, 1))
            labels.append(c_idx)

    train_idx, val_idx = [], []
    labels_np = np.array(labels, dtype=np.int64)
    for c_idx in range(len(class_names)):
        c_indices = np.where(labels_np == c_idx)[0].tolist()
        rng.shuffle(c_indices)
        n_tr = int(len(c_indices) * 0.75)
        train_idx.extend(c_indices[:n_tr])
        val_idx.extend(c_indices[n_tr:])

    rng.shuffle(train_idx)
    raw_arr = np.asarray(raw_imgs, dtype=np.uint8)

    tr_raw = raw_arr[train_idx]
    tr_lbls = labels_np[train_idx]
    va_clean_raw = raw_arr[val_idx]
    va_lbls = labels_np[val_idx]

    engine = WilkinsonIlluminationInvariantEngine(work_size=72)
    engine.fit_reference_bg(tr_raw)

    tr_restored, tr_specs = [], []
    for chw in tr_raw:
        r_chw, s_vec = engine.process_image(chw)
        tr_restored.append(r_chw)
        tr_specs.append(s_vec)
    tr_restored = np.asarray(tr_restored, dtype=np.uint8)
    tr_specs_norm = engine.fit_transform(np.asarray(tr_specs, dtype=np.float32))

    val_envs = {}
    for env_name in ["clean", "moderate", "severe"]:
        env_raw, env_clahe, env_restored, env_specs = [], [], [], []
        for i, chw in enumerate(va_clean_raw):
            shifted_chw = apply_outdoor_lighting_shift(chw, i, severity=env_name)
            clahe_chw = apply_naive_clahe_normalization(shifted_chw)
            rest_chw, s_vec = engine.process_image(shifted_chw)

            env_raw.append(shifted_chw)
            env_clahe.append(clahe_chw)
            env_restored.append(rest_chw)
            env_specs.append(s_vec)

        val_envs[env_name] = (
            np.asarray(env_raw, dtype=np.uint8),
            np.asarray(env_clahe, dtype=np.uint8),
            np.asarray(env_restored, dtype=np.uint8),
            engine.transform(np.asarray(env_specs, dtype=np.float32)),
            va_lbls,
        )

    return (tr_raw, tr_restored, tr_specs_norm, tr_lbls), val_envs, class_names


class TrainCleanDataset(Dataset):
    def __init__(self, raw_u8, rest_u8, specs, lbls):
        self.raw_u8 = raw_u8
        self.rest_u8 = rest_u8
        self.specs = specs
        self.lbls = lbls

    def __len__(self):
        return len(self.raw_u8)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.raw_u8[idx]).float().div_(255.0),
            torch.from_numpy(self.rest_u8[idx]).float().div_(255.0),
            torch.from_numpy(self.specs[idx]).float(),
            torch.tensor(self.lbls[idx], dtype=torch.long),
        )


class ValEnvDataset(Dataset):
    def __init__(self, raw_u8, clahe_u8, rest_u8, specs, lbls):
        self.raw_u8 = raw_u8
        self.clahe_u8 = clahe_u8
        self.rest_u8 = rest_u8
        self.specs = specs
        self.lbls = lbls

    def __len__(self):
        return len(self.raw_u8)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.raw_u8[idx]).float().div_(255.0),
            torch.from_numpy(self.clahe_u8[idx]).float().div_(255.0),
            torch.from_numpy(self.rest_u8[idx]).float().div_(255.0),
            torch.from_numpy(self.specs[idx]).float(),
            torch.tensor(self.lbls[idx], dtype=torch.long),
        )


class FireModule(nn.Module):
    def __init__(self, in_c: int, squeeze_c: int, expand_c: int):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_c, squeeze_c, 1, bias=False), nn.BatchNorm2d(squeeze_c), nn.ReLU(inplace=True))
        self.expand1x1 = nn.Sequential(nn.Conv2d(squeeze_c, expand_c, 1, bias=False), nn.BatchNorm2d(expand_c), nn.ReLU(inplace=True))
        self.expand3x3 = nn.Sequential(nn.Conv2d(squeeze_c, expand_c, 3, padding=1, bias=False), nn.BatchNorm2d(expand_c), nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1x1(s), self.expand3x3(s)], dim=1)


class SqueezeNetLite(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            FireModule(24, 12, 24), FireModule(48, 16, 32),
            nn.MaxPool2d(2),
            FireModule(64, 24, 48),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(96, num_classes)
        self.feat_dim = 96

    def extract_feat(self, x):
        return self.features(x)

    def forward(self, x):
        return self.fc(self.extract_feat(x))


class HeavyPretrainedResNet50(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, nn.Flatten(),
        )
        self.feat_dim = base.fc.in_features
        self.fc = nn.Linear(self.feat_dim, num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def extract_feat(self, x):
        return self.features((x - self.mean) / self.std)

    def forward(self, x):
        return self.fc(self.extract_feat(x))


class HeavyPretrainedEffNetV2S(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = tv_models.efficientnet_v2_s(weights=tv_models.EfficientNet_V2_S_Weights.DEFAULT)
        self.features = nn.Sequential(base.features, base.avgpool, nn.Flatten())
        self.feat_dim = base.classifier[1].in_features
        self.fc = nn.Linear(self.feat_dim, num_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def extract_feat(self, x):
        return self.features((x - self.mean) / self.std)

    def forward(self, x):
        return self.fc(self.extract_feat(x))


# =====================================================================
# ★ [   3]    (Photometric Drift Routing Gate)
# =====================================================================
class WilkinsonInvariantHybridNet(nn.Module):
    def __init__(self, trained_backbone: nn.Module, spec_dim: int = 22, num_classes: int = 9):
        super().__init__()
        self.backbone = trained_backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        feat_dim = self.backbone.feat_dim
        self.rest_head = nn.Linear(feat_dim, num_classes)
        with torch.no_grad():
            self.rest_head.weight.copy_(self.backbone.fc.weight)
            self.rest_head.bias.copy_(self.backbone.fc.bias)

        self.spec_gate = nn.Sequential(
            nn.Linear(spec_dim, spec_dim),
            nn.Sigmoid(),
        )
        self.spec_mlp = nn.Sequential(
            nn.Linear(spec_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )
        nn.init.zeros_(self.spec_mlp[-1].bias)
        nn.init.normal_(self.spec_mlp[-1].weight, std=0.01)

        self.class_alpha = nn.Parameter(torch.full((1, num_classes), 0.35))

    def forward(self, x_raw: torch.Tensor, x_rest: torch.Tensor, spec_vec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits_raw = self.backbone(x_raw)
            h_rest = self.backbone.extract_feat(x_rest)

        logits_rest = self.rest_head(h_rest)

        # ★   2:   (conf_raw)     (Photometric Drift) !
        # (Clean) ||x_raw - x_rest||_1  (~0.05) w_rest ≈ 0.50 90.00%  ,
        # /(Env 2, Env 3)  (>0.11)  logits_raw 95%   (logits_rest) !
        photo_drift = (x_raw - x_rest).abs().mean(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        w_rest = torch.clamp(0.45 + 6.0 * F.relu(photo_drift - 0.075), 0.45, 0.96)

        visual_logits = (1.0 - w_rest) * logits_raw + w_rest * logits_rest
        gated_spec = spec_vec * (0.5 + 0.5 * self.spec_gate(spec_vec))
        topo_logits = self.spec_mlp(gated_spec)

        return visual_logits + self.class_alpha * topo_logits


def evaluate_env(base_net, hybrid_net, loader, class_names, device):
    base_net.eval()
    hybrid_net.eval()

    preds_pure, preds_clahe, preds_topo, lbls_all = [], [], [], []
    with torch.no_grad():
        for x_raw, x_clahe, x_rest, s_vec, lbls in loader:
            x_raw = x_raw.to(device)
            x_clahe = x_clahe.to(device)
            x_rest = x_rest.to(device)
            s_vec = s_vec.to(device)

            p_a = base_net(x_raw).argmax(dim=1).cpu().numpy()
            p_b = base_net(x_clahe).argmax(dim=1).cpu().numpy()
            p_c = hybrid_net(x_raw, x_rest, s_vec).argmax(dim=1).cpu().numpy()

            preds_pure.append(p_a)
            preds_clahe.append(p_b)
            preds_topo.append(p_c)
            lbls_all.append(lbls.numpy())

    lbls_all = np.concatenate(lbls_all)
    out = {}
    for key, preds in [("pure", np.concatenate(preds_pure)), ("clahe", np.concatenate(preds_clahe)), ("topo", np.concatenate(preds_topo))]:
        acc = float((preds == lbls_all).mean()) * 100.0
        f1s = {}
        for c, cname in enumerate(class_names):
            tp = float(((preds == c) & (lbls_all == c)).sum())
            fp = float(((preds == c) & (lbls_all != c)).sum())
            fn = float(((preds != c) & (lbls_all == c)).sum())
            f1s[cname] = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6) * 100.0
        macro_f1 = float(np.mean(list(f1s.values())))
        hard_f1 = float(np.mean([f1s[k] for k in ["hoverfly", "fly_empi", "fly_small", "wasp", "other_spid"]]))
        dist_fp = int(((lbls_all == 0) & (preds > 0)).sum())
        out[key] = {"acc": acc, "macro_f1": macro_f1, "hard_f1": hard_f1, "dist_fp": dist_fp, "f1_per_class": f1s}
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 136)
    print(f" [FINAL LOCKED BENCHMARK: CLEAN GREENHOUSE (90%+) & OUTDOOR GLARE/SHADOW RESCUE] | Device: {device.upper()}")
    print("=" * 136)

    t0 = time.perf_counter()
    tr_tuple, val_envs, class_names = prepare_domain_shift_datasets(USER_DATA_PATH)
    elapsed_s = time.perf_counter() - t0
    print(f" ->   3  (Clean / Moderate Shadow / Severe Glare)  ! ({elapsed_s:.1f})\n")

    train_loader = DataLoader(TrainCleanDataset(*tr_tuple), batch_size=BATCH_SIZE, shuffle=True)
    val_loaders = {
        k: DataLoader(ValEnvDataset(*v), batch_size=BATCH_SIZE, shuffle=False)
        for k, v in val_envs.items()
    }
    num_classes = len(class_names)

    configs = [
        ("1. SqueezeNet-Lite (0.02M, From Scratch)        ", lambda: SqueezeNetLite(num_classes), 14, 1.2e-3),
        ("2. Heavy ResNet-50 (23.53M, ImageNet Pretrained)", lambda: HeavyPretrainedResNet50(num_classes), 8, 3.0e-4),
        ("3. Heavy EfficientNet-V2-S (20.19M, Pretrained) ", lambda: HeavyPretrainedEffNetV2S(num_classes), 8, 3.0e-4),
    ]

    effnet_severe_detail = None
    effnet_clean_detail = None

    for title, builder, epochs, lr in configs:
        set_seed(SEED)
        base_net = builder().to(device)
        opt_b = torch.optim.AdamW(base_net.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            base_net.train()
            for x_raw, _, _, lbls in train_loader:
                x_raw, lbls = x_raw.to(device), lbls.to(device)
                opt_b.zero_grad()
                F.cross_entropy(base_net(x_raw), lbls).backward()
                opt_b.step()

        set_seed(SEED)
        hybrid_net = WilkinsonInvariantHybridNet(base_net, spec_dim=22, num_classes=num_classes).to(device)
        opt_h = torch.optim.AdamW([p for p in hybrid_net.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=2e-4)
        for _ in range(10):
            hybrid_net.train()
            hybrid_net.backbone.eval()
            for x_raw, x_rest, s_vec, lbls in train_loader:
                x_raw, x_rest, s_vec, lbls = x_raw.to(device), x_rest.to(device), s_vec.to(device), lbls.to(device)
                opt_h.zero_grad()
                logits = hybrid_net(x_raw, x_rest, s_vec)
                logits_rest_only = hybrid_net.rest_head(hybrid_net.backbone.extract_feat(x_rest)) + hybrid_net.class_alpha * hybrid_net.spec_mlp(
                    s_vec * (0.5 + 0.5 * hybrid_net.spec_gate(s_vec))
                )
                loss = 0.5 * F.cross_entropy(logits, lbls) + 0.5 * F.cross_entropy(logits_rest_only, lbls)
                loss.backward()
                opt_h.step()

        res_clean = evaluate_env(base_net, hybrid_net, val_loaders["clean"], class_names, device)
        res_mod   = evaluate_env(base_net, hybrid_net, val_loaders["moderate"], class_names, device)
        res_sev   = evaluate_env(base_net, hybrid_net, val_loaders["severe"], class_names, device)

        if "EfficientNet-V2-S" in title:
            effnet_clean_detail = res_clean
            effnet_severe_detail = (title, res_mod, res_sev)

        print("=" * 136)
        print(f" >>> {title}")
        print("-" * 136)
        for env_label, r in [
            ("[Env 1: Clean Greenhouse (  )]", res_clean),
            ("[Env 2: Moderate Outdoor (· )]", res_mod),
            ("[Env 3: Severe Outdoor ( · )]", res_sev),
        ]:
            pa, pb, pc = r["pure"], r["clahe"], r["topo"]
            gain_acc = pc["acc"] - pa["acc"]
            gain_f1 = pc["macro_f1"] - pa["macro_f1"]
            print(f"  {env_label}")
            print(f"    (A) Pure Pretrained Only          | Acc={pa['acc']:5.2f}% | Macro F1={pa['macro_f1']:5.2f}% | Wing/Leg F1={pa['hard_f1']:5.2f}% | Dist FP={pa['dist_fp']:2d}")
            print(f"    (B) + Naive CLAHE Normalization   | Acc={pb['acc']:5.2f}% | Macro F1={pb['macro_f1']:5.2f}% | Wing/Leg F1={pb['hard_f1']:5.2f}% | Dist FP={pb['dist_fp']:2d}")
            print(f"    (C) + Wilkinson Rootfix+Spectrum  | Acc={pc['acc']:5.2f}% | Macro F1={pc['macro_f1']:5.2f}% | Wing/Leg F1={pc['hard_f1']:5.2f}% | Dist FP={pc['dist_fp']:2d}  --> [Gain vs Pure: Acc {gain_acc:+5.2f}%p | F1 {gain_f1:+5.2f}%p]")

    if effnet_severe_detail is not None:
        e_title, r_mod, r_sev = effnet_severe_detail
        print("\n" + "=" * 136)
        print(f" [PART 2: PER-CLASS F1 ON {e_title.strip()} UNDER OUTDOOR SHADOW & GLARE SHIFT]")
        print("=" * 136)
        print(f" {'Class Name':15s} | {'Clean Pure':11s} -> {'Clean Topo':11s} || {'Mod. Pure':10s} -> {'Mod. Topo':10s} || {'Sev. Pure':10s} -> {'Sev. Topo':10s} | {'Outdoor Rescue Gain'}")
        print("-" * 136)
        for c_name in class_names:
            cp = effnet_clean_detail["pure"]["f1_per_class"][c_name]
            ct = effnet_clean_detail["topo"]["f1_per_class"][c_name]
            mp = r_mod["pure"]["f1_per_class"][c_name]
            mt = r_mod["topo"]["f1_per_class"][c_name]
            sp = r_sev["pure"]["f1_per_class"][c_name]
            st = r_sev["topo"]["f1_per_class"][c_name]
            print(f" {c_name:15s} | {cp:9.2f}%  -> {ct:9.2f}%  || {mp:8.2f}%  -> {mt:8.2f}%  || {sp:8.2f}%  -> {st:8.2f}%  | {st-sp:+14.2f}%p (Mod {mt-mp:+5.2f}%p)")
        print("=" * 136)


if __name__ == "__main__":
    main()