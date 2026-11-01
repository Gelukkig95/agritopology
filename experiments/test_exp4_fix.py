import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PHENOBENCH_EXP_DIR = ROOT_DIR / "experiments" / "phenobench"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PHENOBENCH_EXP_DIR))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_baseline import PhenoBenchDataset, BASE_DIR, IMG_SIZE
from torch_morph_rcc8.insect_topo import AgriTopoExtractor
from test_exp1234_all import (
    extract_dataset_with_raw_patches,
    train_eval_topo_mlp,
    perturb_patches_gamma_contrast,
    perturb_patches_boundary_shadow,
)


def normalize_patch_illumination_and_border(patch_chw: np.ndarray) -> np.ndarray:
    """
    [   ·  ]
    1.  5px  (  )   Median   
    2.  5%~95%   (Percentile Contrast Stretch) /   
    """
    p = patch_chw.astype(np.float32).copy()
    _, H, W = p.shape
    inner = p[:, 5 : H - 5, 5 : W - 5]

    # 1)        (Reflect)   
    border_mean = (p[:, :5, :].mean() + p[:, -5:, :].mean()) * 0.5
    inner_mean = inner.mean()
    if border_mean < inner_mean * 0.45:
        p_hwc = cv2.copyMakeBorder(
            inner.transpose(1, 2, 0), 5, 5, 5, 5, borderType=cv2.BORDER_REFLECT_101
        )
        p = p_hwc.transpose(2, 0, 1)

    # 2) /    (5th ~ 95th percentile -> [10, 245])
    out = np.zeros_like(p)
    for c in range(3):
        ch = p[c]
        lo, hi = np.percentile(ch, 5), np.percentile(ch, 95)
        if hi - lo > 1e-3:
            norm_c = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
            out[c] = norm_c * 235.0 + 10.0
        else:
            out[c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_invariant_features(extractor: AgriTopoExtractor, patches_u8: np.ndarray) -> np.ndarray:
    feats = []
    for p in patches_u8:
        p_clean = normalize_patch_illumination_and_border(p)
        feats.append(extractor.extract_single(p_clean))
    return np.array(feats, dtype=np.float32)


def main():
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    extractor, _, y_tr, _, y_val, P_tr_u8, P_val_u8 = extract_dataset_with_raw_patches(val_ds, max_samples=450)

    # ·    14D   
    X_tr_raw = extract_invariant_features(extractor, P_tr_u8)
    X_val_raw = extract_invariant_features(extractor, P_val_u8)
    mean = X_tr_raw.mean(axis=0, keepdims=True)
    std = X_tr_raw.std(axis=0, keepdims=True) + 1e-6

    X_tr = torch.from_numpy((X_tr_raw - mean) / std)
    X_val = torch.from_numpy((X_val_raw - mean) / std)

    acc_clean, model = train_eval_topo_mlp(X_tr, y_tr, X_val, y_val, act="silu")

    # 1) · (y=1.85) 
    P_val_gamma = perturb_patches_gamma_contrast(P_val_u8)
    X_gamma_raw = extract_invariant_features(extractor, P_val_gamma)
    X_gamma = torch.from_numpy((X_gamma_raw - mean) / std)
    with torch.no_grad():
        acc_gamma = float((model(X_gamma).argmax(dim=1) == y_val).float().mean().item()) * 100.0

    # 2)    
    P_val_border = perturb_patches_boundary_shadow(P_val_u8)
    X_border_raw = extract_invariant_features(extractor, P_val_border)
    X_border = torch.from_numpy((X_border_raw - mean) / std)
    with torch.no_grad():
        acc_border = float((model(X_border).argmax(dim=1) == y_val).float().mean().item()) * 100.0

    print("=" * 88)
    print(" [ 4     vs    (14D  )]")
    print("=" * 88)
    print(f" 1)           : 88.15% -> {acc_clean:5.2f}%")
    print(f" 2) ·  (y=1.85) : 38.52% -> {acc_gamma:5.2f}% (+{acc_gamma - 38.52:.2f}%p  !)")
    print(f" 3)     : 15.56% -> {acc_border:5.2f}% (+{acc_border - 15.56:.2f}%p  !)")
    print("=" * 88)


if __name__ == "__main__":
    main()
