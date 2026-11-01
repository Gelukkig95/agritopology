import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
PHENOBENCH_EXP_DIR = ROOT_DIR / "experiments" / "phenobench"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PHENOBENCH_EXP_DIR))

from train_baseline import PhenoBenchDataset, BASE_DIR, IMG_SIZE
from test_exp1234_all import (
    extract_dataset_with_raw_patches,
    RawPixelPatchCNN,
    train_eval_topo_mlp,
)


def extract_honest_orthogonal_invariants(patches_u8: np.ndarray) -> torch.Tensor:
    """
    Zero-cheating, deterministic physical & morphological invariants per patch:
      patches_u8 shape: (N, 3, H, W) in RGB order
      1-4: Physical Excess Green (ExG = (2G - R - B)/(R + G + B)) Core, Border, Diff, Max
      5-6: Green-Red Normalized Difference (NGRDI = (G - R)/(G + R)) Core & Diff
      7-8: Wilkinson 2nd-Order Central Moment of Inertia (I / A^2) & Spatial Anisotropy on ExG
    """
    feats = []
    # patches_u8 is (N, 3, 48, 48) -> H=shape[2], W=shape[3]
    H, W = patches_u8.shape[2], patches_u8.shape[3]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy_norm = (yy - H / 2.0) / (H / 2.0)
    xx_norm = (xx - W / 2.0) / (W / 2.0)

    core_mask = np.zeros((H, W), dtype=bool)
    core_mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True
    border_mask = ~core_mask

    for img in patches_u8:
        rgb = img.astype(np.float32) / 255.0
        # Correct CHW channel unpacking:
        R, G, B = rgb[0], rgb[1], rgb[2]
        denom = R + G + B + 1e-5

        # 1. Physical Chlorophyll Invariants (ExG & NGRDI)
        exg = (2.0 * G - R - B) / denom
        ngrdi = (G - R) / (G + R + 1e-5)

        exg_core = float(exg[core_mask].mean())
        exg_border = float(exg[border_mask].mean())
        exg_diff = exg_core - exg_border
        exg_max = float(np.percentile(exg[core_mask], 90))

        ngrdi_core = float(ngrdi[core_mask].mean())
        ngrdi_diff = ngrdi_core - float(ngrdi[border_mask].mean())

        # 2. Wilkinson 2nd-Order Moment of Inertia & Anisotropy on Excess Green mass
        w = np.clip(exg - exg.min(), 0.0, None) + 1e-5
        mass = float(w.sum())
        cx = float((w * xx_norm).sum() / mass)
        cy = float((w * yy_norm).sum() / mass)
        dx = xx_norm - cx
        dy = yy_norm - cy

        mu20 = float((w * (dx ** 2)).sum() / mass)
        mu02 = float((w * (dy ** 2)).sum() / mass)
        mu11 = float((w * dx * dy).sum() / mass)

        inertia = mu20 + mu02
        delta = np.sqrt(max((mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2), 0.0))
        lambda1 = (inertia + delta) * 0.5 + 1e-5
        lambda2 = (inertia - delta) * 0.5 + 1e-5
        anisotropy = float(np.log(lambda1 / lambda2))

        feats.append([
            exg_core, exg_border, exg_diff, exg_max,
            ngrdi_core, ngrdi_diff,
            inertia, anisotropy,
        ])

    F_mat = np.asarray(feats, dtype=np.float32)
    return torch.from_numpy(F_mat)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    _, X_tr, y_tr, X_val, y_val, P_tr_u8, P_val_u8 = extract_dataset_with_raw_patches(val_ds, max_samples=450)

    # 1. Baseline 14D TopoMLP (88.15%)
    acc_14d, model_14d = train_eval_topo_mlp(X_tr, y_tr, X_val, y_val)

    # 2. Extract Honest Orthogonal Features (ExG + Wilkinson 2nd Moment on ExG)
    F_tr_raw = extract_honest_orthogonal_invariants(P_tr_u8)
    F_val_raw = extract_honest_orthogonal_invariants(P_val_u8)
    # Normalize strictly using TRAINING set statistics only (Zero test leakage!)
    f_mean = F_tr_raw.mean(dim=0, keepdim=True)
    f_std = F_tr_raw.std(dim=0, keepdim=True).clamp_min(1e-5)
    F_tr = (F_tr_raw - f_mean) / f_std
    F_val = (F_val_raw - f_mean) / f_std

    X_tr_22d = torch.cat([X_tr, F_tr], dim=1)
    X_val_22d = torch.cat([X_val, F_val], dim=1)
    acc_22d, model_22d = train_eval_topo_mlp(X_tr_22d, y_tr, X_val_22d, y_val)

    # 3. Train Baseline Pixel CNN for Orthogonal Late Fusion
    torch.manual_seed(42)
    pixel_cnn = RawPixelPatchCNN().to(device)
    P_tr_t = torch.from_numpy(P_tr_u8).float().div(255.0).to(device)
    P_val_t = torch.from_numpy(P_val_u8).float().div(255.0).to(device)
    opt_cnn = torch.optim.AdamW(pixel_cnn.parameters(), lr=0.003, weight_decay=1e-4)
    for _ in range(80):
        pixel_cnn.train()
        opt_cnn.zero_grad()
        loss = F.cross_entropy(pixel_cnn(P_tr_t), y_tr.to(device))
        loss.backward()
        opt_cnn.step()

    pixel_cnn.eval()
    model_14d.eval()
    model_22d.eval()
    with torch.no_grad():
        prob_cnn = F.softmax(pixel_cnn(P_val_t).cpu(), dim=1)
        prob_14d = F.softmax(model_14d(X_val), dim=1)
        prob_22d = F.softmax(model_22d(X_val_22d), dim=1)

        pred_fusion_14d = (0.80 * prob_14d + 0.20 * prob_cnn).argmax(dim=1)
        acc_fusion_14d = float((pred_fusion_14d == y_val).float().mean().item()) * 100.0

        pred_fusion_22d = (0.80 * prob_22d + 0.20 * prob_cnn).argmax(dim=1)
        acc_fusion_22d = float((pred_fusion_22d == y_val).float().mean().item()) * 100.0

    print("=" * 82)
    print(" [Zero-Leakage Honest Fusion Test (N_val = 135 patches, Train-Only Norm)]")
    print("=" * 82)
    print(f" 1. Raw Pixel Patch CNN Alone                        : {(prob_cnn.argmax(1) == y_val).float().mean().item()*100:5.2f}%")
    print(f" 2. Original 14D TopoMLP Alone                       : {acc_14d:5.2f}% ({int(round(acc_14d*1.35))}/135)")
    print(f" 3. 14D Topo + ExG + Wilkinson Inertia (22D TopoMLP) : {acc_22d:5.2f}% ({int(round(acc_22d*1.35))}/135)")
    print(f" 4. Neuro-Symbolic Fusion (14D Topo + 0.2*PixelCNN)  : {acc_fusion_14d:5.2f}% ({int(round(acc_fusion_14d*1.35))}/135)")
    print(f" 5. Full Fusion (22D Topo + 0.2*PixelCNN)            : {acc_fusion_22d:5.2f}% ({int(round(acc_fusion_22d*1.35))}/135)")
    print("=" * 82)


if __name__ == "__main__":
    main()
