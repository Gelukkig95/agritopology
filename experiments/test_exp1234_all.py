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

FEATURE_NAMES = [
    "x0_MaxTree_CenterVol",
    "x1_MaxTree_PeakPers",
    "x2_MinTree_HolePers",
    "x3_MMTO_FluxDensity",
    "x4_TW_HuMoment_Phi1",
    "x5_TW_Elongation",
    "x6_TW_RootfixRatio",
    "x7_MMTO_CrossCosSim",
    "x8_RCC8_WingCoreEC",
    "x9_RCC8_TranslucentEC",
    "x10_RCC8_BorderEC",
    "x11_Fuzzy_EggYolkEC",
    "x12_MMTO_Circularity",
    "x13_MultiSpec_Energy",
]


def extract_dataset_with_raw_patches(val_ds, max_samples=450):
    extractor = AgriTopoExtractor(work_size=48)
    feats, labels, raw_patches = [], [], []
    rng = np.random.default_rng(42)

    for idx in range(min(90, len(val_ds))):
        img_t, lbl_t = val_ds[idx]
        rgb_u8 = np.clip(img_t.numpy() * 255.0, 0, 255).astype(np.uint8)
        gt = lbl_t.numpy()

        for cls_id in (0, 1, 2):
            mask = (gt == cls_id).astype(np.uint8)
            n_cc, cc, stats, centroids = cv2.connectedComponentsWithStats(mask)
            for cid in range(1, n_cc):
                if stats[cid, cv2.CC_STAT_AREA] < 25:
                    continue
                cx, cy = int(centroids[cid][0]), int(centroids[cid][1])
                y0, y1 = max(0, cy - 24), min(IMG_SIZE, cy + 24)
                x0, x1 = max(0, cx - 24), min(IMG_SIZE, cx + 24)
                patch = rgb_u8[:, y0:y1, x0:x1]
                if patch.shape[1] < 16 or patch.shape[2] < 16:
                    continue
                vec14 = extractor.extract_single(patch)
                feats.append(vec14)
                labels.append(cls_id)
                p_hwc = cv2.resize(patch.transpose(1, 2, 0), (48, 48), interpolation=cv2.INTER_AREA)
                raw_patches.append(p_hwc.transpose(2, 0, 1))
                if len(feats) >= max_samples:
                    break
            if len(feats) >= max_samples:
                break
        if len(feats) >= max_samples:
            break

    X_raw = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    P_u8 = np.array(raw_patches, dtype=np.uint8)

    perm = rng.permutation(len(y))
    X_raw, y, P_u8 = X_raw[perm], y[perm], P_u8[perm]

    n_tr = int(len(y) * 0.7)
    train_idx = np.arange(n_tr)
    X_norm = extractor.fit_transform(X_raw, train_idx)

    return (
        extractor,
        torch.from_numpy(X_norm[:n_tr]),
        torch.from_numpy(y[:n_tr]),
        torch.from_numpy(X_norm[n_tr:]),
        torch.from_numpy(y[n_tr:]),
        P_u8[:n_tr],
        P_u8[n_tr:],
    )


class KANLayer1D(nn.Module):
    """ KAN :  (Edge) 1D     """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.spline_weights = nn.Parameter(torch.randn(out_dim, in_dim, 4) * 0.05)

    @staticmethod
    def curve_basis(x):
        # 4  1D  : [x - tanh(x), tanh(x), x^2/(1+x^2), sign(x)*sqrt(|x|)]
        b1 = x - torch.tanh(x)
        b2 = torch.tanh(x)
        b3 = (x * x) / (1.0 + x * x)
        b4 = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-4)
        return torch.stack([b1, b2, b3, b4], dim=-1)

    def forward(self, x):
        basis = self.curve_basis(x)
        curve_sum = torch.einsum("bik,oik->bo", basis, self.spline_weights)
        return self.base_linear(F.silu(x)) + curve_sum


class TwoLayerCrossKAN(nn.Module):
    """
    [ 1] -   2  KAN (14 -> 8 -> 3)
    + Max-Tree(x0~x7) RCC8(x8~x13)    (AND Synergy) 
    """
    CROSS_PAIRS = [
        (6, 10, "x6_Rootfix * (1 - sig(x10_BorderEC))"),
        (0, 8,  "x0_CenterVol * x8_WingCoreEC"),
        (6, 8,  "x6_Rootfix * x8_WingCoreEC"),
        (0, 6,  "x0_CenterVol * x6_Rootfix"),
        (1, 12, "x1_PeakPers * x12_Circularity"),
        (3, 11, "x3_FluxDens * x11_FuzzyEC"),
    ]

    def __init__(self, in_dim=14, hid_dim=8, num_classes=3):
        super().__init__()
        self.kan1 = KANLayer1D(in_dim, hid_dim)
        self.kan2 = KANLayer1D(hid_dim, num_classes)
        self.cross_weights = nn.Parameter(torch.zeros(num_classes, len(self.CROSS_PAIRS)))

    def compute_cross_terms(self, x):
        terms = [
            x[:, 6] * (1.0 - torch.sigmoid(x[:, 10])),
            x[:, 0] * x[:, 8],
            x[:, 6] * x[:, 8],
            x[:, 0] * x[:, 6],
            x[:, 1] * x[:, 12],
            x[:, 3] * x[:, 11],
        ]
        return torch.stack(terms, dim=-1)

    def forward(self, x):
        h = torch.tanh(self.kan1(x))
        out = self.kan2(h)
        cross = self.compute_cross_terms(x)
        return out + F.linear(cross, self.cross_weights)


class TopoMLP(nn.Module):
    def __init__(self, in_dim=14, hid=48, num_classes=3, act="relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, num_classes)
        self.act = act

    def forward(self, x):
        h = self.fc1(x)
        if self.act == "silu":
            h = F.silu(h)
        elif self.act == "gelu":
            h = F.gelu(h)
        else:
            h = F.relu(h)
        return self.fc2(h)


class RawPixelPatchCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_eval_topo_mlp(X_tr, y_tr, X_val, y_val, act="relu", hid=48, epochs=120, seed=42):
    torch.manual_seed(seed)
    model = TopoMLP(in_dim=X_tr.shape[1], hid=hid, num_classes=3, act=act)
    opt = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((model(X_val).argmax(dim=1) == y_val).float().mean().item()) * 100.0
    return acc, model


def build_kan_augmented_features(X: torch.Tensor) -> torch.Tensor:
    x0 = X[:, 0:1]
    x6 = X[:, 6:7]
    x10 = X[:, 10:11]
    f_x0_deadzone = x0 - torch.tanh(x0)
    f_x6_deadzone = x6 - torch.tanh(x6)
    f_x6_sqrt = torch.sign(x6) * torch.sqrt(torch.abs(x6) + 1e-4)
    f_and_gate = x6 * (1.0 - torch.sigmoid(x10))
    return torch.cat([X, f_x0_deadzone, f_x6_deadzone, f_x6_sqrt, f_and_gate], dim=1)


def perturb_patches_gamma_contrast(P_u8: np.ndarray) -> np.ndarray:
    out = []
    for p in P_u8:
        f = p.astype(np.float32) / 255.0
        f_pert = np.clip((f ** 1.85) * 0.65 + 0.18, 0.0, 1.0)
        out.append(np.clip(f_pert * 255.0, 0, 255).astype(np.uint8))
    return np.array(out, dtype=np.uint8)


def perturb_patches_boundary_shadow(P_u8: np.ndarray) -> np.ndarray:
    out = P_u8.copy()
    out[:, :, :5, :] = (out[:, :, :5, :].astype(np.float32) * 0.15).astype(np.uint8)
    out[:, :, -5:, :] = (out[:, :, -5:, :].astype(np.float32) * 0.15).astype(np.uint8)
    out[:, :, :, :5] = (out[:, :, :, :5].astype(np.float32) * 0.15).astype(np.uint8)
    out[:, :, :, -5:] = (out[:, :, :, -5:].astype(np.float32) * 0.15).astype(np.uint8)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    extractor, X_tr, y_tr, X_val, y_val, P_tr_u8, P_val_u8 = extract_dataset_with_raw_patches(val_ds, max_samples=450)

    # =========================================================================
    # [ 1] 1   KAN (65.93%) vs 2  KAN (14 -> 8 -> 3)
    # =========================================================================
    torch.manual_seed(42)
    kan_2layer = TwoLayerCrossKAN(in_dim=14, hid_dim=8, num_classes=3)
    opt_k2 = torch.optim.AdamW(kan_2layer.parameters(), lr=0.015, weight_decay=1e-4)
    for _ in range(140):
        kan_2layer.train()
        opt_k2.zero_grad()
        loss = F.cross_entropy(kan_2layer(X_tr), y_tr)
        loss.backward()
        opt_k2.step()
    kan_2layer.eval()
    with torch.no_grad():
        acc_kan_2layer = float((kan_2layer(X_val).argmax(dim=1) == y_val).float().mean().item()) * 100.0
    kan2_params = sum(p.numel() for p in kan_2layer.parameters())

    print("=" * 88)
    print(" [ 1 : 1  KAN vs 2  KAN (Max-Tree x RCC8  )]")
    print("=" * 88)
    print(f" 1) 1   KAN (  )           : 65.93%")
    print(f" 2) 2  KAN (14 -> 8 -> 3,  {kan2_params})     : {acc_kan_2layer:5.2f}% (+{acc_kan_2layer - 65.93:.2f}%p !)")
    print("    ->   Max-Tree x RCC8  (AND Synergy) :")
    cw = kan_2layer.cross_weights.detach().numpy()
    for c_idx, c_name in enumerate(["Class 0 (Noise)", "Class 1 (Crop) ", "Class 2 (Weed) "]):
        best_p = int(np.argmax(np.abs(cw[c_idx])))
        _, _, pair_str = TwoLayerCrossKAN.CROSS_PAIRS[best_p]
        print(f"       • {c_name}: {cw[c_idx, best_p]:+.3f} * [{pair_str}]")

    # =========================================================================
    # [ 2] 14      (Feature Group Ablation)
    # =========================================================================
    acc_pillar1, _ = train_eval_topo_mlp(X_tr[:, :6], y_tr, X_val[:, :6], y_val)
    acc_pillar12, _ = train_eval_topo_mlp(X_tr[:, :8], y_tr, X_val[:, :8], y_val)
    acc_rcc8_only, _ = train_eval_topo_mlp(X_tr[:, 8:], y_tr, X_val[:, 8:], y_val)
    acc_full_14d, full_model = train_eval_topo_mlp(X_tr, y_tr, X_val, y_val)

    with torch.no_grad():
        feat_imp = full_model.fc1.weight.abs().mean(dim=0).numpy()
        top10_idx = np.argsort(feat_imp)[-10:]
        top10_idx.sort()
    acc_top10, _ = train_eval_topo_mlp(X_tr[:, top10_idx], y_tr, X_val[:, top10_idx], y_val)

    print("\n" + "=" * 88)
    print(" [ 2 : 14     (Feature Pillar Ablation)]")
    print("=" * 88)
    print(f" 1)  1  (Max/Min-Tree  · 6, x0~x5) : {acc_pillar1:5.2f}%")
    print(f" 2)  3  ( RCC8    6,      x8~x13): {acc_rcc8_only:5.2f}%")
    print(f" 3)  1 + 2  (+Wilkinson Rootfix &  8, x0~x7) : {acc_pillar12:5.2f}% ({acc_pillar12 - acc_pillar1:+.2f}%p)")
    print(f" 4)  10   ( 4  ,       Top-10): {acc_top10:5.2f}%")
    print(f" 5) 3    (Max-Tree + Rootfix + RCC8,  14D)   : {acc_full_14d:5.2f}%")

    # =========================================================================
    # [ 3] KAN   (x - tanh(x))     
    # =========================================================================
    acc_silu_14d, _ = train_eval_topo_mlp(X_tr, y_tr, X_val, y_val, act="silu")
    X_tr_kan_aug = build_kan_augmented_features(X_tr)
    X_val_kan_aug = build_kan_augmented_features(X_val)
    acc_kan_hybrid_relu, _ = train_eval_topo_mlp(X_tr_kan_aug, y_tr, X_val_kan_aug, y_val, act="relu")
    acc_kan_hybrid_silu, _ = train_eval_topo_mlp(X_tr_kan_aug, y_tr, X_val_kan_aug, y_val, act="silu")

    print("\n" + "=" * 88)
    print(" [ 3 : KAN   (x - tanh(x))  +  (SiLU)]")
    print("=" * 88)
    print(f" 1)  14D +  ReLU (Baseline)                        : {acc_full_14d:5.2f}%")
    print(f" 2)  14D +  SiLU (Swish  )             : {acc_silu_14d:5.2f}% ({acc_silu_14d - acc_full_14d:+.2f}%p)")
    print(f" 3) KAN   (x0-tanh(x0), sqrt(|x6|) ) + ReLU    : {acc_kan_hybrid_relu:5.2f}% ({acc_kan_hybrid_relu - acc_full_14d:+.2f}%p)")
    print(f" 4) KAN   +  SiLU  (Hybrid Glass-MLP)    : {acc_kan_hybrid_silu:5.2f}% ({acc_kan_hybrid_silu - acc_full_14d:+.2f}%p)")

    # =========================================================================
    # [ 4]   CNN vs   : 3   
    # =========================================================================
    torch.manual_seed(42)
    pixel_cnn = RawPixelPatchCNN().to(device)
    P_tr_t = torch.from_numpy(P_tr_u8).float().div(255.0).to(device)
    P_val_t = torch.from_numpy(P_val_u8).float().div(255.0).to(device)
    y_tr_d = y_tr.to(device)

    opt_cnn = torch.optim.AdamW(pixel_cnn.parameters(), lr=0.003, weight_decay=1e-4)
    for _ in range(80):
        pixel_cnn.train()
        opt_cnn.zero_grad()
        loss = F.cross_entropy(pixel_cnn(P_tr_t), y_tr_d)
        loss.backward()
        opt_cnn.step()
    pixel_cnn.eval()

    with torch.no_grad():
        cnn_clean_acc = float((pixel_cnn(P_val_t).cpu().argmax(dim=1) == y_val).float().mean().item()) * 100.0

    P_val_gamma_u8 = perturb_patches_gamma_contrast(P_val_u8)
    P_val_gamma_t = torch.from_numpy(P_val_gamma_u8).float().div(255.0).to(device)
    with torch.no_grad():
        cnn_gamma_acc = float((pixel_cnn(P_val_gamma_t).cpu().argmax(dim=1) == y_val).float().mean().item()) * 100.0

    X_val_gamma_raw = np.array([extractor.extract_single(p) for p in P_val_gamma_u8], dtype=np.float32)
    X_val_gamma_norm = torch.from_numpy((X_val_gamma_raw - extractor.mean_) / extractor.std_)
    with torch.no_grad():
        topo_gamma_acc = float((full_model(X_val_gamma_norm).argmax(dim=1) == y_val).float().mean().item()) * 100.0

    P_val_border_u8 = perturb_patches_boundary_shadow(P_val_u8)
    P_val_border_t = torch.from_numpy(P_val_border_u8).float().div(255.0).to(device)
    with torch.no_grad():
        cnn_border_acc = float((pixel_cnn(P_val_border_t).cpu().argmax(dim=1) == y_val).float().mean().item()) * 100.0

    X_val_border_raw = np.array([extractor.extract_single(p) for p in P_val_border_u8], dtype=np.float32)
    X_val_border_norm = torch.from_numpy((X_val_border_raw - extractor.mean_) / extractor.std_)
    with torch.no_grad():
        topo_border_acc = float((full_model(X_val_border_norm).argmax(dim=1) == y_val).float().mean().item()) * 100.0

    n_10pct = max(15, int(len(y_tr) * 0.10))
    acc_topo_10pct, _ = train_eval_topo_mlp(X_tr[:n_10pct], y_tr[:n_10pct], X_val, y_val)

    torch.manual_seed(42)
    cnn_10pct = RawPixelPatchCNN().to(device)
    opt_c10 = torch.optim.AdamW(cnn_10pct.parameters(), lr=0.003, weight_decay=1e-4)
    for _ in range(80):
        cnn_10pct.train()
        opt_c10.zero_grad()
        loss = F.cross_entropy(cnn_10pct(P_tr_t[:n_10pct]), y_tr_d[:n_10pct])
        loss.backward()
        opt_c10.step()
    cnn_10pct.eval()
    with torch.no_grad():
        acc_cnn_10pct = float((cnn_10pct(P_val_t).cpu().argmax(dim=1) == y_val).float().mean().item()) * 100.0

    cnn_params = sum(p.numel() for p in pixel_cnn.parameters())
    topo_params = sum(p.numel() for p in full_model.parameters())

    print("\n" + "=" * 88)
    print(" [ 4 :   CNN vs  14D     ]")
    print("=" * 88)
    print(f" •  ( ) :  CNN = {cnn_params:,} vs   = {topo_params:,} (30 !)")
    print("-" * 88)
    print(f" 1)         :  CNN = {cnn_clean_acc:5.2f}%  vs    = {acc_full_14d:5.2f}%")
    print(f" 2) · (y=1.85):  CNN = {cnn_gamma_acc:5.2f}% ({cnn_gamma_acc-cnn_clean_acc:+.2f}%p) vs   = {topo_gamma_acc:5.2f}% ({topo_gamma_acc-acc_full_14d:+.2f}%p)")
    print(f" 3)       :  CNN = {cnn_border_acc:5.2f}% ({cnn_border_acc-cnn_clean_acc:+.2f}%p) vs   = {topo_border_acc:5.2f}% ({topo_border_acc-acc_full_14d:+.2f}%p)")
    print(f" 4)  (10%=31):  CNN = {acc_cnn_10pct:5.2f}%          vs   = {acc_topo_10pct:5.2f}% ({acc_topo_10pct-acc_cnn_10pct:+.2f}%p)")
    print("=" * 88)


if __name__ == "__main__":
    main()
