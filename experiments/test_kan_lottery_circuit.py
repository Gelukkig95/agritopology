import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PHENOBENCH_EXP_DIR = ROOT_DIR / "experiments" / "phenobench"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PHENOBENCH_EXP_DIR))

import copy
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


def extract_real_14d_dataset(val_ds, max_samples=450):
    extractor = AgriTopoExtractor(work_size=48)
    feats, labels = [], []
    rng = np.random.default_rng(42)

    for idx in range(min(90, len(val_ds))):
        img_t, lbl_t = val_ds[idx]
        rgb_u8 = np.clip(img_t.numpy() * 255.0, 0, 255).astype(np.uint8)
        gt = lbl_t.numpy()

        #  0: Background/ ,  1:  (Crop) ,  2:  (Weed) 
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
                if len(feats) >= max_samples:
                    break
            if len(feats) >= max_samples:
                break
        if len(feats) >= max_samples:
            break

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    n_tr = int(len(y) * 0.7)
    train_idx = np.arange(n_tr)
    X_norm = extractor.fit_transform(X, train_idx)
    return (
        torch.from_numpy(X_norm[:n_tr]),
        torch.from_numpy(y[:n_tr]),
        torch.from_numpy(X_norm[n_tr:]),
        torch.from_numpy(y[n_tr:]),
    )


class DenseBlackBoxMLP(nn.Module):
    def __init__(self, in_dim=14, hid=48, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, num_classes)

    def forward(self, x, mask1=None, mask2=None):
        w1 = self.fc1.weight if mask1 is None else self.fc1.weight * mask1
        w2 = self.fc2.weight if mask2 is None else self.fc2.weight * mask2
        h = F.relu(F.linear(x, w1, self.fc1.bias))
        return F.linear(h, w2, self.fc2.bias)


class GlassBoxKANHead(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Decision Head:
     (Edge) 5   [x, x^2, log(1+|x|), sqrt(|x|), tanh(x)]  
         (Symbolic Formula) .
    """
    BASIS_NAMES = ["x", "x^2", "log(1+|x|)", "sqrt(|x|)", "tanh(x)"]

    def __init__(self, in_dim=14, num_classes=3):
        super().__init__()
        # (num_classes, in_dim, 5_basis)
        self.edge_weights = nn.Parameter(torch.randn(num_classes, in_dim, 5) * 0.05)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    @staticmethod
    def compute_basis(x):
        # x: (B, 14) -> (B, 14, 5)
        b1 = x
        b2 = (x * x).clamp(-4.0, 4.0)
        b3 = torch.log1p(torch.abs(x)) * torch.sign(x)
        b4 = torch.sqrt(torch.abs(x) + 1e-4) * torch.sign(x)
        b5 = torch.tanh(x)
        return torch.stack([b1, b2, b3, b4, b5], dim=-1)

    def forward(self, x, edge_mask=None):
        basis = self.compute_basis(x)  # (B, 14, 5)
        w = self.edge_weights if edge_mask is None else self.edge_weights * edge_mask
        #    p 1D  phi_{q,p}(x_p) 
        logits = torch.einsum("bpk,cpk->bc", basis, w) + self.bias
        return logits


def train_and_eval_mlp(model, X_tr, y_tr, X_val, y_val, mask1=None, mask2=None, epochs=120):
    opt = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(X_tr, mask1, mask2), y_tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(X_val, mask1, mask2).argmax(dim=1)
        return float((preds == y_val).float().mean().item()) * 100.0


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    X_tr, y_tr, X_val, y_val = extract_real_14d_dataset(val_ds, max_samples=450)

    # 1.  Dense MLP ( 100% )
    base_mlp = DenseBlackBoxMLP()
    theta_0 = copy.deepcopy(base_mlp.state_dict())
    total_mlp_params = base_mlp.fc1.weight.numel() + base_mlp.fc2.weight.numel()
    acc_dense = train_and_eval_mlp(base_mlp, X_tr, y_tr, X_val, y_val)

    # 2. Iterative Magnitude Pruning (IMP)  10%   (90% ) 
    m1 = torch.ones_like(base_mlp.fc1.weight)
    m2 = torch.ones_like(base_mlp.fc2.weight)
    for step in range(5):
        keep_ratio = (1.0 - 0.90) ** ((step + 1) / 5.0)
        all_w = torch.cat([(base_mlp.fc1.weight * m1).abs().flatten(), (base_mlp.fc2.weight * m2).abs().flatten()])
        thr = torch.quantile(all_w, 1.0 - keep_ratio)
        m1 = (base_mlp.fc1.weight.abs() >= thr).float()
        m2 = (base_mlp.fc2.weight.abs() >= thr).float()
        # Rewind to original initialization theta_0 (Lottery Ticket Hypothesis!)
        base_mlp.load_state_dict(copy.deepcopy(theta_0))
        acc_lt = train_and_eval_mlp(base_mlp, X_tr, y_tr, X_val, y_val, m1, m2, epochs=120)

    active_lt_params = int(m1.sum().item() + m2.sum().item())

    # 3. :  10%  '  '   (Random Re-init)
    rand_mlp = DenseBlackBoxMLP()
    acc_rand_reinit = train_and_eval_mlp(rand_mlp, X_tr, y_tr, X_val, y_val, m1, m2, epochs=120)

    # 4.  KAN   + 90%   (   )
    kan = GlassBoxKANHead()
    opt_k = torch.optim.AdamW(kan.parameters(), lr=0.02, weight_decay=1e-4)
    for _ in range(150):
        opt_k.zero_grad()
        logits = kan(X_tr)
        l1_reg = 0.008 * kan.edge_weights.abs().mean()
        loss = F.cross_entropy(logits, y_tr) + l1_reg
        loss.backward()
        opt_k.step()

    with torch.no_grad():
        w_abs = kan.edge_weights.abs()
        thr_k = torch.quantile(w_abs.flatten(), 0.90)  # 90%  ( 10%  !)
        k_mask = (w_abs >= thr_k).float()

    for _ in range(80):
        opt_k.zero_grad()
        loss = F.cross_entropy(kan(X_tr, k_mask), y_tr)
        loss.backward()
        opt_k.step()

    with torch.no_grad():
        acc_kan = float((kan(X_val, k_mask).argmax(dim=1) == y_val).float().mean().item()) * 100.0
        active_kan_params = int(k_mask.sum().item())
        total_kan_params = kan.edge_weights.numel()

    print("=" * 94)
    print(" [ : 14D     MLP vs  (90% ) vs  KAN]")
    print("=" * 94)
    print(f" 1.   ReLU MLP ( 100%  : {total_mlp_params:3d}) ->  = {acc_dense:5.2f}%")
    print(f" 2.      (  10%  : {active_lt_params:3d}) ->  = {acc_rand_reinit:5.2f}% ( /)")
    print(f" 3.    (IMP)   (  10%  : {active_lt_params:3d}) ->  = {acc_lt:5.2f}% (90%   !)")
    print(f" 4.  KAN       (  10%  : {active_kan_params:3d}/{total_kan_params}) ->  = {acc_kan:5.2f}%")
    print("-" * 94)
    print(" [ KAN    (Causal Circuit) &   ]")
    print("-" * 94)

    cls_names = ["Class 0 (Background/Noise)", "Class 1 (Main Crop)", "Class 2 (Target Weed)"]
    w_final = (kan.edge_weights * k_mask).detach().cpu().numpy()
    for c in range(3):
        terms = []
        for p in range(14):
            for k_idx in range(5):
                coef = w_final[c, p, k_idx]
                if abs(coef) > 1e-4:
                    fn_str = GlassBoxKANHead.BASIS_NAMES[k_idx].replace("x", FEATURE_NAMES[p])
                    terms.append((abs(coef), f"{coef:+.2f}*{fn_str}"))
        terms.sort(key=lambda x: x[0], reverse=True)
        top_formula = " ".join([t[1] for t in terms[:4]])
        print(f" • {cls_names[c]:<26}: Score = {top_formula}")
    print("=" * 94)


if __name__ == "__main__":
    main()
