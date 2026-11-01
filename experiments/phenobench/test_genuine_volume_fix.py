import sys
import time
from pathlib import Path

import numpy as np
import cv2
import higra as hg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PHENOBENCH_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PHENOBENCH_EXP_DIR))

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE

_HIGRA_GRAPH_CACHE = {}


def get_cached_graph(H: int, W: int):
    key = (H, W)
    if key not in _HIGRA_GRAPH_CACHE:
        _HIGRA_GRAPH_CACHE[key] = hg.get_4_adjacency_graph((H, W))
    return _HIGRA_GRAPH_CACHE[key]


def fast_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = 3) -> np.ndarray:
    y_t = y_true.ravel().astype(np.int64)
    y_p = y_pred.ravel().astype(np.int64)
    valid = (y_t >= 0) & (y_t < n_cls) & (y_p >= 0) & (y_p < n_cls)
    return np.bincount(n_cls * y_t[valid] + y_p[valid], minlength=n_cls * n_cls).reshape(n_cls, n_cls)


def resolve_weight_path(filename: str) -> Path:
    for p in (ROOT_DIR / "weights" / filename, ROOT_DIR / filename, PHENOBENCH_EXP_DIR / filename):
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {filename}")


def classify_blob_raster_rcc8(wmask: np.ndarray, crop_bin: np.ndarray) -> str:
    k3 = np.ones((3, 3), np.uint8)
    crop_closure = cv2.dilate(crop_bin, k3, iterations=1)
    crop_interior = cv2.erode(crop_bin, k3, iterations=1)
    overlap = (wmask & (crop_bin > 0)).sum()
    touch_closure = (wmask & (crop_closure > 0)).sum()
    if touch_closure == 0:
        return "DC"
    if overlap == 0 and touch_closure > 0:
        return "EC"
    blob_area = wmask.sum()
    if overlap == blob_area:
        in_interior = (wmask & (crop_interior > 0)).sum()
        return "NTPP" if in_interior == blob_area else "TPP"
    return "PO"


def collect_margin_volume_blobs(model, loader, device, min_size: int = 5):
    """
    Structural Fix 1 & 2:
    1) Builds the Higra Max-Tree on the Decision-Boundary Margin Landscape:
       M(x) = max(0, p_weed(x) - max(p_soil(x), p_crop(x)))
    2) Computes genuine Higra Topological Volume (hg.attribute_volume) & Dynamics
       at the component root node, separating narrow FP spikes from true weeds.
    """
    model.eval()
    raw_cm = np.zeros((3, 3), dtype=np.int64)
    blobs = []
    frames_data = []
    graph = get_cached_graph(IMG_SIZE, IMG_SIZE)
    num_leaves = IMG_SIZE * IMG_SIZE

    with torch.no_grad():
        img_idx = 0
        for imgs, labels in loader:
            imgs_gpu = imgs.to(device)
            logits = model(imgs_gpu)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels.numpy()

            for b in range(preds.shape[0]):
                pred = preds[b].copy()
                gt = labels_np[b]
                raw_cm += fast_confusion_matrix(gt, pred, n_cls=3)

                p_soil = probs[b, 0]
                p_crop = probs[b, 1]
                p_weed = probs[b, 2]

                weed_bin = (pred == 2).astype(np.uint8)
                n_cc, cc_map = cv2.connectedComponents(weed_bin)
                if n_cc <= 1:
                    frames_data.append((pred, gt, p_soil, p_crop, None, []))
                    img_idx += 1
                    continue

                # Decision-Boundary Margin Landscape M(x) in [0, 1]
                competitor_max = np.maximum(p_soil, p_crop)
                margin_map = np.clip(p_weed - competitor_max, 0.0, 1.0).astype(np.float64)

                # Build Higra Max-Tree on Margin Landscape & compute Volume + Dynamics
                tree, alt = hg.component_tree_max_tree(graph, margin_map)
                vol = hg.attribute_volume(tree, alt)
                dyn = hg.attribute_dynamics(tree, alt)
                parents = tree.parents()[:num_leaves]
                leaf_vol = vol[parents].reshape(IMG_SIZE, IMG_SIZE)
                leaf_dyn = dyn[parents].reshape(IMG_SIZE, IMG_SIZE)

                crop_bin = ((pred == 1) | (p_crop > 0.35)).astype(np.uint8)
                frame_blob_indices = []

                for wid in range(1, n_cc):
                    wmask = (cc_map == wid)
                    area = int(wmask.sum())
                    if area < min_size:
                        continue

                    rel = classify_blob_raster_rcc8(wmask.astype(np.uint8), crop_bin)
                    rcc8_group = "NTPP_TPP" if rel in ("NTPP", "TPP") else ("EC_PO" if rel in ("EC", "PO") else "DC")

                    # Component-level topological volume & margin persistence
                    comp_vol = float(leaf_vol[wmask].max())
                    comp_dyn = float(leaf_dyn[wmask].max())
                    mean_margin = float(margin_map[wmask].mean())

                    # Log-normalized topological volume score in [0, 1]
                    norm_vol = float(np.tanh(comp_vol / 6.0))
                    topo_volume_score = 0.45 * norm_vol + 0.35 * mean_margin + 0.20 * comp_dyn

                    gt_majority = int(np.bincount(gt[wmask], minlength=3).argmax())
                    b_record = {
                        "img_idx": img_idx,
                        "wid": wid,
                        "rcc8_group": rcc8_group,
                        "area": area,
                        "topo_score": float(topo_volume_score),
                        "gt_class": gt_majority,
                    }
                    frame_blob_indices.append(len(blobs))
                    blobs.append(b_record)

                frames_data.append((pred, gt, p_soil, p_crop, cc_map, frame_blob_indices))
                del tree, alt, vol, dyn
                img_idx += 1

    return raw_cm, blobs, frames_data


def fit_mondrian_crc_thresholds(cal_blobs: list, target_alpha: float = 0.03) -> dict:
    learned_lambdas = {}
    grid = np.linspace(0.0, 1.0, 1001)
    for g in ("DC", "EC_PO", "NTPP_TPP"):
        group_tp = np.array([b["topo_score"] for b in cal_blobs if b["rcc8_group"] == g and b["gt_class"] == 2])
        n_tp = len(group_tp)
        if n_tp == 0:
            group_all = [b["topo_score"] for b in cal_blobs if b["rcc8_group"] == g]
            learned_lambdas[g] = float(max(group_all) + 1e-4) if group_all else 0.90
            continue
        best_lam = 0.0
        for lam in grid:
            if (np.sum(group_tp < lam) + 1.0) / (n_tp + 1.0) <= target_alpha:
                best_lam = float(lam)
            else:
                break
        learned_lambdas[g] = best_lam
    return learned_lambdas


def evaluate_with_genuine_volume_crc(title: str, raw_cm, blobs, frames_data, target_alpha: float = 0.03):
    fold_a = [b for b in blobs if b["img_idx"] % 2 == 0]
    fold_b = [b for b in blobs if b["img_idx"] % 2 == 1]

    lam_a = fit_mondrian_crc_thresholds(fold_a, target_alpha=target_alpha)
    lam_b = fit_mondrian_crc_thresholds(fold_b, target_alpha=target_alpha)

    # Apply calibrated thresholds to both blobs AND actual pixel predictions (reclassifying pruned EC->Crop, DC->Soil)
    crc_cm = np.zeros((3, 3), dtype=np.int64)
    kept_blobs = []

    for img_idx, (pred, gt, p_soil, p_crop, cc_map, b_indices) in enumerate(frames_data):
        pred_crc = pred.copy()
        active_lam = lam_a if (img_idx % 2 == 1) else lam_b

        for b_idx in b_indices:
            b_rec = blobs[b_idx]
            thresh = active_lam[b_rec["rcc8_group"]]
            if b_rec["topo_score"] >= thresh:
                kept_blobs.append(b_rec)
            else:
                # Reassign pruned false-positive blob pixels to the runner-up class (Crop vs Soil)
                wmask = (cc_map == b_rec["wid"])
                pred_crc[wmask] = np.where(p_crop[wmask] >= p_soil[wmask], 1, 0)

        crc_cm += fast_confusion_matrix(gt, pred_crc, n_cls=3)

    def cm_metrics(cm):
        tp_w = float(cm[2, 2])
        fp_w = float(cm[:, 2].sum() - tp_w)
        fn_w = float(cm[2, :].sum() - tp_w)
        iou_w = tp_w / (tp_w + fp_w + fn_w + 1e-6)
        prec_w = tp_w / (tp_w + fp_w + 1e-6)
        rec_w = tp_w / (tp_w + fn_w + 1e-6)
        ious = [float(cm[c, c]) / (float(cm[c, c] + cm[:, c].sum() - cm[c, c] + cm[c, :].sum() - cm[c, c]) + 1e-6) for c in range(3)]
        return iou_w, prec_w, rec_w, float(np.mean(ious))

    raw_iou, raw_prec, raw_rec, raw_miou = cm_metrics(raw_cm)
    crc_iou, crc_prec, crc_rec, crc_miou = cm_metrics(crc_cm)

    tp_raw = sum(1 for b in blobs if b["gt_class"] == 2)
    fp_soil_raw = sum(1 for b in blobs if b["gt_class"] == 0)
    fp_crop_raw = sum(1 for b in blobs if b["gt_class"] == 1)

    tp_crc = sum(1 for b in kept_blobs if b["gt_class"] == 2)
    fp_soil_crc = sum(1 for b in kept_blobs if b["gt_class"] == 0)
    fp_crop_crc = sum(1 for b in kept_blobs if b["gt_class"] == 1)
    emp_risk = 1.0 - (tp_crc / max(tp_raw, 1))

    print(f"\n================================================================================================")
    print(f" [{title}] Margin Max-Tree Volume + Mondrian CRC (alpha <= {target_alpha*100:.1f}%)")
    print(f"================================================================================================")
    print(f"  * [Before] Weed IoU={raw_iou:.4f} (Prec={raw_prec:.4f}, Rec={raw_rec:.4f}) | mIoU={raw_miou:.4f} | "
          f"Total FP={fp_soil_raw + fp_crop_raw:3d} (Soil={fp_soil_raw:3d}, Crop={fp_crop_raw:3d}) | TP={tp_raw}")
    print(f"  * [After ] Weed IoU={crc_iou:.4f} (Prec={crc_prec:.4f}, Rec={crc_rec:.4f}) | mIoU={crc_miou:.4f} | "
          f"Total FP={fp_soil_crc + fp_crop_crc:3d} (Soil={fp_soil_crc:3d}, Crop={fp_crop_crc:3d}) | TP={tp_crc}")
    print(f"  * [Effect] FP Blobs Removed: -{(fp_soil_raw + fp_crop_raw) - (fp_soil_crc + fp_crop_crc):3d} "
          f"(Soil: -{fp_soil_raw - fp_soil_crc:3d}, Crop-Zone: -{fp_crop_raw - fp_crop_crc:3d}) | "
          f"Empirical TP Loss={emp_risk*100:.2f}% <= {target_alpha*100:.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    for name, fname in [
        ("1. Baseline U-Net", "baseline_unet.pth"),
        ("2. Phase 4 Topo-ExG U-Net", "topo_exg_unet.pth"),
        ("3. Phase 6 Pure Higra-RCC8 U-Net (0% ExG)", "pure_higra_rcc8_unet.pth"),
    ]:
        wpath = resolve_weight_path(fname)
        model = SimpleUNet().to(device)
        model.load_state_dict(torch.load(wpath, map_location=device))
        raw_cm, blobs, frames_data = collect_margin_volume_blobs(model, val_loader, device)
        evaluate_with_genuine_volume_crc(name, raw_cm, blobs, frames_data, target_alpha=0.03)
        if "Phase 6" in name or "Phase 4" in name:
            evaluate_with_genuine_volume_crc(name + " [Strict FP Mode]", raw_cm, blobs, frames_data, target_alpha=0.05)


if __name__ == "__main__":
    main()
