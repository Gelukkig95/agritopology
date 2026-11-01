import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PHENOBENCH_EXP_DIR = ROOT_DIR / "experiments" / "phenobench"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PHENOBENCH_EXP_DIR))

import cv2
import numpy as np
import higra as hg
from scipy import ndimage
import torch
import torch.nn.functional as F

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE
from torch_morph_rcc8.diagnostics import kuratowski_closure, kuratowski_interior

GRAPH_256 = hg.get_4_adjacency_graph((IMG_SIZE, IMG_SIZE))


def find_weights() -> Path:
    for p in [
        ROOT_DIR / "weights" / "baseline_unet.pth",
        ROOT_DIR / "baseline_unet.pth",
        PHENOBENCH_EXP_DIR / "baseline_unet.pth",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("baseline_unet.pth    .")


def apply_nonrigid_and_lighting_corruption(img_chw: np.ndarray, gt_hw: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    H, W = gt_hw.shape

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = ndimage.gaussian_filter(rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32), sigma=12) * 18.0
    dy = ndimage.gaussian_filter(rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32), sigma=12) * 18.0
    map_x = np.clip(xx + dx, 0, W - 1)
    map_y = np.clip(yy + dy, 0, H - 1)

    img_hwc = img_chw.transpose(1, 2, 0).copy()
    warped_img = cv2.remap(img_hwc, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped_gt = cv2.remap(gt_hw.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    angle = rng.uniform(0, np.pi)
    stripe = np.sin(((xx - H / 2) * np.cos(angle) + (yy - W / 2) * np.sin(angle)) / 9.0)
    shadow_bars = (0.72 + 0.28 * stripe)[:, :, None].astype(np.float32)
    corrupted = warped_img * shadow_bars

    for _ in range(12):
        cy, cx = int(rng.integers(15, H - 15)), int(rng.integers(15, W - 15))
        r = int(rng.integers(3, 6))
        yy_c, xx_c = np.ogrid[-cy : H - cy, -cx : W - cx]
        disk = (yy_c * yy_c + xx_c * xx_c) <= r * r
        corrupted[disk, 0] *= 0.65
        corrupted[disk, 1] = np.clip(corrupted[disk, 1] * 1.25 + 0.05, 0.0, 1.0)
        corrupted[disk, 2] *= 0.65

    for _ in range(2):
        y_line = int(rng.integers(40, H - 40))
        corrupted[y_line : y_line + 2, :, :] *= 0.25

    corrupted = np.clip(corrupted, 0.0, 1.0)
    return corrupted.transpose(2, 0, 1), warped_gt.astype(np.int64)


def apply_topdown_maxtree_rcc8_constraint_fixed(probs_chw: np.ndarray, img_chw: np.ndarray) -> np.ndarray:
    p_soil, p_crop, p_weed = probs_chw[0], probs_chw[1], probs_chw[2]
    raw_pred = probs_chw.argmax(axis=0)
    H, W = raw_pred.shape

    #    (ExG)   Background   
    r, g, b = img_chw[0], img_chw[1], img_chw[2]
    exg = 2.0 * g - r - b
    exg_bg = cv2.GaussianBlur(exg, (0, 0), sigmaX=15)
    exg_prom = np.clip(exg - exg_bg, 0.0, 1.0)

    # 1. Crop(Crop) :   (), (2px)    (Closing) +  (NTPP) 
    crop_base = (raw_pred == 1)
    # / ( 2px)  Crop  5x5   ,       
    crop_closed = ndimage.binary_closing(crop_base, structure=np.ones((5, 5)))
    crop_bridged = crop_base | (crop_closed & ((p_crop >= 0.12) | (exg_prom >= 0.04)))
    crop_filled = ndimage.binary_fill_holes(crop_bridged)

    # 2. Higra Max-Tree (    )     
    comb_field = np.round(np.clip(0.6 * p_weed + 0.4 * (exg_prom * 3.0), 0.0, 1.0) * 255.0) / 255.0
    tree, alt = hg.component_tree_max_tree(GRAPH_256, comb_field.astype(np.float64))
    area = hg.attribute_area(tree)
    dyn = hg.attribute_dynamics(tree, alt)
    #  6px         (Rootfix/Opening)
    invalid_nodes = (area < 8) | (dyn < 0.18)
    invalid_nodes[: H * W] = False
    clean_recon = hg.reconstruct_leaf_data(tree, alt, invalid_nodes).reshape(H, W)

    # 3. Weed(Weed)  RCC8 + Max-Tree 
    weed_raw = (raw_pred == 2).astype(np.uint8)
    n_w, w_cc = cv2.connectedComponents(weed_raw)
    weed_clean = np.zeros((H, W), dtype=bool)
    crop_dilated = kuratowski_closure(crop_filled)

    for wid in range(1, n_w):
        wmask = (w_cc == wid)
        b_area = int(wmask.sum())
        if b_area < 5:
            continue

        #  A (RCC8-NTPP/TPP):  Crop   False Positive Crop 
        if (wmask & crop_filled).sum() >= 0.7 * b_area:
            continue

        #  B: Crop  (EC)    Crop    Crop  False Positive 
        touches_crop = (wmask & crop_dilated).any()
        if touches_crop and b_area < 25 and float(p_crop[wmask].mean()) > 0.18:
            continue

        #  C (Max-Tree   &   ):
        #        Max-Tree     
        recon_peak = float(clean_recon[wmask].max())
        mean_w = float(p_weed[wmask].mean())
        if recon_peak >= 0.38 and mean_w >= 0.52:
            weed_clean |= wmask

    out_pred = raw_pred.copy()
    #   Weed 2 (Crop  ) 
    rejected_weed = (raw_pred == 2) & (~weed_clean)
    out_pred[rejected_weed] = np.where(p_crop[rejected_weed] > p_soil[rejected_weed], 1, 0)
    #  Crop      
    out_pred[crop_filled & (out_pred != 2)] = 1
    return out_pred


def count_fragmentation_errors(gt_bin: np.ndarray, pred_bin: np.ndarray) -> int:
    n_gt, gt_cc = cv2.connectedComponents(gt_bin.astype(np.uint8))
    shattered_pieces = 0
    for gid in range(1, n_gt):
        gmask = (gt_cc == gid)
        if gmask.sum() < 40:
            continue
        n_p, _ = cv2.connectedComponents((pred_bin & gmask).astype(np.uint8))
        if n_p - 1 > 1:
            shattered_pieces += (n_p - 2)
    return shattered_pieces


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w_path = find_weights()
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()

    n_eval = 80
    fp_blobs_bu, fp_blobs_td = 0, 0
    shatter_bu, shatter_td = 0, 0
    cm_bu = np.zeros((3, 3), dtype=np.int64)
    cm_td = np.zeros((3, 3), dtype=np.int64)

    with torch.no_grad():
        for idx in range(n_eval):
            img_t, lbl_t = val_ds[idx]
            corrupted_chw, warped_gt = apply_nonrigid_and_lighting_corruption(img_t.numpy(), lbl_t.numpy(), seed=1000 + idx)

            inp = torch.from_numpy(corrupted_chw).unsqueeze(0).to(device)
            probs = F.softmax(model(inp), dim=1)[0].cpu().numpy()

            pred_bu = probs.argmax(axis=0)
            pred_td = apply_topdown_maxtree_rcc8_constraint_fixed(probs, corrupted_chw)

            cm_bu += np.bincount(3 * warped_gt.ravel() + pred_bu.ravel(), minlength=9).reshape(3, 3)
            cm_td += np.bincount(3 * warped_gt.ravel() + pred_td.ravel(), minlength=9).reshape(3, 3)

            for mode_name, pr in [("bu", pred_bu), ("td", pred_td)]:
                n_w, w_cc = cv2.connectedComponents((pr == 2).astype(np.uint8))
                fps = sum(1 for wid in range(1, n_w) if (w_cc == wid).sum() >= 5 and (warped_gt[w_cc == wid] == 2).sum() == 0)
                if mode_name == "bu":
                    fp_blobs_bu += fps
                else:
                    fp_blobs_td += fps

            shatter_bu += count_fragmentation_errors(warped_gt == 1, pred_bu == 1)
            shatter_td += count_fragmentation_errors(warped_gt == 1, pred_td == 1)

    def get_iou(cm, c):
        tp = float(cm[c, c])
        return tp / (tp + float(cm[:, c].sum() - tp) + float(cm[c, :].sum() - tp) + 1e-6) * 100.0

    bu_weed_iou, td_weed_iou = get_iou(cm_bu, 2), get_iou(cm_td, 2)
    bu_crop_iou, td_crop_iou = get_iou(cm_bu, 1), get_iou(cm_td, 1)

    print("=" * 92)
    print(" [   (n=80): Bottom-Up CNN  vs Top-Down (Max-Tree + RCC8)]")
    print("=" * 92)
    print(f" 1.    ( False Positive FP  ):")
    print(f"    • Bottom-Up CNN          : {fp_blobs_bu:4d}")
    print(f"    • +Top-Down (Max-Tree+RCC8) : {fp_blobs_td:4d} "
          f"({(fp_blobs_td - fp_blobs_bu)/max(fp_blobs_bu,1)*100:+.1f}%)")
    print("-" * 92)
    print(f" 2.  ·    ( /  ):")
    print(f"    • Bottom-Up CNN          : {shatter_bu:4d}  ")
    print(f"    • +Top-Down (Max-Tree+RCC8) : {shatter_td:4d} "
          f"({(shatter_td - shatter_bu)/max(shatter_bu,1)*100:+.1f}%)")
    print("-" * 92)
    print(f" 3. Total   (Pixel IoU):")
    print(f"    • (Weed) IoU            : {bu_weed_iou:5.2f}% -> {td_weed_iou:5.2f}% ({td_weed_iou - bu_weed_iou:+.2f}%p)")
    print(f"    •  (Crop) IoU         : {bu_crop_iou:5.2f}% -> {td_crop_iou:5.2f}% ({td_crop_iou - bu_crop_iou:+.2f}%p)")
    print("=" * 92)


if __name__ == "__main__":
    main()
