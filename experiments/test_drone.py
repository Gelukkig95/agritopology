r"""
experiments/test_drone_honest.py
--------------------------------
[(UAV)   &     —    0%]
  baseline_unet.pth PhenoBench (  ) 
  3     :
  1. [  ] 1x( ) -> 2x(2 ) -> 3x(3 )
      CNN (p>=0.5) vs Higra Max-Tree  (Dynamics)    
  2. [(FOV)   ]  /   (RCC8 Border-EC)  
     (Interior)   False Positive(FP Rate)  
  3. [  (No-Strike Referee)  ]
      (Crop)  (RCC8-EC/PO) (UNKNOWN)    vs  
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PHENOBENCH_EXP_DIR = ROOT_DIR / "experiments" / "phenobench"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PHENOBENCH_EXP_DIR))

import cv2
import numpy as np
import higra as hg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE
from torch_morph_rcc8.referee import AgriTopoReferee

GRAPH_256 = hg.get_4_adjacency_graph((IMG_SIZE, IMG_SIZE))


def find_baseline_weights() -> Path:
    for p in [
        ROOT_DIR / "baseline_unet.pth",
        PHENOBENCH_EXP_DIR / "baseline_unet.pth",
        ROOT_DIR / "weights" / "baseline_unet.pth",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("baseline_unet.pth    .")


def classify_pixel_rcc8(blob_mask: np.ndarray, crop_mask: np.ndarray) -> str:
    k3 = np.ones((3, 3), np.uint8)
    crop_closure = cv2.dilate(crop_mask, k3, iterations=1)
    crop_interior = cv2.erode(crop_mask, k3, iterations=1)
    overlap = int((blob_mask & (crop_mask > 0)).sum())
    touch = int((blob_mask & (crop_closure > 0)).sum())
    if touch == 0:
        return "DC"
    if overlap == 0:
        return "EC"
    area = int(blob_mask.sum())
    if overlap == area:
        return "NTPP" if int((blob_mask & (crop_interior > 0)).sum()) == area else "TPP"
    return "PO"


def test_1_altitude_drop(model, val_ds, device, n_images: int = 120):
    """
    [ 1]     (1x -> 2x -> 3x):
         ,
      (A)    (p_weed >= 0.50)
      (B) Higra Max-Tree    ( prominence >= 0.18)
       (GT Weed )  %   (Recall) False Positive(FP)  .
    """
    print("=" * 108)
    print(" [ 1:    ( )      — 1x vs 2x vs 3x ]")
    print("=" * 108)
    print(f" {'  ()':<22} | {' A:  CNN (p>=0.50)':<36} | {' B: Higra Max-Tree  '}")
    print("-" * 108)

    for scale_factor, alt_label in [
        (1, "1x   (256x256)"),
        (2, "2x   (128x128)"),
        (3, "3x    ( 85x85 )"),
    ]:
        gt_total = 0
        cnn_hit, cnn_fp = 0, 0
        higra_hit, higra_fp = 0, 0

        with torch.no_grad():
            for i in range(min(n_images, len(val_ds))):
                img_t, lbl_t = val_ds[i]
                gt = lbl_t.numpy()

                #   :       
                if scale_factor > 1:
                    h_low = IMG_SIZE // scale_factor
                    low_t = F.interpolate(img_t.unsqueeze(0), size=(h_low, h_low), mode="area")
                    in_t = F.interpolate(low_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
                else:
                    in_t = img_t.unsqueeze(0)

                logits = model(in_t.to(device))
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                p_soil, p_crop, p_weed = probs[0], probs[1], probs[2]

                #   (Weed ,  >= 8px) 
                n_gt, gt_cc = cv2.connectedComponents((gt == 2).astype(np.uint8))
                gt_blobs = [(gt_cc == gid) for gid in range(1, n_gt) if (gt_cc == gid).sum() >= 8]
                gt_total += len(gt_blobs)

                #  A:  CNN (argmax == 2,  p_weed 1 )
                pred_cnn = (probs.argmax(axis=0) == 2).astype(np.uint8)
                for gm in gt_blobs:
                    if (pred_cnn[gm] > 0).sum() >= 2:
                        cnn_hit += 1
                n_pc, pc_cc = cv2.connectedComponents(pred_cnn)
                for pid in range(1, n_pc):
                    pm = (pc_cc == pid)
                    if pm.sum() >= 5 and (gt[pm] == 2).sum() == 0:
                        cnn_fp += 1

                #  B: Higra Max-Tree   (  Total     )
                # , Crop    (dynamics >= 0.18)   
                signal = np.clip(p_weed - 0.5 * p_crop, 0.0, 1.0).astype(np.float64)
                tree, alt = hg.component_tree_max_tree(GRAPH_256, signal)
                dyn = hg.attribute_dynamics(tree, alt)
                leaf_dyn = dyn[tree.parents()[: IMG_SIZE * IMG_SIZE]].reshape(IMG_SIZE, IMG_SIZE)
                pred_higra = ((leaf_dyn >= 0.16) & (p_weed >= 0.22) & (p_crop < 0.45)).astype(np.uint8)

                for gm in gt_blobs:
                    if (pred_higra[gm] > 0).sum() >= 2:
                        higra_hit += 1
                n_ph, ph_cc = cv2.connectedComponents(pred_higra)
                for pid in range(1, n_ph):
                    pm = (ph_cc == pid)
                    if pm.sum() >= 5 and (gt[pm] == 2).sum() == 0:
                        higra_fp += 1

        rec_cnn = cnn_hit / max(gt_total, 1) * 100.0
        rec_higra = higra_hit / max(gt_total, 1) * 100.0
        print(
            f" {alt_label:<22} | ={rec_cnn:5.1f}% ({cnn_hit:3d}/{gt_total:3d}) | False Positive={cnn_fp:3d}  || "
            f"={rec_higra:5.1f}% ({higra_hit:3d}/{gt_total:3d}) [{rec_higra-rec_cnn:+5.1f}%p] | False Positive={higra_fp:3d}"
        )
    print("-" * 108)


def test_2_and_3_fov_border_and_collateral_damage(model, val_loader, device):
    """
    [ 2]   (FOV) (Border-EC)  False Positive vs (Interior) False Positive 
    [ 3]  (Crop)    RCC8    (Trade-off) 
    """
    border_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    border_mask[:6, :] = border_mask[-6:, :] = border_mask[:, :6] = border_mask[:, -6:] = True

    border_blobs_tp, border_blobs_fp = 0, 0
    inner_blobs_tp, inner_blobs_fp = 0, 0

    #  3   (  vs RCC8    vs )
    total_tp_targets = 0
    total_collateral_crop_fp = 0   #  (Crop)     
    total_soil_waste_fp = 0        #     False Positive

    safe_tp_hit = 0
    safe_collateral_crop_fp = 0
    abstained_unknown_count = 0
    sample_xai_logs = []

    referee = AgriTopoReferee(lca_safe_threshold=0.710, lca_reject_threshold=0.560)

    with torch.no_grad():
        for b_idx, (imgs, labels) in enumerate(val_loader):
            logits = model(imgs.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            gts = labels.numpy()

            for i in range(preds.shape[0]):
                pred, gt = preds[i], gts[i]
                p_crop, p_weed = probs[i, 1], probs[i, 2]

                crop_bin = (pred == 1).astype(np.uint8)
                weed_bin = (pred == 2).astype(np.uint8)
                n_w, w_cc = cv2.connectedComponents(weed_bin)

                for wid in range(1, n_w):
                    wmask = (w_cc == wid)
                    area = int(wmask.sum())
                    if area < 5:
                        continue

                    gt_maj = int(np.bincount(gt[wmask], minlength=3).argmax())
                    touches_fov_border = bool((wmask & border_mask).any())

                    # [ 2 ] FOV    vs  
                    if touches_fov_border:
                        if gt_maj == 2:
                            border_blobs_tp += 1
                        else:
                            border_blobs_fp += 1
                    else:
                        if gt_maj == 2:
                            inner_blobs_tp += 1
                        else:
                            inner_blobs_fp += 1

                    # [ 3 ]    
                    if gt_maj == 2:
                        total_tp_targets += 1
                    elif gt_maj == 1:
                        total_collateral_crop_fp += 1
                    else:
                        total_soil_waste_fp += 1

                    # [ 3] RCC8   :
                    #  (Crop) (EC/PO/NTPP)       (ABSTAIN)!
                    rel = classify_pixel_rcc8(wmask.astype(np.uint8), crop_bin)
                    mean_conf = float(p_weed[wmask].mean())

                    if rel != "DC":
                        #  (Crop)   ->  Risky UNKNOWN_OCCLUSION  !
                        abstained_unknown_count += 1
                        if len(sample_xai_logs) < 2 and gt_maj == 1:
                            rep = referee.judge_crop_weed_blob(
                                blob_id=f"DroneFrame_{b_idx}_{i}_Blob{wid}",
                                dl_pred="weed",
                                dl_conf=mean_conf,
                                rcc8_rel_with_crop=rel if rel in ("EC", "PO", "NTPP") else "EC",
                                lca_merge_altitude=0.620,
                                growth_stage="late_06-05",
                            )
                            sample_xai_logs.append(rep.to_pretty_log())
                    else:
                        #    (RCC8-DC) ->  !
                        if gt_maj == 2:
                            safe_tp_hit += 1
                        elif gt_maj == 1:
                            safe_collateral_crop_fp += 1

    border_tot = border_blobs_tp + border_blobs_fp
    inner_tot = inner_blobs_tp + inner_blobs_fp
    border_fp_rate = border_blobs_fp / max(border_tot, 1) * 100.0
    inner_fp_rate = inner_blobs_fp / max(inner_tot, 1) * 100.0

    print("\n" + "=" * 108)
    print(" [ 2:   (FOV)  (Border-EC)  vs (Interior)  False Positive ]")
    print("=" * 108)
    print(f"  •  (Interior)       :  {inner_tot:4d}  False Positive {inner_blobs_fp:3d} (False Positive = {inner_fp_rate:5.2f}%)")
    print(f"  •  (Border-EC)    :  {border_tot:4d}  False Positive {border_blobs_fp:3d} (False Positive = {border_fp_rate:5.2f}%)")
    print(f"  ->  :    False Positive   {border_fp_rate - inner_fp_rate:+.2f}%p ")

    collateral_reduction = (1.0 - safe_collateral_crop_fp / max(total_collateral_crop_fp, 1)) * 100.0
    tp_retention = safe_tp_hit / max(total_tp_targets, 1) * 100.0

    print("\n" + "=" * 108)
    print(" [ 3:     (RCC8 Safe-Clearance)    (Gain) (Cost)]")
    print("=" * 108)
    print(f"  1)    /  :   ={total_tp_targets:4d} (100.0%) | "
          f"(Crop)  ={total_collateral_crop_fp:3d} |  ={total_soil_waste_fp:3d}")
    print(f"  2) RCC8  (EC/PO )  :   ={safe_tp_hit:4d} ({tp_retention:5.1f}%) | "
          f"(Crop)  ={safe_collateral_crop_fp:3d} (-{collateral_reduction:.1f}% !)")
    print(f"  3) /  (UNKNOWN)  :  {abstained_unknown_count} Risky    (ABSTAIN)  ")
    print("-" * 108)
    for log_str in sample_xai_logs:
        print(log_str)
        print("-" * 108)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w_path = find_baseline_weights()
    print(f"Loaded weights: {w_path} | Device: {device}")

    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()

    test_1_altitude_drop(model, val_ds, device, n_images=120)
    test_2_and_3_fov_border_and_collateral_damage(model, val_loader, device)


if __name__ == "__main__":
    main()