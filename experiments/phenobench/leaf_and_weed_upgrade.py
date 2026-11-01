import os
import cv2
import numpy as np
import torch
import higra as hg
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8
from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE

TRAIN_DIR = r"D:\Download\PhenoBench-v110\PhenoBench\train"
PLANT_DIR = f"{TRAIN_DIR}/plant_instances"
LEAF_DIR = f"{TRAIN_DIR}/leaf_instances"
SEM_DIR = f"{TRAIN_DIR}/semantics"
IMG_DIR = f"{TRAIN_DIR}/images"


def compute_exg_norm(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2.0 * g - r - b
    return (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)


# ==============================================================================
# PART 1:  (plant_instances)  (leaf_instances)   
# ==============================================================================
def count_leaves_naive_cc(plant_mask):
    """1.  (OpenCV ):     1 """
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(plant_mask)
    return sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 40)


def count_leaves_global_watershed(plant_mask):
    """2.  OpenCV   (0.5*max):       """
    dist = cv2.distanceTransform(plant_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return 0
    peaks = (dist >= 0.45 * dist.max()).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(peaks)
    return sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 5)


def count_leaves_higra_maxtree(plant_mask, exg_norm, min_leaf_area=40):
    """
    3. ★  :  Higra Max-tree  (Dynamics)    
      (Distance Transform)  (ExG)    ,
       (Saddle LCA)   (Dynamics) 
        (Peak)   .
    """
    ys, xs = np.where(plant_mask > 0)
    if len(ys) < min_leaf_area:
        return 0
    y0, y1 = max(0, ys.min() - 2), min(plant_mask.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(plant_mask.shape[1], xs.max() + 3)

    sub_mask = plant_mask[y0:y1, x0:x1]
    sub_exg = exg_norm[y0:y1, x0:x1]

    #        + ExG   
    dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5).astype(np.float32)
    dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
    if dist_smooth.max() > 0:
        dist_norm = dist_smooth / (dist_smooth.max() + 1e-6)
    else:
        return 1

    topo_field = (0.80 * dist_norm + 0.20 * sub_exg) * sub_mask
    topo_field = cv2.GaussianBlur(topo_field, (3, 3), 0).astype(np.float64)

    graph = hg.get_4_adjacency_graph(topo_field.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, topo_field)

    area = hg.attribute_area(tree)
    dynamics = hg.attribute_dynamics(tree, altitudes)

    #    :   (Saddle)  (dynamics >= 0.08)  
    valid_nodes = (area >= 8) & (area <= 2500) & (dynamics >= 0.085) & (altitudes >= 0.15)

    parents = tree.parents()
    node_has_valid_ancestor = valid_nodes.copy()
    for node in range(tree.root() - 1, -1, -1):
        if node_has_valid_ancestor[parents[node]]:
            node_has_valid_ancestor[node] = True

    pixel_peaks = node_has_valid_ancestor[:tree.num_leaves()].reshape(topo_field.shape)
    peak_mask = (pixel_peaks & (sub_mask > 0)).astype(np.uint8)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(peak_mask)
    count = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 5)
    return max(1, count)


def run_leaf_decomposition_experiment():
    print("=" * 76)
    print("[ 1]   ' (leaf_instances)'     ")
    print("=" * 76)
    files = sorted(os.listdir(PLANT_DIR))
    #  (05-15), (05-26), (06-05)  40 
    for date_str in ["05-15", "05-26", "06-05"]:
        date_files = [f for f in files if f.startswith(date_str)][:40]
        gt_counts, cc_errs, ws_errs, hg_errs = [], [], [], []

        for fname in date_files:
            plant_inst = cv2.imread(f"{PLANT_DIR}/{fname}", cv2.IMREAD_UNCHANGED)
            leaf_inst = cv2.imread(f"{LEAF_DIR}/{fname}", cv2.IMREAD_UNCHANGED)
            sem = cv2.imread(f"{SEM_DIR}/{fname}", cv2.IMREAD_UNCHANGED)
            img = cv2.imread(f"{IMG_DIR}/{fname}", cv2.IMREAD_COLOR)
            if plant_inst is None or leaf_inst is None or sem is None or img is None:
                continue

            exg_norm = compute_exg_norm(img)
            for pid in [i for i in np.unique(plant_inst) if i != 0]:
                pmask = (plant_inst == pid).astype(np.uint8)
                if pmask.sum() < 1200:
                    continue
                # Crop(crop: 1  3)  
                maj_sem = np.bincount(sem[pmask > 0], minlength=5).argmax()
                if maj_sem not in (1, 3):
                    continue

                #        (leaf_instances)  
                leaves_in_plant = leaf_inst[pmask > 0]
                unique_lids, lid_counts = np.unique(leaves_in_plant, return_counts=True)
                true_leaf_cnt = sum(1 for lid, cnt in zip(unique_lids, lid_counts) if lid != 0 and cnt >= 40)
                if true_leaf_cnt < 2:
                    continue

                c_cc = count_leaves_naive_cc(pmask)
                c_ws = count_leaves_global_watershed(pmask)
                c_hg = count_leaves_higra_maxtree(pmask, exg_norm)

                gt_counts.append(true_leaf_cnt)
                cc_errs.append(abs(c_cc - true_leaf_cnt))
                ws_errs.append(abs(c_ws - true_leaf_cnt))
                hg_errs.append(abs(c_hg - true_leaf_cnt))

        print(f"• {date_str} ( {len(gt_counts):3d},  Mean  {np.mean(gt_counts):.2f}) | "
              f" Error: {np.mean(cc_errs):.2f} | "
              f" Error: {np.mean(ws_errs):.2f} | "
              f"★ Higra Max-tree Error: {np.mean(hg_errs):.2f}")


# ==============================================================================
# PART 2: Weed  IoU   +    (  0% 2-Fold CV)
# ==============================================================================
def compute_exg_from_tensor(img_tensor):
    img = img_tensor.numpy()
    r, g, b = img[0], img[1], img[2]
    return 2.0 * g - r - b


def run_weed_iou_and_kpi_upgrade():
    print("\n" + "=" * 76)
    print("[ 2] 2-Fold CV Weed  IoU  +      ")
    print("=" * 76)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))
    model.eval()

    samples, blobs = [], []
    kernel_5 = np.ones((5, 5), np.uint8)

    with torch.no_grad():
        for imgs, labels in val_loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels.numpy()

            for i in range(preds.shape[0]):
                pred = preds[i]
                gt = labels_np[i]
                p_weed, p_crop = probs[i, 2], probs[i, 1]
                exg = compute_exg_from_tensor(imgs[i])

                crop_bin = (pred == 1).astype(np.uint8)
                weed_bin = (pred == 2).astype(np.uint8)
                n_crop, crop_cc = cv2.connectedComponents(crop_bin)
                n_weed, weed_cc = cv2.connectedComponents(weed_bin)

                s_idx = len(samples)
                samples.append({"pred": pred, "label": gt, "probs": probs[i], "exg": exg, "weed_cc": weed_cc})
                if n_weed <= 1:
                    continue

                crop_polys = {}
                for cid in range(1, n_crop):
                    m = (crop_cc == cid).astype(np.uint8)
                    if m.sum() >= 10:
                        poly = mask_to_polygon(m)
                        if poly is not None:
                            crop_polys[cid] = poly

                for wid in range(1, n_weed):
                    wmask = (weed_cc == wid).astype(np.uint8)
                    size = int(wmask.sum())
                    if size < 5:
                        continue
                    gt_maj = np.bincount(gt[wmask > 0], minlength=3).argmax()
                    dil = cv2.dilate(wmask, kernel_5)

                    wpoly = mask_to_polygon(wmask)
                    best_rel = RCC8.DC
                    if wpoly is not None:
                        for cid, cpoly in crop_polys.items():
                            rel = classify_rcc8(wpoly, cpoly)
                            if rel != RCC8.DC:
                                best_rel = rel
                                break

                    blobs.append({
                        "sample_idx": s_idx, "wid": wid, "size": size,
                        "is_tp": gt_maj == 2, "is_dc": best_rel == RCC8.DC,
                        "pweed": float(p_weed[wmask > 0].mean()),
                        "pweed_max": float(p_weed[wmask > 0].max()),
                        "pcrop_surr": float(p_crop[dil > 0].mean()),
                        "exg": float(exg[wmask > 0].mean())
                    })

    # 2-Fold :  IoU  ( False Positive  ) 
    keep_mask = [True] * len(blobs)
    for parity in [0, 1]:
        tr_idx = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 == parity) and b["size"] < 25]
        te_idx = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 != parity) and b["size"] < 25]

        for is_dc_flag in [True, False]:
            tr_sub = [i for i in tr_idx if blobs[i]["is_dc"] == is_dc_flag]
            te_sub = [i for i in te_idx if blobs[i]["is_dc"] == is_dc_flag]
            if not tr_sub or not te_sub:
                continue

            X_tr = np.array([[blobs[i]["size"], blobs[i]["pweed"], blobs[i]["pweed_max"],
                              blobs[i]["pcrop_surr"], blobs[i]["exg"]] for i in tr_sub])
            y_tr = np.array([1 if blobs[i]["is_tp"] else 0 for i in tr_sub])
            X_te = np.array([[blobs[i]["size"], blobs[i]["pweed"], blobs[i]["pweed_max"],
                              blobs[i]["pcrop_surr"], blobs[i]["exg"]] for i in te_sub])

            clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
            tr_probs = clf.predict_proba(X_tr)[:, 1]
            te_probs = clf.predict_proba(X_te)[:, 1]

            #  (tr_sub)  ( FP  - 1.35 *  TP )   
            best_thr, best_gain = 0.0, 0.0
            for cand_thr in np.linspace(0.05, 0.75, 36):
                fp_px = sum(blobs[i]["size"] for j, i in enumerate(tr_sub) if tr_probs[j] < cand_thr and not blobs[i]["is_tp"])
                tp_px = sum(blobs[i]["size"] for j, i in enumerate(tr_sub) if tr_probs[j] < cand_thr and blobs[i]["is_tp"])
                gain = fp_px - 1.35 * tp_px
                if gain > best_gain:
                    best_gain = gain
                    best_thr = cand_thr

            for j, idx in enumerate(te_sub):
                keep_mask[idx] = bool(te_probs[j] >= best_thr)

    # 1: RCC8   False Positive  
    pruned_preds = [s["pred"].copy() for s in samples]
    for bi, b in enumerate(blobs):
        if not keep_mask[bi]:
            s = samples[b["sample_idx"]]
            wmask = (s["weed_cc"] == b["wid"])
            p_bg, p_crop = s["probs"][0][wmask], s["probs"][1][wmask]
            pruned_preds[b["sample_idx"]][wmask] = np.where(p_crop > p_bg, 1, 0)

    # 2:   Weed   (Hysteresis)     
    upgraded_preds = []
    kernel_3 = np.ones((3, 3), np.uint8)
    for s_idx, s in enumerate(samples):
        cur = pruned_preds[s_idx].copy()
        p_bg, p_crop, p_weed = s["probs"][0], s["probs"][1], s["probs"][2]
        exg = s["exg"]

        #  Weed(p_weed > 0.80)    ,  Background  (0.38~0.50)    
        high_conf_weed = ((cur == 2) & (p_weed > 0.80)).astype(np.uint8)
        connected_zone = cv2.dilate(high_conf_weed, kernel_3) > 0
        hysteresis_recover = (
            connected_zone
            & (cur == 0)
            & (p_weed >= 0.38)
            & (p_crop < 0.08)
            & (exg > 0.16)
        )
        cur[hysteresis_recover] = 2
        upgraded_preds.append(cur)

    all_gt = np.concatenate([s["label"].ravel() for s in samples])
    fp_raw = sum(1 for b in blobs if not b["is_tp"])
    fp_kept = sum(1 for i, b in enumerate(blobs) if not b["is_tp"] and keep_mask[i])

    for title, pred_list, fp_cnt in [
        ("0. No Correction (Raw Baseline)", [s["pred"] for s in samples], fp_raw),
        ("1. RCC8 IoU- False Positive  (2-Fold CV)", pruned_preds, fp_kept),
        ("2. ★ RCC8 False Positive  +  (Hysteresis) ", upgraded_preds, fp_kept),
    ]:
        all_p = np.concatenate([p.ravel() for p in pred_list])
        cm = confusion_matrix(all_gt, all_p, labels=[0, 1, 2])
        c = 2
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        iou = tp / (tp + fp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        rem = fp_raw - fp_cnt
        print(f"\n[{title}]")
        print(f"  • Weed   : IoU = {iou:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f}")
        print(f"  •   : {fp_cnt} ( {fp_raw}  -{rem} , -{rem/fp_raw*100:.1f}%)")


if __name__ == "__main__":
    run_leaf_decomposition_experiment()
    run_weed_iou_and_kpi_upgrade()