import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE


def compute_exg_from_tensor(img_tensor):
    img = img_tensor.numpy()
    r, g, b = img[0], img[1], img[2]
    return 2 * g - r - b


def extract_all(model, loader, device, min_size=5):
    model.eval()
    samples, blobs = [], []
    kernel_5 = np.ones((5, 5), np.uint8)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs_gpu = imgs.to(device)
            logits = model(imgs_gpu)
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

                sample_idx = len(samples)
                samples.append({"pred": pred, "label": gt, "probs": probs[i], "weed_cc": weed_cc})
                if n_weed <= 1:
                    continue

                crop_polys = {}
                for cid in range(1, n_crop):
                    m = (crop_cc == cid).astype(np.uint8)
                    if m.sum() < 10:
                        continue
                    poly = mask_to_polygon(m)
                    if poly is not None:
                        crop_polys[cid] = poly

                for wid in range(1, n_weed):
                    wmask = (weed_cc == wid).astype(np.uint8)
                    size = int(wmask.sum())
                    if size < min_size:
                        continue

                    gt_vals = gt[wmask > 0]
                    gt_majority = np.bincount(gt_vals, minlength=3).argmax()
                    mean_pweed = float(p_weed[wmask > 0].mean())
                    dilated = cv2.dilate(wmask, kernel_5)
                    mean_pcrop_surround = float(p_crop[dilated > 0].mean())
                    mean_exg = float(exg[wmask > 0].mean())

                    wpoly = mask_to_polygon(wmask)
                    best_rel = RCC8.DC
                    if wpoly is not None:
                        for cid, cpoly in crop_polys.items():
                            rel = classify_rcc8(wpoly, cpoly)
                            if rel != RCC8.DC:
                                best_rel = rel
                                break

                    if gt_majority == 2:
                        group = "TP"
                    elif gt_majority == 0 and best_rel == RCC8.DC:
                        group = "B"
                    elif gt_majority == 1 and best_rel != RCC8.DC:
                        group = "C"
                    elif gt_majority == 1 and best_rel == RCC8.DC:
                        group = "D"
                    else:
                        group = "OTHER"

                    blobs.append({
                        "sample_idx": sample_idx, "wid": wid, "group": group,
                        "is_dc": best_rel == RCC8.DC, "size": size, "pweed": mean_pweed,
                        "pcrop_surround": mean_pcrop_surround, "exg": mean_exg
                    })
    return samples, blobs


def cross_validated_filter(blobs, use_rcc8_branch=True, max_filter_size=30, target_recall=0.96):
    """
        2-Fold :
          ,      .
      (>= max_filter_size)  IoU     (True).
    """
    keep_mask = [True] * len(blobs)

    for train_parity in [0, 1]:
        train_idx = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 == train_parity) and b["size"] < max_filter_size]
        test_idx = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 != train_parity) and b["size"] < max_filter_size]

        if not use_rcc8_branch:
            #  A:     +  
            X_tr = np.array([[blobs[i]["size"], blobs[i]["pweed"]] for i in train_idx])
            y_tr = np.array([1 if blobs[i]["group"] == "TP" else 0 for i in train_idx])
            X_te = np.array([[blobs[i]["size"], blobs[i]["pweed"]] for i in test_idx])

            if len(set(y_tr)) > 1 and len(X_te) > 0:
                clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
                tr_scores = clf.predict_proba(X_tr)[:, 1]
                _, tpr, thr = roc_curve(y_tr, tr_scores)
                t_val = thr[np.argmin(np.abs(tpr - target_recall))]

                te_scores = clf.predict_proba(X_te)[:, 1]
                for j, idx in enumerate(test_idx):
                    keep_mask[idx] = bool(te_scores[j] >= t_val)
        else:
            #  B: RCC8  (DC  vs EC/PO ) + ExG   Crop  
            for is_dc_flag in [True, False]:
                tr_sub = [i for i in train_idx if blobs[i]["is_dc"] == is_dc_flag]
                te_sub = [i for i in test_idx if blobs[i]["is_dc"] == is_dc_flag]
                if not tr_sub or not te_sub:
                    continue

                if is_dc_flag:
                    X_tr = np.array([[blobs[i]["size"], blobs[i]["pweed"], blobs[i]["exg"]] for i in tr_sub])
                    X_te = np.array([[blobs[i]["size"], blobs[i]["pweed"], blobs[i]["exg"]] for i in te_sub])
                else:
                    X_tr = np.array([[blobs[i]["size"], blobs[i]["pweed"], blobs[i]["pcrop_surround"], blobs[i]["exg"]] for i in tr_sub])
                    X_te = np.array([[blobs[i]["size"], blobs[i]["pweed"], blobs[i]["pcrop_surround"], blobs[i]["exg"]] for i in te_sub])

                y_tr = np.array([1 if blobs[i]["group"] == "TP" else 0 for i in tr_sub])
                if len(set(y_tr)) > 1:
                    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
                    tr_scores = clf.predict_proba(X_tr)[:, 1]
                    _, tpr, thr = roc_curve(y_tr, tr_scores)
                    t_val = thr[np.argmin(np.abs(tpr - target_recall))]

                    te_scores = clf.predict_proba(X_te)[:, 1]
                    for j, idx in enumerate(te_sub):
                        keep_mask[idx] = bool(te_scores[j] >= t_val)

    return keep_mask


def evaluate_and_print(name, samples, blobs, keep_mask):
    corrected = [s["pred"].copy() for s in samples]
    for bi, blob in enumerate(blobs):
        if keep_mask[bi]:
            continue
        s = samples[blob["sample_idx"]]
        wmask = (s["weed_cc"] == blob["wid"])
        probs = s["probs"]
        p_bg, p_crop = probs[0][wmask], probs[1][wmask]
        corrected[blob["sample_idx"]][wmask] = np.where(p_crop > p_bg, 1, 0)

    all_preds = np.concatenate([p.ravel() for p in corrected])
    all_labels = np.concatenate([s["label"].ravel() for s in samples])
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    c = 2
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + fn - fn + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    tp_total = sum(1 for b in blobs if b["group"] == "TP")
    tp_kept = sum(1 for i, b in enumerate(blobs) if b["group"] == "TP" and keep_mask[i])
    fp_total = sum(1 for b in blobs if b["group"] in ("B", "C", "D"))
    fp_rem = sum(1 for i, b in enumerate(blobs) if b["group"] in ("B", "C", "D") and not keep_mask[i])
    b_rem = sum(1 for i, b in enumerate(blobs) if b["group"] == "B" and not keep_mask[i])
    c_rem = sum(1 for i, b in enumerate(blobs) if b["group"] == "C" and not keep_mask[i])
    d_rem = sum(1 for i, b in enumerate(blobs) if b["group"] == "D" and not keep_mask[i])

    print(f"\n=== [{name}] ===")
    print(f"•   (Weed): IoU={iou:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")
    print(f"•  Weed  : {tp_kept}/{tp_total} ({tp_kept/tp_total*100:.1f}%)")
    print(f"•  False Positive(FP)  : {fp_rem}/{fp_total} ({fp_rem/fp_total*100:.1f}%)")
    print(f"  - B(  540): {b_rem}  ({b_rem/540*100:.1f}%)")
    print(f"  - C(Crop   66): {c_rem}  ({c_rem/66*100:.1f}%)")
    print(f"  - D(Crop  101): {d_rem}  ({d_rem/101*100:.1f}%)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))

    print("   2-Fold   ...")
    samples, blobs = extract_all(model, val_loader, device)

    evaluate_and_print("0. No Correction (Raw Baseline)", samples, blobs, [True] * len(blobs))

    keep_a = cross_validated_filter(blobs, use_rcc8_branch=False, max_filter_size=30, target_recall=0.96)
    evaluate_and_print("1.  A (2-Fold CV,  +)", samples, blobs, keep_a)

    keep_b = cross_validated_filter(blobs, use_rcc8_branch=True, max_filter_size=30, target_recall=0.96)
    evaluate_and_print("2.  B (2-Fold CV,  RCC8  + ExG + Crop)", samples, blobs, keep_b)


if __name__ == "__main__":
    main()