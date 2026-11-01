import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE


def compute_exg_from_tensor(img_tensor):
    img = img_tensor.numpy()
    r, g, b = img[0], img[1], img[2]
    return 2 * g - r - b


def evaluate_predictions(name, all_labels, all_preds, fp_Before=None, fp_After=None):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    c = 2
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    print(f"\n=== [{name}] ===")
    print(f"• Weed  : IoU={iou:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")
    if fp_Before is not None and fp_After is not None:
        removed = fp_Before - fp_After
        print(f"•  (FP ) : {fp_Before} -> {fp_After} ({removed} , -{removed/fp_Before*100:.1f}%)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))
    model.eval()

    raw_preds_list, pruned_preds_list, boosted_preds_list, labels_list = [], [], [], []
    fp_blobs_raw, fp_blobs_pruned = 0, 0

    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_5 = np.ones((5, 5), np.uint8)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs_gpu = imgs.to(device)
            logits = model(imgs_gpu)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels.numpy()

            for i in range(preds.shape[0]):
                raw = preds[i].copy()
                pruned = preds[i].copy()
                gt = labels_np[i]
                p_bg, p_crop, p_weed = probs[i, 0], probs[i, 1], probs[i, 2]
                exg = compute_exg_from_tensor(imgs[i])

                crop_bin = (raw == 1).astype(np.uint8)
                weed_bin = (raw == 2).astype(np.uint8)
                n_crop, crop_cc = cv2.connectedComponents(crop_bin)
                n_weed, weed_cc = cv2.connectedComponents(weed_bin)

                crop_polys = {}
                for cid in range(1, n_crop):
                    m = (crop_cc == cid).astype(np.uint8)
                    if m.sum() >= 10:
                        poly = mask_to_polygon(m)
                        if poly is not None:
                            crop_polys[cid] = poly

                # 1:  Weed ,    RCC8   
                for wid in range(1, n_weed):
                    wmask = (weed_cc == wid).astype(np.uint8)
                    size = int(wmask.sum())
                    if size < 5:
                        continue

                    gt_majority = np.bincount(gt[wmask > 0], minlength=3).argmax()
                    is_fp = (gt_majority != 2)
                    if is_fp:
                        fp_blobs_raw += 1

                    # :  30   Weed    
                    if size >= 30:
                        if is_fp:
                            fp_blobs_pruned += 1
                        continue

                    mean_pweed = float(p_weed[wmask > 0].mean())
                    mean_exg = float(exg[wmask > 0].mean())
                    dilated = cv2.dilate(wmask, kernel_5)
                    mean_pcrop_surr = float(p_crop[dilated > 0].mean())

                    wpoly = mask_to_polygon(wmask)
                    best_rel = RCC8.DC
                    if wpoly is not None:
                        for cid, cpoly in crop_polys.items():
                            rel = classify_rcc8(wpoly, cpoly)
                            if rel != RCC8.DC:
                                best_rel = rel
                                break

                    remove_blob = False
                    if best_rel == RCC8.DC:
                        #   : (ExG)      
                        if mean_exg < 0.095 or (mean_pweed < 0.82 and mean_exg < 0.12):
                            remove_blob = True
                    else:
                        # Crop (EC/PO/NTPP)  :  Crop     Crop 
                        if mean_pcrop_surr > 0.12 and mean_pweed < 0.88:
                            remove_blob = True

                    if remove_blob:
                        pruned[wmask > 0] = np.where(p_crop[wmask > 0] > p_bg[wmask > 0], 1, 0)
                    else:
                        if is_fp:
                            fp_blobs_pruned += 1

                # 2:   Weed(DC)    Weed   (Topological Shell Recovery)
                boosted = pruned.copy()
                surviving_weed = (pruned == 2).astype(np.uint8)
                dilated_weed = cv2.dilate(surviving_weed, kernel_3)
                shell = (dilated_weed > 0) & (pruned == 0)  # Weed   Background(Soil)  

                #    ExG()   Weed    Crop    
                recover_mask = shell & (exg > 0.14) & (p_weed > 0.22) & (p_crop < 0.15)
                boosted[recover_mask] = 2

                raw_preds_list.append(raw.ravel())
                pruned_preds_list.append(pruned.ravel())
                boosted_preds_list.append(boosted.ravel())
                labels_list.append(gt.ravel())

    all_labels = np.concatenate(labels_list)
    evaluate_predictions("0. No Correction (Raw Baseline)", all_labels, np.concatenate(raw_preds_list))
    evaluate_predictions("1.    RCC8+ExG   ( )",
                         all_labels, np.concatenate(pruned_preds_list), fp_blobs_raw, fp_blobs_pruned)
    evaluate_predictions("2. ★   +    (  + IoU )",
                         all_labels, np.concatenate(boosted_preds_list), fp_blobs_raw, fp_blobs_pruned)


if __name__ == "__main__":
    main()