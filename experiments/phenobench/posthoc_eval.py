import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE


def apply_posthoc_override(pred_mask, min_weed_size=5, min_crop_size=10):
    """
    pred_mask: (H,W)  , 0=Background, 1=Crop, 2=Weed
    NTPP/TPP(Crop   Weed ) Crop .
    EC/PO( ) Risky(Phase 1~2 )  .
    """
    crop_bin = (pred_mask == 1).astype(np.uint8)
    weed_bin = (pred_mask == 2).astype(np.uint8)

    n_crop, crop_labels = cv2.connectedComponents(crop_bin)
    n_weed, weed_labels = cv2.connectedComponents(weed_bin)

    if n_crop <= 1 or n_weed <= 1:
        return pred_mask.copy()

    crop_polys = {}
    for cid in range(1, n_crop):
        mask = (crop_labels == cid).astype(np.uint8)
        if mask.sum() < min_crop_size:
            continue
        poly = mask_to_polygon(mask)
        if poly is not None:
            crop_polys[cid] = poly

    corrected = pred_mask.copy()
    for wid in range(1, n_weed):
        wmask = (weed_labels == wid).astype(np.uint8)
        if wmask.sum() < min_weed_size:
            continue
        wpoly = mask_to_polygon(wmask)
        if wpoly is None:
            continue

        for cid, cpoly in crop_polys.items():
            rel = classify_rcc8(wpoly, cpoly)
            if rel in (RCC8.NTPP, RCC8.TPP):
                corrected[weed_labels == wid] = 1
                break

    return corrected


def compute_metrics(labels, preds, class_names=("Background", "Crop", "Weed")):
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    ious = []
    for c in range(len(class_names)):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        ious.append(iou)
        print(f"{class_names[c]}: IoU={iou:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    print(f"mIoU: {np.mean(ious):.3f}")
    return ious


def evaluate_with_posthoc(model, loader, device):
    model.eval()
    raw_all, corrected_all, labels_all = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(preds.shape[0]):
                raw = preds[i]
                corrected = apply_posthoc_override(raw)
                raw_all.append(raw.ravel())
                corrected_all.append(corrected.ravel())

            labels_all.append(labels.numpy().reshape(labels.shape[0], -1))

    raw_all = np.concatenate(raw_all)
    corrected_all = np.concatenate(corrected_all)
    labels_all = np.concatenate([l.ravel() for l in labels_all])

    print("\n=== Before Correction (Raw) ===")
    compute_metrics(labels_all, raw_all)

    print("\n=== After Correction (NTPP/TPP ) ===")
    compute_metrics(labels_all, corrected_all)

    n_changed = np.sum(raw_all != corrected_all)
    n_correct_fix = np.sum((raw_all != corrected_all) & (labels_all == 1))
    n_wrong_fix = np.sum((raw_all != corrected_all) & (labels_all == 2))
    print(f"\n[ ]")
    print(f"   : {n_changed}")
    if n_changed > 0:
        print(f"  -  False Positive(Crop)  : {n_correct_fix} ({n_correct_fix/n_changed*100:.1f}%)")
        print(f"  -  Weed       : {n_wrong_fix} ({n_wrong_fix/n_changed*100:.1f}%)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))

    evaluate_with_posthoc(model, val_loader, device)


if __name__ == "__main__":
    main()