import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE


def diagnose_false_positives(model, loader, device, min_size=5):
    model.eval()
    relation_counter = Counter()
    gt_label_counter = Counter()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for i in range(preds.shape[0]):
                pred = preds[i]
                gt = labels_np[i]

                crop_bin = (pred == 1).astype(np.uint8)
                weed_bin = (pred == 2).astype(np.uint8)

                n_crop, crop_cc = cv2.connectedComponents(crop_bin)
                n_weed, weed_cc = cv2.connectedComponents(weed_bin)
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
                    if wmask.sum() < min_size:
                        continue

                    gt_vals = gt[wmask > 0]
                    gt_majority = np.bincount(gt_vals, minlength=3).argmax()
                    if gt_majority == 2:
                        continue  #  Weed( ) False Positive  

                    wpoly = mask_to_polygon(wmask)
                    if wpoly is None:
                        continue

                    best_rel = RCC8.DC
                    for cid, cpoly in crop_polys.items():
                        rel = classify_rcc8(wpoly, cpoly)
                        if rel != RCC8.DC:
                            best_rel = rel
                            break

                    relation_counter[best_rel.name] += 1
                    gt_name = {0: "Background(Soil)", 1: "Crop"}.get(gt_majority, "Other")
                    gt_label_counter[gt_name] += 1

    return relation_counter, gt_label_counter


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))

    rel_counts, gt_counts = diagnose_false_positives(model, val_loader, device)
    total = sum(rel_counts.values())

    print(f"\n Weed False Positive  : {total}")
    print("\n[False Positive  RCC8   —   Crop  ]")
    for rel, cnt in rel_counts.most_common():
        print(f"  {rel}: {cnt} ({cnt/total*100:.1f}%)")

    print("\n[False Positive     ]")
    for label, cnt in gt_counts.most_common():
        print(f"  {label}: {cnt} ({cnt/total*100:.1f}%)")


if __name__ == "__main__":
    main()