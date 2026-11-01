import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE


def compute_exg_from_tensor(img_tensor):
    img = img_tensor.numpy()
    r, g, b = img[0], img[1], img[2]
    return 2 * g - r - b


def analyze_blobs(model, loader, device, min_size=5):
    model.eval()
    groups = {"TP_Weed": [], "B_BackgroundDC": [], "C_CropEC_PO": [], "D_CropDC": []}

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
                p_weed = probs[i, 2]
                p_crop = probs[i, 1]
                exg = compute_exg_from_tensor(imgs[i])

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

                kernel = np.ones((5, 5), np.uint8)

                for wid in range(1, n_weed):
                    wmask = (weed_cc == wid).astype(np.uint8)
                    if wmask.sum() < min_size:
                        continue

                    gt_vals = gt[wmask > 0]
                    gt_majority = np.bincount(gt_vals, minlength=3).argmax()

                    size = int(wmask.sum())
                    mean_pweed = float(p_weed[wmask > 0].mean())
                    dilated = cv2.dilate(wmask, kernel)
                    mean_pcrop_surround = float(p_crop[dilated > 0].mean())
                    mean_exg = float(exg[wmask > 0].mean())
                    signal = {"size": size, "mean_pweed": mean_pweed,
                              "mean_pcrop_surround": mean_pcrop_surround, "mean_exg": mean_exg}

                    if gt_majority == 2:
                        groups["TP_Weed"].append(signal)
                        continue

                    wpoly = mask_to_polygon(wmask)
                    best_rel = RCC8.DC
                    if wpoly is not None:
                        for cid, cpoly in crop_polys.items():
                            rel = classify_rcc8(wpoly, cpoly)
                            if rel != RCC8.DC:
                                best_rel = rel
                                break

                    if gt_majority == 0 and best_rel == RCC8.DC:
                        groups["B_BackgroundDC"].append(signal)
                    elif gt_majority == 1 and best_rel != RCC8.DC:
                        groups["C_CropEC_PO"].append(signal)
                    elif gt_majority == 1 and best_rel == RCC8.DC:
                        groups["D_CropDC"].append(signal)

    return groups


def print_stats(groups):
    print(f"\n{'':15s}{'':>6s}  {'':>14s}  {'p_weed':>14s}  {'p_crop':>14s}  {'ExG':>14s}")
    for name, items in groups.items():
        if not items:
            print(f"{name:15s}{'0':>6s}")
            continue
        sizes = np.array([x["size"] for x in items])
        pweeds = np.array([x["mean_pweed"] for x in items])
        pcrops = np.array([x["mean_pcrop_surround"] for x in items])
        exgs = np.array([x["mean_exg"] for x in items])
        print(f"{name:15s}{len(items):>6d}  "
              f"{sizes.mean():>6.1f}±{sizes.std():<6.1f}  "
              f"{pweeds.mean():>6.3f}±{pweeds.std():<6.3f}  "
              f"{pcrops.mean():>6.3f}±{pcrops.std():<6.3f}  "
              f"{exgs.mean():>6.3f}±{exgs.std():<6.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))

    groups = analyze_blobs(model, val_loader, device)
    print_stats(groups)


if __name__ == "__main__":
    main()