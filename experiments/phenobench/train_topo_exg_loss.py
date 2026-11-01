import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE, dice_loss

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dilate_prob(p, kernel_size=5):
    pad = kernel_size // 2
    return F.max_pool2d(p, kernel_size=kernel_size, stride=1, padding=pad)


def erode_prob(p, kernel_size=5):
    pad = kernel_size // 2
    return -F.max_pool2d(-p, kernel_size=kernel_size, stride=1, padding=pad)


def topo_exg_guided_loss(logits, imgs, labels):
    """
    False Positive     In-Training ·  :
    1. ExG-Gated Soil Suppression (B 76.8%  False Positive ):
       (ExG = 2G - R - B)    p_weed   
    2. Morphological Closing & Dilation Crop-Zone Suppression (C/D 23.2% Crop False Positive ):
       5x5  (Closing)     Crop / Weed False Positive 
    """
    probs = F.softmax(logits, dim=1)
    p_crop = probs[:, 1]
    p_weed = probs[:, 2]

    r, g, b = imgs[:, 0], imgs[:, 1], imgs[:, 2]
    exg = 2.0 * g - r - b

    non_weed_mask = (labels != 2).float()

    # 1. (-ExG) Weed False Positive 
    soil_gate = F.relu(0.10 - exg) / 0.10
    loss_soil_fp = (soil_gate * p_weed * non_weed_mask).mean()

    # 2.  (Closing)  Crop    False Positive 
    gt_crop_float = (labels == 1).float().unsqueeze(1)
    dilated_crop_zone = dilate_prob(gt_crop_float, kernel_size=5).squeeze(1)
    closed_pred_crop = erode_prob(dilate_prob(p_crop.unsqueeze(1), 5), 5).squeeze(1).detach()

    crop_zone_weight = torch.clamp(0.5 * dilated_crop_zone + 0.5 * closed_pred_crop, 0.0, 1.0)
    loss_crop_fp = (crop_zone_weight * p_weed * non_weed_mask).mean()

    # 3.    Weed(TP)  Recall  
    weed_mask = (labels == 2).float()
    green_gate = torch.clamp(exg / 0.15, 0.0, 1.0)
    loss_weed_recall = (green_gate * (1.0 - p_weed) * weed_mask).mean()

    return loss_soil_fp + loss_crop_fp + 0.5 * loss_weed_recall


def evaluate_pixel_and_blobs(name, model, loader, device, min_size=5):
    model.eval()
    all_preds, all_labels = [], []
    tp_blobs, fp_blobs_soil, fp_blobs_crop = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs_gpu = imgs.to(device)
            logits = model(imgs_gpu)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.numpy()

            for i in range(preds.shape[0]):
                pred, gt = preds[i], labels_np[i]
                all_preds.append(pred.ravel())
                all_labels.append(gt.ravel())

                weed_bin = (pred == 2).astype(np.uint8)
                n_weed, weed_cc = cv2.connectedComponents(weed_bin)
                for wid in range(1, n_weed):
                    wmask = (weed_cc == wid)
                    if wmask.sum() < min_size:
                        continue
                    gt_majority = np.bincount(gt[wmask], minlength=3).argmax()
                    if gt_majority == 2:
                        tp_blobs += 1
                    elif gt_majority == 0:
                        fp_blobs_soil += 1
                    else:
                        fp_blobs_crop += 1

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    print(f"\n=== [{name}] ===")
    ious = []
    for c, cname in enumerate(["Background", "Crop", "Weed"]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        ious.append(iou)
        print(f"  {cname}: IoU={iou:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
    print(f"  mIoU: {np.mean(ious):.4f}")
    print(f"  [  KPI]  False Positive(FP) : {fp_blobs_soil + fp_blobs_crop} "
          f"( : {fp_blobs_soil}, Crop : {fp_blobs_crop}) |  Weed(TP) : {tp_blobs}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} (SEED={SEED}   )")

    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # 1.  Baseline U-Net 
    base_model = SimpleUNet().to(device)
    base_model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))
    evaluate_pixel_and_blobs("0.  Baseline U-Net  (CE + Dice)", base_model, val_loader, device)

    # 2.  SEED=42 Topo-ExG Loss  
    set_seed(SEED)
    train_ds = PhenoBenchDataset(BASE_DIR, "train", IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)

    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_topo = 0.6

    print("\n[Topo-ExG In-Training Loss 20   ...]")
    for epoch in range(20):
        model.train()
        tot_main, tot_topo = 0.0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)

            main_loss = F.cross_entropy(logits, labels) + dice_loss(logits, labels)
            t_loss = topo_exg_guided_loss(logits, imgs, labels)
            loss = main_loss + lambda_topo * t_loss

            loss.backward()
            optimizer.step()
            tot_main += main_loss.item()
            tot_topo += t_loss.item()

        n = len(train_loader)
        print(f"Epoch {epoch+1:2d}/20: main_loss={tot_main/n:.4f}, topo_exg_penalty={tot_topo/n:.4f}")

    torch.save(model.state_dict(), "topo_exg_unet.pth")
    evaluate_pixel_and_blobs("1. ★ Topo-ExG Loss  U-Net (  0.0ms)", model, val_loader, device)


if __name__ == "__main__":
    main()