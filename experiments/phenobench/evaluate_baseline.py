import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE


def evaluate(model, loader, device, num_classes=3):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds.ravel())
            all_labels.append(labels.numpy().ravel())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    class_names = ["Background", "Crop", "Weed"]
    ious = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        ious.append(iou)
        print(f"{class_names[c]}: IoU={iou:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    print(f"mIoU: {np.mean(ious):.3f}")
    return ious, cm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))

    evaluate(model, val_loader, device)


if __name__ == "__main__":
    main()