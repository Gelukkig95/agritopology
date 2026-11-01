import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = r"D:\Download\PhenoBench-v110\PhenoBench"
IMG_SIZE = 256


class PhenoBenchDataset(Dataset):
    def __init__(self, base_dir, split="train", img_size=256):
        self.images_dir = f"{base_dir}/{split}/images"
        self.semantics_dir = f"{base_dir}/{split}/semantics"
        self.files = sorted(os.listdir(self.images_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(f"{self.images_dir}/{fname}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        semantics = cv2.imread(f"{self.semantics_dir}/{fname}", cv2.IMREAD_UNCHANGED)
        semantics = cv2.resize(semantics, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        label = np.zeros_like(semantics, dtype=np.int64)
        label[(semantics == 1) | (semantics == 3)] = 1  # Crop
        label[(semantics == 2) | (semantics == 4)] = 2  # Weed
        label = torch.from_numpy(label)

        return img, label


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )
        self.enc1 = block(in_ch, base_ch)
        self.enc2 = block(base_ch, base_ch*2)
        self.enc3 = block(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(base_ch*4, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = block(base_ch*2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def dice_loss(logits, target, num_classes=3, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_onehot, dims)
    cardinality = torch.sum(probs + target_onehot, dims)
    dice = (2. * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds = PhenoBenchDataset(BASE_DIR, "train", IMG_SIZE)
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels) + dice_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "baseline_unet.pth")
    print(" : baseline_unet.pth")


if __name__ == "__main__":
    main()

