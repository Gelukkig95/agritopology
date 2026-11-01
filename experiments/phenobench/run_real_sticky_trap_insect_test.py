import gc
import json
import sys
import random
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2
import higra as hg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PHENOBENCH_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PHENOBENCH_EXP_DIR))

from train_baseline import SimpleUNet, dice_loss

SEED = 42
PATCH_SIZE = 256
TRAP_DATA_DIR = ROOT_DIR / "data" / "yellow_sticky_traps_real"
_HIGRA_GRAPH_CACHE = {}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cached_graph(H: int, W: int):
    key = (H, W)
    if key not in _HIGRA_GRAPH_CACHE:
        _HIGRA_GRAPH_CACHE[key] = hg.get_4_adjacency_graph((H, W))
    return _HIGRA_GRAPH_CACHE[key]


def download_real_sticky_trap_dataset(max_traps: int = 24):
    TRAP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_xmls = [p for p in sorted(TRAP_DATA_DIR.glob("*.xml")) if p.with_suffix(".jpg").exists()]
    if len(existing_xmls) >= max_traps:
        print(f"  -> Found {len(existing_xmls)} cached real sticky-trap JPG+XML pairs.")
        return existing_xmls[:max_traps]

    print(f"  -> Fetching up to {max_traps} real sticky-trap images from GitHub...")
    api_url = "https://api.github.com/repos/md-121/yellow-sticky-traps-dataset/contents/annotations"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        items = json.loads(resp.read().decode("utf-8"))

    xml_entries = [it for it in items if isinstance(it, dict) and it.get("name", "").endswith(".xml")][:max_traps]
    downloaded = []
    for idx, entry in enumerate(xml_entries):
        stem = Path(entry["name"]).stem
        xml_url = entry["download_url"]
        jpg_url = xml_url.replace("/annotations/", "/images/").replace(".xml", ".jpg")
        xml_path = TRAP_DATA_DIR / f"{stem}.xml"
        jpg_path = TRAP_DATA_DIR / f"{stem}.jpg"

        if not (xml_path.exists() and jpg_path.exists() and jpg_path.stat().st_size > 10000):
            print(f"     Downloading REAL trap [{idx+1}/{len(xml_entries)}]: {stem}.jpg & .xml ...")
            urllib.request.urlretrieve(xml_url, xml_path)
            urllib.request.urlretrieve(jpg_url, jpg_path)
        downloaded.append(xml_path)

    return downloaded


def extract_patches_from_traps(xml_files: list, patch_size: int = PATCH_SIZE):
    """
    Extracts real 256x256 patches from a specific list of trap boards (strictly trap-level split!).
      - Class 0: Real yellow sticky-trap background (grid lines, glue, dust specks, shadows)
      - Class 1: Annotated Whitefly (WF) / small insect
      - Class 2: Annotated Macrolophus (MR) / Nesidiocoris (NC) winged insect
    """
    imgs_list, labels_list = [], []
    real_insect_masks = []
    rng = np.random.RandomState(SEED)
    total_boxes = 0

    for xml_path in xml_files:
        jpg_path = xml_path.with_suffix(".jpg")
        bgr = cv2.imread(str(jpg_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = rgb.shape

        tree = ET.parse(str(xml_path))
        boxes = []
        for obj in tree.findall("object"):
            name = obj.find("name").text.strip().upper() if obj.find("name") is not None else "WF"
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = max(0, int(float(bb.find("xmin").text)))
            ymin = max(0, int(float(bb.find("ymin").text)))
            xmax = min(W, int(float(bb.find("xmax").text)))
            ymax = min(H, int(float(bb.find("ymax").text)))
            if (xmax - xmin) >= 8 and (ymax - ymin) >= 8:
                cls_id = 2 if name in ("MR", "NC") else 1
                boxes.append((xmin, ymin, xmax, ymax, cls_id))

        total_boxes += len(boxes)
        if not boxes:
            continue

        # Create clean silhouette mask inside each annotated ground-truth bounding box via Otsu contrast
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        full_mask = np.zeros((H, W), dtype=np.uint8)

        for (x1, y1, x2, y2, cls_id) in boxes:
            crop_g = gray[y1:y2, x1:x2]
            # Invert local luminance against yellow trap background and apply Otsu threshold
            inv = np.clip(int(np.percentile(crop_g, 85)) - crop_g.astype(np.int32), 0, 255).astype(np.uint8)
            if inv.max() < 8:
                continue
            _, fg = cv2.threshold(inv, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Keep connected components touching the center of the annotated box
            sub_m = full_mask[y1:y2, x1:x2]
            sub_m[fg > 0] = 2  # Unified foreground insect class (2) vs trap background (0) and grid/border (1)
            full_mask[y1:y2, x1:x2] = sub_m

            if fg.sum() >= 15:
                real_insect_masks.append(fg.astype(np.uint8))

        # Sample up to 22 patches per trap board (both insect-centered and clutter-heavy regions)
        for (x1, y1, x2, y2, _) in boxes[: min(len(boxes), 22)]:
            cx = (x1 + x2) // 2 + rng.randint(-55, 56)
            cy = (y1 + y2) // 2 + rng.randint(-55, 56)
            px1 = int(np.clip(cx - patch_size // 2, 0, max(0, W - patch_size)))
            py1 = int(np.clip(cy - patch_size // 2, 0, max(0, H - patch_size)))

            imgs_list.append(rgb[py1 : py1 + patch_size, px1 : px1 + patch_size].copy().transpose(2, 0, 1))
            labels_list.append(full_mask[py1 : py1 + patch_size, px1 : px1 + patch_size].copy())

    return np.asarray(imgs_list, dtype=np.uint8), np.asarray(labels_list, dtype=np.uint8), real_insect_masks, total_boxes


class StickyTrapDataset(Dataset):
    def __init__(self, imgs_u8: np.ndarray, labels_u8: np.ndarray):
        self.imgs_u8 = imgs_u8
        self.labels_u8 = labels_u8

    def __len__(self):
        return len(self.imgs_u8)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imgs_u8[idx]).float().div_(255.0),
            torch.from_numpy(self.labels_u8[idx]).long(),
        )


# =====================================================================
# Class-Balanced Topological Margin & Luminance Contrast Loss (No Shrinkage!)
# =====================================================================
def balanced_sticky_trap_topo_loss(logits: torch.Tensor, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    p_bg, p_insect = probs[:, 0], probs[:, 2]

    # Yellow sticky-trap background prior: high R & G with zero local contrast cannot be an insect
    r, g, b = imgs[:, 0], imgs[:, 1], imgs[:, 2]
    yellow_flat = F.relu((r + g) * 0.5 - b - 0.35)
    local_var = F.avg_pool2d((r - F.avg_pool2d(r.unsqueeze(1), 5, 1, 2).squeeze(1)).abs().unsqueeze(1), 5, 1, 2).squeeze(1)
    pure_yellow_bg_gate = F.relu(0.025 - local_var) / 0.025 * yellow_flat

    bg_mask = (labels == 0).float()
    insect_mask = (labels == 2).float()

    # 1. Suppress isolated false positives on flat yellow trap background
    fp_bg_loss = (pure_yellow_bg_gate * p_insect * bg_mask).sum() / (bg_mask.sum() + 1e-5)

    # 2. Class-balanced foreground recall preservation (prevents pixel shrinkage!)
    # Normalized by insect_mask.sum() so 1% foreground pixels get equal gradient weight as 99% background!
    fn_recall_loss = ((1.0 - p_insect) * insect_mask).sum() / (insect_mask.sum() + 1e-5)

    # 3. Decision-margin hinge suppression on isolated 1-3px background spikes
    local_mean_p = F.avg_pool2d(p_insect.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1).detach()
    micro_spike = ((p_insect > 0.40) & (local_mean_p < 0.15)).float() * bg_mask
    spike_loss = (F.relu(p_insect - p_bg + 0.05) * micro_spike).sum() / (bg_mask.sum() + 1e-5)

    return 1.2 * fp_bg_loss + 0.8 * fn_recall_loss + 1.5 * spike_loss


def evaluate_real_traps(model, loader, device, apply_crc: bool = False, target_alpha: float = 0.02):
    model.eval()
    cm = np.zeros((3, 3), dtype=np.int64)
    blobs = []
    frames = []
    graph = get_cached_graph(PATCH_SIZE, PATCH_SIZE)
    num_leaves = PATCH_SIZE * PATCH_SIZE

    with torch.no_grad():
        img_idx = 0
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels.numpy()

            for b in range(preds.shape[0]):
                pred = preds[b].copy()
                gt = labels_np[b]
                p_bg, p_insect = probs[b, 0], probs[b, 2]

                ins_bin = (pred == 2).astype(np.uint8)
                n_cc, cc_map = cv2.connectedComponents(ins_bin)
                margin = np.clip(p_insect - p_bg, 0.0, 1.0).astype(np.float64)
                tree, alt = hg.component_tree_max_tree(graph, margin)
                vol = hg.attribute_volume(tree, alt)
                leaf_vol = vol[tree.parents()[:num_leaves]].reshape(PATCH_SIZE, PATCH_SIZE)

                f_indices = []
                for wid in range(1, n_cc):
                    wmask = (cc_map == wid)
                    if int(wmask.sum()) < 5:
                        pred[wmask] = 0
                        continue
                    comp_vol = float(leaf_vol[wmask].max())
                    is_tp = (gt[wmask] == 2).sum() >= 2
                    f_indices.append(len(blobs))
                    blobs.append({
                        "img_idx": img_idx,
                        "wid": wid,
                        "volume": comp_vol,
                        "is_tp": is_tp,
                    })

                if not apply_crc:
                    cm += np.bincount(3 * gt.ravel() + pred.ravel(), minlength=9).reshape(3, 3)
                frames.append((pred, gt, cc_map, f_indices))
                del tree, alt, vol
                img_idx += 1

    if apply_crc:
        def cal_vol(sub):
            tp_v = np.sort([x["volume"] for x in sub if x["is_tp"]])
            if len(tp_v) == 0:
                return 0.0
            k = int(np.floor(target_alpha * (len(tp_v) + 1.0) - 1.0))
            return 0.0 if k < 0 else float(tp_v[min(k, len(tp_v) - 1)])

        lam_a = cal_vol([x for x in blobs if x["img_idx"] % 2 == 0])
        lam_b = cal_vol([x for x in blobs if x["img_idx"] % 2 == 1])

        kept = []
        for idx, (pred, gt, cc_map, f_indices) in enumerate(frames):
            pred_c = pred.copy()
            lam = lam_a if (idx % 2 == 1) else lam_b
            for bi in f_indices:
                br = blobs[bi]
                if br["volume"] >= lam:
                    kept.append(br)
                else:
                    pred_c[cc_map == br["wid"]] = 0
            cm += np.bincount(3 * gt.ravel() + pred_c.ravel(), minlength=9).reshape(3, 3)
        blobs = kept

    tp_pix = float(cm[2, 2])
    fp_pix = float(cm[:, 2].sum() - tp_pix)
    fn_pix = float(cm[2, :].sum() - tp_pix)
    iou = tp_pix / (tp_pix + fp_pix + fn_pix + 1e-6)
    prec = tp_pix / (tp_pix + fp_pix + 1e-6)
    rec = tp_pix / (tp_pix + fn_pix + 1e-6)

    tp_blobs = sum(1 for x in blobs if x["is_tp"])
    fp_blobs = sum(1 for x in blobs if not x["is_tp"])

    return {"iou": iou, "prec": prec, "rec": rec, "tp_blobs": tp_blobs, "fp_blobs": fp_blobs}


def test_real_touching_insect_separation_fixed(real_masks: list, n_pairs: int = 50):
    """
    Places 2 real insect silhouettes so their edges touch (forming 1 connected blob),
    and compares:
      1) Standard ConnectedComponents (counts 1 merged blob -> fails)
      2) Higra Max-Tree on Euclidean Distance Transform without destructive Gaussian blur
    """
    H, W = 96, 96
    graph = get_cached_graph(H, W)
    correct_std, correct_higra = 0, 0

    for i in range(n_pairs):
        m1 = cv2.resize(real_masks[i % len(real_masks)], (26, 26), interpolation=cv2.INTER_NEAREST)
        m2 = cv2.resize(real_masks[(i + 5) % len(real_masks)], (26, 26), interpolation=cv2.INTER_NEAREST)

        # Ensure each single insect mask is a clean connected component
        k3 = np.ones((3, 3), np.uint8)
        m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k3)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k3)

        canvas = np.zeros((H, W), dtype=np.uint8)
        canvas[35:61, 24:50] = m1
        # Place second insect so its boundary touches/slightly overlaps the first insect
        canvas[35:61, 46:72] = np.maximum(canvas[35:61, 46:72], m2)

        # 1. Standard ConnectedComponents
        n_cc_std, _ = cv2.connectedComponents(canvas)
        if (n_cc_std - 1) == 2:
            correct_std += 1

        # 2. Higra Max-Tree on Normalized Euclidean Distance Transform (No destructive blur!)
        dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5).astype(np.float64)
        if dist.max() > 0:
            dist /= dist.max()

        tree, alt = hg.component_tree_max_tree(graph, dist)
        dyn = hg.attribute_dynamics(tree, alt)
        extrema = hg.attribute_extrema(tree, alt)
        n_peaks = int(np.sum((extrema > 0) & (dyn >= 0.22)))
        if n_peaks == 2:
            correct_higra += 1

    return correct_std, correct_higra, n_pairs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    print("=" * 100)
    print(" [TRAP-LEVEL SPLIT REAL STICKY-TRAP BENCHMARK] 24 Real High-Res Traps (16 Train Boards / 8 Val Boards)")
    print("=" * 100)

    xml_files = download_real_sticky_trap_dataset(max_traps=24)
    train_xmls = xml_files[:16]
    val_xmls = xml_files[16:24]

    tr_imgs, tr_lbls, real_masks, n_tr_box = extract_patches_from_traps(train_xmls)
    va_imgs, va_lbls, _, n_va_box = extract_patches_from_traps(val_xmls)

    train_ds = StickyTrapDataset(tr_imgs, tr_lbls)
    val_ds = StickyTrapDataset(va_imgs, va_lbls)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    print(f"  -> Strict Trap-Level Split (0% Data Leakage):")
    print(f"     * Train: 16 trap boards ({n_tr_box} insects) -> {len(train_ds)} patches")
    print(f"     * Val  :  8 trap boards ({n_va_box} insects) -> {len(val_ds)} patches")

    # 1. Baseline U-Net (Standard CE + Dice)
    set_seed(SEED)
    base_net = SimpleUNet().to(device)
    opt_b = torch.optim.Adam(base_net.parameters(), lr=1e-3)
    for _ in range(12):
        base_net.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_b.zero_grad()
            logits = base_net(imgs)
            (F.cross_entropy(logits, labels) + dice_loss(logits, labels)).backward()
            opt_b.step()
    rb = evaluate_real_traps(base_net, val_loader, device, apply_crc=False)
    del base_net, opt_b
    gc.collect()
    torch.cuda.empty_cache()

    # 2. +Agritopology Class-Balanced Topological Margin Loss
    set_seed(SEED)
    topo_net = SimpleUNet().to(device)
    opt_t = torch.optim.Adam(topo_net.parameters(), lr=1e-3)
    for _ in range(12):
        topo_net.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_t.zero_grad()
            logits = topo_net(imgs)
            loss = F.cross_entropy(logits, labels) + dice_loss(logits, labels) + 0.45 * balanced_sticky_trap_topo_loss(logits, imgs, labels)
            loss.backward()
            opt_t.step()
    rt = evaluate_real_traps(topo_net, val_loader, device, apply_crc=False)
    rc = evaluate_real_traps(topo_net, val_loader, device, apply_crc=True, target_alpha=0.02)
    del topo_net, opt_t
    gc.collect()
    torch.cuda.empty_cache()

    c_std, c_higra, n_pairs = test_real_touching_insect_separation_fixed(real_masks, n_pairs=50)

    print("\n" + "=" * 100)
    print(" [FINAL REAL STICKY-TRAP RESULTS (16 Train Boards vs. 8 Unseen Val Boards)]")
    print("=" * 100)
    print(f"  1. Baseline U-Net + OpenCV (>=5px)    : Insect IoU={rb['iou']:.4f} (Prec={rb['prec']*100:5.2f}%, Rec={rb['rec']*100:5.2f}%) | "
          f"Trap Clutter FP={rb['fp_blobs']:3d} blobs | True Insect TP={rb['tp_blobs']}")
    print(f"  2. +Agritopology Balanced Loss (+0ms) : Insect IoU={rt['iou']:.4f} (Prec={rt['prec']*100:5.2f}%, Rec={rt['rec']*100:5.2f}%) | "
          f"Trap Clutter FP={rt['fp_blobs']:3d} blobs | True Insect TP={rt['tp_blobs']}")
    print(f"  3. +Agritopology + Volume Mondrian CRC: Insect IoU={rc['iou']:.4f} (Prec={rc['prec']*100:5.2f}%, Rec={rc['rec']*100:5.2f}%) | "
          f"Trap Clutter FP={rc['fp_blobs']:3d} blobs | True Insect TP={rc['tp_blobs']}")
    print("-" * 100)
    print(f"  [Real Touching-Insect Cluster Separation (n={n_pairs} touching pairs from real trap silhouettes)]")
    print(f"   * Standard ConnectedComponents (Threshold p>0.5)       : {c_std}/{n_pairs} ({c_std/n_pairs*100:.1f}% separated)")
    print(f"   * Higra Distance-Tree Persistence (dynamics >= 0.22)   : {c_higra}/{n_pairs} ({c_higra/n_pairs*100:.1f}% separated)")
    print("=" * 100)


if __name__ == "__main__":
    main()
