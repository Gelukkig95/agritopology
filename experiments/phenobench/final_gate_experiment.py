import os
import random
import cv2
import numpy as np
import higra as hg
from sklearn.metrics import roc_auc_score, roc_curve

base = r"D:\Download\PhenoBench-v110\PhenoBench\train"
semantics_dir = f"{base}/semantics"
plant_dir = f"{base}/plant_instances"
images_dir = f"{base}/images"

CROP_LABELS = {1, 3}
WEED_LABELS = {2, 4}

GRAPH_1024 = hg.get_4_adjacency_graph((1024, 1024))


def classify_plant_instance(pid, plant_inst, semantics):
    mask = plant_inst == pid
    vals, counts = np.unique(semantics[mask], return_counts=True)
    majority = vals[np.argmax(counts)]
    if majority in CROP_LABELS:
        return "crop"
    if majority in WEED_LABELS:
        return "weed"
    return "unknown"


def compute_exg(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 2 * g - r - b


def inject_ecpo_error(crop_mask, patch_size=15):
    boundary_zone = crop_mask - cv2.erode(crop_mask, np.ones((15, 15), np.uint8))
    ys, xs = np.where(boundary_zone > 0)
    if len(ys) == 0:
        return None
    idx = random.randrange(len(ys))
    cy, cx = ys[idx], xs[idx]
    half = patch_size // 2
    y0, y1 = max(0, cy - half), min(crop_mask.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(crop_mask.shape[1], cx + half)
    fake_weed = np.zeros_like(crop_mask)
    fake_weed[y0:y1, x0:x1] = crop_mask[y0:y1, x0:x1]
    return fake_weed if fake_weed.sum() > 10 else None


def merge_altitude_from_masks(tree, altitudes, flat_exg, mask_a, mask_b, dilate_px=3):
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    dil_a = cv2.dilate(mask_a, kernel)
    dil_b = cv2.dilate(mask_b, kernel)
    if not np.any(dil_a & dil_b):
        return None
    idx_a = np.flatnonzero(mask_a)
    idx_b = np.flatnonzero(mask_b)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None
    peak_a = int(idx_a[np.argmax(flat_exg[idx_a])])
    peak_b = int(idx_b[np.argmax(flat_exg[idx_b])])
    lca_node = tree.lowest_common_ancestor(peak_a, peak_b)
    return float(altitudes[lca_node])


def process_file_final(filename):
    semantics = cv2.imread(f"{semantics_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    plant_inst = cv2.imread(f"{plant_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    img = cv2.imread(f"{images_dir}/{filename}", cv2.IMREAD_COLOR)
    if semantics is None or plant_inst is None or img is None:
        return [], []

    exg = compute_exg(img)
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    fg_mask = cv2.dilate((plant_inst > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    exg_masked = np.where(fg_mask > 0, exg_norm, 0.0).astype(np.float64)

    graph = GRAPH_1024 if exg_masked.shape == (1024, 1024) else hg.get_4_adjacency_graph(exg_masked.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, exg_masked)
    flat_exg = exg_masked.ravel()

    plant_ids = [i for i in np.unique(plant_inst) if i != 0]
    if not plant_ids:
        return [], []

    plant_classes = {pid: classify_plant_instance(pid, plant_inst, semantics) for pid in plant_ids}
    masks = {pid: (plant_inst == pid).astype(np.uint8) for pid in plant_ids}

    # (A)  Weed-Crop  — override    (label=0)
    real_scores = []
    weed_ids = [pid for pid, cls in plant_classes.items() if cls == "weed"]
    crop_ids = [pid for pid, cls in plant_classes.items() if cls == "crop"]
    for wid in weed_ids:
        for cid in crop_ids:
            val = merge_altitude_from_masks(tree, altitudes, flat_exg, masks[wid], masks[cid])
            if val is not None and val > 0:
                real_scores.append(val)

    # (B)    — override    (label=1)
    fake_scores = []
    for cid in crop_ids:
        if masks[cid].sum() < 500:
            continue
        fake_weed = inject_ecpo_error(masks[cid])
        if fake_weed is not None:
            val = merge_altitude_from_masks(tree, altitudes, flat_exg, fake_weed, masks[cid])
            if val is not None and val > 0:
                fake_scores.append(val)

    return real_scores, fake_scores


all_files = sorted(os.listdir(semantics_dir))
files_0605 = [f for f in all_files if f.split("_")[0] == "06-05"]
print(f"06-05 Files: {len(files_0605)}")

real_all, fake_all = [], []
for idx, f in enumerate(files_0605):
    r, fk = process_file_final(f)
    real_all.extend(r)
    fake_all.extend(fk)
    if (idx + 1) % 50 == 0:
        print(f"{idx+1}/{len(files_0605)} (Weed {len(real_all)},  {len(fake_all)})")

print(f"\n Weed-Crop (override  ): Mean {np.mean(real_all):.3f} (n={len(real_all)})")
print(f"  (override  ): Mean {np.mean(fake_all):.3f} (n={len(fake_all)})")

labels = [0]*len(real_all) + [1]*len(fake_all)
scores = real_all + fake_all
auc = roc_auc_score(labels, scores)
print(f"\nAUC (Weed vs  ): {auc:.3f}")

fpr, tpr, thresholds = roc_curve(labels, scores)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f" : {best_threshold:.3f}")
print(f"   : ()={tpr[best_idx]*100:.1f}%,  ( Risky)={fpr[best_idx]*100:.1f}%")
print(f"(:    override Risky 31.0%)")

print("\n===    ===")
for target_recall in [0.5, 0.7, 0.8, 0.9]:
    idx = np.argmin(np.abs(tpr - target_recall))
    print(f" {tpr[idx]*100:.1f}%  → Risky {fpr[idx]*100:.1f}% ( {thresholds[idx]:.3f})")

print()
for target_fpr in [0.05, 0.10, 0.15, 0.20, 0.25]:
    idx = np.argmin(np.abs(fpr - target_fpr))
    print(f"Risky {fpr[idx]*100:.1f}%  →  {tpr[idx]*100:.1f}% ( {thresholds[idx]:.3f})")