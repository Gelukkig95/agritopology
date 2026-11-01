import os
import cv2
import numpy as np
import higra as hg
from scipy.stats import mannwhitneyu
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

base = r"D:\Download\PhenoBench-v110\PhenoBench\train"
semantics_dir = f"{base}/semantics"
plant_dir = f"{base}/plant_instances"
images_dir = f"{base}/images"

CROP_LABELS = {1, 3}
WEED_LABELS = {2, 4}

# 1024x1024  4-      1   ( )
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


def maxtree_merge_altitude(tree, altitudes, flat_exg, mask_a, mask_b, dilate_px=3):
    """
          Max-tree LCA() 
           (Merge Altitude) 
    """
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    dil_a = cv2.dilate(mask_a, kernel)
    dil_b = cv2.dilate(mask_b, kernel)

    #        
    if not np.any(dil_a & dil_b):
        return None

    #  A  B ExG     (Peak pixel) 1 
    idx_a = np.flatnonzero(mask_a)
    idx_b = np.flatnonzero(mask_b)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None

    peak_a = int(idx_a[np.argmax(flat_exg[idx_a])])
    peak_b = int(idx_b[np.argmax(flat_exg[idx_b])])

    # Max-tree    (LCA)    (altitude) 
    lca_node = tree.lowest_common_ancestor(peak_a, peak_b)
    return float(altitudes[lca_node])


def process_file_higra(filename, same_class=False):
    semantics = cv2.imread(f"{semantics_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    plant_inst = cv2.imread(f"{plant_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    img = cv2.imread(f"{images_dir}/{filename}", cv2.IMREAD_COLOR)
    if semantics is None or plant_inst is None or img is None:
        return []

    exg = compute_exg(img)
    exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)

    # Background(Soil)            0 
    fg_mask = cv2.dilate((plant_inst > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    exg_masked = np.where(fg_mask > 0, exg_norm, 0.0).astype(np.float64)

    # ★ :  Total   Max-tree  1 !
    graph = GRAPH_1024 if exg_masked.shape == (1024, 1024) else hg.get_4_adjacency_graph(exg_masked.shape)
    tree, altitudes = hg.component_tree_max_tree(graph, exg_masked)
    flat_exg = exg_masked.ravel()

    plant_ids = [i for i in np.unique(plant_inst) if i != 0]
    if not plant_ids:
        return []

    plant_classes = {pid: classify_plant_instance(pid, plant_inst, semantics) for pid in plant_ids}
    masks = {pid: (plant_inst == pid).astype(np.uint8) for pid in plant_ids}

    scores = []
    if same_class:
        crop_ids = [pid for pid, cls in plant_classes.items() if cls == "crop"]
        for i, cid1 in enumerate(crop_ids):
            for cid2 in crop_ids[i + 1:]:
                val = maxtree_merge_altitude(tree, altitudes, flat_exg, masks[cid1], masks[cid2])
                if val is not None and val > 0:
                    scores.append(val)
    else:
        weed_ids = [pid for pid, cls in plant_classes.items() if cls == "weed"]
        crop_ids = [pid for pid, cls in plant_classes.items() if cls == "crop"]
        for wid in weed_ids:
            for cid in crop_ids:
                val = maxtree_merge_altitude(tree, altitudes, flat_exg, masks[wid], masks[cid])
                if val is not None and val > 0:
                    scores.append(val)
    return scores


all_files = sorted(os.listdir(semantics_dir))
files_0605 = [f for f in all_files if f.split("_")[0] == "06-05"]
print(f"06-05 Files: {len(files_0605)}")

weedcrop_higra = []
for idx, f in enumerate(files_0605):
    weedcrop_higra.extend(process_file_higra(f, same_class=False))
    if (idx + 1) % 50 == 0:
        print(f"{idx+1}/{len(files_0605)} (Weed-Crop  {len(weedcrop_higra)})")

cropcrop_higra = []
for idx, f in enumerate(files_0605):
    cropcrop_higra.extend(process_file_higra(f, same_class=True))
    if (idx + 1) % 50 == 0:
        print(f"{idx+1}/{len(files_0605)} (Crop-Crop  {len(cropcrop_higra)})")

print(f"\n[ Higra Max-tree (LCA  ) ]")
print(f"Weed-Crop: Mean {np.mean(weedcrop_higra):.3f}, Median {np.median(weedcrop_higra):.3f} (n={len(weedcrop_higra)})")
print(f"Crop-Crop: Mean {np.mean(cropcrop_higra):.3f}, Median {np.median(cropcrop_higra):.3f} (n={len(cropcrop_higra)})")

stat, p = mannwhitneyu(weedcrop_higra, cropcrop_higra, alternative='two-sided')
print(f"U: p-value={p:.4f}")

pooled_std = np.sqrt((np.var(weedcrop_higra) + np.var(cropcrop_higra)) / 2)
cohens_d = (np.mean(cropcrop_higra) - np.mean(weedcrop_higra)) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")


from sklearn.metrics import roc_auc_score

labels = [0]*len(weedcrop_higra) + [1]*len(cropcrop_higra)
scores = weedcrop_higra + cropcrop_higra
auc = roc_auc_score(labels, scores)
print(f"AUC: {auc:.3f}")