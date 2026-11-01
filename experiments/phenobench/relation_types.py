import os
import cv2
import numpy as np
from collections import Counter
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

base = r"D:\Download\PhenoBench-v110\PhenoBench\train"
semantics_dir = f"{base}/semantics"
plant_dir = f"{base}/plant_instances"

CROP_LABELS = {1, 3}
WEED_LABELS = {2, 4}


def classify_plant_instance(pid, plant_inst, semantics):
    mask = plant_inst == pid
    vals, counts = np.unique(semantics[mask], return_counts=True)
    majority = vals[np.argmax(counts)]
    if majority in CROP_LABELS:
        return "crop"
    if majority in WEED_LABELS:
        return "weed"
    return "unknown"


def process_file_relation_types(filename):
    semantics = cv2.imread(f"{semantics_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    plant_inst = cv2.imread(f"{plant_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    if semantics is None or plant_inst is None:
        return []

    plant_ids = [i for i in np.unique(plant_inst) if i != 0]
    if not plant_ids:
        return []

    plant_classes = {pid: classify_plant_instance(pid, plant_inst, semantics) for pid in plant_ids}
    weed_ids = [pid for pid, cls in plant_classes.items() if cls == "weed"]
    crop_ids = [pid for pid, cls in plant_classes.items() if cls == "crop"]
    if not weed_ids or not crop_ids:
        return []

    plant_polygons = {}
    for pid in weed_ids + crop_ids:
        mask = (plant_inst == pid).astype(np.uint8)
        poly = mask_to_polygon(mask)
        if poly is not None:
            plant_polygons[pid] = poly

    relations = []
    for wid in weed_ids:
        if wid not in plant_polygons:
            continue
        for cid in crop_ids:
            if cid not in plant_polygons:
                continue
            rel = classify_rcc8(plant_polygons[wid], plant_polygons[cid])
            if rel != RCC8.DC:
                relations.append(rel)
    return relations


all_files = sorted(os.listdir(semantics_dir))
files_0605 = [f for f in all_files if f.split("_")[0] == "06-05"]

all_relations = []
for idx, f in enumerate(files_0605):
    all_relations.extend(process_file_relation_types(f))
    if (idx + 1) % 50 == 0:
        print(f"{idx+1}/{len(files_0605)} Processed... ( {len(all_relations)})")

counter = Counter([r.name for r in all_relations])
print("\n06-05 Risky  RCC8   :")
for rel_name, count in counter.most_common():
    pct = count / len(all_relations) * 100
    print(f"  {rel_name}: {count} ({pct:.1f}%)")