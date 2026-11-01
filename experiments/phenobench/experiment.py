import os
import cv2
import numpy as np
from collections import defaultdict
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


def process_file(filename):
    semantics = cv2.imread(f"{semantics_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    plant_inst = cv2.imread(f"{plant_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    if semantics is None or plant_inst is None:
        return 0, 0

    plant_ids = [i for i in np.unique(plant_inst) if i != 0]
    if not plant_ids:
        return 0, 0

    plant_classes = {pid: classify_plant_instance(pid, plant_inst, semantics) for pid in plant_ids}
    weed_ids = [pid for pid, cls in plant_classes.items() if cls == "weed"]
    crop_ids = [pid for pid, cls in plant_classes.items() if cls == "crop"]

    if not weed_ids or not crop_ids:
        return len(weed_ids), 0

    plant_polygons = {}
    for pid in weed_ids + crop_ids:
        mask = (plant_inst == pid).astype(np.uint8)
        poly = mask_to_polygon(mask)
        if poly is not None:
            plant_polygons[pid] = poly

    risky = 0
    for wid in weed_ids:
        if wid not in plant_polygons:
            continue
        for cid in crop_ids:
            if cid not in plant_polygons:
                continue
            rel = classify_rcc8(plant_polygons[wid], plant_polygons[cid])
            if rel != RCC8.DC:
                risky += 1
                break

    return len(weed_ids), risky


files = sorted(os.listdir(semantics_dir))
print(f" {len(files)}  ")

date_stats = defaultdict(lambda: {"weeds": 0, "risky": 0})
total_weeds = 0
total_risky = 0

for idx, f in enumerate(files):
    date_prefix = f.split("_")[0]  # "05-15" 
    n_weeds, n_risky = process_file(f)

    date_stats[date_prefix]["weeds"] += n_weeds
    date_stats[date_prefix]["risky"] += n_risky
    total_weeds += n_weeds
    total_risky += n_risky

    if (idx + 1) % 100 == 0:
        print(f"{idx+1}/{len(files)} Processed...")

print(f"\n===  Risky Ratio ===")
for date, stats in sorted(date_stats.items()):
    pct = stats["risky"] / stats["weeds"] * 100 if stats["weeds"] > 0 else 0
    print(f"{date}: Weed {stats['weeds']}, Risky {stats['risky']}, Ratio {pct:.1f}%")

print(f"\n=== Total ===")
print(f"Weed {total_weeds}, Risky {total_risky}, Ratio {total_risky/total_weeds*100:.2f}%")