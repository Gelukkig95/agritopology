import os
import random
import cv2
import numpy as np
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


def inject_ntpp_error(crop_mask, patch_size=15):
    """Crop      ' Weed' """
    eroded = cv2.erode(crop_mask, np.ones((25, 25), np.uint8))
    ys, xs = np.where(eroded > 0)
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


def inject_ecpo_error(crop_mask, patch_size=15):
    """Crop     ' Weed' """
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


def test_override(crop_mask, fake_weed_mask):
    crop_poly = mask_to_polygon(crop_mask)
    weed_poly = mask_to_polygon(fake_weed_mask)
    if crop_poly is None or weed_poly is None:
        return None
    return classify_rcc8(weed_poly, crop_poly)


def run_injection_experiment(files, label):
    ntpp_success, ntpp_total = 0, 0
    ecpo_success, ecpo_total = 0, 0

    for f in files:
        semantics = cv2.imread(f"{semantics_dir}/{f}", cv2.IMREAD_UNCHANGED)
        plant_inst = cv2.imread(f"{plant_dir}/{f}", cv2.IMREAD_UNCHANGED)
        if semantics is None or plant_inst is None:
            continue

        plant_ids = [i for i in np.unique(plant_inst) if i != 0]
        plant_classes = {pid: classify_plant_instance(pid, plant_inst, semantics) for pid in plant_ids}
        crop_ids = [pid for pid, cls in plant_classes.items() if cls == "crop"]

        for cid in crop_ids:
            crop_mask = (plant_inst == cid).astype(np.uint8)
            if crop_mask.sum() < 500:
                continue

            fake_weed = inject_ntpp_error(crop_mask)
            if fake_weed is not None:
                rel = test_override(crop_mask, fake_weed)
                ntpp_total += 1
                if rel in (RCC8.NTPP, RCC8.TPP):
                    ntpp_success += 1

            fake_weed2 = inject_ecpo_error(crop_mask)
            if fake_weed2 is not None:
                rel2 = test_override(crop_mask, fake_weed2)
                ecpo_total += 1
                if rel2 in (RCC8.EC, RCC8.PO, RCC8.TPP, RCC8.NTPP):
                    ecpo_success += 1

    print(f"\n=== {label} ===")
    if ntpp_total:
        print(f"NTPP  : {ntpp_success}/{ntpp_total} ({ntpp_success/ntpp_total*100:.1f}%)")
    if ecpo_total:
        print(f"EC/PO  : {ecpo_success}/{ecpo_total} ({ecpo_success/ecpo_total*100:.1f}%)")


all_files = sorted(os.listdir(semantics_dir))
files_0515 = [f for f in all_files if f.split("_")[0] == "05-15"][:100]
files_0605 = [f for f in all_files if f.split("_")[0] == "06-05"][:100]

run_injection_experiment(files_0515, "05-15 ()")
run_injection_experiment(files_0605, "06-05 ()")