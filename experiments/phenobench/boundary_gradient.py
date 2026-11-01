import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

base = r"D:\Download\PhenoBench-v110\PhenoBench\train"
semantics_dir = f"{base}/semantics"
plant_dir = f"{base}/plant_instances"
images_dir = f"{base}/images"

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


def boundary_gradient_strength(boundary_geom, sobel_mag, img_shape):
    if boundary_geom.is_empty:
        return None
    if boundary_geom.geom_type == "MultiLineString":
        coords = []
        for line in boundary_geom.geoms:
            coords.extend(list(line.coords))
    elif boundary_geom.geom_type == "LineString":
        coords = list(boundary_geom.coords)
    elif boundary_geom.geom_type == "Point":
        coords = [(boundary_geom.x, boundary_geom.y)]
    elif boundary_geom.geom_type == "MultiPoint":
        coords = [(p.x, p.y) for p in boundary_geom.geoms]
    else:
        return None

    H, W = img_shape
    values = []
    for x, y in coords:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < H and 0 <= xi < W:
            values.append(sobel_mag[yi, xi])
    if not values:
        return None
    return float(np.mean(values))


def process_file_gradients(filename):
    semantics = cv2.imread(f"{semantics_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    plant_inst = cv2.imread(f"{plant_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    img = cv2.imread(f"{images_dir}/{filename}", cv2.IMREAD_COLOR)
    if semantics is None or plant_inst is None or img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

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

    gradients = []
    for wid in weed_ids:
        if wid not in plant_polygons:
            continue
        for cid in crop_ids:
            if cid not in plant_polygons:
                continue
            rel = classify_rcc8(plant_polygons[wid], plant_polygons[cid])
            if rel == RCC8.DC:
                continue
            boundary_intersection = plant_polygons[wid].boundary.intersection(plant_polygons[cid].boundary)
            grad_value = boundary_gradient_strength(boundary_intersection, sobel_mag, gray.shape)
            if grad_value is not None:
                gradients.append(grad_value)
    return gradients


def process_file_cropcrop_gradients(filename):
    semantics = cv2.imread(f"{semantics_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    plant_inst = cv2.imread(f"{plant_dir}/{filename}", cv2.IMREAD_UNCHANGED)
    img = cv2.imread(f"{images_dir}/{filename}", cv2.IMREAD_COLOR)
    if semantics is None or plant_inst is None or img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    plant_ids = [i for i in np.unique(plant_inst) if i != 0]
    if not plant_ids:
        return []

    plant_classes = {pid: classify_plant_instance(pid, plant_inst, semantics) for pid in plant_ids}
    crop_ids = [pid for pid, cls in plant_classes.items() if cls == "crop"]
    if len(crop_ids) < 2:
        return []

    plant_polygons = {}
    for pid in crop_ids:
        mask = (plant_inst == pid).astype(np.uint8)
        poly = mask_to_polygon(mask)
        if poly is not None:
            plant_polygons[pid] = poly

    gradients = []
    for i, cid1 in enumerate(crop_ids):
        if cid1 not in plant_polygons:
            continue
        for cid2 in crop_ids[i + 1:]:
            if cid2 not in plant_polygons:
                continue
            rel = classify_rcc8(plant_polygons[cid1], plant_polygons[cid2])
            if rel == RCC8.DC:
                continue
            boundary_intersection = plant_polygons[cid1].boundary.intersection(plant_polygons[cid2].boundary)
            grad_value = boundary_gradient_strength(boundary_intersection, sobel_mag, gray.shape)
            if grad_value is not None:
                gradients.append(grad_value)
    return gradients


# ────────────────────────────────────────────
#  
# ────────────────────────────────────────────

all_files = sorted(os.listdir(semantics_dir))
files_0605 = [f for f in all_files if f.split("_")[0] == "06-05"]
print(f"06-05  Files: {len(files_0605)}")

# (1) Weed-Crop  (06-05)
all_gradients = []
for idx, f in enumerate(files_0605):
    grads = process_file_gradients(f)
    all_gradients.extend(grads)
    if (idx + 1) % 50 == 0:
        print(f"{idx+1}/{len(files_0605)} Processed... (Weed-Crop  {len(all_gradients)})")

print(f"\n   Samples: {len(all_gradients)}")
if all_gradients:
    print(f"Mean: {np.mean(all_gradients):.2f}, Std: {np.std(all_gradients):.2f}")

plt.figure(figsize=(6, 4))
plt.hist(all_gradients, bins=30, color="steelblue", edgecolor="black")
plt.xlabel("Boundary gradient magnitude (Sobel)")
plt.ylabel("Count")
plt.title("Boundary Gradient at Weed-Crop Contact (06-05)")
plt.savefig("boundary_gradient_hist.png", dpi=150)
plt.close()

# (2) Weed-Crop  (05-15, )
files_0515 = [f for f in all_files if f.split("_")[0] == "05-15"]
print(f"\n05-15  Files: {len(files_0515)}")

gradients_0515 = []
for idx, f in enumerate(files_0515):
    grads = process_file_gradients(f)
    gradients_0515.extend(grads)
    if (idx + 1) % 50 == 0:
        print(f"{idx+1}/{len(files_0515)} Processed... (  {len(gradients_0515)})")

print(f"05-15 Samples: {len(gradients_0515)}")
if gradients_0515:
    print(f"05-15 Mean: {np.mean(gradients_0515):.2f}, Std: {np.std(gradients_0515):.2f}")

stat, p_value = mannwhitneyu(all_gradients, gradients_0515, alternative='two-sided')
print(f"\n[05-15 vs 06-05] Mann-Whitney U: statistic={stat:.2f}, p-value={p_value:.4f}")
print(f"06-05 Median: {np.median(all_gradients):.2f}")
print(f"05-15 Median: {np.median(gradients_0515):.2f}")

# (3) Crop-Crop  (06-05  ,   )
cropcrop_gradients_0605 = []
for idx, f in enumerate(files_0605):
    grads = process_file_cropcrop_gradients(f)
    cropcrop_gradients_0605.extend(grads)
    if (idx + 1) % 50 == 0:
        print(f"{idx+1}/{len(files_0605)} Processed... (Crop-Crop  {len(cropcrop_gradients_0605)})")

print(f"\n06-05 Crop-Crop Samples: {len(cropcrop_gradients_0605)}")
if cropcrop_gradients_0605:
    print(f"Crop-Crop Median: {np.median(cropcrop_gradients_0605):.2f}")

if len(cropcrop_gradients_0605) > 0:
    stat2, p_value2 = mannwhitneyu(all_gradients, cropcrop_gradients_0605, alternative='two-sided')
    print(f"[Crop-Crop vs Weed-Crop, 06-05 ] U: p-value={p_value2:.4f}")

# (4)  
plt.figure(figsize=(7, 5))
plt.hist(all_gradients, bins=30, alpha=0.5, label="06-05 weed-crop (risky)", color="red")
plt.hist(gradients_0515, bins=30, alpha=0.5, label="05-15 weed-crop (safe)", color="green")
if cropcrop_gradients_0605:
    plt.hist(cropcrop_gradients_0605, bins=30, alpha=0.5, label="06-05 crop-crop", color="blue")
plt.xlabel("Boundary gradient magnitude (Sobel)")
plt.ylabel("Count")
plt.title("Boundary Gradient Comparison")
plt.legend()
plt.savefig("boundary_gradient_compare.png", dpi=150)
plt.show()