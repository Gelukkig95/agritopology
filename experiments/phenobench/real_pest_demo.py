import os
import cv2
import numpy as np
import higra as hg
import matplotlib.pyplot as plt

BASE_TRAIN = r"D:\Download\PhenoBench-v110\PhenoBench\train"
LEAF_DIR = f"{BASE_TRAIN}/leaf_instances"
IMG_DIR = f"{BASE_TRAIN}/images"


def compute_exg_norm(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2.0 * g - r - b
    return (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)


def inject_realistic_damage_on_real_leaf(raw_leaf, clean_mask, exg_norm, n_holes=3, n_shadows=4):
    """
     (leaf_instances)   :
    1.     ((clean_mask > 0) & (raw_leaf == 0)) 5x5     ExG Median 
    2.   (±0.08) +   (  ExG) +  ( ) 
    """
    H, W = clean_mask.shape
    inner = cv2.erode(raw_leaf, np.ones((5, 5), np.uint8))
    leaf_pixels = exg_norm[inner > 0]
    bg_pixels = exg_norm[clean_mask == 0]
    if len(leaf_pixels) < 300 or len(bg_pixels) < 500:
        return None

    leaf_med = float(np.median(leaf_pixels))
    bg_med = float(np.median(bg_pixels))
    field = exg_norm.copy()

    # [ ]      (5x5)      Median 
    natural_gaps = ((clean_mask > 0) & (raw_leaf == 0)).astype(np.uint8)
    if natural_gaps.sum() > 0:
        gap_with_rim = cv2.dilate(natural_gaps, np.ones((5, 5), np.uint8))
        field[(gap_with_rim > 0) & (clean_mask > 0)] = leaf_med

    #         
    eroded_clean = cv2.erode(clean_mask, np.ones((7, 7), np.uint8))
    rim_mask = (clean_mask > 0) & (eroded_clean == 0)
    field[rim_mask] = np.maximum(field[rim_mask], leaf_med)

    x_grid = np.linspace(-0.08, 0.08, W, dtype=np.float32)
    field = np.clip(field + np.tile(x_grid, (H, 1)), 0.0, 1.0)

    safe_zone = cv2.erode(clean_mask, np.ones((19, 19), np.uint8))
    ys, xs = np.where(safe_zone > 0)
    if len(ys) < 80:
        return None

    used = []
    def pick_pt(min_d=18):
        for _ in range(60):
            idx = np.random.randint(len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            if all((cy - uy) ** 2 + (cx - ux) ** 2 >= min_d ** 2 for uy, ux in used):
                used.append((cy, cx))
                return cy, cx
        return None

    gt_mask = np.zeros_like(clean_mask)
    actual_holes = 0
    for _ in range(n_holes):
        pt = pick_pt()
        if pt is None:
            continue
        cy, cx = pt
        r = int(np.random.randint(4, 7))
        cv2.circle(field, (cx, cy), r, bg_med, -1)
        cv2.circle(gt_mask, (cx, cy), r, 1, -1)
        actual_holes += 1

    for _ in range(n_shadows):
        pt = pick_pt()
        if pt is None:
            continue
        cy, cx = pt
        r = int(np.random.randint(4, 7))
        stain = np.zeros_like(clean_mask)
        cv2.circle(stain, (cx, cy), r, 1, -1)
        drop = float(np.random.uniform(0.13, 0.17))
        field[stain > 0] = np.clip(field[stain > 0] - drop, 0.0, 1.0)

    return field, gt_mask, actual_holes


def detect_area_filter_mask(field, clean_mask, min_area=15):
    leaf_vals = field[clean_mask > 0]
    thr = float(np.median(leaf_vals) - 0.13)
    defect_bin = ((field < thr) & (clean_mask > 0)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_bin)
    out = np.zeros_like(clean_mask)
    cnt = 0
    for i in range(1, n_labels):
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= 350:
            out[labels == i] = 1
            cnt += 1
    return out, cnt


def detect_higra_tda_mask(field, clean_mask, min_area=15, persistence_thr=0.20):
    smoothed = cv2.GaussianBlur(field, (3, 3), 0)
    inside_field = np.where(clean_mask > 0, smoothed, 1.0).astype(np.float64)

    ys, xs = np.where(clean_mask > 0)
    y0, y1 = max(0, ys.min() - 2), min(clean_mask.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(clean_mask.shape[1], xs.max() + 3)
    sub_field = inside_field[y0:y1, x0:x1]
    sub_mask = clean_mask[y0:y1, x0:x1]

    graph = hg.get_4_adjacency_graph(sub_field.shape)
    tree, altitudes = hg.component_tree_min_tree(graph, sub_field)
    area = hg.attribute_area(tree)
    dynamics = hg.attribute_dynamics(tree, altitudes)

    leaf_med = float(np.median(smoothed[clean_mask > 0]))
    valid_nodes = (area >= min_area) & (area <= 350) & (dynamics >= persistence_thr) & (altitudes < leaf_med - 0.18)

    parents = tree.parents()
    node_has_valid_ancestor = valid_nodes.copy()
    for node in range(tree.root() - 1, -1, -1):
        if node_has_valid_ancestor[parents[node]]:
            node_has_valid_ancestor[node] = True

    pixel_valid = node_has_valid_ancestor[:tree.num_leaves()].reshape(sub_field.shape)
    basin_sub = (pixel_valid & (sub_mask > 0)).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(basin_sub)
    out_sub = np.zeros_like(basin_sub)
    cnt = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out_sub[labels == i] = 1
            cnt += 1

    full_out = np.zeros_like(clean_mask)
    full_out[y0:y1, x0:x1] = out_sub
    return full_out, cnt


def main():
    np.random.seed(42)
    all_files = sorted(os.listdir(LEAF_DIR))
    files_80 = all_files[:80]

    print("=== [1.  :  (leaf_instances) 80 +  Higra Min-tree ] ===")
    print(f"{'':>10s} | {'':>8s} | {' (Error)':>18s} | {'Higra TDA (Error)':>20s}")
    print("-" * 68)

    for n_stains in [0, 3, 6, 10]:
        true_list, area_err_list, tda_err_list = [], [], []
        area_meas_list, tda_meas_list = [], []

        for f in files_80:
            leaf_inst = cv2.imread(f"{LEAF_DIR}/{f}", cv2.IMREAD_UNCHANGED)
            img_bgr = cv2.imread(f"{IMG_DIR}/{f}", cv2.IMREAD_COLOR)
            if leaf_inst is None or img_bgr is None:
                continue
            exg_norm = compute_exg_norm(img_bgr)

            for lid in [i for i in np.unique(leaf_inst) if i != 0]:
                raw_leaf = (leaf_inst == lid).astype(np.uint8)
                if raw_leaf.sum() < 2000:
                    continue
                contours, _ = cv2.findContours(raw_leaf * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                clean_mask = np.zeros_like(raw_leaf)
                cv2.drawContours(clean_mask, contours, -1, 1, -1)

                res = inject_realistic_damage_on_real_leaf(raw_leaf, clean_mask, exg_norm, n_holes=3, n_shadows=n_stains)
                if res is None:
                    continue
                field, _, actual_h = res

                _, c_area = detect_area_filter_mask(field, clean_mask, min_area=15)
                _, c_tda = detect_higra_tda_mask(field, clean_mask, min_area=15, persistence_thr=0.20)

                true_list.append(actual_h)
                area_meas_list.append(c_area)
                tda_meas_list.append(c_tda)
                area_err_list.append(abs(c_area - actual_h))
                tda_err_list.append(abs(c_tda - actual_h))
                break

        print(f"{n_stains:>12d} | {np.mean(true_list):>10.2f} | "
              f"{np.mean(area_meas_list):>8.2f} (Error {np.mean(area_err_list):>4.2f}) | "
              f"{np.mean(tda_meas_list):>9.2f} (Error {np.mean(tda_err_list):>4.2f})")

    # 2.     06-05  4 
    np.random.seed(42)
    files_0605 = [f for f in all_files if f.startswith("06-05")][:50]
    samples = []
    for fname in files_0605:
        leaf_inst = cv2.imread(f"{LEAF_DIR}/{fname}", cv2.IMREAD_UNCHANGED)
        img_bgr = cv2.imread(f"{IMG_DIR}/{fname}", cv2.IMREAD_COLOR)
        if leaf_inst is None or img_bgr is None:
            continue
        exg_norm = compute_exg_norm(img_bgr)

        for lid in [i for i in np.unique(leaf_inst) if i != 0]:
            raw_leaf = (leaf_inst == lid).astype(np.uint8)
            if raw_leaf.sum() < 2500:
                continue
            contours, _ = cv2.findContours(raw_leaf * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(raw_leaf)
            cv2.drawContours(clean_mask, contours, -1, 1, -1)

            res = inject_realistic_damage_on_real_leaf(raw_leaf, clean_mask, exg_norm, n_holes=3, n_shadows=4)
            if res is None:
                continue
            field, gt_mask, actual_holes = res

            ys, xs = np.where(clean_mask > 0)
            y0, y1 = max(0, ys.min() - 8), min(clean_mask.shape[0], ys.max() + 8)
            x0, x1 = max(0, xs.min() - 8), min(clean_mask.shape[1], xs.max() + 8)

            samples.append({
                "title": f"{fname[:11]} (Leaf {lid})",
                "field": field,
                "clean_mask": clean_mask,
                "gt_mask": gt_mask,
                "gt_holes": actual_holes,
                "bbox": (y0, y1, x0, x1)
            })
            break
        if len(samples) >= 4:
            break

    fig, axes = plt.subplots(len(samples), 3, figsize=(11, 3.0 * len(samples)))
    print(f"\n{'':>26s} | {'':>8s} | {' (False Positive)':>20s} | {'Higra TDA ':>14s}")
    print("-" * 78)

    for row, s in enumerate(samples):
        area_mask, area_cnt = detect_area_filter_mask(s["field"], s["clean_mask"])
        tda_mask, tda_cnt = detect_higra_tda_mask(s["field"], s["clean_mask"])
        y0, y1, x0, x1 = s["bbox"]

        print(f"{s['title']:>26s} | {s['gt_holes']:>7d} | {area_cnt:>19d} | {tda_cnt:>13d}")

        axes[row, 0].imshow(s["field"][y0:y1, x0:x1], cmap="YlGn")
        axes[row, 0].set_title(f"{s['title']} (True Holes: {s['gt_holes']}, Shadows: 4)")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(area_mask[y0:y1, x0:x1], cmap="Reds")
        axes[row, 1].set_title(f"Area Filter >=15px (Detected: {area_cnt})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(tda_mask[y0:y1, x0:x1], cmap="Greens")
        axes[row, 2].set_title(f"Higra Min-Tree TDA (Detected: {tda_cnt})")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig("real_pest_comparison.png", dpi=150)
    print("\n/    : real_pest_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()