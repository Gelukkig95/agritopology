import time
import numpy as np
import cv2
import higra as hg


class RQ1DisturbanceFilter:
    """Higra Max-Tree and RCC8 disturbance filter for camera trap imagery."""
    def __init__(self):
        self._graph_cache = {}

    def _get_graph(self, H: int, W: int):
        key = (H, W)
        if key not in self._graph_cache:
            self._graph_cache[key] = hg.get_4_adjacency_graph((H, W))
        return self._graph_cache[key]

    def analyze_frame(self, panel_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        H, W, _ = panel_bgr.shape
        graph = self._get_graph(H, W)
        num_leaves = H * W

        rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. Reference background color contrast (detects large boundary occlusions)
        bg_ref = np.median(rgb[int(H * 0.65):, :int(W * 0.35)].reshape(-1, 3), axis=0)
        color_diff = np.abs(rgb - bg_ref).mean(axis=2)

        # 2. Local dark and bright specular fields
        local_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=18)
        dark_field = np.clip(local_bg - gray, 0, None).astype(np.float64)
        bright_field = np.clip(gray - local_bg, 0, None).astype(np.float64)

        valid_zone = np.zeros((H, W), dtype=np.uint8)
        valid_zone[int(H * 0.04):int(H * 0.96), int(W * 0.04):int(W * 0.96)] = 1

        # 3. Higra Max-Tree construction and volume attribute computation
        d_norm = (dark_field * valid_zone) / (dark_field.max() + 1e-5)
        tree_d, alt_d = hg.component_tree_max_tree(graph, d_norm)
        vol_d = hg.attribute_volume(tree_d, alt_d)
        leaf_vol_d = vol_d[tree_d.parents()[:num_leaves]].reshape(H, W)

        large_occ_mask = ((color_diff > 16.0) & (gray > np.median(gray) + 6.0)).astype(np.uint8)
        large_occ_mask = cv2.morphologyEx(large_occ_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        large_occ_ratio = float(large_occ_mask.sum()) / float(H * W)

        thr_d = max(4.0, float(np.percentile(dark_field[valid_zone > 0], 96) * 0.40))
        dark_bin = ((dark_field >= thr_d) & (valid_zone > 0)).astype(np.uint8)
        dark_bin = cv2.morphologyEx(dark_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(dark_bin)

        best_id, best_vol = 0, 0.0
        for cid in range(1, n_cc):
            if stats[cid, cv2.CC_STAT_AREA] < 8:
                continue
            v = float(leaf_vol_d[cc_map == cid].max()) * float(stats[cid, cv2.CC_STAT_AREA])
            if v > best_vol:
                best_vol = v
                best_id = cid

        obj_mask = (cc_map == best_id).astype(np.uint8) if best_id > 0 else np.zeros((H, W), dtype=np.uint8)
        if best_id > 0:
            x, y, bw, bh, area = stats[best_id]
            span_ratio = float(max(bw, bh)) / float(H)
            solidity = float(area) / float(max(bw * bh, 1))
            dil_obj = cv2.dilate(obj_mask, np.ones((9, 9), np.uint8), iterations=1)
            max_dark_in_obj = float(dark_field[dil_obj > 0].max()) + 1e-5
            max_bright_in_obj = float(bright_field[dil_obj > 0].max())
            specular_ratio = max_bright_in_obj / max_dark_in_obj
        else:
            x, y, bw, bh, area = 0, 0, 0, 0, 0
            span_ratio, solidity, specular_ratio = 0.0, 0.0, 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Topological and RCC8 rule evaluation
        if large_occ_ratio >= 0.14:
            pred_label = "DISTURBANCE: Hand / Large Occlusion"
            vis_mask = large_occ_mask
            color_box = (0, 0, 255)
        elif span_ratio >= 0.55 and solidity < 0.25:
            pred_label = "DISTURBANCE: Spider Web / Filament"
            vis_mask = obj_mask
            color_box = (0, 140, 255)
        elif (specular_ratio >= 0.35 and span_ratio < 0.22) or (span_ratio < 0.15):
            pred_label = "DISTURBANCE: Rain Droplet (Ring+Hole)"
            vis_mask = cv2.dilate(obj_mask, np.ones((5, 5), np.uint8), iterations=1)
            color_box = (255, 180, 0)
        else:
            pred_label = "TRUE INSECT (Pass to Classifier!)"
            vis_mask = obj_mask
            color_box = (0, 220, 0)

        return {
            "pred": pred_label,
            "occ_ratio": large_occ_ratio * 100.0,
            "span_ratio": span_ratio * 100.0,
            "solidity": solidity,
            "rcc8_border": int(span_ratio >= 0.55),
            "specular_ratio": specular_ratio,
            "higra_vol": best_vol,
            "ms": elapsed_ms,
            "vis_mask": vis_mask,
            "bbox": (x, y, bw, bh),
            "color": color_box,
        }
