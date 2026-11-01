import os
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT_DIR = Path(r"C:\Users\geluk\agritopology")
sys.path.insert(0, str(ROOT_DIR))

from torch_morph_rcc8.rq1_filter import RQ1DisturbanceFilter

CAP_PATH = ROOT_DIR / "data" / "rq1_slide_4photos.png"
OUT_PATH = ROOT_DIR / "data" / "rq1_higra_4cases_result.png"


def split_four_slide_panels(bgr_strip: np.ndarray):
    H, W, _ = bgr_strip.shape
    gray = cv2.cvtColor(bgr_strip, cv2.COLOR_BGR2GRAY)
    col_darkness = 255.0 - gray.mean(axis=0)
    is_panel_col = col_darkness > (col_darkness.min() + 0.25 * (col_darkness.max() - col_darkness.min()))

    segments = []
    in_seg, start = False, 0
    for x in range(W):
        if is_panel_col[x] and not in_seg:
            in_seg, start = True, x
        elif not is_panel_col[x] and in_seg:
            in_seg = False
            if (x - start) > W * 0.12:
                segments.append((start, x))
    if in_seg and (W - start) > W * 0.12:
        segments.append((start, W))

    panels = []
    if len(segments) == 4:
        for (x1, x2) in segments:
            panels.append(bgr_strip[:, x1:x2].copy())
    else:
        step = W // 4
        for i in range(4):
            panels.append(bgr_strip[:, i * step : (i + 1) * step].copy())

    clean_panels = []
    for p in panels:
        ph, pw, _ = p.shape
        my, mx = int(ph * 0.08), int(pw * 0.08)
        cropped = p[my : ph - my, mx : pw - mx]
        clean_panels.append(cv2.resize(cropped, (240, 240), interpolation=cv2.INTER_AREA))
    return clean_panels


def main():
    if not CAP_PATH.exists():
        print(f"[!] {CAP_PATH}  .    4 Win+Shift+S   !")
        return

    bgr = cv2.imread(str(CAP_PATH))
    panels = split_four_slide_panels(bgr)
    rq1_filter = RQ1DisturbanceFilter()

    gt_names = [
        "Photo 1 (Spider Web)",
        "Photo 2 (Person's Hand)",
        "Photo 3 (Rain Droplet)",
        "Photo 4 (Real Insect)",
    ]

    print("=" * 114)
    print(" [AGRITOPOLOGY LIBRARY — RQ1 FILTER TEST] from torch_morph_rcc8.rq1_filter import RQ1DisturbanceFilter")
    print("=" * 114)
    print(f" {'Slide Photo':23s} | {'OccArea%':8s} | {'Span%':6s} | {'Solidity':8s} | {'SpecHighlight':13s} | {'Time':6s} | {'agritopology Decision'}")
    print("-" * 114)

    vis_cards = []
    for idx, p in enumerate(panels):
        res = rq1_filter.analyze_frame(p)
        print(f" {gt_names[idx]:23s} | {res['occ_ratio']:6.1f}%  | {res['span_ratio']:5.1f}% | {res['solidity']:8.3f} | "
              f"{res['specular_ratio']:13.2f} | {res['ms']:4.1f}ms | --> {res['pred']}")

        card = p.copy()
        contours, _ = cv2.findContours(res["vis_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(card, contours, -1, res["color"], 2)
        cv2.rectangle(card, (0, 0), (239, 32), (25, 25, 25), -1)
        cv2.putText(card, res["pred"][:32], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, res["color"], 1, cv2.LINE_AA)
        vis_cards.append(card)

    print("=" * 114)
    combined = np.hstack(vis_cards)
    cv2.imwrite(str(OUT_PATH), combined)
    print(f" ->     : {OUT_PATH}")
    try:
        os.startfile(str(OUT_PATH))
    except Exception:
        pass


if __name__ == "__main__":
    main()
