import time
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

from train_baseline import PhenoBenchDataset, SimpleUNet, BASE_DIR, IMG_SIZE

DIM = 10000  # 10,000  


class PureVSAEngine:
    """
       10,000 VSA :
       (RCC8  +  +  + Crop + ExG)
    (*) (+)     .
    """
    def __init__(self, dim=DIM, n_levels=25, seed=42):
        self.dim = dim
        self.n_levels = n_levels
        rng = np.random.default_rng(seed)

        # (Role)   
        self.role_rcc8 = rng.choice([-1, 1], size=dim).astype(np.int8)
        self.role_size = rng.choice([-1, 1], size=dim).astype(np.int8)
        self.role_conf = rng.choice([-1, 1], size=dim).astype(np.int8)
        self.role_surr = rng.choice([-1, 1], size=dim).astype(np.int8)
        self.role_exg = rng.choice([-1, 1], size=dim).astype(np.int8)

        # RCC8 8   
        self.rcc8_hvs = {rel.name: rng.choice([-1, 1], size=dim).astype(np.int8) for rel in RCC8}
        self.level_hvs = self._build_level_hvs(rng)

    def _build_level_hvs(self, rng):
        base = rng.choice([-1, 1], size=self.dim).astype(np.int8)
        levels = [base.copy()]
        flip_step = self.dim // (2 * self.n_levels)
        perm = rng.permutation(self.dim)
        curr = base.copy()
        for s in range(1, self.n_levels + 1):
            idx = perm[(s - 1) * flip_step : s * flip_step]
            curr = curr.copy()
            curr[idx] *= -1
            levels.append(curr)
        return np.array(levels, dtype=np.int8)

    def quantize(self, val, vmin, vmax):
        norm = np.clip((val - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        idx = int(round(norm * self.n_levels))
        return self.level_hvs[idx]

    def encode(self, rcc8_name, size, pweed, pcrop_surr, exg):
        """ if-else   (*)       """
        hv_rel = self.role_rcc8 * self.rcc8_hvs[rcc8_name]
        hv_sz = self.role_size * self.quantize(size, 5.0, 30.0)
        hv_cf = self.role_conf * self.quantize(pweed, 0.50, 1.00)
        hv_sr = self.role_surr * self.quantize(pcrop_surr, 0.00, 0.40)
        hv_ex = self.role_exg * self.quantize(exg, 0.00, 0.25)

        #  (hv_rel)      if-else   
        summed = (
            hv_rel.astype(np.int16)
            + (hv_rel * hv_ex)
            + (hv_rel * hv_sr)
            + (hv_rel * hv_cf)
            + hv_sz + hv_cf + hv_ex
        )
        return np.where(summed >= 0, 1, -1).astype(np.int8)

    @staticmethod
    def bundle(hvs):
        summed = np.sum(hvs, axis=0, dtype=np.int32)
        return np.where(summed >= 0, 1, -1).astype(np.int8)

    @staticmethod
    def cosine_sim(a, b):
        return float(np.dot(a.astype(np.int32), b.astype(np.int32))) / len(a)


def compute_exg_from_tensor(img_tensor):
    img = img_tensor.numpy()
    return 2.0 * img[1] - img[0] - img[2]


def extract_all(model, loader, device, min_size=5):
    model.eval()
    samples, blobs = [], []
    kernel_5 = np.ones((5, 5), np.uint8)

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels_np = labels.numpy()

            for i in range(preds.shape[0]):
                pred, gt = preds[i], labels_np[i]
                p_weed, p_crop = probs[i, 2], probs[i, 1]
                exg = compute_exg_from_tensor(imgs[i])

                crop_bin = (pred == 1).astype(np.uint8)
                weed_bin = (pred == 2).astype(np.uint8)
                n_crop, crop_cc = cv2.connectedComponents(crop_bin)
                n_weed, weed_cc = cv2.connectedComponents(weed_bin)

                s_idx = len(samples)
                samples.append({"pred": pred, "label": gt, "probs": probs[i], "weed_cc": weed_cc})
                if n_weed <= 1:
                    continue

                crop_polys = {}
                for cid in range(1, n_crop):
                    m = (crop_cc == cid).astype(np.uint8)
                    if m.sum() >= 10:
                        poly = mask_to_polygon(m)
                        if poly is not None:
                            crop_polys[cid] = poly

                for wid in range(1, n_weed):
                    wmask = (weed_cc == wid).astype(np.uint8)
                    size = int(wmask.sum())
                    if size < min_size:
                        continue
                    gt_maj = np.bincount(gt[wmask > 0], minlength=3).argmax()
                    wpoly = mask_to_polygon(wmask)
                    best_rel = RCC8.DC
                    if wpoly is not None:
                        for cid, cpoly in crop_polys.items():
                            rel = classify_rcc8(wpoly, cpoly)
                            if rel != RCC8.DC:
                                best_rel = rel
                                break

                    if gt_maj == 2:
                        group = "TP"
                    elif gt_maj == 0 and best_rel == RCC8.DC:
                        group = "B"
                    elif gt_maj == 1 and best_rel != RCC8.DC:
                        group = "C"
                    elif gt_maj == 1 and best_rel == RCC8.DC:
                        group = "D"
                    else:
                        group = "OTHER"

                    dil = cv2.dilate(wmask, kernel_5)
                    blobs.append({
                        "sample_idx": s_idx, "wid": wid, "group": group,
                        "rel": best_rel.name, "size": size,
                        "pweed": float(p_weed[wmask > 0].mean()),
                        "pcrop_surround": float(p_crop[dil > 0].mean()),
                        "exg": float(exg[wmask > 0].mean())
                    })
    return samples, blobs


def evaluate_and_print(name, samples, blobs, keep_mask):
    corrected = [s["pred"].copy() for s in samples]
    for bi, blob in enumerate(blobs):
        if keep_mask[bi]:
            continue
        s = samples[blob["sample_idx"]]
        wmask = (s["weed_cc"] == blob["wid"])
        probs = s["probs"]
        corrected[blob["sample_idx"]][wmask] = np.where(probs[1][wmask] > probs[0][wmask], 1, 0)

    all_preds = np.concatenate([p.ravel() for p in corrected])
    all_labels = np.concatenate([s["label"].ravel() for s in samples])
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    c = 2
    tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
    iou = tp / (tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)

    tp_tot = sum(1 for b in blobs if b["group"] == "TP")
    tp_kept = sum(1 for i, b in enumerate(blobs) if b["group"] == "TP" and keep_mask[i])
    fp_tot = sum(1 for b in blobs if b["group"] in ("B", "C", "D"))
    fp_rem = sum(1 for i, b in enumerate(blobs) if b["group"] in ("B", "C", "D") and not keep_mask[i])
    b_rem = sum(1 for i, b in enumerate(blobs) if b["group"] == "B" and not keep_mask[i])

    print(f"\n=== [{name}] ===")
    print(f"  •   (Weed): IoU={iou:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
    print(f"  •  Weed  : {tp_kept}/{tp_tot} ({tp_kept/tp_tot*100:.1f}%)")
    print(f"  •  False Positive(FP)  : {fp_rem}/{fp_tot} ({fp_rem/fp_tot*100:.1f}%) [ B : {b_rem}/540 ({b_rem/540*100:.1f}%)]")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = PhenoBenchDataset(BASE_DIR, "val", IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("baseline_unet.pth", map_location=device))

    print("    10,000 VSA  ...")
    samples, blobs = extract_all(model, val_loader, device)
    vsa = PureVSAEngine(dim=DIM)

    t0 = time.perf_counter()
    hvs = np.array([vsa.encode(b["rel"], b["size"], b["pweed"], b["pcrop_surround"], b["exg"]) for b in blobs])
    encode_ms = (time.perf_counter() - t0) * 1000.0 / len(blobs)

    keep_vsa = [True] * len(blobs)
    vsa_scores = np.zeros(len(blobs), dtype=np.float32)

    t1 = time.perf_counter()
    # filter_scorecard_cv.py 100%   (size < 30, target_recall = 0.96, 2-Fold CV)
    # : if is_dc_flag     VSA  (bundle) 1 !
    for parity in [0, 1]:
        tr_tp = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 == parity) and b["size"] < 30 and b["group"] == "TP"]
        tr_fp = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 == parity) and b["size"] < 30 and b["group"] != "TP"]
        tr_all = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 == parity) and b["size"] < 30]
        te_all = [i for i, b in enumerate(blobs) if (b["sample_idx"] % 2 != parity) and b["size"] < 30]

        proto_tp = vsa.bundle(hvs[tr_tp])
        proto_fp = vsa.bundle(hvs[tr_fp])

        tr_scores = np.array([vsa.cosine_sim(hvs[i], proto_tp) - vsa.cosine_sim(hvs[i], proto_fp) for i in tr_all])
        y_tr = np.array([1 if blobs[i]["group"] == "TP" else 0 for i in tr_all])
        _, tpr, thr = roc_curve(y_tr, tr_scores)
        t_val = thr[np.argmin(np.abs(tpr - 0.96))]

        for idx in te_all:
            sc = vsa.cosine_sim(hvs[idx], proto_tp) - vsa.cosine_sim(hvs[idx], proto_fp)
            vsa_scores[idx] = sc
            keep_vsa[idx] = bool(sc >= t_val)

    infer_ms = (time.perf_counter() - t1) * 1000.0 / len(blobs)

    evaluate_and_print("0. No Correction (Raw Baseline)", samples, blobs, [True] * len(blobs))
    evaluate_and_print("1. ★  10,000 VSA (if-else  0,  0, 2-Fold CV)", samples, blobs, keep_vsa)
    print(f"\n  • [ ]   ( if-else 2 ): TP 97.7%   FP 79/707 (11.2%) ")
    print(f"  • [VSA ]  + : {encode_ms + infer_ms:.4f} ms")

    rejected = [i for i, b in enumerate(blobs) if b["group"] in ("B", "C", "D") and not keep_vsa[i]]
    if rejected:
        ex_i = rejected[0]
        unbound_rel = hvs[ex_i] * vsa.role_rcc8
        rec_rel = max(vsa.rcc8_hvs.items(), key=lambda kv: vsa.cosine_sim(unbound_rel, kv[1]))[0]
        print(f"  • [VSA (Unbinding)  ] Blob #{ex_i}:  =RCC8.{rec_rel} (=RCC8.{blobs[ex_i]['rel']})")


if __name__ == "__main__":
    main()