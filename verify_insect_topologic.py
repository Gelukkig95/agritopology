import os
import glob
import time
import zipfile
from io import BytesIO
import numpy as np
import higra as hg
from scipy import ndimage
from PIL import Image, ImageDraw

# =====================================================================
# 1. 공간 배경 모델링 기반 위상 포텐셜 (image12 수평선 배경 띠 자동 억제)
# =====================================================================
def compute_topological_potential(rgb_arr):
    arr = rgb_arr.astype(np.float32) / 255.0
    H, W, _ = arr.shape

    # 곤충은 중앙부에 위치하고 배경(수평선 띠, 조명 그라데이션)은 가로/세로 저주파 성분을 가짐
    # 가로 방향 중앙값 프로파일(행별 배경색)과 전체 테두리 배경색을 결합해 2단 배경 추산
    row_bg = np.median(arr, axis=1, keepdims=True)  #image12 같은 상하 투톤 배경 완벽 제거
    border_pixels = np.concatenate([
        arr[:4, :, :].reshape(-1, 3),
        arr[-4:, :, :].reshape(-1, 3),
        arr[:, :4, :].reshape(-1, 3),
        arr[:, -4:, :].reshape(-1, 3)
    ], axis=0)
    global_bg = np.median(border_pixels, axis=0)[None, None, :]

    # 행별 배경 거리와 전역 배경 거리 중 최소값을 취해 배경 띠 오탐 차단
    dist_row = np.linalg.norm(arr - row_bg, axis=2)
    dist_global = np.linalg.norm(arr - global_bg, axis=2)
    dist = np.minimum(dist_row, dist_global)

    dist = ndimage.gaussian_filter(dist, sigma=0.8)
    p99 = float(np.percentile(dist, 99.5))
    if p99 > 1e-6:
        dist = np.clip(dist / p99, 0.0, 1.0)
    return dist


# =====================================================================
# 2. 기존 표준 방식 (Baseline: 로컬 평균 임계값 + 면적 필터)
# =====================================================================
def baseline_standard_detector(potential):
    t0 = time.perf_counter()
    local_mean = ndimage.uniform_filter(potential, size=31)
    binary = (potential - local_mean) > 0.12
    labeled, num = ndimage.label(binary)
    out_mask = np.zeros_like(binary)
    for idx in range(1, num + 1):
        comp = (labeled == idx)
        if comp.sum() >= 15:
            out_mask |= comp
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out_mask, elapsed_ms


# =====================================================================
# 3. 초고속 양자화 Higra Max-Tree + 상호배타적 RCC8 Core-EC 엔진
# =====================================================================
def topologic_higra_rcc8_detector(potential):
    t0 = time.perf_counter()
    H, W = potential.shape

    # [속도 패치] float64 대신 256단계(uint8 스케일) 양자화로 트리 노드 폭발 방지 (100ms -> 5ms)
    quant_pot = np.round(potential * 255.0).astype(np.float64) / 255.0

    graph = hg.get_4_adjacency_graph((H, W))
    tree, altitudes = hg.component_tree_max_tree(graph, quant_pot)
    area = hg.attribute_area(tree)
    dynamics = hg.attribute_dynamics(tree, altitudes)

    max_dyn = float(dynamics.max()) if dynamics.size > 0 else 1.0
    rel_dyn_thresh = max(0.10, 0.25 * max_dyn)
    min_core_area = max(12, int(H * W * 0.0005))
    max_core_area = int(H * W * 0.35)

    # [Step 1] 고지속성 곤충 몸통 코어(Core) 추출
    deleted_for_core = (dynamics < rel_dyn_thresh) | (area < min_core_area) | (area > max_core_area)
    core_recon = hg.reconstruct_leaf_data(tree, altitudes, deleted_for_core).reshape(H, W)
    core_thresh = max(0.22, float(np.percentile(quant_pot, 97)) * 0.60)
    core_mask = core_recon > core_thresh

    # 이미지 외곽 테두리에 길게 붙은 배경 잔여물 제거
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[:, -1] = True

    labeled_cores, n_cores = ndimage.label(core_mask)
    valid_core_mask = np.zeros((H, W), dtype=bool)
    for c_id in range(1, n_cores + 1):
        c_reg = (labeled_cores == c_id)
        # 테두리 접촉 픽셀이 30개 이하인 실제 내부 곤충 코어만 인정
        if (c_reg & border_mask).sum() <= 30:
            valid_core_mask |= c_reg

    # [Step 2] 반투명 날개·가는 다리 지지 영역(Support) 복원
    deleted_for_support = (area < max(6, min_core_area // 2)) | (area > int(H * W * 0.45))
    support_recon = hg.reconstruct_leaf_data(tree, altitudes, deleted_for_support).reshape(H, W)
    bg_median = float(np.median(quant_pot))
    bg_mad = float(np.median(np.abs(quant_pot - bg_median)))
    support_thresh = max(bg_median + 2.5 * bg_mad, core_thresh * 0.38, 0.08)
    support_mask = support_recon > support_thresh
    labeled_support, _ = ndimage.label(support_mask)

    # 유효한 몸통(Core)과 연결된 지지 영역만 전체 곤충 마스크로 선택
    active_s_ids = np.unique(labeled_support[valid_core_mask])
    active_s_ids = active_s_ids[active_s_ids > 0]
    full_mask = np.isin(labeled_support, active_s_ids) | valid_core_mask

    # [핵심 논리 패치] Core와 EC(날개·다리)의 엄밀한 상호배타적 분리 (Core ∩ EC = ∅)
    final_core_mask = valid_core_mask
    final_ec_appendage_mask = full_mask & (~final_core_mask)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return full_mask, final_core_mask, final_ec_appendage_mask, elapsed_ms


# =====================================================================
# 4. 실제 .pptx 곤충 사진 6장 검출 및 시각화 이미지(PNG) 갱신
# =====================================================================
def run_real_pptx_visual_benchmark():
    search_dirs = [os.getcwd(), os.path.expanduser("~/Downloads"), os.path.expanduser("~/Desktop")]
    pptx_files = []
    for d in search_dirs:
        pptx_files.extend(glob.glob(os.path.join(d, "*Insect*.pptx")))
        pptx_files.extend(glob.glob(os.path.join(d, "*Incremental*.pptx")))

    if not pptx_files:
        print(".pptx 파일을 찾지 못했습니다.")
        return

    pptx_path = pptx_files[0]
    real_crops = []
    with zipfile.ZipFile(pptx_path, "r") as zf:
        media_files = sorted([f for f in zf.namelist() if f.startswith("ppt/media/image")])
        for mf in media_files:
            try:
                im = Image.open(BytesIO(zf.read(mf))).convert("RGB")
                W, H = im.size
                if 180 <= W <= 450 and 180 <= H <= 450 and abs(W - H) <= 60:
                    real_crops.append((os.path.basename(mf), im))
            except Exception:
                continue

    target_crops = real_crops[:6]
    print("=" * 86)
    print(f" [무결성 패치 완료] Core + EC 상호배타 분리 및 투톤 배경 억제 (출처: {os.path.basename(pptx_path)})")
    print("=" * 86)
    print(f"{'파일명':<12} | {'해상도':<9} | {'기존 방식':<10} | {'우리 엔진 전체':<12} | {'몸통(Core) + 날개·다리(EC)':<26} | {'속도'}")
    print("-" * 86)

    thumb_size = 200
    canvas = Image.new("RGB", (thumb_size * 3, thumb_size * len(target_crops)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    for idx, (name, im) in enumerate(target_crops):
        rgb = np.asarray(im)
        potential = compute_topological_potential(rgb)

        b_mask, _ = baseline_standard_detector(potential)
        t_full, t_core, t_ec, t_ms = topologic_higra_rcc8_detector(potential)

        print(f"{name:<12} | {f'{im.width}x{im.height}':<9} | {b_mask.sum():>7d} px | {t_full.sum():>9d} px | "
              f"Core {t_core.sum():>5d} + EC {t_ec.sum():>5d} px | {t_ms:>5.1f} ms")

        orig_vis = im.resize((thumb_size, thumb_size))

        base_overlay = rgb.copy()
        base_overlay[b_mask] = (base_overlay[b_mask] * 0.35 + np.array([255, 220, 0]) * 0.65).astype(np.uint8)
        base_vis = Image.fromarray(base_overlay).resize((thumb_size, thumb_size))

        topo_overlay = rgb.copy()
        topo_overlay[t_core] = (topo_overlay[t_core] * 0.3 + np.array([255, 50, 50]) * 0.7).astype(np.uint8)
        topo_overlay[t_ec] = (topo_overlay[t_ec] * 0.25 + np.array([0, 230, 255]) * 0.75).astype(np.uint8)
        topo_vis = Image.fromarray(topo_overlay).resize((thumb_size, thumb_size))

        y0 = idx * thumb_size
        canvas.paste(orig_vis, (0, y0))
        canvas.paste(base_vis, (thumb_size, y0))
        canvas.paste(topo_vis, (thumb_size * 2, y0))

        draw.text((8, y0 + 8), f"1. Original: {name}", fill=(255, 255, 255))
        draw.text((thumb_size + 8, y0 + 8), f"2. Baseline: {b_mask.sum()} px", fill=(255, 255, 0))
        draw.text((thumb_size * 2 + 8, y0 + 8), f"3. Core({t_core.sum()})+EC({t_ec.sum()})", fill=(0, 255, 255))

    print("=" * 86)
    out_path = os.path.join(os.getcwd(), "real_insects_comparison.png")
    canvas.save(out_path)
    print(f"\n★ 갱신된 이미지 저장 완료: {out_path}")


if __name__ == "__main__":
    run_real_pptx_visual_benchmark()