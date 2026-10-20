import time
import numpy as np
import higra as hg
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
import torch.nn.functional as F

from topologic_cnn_layer import TopologicLightweightInsectNet, DifferentiableTopoRCC8Function


def make_synthetic_batch(H, W):
    """곤충(몸통 Core + 반투명 날개 EC) 및 배경 먼지(DC)가 포함된 테스트 텐서 생성"""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((1, 3, H, W), 0.15, dtype=np.float32)
    cy, cx = H // 2, W // 2
    r_scale = max(1.0, H / 64.0)

    body = ((yy - cy)**2 / (5 * r_scale)**2 + (xx - cx)**2 / (4 * r_scale)**2) <= 1.0
    wings = (
        (((yy - (cy + int(6 * r_scale)))**2 / (10 * r_scale)**2 + (xx - (cx - int(5 * r_scale)))**2 / (4 * r_scale)**2) <= 1.0)
        | (((yy - (cy + int(6 * r_scale)))**2 / (10 * r_scale)**2 + (xx - (cx + int(5 * r_scale)))**2 / (4 * r_scale)**2) <= 1.0)
    ) & (~body)
    dust = ((yy - int(H * 0.2))**2 + (xx - int(W * 0.8))**2) <= (2.0 * r_scale)**2

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.015, size=(1, 3, H, W)).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    img[:, :, body] += 0.65
    img[:, :, wings] += 0.16
    img[:, :, dust] += 0.18
    return img, body, wings, dust


def benchmark_point_cloud_vr_persistence(pot_2d):
    """
    기존 점군(Point-Cloud) 기반 위상 지속성(Vietoris-Rips 0D/1D 거리 행렬 필터레이션) 속도 측정:
    활성 픽셀 K개에 대해 O(K^2 ~ K^3) 거리 행렬 및 스패닝 트리를 계산할 때 발생하는 병목 실측
    """
    t0 = time.perf_counter()
    # 상위 활성 픽셀(최대 1,800개 샘플링)을 점군으로 추출
    thresh = np.percentile(pot_2d, 85)
    pts = np.argwhere(pot_2d >= thresh).astype(np.float32)
    if len(pts) > 1800:
        pts = pts[:1800]
    # 쌍별 유클리드 거리 행렬 O(K^2) + 1-스켈레톤 Rips 삼각형 클리크 스캔 O(K^3 일부)
    dist_mat = squareform(pdist(pts))
    _mst = minimum_spanning_tree(dist_mat)
    # 1차원 루프(구멍) 체크를 위한 근접 그래프 엣지 교차 연산
    adj = (dist_mat < 3.5).astype(np.float32)
    _triangles = np.trace(adj @ adj @ adj) / 6.0
    return (time.perf_counter() - t0) * 1000.0


def benchmark_unquantized_float64_tree(pot_2d):
    """양자화 없는 순수 float64 정밀도 트리 + 개별 노드 루프 연산 속도 측정"""
    t0 = time.perf_counter()
    H, W = pot_2d.shape
    graph = hg.get_4_adjacency_graph((H, W))
    # 미세 노이즈가 섞인 float64 그대로 트리를 구축하면 노드 수가 최대화됨
    tree, alt = hg.component_tree_max_tree(graph, pot_2d.astype(np.float64))
    area = hg.attribute_area(tree)
    dyn = hg.attribute_dynamics(tree, alt)
    # 로컬 배경 필터링(grey_opening) 포함 시의 고전 파이프라인 비용
    _bg = ndimage.grey_opening(pot_2d, size=(15, 15))
    _recon = hg.reconstruct_leaf_data(tree, alt, (dyn < 0.2) | (area < 10))
    return (time.perf_counter() - t0) * 1000.0


if __name__ == "__main__":
    torch.manual_seed(42)
    model = TopologicLightweightInsectNet(num_classes=10)
    target = torch.tensor([0])

    # 1. 워밍업 (최초 C++ Autograd & Higra 메모리 할당 시간 제외)
    warm_np, _, _, _ = make_synthetic_batch(64, 64)
    warm_x = torch.from_numpy(warm_np).requires_grad_(True)
    for _ in range(3):
        out, _, _ = model(warm_x)
        F.cross_entropy(out, target).backward()
        model.zero_grad()

    print("=" * 92)
    print(" [실측 검증 1] 해상도별 우리 엔진(TopoRCC8ConvBlock) 실제 처리 속도 및 실시간(30 FPS) 판정")
    print("=" * 92)
    print(f"{'입력 해상도 (용도)':<28} | {'추론 속도 (Forward)':<20} | {'학습 속도 (Fwd+Bwd)':<20} | {'실시간(33.3ms) 기준'}")
    print("-" * 92)

    resolutions = [
        (64, 64, "64x64 (CNN 특징맵)"),
        (128, 128, "128x128 (소형 크롭)"),
        (200, 200, "200x200 (RQ1 슬라이드10)"),
        (300, 300, "300x300 (고해상도 트랩)")
    ]

    n_runs = 20
    for H, W, label in resolutions:
        img_np, _, _, _ = make_synthetic_batch(H, W)

        # (A) 추론 전용 (Inference Only - torch.no_grad)
        x_inf = torch.from_numpy(img_np)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x_inf)
        inf_ms = ((time.perf_counter() - t0) * 1000.0) / n_runs
        inf_fps = 1000.0 / inf_ms

        # (B) 학습 전용 (Forward + loss.backward)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            x_trn = torch.from_numpy(img_np).requires_grad_(True)
            logits, _, _ = model(x_trn)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            model.zero_grad()
        trn_ms = ((time.perf_counter() - t0) * 1000.0) / n_runs

        status = f"통과 ({33.3/inf_ms:.1f}배 빠름)" if inf_ms <= 33.3 else "배치/다운샘플 권장"
        print(f"{label:<28} | {inf_ms:>6.2f} ms ({inf_fps:>5.0f} FPS) | {trn_ms:>6.2f} ms ({1000/trn_ms:>5.0f} FPS) | {status}")

    print("=" * 92)

    # 2. 방식별 직접 비교 (200x200 RQ1 이미지 기준)
    img_200, _, _, _ = make_synthetic_batch(200, 200)
    pot_200 = img_200[0, 0]

    # (1) 우리 양자화 Higra + RCC8 레이어
    x_200 = torch.from_numpy(img_200)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = DifferentiableTopoRCC8Function.apply(x_200, 0.22, 1.50)
    our_ms = ((time.perf_counter() - t0) * 1000.0) / 10.0

    # (2) 비양자화 Float64 + 고전 모폴로지(grey_opening) 파이프라인
    unquant_ms = benchmark_unquantized_float64_tree(pot_200)

    # (3) 점군(Point-Cloud) 거리 행렬 기반 위상 지속성 (Vietoris-Rips 스타일)
    vr_ms = benchmark_point_cloud_vr_persistence(pot_200)

    print("\n" + "=" * 92)
    print(" [실측 검증 2] 동일 200x200 이미지에서 위상 연산 방식별 속도 비교 (내 PC CPU 실측)")
    print("=" * 92)
    print(f" 1. 우리 엔진 (8-bit 양자화 Higra O(N) + RCC8)      : {our_ms:>8.2f} ms  (기준 1.0배)")
    print(f" 2. 비양자화 Float64 트리 + 고전 모폴로지 배경 제거 : {unquant_ms:>8.2f} ms  ({unquant_ms/our_ms:>4.1f}배 느림)")
    print(f" 3. 점군(Point-Cloud) 거리행렬 위상 지속성 (Rips)   : {vr_ms:>8.2f} ms  ({vr_ms/our_ms:>4.1f}배 느림)")
    print("=" * 92)