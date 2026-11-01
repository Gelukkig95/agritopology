import numpy as np
import higra as hg
import torch

from torch_morph_rcc8 import (
    kuratowski_opening,
    classify_rcc8_grid_jepd,
    compute_rcc8_matrix,
    DifferentiableFuzzyRCC8Loss,
    TopoRCC8Transformer,
    extract_higra_rcc8_tokens,
)
from torch_morph_rcc8.diagnostics import gpu_pebay_moment_merge


def test_tier1_kuratowski_and_rcc8_jepd_axioms():
    """Verifies Randell et al. (1992) RCC-8 JEPD axioms and Kuratowski opening idempotence."""
    rng = np.random.default_rng(42)
    valid_relations = {"DC", "EC", "PO", "EQ", "TPP", "NTPP", "TPPi", "NTPPi"}
    for _ in range(30):
        A = rng.random((32, 32)) > 0.65
        B = rng.random((32, 32)) > 0.65
        op_A = kuratowski_opening(A)
        assert np.array_equal(kuratowski_opening(op_A), op_A)
        assert classify_rcc8_grid_jepd(A, B) in valid_relations

    masks = torch.zeros((3, 16, 16))
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 6:10] = 1.0
    masks[2, 10:14, 10:14] = 1.0
    mat = compute_rcc8_matrix(masks)
    assert int(mat[0, 1].item()) == 1  # EC
    assert int(mat[0, 2].item()) == 0  # DC


def test_tier1_monotonic_persistence_invariance():
    """Verifies invariance of Salembier et al. (1998) Max-Tree dynamics under strictly monotonic intensity scaling."""
    rng = np.random.default_rng(42)
    X = rng.random((48, 48)).astype(np.float64)
    graph = hg.get_4_adjacency_graph((48, 48))
    t1, a1 = hg.component_tree_max_tree(graph, X)
    t2, a2 = hg.component_tree_max_tree(graph, 3.7 * X + 1.25)
    d1, d2 = hg.attribute_dynamics(t1, a1), hg.attribute_dynamics(t2, a2)
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(d1)), np.argsort(np.argsort(d2)))[0, 1])
    assert rank_corr > 0.999


def test_tier1_bloch_fuzzy_rcc8_gradcheck():
    """Verifies analytical gradients of Bloch (2005) Eq. (16), (22), (36) via torch.autograd.gradcheck."""
    torch.manual_seed(42)
    loss_fn = DifferentiableFuzzyRCC8Loss(kernel_size=3)
    mu = torch.sigmoid(torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True))
    nu = torch.sigmoid(torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True))

    def func(a, b):
        out = loss_fn(a, b)
        return out["fuzzy_EC"] + out["fuzzy_NTPP"]

    assert torch.autograd.gradcheck(func, (mu, nu), eps=1e-6, atol=1e-4)


def test_tier2_tw2026_pebay_moments_and_higra_tokens():
    """Verifies Teeninga & Wilkinson (2026) Sec. 4.4 Pebay parallel moments and Higra-RCC8 token extraction."""
    vals = torch.tensor([1.2, -0.5, 3.4, 0.8], dtype=torch.float64)
    area = torch.ones(4, dtype=torch.float64)
    mean = vals.clone()
    m2 = torch.zeros(4, dtype=torch.float64)

    gpu_pebay_moment_merge(area, mean, m2, torch.tensor([0, 2]), torch.tensor([1, 3]))
    gpu_pebay_moment_merge(area, mean, m2, torch.tensor([0]), torch.tensor([2]))

    assert torch.isclose(mean[0], vals.mean(), atol=1e-12)
    assert torch.isclose(m2[0] / 4.0, vals.var(unbiased=False), atol=1e-12)

    yy, xx = np.mgrid[0:64, 0:64]
    scene = np.full((64, 64), 0.10, dtype=np.float32)
    body = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 25
    wing = (((yy - 32) ** 2 + (xx - 40) ** 2) <= 16) & (~body)
    scene[body], scene[wing] = 0.80, 0.28
    feats, ec_mat, _, _ = extract_higra_rcc8_tokens(scene)
    model = TopoRCC8Transformer(d_model=8)
    probs, _ = model(feats, ec_mat)
    assert probs.shape[0] == feats.shape[0]
