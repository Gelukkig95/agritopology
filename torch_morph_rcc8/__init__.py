r"""
torch_morph_rcc8
----------------
Differentiable Fuzzy RCC-8 Spatial Logic & Max-Tree Morphological Operators in PyTorch.
"""

from .mask_utils import mask_to_polygon
from .rcc8 import RCC8, classify_rcc8
from .modules import (
    compute_rcc8_ec_matrix,
    extract_higra_rcc8_tokens,
    TopoRCC8Transformer,
)
from .diagnostics import (
    RCC8_NAMES,
    morph_dilation,
    morph_erosion,
    kuratowski_closure,
    kuratowski_interior,
    kuratowski_opening,
    classify_rcc8_grid_jepd,
    compute_rcc8_matrix,
    DifferentiableFuzzyRCC8Loss,
    gpu_pebay_moment_merge,
    gpu_tree_leaffix_accumulate,
    gpu_tree_rootfix_reconstruct,
    ConformalRiskCalibrator,
)
from .insect_topo import (
    AgriTopoExtractor,
    AgriTopoResidualNet,
    FastTensorRasterRCC8,
    ConditionAdaptiveMondrianCRC,
    StrictMondrianCRC,
)
from .rq1_filter import RQ1DisturbanceFilter
from .referee import DecisionState, TopologicalProofReport, AgriTopoReferee

__all__ = [
    "RCC8",
    "RCC8_NAMES",
    "classify_rcc8",
    "mask_to_polygon",
    "morph_dilation",
    "morph_erosion",
    "kuratowski_closure",
    "kuratowski_interior",
    "kuratowski_opening",
    "classify_rcc8_grid_jepd",
    "compute_rcc8_matrix",
    "compute_rcc8_ec_matrix",
    "DifferentiableFuzzyRCC8Loss",
    "gpu_pebay_moment_merge",
    "gpu_tree_leaffix_accumulate",
    "gpu_tree_rootfix_reconstruct",
    "extract_higra_rcc8_tokens",
    "TopoRCC8Transformer",
    "ConformalRiskCalibrator",
    "AgriTopoExtractor",
    "AgriTopoResidualNet",
    "FastTensorRasterRCC8",
    "ConditionAdaptiveMondrianCRC",
    "StrictMondrianCRC",
    "RQ1DisturbanceFilter",
    "DecisionState",
    "TopologicalProofReport",
    "AgriTopoReferee",
]
