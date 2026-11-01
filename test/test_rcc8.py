import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from torch_morph_rcc8.mask_utils import mask_to_polygon
from torch_morph_rcc8.rcc8 import classify_rcc8, RCC8

SHAPE = (100, 100)


def make_rect_mask(row_slice, col_slice):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[row_slice, col_slice] = 1
    return mask


def test_dc_disconnected():
    a = make_rect_mask(slice(10, 30), slice(10, 30))
    b = make_rect_mask(slice(60, 80), slice(60, 80))
    assert classify_rcc8(mask_to_polygon(a), mask_to_polygon(b)) == RCC8.DC


def test_ec_externally_connected():
    a = make_rect_mask(slice(20, 50), slice(20, 50))
    b = make_rect_mask(slice(50, 80), slice(20, 50))
    assert classify_rcc8(mask_to_polygon(a), mask_to_polygon(b)) == RCC8.EC


def test_po_partial_overlap():
    a = make_rect_mask(slice(20, 50), slice(20, 50))
    b = make_rect_mask(slice(35, 65), slice(35, 65))
    assert classify_rcc8(mask_to_polygon(a), mask_to_polygon(b)) == RCC8.PO


def test_eq_equal():
    a = make_rect_mask(slice(20, 50), slice(20, 50))
    b = make_rect_mask(slice(20, 50), slice(20, 50))
    assert classify_rcc8(mask_to_polygon(a), mask_to_polygon(b)) == RCC8.EQ


def test_ntpp_and_ntppi():
    outer = make_rect_mask(slice(10, 90), slice(10, 90))
    inner = make_rect_mask(slice(40, 60), slice(40, 60))  # Strictly inside interior
    poly_outer, poly_inner = mask_to_polygon(outer), mask_to_polygon(inner)
    assert classify_rcc8(poly_inner, poly_outer) == RCC8.NTPP
    assert classify_rcc8(poly_outer, poly_inner) == RCC8.NTPPi


def test_tpp_and_tppi():
    outer = make_rect_mask(slice(10, 90), slice(10, 90))
    inner = make_rect_mask(slice(10, 60), slice(40, 60))  # Shares top boundary with outer
    poly_outer, poly_inner = mask_to_polygon(outer), mask_to_polygon(inner)
    assert classify_rcc8(poly_inner, poly_outer) == RCC8.TPP
    assert classify_rcc8(poly_outer, poly_inner) == RCC8.TPPi
