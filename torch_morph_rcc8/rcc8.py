from enum import Enum
from shapely.geometry import Polygon


class RCC8(Enum):
    """
    Eight Jointly Exhaustive and Pairwise Disjoint (JEPD) base relations
    of the Region Connection Calculus (Randell, Cui, & Cohn, 1992, KR '92, pp. 165-176).
    """
    DC = "disconnected"
    EC = "externally_connected"
    PO = "partial_overlap"
    EQ = "equal"
    TPP = "tangential_proper_part"
    NTPP = "non_tangential_proper_part"
    TPPi = "tangential_proper_part_inverse"
    NTPPi = "non_tangential_proper_part_inverse"


def classify_rcc8(a: Polygon, b: Polygon) -> RCC8:
    """Classifies the topological relation between two 2D Shapely polygons."""
    if a.equals(b):
        return RCC8.EQ
    if a.disjoint(b):
        return RCC8.DC
    if a.touches(b):
        return RCC8.EC

    a_in_b = a.within(b)
    b_in_a = b.within(a)

    if a_in_b:
        return RCC8.TPP if a.boundary.intersects(b.boundary) else RCC8.NTPP
    if b_in_a:
        return RCC8.TPPi if a.boundary.intersects(b.boundary) else RCC8.NTPPi
    return RCC8.PO
