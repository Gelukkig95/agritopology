import numpy as np
from skimage import measure
from shapely.geometry import Polygon


def mask_to_polygon(mask: np.ndarray, level: float = 0.5) -> Polygon | None:
    """
    Converts a 2D binary/probability mask into a continuous Shapely Polygon
    using marching squares (default level=0.5 treats pixels as unit areas).
    """
    contours = measure.find_contours(mask.astype(float), level=level)
    if not contours:
        return None
    largest = max(contours, key=len)
    if len(largest) < 3:
        return None
    coords = largest[:, [1, 0]]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly
