"""Extract building footprints from a user image (e.g. an OSM-map screenshot or a
footprint mask) -> polygons in meters, for the web demo's image -> town flow.

Heuristic: buildings are the distinct blobs against a dominant background. Otsu threshold,
take the minority class as buildings (with an `invert` override), drop tiny specks, and
contour each region. Works well on footprint masks and high-contrast maps.
"""

from __future__ import annotations

import io
from typing import List, Tuple

import numpy as np


def extract_footprints(img_bytes: bytes, n_max: int = 50, min_area_frac: float = 0.0008,
                       invert: bool = False) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Return [(polygon_px (P,2) as (col,row), ...)] and image (H,W). Largest blobs first."""
    from PIL import Image
    from skimage import measure, filters, morphology

    im = Image.open(io.BytesIO(img_bytes)).convert("L")
    a = np.asarray(im, dtype=np.float64) / 255.0
    H, W = a.shape
    try:
        thr = filters.threshold_otsu(a)
    except Exception:
        thr = 0.5
    mask = a < thr                       # darker = building (common in maps/masks)
    if invert:
        mask = ~mask
    if mask.mean() > 0.5:                # buildings should be the minority -> flip if not
        mask = ~mask
    mask = morphology.remove_small_objects(mask, min_size=int(min_area_frac * a.size))
    mask = morphology.remove_small_holes(mask, area_threshold=int(0.0003 * a.size))
    lbl = measure.label(mask)
    regions = sorted(measure.regionprops(lbl), key=lambda r: -r.area)[:n_max]
    polys = []
    for r in regions:
        sub = (lbl == r.label)
        cs = measure.find_contours(sub.astype(float), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        if len(c) < 4:
            continue
        if len(c) > 16:
            c = c[np.linspace(0, len(c) - 1, 16, dtype=int)]
        polys.append(c[:, ::-1].astype(np.float32))   # (row,col) -> (x=col, y=row)
    return polys, (H, W)


def to_meters(polys_px, img_hw, meters_across: float):
    """Scale pixel polygons to a metric, centered frame. Image x->world x, image y->world z
    (flipped so 'up' in the image is +z). Returns list of (polygon_local_m, centroid_xz)."""
    H, W = img_hw
    scale = meters_across / max(W, 1)
    out = []
    for p in polys_px:
        m = p.copy().astype(np.float32)
        m[:, 0] = (m[:, 0] - W / 2) * scale          # x
        m[:, 1] = (H / 2 - m[:, 1]) * scale          # z (flip)
        cen = m.mean(0)
        out.append((m - cen, cen))                    # local poly + world centroid
    return out
