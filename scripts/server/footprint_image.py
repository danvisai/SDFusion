"""Extract building footprints from a user image (e.g. an OSM-map screenshot or a
footprint mask) -> polygons in meters, for the web demo's image -> town flow.

Heuristic: buildings are the distinct blobs against a dominant background. Otsu threshold,
take the minority class as buildings (with an `invert` override), drop tiny specks, and
contour each region. Works well on footprint masks and high-contrast maps.
"""

from __future__ import annotations

import io
import warnings
from typing import List, Optional, Tuple

import numpy as np

# Safety valve for `simplify_px`: every polygon vertex becomes a draggable handle in the town
# editor, so a pathological outline must not be allowed to emit thousands of them.
MAX_POLY_POINTS = 200


def _otsu_threshold(u8: np.ndarray) -> float:
    """Otsu split of a uint8 image, returned in the [0, 1] domain the caller works in.

    Deliberately computed on the **uint8** array: skimage 0.26 + numpy 2.2 raise inside
    `threshold_otsu`'s histogram for a float image in [0, 1] ("operands could not be broadcast
    together with shapes (256,) (258,)"). That used to be swallowed by a bare `except`, so every
    extraction silently ran at a fixed 0.5 instead of the real threshold -- 0.275 on the Munich
    sample. High-contrast maps survive that; anything else quietly mis-segments.
    """
    from skimage import filters
    try:
        return float(filters.threshold_otsu(u8)) / 255.0
    except Exception as e:                       # keep working, but never silently
        warnings.warn(f"threshold_otsu failed ({type(e).__name__}: {e}); falling back to 0.5, "
                      f"which is only safe on near-binary images", RuntimeWarning, stacklevel=2)
        return 0.5


def _simplify_corners(contour: np.ndarray, tol_px: float,
                      max_points: int = MAX_POLY_POINTS) -> np.ndarray:
    """Douglas-Peucker simplification: spend vertices where the outline TURNS.

    The alternative this replaces -- resampling to a fixed count uniformly by index -- spends most
    of its budget on long straight runs and can miss corners entirely, which on architecture is
    exactly the wrong trade: a rectangle needs 4 points and a city block needs 40, and neither is
    served by 16 evenly-spaced ones.
    """
    from skimage import measure

    def _open(p):                                # approximate_polygon returns a closed ring
        return p[:-1] if len(p) > 3 and np.allclose(p[0], p[-1]) else p

    p = _open(measure.approximate_polygon(contour, tolerance=tol_px))
    while len(p) > max_points and tol_px < 64:   # bounded: coarsen until it fits the handle budget
        tol_px = max(tol_px * 1.5, 0.25)         # max(): tolerance 0 means "keep everything", and
        p = _open(measure.approximate_polygon(contour, tolerance=tol_px))   # 0*1.5 never escalates
    return p


def building_mask(u8: np.ndarray, min_area_frac: float = 0.0008,
                  invert: bool = False) -> np.ndarray:
    """Grey image -> boolean building mask. Public so tests score polygons against the very mask
    they were traced from, rather than keeping a second copy of this recipe that can drift."""
    from skimage import morphology
    a = u8.astype(np.float64) / 255.0
    # `<=`, not `<`: skimage's convention is that foreground is strictly ABOVE the returned
    # threshold, so the dark class is everything up to and including it. With `<`, a two-level
    # image -- which is exactly what these OSM renders are, 70 on 245 -- gets Otsu's 70 back and
    # then excludes every building pixel, yielding an empty mask.
    mask = a <= _otsu_threshold(u8)      # darker = building (common in maps/masks)
    if invert:
        mask = ~mask
    if mask.mean() > 0.5:                # buildings should be the minority -> flip if not
        mask = ~mask
    mask = morphology.remove_small_objects(mask, min_size=int(min_area_frac * a.size))
    return morphology.remove_small_holes(mask, area_threshold=int(0.0003 * a.size))


def extract_footprints(img_bytes: bytes, n_max: int = 50, min_area_frac: float = 0.0008,
                       invert: bool = False, simplify_px: Optional[float] = None
                       ) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Return [(polygon_px (P,2) as (col,row), ...)] and image (H,W). Largest blobs first.

    `simplify_px` picks how each contour is reduced to a polygon:
      None (default) -- resample to 16 points uniformly by index. Kept as the default because it
                        is what `inference_service.py`'s older `/generate_from_image` shipped with.
      float          -- Douglas-Peucker at that pixel tolerance (see `_simplify_corners`). Measured
                        against the mask it came from, 1.0 beats the uniform default on every
                        bundled sample and usually with FEWER vertices: Munich 0.882 -> 0.914 union
                        IoU (ceiling 0.920), Lafayette 0.984 -> 0.995 at half the vertex count, and
                        synthetic blocks stay exact at 4 points per building instead of 16.
    """
    from PIL import Image
    from skimage import measure, morphology

    im = Image.open(io.BytesIO(img_bytes)).convert("L")
    u8 = np.asarray(im)
    H, W = u8.shape
    mask = building_mask(u8, min_area_frac=min_area_frac, invert=invert)
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
        if simplify_px is None:
            if len(c) > 16:
                c = c[np.linspace(0, len(c) - 1, 16, dtype=int)]
        else:
            c = _simplify_corners(c, simplify_px)
            if len(c) < 3:
                continue                              # coarsened away to nothing
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
