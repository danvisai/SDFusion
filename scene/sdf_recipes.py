"""Style recipes that compose primitive SDFs into a styled building from a
footprint + target height + seed.

Each recipe is a function:
    recipe(polygon_xz: np.ndarray, target_height: float, seed: int = 0,
           **params) -> SDF
Returns an SDF callable over (Q, 3) torch points in world coordinates,
oriented Y-up, with the polygon laid in the XZ plane.

The 8 categorical styles match the cross-cutting style bank in the plan:
    modern, colonial, victorian, industrial, craftsman,
    mediterranean, contemporary, public_civic.

Each recipe is intentionally cheap to evaluate (10s of primitives) and
produces visually distinct shapes; intricate window/door subtractions can be
layered on later without changing the recipe signature.
"""
from __future__ import annotations
import math
from typing import Callable, Optional

import numpy as np
import torch

try:                                  # true polygon offsets for parapets/eaves (see _offset_polygon)
    from shapely.geometry import Polygon as _Polygon
except ImportError:                   # degrade to skipping those features, never to spikes
    _Polygon = None

from scene.sdf_primitives import (
    SDF,
    sdf_box, sdf_cone_y, sdf_cylinder_y, sdf_gable_roof, sdf_hip_roof,
    sdf_polygon_prism, sdf_rounded_box, sdf_smooth_union, sdf_sphere,
    sdf_subtract, sdf_translate, sdf_union, sdf_intersect,
)


STYLES = (
    "modern", "colonial", "victorian", "industrial",
    "craftsman", "mediterranean", "contemporary", "public_civic",
)


# --- helpers ----------------------------------------------------------------

def _polygon_bbox(poly: np.ndarray):
    """Return (cx, cz, width, depth) for an XZ polygon."""
    x_min, z_min = poly.min(axis=0)
    x_max, z_max = poly.max(axis=0)
    return float((x_min + x_max) / 2.0), float((z_min + z_max) / 2.0), \
           float(x_max - x_min), float(z_max - z_min)


def _body(poly: np.ndarray, height: float) -> SDF:
    """The footprint-faithful body (polygon extrude from y=0 to y=height)."""
    return sdf_polygon_prism(poly, height)


def _clip_roof_to_footprint(roof: SDF, poly: np.ndarray, y_min: float,
                            y_max: float) -> SDF:
    """Intersect a roof SDF with a tall prism over the polygon so the roof
    extent is clipped to the footprint XZ outline (eaves stop at facade)."""
    prism = sdf_translate(
        sdf_polygon_prism(poly, y_max - y_min), (0.0, y_min, 0.0)
    )
    return sdf_intersect(roof, prism)


# --- recipes (each: (poly, height, seed, **params) -> SDF) ------------------

def recipe_modern(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Flat roof, slight parapet, optional small mechanical box."""
    rng = np.random.default_rng(seed)
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    # Parapet: a thin ring around the top edge.
    parapet_h = max(0.4, height * 0.05)
    parapet = sdf_translate(sdf_polygon_prism(poly, parapet_h),
                            (0.0, height, 0.0))
    inner_poly = _shrink_polygon(poly, max(min(w, d) * 0.04, 0.15))
    if inner_poly is None:
        # Footprint too thin to carry a ring: a solid roof slab is the honest result. The old
        # radial shrink instead produced a self-intersecting inner polygon here, and the
        # subtraction left the spike slivers diagnosed in #51.
        out = sdf_union(body, parapet)
    else:
        parapet_inner = sdf_translate(
            sdf_polygon_prism(inner_poly, parapet_h + 0.2),
            (0.0, height - 0.1, 0.0),
        )
        out = sdf_union(body, sdf_subtract(parapet, parapet_inner))
    # Mechanical box on the roof (50% chance, off-center).
    if rng.random() < 0.6:
        mech_w = min(w, d) * 0.18
        mech_d = mech_w
        mech_h = max(0.7, height * 0.07)
        off_x = (rng.random() - 0.5) * w * 0.35
        off_z = (rng.random() - 0.5) * d * 0.35
        mech = sdf_translate(
            sdf_box((mech_w / 2.0, mech_h / 2.0, mech_d / 2.0)),
            (cx + off_x, height + mech_h / 2.0 + parapet_h * 0.4, cz + off_z),
        )
        out = sdf_union(out, mech)
    return out


def recipe_colonial(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Body + symmetric gable roof; optional center chimney."""
    rng = np.random.default_rng(seed)
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    roof_h = min(w, d) * 0.45  # pitch ~30 deg
    # Ridge along the longer axis.
    long_axis_x = w >= d
    if long_axis_x:
        roof = sdf_gable_roof(width=w, depth=d, height=height,
                              roof_height=roof_h, center_xz=(cx, cz))
    else:
        # Rotate gable: ridge along Z by swapping width/depth roles. We achieve
        # this by intersecting with a rotated polygon; simpler is to use a hip
        # with the long side as ridge. For v1 keep gable axis-aligned to whichever
        # is longer.
        roof = sdf_gable_roof(width=d, depth=w, height=height,
                              roof_height=roof_h, center_xz=(cx, cz))
        # Note: a true axis-aware gable would require oriented bbox; this is a
        # reasonable approximation for axis-aligned-ish polygons.
    roof = _clip_roof_to_footprint(roof, poly, height, height + roof_h)
    out = sdf_union(body, roof)
    if rng.random() < 0.7:
        ch_w = max(0.4, min(w, d) * 0.07)
        ch_h = roof_h * 0.85 + 0.6
        chimney = sdf_translate(
            sdf_box((ch_w, ch_h / 2.0, ch_w)),
            (cx + (rng.random() - 0.5) * w * 0.25, height + ch_h / 2.0 + roof_h * 0.4, cz),
        )
        out = sdf_union(out, chimney)
    return out


def recipe_victorian(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Body + hipped roof + small central tower with cone spire."""
    rng = np.random.default_rng(seed)
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    roof_h = min(w, d) * 0.40
    roof = sdf_hip_roof(width=w, depth=d, height=height, roof_height=roof_h,
                        center_xz=(cx, cz))
    roof = _clip_roof_to_footprint(roof, poly, height, height + roof_h)
    out = sdf_union(body, roof)
    # Tower
    tower_r = min(w, d) * 0.16
    tower_h = height * 0.4
    tower_top = height + roof_h + tower_h
    tower = sdf_translate(
        sdf_cylinder_y(tower_r, tower_h),
        (cx + w * 0.15, height + roof_h + tower_h / 2.0, cz + d * 0.15),
    )
    # Conical spire
    spire_h = tower_h * 0.8
    spire = sdf_translate(
        sdf_cone_y(angle_deg=20.0, height=spire_h),
        (cx + w * 0.15, tower_top, cz + d * 0.15),
    )
    out = sdf_union(out, tower)
    out = sdf_union(out, spire)
    # Decorative bay window (smooth-union'd box near the front facade)
    bay_w = min(w, d) * 0.2
    bay_h = height * 0.55
    bay_d = bay_w * 0.6
    bay = sdf_translate(
        sdf_box((bay_w / 2.0, bay_h / 2.0, bay_d / 2.0)),
        (cx - w * 0.25, bay_h / 2.0, cz + d / 2.0 + bay_d * 0.4),
    )
    out = sdf_smooth_union(out, bay, k=0.4)
    return out


def recipe_industrial(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Tall body + flat roof with cornice + small rooftop unit."""
    rng = np.random.default_rng(seed)
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    # Roof slab
    roof_slab = sdf_translate(sdf_polygon_prism(poly, 0.3),
                              (0.0, height, 0.0))
    # Slightly bigger overhang (eaves)
    eaves = sdf_translate(sdf_polygon_prism(_expand_polygon(poly, min(w, d) * 0.03), 0.18),
                          (0.0, height - 0.05, 0.0))
    out = sdf_union(body, sdf_union(roof_slab, eaves))
    # Vent stack
    stack_r = max(0.3, min(w, d) * 0.05)
    stack_h = height * 0.20
    stack = sdf_translate(
        sdf_cylinder_y(stack_r, stack_h),
        (cx + w * 0.18, height + stack_h / 2.0 + 0.3, cz + d * 0.05),
    )
    out = sdf_union(out, stack)
    return out


def recipe_craftsman(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Low-pitch hip roof, large eaves, optional porch slab."""
    rng = np.random.default_rng(seed)
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    roof_h = min(w, d) * 0.20
    # Slightly oversized roof for eaves
    eaves_poly = _expand_polygon(poly, min(w, d) * 0.03)
    eaves_body = sdf_translate(sdf_polygon_prism(eaves_poly, 0.20),
                               (0.0, height - 0.05, 0.0))
    roof_inner = sdf_hip_roof(width=w * 1.02, depth=d * 1.02, height=height,
                              roof_height=roof_h, center_xz=(cx, cz))
    roof = _clip_roof_to_footprint(roof_inner, eaves_poly, height, height + roof_h)
    out = sdf_union(body, sdf_union(eaves_body, roof))
    if rng.random() < 0.5:
        porch_w = w * 0.55
        porch_d = max(min(w, d) * 0.2, 1.0)
        porch_h = max(0.3, height * 0.03)
        porch = sdf_translate(
            sdf_box((porch_w / 2.0, porch_h / 2.0, porch_d / 2.0)),
            (cx, porch_h / 2.0, cz + d / 2.0 + porch_d / 2.0),
        )
        out = sdf_union(out, porch)
    return out


def recipe_mediterranean(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Very low-pitch hip + edge band giving terra-cotta tile feel."""
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    roof_h = min(w, d) * 0.14
    roof = sdf_hip_roof(width=w * 1.04, depth=d * 1.04, height=height,
                        roof_height=roof_h, center_xz=(cx, cz))
    eaves_poly = _expand_polygon(poly, min(w, d) * 0.04)
    roof = _clip_roof_to_footprint(roof, eaves_poly, height, height + roof_h)
    edge_band = sdf_translate(sdf_polygon_prism(eaves_poly, 0.25),
                              (0.0, height, 0.0))
    return sdf_union(body, sdf_union(edge_band, roof))


def recipe_contemporary(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Offset volume stack: main body + a shifted upper box."""
    rng = np.random.default_rng(seed)
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    upper_h = height * 0.45
    upper_w = w * 0.65
    upper_d = d * 0.7
    offx = (rng.random() - 0.5) * w * 0.30
    offz = (rng.random() - 0.5) * d * 0.30
    upper = sdf_translate(sdf_box((upper_w / 2.0, upper_h / 2.0, upper_d / 2.0)),
                          (cx + offx, height + upper_h / 2.0, cz + offz))
    return sdf_smooth_union(body, upper, k=0.4)


def recipe_public_civic(poly: np.ndarray, height: float, seed: int = 0, **_) -> SDF:
    """Body + central dome + small symmetric flanking volumes."""
    body = _body(poly, height)
    cx, cz, w, d = _polygon_bbox(poly)
    # Central dome
    dome_r = min(w, d) * 0.28
    dome = sdf_translate(sdf_sphere(dome_r), (cx, height + dome_r * 0.55, cz))
    # Drum (cylinder) under the dome
    drum_h = dome_r * 0.6
    drum = sdf_translate(sdf_cylinder_y(dome_r * 0.95, drum_h),
                         (cx, height + drum_h / 2.0, cz))
    out = sdf_union(body, sdf_union(drum, dome))
    # Two flanking lower volumes left/right of dome
    flank_w = w * 0.18
    flank_h = height * 0.25
    flank_d = d * 0.5
    left = sdf_translate(
        sdf_box((flank_w / 2.0, flank_h / 2.0, flank_d / 2.0)),
        (cx - w * 0.35, height + flank_h / 2.0, cz),
    )
    right = sdf_translate(
        sdf_box((flank_w / 2.0, flank_h / 2.0, flank_d / 2.0)),
        (cx + w * 0.35, height + flank_h / 2.0, cz),
    )
    out = sdf_union(out, sdf_union(left, right))
    return out


# --- registry + driver -------------------------------------------------------

RECIPES: dict[str, Callable[..., SDF]] = {
    "modern": recipe_modern,
    "colonial": recipe_colonial,
    "victorian": recipe_victorian,
    "industrial": recipe_industrial,
    "craftsman": recipe_craftsman,
    "mediterranean": recipe_mediterranean,
    "contemporary": recipe_contemporary,
    "public_civic": recipe_public_civic,
}


def build_styled_sdf(style: str, polygon_xz: np.ndarray, target_height: float,
                     seed: int = 0, **params) -> SDF:
    if style not in RECIPES:
        raise KeyError(f"Unknown style '{style}'. Known: {list(RECIPES)}")
    return RECIPES[style](polygon_xz, target_height, seed, **params)


# --- 2D polygon helpers ------------------------------------------------------

def _offset_polygon(poly: np.ndarray, amount: float) -> Optional[np.ndarray]:
    """True polygon offset: positive `amount` outward, negative inward.

    Replaces a radial scale-toward-the-centroid, which was wrong for **concave** footprints
    (#51). A real OSM L/U/Z plan frequently has its centroid *outside* the polygon, so scaling
    vertices toward it produced a self-intersecting result; subtracting that from the outer
    prism left thin slivers, which marching cubes rendered as a forest of vertical spikes.
    Spikes appeared on Munich's irregular footprints and never on rectangular ones — exactly
    the signature of a centroid-relative operation meeting a concave polygon.

    Mitre joins keep building corners sharp; round joins would bevel every corner.

    Returns None when the offset collapses the polygon (an inward offset wider than the local
    half-width), so callers skip the feature rather than build a degenerate sliver. Also
    returns None if shapely is unavailable — losing a parapet is preferable to drawing spikes.
    """
    if _Polygon is None:
        return None
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return None
    g = _Polygon(p)
    if not g.is_valid:
        g = g.buffer(0)                      # repair a self-intersecting input ring
    out = g.buffer(amount, join_style=2, mitre_limit=4.0)   # 2 = mitre
    if out.is_empty:
        return None
    if out.geom_type != "Polygon":           # inward offset can split into several pieces
        out = max(out.geoms, key=lambda q: q.area)
    coords = np.asarray(out.exterior.coords[:-1], dtype=np.float32)
    return coords if len(coords) >= 3 else None


def _shrink_polygon(poly: np.ndarray, amount: float) -> Optional[np.ndarray]:
    """Inward offset by `amount`. None when the footprint is too thin to carry it."""
    return _offset_polygon(poly, -abs(float(amount)))


def _expand_polygon(poly: np.ndarray, amount: float) -> np.ndarray:
    """Outward offset by `amount`. Always yields a polygon (falls back to the input)."""
    out = _offset_polygon(poly, abs(float(amount)))
    return out if out is not None else np.asarray(poly, dtype=np.float32)
