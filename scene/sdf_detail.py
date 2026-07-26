"""Procedural architectural detail primitives (additive — does NOT touch
sdf_primitives.py / sdf_recipes.py).

Turns a boxy recipe building into a less-boxy one by composing solid, differentiable
detail onto its SDF:
  - floor bands / cornices  (union thin ledges around the perimeter at each floor line)
  - a plinth                (union a slightly wider base block)
  - window recesses         (subtract a repeated grid of window boxes from the facades)

All detail is generated from the building's footprint bbox + height via IQ-style limited
domain repetition, so a full window grid is one cheap SDF eval. Solid by construction
(union ledges / subtract shallow recesses) — it never opens the interior, which is exactly
what the broken-GT supervision could not guarantee. Differentiable in its parameters, so a
generative head can later predict them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from scene.sdf_primitives import (SDF, sdf_cylinder_y, sdf_cone_y, sdf_box,
                                   sdf_translate, sdf_union, sdf_smooth_union)


def _box_dist(q: torch.Tensor, he: torch.Tensor) -> torch.Tensor:
    """Signed distance to an axis-aligned box (local coords q, half-extents he)."""
    d = q.abs() - he
    return torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1) + torch.clamp(d.amax(dim=-1), max=0.0)


def _rep1d(c: torch.Tensor, spacing: float, lo: float, n: int) -> torch.Tensor:
    """Limited 1-D domain repetition: local offset of c from the nearest of `n` cell
    centers spaced `spacing` apart starting at `lo`."""
    idx = torch.round((c - lo) / spacing).clamp(0, n - 1)
    return c - (lo + idx * spacing)


@dataclass
class DetailParams:
    # floors
    floor_h: float = 3.2            # storey height (m)
    floor0: float = 1.2            # first band/window height above base
    # window grid
    windows: bool = True
    win_h: float = 1.5            # window height
    win_w: float = 1.1            # window width
    win_inset: float = 0.35       # recess depth into the facade
    win_spacing: float = 2.6      # horizontal spacing between window centers
    win_margin: float = 1.4       # keep windows this far from corners
    # bands / cornice
    bands: bool = True
    band_protrude: float = 0.22
    band_h: float = 0.28
    cornice_protrude: float = 0.5
    cornice_h: float = 0.6
    # plinth
    plinth: bool = True
    plinth_h: float = 1.0
    plinth_expand: float = 0.4
    blend: float = 0.06            # smooth-union/subtract radius (0 = hard)


# ---------------------------------------------------------------------------
# Generative detail: a per-style PRIOR over the visible detail params, with real
# within-style variance (one-to-many by design) so a trained head that samples it gives
# genuine facade diversity — unlike the recipe params, whose conditional was ~deterministic.
# The vector below is what the generative detail head predicts (the "required params").
# ---------------------------------------------------------------------------

DETAIL_FIELDS = ["floor_h", "win_h", "win_w", "win_inset", "win_spacing", "win_margin",
                 "band_protrude", "band_h", "cornice_protrude", "cornice_h",
                 "plinth_h", "plinth_expand"]

# (mean, std) per field per style. Means encode architectural character; std gives the
# within-style diversity a generative head can sample.
_M = {
    #            floor_h  win_h  win_w win_ins win_sp win_mrg band_pr band_h corn_pr corn_h plnth_h plnth_x
    "modern":        [3.4, 1.9, 1.6, 0.30, 2.3, 1.3, 0.10, 0.18, 0.30, 0.40, 0.8, 0.25],
    "contemporary":  [3.6, 2.2, 2.0, 0.22, 2.6, 1.4, 0.06, 0.14, 0.22, 0.30, 0.7, 0.20],
    "colonial":      [3.0, 1.4, 1.0, 0.34, 2.1, 1.5, 0.20, 0.26, 0.45, 0.55, 1.0, 0.30],
    "victorian":     [3.1, 2.1, 0.9, 0.40, 2.0, 1.5, 0.28, 0.32, 0.70, 0.85, 1.1, 0.40],
    "industrial":    [4.0, 1.6, 2.1, 0.28, 3.4, 1.6, 0.05, 0.12, 0.18, 0.25, 0.6, 0.18],
    "craftsman":     [3.0, 1.3, 1.2, 0.32, 2.2, 1.4, 0.18, 0.24, 0.55, 0.65, 0.8, 0.30],
    "mediterranean": [3.2, 1.5, 1.0, 0.30, 2.6, 1.6, 0.14, 0.20, 0.40, 0.45, 0.9, 0.28],
    "public_civic":  [3.8, 2.3, 1.3, 0.40, 2.5, 1.7, 0.24, 0.30, 0.65, 0.80, 1.3, 0.45],
}
_S = {  # per-style std (within-style diversity). ~15-30% of mean on the visible knobs.
    s: [0.3, 0.35, 0.30, 0.07, 0.4, 0.2, 0.06, 0.06, 0.14, 0.16, 0.25, 0.10] for s in _M
}

DETAIL_LO = np.array([2.4, 0.8, 0.6, 0.12, 1.6, 0.8, 0.0, 0.06, 0.05, 0.10, 0.3, 0.05])
DETAIL_HI = np.array([4.8, 2.8, 2.4, 0.55, 4.2, 2.2, 0.45, 0.45, 0.95, 1.10, 1.8, 0.65])


def sample_detail_vector(style: str, rng: np.random.Generator) -> np.ndarray:
    m = np.array(_M.get(style, _M["modern"]))
    s = np.array(_S.get(style, _S["modern"]))
    return np.clip(rng.normal(m, s), DETAIL_LO, DETAIL_HI).astype(np.float32)


def vector_to_params(vec, **overrides) -> "DetailParams":
    v = np.clip(np.asarray(vec, np.float64), DETAIL_LO, DETAIL_HI)
    kw = {f: float(v[i]) for i, f in enumerate(DETAIL_FIELDS)}
    kw.update(overrides)
    return DetailParams(**kw)


# ---------------------------------------------------------------------------
# RICH LANDMARK primitives — grounded in the BuildingNet part labels
# (dome=id22, tower/minaret=id7, stairs/podium=id17), placed per building class
# by the real per-class occurrence the labels give. Solid by construction (all unions).
# ---------------------------------------------------------------------------

def sdf_dome(center, radius, dome_h) -> SDF:
    """Ellipsoid dome (half above the base when unioned at roof level)."""
    cx, cy, cz = center

    def f(p):
        x = (p[:, 0] - cx) / radius
        y = (p[:, 1] - cy) / max(dome_h, 1e-3)
        z = (p[:, 2] - cz) / radius
        return (torch.sqrt(x * x + y * y + z * z) - 1.0) * min(radius, dome_h)
    return f


def sdf_minaret(center, radius, height, spire_ratio=0.35) -> SDF:
    """Tower/minaret: cylinder shaft + conical spire on top.

    The spire's cone angle is derived from the shaft radius — sdf_cone_y(angle, h) has
    base radius h*tan(angle), so a fixed angle gave tall towers a giant base skirt
    (h=37m -> ~10m radius sheet; found via the detailizer pair factory 2026-06-11)."""
    import math as _m
    cx, cy, cz = center
    shaft_h = height * (1 - spire_ratio)
    spire_h = max(height * spire_ratio, 1e-3)
    shaft = sdf_translate(sdf_cylinder_y(radius, shaft_h), (cx, cy + shaft_h / 2, cz))
    ang = _m.degrees(_m.atan2(radius * 1.25, spire_h))   # base ~25% wider than the shaft
    spire = sdf_translate(sdf_cone_y(ang, spire_h), (cx, cy + shaft_h, cz))
    return sdf_union(shaft, spire)


def sdf_steps(center, half_w, half_d, base_y, n=4, rise=0.4, run=0.6) -> SDF:
    """Stepped podium/stairs at the base — `n` stacked boxes shrinking upward."""
    parts = []
    for i in range(n):
        hw = half_w + (n - i) * run
        hd = half_d + (n - i) * run
        yc = base_y + i * rise + rise / 2
        parts.append(sdf_translate(sdf_box((hw, rise / 2, hd)), (center[0], yc, center[1])))
    s = parts[0]
    for q in parts[1:]:
        s = sdf_union(s, q)
    return s


def apply_roof_shape(sdf: SDF, footprint, height, shape) -> SDF:
    """Union a roof primitive (gabled/hipped/pyramidal/dome) onto a building; flat = no-op.

    The roof solid is CLIPPED to the footprint prism: the gable/hip primitives span the
    axis-aligned bbox (and include a full-bbox body), which fattens any rotated or
    non-rectangular footprint up to its bounding box (found via the detailizer pair
    factory, 2026-06-11 — fine buildings gained ~250k voxels)."""
    from scene.sdf_primitives import sdf_gable_roof, sdf_hip_roof, sdf_polygon_prism, sdf_intersect
    if not shape or str(shape).lower() == "flat":
        return sdf
    x0, _, z0, x1, _, z1 = _bbox(footprint, height)
    w, d = float(x1 - x0), float(z1 - z0); cx, cz = float((x0 + x1) / 2), float((z0 + z1) / 2)
    rh = 0.35 * min(w, d); s = str(shape).lower()
    if any(k in s for k in ["gabl", "gambrel", "saltbox", "pitch"]):
        roof = sdf_gable_roof(w, d, height, rh, center_xz=(cx, cz))
    elif any(k in s for k in ["hip", "pyramid"]):
        roof = sdf_hip_roof(w, d, height, rh * (1.4 if "pyram" in s else 1.0), center_xz=(cx, cz))
    elif any(k in s for k in ["dome", "onion", "round"]):
        roof = sdf_dome((cx, height, cz), min(w, d) * 0.5, min(w, d) * 0.4)
    else:
        return sdf
    prism = sdf_polygon_prism(np.asarray(footprint, np.float32), height + rh + 1.0)
    eave = height - 0.5

    def roof_cap(p):                     # keep only the cap: the primitives include a
        return torch.maximum(roof(p), eave - p[..., 1])   # full body below the eave
    return sdf_union(sdf, sdf_intersect(roof_cap, prism))


def add_landmarks(base: SDF, footprint, height, *, dome=False, dome_h_ratio=0.55,
                  n_towers=0, tower_h_ratio=1.4, steps=False, blend=None) -> SDF:
    """Compose dome / corner towers / base steps onto a building COHERENTLY (Track-1 fix for
    the floating/duplicated-cones problem — see memory project_part_coherence_research):
      - towers sit FLUSH at the footprint corners (tangent INSIDE so they overlap the walls),
        base on the ground, piercing the roof -> they read as attached corner towers;
      - placement is SYMMETRIC and CAPPED (1=front-centre, 2=front pair, 4=all corners) so we
        never get a forest of disconnected cones;
      - the dome is sized to sit ON the roof (radius <= footprint);
      - EVERYTHING is smooth-unioned (IQ blend) so parts FUSE into the envelope, not stuck on.
    (Real learned coherence is Track 2: a part-proxy + global-mixing model.)"""
    x0, _, z0, x1, _, z1 = _bbox(footprint, height)
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    wx, wz = (x1 - x0) / 2, (z1 - z0) / 2
    rmin = min(wx, wz)
    bl = float(blend if blend is not None else max(0.06 * min(2 * wx, 2 * wz), 0.5))  # size-relative fuse
    parts = []
    if dome:
        r = min(rmin * 0.6, rmin)                      # sits on the roof, radius within footprint
        parts.append(sdf_dome((cx, height, cz), r, r * dome_h_ratio * 2))
    n = int(max(0, min(n_towers, 4)))
    tr = rmin * 0.18
    sel = {1: [(0, -1)], 2: [(-1, -1), (1, -1)],
           3: [(-1, -1), (1, -1), (0, 1)],
           4: [(-1, -1), (1, -1), (-1, 1), (1, 1)]}.get(n, [])
    for (sx, sz) in sel:
        tx = cx + sx * max(wx - tr, 0.0)               # tangent inside the corner -> overlaps the wall
        tz = cz + sz * max(wz - tr, 0.0)
        parts.append(sdf_minaret((tx, 0.0, tz), tr, height * tower_h_ratio))
    if steps:
        from scene.sdf_primitives import sdf_polygon_prism
        run = rmin * 0.12
        raw_steps = sdf_steps((cx, cz), wx * 0.9, wz * 0.9, 0.0,
                              n=4, rise=max(height * 0.04, 0.3), run=run)
        # clip to the (dilated) footprint so the platform follows the actual polygon —
        # the raw steps span the axis-aligned bbox, a huge apron on rotated footprints
        prism = sdf_polygon_prism(np.asarray(footprint, np.float32), height)
        margin = 4 * run

        def clipped_steps(p, _s=raw_steps, _pr=prism, _m=margin):
            return torch.maximum(_s(p), _pr(p) - _m)
        parts.append(clipped_steps)

    def f(p):
        v = base(p)
        for s in parts:
            sv = s(p)
            h = torch.clamp(0.5 + 0.5 * (sv - v) / bl, 0.0, 1.0)   # IQ smooth-union -> fuse, not float
            v = v * h + sv * (1 - h) - bl * h * (1 - h)
        return v
    return f


# Per-class landmark occurrence — REAL numbers from the BuildingNet part labels
# (dome=id22, tower=id7, steps=id17), measured by scripts/server/test_landmarks.py.
# Religious buildings carry towers 78% / domes 38%; commercial/residential ~none.
CLASS_LANDMARK_PROB = {
    "RELIGIOUS":   {"dome": 0.38, "tower": 0.78, "steps": 0.03},
    "PUBLIC":      {"dome": 0.05, "tower": 0.05, "steps": 0.04},
    "COMMERCIAL":  {"dome": 0.00, "tower": 0.03, "steps": 0.02},
    "RESIDENTIAL": {"dome": 0.00, "tower": 0.00, "steps": 0.22},  # stoops/porches
}


def sample_landmarks(building_class: str, rng: np.random.Generator) -> dict:
    """Sample which landmarks a building gets, from the real per-class occurrence."""
    p = CLASS_LANDMARK_PROB.get(building_class.upper(), CLASS_LANDMARK_PROB["RESIDENTIAL"])
    dome = rng.random() < p["dome"]
    n_towers = (4 if rng.random() < 0.5 else 2) if rng.random() < p["tower"] else 0
    steps = rng.random() < p["steps"]
    return {"dome": dome, "n_towers": n_towers, "steps": steps}


# Real per-class FACADE stats from BuildingNet labels
# (scripts/extract_buildingnet_detail_stats.py -> outputs/buildingnet_detail_stats/stats.json):
# glazing = window/(window+wall) point ratio; roof pitched fraction.
CLASS_GLAZING = {"COMMERCIAL": 0.162, "PUBLIC": 0.066, "RELIGIOUS": 0.117, "RESIDENTIAL": 0.173}
CLASS_ROOF_PITCHED = {"COMMERCIAL": 0.15, "PUBLIC": 0.10, "RELIGIOUS": 0.27, "RESIDENTIAL": 0.08}


def ground_glazing(p: "DetailParams", building_class: str) -> "DetailParams":
    """Scale window area so the facade window-coverage matches the REAL per-class glazing
    ratio (preserves the style's window aspect + spacing)."""
    target = CLASS_GLAZING.get(building_class.upper(), 0.15)
    cov = (p.win_w * p.win_h) / max(p.win_spacing * p.floor_h, 1e-6)
    scale = float(np.clip((target / max(cov, 1e-4)) ** 0.5, 0.45, 1.9))
    p.win_w = float(np.clip(p.win_w * scale, DETAIL_LO[2], DETAIL_HI[2]))
    p.win_h = float(np.clip(p.win_h * scale, DETAIL_LO[1], DETAIL_HI[1]))
    return p


def sample_roof_shape(building_class: str, osm_roof, rng: np.random.Generator):
    """Roof type: real OSM `roof:shape` if present, else fall back to the BuildingNet
    class pitched-probability (religious mostly pitched, commercial/residential flat)."""
    if isinstance(osm_roof, str) and osm_roof:
        return osm_roof
    if rng.random() < CLASS_ROOF_PITCHED.get(building_class.upper(), 0.1):
        return rng.choice(["gabled", "hipped"])
    return "flat"


def _bbox(footprint, height):
    poly = np.asarray(footprint, dtype=np.float64)
    x0, z0 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x1, z1 = float(poly[:, 0].max()), float(poly[:, 1].max())
    return x0, 0.0, z0, x1, float(height), z1


def add_facade_detail(base: SDF, footprint, height, p: DetailParams = None) -> SDF:
    """Return a new SDF = base ∪ bands ∪ plinth, − window recesses."""
    p = p or DetailParams()
    x0, y0, z0, x1, y1, z1 = _bbox(footprint, height)
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    wx, wz = (x1 - x0) / 2, (z1 - z0) / 2
    n_floors = max(int((y1 - p.floor0) / p.floor_h), 1)
    nx = max(int((2 * wx - 2 * p.win_margin) / p.win_spacing), 1)  # windows along x (on z-faces)
    nz = max(int((2 * wz - 2 * p.win_margin) / p.win_spacing), 1)  # windows along z (on x-faces)
    x_lo = cx - (nx - 1) * p.win_spacing / 2
    z_lo = cz - (nz - 1) * p.win_spacing / 2
    def f(q: torch.Tensor) -> torch.Tensor:
        px, py, pz = q[:, 0], q[:, 1], q[:, 2]
        base_v = base(q)
        out = base_v

        # bands / cornice / plinth = the MASSING dilated by the protrusion, cut to a
        # y-slab — they follow the actual walls (setbacks, rotations, any polygon),
        # unlike the old bbox boxes which grew plates/aprons on non-box massing
        # (found via the detailizer pair factory, 2026-06-11)
        if p.bands:
            yb = _rep1d(py, p.floor_h, p.floor0, n_floors)
            band = torch.maximum(base_v - p.band_protrude, yb.abs() - p.band_h / 2)
            qyc = (py - (y1 - p.cornice_h / 2)).abs() - p.cornice_h / 2
            corn = torch.maximum(base_v - p.cornice_protrude, qyc)
            out = torch.minimum(out, torch.minimum(band, corn))

        if p.plinth:
            qyp = (py - p.plinth_h / 2).abs() - p.plinth_h / 2
            plinth = torch.maximum(base_v - p.plinth_expand, qyp)
            out = torch.minimum(out, plinth)

        if p.windows:
            yw = _rep1d(py, p.floor_h, p.floor0 + p.floor_h * 0.0, n_floors)
            he = torch.tensor([p.win_inset, p.win_h / 2, p.win_w / 2], device=q.device)
            # windows on the two x-normal faces (vary along z): box at nearer x face
            facex = torch.where((px - x0).abs() < (px - x1).abs(), x0, x1)
            zw = _rep1d(pz, p.win_spacing, z_lo, nz)
            winx = _box_dist(torch.stack([px - facex, yw, zw], -1), he)
            # windows on the two z-normal faces (vary along x)
            facez = torch.where((pz - z0).abs() < (pz - z1).abs(), z0, z1)
            xw = _rep1d(px, p.win_spacing, x_lo, nx)
            he2 = torch.tensor([p.win_w / 2, p.win_h / 2, p.win_inset], device=q.device)
            winz = _box_dist(torch.stack([xw, yw, pz - facez], -1), he2)
            windows = torch.minimum(winx, winz)
            out = torch.maximum(out, -windows)  # subtract the recesses

        return out

    return f


def add_door(base: SDF, footprint, height, door_w=1.8, door_h=2.5, inset=0.5,
             canopy=True) -> SDF:
    """Carve a ground-floor ENTRANCE door, centered on the building's FRONT (longer) face,
    + a thin protruding lintel/canopy above it. A universal architectural element — every
    building reads as a building partly because it has a recognizable ground entrance.
    Recess = subtract (like windows); canopy = union (solid)."""
    x0, _, z0, x1, _, z1 = _bbox(footprint, height)
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    wx, wz = (x1 - x0) / 2, (z1 - z0) / 2
    along_x = (2 * wx) >= (2 * wz)         # longer side is x -> door on a z-normal face
    dh = float(min(door_h, height * 0.9))
    dw = float(min(door_w, (2 * (wx if along_x else wz)) * 0.6))

    def f(q: torch.Tensor) -> torch.Tensor:
        out = base(q)
        px, py, pz = q[:, 0], q[:, 1], q[:, 2]
        if along_x:
            he = torch.tensor([dw / 2, dh / 2, inset], device=q.device)
            local = torch.stack([px - cx, py - dh / 2, pz - z0], -1)
        else:
            he = torch.tensor([inset, dh / 2, dw / 2], device=q.device)
            local = torch.stack([px - x0, py - dh / 2, pz - cz], -1)
        out = torch.maximum(out, -_box_dist(local, he))   # carve the doorway
        if canopy:
            cy = dh + 0.22
            if along_x:
                qx = (px - cx).abs() - (dw / 2 + 0.4)
                qz = (pz - z0).abs() - 0.5
            else:
                qx = (px - x0).abs() - 0.5
                qz = (pz - cz).abs() - (dw / 2 + 0.4)
            qy = (py - cy).abs() - 0.12
            lintel = _box_dist(torch.stack([qx, qy, qz], -1), torch.zeros(3, device=q.device))
            out = torch.minimum(out, lintel)
        return out
    return f


# ---------------------------------------------------------------------------
# ADDED-ELEMENT detail: a user-placed crude primitive (sculpt "add") gets its OWN facade
# treatment instead of staying a bare box — the gap found 2026-07-02 (SDXL texture-bakes
# geometry but can't invent structure; a bare CSG-added box stays bare). Classification is
# the same SHAPE->ARCH rule prototyped in scripts/server/facade_grammar.py (canonical here;
# facade_grammar imports it) — moved to scene/ so this module doesn't depend on scripts/.
# ---------------------------------------------------------------------------

def classify_shape(prim_size, prim_center, bbox, mode="add"):
    """Map the PLACED PRIMITIVE'S SHAPE -> an architectural element type: flat-horizontal ->
    balcony, flat plane / small -> window, tall@ground -> door, chunky -> bay, tall slender
    -> pilaster, thin-but-LARGE (relative to the building) -> wall. `prim_size` = half-extents.
    `bbox` = flat (x0,y0,z0,x1,y1,z1), the _bbox() convention used throughout this module. Any
    consistent unit system (prim_size/prim_center/bbox all in the same units — world meters
    here); the building's own scale (from `bbox`) is what separates a window-scale thin plane
    from a wall-scale one, since geometry alone can't tell a big pane from a small one."""
    hx, hy, hz = [abs(float(v)) for v in prim_size[:3]]
    x0, y0, z0, x1, y1, z1 = bbox
    lateral = max(hx, hz)
    ground = float(prim_center[1]) < y0 + 0.28 * (y1 - y0)
    bldg_scale = max(x1 - x0, z1 - z0, y1 - y0, 1e-6)
    if mode == "subtract":
        return "door" if (ground and hy > 0.12) else "window"
    if hy < 0.55 * lateral:                       # wide & short -> horizontal slab
        return "balcony"
    if hy > 1.7 * lateral:                         # tall & slender
        return "door" if ground else "pilaster"
    if min(hx, hy, hz) < 0.45 * np.median([hx, hy, hz]):   # a thin plane
        if 2 * lateral > 0.3 * bldg_scale or 2 * hy > 0.3 * bldg_scale:
            return "wall"                          # thin but LARGE -> a wall segment/panel
        return "door" if (ground and hy > 0.14) else "window"
    return "bay"                                   # a chunky box -> protruding bay


# Styles that read as favoring round/arched openings over flat-top ones (Mediterranean
# arcades, Romanesque/Gothic-leaning public+religious civic work, Victorian/Craftsman trim,
# colonial fanlights). "modern"/"contemporary"/"industrial" stay flat-top.
ARCH_STYLES = {"mediterranean", "colonial", "victorian", "public_civic", "craftsman"}


def _recess_dist(depth, y, lat, inset, half_h, half_w, arched=False):
    """SDF for a FRESH (single-shot) window/door-style recess (negative = inside the cavity):
    a flat-top box, or an ARCHED (round-top) opening — straight jamb sides up to the
    springline, a semicircular cap above (real arch construction: radius = half_w, inscribed
    within the same overall half_h so it reads correctly the first time it's carved). Finite
    in `depth` (clamped to +/-inset). `depth`/`y`/`lat` are (N,) tensors; `inset`/`half_h`/
    `half_w` are scalars fixed at construction time."""
    depth_d = depth.abs() - inset
    if not arched:
        box2d = torch.maximum(lat.abs() - half_w, y.abs() - half_h)
        return torch.maximum(box2d, depth_d)
    r = min(half_w, half_h * 1.6)
    straight_top = half_h - r
    y_c = (straight_top - half_h) / 2.0
    rect_hh = (straight_top + half_h) / 2.0
    rect2d = torch.maximum(lat.abs() - half_w, (y - y_c).abs() - rect_hh)
    disk2d = torch.sqrt(lat * lat + (y - straight_top) ** 2 + 1e-12) - r
    open2d = torch.minimum(rect2d, disk2d)
    return torch.maximum(open2d, depth_d)


def _nearest_wall_axis_sign(center, bbox):
    """Which of the 4 cardinal exterior walls a point is closest to -> (axis, sign):
    axis 0 = x-normal wall, axis 2 = z-normal wall; sign = +1/-1 = hi/lo face."""
    x0, _, z0, x1, _, z1 = bbox
    cx, _, cz = center
    cands = {(0, 1): abs(cx - x1), (0, -1): abs(cx - x0),
             (2, 1): abs(cz - z1), (2, -1): abs(cz - z0)}
    return min(cands, key=cands.get)


def add_tower_element_detail(base: SDF, center, half_size, style="modern", seed=None) -> SDF:
    """A user-added tower/pilaster-shaped primitive gets its OWN vertical window band (one
    column per face, scaled to fit its own height) + a cap ledge, so it reads as a facade
    element rather than a bare block. `center`/`half_size` in the SAME world-meter frame as
    `base` (Y measured from the building's ground) — mirrors add_facade_detail's window math,
    scoped to the primitive's own local span instead of the whole building."""
    rng = np.random.default_rng(0 if seed is None else int(seed))
    p = vector_to_params(sample_detail_vector(style, rng))
    arched = style in ARCH_STYLES
    cx, cy, cz = [float(v) for v in center]
    hx, hy, hz = [abs(float(v)) for v in half_size]
    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx
    z0, z1 = cz - hz, cz + hz
    floor_h = min(p.floor_h, max(hy * 0.9, 0.6))
    n_floors = max(int((2 * hy) / floor_h), 1)
    win_w = min(p.win_w, hz, hx) * 0.8
    win_h = min(p.win_h, floor_h * 0.55)
    inset = min(p.win_inset, min(hx, hz) * 0.6)

    def f(q: torch.Tensor) -> torch.Tensor:
        px, py, pz = q[:, 0], q[:, 1], q[:, 2]
        out = base(q)
        yw = _rep1d(py, floor_h, y0 + floor_h * 0.5, n_floors)
        facex = torch.where((px - x0).abs() < (px - x1).abs(), x0, x1)
        winx = _recess_dist(px - facex, yw, pz - cz, inset, win_h / 2, win_w / 2, arched)
        facez = torch.where((pz - z0).abs() < (pz - z1).abs(), z0, z1)
        winz = _recess_dist(pz - facez, yw, px - cx, inset, win_h / 2, win_w / 2, arched)
        windows = torch.minimum(winx, winz)
        span = (py - cy).abs() - hy                 # stay within the primitive's own extent
        windows = torch.maximum(windows, span)
        out = torch.maximum(out, -windows)           # carve the recesses
        qyc = (py - (y1 - 0.15)).abs() - 0.15
        cap = torch.maximum(out - 0.12, qyc)
        out = torch.minimum(out, cap)                # a small cap ledge at the top
        return out
    return f


def add_balcony_railing(base: SDF, center, half_size, wall_axis, wall_sign,
                        style="modern", seed=None) -> SDF:
    """A user-added wide/short (balcony-shaped) primitive gets a railing along its OUTWARD
    edge, so it reads as a balcony rather than a bare slab. Solid PARAPET wall, not thin
    balusters: at typical bake resolutions (~96 voxels over a whole building, ~0.2-0.3m/voxel)
    a realistic ~3-5cm baluster is sub-voxel and vanishes (found 2026-07-02 testing this).
    `wall_axis`/`wall_sign` from _nearest_wall_axis_sign — which exterior wall the balcony
    protrudes from, i.e. which edge of the slab faces away from the building."""
    del style, seed   # reserved for future per-style parapet height/thickness variance
    cx, cy, cz = [float(v) for v in center]
    hx, hy, hz = [abs(float(v)) for v in half_size]
    # wall_t is thicker than a real parapet (~10-15cm) because bake grids sample at
    # ~0.2-0.3m/voxel over a whole building — anything thinner than that is sub-voxel and
    # vanishes between sample points (confirmed empirically 2026-07-02: 0.10m -> 0 voxels
    # changed). min(hx,hz)*0.5 caps it so it never exceeds the balcony's own depth/width.
    rail_h = 0.95
    wall_t = max(0.35, min(hx, hz) * 0.5)
    top_y = cy + hy
    if wall_axis == 0:
        edge = cx + wall_sign * hx
        lat_c, half_lat = cz, hz
    else:
        edge = cz + wall_sign * hz
        lat_c, half_lat = cx, hx

    def f(q: torch.Tensor) -> torch.Tensor:
        out = base(q)
        px, py, pz = q[:, 0], q[:, 1], q[:, 2]
        lat = (pz if wall_axis == 0 else px) - lat_c
        depth = (px if wall_axis == 0 else pz) - edge
        he = torch.tensor([wall_t / 2, rail_h / 2, half_lat], device=q.device)
        parapet = _box_dist(torch.stack([depth, py - (top_y + rail_h / 2), lat], -1), he)
        return torch.minimum(out, parapet)
    return f


def add_wall_element_detail(base: SDF, center, half_size, style="modern", seed=None) -> SDF:
    """A user-added WALL segment (thin but LARGE relative to the building — a garden wall,
    courtyard wall, or facade extension, as opposed to a window/door-scale thin plane) gets a
    coping cap along its top edge, like a real capped wall, instead of reading as a bare slab.
    Unlike balcony/bay (which protrude from a specific building wall), a freestanding wall's
    own orientation is whichever of its X/Z half-extents is SMALLER = its thickness axis — it
    doesn't need (or necessarily match) the nearest building wall."""
    del style, seed   # reserved for future per-style coping profile variance
    cx, cy, cz = [float(v) for v in center]
    hx, hy, hz = [abs(float(v)) for v in half_size]
    # cap_h thicker than a real coping stone (~10-15cm): sub-voxel at typical bake resolution
    # (~0.2-0.3m/voxel) vanishes between sample points (same issue found + fixed for the
    # balcony parapet 2026-07-02).
    cap_h, cap_overhang = 0.35, 0.06
    top_y = cy + hy
    thin_x = hx <= hz                              # thickness runs along X (length along Z)
    depth_hh = (hx if thin_x else hz) + cap_overhang
    half_lat = (hz if thin_x else hx) + cap_overhang

    def f(q: torch.Tensor) -> torch.Tensor:
        out = base(q)
        px, py, pz = q[:, 0], q[:, 1], q[:, 2]
        depth = (px - cx) if thin_x else (pz - cz)
        lat = (pz - cz) if thin_x else (px - cx)
        he = torch.tensor([depth_hh, cap_h / 2, half_lat], device=q.device)
        cap = _box_dist(torch.stack([depth, py - (top_y + cap_h / 2), lat], -1), he)
        return torch.minimum(out, cap)
    return f


def add_bay_window_detail(base: SDF, center, half_size, wall_axis, wall_sign,
                          style="modern", seed=None) -> SDF:
    """A user-added chunky (bay-shaped) primitive gets a window on its outward face + a thin
    cap ledge over its own footprint, so it reads as a projecting bay window rather than a
    blank box (previously bay-shaped adds got the tower treatment, a poor fit for a squat
    chunky mass with only 1-2 floors' worth of height)."""
    rng = np.random.default_rng(0 if seed is None else int(seed))
    p = vector_to_params(sample_detail_vector(style, rng))
    arched = style in ARCH_STYLES
    cx, cy, cz = [float(v) for v in center]
    hx, hy, hz = [abs(float(v)) for v in half_size]
    top_y = cy + hy
    inset = min(p.win_inset, min(hx, hy, hz) * 0.5)
    if wall_axis == 0:
        face, win_hw = cx + wall_sign * hx, hz * 0.7
    else:
        face, win_hw = cz + wall_sign * hz, hx * 0.7
    win_hh = hy * 0.6

    def f(q: torch.Tensor) -> torch.Tensor:
        out = base(q)
        px, py, pz = q[:, 0], q[:, 1], q[:, 2]
        depth = (px if wall_axis == 0 else pz) - face
        lat = (pz if wall_axis == 0 else px) - (cz if wall_axis == 0 else cx)
        window = _recess_dist(depth, py - cy, lat, inset, win_hh, win_hw, arched)
        out = torch.maximum(out, -window)             # carve the bay's own window
        he = torch.tensor([hx + 0.05, 0.06, hz + 0.05], device=q.device)
        cap = _box_dist(torch.stack([px - cx, py - (top_y + 0.06), pz - cz], -1), he)
        return torch.minimum(out, cap)                # a small ledge/roof over the bay
    return f


def add_opening_element_detail(base: SDF, center, half_size, body_bbox,
                               style="modern", seed=None) -> SDF:
    """A user-added window/door-shaped primitive (a small protruding box — shutter, AC unit,
    signage, added door leaf) gets a proper inset pane/panel carved into its outward face,
    ARCHED when the style favors it, instead of reading as a featureless block."""
    del seed
    arched = style in ARCH_STYLES
    cx, cy, cz = [float(v) for v in center]
    hx, hy, hz = [abs(float(v)) for v in half_size]
    axis, sign = _nearest_wall_axis_sign(center, body_bbox)
    if axis == 0:
        face, lat_c, half_lat, depth_extent = cx + sign * hx, cz, hz, hx
    else:
        face, lat_c, half_lat, depth_extent = cz + sign * hz, cx, hx, hz
    # inset (the pane recess's depth) uses most of the primitive's OWN thickness along the
    # depth axis, so it never pokes through — but if that thickness itself is sub-voxel at
    # bake resolution (~0.2-0.3m/voxel), the recess will be too, same class of issue fixed
    # for the balcony/wall treatments 2026-07-02; there's no fix for that here but a floor
    # keeps it from vanishing on marginal cases.
    inset = max(0.12, depth_extent * 0.85)

    def f(q: torch.Tensor) -> torch.Tensor:
        out = base(q)
        px, py, pz = q[:, 0], q[:, 1], q[:, 2]
        depth = (px if axis == 0 else pz) - face
        lat = (pz if axis == 0 else px) - lat_c
        pane = _recess_dist(depth, py - cy, lat, inset, hy * 0.75, half_lat * 0.7, arched)
        return torch.maximum(out, -pane)
    return f


def add_element_detail(base: SDF, op: dict, body_bbox, style="modern", seed=None) -> SDF:
    """Give a user-ADDED primitive (an EditOp dict, mode='add', box) its own architectural
    detail instead of leaving it a bare block: classify its shape (classify_shape) then
    dispatch to the matching treatment (tower/pilaster -> window band; balcony -> parapet;
    bay -> bay window; wall -> coping cap; window/door -> inset pane). `op["center"]`/
    `op["size"]` and `body_bbox` must be in the SAME world-meter frame as `base` (Y from the
    building's ground) — the caller converts from whatever frame the raw edit arrived in.
    Chains onto `base`, so call this AFTER the whole-building compose_detail treatment."""
    if str(op.get("mode", "add")) != "add" or str(op.get("kind", "box")) != "box":
        return base
    center = tuple(float(v) for v in op["center"])
    half_size = tuple(float(v) for v in op["size"][:3])
    kind = classify_shape(half_size, center, body_bbox, mode="add")
    if kind == "pilaster":
        return add_tower_element_detail(base, center, half_size, style=style, seed=seed)
    if kind == "balcony":
        axis, sign = _nearest_wall_axis_sign(center, body_bbox)
        return add_balcony_railing(base, center, half_size, axis, sign, style=style, seed=seed)
    if kind == "bay":
        axis, sign = _nearest_wall_axis_sign(center, body_bbox)
        return add_bay_window_detail(base, center, half_size, axis, sign, style=style, seed=seed)
    if kind == "wall":
        return add_wall_element_detail(base, center, half_size, style=style, seed=seed)
    if kind in ("window", "door"):
        return add_opening_element_detail(base, center, half_size, body_bbox, style=style, seed=seed)
    return base
