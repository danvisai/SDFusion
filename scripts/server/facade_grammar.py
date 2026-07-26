"""Exterior facade GRAMMAR — the exterior-by-construction coherent-add (replaces the interior-
pulling set-refiner; see memory/feedback_exterior_design_focus).

A facade = a floors x bays CELL GRID on each of the 4 exterior walls (FacAID/Pro-DG doctrine).
Alignment/rhythm are true BY CONSTRUCTION, and every cell sits ON an exterior wall plane, so a
user-added primitive can only ever become an EXTERIOR element — it physically cannot be pulled
inside. Grid pitch/spacing default to values MEASURED from the clean LoD3 facades (99% row-aligned).

`coherent_add(building_bbox, primitive)` -> a grid-aligned window/door EditOp on the nearest
exterior cell. The grid params can be SAMPLED from the generative facade head (DetailParams) so
different buildings get different — but always coherent — facades.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scene.sdf_detail import classify_shape  # noqa: F401 (re-exported; canonical def moved here 2026-07-02)

# LoD3-measured facade grid (normalized cube units) — scripts/foundations probe 2026-06-15
LOD3_GRID = dict(floor_pitch=0.30, bay_spacing=0.20, win_h=0.13, win_w=0.10,
                 margin=0.14, depth=0.07, floor0_frac=0.6)


def build_grid(bbox, params=None):
    """bbox=(lo,hi) in cube coords. Returns (cells, params). Each cell = exterior wall position +
    oriented window half-extent + (wall, row, col) index."""
    p = {**LOD3_GRID, **(params or {})}
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    y0, y1 = lo[1], hi[1]
    floor0 = y0 + p["floor_pitch"] * p["floor0_frac"]
    rows = [floor0 + i * p["floor_pitch"] for i in range(99)
            if floor0 + i * p["floor_pitch"] <= y1 - 0.03] or [(y0 + y1) / 2]
    # (normal axis, wall coord, lateral axis, lateral lo, lateral hi)
    walls = [(0, hi[0], 2, lo[2], hi[2]), (0, lo[0], 2, lo[2], hi[2]),
             (2, hi[2], 0, lo[0], hi[0]), (2, lo[2], 0, lo[0], hi[0])]
    cells = []
    for wi, (nax, nc, lax, llo, lhi) in enumerate(walls):
        span = (lhi - llo) - 2 * p["margin"]
        ncols = max(int(span / p["bay_spacing"]) + 1, 1) if span > 0 else 1
        c0 = (llo + lhi) / 2 - (ncols - 1) * p["bay_spacing"] / 2
        cols = [c0 + j * p["bay_spacing"] for j in range(ncols)]
        for ri, ry in enumerate(rows):
            for ci, cx in enumerate(cols):
                center = np.zeros(3); center[1] = ry; center[nax] = nc; center[lax] = cx
                half = np.zeros(3); half[1] = p["win_h"] / 2
                half[nax] = p["depth"]; half[lax] = p["win_w"] / 2
                cells.append(dict(center=center, half=half, wall=wi, row=ri, col=ci))
    return cells, p


def nearest_cell(cells, point):
    pc = np.asarray(point, float)
    return min(cells, key=lambda c: float(np.linalg.norm(c["center"] - pc)))


# (normal axis, wall coord) per wall index -> outward unit normal
def _wall_normal(bbox, wall):
    nax, s = [(0, +1), (0, -1), (2, +1), (2, -1)][wall]
    n = np.zeros(3); n[nax] = s
    return n, nax


def snap_cell(bbox, point, params=None):
    """Nearest exterior cell + its outward normal (so a construction knows which way to protrude)."""
    cells, p = build_grid(bbox, params)
    c = nearest_cell(cells, point)
    n, nax = _wall_normal(bbox, c["wall"])
    lax = 2 if nax == 0 else 0
    return dict(center=c["center"], normal=n, normal_axis=nax, lateral_axis=lax,
                row=c["row"], wall=c["wall"]), p


def coherent_add(bbox, prim_center, prim_size, mode="subtract", params=None, grp="gI"):
    """Snap a placed primitive to the nearest EXTERIOR facade cell -> a grid-aligned window/door
    EditOp (cube coords). Exterior + aligned by construction; never interior. Returns None if the
    facade has no cells. `prim_size` half-extents drive door-vs-window typing on the ground row."""
    cells, p = build_grid(bbox, params)
    if not cells:
        return None
    cell = nearest_cell(cells, prim_center)
    is_door = (mode == "subtract") and cell["row"] == 0 and float(prim_size[1]) > p["win_h"]
    half = cell["half"].copy()
    if is_door:
        half[1] = p["win_h"] * 1.15                        # taller ground-floor opening
        cell["center"][1] = bbox[0][1] + half[1] + 0.02    # seat on the ground
    return dict(kind="box", center=[float(v) for v in cell["center"]],
                size=[float(v) for v in half], mode="subtract", smooth=0.0,
                det="door" if is_door else "window", grp=grp,
                cell=[int(cell["wall"]), int(cell["row"]), int(cell["col"])])


def full_facade_ops(bbox, params=None):
    """Instantiate EVERY cell as a window recess = the coherent base facade ('other pieces')."""
    cells, _ = build_grid(bbox, params)
    return [dict(kind="box", center=[float(v) for v in c["center"]],
                 size=[float(v) for v in c["half"]], mode="subtract", smooth=0.0, det="window")
            for c in cells]
