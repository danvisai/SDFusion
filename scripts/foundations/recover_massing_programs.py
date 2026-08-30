"""Recover a semantic layer program for every real building, and measure how well it fits.

#10's question, answered on our own corpus rather than argued from the literature: can constrained
architectural volumes express real LoD2 massing, in how many operations, and with what residual?

WHY THIS IS A HEIGHT-MAP FITTER AND NOT A 3-D CSG FITTER
--------------------------------------------------------
Measured on the pre-registered 714 held-out buildings before any of this was written:

  * `missing` of the blockout against GT is **0.000000 on 714/714** -- the real building is always
    entirely inside its own extruded footprint, so nothing is ever *added* and the task is purely
    subtractive.
  * **100.0%** of the carve volume sits *above the topmost GT voxel in its column*. Through-voids
    (courtyard / passage / light well) account for **0 voxels**, and overhangs for 71 out of
    4,324,919.
  * Only **4 columns in 1,072,438** are not a solid run from the base.

So every building in this corpus is exactly `{(z,y,x) : y0 <= y <= top(z,x)}` -- a 64x64 **height
map**, not a general volume. That is the same object ArcPro's `CreateLayer` grammar produces
(vertically extruded polygonal layers), which is why a layer vocabulary is the right one here and
why `SubtractCourtyard` / `CutNotch` are dead operations on this data: they can never fire.

Fitting in height-map space rather than on the 64^3 grid is therefore not an approximation. It is
the exact representation, and it makes the containment invariant trivial to enforce: a fitted
height may never drop below the target height, so the program can never cut into GT and
`collapse_rate` is 0 by construction.

THE VOCABULARY
--------------
    Layer(height, polygon)        one connected region flattened to one height  (ArcPro CreateLayer)
    CutRoof(kind, eaves, rate)    height falls off with distance from the footprint edge;
                                  kind=hip erodes on all sides, gable_x / gable_z on one axis
    Ramp(region, slope)           the tightest PLANE above the target over one region -- the shed
                                  roof `CutRoof` cannot express, at arbitrary rotation

`ApplySetback` is not a separate operation: in a height field a setback *is* a Layer whose polygon
is the inward offset of the footprint, and the fitter finds it as one.

Output is a semantic program per building plus the recovery statistics. It trains nothing, touches
no GPU, and does not modify the active #92 experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.eval_massing_arms import (            # noqa: E402
    RES, S_STAR_VOXELS, COLLAPSE_MISSING, volume_split, footprint_split, fp_iou, vs_input,
)

H5 = REPO / "data/real_massing_v1/real.h5"
SHIP714 = REPO / "execution/artifacts/massing_arms_eval_ship714.json"

# A building only needs a carve at all if its blockout over-fills by more than this. Matches the
# `allowance` vocabulary in CONTEXT.md: a decision, recorded in one place so it cannot drift.
CARVE_NEEDED = 0.02

# Snap before flooring a ramp plane, so a value that is mathematically integral is not read one
# level lower because the linear program returned it a few ulps short. See `_ramp_candidates`.
FLOOR_EPS = 1e-9


# ----------------------------------------------------------------------------------------------
# the height field
# ----------------------------------------------------------------------------------------------

def height_field(gt_occ: np.ndarray, fp: np.ndarray):
    """GT occupancy -> (y0, y1, target height map in voxels above y0).

    `top` is 0 off the footprint and in [1 .. y1-y0+1] on it. The blockout is the constant map
    `y1-y0+1`, so `blockout - target` is exactly the per-column carve depth.
    """
    ys = np.nonzero(gt_occ.any(axis=(0, 2)))[0]
    if not len(ys):
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    sub = gt_occ[:, y0:y1 + 1, :]                              # [z, h, x]
    top = (sub.shape[1] - 1 - np.argmax(sub[:, ::-1, :], axis=1)) + 1
    return y0, y1, np.where(fp, top, 0).astype(np.int16)


def occupancy(fp: np.ndarray, y0: int, h: np.ndarray) -> np.ndarray:
    """Height map -> 64^3 occupancy, the inverse of `height_field`."""
    yy = np.arange(RES)[None, :, None]
    return fp[:, None, :] & (yy >= y0) & (yy < y0 + h[:, None, :].astype(np.int32))


def occ_to_field(occ: np.ndarray) -> np.ndarray:
    """Occupancy -> signed EDT on the corpus scale, identical to `blockout_sdf`'s convention.

    Scoring downstream reads `field <= 0`, so this keeps the recovered arm on exactly the same
    footing as `blockout` and `gt` in the shipped harness rather than inventing a second path.
    """
    inside = ndimage.distance_transform_edt(occ)
    outside = ndimage.distance_transform_edt(~occ)
    return ((outside - inside) * (2.0 / (RES - 1))).astype(np.float32)


def _dist_axis(fp: np.ndarray, axis: int) -> np.ndarray:
    """Cells to the nearest non-footprint cell along ONE axis (1 on the boundary cell).

    A gable roof insets along a single axis, so an isotropic distance transform cannot express it;
    a hip roof insets on all sides and needs the isotropic one. Both are required.
    """
    m = fp if axis == 1 else fp.T
    n = m.shape[1]
    fwd = np.zeros(m.shape, np.int16)
    acc = np.zeros(m.shape[0], np.int16)
    for i in range(n):
        acc = np.where(m[:, i], acc + 1, 0)
        fwd[:, i] = acc
    bwd = np.zeros(m.shape, np.int16)
    acc = np.zeros(m.shape[0], np.int16)
    for i in range(n - 1, -1, -1):
        acc = np.where(m[:, i], acc + 1, 0)
        bwd[:, i] = acc
    out = np.minimum(fwd, bwd)
    return out if axis == 1 else out.T


# ----------------------------------------------------------------------------------------------
# the fitter
# ----------------------------------------------------------------------------------------------

ROOF_RATES = (0.5, 1.0, 1.5, 2.0, 3.0)


def _roof_candidates(fp, dists, target, h):
    """Every (kind, eaves, rate) roof that stays at or above the target height everywhere."""
    for kind, d in dists.items():
        for rate in ROOF_RATES:
            slope = (d.astype(np.float32) - 1.0) * rate
            for eaves in range(1, int(h.max()) + 1):
                cand = np.minimum(h, np.floor(eaves + slope)).astype(np.int16)
                cand = np.where(fp, np.maximum(cand, 1), 0)
                if (cand[fp] < target[fp]).any():
                    continue                                   # would cut into GT -- rejected
                gain = int((h[fp] - cand[fp]).sum())
                if gain > 0:
                    yield gain, cand, dict(op="CutRoof", kind=kind, eaves=int(eaves),
                                           rate=float(rate))


def _layer_candidates(fp, target, h):
    """Every single connected polygon that can be flattened to one height.

    A Layer is ONE polygon by definition, so a candidate height whose eligible region breaks into
    several components is offered once per component rather than as a disconnected set.
    """
    for v in np.unique(target[fp]):
        region = fp & (target <= v) & (h > v)
        if not region.any():
            continue
        lab, n = ndimage.label(region)
        for c in range(1, n + 1):
            piece = lab == c
            gain = int((h[piece] - v).sum())
            if gain > 0:
                cand = np.where(piece, np.int16(v), h)
                yield gain, cand, dict(op="Layer", height=int(v), area=int(piece.sum()),
                                       components=1, _region=piece)


def _ramp_candidates(fp, target, h, max_regions: int = 3):
    """The tightest PLANE that stays at or above the target over a surplus region.

    Why a general plane and not an axis-aligned ramp: buildings sit at arbitrary rotations on the
    grid, so a shed roof's fall line almost never lines up with x or z. And why a plane at all --
    `CutRoof` measures distance to the *nearest* footprint edge, which is symmetric, so it can
    express a gable or a hip but never a **shed**. The worst-residual trace was entirely smooth
    roof ramps that `CutRoof` could not fire on, which the fitter then approximated with a
    staircase of flat `Layer`s.

    Finding the plane is a 3-variable linear program: minimise the surplus `sum(a + b*x + c*z -
    target)` subject to `a + b*x + c*z >= target` on every cell of the region, so the result is the
    optimal ramp rather than a sampled guess. Because `target` is integral and the plane dominates
    it, `floor(plane) >= target` still holds and the containment invariant survives the rounding.

    Offered per connected surplus region, so a gable is recoverable as two opposing ramps.
    """
    from scipy.optimize import linprog

    surplus = fp & (h > target)
    if not surplus.any():
        return
    lab, n = ndimage.label(surplus)
    order = sorted(range(1, n + 1), key=lambda c: -int((lab == c).sum()))[:max_regions]
    zz_g, xx_g = np.mgrid[0:RES, 0:RES]
    for c in order:
        piece = lab == c
        zz, xx = np.nonzero(piece)
        if len(xx) < 3:
            continue
        t = target[piece].astype(float)
        ones = np.ones(len(xx))
        # -(a + b*x + c*z) <= -target   <=>   plane >= target
        A = -np.stack([ones, xx.astype(float), zz.astype(float)], 1)
        obj = np.array([ones.sum(), xx.sum(), zz.sum()], float)
        try:
            r = linprog(obj, A_ub=A, b_ub=-t, bounds=[(None, None)] * 3, method="highs")
        except Exception:
            continue
        if not r.success:
            continue
        a, b, cz = r.x
        # `+FLOOR_EPS`: the optimal plane *touches* the binding target heights, so its value at a
        # voxel centre is an exact integer far more often than chance -- and `linprog` returns that
        # integer as 45.999999999999996 about half the time, which `floor` then reads as 45. That is
        # round-off deciding geometry. Snapping first makes the discretisation depend on the plane
        # rather than on the solver's last bit, and is what lets the same program compile to the
        # same solid through `scene/sdf_edit.py` (#128).
        plane = np.floor(a + b * xx_g + cz * zz_g + FLOOR_EPS)
        cand = np.where(piece, np.minimum(h, plane).astype(np.int16), h)
        cand = np.where(fp, np.maximum(cand, 1), 0).astype(np.int16)
        if (cand[fp] < target[fp]).any():
            continue                                       # rounding guard; never cut GT
        gain = int((h[fp] - cand[fp]).sum())
        if gain > 0:
            yield gain, cand, dict(op="Ramp", area=int(piece.sum()),
                                   slope=[round(float(b), 4), round(float(cz), 4)],
                                   plane=[float(a), float(b), float(cz)], _region=piece)


VOCABULARY = ("Layer", "CutRoof", "Ramp")


def _all_candidates(fp, dists, target, h, ops_allowed=VOCABULARY):
    """Every operation the vocabulary can offer against the current height map.

    `ops_allowed` restricts it. The default is the whole vocabulary and is what every recovery
    number on this project's record was measured with; #6 fits its training labels with `CutRoof`
    withheld, because a `CutRoof` surface is a distance transform rather than a plane and so cannot
    be carried by the (type, plane) slot the generator predicts. That exclusion is a decision with a
    price, and the price is measured rather than assumed -- `--ops_allowed Layer Ramp` re-runs the
    recovery so the two residuals can be read side by side.
    """
    if "CutRoof" in ops_allowed:
        yield from _roof_candidates(fp, dists, target, h)
    if "Layer" in ops_allowed:
        yield from _layer_candidates(fp, target, h)
    if "Ramp" in ops_allowed:
        yield from _ramp_candidates(fp, target, h)


def _dists_for(fp):
    return dict(hip=ndimage.distance_transform_edt(fp).astype(np.int16),
                gable_x=_dist_axis(fp, 1), gable_z=_dist_axis(fp, 0))


def fit_program(fp, y0, y1, target, max_ops=4, allowance=CARVE_NEEDED,
                ops_allowed=VOCABULARY):
    """Greedy: repeatedly take the operation that removes the most surplus without cutting GT."""
    full = np.int16(y1 - y0 + 1)
    h = np.where(fp, full, 0).astype(np.int16)
    gt_vox = int(target[fp].sum())
    dists = _dists_for(fp)
    ops = []
    for _ in range(max_ops):
        surplus = int((h[fp] - target[fp]).sum())
        if gt_vox and surplus / gt_vox <= allowance:
            break
        best = max(_all_candidates(fp, dists, target, h, ops_allowed),
                   key=lambda t: t[0], default=None)
        if best is None or best[0] <= 0:
            break
        gain, h, meta = best
        meta["removed_voxels"] = int(gain)
        # the residual AFTER this operation, so one K=16 run yields the whole simplicity curve
        # instead of re-fitting the corpus once per K
        meta["residual_extra"] = (float((h[fp] - target[fp]).sum() / gt_vox) if gt_vox else 0.0)
        ops.append(meta)
    return ops, h


def fit_program_beam(fp, y0, y1, target, max_ops=4, allowance=CARVE_NEEDED,
                     beam=6, branch=6, ops_allowed=VOCABULARY):
    """Beam search over programs, because greedy is provably myopic on gable roofs.

    The worst-residual trace after `Ramp` landed was entirely **symmetric double ramps**: a gable
    rises from both eaves to a ridge, so no single plane dominates it and it needs two opposing
    `Ramp`s. Greedy never gets there -- one large flat `Layer` always wins the immediate gain, and
    by the time the surplus has split into the two regions that would each take a ramp, the
    operation budget is spent. That is a search failure, not a missing operation: at K=16 greedy
    already reaches 3-D IoU 0.9981, so the vocabulary is sufficient and only the order is wrong.

    Beams are de-duplicated by the height map itself rather than by the operation list, since two
    different orders that reach the same massing are the same program for every purpose here.
    """
    import heapq

    full = np.int16(y1 - y0 + 1)
    h0 = np.where(fp, full, 0).astype(np.int16)
    gt_vox = int(target[fp].sum())
    dists = _dists_for(fp)
    surplus = lambda hh: int((hh[fp] - target[fp]).sum())

    beams = [(surplus(h0), h0, [])]
    for _ in range(max_ops):
        nxt, seen = [], set()
        for sur, h, ops in beams:
            if gt_vox and sur / gt_vox <= allowance:
                nxt.append((sur, h, ops))                  # already good enough: carry it forward
                continue
            top = heapq.nlargest(branch, _all_candidates(fp, dists, target, h, ops_allowed),
                                 key=lambda t: t[0])
            for gain, hh, meta in top:
                if gain <= 0:
                    continue
                key = hh.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                m = dict(meta)
                m["removed_voxels"] = int(gain)
                s2 = surplus(hh)
                m["residual_extra"] = (float(s2 / gt_vox) if gt_vox else 0.0)
                nxt.append((s2, hh, ops + [m]))
        if not nxt:
            break
        nxt.sort(key=lambda t: (t[0], len(t[2])))
        beams = nxt[:beam]
    best = min(beams, key=lambda t: (t[0], len(t[2])))

    # A beam search is NOT guaranteed to dominate greedy: the greedy path can be pruned at an
    # intermediate step by siblings that look better then and end worse. Measured -- id 16764 went
    # 0.152 greedy -> 0.159 beam. Greedy is cheap, so run it too and keep whichever program is
    # actually better. This makes the beam a monotone improvement by construction.
    g_ops, g_h = fit_program(fp, y0, y1, target, max_ops, allowance, ops_allowed)
    if surplus(g_h) < best[0]:
        return g_ops, g_h
    return best[2], best[1]


# ----------------------------------------------------------------------------------------------
# #6 -- a fitted program as a fixed set of typed slots, which is what a network can predict
# ----------------------------------------------------------------------------------------------

K_OPS = 4
SLOT_TYPES = ("Layer", "Ramp")


def plane_surface(plane, eps: float = FLOOR_EPS, res: int = RES) -> np.ndarray:
    """`height = floor(a + b*x + c*z + eps)` over the whole plan. One spelling, two epsilons.

    ⚠️ The snap is a parameter because the right value depends on the PRECISION of the plane, not on
    the geometry. `linprog` returns float64 and 1e-9 is the correct snap for it; #6 stores a slot's
    plane in the label cache and predicts it in **float32**, where a value that is mathematically 30
    arrives as 29.9999996 and `floor` reads 29 -- measured on 96 of 1,280 columns of a plain shed
    roof. Both callers floor the same expression, so they are written once here and the epsilon is
    the only thing that varies; `test_the_slots_replay_the_fitted_height_map_exactly` is what pins
    that the two choices still agree on the same building.
    """
    zz, xx = np.mgrid[0:res, 0:res]
    a, b, c = (float(v) for v in plane)
    return np.floor(a + b * xx + c * zz + eps)


def program_to_slots(fp, extent, ops, k_ops: int = K_OPS):
    """A fitted program -> (per-column slot assignment, per-slot type, per-slot plane).

    🔑 **The bridge #6 turns on.** The fitter *searches a sequence*: each operation is applied to the
    result of the last, so op 3 only means anything given ops 1 and 2. A network cannot easily emit
    that -- and it does not have to, because of one property of this vocabulary:

        every operation only ever LOWERS the height map.

    So the final height of a column is simply the value written by whichever operation last touched
    it, and recording that **owner** per column replays the entire cascade in a single pass. The
    sequence collapses into a set of typed slots plus one assignment per column, with no loss --
    which `test_the_slots_replay_the_fitted_height_map_exactly` pins on four surface families.

    That is the whole reason a program is the right answer to #127's mound. Each column's height
    comes from a slot the whole region shares, so a ridge line is one decision instead of 4,096
    independent ones, and the *joint commitment* #127 found neither supervision nor decoding could
    supply is made by the representation instead.

    Returned in the fitter's own voxel convention -- `height = a + b*x + c*z` above `y0` -- because
    that is what `plane` already means everywhere else in this module. The network's scale-free
    convention is `train_height_map_generator.plane_to_normalised`, converted at exactly one place.

      assign  [RES, RES] uint8, in 0..k_ops. `k_ops` is the UNCARVED class: no operation touched
              this column and it keeps the blockout. Off the footprint is always `k_ops`.
      types   [k_ops] int8, an index into `SLOT_TYPES`, or -1 for a slot the program does not use.
      planes  [k_ops, 3] float32.

    Slots come back sorted by **descending owned area**. #6 asks how to canonicalise, and this is the
    cheapest form that always exists: a set head has no natural slot order, so without it two fits
    that found the same program in a different order would supervise contradictory labels and the
    arm would need a matching loss to paper over a problem it need not have.

    ⚠️ `CutRoof` is refused rather than approximated. Its surface is a distance transform, not a
    plane, so no (type, plane) slot can carry it; fit the labels with `ops_allowed=("Layer", "Ramp")`
    and the case cannot arise. Silently least-squares-fitting a plane through it would put a target
    in the cache that the compiler provably cannot reproduce.
    """
    m = np.asarray(fp, bool)
    e = int(extent)
    h = np.where(m, np.int16(e), 0).astype(np.int16)
    owner = np.full(m.shape, k_ops, np.uint8)
    kinds, planes_vox = [], []
    for i, op in enumerate(ops[:k_ops]):
        kind = op["op"]
        region = op.get("_region")
        if region is None:
            region = _rings_to_mask(op["region"])
        region = np.asarray(region, bool) & m
        if kind == "Layer":
            plane = (float(op["height"]), 0.0, 0.0)
        elif kind == "Ramp":
            plane = tuple(float(v) for v in op["plane"])
        else:
            raise ValueError(
                f"'{kind}' has no plane, so it cannot be a slot -- fit the labels with "
                f"ops_allowed=('Layer', 'Ramp')")
        surf = plane_surface(plane)                # float64 from the fitter: the 1e-9 snap
        cand = np.where(region, np.minimum(h, surf), h)
        cand = np.where(m, np.maximum(cand, 1), 0).astype(np.int16)
        owner[m & (cand < h)] = i
        h = cand
        kinds.append(SLOT_TYPES.index(kind))
        planes_vox.append(plane)

    # canonical order: descending owned area, unused slots last and typed -1
    areas = [int((owner == i).sum()) for i in range(len(kinds))]
    keep = sorted((i for i in range(len(kinds)) if areas[i] > 0), key=lambda i: -areas[i])
    types = np.full(k_ops, -1, np.int8)
    planes = np.zeros((k_ops, 3), np.float32)
    remap = np.full(k_ops + 1, k_ops, np.uint8)
    for new, old in enumerate(keep):
        types[new] = kinds[old]
        planes[new] = planes_vox[old]
        remap[old] = new
    return remap[owner], types, planes


# ----------------------------------------------------------------------------------------------
# visual trace
# ----------------------------------------------------------------------------------------------

def build_montage(cases, out: Path, cell: int = 128) -> Path:
    """Height-map trace: footprint | GT | recovered | residual, one row per building.

    Rendered as height maps rather than shaded 3-D views on purpose. The corpus *is* a height
    field (see the module docstring), so this shows the actual object under fit rather than a
    projection of it, and it stays CPU-only while the A100 is busy with #92.
    """
    from PIL import Image, ImageDraw

    def tile(a, vmax, colour):
        """One 64x64 map -> an upscaled RGB tile. `colour` False renders a binary mask."""
        a = np.asarray(a, np.float32)
        if colour:
            n = np.clip(a / max(vmax, 1e-6), 0, 1)
            # dark blue (low) -> yellow (high); off-footprint stays near-black
            rgb = np.stack([np.clip(1.6 * n - 0.3, 0, 1),
                            np.clip(1.5 * n - 0.1, 0, 1),
                            np.clip(0.9 - 1.1 * n, 0, 1)], -1)
            rgb[a <= 0] = 0.06
        else:
            rgb = np.repeat((a > 0).astype(np.float32)[..., None], 3, -1) * 0.85 + 0.06
        img = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
        return img.resize((cell, cell), Image.NEAREST)

    cols = ["footprint", "GT height", "recovered", "residual (surplus)"]
    pad, head = 4, 22
    W = len(cols) * (cell + pad) + pad
    H = head + len(cases) * (cell + pad) + pad
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((pad + j * (cell + pad) + 4, 6), c, fill=(210, 210, 215))
    for i, cs in enumerate(cases):
        y = head + i * (cell + pad) + pad
        vmax = float(max(cs["target"].max(), cs["fitted"].max(), 1))
        for j, im in enumerate([tile(cs["fp"], 1, False), tile(cs["target"], vmax, True),
                                tile(cs["fitted"], vmax, True),
                                tile(cs["fitted"] - cs["target"], vmax, True)]):
            sheet.paste(im, (pad + j * (cell + pad), y))
        d.text((pad + 4, y + 4), f"id {cs['id']}", fill=(255, 255, 255))
        d.text((pad + 4, y + cell - 14),
               f"{cs['n_ops']} ops  extra {cs['extra']:.3f}", fill=(255, 240, 160))
        d.text((pad + 2 * (cell + pad) + 4, y + cell - 14),
               " > ".join(cs["ops"][:4]) or "empty", fill=(160, 230, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def finalise_program(ops):
    """Turn each selected operation's region mask into polygon rings, in place.

    Carried as a mask through the search because a beam expands thousands of candidates and only a
    handful survive; tracing every one would dominate the fit. What lands in the artifact is the
    ring list, so the program is **replayable** -- `scene.sdf_edit.layer_program_to_ops` rebuilds
    the identical solid from it, and a program that cannot be replayed is not a recipe (#128).
    """
    from scene.sdf_edit import mask_to_rings

    out = []
    for op in ops:
        op = dict(op)
        mask = op.pop("_region", None)
        if mask is not None:
            op["region"] = [r.tolist() for r in mask_to_rings(mask)]
        out.append(op)
    return out


# ----------------------------------------------------------------------------------------------
# the trip through the edit stack (#128)
# ----------------------------------------------------------------------------------------------

def _rings_to_mask(rings, res: int = RES) -> np.ndarray:
    """Polygon rings (voxel-index units, outer first) -> the cells they cover.

    Uses matplotlib's point-in-polygon rather than our own SDF, so the check below compares two
    independent implementations instead of one implementation with itself.
    """
    from matplotlib.path import Path as MplPath

    zz, xx = np.mgrid[0:res, 0:res]
    pts = np.stack([xx.ravel(), zz.ravel()], axis=1).astype(float)
    mask = np.zeros(len(pts), bool)
    for i, ring in enumerate(rings):
        hit = MplPath(np.asarray(ring, float)).contains_points(pts)
        mask = hit if i == 0 else (mask & ~hit)
    return mask.reshape(res, res)


def replay_program(fp: np.ndarray, y0: int, y1: int, program) -> np.ndarray:
    """Re-run a serialised program in height-map space, reading only what the artifact stores.

    The fitter returns its height map as a by-product of the search. This interprets the written
    program instead, so a disagreement means the artifact is not self-contained -- which is the
    difference between a recorded *program* and a recorded *result*.

    It repeats the Layer/Ramp/CutRoof cascade that `scene.sdf_edit.layer_program_to_ops` also walks,
    and that repetition is the point: one reads the program into height-map space and the other into
    an SDF, and a check is only worth running while the two are written independently.

    🔑 **Every operation takes a MIN, so the program COMMUTES** (#4). `Layer` used to *set*
    (`where(region, v, h)`), which can raise a column an earlier operation had lowered -- and the
    SDF path cannot, because it composes with `sdf_subtract` and subtracting A then B is
    subtracting their union. The two therefore agreed on order only because #10's fitter never
    emits an operation that would raise a column, which is a property of the **search**, not of the
    algebra: a hand-authored or generated program is under no obligation to have it.

    Measured over 250 recovered programs before the change: 78% have two operations whose regions
    overlap and a permutation changed the building on **68.8%** of them; taking the min instead
    changed the result on **0 of 250** and left **0 of 2,000** permutations changed. So
    commutativity cost nothing and bought a canonical form, an equivalence test, and deletion of
    any operation rather than only the last (`EditableBuilding.remove`).
    """
    h = np.where(fp, np.int16(y1 - y0 + 1), 0).astype(np.int16)
    dists = _dists_for(fp)
    for op in program:
        kind = op["op"]
        if kind == "Layer":
            region = _rings_to_mask(op["region"]) & fp
            cand = np.where(region, np.minimum(h, np.int16(op["height"])), h)
        elif kind == "Ramp":
            region = _rings_to_mask(op["region"]) & fp
            cand = np.where(region, np.minimum(h, plane_surface(op["plane"])), h)
        elif kind == "CutRoof":
            slope = (dists[op["kind"]].astype(np.float32) - 1.0) * float(op["rate"])
            cand = np.minimum(h, np.floor(int(op["eaves"]) + slope))
        else:
            raise ValueError(f"unknown operation '{kind}'")
        h = np.where(fp, np.maximum(cand.astype(np.int16), 1), 0).astype(np.int16)
    return h


def verify_edit_stack(cases, out_dir: Path, sheet_rows: int = 6):
    """Load each recovered program into `EditableBuilding` and check it is the same building.

    Three claims, each measured rather than asserted:
      1. the **serialised** program replays to the height map the fitter found;
      2. the SDF composition and the voxel compiler agree, voxel for voxel;
      3. `undo()` returns the building to the program's previous state -- reversibility, which
         `CONTEXT.md` calls the load-bearing claim, holding through the new operations.

    Reported in three groups, because a roof is compiled differently from a layer and the residual
    means something different in each:
      * `Layer` / `Ramp` -- **exact**, and the overwhelming majority of recovered operations;
      * `CutRoof(hip)` -- a cap over the region's outline distance;
      * `CutRoof(gable_x|gable_z)` -- a clause per eave, clipped to the rows its wall spans.
    The two roof forms read a *continuous* distance to the wall where the recovered rule reads a
    discrete transform over cells, so they can disagree by up to half a voxel at a corner. Measured
    on the 12 corpus buildings whose programs contain a roof, that is a median IoU of 0.9994 (hip)
    and 1.0000 (gable), never below 0.9892.
    """
    from scene.sdf_edit import EditableBuilding, footprint_envelope_sdf, layer_program_to_ops
    from scene.sdf_primitives import grid_to_mesh, sample_grid

    bbox = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    # One grid sample serves both the occupancy and the mesh. Evaluating a composed program costs
    # O(points x polygon vertices) per operation and the corpus's regions run to hundreds of
    # vertices, so sampling twice for the same building is the difference between minutes and tens
    # of minutes across a run.
    grid_of = lambda b: sample_grid(b.composed(), RES, bbox, device="cpu", chunk=1 << 14)

    rows, sheet = [], []
    for case in cases:
        fp, y0, y1 = case["fp"], case["y0"], case["y1"]
        program, fitted = case["program"], case["fitted"]

        replayed = replay_program(fp, y0, y1, program)
        ops = layer_program_to_ops(program, fp, y0, y1, res=RES)
        eb = EditableBuilding(footprint_envelope_sdf(fp, y0, y1, res=RES), ops)
        grid = grid_of(eb)
        occ_sdf = (grid <= 0.0).numpy()
        occ_ref = occupancy(fp, y0, fitted)
        mesh = grid_to_mesh(grid, bbox, iso=0.0)

        undo_iou = None
        if ops:
            eb.undo()
            undo_iou = volume_split(
                (grid_of(eb) <= 0.0).numpy(),
                occupancy(fp, y0, replay_program(fp, y0, y1, program[:-1])))["vol_iou"]

        rows.append(dict(id=int(case["id"]), n_ops=len(program),
                         ops=[o["op"] for o in program],
                         replay_exact=bool((replayed == fitted).all()),
                         sdf_iou=volume_split(occ_sdf, occ_ref)["vol_iou"],
                         undo_iou=undo_iou,
                         roof_kinds=[o["kind"] for o in program if o["op"] == "CutRoof"],
                         vertices=(0 if mesh is None else int(len(mesh.vertices))),
                         faces=(0 if mesh is None else int(len(mesh.faces)))))
        if len(sheet) < sheet_rows:
            top = np.where(occ_sdf.any(axis=1),
                           RES - np.argmax(occ_sdf[:, ::-1, :], axis=1) - y0, 0).astype(np.int16)
            sheet.append((case["id"], fp, fitted, np.where(fp, top, 0)))

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = None
    if sheet:
        from PIL import Image, ImageDraw
        panels = [[render_iso(h, fp) for h in (voxels, sdf)] for _i, fp, voxels, sdf in sheet]
        w = max(im.width for row in panels for im in row)
        ht = max(im.height for row in panels for im in row)
        LBL = 18
        canvas = Image.new("RGB", (2 * w, len(panels) * (ht + LBL) + LBL), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for col, name in enumerate(("voxel compiler", "SDF edit stack")):
            d.text((col * w + 8, 4), name, fill=(0, 0, 0))
        for r, (row, (bid, *_)) in enumerate(zip(panels, sheet)):
            y = LBL + r * (ht + LBL)
            d.text((8, y + ht), f"id {bid}", fill=(0, 0, 0))
            for c, im in enumerate(row):
                canvas.paste(im, (c * w, y))
        sheet_path = out_dir / "edit_stack.png"
        canvas.save(sheet_path)
    return rows, sheet_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_from", default=str(SHIP714),
                    help="replay a pinned id set; default is the pre-registered 714")
    ap.add_argument("--n", type=int, default=0, help="0 = every id in the set")
    ap.add_argument("--max_ops", type=int, default=4)
    ap.add_argument("--allowance", type=float, default=CARVE_NEEDED)
    ap.add_argument("--out", default="execution/artifacts/program_recovery_714.json")
    ap.add_argument("--beam", type=int, default=1,
                    help="1 = greedy (the default all earlier numbers used); >1 runs a "
                         "beam search of this width over programs")
    ap.add_argument("--branch", type=int, default=6,
                    help="candidates expanded per beam per step")
    ap.add_argument("--ops_allowed", nargs="*", default=list(VOCABULARY), choices=VOCABULARY,
                    help="restrict the vocabulary. `--ops_allowed Layer Ramp` is what #6 fits its "
                         "training labels with, because a CutRoof surface is a distance transform "
                         "rather than a plane and no (type, plane) slot can carry it; running it "
                         "against the default measures what that exclusion costs")
    ap.add_argument("--montage", type=int, default=0,
                    help="rows per sheet; emits a worst-N and a representative-N trace")
    ap.add_argument("--verify_edit_stack", type=int, default=0,
                    help="load this many recovered programs into EditableBuilding and check the "
                         "SDF composition, the replay and undo against the voxel compiler (#128)")
    args = ap.parse_args()

    ids = [int(i) for i in json.load(open(args.ids_from))["ids"]]
    if args.n:
        ids = ids[:args.n]
    print(f"[ids] {len(ids)} buildings from {args.ids_from}", flush=True)

    rows, cases, bridge_cases, all_cases, t0 = {}, [], [], [], time.time()
    with h5py.File(H5, "r") as g:
        for k, b in enumerate(ids):
            gt = np.asarray(g["sdf"][b], np.float32) <= 0
            fp = np.asarray(g["footprint"][b]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, target = hf
            bo_occ = occupancy(fp, y0, np.where(fp, np.int16(y1 - y0 + 1), 0).astype(np.int16))
            if args.beam > 1:
                ops, h = fit_program_beam(fp, y0, y1, target, args.max_ops, args.allowance,
                                          args.beam, args.branch, tuple(args.ops_allowed))
            else:
                ops, h = fit_program(fp, y0, y1, target, args.max_ops, args.allowance,
                                     tuple(args.ops_allowed))
            occ = occupancy(fp, y0, h)

            ops = finalise_program(ops)
            row = dict(fp_iou=fp_iou(occ, fp), n_ops=len(ops),
                       ops=[o["op"] for o in ops], program=ops)
            row.update(volume_split(occ, gt))
            row.update(footprint_split(occ, fp))
            row["vs_input"] = vs_input(occ, bo_occ)
            row["blockout_extra"] = volume_split(bo_occ, gt)["extra"]
            rows[str(b)] = row
            if args.montage:
                cases.append(dict(id=b, fp=fp.copy(), target=target.copy(), fitted=h.copy(),
                                  n_ops=len(ops), extra=row["extra"],
                                  ops=[o["op"] for o in ops]))
            if args.verify_edit_stack:
                case = dict(id=b, fp=fp.copy(), y0=y0, y1=y1, fitted=h.copy(), program=ops)
                all_cases.append(case)
                if len(bridge_cases) < args.verify_edit_stack:
                    bridge_cases.append(case)
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), n=len(rows),
                             gt_h5=str(H5.relative_to(REPO)), ids_from=args.ids_from,
                             max_ops=args.max_ops, allowance=args.allowance, beam=args.beam,
                             vocabulary=list(args.ops_allowed)),
                   ids=[int(b) for b in rows], per_building=rows), open(out, "w"), indent=1)
    print(f"[artifact] {out}", flush=True)
    if args.montage:
        carve = [c for c in cases if rows[str(c["id"])]["blockout_extra"] >= CARVE_NEEDED]
        worst = sorted(carve, key=lambda c: -c["extra"])[:args.montage]
        rep = sorted(carve, key=lambda c: c["extra"])[len(carve) // 2:][:args.montage]
        for tag, sel in (("worst", worst), ("representative", rep)):
            if sel:
                p = build_montage(sel, REPO / f"outputs/program_recovery/{tag}.png")
                print(f"[montage] {p}", flush=True)
    if bridge_cases:
        # Compiling is cheap where sampling the composed field is not, so every program is put
        # through it and only a sample is checked voxel for voxel. It is the coverage question the
        # sample cannot answer: whether any recovered program is inexpressible at all.
        from scene.sdf_edit import layer_program_to_ops
        refused = []
        for case in all_cases:
            try:
                layer_program_to_ops(case["program"], case["fp"], case["y0"], case["y1"], res=RES)
            except Exception as exc:
                refused.append((int(case["id"]), f"{type(exc).__name__}: {exc}"))
        bridge, sheet = verify_edit_stack(bridge_cases, REPO / "outputs/program_recovery")
        art = Path(args.out).with_name(Path(args.out).stem + "_edit_stack.json")
        json.dump(dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), n=len(bridge),
                                 res=RES, source=args.out), per_building=bridge),
                  open(art, "w"), indent=1)
        plain = [r for r in bridge if not r["roof_kinds"]]
        hip = [r for r in bridge if r["roof_kinds"] and set(r["roof_kinds"]) == {"hip"}]
        gable = [r for r in bridge if set(r["roof_kinds"]) - {"hip"}]
        print(f"\n-- #128 EDIT-STACK BRIDGE  n={len(bridge)} sampled of {len(all_cases)}")
        print(f"   every recovered program compiles to EditOps:         "
              f"{len(all_cases) - len(refused)}/{len(all_cases)}")
        for bid, why in refused[:5]:
            print(f"      refused: id {bid}  {why}")
        print(f"   serialised program replays to the fitted height map: "
              f"{sum(r['replay_exact'] for r in bridge)}/{len(bridge)}")
        print(f"   composed SDF == voxel compiler, Layer/Ramp only:     "
              f"{sum(r['sdf_iou'] == 1.0 for r in plain)}/{len(plain)}")
        for tag, sel in (("with a hipped CutRoof (outline cap)   ", hip),
                         ("with a gabled CutRoof (per-axis run)  ", gable)):
            if sel:
                v = [r["sdf_iou"] for r in sel]
                print(f"   {tag}: median IoU {np.median(v):.4f}  min {min(v):.4f}  on {len(sel)}")
        undone = [r for r in bridge if r["undo_iou"] is not None]
        exact_undo = [r for r in undone if not r["roof_kinds"]]
        print(f"   undo() returns the previous program state:            "
              f"{sum(r['undo_iou'] == 1.0 for r in exact_undo)}/{len(exact_undo)}")
        roof_undo = [r for r in undone if r["roof_kinds"]]
        if roof_undo:
            print(f"   undo() on a program ending in a roof:                 "
                  f"median IoU {np.median([r['undo_iou'] for r in roof_undo]):.4f} "
                  f"on {len(roof_undo)}")
        print(f"[artifact] {art}")
        if sheet:
            print(f"[sheet] {sheet}")
    report(rows, args)


def report(rows, args) -> None:
    """Split the two populations, never pooled -- #80's bimodal result is the precedent."""
    carve = {b: r for b, r in rows.items() if r["blockout_extra"] >= CARVE_NEEDED}
    flat = {b: r for b, r in rows.items() if r["blockout_extra"] < CARVE_NEEDED}

    def block(name, d):
        if not d:
            print(f"\n{name}: none")
            return
        med = lambda k: float(np.median([r[k] for r in d.values()]))
        coll = float(np.mean([r["missing"] >= COLLAPSE_MISSING for r in d.values()]))
        print(f"\n{name}  (n={len(d)})")
        print(f"  fp-IoU        {med('fp_iou'):.4f}      vs_input      {med('vs_input'):.4f}")
        print(f"  missing       {med('missing'):.6f}    collapse_rate {coll:.4f}")
        print(f"  3D IoU        {med('vol_iou'):.4f}      n_ops         {med('n_ops'):.1f}")
        print(f"  extra  before {med('blockout_extra'):.4f}  ->  after {med('extra'):.4f}")
        under = float(np.mean([r["extra"] <= args.allowance for r in d.values()]))
        print(f"  reach the allowance ({args.allowance}): {under*100:.1f}%")

    print("\n" + "=" * 78)
    block("CARVE-NEEDING buildings", carve)
    block("ALREADY-FLAT buildings", flat)
    if carve:
        names = [o for r in carve.values() for o in r["ops"]]
        vol = {}
        for r in carve.values():
            for o in r["program"]:
                vol[o["op"]] = vol.get(o["op"], 0) + o["removed_voxels"]
        tv = sum(vol.values()) or 1
        print("\noperation mix on carve-needing buildings")
        for k in sorted(vol, key=lambda k: -vol[k]):
            print(f"  {k:<10} used {names.count(k):>5}x   {vol[k]/tv*100:5.1f}% of removed volume")
    print("=" * 78)


# ----------------------------------------------------------------------------------------------
# isometric render
# ----------------------------------------------------------------------------------------------

def render_iso(h, fp, cell: int = 6, pad: int = 20, base=(196, 198, 203)):
    """Shaded isometric view of a height map, drawn on the CPU.

    The harness's `render_world` goes through pyrender/EGL and hangs on this node while the four
    #92 arms hold the GPU. It is also more machinery than this needs: the corpus is a height field,
    so the massing can be drawn exactly as one top face plus two side faces per column, with a
    painter's-algorithm ordering by (x + z). No marching cubes, no mesh, no GPU, and the result is
    the true geometry rather than an isosurface approximation of it.
    """
    from PIL import Image, ImageDraw

    H = np.asarray(h, np.int32)
    Z, X = H.shape
    cos30, sin30, hs = 0.866, 0.5, cell * 0.62
    sx = lambda x, z: (x - z) * cos30 * cell
    sy = lambda x, z, v: (x + z) * sin30 * cell - v * hs
    x0, x1 = sx(0, Z), sx(X, 0)
    y0, y1 = sy(0, 0, int(H.max())), sy(X, Z, 0)
    W, Ht = int(x1 - x0) + 2 * pad, int(y1 - y0) + 2 * pad
    ox, oy = -x0 + pad, -y0 + pad
    img = Image.new("RGB", (W, Ht), (255, 255, 255))
    d = ImageDraw.Draw(img)

    gz, gx = np.gradient(H.astype(np.float64))
    lam = 1.0 / np.sqrt(gx ** 2 + gz ** 2 + 1.0)              # Lambert against a vertical light
    # round to integers so neighbouring columns share exact vertices -- without this the
    # side faces are separated by hairline background gaps and the massing looks combed
    P = lambda x, z, v: (round(sx(x, z) + ox), round(sy(x, z, v) + oy))
    shade = lambda f: tuple(int(np.clip(c * f, 0, 255)) for c in base)

    order = sorted(((x + z, z, x) for z in range(Z) for x in range(X) if fp[z, x]))
    for _, z, x in order:
        v = int(H[z, x])
        if v <= 0:
            continue
        nx = int(H[z, x + 1]) if x + 1 < X and fp[z, x + 1] else 0
        nz = int(H[z + 1, x]) if z + 1 < Z and fp[z + 1, x] else 0
        d.polygon([P(x, z, v), P(x + 1, z, v), P(x + 1, z + 1, v), P(x, z + 1, v)],
                  fill=shade(0.62 + 0.55 * lam[z, x]))
        if v > nx:
            d.polygon([P(x + 1, z, v), P(x + 1, z + 1, v),
                       P(x + 1, z + 1, nx), P(x + 1, z, nx)], fill=shade(0.74))
        if v > nz:
            d.polygon([P(x, z + 1, v), P(x + 1, z + 1, v),
                       P(x + 1, z + 1, nz), P(x, z + 1, nz)], fill=shade(0.52))
    return img


def build_iso_sheet(cases, out: Path, cell: int = 6) -> Path:
    """Real building beside recovered building, one row each, as shaded 3-D massing."""
    from PIL import Image, ImageDraw

    tiles = [(c, render_iso(c["target"], c["fp"], cell), render_iso(c["fitted"], c["fp"], cell))
             for c in cases]
    tw = max(max(a.width, b.width) for _, a, b in tiles)
    th = max(max(a.height, b.height) for _, a, b in tiles)
    head, pad, lab = 26, 10, 30
    W = 2 * tw + 3 * pad
    sheet = Image.new("RGB", (W, head + len(tiles) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), "REAL BUILDING", fill=(0, 0, 0))
    d.text((pad + tw + pad, 8), "RECOVERED BY THE PROGRAM", fill=(0, 0, 0))
    for i, (c, a, b) in enumerate(tiles):
        y = head + i * (th + lab)
        sheet.paste(a, (pad + (tw - a.width) // 2, y + (th - a.height) // 2))
        sheet.paste(b, (2 * pad + tw + (tw - b.width) // 2, y + (th - b.height) // 2))
        d.text((pad, y + th + 6),
               f"id {c['id']}   {c['n_ops']} ops: {' > '.join(c['ops']) or 'empty'}"
               f"   surplus left {c['extra']:.3f}", fill=(40, 40, 40))
        d.line([(0, y + th + lab - 2), (W, y + th + lab - 2)], fill=(225, 225, 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


if __name__ == "__main__":
    main()
