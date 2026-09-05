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

THE POLYGON VERTEX BUDGET (#131)
--------------------------------
`--vertex_budget` re-reads a finished artifact and asks what those polygons actually cost. Every
recovered region is an exact voxel-boundary ring at a median of **94 vertices**, which is a raster
trace rather than an architectural region, so a program's real DSL token cost -- and any claim about
program simplicity resting on `dl_ops`, which counts operations and ignores their vertices -- is
measured with the dominant term missing. Cutting each polygon back until one more deletion would
move a cell takes the median region to **58** vertices and a program from **578 tokens to 342**
with the geometry unchanged to the voxel. Below that a budget is a fidelity trade, and 🔑 it is a
worse one than the median `extra` suggests: a region that shrinks abandons its columns to the full
envelope height, so the surplus stands up as spikes rather than spreading over the roof.

Output is a semantic program per building plus the recovery statistics. It trains nothing, touches
no GPU, and does not modify the active #92 experiment.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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


# ----------------------------------------------------------------------------------------------
# #9 / #149 -- a soft, per-axis block-coordination bias on top of the same fitter
# ----------------------------------------------------------------------------------------------

# Soft by construction (#9's own decision): a full-match bonus can only ever tip a choice between
# candidates within this fraction of each other's raw gain. It can never make the fitter prefer a
# candidate that removes dramatically less surplus, so a program's real quality (`missing`/`extra`/
# collapse) cannot regress beyond noise -- #149's second acceptance criterion.
#
# Measured against no bias at all, `FitBias(roof_family="ramp")`, on the first ids of
# `SHIP714` read from `H5`: Ramp's share of chosen ops rose 0.541 -> 0.647 under `fit_program`
# (n=60) and 0.508 -> 0.610 under `fit_program_beam` at beam=branch=4 (n=40), while mean `extra`
# moved 0.006981 -> 0.006990 on the first and did not move at all on the second -- a measurable
# shift in family (#149 acceptance criterion 2) at no cost in residual (its other half).
# ⚠️ Measured ad hoc, from a throwaway script, and NOT reproduced by anything committed: no CLI
# flag reaches `bias` (that is #150/#151's job) and this file's tests are corpus-free by
# convention. Re-deriving it means writing that loop again, not re-running a recorded command --
# unlike `--measure_commutativity`, which is a real re-runnable check.
#
# 🔑 The within-type axes are real but deliberately WEAK, and the difference matters to a caller.
# On the first 120 `SHIP714` ids: a `height_rhythm` aimed at each building's own lowest available
# `Layer` step changed the recovered program on 7 of 120, while a fixed `height_rhythm=(3,6,9)`
# changed 0 of 120 -- because a fixed rhythm value simply is not within
# `HEIGHT_RHYTHM_TOLERANCE_VOX` of any step that building had on offer. So a block program that
# names absolute levels will often be a silent no-op on any given footprint; one that names a
# level the footprint can actually reach will land. That is the soft prior behaving as #9
# specified (it may never force a step a building does not have), not the axis failing.
BIAS_WEIGHT = 0.15

AZIMUTH_TOLERANCE_DEG = 20.0        # a compass quadrant's worth of slack around the target azimuth
HEIGHT_RHYTHM_TOLERANCE_VOX = 1     # one voxel of slack -- the fitter's own integer quantisation grain
SETBACK_TOLERANCE_VOX = 1           # ditto, for an inset depth measured on the same integer grid

# One family name per `VOCABULARY` kind. A kind absent here simply never matches a `roof_family`
# bias (`_family_bonus` reads it with `.get`) rather than breaking the fitter.
_ROOF_FAMILY_OF = {"Layer": "flat", "CutRoof": "cut_roof", "Ramp": "ramp"}
_VALID_ROOF_FAMILIES = frozenset(_ROOF_FAMILY_OF.values())


@dataclass(frozen=True)
class FitBias:
    """#9's explicit block program, restricted to what one footprint's fit needs to see: up to
    four independent, optional targets. Every field defaults to `None` and an unset field
    contributes nothing (see `_family_bonus`/`_within_type_bonus`) -- so `FitBias()`, like
    `bias=None`, reproduces the unbiased fitter exactly, and setting only one field never perturbs
    how the other three would have been chosen (enforced by `_select`'s two-stage ranking, not
    just by each axis's own zero-contribution-when-unset rule -- see its docstring for why the
    single check alone was not enough).

    height_rhythm -- target `Layer` step height(s) in voxels, above `y0`. A single value or a
        sequence (a "rhythm" of several acceptable levels); a candidate matches if it lands within
        `HEIGHT_RHYTHM_TOLERANCE_VOX` of any of them.
    roof_family -- one of "flat" (prefer `Layer`), "cut_roof", or "ramp". The only axis allowed to
        shift which operation TYPE wins; see `_select`.
    setback -- target inward-inset depth in voxels for a `Layer` read as a setback (#4: a setback
        *is* a Layer whose polygon is the inward offset of the footprint). Measured as the
        candidate region's own minimum distance-to-footprint-edge.
    azimuth -- target `Ramp` direction in degrees, measured up the slope and in #129's own
        `arctan2(Cx, Bz)` convention, so a value read off a trained plane head means here what it
        means there. See `_within_type_bonus` for why that argument order is load-bearing.
    """
    height_rhythm: Optional[Union[float, Sequence[float]]] = None
    roof_family: Optional[str] = None
    setback: Optional[float] = None
    azimuth: Optional[float] = None

    def __post_init__(self):
        if self.roof_family is not None and self.roof_family not in _VALID_ROOF_FAMILIES:
            raise ValueError(f"roof_family must be one of {sorted(_VALID_ROOF_FAMILIES)} or "
                             f"None, got {self.roof_family!r}")

    def is_empty(self) -> bool:
        return (self.height_rhythm is None and self.roof_family is None
                and self.setback is None and self.azimuth is None)


def _family_bonus(meta: dict, bias: FitBias) -> float:
    """1.0 if this candidate's own operation type matches `bias.roof_family`, else 0.0.

    The ONLY bonus `_select` lets compete ACROSS operation types. Unset (`None`), it is always
    0.0 -- which is what makes leaving `roof_family` alone a true no-op on the type decision,
    whatever `height_rhythm`/`setback`/`azimuth` are doing (see `_select`).

    An operation kind with no family named for it scores 0.0 rather than raising, matching how
    the rest of the file's `op` dispatches skip what they don't recognise.
    """
    if bias.roof_family is None:
        return 0.0
    return float(_ROOF_FAMILY_OF.get(meta["op"]) == bias.roof_family)


def _within_type_bonus(meta: dict, bias: FitBias, dists) -> float:
    """[0, 1]: how well one candidate matches the axes that only ever compare candidates of its
    OWN operation type against each other -- `height_rhythm`/`setback` for `Layer`, `azimuth` for
    `Ramp` -- averaged over whichever of those apply to this candidate's type. Never used to
    compare candidates of different types; see `_select`.
    """
    op = meta["op"]
    hits = checks = 0

    if op == "Layer":
        if bias.height_rhythm is not None:
            checks += 1
            targets = ((bias.height_rhythm,) if np.isscalar(bias.height_rhythm)
                      else tuple(bias.height_rhythm))
            hits += int(any(abs(meta["height"] - t) <= HEIGHT_RHYTHM_TOLERANCE_VOX
                            for t in targets))
        if bias.setback is not None:
            checks += 1
            depth = int(dists["hip"][meta["_region"]].min())
            hits += int(abs(depth - bias.setback) <= SETBACK_TOLERANCE_VOX)

    if op == "Ramp" and bias.azimuth is not None:
        checks += 1
        # ⚠️ `atan2(x_coeff, z_coeff)`, in that order, because #129's `plane_to_bins` reads
        # `arctan2(Cx, Bz)` off a plane spelled `(A, Bz, Cx)` -- `Cx` on x, `Bz` on z. This
        # fitter spells the same plane `[a, b, cz]`, `b` on x and `cz` on z, so #129's azimuth
        # is `atan2(b, cz)` here. Passing them the other way round (the arguments were swapped
        # until review caught it) mirrors every angle about 45 degrees, which is silent: a
        # caller's target still matches *something*, just the wrong ramps.
        b, cz = meta["slope"]
        az = math.degrees(math.atan2(b, cz)) % 360.0
        diff = abs(az - bias.azimuth) % 360.0
        hits += int(min(diff, 360.0 - diff) <= AZIMUTH_TOLERANCE_DEG)

    return hits / checks if checks else 0.0


def _select(candidates, bias: Optional[FitBias], dists, n: int):
    """Pick the best `n` fitter candidates, honouring `bias` without letting any axis reach
    outside the decision it owns.

    🔑🔑 A single flat score (gain scaled by every matched axis at once) cannot keep the four axes
    independent: a `height_rhythm` match on a `Layer` candidate would out-bid a `CutRoof` it was
    never asked to compete against, silently acting as an unrequested `roof_family="flat"`
    preference whenever it fired. That is #149's acceptance criterion 4, "supplying only one
    leaves the other three exactly as the unbiased fitter would have chosen them".

    ⚠️ The first fix for it -- rank each type's own entrant, then rank those against each other on
    each type's best RAW gain -- was correct at `n = 1` and **wrong for every `n > 1`**, which is
    the whole beam path (`n = branch`). Every entrant of a type carried that one type's `raw_best`
    as its key, so `nlargest` drained a type's entire list before reaching the next type: a bias
    matching *nothing at all* still replaced two `CutRoof` branches with two much worse `Layer`
    ones, and moved 7 of the first 120 real buildings. Both review axes caught it; the tests did
    not, because every one of them exercised `n = 1`.

    So the split is by DECISION, and the two decisions are made in this order:

    1. **Which positions each operation type occupies** -- raw gain, shifted only by
       `roof_family`. `_family_bonus` depends on nothing but the type, and is 0.0 when
       `roof_family` is unset, so with it unset this ranking IS the unbiased one, candidate for
       candidate.
    2. **Which member of a type fills each position that type won** -- `_within_type_bonus` only.
       A type's positions are already fixed by step 1, so these axes permute occupants and can
       never move the type itself.

    Hence: within-type axes alone leave the returned types exactly as unbiased (only *which*
    `Layer` changes, never that a `Layer` displaced a `CutRoof`); `roof_family` alone leaves each
    type's internal order exactly as unbiased; a bias matching nothing is the identity; and no
    bias at all short-circuits to the fitter's original `heapq.nlargest`, bit-identical to before
    #149 (acceptance criterion 1).

    The returned list is deliberately not re-sorted by raw gain: both callers use it as a set
    (`fit_program` takes `n = 1`; the beam re-ranks everything it expands by true surplus).
    """
    candidates = list(candidates)
    if not candidates:
        return []
    if bias is None or bias.is_empty():
        return heapq.nlargest(n, candidates, key=lambda t: t[0])

    # 1. the positions, and so the type filling each one
    family_key = lambda t: t[0] * (1.0 + BIAS_WEIGHT * _family_bonus(t[2], bias))
    positions = heapq.nlargest(n, candidates, key=family_key)

    # 2. the occupant of each position, drawn from its own type in within-type order
    within_key = lambda t: t[0] * (1.0 + BIAS_WEIGHT * _within_type_bonus(t[2], bias, dists))
    ranked_by_type: dict = {}
    for t in candidates:
        ranked_by_type.setdefault(t[2]["op"], []).append(t)
    for op, group in ranked_by_type.items():
        ranked_by_type[op] = sorted(group, key=within_key, reverse=True)

    taken: dict = {}
    out = []
    for t in positions:
        op = t[2]["op"]
        out.append(ranked_by_type[op][taken.get(op, 0)])
        taken[op] = taken.get(op, 0) + 1
    return out


def fit_program(fp, y0, y1, target, max_ops=4, allowance=CARVE_NEEDED,
                ops_allowed=VOCABULARY, bias: Optional[FitBias] = None):
    """Greedy: repeatedly take the operation that removes the most surplus without cutting GT.

    `bias` (#149) only ever changes which candidate this ranks highest; it never changes which
    candidates exist, so containment (every candidate `_all_candidates` yields already stays at or
    above `target`) is untouched by it.
    """
    full = np.int16(y1 - y0 + 1)
    h = np.where(fp, full, 0).astype(np.int16)
    gt_vox = int(target[fp].sum())
    dists = _dists_for(fp)
    ops = []
    for _ in range(max_ops):
        surplus = int((h[fp] - target[fp]).sum())
        if gt_vox and surplus / gt_vox <= allowance:
            break
        picked = _select(_all_candidates(fp, dists, target, h, ops_allowed), bias, dists, n=1)
        best = picked[0] if picked else None
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
                     beam=6, branch=6, ops_allowed=VOCABULARY, bias: Optional[FitBias] = None):
    """Beam search over programs, because greedy is provably myopic on gable roofs.

    The worst-residual trace after `Ramp` landed was entirely **symmetric double ramps**: a gable
    rises from both eaves to a ridge, so no single plane dominates it and it needs two opposing
    `Ramp`s. Greedy never gets there -- one large flat `Layer` always wins the immediate gain, and
    by the time the surplus has split into the two regions that would each take a ramp, the
    operation budget is spent. That is a search failure, not a missing operation: at K=16 greedy
    already reaches 3-D IoU 0.9981, so the vocabulary is sufficient and only the order is wrong.

    Beams are de-duplicated by the height map itself rather than by the operation list, since two
    different orders that reach the same massing are the same program for every purpose here.

    `bias` (#149) shapes which candidates this function's own branch-selection step *proposes* to
    expand each beam with; it never touches the per-round beam-survival cut or the final
    "actually better" comparison below, both of which stay ranked by true surplus alone. That is
    deliberate, not a gap: letting bias influence which beam SURVIVES (rather than only which
    candidate gets tried) would let a worse-quality lineage outlive a genuinely better one purely
    for matching the bias, which is exactly what would put `missing`/`extra`/collapse at risk of
    moving beyond noise -- #149's acceptance criterion 2. The greedy fallback is given the same
    `bias` so the two programs being compared for "actually better" were fit under the same terms.
    """
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
            top = _select(_all_candidates(fp, dists, target, h, ops_allowed), bias, dists, n=branch)
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
    g_ops, g_h = fit_program(fp, y0, y1, target, max_ops, allowance, ops_allowed, bias=bias)
    if surplus(g_h) < best[0]:
        return g_ops, g_h
    return best[2], best[1]


# ----------------------------------------------------------------------------------------------
# #9 / #150 -- the explicit block program: one FitBias, applied to a named set of footprints
# ----------------------------------------------------------------------------------------------

class UnknownFootprintError(KeyError):
    """Raised by `BlockProgram.apply` when it names a footprint id absent from the mapping it is
    given, rather than silently skipping it. Deliberately a `KeyError` subclass -- footprint
    lookup elsewhere in this codebase is ordinary dict access, so a future caller wrapping that
    lookup in `except KeyError:` should not have to learn a second exception type just because
    the miss happened inside a `BlockProgram` instead of a plain dict. Its own message names
    every offending id at once, which a bare `KeyError` on the first miss would not."""


# One footprint's fitter inputs, spelled out once rather than left as a bare `Tuple`: exactly
# `fit_program_beam`'s own leading positional arguments, in that order.
FootprintFit = Tuple[np.ndarray, int, int, np.ndarray]


@dataclass(frozen=True)
class BlockProgram:
    """#9's explicit block program: a selected set of footprints plus up to four optional,
    independently-selectable coordination targets. Per #9's "block identity is ephemeral"
    decision, this object has no stable id of its own and is never persisted; it exists only for
    the duration of one `apply()` call.

    🔑 The four fields deliberately restate `FitBias`'s own four, rather than this class simply
    holding a `bias: FitBias` field -- they are two different layers, not one duplicated: this is
    #9's domain object (a *block's* coordinated decision, over a named set of footprints), and
    `FitBias` is #149's fitter-internal search parameter (what one fit call is biased by). Keeping
    them distinct means #150's own type can change independently of how #149's search happens to
    take its bias -- `to_bias()` is the one seam between them.

    Applying it means a **full re-fit** of every named footprint (#9's decision, not a parameter
    nudge) through #10's constrained beam-search fitter, biased by exactly the axes this object
    sets -- one `FitBias`, constructed once and applied identically everywhere, so a coordinated
    decision cannot silently drift from one footprint to the next. `FitBias`/`_select` (#149)
    already guarantee an unset axis is a true no-op; this object's own job is only to build that
    one bias correctly and apply it uniformly, not to re-derive that guarantee.
    """
    footprint_ids: Tuple[Any, ...]
    height_rhythm: Optional[Union[float, Sequence[float]]] = None
    roof_family: Optional[str] = None
    setback: Optional[float] = None
    azimuth: Optional[float] = None

    def __post_init__(self):
        # frozen dataclasses still allow this at construction time (`object.__setattr__`, not
        # `self.x =`) -- accepting a caller's list here rather than only a tuple, without losing
        # the hashability `frozen=True` otherwise implies.
        object.__setattr__(self, "footprint_ids", tuple(self.footprint_ids))
        self.to_bias()          # fails fast on an invalid roof_family, reusing FitBias's own check

    def to_bias(self) -> FitBias:
        """The single `FitBias` every footprint named here is re-fit under."""
        return FitBias(height_rhythm=self.height_rhythm, roof_family=self.roof_family,
                       setback=self.setback, azimuth=self.azimuth)

    def apply(self, footprints: Dict[Any, FootprintFit]) -> Dict[Any, Tuple[list, np.ndarray]]:
        """Re-fit every named footprint under this program's bias.

        `footprints` maps a footprint id to `(fp, y0, y1, target)` -- exactly `fit_program_beam`'s
        own positional arguments, so this adds no footprint representation of its own; #151 is
        what turns a result into committed `EditOp`s.

        Every id is checked against `footprints` before any footprint is fit: a missing one is a
        caller error, not a partial result, so `UnknownFootprintError` names every offending id at
        once and nothing is fit at all rather than fitting some and skipping the rest.
        """
        missing = [fid for fid in self.footprint_ids if fid not in footprints]
        if missing:
            raise UnknownFootprintError(
                f"BlockProgram names {len(missing)} footprint id(s) absent from the given "
                f"footprints: {missing!r}")

        bias = self.to_bias()
        return {fid: fit_program_beam(*footprints[fid], bias=bias)
                for fid in self.footprint_ids}


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


def program_floor(program):
    """#134's confound-control arm: the lowest height ANY `Layer` op in `program` specifies, or
    `None` if it has none.

    #131 diagnosed the spike as structural: a column no operation's region covers reverts to the
    FULL envelope height, because `replay_program`'s cascade starts there and every op only ever
    lowers within its own region. This is the "base `Layer` under the cascade" #131 named and left
    untested -- a per-building, data-derived floor requiring no information beyond what already
    recovered this program, not a tuned constant.
    """
    heights = [int(op["height"]) for op in program if op["op"] == "Layer"]
    return min(heights) if heights else None


def replay_program(fp: np.ndarray, y0: int, y1: int, program, floor: int = None) -> np.ndarray:
    """Re-run a serialised program in height-map space, reading only what the artifact stores.

    ⚠️ #134's control arm: `floor`, when given, is the height an UNCOVERED column starts at
    instead of the full envelope extent -- #131's own "base `Layer` under the cascade" (its "What
    this does not settle", item 1), implemented as changing the cascade's own starting value
    rather than literally prepending an operation, since every op already takes a MIN against
    whatever `h` already is. `None` (the default) is the existing, unchanged behaviour: every
    caller before #134 is unaffected.

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
    overlap and a permutation changed the building on **69.6%** of them; taking the min instead
    changed the result on **0 of 250** and left **0 of 2,000** permutations changed. So
    commutativity cost nothing and bought a canonical form, an equivalence test, and deletion of
    any operation rather than only the last (`EditableBuilding.remove`).
    """
    extent = y1 - y0 + 1
    start = min(int(floor), extent) if floor is not None else extent
    h = np.where(fp, np.int16(start), 0).astype(np.int16)
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


def replay_program_ordered(fp: np.ndarray, y0: int, y1: int, program) -> np.ndarray:
    """`replay_program` as it was BEFORE #4: `Layer` SETS its region rather than taking a min.

    ⚠️ Kept deliberately, and used only by `measure_commutativity`. #4's decision rests on a
    before/after number -- permuting the operations changed the building on 69.6% of programs under
    these semantics and 0% under the min -- and a "before" that cannot be re-run is an anecdote
    rather than a measurement. Nothing in the pipeline calls this.
    """
    h = np.where(fp, np.int16(y1 - y0 + 1), 0).astype(np.int16)
    dists = _dists_for(fp)
    for op in program:
        kind = op["op"]
        if kind == "Layer":
            region = _rings_to_mask(op["region"]) & fp
            h = np.where(region, np.int16(op["height"]), h).astype(np.int16)
            continue
        if kind == "Ramp":
            region = _rings_to_mask(op["region"]) & fp
            cand = np.where(region, np.minimum(h, plane_surface(op["plane"])), h)
        elif kind == "CutRoof":
            slope = (dists[op["kind"]].astype(np.float32) - 1.0) * float(op["rate"])
            cand = np.minimum(h, np.floor(int(op["eaves"]) + slope))
        else:
            raise ValueError(f"unknown operation '{kind}'")
        h = np.where(fp, np.maximum(cand.astype(np.int16), 1), 0).astype(np.int16)
    return h


def measure_commutativity(rows, n: int = 250, perms: int = 8, seed: int = 0) -> dict:
    """🔑🔑 #4's central measurement: is the serialised algebra ORDERED or COMMUTATIVE?

    The SDF compiler composes every operation with `sdf_subtract`, and subtracting A then B is
    subtracting their union, so `EditableBuilding` commutes by construction. The height-map replay
    did not, because `Layer` used to *set* its region -- which can RAISE a column an earlier
    operation lowered. This runs both readings over real recovered programs and reports the four
    numbers #4's decision was made on, so the decision is re-checkable rather than quoted.

    `rows` is `per_building` from a `program_recovery_*.json` artifact.
    """
    import itertools
    import random

    import h5py

    rng = random.Random(seed)
    out = dict(n=0, overlapping=0, ordered_permutation_changed=0,
               min_differs_from_set=0, raises_a_column=0, commuting_permutation_changed=0,
               perms=perms)
    with h5py.File(H5, "r") as g:
        for b, row in rows.items():
            prog = row.get("program") or []
            if len(prog) < 2:
                continue
            gt = np.asarray(g["sdf"][int(b)], np.float32) <= 0
            fp = np.asarray(g["footprint"][int(b)]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, _ = hf
            out["n"] += 1
            masks = [(_rings_to_mask(o["region"]) & fp) if o.get("region") is not None else fp
                     for o in prog]
            if any((masks[i] & masks[j]).any()
                   for i, j in itertools.combinations(range(len(prog)), 2)):
                out["overlapping"] += 1
            set_base = replay_program_ordered(fp, y0, y1, prog)
            min_base = replay_program(fp, y0, y1, prog)
            if not np.array_equal(set_base, min_base):
                out["min_differs_from_set"] += 1
            h = np.where(fp, np.int16(y1 - y0 + 1), 0).astype(np.int16)
            for k in range(len(prog)):
                nxt = replay_program_ordered(fp, y0, y1, prog[:k + 1])
                if (nxt[fp] > h[fp]).any():
                    out["raises_a_column"] += 1
                    break
                h = nxt
            for tag, replay, base in (("ordered_permutation_changed", replay_program_ordered,
                                       set_base),
                                      ("commuting_permutation_changed", replay_program, min_base)):
                for _ in range(perms):
                    p = prog[:]
                    rng.shuffle(p)
                    if not np.array_equal(replay(fp, y0, y1, p), base):
                        out[tag] += 1
                        break
            if out["n"] >= n:
                break
    return out


def report_commutativity(d: dict) -> None:
    n = max(d["n"], 1)
    print(f"\n-- #4 COMMUTATIVITY  n={d['n']} programs of >=2 operations, "
          f"{d['perms']} permutations each")
    for key, label in (("overlapping", "two operation regions overlap"),
                       ("ordered_permutation_changed",
                        "Layer-as-SET: a permutation changes the building"),
                       ("min_differs_from_set", "Layer-as-MIN differs from Layer-as-SET"),
                       ("raises_a_column", "some operation RAISES a column"),
                       ("commuting_permutation_changed",
                        "Layer-as-MIN: a permutation changes the building")):
        print(f"   {label:<52} {d[key]:>4}  ({d[key]/n:.3f})")
    print("   -> the algebra is COMMUTATIVE iff the last row is 0")


# ----------------------------------------------------------------------------------------------
# the polygon vertex budget (#131)
# ----------------------------------------------------------------------------------------------

# #4 and #128 both flagged this and neither started it. Every recovered region is an EXACT
# voxel-boundary ring at a median of 94 vertices per region, which is a raster trace rather than an
# architectural region: nobody can read it, a generator would have to emit it, and so the real DSL
# token cost of a program -- and any claim about program simplicity resting on `dl_ops`, which
# counts operations and ignores their vertices -- is measured with one term missing.
VERTEX_BUDGETS = (4, 6, 8, 12, 16, 24, 94)


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def _segments_cross(p1, p2, p3, p4) -> bool:
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _chord_is_simple(ring, i) -> bool:
    """The shortcut that skips vertex `i` must not cross a non-adjacent edge of its own ring."""
    n = len(ring)
    a, b = ring[(i - 1) % n], ring[(i + 1) % n]
    skip = {(i - 1) % n, i, (i + 1) % n}
    for j in range(n):
        k = (j + 1) % n
        if j in skip or k in skip:
            continue
        if _segments_cross(a, b, ring[j], ring[k]):
            return False
    return True


def _triangle_cells(a, v, b, res: int = RES, eps: float = -1e-9) -> np.ndarray:
    """The integer cell centres inside triangle (a, v, b).

    Ring vertices sit on half-voxel offsets so a voxel centre never lies on an *axis-aligned* edge
    (`mask_to_rings`), but a 45-degree shortcut between two of them passes exactly through one --
    and on this corpus that tie is the common case, not the rare one. `eps` picks the side:

      * `eps < 0` counts the boundary as inside. Conservative, and what the containment rule wants:
        it can refuse a deletion that was in fact safe, never admit one that was not.
      * `eps > 0` counts only cells strictly inside. A cell there really does change hands, so this
        is the test for "this deletion definitely moves the geometry".
    """
    xs, zs = (a[0], v[0], b[0]), (a[1], v[1], b[1])
    x0, x1 = max(int(np.ceil(min(xs))), 0), min(int(np.floor(max(xs))), res - 1)
    z0, z1 = max(int(np.ceil(min(zs))), 0), min(int(np.floor(max(zs))), res - 1)
    if x1 < x0 or z1 < z0:
        return np.empty((0, 2), int)
    gx, gz = np.meshgrid(np.arange(x0, x1 + 1), np.arange(z0, z1 + 1), indexing="ij")
    p = np.stack([gx.ravel(), gz.ravel()], 1).astype(float)
    s = 1.0 if _cross(a, v, b) > 0 else -1.0
    keep = np.ones(len(p), bool)
    for u, w in ((a, v), (v, b), (b, a)):
        keep &= (s * ((w[0] - u[0]) * (p[:, 1] - u[1]) - (w[1] - u[1]) * (p[:, 0] - u[0]))) >= eps
    return p[keep].astype(int)


# ================================================================================================
# #134 -- fit few-vertex regions directly, instead of trimming exact rings
#
# #131 priced every budget as the cost of DELETING vertices from the exact ring down to it. It left
# open whether a fitter that placed a FEW vertices directly, rather than trimming many down, would
# land somewhere better. RADmesh (ECCV 2026) optimizes on a deliberately coarse discretization and
# re-discretizes on a coarse-to-fine schedule, carrying its optimizer state across each step by
# barycentric interpolation. The mechanism transfers here as: start a region at a handful of
# vertices, then GROW it -- insert the vertex that recovers the most currently-uncovered area at
# each step -- so budget=16's polygon is built ON TOP OF budget=8's, never restarted.
# ================================================================================================


def _triangle_contained(a, v, b, exact_mask, res: int):
    """Triangle `(a, v, b)`'s cells, or `None` if any of them falls outside `exact_mask`.

    #134's one shared primitive: the identical cell-level containment test `simplify_region`'s
    `contained` rule already runs for deletion, factored out so `_seed_triangle` and
    `_fit_outer_ring_direct` -- both admitting an INSERTION instead of a deletion -- share one
    implementation of "is this triangle safe to add" rather than each re-stating it.
    """
    cells = _triangle_cells(a, v, b, res)
    if len(cells) and not exact_mask[cells[:, 1], cells[:, 0]].all():
        return None
    return cells


def _seed_triangle(ring, ri: int, exact_mask, res: int):
    """#134: the coarsest possible contained start for one ring: an "ear" -- three consecutive
    vertices whose triangle is entirely inside `exact_mask`. Ear-clipping theory guarantees at
    least one exists for any simple polygon (a ring here always is one), so this is always found,
    not a heuristic that can fail on a well-formed ring; the explicit containment re-check is
    defensive. An ear is a corner CONVEX in ring-space -- `turn > 0` for the outer ring, `turn < 0`
    for a hole, the exact conditions `simplify_region`'s own `contained` rule already uses,
    mirrored.
    """
    n = len(ring)
    if n <= 3:
        return list(range(n))
    for i in range(n):
        a, v, b = ring[(i - 1) % n], ring[i], ring[(i + 1) % n]
        turn = _cross(a, v, b)
        is_ear = (turn > 0) if ri == 0 else (turn < 0)
        if not is_ear:
            continue
        if _triangle_contained(a, v, b, exact_mask, res) is None:
            continue
        return sorted({(i - 1) % n, i, (i + 1) % n})
    return None


def _fit_outer_ring_direct(ring, exact_mask, res: int, budget: int):
    """#134: grow the OUTER ring from a 3-vertex seed toward `exact_mask`, inserting the vertex
    that recovers the most currently-uncovered area at each step, subject to `_triangle_contained`
    -- the same cell-level test `simplify_region`'s `contained` rule uses for deletion, shared here
    rather than restated, applied to insertion instead (`grows = turn > 0`, the mirror image of
    the deletion rule's `turn < 0`).

    ⚠️ Scoped to the outer ring only (#134's own scope decision): a hole's own vertices stay on the
    existing, already-measured `contained` trimming path (`simplify_region` handles this by
    allocating `budget` across all rings; a caller here passes the remainder after hole trimming).
    Holes are the minority of the vertex count (#131: holes are usually near their own floor
    already), and growing them too would need a signed incremental-coverage update (a hole
    REMOVES area rather than adding it) this scope does not need to build.

    Each step's kept-index set is a strict superset of the previous step's -- RADmesh's "carry the
    fit forward across re-discretization" -- by construction, not by extra bookkeeping.
    """
    n = len(ring)
    seed = _seed_triangle(ring, 0, exact_mask, res)
    if seed is None:                                      # defensive: a malformed ring
        return ring
    kept = set(seed)
    if len(kept) >= n:
        return ring
    covered = np.zeros_like(exact_mask)
    seed_cells = _triangle_cells(ring[seed[0]], ring[seed[1]], ring[seed[2]], res)
    if len(seed_cells):
        covered[seed_cells[:, 1], seed_cells[:, 0]] = True

    while len(kept) < min(budget, n):
        best = None                                       # (gain, j, cells)
        idxs = sorted(kept)
        for pos, i in enumerate(idxs):
            nxt = idxs[(pos + 1) % len(idxs)]
            span = (nxt - i) % n
            for step in range(1, span):
                j = (i + step) % n
                a, v, b = ring[i], ring[j], ring[nxt]
                if _cross(a, v, b) <= 0:                  # not a bulge outward: no area to gain
                    continue
                cells = _triangle_contained(a, v, b, exact_mask, res)
                if cells is None:
                    continue                              # would add a cell outside the exact region
                gain = (int((exact_mask[cells[:, 1], cells[:, 0]]
                            & ~covered[cells[:, 1], cells[:, 0]]).sum()) if len(cells) else 0)
                if best is None or gain > best[0]:
                    best = (gain, j, cells)
        if best is None or best[0] <= 0:
            break                                          # no admissible insertion still gains area
        _, j, cells = best
        kept.add(j)
        if len(cells):
            covered[cells[:, 1], cells[:, 0]] = True

    return [ring[i] for i in sorted(kept)]


def _simplify_region_direct(rings, budget: int, exact_mask, res: int):
    """#134's `direct` rule: the outer ring GROWS toward the exact region (`_fit_outer_ring_
    direct`); holes are reduced to their own LOSSLESS floor first -- free, no fidelity cost, via
    the existing `lossless` rule on each hole ring by itself -- and spend whatever that floor
    costs before the outer ring gets the remainder.

    ⚠️ Holes are not GROWN under this rule -- a deliberate scope decision, not an oversight. They
    are the minority of the vertex count (15.7% of regions carry one at all, #131) and are usually
    already small (a median hole is a single cell: 4 vertices, which the lossless floor cannot
    reduce further -- #131's own finding, "a speckle hole is 4 vertices that cannot be spent").
    #134 is scoped to the outer boundary, where the vertex budget and the spike problem #131
    diagnosed both concentrate. A pathological region with many holes (#131 records one with 156)
    can still starve the outer ring down to its 3-vertex floor regardless of budget -- an honest,
    documented limitation of this scope, not a silent wrong answer: the outer ring's own growth
    still respects every hole correctly regardless, since `exact_mask` already excludes hole
    cells, so a growth step that would bulge into one fails the same containment test as any
    other out-of-region cell.
    """
    outer, holes = rings[0], rings[1:]
    reduced_holes = []
    for h in holes:
        if len(h) <= 3:
            reduced_holes.append(h)
            continue
        hole_mask = _rings_to_mask([h], res)
        reduced_holes.append(simplify_region([h], 0, hole_mask, res, "lossless")[0])
    hole_verts = sum(len(h) for h in reduced_holes)
    outer_budget = max(3, budget - hole_verts)
    grown = _fit_outer_ring_direct([[float(v[0]), float(v[1])] for v in outer], exact_mask, res,
                                   outer_budget)
    return [grown] + [[[float(v[0]), float(v[1])] for v in h] for h in reduced_holes]


def simplify_region(rings, budget: int, exact_mask, res: int = RES, rule: str = "contained"):
    """One region's rings -> the same region under a vertex budget, cheapest corner first.

    Greedy least-area vertex deletion (Visvalingam) run over **every ring of the region at once**,
    so the budget is a per-region total and a hole competes with the outer boundary for it.

    🔑 With `exact_mask` given, a deletion is admitted only when every cell it would ADD to the
    region is already inside the exact region. That keeps the simplified region a subset of the
    exact one **at the cell level, which is the level the compiler rasterizes at** -- so #10's
    containment guarantee survives: a region that only ever shrinks can only leave surplus, and can
    never cut into GT. `missing` and `collapse_rate` stay 0 by construction, and the whole cost of
    the budget lands on `extra`, where it is visible.

    ⚠️ The cell level is the right level and the polygon level is the wrong one. A chord across a
    staircase of half-voxel steps bulges outside the exact *polygon* while covering not one new
    voxel *centre*: an exact 62-vertex diagonal trace is the same 30 cells as a 4-vertex triangle.
    Constraining the polygon instead would refuse that for nothing.

    Three rules, and they differ in that one test alone:

      * `contained` -- admit a deletion only if every cell it ADDS is already in the exact region.
        ⚠️ The sweep's artifact keys and table column call this arm `inner`; same rule, older name,
        kept so a published table and the artifact behind it do not disagree.
        Needs `exact_mask`. The region can only shrink, so the whole cost lands on `extra`.
      * `lossless`  -- admit a deletion only if its triangle holds NO cell centre at all, so the
        rasterized region does not change by one cell. Run to `budget=0` this answers the question
        the ticket actually asks: **how many vertices does this region need?** Every vertex above
        that count is the rasterizer's, not the architecture's.
      * `free`      -- no test. It may also delete a ring outright once it is down to a triangle,
        which is how a one-cell speckle hole disappears -- by swallowing its cell.
      * `direct`    -- #134: not a deletion rule at all. Grows the outer ring from a coarse seed
        toward the exact region instead of trimming the exact ring down -- see
        `_fit_outer_ring_direct`. The one thing it shares with `contained` is the same cell-level
        containment test, reused for insertion rather than deletion.

    ⚠️ Marching squares is the obvious tool for this and is wrong here (`mask_to_rings` says why):
    it chamfers every corner diagonally, handing a plain rectangular shed four 45-degree eaves it
    does not have. This deletes *existing* vertices and never invents one, so a rectangle stays a
    rectangle at every budget -- the check `test_simplify_region_keeps_a_plain_shed` pins.
    """
    if rule == "direct":
        return _simplify_region_direct(rings, budget, exact_mask, res)
    rs = [[[float(v[0]), float(v[1])] for v in r] for r in rings]
    while sum(len(r) for r in rs) > budget:
        cand = []
        for ri, ring in enumerate(rs):
            n = len(ring)
            if n <= 3:
                if rule == "free" and n == 3:
                    cand.append((abs(_cross(*ring)) / 2.0, ri, None, False))
                continue
            for i in range(n):
                a, v, b = ring[(i - 1) % n], ring[i], ring[(i + 1) % n]
                turn = _cross(a, v, b)
                tie = False
                if rule == "lossless":
                    if len(_triangle_cells(a, v, b, res, +1e-9)):
                        continue                     # a cell centre is strictly inside: it moves
                    # a centre exactly ON the shortcut is a tie the rasterizer has to settle
                    tie = bool(len(_triangle_cells(a, v, b, res)))
                # ring 0 is the outer boundary and the rest are holes, subtracted by position. So a
                # REFLEX corner of the outer ring bulges outward, and a CONVEX corner of a hole
                # shrinks the hole: both hand the region cells it did not have.
                elif rule == "contained" and ((turn < 0) if ri == 0 else (turn > 0)):
                    cells = _triangle_cells(a, v, b, res)
                    if len(cells) and not exact_mask[cells[:, 1], cells[:, 0]].all():
                        continue
                cand.append((abs(turn) / 2.0, ri, i, tie))
        cand.sort(key=lambda c: c[0])
        for _, ri, i, tie in cand:
            if i is None:
                rs.pop(ri)
                break
            if not _chord_is_simple(rs[ri], i):
                continue
            if tie:
                trial = [r[:] for r in rs]
                trial[ri].pop(i)
                if not np.array_equal(_rings_to_mask([r for r in trial if len(r) >= 3], res),
                                      exact_mask):
                    continue
            rs[ri].pop(i)
            break
        else:
            break                    # nothing admissible is left, so this budget is not reachable
    return [r for r in rs if len(r) >= 3]


def simplify_program(program, budget: int, rule: str = "contained", res: int = RES):
    """Re-cut every polygon in a program to `budget` vertices, and measure what it cost.

    Returns the rewritten program and the per-region ledger: vertices before and after, cells the
    region gained (the containment breach, which must be 0 under `contained`) and cells it gave up.
    """
    out, ledger = [], []
    for op in program:
        op = dict(op)
        rings = op.get("region")
        if rings:
            exact = _rings_to_mask(rings, res)
            simp = simplify_region(rings, budget, exact, res, rule)
            got = _rings_to_mask(simp, res) if simp else np.zeros((res, res), bool)
            ledger.append(dict(op=op["op"], before=sum(len(r) for r in rings),
                               after=sum(len(r) for r in simp),
                               rings_before=len(rings), rings_after=len(simp),
                               met=bool(sum(len(r) for r in simp) <= budget),
                               added=int((got & ~exact).sum()), lost=int((exact & ~got).sum()),
                               area=int(exact.sum())))
            op["region"] = simp
        out.append(op)
    return out, ledger


def dsl_tokens(program) -> int:
    """A program's real token cost, counted the way a generator would have to emit it.

    One token for the operation name, one per scalar parameter, one per ring (its separator), and
    **two per vertex**. `dl_ops` counts the first of those and ignores the rest, which is exactly
    the missing term: at the exact ring a two-operation roof costs 4 tokens by `dl_ops` and around
    380 by this.
    """
    n = 0
    for op in program:
        if op["op"] == "Layer":
            n += 2                                            # Layer, height
        elif op["op"] == "Ramp":
            n += 4                                            # Ramp, plane a/b/c
        elif op["op"] == "CutRoof":
            n += 4                                            # CutRoof, kind, eaves, rate
        else:
            raise ValueError(f"unknown operation '{op['op']}'")
        for ring in op.get("region", []):
            n += 1 + 2 * len(ring)
    return n


_H5_HANDLE = None


def _h5():
    """One h5 handle per worker process, opened on first use rather than pickled."""
    global _H5_HANDLE
    if _H5_HANDLE is None:
        _H5_HANDLE = h5py.File(H5, "r")
    return _H5_HANDLE


def _budget_case(task):
    """Score one building's program at the exact ring and at every budget, both arms.

    Runs in a worker process. The exact-ring row is not read from the recovery artifact but
    re-derived through the same replay as every other row, so the control and the arms differ only
    in the polygons -- and its agreement with the recorded numbers is itself a check that the
    artifact is a program rather than a result (#128).
    """
    bid, program, budgets = task
    g = _h5()
    gt = np.asarray(g["sdf"][bid], np.float32) <= 0
    fp = np.asarray(g["footprint"][bid]) > 0
    hf = height_field(gt, fp)
    if hf is None:
        return bid, None
    y0, y1, target = hf
    extent = y1 - y0 + 1
    bo_occ = occupancy(fp, y0, np.where(fp, np.int16(extent), 0).astype(np.int16))

    from scripts.foundations.train_height_map_generator import roof_description_length

    def score(prog, ledger, floor=None):
        h = replay_program(fp, y0, y1, prog, floor=floor)
        occ = occupancy(fp, y0, h)
        row = volume_split(occ, gt)
        row["vs_input"] = vs_input(occ, bo_occ)
        dl = roof_description_length(h, fp, y0, extent)
        row["dl_ops"], row["dl_planar_fraction"] = dl["ops"], dl["planar_fraction"]
        row["tokens"] = dsl_tokens(prog)
        row["verts"] = sum(sum(len(r) for r in o.get("region", [])) for o in prog)
        row["regions"] = len(ledger)
        row["regions_met"] = sum(1 for e in ledger if e["met"])
        row["regions_added"] = sum(1 for e in ledger if e["added"])
        row["cells_added"] = sum(e["added"] for e in ledger)
        row["cells_lost"] = sum(e["lost"] for e in ledger)
        row["cells_region"] = sum(e["area"] for e in ledger)
        row["region_verts"] = [e["after"] for e in ledger]
        row["region_verts_exact"] = [e["before"] for e in ledger]
        # ⚠️ the median `extra` hides WHERE the surplus goes. A region that pulls back leaves the
        # columns it abandoned standing at the full envelope height, and the montage shows those as
        # spikes -- a visible fault, not an approximation, so they are priced beside the median.
        surplus = (h.astype(np.int32) - target.astype(np.int32))[fp]
        row["surplus_max"] = int(surplus.max()) if surplus.size else 0
        row["spike_columns"] = int((surplus > S_STAR_VOXELS).sum())
        return row

    exact_ledger = [dict(op=o["op"], before=sum(len(r) for r in o["region"]),
                         after=sum(len(r) for r in o["region"]), rings_before=len(o["region"]),
                         rings_after=len(o["region"]), met=True, added=0, lost=0,
                         area=int(_rings_to_mask(o["region"]).sum()))
                    for o in program if o.get("region")]
    out = {"exact": score(program, exact_ledger)}
    # the vertices the regions actually NEED: simplified until the rasterized cells would change,
    # with no budget at all. Everything above this count is the rasterizer's, not the building's.
    out["lossless"] = score(*simplify_program(program, 0, "lossless"))
    floor = program_floor(program)                        # #134's confound control, computed once
    for v in budgets:
        for arm, rule in (("inner", "contained"), ("free", "free"), ("direct", "direct")):
            prog_v, ledger_v = simplify_program(program, v, rule)
            out[f"{arm}{v}"] = score(prog_v, ledger_v)
            if arm == "inner":
                # does a base-Layer floor ALONE -- no change to the fitter -- already fix the
                # spike `inner` leaves? Isolates the floor's own effect from the search's, so a
                # `direct` win cannot be attributed to the wrong cause (#134's "confound to
                # control first").
                out[f"floor{v}"] = score(prog_v, ledger_v, floor=floor)
    return bid, out


def measure_vertex_budget(rows, budgets=VERTEX_BUDGETS, workers: int = 0) -> dict:
    """#131, on the pinned carve-needing subset: what does each vertex budget cost the building?

    Only carve-needing buildings are scored, per #126: an already-flat building has no operations,
    so no polygon, so nothing a budget could change -- pooling them in would dilute every number
    with 303 rows that are identical in every column.
    """
    import multiprocessing as mp
    # imported HERE, before the pool forks, so 62 workers share one copy of it and of torch through
    # copy-on-write instead of each paying the import. Unused in this frame by design.
    from scripts.foundations.train_height_map_generator import roof_description_length  # noqa: F401

    carve = [(int(b), r["program"], tuple(budgets)) for b, r in rows.items()
             if r["blockout_extra"] >= CARVE_NEEDED]
    n = workers or min(len(carve), max(mp.cpu_count() - 2, 1))
    print(f"[#131] {len(carve)} carve-needing buildings x {1 + 2 * len(budgets)} configurations "
          f"on {n} workers", flush=True)
    t0, out = time.time(), {}
    with mp.get_context("fork").Pool(n) as pool:
        for k, (bid, res) in enumerate(pool.imap_unordered(_budget_case, carve, chunksize=1)):
            if res is not None:
                out[str(bid)] = res
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(carve)}  {time.time()-t0:.0f}s", flush=True)
    return dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), question="#131",
                          n=len(out), budgets=list(budgets), allowance=CARVE_NEEDED,
                          collapse_missing=COLLAPSE_MISSING),
                per_building=out)


def report_vertex_budget(art: dict) -> None:
    """The fidelity-against-budget table, with #126's guards on every row and not only the headline."""
    pb, budgets = art["per_building"], art["meta"]["budgets"]
    med = lambda cfg, k: float(np.median([r[cfg][k] for r in pb.values()]))
    tot = lambda cfg, k: int(sum(r[cfg][k] for r in pb.values()))
    coll = lambda cfg: float(np.mean([r[cfg]["missing"] >= COLLAPSE_MISSING for r in pb.values()]))

    print(f"\n{'=' * 108}")
    print(f"#131 THE POLYGON VERTEX BUDGET   n={len(pb)} carve-needing buildings")
    print(f"{'=' * 108}")
    print(f"{'budget':>8} {'arm':<6} | {'verts':>6} {'tokens':>7} | {'missing':>8} {'extra':>7} "
          f"{'vs_input':>8} {'collapse':>8} | {'ops':>4} {'planar':>6} | {'spike':>6} {'spiked':>7} "
          f"| {'contained':>9} {'met':>5} {'cells+':>7}")
    print("-" * 118)

    def line(label, arm, cfg):
        regions, met = tot(cfg, "regions"), tot(cfg, "regions_met")
        breached = tot(cfg, "regions_added")
        spiked = float(np.mean([r[cfg]["spike_columns"] > 0 for r in pb.values()]))
        print(f"{label:>8} {arm:<6} | {med(cfg, 'verts'):>6.0f} {med(cfg, 'tokens'):>7.0f} | "
              f"{med(cfg, 'missing'):>8.4f} {med(cfg, 'extra'):>7.4f} "
              f"{med(cfg, 'vs_input'):>8.4f} {coll(cfg):>8.4f} | "
              f"{med(cfg, 'dl_ops'):>4.1f} {med(cfg, 'dl_planar_fraction'):>6.2f} | "
              f"{med(cfg, 'surplus_max'):>6.0f} {spiked:>7.3f} "
              f"| {1 - breached / max(regions, 1):>9.4f} {met / max(regions, 1):>5.2f} "
              f"{tot(cfg, 'cells_added'):>7}")

    line("exact", "-", "exact")
    line("needed", "-", "lossless")
    any_row = next(iter(pb.values()))
    for v in budgets:
        print("-" * 118)
        # #134 adds `direct`/`floor`; detected rather than hardcoded so this still reports an
        # OLD #131-only artifact (inner/free alone) without a KeyError.
        for arm in ("inner", "floor", "direct", "free"):
            if f"{arm}{v}" in any_row:
                line(str(v), arm, f"{arm}{v}")
    print("-" * 118)
    print("  `contained` = fraction of regions that gained NO cell, so #10's guarantee holds and")
    print("  `missing`/collapse stay 0 by construction.  `met` = fraction that reached the budget.")
    print("  `needed` = every polygon cut back until one more deletion would move a cell: same")
    print("  building to the voxel, so the difference from `exact` is pure rasterizer.")
    print(f"  `spike` = median worst column surplus in voxels; `spiked` = fraction of buildings")
    print(f"  with any column more than s* = {S_STAR_VOXELS} voxels proud of GT (ADR 0004).")

    ex = np.array([v for r in pb.values() for v in r["exact"]["region_verts_exact"]])
    nd = np.array([v for r in pb.values() for v in r["lossless"]["region_verts"]])
    print(f"\n  VERTICES PER REGION, over all {len(ex)} regions   "
          f"(the ticket asks for the distribution, not the median)")
    print(f"{'':>14}{'min':>6}{'p10':>6}{'p25':>6}{'median':>8}{'p75':>6}{'p90':>6}{'p99':>6}"
          f"{'max':>6}   {'<=4':>6} {'<=8':>6} {'<=16':>6}")
    for label, a in (("exact ring", ex), ("needed", nd)):
        pct = lambda q: np.percentile(a, q)
        print(f"  {label:<12}{a.min():>6}{pct(10):>6.0f}{pct(25):>6.0f}{np.median(a):>8.0f}"
              f"{pct(75):>6.0f}{pct(90):>6.0f}{pct(99):>6.0f}{a.max():>6}   "
              f"{np.mean(a <= 4):>6.2f} {np.mean(a <= 8):>6.2f} {np.mean(a <= 16):>6.2f}")
    print(f"{'=' * 108}")


def build_budget_sheet(cases, out: Path, budget: int, cell: int = 6) -> Path:
    """The real building, the exact-ring program, and the same program at one vertex budget."""
    from PIL import Image, ImageDraw

    cols = ["REAL BUILDING", "EXACT RING (median 94 verts)",
            "CHOSEN: LOSSLESS FLOOR (median 58)",
            f"BUDGET {budget}, CONTAINED", f"BUDGET {budget}, FREE"]
    tiles = [(c, [render_iso(h, c["fp"], cell) for h in
                  (c["target"], c["exact"], c["lossless"], c["inner"], c["free"])]) for c in cases]
    tw = max(max(t.width for t in ts) for _, ts in tiles)
    th = max(max(t.height for t in ts) for _, ts in tiles)
    head, pad, lab = 26, 10, 44
    W = len(cols) * (tw + pad) + pad
    sheet = Image.new("RGB", (W, head + len(tiles) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((pad + j * (tw + pad), 8), c, fill=(0, 0, 0))
    for i, (c, ts) in enumerate(tiles):
        y = head + i * (th + lab)
        for j, t in enumerate(ts):
            sheet.paste(t, (pad + j * (tw + pad) + (tw - t.width) // 2, y + (th - t.height) // 2))
        d.text((pad, y + th + 4),
               f"id {c['id']}   {' > '.join(c['ops']) or 'empty'}   "
               f"{c['verts_exact']} verts -> {c['verts_lossless']} lossless (CHOSEN) / "
               f"{c['verts_inner']} contained / {c['verts_free']} free",
               fill=(40, 40, 40))
        d.text((pad, y + th + 20),
               f"surplus  exact {c['extra_exact']:.4f}   lossless {c['extra_lossless']:.4f}   "
               f"contained {c['extra_inner']:.4f}   free {c['extra_free']:.4f}  "
               f"(free cuts into GT: missing {c['missing_free']:.4f})",
               fill=(120, 40, 40))
        d.line([(0, y + th + lab - 2), (W, y + th + lab - 2)], fill=(225, 225, 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def budget_montage(rows, art: dict, budget: int, out: Path, n: int = 6) -> Path:
    """Six representative carve-needing buildings, rendered at the exact ring and at `budget`."""
    pb = art["per_building"]
    ranked = sorted(pb, key=lambda b: pb[b][f"inner{budget}"]["extra"])
    picks = [ranked[int(round(q * (len(ranked) - 1)))]
             for q in np.linspace(0.1, 0.9, n)]
    cases = []
    with h5py.File(H5, "r") as g:
        for b in picks:
            gt = np.asarray(g["sdf"][int(b)], np.float32) <= 0
            fp = np.asarray(g["footprint"][int(b)]) > 0
            y0, y1, target = height_field(gt, fp)
            prog = rows[b]["program"]
            hs = {"exact": replay_program(fp, y0, y1, prog)}
            # the CHOSEN answer is the lossless floor, so it gets a column of its own; the budget
            # columns beside it are what was rejected and why
            for arm, rule, v in (("lossless", "lossless", 0), ("inner", "contained", budget),
                                 ("free", "free", budget)):
                p, _ = simplify_program(prog, v, rule)
                hs[arm] = replay_program(fp, y0, y1, p)
            cases.append(dict(id=int(b), fp=fp, target=target, ops=rows[b]["ops"], **hs,
                              verts_exact=pb[b]["exact"]["verts"],
                              verts_lossless=pb[b]["lossless"]["verts"],
                              extra_lossless=pb[b]["lossless"]["extra"],
                              verts_inner=pb[b][f"inner{budget}"]["verts"],
                              verts_free=pb[b][f"free{budget}"]["verts"],
                              extra_exact=pb[b]["exact"]["extra"],
                              extra_inner=pb[b][f"inner{budget}"]["extra"],
                              extra_free=pb[b][f"free{budget}"]["extra"],
                              missing_free=pb[b][f"free{budget}"]["missing"]))
    return build_budget_sheet(cases, out, budget)


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
    ap.add_argument("--measure_commutativity", type=int, default=0, metavar="N",
                    help="replay N committed programs under permutation, under both the old "
                         "Layer-as-SET reading and the current Layer-as-MIN one, and report #4's "
                         "ordering table. Reads --out as an EXISTING artifact and writes nothing")
    ap.add_argument("--vertex_budget", nargs="*", type=int, default=None, metavar="V",
                    help="#131: re-cut every polygon in --out to each vertex budget and re-score "
                         "the compiled building, under both the containment-preserving arm and the "
                         "unconstrained one. Reads --out as an EXISTING artifact and writes a "
                         "companion; no fit is re-run. Bare flag = the pre-registered budgets")
    ap.add_argument("--budget_montage", type=int, default=0, metavar="V",
                    help="with --vertex_budget, render six buildings at budget V beside the "
                         "exact-ring fit")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes for --vertex_budget; 0 = all but two cores")
    ap.add_argument("--verify_edit_stack", type=int, default=0,
                    help="load this many recovered programs into EditableBuilding and check the "
                         "SDF composition, the replay and undo against the voxel compiler (#128)")
    args = ap.parse_args()

    if args.measure_commutativity:
        # a measurement over ALREADY-RECOVERED programs: it re-runs no fit and writes no artifact,
        # so it cannot disturb the record it is checking
        rows = json.load(open(args.out))["per_building"]
        report_commutativity(measure_commutativity(rows, n=args.measure_commutativity))
        return

    if args.vertex_budget is not None:
        # a measurement over ALREADY-RECOVERED programs, like --measure_commutativity: it re-runs
        # no fit and never rewrites --out, so it cannot disturb the record it is reading
        rows = json.load(open(args.out))["per_building"]
        art = measure_vertex_budget(rows, args.vertex_budget or VERTEX_BUDGETS, args.workers)
        p = Path(args.out).with_name(Path(args.out).stem + "_vertex_budget.json")
        json.dump(art, open(p, "w"), indent=1)
        print(f"[artifact] {p}", flush=True)
        report_vertex_budget(art)
        if args.budget_montage:
            sheet = budget_montage(rows, art, args.budget_montage,
                                   REPO / "outputs/program_recovery/vertex_budget.png")
            print(f"[montage] {sheet}")
        return

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
