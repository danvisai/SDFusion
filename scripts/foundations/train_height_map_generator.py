"""Does a footprint-conditioned height-map generator carve, or does it learn identity too?

#127's question. Every generator on this project's record -- #69 through #92, six arms across two
representations -- converged to returning its own input. This asks whether that was the *model* or
the *output space*, by moving the output space to the one #10 measured the corpus actually to be:
a 64x64 height map.

WHAT IS PREDICTED, AND WHY IT IS THE CARVE AND NOT THE HEIGHT
-------------------------------------------------------------
The label is the per-column **carve depth** `d = extent - top`, classified over 64 levels, not the
absolute top and not a regression. Three reasons, in the order they matter:

  * **Depth makes the arm purely subtractive.** `apply_depth` clamps to `[1, extent]`, so a
    prediction can never exceed the blockout it started from and `extra` can never come out worse
    than doing nothing. #10 measured `missing`=0 on 714/714 -- the real building is always inside
    its own extruded footprint -- so subtractive-only is the corpus's own structure, not a
    convenience.
  * **Classification, not MSE.** MSE returns the conditional mean, which on a bimodal roof
    distribution (flat top / pitched) is a roof nobody built -- the same regression-to-the-mean that
    produced the no-op. `--objective mse` exists to *test* that claim rather than assume it, and is
    scored as its own arm.
  * **The labels are exact integers already.** No quantisation, no codec, no latent. #10's
    reconstruction residual over the pinned 714 is 71 voxels in 4.3M.

WHAT THE OUTPUT SPACE GIVES FOR FREE, STATED PRECISELY
------------------------------------------------------
#127 claims a height map is "footprint-exact, collapse-impossible, and `missing` and `collapse_rate`
are 0 by clamping". Two of those are true and one is not, and the tests pin the difference:

  * footprint-exact  -- TRUE. `apply_depth` writes exactly the footprint mask, so fp-IoU is 1.0000
                        by construction for every prediction, good or bad.
  * a valid solid    -- TRUE. Every footprint column keeps at least one voxel, so no prediction can
                        punch a hole through the plan or return a hollow shell (#80's failure).
  * `missing` = 0    -- FALSE. Over-carving still eats GT. The collapse rate is measured on the
                        model's own output and published beside every number, exactly as #126
                        requires of the alternative-building arm that collapses on 16.7%.

THE BAR, PRE-REGISTERED BEFORE THE FIRST RUN
--------------------------------------------
Fixed here so a result cannot re-litigate it (map #87's discipline, and #10's record of stopping at
a dip and being wrong twice). Scored on the **411 carve-needing** buildings of the pinned 714 --
303 need no carve at all and a 42% no-op majority flatters every aggregate (#126 point 4).

  PASS   median `extra` strictly below the **1-NN retrieval** arm's, measured on the same rows in
         the same run. 1-NN is the bar, not the blockout (#127).
  GUARD  collapse rate no worse than 1-NN's, and `vs_input` < 0.98 -- an arm that did not move has
         not been measured as a generator at all (#75).
  KILL   median `extra` at or above the blockout's. That is identity, and it answers #127 "no".

The aggregate 3D IoU is reported to the right of the bar and is a diagnostic, never a gate: #126
demoted it because its median cannot rank a real building above the envelope.

Run -- one arm at a time, then score them together against the baselines:

    P="env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/train_height_map_generator.py"
    $P --objective ce  --tag heightmap_ce  --epochs 40 --montage 0   # ~16 min on one A100
    $P --objective mse --tag heightmap_mse --epochs 40 --montage 0
    $P --ckpt heightmap_ce=outputs/height_map_generator/heightmap_ce.pt \
              heightmap_mse=outputs/height_map_generator/heightmap_mse.pt \
              --median_decode --montage 6

The 3D montage says whether an arm reads as a building. The plan-view pair says *why* -- the height
map shows where the volume went, the normal map shows whether what is left is made of planes -- and
it scores nothing, so it runs from finished checkpoints in seconds:

    $P --ckpt heightmap_ce=outputs/height_map_generator/heightmap_ce.pt \
              heightmap_planes=outputs/height_map_generator/heightmap_planes.pt \
              --median_decode --maps_only --maps 4          # best/representative/worst sheets
    $P --ckpt ... --median_decode --maps_only --maps_ids 1341 19229 20650   # named buildings
    $P --ckpt ... --maps_only --maps_arms heightmap_ce_median heightmap_ce_slope_median

The joint SLOPE term is an arm like any other and is off by default, so every arm on the record is
unaffected. Its pre-registered weight is 1.0 (`docs/wayfinding/solid-first-subtractive-modeling/
127-height-map-generator.md`):

    $P --objective ce --slope_weight 1.0 --tag heightmap_ce_slope --epochs 40 --montage 0 --no_form

The first invocation builds `outputs/height_map_generator/height_fields.npz` from the corpus, which
takes ~12 minutes and is then reused by everything else.


#6 -- THE PROGRAM ARM, AND WHY IT IS AN ARM OF THIS SCRIPT AND NOT A NEW ONE
============================================================================
#127 closed with the open problem moved from *amount* to *form*: every trained arm removes about the
right volume and none produces planes and ridges. It measured the cause from both directions --
supervision could not put planes in (the slope term bought description length and reversed
planarity) and decoding could not take a roof out (an oracle quantile chosen per building with the
answer in hand buys 12% of the symmetric difference and **exactly zero shape**). What neither can
supply is a **joint commitment**: one hypothesis chosen across a run of columns rather than 4,096
independent summaries, whose pointwise median is a mound that is none of them.

#6 asks which learned formulation makes that commitment. The answer here is chosen by measurement:

  * **Predict the program, not the surface.** K=4 typed slots -- each a `Layer` (flat) or a `Ramp`
    (a plane) -- plus one assignment per column over the slots and an UNCARVED class. Every column
    in a region gets its height from the slot the region shares, so a ridge line is one decision.
  * **Supervise it with #10's fitter, exactly.** Measured before the arm was designed: the fitter is
    deterministic, sees GT, and costs 0.2 s per building, so the whole 35,623-row corpus labels in
    **56 s** on 48 cores. The literature #6 names reaches for pseudo-labels, RL or a differentiable
    relaxation because exact programs are usually unavailable. Here they are not, so none of that
    machinery is bought, and the surface loss whose flat optimum sank the plane head is not used at
    all.
  * **Canonicalise by area, not by matching.** A set head has no natural slot order, so slots are
    sorted by owned area. That removes the permutation problem outright rather than paying for a
    Hungarian loss to tolerate it.
  * **`CutRoof` is withheld from the label vocabulary**, because its surface is a distance transform
    and no (type, plane) slot can carry it. Measured, not assumed: it was 13 of 1,246 operations,
    and dropping it moves the fit's median `extra` on the 411 from **0.0030 to 0.0035**.

Three facts measured before the run, which are what make the formulation worth a run at all:

    compiled label ceiling, on the 411   extra 0.0035   missing 0.0000   3D IoU 0.9965
    its form                             2.0 ops, planar_fraction 0.50 -- EQUAL to the real building
    robustness                           param noise of 0.10 *of the building's own height* still
                                         scores extra 0.0379, and randomising a QUARTER of the
                                         column assignments still scores 0.0325 -- both below the
                                         served per-column arm's 0.0603

That last one is the answer to the obvious objection to supervising parameters while scoring a
surface: this output space degrades gracefully, so the arm has to be roughly right, not exact.

THE BAR FOR #6, PRE-REGISTERED BEFORE THE FIRST RUN
---------------------------------------------------
Same 411 rows, same discipline. The reference numbers are #127's, re-read on those rows.

  PASS   BOTH halves of form at once -- median `dl_ops` <= 3.0 AND median `dl_planar_fraction`
         >= 0.40 -- AND median `extra` strictly below the served CE+median arm's **0.0603**.
         ⚠️ Both halves, because #127's plane head reached 3.0 ops with planar_fraction **0.00**:
         it swapped a mound for a terrace, and a single-number form metric would have shipped it.
  GUARD  collapse rate no worse than 1-NN's, and `vs_input` < 0.98 (#75).
  KILL   median `dl_planar_fraction` <= 0.20, the served per-column arm's own value. That is the
         terrace failure repeating in a third representation, and it answers #6 "not this way".

  for reference, on those same rows:  GT 2.0 ops / 0.50 planar
                                      CE+median (served) extra 0.0603 / 6.0 / 0.20
                                      planes K=6         extra 0.0772 / 3.0 / 0.00
                                      1-NN retrieval     extra 0.1031 / 2.0 / 0.17

    $P --objective program --tag heightmap_program --epochs 40 --montage 0

The first program run builds `outputs/height_map_generator/program_labels.npz` (~1 min, 48 cores).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.eval_massing_arms import (              # noqa: E402
    COLLAPSE_MISSING, RES, fp_iou, footprint_split, volume_split, vs_input,
)
from scripts.foundations.measure_scoring_optimum import (        # noqa: E402
    compare_to_envelope, transplant_height,
)
from scripts.foundations.recover_massing_programs import (       # noqa: E402
    CARVE_NEEDED, FLOOR_EPS, H5, K_OPS, SHIP714, SLOT_TYPES, fit_program_beam, height_field,
    occupancy, program_to_slots, render_iso,
)

LATENTS = REPO / "data/real_massing_v1/vecset_latents.h5"
WORK = REPO / "outputs/height_map_generator"
CACHE = WORK / "height_fields.npz"
PROGRAM_CACHE = WORK / "program_labels.npz"

# One class per voxel of carve depth. Measured over all 35,623 corpus rows: the deepest carve is
# 59 voxels of a 60-voxel extent, and no column is ever cut below 1, so depth lies in [0, 63] and 64
# covers the label range exactly with nothing clipped away at training time.
DEPTH_CLASSES = RES

# Which decode the joint slope term takes its hard forward at. NOT a flag: #127 measured that the
# mode under-carves and the arm is served at its posterior median, so the term must see the surface
# the arm is actually read at. Exposing it as a sweepable option would be selecting on the answer.
SLOPE_DECODE_QUANTILE = 0.5

# Buildings held back from training to select the checkpoint. Drawn from the TRAINING rows -- the
# pinned 714 are never seen, not even for early stopping.
VAL_BUILDINGS = 1000

# #6's slot vocabulary, re-exported under this module's name so the arm's own tests and the demo
# read one spelling. `Layer` is flat and `Ramp` is tilted, and which of the two a slot is is a
# DISCRETE prediction -- that is the single mechanical difference from the plane head below, whose
# slopes were free to decay to zero under L1 and did, from two separate initialisations.
PROGRAM_TYPES = SLOT_TYPES

# The three program-supervision terms are summed at equal weight. Fixed a priori and NOT swept:
# every term is already in commensurate units (two cross-entropies in nats, one L1 in units of the
# building's own height), and this ticket's record has two near-misses from reading a curve as a
# trend. A sweep here would be selecting on the answer.
PROGRAM_TERM_WEIGHTS = dict(assign=1.0, type=1.0, param=1.0)

# ⚠️ `recover_massing_programs.FLOOR_EPS` is 1e-9, which is the right snap for the float64 plane
# `linprog` returns. A slot's plane is stored in the label cache and predicted by the network in
# **float32**, where a value that is mathematically 30 arrives as 29.9999996 -- and `floor` then
# reads it as 29. Measured: 96 of 1,280 columns of a plain shed roof, every one of them a plane
# touching an integral target exactly, which is precisely the case the fitter's own snap exists to
# protect. float32 resolution at the top of a 64-grid is ~4e-6, so this is ~25x the noise and still
# four thousand times smaller than the half-voxel that would change a genuine geometric decision.
PLANE_FLOOR_EPS = 1e-4

N_REGIONS = 3          # source corpora: 0 NL / 1 DE / 2 JP, the `region` column of the latent cache
# footprint mask, conditioned extent, log height in metres, distance-to-edge, region one-hot.
# Pinned by `test_the_channel_count_matches_the_model_input` so the two cannot drift apart.
COND_CHANNELS = 4 + N_REGIONS


# ==================================================================================================
# the label, and the invariants of the output space
# ==================================================================================================

def carve_depth(top: np.ndarray, fp: np.ndarray, extent: int) -> np.ndarray:
    """Height map -> per-column carve depth below the blockout. 0 off the footprint."""
    m = np.asarray(fp, bool)
    return np.where(m, int(extent) - np.asarray(top, np.int32), 0).astype(np.int16)


def apply_depth(fp: np.ndarray, extent: int, depth: np.ndarray) -> np.ndarray:
    """Carve depth -> height map, clamped so the result is a valid solid whatever was predicted.

    The clamp is the whole structural argument of #127 and it is deliberately total: it accepts any
    array at all, including negative and out-of-range depths, and still returns a height map that is
    footprint-exact and at least one voxel deep on every footprint column, never taller than the
    blockout. A prediction can therefore be *wrong*, but never *invalid*.
    """
    m = np.asarray(fp, bool)
    e = int(extent)
    h = np.clip(e - np.asarray(depth, np.int32), 1, max(e, 1))
    return np.where(m, h, 0).astype(np.int16)


def height_split(pred: np.ndarray, target: np.ndarray) -> dict:
    """`volume_split` computed in column space, for two height maps sharing a base level.

    Exactly equal to voxelising both and calling `volume_split` -- a column is a solid run from the
    same `y0` in both, so the intersection is `min` per column -- and about 200x cheaper, which is
    what makes it affordable once per epoch on the validation split. `test_height_split_agrees_with_
    volume_split` is the pin; the scored arms still go through `volume_split` on real occupancy so
    the reported numbers stay on the same path as every other arm on this project's record.
    """
    p, t = np.asarray(pred, np.int64), np.asarray(target, np.int64)
    inter, av, gv = int(np.minimum(p, t).sum()), int(p.sum()), int(t.sum())
    union = av + gv - inter
    return dict(vol_iou=float(inter / union) if union else 0.0,
                missing=float((gv - inter) / gv) if gv else 0.0,
                extra=float((av - inter) / gv) if gv else 0.0)


def roof_description_length(surface: np.ndarray, fp: np.ndarray, y0: int, extent: int,
                            max_ops: int = 16, allowance: float = CARVE_NEEDED) -> dict:
    """🔑 The form metric. **How many architectural operations explain this roof?**

    Three amplitude statistics failed at this (`roof_shape_stats`), because GT is itself terraced at
    64^3 and no measure of step size can tell a discretised plane from a mound. What separates them
    is not amplitude, it is *organisation*: a real roof is a handful of planes meeting at ridges, and
    a mound is a continuum of orientations. So the question is asked in the project's own vocabulary
    -- #10's `Layer` / `Ramp` / `CutRoof` fitter is run on the arm's OWN surface, and what is
    reported is the **description length**: the number of operations needed to explain it.

    Validated on shapes whose answer is known by construction (`test_roof_description_length`):

        flat roof              1 op    Layer
        shed (one plane)       1 op    Ramp
        gable (two planes)     2 ops   CutRoof > Ramp
        hip (four planes)      4 ops
        two-step setback       2 ops   Layer > Layer
        a dome                 9 ops   and mostly Layers -- contour terraces, not planes
        noise                 16+ ops  and still not explained

    🔑 The **operation mix** is as diagnostic as the count. Architecture spends its budget on `Ramp`
    and `CutRoof`, which are planes; a mound cannot be explained by planes, so the fitter falls back
    to stacking flat `Layer`s, which is exactly the concentric contour banding the montages show.

    ⚠️ This metric is **not carve-aware, by design**. The footprint envelope is one flat plane and
    scores 1 op, which is *correct* -- the envelope genuinely is planar. Form is a separate axis from
    surplus, and `extra` / `missing` / `vs_input` already police whether the arm acted. Read the two
    together and never this one alone.
    """
    from scripts.foundations.recover_massing_programs import fit_program

    m = np.asarray(fp, bool)
    surf = np.asarray(surface, np.int16)
    prog, fitted = fit_program(m, int(y0), int(y0) + int(extent) - 1, surf, max_ops, allowance)
    vox = int(surf[m].sum())
    residual = float((fitted[m] - surf[m]).sum() / vox) if vox else 0.0
    mix = [o["op"] for o in prog]
    planar = sum(1 for o in mix if o in ("Ramp", "CutRoof"))
    return dict(ops=len(prog), residual=residual, explained=bool(residual <= allowance),
                planar_ops=planar,
                planar_fraction=float(planar / len(mix)) if mix else 0.0)


def roof_shape_stats(h: np.ndarray, fp: np.ndarray) -> dict:
    """Three attempts at a scalar for "does this roof look like a building", and all three fail.

    ⚠️ Recorded as a **negative result**, not as a scorecard column that works. The montages show a
    clear difference -- real roofs and the retrieved ones are flat planes meeting at ridges, while
    every trained arm returns a rounded mound with concentric contours -- and none of these
    statistics separates them. Measured on the carve-needing 411:

        arm            relief  curvature  speckle      the eye says
        gt               0.46      0.634    0.000      planes and ridges
        nn_retrieval     0.32      0.454    0.000      planes and ridges (it copies one)
        heightmap_ce     0.47      0.778    0.000      a mound, plus visible speckle
        ..._ce_median    0.40      0.509    0.000      a mound
        heightmap_mse    0.28      0.492    0.000      a mound

    `relief` ranks the worst-looking arm closest to GT and `curvature` ranks two of the mounds
    *smoother* than a real building, so both order the arms nearly opposite to the eye. The cause is
    that **GT is itself terraced at 64^3**: a pitched roof discretises to a staircase, so an
    amplitude statistic cannot tell a discretised plane from a mound. What differs is the
    *organisation* of the steps -- parallel runs against closed contours -- which is a directional
    property none of these three measures.

    This is the same wall map #34 hit ("roughness is prior-side; 2 scalar metrics failed") and #71
    ("ribbing is not melt"). They are kept, computed and published so the attempt is on the record
    and re-checkable, and the visual criterion stays the one that decides.

        relief     mean |height step| between adjacent footprint columns
        curvature  mean |second difference| along each axis; 0 on any plane at any slope
        speckle    fraction of interior columns that are a strict local extremum over 4 neighbours
    """
    a, m = np.asarray(h, np.int32), np.asarray(fp, bool)
    step = np.concatenate([np.abs(a[:, :-1] - a[:, 1:])[m[:, :-1] & m[:, 1:]],
                           np.abs(a[:-1, :] - a[1:, :])[m[:-1, :] & m[1:, :]]])
    curv = np.concatenate([
        np.abs(a[:, :-2] - 2 * a[:, 1:-1] + a[:, 2:])[m[:, :-2] & m[:, 1:-1] & m[:, 2:]],
        np.abs(a[:-2, :] - 2 * a[1:-1, :] + a[2:, :])[m[:-2, :] & m[1:-1, :] & m[2:, :]]])
    core = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    nb = np.stack([a[:-2, 1:-1], a[2:, 1:-1], a[1:-1, :-2], a[1:-1, 2:]])
    ext = ((a[1:-1, 1:-1] > nb).all(0) | (a[1:-1, 1:-1] < nb).all(0)) & core
    return dict(relief=float(step.mean()) if len(step) else 0.0,
                curvature=float(curv.mean()) if len(curv) else 0.0,
                speckle=float(ext.sum() / core.sum()) if core.any() else 0.0)


def envelope_depth(fp: np.ndarray) -> np.ndarray:
    """The do-nothing prediction: carve nothing, which `apply_depth` renders as the blockout."""
    return np.zeros(np.shape(fp), np.int16)


# ==================================================================================================
# the conditioning -- footprint, conditioned height, region. Nothing else may enter.
# ==================================================================================================

def condition_channels(fp: np.ndarray, extent: int, height_m: float, region: int) -> np.ndarray:
    """[C, Z, X] network input built from #127's conditioning ONLY.

    The signature is the leakage guard: there is no argument through which the target height field
    could reach the model, and `test_two_buildings_with_the_same_conditioning_get_identical_input`
    pins it. Two real buildings with the same footprint, height and region are genuinely
    indistinguishable inputs -- #126 measured that they still differ by a median 3D IoU of 0.886,
    which is the irreducible ambiguity this arm is working inside.

    The distance transform is a deterministic function of the footprint, not new information. It is
    supplied because #10 found the roof operations are functions of distance-to-edge (a hip erodes
    on all sides, a gable on one), and a small convolutional net would otherwise spend capacity
    rediscovering it.
    """
    m = np.asarray(fp, bool)
    edt = ndimage.distance_transform_edt(m).astype(np.float32) / 8.0
    ch = [m.astype(np.float32),
          np.full(m.shape, float(extent) / RES, np.float32),
          np.full(m.shape, float(np.log1p(max(height_m, 0.0))) / 4.0, np.float32),
          np.clip(edt, 0.0, 4.0)]
    for r in range(N_REGIONS):
        ch.append(np.full(m.shape, 1.0 if int(region) == r else 0.0, np.float32))
    return np.stack(ch).astype(np.float32)


def decode_logits(logits: np.ndarray, fp: np.ndarray, extent: int,
                  quantile: float | None = None) -> np.ndarray:
    """[K, Z, X] logits -> height map. **Argmax by default**, never by expectation.

    Taking the mean of the predicted distribution would reintroduce at decode time exactly the
    regression-to-the-mean the classification objective exists to avoid: a column whose posterior is
    split between "flat at full height" and "cut to the eaves" has a mean at neither. Argmax is what
    the pre-registered arm decodes with.

    `quantile` decodes the ordinal posterior's q-quantile instead -- the smallest depth whose
    cumulative probability reaches q. It exists because depth is **ordinal**, and the mode of an
    ordinal posterior is a biased estimator of it when one class dominates: with 54% of columns
    carrying depth 0, a column whose posterior is genuinely spread over 0..12 can have its mode at 0
    while its median is at 6. The quantile is fixed a priori at 0.5 by decision theory (the Bayes
    act under absolute error), NOT fitted -- and it is reported as a decode ablation beside the
    pre-registered arm, never in place of it.
    """
    z = np.asarray(logits, np.float64)
    if quantile is None:
        return apply_depth(fp, extent, np.argmax(z, axis=0).astype(np.int16))
    p = np.exp(z - z.max(axis=0, keepdims=True))
    cdf = np.cumsum(p / p.sum(axis=0, keepdims=True), axis=0)
    return apply_depth(fp, extent, np.argmax(cdf >= quantile, axis=0).astype(np.int16))


# ==================================================================================================
# the zero-training baselines #127 names
# ==================================================================================================

def mean_relative_depth(depths: np.ndarray, fps: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """The corpus's mean roof, per grid cell, as a fraction of the building's own height.

    This is the *unconditional* conditional-mean -- the arm #127's design note warns an MSE
    objective converges to. Relative rather than absolute because the corpus normalises each
    building into the grid: averaging voxel depths across a 6-voxel and a 60-voxel building would
    measure the height distribution, not the roof.

    Cells no footprint covers get 0 rather than NaN, so the profile is defined everywhere.
    """
    f = np.asarray(fps, bool)
    rel = np.where(f, np.asarray(depths, np.float32) /
                   np.maximum(np.asarray(extents, np.float32), 1)[:, None, None], 0.0)
    cover = f.sum(0).astype(np.float32)
    return np.divide(rel.sum(0), cover, out=np.zeros(rel.shape[1:], np.float32), where=cover > 0)


def mean_roof_height(profile: np.ndarray, fp: np.ndarray, extent: int) -> np.ndarray:
    """The mean profile rendered on this footprint at this conditioned height."""
    return apply_depth(fp, extent, np.rint(np.asarray(profile, np.float32) * int(extent)))


def retrieve_nn(query_fps: np.ndarray, bank_fps: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Index into `bank_fps` of the footprint-IoU-nearest bank row, for each query.

    Hyper-parameter free on purpose. The footprint is the shape half of the conditioning, and the
    height half is supplied exactly by `transplant_height`'s rescale, so a distance that mixed the
    two would need a weight -- and a *baseline* with a tuned weight is not a baseline. The bank is
    built from training rows only, so a held-out building can never retrieve itself.
    """
    q = np.asarray(query_fps, bool).reshape(len(query_fps), -1).astype(np.float32)
    b = np.asarray(bank_fps, bool).reshape(len(bank_fps), -1).astype(np.float32)
    qa, ba = q.sum(1), b.sum(1)
    out = np.zeros(len(q), np.int64)
    for s in range(0, len(q), chunk):
        inter = q[s:s + chunk] @ b.T
        union = qa[s:s + chunk, None] + ba[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        out[s:s + chunk] = np.argmax(iou, axis=1)
    return out


# ==================================================================================================
# the corpus as height fields, cached once
# ==================================================================================================

def build_cache(path: Path = CACHE, force: bool = False) -> dict:
    """Every corpus row as (footprint, base level, extent, target height map) + its conditioning.

    Keyed by the **latent cache**'s rows, because that file carries `held_out` -- the one split all
    of this project's arms have been scored against. Reading the 64^3 SDFs once and keeping only the
    height field turns 37 GB into 165 MB, which is the whole reason this task trains in minutes.
    """
    import h5py

    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    with h5py.File(LATENTS, "r") as f:
        rows = f["row"][:].astype(np.int32)
        held = (f["held_out"][:] == 1).astype(np.uint8)
        region = f["region"][:].astype(np.int8)
        height_m = f["height_m"][:].astype(np.float32)
    n = len(rows)
    fps = np.zeros((n, RES, RES), np.uint8)
    targets = np.zeros((n, RES, RES), np.uint8)
    y0s = np.zeros(n, np.int16)
    extents = np.zeros(n, np.int16)
    ok = np.zeros(n, np.uint8)
    t0 = time.time()
    with h5py.File(H5, "r") as g:
        for k, b in enumerate(rows):
            gt = np.asarray(g["sdf"][int(b)], np.float32) <= 0
            fp = np.asarray(g["footprint"][int(b)]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, target = hf
            fps[k] = fp
            targets[k] = np.clip(target, 0, 255)
            y0s[k], extents[k], ok[k] = y0, y1 - y0 + 1, 1
            if (k + 1) % 5000 == 0:
                print(f"  [cache] {k+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    out = dict(row=rows, held=held, region=region, height_m=height_m,
               fp=fps, target=targets, y0=y0s, extent=extents, ok=ok)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[cache] {path}  n={int(ok.sum())}/{n}  {time.time()-t0:.0f}s", flush=True)
    return out


# ==================================================================================================
# the model
# ==================================================================================================

OBJECTIVES = ("ce", "mse", "quantile", "planes", "program")


def head_channels(objective: str) -> int:
    """`ce` predicts a distribution over depths; the regressions predict one number per column."""
    return DEPTH_CLASSES if objective == "ce" else 1


def make_model(objective: str, width: int, k_planes: int):
    """The one place an objective chooses an architecture."""
    if objective == "program":
        return build_program_model(K_OPS, width)
    if objective == "planes":
        return build_plane_model(k_planes, width)
    return build_model(head_channels(objective), width)


def forward_heights(model, x, ext, objective: str):
    """Any objective -> a height map in voxels, differentiably. `ce` is excluded: it predicts a
    distribution, and collapsing it to a height before the loss is exactly the mistake this ticket
    found (its loss must see the classes, not their summary)."""
    out = model(x)
    if objective == "planes":
        return compose_planes(out[0], out[1], ext)
    return (1.0 - out[:, 0]) * ext[:, None, None]      # relative depth -> height, in voxels


def per_column_loss(out, y, ext, objective: str, quantile: float):
    """The training loss for one objective, per footprint column, in **voxel units**.

    All three live here rather than as an if/else at each call site, because they are read against
    each other: this ticket's result is that the three differ in *which statistic of the posterior*
    they target, and that is only legible with them side by side.

        ce        cross-entropy over 64 depth classes. Bayes act: the **mode**.
        mse       squared error on relative depth, rescaled to voxels. Bayes act: the **mean**.
        quantile  the pinball loss at `quantile`. Bayes act: that **quantile** -- at q=0.5, the
                  **median**, which is what #127 found the CE arm had to be decoded at anyway.

    🔑 At q=0.5 the pinball loss is L1 up to a factor of 2, so this arm is median regression and
    nothing more exotic. The point is not the loss's novelty, it is that the objective and the
    decode finally name the same statistic: the CE arm was trained for its mode and read at its
    median, and that mismatch is worth `extra` 0.1178 against 0.0603.
    """
    import torch
    import torch.nn.functional as F

    if objective == "ce":
        return F.cross_entropy(out, y.clamp(0, DEPTH_CLASSES - 1), reduction="none")
    if objective == "planes":
        # `out` is already a composed height map; the target height is extent - depth
        err = out - (ext[:, None, None] - y.float())
        return torch.maximum(quantile * -err, (quantile - 1.0) * -err)
    err = (out[:, 0] - y.float() / ext[:, None, None]) * ext[:, None, None]   # voxels, signed
    if objective == "mse":
        return err ** 2
    return torch.maximum(quantile * -err, (quantile - 1.0) * -err)


def differentiable_depth(out, ext, objective: str, quantile: float | None):
    """Network output -> per-column carve depth in voxels, differentiably, with a HARD forward.

    The slope term needs a *height* to take differences of, and a cross-entropy head predicts a
    distribution. Collapsing it with the softmax expectation would measure the slope of a BLENDED
    field, and this ticket's own finding (`compose_planes`) is that a smooth blend of surfaces is a
    mound -- the exact failure being fixed. So the forward pass takes the arm's real decode, and the
    gradient flows back through the soft probabilities: the same straight-through the plane head
    already uses, for the same reason.

    `quantile` picks which decode: `None` is the mode (the pre-registered decode) and 0.5 the
    median (the decode the arm is actually served at). The regressions have no distribution to
    collapse, so their own predicted depth passes straight through.

    ⚠️ The depth is returned UNCLAMPED, so it is `apply_depth`'s input rather than its output. On a
    column the model would carve away entirely the clamp returns a flat one-voxel slab, and a term
    reading that would have no gradient exactly where the prediction is worst.
    """
    import torch

    if objective == "planes":
        return ext[:, None, None] - out                     # `out` is already a height map
    if objective != "ce":
        return out[:, 0] * ext[:, None, None]               # relative depth -> voxels
    p = torch.softmax(out, dim=1)
    levels = torch.arange(DEPTH_CLASSES, device=out.device, dtype=p.dtype).view(1, -1, 1, 1)
    with torch.no_grad():
        # the cumulative sum in float64, because `decode_logits` serves this same posterior in
        # float64: a column whose cdf lands on the quantile picks a different class in the two
        # precisions, and then the term would be shaping a surface the arm is never read at. No
        # gradient passes through the index, so the cast costs a temporary and nothing else.
        idx = (p.argmax(1, keepdim=True) if quantile is None else
               (torch.cumsum(p.double(), 1) >= quantile).float().argmax(1, keepdim=True))
    hard = torch.zeros_like(p).scatter_(1, idx, 1.0)
    # grouped so the straight-through residual is EXACTLY zero in the forward pass: `hard + p - p`
    # left to right rounds through an intermediate and returns 5.9999995 where the decode says 6,
    # and this value is compared against `decode_logits` by test, not just used as a gradient path
    return ((hard + (p - p.detach())) * levels).sum(1)


def slope_loss(depth, y, mask):
    """🔑 The joint term: L1 between the prediction's first differences and GT's, per column PAIR.

    Every objective in `per_column_loss` scores each of the 4,096 plan columns on its own, so the
    ridge line -- a property of a *run* of columns -- is not in any of them, and a mound and a hip
    roof that remove the same volume cost the same. This term is the quantity the normal map draws,
    moved from the picture into the objective: a pitched plane is a constant step along a run and a
    hard break at the ridge, a mound is a step that drifts everywhere.

    Two properties make it an addition to the per-column loss rather than a replacement:

      * it is **blind to a constant offset**, so it says nothing about how deep to carve -- that
        stays cross-entropy's job -- and only about how the carve is arranged;
      * it matches GT's steps rather than minimising them, so a sharp ridge is **free**. A term that
        merely penalised roughness would prefer a rounded ridge, which is the mound again.

    ⚠️ It shapes the loss, not the architecture: at inference the head still emits one posterior per
    column independently. #127's diagnosis is that per-column *independence* is what produces a
    mound, so this is a probe of how much of that can be recovered by supervision alone, and a
    negative result is a real answer to that question.

    Only pairs with both columns inside the footprint are counted: off the footprint there is no
    surface, and the footprint wall is a vertical cliff that would otherwise dominate every edge.
    """
    d, t, m = depth.float(), y.float(), mask.bool()
    dz, tz, mz = d[:, 1:] - d[:, :-1], t[:, 1:] - t[:, :-1], m[:, 1:] & m[:, :-1]
    dx, tx, mx = d[:, :, 1:] - d[:, :, :-1], t[:, :, 1:] - t[:, :, :-1], m[:, :, 1:] & m[:, :, :-1]
    num = ((dz - tz).abs() * mz).sum() + ((dx - tx).abs() * mx).sum()
    return num / (mz.sum() + mx.sum()).clamp(min=1)


def decode_prediction(out_k: np.ndarray, fp: np.ndarray, extent: int, objective: str,
                      quantile: float | None) -> np.ndarray:
    """One network output -> one height map. The inverse of `per_column_loss`, kept beside it.

    ⚠️ `quantile` means two different things by objective and that is deliberate: for `ce` it picks
    which statistic to read OUT of a distribution the training never committed to, and for the
    regressions the statistic was fixed at training time and the argument is ignored. Reading a
    trained median at some other quantile is not possible, which is exactly the property that makes
    the `quantile` arm honest and the CE arm's post-hoc median a decode ablation.
    """
    if objective == "ce":
        return decode_logits(out_k, fp, extent, quantile)
    if objective == "program":
        # ⚠️ argmax on both discrete heads, never a blend: a softmax mixture of two slots is a
        # surface belonging to neither, which is #127's mound arriving by a third route. The one
        # place the network's scale-free plane is converted back to the fitter's voxel convention.
        a, t, p = out_k
        return compile_program(np.argmax(a, axis=0).astype(np.uint8),
                               np.argmax(t, axis=-1).astype(np.int8),
                               np.stack([plane_to_voxel(p[k], extent) for k in range(len(p))]),
                               fp, extent)
    if objective == "planes":
        return apply_depth(fp, extent, extent - np.rint(out_k))     # out_k is a height map
    return apply_depth(fp, extent, np.rint(out_k[0] * extent))


def build_model(out_channels: int, width: int = 64):
    """A small U-Net over the 64x64 plan. ~4M parameters against A2's 49M and map-24's 947M.

    Depth is chosen so the bottleneck is 8x8 -- one cell there sees an eighth of the plan, which is
    the scale a setback or a ridge line lives at. Nothing here is novel and nothing needs to be:
    #127 is a question about the output space, so the network is the cheapest thing that can answer
    it, and a bigger one would confound the answer.
    """
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(COND_CHANNELS, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.bot = block(4 * w, 4 * w)
            self.d3, self.d2, self.d1 = block(8 * w, 2 * w), block(4 * w, w), block(2 * w, w)
            self.head = nn.Conv2d(w, out_channels, 1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

        def forward(self, x):
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            s3 = self.e3(self.pool(s2))
            b = self.bot(self.pool(s3))
            x = self.d3(torch.cat([self.up(b), s3], 1))
            x = self.d2(torch.cat([self.up(x), s2], 1))
            x = self.d1(torch.cat([self.up(x), s1], 1))
            return self.head(x)

    return UNet()


# ==================================================================================================
# the planar head -- #127's form gap attacked in the representation, not in the loss
# ==================================================================================================

def compose_planes(logits, params, extent, hard: bool = True):
    """K planes plus a per-column assignment -> one height map, **piecewise-planar by construction**.

    🔑 The design move that already worked twice on this ticket: put the invariant in the
    representation rather than in the loss. A clamped height map made *validity* free -- no floating
    voxels are representable. This makes *planarity* free: the output is K planes and an assignment,
    so its description length is at most K by construction, and a mound is not representable at all.

    Why an assignment and not just `min` over the planes. A gable IS the min of two opposing planes,
    and so is a hip -- but a **setback** is not: two flat roofs at different heights over different
    parts of the plan have a min that is just the lower one everywhere. #10 measured `Layer` at
    **75.4%** of all removed volume, so setbacks are the majority of the corpus and a min-only
    composition would be unable to express most of it. The assignment is what makes each operation
    a *region*, which is exactly what `Layer` and `Ramp` are.

    ⚠️ Hard assignment forward, soft gradient backward (straight-through). A softmax BLEND of planes
    is smooth, and a smooth blend of planes is a mound -- the exact failure being fixed. So the
    forward pass must be hard even though that is what makes the gradient awkward.
    """
    import torch

    b, k, res, _ = logits.shape
    zz = torch.linspace(-0.5, 0.5, res, device=logits.device).view(1, 1, res, 1)
    xx = torch.linspace(-0.5, 0.5, res, device=logits.device).view(1, 1, 1, res)
    a, bz, cx = params[..., 0:1, None], params[..., 1:2, None], params[..., 2:3, None]
    planes = (a + bz * zz + cx * xx) * extent.view(b, 1, 1, 1)          # [B,K,res,res], in voxels
    soft = torch.softmax(logits, dim=1)
    if not hard:
        return (soft * planes).sum(1)
    onehot = torch.zeros_like(soft).scatter_(1, soft.argmax(1, keepdim=True), 1.0)
    w = onehot + soft - soft.detach()                                   # straight-through
    return (w * planes).sum(1)


def build_plane_model(k_planes: int, width: int = 64):
    """The same U-Net trunk, with two heads: a per-column assignment and K global plane parameters.

    The planes are **global per building** and the assignment is **spatial**, which is the split the
    vocabulary already has: a `Ramp` is one plane over one region. Pooling the bottleneck is what
    makes a plane a property of the whole building rather than of a neighbourhood, so a ridge line
    stays straight across the plan instead of drifting -- the per-column independence that produced
    the mound is exactly what this removes.
    """
    import torch
    import torch.nn as nn

    trunk = build_model(width, width)          # reuse the tested U-Net; its head becomes features

    class PlaneNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.assign = nn.Conv2d(width, k_planes, 1)
            self.params = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                        nn.Linear(4 * width, k_planes * 3))
            # ⚠️ The initialisation is load-bearing, and the first version of it was wrong. It set
            # every slope to exactly 0 and crushed the head's weights by 100x, so the planes began
            # flat and STAYED flat: measured after 40 epochs, a plane tilted a median of 0.21 voxels
            # across the whole plan. The model became six horizontal terraces -- a ziggurat, which is
            # #10's own name for this failure -- and its `planar_fraction` fell to 0.00 while the
            # per-column model managed 0.20. Planarity was free; SLOPE was not, because a flat region
            # is a strong local optimum under L1 and the straight-through gradient never escaped it.
            #
            # So the planes now start DIVERSE in slope as well as in height: half flat, half tilted
            # by half an extent across the plan in evenly spread directions. Buildings sit at
            # arbitrary grid rotations (#10), so the directions must cover the circle rather than the
            # axes, and the corpus is 54% flat columns, so the flat half is not optional either.
            with torch.no_grad():
                self.params[-1].weight.mul_(0.1)
                bias = torch.zeros(k_planes, 3)
                bias[:, 0] = torch.linspace(0.55, 1.0, k_planes)
                tilted = k_planes // 2
                ang = torch.linspace(0, float(np.pi), tilted + 1)[:tilted]
                bias[k_planes - tilted:, 1] = 0.5 * torch.cos(ang)
                bias[k_planes - tilted:, 2] = 0.5 * torch.sin(ang)
                self.params[-1].bias.copy_(bias.reshape(-1))
            self.k = k_planes

        def forward(self, x):
            f = self.trunk(x)
            p = self.params(f.mean(dim=(2, 3))).view(-1, self.k, 3)
            return self.assign(f), p

    return PlaneNet()


# ==================================================================================================
# #6 -- the program arm. The form gap attacked in the OUTPUT VOCABULARY, not the loss and not a
# soft composition of planes.
# ==================================================================================================

def plane_to_normalised(plane, extent) -> np.ndarray:
    """The fitter's voxel plane `a + b*x + c*z` -> the network's scale-free `(A, Bz, Cx)`.

    The network predicts a roof in units of the building's own height, on plan coordinates running
    -0.5..0.5 -- the convention `compose_planes` already uses, so the two heads stay readable
    against each other. It matters for the same reason `mean_relative_depth` is relative: a 6-voxel
    and a 60-voxel building must not be asked to regress the same number, or the parameter loss
    measures the corpus's height distribution instead of its roofs.

        height_voxels(z, x) = (A + Bz*zn + Cx*xn) * extent,   zn, xn = (i - (RES-1)/2) / (RES-1)
    """
    a, b, c = (float(v) for v in np.asarray(plane, np.float64))
    e = max(float(extent), 1.0)
    return np.array([(a + 0.5 * (RES - 1) * (b + c)) / e,
                     c * (RES - 1) / e,
                     b * (RES - 1) / e], np.float64)


def plane_to_voxel(params, extent) -> np.ndarray:
    """The inverse of `plane_to_normalised`, so the compiler only ever speaks one convention."""
    A, Bz, Cx = (float(v) for v in np.asarray(params, np.float64))
    e = float(extent)
    b, c = Cx * e / (RES - 1), Bz * e / (RES - 1)
    return np.array([A * e - 0.5 * (RES - 1) * (b + c), b, c], np.float64)


def compile_program(assign, types, planes, fp, extent) -> np.ndarray:
    """A predicted program -> one height map. The output space of #6's arm.

    🔑 **What this makes free, and it is a different thing from what #127's two representations
    made free.** The clamped height map made *validity* free; the plane head was meant to make
    *planarity* free and did, and it still terraced, because a plane whose slope may drift to zero
    is a flat region wearing a plane's name. Here the slot's **type** is a discrete prediction the
    compiler obeys: `Layer` ignores the slope it was handed and `Ramp` compiles the plane it was
    given, so "flat" and "pitched" are different answers rather than the same answer at different
    magnitudes. A slot cannot quietly become a terrace.

    And it is **joint** by construction, which is the property #127 measured to be missing from both
    ends: the ridge line falls out of one shared plane across a whole region, rather than out of
    4,096 columns that each summarised their own posterior and averaged a family of roofs into a
    mound.

    ⚠️ Total, exactly like `apply_depth`, and for the same reason. It accepts any assignment, any
    type and any plane at all -- including the wildly out-of-range params an untrained head emits --
    and still returns a footprint-exact height map with at least one voxel under every footprint
    column and nothing above the blockout. A run may then fail for a bad answer, never for an
    unrepresentable one.

    `planes` is in the fitter's voxel convention, so a program straight out of `program_to_slots`
    compiles without conversion and a *predicted* one is converted once, in `decode_prediction`.
    """
    m = np.asarray(fp, bool)
    e = int(extent)
    a = np.asarray(assign)
    t = np.asarray(types, np.int32)
    p = np.asarray(planes, np.float64)
    zz, xx = np.mgrid[0:RES, 0:RES]
    h = np.full(m.shape, float(e), np.float64)
    ramp = SLOT_TYPES.index("Ramp")
    for k in range(len(p)):
        sel = a == k
        if not sel.any():
            continue
        # a slot that is inactive (-1) or typed `Layer` is flat: its slope is not read at all
        surf = (p[k, 0] + p[k, 1] * xx + p[k, 2] * zz if t[k] == ramp
                else np.full(m.shape, p[k, 0], np.float64))
        h = np.where(sel, np.floor(surf + PLANE_FLOOR_EPS), h)
    return np.where(m, np.clip(h, 1, max(e, 1)), 0).astype(np.int16)


def program_loss(out, labels, mask):
    """🔑 #6's training strategy, in one function: supervise the **program**, never the surface.

    #127 established the trap this avoids, twice and from both directions. Supervision on the
    surface could not put planes in -- an L1 has a flat region as a strong local optimum, so the
    plane head's slopes collapsed to 0.25 voxels across a 40-voxel building from two different
    initialisations -- and no decode could take a roof out of a per-column posterior. So no term
    here reads the compiled height map at all. Each term sees a piece of the program:

        assign  cross-entropy per footprint column over the K slots plus the UNCARVED class. This
                is where the *regions* are learned, and it is a segmentation, not a height.
        type    cross-entropy per ACTIVE slot over (Layer, Ramp). The discrete flat-or-pitched
                decision that a straight-through slope could never make.
        param   L1 on the plane, in units of the building's own height, per active slot -- and on
                the OFFSET ONLY for a slot typed `Layer`, because a flat roof's slope is not a
                quantity the label has an opinion about and regressing it towards zero would spend
                capacity teaching the model something the compiler already ignores.

    ⚠️ Inactive slots contribute nothing to any term. A building the fitter explained in two
    operations must not be pushed to invent four; #10 measured a median of 4 and a mode of 4 at the
    budget, but 59 of the 411 carve-needing buildings need exactly one.
    """
    import torch
    import torch.nn.functional as F

    assign_logits, type_logits, params = out
    assign, types, planes = labels
    m = mask.bool()

    ce = F.cross_entropy(assign_logits, assign, reduction="none")
    l_assign = (ce * m).sum() / m.sum().clamp(min=1)

    active = types >= 0
    n_active = active.sum().clamp(min=1)
    l_type = (F.cross_entropy(type_logits[active], types[active], reduction="sum") / n_active
              if bool(active.any()) else assign_logits.sum() * 0.0)

    # a Layer's slope is not in the label, so it is not in the loss: weight the offset everywhere
    # and the two slope components only where the slot is a Ramp
    is_ramp = (types == SLOT_TYPES.index("Ramp")) & active
    w = torch.stack([active.float(), is_ramp.float(), is_ramp.float()], dim=-1)
    l_param = ((params - planes).abs() * w).sum() / w.sum().clamp(min=1)

    return (PROGRAM_TERM_WEIGHTS["assign"] * l_assign +
            PROGRAM_TERM_WEIGHTS["type"] * l_type +
            PROGRAM_TERM_WEIGHTS["param"] * l_param)


def build_program_model(k_ops: int, width: int = 64):
    """The same U-Net trunk, with an assignment head and a slot head. ~3.6M parameters.

    The split is the vocabulary's own: an operation is **one plane over one region**, so the plane
    is pooled to a property of the whole building and the region stays spatial. That is what keeps a
    ridge line straight across the plan instead of drifting -- the per-column independence #127
    diagnosed as the cause of the mound is exactly what pooling removes.

    The slot head emits type logits and plane parameters together, from the same pooled feature, so
    a slot's "am I flat" decision and its slope are read off one representation rather than two.
    """
    import torch
    import torch.nn as nn

    trunk = build_model(width, width)             # the tested U-Net; its head becomes features
    n_type = len(SLOT_TYPES)

    class ProgramNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.k = k_ops
            self.assign = nn.Conv2d(width, k_ops + 1, 1)
            self.slots = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                       nn.Linear(4 * width, k_ops * (n_type + 3)))
            # ⚠️ #127's plane head recorded that this initialisation is load-bearing: starting every
            # plane flat left them flat after 40 epochs. The labels here supervise the slope
            # directly, so the failure cannot repeat for that reason -- but the slots are still
            # canonicalised by AREA, so slot 0 sees mostly large flat setbacks and the later slots
            # the small pitched pieces, and starting them identical wastes the early epochs
            # separating them. Heights spread over the top of the building, slopes spread around
            # the circle because buildings sit at arbitrary grid rotations (#10).
            with torch.no_grad():
                self.slots[-1].weight.mul_(0.1)
                bias = torch.zeros(k_ops, n_type + 3)
                bias[:, n_type] = torch.linspace(0.9, 0.5, k_ops)
                ang = torch.linspace(0, float(np.pi), k_ops + 1)[:k_ops]
                bias[:, n_type + 1] = 0.25 * torch.cos(ang)
                bias[:, n_type + 2] = 0.25 * torch.sin(ang)
                self.slots[-1].bias.copy_(bias.reshape(-1))

        def forward(self, x):
            f = self.trunk(x)
            s = self.slots(f.mean(dim=(2, 3))).view(-1, self.k, n_type + 3)
            return self.assign(f), s[..., :n_type], s[..., n_type:]

    return ProgramNet()


def _d4_program(assign, types, planes, k: int, flip: bool, k_ops: int = K_OPS):
    """One plan symmetry applied to a PROGRAM, so #6's arm keeps the 8x augmentation every other
    arm on this ticket trains with.

    The assignment is an image and rotates with the footprint. A plane is not: `height = a + b*x +
    c*z` has to be re-expressed in the rotated frame, and getting that wrong would silently train
    the arm on roofs tilted the wrong way -- an error no shape test on the footprint would catch,
    which is why `test_the_augmented_program_compiles_to_the_augmented_surface` compares the
    compiled surfaces rather than the parameters.

    `np.rot90(a)[z, x] = a[x, n-1-z]`, so height'(z, x) = a + b*(n-1-z) + c*x, and the flip
    `a[:, ::-1]` sends x -> n-1-x. Both are applied in the same order as `_d4`.
    """
    n = RES - 1
    a, b, c = planes[:, 0].copy(), planes[:, 1].copy(), planes[:, 2].copy()
    for _ in range(k % 4):
        a, b, c = a + b * n, c.copy(), -b
    if flip:
        a, b = a + b * n, -b
    out = np.stack([a, b, c], axis=1).astype(np.float32)
    ass = np.rot90(assign, k)
    if flip:
        ass = ass[:, ::-1]
    return np.ascontiguousarray(ass), types.copy(), out


def build_program_cache(cache: dict, path: Path = PROGRAM_CACHE, force: bool = False,
                        workers: int = 0, beam: int = 12, branch: int = 6) -> dict:
    """#10's fitter run over the whole corpus, decomposed into slots. #6's supervision.

    🔑 This is why #6 is a supervised-learning ticket and not a program-induction one. The literature
    #6 names reaches for pseudo-labels, RL or a differentiable relaxation precisely because exact
    programs are usually unavailable -- and here they are not: the fitter is deterministic, sees GT,
    reaches a median `extra` of 0.003, and costs **0.2 s per building**, so the entire 35,776-row
    corpus labels in under three minutes on this machine's 64 cores. Measured before the arm was
    designed, and it is the fact that chose the formulation.

    ⚠️ Fitted with `CutRoof` withheld, so every operation is a plane and `program_to_slots` is
    lossless. `CutRoof` was 13 of 1,246 operations (1.0%) in the committed 714-building recovery,
    and `--ops_allowed` on the recovery script measures what withholding it costs rather than
    assuming it away.
    """
    import multiprocessing as mp

    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    idx = np.nonzero(cache["ok"] > 0)[0]
    jobs = [(cache["fp"][i] > 0, int(cache["extent"][i]),
             cache["target"][i].astype(np.int16), beam, branch) for i in idx]
    n = len(cache["ok"])
    assign = np.full((n, RES, RES), K_OPS, np.uint8)
    types = np.full((n, K_OPS), -1, np.int8)
    planes = np.zeros((n, K_OPS, 3), np.float32)
    residual = np.zeros(n, np.float32)
    t0 = time.time()
    workers = workers or min(mp.cpu_count(), 48)
    with mp.Pool(workers) as pool:
        for done, (i, out) in enumerate(zip(idx, pool.imap(_fit_one_slots, jobs, chunksize=8))):
            assign[i], types[i], planes[i], residual[i] = out
            if (done + 1) % 5000 == 0:
                print(f"  [program] {done+1}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)
    out = dict(assign=assign, types=types, planes=planes, residual=residual,
               ok=cache["ok"], row=cache["row"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[program] {path}  n={len(idx)}  median fitted extra "
          f"{float(np.median(residual[idx])):.4f}  {time.time()-t0:.0f}s", flush=True)
    return out


def _fit_one_slots(job):
    """One building's program labels. Module level so `multiprocessing` can pickle it."""
    fp, extent, target, beam, branch = job
    ops, fitted = fit_program_beam(fp, 0, extent - 1, target, max_ops=K_OPS,
                                   beam=beam, branch=branch, ops_allowed=SLOT_TYPES)
    assign, types, planes = program_to_slots(fp, extent, ops)
    vox = int(target[fp].sum())
    residual = float((fitted[fp] - target[fp]).sum() / vox) if vox else 0.0
    return assign, types, planes, residual


def _d4(fp, target, k: int, flip: bool):
    """One of the 8 plan symmetries, applied to footprint and label together.

    Buildings sit at arbitrary grid rotations already (#10: an axis-aligned ramp could not fix the
    shed-roof residual), so the symmetry group is a property of the corpus rather than an assumption
    imposed on it. The conditioning is rebuilt from the rotated footprint, so nothing can desync.
    """
    fp, target = np.rot90(fp, k), np.rot90(target, k)
    if flip:
        fp, target = fp[:, ::-1], target[:, ::-1]
    return np.ascontiguousarray(fp), np.ascontiguousarray(target)


class HeightFieldSet:
    """Conditioning + label for one split, materialised on demand so augmentation stays honest.

    `program` adds #6's slot labels alongside the per-column depth rather than in place of it: the
    depth label is still what the validation geometry is measured against, so the program arm is
    selected by exactly the rule every other arm on this ticket was.
    """

    def __init__(self, cache: dict, idx: np.ndarray, augment: bool, seed: int = 0,
                 program: dict | None = None):
        self.fp = cache["fp"][idx] > 0
        self.target = cache["target"][idx].astype(np.int16)
        self.extent = cache["extent"][idx].astype(np.int32)
        self.height_m = cache["height_m"][idx]
        self.region = cache["region"][idx].astype(np.int32)
        self.augment, self.rng = augment, np.random.default_rng(seed)
        self.program = None
        if program is not None:
            self.program = dict(assign=program["assign"][idx], types=program["types"][idx],
                                planes=program["planes"][idx])

    def __len__(self):
        return len(self.fp)

    def batch(self, sel: np.ndarray):
        xs, ys, pa, pt, pp = [], [], [], [], []
        for i in sel:
            fp, target = self.fp[i], self.target[i]
            k, flip = (int(self.rng.integers(4)), bool(self.rng.integers(2))) if self.augment \
                else (0, False)
            if self.augment:
                fp, target = _d4(fp, target, k, flip)
            xs.append(condition_channels(fp, int(self.extent[i]), float(self.height_m[i]),
                                         int(self.region[i])))
            ys.append(carve_depth(target, fp, int(self.extent[i])))
            if self.program is not None:
                # ⚠️ the SAME symmetry as the footprint above, drawn once: a program augmented
                # independently of its own plan would supervise a roof on the wrong building
                a, t, p = _d4_program(self.program["assign"][i], self.program["types"][i],
                                      self.program["planes"][i], k, flip)
                pa.append(a)
                pt.append(t)
                pp.append(np.stack([plane_to_normalised(p[j], int(self.extent[i]))
                                    for j in range(len(p))]))
        prog = (np.stack(pa).astype(np.int64), np.stack(pt).astype(np.int64),
                np.stack(pp).astype(np.float32)) if self.program is not None else None
        return (np.stack(xs), np.stack(ys).astype(np.int64),
                self.extent[sel].astype(np.float32), prog)


def train(cache: dict, args) -> Path:
    """Train one arm and return its selected checkpoint.

    ⚠️ Selection is on a validation split drawn from the TRAINING rows. The pinned 714 are not read
    here at all. This project's record has two near-misses from reading a training curve as a trend
    (#80, twice), so the checkpoint is chosen by a held-in number and the whole curve is written to
    the artifact rather than summarised.

    🔑 It is chosen on the **geometry**, not on the loss. #75/#76 measured that neither the training
    loss nor latent distance tracked the goal on this project, and #76 found latent distance was
    *wrong-signed* pooled across error families. A cross-entropy is a proxy; the height field it
    decodes to is the thing, and it costs one argmax per validation building per epoch to measure
    directly.

    ⚠️ The criterion is `missing + extra` -- the symmetric difference, normalised by GT volume --
    and NOT `extra` alone. Selecting on `extra` was tried first and is unsound in a way that only
    showed up on the MSE arm: an arm that carves the building away scores `extra` 0, so the rule
    picked that arm's **first epoch** (`extra` 0.039, `missing` 0.082) and would then have failed
    the collapse guard for a reason belonging to the selection rule rather than to the objective.
    The symmetric difference cannot be gamed from either end -- no-op and over-carve are both
    penalised -- and it needs no threshold. Validation loss is recorded for the curve and breaks
    ties.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pool = np.nonzero((cache["ok"] > 0) & (cache["held"] == 0))[0]
    perm = np.random.default_rng(args.seed).permutation(len(pool))
    val_idx, tr_idx = pool[perm[:VAL_BUILDINGS]], pool[perm[VAL_BUILDINGS:]]
    prog = (build_program_cache(cache, force=args.rebuild_program_cache)
            if args.objective == "program" else None)
    tr = HeightFieldSet(cache, tr_idx, augment=not args.no_aug, seed=args.seed, program=prog)
    va = HeightFieldSet(cache, val_idx, augment=False, program=prog)
    print(f"[train] {len(tr)} buildings, {len(va)} validation, objective={args.objective}, "
          f"device={dev}", flush=True)

    model = make_model(args.objective, args.width, args.k_planes).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(len(tr) // args.batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    print(f"[train] {n_par/1e6:.2f}M parameters, {steps} steps", flush=True)

    def loss_of(x, y, ext, prog_labels=None):
        m = x[:, 0] > 0                                   # footprint columns only
        if args.objective == "program":
            # 🔑 the program arm never sees its own compiled surface during training. No `slope_
            # weight` either: the joint structure is in the output space now, and #127 measured
            # that adding it to the loss buys description length without buying planes.
            return program_loss(model(x), prog_labels, m)
        out = (forward_heights(model, x, ext, args.objective) if args.objective == "planes"
               else model(x))
        per = per_column_loss(out, y, ext, args.objective, args.quantile)
        loss = (per * m).sum() / m.sum().clamp(min=1)
        if args.slope_weight:
            loss = loss + args.slope_weight * slope_loss(
                differentiable_depth(out, ext, args.objective, SLOPE_DECODE_QUANTILE), y, m)
        return loss

    def to_dev(b):
        x, y, e, p = b
        return (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
                torch.from_numpy(e).to(dev),
                tuple(torch.from_numpy(t).to(dev) for t in p) if p is not None else None)

    curve, best, best_path = [], (float("inf"), float("inf")), WORK / f"{args.tag}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)
    val_carve = np.array([
        height_split(apply_depth(va.fp[i], int(va.extent[i]), envelope_depth(va.fp[i])),
                     va.target[i])["extra"] >= CARVE_NEEDED for i in range(len(va))])
    print(f"[train] selecting on validation missing+extra over "
          f"{int(val_carve.sum())}/{len(va)} carve-needing validation buildings", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        run = 0.0
        for s in range(0, len(order) - args.batch + 1, args.batch):
            x, y, e, p = to_dev(tr.batch(order[s:s + args.batch]))
            loss = loss_of(x, y, e, p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
        run /= max(len(order) // args.batch, 1)
        model.eval()
        vl, ve, vm = _validate(model, va, val_carve, args.objective, args.quantile, dev)
        curve.append(dict(epoch=ep + 1, train=run, val=vl, val_extra=ve, val_missing=vm,
                          val_symmetric=ve + vm))
        mark = ""
        if (ve + vm, vl) < best:
            best, mark = (ve + vm, vl), "  <- best"
            torch.save(dict(state=model.state_dict(), objective=args.objective, width=args.width,
                            quantile=args.quantile, k_planes=args.k_planes,
                            slope_weight=args.slope_weight,
                            slope_decode_quantile=SLOPE_DECODE_QUANTILE,
                            epoch=ep + 1, val=vl, val_extra=ve, val_missing=vm,
                            val_symmetric=ve + vm, params=n_par), best_path)
        print(f"  epoch {ep+1:>3}/{args.epochs}  train {run:.4f}  val {vl:.4f}  "
              f"val extra {ve:.4f}  val miss {vm:.4f}  sym {ve+vm:.4f}  "
              f"{time.time()-t0:.0f}s{mark}", flush=True)
    json.dump(curve, open(WORK / f"{args.tag}_curve.json", "w"), indent=1)
    print(f"[train] best validation missing+extra {best[0]:.4f} (loss {best[1]:.4f}) -> "
          f"{best_path}", flush=True)
    return best_path


def _validate(model, va, carve_mask, objective: str, quantile: float, dev) -> tuple:
    """Validation loss AND the geometric quantity the ticket is judged on, on held-in buildings."""
    import torch

    # ⚠️ The CE arm is validated at its ARGMAX, which is what it is trained for. Validating it at
    # the median would select a checkpoint for a decode the training never committed to, and the
    # post-hoc median has to stay an ablation of a finished model rather than a training signal.
    decode_q = None if objective == "ce" else quantile
    losses, splits = [], []
    with torch.no_grad():
        for s in range(0, len(va), 128):
            sel = np.arange(s, min(s + 128, len(va)))
            x, y, e, p = va.batch(sel)
            xt, yt = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
            et = torch.from_numpy(e).to(dev)
            m = xt[:, 0] > 0
            if objective == "program":
                out = model(xt)
                losses.append(float(program_loss(
                    out, tuple(torch.from_numpy(t).to(dev) for t in p), m).detach()))
                o = [tuple(t[k].cpu().numpy() for t in out) for k in range(len(sel))]
            else:
                out = (forward_heights(model, xt, et, objective) if objective == "planes"
                       else model(xt))
                per = per_column_loss(out, yt, et, objective, quantile)
                losses.append(float(((per * m).sum() / m.sum().clamp(min=1)).detach()))
                o = out.cpu().numpy()
            for k, i in enumerate(sel):
                ext, fp = int(va.extent[i]), va.fp[i]
                splits.append(height_split(
                    decode_prediction(o[k], fp, ext, objective, decode_q), va.target[i]))
    carve = [d for d, m in zip(splits, carve_mask) if m]
    return (float(np.mean(losses)),
            float(np.median([d["extra"] for d in carve])) if carve else float("nan"),
            float(np.median([d["missing"] for d in carve])) if carve else float("nan"))


def predict(ckpt: Path, held: dict, batch: int = 64, cpu: bool = False,
            quantile: float | None = None):
    """Height maps for the pinned buildings from a trained checkpoint, and how it was selected.

    The provenance travels with the prediction rather than with the command line: a `--ckpt` rerun
    scores a file trained by some earlier invocation, and recording the flags of the *rerun* would
    put a number in the artifact that did not produce the checkpoint beside it.
    """
    import torch

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model = make_model(d["objective"], d["width"], d.get("k_planes", 6)).to(dev)
    model.load_state_dict(d["state"])
    model.eval()
    out = np.zeros((len(held["fp"]), RES, RES), np.int16)
    with torch.no_grad():
        for s in range(0, len(out), batch):
            sel = range(s, min(s + batch, len(out)))
            x = np.stack([condition_channels(held["fp"][i], int(held["extent"][i]),
                                             float(held["height_m"][i]), int(held["region"][i]))
                          for i in sel])
            xt = torch.from_numpy(x).to(dev)
            if d["objective"] == "planes":
                et = torch.tensor([float(held["extent"][i]) for i in sel], device=dev)
                y = forward_heights(model, xt, et, "planes").cpu().numpy()
            elif d["objective"] == "program":
                heads = model(xt)
                y = [tuple(t[k].cpu().numpy() for t in heads) for k in range(len(list(sel)))]
            else:
                y = model(xt).cpu().numpy()
            for k, i in enumerate(sel):
                out[i] = decode_prediction(y[k], held["fp"][i], int(held["extent"][i]),
                                           d["objective"], quantile)
    # the whole training curve travels into the artifact, not a summary of it: this project has
    # twice recommended stopping at a dip that recovered (#80), and a curve nobody can re-read is
    # how that happens a third time.
    curve = ckpt.with_name(ckpt.stem + "_curve.json")
    return out, dict(path=str(ckpt), objective=d["objective"], width=d["width"],
                     decode=("argmax" if d["objective"] == "ce" and quantile is None else
                             f"posterior q={quantile}" if d["objective"] == "ce" else
                             "compiled program (argmax slot, argmax type)"
                             if d["objective"] == "program" else
                             f"regression (trained at q={d.get('quantile')})"
                             if d["objective"] == "quantile" else "regression"),
                     trained_quantile=d.get("quantile"),
                     slope_weight=d.get("slope_weight", 0.0),
                     slope_decode_quantile=d.get("slope_decode_quantile"),
                     epoch=d.get("epoch"), val_loss=d.get("val"), val_extra=d.get("val_extra"),
                     val_missing=d.get("val_missing"), val_symmetric=d.get("val_symmetric"),
                     params=d.get("params"), selected_on="validation missing+extra",
                     curve=json.load(open(curve)) if curve.exists() else None)


# ==================================================================================================
# scoring
# ==================================================================================================

def score_arm(heights: np.ndarray, held: dict, form: bool = True) -> list:
    """One row of metrics per pinned building, in the order #126 decided they must be read."""
    rows = []
    for i in range(len(heights)):
        fp, y0, extent = held["fp"][i], int(held["y0"][i]), int(held["extent"][i])
        gt = occupancy(fp, y0, held["target"][i])
        bo = occupancy(fp, y0, apply_depth(fp, extent, envelope_depth(fp)))
        occ = occupancy(fp, y0, heights[i])
        r = dict(id=int(held["row"][i]),
                 # #127's actual question in plan view: of the footprint columns, what fraction did
                 # the arm cut at all, against the fraction GT cuts? `extra` says how much surplus
                 # is left; this says whether the arm ACTED, and on how much of the building.
                 carved_cols=float((heights[i][fp] < extent).mean()),
                 gt_carved_cols=float((held["target"][i][fp] < extent).mean()),
                 **{f"roof_{k}": v for k, v in roof_shape_stats(heights[i], fp).items()},
                 **{f"gt_roof_{k}": v for k, v in
                    roof_shape_stats(held["target"][i], fp).items()},
                 **({f"dl_{k}": v for k, v in roof_description_length(
                        heights[i], fp, y0, extent).items()} if form else {}),
                 **({f"gt_dl_{k}": v for k, v in roof_description_length(
                        held["target"][i], fp, y0, extent).items()} if form else {}))
        r.update(volume_split(occ, gt))
        r.update(footprint_split(occ, fp))
        r["fp_iou"] = fp_iou(occ, fp)
        r["vs_input"] = vs_input(occ, bo)
        r["blockout_extra"] = volume_split(bo, gt)["extra"]
        rows.append(r)
    return rows


def summarise(rows: list) -> dict:
    med = lambda k: float(np.median([r[k] for r in rows])) if rows else float("nan")
    return dict(n=len(rows), missing=med("missing"), extra=med("extra"),
                vs_input=med("vs_input"), carved_cols=med("carved_cols"),
                gt_carved_cols=med("gt_carved_cols"),
                **{k: med(k) for k in ("roof_relief", "roof_curvature", "roof_speckle",
                                       "gt_roof_relief", "gt_roof_curvature", "gt_roof_speckle")},
                **({k: med(k) for k in ("dl_ops", "dl_planar_fraction", "dl_residual",
                                        "gt_dl_ops", "gt_dl_planar_fraction")}
                   if rows and "dl_ops" in rows[0] else {}),
                **(dict(dl_explained=float(np.mean([r["dl_explained"] for r in rows])))
                   if rows and "dl_ops" in rows[0] else {}),
                collapse_rate=float(np.mean([r["missing"] >= COLLAPSE_MISSING for r in rows]))
                if rows else float("nan"),
                fp_iou=med("fp_iou"), spill=med("spill"), vol_iou=med("vol_iou"))


def verdict(arms: dict, pop: str) -> dict:
    """The pre-registered bar, evaluated mechanically so the write-up cannot soften it."""
    out = {}
    bo, nn = arms["blockout"][pop], arms["nn_retrieval"][pop]
    for name, a in arms.items():
        if name in ("blockout", "nn_retrieval"):
            continue
        s = a[pop]
        out[name] = dict(
            beats_1nn_extra=bool(s["extra"] < nn["extra"]),
            collapse_no_worse_than_1nn=bool(s["collapse_rate"] <= nn["collapse_rate"]),
            moved=bool(s["vs_input"] < 0.98),
            killed_identity=bool(s["extra"] >= bo["extra"]),
        )
        out[name]["pass"] = bool(out[name]["beats_1nn_extra"] and
                                 out[name]["collapse_no_worse_than_1nn"] and out[name]["moved"])
    return out


REFERENCE = {
    # arm -> (committed artifact, key under `per_building`). Quoted, never recomputed: these are
    # this project's record on the SAME pinned 714, and re-deriving them here would risk a second,
    # silently different number for an arm that already has one.
    "a2_s0.5 (shipped)": ("execution/artifacts/massing_arms_eval_ship714.json", "a2_s0.5"),
    "deployed_map24": ("execution/artifacts/massing_arms_eval_ship714.json", "deployed_map24"),
    "codec_ceiling": ("execution/artifacts/massing_arms_eval_ship714.json", "codec_ceiling"),
    "program K=16 (sees GT)": ("execution/artifacts/program_recovery_714.json", None),
}


def reference_arms(carve_ids: set) -> dict:
    """This project's arms of record, re-summarised on exactly the rows scored here.

    #126's rule, applied to the write-up as well as to the run: an arm quoted from one population
    beside an arm measured on another is how "19% surplus reduction" became 11.8% like-for-like on
    map #87. The medians are recomputed from the committed per-building rows, so the population is
    the same 411 buildings whatever those artifacts summarised themselves over.
    """
    out = {}
    for name, (path, key) in REFERENCE.items():
        f = REPO / path
        if not f.exists():
            continue
        doc = json.load(open(f))
        pb = doc["per_building"]
        pb = pb[key] if key else pb
        rows = [r for b, r in pb.items() if int(b) in carve_ids]
        if not rows:
            continue
        med = lambda k: float(np.median([r[k] for r in rows if k in r]))
        out[name] = dict(n=len(rows), missing=med("missing"), extra=med("extra"),
                         vs_input=med("vs_input") if "vs_input" in rows[0] else None,
                         collapse_rate=float(np.mean([r["missing"] >= COLLAPSE_MISSING
                                                      for r in rows])),
                         vol_iou=med("vol_iou"), source=path)
    return out


def montage(cases, out: Path, cell: int = 5) -> Path:
    """Real building beside every arm, as shaded massing. The human's criterion, not a number.

    #10 recorded three separate occasions where reading a picture corrected a conclusion the scalar
    metric supported, so the arms are rendered side by side on the same buildings rather than
    summarised.
    """
    from PIL import Image, ImageDraw

    names = list(cases[0]["arms"])
    tiles = [[render_iso(c["target"], c["fp"], cell)] +
             [render_iso(c["arms"][n], c["fp"], cell) for n in names] for c in cases]
    tw = max(t.width for row in tiles for t in row)
    th = max(t.height for row in tiles for t in row)
    head, pad, lab, cols = 26, 8, 34, len(names) + 1
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad,
                              head + len(tiles) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, title in enumerate(["REAL BUILDING"] + [n.upper() for n in names]):
        d.text((pad + j * (tw + pad), 8), title, fill=(0, 0, 0))
    for i, row in enumerate(tiles):
        y = head + i * (th + lab)
        for j, t in enumerate(row):
            sheet.paste(t, (pad + j * (tw + pad) + (tw - t.width) // 2, y + (th - t.height) // 2))
        c = cases[i]
        d.text((pad, y + th + 4), f"id {c['id']}   " + "   ".join(
            f"{n} extra {c['extra'][n]:.3f}" for n in names), fill=(40, 40, 40))
        d.line([(0, y + th + lab - 2), (sheet.width, y + th + lab - 2)], fill=(225, 225, 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


# ==================================================================================================
# plan-view maps -- the same surfaces the montage draws in 3D, read as height and as slope
# ==================================================================================================

LEGEND_W = 1000          # the map sheet's legends need this much width whatever the tiles need


def height_rgb(h: np.ndarray, fp: np.ndarray, extent: int, contour: int = 2,
               lo: int = 0) -> np.ndarray:
    """[Z, X, 3] plan view of a height map, coloured by level, with iso-contours every `contour`.

    The ramp is shared by every arm on the row and by the real building, so a colour means the same
    height across the row and the arms can be compared by eye. It spans `lo`..extent rather than
    0..extent: the deepest level any arm on that row reaches is usually well above the base, and
    stretching the ramp over the empty part of the range spends most of the colours on heights no
    arm predicted. `lo` is the deepest level anything on the row reaches, the real building
    included, so every arm is read against one scale and none is flattered by its own.

    Contours are drawn because the open question is **form**: a mound and a hip roof can carve the
    same volume and score the same `extra`, and closed concentric rings against a few straight bands
    separate them at a glance where shading does not.
    """
    from matplotlib import colormaps

    m = np.asarray(fp, bool)
    e = max(int(extent), 1)
    lvl = np.clip(np.asarray(h, np.int32), 0, e)
    t = (lvl - int(lo)) / max(e - int(lo), 1)
    rgb = (np.asarray(colormaps["turbo"](np.where(m, np.clip(t, 0.0, 1.0), 0.0)))[..., :3]
           * 255).astype(np.uint8)
    if contour:
        band = lvl // max(int(contour), 1)
        edge = np.zeros_like(m)
        edge[:, :-1] |= m[:, :-1] & m[:, 1:] & (band[:, :-1] != band[:, 1:])
        edge[:-1, :] |= m[:-1, :] & m[1:, :] & (band[:-1, :] != band[1:, :])
        rgb[edge] = (rgb[edge] * 0.45).astype(np.uint8)
    rgb[~m] = 246
    return rgb


def normal_rgb(h: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """[Z, X, 3] plan view of the top surface's unit normal, R=x-slope, G=z-slope, B=up.

    🔑 This reads the *derivative* of the height field, which is where a roof and a mound differ.
    `roof_shape_stats` failed because GT is itself terraced at 64^3 and no amplitude statistic can
    tell a discretised plane from a dome; slope can, and directly: a pitched plane is one flat
    colour, a ridge is a hard seam between two of them, and a dome is a continuous rainbow. Flat
    tops come out pale blue, which is the familiar normal-map convention.

    The gradient is taken with off-footprint columns filled from the nearest footprint column, so
    the footprint wall -- a vertical cliff carrying no roof information -- does not paint a false
    slope around every building.
    """
    m = np.asarray(fp, bool)
    H = np.asarray(h, np.float64)
    if m.any():
        H = H[tuple(ndimage.distance_transform_edt(~m, return_indices=True)[1])]
    gz, gx = np.gradient(H)
    n = np.stack([-gx, -gz, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=0, keepdims=True)
    rgb = ((n.transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb[~m] = 246
    return rgb


def _normal_key(size: int = 96) -> np.ndarray:
    """The legend for `normal_rgb`: every direction a roof can face, drawn as a hemisphere."""
    v, u = np.mgrid[0:size, 0:size] / (size / 2.0) - 1.0
    up = np.sqrt(np.clip(1.0 - (u ** 2 + v ** 2), 0.0, 1.0))
    rgb = (np.stack([u, v, up], -1) * 0.5 + 0.5) * 255
    rgb[(u ** 2 + v ** 2) > 1.0] = 246
    return rgb.astype(np.uint8)


def map_sheet(cases, out: Path, cell: int = 6, contour: int = 2) -> Path:
    """Height map and normal map, real building beside every arm, one row per building.

    The 3D montage answers "would you take this over the extruded footprint". These two views
    answer *why*: the height map shows where the volume went, and the normal map shows whether what
    is left is made of planes. They are drawn from the same `int16` height maps the arms are scored
    on, so nothing here is a separate rendering path that could disagree with the numbers.
    """
    from PIL import Image, ImageDraw

    names = list(cases[0]["arms"])
    def tile(a, box):
        z0, z1, x0, x1 = box
        return Image.fromarray(a[z0:z1, x0:x1]).resize(((x1 - x0) * cell, (z1 - z0) * cell),
                                                       Image.NEAREST)

    rows, lo = [], []
    for c in cases:
        zs, xs = np.nonzero(c["fp"])
        box = (max(zs.min() - 1, 0), min(zs.max() + 2, c["fp"].shape[0]),
               max(xs.min() - 1, 0), min(xs.max() + 2, c["fp"].shape[1]))
        lo.append(min(int(np.asarray(a)[c["fp"]].min()) for a in c["arms"].values()))
        rows.append([t for n in names
                     for t in (tile(height_rgb(c["arms"][n], c["fp"], c["extent"], contour,
                                               lo[-1]), box),
                               tile(normal_rgb(c["arms"][n], c["fp"]), box))])

    tw = max(t.width for r in rows for t in r)
    th = max(t.height for r in rows for t in r)
    head, pad, lab, foot = 40, 10, 30, 150
    cols = 2 * len(names)
    # LEGEND_W is a floor on the sheet, not a decoration: `--maps_arms` narrows the sheet to one
    # arm, and a canvas sized only by the tiles silently clips the key that says what a colour means
    sheet = Image.new("RGB", (max(cols * (tw + pad) + pad, LEGEND_W),
                              head + len(rows) * (th + lab) + foot), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, n in enumerate(names):
        x = pad + 2 * j * (tw + pad)
        d.text((x, 8), n.upper(), fill=(0, 0, 0))
        d.text((x, 24), "height", fill=(90, 90, 96))
        d.text((x + tw + pad, 24), "normals (slope)", fill=(90, 90, 96))
        if j:
            d.line([(x - pad // 2, 0), (x - pad // 2, head + len(rows) * (th + lab))],
                   fill=(210, 210, 215))
    for i, r in enumerate(rows):
        y = head + i * (th + lab)
        for j, t in enumerate(r):
            sheet.paste(t, (pad + j * (tw + pad) + (tw - t.width) // 2, y + (th - t.height) // 2))
        c = cases[i]
        d.text((pad, y + th + 6), f"id {c['id']}   extent {int(c['extent'])} vx"
               f"   {c['height_m']:.1f} m   ramp {lo[i]}-{int(c['extent'])} vx   " + "   ".join(
                   f"{n} extra {c['extra'][n]:.3f}" for n in names if n in c["extra"]),
               fill=(40, 40, 40))
        d.line([(0, y + th + lab - 2), (sheet.width, y + th + lab - 2)], fill=(228, 228, 232))

    # the two legends: what a colour means on each half of the sheet
    y = head + len(rows) * (th + lab) + 16
    from matplotlib import colormaps
    ramp = (np.asarray(colormaps["turbo"](np.linspace(0, 1, 256)))[None, :, :3]
            * 255).astype(np.uint8)
    sheet.paste(Image.fromarray(np.repeat(ramp, 22, 0)).resize((512, 22)), (pad, y + 16))
    d.text((pad, y), "HEIGHT   shared per row: the deepest level any arm reaches -> the extent",
           fill=(0, 0, 0))
    d.text((pad, y + 42), "deepest carve on the row", fill=(60, 60, 60))
    d.text((pad + 400, y + 42), "uncarved (the blockout)", fill=(60, 60, 60))
    kx = pad + 620
    sheet.paste(Image.fromarray(_normal_key()), (kx, y + 12))
    d.text((kx, y), "NORMALS   which way the roof faces", fill=(0, 0, 0))
    d.text((kx + 108, y + 20), "pale blue = flat   |   one flat colour = one pitched plane",
           fill=(60, 60, 60))
    d.text((kx + 108, y + 38), "hard seam = a ridge   |   smooth rainbow = a mound, not a roof",
           fill=(60, 60, 60))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def sheet_picks(rank, eligible, per_sheet: int) -> dict:
    """best / representative / worst, ranked by ONE arm's surplus over the eligible rows.

    🔑 Both sheet writers rank by the **first trained arm**, which is the pre-registered one, so
    "worst" means worst for the arm the bar was written for and not for whichever arm happens to be
    first in the dict. The rule lived at the montage call site while the map sheets silently ranked
    by the blockout; it is one function now so the two cannot drift apart again.
    """
    by = sorted(eligible, key=lambda i: rank[i])
    h = len(by) // 2
    return dict(best=by[:per_sheet], representative=by[h:h + per_sheet], worst=by[-per_sheet:])


def _pick_arms(heights: dict, names) -> dict:
    """The arms to draw, in the order asked for. Unknown names are an error, not a silent drop:
    a sheet quietly missing the arm it was made to show is worse than no sheet."""
    if not names:
        return heights
    missing = [n for n in names if n not in heights]
    if missing:
        sys.exit(f"--maps_arms: no such arm {missing}; have {list(heights)}")
    return {n: heights[n] for n in names}


def write_map_sheets(held: dict, heights: dict, per_sheet: int, ids=None,
                     rank_by: str | None = None) -> None:
    """Pick the buildings and write the map sheets, ranked by the arm the caller names.

    `extra` here comes from `height_split`, the column-space identity of `volume_split` -- the sheet
    is a picture of the same surfaces the artifact scores, and its captions must not come from a
    second, differently-computed number.
    """
    key = rank_by or next(iter(heights))
    extra = {name: [height_split(h[i], held["target"][i])["extra"] for i in range(len(h))]
             for name, h in heights.items()}
    if ids:
        want = {int(i) for i in ids}
        rows = {int(r): i for i, r in enumerate(held["row"])}
        if not want <= set(rows):
            sys.exit(f"--maps_ids: not in the pinned population: {sorted(want - set(rows))}")
        picks = {"picked": [rows[i] for i in ids]}
    else:
        # the carve-needing subset only: on a building whose envelope is already right, a map sheet
        # shows two flat rectangles and says nothing about form (#126 point 4)
        carve = [i for i in range(len(held["fp"]))
                 if height_split(apply_depth(held["fp"][i], int(held["extent"][i]),
                                             envelope_depth(held["fp"][i])),
                                 held["target"][i])["extra"] >= CARVE_NEEDED]
        picks = sheet_picks(extra[key], carve, per_sheet)
    for tag, sub in picks.items():
        cases = [dict(id=int(held["row"][i]), fp=held["fp"][i], extent=int(held["extent"][i]),
                      height_m=float(held["height_m"][i]),
                      arms={"real building": held["target"][i],
                            **{a: heights[a][i] for a in heights}},
                      extra={a: extra[a][i] for a in heights}) for i in sub]
        if cases:
            print(f"[maps] {map_sheet(cases, WORK / f'maps_{tag}.png')}", flush=True)


def report(res: dict) -> None:
    print("\n" + "=" * 100)
    print("the aggregate is right of the bar: #126 demoted it, so it may not head the row")
    for pop, label in (("carve", "CARVE-NEEDING buildings -- the population the bar is set on"),
                       ("flat", "ALREADY-FLAT buildings -- reported, never pooled"),
                       ("all", "all pinned buildings")):
        print(f"\n== {label} (n={res['arms']['blockout'][pop]['n']}) ==")
        print(f"{'arm':22s} {'miss':>7} {'extra':>7} {'vs_inp':>7} {'collapse':>9} "
              f"{'>env:xtr':>9} {'carved':>7} {'FORM:ops':>9} {'planar':>7} | {'(3D IoU)':>9}   "
              f"(GT: carves {res['arms']['blockout'][pop]['gt_carved_cols']:.3f} of columns, "
              f"form {res['arms']['blockout'][pop].get('gt_dl_ops', float('nan')):.1f} ops, "
              f"planar {res['arms']['blockout'][pop].get('gt_dl_planar_fraction', float('nan')):.2f})")
        for name, a in res["arms"].items():
            s = a[pop]
            w = a["beats_envelope_extra"][pop]["rate_ex_ties"]
            print(f"{name:22s} {s['missing']:>7.4f} {s['extra']:>7.4f} {s['vs_input']:>7.4f} "
                  f"{s['collapse_rate']:>9.4f} {w:>9.3f} {s['carved_cols']:>7.3f} "
                  f"{s.get('dl_ops', float('nan')):>9.1f} "
                  f"{s.get('dl_planar_fraction', float('nan')):>7.2f} | {s['vol_iou']:>9.4f}")
    if res.get("reference"):
        print("\n== this project's arms of record, re-summarised on the SAME carve-needing rows ==")
        for name, a in res["reference"].items():
            vi = "     -" if a["vs_input"] is None else f"{a['vs_input']:>7.4f}"
            print(f"{name:22s} {a['missing']:>7.4f} {a['extra']:>7.4f} {vi} "
                  f"{a['collapse_rate']:>9.4f} {'':>9} {'':>7} {'':>9} {'':>7} | "
                  f"{a['vol_iou']:>9.4f}")

    print("\n== the pre-registered bar, on the carve-needing subset ==")
    for name, v in res["verdict"].items():
        print(f"  {name:22s} beats 1-NN `extra` {str(v['beats_1nn_extra']):>5}   "
              f"collapse ok {str(v['collapse_no_worse_than_1nn']):>5}   "
              f"moved {str(v['moved']):>5}   ->  {'PASS' if v['pass'] else 'NOT MET'}"
              + ("   [KILL: identity]" if v["killed_identity"] else ""))
    print("=" * 100)


# ==================================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_from", default=str(SHIP714))
    ap.add_argument("--objective", default="ce", choices=OBJECTIVES,
                    help="which statistic of the per-column posterior to target: ce -> the mode, "
                         "mse -> the mean, quantile -> --quantile (0.5 = the median)")
    ap.add_argument("--k_planes", type=int, default=6,
                    help="planes for --objective planes. #10 measured a median of 5 operations to "
                         "explain a real roof and 9 at p75, so 6 is the median-plus with room")
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="the pinball loss's quantile; used by --objective quantile ONLY -- the "
                         "slope term reads its own fixed SLOPE_DECODE_QUANTILE. 0.5 is "
                         "the median and is the value #127 pre-committed to -- sweeping it trades "
                         "`missing` against `extra` directly and would be selecting on the answer")
    ap.add_argument("--slope_weight", type=float, default=0.0,
                    help="weight on the joint SLOPE term, added to the per-column loss and never "
                         "in place of it. 0 disables it, which is every arm on #127's record; the "
                         "pre-registered value is 1.0, fixed a priori as a 20%% share of the "
                         "converged loss (CE 1.5552, slope 0.3090) and deliberately not swept")
    ap.add_argument("--tag", default=None, help="run name; defaults to the objective")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_aug", action="store_true", help="disable the 8 plan symmetries")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--rebuild_program_cache", action="store_true",
                    help="re-fit #6's slot labels over the whole corpus (~1 min on 48 cores)")
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="score these checkpoints instead of training (name=path or path)")
    ap.add_argument("--no_form", action="store_true",
                    help="skip the description-length form metric. It fits a Layer/Ramp/CutRoof "
                         "program to every arm's own surface, which is the only measure found that "
                         "separates a roof from a mound -- and it costs ~0.07s per building per arm")
    ap.add_argument("--median_decode", action="store_true",
                    help="add a second arm per CE checkpoint decoding the posterior MEDIAN rather "
                         "than the mode. A decode ablation reported beside the pre-registered arm, "
                         "never in place of it")
    ap.add_argument("--montage", type=int, default=6, help="buildings per sheet; 0 disables")
    ap.add_argument("--maps", type=int, default=0,
                    help="buildings per height/normal MAP sheet -- the plan-view pair that shows "
                         "where the volume went and whether what is left is made of planes")
    ap.add_argument("--maps_ids", type=int, nargs="*", default=None,
                    help="render exactly these corpus row ids, e.g. the ones already on a montage")
    ap.add_argument("--maps_arms", nargs="*", default=None,
                    help="restrict the map sheet to these arms. Seven arms is 16 columns and "
                         "unreadable at any print size, and an unreadable figure decides nothing")
    ap.add_argument("--maps_only", action="store_true",
                    help="write the map sheets from --ckpt and exit, skipping the scored run")
    ap.add_argument("--out", default="execution/artifacts/height_map_generator_714.json")
    args = ap.parse_args()
    args.tag = args.tag or f"heightmap_{args.objective}"

    cache = build_cache(force=args.rebuild_cache)

    ckpts = {}
    if args.ckpt:
        for spec in args.ckpt:
            name, _, path = spec.rpartition("=")
            ckpts[name or Path(path).stem] = Path(path)
    else:
        ckpts[args.tag] = train(cache, args)

    # ---- the pinned population, in the pinned order -------------------------------------------
    ids = [int(i) for i in json.load(open(args.ids_from))["ids"]]
    row_to_idx = {int(r): i for i, r in enumerate(cache["row"])}
    sel = np.array([row_to_idx[i] for i in ids if i in row_to_idx and cache["ok"][row_to_idx[i]]])
    held = {k: cache[k][sel] for k in ("row", "fp", "target", "y0", "extent", "region", "height_m")}
    held["fp"] = held["fp"] > 0
    held["target"] = held["target"].astype(np.int16)
    print(f"[ids] {len(sel)} pinned buildings from {args.ids_from}", flush=True)

    if args.maps_only:
        # The sheet only needs the arms' own height maps, so it skips 1-NN retrieval, the mean roof
        # and the form fitter -- minutes of work that would produce nothing this figure draws.
        if not args.ckpt:
            sys.exit("--maps_only scores nothing, so it needs --ckpt to say what to draw")
        heights = {}
        for name, path in ckpts.items():
            heights[name], meta = predict(path, held, cpu=args.cpu)
            if args.median_decode and meta["objective"] == "ce":
                heights[f"{name}_median"], _ = predict(path, held, cpu=args.cpu, quantile=0.5)
        arms = _pick_arms(heights, args.maps_arms)
        write_map_sheets(held, arms, args.maps or 6, args.maps_ids, rank_by=next(iter(arms)))
        return

    # ---- the arms -------------------------------------------------------------------------------
    train_idx = np.nonzero((cache["ok"] > 0) & (cache["held"] == 0))[0]
    bank_fp = cache["fp"][train_idx] > 0
    bank_target = cache["target"][train_idx].astype(np.int16)
    bank_extent = cache["extent"][train_idx].astype(np.int32)
    print(f"[bank] {len(train_idx)} training buildings for retrieval and for the mean roof",
          flush=True)

    heights = {"blockout": np.stack([apply_depth(held["fp"][i], int(held["extent"][i]),
                                                 envelope_depth(held["fp"][i]))
                                     for i in range(len(sel))])}

    bank_depth = np.stack([carve_depth(bank_target[i], bank_fp[i], int(bank_extent[i]))
                           for i in range(len(train_idx))])
    profile = mean_relative_depth(bank_depth, bank_fp, bank_extent)
    heights["mean_roof"] = np.stack([mean_roof_height(profile, held["fp"][i],
                                                      int(held["extent"][i]))
                                     for i in range(len(sel))])

    t0 = time.time()
    nn = retrieve_nn(held["fp"], bank_fp)
    heights["nn_retrieval"] = np.stack([
        transplant_height(bank_target[j], bank_fp[j], int(bank_extent[j]),
                          held["fp"][i], int(held["extent"][i]))
        for i, j in enumerate(nn)])
    print(f"[1-NN] retrieved in {time.time()-t0:.0f}s  "
          f"(median footprint IoU to the retrieved row reported in the artifact)", flush=True)

    ckpt_meta = {}
    for name, path in ckpts.items():
        heights[name], ckpt_meta[name] = predict(path, held, cpu=args.cpu)
        if args.median_decode and ckpt_meta[name]["objective"] == "ce":
            alt = f"{name}_median"
            heights[alt], ckpt_meta[alt] = predict(path, held, cpu=args.cpu, quantile=0.5)

    # ---- score, split by population, never pooled -----------------------------------------------
    rows = {name: score_arm(h, held, form=not args.no_form)
            for name, h in heights.items()}
    carve_mask = np.array([r["blockout_extra"] >= CARVE_NEEDED for r in rows["blockout"]])
    pops = {p: np.nonzero(m)[0] for p, m in
            dict(all=np.ones(len(carve_mask), bool), carve=carve_mask, flat=~carve_mask).items()}

    arms = {}
    for name, rr in rows.items():
        a = {p: summarise([rr[i] for i in idx]) for p, idx in pops.items()}
        a["beats_envelope_extra"], a["beats_envelope_iou"] = {}, {}
        for p, idx in pops.items():
            # paired against the SAME building's envelope, by index -- #126's like-for-like rule
            paired = [dict(arm=dict(extra=rr[i]["extra"], vol_iou=rr[i]["vol_iou"]),
                           blockout=dict(extra=rows["blockout"][i]["extra"],
                                         vol_iou=rows["blockout"][i]["vol_iou"])) for i in idx]
            a["beats_envelope_extra"][p] = compare_to_envelope(paired, "arm", "extra", False)
            a["beats_envelope_iou"][p] = compare_to_envelope(paired, "arm", "vol_iou", True)
        arms[name] = a

    res = dict(
        meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), question="#127",
                  ids_from=args.ids_from, gt_h5=str(H5.relative_to(REPO)),
                  n_pinned=len(sel), n_carve=int(carve_mask.sum()),
                  n_train=len(train_idx), depth_classes=DEPTH_CLASSES,
                  checkpoints=ckpt_meta, run_flags=dict(
                      epochs=args.epochs, batch=args.batch, lr=args.lr, width=args.width,
                      seed=args.seed, augment=not args.no_aug, trained_here=not bool(args.ckpt))),
        arms=arms, verdict=verdict(arms, "carve"),
        reference=reference_arms({int(held["row"][i]) for i in pops["carve"]}),
        nn_footprint_iou=float(np.median([
            float((held["fp"][i] & bank_fp[j]).sum()) / max(float((held["fp"][i] | bank_fp[j]).sum()), 1)
            for i, j in enumerate(nn)])),
        per_building={name: rr for name, rr in rows.items()},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    report(res)
    print(f"\n[artifact] {out}")

    # ranked by the FIRST trained arm, which is the pre-registered one -- so "worst" means worst
    # for the arm the bar was written for, not for whichever arm happens to lead the dict
    model_names = [n for n in heights if n in ckpts]
    key = model_names[0] if model_names else "nn_retrieval"

    if args.maps or args.maps_ids:
        arms = _pick_arms(heights, args.maps_arms)
        write_map_sheets(held, arms, args.maps or 6, args.maps_ids,
                         rank_by=key if key in arms else next(iter(arms)))

    if args.montage:
        picks = sheet_picks([r["extra"] for r in rows[key]], pops["carve"].tolist(), args.montage)
        for tag, sub in picks.items():
            cases = [dict(id=int(held["row"][i]), fp=held["fp"][i], target=held["target"][i],
                          arms={n: heights[n][i] for n in heights},
                          extra={n: rows[n][i]["extra"] for n in heights}) for i in sub]
            if cases:
                print(f"[montage] {montage(cases, WORK / f'{tag}.png')}")


if __name__ == "__main__":
    main()
