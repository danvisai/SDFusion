"""#125: the authentic whole-volume A2 voxel-correction feasibility experiment.

Question, and only question:

    Can a small, deterministic occupancy-space editor make useful architectural massing
    changes to real A2 outputs -- ANY voxel eligible, not just a roof or surface band --
    without collapsing the building, spilling past its footprint, or learning to return
    its input?

This supersedes the original throwaway prototype (#119/#120's ancestor). That version used a
source-derived roof/surface `edit_mask` and a KEEP/ADD/REMOVE action lattice. #125 requires both
removed: the primary endpoint is now absolute, dense, binary occupancy at every voxel, and there
is no spatial mask -- only a shared, minimal, deterministic footprint sanitizer applied
identically to the learned candidate and to every baseline arm.

Five commands form the falsifiable experiment::

    # CPU-only plumbing check; does not load A2/Dora or touch the training GPU.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py smoke

    # CPU: fix the 384-train / 96-screen / 714-full row manifest before any generation.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py manifest \
      --out outputs/voxel_editor_prototype/manifest.json

    # GPU: materialise authentic (A2 decoded field, real occupancy) pairs for one role.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py cache --role train \
      --manifest outputs/voxel_editor_prototype/manifest.json \
      --out outputs/voxel_editor_prototype/train.h5

    # GPU recommended, CPU possible: fit the absolute-occupancy corrector on TRAIN only.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py train \
      --cache outputs/voxel_editor_prototype/train.h5

    # One-time screen on the 96-row cache, then (only if it passes) evaluate-full on 714.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py evaluate \
      --cache outputs/voxel_editor_prototype/screen.h5 \
      --checkpoint outputs/voxel_editor_prototype/editor.pth

The cache branches from `DoraCodec.decode_grid` directly and never meshes the field and
voxelises it again. Meshing remains a visualisation/export concern (surface realization, below),
not a data conversion.

Frozen by #125 and not swept here: A2 checkpoint step 240,000
(sha256 `643aed0896e2edc36ab3ecb073da63847881dc2ba95459eed896a19b65fed04d`), strength 0.5, 20
projection steps, guidance 1.0, master seed 0, exactly one sample per building.

This file remains self-contained and named `prototype`. It is not a production model, service,
dataset format, or new architectural commitment (#125's Out of Scope), and a passing result does
not by itself authorize production integration, stochastic diffusion, or recipe closure.

⚠️ **The cohorts in the commands above are #125's and are superseded.** #118 settled the bar on
2026-09-18 against the BuildingWorld corpus: a 250-row family-stratified screen, #178's own
24,464-row nine-city held-out slice as the confirmation, the pinned 714 demoted to a regression
guard, and training on the full held-out=0 bank. `build_manifest`/`select_screen_rows` below are
still the three-region 384/96/714 design and must be rebuilt by the implementing pass;
`GATE_SCREEN_N` and `GATE_CONFIRMATION_N` already hold the settled values and the gate enforces
them, so a cohort that was not rebuilt fails the bar rather than passing quietly. See
`docs/wayfinding/whole-volume-voxel-transform/118-gates.md`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.frozen_corpus import FROZEN_CORPUS_SHA256, open_real_corpus  # noqa: E402

RES = 64
DEFAULT_A2 = REPO / "weights/massing-vecset/vecset_v4_surf.pth"
DEFAULT_LATENTS = REPO / "data/real_massing_v1/vecset_latents.h5"
DEFAULT_REAL = REPO / "data/real_massing_v1/real.h5"
DEFAULT_OUT = REPO / "outputs/voxel_editor_prototype"

# Regions, matching `scripts/foundations/watch_checkpoints.py:REGION_NAMES` and the `region`
# column of `vecset_latents.h5`: 0 NL (3D BAG), 1 DE (NRW), 2 JP (PLATEAU).
N_REGIONS = 3
REGION_NAMES = {0: "bag_nl", 1: "nrw_de", 2: "plateau_jp"}

# #125: freeze the accepted shipped A2 operating point before any editor training.
A2_CHECKPOINT_SHA256 = "643aed0896e2edc36ab3ecb073da63847881dc2ba95459eed896a19b65fed04d"
A2_STRENGTH = 0.5
A2_STEPS = 20
A2_GUIDANCE = 1.0
A2_MASTER_SEED = 0

# #125: the preregistered cohort sizes. Opportunity is split evenly BAG/NRW; identity control is
# 19/19/20 BAG/NRW/PLATEAU. The screen is a flat 32-per-region draw from the sealed 714.
MANIFEST_VERSION = 1
MANIFEST_SALT = "voxel-editor-v1"
TRAIN_OPPORTUNITY_QUOTA = {0: 163, 1: 163}
TRAIN_IDENTITY_QUOTA = {0: 19, 1: 19, 2: 20}
TRAIN_N = sum(TRAIN_OPPORTUNITY_QUOTA.values()) + sum(TRAIN_IDENTITY_QUOTA.values())  # 384
SCREEN_PER_REGION = 32
EXPECTED_FULL_POPULATION_N = 714
# `SCREEN_N` (= 96) is gone: #118 replaced it with `GATE_SCREEN_N` and the screen is no longer a
# flat per-region draw. `SCREEN_PER_REGION` still drives `select_screen_rows` until the
# implementing pass rebuilds the cohorts on BuildingWorld.

# Mirrors `scripts.foundations.eval_massing_arms`' constants of the same name (checked equal by
# `test_prototype_voxel_editor.py`). Not imported directly: that module's own top-level imports
# pull in torch and scipy, which would make importing THIS module for its pure occupancy logic
# unconditionally require both -- exactly what this file's lazy-import convention avoids elsewhere.
S_STAR_VOXELS = 3          # ADR 0004: detail scale s* = 1.0 m ~= 3 voxels @64^3. Fixed a priori.
COLLAPSE_MISSING = 0.15    # #80: solid iff missing < 15%.
C2_ALLOWANCE = 0.05        # #85's criterion 2, allowance chosen by the owner on 2026-08-07.

KEEP, ADD, REMOVE = 0, 1, 2

# #125: fixed a priori, before any corrector has been run, so no stratum boundary is chosen to
# flatter a result. Both strata use 1 - vol_iou against the real target.
E_STRATUM_BOUNDS = (0.05, 0.20)   # envelope (footprint+height, no A2) error
A_STRATUM_BOUNDS = (0.05, 0.20)   # authentic A2 (raw decode) error

# -------------------------------------------------------------------------------------------------
# #118's preregistered bar. Settled by interview with the ticket owner on 2026-09-18; see
# docs/wayfinding/whole-volume-voxel-transform/118-gates.md for the derivation of every number
# here and for what each one is quoted from. Nothing below was chosen by taste.
# -------------------------------------------------------------------------------------------------

# Simon (1989)'s two-stage design: stop for futility at <= r1 wins of n1. n1 = 250 kills a useless
# arm 92.7% of the time against 62.0% at #125's original 96, for the same ~4% power cost.
GATE_NULL_P0 = 0.50                 # a paired win rate no better than a coin flip
GATE_SCREEN_ALTERNATIVE_P1 = 0.60   # the alternative the screen is powered against
GATE_SCREEN_N = 250
# #178's own held-out population, so its quoted floors apply unchanged. Scoring #118's arm on any
# other population silently voids every one of them.
GATE_CONFIRMATION_N = 24464
# `guided_edit_completion_proxy.UNDERSAMPLED_N`: below this a rate is reported, never trusted.
MIN_DECIDED_COMPARISONS = 30
# ⚠️ Not settled in the #118 interview -- a mechanical consequence of computing the futility
# boundary on decided comparisons, recorded here rather than left implicit. Ties are excluded from
# every rate (#126), so a boundary read off a thin decided set would be cheap to clear; at least
# half the cohort must have produced a decided comparison on each axis.
MIN_DECIDED_FRACTION = 0.5
# CONTEXT.md's standing guard, and `buildingworld_baseline.quoted_bar`'s `max_vs_input`. #125's
# placeholder wrote 0.99, which is looser than the project's own rule; 0.98 binds.
MAX_VS_INPUT = 0.98
# CONTEXT.md: 1-NN retrieval's collapse rate. "A generator that destroys more buildings than naive
# retrieval is not servable whatever else it scores."
KILL_COLLAPSE_RATE = 0.1582

# ICH E10 assay sensitivity: the run is uninterpretable unless the control behaved as expected.
# `massing_arms_eval_ship714.json` `summary.a2_s0.5.extra`, the shipped operating point's own
# number on the pinned 714 -- the population CONTEXT.md designates this route's regression control.
A2_PINNED_714_EXTRA_OF_RECORD = 0.09219727319195108
# The same artifact's `noise_floor.median_range.extra`: this project's measured seed-to-seed range
# for `extra`. The cache re-seeds per corpus row rather than by loop order (#125), so the cached
# baseline is a different draw of the same operating point, and that is exactly the term this
# tolerance has to cover.
EXTRA_SEED_NOISE_RANGE = 0.04


class RoleIsolationError(RuntimeError):
    """Raised when code tries to open a voxel cache under the wrong role.

    #125: "the trainer may open only the training cache. Screening and full target arrays must
    live in separate artifacts unavailable to the training process." `open_training_cache`
    always passes `expected_role="train"`; there is no other function in this module through
    which the trainer can reach a screen/full-role file.
    """


# -------------------------------------------------------------------------------------------------
# Digests: checkpoint identity, per-row content, and row-list identity.
# -------------------------------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_frozen_a2_checkpoint(path: Path) -> None:
    """#125: the active experimental A2 run is ineligible for this corpus. Fail loudly, not
    silently, if `path` is not the exact accepted shipped operating point."""
    actual = sha256_file(Path(path))
    if actual != A2_CHECKPOINT_SHA256:
        raise ValueError(
            f"#125: {path} does not match the frozen A2 checkpoint (expected sha256 "
            f"{A2_CHECKPOINT_SHA256}, got {actual}). This corpus may only be generated from the "
            "accepted shipped step-240000 checkpoint; the active experimental A2 run, or any "
            "other checkpoint, is ineligible.")


def row_list_digest(rows) -> str:
    payload = np.array(sorted(int(r) for r in rows), dtype="<i8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def row_content_digest(row: int, region: int, height_m: float, y0: int, y1: int,
                       footprint: np.ndarray, source_occ: np.ndarray,
                       target_occ: np.ndarray) -> str:
    """SHA-256 over the exact categorical content a cache row commits to.

    Explicit widths and byte order, packed bits for volumes, mirroring
    `utils.frozen_corpus.row_identity_sha256`'s reproducibility discipline. Does not cover
    `source_field` (float16 conditioning is not the categorical source of truth) or
    `envelope_occ` (a pure function of footprint + target's own extent, already covered).
    """
    h = hashlib.sha256()
    h.update(np.array([row], "<i8").tobytes())
    h.update(np.array([region], "<i1").tobytes())
    h.update(np.array([height_m], "<f4").tobytes())
    h.update(np.array([y0, y1], "<i2").tobytes())
    h.update(np.packbits(np.asarray(footprint, bool)).tobytes())
    h.update(np.packbits(np.asarray(source_occ, bool)).tobytes())
    h.update(np.packbits(np.asarray(target_occ, bool)).tobytes())
    return h.hexdigest()


# -------------------------------------------------------------------------------------------------
# Pure occupancy logic. Array axes follow this repository's convention: [z, y, x], y is vertical.
# -------------------------------------------------------------------------------------------------


def volume_metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate, target = np.asarray(candidate, bool), np.asarray(target, bool)
    inter = int((candidate & target).sum())
    union = int((candidate | target).sum())
    target_volume = max(int(target.sum()), 1)
    return {
        "vol_iou": float(inter / union) if union else 1.0,
        "missing": float((target & ~candidate).sum() / target_volume),
        "extra": float((candidate & ~target).sum() / target_volume),
    }


def fill_sealed_cavities(occ: np.ndarray) -> np.ndarray:
    """Close interior voids, leaving courtyards and passages untouched.

    #118, owner decision: interior fill is not architecture. A building modelled hollow and the
    same building modelled solid are one building seen from outside, and ISO 19107 treats an
    interior shell as a first-class feature (#117 declines to fail cavities for the same reason).
    A raw volume comparison charges every voxel of air inside a shell target as the candidate's
    surplus, which is bookkeeping about air -- and #117 measured ~4.6% of BuildingWorld (~70,600
    rows) arriving as 1-3 voxel skins, Calgary ~97% of them, so it is not a rare corner.

    🔑 It also puts this arm on the footing of the bar it is graded against. #178's 1-NN floors are
    computed by `train_height_map_generator.height_split` in height-column space, where a column is
    a solid run and an interior cannot exist. Grading a hollow-sensitive candidate against a
    hollow-blind floor compares two different measurements.

    A cavity is sealed iff it is unreachable from outside, which is `hollow_shell_voxels`' own
    test, so a courtyard open to the sky or a passage clear through the building survives.
    """
    occ = np.asarray(occ, bool)
    return occ | hollow_shell_voxels(occ)


def sealed_volume_metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """`volume_metrics` after sealing both sides, so hollow-vs-solid cannot reach any number."""
    return volume_metrics(fill_sealed_cavities(candidate), fill_sealed_cavities(target))


def occupancy_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def vertical_extent(occ: np.ndarray) -> tuple[int, int] | None:
    ys = np.nonzero(np.asarray(occ, bool).any(axis=(0, 2)))[0]
    return (int(ys.min()), int(ys.max())) if len(ys) else None


def envelope_occupancy(footprint: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Extrude the conditioning footprint across the target's specified vertical extent."""
    out = np.zeros_like(target, dtype=bool)
    ext = vertical_extent(target)
    if ext is not None:
        out[:, ext[0] : ext[1] + 1, :] = np.asarray(footprint, bool)[:, None, :]
    return out


def sanitize_footprint(occ: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """The shared minimal deterministic sanitizer: footprint intersection, nothing else.

    #125: applied IDENTICALLY to the learned candidate and to every baseline arm, so deleting
    spill outside the footprint can never be credited as a learned architectural correction. If
    a stronger deterministic repair is ever introduced it creates a new named corpus and must
    still be applied identically to the baseline -- it does not replace this function.
    """
    return np.asarray(occ, bool) & np.asarray(footprint, bool)[:, None, :]


def occupancy_to_sdf(occ: np.ndarray) -> np.ndarray:
    """Crisp metric signed EDT; negative is inside. `field <= 0 == occ` everywhere except the
    degenerate all-empty/all-solid volume."""
    from scipy import ndimage

    occ = np.asarray(occ, bool)
    spacing = 2.0 / max(occ.shape[0] - 1, 1)
    return ((ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ))
            * spacing).astype(np.float32)


def derive_action(source_occ: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """KEEP/ADD/REMOVE implied by (source, occ). A reporting/reweighting convenience, never a
    stored state -- see `apply_action_to_source` and the reduction identity it proves."""
    source_occ, occ = np.asarray(source_occ, bool), np.asarray(occ, bool)
    action = np.full(source_occ.shape, KEEP, np.uint8)
    action[~source_occ & occ] = ADD
    action[source_occ & ~occ] = REMOVE
    return action


def apply_action_to_source(source_occ: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Reduce a KEEP/ADD/REMOVE action lattice back to absolute occupancy.

    #125: "[an auxiliary action loss] must reduce immediately to absolute occupancy and is never
    the stored state." This function exists only to make that reduction an identity that can be
    tested (`apply_action_to_source(s, derive_action(s, occ)) == occ` for any `occ`); the
    corrector's forward pass never constructs or consumes an action lattice.
    """
    out = np.asarray(source_occ, bool).copy()
    action = np.asarray(action)
    out[action == ADD] = True
    out[action == REMOVE] = False
    return out


def action_weight_map(source_occ: np.ndarray, target_occ: np.ndarray,
                      keep_weight: float = 0.12, edit_weight: float = 1.0) -> np.ndarray:
    """Per-voxel training weight from the KEEP/ADD/REMOVE action implied by (source, target).

    This is the auxiliary signal #125 permits: it reweights the SAME absolute-occupancy
    cross-entropy so the rare ADD/REMOVE voxels are not drowned out by an overwhelming KEEP
    majority. There is no second output, no separate head, and no persisted action state.
    """
    action = derive_action(source_occ, target_occ)
    return np.where(action == KEEP, keep_weight, edit_weight).astype(np.float32)


# -------------------------------------------------------------------------------------------------
# Validity contract. #117 (the hard validity contract) remains an open human decision; these are
# the explicit checks #125's own Implementation/Testing Decisions ask for, stated so they can be
# measured and revisited rather than assumed.
# -------------------------------------------------------------------------------------------------


def flood_fill_exterior(empty: np.ndarray) -> np.ndarray:
    """Empty voxels reachable from outside the grid, 6-connectivity, through empty space only.

    A courtyard open to the sky or a passage clear through the building both touch a boundary
    face of the volume and are reachable; a true hollow shell -- solid on every side -- is not.
    """
    from scipy import ndimage

    empty = np.asarray(empty, bool)
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, n = ndimage.label(empty, structure=structure)
    if n == 0:
        return np.zeros_like(empty)
    boundary_labels = (
        set(np.unique(labeled[0, :, :])) | set(np.unique(labeled[-1, :, :]))
        | set(np.unique(labeled[:, 0, :])) | set(np.unique(labeled[:, -1, :]))
        | set(np.unique(labeled[:, :, 0])) | set(np.unique(labeled[:, :, -1]))
    )
    boundary_labels.discard(0)
    return np.isin(labeled, list(boundary_labels))


def hollow_shell_voxels(occ: np.ndarray) -> np.ndarray:
    """Empty voxels sealed off from the outside -- an unintended cavity, not an admissible
    courtyard or passage (both of those remain reachable from a boundary face)."""
    occ = np.asarray(occ, bool)
    empty = ~occ
    return empty & ~flood_fill_exterior(empty)


def connected_components(occ: np.ndarray):
    from scipy import ndimage

    occ = np.asarray(occ, bool)
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    return ndimage.label(occ, structure=structure)


def ground_connected_ok(occ: np.ndarray) -> bool:
    """Every solid connected component touches y=0 -- no floating fragments or islands.

    Multiple components ARE allowed (a two-blob footprint gives GT itself 1-2 components), so
    this does not require single-component massing; it only forbids a component with no ground
    contact at all.
    """
    occ = np.asarray(occ, bool)
    if not occ.any():
        return False
    labeled, _ = connected_components(occ)
    ground_labels = set(np.unique(labeled[:, 0, :])) - {0}
    all_labels = set(np.unique(labeled)) - {0}
    return ground_labels == all_labels


def min_thickness_survival(occ: np.ndarray, s_star: int = S_STAR_VOXELS) -> float:
    """Fraction of solid volume surviving an erosion by radius floor(s_star/2).

    A wafer-thin wall or slab disappears under this erosion; massing built at or above the
    project's detail scale s* (ADR 0004, 3 voxels @64^3) mostly survives. A volume-fraction
    proxy, not a per-wall guarantee, pending #117's formal contract.
    """
    from scipy import ndimage

    occ = np.asarray(occ, bool)
    total = int(occ.sum())
    if total == 0:
        return 0.0
    radius = max(int(s_star) // 2, 1)
    # border_value=1: treat the outside of the 64^3 GRID as solid for this erosion, so a
    # building that legitimately reaches the domain boundary (every building's y=0 ground
    # layer, or a wall near the x/z edge) is not penalised as "thin" merely for touching the
    # array's own edge -- only a real empty neighbour should count as a nearby surface.
    eroded = ndimage.binary_erosion(occ, iterations=radius, border_value=1)
    return float(int(eroded.sum()) / total)


def validity_report(occ: np.ndarray, footprint: np.ndarray, s_star: int = S_STAR_VOXELS,
                    min_thickness_survival_threshold: float = 0.5) -> dict:
    """Explicit validity outcomes for one candidate massing. Every field is always populated --
    nothing here is a hidden repair; only `sanitize_footprint` ever changes the candidate."""
    occ = np.asarray(occ, bool)
    footprint3 = np.asarray(footprint, bool)[:, None, :]
    spill = int((occ & ~footprint3).sum())
    hollow = int(hollow_shell_voxels(occ).sum())
    _, n_components = connected_components(occ)
    thickness = min_thickness_survival(occ, s_star)
    report = {
        "nonempty": bool(occ.any()),
        "footprint_exact": spill == 0,
        "spill_voxels": spill,
        "ground_connected": ground_connected_ok(occ),
        "n_connected_components": int(n_components),
        "hollow_shell_voxels": hollow,
        "no_hollow_shell": hollow == 0,
        "min_thickness_survival": thickness,
        "min_thickness_ok": thickness >= min_thickness_survival_threshold,
    }
    report["valid"] = bool(
        report["nonempty"] and report["footprint_exact"] and report["ground_connected"]
        and report["no_hollow_shell"] and report["min_thickness_ok"]
    )
    return report


def validity_projection_delta(raw_occ: np.ndarray, sanitized_occ: np.ndarray) -> dict:
    """How much the deterministic sanitizer changed a candidate. Always computed and reported
    (#125: "no unreported validity projection") -- including when it is exactly zero."""
    raw_occ, sanitized_occ = np.asarray(raw_occ, bool), np.asarray(sanitized_occ, bool)
    changed = int((raw_occ != sanitized_occ).sum())
    return {
        "projection_changed_voxels": changed,
        "projection_changed_fraction": float(changed / max(int(raw_occ.sum()), 1)),
    }


# -------------------------------------------------------------------------------------------------
# Surface realization: two arms over IDENTICAL corrected occupancy.
# -------------------------------------------------------------------------------------------------


def signs_consistent(field: np.ndarray, occ: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(field) <= 0, np.asarray(occ, bool)))


def recover_surface_control(occ: np.ndarray) -> np.ndarray:
    """The deterministic control: signed Euclidean-distance recovery. Sign-exact by
    construction (`occupancy_to_sdf`)."""
    return occupancy_to_sdf(occ)


def recover_surface_narrowband(occ: np.ndarray, source_field: np.ndarray,
                               band_voxels: float = 3.0) -> np.ndarray:
    """The at-most-one continuous narrow-band candidate #125 permits.

    Starts from the crisp control field and blends toward the A2 source field's own shape
    evidence within `band_voxels` of the corrected surface -- but only where doing so cannot
    flip which side of the surface a grid sample is on. A convex combination of two same-signed
    values is always the same sign, so the sign-exactness assertion below is a proof, not a hope.
    """
    control = occupancy_to_sdf(occ)
    src = np.asarray(source_field, np.float32)
    spacing = 2.0 / max(occ.shape[0] - 1, 1)
    band = np.abs(control) <= (band_voxels * spacing)
    same_sign = (src <= 0) == (control <= 0)
    take = band & same_sign
    blended = control.copy()
    blended[take] = 0.5 * control[take] + 0.5 * src[take]
    if not signs_consistent(blended, occ):
        raise RuntimeError("#125: narrowband recovery failed to preserve occupancy signs")
    return blended.astype(np.float32)


def sharp_normal_error_for_arms(fields: dict, device, views: int = 22, size: int = 256) -> dict:
    """Thin, nonzero-views-enforcing wrapper over the fixed-ID evaluator's own SNE (#79), so this
    experiment extends that instrument instead of inventing a second one."""
    from scripts.foundations.eval_massing_arms import sharp_normal_error

    if views <= 0:
        raise ValueError("#125: Sharp Normal Error must run with nonzero views")
    return sharp_normal_error(fields, list(fields.keys()), device, views=views, size=size)


# -------------------------------------------------------------------------------------------------
# Model: absolute-occupancy corrector. No spatial mask, no action head.
# -------------------------------------------------------------------------------------------------

IN_CHANNELS = 7 + N_REGIONS  # source_occ, field, footprint, y, height, y0, y1, region one-hot


def model_input(source_occ: np.ndarray, source_field: np.ndarray, footprint: np.ndarray,
                height_m: float, y0: int, y1: int, region: int) -> np.ndarray:
    """Build the corrector's conditioning tensor.

    #125: conditioned only on source-side information already available at inference -- source
    occupancy, the normalized source field, footprint, declared height, explicit normalized
    vertical extent, and region. Targets, identity labels, opportunity strata, and target-derived
    masks are never inputs; this signature has no parameter through which any of them could
    reach the model.
    """
    field = np.asarray(source_field, np.float32)
    src = np.asarray(source_occ, bool).astype(np.float32) * 2.0 - 1.0
    nz, ny, nx = field.shape
    fp3 = np.broadcast_to(np.asarray(footprint, np.float32)[:, None, :], field.shape)
    y = np.broadcast_to(np.linspace(-1, 1, ny, dtype=np.float32)[None, :, None], field.shape)
    h = np.full(field.shape, np.clip(float(height_m) / 30.0, 0.0, 2.0), np.float32)
    y0n = np.full(field.shape, 2.0 * float(y0) / max(ny - 1, 1) - 1.0, np.float32)
    y1n = np.full(field.shape, 2.0 * float(y1) / max(ny - 1, 1) - 1.0, np.float32)
    region_channels = [np.full(field.shape, float(region == r), np.float32) for r in range(N_REGIONS)]
    return np.stack([src, np.tanh(field / 0.10), fp3, y, h, y0n, y1n, *region_channels])


def build_corrector(base: int = 8, in_channels: int = IN_CHANNELS, identity_bias: float = 3.0):
    """A deliberately small deterministic 3-D corrector. One absolute occupancy logit per voxel.

    Every voxel passes through the same convolutional path and can change; there is no mask.
    `identity_bias` only shifts the network's STARTING point toward the source occupancy (a fixed
    additive constant derived from input channel 0, not a learned or persisted state), so
    training starts near a no-op and earns every edit -- the whole-volume generalisation of the
    old action head's "begin as KEEP" warm start.
    """
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            groups = 4 if cout % 4 == 0 else 1
            self.net = nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
                nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
            )

        def forward(self, x):
            return self.net(x)

    class OccupancyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = base
            self.e1 = Block(in_channels, c)
            self.e2 = Block(c, 2 * c)
            self.b = Block(2 * c, 4 * c)
            self.pool = nn.MaxPool3d(2)
            self.u2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, 2)
            self.d2 = Block(4 * c, 2 * c)
            self.u1 = nn.ConvTranspose3d(2 * c, c, 2, 2)
            self.d1 = Block(2 * c, c)
            self.out = nn.Conv3d(c, 1, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.identity_bias = identity_bias

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            b = self.b(self.pool(e2))
            d2 = self.d2(torch.cat((self.u2(b), e2), dim=1))
            d1 = self.d1(torch.cat((self.u1(d2), e1), dim=1))
            delta = self.out(d1)[:, 0]
            return delta + self.identity_bias * x[:, 0]

    return OccupancyUNet()


def occupancy_from_logits(logits) -> np.ndarray:
    return np.asarray(logits) > 0


# -------------------------------------------------------------------------------------------------
# Row selection: fixed salted-hash admission, independent of iteration order.
# -------------------------------------------------------------------------------------------------


def _salted_score(salt: str, purpose: str, row: int) -> int:
    """A deterministic pseudo-uniform score in [0, 2**64) for (salt, purpose, row).

    A pure function of its three arguments -- selecting on this score and breaking ties by row id
    is therefore independent of what order candidate rows are iterated in, and independent of
    which OTHER rows happen to be in the candidate pool.
    """
    digest = hashlib.sha256(f"{salt}:{purpose}:{int(row)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _candidate_rows(latents: Path):
    import h5py

    with h5py.File(latents, "r") as h:
        rows = h["row"][:].astype(int)
        held = h["held_out"][:].astype(bool)
        regions = h["region"][:].astype(int)
        heights = h["height_m"][:].astype(float)
        footprints = h["footprint"][:]
    return rows, held, regions, heights, footprints


def _has_non_degenerate_target(real_handle, row: int) -> bool:
    target = np.asarray(real_handle["sdf"][int(row)]) <= 0
    return vertical_extent(target) is not None


def _filter_non_degenerate(rows, real_handle) -> list[int]:
    """Pure filtering logic, directly testable against any small `real_handle`-like object with
    an `sdf` dataset -- no frozen-corpus identity gate. `full_population_rows` is the production
    wrapper that opens the real corpus (with that gate) and fixes the expected count."""
    return [row for row in rows if _has_non_degenerate_target(real_handle, row)]


def full_population_rows(latents: Path, real: Path,
                         expected_n: int = EXPECTED_FULL_POPULATION_N) -> list[int]:
    """#125: "the fixed 714 non-degenerate held-out rows." Computed, not hardcoded, so a corpus
    change that shrinks or grows the held-out set fails loudly instead of silently drifting."""
    rows, held, _, _, _ = _candidate_rows(latents)
    held_rows = sorted(int(r) for r in rows[held])
    with open_real_corpus(real) as h:
        out = _filter_non_degenerate(held_rows, h)
    if len(out) != expected_n:
        raise RuntimeError(
            f"#125: expected {expected_n} non-degenerate held-out rows, found "
            f"{len(out)}. The held-out corpus has drifted; do not silently rebase this number.")
    return out


def select_screen_rows(full_rows: list[int], region_of: dict, salt: str = MANIFEST_SALT) -> list[int]:
    """#125: "96 outcome-blind held-out rows, 32 from each region." A flat per-region draw from
    the full population -- no opportunity/identity filtering, since that would look at the
    target's relationship to the envelope (not an A2/editor outcome, but #125 asks this cohort to
    be chosen with no such shaping at all, unlike the training cohort)."""
    by_region: dict[int, list[int]] = {}
    for row in full_rows:
        by_region.setdefault(region_of[row], []).append(row)
    out: list[int] = []
    for region in sorted(by_region):
        candidates = by_region[region]
        if len(candidates) < SCREEN_PER_REGION:
            raise RuntimeError(
                f"#125: region {region} has only {len(candidates)} held-out rows; needs "
                f"{SCREEN_PER_REGION} for the screening cohort")
        ranked = sorted(candidates, key=lambda r: (_salted_score(salt, "screen", r), r))
        out.extend(ranked[:SCREEN_PER_REGION])
    return sorted(out)


def _select_train_rows_from_handle(candidates: list[int], index_of: dict, regions: np.ndarray,
                                   footprints: np.ndarray, real_handle, salt: str,
                                   opportunity_quota: dict, identity_quota: dict) -> list[int]:
    """Pure selection logic over an already-open real-corpus handle. Directly testable with a
    tiny synthetic handle and tiny quotas; `select_train_rows` is the production wrapper that
    opens the real corpus (with the frozen-identity gate) and fixes the preregistered quotas."""
    ranked = sorted(candidates, key=lambda r: (_salted_score(salt, "train", r), r))
    opportunity: dict[int, list[int]] = {region: [] for region in opportunity_quota}
    identity: dict[int, list[int]] = {region: [] for region in identity_quota}
    for row in ranked:
        li = index_of[row]
        region = int(regions[li])
        if not _has_non_degenerate_target(real_handle, row):
            continue
        target = np.asarray(real_handle["sdf"][row]) <= 0
        is_identity = np.array_equal(envelope_occupancy(footprints[li], target), target)
        bucket, quota = (identity, identity_quota) if is_identity else (opportunity, opportunity_quota)
        if region in bucket and len(bucket[region]) < quota[region]:
            bucket[region].append(row)
        if (all(len(opportunity[r]) >= q for r, q in opportunity_quota.items())
                and all(len(identity[r]) >= q for r, q in identity_quota.items())):
            break

    train = sorted(sum(opportunity.values(), []) + sum(identity.values(), []))
    want = sum(opportunity_quota.values()) + sum(identity_quota.values())
    if len(train) != want:
        # Namespaced, not merged: opportunity and identity share region keys (0 and 1 both
        # appear in both quotas in production), so a single merged dict would silently drop one
        # cohort's shortfall.
        opp_short = {r: opportunity_quota[r] - len(opportunity[r]) for r in opportunity_quota}
        id_short = {r: identity_quota[r] - len(identity[r]) for r in identity_quota}
        raise RuntimeError(f"#125: could only fill {len(train)}/{want} training rows; "
                          f"opportunity shortfall by region {opp_short}, "
                          f"identity shortfall by region {id_short}")
    return train


def select_train_rows(latents: Path, real: Path, salt: str = MANIFEST_SALT,
                      opportunity_quota: dict = TRAIN_OPPORTUNITY_QUOTA,
                      identity_quota: dict = TRAIN_IDENTITY_QUOTA) -> list[int]:
    """#125's training cohort: 326 envelope-correction opportunities split evenly BAG/NRW, plus
    58 byte-identical envelope controls split 19/19/20 BAG/NRW/PLATEAU. Drawn from NON-held-out
    rows only (disjoint from the screen/full population by construction), ranked by a fixed
    salted hash rather than an RNG shuffle so the result never depends on loop or file order.
    """
    rows, held, regions, _, footprints = _candidate_rows(latents)
    index_of = {int(r): i for i, r in enumerate(rows)}
    non_held = sorted(int(r) for r in rows[~held])
    with open_real_corpus(real) as h:
        return _select_train_rows_from_handle(non_held, index_of, regions, footprints, h, salt,
                                             opportunity_quota, identity_quota)


def assert_disjoint_rows(a: list[int], b: list[int], label: str = "row sets") -> None:
    overlap = sorted(set(a) & set(b))
    if overlap:
        raise ValueError(f"#125: {label} must be disjoint; {len(overlap)} rows overlap, "
                         f"e.g. {overlap[:5]}")


# -------------------------------------------------------------------------------------------------
# Manifest: fixed row lists + digests, built once, verified before every later read.
# -------------------------------------------------------------------------------------------------


def build_manifest(latents: Path, real: Path, a2_checkpoint: Path,
                   salt: str = MANIFEST_SALT) -> dict:
    full_rows = full_population_rows(latents, real)
    rows, _, regions, _, _ = _candidate_rows(latents)
    region_of = {int(r): int(g) for r, g in zip(rows, regions)}
    screen_rows = select_screen_rows(full_rows, region_of, salt)
    train_rows = select_train_rows(latents, real, salt)
    assert_disjoint_rows(train_rows, full_rows, "training rows and the held-out population")

    return {
        "version": MANIFEST_VERSION,
        "salt": salt,
        "a2_checkpoint": {"path": str(Path(a2_checkpoint).resolve()),
                          "sha256": A2_CHECKPOINT_SHA256},
        "generation": {"strength": A2_STRENGTH, "steps": A2_STEPS, "guidance": A2_GUIDANCE,
                      "master_seed": A2_MASTER_SEED},
        "real_corpus_path": str(Path(real).resolve()),
        "real_corpus_sha256": FROZEN_CORPUS_SHA256,
        "latents_path": str(Path(latents).resolve()),
        "roles": {
            "train": {"rows": train_rows, "row_list_digest": row_list_digest(train_rows)},
            "screen": {"rows": screen_rows, "row_list_digest": row_list_digest(screen_rows)},
            "full": {"rows": full_rows, "row_list_digest": row_list_digest(full_rows)},
        },
        "sealed_complement_rows": sorted(set(full_rows) - set(screen_rows)),
    }


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def assert_manifest_matches_corpus(manifest: dict, latents: Path, real: Path) -> None:
    """#125: "fail if row-list or content digests differ." Recomputes row lists fresh from the
    corpus and compares digests; never trusts the manifest's own row lists as ground truth."""
    fresh = build_manifest(latents, real, manifest["a2_checkpoint"]["path"], manifest["salt"])
    if fresh["real_corpus_sha256"] != manifest["real_corpus_sha256"]:
        raise ValueError("#125: real corpus identity has changed since this manifest was built")
    for role in ("train", "screen", "full"):
        expected = manifest["roles"][role]["row_list_digest"]
        actual = fresh["roles"][role]["row_list_digest"]
        if actual != expected:
            raise ValueError(f"#125: manifest role {role!r} row-list digest mismatch (expected "
                             f"{expected}, recomputed {actual}); the corpus or selection has "
                             "drifted since this manifest was fixed")


def manifest_command(args) -> None:
    manifest = build_manifest(Path(args.latents), Path(args.real), Path(args.a2), args.salt)
    save_manifest(manifest, Path(args.out))
    print(json.dumps({role: {"n": len(v["rows"]), "digest": v["row_list_digest"]}
                      for role, v in manifest["roles"].items()}, indent=2))
    print(f"[manifest] wrote {args.out}")


# -------------------------------------------------------------------------------------------------
# Cache: one HDF5 file per role, isolated by construction.
# -------------------------------------------------------------------------------------------------


@dataclass
class Example:
    row: int
    region: int
    height_m: float
    y0: int
    y1: int
    footprint: np.ndarray
    source_field: np.ndarray
    source_occ: np.ndarray
    sanitized_source_occ: np.ndarray
    target: np.ndarray
    envelope: np.ndarray


class VoxelCache:
    """Small, single-process reader; HDF5 handles are never shared across DataLoader workers."""

    def __init__(self, path: Path, real: str | Path | None = None, expected_role: str | None = None):
        import h5py

        self.path = Path(path)
        with h5py.File(self.path, "r") as h:
            source = real if real is not None else h.attrs.get("real_corpus_path")
            if source is None:
                raise ValueError("#125: voxel cache has no recorded raw corpus path; "
                                 "pass --real to identify its source, or rebuild the cache")
            self.real_path = Path(source).resolve()
            with open_real_corpus(self.real_path):
                pass
            role = h.attrs.get("role")
            if expected_role is not None and role != expected_role:
                raise RoleIsolationError(
                    f"{self.path} has role {role!r}; this reader requires {expected_role!r}. "
                    "#125: the trainer may open only the training cache.")
            self.role = role
            manifest_digest = h.attrs.get("row_list_digest")
            rows = h["row"][:]
            if manifest_digest is not None and row_list_digest(rows) != manifest_digest:
                raise ValueError(f"{self.path}: row-list digest mismatch against its own "
                                 "recorded manifest digest; rows were added, removed, or "
                                 "reordered after this cache's manifest was fixed")
            self.n = int(len(rows))

    def __len__(self):
        return self.n

    def get(self, i: int) -> Example:
        import h5py

        with h5py.File(self.path, "r") as h:
            return Example(
                row=int(h["row"][i]),
                region=int(h["region"][i]),
                height_m=float(h["height_m"][i]),
                y0=int(h["y0"][i]),
                y1=int(h["y1"][i]),
                footprint=np.asarray(h["footprint"][i], bool),
                source_field=np.asarray(h["source_field"][i], np.float32),
                source_occ=np.asarray(h["source_occ"][i], bool),
                sanitized_source_occ=np.asarray(h["sanitized_source_occ"][i], bool),
                target=np.asarray(h["target_occ"][i], bool),
                envelope=np.asarray(h["envelope_occ"][i], bool),
            )


def open_training_cache(path: Path, real: str | Path | None = None) -> VoxelCache:
    """The only function in this module through which training code may open a cache. Raises
    `RoleIsolationError` on anything not built with `role="train"`.

    #125: "cache manifests and content digests verified before training, so that corpus
    replacement or row drift fails loudly." Reverifies every row's content digest against the
    stored arrays on every open, not just the cache's own self-recorded row-list attribute.
    """
    verify_cache_integrity(Path(path))
    return VoxelCache(path, real=real, expected_role="train")


def _create_cache(path: Path, attrs: dict) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs.update(attrs)
        for name, dtype in (("row", "i8"), ("region", "i1"), ("height_m", "f4"),
                            ("y0", "i2"), ("y1", "i2")):
            h.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(128,), dtype=dtype)
        h.create_dataset("row_digest", shape=(0,), maxshape=(None,), chunks=(128,), dtype="S64")
        h.create_dataset("footprint", shape=(0, RES, RES), maxshape=(None, RES, RES),
                         chunks=(1, RES, RES), compression="gzip", dtype="u1")
        for name, dtype in (("source_field", "f2"), ("source_occ", "u1"),
                            ("sanitized_source_occ", "u1"), ("target_occ", "u1"),
                            ("envelope_occ", "u1")):
            h.create_dataset(name, shape=(0, RES, RES, RES), maxshape=(None, RES, RES, RES),
                             chunks=(1, RES, RES, RES), compression="gzip",
                             compression_opts=1, dtype=dtype)


def _append_cache(path: Path, values: dict) -> None:
    import h5py

    with h5py.File(path, "a") as h:
        n = len(h["row"])
        for name, value in values.items():
            ds = h[name]
            ds.resize(n + 1, axis=0)
            ds[n] = value
        h.flush()


def _finalize_cache(path: Path, expected_row_list_digest: str) -> None:
    """Verify what was actually written matches the manifest's recorded row list before the
    cache is considered usable, and record that digest on the file itself."""
    import h5py

    with h5py.File(path, "r") as h:
        rows = h["row"][:]
    actual = row_list_digest(rows)
    if actual != expected_row_list_digest:
        raise RuntimeError(f"#125: {path} was written with a different row set than its "
                          f"manifest role recorded (expected digest {expected_row_list_digest}, "
                          f"got {actual}); refusing to mark it usable")
    with h5py.File(path, "a") as h:
        h.attrs["row_list_digest"] = actual


def verify_cache_integrity(path: Path) -> dict:
    """Recompute every row's content digest and the file's row-list digest from the stored
    arrays and compare against what was recorded at generation time. Raises on any mismatch."""
    import h5py

    report = {"path": str(path), "n": 0, "mismatched_rows": []}
    with h5py.File(path, "r") as h:
        n = len(h["row"])
        report["n"] = n
        stored_rows = h["row"][:]
        if len(set(int(r) for r in stored_rows)) != n:
            raise ValueError(f"#125: {path} contains duplicate rows")
        recorded_digest = h.attrs.get("row_list_digest")
        if recorded_digest is not None and row_list_digest(stored_rows) != recorded_digest:
            raise ValueError(f"#125: {path} row-list digest no longer matches its own manifest")
        for i in range(n):
            expected = h["row_digest"][i].decode() if isinstance(h["row_digest"][i], bytes) \
                else h["row_digest"][i]
            actual = row_content_digest(
                int(h["row"][i]), int(h["region"][i]), float(h["height_m"][i]),
                int(h["y0"][i]), int(h["y1"][i]), h["footprint"][i],
                h["source_occ"][i], h["target_occ"][i])
            if actual != expected:
                report["mismatched_rows"].append(int(h["row"][i]))
    if report["mismatched_rows"]:
        raise ValueError(f"#125: {path} content-digest mismatch on rows "
                         f"{report['mismatched_rows'][:5]}")
    return report


def assert_disjoint_roles(train_path: Path, screen_path: Path, full_path: Path) -> dict:
    """Isolation, checked on disk rather than assumed: no row id appears in more than one of
    {train, screen}, and every screen row is a member of full."""
    import h5py

    def rows_of(path):
        with h5py.File(path, "r") as h:
            return set(int(r) for r in h["row"][:])

    train_rows, screen_rows, full_rows = rows_of(train_path), rows_of(screen_path), rows_of(full_path)
    assert_disjoint_rows(train_rows, screen_rows, "train and screen")
    assert_disjoint_rows(train_rows, full_rows, "train and full")
    missing = screen_rows - full_rows
    if missing:
        raise ValueError(f"#125: {len(missing)} screen rows are not members of full, e.g. "
                         f"{sorted(missing)[:5]}")
    return {"train_n": len(train_rows), "screen_n": len(screen_rows), "full_n": len(full_rows),
           "screen_subset_of_full": True}


def build_replay_envelope(manifest: dict, editor_checkpoint: Path | None = None) -> dict:
    """#125: "a replay envelope containing source recipe revision, code/model/codec content
    digests, encoder sampling seed, projection settings/noise identity, editor identity/settings,
    runtime identity, and expected source/output digests." Audit and reroll only, within a
    pinned execution envelope -- never a claim of independent editability (#125's Out of Scope).
    """
    import torch

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                          text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_rev = None
    envelope = {
        "git_revision": git_rev,
        "a2_checkpoint": manifest["a2_checkpoint"],
        "generation": manifest["generation"],
        "encoder_seed_formula": "master_seed * 1000003 + row, reseeded immediately before "
                                "every envelope encoding",
        "manifest_row_list_digests": {role: v["row_list_digest"]
                                      for role, v in manifest["roles"].items()},
        "runtime": {"python": sys.version, "torch": torch.__version__,
                   "cuda": torch.version.cuda if torch.cuda.is_available() else None},
    }
    if editor_checkpoint is not None and Path(editor_checkpoint).exists():
        envelope["editor_checkpoint_sha256"] = sha256_file(Path(editor_checkpoint))
    return envelope


def normalize_latent_stats(checkpoint: dict, device: str = "cpu"):
    """The checkpoint's latent mean/std as tensors on `device`, whatever it stored them as.

    `train_vecset.py` persists `ds.mu`/`ds.sd` verbatim, so a checkpoint may carry Python floats,
    NumPy scalars, or tensors depending on the dataset that produced it. The frozen A2 source
    (`vecset_v4_surf` @240k) stores floats, which is why the unguarded `.to(device)` this replaces
    could never have run. `torch.as_tensor` is what the surface-loss probes already use.
    """
    import torch

    for key in ("latent_mu", "latent_sd"):
        if key not in checkpoint:
            raise KeyError(f"#125: checkpoint has no {key}; it cannot denormalise A2 latents")
    return (torch.as_tensor(checkpoint["latent_mu"], device=device),
            torch.as_tensor(checkpoint["latent_sd"], device=device))


def cache_command(args) -> None:
    import h5py
    import torch

    manifest = load_manifest(Path(args.manifest))
    # #125: "fail if row-list or content digests differ" -- reverify the manifest against the
    # corpus now, not just trust the JSON on disk, so corpus replacement or row drift between
    # `manifest` and `cache` time fails loudly instead of silently generating the wrong rows.
    assert_manifest_matches_corpus(manifest, Path(manifest["latents_path"]),
                                   Path(manifest["real_corpus_path"]))
    assert_frozen_a2_checkpoint(manifest["a2_checkpoint"]["path"])
    role_rows = manifest["roles"][args.role]["rows"]
    expected_digest = manifest["roles"][args.role]["row_list_digest"]

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace this prototype cache")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA/ROCm is not visible. Run cache when a GPU is available, or "
                         "explicitly request --device cpu (very slow).")

    _create_cache(out, {
        "purpose": "#125 authentic A2 -> real occupancy whole-volume corrector cache",
        "role": args.role,
        "real_corpus_path": manifest["real_corpus_path"],
        "a2_checkpoint": manifest["a2_checkpoint"]["path"],
        "a2_checkpoint_sha256": manifest["a2_checkpoint"]["sha256"],
        "strength": manifest["generation"]["strength"],
        "steps": manifest["generation"]["steps"],
        "guidance": manifest["generation"]["guidance"],
        "master_seed": manifest["generation"]["master_seed"],
        "salt": manifest["salt"],
    })

    from models.networks.vecset_denoiser import VecsetDenoiser
    from models.networks.vecset_projection import SetSDEdit
    from models.shape_codec import Building, DoraCodec
    from scripts.foundations.dora_roundtrip_probe import load_dora
    from scripts.foundations.eval_massing_arms import blockout_sdf
    from scripts.foundations.baseline_gate_eval import mesh_sdf_surface
    from scripts.foundations.vecset_ceiling_probe import TRUNC, verts_to_world

    device = args.device
    seed = manifest["generation"]["master_seed"]
    ck = torch.load(manifest["a2_checkpoint"]["path"], map_location="cpu", weights_only=False)
    ca = ck["args"]
    net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=ca["width"],
                         depth=ca["depth"], heads=ca["heads"],
                         footprint_res=ck["footprint_res"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    op = SetSDEdit(net, timesteps=ca["timesteps"])
    # `train_vecset.py` writes whatever its dataset exposes, and the frozen A2 checkpoint stores
    # these as Python floats, not tensors -- `.to(device)` raised AttributeError here before the
    # first row could be generated. Every other call site in the repo already normalises.
    mu, sd = normalize_latent_stats(ck, device)
    codec = DoraCodec(load_dora(device))

    with h5py.File(manifest["latents_path"], "r") as lat, open_real_corpus(manifest["real_corpus_path"]) as real:
        latent_row = {int(row): i for i, row in enumerate(lat["row"][:])}
        t0 = time.time()
        for k, row in enumerate(role_rows):
            li = latent_row[row]
            fp = np.asarray(lat["footprint"][li], bool)
            target = np.asarray(real["sdf"][row], np.float32) <= 0
            ext = vertical_extent(target)
            if ext is None:
                raise RuntimeError(f"manifest row {row} has an empty target; the manifest "
                                  "should have excluded this as degenerate")
            y0, y1 = ext
            bo = blockout_sdf(fp, y0, y1)
            verts, faces = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
            if verts is None:
                raise RuntimeError(f"manifest row {row} has an unmeshable envelope")
            codec.reseed(seed * 1000003 + row)
            z0 = (codec.encode(Building(verts=verts_to_world(verts), faces=faces)).float() - mu) / sd
            fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(device)
            height = torch.tensor([float(lat["height_m"][li])], device=device)
            region = torch.tensor([int(lat["region"][li])], device=device)
            zp = op.project(z0, fpt, height, region, strength=manifest["generation"]["strength"],
                            steps=manifest["generation"]["steps"],
                            guidance=manifest["generation"]["guidance"],
                            seed=seed * 1000003 + row)
            with torch.no_grad():
                source_field_f32 = codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]
            # #125: derive occupancy from the float32 field BEFORE storage quantization, and
            # persist it separately from the float16 conditioning field.
            source_occ = source_field_f32 <= 0
            sanitized = sanitize_footprint(source_occ, fp)
            digest = row_content_digest(row, int(lat["region"][li]), float(lat["height_m"][li]),
                                        y0, y1, fp, source_occ, target)
            _append_cache(out, {
                "row": row,
                "region": int(lat["region"][li]),
                "height_m": float(lat["height_m"][li]),
                "y0": y0,
                "y1": y1,
                "row_digest": digest.encode(),
                "footprint": fp.astype(np.uint8),
                "source_field": source_field_f32.astype(np.float16),
                "source_occ": source_occ.astype(np.uint8),
                "sanitized_source_occ": sanitized.astype(np.uint8),
                "target_occ": target.astype(np.uint8),
                "envelope_occ": envelope_occupancy(fp, target).astype(np.uint8),
            })
            if (k + 1) % 10 == 0 or k + 1 == len(role_rows):
                print(f"[cache:{args.role}] {k+1}/{len(role_rows)}  {time.time()-t0:.0f}s", flush=True)
    _finalize_cache(out, expected_digest)
    envelope_path = out.with_suffix(out.suffix + ".replay_envelope.json")
    envelope_path.write_text(json.dumps(build_replay_envelope(manifest), indent=2) + "\n")
    print(f"[cache] wrote authentic {args.role} pairs to {out}")
    print(f"[cache] wrote replay envelope to {envelope_path}")


def verify_command(args) -> None:
    reports = {}
    for role, path in (("train", args.train), ("screen", args.screen), ("full", args.full)):
        if path:
            reports[role] = verify_cache_integrity(Path(path))
    if args.train and args.screen and args.full:
        reports["isolation"] = assert_disjoint_roles(Path(args.train), Path(args.screen), Path(args.full))
    print(json.dumps(reports, indent=2))


# -------------------------------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------------------------------


def _tensor_example(ex: Example, augment: bool = False):
    import torch

    inp = model_input(ex.source_occ, ex.source_field, ex.footprint, ex.height_m, ex.y0, ex.y1,
                      ex.region)
    target = ex.target.astype(np.float32)
    weight = action_weight_map(ex.source_occ, ex.target)
    if augment:
        if np.random.random() < 0.5:
            inp, target, weight = inp[..., ::-1].copy(), target[..., ::-1].copy(), weight[..., ::-1].copy()
        if np.random.random() < 0.5:
            inp, target, weight = inp[:, ::-1].copy(), target[::-1].copy(), weight[::-1].copy()
    return torch.from_numpy(inp), torch.from_numpy(target), torch.from_numpy(weight)


def _batch(examples, augment: bool):
    import torch

    items = [_tensor_example(ex, augment) for ex in examples]
    return (torch.stack([x[0] for x in items]), torch.stack([x[1] for x in items]),
           torch.stack([x[2] for x in items]))


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def train_command(args) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device else _default_device()
    ds = open_training_cache(Path(args.cache), real=args.real)
    if not len(ds):
        raise SystemExit("cache has no training rows")
    model = build_corrector(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    losses = []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        indices = rng.integers(0, len(ds), size=args.batch)
        examples = [ds.get(int(i)) for i in indices]
        x, target, weight = _batch(examples, augment=True)
        x, target, weight = x.to(device), target.to(device), weight.to(device)
        logits = model(x)
        per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"[train {step+1}/{args.steps}] loss={losses[-1]:.5f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    checkpoint = Path(args.out)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "base": args.base,
        "cache": str(Path(args.cache).resolve()),
        "steps": args.steps,
        "seed": args.seed,
        "loss_first": losses[0],
        "loss_last_20": float(np.mean(losses[-20:])),
    }, checkpoint)
    print(f"[train] wrote {checkpoint}")


# -------------------------------------------------------------------------------------------------
# Evaluate: screening (96) and full (714 + sealed 618 complement)
# -------------------------------------------------------------------------------------------------


def error_stratum(error: float, bounds: tuple[float, float] = (0.05, 0.20)) -> str:
    low, high = bounds
    if error <= low:
        return "small"
    if error <= high:
        return "medium"
    return "large"


def _predict(model, ex: Example, device: str) -> np.ndarray:
    import torch

    x, _, _ = _tensor_example(ex, augment=False)
    with torch.no_grad():
        logits = model(x[None].to(device)).cpu().numpy()[0]
    return occupancy_from_logits(logits)


def _surface_realization_diagnostics(sanitized_predicted: np.ndarray,
                                     source_field: np.ndarray) -> dict:
    """Compute both surface-realization arms over the SAME corrected occupancy and report a
    cheap CPU-only diagnostic. Full fields are never embedded in a per-row artifact (a 64^3
    float32 field per row would make the JSON report enormous); the montage/SNE pass (#125's
    "fixed-frame plan/facade/isometric/section montages for human review") is a separate,
    GPU/pytorch3d-dependent step this session did not run -- see `sharp_normal_error_for_arms`.
    """
    control = recover_surface_control(sanitized_predicted)
    narrowband = recover_surface_narrowband(sanitized_predicted, source_field)
    return {
        "control_narrowband_sign_exact": signs_consistent(narrowband, sanitized_predicted),
        "control_narrowband_max_abs_diff": float(np.abs(narrowband - control).max()),
    }


def _criterion2_pass(split: dict, allowance: float = C2_ALLOWANCE) -> bool:
    """#85's criterion 2: spill and uncovered beyond the s* band, each within the allowance.

    Fringe is excluded on purpose -- it is a 64^3 discretisation effect present even when the
    model is right, which CONTEXT.md says is "reported and ignored".
    """
    return bool(split["spill"] <= allowance and split["uncovered"] <= allowance)


def _score_row(ex: Example, predicted_occ: np.ndarray) -> dict:
    from scripts.foundations.eval_massing_arms import footprint_split

    sanitized_source = ex.sanitized_source_occ
    sanitized_predicted = sanitize_footprint(predicted_occ, ex.footprint)
    # #118: every volume comparison is made on SEALED occupancy, so whether a building was
    # modelled hollow or solid cannot reach a score. See `fill_sealed_cavities`.
    source_metrics = sealed_volume_metrics(ex.source_occ, ex.target)
    sanitized_metrics = sealed_volume_metrics(sanitized_source, ex.target)
    predicted_metrics = sealed_volume_metrics(sanitized_predicted, ex.target)
    envelope_metrics = sealed_volume_metrics(ex.envelope, ex.target)
    identity_envelope = bool(np.array_equal(ex.envelope, ex.target))
    c2_predicted = footprint_split(predicted_occ, ex.footprint)
    c2_source = footprint_split(ex.source_occ, ex.footprint)
    e_error = 1.0 - envelope_metrics["vol_iou"]
    a_error = 1.0 - source_metrics["vol_iou"]
    return {
        "row": ex.row,
        "region": ex.region,
        "region_name": REGION_NAMES.get(ex.region, str(ex.region)),
        "source": source_metrics,
        "sanitized_source": sanitized_metrics,
        "predicted": predicted_metrics,
        "envelope": envelope_metrics,
        "vs_input": occupancy_iou(sanitized_predicted, sanitized_source),
        "vs_raw_source": occupancy_iou(sanitized_predicted, ex.source_occ),
        "identity_envelope": identity_envelope,
        "e_stratum": error_stratum(e_error, E_STRATUM_BOUNDS),
        "a_stratum": error_stratum(a_error, A_STRATUM_BOUNDS),
        "collapse_source": source_metrics["missing"] >= COLLAPSE_MISSING,
        "collapse_sanitized": sanitized_metrics["missing"] >= COLLAPSE_MISSING,
        "collapse_predicted": predicted_metrics["missing"] >= COLLAPSE_MISSING,
        "spill_raw_source": c2_source,
        "spill_raw_predicted": c2_predicted,
        # #85's criterion 2, at ADR 0004's fixed s* tolerance and the allowance the owner chose on
        # 2026-08-07. #118 grades it: the owner's "does it follow the footprint" requirement.
        "footprint_c2_pass_predicted": _criterion2_pass(c2_predicted),
        "footprint_c2_pass_source": _criterion2_pass(c2_source),
        "validity_predicted": validity_report(sanitized_predicted, ex.footprint),
        "validity_sanitized_source": validity_report(sanitized_source, ex.footprint),
        "validity_projection": validity_projection_delta(predicted_occ, sanitized_predicted),
        "surface_realization": _surface_realization_diagnostics(sanitized_predicted,
                                                                ex.source_field),
    }


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("#125: cannot summarize an empty row set")

    def med(arm: str, metric: str) -> float:
        return float(np.median([r[arm][metric] for r in rows]))

    opportunity = [r for r in rows if not r["identity_envelope"]]
    identity = [r for r in rows if r["identity_envelope"]]
    # #118: the two surplus axes are never summed -- CONTEXT.md, "not symmetric in consequence":
    # surplus is a building that looks unfinished, `missing` is a building with a trench through
    # it. Each gets its own paired win record against the arm's own A2 input, ties excluded.
    # ⚠️ Scoped to OPPORTUNITY rows, which is the form #118 asks for ("opportunity-row win rate")
    # and #125's own screening gate used. Identity rows -- where the envelope is already the
    # answer -- belong to the non-degradation clauses instead; letting them into the win rate
    # would score "correctly left it alone" against the arm on the corpus's own no-op majority.
    scored = opportunity or rows
    win_extra = paired_win_record(
        [(r["predicted"]["extra"], r["sanitized_source"]["extra"]) for r in scored])
    win_missing = paired_win_record(
        [(r["predicted"]["missing"], r["sanitized_source"]["missing"]) for r in scored])
    win_extra_all_rows = paired_win_record(
        [(r["predicted"]["extra"], r["sanitized_source"]["extra"]) for r in rows])
    beats_envelope_extra = paired_win_record(
        [(r["predicted"]["extra"], r["envelope"]["extra"]) for r in rows])
    delta_vs_sanitized = float(np.median(
        [r["predicted"]["vol_iou"] - r["sanitized_source"]["vol_iou"] for r in rows]))
    strict_win_sanitized = float(np.mean(
        [r["predicted"]["vol_iou"] > r["sanitized_source"]["vol_iou"] for r in rows]))
    strict_win_opportunity = float(np.mean(
        [r["predicted"]["vol_iou"] > r["sanitized_source"]["vol_iou"]
         for r in opportunity])) if opportunity else 0.0
    strict_win_envelope = float(np.mean(
        [r["predicted"]["vol_iou"] > r["envelope"]["vol_iou"] for r in rows]))
    identity_delta = float(np.median(
        [r["predicted"]["vol_iou"] - r["sanitized_source"]["vol_iou"] for r in identity]
    )) if identity else None

    by_region = {}
    for r in rows:
        by_region.setdefault(r["region_name"], []).append(r)
    region_summary = {name: {"n": len(rs), "predicted_vol_iou_median":
                             float(np.median([x["predicted"]["vol_iou"] for x in rs]))}
                      for name, rs in by_region.items()}

    def stratum_summary(key: str) -> dict:
        out = {}
        for label in ("small", "medium", "large"):
            subset = [r for r in rows if r[key] == label]
            if subset:
                out[label] = {"n": len(subset), "predicted_vol_iou_median":
                             float(np.median([x["predicted"]["vol_iou"] for x in subset])),
                             "delta_vs_sanitized_median": float(np.median(
                                 [x["predicted"]["vol_iou"] - x["sanitized_source"]["vol_iou"]
                                  for x in subset]))}
        return out

    n_invalid_predicted = int(sum(not r["validity_predicted"]["valid"] for r in rows))
    return {
        "n": len(rows),
        "n_opportunity": len(opportunity),
        "n_identity_envelope": len(identity),
        "source_vol_iou_median": med("source", "vol_iou"),
        "sanitized_source_vol_iou_median": med("sanitized_source", "vol_iou"),
        "predicted_vol_iou_median": med("predicted", "vol_iou"),
        "envelope_vol_iou_median": med("envelope", "vol_iou"),
        "delta_vs_sanitized_source_median": delta_vs_sanitized,
        "strict_win_rate_vs_sanitized_source": strict_win_sanitized,
        "strict_win_rate_vs_sanitized_source_opportunity": strict_win_opportunity,
        "strict_beats_envelope_rate": strict_win_envelope,
        "collapse_rate_source": float(np.mean([r["collapse_source"] for r in rows])),
        "collapse_rate_sanitized_source": float(np.mean([r["collapse_sanitized"] for r in rows])),
        "collapse_rate_predicted": float(np.mean([r["collapse_predicted"] for r in rows])),
        "vs_input_median": float(np.median([r["vs_input"] for r in rows])),
        "identity_predicted_delta_median": identity_delta,
        "n_predicted_invalid": n_invalid_predicted,
        # ---- #118's registered bar reads these. Every one is a paired difference against the
        # arm's own A2 input, or a rate on the same rows; none is an aggregate-IoU threshold,
        # which #126 forbids this ticket from using.
        "win_extra": win_extra,                       # opportunity rows: what the gate reads
        "win_missing": win_missing,                   # opportunity rows: what the gate reads
        "win_extra_all_rows": win_extra_all_rows,     # reported, never gated
        "n_scored_for_wins": len(scored),
        "beats_envelope_extra": beats_envelope_extra,
        "predicted_extra_median": med("predicted", "extra"),
        "predicted_missing_median": med("predicted", "missing"),
        "envelope_extra_median": med("envelope", "extra"),
        "envelope_missing_median": med("envelope", "missing"),
        "vs_input_median_opportunity": float(np.median(
            [r["vs_input"] for r in opportunity])) if opportunity else 1.0,
        "identity_extra_delta_median": float(np.median(
            [r["predicted"]["extra"] - r["sanitized_source"]["extra"] for r in identity]
        )) if identity else 0.0,
        "identity_missing_delta_median": float(np.median(
            [r["predicted"]["missing"] - r["sanitized_source"]["missing"] for r in identity]
        )) if identity else 0.0,
        "footprint_c2_pass_rate_predicted": float(np.mean(
            [r["footprint_c2_pass_predicted"] for r in rows])),
        "footprint_c2_pass_rate_source": float(np.mean(
            [r["footprint_c2_pass_source"] for r in rows])),
        "invalid_rate_predicted": n_invalid_predicted / len(rows),
        "invalid_rate_source": float(np.mean(
            [not r["validity_sanitized_source"]["valid"] for r in rows])),
        "by_region": region_summary,
        "by_e_stratum": stratum_summary("e_stratum"),
        "by_a_stratum": stratum_summary("a_stratum"),
    }


def wilson_lower_bound(wins: int, decided: int, z: float = 1.6449) -> float:
    """One-sided 95% Wilson score lower bound on a proportion.

    Brown, Cai & DasGupta (2001) recommend Wilson over Wald at these n; the Wald interval's
    coverage is erratic and "common textbook prescriptions regarding its safety are misleading".
    """
    if decided <= 0:
        return 0.0
    p = wins / decided
    denominator = 1.0 + z * z / decided
    centre = (p + z * z / (2 * decided)) / denominator
    half = z * math.sqrt(p * (1 - p) / decided + z * z / (4 * decided * decided)) / denominator
    return centre - half


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def futility_boundary(n: int, p1: float, alpha: float = 0.05) -> int:
    """Simon (1989)'s stage-1 boundary: stop for futility at or below this many wins of `n`.

    The largest r1 whose chance of occurring under the alternative worth detecting is at most
    `alpha`, so screening out an arm that really is at `p1` costs at most `alpha` of the power.
    """
    return max(k for k in range(n + 1) if _binomial_cdf(k, n, p1) <= alpha)


def probability_of_early_stop(n: int, p1: float, p0: float = GATE_NULL_P0,
                              alpha: float = 0.05) -> float:
    """How often `futility_boundary` kills an arm that is genuinely no better than `p0`."""
    return _binomial_cdf(futility_boundary(n, p1, alpha), n, p0)


def paired_win_record(pairs, lower_is_better: bool = True) -> dict:
    """Row-by-row wins of an arm over a reference, with ties excluded from the rate.

    #126 requires #118 to name both the metric and the tie handling, because pooling ties into
    the denominator is what turned a real 60% into a 46% "coin flip" on its own 72 rows: on 17 of
    them the alternative's roof simply *was* the envelope. Ties are not losses. They are excluded
    here and published, so a rate resting on a handful of decided rows is visible rather than
    hidden -- `screening_gate` refuses one below `MIN_DECIDED_COMPARISONS`.
    """
    wins = losses = ties = 0
    for arm, reference in pairs:
        if arm == reference:
            ties += 1
        elif (arm < reference) == lower_is_better:
            wins += 1
        else:
            losses += 1
    decided = wins + losses
    return {"wins": wins, "losses": losses, "ties": ties, "decided": decided,
            "rate": (wins / decided) if decided else 0.0,
            "wilson_lower_bound": wilson_lower_bound(wins, decided)}


def assay_sensitivity(cached_baseline_extra: float,
                      reference: float = A2_PINNED_714_EXTRA_OF_RECORD,
                      tolerance: float = EXTRA_SEED_NOISE_RANGE) -> dict:
    """ICH E10: a comparison is uninterpretable unless the control behaved as expected.

    Every #118 number is a paired difference against the cached A2 baseline, so if the cache did
    not reproduce the shipped operating point the whole run is measuring something else while
    still returning a tidy verdict. The control is the pinned 714, where A2 has a committed
    number; the tolerance is this project's own measured seed-to-seed range for `extra`, which is
    exactly the term #125's per-row reseeding introduces.
    """
    delta = abs(float(cached_baseline_extra) - reference)
    return {"measured": float(cached_baseline_extra), "reference": reference,
            "tolerance": tolerance, "delta": delta,
            "baseline_reproduces_arm_of_record": delta <= tolerance}


def _cleared_futility(record: dict) -> bool:
    """Simon's boundary on the number of DECIDED comparisons, which is the sign test's own n.

    ⚠️ Not on the cohort size. `wins` counts decided rows only, so measuring it against a
    boundary computed on all 250 would silently score every tie as a failure to win -- the exact
    pooling #126 forbids. `enough_decided_comparisons` is what stops a thin decided set from
    buying a cheap boundary instead.
    """
    decided = record.get("decided", 0)
    if decided < 1:
        return False
    return record.get("wins", 0) > futility_boundary(decided, GATE_SCREEN_ALTERNATIVE_P1)


def screening_gate(summary: dict) -> dict:
    """#118's stage-1 futility boundary, evaluated mechanically so prose cannot soften it.

    An AND of clauses, which is Berger (1982)'s intersection-union test: each may be run at the
    same level and the conjunction is still level-alpha, so stacking clauses costs no multiplicity
    correction. (`kill_clauses` is a disjunction and deliberately carries no alpha at all.)

    ⚠️ A measurement that was not taken reads as ABSENT, never as a pass -- `verdict()`'s own
    `--no_form` rule in `train_height_map_generator`. Both human review and Sharp Normal Error
    have to actually have been run, and every SNE artifact this project has ever committed carries
    `views: 0`.
    """
    extra, missing = summary["win_extra"], summary["win_missing"]
    # Both floors must hold: an absolute one so a rate is never read off a handful of rows, and a
    # relative one so an arm that ties on most of the cohort cannot win on a thin decided slice.
    scored = summary.get("n_scored_for_wins", summary["n"])
    min_decided = max(MIN_DECIDED_COMPARISONS, math.ceil(scored * MIN_DECIDED_FRACTION))
    return {
        "screen_n_at_least_250": summary["n"] >= GATE_SCREEN_N,
        "enough_decided_comparisons":
            min(extra["decided"], missing["decided"]) >= min_decided,
        "extra_wins_clear_futility_boundary": _cleared_futility(extra),
        "missing_wins_clear_futility_boundary": _cleared_futility(missing),
        "moved_on_opportunity_rows": summary["vs_input_median_opportunity"] < MAX_VS_INPUT,
        "no_collapse_regression_vs_source":
            summary["collapse_rate_predicted"] <= summary["collapse_rate_source"],
        "identity_rows_not_degraded":
            summary["identity_missing_delta_median"] <= 0.0
            and summary["identity_extra_delta_median"] <= 0.0,
        "footprint_no_worse_than_source":
            summary["footprint_c2_pass_rate_predicted"]
            >= summary["footprint_c2_pass_rate_source"],
        "validity_no_worse_than_source":
            summary["invalid_rate_predicted"] <= summary["invalid_rate_source"],
        "human_review_passed": summary.get("human_review_pass") is True,
        "sharp_normal_error_measured":
            summary.get("sharp_normal_error_views", 0) > 0
            and summary.get("sharp_normal_error_narrowband_no_worse") is True,
    }


def kill_clauses(summary: dict) -> dict:
    """The pre-registered sentences that answer #113 "no" outright.

    Point-estimate facts, not significance tests, and deliberately so. A PASS gate is a
    conjunction and Berger (1982) makes it free; a KILL gate is a DISJUNCTION, where three
    clauses each run at 5% would give roughly a 14% false-kill rate. A KILL has to be a plainly
    visible failure rather than a marginal p-value, so no alpha is attached to any of these.
    """
    dominated_on_extra = summary["predicted_extra_median"] >= summary["envelope_extra_median"]
    dominated_on_missing = summary["predicted_missing_median"] >= summary["envelope_missing_median"]
    return {
        # `train_height_map_generator.verdict`'s `killed_identity`, made two-sided: the arm is
        # dead only if doing nothing was at least as good on BOTH axes. Beating the envelope on
        # one of them is a real, if partial, transform and is not killed here.
        "killed_not_transforming": bool(dominated_on_extra and dominated_on_missing),
        "killed_collapse_over_1nn": summary["collapse_rate_predicted"] > KILL_COLLAPSE_RATE,
    }


def _load_corrector_checkpoint(checkpoint: Path, device: str):
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_corrector(int(ck["base"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def evaluate_model(model, ds: VoxelCache, device: str, report_path: Path,
                   editor_checkpoint_sha256: str | None = None) -> dict:
    if not len(ds):
        raise SystemExit("cache has no rows to evaluate")
    model.eval()
    rows = []
    for i in range(len(ds)):
        ex = ds.get(i)
        predicted_occ = _predict(model, ex, device)
        rows.append(_score_row(ex, predicted_occ))
        if (i + 1) % 20 == 0:
            print(f"[eval] {i+1}/{len(ds)}", flush=True)
    summary = summarize_rows(rows)
    gate = screening_gate(summary)
    summary["screening_gate"] = gate
    summary["screening_gate_pass"] = bool(all(gate.values()))
    # #125: "freeze and hash the editor before opening or scoring full-population targets" --
    # recorded here so `evaluate_full_command` can refuse to score a DIFFERENT checkpoint than
    # the one that actually passed this screen.
    summary["editor_checkpoint_sha256"] = editor_checkpoint_sha256
    artifact = {"summary": summary, "rows": rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[eval] wrote {report_path}")
    return artifact


def evaluate_command(args) -> None:
    device = args.device if args.device else _default_device()
    model = _load_corrector_checkpoint(Path(args.checkpoint), device)
    ds = VoxelCache(Path(args.cache), real=args.real, expected_role="screen")
    editor_sha = sha256_file(Path(args.checkpoint))
    evaluate_model(model, ds, device, Path(args.report), editor_checkpoint_sha256=editor_sha)


def confirmation_gate(summary: dict, screen_summary: dict, buildingworld_verdict: dict) -> dict:
    """#118's stage-2 bar, on #178's own held-out population so its floors apply unchanged.

    `buildingworld_verdict` is the result of `buildingworld_baseline.buildingworld_verdict`
    against a #118 registration -- one built with `include_missing=True`, so both surplus axes
    are quoted from 1-NN per #170's falsification rule. That call carries the per-city and
    per-family floors; what this function adds is the handful of clauses #178's bar has no
    place for, because a carve-only height map cannot fail them.
    """
    envelope = summary["beats_envelope_extra"]
    return {
        "confirmation_n_is_the_178_population": summary["n"] >= GATE_CONFIRMATION_N,
        # #126: name the metric and the tie handling or it is not pre-registered. `extra`, ties
        # excluded and published. The owner set "more decided rows than not"; at n=24,464 the
        # 95% lower bound clears 0.50 at 12,361 wins, so the statistical form costs ~0.5 points.
        "beats_envelope_over_half_of_decided_rows":
            envelope["decided"] >= MIN_DECIDED_COMPARISONS
            and wilson_lower_bound(envelope["wins"], envelope["decided"]) > GATE_NULL_P0,
        "buildingworld_floors_pass": bool(buildingworld_verdict.get("numeric_pass")),
        "no_collapse_regression_vs_screen":
            summary["collapse_rate_predicted"] <= screen_summary["collapse_rate_predicted"],
        "validity_no_worse_than_screen":
            summary["invalid_rate_predicted"] <= screen_summary["invalid_rate_predicted"],
        "footprint_no_worse_than_source":
            summary["footprint_c2_pass_rate_predicted"]
            >= summary["footprint_c2_pass_rate_source"],
        "human_review_passed": summary.get("human_review_pass") is True,
        "sharp_normal_error_measured":
            summary.get("sharp_normal_error_views", 0) > 0
            and summary.get("sharp_normal_error_narrowband_no_worse") is True,
        "assay_sensitivity_holds":
            bool(summary.get("assay_sensitivity", {})
                 .get("baseline_reproduces_arm_of_record")),
    }


def evaluate_full_command(args) -> None:
    device = args.device if args.device else _default_device()
    editor_sha = sha256_file(Path(args.checkpoint))
    screen_report = json.loads(Path(args.screen_report).read_text())
    screen_summary = screen_report["summary"]
    # #125: "Only a passing, frozen editor checkpoint may be evaluated on the fixed 714-building
    # population." Both conditions are checked, not just one: the referenced screen must have
    # passed its own gate, AND it must be a screen of THIS exact checkpoint -- otherwise a
    # checkpoint that never earned a passing screen (or a different one that happens to share a
    # report path) could be scored on the full population.
    if not screen_summary.get("screening_gate_pass"):
        raise SystemExit(f"#125: {args.screen_report} did not pass its screening gate; only a "
                         "passing, frozen editor checkpoint may be evaluated on the full "
                         "714-building population")
    if screen_summary.get("editor_checkpoint_sha256") != editor_sha:
        raise SystemExit(f"#125: {args.checkpoint} (sha256 {editor_sha}) is not the checkpoint "
                         f"{args.screen_report} recorded passing its screen (sha256 "
                         f"{screen_summary.get('editor_checkpoint_sha256')})")
    model = _load_corrector_checkpoint(Path(args.checkpoint), device)

    full_ds = VoxelCache(Path(args.cache), real=args.real, expected_role="full")
    manifest = load_manifest(Path(args.manifest))
    screen_rows = set(manifest["roles"]["screen"]["rows"])

    rows = []
    for i in range(len(full_ds)):
        ex = full_ds.get(i)
        predicted_occ = _predict(model, ex, device)
        rows.append(_score_row(ex, predicted_occ))
        if (i + 1) % 40 == 0:
            print(f"[eval-full] {i+1}/{len(full_ds)}", flush=True)

    full_summary = summarize_rows(rows)
    sealed_rows = [r for r in rows if r["row"] not in screen_rows]
    sealed_summary = summarize_rows(sealed_rows) if sealed_rows else None
    # #118: `confirmation_gate` also needs the #178 per-city/per-family verdict, the montage
    # review and the SNE run, none of which this command produces. They are merged in by the
    # confirmation pass; until then those clauses read false, which is the intended behaviour --
    # a measurement that was not taken is absent, never a pass.
    gate = confirmation_gate(full_summary, screen_summary, {"numeric_pass": False})
    kills = kill_clauses(full_summary)
    artifact = {
        "all_rows": {"summary": full_summary},
        "sealed_complement": {"summary": sealed_summary},
        "confirmation_gate": gate,
        "confirmation_gate_pass": bool(all(gate.values())),
        "kill_clauses": kills,
        "killed": bool(any(kills.values())),
        "rows": rows,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(json.dumps({"all_rows": full_summary, "sealed_complement": sealed_summary,
                     "confirmation_gate": gate, "kill_clauses": kills}, indent=2, default=str))
    print(f"[eval-full] wrote {args.report}")


# -------------------------------------------------------------------------------------------------
# CPU synthetic smoke: validates the data/absolute-state/training plumbing, not the research claim.
# -------------------------------------------------------------------------------------------------


def synthetic_examples(n: int = 12, res: int = 16, seed: int = 0) -> list[Example]:
    """Source/target pairs requiring BOTH an addition and a removal deep inside the mass -- away
    from any roof or surface band the old (now-removed) mask would have defined. With no mask,
    the absolute-occupancy representation and endpoint can express both by construction.
    """
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(n):
        fp = np.zeros((res, res), bool)
        margin = 2 + i % 2
        fp[margin : res - margin, margin : res - margin] = True
        y0, y1 = 2, 12
        source = np.zeros((res, res, res), bool)
        source[:, y0 : y1 + 1, :] = fp[:, None, :]
        target = source.copy()

        # Both edits sit >= 2 voxels inside every box wall (footprint boundary, ground, roof),
        # regardless of which `margin` this row uses -- see the smoke test's surface-distance
        # check, which measures against this pristine box, not the post-edit source.
        mid = res // 2
        lo, hi = margin + 2, res - margin - 2
        target[mid - 1 : mid + 1, y0 + 3 : y0 + 6, lo:hi] = False  # REMOVE: an interior passage

        pz = px = res // 2 - 1
        target[pz : pz + 2, y0 + 2 : y0 + 4, px : px + 2] = True   # ADD: fill an interior pocket
        source[pz : pz + 2, y0 + 2 : y0 + 4, px : px + 2] = False

        source_field = occupancy_to_sdf(source)
        source_field += rng.normal(0, 0.002, source.shape).astype(np.float32)
        sanitized = sanitize_footprint(source, fp)
        examples.append(Example(i, i % N_REGIONS, 8.0 + i % 4, y0, y1, fp, source_field,
                               source, sanitized, target, envelope_occupancy(fp, target)))
    return examples


def smoke_command(args) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Absolute-state reduction identity: for ANY (source, occ) pair, deriving an action lattice
    # and applying it back to source must reproduce occ exactly. This is what "never the stored
    # state" means operationally.
    rng = np.random.default_rng(args.seed)
    for _ in range(20):
        shape = (6, 6, 6)
        s = rng.random(shape) < 0.5
        o = rng.random(shape) < 0.5
        if not np.array_equal(apply_action_to_source(s, derive_action(s, o)), o):
            raise SystemExit("smoke failed: action reduction is not an identity")

    examples = synthetic_examples(seed=args.seed)
    # Whole-volume eligibility: measure distance to the PRISTINE box's own surface (footprint
    # wall, ground, roof) -- not the post-edit source, where a carved pocket trivially creates
    # its own zero-distance "surface" right where the edit is. The constructed add/remove sites
    # must be >= 2 voxels inside every box wall, i.e. outside any near-surface band a masked
    # representation could have defined.
    from scipy import ndimage
    for ex in examples:
        box = envelope_occupancy(ex.footprint, ex.source_occ | ex.target)
        # Unsigned voxel distance to box's own surface: at any single voxel, exactly one of
        # edt(box)/edt(~box) is trivially 0 (that voxel is a "background" point for one of the
        # two arrays), so combine them as a signed difference (as `occupancy_to_sdf` does), not
        # a minimum -- `np.minimum` here would be 0 everywhere by construction.
        box_dist = np.abs(ndimage.distance_transform_edt(~box) - ndimage.distance_transform_edt(box))
        changed = ex.source_occ != ex.target
        if changed.any() and float(box_dist[changed].min()) < 1.5:
            raise SystemExit("smoke failed: constructed edit is not away from the box surface")

    model = build_corrector(base=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for step in range(args.steps):
        x, target, weight = _batch(examples, augment=True)
        logits = model(x)
        per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    before, after = [], []
    model.eval()
    with torch.no_grad():
        for ex in examples:
            predicted = _predict(model, ex, "cpu")
            sanitized_predicted = sanitize_footprint(predicted, ex.footprint)
            before.append(volume_metrics(ex.sanitized_source_occ, ex.target)["vol_iou"])
            after.append(volume_metrics(sanitized_predicted, ex.target)["vol_iou"])
            if (sanitized_predicted & ~np.asarray(ex.footprint, bool)[:, None, :]).any():
                raise SystemExit("smoke failed: sanitizer left spill outside the footprint")

    report = {
        "meaning": "plumbing check only; synthetic examples are not evidence for the A2 hypothesis",
        "steps": args.steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "sanitized_source_iou_median": float(np.median(before)),
        "edited_iou_median": float(np.median(after)),
    }
    print(json.dumps(report, indent=2))
    if not np.isfinite(losses[-1]) or losses[-1] >= losses[0]:
        raise SystemExit("smoke failed: loss did not decrease")
    if report["edited_iou_median"] <= report["sanitized_source_iou_median"]:
        raise SystemExit("smoke failed: the corrector did not improve on the sanitized source "
                         "-- whole-volume interior edits should be learnable on this toy task")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="CPU synthetic plumbing check")
    smoke.add_argument("--steps", type=int, default=80)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(func=smoke_command)

    manifest = sub.add_parser("manifest", help="CPU: fix the 384/96/714 row manifest")
    manifest.add_argument("--out", default=str(DEFAULT_OUT / "manifest.json"))
    manifest.add_argument("--a2", default=str(DEFAULT_A2))
    manifest.add_argument("--latents", default=str(DEFAULT_LATENTS))
    manifest.add_argument("--real", default=str(DEFAULT_REAL))
    manifest.add_argument("--salt", default=MANIFEST_SALT)
    manifest.set_defaults(func=manifest_command)

    cache = sub.add_parser("cache", help="GPU: cache authentic A2 output -> real target pairs")
    cache.add_argument("--role", choices=("train", "screen", "full"), required=True)
    cache.add_argument("--manifest", default=str(DEFAULT_OUT / "manifest.json"))
    cache.add_argument("--out", required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--overwrite", action="store_true")
    cache.set_defaults(func=cache_command)

    verify = sub.add_parser("verify", help="content-digest and role-isolation audit")
    verify.add_argument("--train", default=None)
    verify.add_argument("--screen", default=None)
    verify.add_argument("--full", default=None)
    verify.set_defaults(func=verify_command)

    train = sub.add_parser("train", help="fit the absolute-occupancy corrector on TRAIN only")
    train.add_argument("--cache", default=str(DEFAULT_OUT / "train.h5"))
    train.add_argument("--real", default=None,
                       help="raw corpus source override for relocated caches; defaults to the "
                            "path recorded at cache creation")
    train.add_argument("--out", default=str(DEFAULT_OUT / "editor.pth"))
    train.add_argument("--steps", type=int, default=800)
    train.add_argument("--batch", type=int, default=2)
    train.add_argument("--base", type=int, default=8)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--device", default=None)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=50)
    train.set_defaults(func=train_command)

    evaluate = sub.add_parser("evaluate", help="one-time screen on the 96-row cache")
    evaluate.add_argument("--cache", default=str(DEFAULT_OUT / "screen.h5"))
    evaluate.add_argument("--real", default=None)
    evaluate.add_argument("--checkpoint", default=str(DEFAULT_OUT / "editor.pth"))
    evaluate.add_argument("--report", default=str(DEFAULT_OUT / "screening.json"))
    evaluate.add_argument("--device", default=None)
    evaluate.set_defaults(func=evaluate_command)

    evaluate_full = sub.add_parser("evaluate-full", help="the 714-row confirmation (only after "
                                   "evaluate's gate passes)")
    evaluate_full.add_argument("--cache", default=str(DEFAULT_OUT / "full.h5"))
    evaluate_full.add_argument("--real", default=None)
    evaluate_full.add_argument("--manifest", default=str(DEFAULT_OUT / "manifest.json"))
    evaluate_full.add_argument("--checkpoint", default=str(DEFAULT_OUT / "editor.pth"))
    evaluate_full.add_argument("--screen-report", default=str(DEFAULT_OUT / "screening.json"))
    evaluate_full.add_argument("--report", default=str(DEFAULT_OUT / "full.json"))
    evaluate_full.add_argument("--device", default=None)
    evaluate_full.set_defaults(func=evaluate_full_command)

    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
