"""#180 -- H3: test #9's multi-footprint block-coordination bias against REAL, product-sourced
footprint sets, not a synthetic assembly of held-out corpus rows pretending to be a block (#8's own
H3 decision).

Block scenes come from the SAME image-import path the town editor actually uses
(`scripts/server/footprint_image.extract_footprints`/`to_meters`), run on the two bundled REAL-place
samples the product ships (`web/samples/munich_oldtown.png`, `lafayette.png`) -- the third bundled
sample, `synthetic_blocks.png`, is excluded on purpose; it is literally named for being one, and
#180's own bar is "not a synthetic assembly". Each image's own extracted layout is split into a few
spatially-adjacent quadrant clusters -- real adjacency (wherever an extracted building's own
centroid actually falls), never a hand-picked grouping. Each footprint is rasterized to a
`(fp, y0, y1)` triple through `scripts/server/town_generate_service.py`'s own, unmodified
`_footprint_normalization`/`_rasterize_footprint`/`_height_voxel_range` -- the exact conversion the
product already applies to a drawn/imported polygon -- so nothing here reimplements how the product
turns a footprint into a corpus-format grid.

A raster import carries no existing massing decision, so every footprint starts from one
deterministic placeholder program (`initial_program_for`): a single flat `Layer`, cut down from the
blockout by a fixed fraction of the envelope. This is NOT a generation-quality claim -- #181/H1a
owns that question -- it exists purely so #9's bias has real surplus to act on, per
`_current_target`'s own warning that a zero-surplus envelope gives every bias nothing to bite into.

Score exactly what #180 asks, nothing else: does every footprint in the block still pass #7's
finalize-time gate after each of #9's four coordination axes (independently) and the combined
program? Per this ticket's own reconciliation note, "gate" means BOTH `finalize_problems` (all
`commit_block_program` itself calls) AND `containment_problems` (a separate function it does not
call) -- run here against each committed occupancy, so the reported result is #7's full bar, not
half of it under the same name. No coordination-consistency statistic is computed; that reading is
qualitative, left to a human reviewing the rendered carving trace (#147, reused unmodified).

Run:  env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/block_coordination_real_footprints.py
Test: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_block_coordination_real_footprints.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "server"))

from scene.sdf_edit import (  # noqa: E402
    EditableBuilding, commit_block_program, containment_problems, footprint_envelope_sdf,
    layer_program_to_ops, mask_components_rings,
)
from scripts.foundations.carving_trace import render_carving_trace, save_carving_trace  # noqa: E402
from scripts.foundations.recover_massing_programs import BlockProgram  # noqa: E402

SAMPLES_DIR = REPO / "scripts/server/web/samples"
REAL_SAMPLES = ("munich_oldtown", "lafayette")   # excludes the bundled "synthetic_blocks" sample
DEFAULT_HEIGHT = 12.0                            # town_generate_service.TownReq's own default
CUT_FRACTIONS: Tuple[float, float] = (0.15, 0.35)  # the placeholder's two cut depths
MIN_CLUSTER = 3                                  # a block needs at least this many footprints
MAX_CLUSTER = 8                                  # ... and no more than this many
AXES: Tuple[str, ...] = ("height_rhythm", "roof_family", "setback", "azimuth")
# "unbiased" runs FIRST: #180's own acceptance criterion is "any regression relative to that
# footprint's OWN UNCOORDINATED FIT", so the uncoordinated baseline is measured directly (an
# explicit `commit_block_program` re-fit with an empty `FitBias`) rather than inferred from the
# pre-coordination placeholder being #7-clean by construction.
CONDITIONS: Tuple[str, ...] = ("unbiased",) + AXES + ("combined",)
UNDERSAMPLED_SCENES = 5


# ------------------------------------------------------------------------------------------------
# real footprint sourcing -- the product's own extraction + rasterization, reused unmodified
# ------------------------------------------------------------------------------------------------

def rasterize_building(points: np.ndarray, height: float) -> Tuple[np.ndarray, int, int]:
    """World-meter polygon -> `(fp, y0, y1)`, via `town_generate_service.py`'s own per-building
    frame -- the exact conversion the product applies to a drawn/imported footprint, not a
    reimplementation of it."""
    from town_generate_service import (  # local import: heavy module, only needed here
        _footprint_normalization, _height_voxel_range, _rasterize_footprint)

    s, cx, cz = _footprint_normalization(points, height)
    fp = _rasterize_footprint(points, s, cx, cz).astype(bool)
    y0, y1 = _height_voxel_range(height, s)
    return fp, y0, y1


def _cluster_quadrants(centroids: np.ndarray, min_size: int = MIN_CLUSTER,
                       max_size: int = MAX_CLUSTER) -> List[List[int]]:
    """Deterministic spatial clustering: split the extracted layout's own bounding box at its
    median centroid into 2x2 quadrants, keep quadrants with enough buildings to read as a block.
    This groups buildings by real adjacency (wherever their own centroid falls), not by a hand
    picked or synthetic assignment."""
    if len(centroids) == 0:
        return []
    mx, mz = np.median(centroids[:, 0]), np.median(centroids[:, 1])
    quad = (centroids[:, 0] >= mx).astype(int) * 2 + (centroids[:, 1] >= mz).astype(int)
    clusters = []
    for q in range(4):
        idx = np.nonzero(quad == q)[0].tolist()
        if len(idx) >= min_size:
            clusters.append(idx[:max_size])
    return clusters


def load_real_block_scenes(samples: Sequence[str] = REAL_SAMPLES,
                           default_height: float = DEFAULT_HEIGHT,
                           min_size: int = MIN_CLUSTER, max_size: int = MAX_CLUSTER) -> List[dict]:
    """#180 acceptance criterion 1. Returns a list of
    `{scene_id, source, envelopes: {footprint_id: (fp, y0, y1)}}`.
    """
    from footprint_image import extract_footprints, to_meters  # local import, see above

    scenes = []
    for name in samples:
        raw = (SAMPLES_DIR / f"{name}.png").read_bytes()
        polys_px, hw = extract_footprints(raw, simplify_px=1.0)
        scaled = to_meters(polys_px, hw, 200.0)
        points = [local + cen for local, cen in scaled]
        if not points:
            continue
        centroids = np.array([p.mean(axis=0) for p in points])
        for ci, idx in enumerate(_cluster_quadrants(centroids, min_size, max_size)):
            envelopes: Dict[str, Tuple[np.ndarray, int, int]] = {}
            for bi in idx:
                fid = f"{name}_{bi}"
                try:
                    fp, y0, y1 = rasterize_building(points[bi], default_height)
                except Exception:
                    continue                          # a degenerate extracted polygon; skip it
                if not fp.any() or y1 <= y0:
                    continue
                # a real extracted polygon can rasterize to >1 connected region at RES=64 (a thin
                # neck or a self-touching contour some real building outlines actually have) --
                # `initial_program_for`'s own Layer needs ONE polygon (#4/#128's own rule), so a
                # split footprint is excluded here rather than crashing deeper in the pipeline.
                _lab, n_comp = ndimage.label(fp)
                if n_comp != 1:
                    continue
                envelopes[fid] = (fp, y0, y1)
            if len(envelopes) >= min_size:
                scenes.append(dict(scene_id=f"{name}_q{ci}", source=name, envelopes=envelopes))
    return scenes


# ------------------------------------------------------------------------------------------------
# the placeholder starting massing
# ------------------------------------------------------------------------------------------------

def initial_program_for(fp: np.ndarray, y0: int, y1: int,
                        cut_fractions: Tuple[float, float] = CUT_FRACTIONS):
    """TWO flat `Layer`s over a simple split of the footprint -- deliberately NOT one uniform cut.

    ⚠️ A single flat cut was tried first and measured to give #9's `roof_family`/`setback`/
    `azimuth` axes nothing to act on: against a spatially UNIFORM target, the tightest `Ramp` a
    bias could offer degenerates to the exact same flat plane as the `Layer` `_select` would pick
    unbiased anyway (a "tightest plane over a flat target" IS that flat height), so a real re-fit
    ran and reproduced the identical one-op program on all 55/55 real footprints in the first full
    run of this ticket -- a correct but nearly vacuous "nothing broke" result, not a genuine test
    of coordination under a real per-axis choice. Splitting the footprint into two differently-cut
    halves (along whichever of rows/columns the footprint's own bounding box spans more, at its
    own median -- real geometry, not an arbitrary axis) gives every axis actual material: a real
    height difference to test `height_rhythm` against, a real inset boundary between the two
    halves for `setback`, and a real non-flat target for `roof_family`/`azimuth` to plane over.

    A half that is itself disconnected (a concave real footprint split by a straight line) becomes
    one `Layer` entry per connected piece, at the same height -- `_layer_candidates`' own
    convention for exactly this case.
    """
    full = y1 - y0 + 1
    zs, xs = np.nonzero(fp)
    split_on_rows = (zs.max() - zs.min()) >= (xs.max() - xs.min())
    if split_on_rows:
        med = np.median(zs)
        idx = np.arange(fp.shape[0])[:, None]
    else:
        med = np.median(xs)
        idx = np.arange(fp.shape[1])[None, :]
    halves = (fp & (idx <= med), fp & (idx > med))

    entries = []
    for half, frac in zip(halves, cut_fractions):
        if not half.any():
            continue
        cut = max(1, full - max(1, int(round(full * frac))))
        for rings in mask_components_rings(half):
            # `area`/`components` are `_layer_candidates`' own diagnostic fields, never read by
            # `_op_for`/`layer_program_to_ops` -- omitted here rather than cargo-culted in.
            entries.append(dict(op="Layer", height=int(cut), region=[r.tolist() for r in rings]))
    return layer_program_to_ops(entries, fp, y0, y1, res=fp.shape[0])


def _fresh_buildings(envelopes: Dict[str, Tuple[np.ndarray, int, int]]) -> Dict[str, EditableBuilding]:
    """A brand-new `EditableBuilding` per footprint, each on its own placeholder program -- called
    once per CONDITION so the 4 axes and the combined run all start from the same state rather than
    cascading from whichever axis ran before them."""
    buildings = {}
    for fid, (fp, y0, y1) in envelopes.items():
        base = footprint_envelope_sdf(fp, y0, y1, res=fp.shape[0])
        buildings[fid] = EditableBuilding(base, initial_program_for(fp, y0, y1))
    return buildings


# ------------------------------------------------------------------------------------------------
# one block scene, every axis and the combined program
# ------------------------------------------------------------------------------------------------

def _axis_kwargs(envelopes: Dict[str, Tuple[np.ndarray, int, int]], condition: str) -> dict:
    """One concrete value per axis -- #180 tests whether coordination stays valid under a real
    bias, not which bias value is best, so one representative value per axis is the bar, not a
    sweep. `height_rhythm` is drawn from the SCENE'S OWN placeholder-cut heights, over BOTH levels
    the two-half placeholder actually creates (#9's own finding: a fixed rhythm value that isn't
    within tolerance of anything a footprint actually has is a silent no-op, not a fair test) --
    `roof_family="ramp"` is the genuinely interesting probe against a two-level (non-flat) starting
    shape, unlike `"flat"`, which the unbiased search would tend to pick anyway."""
    heights = []
    for fp, y0, y1 in envelopes.values():
        full = y1 - y0 + 1
        for frac in CUT_FRACTIONS:
            heights.append(full - max(1, int(round(full * frac))))
    height_rhythm = float(np.median(heights))
    values = dict(height_rhythm=height_rhythm, roof_family="ramp", setback=2.0, azimuth=45.0)
    if condition == "unbiased":
        return {}
    if condition == "combined":
        return values
    return {condition: values[condition]}


def run_block_scene(scene: dict, trace_dir: "Path | None" = None) -> dict:
    """Run every condition against one block scene. Returns
    `{scene_id, source, n_footprints, footprint_results: {fid: [{condition, gate_pass, problems}]}}`.
    """
    envelopes = scene["envelopes"]
    fids = tuple(envelopes.keys())
    footprint_results: Dict[str, list] = {fid: [] for fid in fids}

    for condition in CONDITIONS:
        buildings = _fresh_buildings(envelopes)
        program = BlockProgram(footprint_ids=fids, **_axis_kwargs(envelopes, condition))
        reports = commit_block_program(program, buildings, envelopes, res=64)

        for fid in fids:
            problems = list(reports[fid])
            fp, y0, y1 = envelopes[fid]
            occ = buildings[fid].to_occupancy(res=fp.shape[0])
            problems += containment_problems(occ, fp, y0, y1)      # see module docstring
            footprint_results[fid].append(dict(condition=condition, problems=problems,
                                               gate_pass=not problems))

        if trace_dir is not None and condition == "combined":
            for fid in fids:
                fp, y0, y1 = envelopes[fid]
                base = footprint_envelope_sdf(fp, y0, y1, res=fp.shape[0])
                trace = render_carving_trace(base, buildings[fid].ops, fp, res=fp.shape[0])
                save_carving_trace(trace, Path(trace_dir) / scene["scene_id"] / fid)

    return dict(scene_id=scene["scene_id"], source=scene["source"], n_footprints=len(fids),
               footprint_results=footprint_results)


# ------------------------------------------------------------------------------------------------
# aggregation -- true N, every regression named, no all-or-nothing rollup
# ------------------------------------------------------------------------------------------------

def aggregate(scene_results: List[dict], n_boot: int = 2000, seed: int = 180) -> dict:
    """`regressions` names every gate failure. Per #180's own acceptance criterion ("any regression
    relative to that footprint's own uncoordinated fit"), a failure is additionally flagged
    `is_regression=True` when that SAME footprint's explicit `"unbiased"` condition passed -- i.e.
    coordination is what broke it, not a pre-existing placeholder problem. A failure under
    `"unbiased"` itself (never observed in this run) is still reported, just not double-counted as
    a coordination regression.

    The overall pass rate carries a bootstrap CI -- #8's own package-wide convention ("every number
    ... states its true, non-inflated N ... with a bootstrap confidence interval"), the same
    `bootstrap_mean_ci` the sibling #179 harness reuses rather than reinventing.
    """
    from scripts.foundations.decide_c2_kill_gate import bootstrap_mean_ci

    n_scenes = len(scene_results)
    per_condition = {c: dict(n_pass=0, n_total=0) for c in CONDITIONS}
    regressions = []
    pass_flags: List[float] = []
    for s in scene_results:
        for fid, rows in s["footprint_results"].items():
            by_condition = {r["condition"]: r for r in rows}
            unbiased_pass = by_condition.get("unbiased", {}).get("gate_pass", False)
            for r in rows:
                pass_flags.append(1.0 if r["gate_pass"] else 0.0)
                per_condition[r["condition"]]["n_total"] += 1
                if r["gate_pass"]:
                    per_condition[r["condition"]]["n_pass"] += 1
                else:
                    regressions.append(dict(
                        scene_id=s["scene_id"], footprint_id=fid, condition=r["condition"],
                        problems=r["problems"],
                        is_regression=(r["condition"] != "unbiased" and unbiased_pass)))

    n_checks = len(pass_flags)
    if n_checks:
        pass_rate, ci_lo, ci_hi = bootstrap_mean_ci(pass_flags, n_boot=n_boot, seed=seed)
    else:
        pass_rate = ci_lo = ci_hi = None
    return dict(n_scenes=n_scenes, n_checks=n_checks, n_pass=int(sum(pass_flags)),
               pass_rate=pass_rate, pass_rate_ci=([ci_lo, ci_hi] if n_checks else None),
               per_condition=per_condition, regressions=regressions,
               undersampled_scenes=n_scenes < UNDERSAMPLED_SCENES)


def report(summary: dict) -> None:
    print(f"\n-- #180 MULTI-FOOTPRINT COORDINATION (H3)  "
         f"scenes={summary['n_scenes']}  footprint-conditions={summary['n_checks']}")
    if summary["undersampled_scenes"]:
        print(f"   ⚠️ undersampled: {summary['n_scenes']} block scenes < {UNDERSAMPLED_SCENES}; "
             f"read as directional, not a generalizable rate")
    lo, hi = summary["pass_rate_ci"]
    print(f"   overall #7 gate pass rate: {summary['pass_rate']:.3f}  "
         f"95% CI [{lo:.3f}, {hi:.3f}]  ({summary['n_pass']}/{summary['n_checks']})")
    for c in CONDITIONS:
        d = summary["per_condition"][c]
        print(f"     {c:<14s} {d['n_pass']}/{d['n_total']}")
    if summary["regressions"]:
        print(f"   {len(summary['regressions'])} named regression(s):")
        for r in summary["regressions"][:10]:
            print(f"     {r['scene_id']}/{r['footprint_id']} [{r['condition']}]: {r['problems'][0]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/block_coordination_real_footprints.json"))
    ap.add_argument("--trace_dir", default=str(REPO / "outputs/block_coordination_traces"))
    ap.add_argument("--no_trace", action="store_true")
    args = ap.parse_args()

    scenes = load_real_block_scenes()
    print(f"[scenes] {len(scenes)} real block scenes "
         f"({', '.join(s['scene_id'] for s in scenes)})", flush=True)

    trace_dir = None if args.no_trace else args.trace_dir
    t0 = time.time()
    results = []
    for s in scenes:
        results.append(run_block_scene(s, trace_dir=trace_dir))
        print(f"  {s['scene_id']}: {len(s['envelopes'])} footprints  "
             f"{time.time() - t0:.0f}s", flush=True)

    summary = aggregate(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), samples=REAL_SAMPLES,
                             default_height=DEFAULT_HEIGHT, cut_fractions=CUT_FRACTIONS,
                             min_cluster=MIN_CLUSTER, max_cluster=MAX_CLUSTER),
                   summary=summary, scenes=results), open(out, "w"), indent=1)
    print(f"[artifact] {out}", flush=True)
    if trace_dir:
        print(f"[traces] {trace_dir}", flush=True)
    report(summary)


if __name__ == "__main__":
    main()
