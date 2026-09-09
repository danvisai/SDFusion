"""#179 -- the H1b guided-edit completion proxy (#8's falsifiable-proof package).

No gesture, accept/reject, or confidence UI exists anywhere in the repository, and building one is
out of scope for a planning-only map (#1) -- #8's own decision is to test the underlying completion
capability HEADLESSLY: take a held-out building's #10-recovered program, synthesize a "rough
gesture" (a box-shaped add/subtract volume, footprint-column-local, at several sizes and
positions) on top of the current solid, and re-run #10's constrained beam-search fitter on the
gesture-modified target to produce a completed program.

Score exactly two things per completion, nothing else (#8's own pass bar):

1. Does the completion pass #7's finalize-time gate (`finalize_problems`)?
2. Does #144's edit-locality invariant hold on THIS re-fit path -- every op outside the gesture's
   own footprint-column footprint must be provably byte-identical to the pre-gesture program? This
   is a fresh check, not a re-litigation of #144: #144 proved locality on a hand-built mixed
   program under `remove_by_id`; this is the fitter's OWN re-fit path, different code entirely.

No gesture-accuracy/IoU score: there is no real "intended" shape a rough box implies, and scoring
the fitter's own output against an invented target would just declare that output ground truth
(NOVELTY_SURVEY.md risk 5 -- the "synthetic-generator ceiling").

WHY THE LOCALITY CHECK IS SYMMETRIC, NOT JUST "every completed op has a match"
-------------------------------------------------------------------------------
#8's own wording ("every op outside the gesture region must be byte-identical to the pre-gesture
program") reads one-directional -- iterate the COMPLETED program's ops. But the acceptance
criterion's own closing clause is "any op outside that region that changed is a reported failure,
not silently ignored" -- and a one-directional check misses exactly one failure mode: an original
op that the re-fit's own op budget silently drops (leaving that area reverted to blockout, a real
geometric change outside the gesture, with nothing in the completed program left to iterate over
and catch it). So this checks both directions: every OUTSIDE-gesture op signature from the
pre-gesture program must reappear in the completion, and vice versa. A set-difference either way is
a locality failure.

WHY AN OP'S "FOOTPRINT COLUMNS" ARE ITS OWN CONTRIBUTION, NOT ITS DECLARED REGION
------------------------------------------------------------------------------------
A `CutRoof` carries no `region` field at all (its distance-based cut applies over the whole
footprint by construction), so classifying it by a declared region would make it ALWAYS overlap the
gesture, never comparably "outside" anything. `op_column_masks` instead replays the program exactly
as `replay_program` does and records, per op, the columns it ACTUALLY lowered given whatever came
before it -- the height-field analogue of #144's own `_contribution` (voxels one operation's own
application toggles). This treats every op kind on the same footing and is the same "columns it
actually changed" the byte-identical check needs.

Run:  env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/guided_edit_completion_proxy.py
Test: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_guided_edit_completion_proxy.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import containment_problems, finalize_problems, layer_program_to_ops  # noqa: E402
from scripts.foundations.decide_c2_kill_gate import bootstrap_mean_ci  # noqa: E402
from scripts.foundations.recover_massing_programs import (  # noqa: E402
    CARVE_NEEDED, H5, finalise_program, fit_program_beam, height_field, occupancy, replay_program,
)

DEFAULT_BOX_FRACS: Tuple[float, ...] = (0.12, 0.22, 0.32)   # #179 acceptance criterion 1: sizes
DEFAULT_N_POSITIONS = 2                                       # ... and positions
DEFAULT_MODES: Tuple[str, ...] = ("add", "subtract")
DEFAULT_DY = 3                        # voxels the gesture raises/lowers the target by
MIN_GESTURE_CELLS = 6                 # reject a placement this degenerate; try another instead
MAX_PLACEMENT_TRIES = 50
DEFAULT_MAX_OPS_MARGIN = 4            # ops beyond the pre-gesture program's own count (see below)
UNDERSAMPLED_N = 30                   # below this, the CI is reported but flagged, not trusted


# ------------------------------------------------------------------------------------------------
# per-op "footprint columns" -- the columns an op actually changed, not its declared region
# ------------------------------------------------------------------------------------------------

def op_column_masks(fp: np.ndarray, y0: int, y1: int,
                    program: Sequence[dict]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Per op in `program`, the [Z, X] bool mask of columns THAT OP actually lowered, plus the
    final height map -- `replay_program`'s own `return_masks=True` path (#179), so a mask here can
    never silently drift from what the corpus's own replay considers the compiled building. Kept as
    a thin wrapper (return order flipped: masks first) for this module's own call sites below.
    """
    h, masks = replay_program(fp, y0, y1, program, return_masks=True)
    return masks, h


def op_signature(op: dict, mask: np.ndarray) -> tuple:
    """A comparable identity for one op: its own type-specific scalar geometry (rounded against
    float noise across separately-run fits) plus the exact columns it changed (`mask.tobytes()`,
    literal byte equality) -- content, not position, the same spirit as #4's `canonical_form` but
    keyed on the op's actual contribution rather than its declared region, so a `CutRoof` (which
    carries no `region` field at all) compares on the same footing as a `Layer`/`Ramp`.
    """
    kind = op["op"]
    if kind == "Layer":
        params: tuple = (round(float(op["height"]), 6),)
    elif kind == "Ramp":
        params = tuple(round(float(v), 6) for v in op["plane"])
    elif kind == "CutRoof":
        params = (op["kind"], round(float(op["eaves"]), 6), round(float(op["rate"]), 6))
    else:
        raise ValueError(f"unknown operation '{kind}'")
    return (kind, params, mask.tobytes())


# ------------------------------------------------------------------------------------------------
# the synthetic gesture generator
# ------------------------------------------------------------------------------------------------

def _footprint_bbox(fp: np.ndarray) -> Tuple[int, int, int, int]:
    zs, xs = np.nonzero(fp)
    return int(zs.min()), int(zs.max()), int(xs.min()), int(xs.max())


def _place_box(fp: np.ndarray, rng: np.random.Generator, bh: int, bw: int,
               min_cells: int) -> "np.ndarray | None":
    """A random `bh` x `bw` box, centered inside the footprint's own bounding box, intersected with
    `fp` -- footprint-column-local BY CONSTRUCTION, never the whole footprint. Retries on a
    placement too small to be a meaningful gesture (e.g. it landed mostly off the footprint)."""
    Z, X = fp.shape
    z0, z1, x0, x1 = _footprint_bbox(fp)
    for _ in range(MAX_PLACEMENT_TRIES):
        zc = int(rng.integers(z0, z1 + 1))
        xc = int(rng.integers(x0, x1 + 1))
        zlo, zhi = max(0, zc - bh // 2), min(Z, zc - bh // 2 + bh)
        xlo, xhi = max(0, xc - bw // 2), min(X, xc - bw // 2 + bw)
        region = np.zeros_like(fp)
        region[zlo:zhi, xlo:xhi] = True
        region &= fp
        if int(region.sum()) >= min_cells:
            return region
    return None


def synthesize_gestures(fp: np.ndarray, h_prog: np.ndarray, full: int, seed: int,
                        box_fracs: Sequence[float] = DEFAULT_BOX_FRACS,
                        n_positions: int = DEFAULT_N_POSITIONS,
                        modes: Sequence[str] = DEFAULT_MODES, dy: int = DEFAULT_DY,
                        min_cells: int = MIN_GESTURE_CELLS) -> List[dict]:
    """Every (size, position, mode) synthetic box gesture for one building, deterministic from
    `seed` so a run is repeatable (#179 acceptance criterion 1: several sizes and positions).

    `target` is `h_prog` everywhere except `region`, where it is raised (`add`, capped at `full`)
    or lowered (`subtract`, floored at 1) by `dy` voxels -- "add"/"subtract" here name what the
    gesture asks the TARGET to become, not an `EditOp` mode: #10's fitter is subtractive-from-
    blockout regardless of which way the gesture moves the local target (see the module docstring).

    One `region` is drawn per (box_frac, position) and reused for both modes, so add and subtract
    are directly compared at the same size and place. A gesture that would be a no-op (the region
    was already sitting at the clip bound `add`/`subtract` asks for) is dropped -- it tests nothing.
    """
    rng = np.random.default_rng(seed)
    z0, z1, x0, x1 = _footprint_bbox(fp)
    height_cells, width_cells = z1 - z0 + 1, x1 - x0 + 1
    full16 = np.int16(full)
    gestures = []
    for box_frac in box_fracs:
        bh = max(2, int(round(height_cells * box_frac)))
        bw = max(2, int(round(width_cells * box_frac)))
        for position in range(n_positions):
            region = _place_box(fp, rng, bh, bw, min_cells)
            if region is None:
                continue
            for mode in modes:
                target = h_prog.copy()
                if mode == "add":
                    target[region] = np.minimum(full16, h_prog[region] + dy)
                else:
                    target[region] = np.maximum(np.int16(1), h_prog[region] - dy)
                if np.array_equal(target, h_prog):
                    continue
                gestures.append(dict(box_frac=box_frac, position=position, mode=mode,
                                     region=region, target=target.astype(np.int16)))
    return gestures


# ------------------------------------------------------------------------------------------------
# one completion: re-fit, then #7's gate and the locality check
# ------------------------------------------------------------------------------------------------

def run_completion(fp: np.ndarray, y0: int, y1: int, pre_program: Sequence[dict],
                   pre_masks: Sequence[np.ndarray], gesture: dict, max_ops: int, beam: int,
                   branch: int, allowance: float) -> dict:
    """Re-run #10's constrained beam-search fitter against `gesture["target"]`, then check the
    completion against #7's finalize-time gate and #144's locality invariant on this re-fit path.
    """
    res = fp.shape[0]
    raw_ops, h = fit_program_beam(fp, y0, y1, gesture["target"], max_ops=max_ops,
                                  allowance=allowance, beam=beam, branch=branch)
    completed = finalise_program(raw_ops)

    edit_ops = layer_program_to_ops(completed, fp, y0, y1, res=res)
    # #7's own two-layer bar, both checks: `finalize_problems` (syntax/architectural-program) is
    # the ONE ticket #179 names, but CONTEXT.md's own standing note is explicit that "the
    # footprint-adherence claim requires both checks on compiled geometry" -- `containment_problems`
    # is a separate function precisely because #145/#146 never merged them. `occupancy(fp, y0, h)`
    # reuses the height map the fit already produced rather than re-deriving it through the SDF path.
    gate_problems = finalize_problems(edit_ops) + containment_problems(
        occupancy(fp, y0, h), fp, y0, y1)

    new_masks, _ = op_column_masks(fp, y0, y1, completed)
    region = gesture["region"]

    def _outside(mask: np.ndarray) -> bool:
        return not (mask & region).any()

    pre_outside = {op_signature(op, m) for op, m in zip(pre_program, pre_masks) if _outside(m)}
    new_outside = {op_signature(op, m) for op, m in zip(completed, new_masks) if _outside(m)}

    locality_problems = []
    for sig in sorted(new_outside - pre_outside, key=repr):
        locality_problems.append(
            f"a {sig[0]} op outside the gesture's footprint columns appears in the completion "
            f"with no byte-identical match in the pre-gesture program")
    for sig in sorted(pre_outside - new_outside, key=repr):
        locality_problems.append(
            f"a {sig[0]} op outside the gesture's footprint columns from the pre-gesture program "
            f"is missing or changed in the completion")

    return dict(gate_problems=gate_problems, gate_pass=not gate_problems,
               locality_problems=locality_problems, locality_pass=not locality_problems,
               n_ops_completed=len(completed), n_ops_pre=len(pre_program),
               n_ops_outside_gesture_pre=len(pre_outside),
               n_ops_outside_gesture_completed=len(new_outside),
               ops_completed=[o["op"] for o in completed])


# ------------------------------------------------------------------------------------------------
# aggregation -- true N, bootstrap CI on both pass rates, flagged if undersampled (#10's own
# sculpt-strength-sweep convention, adopted project-wide by #8)
# ------------------------------------------------------------------------------------------------

def aggregate(rows: Sequence[dict], n_boot: int = 2000, seed: int = 0,
             undersampled_n: int = UNDERSAMPLED_N) -> dict:
    n = len(rows)
    if n == 0:
        return dict(n=0, undersampled=True, gate_pass_rate=None, gate_ci=None,
                   locality_pass_rate=None, locality_ci=None)
    gate_pt, gate_lo, gate_hi = bootstrap_mean_ci(
        [1.0 if r["gate_pass"] else 0.0 for r in rows], n_boot=n_boot, seed=seed)
    loc_pt, loc_lo, loc_hi = bootstrap_mean_ci(
        [1.0 if r["locality_pass"] else 0.0 for r in rows], n_boot=n_boot, seed=seed)
    return dict(n=n, undersampled=n < undersampled_n,
               gate_pass_rate=gate_pt, gate_ci=[gate_lo, gate_hi],
               locality_pass_rate=loc_pt, locality_ci=[loc_lo, loc_hi])


# ------------------------------------------------------------------------------------------------
# CLI: run over a sample of held-out, carve-needing buildings against the real corpus
# ------------------------------------------------------------------------------------------------

def _load_carve_needing_ids(program_recovery_path: Path) -> Dict[str, dict]:
    art = json.load(open(program_recovery_path))
    return {bid: row for bid, row in art["per_building"].items() if row["n_ops"] > 0}


def report(summary: dict) -> None:
    print(f"\n-- #179 GUIDED-EDIT COMPLETION PROXY (H1b)  n={summary['n']}")
    if summary["n"] == 0:
        print("   no gesture/building pairs produced -- nothing to report")
        return
    if summary["undersampled"]:
        print(f"   ⚠️ undersampled: n={summary['n']} < {UNDERSAMPLED_N}; CI below is reported, "
             f"not trusted")
    gp, (glo, ghi) = summary["gate_pass_rate"], summary["gate_ci"]
    lp, (llo, lhi) = summary["locality_pass_rate"], summary["locality_ci"]
    print(f"   #7 gate (finalize + containment):     {gp:.3f}  95% CI [{glo:.3f}, {ghi:.3f}]")
    print(f"   #144 locality-on-refit pass rate:     {lp:.3f}  95% CI [{llo:.3f}, {lhi:.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program_recovery",
                    default=str(REPO / "execution/artifacts/program_recovery_714.json"),
                    help="#10's own recovered-program artifact; the 'pre-gesture program' and "
                         "'current solid' this ticket builds on")
    ap.add_argument("--n_buildings", type=int, default=30,
                    help="sampled from the carve-needing (n_ops > 0) population only -- an "
                         "already-flat building has an empty pre-gesture program, which makes "
                         "locality vacuous rather than a meaningful test")
    ap.add_argument("--seed", type=int, default=179)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--branch", type=int, default=10)
    ap.add_argument("--allowance", type=float, default=CARVE_NEEDED)
    ap.add_argument("--max_ops_margin", type=int, default=DEFAULT_MAX_OPS_MARGIN,
                    help="the completion's op budget is the pre-gesture program's own op count "
                         "plus this margin, so the re-fit is never budget-starved relative to "
                         "what #10's fitter already spent recreating everything but the gesture")
    ap.add_argument("--box_fracs", nargs="*", type=float, default=list(DEFAULT_BOX_FRACS))
    ap.add_argument("--n_positions", type=int, default=DEFAULT_N_POSITIONS)
    ap.add_argument("--dy", type=int, default=DEFAULT_DY)
    ap.add_argument("--out",
                    default=str(REPO / "execution/artifacts/guided_edit_completion_proxy.json"))
    args = ap.parse_args()

    carve_needing = _load_carve_needing_ids(Path(args.program_recovery))
    rng = np.random.default_rng(args.seed)
    all_ids = sorted(int(b) for b in carve_needing)
    n_pick = min(args.n_buildings, len(all_ids))
    ids = sorted(int(i) for i in rng.choice(all_ids, size=n_pick, replace=False))
    print(f"[ids] {len(ids)} carve-needing buildings of {len(all_ids)} sampled from "
         f"{args.program_recovery}", flush=True)

    rows: List[dict] = []
    t0 = time.time()
    with h5py.File(H5, "r") as g:
        for k, bid in enumerate(ids):
            gt = np.asarray(g["sdf"][bid], np.float32) <= 0
            fp = np.asarray(g["footprint"][bid]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, _gt_target = hf
            full = int(y1 - y0 + 1)
            pre_program = carve_needing[str(bid)]["program"]
            h_prog = replay_program(fp, y0, y1, pre_program)
            pre_masks, _ = op_column_masks(fp, y0, y1, pre_program)

            gestures = synthesize_gestures(fp, h_prog, full, seed=args.seed * 1_000_003 + bid,
                                           box_fracs=tuple(args.box_fracs),
                                           n_positions=args.n_positions, dy=args.dy)
            max_ops = len(pre_program) + args.max_ops_margin
            for gi, gesture in enumerate(gestures):
                res = run_completion(fp, y0, y1, pre_program, pre_masks, gesture, max_ops,
                                     args.beam, args.branch, args.allowance)
                rows.append(dict(id=int(bid), gesture_index=gi, mode=gesture["mode"],
                                 box_frac=gesture["box_frac"], position=gesture["position"],
                                 gesture_cells=int(gesture["region"].sum()), **res))
            if (k + 1) % 5 == 0:
                print(f"  {k + 1}/{len(ids)} buildings, {len(rows)} gesture pairs, "
                     f"{time.time() - t0:.0f}s", flush=True)

    summary = aggregate(rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                             program_recovery=args.program_recovery, n_buildings=len(ids),
                             seed=args.seed, beam=args.beam, branch=args.branch,
                             allowance=args.allowance, max_ops_margin=args.max_ops_margin,
                             box_fracs=args.box_fracs, n_positions=args.n_positions, dy=args.dy),
                   summary=summary, rows=rows),
              open(out, "w"), indent=1)
    print(f"[artifact] {out}", flush=True)
    report(summary)


if __name__ == "__main__":
    main()
