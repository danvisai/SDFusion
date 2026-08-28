"""Where does the paired massing metric put its optimum -- on a real building, or on the envelope?

#126's question, made reproducible. The ticket asserts that pairs of real held-out buildings with
near-identical footprints and heights score a median 3D IoU of **0.674** against each other, against
the blockout's 0.8125, and concludes that "doing nothing beats generating well" -- that the paired
metric's optimum is the extruded footprint rather than a real building. That number was computed
ad-hoc and never committed, so nothing in the repository could reproduce or check it. This does.

WHY THE ARM HAS TO BE CONSTRUCTED CAREFULLY
-------------------------------------------
"A plausible alternative real building" is a *stand-in for a generator's output*, so it is only fair
evidence if it is offered the way a generator would have to offer it. Two constructions are possible
and they do not measure the same thing:

  * `alt_raw`   -- building b's own occupancy, scored against building a's GT. b sits on its own
                   footprint, so up to 10% of the footprint disagreement enters the score. A
                   footprint-conditioned generator cannot make that error: its footprint is given.
  * `alt_exact` -- b's roof PROFILE rendered on a's footprint at a's height (`transplant_height`).
                   Footprint-exact and height-exact by construction, so what it is charged for is
                   exactly the part a generator would have to invent: the shape of the roof.

Reporting only `alt_raw` would charge the alternative for an error the generator being modelled
never makes, and the decision would rest on an artefact of the construction.

AND THE POPULATIONS HAVE TO MATCH
---------------------------------
The ticket's headline compares 0.674, a median over ~250 look-alike pairs, against 0.8125, a median
over the 411 carve-needing buildings. Those are different populations, and this project has been
caught by exactly that before -- map #87 records a "19% surplus reduction" that was 11.8%
like-for-like once the arms were compared on the same buildings. So every arm here is scored on the
**same ordered pairs**, and the blockout is recomputed on that population rather than quoted from
another one.

CPU only. Trains nothing, touches no GPU, reads the corpus and the pinned 714.
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
    COLLAPSE_MISSING, volume_split, footprint_split, fp_iou, vs_input,
)
from scripts.foundations.recover_massing_programs import (     # noqa: E402
    CARVE_NEEDED, H5, SHIP714, height_field, occupancy,
)

# The pair filter, taken verbatim from #126 so the reproduction is of the stated measurement and not
# of a re-tuned one. Both are recorded here so they cannot drift silently between runs.
FP_MATCH = 0.90     # footprint IoU above which two buildings are answers for the same conditioning
H_TOL = 0.05        # and their heights within 5% of the taller

# The envelope first, then the ladder. Named once so the scorecard and the artifact cannot drift.
ARMS = ("blockout", "alt_raw", "alt_aligned", "alt_exact")


def matched_pairs(fps: np.ndarray, extents: np.ndarray,
                  fp_thresh: float = FP_MATCH, h_tol: float = H_TOL):
    """Unordered index pairs of buildings that are plausible answers for the same conditioning.

    Both conditions are on the *conditioning*, never on the massing: two buildings qualify because a
    generator could not tell their inputs apart, not because their roofs agree. That is the whole
    point -- the arm has to be blind to the answer.
    """
    f = np.asarray(fps, bool).reshape(len(fps), -1)
    e = np.asarray(extents, np.float64)
    area = f.sum(1).astype(np.float64)
    inter = (f.astype(np.float32) @ f.astype(np.float32).T).astype(np.float64)
    union = area[:, None] + area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(union), where=union > 0)
    taller = np.maximum(np.maximum(e[:, None], e[None, :]), 1e-9)
    hrel = np.abs(e[:, None] - e[None, :]) / taller
    i, j = np.nonzero(np.triu((iou >= fp_thresh) & (hrel <= h_tol), 1))
    return list(zip(i.tolist(), j.tolist()))


def transplant_height(h_src: np.ndarray, fp_src: np.ndarray, extent_src: int,
                      fp_dst: np.ndarray, extent_dst: int) -> np.ndarray:
    """Building b's roof profile, rendered on a's footprint at a's height.

    Height is a user input (#81) and the footprint is the conditioning, so an alternative is only a
    fair stand-in for a generator's output if it is offered footprint-exact and at the conditioned
    height. What is borrowed is the *shape* of the roof -- the part a generator has to invent.

    A destination cell the source footprint does not cover takes its nearest source value rather
    than 0, so the transplant is a roof everywhere rather than a roof with holes punched where the
    two footprints disagree.
    """
    src, dst = np.asarray(fp_src, bool), np.asarray(fp_dst, bool)
    if not dst.any():
        return np.zeros(dst.shape, np.int16)
    if src.any():
        _, idx = ndimage.distance_transform_edt(~src, return_indices=True)
        prof = np.asarray(h_src, np.float32)[tuple(idx)]
    else:
        prof = np.full(dst.shape, float(extent_src), np.float32)
    scaled = prof * (float(extent_dst) / max(int(extent_src), 1))
    h = np.clip(np.rint(scaled), 1, max(int(extent_dst), 1)).astype(np.int16)
    return np.where(dst, h, 0).astype(np.int16)


def load_height_fields(ids, h5_path: Path = H5):
    """The pinned buildings as the height fields #10 proved they exactly are, plus that proof's guard.

    #10 measured the corpus to be a 64x64 height map on 714/714. This re-checks it per building
    rather than trusting it, and returns the residual so the report can state it instead of assuming
    it: every number below is computed on the reconstruction, so a non-zero residual would mean the
    arms are being scored against something that is not quite GT.
    """
    out, mismatch = [], 0
    with h5py.File(h5_path, "r") as g:
        for b in ids:
            gt = np.asarray(g["sdf"][b], np.float32) <= 0
            fp = np.asarray(g["footprint"][b]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, target = hf
            mismatch += int((occupancy(fp, y0, target) ^ gt).sum())
            out.append(dict(id=int(b), fp=fp, y0=y0, target=target,
                            extent=int(y1 - y0 + 1)))
    return out, mismatch


def envelope_height(b: dict) -> np.ndarray:
    """The blockout's height map: every footprint column filled to the conditioned height."""
    return np.where(b["fp"], np.int16(b["extent"]), 0).astype(np.int16)


def score_pair(a: dict, b: dict) -> dict:
    """Building b offered as the answer for building a's conditioning, as a LADDER of three arms.

    Each rung removes one thing a footprint-conditioned generator is not free to get wrong, so the
    gap between rungs attributes #126's number to a cause instead of leaving it as one aggregate:

        alt_raw      b exactly as it sits in the corpus -- its own base level, its own footprint.
        alt_aligned  b moved to a's base level. Removes grid placement, which is not architecture:
                     the pinned buildings sit at 28 distinct y0, and the pair filter never
                     constrained it.
        alt_exact    b's roof profile on a's footprint at a's height. Removes the footprint
                     disagreement the filter still allows (up to 10%). What is left is roof shape --
                     the only part a generator actually has to invent.
    """
    gt = occupancy(a["fp"], a["y0"], a["target"])
    bo = occupancy(a["fp"], a["y0"], envelope_height(a))
    arms = dict(
        blockout=bo,
        alt_raw=occupancy(b["fp"], b["y0"], b["target"]),
        alt_aligned=occupancy(b["fp"], a["y0"], b["target"]),
        alt_exact=occupancy(a["fp"], a["y0"],
                            transplant_height(b["target"], b["fp"], b["extent"],
                                              a["fp"], a["extent"])),
    )
    row = dict(target=a["id"], alternative=b["id"], dy0=int(a["y0"]) - int(b["y0"]))
    for name, occ in arms.items():
        s = volume_split(occ, gt)
        row[name] = dict(vol_iou=s["vol_iou"], missing=s["missing"], extra=s["extra"],
                         fp_iou=fp_iou(occ, a["fp"]), vs_input=vs_input(occ, bo),
                         spill=footprint_split(occ, a["fp"])["spill"])
    row["carve_needed"] = bool(row["blockout"]["extra"] >= CARVE_NEEDED)
    return row


def _median(rows, arm, key):
    v = [r[arm][key] for r in rows]
    return float(np.median(v)) if v else float("nan")


def summarise(rows, arms=ARMS) -> dict:
    """Medians per arm, plus the quartiles #126 quotes, on whatever population is handed in.

    `beats_envelope` is carried per metric and not just per arm, because that split is the whole
    answer to #126: an arm can be unable to beat the envelope on the aggregate while beating it
    decisively on the decomposition, and "the metric's optimum is the envelope" is then a statement
    about the metric chosen rather than about the arm.
    """
    out = dict(n=len(rows))
    for arm in arms:
        v = np.array([r[arm]["vol_iou"] for r in rows], np.float64)
        out[arm] = dict(
            # the headline, in the order this ticket decided it must be read
            missing=_median(rows, arm, "missing"), extra=_median(rows, arm, "extra"),
            vs_input=_median(rows, arm, "vs_input"),
            collapse_rate=float(np.mean([r[arm]["missing"] >= COLLAPSE_MISSING for r in rows])),
            fp_iou=_median(rows, arm, "fp_iou"), spill=_median(rows, arm, "spill"),
            beats_envelope_extra=compare_to_envelope(rows, arm, "extra", False),
            # diagnostics: this ticket demoted the aggregate, so it may not lead the row
            vol_iou=_median(rows, arm, "vol_iou"),
            p25=float(np.percentile(v, 25)) if len(v) else float("nan"),
            p75=float(np.percentile(v, 75)) if len(v) else float("nan"),
            beats_envelope_iou=compare_to_envelope(rows, arm, "vol_iou", True),
        )
    return out


def compare_to_envelope(rows, arm, key, higher_is_better: bool) -> dict:
    """How `arm` fares against the blockout it is scored beside, counting TIES separately.

    🔑 Ties are not losses and must not be pooled into a win rate. On the carve-needing offers a
    real building's roof simply **is** the envelope on 17 of 72 -- the building genuinely has a flat
    top at full height -- and folding those into the denominator turns a 60% win into a 46%
    "coin flip". Both are reported so neither reading can be quoted alone.
    """
    if not rows:
        return dict(wins=0, losses=0, ties=0, rate=float("nan"), rate_ex_ties=float("nan"))
    wins = losses = ties = 0
    for r in rows:
        a, b = r[arm][key], r["blockout"][key]
        if a == b:
            ties += 1
        elif (a > b) == higher_is_better:
            wins += 1
        else:
            losses += 1
    decided = wins + losses
    return dict(wins=wins, losses=losses, ties=ties, rate=float(wins / len(rows)),
                rate_ex_ties=float(wins / decided) if decided else float("nan"))


def score_population(bl, pairs) -> list:
    """Every matched pair scored in both directions -- each direction is one offered answer."""
    rows = []
    for i, j in pairs:
        rows.append(score_pair(bl[i], bl[j]))
        rows.append(score_pair(bl[j], bl[i]))
    return rows


def threshold_sweep(bl, fps, extents, h_tol: float = H_TOL,
                    thresholds=(0.80, 0.85, 0.90, 0.95)) -> dict:
    """The whole measurement re-run at each footprint threshold, arms and all.

    #126 reports its measurement over **250 pairs**; the filter it states admits far fewer on the
    pinned 714, so the population is swept rather than fixed. The arms are re-scored at each
    threshold and not just counted, because the robustness claim this ticket rests on -- that
    `alt_exact` is insensitive to how the population is drawn, where `alt_raw` is not -- is a claim
    about the ARMS. Publishing only the counts would leave that claim in the same un-committed state
    this ticket criticises #126 for.
    """
    out = {}
    for t in thresholds:
        pairs = matched_pairs(fps, extents, t, h_tol)
        rows = score_population(bl, pairs)
        carve = [r for r in rows if r["carve_needed"]]
        out[f"{t:.2f}"] = dict(n_pairs=len(pairs), all_pairs=summarise(rows),
                               carve_pairs=summarise(carve))
    return out


def report(res: dict) -> None:
    hdr = f"{'arm':12s} {'n':>5} {'miss':>7} {'extra':>7} {'vs_inp':>7} {'collapse':>9} " \
          f"{'>env:xtr':>9} {'fp_iou':>7} | {'(3D IoU)':>9} {'p25':>7} {'p75':>7} {'>env:IoU':>9}"
    note = "   (>env rates EXCLUDE ties; wins/losses/ties are in the artifact)"
    print("the aggregate is right of the bar: #126 demoted it, so it may not head the row")
    print(note)
    for pop, label in (("all_pairs", "every matched pair"),
                       ("carve_pairs", "carve-needing subset of the same pairs")):
        s = res[pop]
        print(f"\n== {label}  (n={s['n']} ordered pairs) ==")
        print(hdr)
        for arm in ARMS:
            a = s[arm]
            print(f"{arm:12s} {s['n']:>5} {a['missing']:>7.4f} {a['extra']:>7.4f} "
                  f"{a['vs_input']:>7.4f} {a['collapse_rate']:>9.4f} "
                  f"{a['beats_envelope_extra']['rate_ex_ties']:>9.3f} {a['fp_iou']:>7.4f} | "
                  f"{a['vol_iou']:>9.4f} {a['p25']:>7.4f} {a['p75']:>7.4f} "
                  f"{a['beats_envelope_iou']['rate_ex_ties']:>9.3f}")
    c = res["corpus"]
    print(f"\n== the populations the ticket compares across ==")
    print(f"blockout 3D IoU over all {c['n']:>3} pinned buildings          {c['blockout_iou_all']:.4f}")
    print(f"blockout 3D IoU over the {c['n_carve']:>3} carve-needing ones   "
          f"{c['blockout_iou_carve']:.4f}")
    print(f"buildings entering at least one matched pair       {c['n_paired']}")
    print(f"height-field reconstruction residual (voxels)      {c['height_field_residual']}")
    print(f"distinct base levels y0 among pinned buildings     {len(c['y0_levels'])}"
          f"  {c['y0_levels']}")
    print(f"matched pairs whose base levels already agree     {c['pairs_same_y0']}")
    if c["threshold_sweep"]:
        print("\n== threshold sweep, carve-needing rows (is the arm sensitive to the draw?) ==")
        print(f"{'fp>=':>6} {'pairs':>6} {'blockout xtr':>13} {'alt_raw xtr':>12} "
              f"{'alt_exact xtr':>14} | {'alt_raw IoU':>12} {'alt_exact IoU':>14}")
        for t, sw in sorted(c["threshold_sweep"].items()):
            k = sw["carve_pairs"]
            if not k["n"]:
                continue
            print(f"{t:>6} {sw['n_pairs']:>6} {k['blockout']['extra']:>13.4f} "
                  f"{k['alt_raw']['extra']:>12.4f} {k['alt_exact']['extra']:>14.4f} | "
                  f"{k['alt_raw']['vol_iou']:>12.4f} {k['alt_exact']['vol_iou']:>14.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_from", default=str(SHIP714),
                    help="replay a pinned id set; default is the pre-registered 714")
    ap.add_argument("--n", type=int, default=0, help="0 = every id in the set")
    ap.add_argument("--fp_match", type=float, default=FP_MATCH)
    ap.add_argument("--h_tol", type=float, default=H_TOL)
    ap.add_argument("--out", default="execution/artifacts/scoring_optimum_714.json")
    ap.add_argument("--sweep", type=int, default=1,
                    help="1 = re-score every arm at each footprint threshold into the artifact")
    args = ap.parse_args()

    ids = [int(i) for i in json.load(open(args.ids_from))["ids"]]
    if args.n:
        ids = ids[:args.n]
    t0 = time.time()
    bl, residual = load_height_fields(ids)
    print(f"[ids] {len(bl)} buildings from {args.ids_from}  ({time.time()-t0:.0f}s)", flush=True)

    fps = np.stack([b["fp"] for b in bl])
    extents = np.array([b["extent"] for b in bl])
    pairs = matched_pairs(fps, extents, args.fp_match, args.h_tol)
    print(f"[pairs] {len(pairs)} unordered matched pairs "
          f"(fp_iou >= {args.fp_match}, height within {args.h_tol:.0%})", flush=True)

    rows = score_population(bl, pairs)

    # the blockout on the populations the ticket quotes across, so the comparison can be made
    # like-for-like instead of across two different sets of buildings
    bo_all = []
    for b in bl:
        gt = occupancy(b["fp"], b["y0"], b["target"])
        bo_all.append(volume_split(occupancy(b["fp"], b["y0"], envelope_height(b)), gt))
    carve = [s for s in bo_all if s["extra"] >= CARVE_NEEDED]

    res = dict(
        meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), ids_from=args.ids_from,
                  gt_h5=str(H5.relative_to(REPO)), fp_match=args.fp_match, h_tol=args.h_tol,
                  n_buildings=len(bl), n_pairs=len(pairs), question="#126"),
        all_pairs=summarise(rows),
        carve_pairs=summarise([r for r in rows if r["carve_needed"]]),
        corpus=dict(
            n=len(bl), n_carve=len(carve),
            blockout_iou_all=float(np.median([s["vol_iou"] for s in bo_all])),
            blockout_iou_carve=float(np.median([s["vol_iou"] for s in carve])) if carve else 0.0,
            n_paired=len({i for p in pairs for i in p}),
            height_field_residual=residual,
            y0_levels=sorted({int(b["y0"]) for b in bl}),
            pairs_same_y0=f"{sum(1 for r in rows if r['dy0'] == 0)}/{len(rows)}",
            threshold_sweep=threshold_sweep(bl, fps, extents, args.h_tol)
            if args.sweep else {},
        ),
        per_pair=rows,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    report(res)
    print(f"\n[artifact] {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
