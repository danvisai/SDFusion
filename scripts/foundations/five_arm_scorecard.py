"""#181 -- H1a: the five-arm autonomous-generation scorecard, against #153's region/tile-stratified
split rather than the legacy pinned 714.

Exactly five arms, nothing more (`blockout`, `1-NN` retrieval, the raw #127/#155 height-map
generator, `fit_decode`, `fit_decode` + #9's coordination bias) -- reused, not reimplemented, via
`train_height_map_generator.py`'s own scoring pipeline (`predict`/`score_arm`/`summarise`/
`verdict`/`fit_decode`), which already computes every metric #181 asks for
(`missing`/`extra`/`vs_input`/collapse/`dl_ops`/`dl_planar_fraction`) and already knows how to build
1-NN and `fit_decode` arms end to end. This module's own job is the THREE things that pipeline does
not already do:

1. **Materialize #153's split.** `stratified_split.make_split` returns an in-memory label array,
   not a persisted id list -- #153 deliberately left it that way (see that module's own docstring:
   dozens of already-closed tickets are pinned to the OLD population, and this is a new,
   prospective split, not a replacement). `materialize_split` runs it and writes the test/val ids
   `--ids_from`/`--bank_exclude_ids` need.

2. **Audit train/test overlap, per this ticket's own reconciliation note.** Existing checkpoints
   were trained under the OLD split; #153's new test population overlaps it heavily (the
   reconciliation note itself cites ~96-97%). `audit_overlap` measures this directly rather than
   trusting the cited figures, so the scorecard's own report can state plainly which arms this is a
   genuine held-out measurement for (`blockout`, `1-NN` once its bank is rebuilt) and which are
   retrospective (the trained generator and `fit_decode`, since "no new training" means the
   checkpoint itself cannot be un-trained on the ~97% of #153's test set it likely already saw).

3. **Sample and orchestrate.** #153's test set is ~7,120 rows -- an order of magnitude past what
   this map's headline comparisons have run at (the pinned population is 714/411). Scoring every
   arm (`fit_decode`'s beam search and the per-arm description-length fit both cost real time per
   building) at that scale is a multi-hour run; a deterministic sample keeps this tractable while
   still clearing a real statistical bar, disclosed explicitly rather than silently shrunk.

No new arm is added: this module never touches `predict`/`fit_program_beam`/`score_arm`'s own logic,
it only supplies the id lists and shells out to the existing, unmodified CLI.

Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
      scripts/foundations/five_arm_scorecard.py --ckpt weights/massing-heightmap/heightmap_ce.pt
Test: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_five_arm_scorecard.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, open_real_corpus  # noqa: E402
from scripts.foundations.recover_massing_programs import CARVE_NEEDED, H5  # noqa: E402
from scripts.foundations.stratified_split import make_split  # noqa: E402
from scripts.foundations.train_height_map_generator import CACHE, build_cache  # noqa: E402

VAL_FRAC = 0.02
TEST_FRAC = 0.02
SPLIT_SEED = 0                    # #153's own published default
SAMPLE_SEED = 181
SAMPLE_N = 1000                   # see module docstring, point 3
UNDERSAMPLED_N = 300
CI_N_BOOT = 2000

ARTIFACTS = REPO / "execution/artifacts"

# #130's own oracle-fed named-baseline numbers, on the legacy pinned 411 carve-needing rows -- quoted,
# never recomputed, exactly this project's own `REFERENCE`-dict convention
# (`train_height_map_generator.py`'s own docstring: "Quoted, never recomputed... re-deriving them
# here would risk a second, silently different number for an arm that already has one"). Source:
# docs/wayfinding/solid-first-subtractive-modeling/130-baselines-diffusion-curriculum.md.
ORACLE_REFERENCE = {
    "ceiling (program recovery, sees GT)": dict(
        extra=0.0035, missing=0.0, vs_input=0.8226, collapse_rate=0.0, dl_ops=2.0,
        dl_planar_fraction=0.50, population="legacy pinned 411 carve-needing, not #153"),
    "ArcPro/CoMa (flatten_ramps, oracle-fed)": dict(
        extra=0.0528, missing=0.0, vs_input=0.8847, collapse_rate=0.0024, dl_ops=1.0,
        dl_planar_fraction=0.00, population="legacy pinned 411 carve-needing, not #153"),
}


# ------------------------------------------------------------------------------------------------
# 1. materialize #153's split
# ------------------------------------------------------------------------------------------------

def materialize_split(h5_path: Path = H5, val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC,
                      seed: int = SPLIT_SEED) -> dict:
    """Runs #153's `make_split` and returns real.h5 ROW INDICES for val/test -- the same id space
    `cache["row"]`/every other arm on this map already uses.

    Restricted to the frozen NL/DE/JP prefix (`FROZEN_SPLIT_N_TOTAL` rows) -- #177's BuildingWorld
    rows appended after it (`source_id == -1`) have no whole-tile structure `tile_key` can use
    (their `bag_id`s are `'<City>#<member>'`, so every BuildingWorld row from one city would
    collapse onto a single giant "tile"), and #153's own split was scoped to the historical
    population it was built and recorded against, not to whatever `real.h5` has grown to since.
    """
    with open_real_corpus(h5_path) as f:
        source_id = f["source_id"][:FROZEN_SPLIT_N_TOTAL]
        bag_id = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
    split, report = make_split(source_id, bag_id, val_frac, test_frac, seed)
    return dict(test_ids=np.nonzero(split == "test")[0], val_ids=np.nonzero(split == "val")[0],
               n_total=len(split), report=report)


def filter_to_cache(ids: Sequence[int], cache: dict) -> np.ndarray:
    """Ids not present (or not `ok`) in the height-field cache are dropped -- 153 of real.h5's
    35,776 rows fail `vecset_latents.h5` encoding entirely; see the module docstring."""
    row_to_idx = {int(r): i for i, r in enumerate(cache["row"])}
    return np.array(sorted(int(i) for i in ids
                          if int(i) in row_to_idx and cache["ok"][row_to_idx[int(i)]]), dtype=np.int64)


def sample_ids(ids: Sequence[int], n: int, seed: int = SAMPLE_SEED) -> np.ndarray:
    ids = np.asarray(sorted(int(i) for i in ids))
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=n, replace=False))


# ------------------------------------------------------------------------------------------------
# 2. audit train/test overlap against the OLD split
# ------------------------------------------------------------------------------------------------

def audit_overlap(test_ids: Sequence[int], cache: dict, n_total: int) -> dict:
    """#181's own reconciliation note, measured directly rather than trusted from the cited
    figures. `bag3d_train` mirrors `datasets/bag3d_dataset.py`'s own split formula exactly (seed 0,
    `perm[2*n_val:]`); `heightmap_pool` is `cache["held"]==0`, the height-map generator's own
    training-eligible pool (BEFORE its own per-checkpoint train/val split, which no saved
    checkpoint records well enough to reproduce -- see the module docstring).
    """
    n_val = max(1, int(0.02 * n_total))
    perm = np.random.default_rng(0).permutation(n_total)
    bag3d_train = set(int(i) for i in perm[2 * n_val:])
    heightmap_pool = set(int(r) for r, h in zip(cache["row"], cache["held"]) if h == 0)
    test_set = set(int(i) for i in test_ids)
    return dict(n_test=len(test_set),
               n_overlap_bag3d_train=len(test_set & bag3d_train),
               n_overlap_heightmap_pool=len(test_set & heightmap_pool))


# ------------------------------------------------------------------------------------------------
# 3. orchestrate: shell out to the existing, unmodified scoring CLI
# ------------------------------------------------------------------------------------------------

def write_ids_json(ids: Sequence[int], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ids": [int(i) for i in ids]}, open(path, "w"))
    return path


def run_scoring_cli(ids_from: Path, exclude_ids: Path, ckpt: Path, out: Path,
                    python: str = str(REPO / "sdfusion/bin/python")) -> None:
    cmd = [python, str(REPO / "scripts/foundations/train_height_map_generator.py"),
          "--ids_from", str(ids_from), "--ckpt", f"ce={ckpt}", "--median_decode",
          "--fit_decode", "--fit_decode_roof_family", "flat",
          "--bank_exclude_ids", str(exclude_ids), "--montage", "0", "--out", str(out)]
    # matches this repo's own `env -u LD_PRELOAD -u LD_LIBRARY_PATH` run convention (REPRODUCING.md)
    # by actually removing the keys, not blanking them -- an unset var and an empty one are not the
    # same thing to every consumer of `LD_LIBRARY_PATH`, even though glibc treats an empty
    # `LD_PRELOAD` as zero entries.
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


# ------------------------------------------------------------------------------------------------
# post-processing: bootstrap CI, the curated five-arm + oracle table
# ------------------------------------------------------------------------------------------------

# The five arms named by #181, mapped to the names `train_height_map_generator.py`'s own CLI
# produces under `--ckpt ce=... --median_decode --fit_decode --fit_decode_roof_family flat` --
# #127's served arm decodes the posterior MEDIAN, not the mode (CONTEXT.md's own "argmax ->
# posterior median" finding), so `ce_median` (not plain `ce`) is "the raw generator" here.
FIVE_ARMS = {
    "blockout": "blockout",
    "1-NN retrieval": "nn_retrieval",
    "raw height-map generator (#127/#155)": "ce_median",
    "fit_decode (#155)": "ce_median_fit",
    "fit_decode + #9 coordination bias": "ce_median_fit_flat",
}


def carve_ids(per_building: Dict[str, list]) -> set:
    return {r["id"] for r in per_building["blockout"] if r["blockout_extra"] >= CARVE_NEEDED}


def bootstrap_median_ci(values: Sequence[float], n_boot: int = CI_N_BOOT, seed: int = 0,
                        ci: float = 0.95) -> Tuple[float, float, float]:
    """A percentile bootstrap CI for the MEDIAN -- `decide_c2_kill_gate.bootstrap_mean_ci`'s own
    pattern, but over the median rather than the mean, because `extra`/`missing` are reported as
    medians EVERYWHERE ELSE on this map (`summarise()`'s own `med()`, #10's "median residual
    extra", every prior arm-comparison doc). Reusing the mean-CI helper here would silently report
    a different statistic than the plain table above it in the same artifact.
    """
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_medians = np.median(arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))], axis=1)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lo, hi = np.percentile(boot_medians, [lo_pct, hi_pct])
    return float(np.median(arr)), float(lo), float(hi)


def arm_ci(per_building: Dict[str, list], arm_key: str, ids: set, metric: str = "extra",
          n_boot: int = CI_N_BOOT, seed: int = SAMPLE_SEED) -> Tuple[float, float, float]:
    vals = [r[metric] for r in per_building[arm_key] if r["id"] in ids]
    return bootstrap_median_ci(vals, n_boot=n_boot, seed=seed)


def build_scorecard(res: dict) -> dict:
    """The curated report: exactly #181's five arms plus #130's oracle references, each carrying
    the true carve-needing N with a bootstrap CI, and #8's own per-axis verdict shape (volume/safety
    vs. architectural form) rather than one collapsed word."""
    per_building = res["per_building"]
    carve = carve_ids(per_building)
    n_carve = len(carve)

    rows = {}
    for label, key in FIVE_ARMS.items():
        summ = res["arms"][key]["carve"]
        pt, lo, hi = arm_ci(per_building, key, carve, "extra")
        rows[label] = dict(
            n=n_carve, extra=pt, extra_ci=[lo, hi], missing=summ["missing"],
            vs_input=summ.get("vs_input"), collapse_rate=summ["collapse_rate"],
            dl_ops=summ.get("dl_ops"), dl_planar_fraction=summ.get("dl_planar_fraction"))

    verdict = res["verdict"]
    axis_verdict = {}
    for label, key in FIVE_ARMS.items():
        if key not in verdict:
            continue                                   # blockout/nn_retrieval: NOT_GENERATORS
        v = verdict[key]
        axis_verdict[label] = dict(
            volume_safety=("PASS" if v["pass"] else "KILL"),
            architectural_form=(
                "PASS" if v.get("form_planar_over_bar") else
                "KILL" if "form_planar_over_bar" in v else "not measured"))

    return dict(n_carve=n_carve, undersampled=n_carve < UNDERSAMPLED_N,
               five_arms=rows, oracle_reference=ORACLE_REFERENCE, axis_verdict=axis_verdict)


def report(scorecard: dict, overlap: dict) -> None:
    print(f"\n-- #181 FIVE-ARM SCORECARD (H1a), #153's split  n_carve={scorecard['n_carve']}")
    if scorecard["undersampled"]:
        print(f"   ⚠️ undersampled: {scorecard['n_carve']} < {UNDERSAMPLED_N}")
    print(f"   overlap audit: {overlap['n_overlap_bag3d_train']}/{overlap['n_test']} of #153's test "
         f"set was in the OLD Bag3d training split; {overlap['n_overlap_heightmap_pool']}/"
         f"{overlap['n_test']} was in the height-map generator's own OLD training-eligible pool")
    for label, r in scorecard["five_arms"].items():
        lo, hi = r["extra_ci"]
        print(f"   {label:<40s} extra {r['extra']:.4f} CI [{lo:.4f},{hi:.4f}]  "
             f"missing {r['missing']:.4f}  collapse {r['collapse_rate']:.4f}"
             + (f"  dl_ops {r['dl_ops']:.2f}  planar {r['dl_planar_fraction']:.2f}"
                if r["dl_ops"] is not None else ""))
    print("   per-axis verdict:")
    for label, v in scorecard["axis_verdict"].items():
        print(f"     {label:<40s} volume/safety={v['volume_safety']:<5s} "
             f"form={v['architectural_form']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="an EXISTING checkpoint (#181: no new training)")
    ap.add_argument("--sample_n", type=int, default=SAMPLE_N)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--out", default=str(ARTIFACTS / "five_arm_scorecard_153.json"))
    ap.add_argument("--raw_out", default=str(ARTIFACTS / "height_map_generator_153_test.json"),
                    help="where train_height_map_generator.py's own full artifact lands")
    ap.add_argument("--skip_scoring", action="store_true",
                    help="reuse --raw_out from a previous run instead of re-scoring")
    args = ap.parse_args()

    t0 = time.time()
    split = materialize_split()
    cache = build_cache(CACHE)

    test_ids_full = filter_to_cache(split["test_ids"], cache)
    val_ids_full = filter_to_cache(split["val_ids"], cache)
    print(f"[split] {len(split['test_ids'])} raw test ids -> {len(test_ids_full)} present in the "
         f"height-field cache", flush=True)

    overlap = audit_overlap(test_ids_full, cache, split["n_total"])

    test_sample = sample_ids(test_ids_full, args.sample_n, args.seed)
    print(f"[sample] {len(test_sample)} of {len(test_ids_full)} test ids "
         f"(seed {args.seed})", flush=True)

    ids_from = write_ids_json(test_sample, ARTIFACTS / "stratified_split_153_test_sample.json")
    exclude_ids = write_ids_json(np.concatenate([test_ids_full, val_ids_full]),
                                 ARTIFACTS / "stratified_split_153_exclude.json")

    if not args.skip_scoring:
        run_scoring_cli(ids_from, exclude_ids, args.ckpt, Path(args.raw_out))
    res = json.load(open(args.raw_out))

    scorecard = build_scorecard(res)
    out = dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), ckpt=str(args.ckpt),
                         sample_n=len(test_sample), n_test_full=len(test_ids_full),
                         seed=args.seed, raw_artifact=str(args.raw_out)),
              overlap=overlap, scorecard=scorecard)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[artifact] {args.out}  ({time.time() - t0:.0f}s total)", flush=True)
    report(scorecard, overlap)


if __name__ == "__main__":
    main()
