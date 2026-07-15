"""Ticket 13 addendum: diagnose why the monolith arm's 73/277 near-empty generations
(gen_occ_frac < 1e-4, ticket 13's own threshold) collapse the way they do. Discovered during a
manual 2AFC pilot in ticket 17's work, not something ticket 13's original run checked.

Finding this reproduces: all 73 near-empty outputs are byte-IDENTICAL to each other -- not
independently collapsed noise, one fixed output. This traces cleanly to an empty (zero-occupancy)
coarse conditioning input: `GaussianDiffusion.ddim_sample`'s own contract is "equal inputs + equal
seed -> bit-identical output" (models/monolith_diffusion.py), and `generate_monolith_arm.py` passes
the same fixed `seed=0` for every building, so every empty-coarse-input building gets the identical
deterministic trajectory. The training data has the same pattern -- a meaningful fraction of
`train_100` pairs also have empty coarse input -- so this is not an eval-time surprise; the model
was repeatedly trained on this exact degenerate case under unweighted MSE loss.

Separately: of the 73, most have real, nonzero (if very sparse) ground-truth geometry -- their
massing signal simply doesn't survive `low_pass_sdf`'s downsample-to-s*-then-upsample round trip.
A small minority have literally zero real ground-truth voxels -- genuinely broken data, worth
excluding from the corpus, but NOT representative of the other sparse-but-real cases (see the
ticket 13 Answer addendum for the full project-owner discussion this came out of, including why a
broad "filter out sparse buildings" fix was rejected).

Out: execution/artifacts/monolith_collapse_diagnosis.json
Run:  TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        ./sdfusion/bin/python scripts/foundations/diagnose_monolith_collapse.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations", "datasets"):
    sys.path.insert(0, str(REPO / _p))

from transform_vs_noise import git_provenance  # noqa: E402

NEAR_EMPTY_THRESHOLD = 1e-4  # ticket 13's own threshold for "near-empty generation"


def classify_by_occupancy(per_building_rows, threshold=NEAR_EMPTY_THRESHOLD):
    """Pure: split monolith_arm_v1 manifest rows into (near_empty_ids, non_empty_ids) by
    `gen_occ_frac`, matching ticket 13's own `localize_monolith_failures` bucketing."""
    near_empty = [r["building"] for r in per_building_rows if r["gen_occ_frac"] < threshold]
    non_empty = [r["building"] for r in per_building_rows if r["gen_occ_frac"] >= threshold]
    return near_empty, non_empty


def group_by_hash(id_to_bytes):
    """Pure: {building_id: raw_bytes} -> {md5_hex: [building_id, ...]}. Buildings sharing a
    hash have byte-identical arrays."""
    groups: dict = {}
    for bid, b in id_to_bytes.items():
        h = hashlib.md5(b).hexdigest()
        groups.setdefault(h, []).append(bid)
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/monolith_collapse_diagnosis.json"))
    a = ap.parse_args()

    import render_facades as rf
    from monolith_pair_dataset import MonolithPairDataset

    monolith_manifest = json.load(open(REPO / "data/monolith_arm_v1/manifest.json"))
    near_empty_ids, non_empty_ids = classify_by_occupancy(monolith_manifest["per_building"])
    print(f"[*] {len(near_empty_ids)} near-empty / {len(non_empty_ids)} non-empty outputs "
          f"(threshold gen_occ_frac < {NEAR_EMPTY_THRESHOLD})")

    grids_dir = Path(monolith_manifest["grids_dir"])
    grid_bytes = {bid: np.load(grids_dir / f"{bid}.npy").tobytes() for bid in near_empty_ids}
    hash_groups = group_by_hash(grid_bytes)
    print(f"[*] {len(near_empty_ids)} near-empty outputs collapse into {len(hash_groups)} "
          f"distinct grid(s)")

    print("[*] checking real ground-truth voxel counts for near-empty-output buildings...")
    real_voxel_counts = {}
    for bid in near_empty_ids:
        grid = np.asarray(rf.load_buildingnet_sdf(bid, working_res=rf.WORKING_RES, device="cpu"))
        real_voxel_counts[bid] = int((grid <= 0).sum())
    zero_voxel_ids = [bid for bid, n in real_voxel_counts.items() if n == 0]
    sparse_but_real_ids = [bid for bid, n in real_voxel_counts.items() if n > 0]

    print("[*] checking coarse-input occupancy for the eval population...")
    test_ids = json.load(open(REPO / "data/splits_v1/test.json"))
    eval_ds = MonolithPairDataset(test_ids, working_res=96, augment=False, device="cpu")
    eval_id_to_idx = {bid: i for i, bid in enumerate(test_ids)}
    eval_coarse_occ = {}
    for bid in near_empty_ids + non_empty_ids:
        item = eval_ds[eval_id_to_idx[bid]]
        eval_coarse_occ[bid] = float((item["coarse"][0].numpy() <= 0).mean())
    near_empty_coarse_occ = [eval_coarse_occ[b] for b in near_empty_ids]
    non_empty_coarse_occ = [eval_coarse_occ[b] for b in non_empty_ids]

    print("[*] checking coarse-input occupancy for train_100 (this is the slow part)...")
    train_ids = json.load(open(REPO / "data/splits_v1/train_100.json"))
    train_ds = MonolithPairDataset(train_ids, working_res=96, augment=False, device="cpu")
    train_empty_coarse = 0
    train_coarse_occ = []
    for i in range(len(train_ids)):
        occ = float((train_ds[i]["coarse"][0].numpy() <= 0).mean())
        train_coarse_occ.append(occ)
        if occ == 0.0:
            train_empty_coarse += 1

    diagnosis = dict(
        n_near_empty_outputs=len(near_empty_ids), n_non_empty_outputs=len(non_empty_ids),
        near_empty_threshold=NEAR_EMPTY_THRESHOLD,
        n_distinct_grids_among_near_empty=len(hash_groups),
        all_near_empty_are_byte_identical=(len(hash_groups) == 1),
        real_ground_truth=dict(
            n_zero_voxel_broken=len(zero_voxel_ids), n_sparse_but_real=len(sparse_but_real_ids),
            zero_voxel_ids=zero_voxel_ids,
            sparse_but_real_voxel_range=[min(real_voxel_counts[b] for b in sparse_but_real_ids),
                                         max(real_voxel_counts[b] for b in sparse_but_real_ids)]
                                         if sparse_but_real_ids else None,
        ),
        eval_coarse_input=dict(
            near_empty_group_mean=float(np.mean(near_empty_coarse_occ)),
            near_empty_group_max=float(np.max(near_empty_coarse_occ)),
            near_empty_group_n_exactly_zero=int(sum(1 for o in near_empty_coarse_occ if o == 0.0)),
            non_empty_group_mean=float(np.mean(non_empty_coarse_occ)),
            non_empty_group_min=float(np.min(non_empty_coarse_occ)),
        ),
        train_100_coarse_input=dict(
            n_pairs=len(train_ids), n_with_empty_coarse=train_empty_coarse,
            fraction_with_empty_coarse=train_empty_coarse / len(train_ids),
            median_occ=float(np.median(train_coarse_occ)), mean_occ=float(np.mean(train_coarse_occ)),
        ),
        **git_provenance(),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(diagnosis, open(a.out, "w"), indent=2)
    print(f"\n[done] all_near_empty_are_byte_identical={diagnosis['all_near_empty_are_byte_identical']}  "
          f"train_100 empty-coarse fraction={diagnosis['train_100_coarse_input']['fraction_with_empty_coarse']:.3f}")
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
