"""Ticket 07: build the real full-data monolith-pair dataset.

For every `train_100` id (ticket 03's sealed, leakage-safe fraction, with the sealed `test` split
excluded a second time here as a defense-in-depth check -- never trust a single gate for "the
sealed test set never appears in a training fraction"): the TARGET is the real BuildingNet SDF at
the locked working resolution (never a synthetic composer output -- there is no composer in this
script's call graph at all). The COARSE input is ADR 0004's locked primary -- a low-pass transform
of that SAME building's real SDF, so source and target stay spatially aligned by construction (no
separate footprint/height re-derivation that could drift). The low-pass is implemented by
resampling down to a grid whose own voxel pitch matches the fixed `s*` and back up to the working
resolution via the SAME trilinear `resample_sdf_grid` every other ticket already resamples with --
one resampling primitive, no new interpolation code path, and no new free hyperparameter (the
coarse resolution is derived from `s*`, which ADR 0004 already fixed a priori).

Validation, never silent filtering (matches tickets 06/09's disclose-don't-drop policy): resolution
and finiteness are asserted per pair (a failure is recorded, not swallowed); axis convention is
checked against BuildingNet's OWN precomputed footprint field (`footprint_alignment_iou`, verified
IoU=1.0 on real data for the assumed H-up / axis=1 convention -- see the ticket answer); occupancy
fractions are recorded, not thresholded, because BuildingNet's real interior-occupancy distribution
has no natural "corrupted" cutoff (ticket 09's own finding).

Out: data/monolith_pairs_v1/{manifest.json, pairs.json, per_pair.json}
     outputs/monolith_pairs_v1/montage.png (real target vs low-pass coarse, one row per class)
Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/foundations/build_monolith_pairs.py [--limit N]
"""
from __future__ import annotations

import os

# Same BLAS-oversubscription guard ticket 09 needed on this node (121 threads/40 cores observed) --
# must run before numpy/torch import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations"):
    sys.path.insert(0, str(REPO / _p))

import render_facades as rf  # noqa: E402
from build_element_library import load_id_list, select_building_ids  # noqa: E402
from make_splits import parse_class  # noqa: E402

WORKING_RES = 96      # ADR 0004: locked shared working resolution
S_STAR_VOXELS = 5     # ADR 0004: s* = 1.0 m = 5 voxels @96^3


def coarse_resolution(working_res: int = WORKING_RES, s_star_vox: float = S_STAR_VOXELS) -> int:
    """Resolution whose own voxel pitch ~= `s*` -- resampling through it band-limits away
    content below the fixed massing/detail boundary."""
    return max(1, round(working_res / s_star_vox))


def low_pass_sdf(sdf, working_res: int = WORKING_RES, s_star_vox: float = S_STAR_VOXELS,
                  device: str = "cpu"):
    """ADR 0004's monolith coarse-input primary: down-then-up trilinear resample through
    `coarse_resolution()`, reusing `render_facades.resample_sdf_grid` for both legs."""
    cres = coarse_resolution(working_res, s_star_vox)
    down = rf.resample_sdf_grid(sdf, cres, device=device)
    return rf.resample_sdf_grid(down, working_res, device=device)


def footprint_alignment_iou(native_occ, stored_footprint) -> float:
    """IoU between a footprint DERIVED from 3D occupancy (`occ.any(axis=1)`, the H-up axis
    convention every eval script assumes) and BuildingNet's OWN precomputed footprint field --
    independent evidence that convention matches the raw data for this specific building."""
    derived = np.asarray(native_occ).any(axis=1)
    stored = np.asarray(stored_footprint).astype(bool)
    if stored.ndim == 3:
        stored = stored[0]
    inter = np.logical_and(derived, stored).sum()
    union = np.logical_or(derived, stored).sum()
    return float(inter / union) if union else 1.0


def validate_pair(building_id: str, target, coarse, working_res: int, footprint_iou: float) -> dict:
    """Structural + sign/axis diagnostics for one (target, coarse) pair. Raises (caught by the
    per-building try/except in `main`, recorded as a failure) on a corrupted pair; otherwise only
    RECORDS occupancy so a reader can judge data quality themselves, never filters on it."""
    assert target.shape == (working_res,) * 3, f"{building_id}: target shape {target.shape}"
    assert coarse.shape == (working_res,) * 3, f"{building_id}: coarse shape {coarse.shape}"
    assert np.isfinite(target).all(), f"{building_id}: non-finite target sdf"
    assert np.isfinite(coarse).all(), f"{building_id}: non-finite coarse sdf"
    return dict(
        building=building_id,
        target_occupancy_frac=float((target <= 0).mean()),
        coarse_occupancy_frac=float((coarse <= 0).mean()),
        target_sdf_min=float(target.min()), target_sdf_max=float(target.max()),
        footprint_axis_iou=footprint_iou,
    )


def _git_provenance():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return dict(git_rev=None, dirty_digest=None)
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    digest = hashlib.sha1(status.encode()).hexdigest()[:12] if status.strip() else None
    return dict(git_rev=rev, dirty_digest=digest)


def _montage(rows, out_path: Path, cell=224):
    """rows: list of (label, [(title, (D,H,W) sdf), ...]) -- one row per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    n_rows, n_cols = len(rows), len(rows[0][1]) if rows else 0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell / 60 * n_cols, cell / 60 * n_rows),
                             subplot_kw={"projection": "3d"}, squeeze=False)
    for ri, (row_label, cells) in enumerate(rows):
        for ci, (title, sdf) in enumerate(cells):
            ax = axes[ri][ci]
            ax.set_axis_off()
            if sdf is not None and (sdf <= 0).sum() > 8:
                try:
                    v, f, *_ = measure.marching_cubes(sdf, 0.0)
                    ax.plot_trisurf(v[:, 2], v[:, 0], f, v[:, 1], color="#c9b790",
                                    edgecolor="none", shade=True)
                    ax.set_xlim(0, sdf.shape[2]); ax.set_ylim(0, sdf.shape[0]); ax.set_zlim(0, sdf.shape[1])
                except Exception:
                    pass
            ax.view_init(elev=14, azim=-60)
            ax.set_title(f"{row_label}\n{title}" if ci == 0 else title, fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default=str(REPO / "data/splits_v1"))
    ap.add_argument("--fraction", default="train_100", help="which splits_v1 fraction to build pairs for")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N ids")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(REPO / "data/monolith_pairs_v1"))
    ap.add_argument("--montage-out", default=str(REPO / "outputs/monolith_pairs_v1/montage.png"))
    ap.add_argument("--montage-per-class", type=int, default=2)
    ap.add_argument("--no-qa", action="store_true")
    a = ap.parse_args()

    split_dir = Path(a.split_dir)
    train_ids = load_id_list(str(split_dir / f"{a.fraction}.json"))
    test_ids = load_id_list(str(split_dir / "test.json"))
    ids = select_building_ids(train_ids, include_ids=None, exclude_ids=test_ids)
    if a.limit:
        ids = ids[: a.limit]

    per_pair, failures = [], []
    montage_by_class: dict[str, list] = {}
    for i, bid in enumerate(ids):
        try:
            native = rf.load_buildingnet_sdf(bid, working_res=64, device=a.device)  # raw, un-resampled
            native_occ = native <= 0
            stored_fp = rf.load_buildingnet_footprint(bid)
            fp_iou = footprint_alignment_iou(native_occ, stored_fp)

            target = rf.resample_sdf_grid(native, WORKING_RES, device=a.device)
            coarse = low_pass_sdf(target, WORKING_RES, S_STAR_VOXELS, device=a.device)
            rec = validate_pair(bid, target, coarse, WORKING_RES, fp_iou)
            rec["class"] = parse_class(bid)
            per_pair.append(rec)

            bucket = montage_by_class.setdefault(rec["class"], [])
            if len(bucket) < a.montage_per_class:
                bucket.append((bid, target, coarse))

            print(f"  [{i + 1}/{len(ids)}] {bid}  target_occ={100 * rec['target_occupancy_frac']:.3f}%  "
                  f"coarse_occ={100 * rec['coarse_occupancy_frac']:.3f}%  fp_axis_iou={fp_iou:.2f}",
                  flush=True)
        except Exception as ex:  # noqa: BLE001
            failures.append(dict(building=bid, error=f"{type(ex).__name__}: {str(ex)[:120]}"))
            print(f"  [{i + 1}/{len(ids)}] {bid} FAILED: {failures[-1]['error']}", flush=True)

    built_ids = sorted(r["building"] for r in per_pair)
    leak = sorted(set(built_ids) & set(test_ids))
    class_balance = dict(sorted(Counter(r["class"] for r in per_pair).items()))
    fp_ious = [r["footprint_axis_iou"] for r in per_pair]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.json").write_text(json.dumps(built_ids, indent=0))
    (out / "per_pair.json").write_text(json.dumps(per_pair, indent=2))

    manifest = dict(
        fraction=a.fraction, split_dir=str(split_dir),
        working_res=WORKING_RES, s_star_voxels=S_STAR_VOXELS,
        coarse_resolution=coarse_resolution(WORKING_RES, S_STAR_VOXELS),
        coarse_input=("low_pass (primary, ADR 0004): resample the same building's real SDF down "
                      "to a grid whose voxel pitch matches s*, then back up to working_res via "
                      "trilinear resample_sdf_grid"),
        n_requested=len(ids), n_built=len(per_pair), n_failed=len(failures), failures=failures,
        leakage_excluded_contributors=leak,
        class_balance=class_balance,
        footprint_axis_check=dict(
            mean_iou=float(np.mean(fp_ious)) if fp_ious else None,
            min_iou=float(np.min(fp_ious)) if fp_ious else None,
            n_below_0_5=sum(1 for v in fp_ious if v < 0.5),
        ),
        occupancy_frac=dict(
            target_mean=float(np.mean([r["target_occupancy_frac"] for r in per_pair])) if per_pair else None,
            coarse_mean=float(np.mean([r["coarse_occupancy_frac"] for r in per_pair])) if per_pair else None,
        ),
        command=f"scripts/foundations/build_monolith_pairs.py --fraction {a.fraction} "
                f"--split-dir {a.split_dir} --out {a.out}",
        **_git_provenance(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    assert not leak, f"LEAKAGE: excluded test ids present in built pairs: {leak[:5]}"

    if not a.no_qa and montage_by_class:
        rows = []
        for cls in sorted(montage_by_class):
            for bid, target, coarse in montage_by_class[cls]:
                rows.append((f"{cls}\n{bid}", [("real target", target), ("low-pass coarse", coarse)]))
        _montage(rows, Path(a.montage_out))
        print(f"[save] {a.montage_out}")

    print(f"\n[done] {len(per_pair)}/{len(ids)} pairs built ({len(failures)} failed) -> {out}")


if __name__ == "__main__":
    main()
