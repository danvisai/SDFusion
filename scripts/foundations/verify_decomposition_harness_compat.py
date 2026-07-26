"""Ticket 12's own acceptance bar: verify the decomposition arm's saved outputs are
resolution/format-compatible with the neutral evaluation harness (`scripts/eval/{render_facades,
fid}.py`, ticket 05) -- NOT the full head-to-head FID/IoU comparison against the monolith, which
is ticket 13's ("Decide the Full-Data C2 Kill-Gate") job. This only proves the handoff works:
the saved 96^3 grids render without shape errors and Inception feature extraction runs on the
result. No paired-IoU is computed here -- the composed grids include detail, and CONTEXT.md is
explicit that only massing fidelity is measured paired (detail fidelity is distributional, never
paired); `generate_decomposition_arm.py`'s own manifest already records the correct paired
massing-only IoU, so recomputing IoU here against the detail-inclusive grid would risk being
misread as a fidelity number this script was never meant to produce.

Out: execution/artifacts/decomposition_harness_compat.json
Run:  TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        ./sdfusion/bin/python scripts/foundations/verify_decomposition_harness_compat.py [--n N]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations"):
    sys.path.insert(0, str(REPO / _p))

import fid as fidmod  # noqa: E402
import render_facades as rf  # noqa: E402
from transform_vs_noise import git_provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="how many saved grids to check")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--img-res", type=int, default=256)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/decomposition_harness_compat.json"))
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = json.load(open(REPO / "data/decomposition_arm_v1/manifest.json"))
    rows = manifest["per_building"][: a.n]
    grids_dir = Path(manifest["grids_dir"])
    if not grids_dir.is_absolute():
        grids_dir = REPO / grids_dir

    cams = rf.orbit_cameras(n_views=a.views)
    checks, failures, gen_imgs, real_imgs = [], [], [], []
    for row in rows:
        bid = row["building"]
        try:
            grid = np.load(grids_dir / f"{bid}.npy").astype(np.float32)
            if grid.shape != (rf.WORKING_RES,) * 3:
                raise ValueError(f"unexpected shape {grid.shape}, expected {(rf.WORKING_RES,) * 3}")
            real96 = rf.load_buildingnet_sdf(bid, working_res=rf.WORKING_RES, device=device)
            views = rf.render_sdf_neutral(grid, cameras=cams, res=a.img_res, device=device)
            real_views = rf.render_sdf_neutral(real96, cameras=cams, res=a.img_res, device=device)
        except Exception as ex:  # noqa: BLE001
            failures.append(dict(building=bid, error=f"{type(ex).__name__}: {str(ex)[:160]}"))
            print(f"  {bid:38s} FAILED: {failures[-1]['error']}", flush=True)
            continue
        gen_imgs.append(views)
        real_imgs.append(real_views)
        checks.append(dict(building=bid, shape=list(grid.shape), n_views_rendered=len(views)))
        print(f"  {bid:38s} shape_ok rendered {len(views)} views", flush=True)

    ext = fidmod.InceptionExtractor(device=device)
    gen_stack = np.stack([im for v in gen_imgs for im in v])
    real_stack = np.stack([im for v in real_imgs for im in v])
    gen_feat, real_feat = ext.features(gen_stack), ext.features(real_stack)
    point, lo, hi = fidmod.bootstrap_fid_ci(gen_feat, real_feat, n_boot=20, seed=0)
    undersampled = bool(fidmod.undersampled(gen_feat, real_feat))
    print(f"[fid] feature extraction ran cleanly: point={point:.1f} ci95=[{lo:.1f},{hi:.1f}] "
          f"undersampled={undersampled}")

    result = dict(
        n_checked=len(checks), n_failed=len(failures), failures=failures,
        checks=checks, fid_sanity=dict(point=point, ci95=[lo, hi], undersampled=undersampled,
                                       n_generated=len(gen_stack), n_real=len(real_stack),
                                       note="sanity check only -- NOT ticket 13's headline "
                                            "comparison, undersampled expected at this n"),
        **git_provenance(),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(f"\n[done] {len(checks)}/{a.n} grids verified compatible with the neutral harness "
          f"({len(failures)} failed)")
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
