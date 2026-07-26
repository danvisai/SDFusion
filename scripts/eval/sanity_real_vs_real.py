"""Real-vs-real FID sanity baseline (ticket 05): the metric floor + finite-sample variation.

Renders real BuildingNet test buildings with the neutral shader (resampled to the locked
WORKING_RES=96, ADR 0004 — never the raw 64³ native grid, so this sanity run is at the SAME
sampling density every headline comparison will use), splits the BUILDINGS in two, and reports
FID(halfA, halfB) with a building-level bootstrap CI (views of one building are correlated, so the
CI resamples whole buildings, not individual views) + a render-determinism check. Establishes the
metric floor before the harness judges either claim; failures and provenance are recorded so no
number floats free of its run context (PRD #16/#19/#35).

Run: TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
     ./sdfusion/bin/python scripts/eval/sanity_real_vs_real.py --n 48 --views 6
"""
from __future__ import annotations

import os

# On a shared many-core node, leaving these unset lets numpy/scipy/torch's BLAS backends each
# spawn a thread pool sized to nproc independently -> massive oversubscription (observed: 121
# threads on a 40-core node) that makes small per-building ops crawl despite 100%+ CPU. Cap BEFORE
# numpy/torch import so their thread pools initialize small; respects an explicit caller override.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))
import fid as fidmod  # noqa: E402
import render_facades as rf  # noqa: E402


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _package_versions():
    import torch
    import torchvision
    return dict(torch=torch.__version__, torchvision=torchvision.__version__)


def render_ids(ids, cameras, sdf_res, img_res, device="cuda"):
    """Per building: render (or record a failure). `sdf_res` is the SDF VOXEL grid resolution
    (ADR 0004's locked, research-critical shared resolution — must match across every arm);
    `img_res` is the rendered image PIXEL resolution (a rendering-quality knob, independent of
    voxel density — sphere-tracing samples the SDF continuously, so a higher pixel resolution
    does not require a higher voxel resolution). Conflating the two would silently degrade image
    quality when someone lowers the working resolution, or vice versa.
    Returns (per_building_views, succeeded_ids, failures) — `succeeded_ids[i]` is the building
    `per_bldg[i]` was rendered from."""
    import time
    per_bldg, succeeded_ids, failures = [], [], []
    for i, bid in enumerate(ids):
        t0 = time.time()
        try:
            g = rf.load_buildingnet_sdf(bid, working_res=sdf_res, device=device)
            per_bldg.append(rf.render_sdf_neutral(g, cameras=cameras, res=img_res, device=device))
            succeeded_ids.append(bid)
            print(f"  [{i + 1}/{len(ids)}] {bid} ok ({time.time() - t0:.1f}s)", flush=True)
        except Exception as ex:  # noqa: BLE001
            failures.append(dict(building=bid, error=f"{type(ex).__name__}: {str(ex)[:80]}"))
            print(f"  [{i + 1}/{len(ids)}] {bid} FAILED ({time.time() - t0:.1f}s)", flush=True)
    return per_bldg, succeeded_ids, failures


def stacked_features(extractor, per_bldg):
    """(features, group_ids): flatten a per-building list of view-lists into one feature array
    plus a matching building-index-per-row group array, for group-aware bootstrapping."""
    imgs = np.stack([im for views in per_bldg for im in views])
    groups = np.repeat(np.arange(len(per_bldg)), [len(v) for v in per_bldg])
    return extractor.features(imgs), groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--sdf-res", type=int, default=rf.WORKING_RES,
                    help="SDF voxel grid resolution (ADR 0004 locked, shared across every arm)")
    ap.add_argument("--img-res", type=int, default=256,
                    help="rendered image pixel resolution (rendering-quality knob, independent "
                         "of --sdf-res)")
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/fid_sanity.json"))
    a = ap.parse_args()

    ids = json.load(open(REPO / "data/splits_v1/test.json"))
    rng = np.random.default_rng(a.seed)
    ids = [ids[i] for i in rng.permutation(len(ids))][: a.n]
    cams = rf.orbit_cameras(n_views=a.views)

    print(f"rendering up to {len(ids)} buildings x {a.views} views "
          f"(sdf {a.sdf_res}^3, img {a.img_res}px)...", flush=True)
    per_bldg, succeeded_ids, failures = render_ids(ids, cams, a.sdf_res, a.img_res)
    for f in failures:
        print(f"[skip] {f['building']}: {f['error']}")

    # determinism / representation parity: re-render the first building, expect bit-identical
    determ = False
    if per_bldg:
        g0 = rf.load_buildingnet_sdf(succeeded_ids[0], working_res=a.sdf_res)
        again = rf.render_sdf_neutral(g0, cameras=cams, res=a.img_res)
        determ = all(np.array_equal(x, y) for x, y in zip(per_bldg[0], again))

    half = len(per_bldg) // 2
    ext = fidmod.InceptionExtractor(device="cuda")
    fa, ga = stacked_features(ext, per_bldg[:half])
    fb, gb = stacked_features(ext, per_bldg[half:2 * half])
    point, lo, hi = fidmod.bootstrap_fid_ci(fa, fb, n_boot=a.n_boot, seed=a.seed,
                                           groups_a=ga, groups_b=gb)

    out = dict(
        provenance=dict(**fidmod.EXTRACTOR_PROVENANCE, weights_url=ext.weights_url,
                        packages=_package_versions(), git_rev=_git_rev()),
        cameras=dict(n_views=a.views, kind="orbit", params=cams),
        sdf_working_res=a.sdf_res, image_res=a.img_res,
        n_requested=len(ids), n_rendered=len(per_bldg), n_failed=len(failures),
        failures=failures,
        split_half_buildings=half, images_per_half=half * a.views,
        render_deterministic=bool(determ),
        bootstrap="group-level (per building)",
        fid_real_vs_real=point, ci95=[lo, hi],
        undersampled_for_reliable_fid=bool(fidmod.undersampled(fa, fb)),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
