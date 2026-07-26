"""Ticket 13: generate the full-data monolith arm's held-out outputs, on the SAME population
ticket 12's decomposition arm already used (`data/splits_v1/test.json`, 277 ids).

`eval_monolith.py` (ticket 11) never ran the monolith through a comparison against
`test.json` at all -- it evaluated occupancy-only on a `train_100`-internal validation slice,
explicitly disclaiming the headline comparison ("tickets 12/13 own that"). This script builds
that missing half.

Contract (reused, not re-derived):
  - Coarse input: `datasets.monolith_pair_dataset.MonolithPairDataset` applied directly to
    `test.json` ids -- ticket 07's own `low_pass_sdf` + `frame_n_input` pipeline, id-agnostic
    by design (no train_100-specific state; leakage-safety lives in which ids you pass it, and
    `test.json` was never in `train_100`'s pairs by construction, verified 0 overlap in ticket
    10/12's own checks).
  - Model: `monolith_v3` (`logs_building/monolith_v3/ckpt/monolith_steps-latest.pth`),
    `eval_monolith.load_model`, `ddim_steps=1000` -- ticket 11's own established value (a
    diagnostic run found 50-step DDIM markedly worse for this from-scratch model than
    Stage3a's 150k-step deployed prior).
  - Unscaling: `ddim_sample`'s output lives in the SAME `TRUNC=0.2`-normalized `[-1,1]` range
    the model trained against (its `clip_x0=1.0` default matches this range exactly) -- no
    existing code un-scales this for rendering/comparison purposes, so `unscale_ddim_output`
    (multiply by `TRUNC`) is new, minimal, and required before this output is comparable to the
    decomposition arm's or real targets' metric SDF values.

Leakage note (asymmetric with the decomposition arm, disclosed not hidden): `monolith_v3` was
trained fresh, only on `train_100`, so ALL 277 `test.json` ids are genuinely held out for it --
unlike the decomposition arm, whose massing half uses the pretrained Stage3a prior (leakage on
224/277, per ticket 09/12's own finding).

Out: data/monolith_arm_v1/{manifest.json, grids/<id>.npy}
Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/foundations/generate_monolith_arm.py [--limit N]
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
for _p in ("scripts/eval", "scripts/foundations", "models", "models/networks", "datasets"):
    sys.path.insert(0, str(REPO / _p))

from eval_harness import iou, fp_iou  # noqa: E402
from eval_monolith import load_model  # noqa: E402
from make_splits import parse_class  # noqa: E402
from monolith_pair_dataset import MonolithPairDataset, TRUNC  # noqa: E402
from transform_vs_noise import git_provenance  # noqa: E402

CKPT = REPO / "logs_building/monolith_v3/ckpt/monolith_steps-latest.pth"
WORKING_RES = 96
DDIM_STEPS = 1000  # ticket 11's own established value, not the codebase's usual 50


def unscale_ddim_output(x, trunc=TRUNC):
    """`ddim_sample`'s output lives in the model's TRUNC-normalized [-1,1] training range --
    multiply back by TRUNC to land in the same metric SDF units real targets and the
    decomposition arm's composed grids use."""
    return x * trunc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N buildings")
    ap.add_argument("--ddim-steps", type=int, default=DDIM_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "data/monolith_arm_v1"))
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] loading monolith_v3 from {CKPT}")
    diffusion, cfg, ckpt_step = load_model(str(CKPT), device)
    print(f"[*] checkpoint step={ckpt_step}  config={cfg}")

    test_ids = json.load(open(REPO / "data/splits_v1/test.json"))
    if a.limit:
        test_ids = test_ids[: a.limit]
    print(f"[*] {len(test_ids)} buildings from data/splits_v1/test.json "
          f"(all genuinely held out for monolith_v3 -- trained fresh on train_100 only)")

    ds = MonolithPairDataset(test_ids, working_res=WORKING_RES, augment=False, device="cpu")
    shape = (1, 1, WORKING_RES, WORKING_RES, WORKING_RES)

    grids_dir = Path(a.out_dir) / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for i, bid in enumerate(test_ids):
        try:
            item = ds[i]
            coarse = item["coarse"][None].to(device)
            gen = diffusion.ddim_sample(coarse, shape=shape, ddim_steps=a.ddim_steps, seed=a.seed)
            gen_np = unscale_ddim_output(gen[0, 0].detach().cpu().numpy()).astype(np.float32)
            target_np = unscale_ddim_output(item["target"][0].numpy()).astype(np.float32)
        except Exception as ex:  # noqa: BLE001
            failures.append(dict(building=bid, error=f"{type(ex).__name__}: {str(ex)[:160]}"))
            print(f"  [{i+1}/{len(test_ids)}] {bid} FAILED: {failures[-1]['error']}", flush=True)
            continue

        gen_occ, target_occ = gen_np <= 0, target_np <= 0
        m_iou, m_fp_iou = iou(gen_occ, target_occ), fp_iou(gen_occ, target_occ)
        np.save(grids_dir / f"{bid}.npy", gen_np.astype(np.float16))
        rows.append(dict(building=bid, building_class=parse_class(bid),
                         gen_occ_frac=float(gen_occ.mean()), target_occ_frac=float(target_occ.mean()),
                         iou=m_iou, fp_iou=m_fp_iou))
        print(f"  [{i+1}/{len(test_ids)}] {bid:38s} iou={m_iou:.3f} fp_iou={m_fp_iou:.3f} "
              f"gen_occ={100*gen_occ.mean():.3f}% target_occ={100*target_occ.mean():.3f}%", flush=True)

    manifest = dict(
        checkpoint=str(CKPT), checkpoint_step=ckpt_step, ddim_steps=a.ddim_steps,
        trunc=TRUNC, working_res=WORKING_RES, leakage="none -- trained fresh on train_100 only",
        n_succeeded=len(rows), n_failed=len(failures), failures=failures,
        per_building=rows, grids_dir=str(grids_dir), **git_provenance(),
    )
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(Path(a.out_dir) / "manifest.json", "w"), indent=2)
    print(f"\n[done] {len(rows)} succeeded, {len(failures)} failed")
    print(f"[save] {Path(a.out_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
