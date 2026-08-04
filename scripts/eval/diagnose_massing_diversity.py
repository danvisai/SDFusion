"""Diagnostic (not a tracked ticket): how diverse is Stage3a's from-noise massing generation
for a FIXED footprint/class/height? User-requested follow-up after ticket 11 -- before
investing in more massing training data (Stage3a already trains on the full ~35k-building
cross-cultural `real_massing_v1` corpus, per the project's own prior data audit), check
whether the deployed prior actually produces varied buildings from identical symbolic input,
or collapses toward one shape per footprint.

Diversity metric matches `scripts/server/compare_heads_diversity.py`'s established
methodology for the (separate) recipe-param diffusion head: mean pairwise (1 - occupancy IoU)
among K samples drawn from the SAME conditioning. 0 = every sample identical (diversity=0,
the deterministic-collapse failure mode); higher = more varied.

Out: outputs/massing_diversity/diagnostic.json (per-footprint diversity scores),
     outputs/massing_diversity/montage_<class>.png (K samples per footprint, visual)
Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH TORCH_HOME=external/torch_hub \
        ./sdfusion/bin/python scripts/eval/diagnose_massing_diversity.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations", "scripts/server"):
    sys.path.insert(0, str(REPO / _p))

from eval_harness import iou  # noqa: E402
from transform_vs_noise import held_out_population  # noqa: E402
from make_splits import parse_class  # noqa: E402


def pairwise_diversity(occs) -> float:
    """Mean pairwise (1 - IoU) among a list of occupancy grids -- 0 if fewer than 2."""
    pairs = list(combinations(range(len(occs)), 2))
    if not pairs:
        return 0.0
    return float(np.mean([1.0 - iou(occs[i], occs[j]) for i, j in pairs]))


def repeat_batch(data: dict, k: int) -> dict:
    """Tile a batch=1 conditioning dict to batch=k so `model.inference`'s per-item DDIM
    randomness (independent initial noise per batch position) gives k independent samples
    from IDENTICAL symbolic conditioning."""
    out = {}
    for key, val in data.items():
        out[key] = val.repeat(k, *([1] * (val.dim() - 1)))
    return out


def _montage(rows, out_path: Path, cell=200):
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
    ap.add_argument("--k-samples", type=int, default=8, help="samples per footprint")
    ap.add_argument("--n-footprints", type=int, default=8, help="held-out footprints to test")
    ap.add_argument("--mode", choices=["noise", "sdedit"], default="noise",
                    help="'noise' = model.inference() (unconditional from-noise generation); "
                         "'sdedit' = model.sdedit() from the same footprint-extrude blockout "
                         "ticket 09 already validated (the actually-used production path)")
    ap.add_argument("--strength", type=float, default=0.5, help="sdedit strength (ticket 09's value)")
    ap.add_argument("--out", default=str(REPO / "outputs/massing_diversity/diagnostic.json"))
    ap.add_argument("--montage-out", default=str(REPO / "outputs/massing_diversity/montage.png"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from refine import Refiner
    from transform_vs_noise import build_condition

    tiers, subtype_to_idx = held_out_population()
    ids = tiers["clean"][: a.n_footprints]
    print(f"[*] {len(ids)} held-out footprints, k={a.k_samples} samples each, mode={a.mode}")

    print("[*] loading the deployed live prior...")
    refiner = Refiner(SimpleNamespace(device=a.device))
    model, _ = refiner._load_sdedit(autoguidance=False)

    per_footprint, rows = [], []
    for bid in ids:
        # build_condition's data['sdf'] is ALREADY the footprint-extrude blockout (ticket 09) --
        # inference() ignores it (footprint/class/height only), sdedit() uses it as the starting
        # point to project from, so the SAME conditioning dict serves both modes.
        data, real_occ, real_sdf = build_condition(bid, subtype_to_idx, a.device)
        batched = repeat_batch(data, a.k_samples)
        if a.mode == "sdedit":
            gen = model.sdedit(batched, strength=a.strength, max_sample=a.k_samples,
                               guide_model=None)  # (K,1,64,64,64)
        else:
            gen = model.inference(batched, max_sample=a.k_samples)  # (K,1,64,64,64)
        occs = [(gen[i, 0].detach().cpu().numpy() <= 0) for i in range(a.k_samples)]
        div = pairwise_diversity(occs)
        cls = parse_class(bid)
        per_footprint.append(dict(building=bid, cls=cls, diversity=div,
                                  occ_fracs=[float(o.mean()) for o in occs]))
        print(f"  {bid:38s} cls={cls:12s} diversity={div:.4f}  "
              f"occ_fracs={[round(float(o.mean()) * 100, 2) for o in occs]}")
        row_cells = [("real", real_sdf)] + [(f"sample {i+1}", gen[i, 0].detach().cpu().numpy())
                                            for i in range(min(a.k_samples, 6))]
        rows.append((f"{bid}\ndiv={div:.3f}", row_cells))

    _montage(rows, Path(a.montage_out))
    divs = [r["diversity"] for r in per_footprint]
    summary = dict(mode=a.mode, strength=a.strength if a.mode == "sdedit" else None,
                   k_samples=a.k_samples, n_footprints=len(ids),
                   mean_diversity=float(np.mean(divs)), min_diversity=float(np.min(divs)),
                   max_diversity=float(np.max(divs)), per_footprint=per_footprint,
                   montage=a.montage_out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)
    print(f"\n[done] mean diversity={summary['mean_diversity']:.4f} "
          f"(0=identical samples, higher=more varied)")
    print(f"[save] {a.out}\n[save] {a.montage_out}")


if __name__ == "__main__":
    main()
