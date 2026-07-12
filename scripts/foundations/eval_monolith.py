"""Ticket 11: held-out behavior + inference-reproducibility check for a trained monolith.

Not the headline C2 comparison (tickets 12/13 own that, against the sealed
`data/splits_v1/test.json`) -- this only asks whether the checkpoint this ticket trained is an
honest, sane baseline: does it produce plausible occupancy on buildings it never trained on,
and is sampling reproducible.

Uses a HIGH ddim_steps count, not the codebase's usual `ddim_steps=50`: a diagnostic run
against an early checkpoint found 50-step DDIM gives markedly worse (higher, less realistic)
occupancy than ~1000-step DDIM for this from-scratch, far-less-mature model than Stage3a's
150k-step deployed prior -- see the ticket answer. This is a property of the model's maturity,
not a hyperparameter tuned to flatter the result.

Out: execution/artifacts/monolith_eval.json, outputs/monolith_v1/montage.png
Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/foundations/eval_monolith.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in ("models", "models/networks", "datasets", "scripts/foundations", "scripts/eval"):
    sys.path.insert(0, str(REPO / _p))

from monolith_unet import MonolithUNet  # noqa: E402
from monolith_diffusion import GaussianDiffusion  # noqa: E402
from monolith_pair_dataset import MonolithPairDataset  # noqa: E402
from make_splits import parse_class  # noqa: E402


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["config"]
    net = MonolithUNet(base_channels=cfg["base_channels"], channel_mults=tuple(cfg["channel_mults"])).to(device)
    net.load_state_dict(state["model"])
    net.eval()
    return GaussianDiffusion(net, timesteps=cfg["timesteps"], device=device), cfg, state["step"]


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
    ap.add_argument("--ckpt", default=str(REPO / "logs_building/monolith_v1/ckpt/monolith_steps-latest.pth"))
    ap.add_argument("--run-manifest", default=str(REPO / "logs_building/monolith_v1/manifest.json"))
    ap.add_argument("--ddim-steps", type=int, default=1000)
    ap.add_argument("--n-quant", type=int, default=40, help="val buildings for the quantitative pass")
    ap.add_argument("--n-montage-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/monolith_eval.json"))
    ap.add_argument("--montage-out", default=str(REPO / "outputs/monolith_v1/montage.png"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    diffusion, cfg, ckpt_step = load_model(a.ckpt, a.device)
    run_manifest = json.load(open(a.run_manifest))
    val_ids = run_manifest["val_ids"]

    # --- inference reproducibility -------------------------------------------------------
    ds_one = MonolithPairDataset(val_ids[:1], augment=False, device="cpu")
    coarse0 = ds_one[0]["coarse"][None].to(a.device)
    shape = (1, 1, 96, 96, 96)
    gen_a = diffusion.ddim_sample(coarse0, shape=shape, ddim_steps=a.ddim_steps, seed=a.seed)
    gen_b = diffusion.ddim_sample(coarse0, shape=shape, ddim_steps=a.ddim_steps, seed=a.seed)
    gen_c = diffusion.ddim_sample(coarse0, shape=shape, ddim_steps=a.ddim_steps, seed=a.seed + 1)
    reproducible_same_seed = bool(torch.equal(gen_a, gen_b))
    differs_other_seed = not bool(torch.equal(gen_a, gen_c))
    print(f"[repro] same-seed bit-identical: {reproducible_same_seed}  "
          f"different-seed differs: {differs_other_seed}")

    # --- quantitative pass over a balanced held-out sample --------------------------------
    by_class = defaultdict(list)
    for bid in val_ids:
        by_class[parse_class(bid)].append(bid)
    quant_ids, per_class_budget = [], max(1, a.n_quant // max(len(by_class), 1))
    for cls in sorted(by_class):
        quant_ids.extend(sorted(by_class[cls])[:per_class_budget])
    quant_ids = quant_ids[: a.n_quant]

    ds_quant = MonolithPairDataset(quant_ids, augment=False, device="cpu")
    per_building = []
    montage_by_class = defaultdict(list)  # filled alongside the quant pass -- one DDIM sample
                                          # per building, never resampled for the montage
    for i, bid in enumerate(quant_ids):
        item = ds_quant[i]
        coarse = item["coarse"][None].to(a.device)
        target_np, coarse_np = item["target"][0].numpy(), item["coarse"][0].numpy()
        gen = diffusion.ddim_sample(coarse, shape=shape, ddim_steps=a.ddim_steps, seed=a.seed)
        gen_np = gen[0, 0].detach().cpu().numpy()
        target_occ, coarse_occ, gen_occ = target_np <= 0, coarse_np <= 0, gen_np <= 0
        cls = parse_class(bid)
        rec = dict(building=bid, cls=cls,
                  target_occ_frac=float(target_occ.mean()),
                  coarse_occ_frac=float(coarse_occ.mean()),
                  gen_occ_frac=float(gen_occ.mean()))
        per_building.append(rec)
        if len(montage_by_class[cls]) < a.n_montage_per_class:
            montage_by_class[cls].append((bid, target_np, coarse_np, gen_np))
        print(f"  [{i + 1}/{len(quant_ids)}] {bid}  target={100*rec['target_occ_frac']:.3f}%  "
              f"coarse={100*rec['coarse_occ_frac']:.3f}%  gen={100*rec['gen_occ_frac']:.3f}%", flush=True)

    gen_fracs = np.asarray([r["gen_occ_frac"] for r in per_building])
    target_fracs = np.asarray([r["target_occ_frac"] for r in per_building])

    # --- qualitative montage: real / coarse / generated, a few buildings per class --------
    rows = []
    for cls in sorted(montage_by_class):
        for bid, target, coarse, gen in montage_by_class[cls]:
            rows.append((f"{cls}\n{bid}", [("real target", target), ("low-pass coarse", coarse),
                                           ("monolith gen", gen)]))
    if rows:
        _montage(rows, Path(a.montage_out))

    manifest = dict(
        checkpoint=a.ckpt, checkpoint_step=ckpt_step, ddim_steps=a.ddim_steps,
        reproducibility=dict(same_seed_bit_identical=reproducible_same_seed,
                            different_seed_differs=differs_other_seed),
        n_quant=len(per_building), per_building=per_building,
        gen_occ_frac=dict(mean=float(gen_fracs.mean()), median=float(np.median(gen_fracs))),
        target_occ_frac=dict(mean=float(target_fracs.mean()), median=float(np.median(target_fracs))),
        montage=a.montage_out if rows else None,
        **_git_provenance(),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(a.out, "w"), indent=2)
    print(f"\n[done] mean gen_occ_frac={manifest['gen_occ_frac']['mean']:.4f}  "
          f"mean target_occ_frac={manifest['target_occ_frac']['mean']:.4f}")
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
