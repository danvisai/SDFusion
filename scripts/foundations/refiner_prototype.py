"""#45 -- CHEAP prototype: does a downstream learned SDF->SDF refiner sharpen a REAL prior
sample without eroding its footprint?

Context (map #34 / ticket #41): the #27-gate-passing LoD2 from-scratch prior (map #24) generates
solid, footprint-matching massing, but the surface is wavy -- a rounded/lumpy field with a rippled
0-contour (ticket #40's honest baseline: GT LoD2 SDFs are crisp, the VQVAE round-trip is crisp, the
PRIOR SAMPLE is wavy). Ticket #41 decided to fix this with a DOWNSTREAM refiner, leaving the prior
untouched (Phase-2's in-loop smoothness regularizer already failed a kill-gate: it won massing IoU
but lost detail fidelity -- the erosion failure mode this prototype must rule out).

This is a proof-of-concept, NOT a finished model: a few hundred synthetic (wavy, crisp) pairs, a few
hundred-to-low-thousands of training steps (minutes), then the make-or-break check -- run the trained
refiner on REAL prior samples (never seen in training) and confirm footprint-IoU does not drop.

Recipe:
  1. Synthesize aligned pairs from GT: crisp target = GT LoD2 SDF `x`. Wavy input = decode(encode(x)
     + sigma*randn). sigma is picked by matching a self-referential surface-band roughness metric
     (mean |Laplacian| weighted near the field's own 0-level) between decoded-wavy GT and REAL prior
     samples -- calibrated once here at sigma=0.15 (see `calibrate_sigma`, and the matching visual
     montage this script also saves) which lands wavy roughness (~0.0067) almost exactly on real
     prior roughness (~0.0064), vs the GT crisp floor (~0.0036).
  2. Train RefineUNet3D (models/networks/refine_unet.py) -- a residual 3D UNet, out = crude + delta,
     zero-initialized output layer so it starts at identity (low risk of eroding footprint) -- to map
     wavy -> crisp target with the near-surface-weighted L1 already defined alongside it.
  3. Validate on REAL model.inference() prior samples (held-out conditioning, never used to build
     training pairs): footprint-IoU before vs after refinement is the key number, plus an honest
     mesh_sdf_surface (continuous SDF @ 0.0) before/after/GT-reference montage.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/refiner_prototype.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(REPO))

from scripts.foundations.baseline_gate_eval import build_opt, mesh_sdf_surface, fp_iou
from models.networks.refine_unet import RefineUNet3D, surface_weighted_l1

CKPT = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"


def surface_roughness(field: torch.Tensor, band_sigma: float = 0.05) -> float:
    """Self-referential surface-band roughness: mean |Laplacian| weighted by exp(-|field|/band_sigma)
    (a band around the field's OWN 0-level). Unlike `_sdf_field_smoothness` in stage3a_model.py (which
    compares a predicted field's curvature against a separate GT target's band), this needs no paired
    target -- so it can score a real prior sample, a decoded-wavy field, or a GT field all on the same
    footing, which is exactly what calibrating sigma against real prior samples requires."""
    f = field
    while f.dim() > 3:
        f = f[0]
    c = f[1:-1, 1:-1, 1:-1]
    lap = (f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
           + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
           + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2] - 6 * c)
    band = torch.exp(-c.abs() / band_sigma)
    return float((band * lap.abs()).sum() / band.sum().clamp_min(1e-8))


def make_wavy(model, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """crisp GT SDF `x` (B,1,64,64,64) -> wavy version: decode(encode(x) + sigma*randn), matching the
    rounded/lumpy character of a real prior sample (frozen VQVAE, no grad)."""
    with torch.no_grad():
        z0 = model.vqvae(x, forward_no_quant=True, encode_only=True)
        zw = z0 + sigma * torch.randn_like(z0)
        return model.vqvae.decode_no_quant(zw)


def calibrate_sigma(model, ds_train, ds_test, device, rng, sigmas, n_probe=8):
    """Report roughness of REAL prior samples vs GT-crisp vs decoded-wavy at each candidate sigma, so
    the choice of sigma is picked by matching a real distribution, not eyeballed alone. Returns the
    real prior samples generated here (region_id, gen_sdf, fp) so validation can reuse them instead of
    re-running DDIM sampling."""
    pick = rng.choice(len(ds_test), size=n_probe, replace=False)
    prior_rough, prior_samples = [], []
    for idx in pick:
        item = ds_test[int(idx)]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
                for k, v in item.items() if torch.is_tensor(v)}
        with torch.no_grad():
            gen = model.inference(data, ddim_steps=100)
        prior_rough.append(surface_roughness(gen))
        prior_samples.append((int(item["region_id"]), gen.detach().clone(),
                               item["fp"].numpy()[0], item["sdf"].numpy()[0]))
    pick_gt = rng.choice(len(ds_train), size=n_probe, replace=False)
    gt_x = [ds_train[int(i)]["sdf"].unsqueeze(0).to(device) for i in pick_gt]
    gt_rough = [surface_roughness(x) for x in gt_x]
    sweep = {}
    for sigma in sigmas:
        rs = [surface_roughness(make_wavy(model, x, sigma)) for x in gt_x]
        sweep[str(sigma)] = float(np.mean(rs))
    report = dict(prior_roughness_mean=float(np.mean(prior_rough)),
                  gt_roughness_mean=float(np.mean(gt_rough)),
                  sigma_sweep_wavy_roughness=sweep)
    return report, prior_samples


def build_pairs_pool(model, ds_train, device, n_pairs, sigma, rng):
    """Cache crisp targets + their raw (pre-noise) VQVAE latents for a fixed pool of n_pairs GT
    buildings from the TRAIN split. Noise is re-sampled fresh every training step (not baked into the
    cache) so the refiner sees varied wavy realizations of the same crisp target."""
    idxs = rng.choice(len(ds_train), size=n_pairs, replace=False)
    targets, z0s = [], []
    for i in idxs:
        x = ds_train[int(i)]["sdf"].unsqueeze(0).to(device)
        with torch.no_grad():
            z0 = model.vqvae(x, forward_no_quant=True, encode_only=True)
        targets.append(x); z0s.append(z0)
    return torch.cat(targets, 0), torch.cat(z0s, 0)  # (N,1,64,64,64), (N,3,16,16,16)


def train_refiner(model, targets, z0s, sigma, steps, batch, lr, delta_scale, base, device, log_every=100):
    refiner = RefineUNet3D(base=base, delta_scale=delta_scale).to(device)
    opt = torch.optim.Adam(refiner.parameters(), lr=lr)
    n = targets.shape[0]
    losses = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        z0b = z0s[idx]
        with torch.no_grad():
            zw = z0b + sigma * torch.randn_like(z0b)
            wavy = model.vqvae.decode_no_quant(zw)
        pred = refiner(wavy)
        loss = surface_weighted_l1(pred, targets[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"  [train {step+1}/{steps}] loss={losses[-1]:.5f} "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)
    return refiner, losses


def save_before_after_montage(rows, path):
    """rows: list of dicts with region, before_sdf, after_sdf, gt_sdf (or None), fp_iou_before,
    fp_iou_after. 3 columns: real prior sample | refined | GT-for-reference. Honest rendering via
    mesh_sdf_surface (continuous SDF @ 0.0) -- never binarize (#43/#39 conventions)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 3 * len(rows)))
    cols = [("prior sample (before)", "before_sdf"), ("refined (after)", "after_sdf"),
            ("GT reference (same footprint)", "gt_sdf")]
    for ri, row in enumerate(rows):
        for ci, (title, key) in enumerate(cols):
            ax = fig.add_subplot(len(rows), 3, ri * 3 + ci + 1, projection="3d"); ax.set_axis_off()
            vol = row.get(key)
            if vol is not None:
                v, f = mesh_sdf_surface(vol)
                if v is not None:
                    ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color=(0.72, 0.68, 0.55), lw=0.1)
                    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
            if ri == 0: ax.set_title(title, fontsize=10)
            if ci == 0:
                sub = f"region {row['region']}\nIoU {row['fp_iou_before']:.2f}→{row['fp_iou_after']:.2f}"
                ax.text2D(-0.15, 0.5, sub, transform=ax.transAxes, fontsize=8)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n_pairs", type=int, default=300, help="# GT buildings in the training pool")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sigma", type=float, default=0.15, help="latent noise std for synth-wavy pairs")
    ap.add_argument("--base", type=int, default=24, help="RefineUNet3D base channel width")
    ap.add_argument("--delta_scale", type=float, default=0.25,
                    help="tanh-bounded residual magnitude; SDF is truncated to +-0.2 (trunc_thres)")
    ap.add_argument("--n_val", type=int, default=6, help="# real prior samples for the before/after check")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from datasets.bag3d_dataset import Bag3dDataset
    from models.stage3a_model import Stage3aModel

    opt = build_opt(device, ckpt=a.ckpt, use_region=True, use_extra_cond=False, use_ema=True)
    print(f"[load] Stage3a (map #24 LoD2 retrain) from {a.ckpt}", flush=True)
    model = Stage3aModel(); model.initialize(opt)

    ds_train = Bag3dDataset(); ds_train.initialize(opt, phase="train")
    ds_test = Bag3dDataset(); ds_test.initialize(opt, phase="test")
    rng = np.random.default_rng(a.seed)

    # 1) sigma calibration report (reuses the same real prior samples for validation below, so we
    #    don't pay for DDIM sampling twice).
    print(f"[calibrate] sigma sweep vs real prior roughness (n_probe=8, n_val={a.n_val})", flush=True)
    calib, prior_samples = calibrate_sigma(
        model, ds_train, ds_test, device, rng,
        sigmas=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40], n_probe=max(8, a.n_val))
    print(f"  prior_roughness_mean={calib['prior_roughness_mean']:.5f}  "
          f"gt_roughness_mean={calib['gt_roughness_mean']:.5f}", flush=True)
    for s, r in calib["sigma_sweep_wavy_roughness"].items():
        print(f"  sigma={s}  wavy_roughness={r:.5f}", flush=True)

    # 2) build the synthetic-pair training pool from TRAIN split, train the refiner.
    print(f"[pairs] caching {a.n_pairs} GT (crisp target, z0) pairs from train split", flush=True)
    targets, z0s = build_pairs_pool(model, ds_train, device, a.n_pairs, a.sigma, rng)
    print(f"[train] RefineUNet3D base={a.base} delta_scale={a.delta_scale} "
          f"steps={a.steps} batch={a.batch} lr={a.lr} sigma={a.sigma}", flush=True)
    refiner, losses = train_refiner(model, targets, z0s, a.sigma, a.steps, a.batch, a.lr,
                                     a.delta_scale, a.base, device)
    refiner.eval()

    ckpt_path = REPO / "outputs/refiner_prototype/refiner_unet_proto.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refiner.state_dict(), ckpt_path)
    print(f"[ckpt] saved {ckpt_path}", flush=True)

    # 3) make-or-break validation: REAL prior samples (held-out test conditioning, from calibrate_sigma
    #    above -- never used to build training pairs), before/after footprint-IoU + honest montage.
    rows = []
    for region_id, gen, real_fp, real_sdf in prior_samples[: a.n_val]:
        gen_occ_before = (gen.detach().cpu().numpy()[0, 0] <= 0)
        iou_before = fp_iou(gen_occ_before, real_fp)
        with torch.no_grad():
            refined = refiner(gen)
        gen_occ_after = (refined.detach().cpu().numpy()[0, 0] <= 0)
        iou_after = fp_iou(gen_occ_after, real_fp)
        rough_before = surface_roughness(gen)
        rough_after = surface_roughness(refined)
        rows.append(dict(
            region=region_id, fp_iou_before=iou_before, fp_iou_after=iou_after,
            roughness_before=rough_before, roughness_after=rough_after,
            before_sdf=gen.detach().cpu().numpy()[0, 0], after_sdf=refined.detach().cpu().numpy()[0, 0],
            gt_sdf=real_sdf,
        ))
        print(f"  [val] region={region_id} fp_iou {iou_before:.3f} -> {iou_after:.3f}  "
              f"roughness {rough_before:.5f} -> {rough_after:.5f}", flush=True)

    montage_path = save_before_after_montage(rows, REPO / "outputs/refiner_prototype/before_after_montage.png")
    print(f"[montage] saved {montage_path}", flush=True)

    iou_before_list = [r["fp_iou_before"] for r in rows]
    iou_after_list = [r["fp_iou_after"] for r in rows]
    rough_before_list = [r["roughness_before"] for r in rows]
    rough_after_list = [r["roughness_after"] for r in rows]
    result = dict(
        ckpt=a.ckpt, sigma=a.sigma, n_pairs=a.n_pairs, steps=a.steps, batch=a.batch, lr=a.lr,
        base=a.base, delta_scale=a.delta_scale, seed=a.seed,
        calibration=calib,
        train_loss_first=losses[0], train_loss_last=float(np.mean(losses[-20:])),
        fp_iou_before_mean=float(np.mean(iou_before_list)), fp_iou_after_mean=float(np.mean(iou_after_list)),
        fp_iou_before_per_sample=iou_before_list, fp_iou_after_per_sample=iou_after_list,
        roughness_before_mean=float(np.mean(rough_before_list)), roughness_after_mean=float(np.mean(rough_after_list)),
        gt_roughness_mean=calib["gt_roughness_mean"],
        montage=str(montage_path), refiner_ckpt=str(ckpt_path),
    )
    out = REPO / "execution/artifacts/refiner_prototype_ticket45.json"
    out.write_text(json.dumps(result, indent=2))
    print("\n=== RESULT ===", flush=True)
    print(f"  fp_iou before={result['fp_iou_before_mean']:.3f} after={result['fp_iou_after_mean']:.3f}", flush=True)
    print(f"  roughness before={result['roughness_before_mean']:.5f} after={result['roughness_after_mean']:.5f} "
          f"(GT floor={result['gt_roughness_mean']:.5f})", flush=True)
    print(f"  artifact: {out}", flush=True)


if __name__ == "__main__":
    main()
