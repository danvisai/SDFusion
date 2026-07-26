"""#46 -- train the wavy->crisp SDF refiner AT SCALE (deployable v1), following the #45 prototype's
proven recipe (scripts/foundations/refiner_prototype.py, LOCKED design): residual RefineUNet3D
(models/networks/refine_unet.py, zero-init output = identity start, tanh-bounded delta ->
footprint-safe), surface_weighted_l1 loss, SYNTHETIC-aligned pairs (input = corrupted GT, target =
that same GT -- never a footprint-conditioned `model.inference` sample, which is a DIFFERENT
massing with no aligned target).

What's new vs the #45 prototype (which used a single fixed latent-noise sigma): the CORRUPTION is
hardened to better match REAL prior waviness, without touching the alignment:
  1. SDEdit-style partial regeneration (primary). `model.sdedit` (models/stage3a_model.py) already
     implements exactly the recipe the ticket calls for: encode GT -> z0, noise z0 to a moderate
     DDIM timestep (`strength`), run the prior's OWN reverse/DDIM steps back to a hat-z0, decode.
     Anchored to the GT's own conditioning (footprint/class/style/height/region), so the massing
     stays aligned while genuine prior waviness (not synthetic latent noise) is introduced.
  2. Sigma-latent-noise augmentation (secondary, cheap, resampled every step): sigma ~
     Uniform(sigma_lo, sigma_hi) per step, same mechanism as the prototype's `make_wavy`, so the
     refiner generalizes across waviness levels instead of overfitting one corruption mode.
Both modes are mixed within every training batch (p_sdedit controls the split). Both are
recalibrated here against real-prior roughness (reuses `calibrate_sigma`/`surface_roughness` from
the prototype) so the synthetic distribution actually matches real samples.

Run (at-scale, ~15-25 min on one 80GB GPU):
  env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/train_refiner.py \
      --n_pairs 2000 --n_sdedit 800 --steps 3000 --batch 8 --base 32

Downstream: scripts/foundations/baseline_gate_eval.py --refine outputs/refiner_v1/refiner_unet_v1.pth
applies the frozen trained refiner as a post-process on the prior's inference SDF for the #27 re-gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(REPO))

from scripts.foundations.baseline_gate_eval import build_opt, fp_iou
from scripts.foundations.refiner_prototype import (
    surface_roughness, calibrate_sigma, save_before_after_montage,
)
from models.networks.refine_unet import RefineUNet3D, surface_weighted_l1

CKPT = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"
POOL_KEYS = ("sdf", "fp", "class_id", "style_id", "height", "region_id")


def save_refiner_ckpt(refiner: RefineUNet3D, path, base: int, delta_scale: float, step: int, **extra):
    """Checkpoint format consumed by `baseline_gate_eval.load_refiner`: architecture hyperparams
    travel WITH the weights so a --refine ckpt reconstructs the exact trained RefineUNet3D (no
    guessing base/delta_scale at gate-eval time)."""
    payload = dict(state_dict=refiner.state_dict(), base=int(base), delta_scale=float(delta_scale),
                   step=int(step), **extra)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def build_conditioned_pool(ds_train, n_pairs: int, rng) -> dict:
    """Like the prototype's `build_pairs_pool`, but retains the FULL per-building conditioning
    (fp/class_id/style_id/height/region_id), not just (sdf, z0) -- required because SDEdit
    corruption (unlike the prototype's pure latent-noise corruption) must condition on the same
    (footprint, class, style, height, region) the GT target itself carries, so the corrupted input
    and the crisp target stay the SAME building. CPU tensors; sliced to device per use."""
    idxs = rng.choice(len(ds_train), size=n_pairs, replace=False)
    items = [ds_train[int(i)] for i in idxs]
    return {k: torch.stack([it[k] for it in items], 0) for k in POOL_KEYS}


def encode_pool_z0(model, targets: torch.Tensor, device, chunk: int = 256) -> torch.Tensor:
    """Batched frozen-VQVAE encode of the whole pool's crisp targets -> z0 latents, chunked so a
    2000-building pool doesn't need a single giant forward pass."""
    out = []
    for i in range(0, targets.shape[0], chunk):
        x = targets[i:i + chunk].to(device)
        with torch.no_grad():
            out.append(model.vqvae(x, forward_no_quant=True, encode_only=True))
    return torch.cat(out, 0)


def calibrate_sdedit_strength(model, pool: dict, device, rng, strengths, ddim_steps: int, n_probe: int = 12) -> dict:
    """SDEdit analog of the prototype's sigma sweep: roughness of SDEdit-corrupted GT (real prior
    reverse-diffusion waviness, not synthetic latent noise) across a strength sweep, batched (one
    `model.sdedit` call per strength over n_probe buildings at once) so the sweep stays cheap.
    Picking `strength` by matching REAL prior roughness (not eyeballed) mirrors calibrate_sigma."""
    n = pool["sdf"].shape[0]
    idx = rng.choice(n, size=min(n_probe, n), replace=False)
    data = {k: v[idx].to(device) for k, v in pool.items()}
    sweep = {}
    for s in strengths:
        with torch.no_grad():
            wavy = model.sdedit(data, strength=s, ddim_steps=ddim_steps, uc_scale=1.0, max_sample=len(idx))
        roughs = [surface_roughness(wavy[i:i + 1]) for i in range(wavy.shape[0])]
        sweep[str(s)] = float(np.mean(roughs))
        print(f"  [sdedit-calib] strength={s:.2f}  roughness={sweep[str(s)]:.5f}", flush=True)
    return sweep


def build_sdedit_pool(model, pool: dict, sdedit_idx: np.ndarray, device, strength_lo: float, strength_hi: float,
                       ddim_steps: int, rng, batch_size: int = 32) -> torch.Tensor:
    """Precompute SDEdit-corrupted ('genuinely wavy, still GT-aligned') versions for a SUBSET of the
    pool, batched for speed. `model.sdedit`'s truncated DDIM loop takes one scalar `strength` per
    call, so strength is jittered PER BATCH (drawn fresh per chunk) rather than per building --
    still gives the pool a spread of corruption levels (recalibration's "generalize across waviness
    levels" -- here for the domain-matched mode, complementing the sigma range for the cheap mode)."""
    n = len(sdedit_idx)
    wavy = torch.empty(n, 1, 64, 64, 64)
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = sdedit_idx[start:start + batch_size]
        data = {k: pool[k][chunk].to(device) for k in pool}
        s = float(rng.uniform(strength_lo, strength_hi))
        with torch.no_grad():
            out = model.sdedit(data, strength=s, ddim_steps=ddim_steps, uc_scale=1.0, max_sample=len(chunk))
        wavy[start:start + len(chunk)] = out.detach().cpu()
        print(f"  [sdedit-pool] {start + len(chunk)}/{n}  strength={s:.2f}  "
              f"({time.time() - t0:.1f}s elapsed)", flush=True)
    return wavy


def train_refiner_mixed(model, pool: dict, targets: torch.Tensor, z0s: torch.Tensor,
                        wavy_sdedit: torch.Tensor, sdedit_idx: np.ndarray,
                        sigma_lo: float, sigma_hi: float, p_sdedit: float,
                        steps: int, batch: int, lr: float, base: int, delta_scale: float,
                        device, ckpt_path=None, ckpt_every: int = 500, log_every: int = 100):
    """Mixed-corruption training loop (the ticket's "mix both corruption modes across training"):
    each step's batch is split between (a) precomputed SDEdit-corrupted inputs (real prior
    waviness, GT-aligned) and (b) on-the-fly sigma-latent-noise inputs with sigma resampled fresh
    per step in [sigma_lo, sigma_hi] (generalizes across waviness levels, same mechanism as the
    prototype's make_wavy). Same RefineUNet3D + surface_weighted_l1 as the prototype -- only the
    corruption source is new."""
    refiner = RefineUNet3D(base=base, delta_scale=delta_scale).to(device)
    opt = torch.optim.Adam(refiner.parameters(), lr=lr)
    n_total = targets.shape[0]
    n_sdedit = len(sdedit_idx)
    sdedit_idx_t = torch.as_tensor(sdedit_idx, device=device, dtype=torch.long)
    wavy_sdedit = wavy_sdedit.to(device)
    losses = []
    t0 = time.time()
    for step in range(steps):
        n_se = min(int(round(batch * p_sdedit)), batch) if n_sdedit > 0 else 0
        n_sig = batch - n_se
        parts_in, parts_tgt = [], []
        if n_se > 0:
            pick = torch.randint(0, n_sdedit, (n_se,), device=device)
            local_idx = sdedit_idx_t[pick]
            parts_in.append(wavy_sdedit[pick])
            parts_tgt.append(targets[local_idx])
        if n_sig > 0:
            pick2 = torch.randint(0, n_total, (n_sig,), device=device)
            z0b = z0s[pick2]
            sigma = torch.empty(n_sig, 1, 1, 1, 1, device=device).uniform_(sigma_lo, sigma_hi)
            with torch.no_grad():
                zw = z0b + sigma * torch.randn_like(z0b)
                wavy = model.vqvae.decode_no_quant(zw)
            parts_in.append(wavy)
            parts_tgt.append(targets[pick2])
        inp = torch.cat(parts_in, 0)
        tgt = torch.cat(parts_tgt, 0)
        pred = refiner(inp)
        loss = surface_weighted_l1(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"  [train {step+1}/{steps}] loss={losses[-1]:.5f} "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)
        if ckpt_path and (step + 1) % ckpt_every == 0:
            save_refiner_ckpt(refiner, ckpt_path, base, delta_scale, step + 1)
    if ckpt_path:
        save_refiner_ckpt(refiner, ckpt_path, base, delta_scale, steps)
    return refiner, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n_pairs", type=int, default=2000, help="# GT buildings in the training pool (train split)")
    ap.add_argument("--n_sdedit", type=int, default=800, help="# of the pool given a precomputed SDEdit-corrupted version")
    ap.add_argument("--p_sdedit", type=float, default=0.5, help="per-step batch fraction: SDEdit-precomputed vs on-the-fly sigma corruption")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sigma_lo", type=float, default=0.10, help="sigma-augmentation lower bound (latent-noise corruption)")
    ap.add_argument("--sigma_hi", type=float, default=0.20, help="sigma-augmentation upper bound")
    ap.add_argument("--sdedit_ddim_steps", type=int, default=100, help="DDIM schedule length for SDEdit corruption (matches production)")
    ap.add_argument("--sdedit_strengths", default="0.15,0.25,0.35,0.45,0.55", help="comma-separated calibration sweep candidates")
    ap.add_argument("--sdedit_strength_lo", type=float, default=None, help="override the calibrated pick's jitter range")
    ap.add_argument("--sdedit_strength_hi", type=float, default=None)
    ap.add_argument("--sdedit_calib_probe", type=int, default=12)
    ap.add_argument("--sdedit_batch", type=int, default=32, help="batch size for precomputing the SDEdit-corrupted pool")
    ap.add_argument("--base", type=int, default=32, help="RefineUNet3D base channel width")
    ap.add_argument("--delta_scale", type=float, default=0.25)
    ap.add_argument("--n_val", type=int, default=24, help="# real prior held-out samples for roughness/fp_iou/montage (>=20 per #46)")
    ap.add_argument("--n_montage", type=int, default=6)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/refiner_v1")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = REPO / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets.bag3d_dataset import Bag3dDataset
    from models.stage3a_model import Stage3aModel

    opt = build_opt(device, ckpt=a.ckpt, use_region=True, use_extra_cond=False, use_ema=True)
    print(f"[load] Stage3a (map #24 LoD2 retrain) from {a.ckpt}", flush=True)
    model = Stage3aModel(); model.initialize(opt)

    ds_train = Bag3dDataset(); ds_train.initialize(opt, phase="train")
    ds_test = Bag3dDataset(); ds_test.initialize(opt, phase="test")
    rng = np.random.default_rng(a.seed)
    t_start = time.time()

    # 1) sigma calibration (reused verbatim from the prototype) -- also yields n_val real prior
    #    samples (held-out test conditioning) reused below for BOTH the sdedit-strength calibration
    #    reference point AND the roughness/fp_iou/montage validation, so DDIM sampling isn't paid
    #    for twice.
    n_probe = max(24, a.n_val)
    print(f"[calibrate-sigma] sigma sweep vs real prior roughness (n_probe={n_probe})", flush=True)
    calib_sigma, prior_samples = calibrate_sigma(
        model, ds_train, ds_test, device, rng,
        sigmas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30], n_probe=n_probe)
    print(f"  prior_roughness_mean={calib_sigma['prior_roughness_mean']:.5f}  "
          f"gt_roughness_mean={calib_sigma['gt_roughness_mean']:.5f}", flush=True)

    # 2) conditioned training pool (crisp targets + full conditioning), then z0 latents.
    print(f"[pool] caching {a.n_pairs} GT (target, conditioning) buildings from train split", flush=True)
    pool = build_conditioned_pool(ds_train, a.n_pairs, rng)
    targets = pool["sdf"].to(device)
    z0s = encode_pool_z0(model, pool["sdf"], device)

    # 3) SDEdit-strength calibration: pick the strength whose corrupted-GT roughness lands closest
    #    to the REAL prior roughness measured above (recalibration, not eyeballed).
    strengths = [float(s) for s in a.sdedit_strengths.split(",")]
    print(f"[calibrate-sdedit] strength sweep {strengths} @ ddim={a.sdedit_ddim_steps}", flush=True)
    sdedit_sweep = calibrate_sdedit_strength(model, pool, device, rng, strengths,
                                             a.sdedit_ddim_steps, n_probe=a.sdedit_calib_probe)
    target_rough = calib_sigma["prior_roughness_mean"]
    best_strength = min(strengths, key=lambda s: abs(sdedit_sweep[str(s)] - target_rough))
    lo = a.sdedit_strength_lo if a.sdedit_strength_lo is not None else max(0.05, best_strength - 0.05)
    hi = a.sdedit_strength_hi if a.sdedit_strength_hi is not None else min(0.90, best_strength + 0.05)
    print(f"  picked strength={best_strength:.2f} (roughness={sdedit_sweep[str(best_strength)]:.5f} "
          f"vs real prior {target_rough:.5f}) -> jitter range [{lo:.2f}, {hi:.2f}]", flush=True)

    # 4) precompute the SDEdit-corrupted subpool (real prior waviness, GT-aligned).
    n_sdedit = min(a.n_sdedit, a.n_pairs)
    sdedit_idx = rng.choice(a.n_pairs, size=n_sdedit, replace=False)
    print(f"[sdedit-pool] precomputing {n_sdedit} SDEdit-corrupted pairs (strength~[{lo:.2f},{hi:.2f}], "
          f"ddim={a.sdedit_ddim_steps}, batch={a.sdedit_batch})", flush=True)
    wavy_sdedit = build_sdedit_pool(model, pool, sdedit_idx, device, lo, hi,
                                    a.sdedit_ddim_steps, rng, batch_size=a.sdedit_batch)

    # 5) train, mixing SDEdit-precomputed + on-the-fly sigma-augmented corruption every step.
    ckpt_path = out_dir / "refiner_unet_v1.pth"
    print(f"[train] RefineUNet3D base={a.base} delta_scale={a.delta_scale} steps={a.steps} "
          f"batch={a.batch} lr={a.lr} p_sdedit={a.p_sdedit} sigma=[{a.sigma_lo},{a.sigma_hi}]", flush=True)
    refiner, losses = train_refiner_mixed(
        model, pool, targets, z0s, wavy_sdedit, sdedit_idx,
        a.sigma_lo, a.sigma_hi, a.p_sdedit, a.steps, a.batch, a.lr, a.base, a.delta_scale,
        device, ckpt_path=ckpt_path, ckpt_every=a.ckpt_every)
    refiner.eval()
    print(f"[ckpt] saved {ckpt_path}", flush=True)

    # 6) validation: >=20 held-out REAL prior samples (never used to build training pairs) --
    #    roughness before/after (the ticket's headline non-gate metric) + fp_iou sanity + montage.
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

    montage_path = save_before_after_montage(rows[: a.n_montage], out_dir / "before_after_montage.png")
    print(f"[montage] saved {montage_path}", flush=True)

    iou_before_list = [r["fp_iou_before"] for r in rows]
    iou_after_list = [r["fp_iou_after"] for r in rows]
    rough_before_list = [r["roughness_before"] for r in rows]
    rough_after_list = [r["roughness_after"] for r in rows]
    result = dict(
        ckpt=a.ckpt, n_pairs=a.n_pairs, n_sdedit=n_sdedit, p_sdedit=a.p_sdedit,
        steps=a.steps, batch=a.batch, lr=a.lr, base=a.base, delta_scale=a.delta_scale,
        sigma_lo=a.sigma_lo, sigma_hi=a.sigma_hi,
        sdedit_ddim_steps=a.sdedit_ddim_steps, sdedit_strength_sweep=sdedit_sweep,
        sdedit_strength_picked=best_strength, sdedit_strength_range=[lo, hi], seed=a.seed,
        calibration_sigma=calib_sigma,
        train_loss_first=losses[0], train_loss_last=float(np.mean(losses[-20:])),
        n_val=len(rows),
        fp_iou_before_mean=float(np.mean(iou_before_list)), fp_iou_after_mean=float(np.mean(iou_after_list)),
        fp_iou_before_per_sample=iou_before_list, fp_iou_after_per_sample=iou_after_list,
        roughness_before_mean=float(np.mean(rough_before_list)), roughness_after_mean=float(np.mean(rough_after_list)),
        roughness_before_per_sample=rough_before_list, roughness_after_per_sample=rough_after_list,
        gt_roughness_mean=calib_sigma["gt_roughness_mean"],
        montage=str(montage_path), refiner_ckpt=str(ckpt_path),
        wall_time_sec=time.time() - t_start,
    )
    # derive the artifact name from --out_dir so distinct runs don't clobber each other's record
    # (out_dir "outputs/refiner_v1" -> "refiner_v1_train.json"; keeps the original name for v1).
    out = REPO / "execution/artifacts" / f"{out_dir.name}_train.json"
    out.write_text(json.dumps(result, indent=2))
    print("\n=== RESULT ===", flush=True)
    print(f"  fp_iou before={result['fp_iou_before_mean']:.3f} after={result['fp_iou_after_mean']:.3f}", flush=True)
    print(f"  roughness (n={len(rows)}) before={result['roughness_before_mean']:.5f} "
          f"after={result['roughness_after_mean']:.5f} (GT floor={result['gt_roughness_mean']:.5f})", flush=True)
    print(f"  wall_time={result['wall_time_sec']:.0f}s", flush=True)
    print(f"  artifact: {out}", flush=True)


if __name__ == "__main__":
    main()
