"""#59 -- latent-space corrector DE-RISK PROTOTYPE (map #58).

Prior findings this experiment tests (do not re-litigate, see docs/wayfinding history):
  - Stage3a's diffusion prior produces LUMPY/WAVY building SDFs. Roughness (surface_roughness,
    band-weighted mean|Laplacian| around the 0-level): GT floor ~0.0041; raw diffusion output
    ~0.0054; the #54 POST-DECODE SDF-space refiner plateaus at ~0.0047 and cannot reach GT.
  - BUT decode(encode(GT)) ~0.0044 ~ GT: the frozen VQVAE decodes CLEAN latents crisply. The
    codec is not the bottleneck -- the diffusion's LATENT is wavy.
  - Hypothesis under test here: a small corrector g operating IN LATENT SPACE (wavy z -> clean
    z), with the codec AND diffusion both frozen, recovers crispness because the frozen crisp
    codec then decodes the corrected latent. SDF-space post-decode correction already failed
    (#54); this is the untested cheap lever.

Recipe (mirrors scripts/foundations/train_refiner.py's aligned-pair machinery, moved into latent
space): build a conditioned pool of GT buildings (`build_conditioned_pool`, reused verbatim),
encode their crisp targets to raw VQVAE latents z0 (`encode_pool_z0`, reused verbatim). Corrupt a
subset via SDEdit (`build_sdedit_pool`/`calibrate_sdedit_strength`, reused verbatim, calibrated to
match REAL prior roughness) and re-encode the SDEdit-wavy SDFs to latents -- these pairs
(z_wavy_sdedit, z0) are genuinely wavy (real prior-style reverse-diffusion artifacts) yet still
GT-aligned (same building). Mixed in with cheaper on-the-fly sigma-latent-noise pairs
(z0 + sigma*randn, z0) for generalization across waviness levels -- this mode needs no VQVAE
decode/re-encode round trip at all since corruption and target are both already in latent space.

Corrector: `LatentCorrectorUNet3D` (models/networks/refine_unet.py) -- residual 3D UNet on
(B,3,16,16,16), zero-init output conv (starts at identity, same contract as `RefineUNet3D`).
Loss: latent L1 (primary) with an optional small decoded-space `surface_weighted_l1` term
(--w_decode, default 0.0 -- try pure latent L1 first per the ticket).

Evaluation (the actual de-risk) runs on the diffusion's ACTUAL held-out real prior samples (never
SDEdit): z = vqvae.encode(wavy_sdf); corrected = vqvae.decode_no_quant(g(z)). Roughness/fp_iou
compare decode(z) ("before", the fair round-tripped baseline -- isolates the corrector's own
effect from any residual codec round-trip artifact) against decode(g(z)) ("after").

Run (smoke): env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/train_latent_corrector.py \
    --steps 6 --n_pairs 16 --n_sdedit 8 --calib_n_probe 4 --sdedit_calib_probe 4 --n_val 4 --n_montage 2
Run (at-scale): env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/train_latent_corrector.py \
    --n_pairs 2000 --n_sdedit 800 --steps 3000 --batch 8 --base 48
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(REPO))

from scripts.foundations.baseline_gate_eval import build_opt, fp_iou
from scripts.foundations.refiner_prototype import surface_roughness, calibrate_sigma
from scripts.foundations.train_refiner import (
    build_conditioned_pool, encode_pool_z0, calibrate_sdedit_strength, build_sdedit_pool,
)
from models.networks.refine_unet import LatentCorrectorUNet3D, surface_weighted_l1

CKPT = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"


def save_corrector_ckpt(corrector: LatentCorrectorUNet3D, path, channels: int, base: int,
                        delta_scale: float, step: int, **extra):
    """Checkpoint format consumed by `load_latent_corrector`: architecture hyperparams travel WITH
    the weights so a ckpt reconstructs the exact trained LatentCorrectorUNet3D (mirrors
    train_refiner.save_refiner_ckpt / baseline_gate_eval.load_refiner for the SDF-space refiner)."""
    payload = dict(state_dict=corrector.state_dict(), channels=int(channels), base=int(base),
                   delta_scale=float(delta_scale), step=int(step), **extra)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_latent_corrector(path, device="cpu") -> LatentCorrectorUNet3D:
    """Load a trained LatentCorrectorUNet3D as a frozen post-process on a VQVAE latent. Zero-init
    output conv means an UNTRAINED checkpoint loaded here is provably the identity map -- see the
    CPU-only contract test in test_latent_corrector.py."""
    state = torch.load(path, map_location=device)
    channels = int(state.get("channels", 3))
    base = int(state.get("base", 48))
    delta_scale = float(state.get("delta_scale", 1.0))
    corrector = LatentCorrectorUNet3D(channels=channels, base=base, delta_scale=delta_scale)
    corrector.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    corrector.eval()
    for p in corrector.parameters():
        p.requires_grad = False
    return corrector.to(device)


def train_corrector_mixed(model, z0s: torch.Tensor, targets_sdf: torch.Tensor,
                          wavy_sdedit_z: torch.Tensor, sdedit_idx: np.ndarray,
                          sigma_lo: float, sigma_hi: float, p_sdedit: float,
                          steps: int, batch: int, lr: float, channels: int, base: int,
                          delta_scale: float, device, w_decode: float = 0.0, band: float = 0.1,
                          ckpt_path=None, ckpt_every: int = 500, log_every: int = 100):
    """Mixed-corruption training loop in LATENT space (train_refiner's `train_refiner_mixed`,
    moved from decoded-SDF-space corruption to latent-space corruption). Each step's batch mixes
    (a) precomputed SDEdit-corrupted latents (real prior waviness, GT-aligned, encoded from
    `build_sdedit_pool`'s decoded output) and (b) on-the-fly sigma-latent-noise pairs (sigma
    resampled fresh per step) -- (b) needs no VQVAE call at all since both input and target already
    live in latent space. Loss is latent L1 (primary); `w_decode`>0 adds a small decoded-space
    surface_weighted_l1 anchor (off by default -- the ticket's "try pure latent L1 first")."""
    corrector = LatentCorrectorUNet3D(channels=channels, base=base, delta_scale=delta_scale).to(device)
    opt = torch.optim.Adam(corrector.parameters(), lr=lr)
    n_total = z0s.shape[0]
    n_sdedit = len(sdedit_idx)
    sdedit_idx_t = torch.as_tensor(sdedit_idx, device=device, dtype=torch.long)
    wavy_sdedit_z = wavy_sdedit_z.to(device)
    losses = []
    t0 = time.time()
    for step in range(steps):
        n_se = min(int(round(batch * p_sdedit)), batch) if n_sdedit > 0 else 0
        n_sig = batch - n_se
        parts_in, parts_tgt, parts_tgt_idx = [], [], []
        if n_se > 0:
            pick = torch.randint(0, n_sdedit, (n_se,), device=device)
            local_idx = sdedit_idx_t[pick]
            parts_in.append(wavy_sdedit_z[pick])
            parts_tgt.append(z0s[local_idx])
            parts_tgt_idx.append(local_idx)
        if n_sig > 0:
            pick2 = torch.randint(0, n_total, (n_sig,), device=device)
            z0b = z0s[pick2]
            sigma = torch.empty(n_sig, 1, 1, 1, 1, device=device).uniform_(sigma_lo, sigma_hi)
            zw = z0b + sigma * torch.randn_like(z0b)
            parts_in.append(zw)
            parts_tgt.append(z0b)
            parts_tgt_idx.append(pick2)
        inp = torch.cat(parts_in, 0)
        tgt = torch.cat(parts_tgt, 0)
        pred = corrector(inp)
        latent_l1 = (pred - tgt).abs().mean()
        loss = latent_l1
        comps = {"latent_l1": float(latent_l1.detach())}
        if w_decode > 0:
            tgt_idx = torch.cat(parts_tgt_idx, 0)
            decoded = model.vqvae.decode_no_quant(pred)
            decode_l1 = surface_weighted_l1(decoded, targets_sdf[tgt_idx], band=band)
            comps["decode_l1"] = float(decode_l1.detach())
            loss = loss + w_decode * decode_l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            extra = " ".join(f"{k}={v:.5f}" for k, v in comps.items())
            print(f"  [train {step+1}/{steps}] loss={losses[-1]:.5f} [{extra}] "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)
        if ckpt_path and (step + 1) % ckpt_every == 0:
            save_corrector_ckpt(corrector, ckpt_path, channels, base, delta_scale, step + 1)
    if ckpt_path:
        save_corrector_ckpt(corrector, ckpt_path, channels, base, delta_scale, steps)
    return corrector, losses


def save_latent_corrector_montage(rows, path):
    """GT | wavy (decode z) | corrected (decode g(z)) montage, >=6 rows. HONEST SHADING (the whole
    point is judging crispness -- a height colormap would hide it): mesh each field with the
    continuous-SDF marching-cubes convention (`mesh_sdf_surface`, @0.0, never binary occupancy),
    then render shaded solid surfaces with a fixed light source, not a value colormap."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from scripts.foundations.baseline_gate_eval import mesh_sdf_surface

    ls = LightSource(azdeg=315, altdeg=50)
    fig = plt.figure(figsize=(9, 3 * len(rows)))
    cols = [("GT", "gt_sdf"), ("wavy (decode z)", "wavy_sdf"), ("corrected (decode g(z))", "corrected_sdf")]
    for ri, row in enumerate(rows):
        for ci, (title, key) in enumerate(cols):
            ax = fig.add_subplot(len(rows), 3, ri * 3 + ci + 1, projection="3d"); ax.set_axis_off()
            vol = row.get(key)
            if vol is not None:
                v, f = mesh_sdf_surface(vol)
                if v is not None:
                    ax.plot_trisurf(v[:, 2], v[:, 0], f, v[:, 1], color=(0.82, 0.75, 0.60),
                                    shade=True, lightsource=ls, edgecolor="none", linewidth=0,
                                    antialiased=False)
                    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
                    ax.view_init(elev=24, azim=-58)
            if ri == 0: ax.set_title(title, fontsize=10)
            if ci == 0:
                sub = (f"region {row['region']}\nrough {row['roughness_before']:.4f}"
                      f"→{row['roughness_after']:.4f}\nIoU {row['fp_iou_before']:.2f}"
                      f"→{row['fp_iou_after']:.2f}")
                ax.text2D(-0.15, 0.5, sub, transform=ax.transAxes, fontsize=8)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n_pairs", type=int, default=2000, help="# GT buildings in the training pool (train split)")
    ap.add_argument("--n_sdedit", type=int, default=800, help="# of the pool given a precomputed SDEdit-corrupted version")
    ap.add_argument("--p_sdedit", type=float, default=0.5, help="per-step batch fraction: SDEdit-precomputed vs on-the-fly sigma corruption")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sigma_lo", type=float, default=0.10, help="latent-noise sigma lower bound (raw-latent units)")
    ap.add_argument("--sigma_hi", type=float, default=0.20, help="latent-noise sigma upper bound")
    ap.add_argument("--sdedit_ddim_steps", type=int, default=100, help="DDIM schedule length for SDEdit corruption (matches production)")
    ap.add_argument("--sdedit_strengths", default="0.15,0.25,0.35,0.45,0.55", help="comma-separated calibration sweep candidates")
    ap.add_argument("--sdedit_strength_lo", type=float, default=None, help="override the calibrated pick's jitter range")
    ap.add_argument("--sdedit_strength_hi", type=float, default=None)
    ap.add_argument("--sdedit_calib_probe", type=int, default=12)
    ap.add_argument("--sdedit_batch", type=int, default=32, help="batch size for precomputing the SDEdit-corrupted pool")
    ap.add_argument("--calib_n_probe", type=int, default=24, help="floor for calibrate_sigma's real-prior probe count (max(this, n_val))")
    ap.add_argument("--channels", type=int, default=3, help="VQVAE raw latent channels")
    ap.add_argument("--base", type=int, default=48, help="LatentCorrectorUNet3D base channel width")
    ap.add_argument("--delta_scale", type=float, default=1.0, help="tanh-bounded residual magnitude (raw latent std ~1/scale_factor ~0.42)")
    ap.add_argument("--w_decode", type=float, default=0.0, help="weight of an optional decoded-space surface_weighted_l1 anchor term (0 = pure latent L1, per the ticket's default)")
    ap.add_argument("--band", type=float, default=0.1, help="near-surface band for the optional decoded-space loss")
    ap.add_argument("--n_val", type=int, default=24, help="# real prior held-out samples for roughness/fp_iou/montage (>=20 per #59)")
    ap.add_argument("--n_montage", type=int, default=6)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/latent_corrector")
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
    # freeze the codec explicitly (the ticket's "codec AND diffusion frozen" -- diffusion is
    # already only ever called under model.inference/sdedit's own @torch.no_grad).
    model.vqvae.eval()
    for p in model.vqvae.parameters():
        p.requires_grad = False

    ds_train = Bag3dDataset(); ds_train.initialize(opt, phase="train")
    ds_test = Bag3dDataset(); ds_test.initialize(opt, phase="test")
    rng = np.random.default_rng(a.seed)
    t_start = time.time()

    # 1) sigma calibration (reused verbatim) -- also yields n_val real prior samples (held-out test
    #    conditioning) reused below for BOTH the sdedit-strength calibration reference point AND the
    #    roughness/fp_iou/montage validation, so DDIM sampling isn't paid for twice.
    n_probe = max(a.calib_n_probe, a.n_val)
    print(f"[calibrate-sigma] sigma sweep vs real prior roughness (n_probe={n_probe})", flush=True)
    calib_sigma, prior_samples = calibrate_sigma(
        model, ds_train, ds_test, device, rng,
        sigmas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30], n_probe=n_probe)
    print(f"  prior_roughness_mean={calib_sigma['prior_roughness_mean']:.5f}  "
          f"gt_roughness_mean={calib_sigma['gt_roughness_mean']:.5f}", flush=True)

    # 2) conditioned training pool (crisp targets + full conditioning), then RAW z0 latents (no
    #    scale_factor -- decode_no_quant expects the raw latent, per the ticket's pitfall note).
    print(f"[pool] caching {a.n_pairs} GT (target, conditioning) buildings from train split", flush=True)
    pool = build_conditioned_pool(ds_train, a.n_pairs, rng)
    targets_sdf = pool["sdf"].to(device)
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

    # 4) precompute the SDEdit-corrupted subpool (real prior waviness, GT-aligned SDF), then encode
    #    to RAW latents -- the actual (z_wavy, z_clean) training pairs.
    n_sdedit = min(a.n_sdedit, a.n_pairs)
    sdedit_idx = rng.choice(a.n_pairs, size=n_sdedit, replace=False)
    print(f"[sdedit-pool] precomputing {n_sdedit} SDEdit-corrupted pairs (strength~[{lo:.2f},{hi:.2f}], "
          f"ddim={a.sdedit_ddim_steps}, batch={a.sdedit_batch})", flush=True)
    wavy_sdedit_sdf = build_sdedit_pool(model, pool, sdedit_idx, device, lo, hi,
                                        a.sdedit_ddim_steps, rng, batch_size=a.sdedit_batch)
    wavy_sdedit_z = encode_pool_z0(model, wavy_sdedit_sdf, device)

    # 5) train, mixing SDEdit-precomputed + on-the-fly sigma-augmented latent corruption every step.
    ckpt_path = out_dir / "latent_corrector.pth"
    print(f"[train] LatentCorrectorUNet3D channels={a.channels} base={a.base} delta_scale={a.delta_scale} "
          f"steps={a.steps} batch={a.batch} lr={a.lr} p_sdedit={a.p_sdedit} sigma=[{a.sigma_lo},{a.sigma_hi}] "
          f"w_decode={a.w_decode}", flush=True)
    corrector, losses = train_corrector_mixed(
        model, z0s, targets_sdf, wavy_sdedit_z, sdedit_idx,
        a.sigma_lo, a.sigma_hi, a.p_sdedit, a.steps, a.batch, a.lr, a.channels, a.base, a.delta_scale,
        device, w_decode=a.w_decode, band=a.band, ckpt_path=ckpt_path, ckpt_every=a.ckpt_every)
    corrector.eval()
    print(f"[ckpt] saved {ckpt_path}", flush=True)

    # 6) validation: >=20 held-out REAL prior samples (never used to build training pairs), the
    #    diffusion's ACTUAL wavy SDF -- not SDEdit. z = encode(wavy_sdf); "before" = decode(z) (fair
    #    round-tripped baseline, isolates the corrector's own effect from any residual codec
    #    round-trip artifact); "after" = decode(g(z)).
    rows = []
    with torch.no_grad():
        for region_id, gen, real_fp, real_sdf in prior_samples[: a.n_val]:
            z = model.vqvae(gen, forward_no_quant=True, encode_only=True)
            wavy_decoded = model.vqvae.decode_no_quant(z)
            corrected_z = corrector(z)
            corrected_decoded = model.vqvae.decode_no_quant(corrected_z)

            wavy_np = wavy_decoded.detach().cpu().numpy()[0, 0]
            corrected_np = corrected_decoded.detach().cpu().numpy()[0, 0]
            iou_before = fp_iou(wavy_np <= 0, real_fp)
            iou_after = fp_iou(corrected_np <= 0, real_fp)
            rough_before = surface_roughness(wavy_decoded)
            rough_after = surface_roughness(corrected_decoded)
            rows.append(dict(
                region=region_id, fp_iou_before=iou_before, fp_iou_after=iou_after,
                roughness_before=rough_before, roughness_after=rough_after,
                wavy_sdf=wavy_np, corrected_sdf=corrected_np, gt_sdf=real_sdf,
            ))
            print(f"  [val] region={region_id} fp_iou {iou_before:.3f} -> {iou_after:.3f}  "
                  f"roughness {rough_before:.5f} -> {rough_after:.5f}", flush=True)

    montage_path = save_latent_corrector_montage(rows[: a.n_montage], out_dir / "montage.png")
    print(f"[montage] saved {montage_path}", flush=True)
    wf_dir = REPO / "docs/wayfinding/diffusion-latent-accuracy"
    wf_dir.mkdir(parents=True, exist_ok=True)
    wf_path = wf_dir / "latent-corrector-montage.png"
    shutil.copyfile(montage_path, wf_path)
    print(f"[montage] copied to {wf_path}", flush=True)

    iou_before_list = [r["fp_iou_before"] for r in rows]
    iou_after_list = [r["fp_iou_after"] for r in rows]
    rough_before_list = [r["roughness_before"] for r in rows]
    rough_after_list = [r["roughness_after"] for r in rows]
    result = dict(
        ckpt=a.ckpt, n_pairs=a.n_pairs, n_sdedit=n_sdedit, p_sdedit=a.p_sdedit,
        steps=a.steps, batch=a.batch, lr=a.lr, channels=a.channels, base=a.base,
        delta_scale=a.delta_scale, w_decode=a.w_decode, band=a.band,
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
        refiner_v1_plateau_roughness=0.0047,  # #54 SDF-space post-decode plateau, the bar to beat
        montage=str(montage_path), corrector_ckpt=str(ckpt_path),
        wall_time_sec=time.time() - t_start,
    )
    out = REPO / "execution/artifacts/latent_corrector_train.json"
    out.write_text(json.dumps(result, indent=2))
    print("\n=== RESULT ===", flush=True)
    print(f"  fp_iou before={result['fp_iou_before_mean']:.3f} after={result['fp_iou_after_mean']:.3f}", flush=True)
    print(f"  roughness (n={len(rows)}) before={result['roughness_before_mean']:.5f} "
          f"after={result['roughness_after_mean']:.5f} (GT floor={result['gt_roughness_mean']:.5f}, "
          f"#54 plateau={result['refiner_v1_plateau_roughness']:.5f})", flush=True)
    print(f"  wall_time={result['wall_time_sec']:.0f}s", flush=True)
    print(f"  artifact: {out}", flush=True)


if __name__ == "__main__":
    main()
