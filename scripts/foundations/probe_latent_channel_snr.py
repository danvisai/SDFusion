"""#70 -- does one global mean/std leave channels the noise schedule destroys, and do those channels matter?

`check_vecset_latent_stats.py` established the distribution: per-channel std spans **59x** on the real
cache, and one global divisor leaves 25% of channels outside [0.5, 2.0]. This script closes the two gaps
that leaves, because the statistic alone does not license a fix.

**1. The mechanism, quantified.** The projection noises to `t_start = strength * (T-1)` and the schedule
adds unit-variance epsilon to every channel alike:

    x_t = sqrt(a_t) * x_0 + sqrt(1 - a_t) * eps

so per channel, after global normalisation to std sigma_c, the signal-to-noise ratio at that step is

    SNR_c = a_t * sigma_c^2 / (1 - a_t)

A channel with sigma_c = 0.026 is at SNR ~1e-3 wherever a well-scaled channel sits at ~1: it is pure
noise, and the denoiser cannot recover what the schedule erased. Its content is effectively **resampled
from the prior on every projection**.

**2. Whether that matters -- the part the statistic cannot answer.** A near-zero-variance channel in a
KL-regularised latent is usually a *collapsed* dimension the decoder ignores. If these channels are dead,
the 59x spread is cosmetic and per-channel normalisation would be actively harmful: it would amplify
encoder noise to unit scale and spend model capacity modelling it. If they carry shape, the spread is a
mechanism for the melt.

So we ablate them **through the real decoder** and measure whether the decoded surface moves:

  * ``full``          -- decode(z), the reference
  * ``low_to_mean``   -- low-variance channels replaced by their per-channel mean (is the content needed?)
  * ``low_shuffled``  -- low-variance channels taken from a DIFFERENT building (do they carry per-shape
                         information, as opposed to a constant the decoder happens to want?)
  * ``high_to_mean``  -- **control arm**: the same NUMBER of highest-variance channels blanked. This must
                         degrade badly. If it does not, the ablation is insensitive and no conclusion
                         about the low channels is licensed either way.

The headline metric is ``IoU(ablated decode, full decode)`` -- a paired, same-building comparison that
isolates the channels' contribution without GT error confounding it. fp-IoU and 3D IoU against GT are
reported alongside so the numbers stay on the map's criteria.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks.vecset_projection import cosine_alphas                   # noqa: E402
from models.shape_codec import DoraCodec                                      # noqa: E402
from scripts.foundations.baseline_gate_eval import fp_iou                     # noqa: E402
from scripts.foundations.dora_roundtrip_probe import load_dora, H5            # noqa: E402
from scripts.foundations.vecset_ceiling_probe import RES                      # noqa: E402

LATENTS = "data/real_massing_v1/vecset_latents.h5"


def snr_table(ch_sd_norm: np.ndarray, timesteps: int, strengths) -> dict:
    """Per-channel SNR at each projection start, using the SAME schedule the model trains on."""
    ac = cosine_alphas(timesteps).numpy()
    out = {}
    for s in strengths:
        t = int(min(max(s, 0.0), 1.0) * (timesteps - 1))
        a = float(ac[t])
        snr = a * ch_sd_norm ** 2 / max(1.0 - a, 1e-12)
        out[f"s{s}"] = {
            "t_start": t,
            "alpha_bar": a,
            "snr_unit_channel": a / max(1.0 - a, 1e-12),
            "snr_min": float(snr.min()),
            "snr_median": float(np.median(snr)),
            "snr_max": float(snr.max()),
            "n_channels_snr_lt_1": int((snr < 1.0).sum()),
            "n_channels_snr_lt_0p1": int((snr < 0.1).sum()),
            "n_channels_snr_lt_0p01": int((snr < 0.01).sum()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="held-out buildings for the ablation")
    ap.add_argument("--rows_stats", type=int, default=400, help="rows for the channel statistics")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="normalised per-channel std below which a channel counts as low-variance")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--strengths", type=float, nargs="*", default=[0.15, 0.2, 0.35, 0.5, 0.65])
    ap.add_argument("--out", default="execution/artifacts/latent_channel_snr.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep: dict = {"thresh_normalised_std": args.thresh}

    # ---- channel statistics on the TRAIN split, exactly as LatentSet computes mu/sd ----
    with h5py.File(LATENTS, "r") as f:
        tr = np.flatnonzero(f["held_out"][:] == 0)
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(tr, size=min(args.rows_stats, len(tr)), replace=False))
        zt = f["latent"][take].astype(np.float32)
    mu, sd = float(zt.mean()), float(zt.std())
    flat = zt.reshape(-1, zt.shape[-1])
    ch_sd_norm = flat.std(axis=0) / sd                     # per-channel std AFTER global normalisation
    ch_mu = flat.mean(axis=0)
    del zt, flat

    order = np.argsort(ch_sd_norm)
    low = np.flatnonzero(ch_sd_norm < args.thresh)
    k = len(low)
    high = order[-k:] if k else np.array([], int)
    rep["global_mu"], rep["global_sd"] = mu, sd
    rep["n_channels"] = int(len(ch_sd_norm))
    rep["n_low"] = int(k)
    rep["low_channels"] = [int(c) for c in low]
    rep["low_channel_std_norm"] = [round(float(ch_sd_norm[c]), 4) for c in low]
    rep["high_channels_control"] = [int(c) for c in high]
    print(f"[stats] mu={mu:+.4f} sd={sd:.4f}   {k}/{len(ch_sd_norm)} channels below "
          f"normalised std {args.thresh}")
    print(f"        low  std_norm: {np.round(ch_sd_norm[low], 3).tolist()}")
    print(f"        high std_norm: {np.round(ch_sd_norm[high], 3).tolist()}")

    # ---- the mechanism: SNR per channel at each projection start ----
    rep["snr"] = snr_table(ch_sd_norm, args.timesteps, args.strengths)
    print(f"\n{'strength':>9} {'t':>5} {'alpha_bar':>10} {'SNR@sd=1':>9} "
          f"{'SNR<1':>6} {'SNR<0.1':>8} {'SNR<0.01':>9}")
    for s in args.strengths:
        d = rep["snr"][f"s{s}"]
        print(f"{s:>9} {d['t_start']:>5} {d['alpha_bar']:>10.4f} {d['snr_unit_channel']:>9.3f} "
              f"{d['n_channels_snr_lt_1']:>6} {d['n_channels_snr_lt_0p1']:>8} "
              f"{d['n_channels_snr_lt_0p01']:>9}")

    if k == 0:
        print("\nno low-variance channels -- ablation skipped")
        Path(args.out).write_text(json.dumps(rep, indent=1))
        return

    # ---- do those channels matter? ablate through the real decoder ----
    codec = DoraCodec(load_dora(dev))
    with h5py.File(LATENTS, "r") as f:
        ho = np.flatnonzero(f["held_out"][:] == 1)[:args.n]
        Z = f["latent"][ho].astype(np.float32)
        FP = f["footprint"][ho]
        ROW = f["row"][ho]

    def decode(z: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(z)[None].to(dev)
        return codec.decode_grid(t, RES).cpu().numpy()[0, 0]

    arms = ["low_to_mean", "low_shuffled", "high_to_mean"]
    acc = {a: {"iou_vs_full": [], "fp": [], "vol": []} for a in arms}
    acc["full"] = {"fp": [], "vol": []}

    with h5py.File(H5, "r") as gt:
        for i in range(len(ho)):
            g = np.asarray(gt["sdf"][int(ROW[i])], np.float32)
            gocc = g <= 0
            fp = FP[i]

            base = decode(Z[i])
            bocc = base <= 0
            acc["full"]["fp"].append(fp_iou(bocc, fp))
            acc["full"]["vol"].append((bocc & gocc).sum() / max((bocc | gocc).sum(), 1))

            for arm in arms:
                z = Z[i].copy()
                if arm == "low_to_mean":
                    z[:, low] = ch_mu[low]
                elif arm == "high_to_mean":
                    z[:, high] = ch_mu[high]
                else:                                       # low_shuffled -- a different building
                    z[:, low] = Z[(i + 1) % len(ho)][:, low]
                d = decode(z)
                occ = d <= 0
                acc[arm]["iou_vs_full"].append((occ & bocc).sum() / max((occ | bocc).sum(), 1))
                acc[arm]["fp"].append(fp_iou(occ, fp))
                acc[arm]["vol"].append((occ & gocc).sum() / max((occ | gocc).sum(), 1))
            print(f"  {i+1}/{len(ho)}", flush=True)

    print(f"\n=== channel ablation through the decoder (n={len(ho)} held-out, medians) ===")
    print(f"{'arm':16s} {'IoU vs full':>12} {'fp-IoU':>8} {'3D IoU':>8}")
    print(f"{'full (ref)':16s} {'-':>12} {np.median(acc['full']['fp']):>8.3f} "
          f"{np.median(acc['full']['vol']):>8.3f}")
    for arm in arms:
        tag = arm + (" [CONTROL]" if arm == "high_to_mean" else "")
        print(f"{tag:16s} {np.median(acc[arm]['iou_vs_full']):>12.4f} "
              f"{np.median(acc[arm]['fp']):>8.3f} {np.median(acc[arm]['vol']):>8.3f}")

    rep["ablation"] = {
        a: {kk: float(np.median(vv)) for kk, vv in d.items()} for a, d in acc.items()
    }
    rep["ablation"]["n"] = len(ho)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
