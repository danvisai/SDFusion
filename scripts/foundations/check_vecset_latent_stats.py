"""#70 -- validate the vecset latent normalisation before any long run.

`train_vecset.py` reduces the whole (N, 2048, 64) latent to ONE scalar mean and ONE scalar std
(`LatentSet.__init__`) and applies them to every entry. Both A2 runs trained through that, and it was
never checked. This script answers whether that is sound, because if it is not, no length of run fixes it.

Three questions, in the order that matters:

1. **Is the normalised space actually unit-scale PER CHANNEL?** The cosine schedule adds noise of unit
   variance to every channel alike. If per-channel stds differ widely, one global divisor leaves some
   channels swamped by noise at low t and others barely perturbed at high t -- the schedule is then not
   well-posed on this tensor, whatever the aggregate std says.
2. **Is the distribution heavy-tailed?** A heavy-tailed latent makes both the schedule and the +/-3 sigma
   clamp mis-scaled. Reported as excess kurtosis per channel.
3. **Do the REAL and BLOCKOUT caches share a scale?** `__getitem__` normalises the blockout latent with
   the statistics of the *real* latents. At inference the generator STARTS from a blockout. If the two
   distributions differ in scale, every projection begins off-distribution by construction -- which would
   be a second, independent cause of the same symptom the aligned-pair run was built to fix.

Sampling: statistics over a random subsample of rows (all 2048 tokens of each), which is ample for
mean/std/kurtosis and avoids reading 9.3 GB twice. Tokens in a vecset are order-agnostic, so per-token
statistics are reported only as an exchangeability sanity check, not as a normalisation candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

REAL = "data/real_massing_v1/vecset_latents.h5"
BLOCKOUT = "data/real_massing_v1/vecset_blockout_latents.h5"


def load_sample(path: str, n_rows: int, held_out: bool, seed: int = 0):
    """Rows from the requested split, matching how LatentSet selects them."""
    with h5py.File(path, "r") as f:
        mask = (f["held_out"][:] == (1 if held_out else 0))
        idx = np.flatnonzero(mask)
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(idx, size=min(n_rows, len(idx)), replace=False))
        # h5py fancy-indexing on the first axis is fine and stays out-of-core until here
        z = f["latent"][take].astype(np.float32)
        rows = f["row"][take]
    return z, take, rows


def describe(z: np.ndarray) -> dict:
    """z: (R, T, C) float32."""
    flat = z.reshape(-1, z.shape[-1])                     # (R*T, C)
    g_mu, g_sd = float(flat.mean()), float(flat.std())

    ch_mu = flat.mean(axis=0)                             # (C,)
    ch_sd = flat.std(axis=0)

    # excess kurtosis per channel, on the globally-normalised values
    zn = (flat - g_mu) / (g_sd or 1.0)
    ch_kurt = ((zn - zn.mean(axis=0)) ** 4).mean(axis=0) / (zn.std(axis=0) ** 4 + 1e-12) - 3.0

    # what one global divisor leaves behind: the per-channel std AFTER global normalisation
    resid_sd = ch_sd / (g_sd or 1.0)

    return {
        "global_mean": g_mu,
        "global_std": g_sd,
        "per_channel_std": {
            "min": float(ch_sd.min()), "p10": float(np.percentile(ch_sd, 10)),
            "median": float(np.median(ch_sd)), "p90": float(np.percentile(ch_sd, 90)),
            "max": float(ch_sd.max()),
            "max_over_min": float(ch_sd.max() / max(ch_sd.min(), 1e-12)),
        },
        "per_channel_mean": {
            "min": float(ch_mu.min()), "median": float(np.median(ch_mu)),
            "max": float(ch_mu.max()),
            "max_abs": float(np.abs(ch_mu).max()),
        },
        "after_global_norm_channel_std": {
            "min": float(resid_sd.min()), "median": float(np.median(resid_sd)),
            "max": float(resid_sd.max()),
            "frac_outside_0p5_to_2p0": float(np.mean((resid_sd < 0.5) | (resid_sd > 2.0))),
        },
        "excess_kurtosis": {
            "min": float(ch_kurt.min()), "median": float(np.median(ch_kurt)),
            "max": float(ch_kurt.max()),
        },
        "tail": {
            "frac_abs_gt_3_global_sd": float(np.mean(np.abs(zn) > 3.0)),
            "frac_abs_gt_5_global_sd": float(np.mean(np.abs(zn) > 5.0)),
            "max_abs_global_sd": float(np.abs(zn).max()),
        },
        "_ch_sd": ch_sd, "_ch_mu": ch_mu, "_ch_kurt": ch_kurt,
    }


def fp16_roundtrip(path: str, n_rows: int = 8) -> dict:
    """What the fp16 cache costs, relative to the latent's own scale.

    normalise/denormalise are exact float32 inverses, so the only lossy step in
    encode -> cache -> normalise -> denormalise -> decode is the fp16 STORE.
    """
    with h5py.File(path, "r") as f:
        z16 = f["latent"][:n_rows]
    z32 = z16.astype(np.float32)
    back = z32.astype(np.float16).astype(np.float32)
    err = np.abs(back - z32)
    return {
        "max_abs_err": float(err.max()),
        "rel_err_vs_std": float(err.max() / (z32.std() + 1e-12)),
        "note": "fp16 store is the only lossy step; normalise/denormalise are exact float32 inverses",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600, help="rows sampled per cache (x2048 tokens each)")
    ap.add_argument("--out", default="execution/artifacts/vecset_latent_stats.json")
    args = ap.parse_args()

    report: dict = {"rows_sampled": args.rows}

    for name, path in (("real", REAL), ("blockout", BLOCKOUT)):
        z, _, _ = load_sample(path, args.rows, held_out=False)
        d = describe(z)
        report[name] = {k: v for k, v in d.items() if not k.startswith("_")}
        report[name]["shape_sampled"] = list(z.shape)
        print(f"\n=== {name} ({path}) ===")
        print(f"  global   mean {d['global_mean']:+.4f}  std {d['global_std']:.4f}")
        p = d["per_channel_std"]
        print(f"  ch std   min {p['min']:.4f}  med {p['median']:.4f}  max {p['max']:.4f}"
              f"   max/min {p['max_over_min']:.2f}x")
        r = d["after_global_norm_channel_std"]
        print(f"  after global norm, ch std: {r['min']:.3f} .. {r['max']:.3f}"
              f"  ({r['frac_outside_0p5_to_2p0']*100:.1f}% outside [0.5, 2.0])")
        k = d["excess_kurtosis"]
        print(f"  excess kurtosis  min {k['min']:+.2f}  med {k['median']:+.2f}  max {k['max']:+.2f}")
        t = d["tail"]
        print(f"  tail  |z|>3sd {t['frac_abs_gt_3_global_sd']*100:.3f}%   "
              f"|z|>5sd {t['frac_abs_gt_5_global_sd']*100:.4f}%   max {t['max_abs_global_sd']:.2f}sd")
        report[name]["_ch_sd_head"] = [float(x) for x in d["_ch_sd"][:8]]

    # --- the cross-cache question: does the blockout share the real latent's scale? ---
    zr, _, _ = load_sample(REAL, args.rows, held_out=False)
    zb, _, _ = load_sample(BLOCKOUT, args.rows, held_out=False)
    mu_r, sd_r = float(zr.mean()), float(zr.std())
    zb_under_real = (zb - mu_r) / sd_r          # exactly what __getitem__ does to a blockout
    report["blockout_under_real_stats"] = {
        "mean": float(zb_under_real.mean()),
        "std": float(zb_under_real.std()),
        "comment": "training normalises the blockout with the REAL latents' mu/sd; ideal is mean 0 std 1",
    }
    print("\n=== blockout normalised by REAL statistics (what training actually does) ===")
    print(f"  mean {zb_under_real.mean():+.4f}   std {zb_under_real.std():.4f}   (ideal 0.0 / 1.0)")

    ch_r = zr.reshape(-1, zr.shape[-1]).std(axis=0)
    ch_b = zb.reshape(-1, zb.shape[-1]).std(axis=0)
    report["per_channel_std_correlation_real_vs_blockout"] = float(np.corrcoef(ch_r, ch_b)[0, 1])
    print(f"  per-channel std correlation real vs blockout: "
          f"{report['per_channel_std_correlation_real_vs_blockout']:.4f}")

    report["fp16_roundtrip_real"] = fp16_roundtrip(REAL)
    print(f"\n=== fp16 cache fidelity ===")
    print(f"  max abs err {report['fp16_roundtrip_real']['max_abs_err']:.2e}  "
          f"= {report['fp16_roundtrip_real']['rel_err_vs_std']:.2e} of the latent std")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
