"""#94: does the surface term's gradient fight the epsilon term's, and does alignment worsen it?

`train_vecset.py` sums the two losses at the model's raw output: ``loss = eps_loss + surf_weight *
surf``, both computed from ``pred``. That sum is the one place all three of #94's candidates make a
different prediction:

- **decoder path intolerant** -- the surface gradient exists but the frozen decoder amplifies it far
  past the epsilon gradient's scale (a large ``surf_grad_norm / eps_grad_norm``), not necessarily an
  opposed direction.
- **displacement** -- alignment does not change the *relationship* between the two gradients, only how
  much epsilon pull remains once the surface term's weight is added; the cosine between them should be
  no different from the encoded regime.
- **disagreement, and it grows** -- the two gradients point in measurably more opposed directions
  (more negative cosine) once the target is aligned.

This probe computes ``d(eps_loss)/d(pred)`` and ``d(surf_loss)/d(pred)`` directly and compares their
cosine similarity, paired per (row, timestep), holding the model and the real target fixed and
changing only which blockout cache -- encoded or aligned -- sources the pair corruption. That is the
single-variable comparison #91 built the aligned cache for.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.probe_surface_loss_order import (
    _load_denoiser, _pair_step_draw, _shared_training_rows,
)

DEFAULT_REAL = REPO / "data/real_massing_v1/vecset_latents_v2.h5"
DEFAULT_ENCODED = REPO / "data/real_massing_v1/vecset_blockout_latents_v2.h5"
DEFAULT_ALIGNED = REPO / "data/real_massing_v1/vecset_blockout_latents_v2_aligned.h5"
DEFAULT_CHECKPOINT = REPO / "logs_building/vecset_v3_pair_long/vecset_denoiser_step180000.pth"
DEFAULT_OUT = REPO / "execution/artifacts/surface_gradient_conflict_probe.json"

REGIMES = ("encoded", "aligned")


def _stat(values: Sequence[float], fn):
    return float(fn(values)) if values else float("nan")


def summarize_conflict(observed: Mapping[str, Sequence[Mapping]]) -> dict:
    """Turn paired per-row gradient measurements into #94's candidate-discriminating report.

    Every row must carry both regimes so each comparison is paired -- same building, timestep, and
    diffusion noise, with only the blockout's token order changing -- which is what lets
    ``delta_cosine`` be attributed to alignment and nothing else.
    """
    by_t = {}
    for t_frac, rows in observed.items():
        if not rows:
            raise ValueError(f"t={t_frac} has no rows")
        for row in rows:
            missing = set(REGIMES) - set(row)
            if missing:
                raise ValueError(f"t={t_frac} row {row.get('row')} is missing regimes: {sorted(missing)}")

        cosines = {regime: [float(row[regime]["cosine"]) for row in rows] for regime in REGIMES}
        norm_ratios = {
            regime: [float(row[regime]["surf_grad_norm"]) / float(row[regime]["eps_grad_norm"])
                     for row in rows if float(row[regime]["eps_grad_norm"]) > 0]
            for regime in REGIMES
        }
        deltas = [row["aligned"]["cosine"] - row["encoded"]["cosine"] for row in rows]

        by_t[str(t_frac)] = {
            "n": len(rows),
            "encoded_cosine_mean": _stat(cosines["encoded"], mean),
            "aligned_cosine_mean": _stat(cosines["aligned"], mean),
            "encoded_cosine_median": _stat(cosines["encoded"], median),
            "aligned_cosine_median": _stat(cosines["aligned"], median),
            "delta_cosine_mean": _stat(deltas, mean),
            "n_more_conflicting_aligned": sum(1 for d in deltas if d < 0),
            "encoded_norm_ratio_mean": _stat(norm_ratios["encoded"], mean),
            "aligned_norm_ratio_mean": _stat(norm_ratios["aligned"], mean),
        }

    return {
        "observed": observed,
        "by_t": by_t,
        "aligned_more_conflicting_at_every_t": bool(by_t) and all(
            row["delta_cosine_mean"] < 0 for row in by_t.values()
        ),
        "gradients_oppose_on_average": bool(by_t) and all(
            row["encoded_cosine_mean"] < 0 and row["aligned_cosine_mean"] < 0 for row in by_t.values()
        ),
    }


def measure(args) -> dict:
    """Compute d(eps_loss)/d(pred) vs d(surf_loss)/d(pred) cosine similarity, paired by row and t."""
    import h5py
    import numpy as np
    import torch
    import torch.nn.functional as F

    from models.shape_codec import DoraCodec
    from models.networks.vecset_projection import cosine_alphas
    from scripts.foundations.dora_roundtrip_probe import load_dora
    from scripts.train_vecset import surface_term
    from utils.numeric_guard import check_numpy

    check_numpy()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, model = _load_denoiser(Path(args.a2), device)
    codec = DoraCodec(load_dora(device), differentiable=True).freeze()
    alphas = cosine_alphas(checkpoint["args"]["timesteps"]).to(device)
    mu = torch.as_tensor(checkpoint["latent_mu"], device=device)
    sd = torch.as_tensor(checkpoint["latent_sd"], device=device)
    rows = _shared_training_rows(Path(args.real_cache), Path(args.encoded_cache), args.n)

    observed = {f"{fraction:.2f}": [] for fraction in args.t_frac}
    t0 = time.time()
    with h5py.File(args.real_cache, "r") as real, \
         h5py.File(args.encoded_cache, "r") as encoded, \
         h5py.File(args.aligned_cache, "r") as aligned:
        for row, real_index, blockout_index in rows:
            z = torch.from_numpy(np.asarray(real["latent"][real_index], np.float32))[None].to(device)
            z = (z - mu) / sd
            footprint = torch.from_numpy(
                np.asarray(real["footprint"][real_index], np.float32)
            )[None, None].to(device)
            height = torch.tensor([float(real["height_m"][real_index])], device=device)
            region = torch.tensor([int(real["region"][real_index])], device=device)

            zb_by_regime = {}
            for name, h5 in (("encoded", encoded), ("aligned", aligned)):
                zb = torch.from_numpy(np.asarray(h5["latent"][blockout_index], np.float32))[None].to(device)
                zb_by_regime[name] = (zb - mu) / sd

            for fraction in args.t_frac:
                key = f"{fraction:.2f}"
                t, alpha, noise, points = _pair_step_draw(
                    row, fraction, checkpoint["args"]["timesteps"], alphas, args.seed,
                    args.surface_points, z.shape, device,
                )
                with torch.no_grad():
                    target_field = codec.query(z * sd + mu, points)

                row_result = {"row": row}
                for name in REGIMES:
                    src = zb_by_regime[name]
                    zt = alpha.sqrt() * src + (1 - alpha).sqrt() * noise
                    eps_target = (zt - alpha.sqrt() * z) / (1 - alpha).sqrt()
                    with torch.no_grad():
                        pred = model(x=zt, t=t, footprint=footprint, height=height,
                                     region=region, drop_cond=False)
                    # `pred_leaf` is a detached copy of the model's output, not the output itself --
                    # each loss below is backpropped independently onto this one leaf so the two
                    # gradients answer "what does each TERM alone want", the quantity #94 is about.
                    # Backward accumulates rather than overwrites, so the leaf's grad must be reset
                    # between the two calls or `surf_grad` would silently include `eps_grad`.
                    pred_leaf = pred.detach().clone().requires_grad_(True)

                    eps_loss = F.mse_loss(pred_leaf, eps_target)
                    eps_loss.backward()
                    eps_grad = pred_leaf.grad.detach().clone()
                    pred_leaf.grad = None

                    x0 = (zt - (1 - alpha).sqrt() * pred_leaf) / alpha.sqrt()
                    got_field = codec.query(x0 * sd + mu, points)
                    surf_loss, _ = surface_term(got_field, target_field, alpha)
                    surf_loss.backward()
                    surf_grad = pred_leaf.grad.detach().clone()

                    cosine = F.cosine_similarity(eps_grad.flatten(), surf_grad.flatten(), dim=0)
                    row_result[name] = {
                        "cosine": float(cosine),
                        "eps_grad_norm": float(eps_grad.norm()),
                        "surf_grad_norm": float(surf_grad.norm()),
                        "eps_loss": float(eps_loss.detach()),
                        "surf_loss": float(surf_loss.detach()),
                    }
                observed[key].append(row_result)
            print(f"  row {row} done ({sum(len(v) for v in observed.values())} regime-samples so far)",
                  flush=True)

    report = summarize_conflict(observed)
    report["meta"] = {
        "checkpoint": str(args.a2),
        "step": int(checkpoint["step"]),
        "real_cache": str(args.real_cache),
        "encoded_cache": str(args.encoded_cache),
        "aligned_cache": str(args.aligned_cache),
        "n": len(rows),
        "rows": [row for row, _, _ in rows],
        "t_frac": args.t_frac,
        "surface_points": args.surface_points,
        "seed": args.seed,
        "device": device,
        "seconds": time.time() - t0,
        "intervention": "hold model, real target, noise and query points fixed; vary only which "
                         "blockout cache (encoded vs aligned) sources the pair-training corruption",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements",
                        help="existing JSON containing an 'observed' mapping; skips model execution")
    parser.add_argument("--real-cache", default=str(DEFAULT_REAL))
    parser.add_argument("--encoded-cache", default=str(DEFAULT_ENCODED))
    parser.add_argument("--aligned-cache", default=str(DEFAULT_ALIGNED))
    parser.add_argument("--a2", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--t-frac", type=float, nargs="+", default=[0.40, 0.55, 0.70])
    parser.add_argument("--surface-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON report path")
    args = parser.parse_args()

    if args.measurements:
        source = json.loads(Path(args.measurements).read_text())
        report = summarize_conflict(source["observed"])
    else:
        report = measure(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
