"""Recover #89's measurement of order sensitivity in the pair-training objective.

The full probe is intentionally kept separate from ``probe_token_order.py``: that script answers
#95 at inference, while this one evaluates the two losses on the actual pair-training path.  The
JSON report is the public seam consumed by the ticket record.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_REAL = REPO / "data/real_massing_v1/vecset_latents_v2.h5"
DEFAULT_BLOCKOUT = REPO / "data/real_massing_v1/vecset_blockout_latents_v2.h5"
DEFAULT_CHECKPOINT = REPO / "logs_building/vecset_v4_surf/vecset_denoiser_step240000.pth"
DEFAULT_OUT = REPO / "execution/artifacts/surface_loss_order_probe.json"
PUBLISHED_SPREAD_PCT = {
    "0.40": {"epsilon": 1.53, "surface": 0.0003},
    "0.55": {"epsilon": 0.97, "surface": 0.0005},
    "0.70": {"epsilon": 0.31, "surface": 0.0001},
}


def _relative_spread(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if mean == 0.0:
        return 0.0 if float(np.ptp(values)) == 0.0 else float("inf")
    return float(100.0 * np.ptp(values) / abs(mean))


def build_report(observed: Mapping[str, Mapping[str, Sequence[float]]]) -> dict:
    """Turn per-ordering losses into the JSON-ready #89 comparison.

    ``sensitivity_ratio`` is epsilon spread divided by surface spread.  The boolean is deliberately
    a strong descriptive threshold, not a new model gate: it records whether surface loss was at
    least two orders of magnitude less order-sensitive at every measured timestep.
    """
    by_t = {}
    for t_frac, losses in observed.items():
        epsilon = list(losses["epsilon"])
        surface = list(losses["surface"])
        if len(epsilon) < 2 or len(surface) < 2:
            raise ValueError("each timestep needs at least two orderings")
        if len(epsilon) != len(surface):
            raise ValueError("epsilon and surface must have the same number of orderings")
        eps_spread = _relative_spread(epsilon)
        surface_spread = _relative_spread(surface)
        ratio = (float("inf") if surface_spread == 0.0 else eps_spread / surface_spread)
        by_t[str(t_frac)] = {
            "n_orderings": len(epsilon),
            "epsilon_spread_pct": eps_spread,
            "surface_spread_pct": surface_spread,
            "sensitivity_ratio": ratio,
        }
    return {
        "observed": observed,
        "by_t": by_t,
        "surface_is_at_least_100x_less_order_sensitive_at_every_t": bool(
            by_t and all(row["sensitivity_ratio"] >= 100.0 for row in by_t.values())
        ),
    }


def _shared_training_rows(real_path: Path, blockout_path: Path, n: int) -> list[tuple[int, int, int]]:
    """Return ``(global row, real index, blockout index)`` round-robin by source region."""
    import h5py

    with h5py.File(real_path, "r") as real, h5py.File(blockout_path, "r") as blockout:
        blockout_index = {int(row): i for i, row in enumerate(blockout["row"][:])}
        groups: dict[int, list[tuple[int, int, int]]] = {}
        held = np.asarray(real["held_out"])
        for real_index, (row, region) in enumerate(zip(real["row"][:], real["region"][:])):
            row = int(row)
            if held[real_index] or row not in blockout_index:
                continue
            groups.setdefault(int(region), []).append((row, real_index, blockout_index[row]))

    selected = []
    offset = 0
    regions = sorted(groups)
    while len(selected) < n and regions:
        next_regions = []
        for region in regions:
            if offset < len(groups[region]):
                selected.append(groups[region][offset])
                next_regions.append(region)
                if len(selected) == n:
                    break
        regions = next_regions
        offset += 1
    if len(selected) < n:
        raise ValueError(f"only {len(selected)} shared training rows are available; requested {n}")
    return selected


def _load_denoiser(path: Path, device: str):
    import torch
    from models.networks.vecset_denoiser import VecsetDenoiser

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model = VecsetDenoiser(
        latent_channels=checkpoint["latent_channels"],
        width=args["width"],
        depth=args["depth"],
        heads=args["heads"],
        footprint_res=checkpoint["footprint_res"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model


def _pair_step_draw(row: int, fraction: float, timesteps: int, alphas, seed: int,
                     surface_points: int, shape: Sequence[int], device: str):
    """Draw the timestep, alpha, noise and query points one pair-training step needs.

    Shared with ``probe_surface_gradient_conflict.py`` (#94): both probes must draw from
    identical seed streams for their measurements to sit on the same footing, not just be
    individually reproducible.
    """
    import torch

    timestep = min(int(round(fraction * timesteps)), timesteps - 1)
    t = torch.tensor([timestep], device=device, dtype=torch.long)
    alpha = alphas[t].view(1, 1, 1)
    generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + row * 97 + timestep)
    noise = torch.randn(tuple(shape), generator=generator).to(device)
    point_generator = torch.Generator(device="cpu").manual_seed(
        seed * 1_000_003 + row * 193 + timestep
    )
    points = (torch.rand((1, surface_points, 3), generator=point_generator) * 2 - 1).to(device)
    return t, alpha, noise, points


def measure(args) -> dict:
    """Measure both losses while changing only the token gauge of the pair source."""
    import h5py
    import torch

    from models.shape_codec import DoraCodec
    from models.networks.vecset_projection import cosine_alphas
    from scripts.foundations.dora_roundtrip_probe import load_dora
    from scripts.train_vecset import surface_term
    from utils.numeric_guard import check_numpy

    check_numpy()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, model = _load_denoiser(Path(args.a2), device)
    codec = DoraCodec(load_dora(device), differentiable=False).freeze()
    alphas = cosine_alphas(checkpoint["args"]["timesteps"]).to(device)
    mu = torch.as_tensor(checkpoint["latent_mu"], device=device)
    sd = torch.as_tensor(checkpoint["latent_sd"], device=device)
    rows = _shared_training_rows(Path(args.real_cache), Path(args.blockout_cache), args.n)

    observed = {
        f"{fraction:.2f}": {"epsilon": [0.0] * args.orderings,
                            "surface": [0.0] * args.orderings}
        for fraction in args.t_frac
    }
    per_building = {}
    t0 = time.time()
    with h5py.File(args.real_cache, "r") as real, h5py.File(args.blockout_cache, "r") as blockout:
        for row, real_index, blockout_index in rows:
            z = torch.from_numpy(np.asarray(real["latent"][real_index], np.float32))[None].to(device)
            zb = torch.from_numpy(np.asarray(blockout["latent"][blockout_index], np.float32))[None].to(device)
            z = (z - mu) / sd
            zb = (zb - mu) / sd
            footprint = torch.from_numpy(
                np.asarray(real["footprint"][real_index], np.float32)
            )[None, None].to(device)
            height = torch.tensor([float(real["height_m"][real_index])], device=device)
            region = torch.tensor([int(real["region"][real_index])], device=device)
            row_result = {}

            for fraction in args.t_frac:
                key = f"{fraction:.2f}"
                t, alpha, noise, points = _pair_step_draw(
                    row, fraction, checkpoint["args"]["timesteps"], alphas, args.seed,
                    args.surface_points, zb.shape, device,
                )
                with torch.no_grad():
                    target_field = codec.query(z * sd + mu, points)

                losses = {"epsilon": [], "surface": []}
                for ordering in range(args.orderings):
                    permutation = (np.arange(zb.shape[1]) if ordering == 0 else
                                   np.random.default_rng(args.seed + row * 389 + ordering)
                                   .permutation(zb.shape[1]))
                    permutation = torch.from_numpy(permutation).to(device)
                    source = zb[:, permutation]
                    ordered_noise = noise[:, permutation]
                    zt = alpha.sqrt() * source + (1 - alpha).sqrt() * ordered_noise
                    epsilon_target = (zt - alpha.sqrt() * z) / (1 - alpha).sqrt()
                    with torch.no_grad():
                        predicted = model(x=zt, t=t, footprint=footprint, height=height,
                                          region=region, drop_cond=False)
                        epsilon_loss = torch.nn.functional.mse_loss(predicted, epsilon_target)
                        x0 = (zt - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
                        got_field = codec.query(x0 * sd + mu, points)
                        surface_loss, _ = surface_term(got_field, target_field, alpha)
                    losses["epsilon"].append(float(epsilon_loss))
                    losses["surface"].append(float(surface_loss))
                    observed[key]["epsilon"][ordering] += float(epsilon_loss) / len(rows)
                    observed[key]["surface"][ordering] += float(surface_loss) / len(rows)
                row_result[key] = losses
            per_building[str(row)] = row_result
            print(f"  row {row} ({len(per_building)}/{len(rows)})", flush=True)

    report = build_report(observed)
    report.update({
        "meta": {
            "checkpoint": str(args.a2),
            "step": int(checkpoint["step"]),
            "real_cache": str(args.real_cache),
            "blockout_cache": str(args.blockout_cache),
            "n": len(rows),
            "rows": [row for row, _, _ in rows],
            "orderings": args.orderings,
            "surface_points": args.surface_points,
            "seed": args.seed,
            "device": device,
            "seconds": time.time() - t0,
            "intervention": "permute blockout tokens and their noise; keep real target order fixed",
        },
        "per_building": per_building,
        "published_spread_pct": PUBLISHED_SPREAD_PCT,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements",
                        help="existing JSON containing an 'observed' mapping; skips model execution")
    parser.add_argument("--real-cache", default=str(DEFAULT_REAL))
    parser.add_argument("--blockout-cache", default=str(DEFAULT_BLOCKOUT))
    parser.add_argument("--a2", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--orderings", type=int, default=5)
    parser.add_argument("--t-frac", type=float, nargs="+", default=[0.40, 0.55, 0.70])
    parser.add_argument("--surface-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON report path")
    args = parser.parse_args()

    if args.measurements:
        source = json.loads(Path(args.measurements).read_text())
        report = build_report(source["observed"])
    else:
        report = measure(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
