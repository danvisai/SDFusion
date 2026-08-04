"""#76 -- does the diffusion need a decoded-surface loss, and what would the gradient path cost?

Two measurements, one per half of the decision.

**A. Does latent distance rank candidates the way surface quality does?** Our diffusion's entire
supervision is `mse_loss(pred, noise)` on latent tokens (`train_vecset.py:166`). A loss is a *ranking
device*: it must say which of two candidate outputs is better. So the question is not whether latent
distance correlates loosely with quality, it is whether it **orders candidates correctly**. Measured as
Spearman rank correlation between latent distance and decoded 3D IoU, over a pool of candidates built
from two different error families:

  * **on-manifold** -- the same mesh re-encoded with a fresh point sample. Far away in latent space
    (FPS reorders the tokens), identical in geometry.
  * **off-manifold** -- isotropic noise at controlled magnitudes.

Reported three ways: pooled over both families, and within each family alone. That split is the whole
point -- a loss that ranks correctly *within* one error family can still be worthless, because during
training the candidates it must order are not drawn from one family.

**B. What does a differentiable query path cost?** `DoraCodec.query` and `.encode` both run under
`torch.no_grad()` (`models/shape_codec.py:152,160`), so today there is **no gradient path** from a
decoded surface back to the denoiser -- the fix is structurally blocked, not merely unimplemented. This
measures what unblocking costs: peak memory and step time with gradients flowing through
`model.decode` + `model.query`, swept over query-point count, and a check that the decoder's own
weights stay frozen while gradients merely pass through to the latent.

Run:
    probe_surface_loss.py --n 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.eval_massing_arms import H5, LATENTS, RES, pick_ids, volume_split  # noqa: E402
from scripts.foundations.diagnose_decoder_tolerance import perturb  # noqa: E402

BASELINE = REPO / "execution/artifacts/massing_arms_eval_baseline.json"
COSINES = [0.999, 0.995, 0.980, 0.935, 0.900]
QUERY_COUNTS = [1024, 4096, 8192, 16384, 32768]


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def phase_a(codec, lf, gt, ids, lat_of, mu, sd, surf, seed, dev):
    """Latent distance vs decoded quality, over two error families."""
    import torch
    from models.shape_codec import Building
    from scene.surface_sampling import to_array_frame

    def decode(zn):
        with torch.no_grad():
            return codec.decode_grid(
                torch.from_numpy(zn * sd + mu)[None].to(dev), RES).cpu().numpy()[0, 0]

    rows = []
    for k, bid in enumerate(ids):
        gocc = np.asarray(gt["sdf"][bid], np.float32) <= 0
        zn = (np.asarray(lf["latent"][lat_of[bid]], np.float32) - mu) / sd
        pocc = decode(zn) <= 0

        cands = []
        if bid in surf:
            v, fc, _ = surf[bid]
            av, af = to_array_frame(v, fc)
            for r in range(2):
                codec.rng = np.random.default_rng(seed * 1000003 + bid + r)
                z2 = (codec.encode(Building(verts=av, faces=af)).float().cpu().numpy()[0] - mu) / sd
                cands.append(("on_manifold", z2))
        for c in COSINES:
            for r in range(2):
                zp, _ = perturb(zn, c, seed=seed * 1000003 + bid * 9973 + int(c * 1000) * 13 + r)
                cands.append(("off_manifold", zp))

        for family, z in cands:
            fld = decode(z)
            rows.append(dict(
                row=int(bid), family=family,
                l2=float(np.linalg.norm(z - zn)),
                l1=float(np.abs(z - zn).mean()),
                cos=float(np.dot(zn.ravel(), z.ravel())
                          / (np.linalg.norm(zn) * np.linalg.norm(z))),
                iou_gt=volume_split(fld <= 0, gocc)["vol_iou"],
                iou_perfect=volume_split(fld <= 0, pocc)["vol_iou"],
            ))
        print(f"  [A {k+1}/{len(ids)}] row {bid}", flush=True)
    return rows


def phase_b(codec, lf, lat_of, ids, mu, sd, dev):
    """Cost of letting gradients flow through decode + query, in both freezing configurations.

    The decoder ships with `requires_grad=True` on every parameter, so simply removing `no_grad` would
    build a graph for all 191.6M of them and accumulate gradients we never use. `requires_grad_(False)`
    is the configuration we would actually train in, and it is measured separately -- the difference is
    the price of forgetting to do it.

    The frozen-ness check is read **immediately after backward**, before any `zero_grad`: checking at
    the end of a loop that zeroes gradients proves nothing at all.
    """
    import torch

    model = codec.model
    z0 = torch.from_numpy(
        ((np.asarray(lf["latent"][lat_of[ids[0]]], np.float32) - mu) / sd) * sd + mu)[None].to(dev)

    def run(nq: int, freeze: bool) -> dict:
        for p in model.parameters():
            p.requires_grad_(not freeze)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        z = z0.clone().requires_grad_(True)
        pts = (torch.rand(1, nq, 3, device=dev) * 2 - 1)
        torch.cuda.synchronize(); t0 = time.time()
        # deliberately NOT through DoraCodec.query -- that wrapper's no_grad is the thing under test
        sdf = model.query(pts.float(), model.decode(z)).float()
        (sdf ** 2).mean().backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_grad = sum(1 for p in model.parameters() if p.grad is not None)   # read BEFORE zero_grad
        rec = dict(
            n_query=nq, freeze_decoder=freeze, sec_per_step=round(dt, 4),
            peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
            latent_grad_ok=bool(z.grad is not None and torch.isfinite(z.grad).all()),
            grad_norm=float(z.grad.norm()) if z.grad is not None else float("nan"),
            decoder_params_with_grad=n_grad,
        )
        model.zero_grad(set_to_none=True)
        return rec

    run(1024, True)                      # warm-up: first CUDA call carries kernel-load cost
    out = [run(nq, True) for nq in QUERY_COUNTS]
    for r in out:
        print(f"  [B frozen]   {r['n_query']:>6} pts  {r['sec_per_step']:.3f}s  "
              f"{r['peak_mem_gb']:>6.2f} GB  latent_grad={r['latent_grad_ok']}  "
              f"decoder_params_with_grad={r['decoder_params_with_grad']}", flush=True)
    unfrozen = run(8192, False)
    print(f"  [B unfrozen] {unfrozen['n_query']:>6} pts  {unfrozen['sec_per_step']:.3f}s  "
          f"{unfrozen['peak_mem_gb']:>6.2f} GB  "
          f"decoder_params_with_grad={unfrozen['decoder_params_with_grad']}", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return dict(sweep_frozen=out, unfrozen_8192=unfrozen,
                n_decoder_params=sum(p.numel() for p in model.parameters()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--latents", default=str(LATENTS))
    ap.add_argument("--ids_from", default=str(BASELINE) if BASELINE.exists() else None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import h5py
    import torch
    from models.shape_codec import DoraCodec
    from scripts.foundations.dora_frozen_gate import load_surfaces
    from scripts.foundations.dora_roundtrip_probe import load_dora

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cand, lat_of = pick_ids(Path(args.latents), args.ids_from)
    ids = cand[:args.n]
    with h5py.File(args.latents, "r") as f:
        probe = np.asarray(f["latent"][:600], np.float32)
        mu, sd = float(probe.mean()), float(probe.std())
    codec = DoraCodec(load_dora(dev))
    surf = load_surfaces()

    print(f"\n--- A: does latent distance rank like surface quality? (n={len(ids)}) ---", flush=True)
    with h5py.File(args.latents, "r") as lf, h5py.File(H5, "r") as gt:
        rows = phase_a(codec, lf, gt, ids, lat_of, mu, sd, surf, args.seed, dev)

    corr = {}
    for label, sub in [("pooled", rows),
                       ("on_manifold_only", [r for r in rows if r["family"] == "on_manifold"]),
                       ("off_manifold_only", [r for r in rows if r["family"] == "off_manifold"])]:
        if not sub:
            continue
        corr[label] = {m: dict(
            vs_iou_gt=spearman([r[m] for r in sub], [r["iou_gt"] for r in sub]),
            vs_iou_perfect=spearman([r[m] for r in sub], [r["iou_perfect"] for r in sub]),
            n=len(sub)) for m in ("l2", "l1", "cos")}

    print(f"\n=== A. Spearman rank correlation, latent distance vs decoded 3D IoU ===")
    print(f"{'candidate pool':20s} {'n':>5} {'L2':>8} {'L1':>8} {'cosine':>8}")
    for label, d in corr.items():
        print(f"{label:20s} {d['l2']['n']:>5} {d['l2']['vs_iou_perfect']:>8.3f} "
              f"{d['l1']['vs_iou_perfect']:>8.3f} {d['cos']['vs_iou_perfect']:>8.3f}")
    print("  (L2/L1 are distances, so a working loss wants NEGATIVE; cosine is a similarity, "
          "so it wants POSITIVE)")

    print(f"\n--- B: cost of a differentiable query path ---", flush=True)
    with h5py.File(args.latents, "r") as lf:
        cost = phase_b(codec, lf, lat_of, ids, mu, sd, dev)
    print(f"\ndecoder is {cost['n_decoder_params']/1e6:.1f}M params; with requires_grad left ON, "
          f"{cost['unfrozen_8192']['decoder_params_with_grad']} of its tensors take gradient "
          f"({cost['unfrozen_8192']['peak_mem_gb']:.2f} GB vs "
          f"{cost['sweep_frozen'][2]['peak_mem_gb']:.2f} GB frozen at the same query count)")

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    suffix = f"_{args.tag}" if args.tag else ""
    art = REPO / f"execution/artifacts/surface_loss_probe{suffix}.json"
    art.write_text(json.dumps(dict(
        meta=dict(git_rev=rev, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  n=len(ids), seed=args.seed, mu=mu, sd=sd, cosines=COSINES),
        correlation=corr, gradient_path=cost, per_candidate=rows), indent=2))
    print(f"\nartifact: {art}", flush=True)


if __name__ == "__main__":
    main()
