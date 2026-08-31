"""#95: does an arbitrary token order at inference break a model trained on aligned pairs?

The answer that decided #91/#92: **order at inference is noise, not bias.** This probe measures the
three things that answer is built on.

**A. The symmetry, stated correctly.** Permuting the blockout *and* its noise permutes the output --
the operator is equivariant. Permuting the blockout *alone* is a **different sample**, not a broken
symmetry, because the noise is independent of token order. ⚠️ #95's own premise said decoded fields
"must be identical" under a token permutation and would have triggered "the whole plan needs
rethinking"; running it literally is what showed the premise was wrong.

**B. What that costs in quality.** The same envelope generated under several orderings, scored per
building. The spread is real but small in aggregate, and **in #92 both arms see the same envelope in
the same order, so it cancels in a paired difference** -- which is a requirement on #92, not a hope.

**C. A canonical order at inference buys nothing.** Morton scored -0.0150 against as-encoded on the
lost run: slightly worse, inside the noise. So #90 chooses on alignment quality and cost alone.

⚠️ The lost run reintroduced the Dutch-only trap here by re-deriving its own sample. This one uses
`pick_ids`, the stratified picker, and prints the region mix so a single-region sample is visible.

    probe_token_order.py --a2 weights/massing-vecset/vecset_v4_surf.pth --n 8 --orderings 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.shape_codec import Building, DoraCodec                              # noqa: E402
from scripts.foundations.baseline_gate_eval import mesh_sdf_surface             # noqa: E402
from scripts.foundations.dora_roundtrip_probe import load_dora, H5              # noqa: E402
from scripts.foundations.eval_massing_arms import (                             # noqa: E402
    LATENTS, _vertical_extent, blockout_sdf, pick_ids, score_arm,
)
from scripts.foundations.vecset_ceiling_probe import RES, TRUNC, verts_to_world  # noqa: E402

OUT = REPO / "execution/artifacts/token_order_inference.json"

#: the lost run's numbers, kept beside the new ones
PUBLISHED = {"equivariant_max_field_diff": 7.36e-04, "tokens_only_max_field_diff": 2.03,
             "range_median": 0.0208, "range_mean": 0.0851, "range_max": 0.5386,
             "morton_delta": -0.0150, "se_median_714": 0.00132}


def _load_a2(path: str, dev: str):
    import torch
    from models.networks.vecset_denoiser import VecsetDenoiser
    from models.networks.vecset_projection import SetSDEdit

    ck = torch.load(path, map_location="cpu", weights_only=False)
    ca = ck["args"]
    net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=ca["width"], depth=ca["depth"],
                         heads=ca["heads"], footprint_res=ck["footprint_res"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return dict(op=SetSDEdit(net, timesteps=ca["timesteps"]), mu=ck["latent_mu"], sd=ck["latent_sd"],
                step=int(ck["step"]))


def _envelope(codec, a2, fp, gt_occ, dev):
    """The blockout latent the generator is handed at inference, plus its token query positions.

    Positions come from #88's capture path, so the Morton arm sorts on the encoder's real query
    points rather than on a stand-in derived from the latent itself.
    """
    bo = blockout_sdf(fp, *_vertical_extent(gt_occ))
    bv, bf = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
    z, pos = codec.encode_with_positions(Building(verts=verts_to_world(bv), faces=bf))
    return (z.float() - a2["mu"]) / a2["sd"], pos


def part_a(codec, a2, fp, gt_occ, ht, rg, dev, strength, steps, guidance, seed) -> dict:
    """The symmetry: permute tokens WITH the noise, then permute tokens ALONE."""
    import torch

    z0, _ = _envelope(codec, a2, fp, gt_occ, dev)
    fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
    h = torch.tensor([ht], device=dev)
    r = torch.tensor([rg], device=dev)
    eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(seed)).to(dev)
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(z0.shape[1])).to(dev)

    def go(z, noise):
        y = a2["op"].project(blockout=z, footprint=fpt, height=h, region=r, strength=strength,
                             steps=steps, guidance=guidance, noise=noise)
        with torch.no_grad():
            return codec.decode_grid(y * a2["sd"] + a2["mu"], RES).cpu().numpy()[0, 0]

    base = go(z0, eps)
    both = go(z0[:, perm], eps[:, perm])
    tokens_only = go(z0[:, perm], eps)
    return {
        "equivariant_max_field_diff": float(np.abs(base - both).max()),
        "tokens_only_max_field_diff": float(np.abs(base - tokens_only).max()),
        "published": {k: PUBLISHED[k] for k in ("equivariant_max_field_diff",
                                                "tokens_only_max_field_diff")},
    }


def part_bc(codec, a2, ids, fp_of, gt_occ, ht_of, rg_of, dev, args) -> dict:
    """Quality under several orderings (B), and a canonical Morton order at inference (C)."""
    import torch

    per, morton_delta = {}, []
    for bid in ids:
        fp = fp_of[bid]
        z0, pos = _envelope(codec, a2, fp, gt_occ[bid], dev)
        fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
        h = torch.tensor([ht_of[bid]], device=dev)
        r = torch.tensor([rg_of[bid]], device=dev)

        def score(z):
            y = a2["op"].project(blockout=z, footprint=fpt, height=h, region=r,
                                 strength=args.strength, steps=args.steps, guidance=args.guidance,
                                 seed=args.seed * 1000003 + bid)
            with torch.no_grad():
                fld = codec.decode_grid(y * a2["sd"] + a2["mu"], RES).cpu().numpy()[0, 0]
            return score_arm(fld, gt_occ[bid], fp)["vol_iou"]

        vals = [score(z0)]
        for o in range(1, args.orderings):
            p = torch.from_numpy(np.random.default_rng(bid * 97 + o).permutation(z0.shape[1])).to(dev)
            vals.append(score(z0[:, p]))
        per[bid] = vals

        # C: a canonical order, the one thing a sort could offer at inference -- sorted on the
        # encoder's own query positions, so this arm is a real Morton order and not a stand-in.
        if args.morton:
            mp = np.argsort(_morton_key(pos))
            morton_delta.append(score(z0[:, torch.from_numpy(mp).to(dev)]) - vals[0])
        print(f"  id {bid}: " + " ".join(f"{v:.3f}" for v in vals) +
              f"   range {max(vals)-min(vals):.4f}", flush=True)

    ranges = [max(v) - min(v) for v in per.values()]
    firsts = [v[0] for v in per.values()]
    out = {
        "n": len(per), "orderings": args.orderings,
        "range_median": float(np.median(ranges)), "range_mean": float(np.mean(ranges)),
        "range_max": float(np.max(ranges)),
        "moved_under_0.05": int(sum(r < 0.05 for r in ranges)),
        "moved_over_0.20": int(sum(r > 0.20 for r in ranges)),
        "se_median_scaled_714": float(np.std(firsts, ddof=1) / np.sqrt(714)) if len(firsts) > 1
        else None,
        "per_building": {str(k): v for k, v in per.items()},
        "published": {k: PUBLISHED[k] for k in ("range_median", "range_mean", "range_max",
                                                "se_median_714")},
    }
    if morton_delta:
        out["morton_delta"] = float(np.mean(morton_delta))
        out["published"]["morton_delta"] = PUBLISHED["morton_delta"]
    return out


def _morton_key(pos: np.ndarray) -> np.ndarray:
    """The canonical space-filling rank of each token's query position."""
    from models.token_alignment import _morton_code
    return _morton_code(np.asarray(pos, np.float64), 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", default=str(REPO / "weights/massing-vecset/vecset_v4_surf.pth"))
    ap.add_argument("--latents", default=str(LATENTS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--orderings", type=int, default=5)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--morton", action="store_true", help="also score a canonical order (indicative)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import h5py
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = DoraCodec(load_dora(dev)).freeze()
    a2 = _load_a2(args.a2, dev)
    print(f"[a2] {args.a2}  step {a2['step']}", flush=True)

    cand, lat_of = pick_ids(Path(args.latents), None)
    ids = cand[:args.n]
    fp_of, gt_occ, ht_of, rg_of = {}, {}, {}, {}
    with h5py.File(H5, "r") as f, h5py.File(args.latents, "r") as g:
        reg = np.asarray(g["region"]) if "region" in g else None
        row = np.asarray(g["row"])
        rg_by_row = {int(r): int(x) for r, x in zip(row, reg)} if reg is not None else {}
        for bid in ids:
            fp_of[bid] = np.asarray(f["footprint"][bid], np.uint8)
            gt_occ[bid] = np.asarray(f["sdf"][bid], np.float32) <= 0
            ht_of[bid] = float(f["height_m"][bid])
            rg_of[bid] = rg_by_row.get(int(bid), 0)
    mix = {}
    for bid in ids:
        mix[rg_of[bid]] = mix.get(rg_of[bid], 0) + 1
    print(f"[order] {len(ids)} buildings, region mix {dict(sorted(mix.items()))}", flush=True)

    t0 = time.time()
    first = ids[0]
    print("[A] token-set symmetry on one building", flush=True)
    a = part_a(codec, a2, fp_of[first], gt_occ[first], ht_of[first], rg_of[first], dev,
               args.strength, args.steps, args.guidance, args.seed)
    print(f"  permute tokens AND noise : max field diff {a['equivariant_max_field_diff']:.2e}   "
          f"(lost run {PUBLISHED['equivariant_max_field_diff']:.2e})")
    print(f"  permute tokens ONLY      : max field diff {a['tokens_only_max_field_diff']:.2f}   "
          f"(lost run {PUBLISHED['tokens_only_max_field_diff']:.2f})")

    print(f"[B] quality under {args.orderings} orderings", flush=True)
    bc = part_bc(codec, a2, ids, fp_of, gt_occ, ht_of, rg_of, dev, args)
    print(f"\n  range: median {bc['range_median']:.4f} · mean {bc['range_mean']:.4f} · "
          f"max {bc['range_max']:.4f}   (lost run {PUBLISHED['range_median']} · "
          f"{PUBLISHED['range_mean']} · {PUBLISHED['range_max']})")
    print(f"  {bc['moved_under_0.05']}/{bc['n']} buildings move < 0.05, "
          f"{bc['moved_over_0.20']} move > 0.20")

    art = {"checkpoint": args.a2, "step": a2["step"], "strength": args.strength,
           "region_mix": {str(k): v for k, v in sorted(mix.items())}, "A": a, "B": bc,
           "seconds": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(art, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
