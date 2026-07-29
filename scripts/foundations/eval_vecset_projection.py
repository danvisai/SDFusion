"""Measure what the A2 generator retains of its codec's ceiling (spec #67).

This is the number the whole effort turns on. Both stacks have a codec ceiling and a generator that
falls short of it; the bet behind A2 is that a token-set generator loses LESS of its ceiling than the
dense-grid one loses of its own:

    dense grid (today)   ceiling 3D IoU 0.995  ->  generator delivers 0.601
    vecset (A2)          ceiling 3D IoU 0.999  ->  measured here

Scored on the criteria that matter -- **footprint-IoU and 3D IoU**, not surface roughness. Roughness was
shown to be anti-correlated with the goal on exactly this comparison: it ranks a melted blob above a
crisp ribbed box (`docs/wayfinding/crisp-massing-vecset/deployed-vs-dora.md`).

The generator is a PROJECTION per ADR 0003: the footprint is extruded into a blockout, encoded, partly
noised, and denoised back. It is never sampled from noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks.vecset_denoiser import VecsetDenoiser                  # noqa: E402
from models.networks.vecset_projection import SetSDEdit                     # noqa: E402
from models.shape_codec import Building, DoraCodec                          # noqa: E402
from scripts.foundations.baseline_gate_eval import fp_iou, mesh_sdf_surface  # noqa: E402
from scripts.foundations.dora_roundtrip_probe import load_dora, H5          # noqa: E402
from scripts.foundations.vecset_ceiling_probe import RES, TRUNC, verts_to_world  # noqa: E402

DEPLOYED = {"fp_iou": 0.863, "vol_iou": 0.601, "codec_vol_iou": 0.995}   # map-#24, measured n=15


def blockout_sdf(fp: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Footprint mask + vertical extent -> a crisp extruded-prism SDF on the corpus grid.

    This is the analytic extrusion prior #53 measured as crisp (0.0035 roughness) and footprint-exact
    (IoU 0.96) -- exactly the "crude blockout" ADR 0003 says generation should project FROM. Signed
    EDT rather than marching a binary mask, so the surface is not pre-staircased.
    """
    from scipy import ndimage
    occ = np.zeros((RES, RES, RES), bool)                 # array axes are [z, y, x]
    occ[:, y0:y1 + 1, :] = fp.astype(bool)[:, None, :]
    if not occ.any():
        return None
    inside = ndimage.distance_transform_edt(occ)
    outside = ndimage.distance_transform_edt(~occ)
    return ((outside - inside) * (2.0 / (RES - 1))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs_building/vecset_v1/vecset_denoiser.pth")
    ap.add_argument("--latents", default="data/real_massing_v1/vecset_latents.h5")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--strength", type=float, nargs="*", default=[0.2, 0.4, 0.6])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/vecset_eval")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import h5py

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=a["width"], depth=a["depth"],
                         heads=a["heads"], footprint_res=ck["footprint_res"]).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    print(f"[model] step {ck['step']}  {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    mu, sd = ck["latent_mu"], ck["latent_sd"]
    op = SetSDEdit(net, timesteps=a["timesteps"])
    codec = DoraCodec(load_dora(dev))

    with h5py.File(args.latents, "r") as f:
        held = np.nonzero(f["held_out"][:] == 1)[0][:args.n]
        rows = f["row"][:][held]
        fps = f["footprint"][:][held]
        regs = f["region"][:][held]
        hts = f["height_m"][:][held]

    res = {s: {"fp": [], "vol": []} for s in args.strength}
    res["blockout"] = {"fp": [], "vol": []}
    with h5py.File(H5, "r") as gt:
        for k, (r, fp, reg, hm) in enumerate(zip(rows, fps, regs, hts)):
            g = np.asarray(gt["sdf"][int(r)], np.float32)
            gocc = g <= 0
            ys = np.nonzero(gocc.any(axis=(0, 2)))[0]
            if len(ys) == 0:
                continue
            bo = blockout_sdf(fp, int(ys.min()), int(ys.max()))
            if bo is None:
                continue

            # the blockout itself, as the honest baseline the projection must improve on
            bocc = bo <= 0
            res["blockout"]["fp"].append(fp_iou(bocc, fp))
            res["blockout"]["vol"].append((bocc & gocc).sum() / max((bocc | gocc).sum(), 1))

            v, fc = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
            if v is None:
                continue
            z0 = codec.encode(Building(verts=verts_to_world(v), faces=fc))
            z0 = ((z0.float() - mu) / sd)
            fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
            ht = torch.tensor([float(hm)], device=dev)
            rg = torch.tensor([int(reg)], device=dev)

            for s in args.strength:
                zp = op.project(blockout=z0, footprint=fpt, height=ht, region=rg,
                                strength=s, steps=args.steps, guidance=args.guidance, seed=0)
                fld = codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]
                occ = fld <= 0
                res[s]["fp"].append(fp_iou(occ, fp))
                res[s]["vol"].append((occ & gocc).sum() / max((occ | gocc).sum(), 1))
            if (k + 1) % 5 == 0:
                print(f"  {k+1}/{len(rows)}", flush=True)

    print(f"\n=== A2 PROJECTION (n={len(res['blockout']['fp'])} held-out) ===")
    print(f"{'arm':26s} {'fp-IoU':>8} {'3D IoU':>8}")
    print(f"{'blockout (input)':26s} {np.median(res['blockout']['fp']):>8.3f} "
          f"{np.median(res['blockout']['vol']):>8.3f}")
    best = None
    for s in args.strength:
        f_, v_ = np.median(res[s]["fp"]), np.median(res[s]["vol"])
        print(f"{'projected s=' + str(s):26s} {f_:>8.3f} {v_:>8.3f}")
        if best is None or v_ > best[1]:
            best = (s, v_, f_)
    print(f"\n--- reference: map-#24 deployed ---")
    print(f"{'deployed [GENERATED]':26s} {DEPLOYED['fp_iou']:>8.3f} {DEPLOYED['vol_iou']:>8.3f}")
    print(f"\nretention of codec ceiling:")
    print(f"  dense grid : 0.601 / 0.995 = {0.601/0.995:.3f}")
    print(f"  vecset     : {best[1]:.3f} / 0.999 = {best[1]/0.999:.3f}   (best strength {best[0]})")
    json.dump({k if isinstance(k, str) else f"s{k}":
               {"fp_iou": float(np.median(v["fp"])), "vol_iou": float(np.median(v["vol"]))}
               for k, v in res.items()}, open(out / "eval.json", "w"), indent=2)
    print(f"-> {out/'eval.json'}")


if __name__ == "__main__":
    main()
