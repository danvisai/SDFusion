"""Encode the recovered LoD2 surfaces into vecset latents, once, for diffusion training.

Encoding is the expensive part and the corpus is fixed, so it is done ahead of time rather than in the
data loader. Everything goes through the `ShapeCodec` contract (spec #68) rather than reaching into the
autoencoder directly -- the same calls the diffusion will use, so a codec swap changes one flag here and
nothing downstream.

Also caches the conditioning the generator needs beside each latent -- footprint, height, region -- so
training reads one file and never touches the source corpus.

Usage:
    precompute_vecset_latents.py --limit 256          # smoke
    precompute_vecset_latents.py                       # the whole corpus
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.shape_codec import Building, DoraCodec                       # noqa: E402
from scripts.foundations.baseline_gate_eval import mesh_sdf_surface       # noqa: E402
from scripts.foundations.vecset_ceiling_probe import TRUNC, verts_to_world  # noqa: E402
from scripts.foundations.dora_roundtrip_probe import load_dora, H5       # noqa: E402
from scripts.foundations.dora_frozen_gate import load_surfaces           # noqa: E402
from scripts.foundations.vecset_ceiling_probe import test_indices        # noqa: E402

OUT = REPO / "data/real_massing_v1/vecset_latents.h5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--n_coarse", type=int, default=8192)
    ap.add_argument("--n_sharp", type=int, default=8192)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--blockout", action="store_true",
                    help="encode the footprint EXTRUSION instead of the real surface, giving the "
                         "aligned-pair partner: what the generator is handed at inference")
    args = ap.parse_args()

    import h5py
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = DoraCodec(load_dora(dev), n_coarse=args.n_coarse, n_sharp=args.n_sharp)

    surf = load_surfaces()
    rows = sorted(surf)
    if args.limit:
        rows = rows[:args.limit]
    held = set(int(i) for i in test_indices(35776))
    print(f"[precompute] {len(rows)} buildings -> {args.out}")

    lat, fps, hts, regs, keep, split = [], [], [], [], [], []
    src_id = {"bag3d": 0, "nrw": 1, "plateau": 2}
    t0 = time.time()
    with h5py.File(H5, "r") as f:
        for n, r in enumerate(rows):
            v, fc, src = surf[r]
            try:
                if args.blockout:
                    # the same extrusion the generator is handed at inference -- encoding it here is
                    # what removes the train/inference distribution gap the diagnostic identified
                    from scripts.foundations.eval_vecset_projection import blockout_sdf
                    g = np.asarray(f["sdf"][r], np.float32)
                    ys = np.nonzero((g <= 0).any(axis=(0, 2)))[0]
                    if len(ys) == 0:
                        continue
                    bo = blockout_sdf(np.asarray(f["footprint"][r], np.uint8),
                                      int(ys.min()), int(ys.max()))
                    if bo is None:
                        continue
                    bv, bf = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
                    if bv is None:
                        continue
                    z = codec.encode(Building(verts=verts_to_world(bv), faces=bf))
                else:
                    z = codec.encode(Building(verts=v, faces=fc))
            except Exception as e:
                print(f"  [skip] row {r}: {type(e).__name__}"); continue
            lat.append(z[0].cpu().numpy().astype(np.float16))   # fp16: 2048x64 per building
            fps.append(np.asarray(f["footprint"][r], np.uint8))
            hts.append(float(f["height_m"][r]))
            regs.append(src_id[src])
            keep.append(r)
            split.append(1 if r in held else 0)                 # 1 = held out, never trained on
            if (n + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {n+1}/{len(rows)}  {el:.0f}s  eta {el/(n+1)*(len(rows)-n-1):.0f}s", flush=True)

    if not lat:
        raise SystemExit("nothing encoded")
    L = np.stack(lat)
    with h5py.File(args.out, "w") as o:
        o.create_dataset("latent", data=L, compression="lzf")
        o.create_dataset("footprint", data=np.stack(fps), compression="lzf")
        o.create_dataset("height_m", data=np.asarray(hts, np.float32))
        o.create_dataset("region", data=np.asarray(regs, np.int32))
        o.create_dataset("row", data=np.asarray(keep, np.int32))
        o.create_dataset("held_out", data=np.asarray(split, np.uint8))
        o.attrs["codec"] = codec.name
        o.attrs["n_coarse"], o.attrs["n_sharp"] = args.n_coarse, args.n_sharp
    print(f"[precompute] {len(L)} latents {L.shape} ({L.nbytes/1e6:.0f} MB fp16), "
          f"{int(np.sum(split))} held out -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
