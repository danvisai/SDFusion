"""The comparison that decides whether the vecset path is an upgrade: what we SHIP vs what Dora gives.

Four arms on the same held-out buildings, all meshed at level 0.0 and shaded identically:

  1. **GT**                  the real LoD2 surface
  2. **map-#24 deployed**    what we ship -- GENERATED from the footprint alone
  3. **VQVAE round-trip**    the current stack's codec ceiling (#56 measured 0.0044)
  4. **Dora round-trip**     the vecset stack's codec ceiling

⚠️ **The tasks are not symmetric, and the asymmetry favours arms 3 and 4.** map-#24 *generates* from a
footprint with no sight of the answer; both round-trip arms are handed the ground-truth surface and only
have to reproduce it. So this is not a like-for-like generative comparison and must not be read as one.

What it *does* answer, which is the live question: for each representation, how good is its decoder's
surface, and how far does its generator fall short of it? #56's finding was that our dense-grid codec is
near-perfect (0.0044 vs GT 0.0041) while our diffusion sits at 0.00552 -- the gap is the diffusion, not
the codec. The bet behind A2 is that a token-set diffusion loses *less* of its codec's ceiling than a
dense-grid diffusion loses of its own.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.baseline_gate_eval import build_opt, mesh_sdf_surface   # noqa: E402
from scripts.foundations.vecset_ceiling_probe import (                           # noqa: E402
    RES, TRUNC, grid_points, verts_to_world, test_indices,
)
from scripts.foundations.dora_roundtrip_probe import (                           # noqa: E402
    load_dora, sample_surface, sample_sharp_edges, H5,
)
from scripts.foundations.dora_frozen_gate import load_surfaces, _revoxel, _rough  # noqa: E402

MAP24 = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ckpt", default=MAP24)
    ap.add_argument("--out", default="outputs/deployed_vs_dora")
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    import h5py, trimesh
    from PIL import Image, ImageDraw
    from datasets.bag3d_dataset import Bag3dDataset
    from models.stage3a_model import Stage3aModel
    from scripts.hunyuan_building_mesh_smoke import render_mesh_png

    opt = build_opt(dev, ckpt=args.ckpt, use_region=True, use_extra_cond=False, use_ema=True)
    opt.bag3d_h5 = str(H5)
    print(f"[load] Stage3a from {args.ckpt}", flush=True)
    stage3a = Stage3aModel(); stage3a.initialize(opt); stage3a.switch_eval()

    ds = Bag3dDataset(); ds.initialize(opt, phase="test")
    glob_of = {int(g): i for i, g in enumerate(ds.idxs)}      # global row -> dataset index

    surf = load_surfaces()
    held = [int(i) for i in test_indices(35776)]
    by_src, picks, i = {}, [], 0
    for s in ("bag3d", "nrw", "plateau"):
        by_src[s] = [r for r in held if r in surf and surf[r][2] == s and r in glob_of]
    while len(picks) < args.n and any(len(v) > i for v in by_src.values()):
        for s in by_src:
            if len(by_src[s]) > i and len(picks) < args.n:
                picks.append(by_src[s][i])
        i += 1

    dora = load_dora(dev)
    pts = grid_points()
    q = torch.from_numpy(pts.astype(np.float32))[None].to(dev)
    rows = []

    with h5py.File(H5, "r") as f:
        for r in picks:
            v, fc, src = surf[r]
            gt_field = _revoxel(v, fc, pts)
            item = ds[glob_of[r]]
            data = {k: (val.unsqueeze(0).to(dev) if torch.is_tensor(val) else val)
                    for k, val in item.items() if torch.is_tensor(val)}
            with torch.no_grad():
                gen = stage3a.inference(data, ddim_steps=opt.ddim_steps,
                                        uc_scale=1.0).cpu().numpy()[0, 0]
                x = item["sdf"].unsqueeze(0).to(dev)
                z = stage3a.vqvae(x, forward_no_quant=True, encode_only=True)
                rt = stage3a.vqvae.decode_no_quant(z).cpu().numpy()[0, 0]

                mesh = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(fc), process=False)
                co = torch.from_numpy(sample_surface(mesh, 8192, rng))[None].to(dev)
                sh = torch.from_numpy(sample_sharp_edges(mesh, 8192, rng))[None].to(dev)
                _, kl, _ = dora.encode(co, sh, sample_posterior=False)
                lat = dora.decode(kl)
                dfield = -torch.cat([dora.query(q[:, j:j + 32768], lat).float()
                                     for j in range(0, q.shape[1], 32768)],
                                    1).view(RES, RES, RES).cpu().numpy()

            arms = [("GT (real LoD2)", gt_field), ("map-#24 deployed  [GENERATED]", gen),
                    ("VQVAE round-trip", rt), ("Dora round-trip", dfield)]
            panels = []
            for label, fld in arms:
                mv, mf = mesh_sdf_surface(np.clip(fld, -TRUNC, TRUNC))
                if mv is None:
                    panels.append((label, None, float("nan"))); continue
                m = trimesh.Trimesh(verts_to_world(mv), mf, process=False)
                g = _rough(_revoxel(m.vertices, m.faces, pts))
                panels.append((label, render_mesh_png(m, image_size=args.size), g))
            rows.append((r, src, panels))
            print(f"row {r} ({src}): " + "  ".join(f"{l.split('[')[0].strip()}={g:.5f}"
                                                   for l, _, g in panels), flush=True)

    agg = {}
    for _, _, panels in rows:
        for label, _, g in panels:
            if np.isfinite(g):
                agg.setdefault(label, []).append(g)
    print("\n=== MEAN roughness (re-voxelised, comparable) ===")
    for label, vals in agg.items():
        print(f"  {label:34s} {np.mean(vals):.5f}")

    S, PAD, HDR = args.size, 8, 24
    canvas = Image.new("RGB", (4 * S + 5 * PAD, HDR + len(rows) * (S + HDR) + PAD), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 6), "What we ship vs the vecset path — same buildings, identical shading. "
                     "NOTE: only map-#24 is generated; the round-trips see the answer.", fill=(0, 0, 0))
    for i, (r, src, panels) in enumerate(rows):
        y = HDR + i * (S + HDR)
        for j, (label, img, g) in enumerate(panels):
            x0 = PAD + j * (S + PAD)
            if img is not None:
                canvas.paste(img.convert("RGB").resize((S, S)), (x0, y + HDR))
            d.text((x0 + 4, y + 5), f"{label}  {g:.5f}", fill=(0, 0, 0))
    p = Path(args.out) / "deployed_vs_dora.png"
    canvas.save(p)
    print("->", p)


if __name__ == "__main__":
    main()
