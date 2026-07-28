"""Render what the frozen gate actually produced: GT vs Dora vs TripoSG, same buildings, side by side.

The gate reported scalars only. Per the effort's standing rule (#36) the **visual is the primary
arbiter** and the scalar is a diagnostic -- and the roughness metric is known to be blind to at least
one artifact class (fine-scale striation on flat faces, #63). So the numbers should not be signed off
without looking at the surfaces they describe.

Every panel is meshed at level 0.0 from a continuous field and shaded identically, so differences are
the geometry rather than the renderer.
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

from scripts.foundations.baseline_gate_eval import mesh_sdf_surface           # noqa: E402
from scripts.foundations.vecset_ceiling_probe import (                        # noqa: E402
    RES, TRUNC, grid_points, verts_to_world, test_indices,
)
from scripts.foundations.dora_roundtrip_probe import (                        # noqa: E402
    _stub_absent_deps, load_dora, sample_surface, sample_sharp_edges, H5,
)
from scripts.foundations.dora_frozen_gate import load_surfaces, _revoxel, _rough  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="outputs/frozen_gate_montage")
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    import h5py, trimesh
    from PIL import Image, ImageDraw
    from scripts.hunyuan_building_mesh_smoke import render_mesh_png

    surf = load_surfaces()
    with h5py.File(H5, "r") as f:
        held = [int(i) for i in test_indices(int(f["sdf"].shape[0]))]
    by_src = {s: [r for r in held if r in surf and surf[r][2] == s]
              for s in ("bag3d", "nrw", "plateau")}
    picks, i = [], 0
    while len(picks) < args.n and any(len(v) > i for v in by_src.values()):
        for s in by_src:
            if len(by_src[s]) > i and len(picks) < args.n:
                picks.append(by_src[s][i])
        i += 1

    dora = load_dora(dev)
    _stub_absent_deps()
    sys.path.insert(0, str(REPO / "external/TripoSG"))
    from triposg.models.autoencoders.autoencoder_kl_triposg import TripoSGVAEModel
    tri = TripoSGVAEModel.from_pretrained(str(REPO / "external/triposg_vae")).eval().to(dev)

    pts = grid_points()
    q = torch.from_numpy(pts.astype(np.float32))[None].to(dev)
    CH = 32768
    rows = []

    for r in picks:
        v, fc, src = surf[r]
        mesh = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(fc), process=False)
        coarse = torch.from_numpy(sample_surface(mesh, 8192, rng))[None].to(dev)
        sharp = torch.from_numpy(sample_sharp_edges(mesh, 8192, rng))[None].to(dev)

        with torch.no_grad():
            _, kl, _ = dora.encode(coarse, sharp, sample_posterior=False)
            lat = dora.decode(kl)
            d_field = -torch.cat([dora.query(q[:, j:j + CH], lat).float()
                                  for j in range(0, q.shape[1], CH)], 1).view(RES, RES, RES).cpu().numpy()
            x = torch.cat([coarse, sharp], dim=1)
            z = tri.encode(x).latent_dist.mode()
            t_field = torch.cat([tri.decode(z, q[:, j:j + CH]).sample.float().squeeze(-1)
                                 for j in range(0, q.shape[1], CH)], 1).view(RES, RES, RES).cpu().numpy()

        panels = []
        # GT arm: the recovered surface itself, through the same re-voxelise path as the gate
        gt_field = _revoxel(v, fc, pts)
        for label, fld in (("GT (real LoD2)", gt_field), ("Dora frozen", d_field),
                           ("TripoSG frozen", t_field)):
            mv, mf = mesh_sdf_surface(np.clip(fld, -TRUNC, TRUNC))
            if mv is None:
                panels.append((label, None, float("nan"))); continue
            m = trimesh.Trimesh(verts_to_world(mv), mf, process=False)
            panels.append((label, render_mesh_png(m, image_size=args.size), _rough(fld)))
        rows.append((r, src, panels))
        print(f"row {r} ({src}) rendered  " +
              "  ".join(f"{l}={g:.5f}" for l, _, g in panels), flush=True)

    S, PAD, HDR = args.size, 8, 24
    W = 3 * S + 4 * PAD
    H = HDR + len(rows) * (S + HDR) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 6), "Frozen round-trip gate — same building, identical shading, meshed at level 0.0",
           fill=(0, 0, 0))
    for i, (r, src, panels) in enumerate(rows):
        y = HDR + i * (S + HDR)
        for j, (label, img, g) in enumerate(panels):
            x0 = PAD + j * (S + PAD)
            if img is not None:
                canvas.paste(img.convert("RGB").resize((S, S)), (x0, y + HDR))
            d.text((x0 + 4, y + 5), f"{label}   roughness {g:.5f}", fill=(0, 0, 0))
    path = Path(args.out) / "frozen_gate_montage.png"
    canvas.save(path)
    print("->", path)


if __name__ == "__main__":
    main()
