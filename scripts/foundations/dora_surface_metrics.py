"""Two follow-ups the grid-Laplacian metric could not answer.

**(1) A resolution-independent surface metric.** `surface_roughness` is a discrete Laplacian on a grid,
so on a genuinely flat surface it collapses as h^2 while a fixed-wavelength ripple's does not. That makes
the GT/Dora *ratio* worsen with resolution (14x at 64^3, 78x at 256^3) even as the decoded surface
visibly improves -- the instrument, not the geometry. Chamfer distance and normal consistency are
measured on the surfaces themselves and have no such artifact.

**(2) Does decode noise track sampling coverage?** The hypothesis: where our sampler left a face
sparsely covered, the encoder had no evidence and the decoder filled it with noise. Dora is not
image-conditioned so there is no literally "unseen side", but sparse point coverage is the direct
analog. Measured per-face rather than argued.
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

from scripts.foundations.baseline_gate_eval import mesh_sdf_surface            # noqa: E402
from scripts.foundations.dora_roundtrip_probe import (                         # noqa: E402
    load_dora, sample_surface, sample_sharp_edges,
)
from scripts.foundations.dora_frozen_gate import load_surfaces                 # noqa: E402


def decode_mesh(model, lat, R, dev, chunk=65536):
    """Query the decoder on an R^3 grid and mesh at level 0.0 -- resolution is a free parameter for a
    query decoder, which is the whole point of the representation."""
    import trimesh
    ax = np.linspace(-1, 1, R)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    q = torch.from_numpy(pts)[None].to(dev)
    with torch.no_grad():
        vals = torch.cat([model.query(q[:, j:j + chunk], lat).float()
                          for j in range(0, q.shape[1], chunk)], 1)
    fld = -vals.view(R, R, R).cpu().numpy()
    v, f = mesh_sdf_surface(np.clip(fld, -0.2, 0.2))
    if v is None:
        return None
    return trimesh.Trimesh(v * (2.0 / (R - 1)) - 1.0, f, process=False)


def surface_metrics(a, b, n=60000, rng=None):
    """Symmetric Chamfer (mean) + normal consistency between two meshes, sampled on the surfaces."""
    import trimesh
    from scipy.spatial import cKDTree
    rng = rng or np.random.default_rng(0)
    pa, fa = trimesh.sample.sample_surface(a, n)
    pb, fb = trimesh.sample.sample_surface(b, n)
    na, nb = a.face_normals[fa], b.face_normals[fb]
    ta, tb = cKDTree(pa), cKDTree(pb)
    dab, iab = tb.query(pa)          # a -> b
    dba, iba = ta.query(pb)          # b -> a
    chamfer = float(dab.mean() + dba.mean()) / 2
    nc = float((np.abs((na * nb[iab]).sum(1)).mean() +
                np.abs((nb * na[iba]).sum(1)).mean()) / 2)
    return chamfer, nc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[4128, 29650, 20133, 22430])
    ap.add_argument("--noisy_row", type=int, default=22430)
    args = ap.parse_args()

    import trimesh
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    model = load_dora(dev)
    surf = load_surfaces()

    print("\n=== (1) resolution-independent surface metrics vs GT ===")
    print(f"{'row':>6} {'R':>5} {'chamfer':>10} {'normal-consist':>15}")
    agg = {}
    for row in args.rows:
        v, fc, _ = surf[row]
        gt = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(fc), process=False)
        coarse = torch.from_numpy(sample_surface(gt, 8192, rng))[None].to(dev)
        sharp = torch.from_numpy(sample_sharp_edges(gt, 8192, rng))[None].to(dev)
        with torch.no_grad():
            _, kl, _ = model.encode(coarse, sharp, sample_posterior=False)
            lat = model.decode(kl)
        for R in (64, 256):
            dm = decode_mesh(model, lat, R, dev)
            if dm is None:
                print(f"{row:>6} {R:>5}  (no surface)"); continue
            ch, nc = surface_metrics(gt, dm, rng=rng)
            agg.setdefault(R, []).append((ch, nc))
            print(f"{row:>6} {R:>5} {ch:>10.5f} {nc:>15.4f}")
    print("\n  mean by resolution (chamfer lower better, normal-consistency 1.0 = perfect):")
    for R in sorted(agg):
        a = np.array(agg[R])
        print(f"    {R:>4}^3  chamfer {a[:,0].mean():.5f}   normal-consistency {a[:,1].mean():.4f}")

    print("\n=== (2) does decode noise track sampling coverage? ===")
    v, fc, _ = surf[args.noisy_row]
    gt = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(fc), process=False)
    area = gt.area_faces
    _, fid = trimesh.sample.sample_surface(gt, 8192)
    cnt = np.bincount(fid, minlength=len(gt.faces))
    print(f"  row {args.noisy_row}: {len(gt.faces)} faces, {len(gt.vertices)} verts")
    print(f"  faces with ZERO uniform samples: {(cnt==0).sum()} / {len(cnt)}")
    print(f"  face area  min={area.min():.2e}  median={np.median(area):.2e}  max={area.max():.2e}")
    print(f"  area share of zero-sample faces: {100*area[cnt==0].sum()/area.sum():.3f}%")
    print(f"  watertight={gt.is_watertight}  volume={gt.volume:.4f}  "
          f"degenerate faces={(area < 1e-9).sum()}")
    order = np.argsort(cnt)[:5]
    print("  least-covered faces (idx, samples, area):",
          [(int(i), int(cnt[i]), float(area[i])) for i in order])


if __name__ == "__main__":
    main()
