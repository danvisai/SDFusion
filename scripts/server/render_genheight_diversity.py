"""Validate the B+.6h height-generation model: is the diversity class-structured & real?

Loads outputs/recipe_diffusion_genheight and, for a fixed footprint:
  - generates K buildings per class (height generated, not given)
  - reports generated slenderness mean/std per class (should rank residential < commercial)
  - renders a sheet: same footprint, K diverse-height generated buildings (3D)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from models.networks import recipe_param_space as ps
from models.networks.diff_recipe import build_diff_recipe
from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
from scene.sdf_primitives import polygon_bbox_with_pad, grid_to_mesh

CK = REPO / "outputs/recipe_diffusion_genheight"


def render(ax, mesh, title):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title + " (empty)", fontsize=7); return
    V, F = mesh.vertices, mesh.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1); col = plt.cm.viridis(0.2 + 0.6 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=col, edgecolors="none"))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=20, azim=-55); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=7)


def main(k=5):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CK / "denoiser.pth", map_location=dev, weights_only=False)
    den = ConditionalDenoiser(cond_dim=ps.COND_DIM, n_params=ck["gen_dim"],
                              hidden=ck["hidden"], depth=ck["depth"]).to(dev)
    den.load_state_dict(ck["model"]); den.eval()
    diff = GaussianDiffusion(ck["timesteps"], device=dev)
    feat, pnorm = ps.load_scalers(CK / "scalers.npz")
    s_mean, s_std = ck["s_mean"], ck["s_std"]

    poly = np.array([[-7, -10], [7, -10], [7, 10], [-7, 10]], dtype=np.float32)  # 14x20 m
    area = 14 * 20; sa = np.sqrt(area)
    probes = [("RESIDENTIAL", "craftsman"), ("RESIDENTIAL", "modern"),
              ("COMMERCIAL", "modern"), ("PUBLIC", "public_civic")]

    print("generated slenderness (height/sqrt area) per class, same footprint:")
    fig = plt.figure(figsize=(2.4 * k, 2.4 * len(probes)))
    for r, (cls, style) in enumerate(probes):
        ci = ps.CLASS_TO_IDX[cls]; si = ps.STYLE_TO_IDX[style]
        c = ps.raw_conditioning(poly, 1.0, ci, si)[None].copy()
        c[:, ps.SLENDERNESS_FEAT_IDX] = 0.0
        ct = torch.tensor(feat.transform(np.repeat(c, k, axis=0)), device=dev)
        with torch.no_grad():
            g = diff.ddim_sample(den, ct, n_params=ck["gen_dim"], steps=50, eta=1.0,
                                 guidance=2.0).cpu().numpy()
        praw = pnorm.inverse(g[:, :ps.MAX_PARAMS], np.full(k, si))
        slen = g[:, ps.MAX_PARAMS] * s_std + s_mean
        heights = slen * sa
        print(f"  {cls:11s}/{style:12s}: slen mean={slen.mean():.2f} std={slen.std():.2f} "
              f"| heights(m)={np.round(np.sort(heights),1)}")
        for j in range(k):
            h = float(max(heights[j], 1.0))
            bbox = polygon_bbox_with_pad(poly, h * 1.5, pad=0.12)
            x0, y0, z0, x1, y1, z1 = bbox; R = 56
            xs = torch.linspace(x0, x1, R, device=dev); ys = torch.linspace(y0, y1, R, device=dev)
            zs = torch.linspace(z0, z1, R, device=dev)
            Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
            pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
            mod = build_diff_recipe(style)[0].to(dev)
            with torch.no_grad():
                sdf = mod(torch.tensor(ps.unpad_params(praw[j], style), device=dev),
                          torch.tensor(poly, device=dev), torch.tensor(h, device=dev),
                          pts).reshape(R, R, R)
            mesh = grid_to_mesh(sdf, bbox, 0.0)
            ax = fig.add_subplot(len(probes), k, r * k + j + 1, projection="3d")
            render(ax, mesh, f"{cls[:4]}/{style[:6]} h={h:.0f}m")
    fig.suptitle("B+.6h: same footprint -> diverse GENERATED heights, per class", fontsize=11)
    fig.tight_layout(); out = CK / "genheight_diversity.png"; fig.savefig(out, dpi=100)
    plt.close(fig); print(f"[save] {out}")


if __name__ == "__main__":
    main()
