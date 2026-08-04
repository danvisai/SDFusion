"""Validate the generative detail head: same procedural base building, K model-SAMPLED
facade variations (real diversity), and a per-style row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine
from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
from models.networks import recipe_param_space as ps
from scene.sdf_edit import recipe_base_sdf
from scene import sdf_detail as det
from scene.sdf_primitives import sample_grid, grid_to_mesh

CK = REPO / "outputs/detail_generator/detail_gen.pth"


def render(ax, mesh, title):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title, fontsize=7); return
    V, F = mesh.vertices, mesh.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1); c = plt.cm.bone(0.25 + 0.6 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=c, edgecolors="k", linewidths=0.05))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=14, azim=-52); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=7)


def main():
    out = REPO / "outputs/detail_generator"; out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CK, map_location=dev, weights_only=False)
    den = ConditionalDenoiser(cond_dim=len(ps.STYLES), n_params=ck["n_dim"],
                              hidden=ck["hidden"], depth=ck["depth"]).to(dev)
    den.load_state_dict(ck["model"]); den.eval()
    diff = GaussianDiffusion(ck["timesteps"], device=dev)
    mean, std = ck["mean"], ck["std"]

    eng = RecipeInferenceEngine()

    def sample_detail(style, k, seed):
        torch.manual_seed(seed)
        si = ps.STYLE_TO_IDX[style]
        c = torch.zeros(k, len(ps.STYLES), device=dev); c[:, si] = 1.0
        with torch.no_grad():
            g = diff.ddim_sample(den, c, n_params=ck["n_dim"], steps=50, eta=1.0).cpu().numpy()
        return np.clip(g * std + mean, det.DETAIL_LO, det.DETAIL_HI)

    def build(style, cls, fp, H, vec):
        params = eng.sample_params(fp, H, cls, style, seed=1)
        base = recipe_base_sdf(style, params, fp, H, device=dev)
        detailed = det.add_facade_detail(base, fp, H, det.vector_to_params(vec))
        poly = np.asarray(fp); pad = 1.5
        bbox = (poly[:, 0].min()-pad, 0.0, poly[:, 1].min()-pad, poly[:, 0].max()+pad, H+1.5, poly[:, 1].max()+pad)
        return grid_to_mesh(sample_grid(detailed, 104, bbox, device=dev), bbox, 0.0)

    # Row 1: same modern building, K model-sampled facades (diversity)
    fp = [[-7, -10], [7, -10], [7, 10], [-7, 10]]; H = 20.0
    K = 4
    vecs = sample_detail("modern", K, seed=7)
    # Row 2: one sample per style (style structure)
    styles = ["modern", "victorian", "industrial", "public_civic"]
    fig = plt.figure(figsize=(3.2 * K, 6.6))
    for j in range(K):
        ax = fig.add_subplot(2, K, j + 1, projection="3d")
        render(ax, build("modern", "COMMERCIAL", fp, H, vecs[j]), f"modern — sample {j+1}")
    for j, s in enumerate(styles):
        v = sample_detail(s, 1, seed=3 + j)[0]
        ax = fig.add_subplot(2, K, K + j + 1, projection="3d")
        render(ax, build(s, "COMMERCIAL", fp, H, v), f"{s} — generated detail")
    fig.suptitle("Generative facade detail: same procedural base, model-sampled detail "
                 "(top: modern variety; bottom: per-style)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "detail_generator.png", dpi=115); plt.close(fig)
    print(f"[save] {out/'detail_generator.png'}")


if __name__ == "__main__":
    main()
