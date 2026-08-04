"""Validate that the B+.6 generative head produces plausible 3D BUILDINGS.

Footprint IoU only checks the top-down silhouette. This renders the actual generated
meshes (matplotlib 3D, headless) so we can eyeball the 3D form and the diversity:

  - gallery   : all 8 styles on a standard footprint (one seed each)
  - diversity : one style across several seeds + guidance presets (3D variation)

Outputs PNGs under outputs/recipe_param_diffusion_b6/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine, DIVERSITY_PRESETS
from models.networks import recipe_param_space as ps

OUT = REPO / "outputs/recipe_param_diffusion_b6"


def render_mesh(ax, mesh, title=""):
    if mesh is None or len(mesh.faces) == 0:
        ax.text2D(0.5, 0.5, "EMPTY", ha="center"); ax.set_title(title, fontsize=8); return
    V, F = mesh.vertices, mesh.faces
    tris = V[F]                                   # (nf, 3, 3) -> (x,y,z)
    # Shade by face height (y) so massing reads clearly.
    fy = tris[:, :, 1].mean(axis=1)
    norm = (fy - fy.min()) / (np.ptp(fy) + 1e-9)
    colors = plt.cm.viridis(0.2 + 0.6 * norm)
    pc = Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=colors,  # plot (x,z,y) so y is up
                          edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    for lo, hi, s in [(x.min(), x.max(), "x"), (z.min(), z.max(), "z"), (y.min(), y.max(), "y")]:
        pass
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try:
        ax.set_box_aspect((np.ptp(x), np.ptp(z), np.ptp(y)))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-50)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=8)


def main():
    eng = RecipeInferenceEngine(grid_res=56)
    poly = np.array([[-6, -8], [6, -8], [6, 8], [-6, 8]], dtype=np.float32)  # 12x16 m
    heights = {"modern": 18, "colonial": 9, "victorian": 11, "industrial": 12,
               "craftsman": 8, "mediterranean": 8, "contemporary": 16, "public_civic": 14}

    # --- gallery: 8 styles ---
    fig = plt.figure(figsize=(16, 8))
    for i, style in enumerate(ps.STYLES):
        b = eng.generate_building(poly, "RESIDENTIAL", heights[style], style, seed=3)
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        import trimesh, io
        m = trimesh.load(io.BytesIO(b.glb), file_type="glb").dump(concatenate=True) if b.glb else None
        render_mesh(ax, m, f"{style}  ({b.n_vertices}v/{b.n_faces}f, h={heights[style]}m)")
    fig.suptitle("B+.6 generated buildings — 8 styles (12x16 m footprint)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "mesh_gallery_styles.png", dpi=95)
    plt.close(fig)
    print(f"[save] {OUT/'mesh_gallery_styles.png'}")

    # --- diversity: modern across seeds, and a style across guidance presets ---
    fig = plt.figure(figsize=(16, 8))
    for j, seed in enumerate(range(4)):
        b = eng.generate_building(poly, "COMMERCIAL", 18, "modern", seed=seed, guidance=3.0)
        ax = fig.add_subplot(2, 4, j + 1, projection="3d")
        import trimesh, io
        m = trimesh.load(io.BytesIO(b.glb), file_type="glb").dump(concatenate=True) if b.glb else None
        render_mesh(ax, m, f"modern seed={seed} (g=3)")
    for j, (name, g) in enumerate(DIVERSITY_PRESETS.items()):
        b = eng.generate_building(poly, "RESIDENTIAL", 8, "victorian", seed=7, guidance=g)
        ax = fig.add_subplot(2, 4, 4 + j + 1, projection="3d")
        import trimesh, io
        m = trimesh.load(io.BytesIO(b.glb), file_type="glb").dump(concatenate=True) if b.glb else None
        render_mesh(ax, m, f"victorian diversity={name} (g={g})")
    fig.suptitle("B+.6 3D diversity — modern across seeds + victorian across guidance", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "mesh_gallery_diversity.png", dpi=95)
    plt.close(fig)
    print(f"[save] {OUT/'mesh_gallery_diversity.png'}")


if __name__ == "__main__":
    main()
