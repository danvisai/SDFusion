"""Sample N random parameter vectors for one style and render the buildings.

This shows the DESIGN SPACE accessible through the parameter vector — what a
diffusion head sampling these params would produce. Use to sanity-check that
each style can produce visibly distinct buildings (not just slight variants).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from models.networks.diff_recipe import (
    build_diff_recipe, make_grid_points, grid_from_flat, bbox_for_polygon,
)

OUT = REPO / "outputs/diff_recipe_diversity"
OUT.mkdir(parents=True, exist_ok=True)


def _rect_polygon(w, d, cx=0.0, cz=0.0):
    return np.array([
        [cx - w/2, cz - d/2], [cx + w/2, cz - d/2],
        [cx + w/2, cz + d/2], [cx - w/2, cz + d/2],
    ], dtype=np.float32)


def _L_polygon(w, d):
    """L-shaped footprint to test the polygon-prism path."""
    return np.array([
        [-w/2, -d/2], [w/2, -d/2], [w/2, 0.0],
        [0.0, 0.0], [0.0, d/2], [-w/2, d/2],
    ], dtype=np.float32)


def render(sdf_grid, bbox, out_path, title=""):
    import skimage.measure as sk
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    g = sdf_grid.detach().cpu().numpy()
    fig = plt.figure(figsize=(3.5, 3.5))
    if not (g.min() < 0 < g.max()):
        plt.text(0.5, 0.5, "no iso=0", ha="center", va="center")
        plt.title(title, fontsize=8); plt.axis("off")
    else:
        verts, faces, _, _ = sk.marching_cubes(g, level=0.0)
        x0, y0, z0, x1, y1, z1 = bbox
        D = g.shape[0]
        s = np.array([(x1-x0), (y1-y0), (z1-z0)], dtype=np.float32) / (D - 1)
        o = np.array([x0, y0, z0], dtype=np.float32)
        # plot_x=world_x, plot_y=world_z (depth), plot_z=world_y (UP)
        verts_world = np.stack([verts[:, 2]*s[0]+o[0],
                                verts[:, 0]*s[2]+o[2],
                                verts[:, 1]*s[1]+o[1]], axis=-1)
        ax = fig.add_subplot(111, projection="3d")
        tri = verts_world[faces]
        poly = Poly3DCollection(tri, alpha=0.75, edgecolor="k", linewidth=0.04)
        poly.set_facecolor((0.65, 0.65, 0.75))
        ax.add_collection3d(poly)
        pad = 0.5
        ax.set_xlim(verts_world[:,0].min()-pad, verts_world[:,0].max()+pad)
        ax.set_ylim(verts_world[:,1].min()-pad, verts_world[:,1].max()+pad)
        ax.set_zlim(verts_world[:,2].min()-pad, verts_world[:,2].max()+pad)
        ax.view_init(elev=22, azim=35); ax.set_axis_off()
        ax.set_title(title, fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()


@torch.no_grad()
def style_design_space(style: str, device: str, n_samples: int = 6, seed: int = 0):
    """Sample n different param vectors + render each."""
    torch.manual_seed(seed); np.random.seed(seed)
    module, default_fn, n_params = build_diff_recipe(style)
    module = module.to(device)
    defaults = default_fn(device)

    # Use a mix of footprints + heights for diversity
    cases = [
        ("rect_wide", _rect_polygon(14, 8),   6.0),
        ("rect_tall", _rect_polygon(8, 8),   14.0),
        ("rect_sml",  _rect_polygon(6, 5),    4.0),
        ("L_shape",   _L_polygon(12, 12),     7.0),
        ("rect_big",  _rect_polygon(16, 12),  9.0),
        ("rect_sq",   _rect_polygon(10, 10), 10.0),
    ]
    cases = cases[:n_samples]

    panels = []
    for i, (label, poly_np, height) in enumerate(cases):
        # Perturb params with strong noise; keep critical occupancy gates ON
        # so the buildings actually display their feature set
        p = defaults.clone() * (1.0 + 0.5 * torch.randn_like(defaults))
        # Force positive occupancy gates (logit > 0 -> active)
        if style == "modern":
            p[2] = torch.rand(()) * 4.0 - 1.0  # mech_active_logit
        if style == "colonial":
            p[1] = torch.rand(()) * 4.0 - 1.0
        if style == "craftsman":
            p[2] = torch.rand(()) * 4.0 - 1.0
        # Snap unstable params (LR-sensitive) to positive ranges
        for j, val in enumerate(p):
            if defaults[j].item() > 0:
                p[j] = p[j].abs() + 1e-4

        poly_t = torch.tensor(poly_np, dtype=torch.float32, device=device)
        h_t = torch.tensor(height, dtype=torch.float32, device=device)
        bbox = bbox_for_polygon(poly_np, height, pad=1.5)
        grid_pts = make_grid_points(64, bbox, device)
        sdf_flat = module(p, poly_t, h_t, grid_pts)
        sdf_grid = grid_from_flat(sdf_flat, 64).cpu()
        out_path = OUT / f"{style}_var{i}_{label}.png"
        render(sdf_grid, bbox, out_path, title=f"{label}\nh={height:.0f}")
        panels.append(out_path)
    return panels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Diversity demo for all 8 styles (device={device})")
    style_panels = {}
    for style in ["modern", "colonial", "victorian", "industrial",
                  "craftsman", "mediterranean", "contemporary", "public_civic"]:
        print(f"  rendering {style}...")
        style_panels[style] = style_design_space(style, device, n_samples=6, seed=42)

    # Combined sheet: rows = styles, cols = variants
    from PIL import Image
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows = len(style_panels)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows))
    for ri, (style, paths) in enumerate(style_panels.items()):
        for ci, p in enumerate(paths):
            im = Image.open(p)
            axes[ri, ci].imshow(im); axes[ri, ci].axis("off")
            if ci == 0:
                axes[ri, ci].text(-0.1, 0.5, style, transform=axes[ri, ci].transAxes,
                                  ha="right", va="center", rotation=0, fontsize=11, fontweight="bold")
    plt.tight_layout()
    sheet = OUT / "design_space_8styles_6variants.png"
    plt.savefig(sheet, dpi=100, bbox_inches="tight"); plt.close()
    print(f"  wrote {sheet}")


if __name__ == "__main__":
    main()
