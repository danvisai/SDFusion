"""Render GT vs fitted recipe meshes for each asset in best_params.npz.

Picks top-K best fits + bottom-K worst fits + a few median ones — gives a
diagnostic view of where the recipe library works and where it fails.
"""

from __future__ import annotations
import sys, h5py
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import skimage.measure as skm

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from models.networks.diff_recipe import build_diff_recipe, make_grid_points
import scripts.fit_recipes_to_buildingnet as fitter


FITS_NPZ = REPO / "outputs/fit_recipes_buildingnet/best_params.npz"
OUT_DIR = REPO / "outputs/fit_recipes_buildingnet/visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_sdf_grid(sdf_grid, bbox, ax, title=""):
    """Render SDF iso=0 via marching cubes.

    Coordinate handling:
      SDF grid axes: (D=z, H=y, W=x) — preprocess/create_sdf.py convention.
      World convention: Y is UP, X is east-west, Z is north-south.
      matplotlib 3D: default Z axis is vertical, so we need plot_z = world_y.
      For sparse/non-watertight BuildingNet GT SDFs the iso=0 contour may be
      very thin; we fall back to iso=0.005 if iso=0 produces too few faces.
    """
    g = sdf_grid.detach().cpu().numpy() if isinstance(sdf_grid, torch.Tensor) else sdf_grid
    iso = 0.0
    if not (g.min() < iso < g.max()):
        # Try a small positive iso for sparse SDFs
        for try_iso in (0.005, 0.01, 0.02, 0.05):
            if g.min() < try_iso < g.max():
                iso = try_iso; break
        else:
            ax.text(0.5, 0.5, f"no iso\nmin={g.min():.3f}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            ax.set_title(title, fontsize=8); ax.axis("off")
            return
    try:
        verts, faces, _, _ = skm.marching_cubes(g, level=iso)
    except Exception:
        ax.text(0.5, 0.5, "MC failed", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8); ax.axis("off")
        return
    if len(verts) < 4:
        ax.text(0.5, 0.5, f"sparse\n{len(verts)} verts", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_title(title, fontsize=8); ax.axis("off")
        return
    x0, y0, z0, x1, y1, z1 = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
    D = g.shape[0]
    s = np.array([(x1-x0), (y1-y0), (z1-z0)], dtype=np.float32) / (D - 1)
    o = np.array([x0, y0, z0], dtype=np.float32)
    # verts[:, 0]=D-index→world z; [:, 1]=H-index→world y; [:, 2]=W-index→world x
    # For matplotlib Y-up rendering: plot_x=world_x, plot_y=world_z (depth), plot_z=world_y (UP)
    vw_plot = np.stack([
        verts[:, 2] * s[0] + o[0],   # plot_x  = world_x  (from W index)
        verts[:, 0] * s[2] + o[2],   # plot_y  = world_z  (from D index, depth)
        verts[:, 1] * s[1] + o[1],   # plot_z  = world_y  (from H index, UP)
    ], axis=-1)
    tri = vw_plot[faces]
    poly = Poly3DCollection(tri, alpha=0.75, edgecolor="k", linewidth=0.03)
    poly.set_facecolor((0.65, 0.65, 0.75))
    ax.add_collection3d(poly)
    pad = 0.05
    ax.set_xlim(vw_plot[:,0].min()-pad, vw_plot[:,0].max()+pad)
    ax.set_ylim(vw_plot[:,1].min()-pad, vw_plot[:,1].max()+pad)
    ax.set_zlim(vw_plot[:,2].min()-pad, vw_plot[:,2].max()+pad)
    ax.view_init(elev=22, azim=35); ax.set_axis_off()
    ax.set_title(title + (f" iso={iso}" if iso != 0.0 else ""), fontsize=8)


@torch.no_grad()
def main():
    if not FITS_NPZ.exists():
        print(f"[!] no fits at {FITS_NPZ}")
        return 1
    data = np.load(FITS_NPZ, allow_pickle=True)["fits"].item()
    rows = []
    for aid, fit in data.items():
        rows.append({"asset_id": aid, **fit})
    rows.sort(key=lambda r: r["iou"], reverse=True)
    n = len(rows)
    # Pick top 4 + bottom 4 + 4 median
    top4 = rows[:4]
    bot4 = rows[-4:]
    med_idx = n // 2
    med4 = rows[max(0, med_idx-2):med_idx+2]
    picks = [("TOP", r) for r in top4] + [("MED", r) for r in med4] + [("BOT", r) for r in bot4]
    print(f"Rendering {len(picks)} fits (top4 + med4 + bot4)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7, 3.5 * n_rows),
                             subplot_kw={"projection": "3d"})
    for ri, (tag, row) in enumerate(picks):
        aid = row["asset_id"]
        sdf_np, fp_np, bbox = fitter.load_asset(aid)
        if sdf_np is None:
            continue
        # GT render
        render_sdf_grid(sdf_np, bbox, axes[ri, 0],
                        title=f"[{tag}] GT — {aid[:34]}\nFP_IoU={row['iou']:.3f}")
        # Recipe render with fitted params
        style = row["style"]
        params = torch.tensor(row["params"], dtype=torch.float32, device=device)
        module, _, _ = build_diff_recipe(style)
        module = module.to(device)
        poly = torch.tensor(row["polygon"], dtype=torch.float32, device=device)
        x0, y0, z0, x1, y1, z1 = bbox.tolist()
        h_t = torch.tensor(y1 - y0, dtype=torch.float32, device=device)
        qp = fitter.make_query_grid(bbox, device)
        qp[:, 1] -= y0
        pred = module(params, poly, h_t, qp).reshape(64, 64, 64)
        render_sdf_grid(pred, bbox, axes[ri, 1],
                        title=f"Fitted {style}\nfp_IoU={row['iou']:.3f}")
    plt.tight_layout()
    out_path = OUT_DIR / "fit_quality_sheet.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
