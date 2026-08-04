"""Phase B+.3 verification — test all 8 differentiable building styles.

For each style:
  1. Forward pass via DiffRecipe<Style> with default params
  2. Forward pass via scene/sdf_recipes (procedural reference)
  3. Compare: sign_match, iso_iou, L1
  4. Render side-by-side, save PNG

Pass criterion: every style achieves sign_match > 0.85 with procedural reference.

Note: procedural recipes have RANDOM mech/chimney/porch placement controlled by
their `seed` arg, while DiffRecipe uses fixed default params. So we don't expect
bit-identical matches — only that the dominant building shape (body + roof +
key features) overlaps strongly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from models.networks.diff_recipe import (
    DIFF_RECIPE_REGISTRY, build_diff_recipe,
    make_grid_points, grid_from_flat, bbox_for_polygon,
)
from scene.sdf_recipes import build_styled_sdf
from scene.sdf_primitives import sample_grid


OUT_DIR = REPO / "outputs/diff_recipe_phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rect_polygon(w: float, d: float, cx: float = 0.0, cz: float = 0.0) -> np.ndarray:
    return np.array([
        [cx - w / 2, cz - d / 2],
        [cx + w / 2, cz - d / 2],
        [cx + w / 2, cz + d / 2],
        [cx - w / 2, cz + d / 2],
    ], dtype=np.float32)


def render_mesh_from_sdf(sdf_grid: torch.Tensor, bbox, out_path: Path,
                         title: str = ""):
    import skimage.measure as sk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    g = sdf_grid.detach().cpu().numpy()
    fig = plt.figure(figsize=(4, 4))
    if not (g.min() < 0 < g.max()):
        plt.text(0.5, 0.5, f"no iso=0\nmin={g.min():.2f} max={g.max():.2f}",
                 ha="center", va="center")
        plt.title(title)
        plt.axis("off")
    else:
        verts, faces, _, _ = sk.marching_cubes(g, level=0.0)
        x0, y0, z0, x1, y1, z1 = bbox
        D = g.shape[0]
        scale = np.array([(x1 - x0), (y1 - y0), (z1 - z0)], dtype=np.float32) / (D - 1)
        origin = np.array([x0, y0, z0], dtype=np.float32)
        # SDF grid axes (D=z, H=y, W=x). matplotlib 3D has Z vertical.
        # plot_x=world_x, plot_y=world_z (depth), plot_z=world_y (UP)
        verts_world = np.stack([
            verts[:, 2] * scale[0] + origin[0],   # plot_x = world_x (W index)
            verts[:, 0] * scale[2] + origin[2],   # plot_y = world_z (D index)
            verts[:, 1] * scale[1] + origin[1],   # plot_z = world_y (H index, UP)
        ], axis=-1)
        ax = fig.add_subplot(111, projection="3d")
        tri = verts_world[faces]
        poly = Poly3DCollection(tri, alpha=0.7, edgecolor="k", linewidth=0.05)
        poly.set_facecolor((0.65, 0.65, 0.75))
        ax.add_collection3d(poly)
        pad = 0.5
        ax.set_xlim(verts_world[:, 0].min() - pad, verts_world[:, 0].max() + pad)
        ax.set_ylim(verts_world[:, 1].min() - pad, verts_world[:, 1].max() + pad)
        ax.set_zlim(verts_world[:, 2].min() - pad, verts_world[:, 2].max() + pad)
        ax.view_init(elev=22, azim=35)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def test_style(style: str, device: str, width: float, depth: float, height: float, seed: int = 0):
    poly_np = _rect_polygon(width, depth)
    poly_t = torch.tensor(poly_np, dtype=torch.float32, device=device)
    h_t = torch.tensor(height, dtype=torch.float32, device=device)

    bbox = bbox_for_polygon(poly_np, height, pad=1.5)
    res = 64
    grid_pts = make_grid_points(res, bbox, device)

    # Procedural reference
    ref_fn = build_styled_sdf(style, poly_np, height, seed=seed)
    ref_grid = sample_grid(ref_fn, res, bbox, device=device).cpu()

    # Differentiable recipe
    module, default_fn, n_params = build_diff_recipe(style)
    module = module.to(device)
    params = default_fn(device)
    diff_sdf_flat = module(params, poly_t, h_t, grid_pts)
    diff_grid = grid_from_flat(diff_sdf_flat, res).cpu()

    sign_match = ((ref_grid <= 0) == (diff_grid <= 0)).float().mean().item()
    iso_inter = ((ref_grid <= 0) & (diff_grid <= 0)).sum().float()
    iso_union = ((ref_grid <= 0) | (diff_grid <= 0)).sum().float()
    iso_iou = (iso_inter / iso_union.clamp_min(1)).item()
    l1 = (ref_grid - diff_grid).abs().mean().item()

    render_mesh_from_sdf(ref_grid, bbox, OUT_DIR / f"{style}_ref.png",
                         title=f"{style} (procedural)")
    render_mesh_from_sdf(diff_grid, bbox, OUT_DIR / f"{style}_diff.png",
                         title=f"{style} (DiffRecipe)")

    return {
        "style": style, "n_params": n_params,
        "sign_match": sign_match, "iso_iou": iso_iou, "L1": l1,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase B+.3 verification — all 8 styles (device={device})")
    width, depth, height = 12.0, 10.0, 8.0
    print(f"  test footprint: {width:.0f} x {depth:.0f}, height {height:.0f}")
    print()

    rows = []
    for style in DIFF_RECIPE_REGISTRY:
        try:
            res = test_style(style, device, width, depth, height, seed=0)
            print(f"  {style:14s}  N={res['n_params']:2d}  sign_match={res['sign_match']:.3f}  iso_iou={res['iso_iou']:.3f}  L1={res['L1']:.3f}")
            rows.append(res)
        except Exception as exc:
            print(f"  {style:14s}  ERROR: {exc}")
            import traceback; traceback.print_exc()
            rows.append({"style": style, "sign_match": 0.0, "error": str(exc)})

    # Combined comparison sheet
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, r in enumerate(rows):
        style = r["style"]
        try:
            ref_im = Image.open(OUT_DIR / f"{style}_ref.png")
            diff_im = Image.open(OUT_DIR / f"{style}_diff.png")
            axes[i, 0].imshow(ref_im); axes[i, 0].set_title(f"{style} ref"); axes[i, 0].axis("off")
            axes[i, 1].imshow(diff_im); axes[i, 1].set_title(f"{style} diff (sign_match={r.get('sign_match',0):.3f})"); axes[i, 1].axis("off")
        except FileNotFoundError:
            axes[i, 0].text(0.5, 0.5, f"{style}: failed", ha="center", va="center")
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
    plt.tight_layout()
    sheet_path = OUT_DIR / "all_styles_sheet.png"
    plt.savefig(sheet_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\n  wrote {sheet_path}")

    # Summary
    print("\n=== SUMMARY ===")
    n_pass = sum(1 for r in rows if r.get("sign_match", 0) > 0.85)
    n_total = len(rows)
    print(f"  styles passing sign_match > 0.85: {n_pass} / {n_total}")
    overall = (n_pass == n_total)
    print("\n" + "=" * 25)
    print(f"  PHASE B+.3 RESULT: {'PASS ✓' if overall else 'PARTIAL ⚠'}  ({n_pass}/{n_total})")
    print("=" * 25)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
