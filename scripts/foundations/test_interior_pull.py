"""Interior-pull regression check for CoherentPartRefiner / integrate_new_part.

The OLDER PartSetRefiner was removed from the demo (2026-06-15, see sculpt.html) because it
"pulled exterior windows inward and carved holes." integrate_new_part uses a DIFFERENT model
(CoherentPartRefiner, marker-conditioned + X-Part locality) and already resnaps to the surface
as a safety net -- but that snap logic (_snap_to_surface: nearest occupied-boundary voxel) can't
distinguish a TRUE exterior wall from an interior-facing courtyard wall on a non-convex footprint.
All prior tests this session used a convex box, which can't exhibit this failure at all.

This test uses a U-shaped (courtyard) footprint -- the hardest case -- and checks BOTH:
  1. quantitatively: |massing SDF| at each op's final center should be small (on/near the
     surface); a large NEGATIVE value means the op ended up buried inside solid mass.
  2. visually: render the composed mesh and look for stray interior cavities/holes.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/foundations/test_interior_pull.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "server"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from recipe_inference import RecipeInferenceEngine
from refine import Refiner
import layout_detail as ld

OUT = REPO / "outputs/part_set_refiner"


def U(w, d, cw, cd):
    """U-shaped footprint (a courtyard notch) -- local coords, centered."""
    return [[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [w/2-cw, d/2], [w/2-cw, d/2-cd],
            [-w/2+cw, d/2-cd], [-w/2+cw, d/2], [-w/2, d/2]]


def sdf_at(grid, pts_world):
    """True signed distance IN WORLD VOXELS at viewer-frame (world cube) op centers,
    read straight off the massing SDF grid (layout (z,y,x)).

    HISTORY (2026-07-08): this used to rebuild an EDT over the OCC-FRAME resampled
    occupancy and then applied the occ->world transform to points that were ALREADY
    world-frame — every wall-mounted op evaluated at a spot shrunk toward the building
    interior and read as "buried". That measurement bug was the entire 2026-07-03
    "propose_detail_ops buries windows on non-convex footprints" report; the ops were
    on-surface all along (re-verified via trilinear world SDF and 96^3 detail renders).
    Measuring against the actual SDF grid removes both the frame mix-up and the EDT
    resampling quantization that inflated boundary readings."""
    R = grid.shape[0]
    vox = 2.0 / (R - 1)
    to_i = lambda v: int(np.clip((v + 1) * 0.5 * (R - 1), 0, R - 1))
    out = []
    for p in pts_world:
        p = np.asarray(p)
        out.append(float(grid[to_i(p[2]), to_i(p[1]), to_i(p[0])]) / vox)
    return np.array(out)


def render(ax, verts, faces, title):
    if len(faces) == 0:
        ax.set_title(title + " (empty)", fontsize=8); return
    tris = verts[faces]
    fy = tris[:, :, 1].mean(1)
    col = plt.cm.viridis(0.15 + 0.7 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=col, edgecolors="none"))
    x, z, y = verts[:, 0], verts[:, 2], verts[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    ax.view_init(elev=35, azim=-50); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=9)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = RecipeInferenceEngine(); r = Refiner(eng, res=64)

    fp = U(20, 16, 6, 8)                              # a U/courtyard footprint -- non-convex
    H, style, cls = 11.0, "modern", "RESIDENTIAL"
    params = eng.sample_params(fp, H, cls, style, seed=4)
    grid, c, s, _hn = r.building_volume(fp, style, params, H, res=64)
    occ = grid <= 0
    print(f"[base] U-shaped courtyard footprint, occ={float(occ.mean()):.3f} center={c} scale={s:.2f}")

    existing_ops = ld.propose_detail_ops(grid, building_class=cls, device=dev, temperature=0.7,
                                         max_ops=16, seed=9)
    print(f"existing ops: {len(existing_ops)} ->", [o.get("det") for o in existing_ops])

    # place the moldy piece RIGHT AT THE COURTYARD NOTCH -- the hardest case: a point here is
    # geometrically close to BOTH an interior courtyard-facing wall and, potentially, a
    # farther true exterior wall. If snap logic is naive it'll prefer the (wrong) nearby
    # interior face.
    new_window = dict(kind="box", center=[0.05, -0.05, 0.15], size=[0.05, 0.07, 0.03],
                      mode="subtract", smooth=0.0, det="window", grp="gNew")
    out_ops, used = ld.integrate_new_part(grid, existing_ops, new_window, building_class=cls, device=dev)
    print(f"used={used}")

    # 1) QUANTITATIVE: sample the massing SDF at every op's final center — flag anything
    # deep inside solid mass. Wall ops legitimately straddle the surface (center on the
    # outermost occupied voxel ~= -1 vox), so the burial threshold is -2.5 voxels.
    centers = [np.asarray(o["center"], np.float32) for o in out_ops]
    sdfs = sdf_at(grid, centers)
    print("\nper-op surface distance (world SDF, voxels; ~-1 = on surface, very negative = BURIED):")
    worst = 0.0
    for o, d in zip(out_ops, sdfs):
        flag = "  <-- INTERIOR PULL?" if d < -2.5 else ""
        print(f"  {o.get('det','?'):8s} grp={o.get('grp','-'):8s} sdf={d:+6.1f} vox{flag}")
        worst = min(worst, d)
    print(f"\nworst (most negative) surface distance: {worst:+.1f} vox  "
          f"({'FAIL -- likely interior pull' if worst < -2.5 else 'PASS -- nothing buried'})")

    # 2) VISUAL: render base vs. after-integration, from an angle that shows the courtyard.
    def compose_grid(ops_):
        from scene.sdf_edit import EditableBuilding, EditOp
        from scene.sdf_primitives import sample_grid
        from refine import volume_to_sdf
        comp = EditableBuilding(volume_to_sdf(grid, dev), [EditOp.from_dict(o) for o in ops_]).composed()
        return sample_grid(comp, 64, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), device=dev).cpu().numpy()

    base_vol = compose_grid(existing_ops)
    after_vol = compose_grid(out_ops)
    fig = plt.figure(figsize=(8, 4))
    for i, (vol, title) in enumerate([(base_vol, "before (existing windows)"),
                                       (after_vol, "after integrate_new_part")]):
        try:
            v, f, *_ = marching_cubes(vol, level=0.0)
            v = (v / 63.0) * 2 - 1
        except Exception:
            v, f = np.zeros((0, 3)), np.zeros((0, 3), int)
        render(fig.add_subplot(1, 2, i + 1, projection="3d"), v, f, title)
    fig.suptitle("Interior-pull check: U-shaped courtyard footprint", fontsize=11)
    fig.tight_layout()
    out_p = OUT / "interior_pull_check.png"
    fig.savefig(out_p, dpi=120); plt.close(fig)
    print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
