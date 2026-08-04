"""Validate the end-to-end OSM->town goal: compose many generated buildings into one scene.

Uses REAL, varied BuildingNet footprints (from the B+.7 fits) as the "tile" input —
each footprint's own best-fit style — generates a building per footprint via the B+.6
head, lays them out on a grid (stand-in for OSM placement), composes into one trimesh
scene, exports .glb/.obj, and renders top-down + isometric previews.

This proves multiple generated buildings compose into a coherent multi-building scene
(the project deliverable), and exercises the recipes on non-rectangular real footprints.

Output: outputs/recipe_param_diffusion_b6/scene_demo.{glb,png}
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine
from models.networks import recipe_param_space as ps

OUT = REPO / "outputs/recipe_param_diffusion_b6"
FITS = REPO / "outputs/fit_recipes_buildingnet/best_params.npz"


def pick_footprints(n_per_style=2, min_iou=0.6):
    """A diverse set of real footprints: up to n_per_style good fits per style, in meters.

    The B+.7 polygons are in normalized Frame-N (~[-1,1]); scale to a plausible metric
    footprint (~15 m span) so the generated buildings are realistically sized.
    """
    d = np.load(FITS, allow_pickle=True)["fits"].item()
    by_style = {}
    for aid, v in d.items():
        if v["iou"] < min_iou:
            continue
        by_style.setdefault(v["style"], []).append((v["iou"], aid, v))
    picks = []
    for style, lst in by_style.items():
        lst.sort(reverse=True)
        for _, aid, v in lst[:n_per_style]:
            poly = np.asarray(v["polygon"], np.float32)
            poly = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
            span = max(np.ptp(poly[:, 0]), np.ptp(poly[:, 1]), 1e-3)
            poly_m = poly * (15.0 / span)               # ~15 m across
            bbox = np.asarray(v["bbox"], np.float32)
            height_m = float((bbox[4] - bbox[1]) * (15.0 / span))
            picks.append({"id": aid, "style": style, "poly": poly_m,
                          "height": max(height_m, 4.0), "cls": aid[:11]})
    return picks


def render_scene(scene_mesh, picks_meta, out_png):
    fig = plt.figure(figsize=(16, 7))
    V, F = scene_mesh.vertices, scene_mesh.faces
    tris = V[F]
    fy = tris[:, :, 1].mean(axis=1)
    colors = plt.cm.terrain(0.25 + 0.5 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=colors,
                                         edgecolors="none"))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, y.max())
    try:
        ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception:
        pass
    ax.view_init(elev=28, azim=-60); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("isometric", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.add_collection(plt.matplotlib.collections.PolyCollection(
        [t[:, [0, 2]] for t in tris], facecolors=colors, edgecolors="none"))
    ax2.set_xlim(x.min(), x.max()); ax2.set_ylim(z.min(), z.max()); ax2.set_aspect("equal")
    ax2.set_title("top-down", fontsize=9); ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle(f"B+.6 generated town — {len(picks_meta)} buildings from real BuildingNet "
                 f"footprints", fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)


def main():
    eng = RecipeInferenceEngine(grid_res=56)
    picks = pick_footprints(n_per_style=2, min_iou=0.6)
    print(f"[scene] {len(picks)} buildings: {[p['style'] for p in picks]}")

    cols = int(np.ceil(np.sqrt(len(picks))))
    spacing = 26.0
    parts, meta = [], []
    for i, p in enumerate(picks):
        b = eng.generate_building(p["poly"], p["cls"], p["height"], p["style"], seed=i)
        if not b.glb:
            print(f"  [skip] {p['id']} ({p['style']}) empty mesh"); continue
        m = trimesh.load(io.BytesIO(b.glb), file_type="glb").to_geometry()
        gx, gy = i % cols, i // cols
        m.apply_translation([gx * spacing, 0, gy * spacing])
        parts.append(m)
        meta.append(p)
        print(f"  {p['style']:13s} {p['id'][:30]:30s} h={p['height']:.1f}m "
              f"verts={b.n_vertices} faces={b.n_faces}")

    scene = trimesh.util.concatenate(parts)
    glb_path = OUT / "scene_demo.glb"
    scene.export(glb_path)
    print(f"[save] {glb_path}  ({len(scene.vertices)} verts, {len(scene.faces)} faces, "
          f"watertight={scene.is_watertight})")
    render_scene(scene, meta, OUT / "scene_demo.png")
    print(f"[save] {OUT/'scene_demo.png'}")


if __name__ == "__main__":
    main()
