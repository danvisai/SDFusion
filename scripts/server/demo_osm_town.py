"""End-to-end capstone: REAL OSM tile -> generated 3D town via the B+.6 head.

Loads a buildings JSON from scene/extract_osm.py (real OSM footprints in local meters +
class + height), assigns a style per class, generates each building with the B+.6
diffusion head, places it at its true OSM centroid, composes one scene, and renders
top-down (with roads) + isometric.

This is the literal project deliverable: OSM footprints -> navigable generated town,
"truly generative from symbolic input, no reference image".

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/server/demo_osm_town.py --osm /tmp/lafayette_tile.json
"""

from __future__ import annotations

import argparse
import io
import json
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
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine
from models.networks import recipe_param_space as ps

OUT = REPO / "outputs/recipe_param_diffusion_b6"

# Class -> candidate styles (cycled for variety). Symbolic style assignment; in the
# product this is a user choice / learned style hint.
STYLE_FOR_CLASS = {
    "RESIDENTIAL": ["craftsman", "colonial", "victorian", "modern"],
    "COMMERCIAL": ["modern", "contemporary", "industrial"],
    "PUBLIC": ["public_civic", "modern"],
    "RELIGIOUS": ["victorian", "public_civic"],
}


def top_class(cls: str) -> str:
    for t in ps.CLASSES:
        if cls.upper().startswith(t):
            return t
    return "RESIDENTIAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", type=Path, default="/tmp/lafayette_tile.json")
    ap.add_argument("--max_buildings", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out_png", type=Path, default=OUT / "osm_town_demo.png")
    ap.add_argument("--out_glb", type=Path, default=OUT / "osm_town_demo.glb")
    args = ap.parse_args()

    raw = json.load(open(args.osm))
    buildings = raw if isinstance(raw, list) else raw.get("buildings", [])
    roads = raw.get("roads", []) if isinstance(raw, dict) else []
    buildings = [b for b in buildings if len(b.get("polygon", [])) >= 3][:args.max_buildings]
    print(f"[osm] {len(buildings)} buildings, {len(roads)} roads from {args.osm}")

    eng = RecipeInferenceEngine(grid_res=56)
    style_counters = {}
    parts, foot_polys, foot_styles = [], [], []
    for i, b in enumerate(buildings):
        poly = np.asarray(b["polygon"], dtype=np.float32)
        poly = poly[:-1] if len(poly) > 1 and np.allclose(poly[0], poly[-1]) else poly
        tc = top_class(b.get("class", "RESIDENTIAL"))
        cand = STYLE_FOR_CLASS.get(tc, ["modern"])
        k = style_counters.get(tc, 0); style = cand[k % len(cand)]; style_counters[tc] = k + 1
        height = float(b.get("height", 10.0))

        res = eng.generate_building(poly, tc, height, style, seed=i, guidance=args.guidance)
        if not res.glb:
            print(f"  [skip] {b.get('id')} empty"); continue
        m = trimesh.load(io.BytesIO(res.glb), file_type="glb").to_geometry()
        # generate_building centred the polygon; place back at its true OSM centroid.
        cx, cz = res.position_xz
        m.apply_translation([cx, 0, cz])
        parts.append(m)
        foot_polys.append(poly); foot_styles.append(style)
        print(f"  {b.get('id'):>10} {tc:11s} -> {style:13s} h={height:.1f}m "
              f"area={b.get('area',0):.0f}m2 verts={res.n_vertices}")

    scene = trimesh.util.concatenate(parts)
    scene.export(args.out_glb)
    print(f"[save] {args.out_glb} ({len(scene.vertices)} verts, {len(scene.faces)} faces)")

    # --- render ---
    fig = plt.figure(figsize=(16, 7))
    V, F = scene.vertices, scene.faces
    tris = V[F]; fy = tris[:, :, 1].mean(axis=1)
    colors = plt.cm.terrain(0.25 + 0.5 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=colors, edgecolors="none"))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, y.max())
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=30, azim=-62); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("generated town (isometric)", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    for seg in roads:
        pts = np.asarray(seg.get("polyline", seg.get("coords", seg)) if isinstance(seg, (list, dict)) else seg)
        if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 2:
            ax2.plot(pts[:, 0], pts[:, 1], color="0.6", lw=2, zorder=0)
    cmap = {s: plt.cm.tab10(i) for i, s in enumerate(ps.STYLES)}
    ax2.add_collection(PolyCollection(foot_polys, facecolors=[cmap[s] for s in foot_styles],
                                      edgecolors="k", linewidths=0.5, alpha=0.85))
    allp = np.concatenate(foot_polys)
    ax2.set_xlim(allp[:, 0].min() - 10, allp[:, 0].max() + 10)
    ax2.set_ylim(allp[:, 1].min() - 10, allp[:, 1].max() + 10)
    ax2.set_aspect("equal"); ax2.set_title("OSM footprints (meters) colored by assigned style", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[s]) for s in sorted(set(foot_styles))]
    ax2.legend(handles, sorted(set(foot_styles)), fontsize=7, loc="upper right")
    fig.suptitle(f"REAL OSM tile -> B+.6 generated town ({len(parts)} buildings)", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=105); plt.close(fig)
    print(f"[save] {args.out_png}")


if __name__ == "__main__":
    main()
