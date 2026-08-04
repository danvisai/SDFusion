"""Capstone: real Munich OSM tile -> generated town with EVERYTHING wired in.

Per building: B+.6 recipe base (trained head) + procedural facade detail (data-grounded
per-style prior) + rich landmarks sampled by REAL per-class occurrence (dome/tower/steps
from BuildingNet labels) + roof from real OSM `roof:shape` (gabled/hipped/pyramidal/dome).

Munich because ~70% of its buildings carry roof:shape + building:levels tags.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/server/demo_munich_town.py --max_buildings 70
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import trimesh

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine
from scene.sdf_edit import recipe_base_sdf
from scene import sdf_detail as det
from scene.sdf_primitives import sample_grid, grid_to_mesh, sdf_gable_roof, sdf_hip_roof, sdf_union

OUT = REPO / "outputs/recipe_param_diffusion_b6"

STYLE_FOR_CLASS = {
    "RELIGIOUS": ["victorian", "public_civic"],
    "PUBLIC": ["public_civic", "modern"],
    "COMMERCIAL": ["modern", "contemporary", "industrial"],
    "RESIDENTIAL": ["colonial", "craftsman", "victorian", "modern"],
}


def osm_class(btag: str) -> str:
    b = (btag or "").lower()
    if any(k in b for k in ["church", "cathedral", "chapel", "mosque", "temple", "synagogue", "basilica", "monaster"]):
        return "RELIGIOUS"
    if any(k in b for k in ["commercial", "office", "retail", "hotel", "industrial", "warehouse", "supermarket"]):
        return "COMMERCIAL"
    if any(k in b for k in ["civic", "public", "government", "museum", "hospital", "school", "university", "hall", "train"]):
        return "PUBLIC"
    return "RESIDENTIAL"


def query_munich(bbox, max_b):
    import osmnx as ox
    n, s, e, w = bbox
    try:
        gdf = ox.features_from_bbox(bbox=(w, s, e, n), tags={"building": True})
    except Exception:
        gdf = ox.features_from_bbox(n, s, e, w, tags={"building": True})
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = ox.projection.project_gdf(gdf)
    buildings = []
    for _, r in gdf.iterrows():
        g = r.geometry
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        ext = np.array(g.exterior.coords)[:, :2]
        if len(ext) < 4 or g.area < 25:
            continue
        cen = ext.mean(0)

        def _num(v):
            import math
            try:
                f = float(v)
                return f if math.isfinite(f) else None
            except (TypeError, ValueError):
                try:
                    return float(str(v).split()[0])
                except Exception:
                    return None
        h = _num(r.get("height")); lv = _num(r.get("building:levels"))
        height = h if h else (lv * 3.3 if lv else 9.0)
        buildings.append({"poly": (ext - cen).astype(np.float32), "centroid": cen,
                          "height": float(np.clip(height, 4, 90)),
                          "cls": osm_class(r.get("building") if isinstance(r.get("building"), str) else ""),
                          "roof": r.get("roof:shape") if isinstance(r.get("roof:shape"), str) else None,
                          "area": float(g.area)})
    buildings.sort(key=lambda b: -b["area"])
    return buildings[:max_b]


def apply_roof(sdf, poly, height, shape):
    if not shape or shape == "flat":
        return sdf
    x0, z0 = poly[:, 0].min(), poly[:, 1].min(); x1, z1 = poly[:, 0].max(), poly[:, 1].max()
    w, d = float(x1 - x0), float(z1 - z0); cx, cz = float((x0 + x1) / 2), float((z0 + z1) / 2)
    rh = 0.35 * min(w, d)
    s = str(shape).lower()
    if any(k in s for k in ["gabl", "gambrel", "saltbox", "pitch"]):
        roof = sdf_gable_roof(w, d, height, rh, center_xz=(cx, cz))
    elif any(k in s for k in ["hip", "pyramid"]):
        roof = sdf_hip_roof(w, d, height, rh * (1.4 if "pyram" in s else 1.0), center_xz=(cx, cz))
    elif any(k in s for k in ["dome", "onion", "round"]):
        roof = det.sdf_dome((cx, height, cz), min(w, d) * 0.5, min(w, d) * 0.4)
    else:
        return sdf
    return sdf_union(sdf, roof)


def compose_building(eng, b, seed, dev):
    cls = b["cls"]; rng = np.random.default_rng(seed)
    style = STYLE_FOR_CLASS[cls][seed % len(STYLE_FOR_CLASS[cls])]
    fp, H = b["poly"], b["height"]
    params = eng.sample_params(fp, H, cls, style, seed=seed)
    sdf = recipe_base_sdf(style, params, fp, H, device=dev)
    # facade detail with window density GROUNDED in the real per-class BuildingNet glazing
    dparams = det.ground_glazing(det.vector_to_params(det.sample_detail_vector(style, rng)), cls)
    sdf = det.add_facade_detail(sdf, fp, H, dparams)
    # roof: real OSM roof:shape if present, else BuildingNet class pitched-probability
    roof_shape = det.sample_roof_shape(cls, b["roof"], rng)
    sdf = apply_roof(sdf, fp, H, roof_shape)
    lm = det.sample_landmarks(cls, rng)
    if lm["dome"] or lm["n_towers"] or lm["steps"]:
        sdf = det.add_landmarks(sdf, fp, H, dome=lm["dome"], n_towers=lm["n_towers"], steps=lm["steps"])
    return sdf, style, lm


def main():
    ap = argparse.ArgumentParser()
    # Munich Altstadt core: Marienplatz + Frauenkirche + St. Peter (churches -> towers/domes)
    ap.add_argument("--bbox", nargs=4, type=float, default=[48.1400, 48.1362, 11.5785, 11.5725])
    ap.add_argument("--max_buildings", type=int, default=70)
    ap.add_argument("--res", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    buildings = query_munich(args.bbox, args.max_buildings)
    nroof = sum(1 for b in buildings if b["roof"])
    from collections import Counter
    print(f"[munich] {len(buildings)} buildings | {nroof} with roof:shape "
          f"({Counter(b['roof'] for b in buildings if b['roof'])}) | classes {Counter(b['cls'] for b in buildings)}")

    eng = RecipeInferenceEngine()
    parts, foot, lmcount = [], [], Counter()
    for i, b in enumerate(buildings):
        sdf, style, lm = compose_building(eng, b, i, dev)
        fp = b["poly"]; H = b["height"]
        pad = 2.0
        bbox = (fp[:, 0].min()-pad, 0.0, fp[:, 1].min()-pad, fp[:, 0].max()+pad,
                H * (1.9 if lm["n_towers"] else 1.5), fp[:, 1].max()+pad)
        mesh = grid_to_mesh(sample_grid(sdf, args.res, bbox, device=dev), bbox, 0.0)
        if mesh is None or len(mesh.faces) == 0:
            continue
        mesh.apply_translation([b["centroid"][0], 0, b["centroid"][1]])
        parts.append(mesh); foot.append(b["centroid"] + fp[:, :2])
        for k, v in lm.items():
            lmcount[k] += (v if isinstance(v, int) else int(bool(v)))
    print(f"[compose] {len(parts)} meshes | landmarks placed: {dict(lmcount)}")

    scene = trimesh.util.concatenate(parts)
    scene.export(OUT / "munich_town.glb")
    print(f"[save] {OUT/'munich_town.glb'} ({len(scene.vertices)} verts, {len(scene.faces)} faces)")

    # render iso + top-down
    fig = plt.figure(figsize=(17, 8))
    V, F = scene.vertices, scene.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1); col = plt.cm.terrain(0.25 + 0.5 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=col, edgecolors="none"))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, y.max())
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=24, azim=-60); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("generated Munich town (B+.6 + facade detail + landmarks + OSM roofs)", fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.add_collection(PolyCollection(foot, facecolors="0.7", edgecolors="k", linewidths=0.3))
    allp = np.concatenate(foot)
    ax2.set_xlim(allp[:, 0].min(), allp[:, 0].max()); ax2.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax2.set_aspect("equal"); ax2.set_title("OSM footprints (m)", fontsize=9); ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle(f"REAL Munich OSM tile -> generated town ({len(parts)} buildings, {nroof} real roof types)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "munich_town.png", dpi=110); plt.close(fig)
    print(f"[save] {OUT/'munich_town.png'}")


if __name__ == "__main__":
    main()
