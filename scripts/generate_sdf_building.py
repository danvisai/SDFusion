"""Generate a styled procedural building from a footprint via SDF recipes.

Standalone CLI; complements scripts/osm_hunyuan_pipeline_smoke.py. Given a
2D polygon footprint (CSV / JSON / npy), a style, and a target height, this
samples the recipe's SDF on a grid, marching-cubes it, and writes an OBJ.

Examples:
    # From an inline rectangle (10m x 6m, height 5m, victorian style)
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/generate_sdf_building.py \
        --style victorian --height 5.0 \
        --polygon '[[-5,-3],[5,-3],[5,3],[-5,3]]' \
        --out /tmp/sdf_victorian.obj

    # From a polygon JSON (first building in an OSM extract)
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/generate_sdf_building.py \
        --style colonial --height 7.0 \
        --polygon_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
        --polygon_index 0 \
        --out /tmp/colonial_east0.obj
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.sdf_primitives import (
    grid_to_mesh, polygon_bbox_with_pad, sample_grid,
)
from scene.sdf_recipes import STYLES, build_styled_sdf


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", required=True, choices=list(STYLES))
    ap.add_argument("--height", type=float, default=6.0,
                    help="Target body height in world units (m).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output OBJ path.")
    ap.add_argument("--polygon", help="Inline polygon JSON: [[x,z], ...]")
    ap.add_argument("--polygon_json",
                    help="OSM-style JSON; uses building[polygon_index]['polygon']")
    ap.add_argument("--polygon_index", type=int, default=0)
    ap.add_argument("--polygon_npy",
                    help=".npy file with (P, 2) polygon vertices in XZ.")
    ap.add_argument("--resolution", type=int, default=128,
                    help="Voxel grid resolution per side for marching cubes.")
    ap.add_argument("--pad", type=float, default=0.10,
                    help="Fractional padding around polygon bbox for sampling.")
    ap.add_argument("--preview_png", help="Optional: save a quick render here.")
    ap.add_argument("--save_sdf", help="Optional: dump (D,H,W) SDF tensor as .npy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_polygon(args) -> np.ndarray:
    if args.polygon:
        return np.asarray(json.loads(args.polygon), dtype=np.float32)
    if args.polygon_npy:
        return np.load(args.polygon_npy).astype(np.float32)
    if args.polygon_json:
        with open(args.polygon_json) as f:
            payload = json.load(f)
        buildings = payload.get("buildings", [])
        if not buildings:
            raise ValueError(f"No buildings in {args.polygon_json}")
        b = buildings[args.polygon_index]
        return np.asarray(b["polygon"], dtype=np.float32)
    raise SystemExit("Provide one of --polygon, --polygon_json, --polygon_npy")


def main() -> None:
    args = parse_args()
    poly = load_polygon(args)
    print(f"[sdfgen] style={args.style} height={args.height} "
          f"polygon shape={poly.shape} res={args.resolution}")

    t0 = time.time()
    sdf = build_styled_sdf(args.style, poly, args.height, seed=args.seed)
    # Pad covers roof + ornaments which can extend ~2x building height.
    bbox = polygon_bbox_with_pad(poly, args.height * 2.5, pad=args.pad)
    print(f"[sdfgen] bbox={bbox}")

    device = torch.device(args.device)
    grid = sample_grid(sdf, args.resolution, bbox, device=str(device))
    print(f"[sdfgen] grid sampled  ({time.time()-t0:.1f}s)  "
          f"range=[{grid.min().item():.3f}, {grid.max().item():.3f}]")

    mesh = grid_to_mesh(grid, bbox)
    if mesh is None:
        raise SystemExit("[sdfgen] marching cubes returned no surface (level=0 not crossed).")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    print(f"[sdfgen] mesh V={len(mesh.vertices):,} F={len(mesh.faces):,}  -> {out_path}")

    if args.save_sdf:
        np.save(args.save_sdf, grid.cpu().numpy())
        print(f"[sdfgen] sdf -> {args.save_sdf}")

    if args.preview_png:
        from scripts.render_buildingnet_objfiles import make_renderer, render_one
        renderer = make_renderer(device, image_size=512, scale=0.35)
        rgb = render_one(mesh, renderer, device)
        Image.fromarray(rgb).save(args.preview_png, optimize=True)
        print(f"[sdfgen] preview -> {args.preview_png}")

    print(f"[sdfgen] done  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
