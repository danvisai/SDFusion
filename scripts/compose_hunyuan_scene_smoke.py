"""Compose simplified Hunyuan building meshes into the demo town layout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import trimesh

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import place_mesh, synthetic_scene


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", default="outputs/hunyuan_retrieval_rank1_mini_simplified")
    ap.add_argument("--pattern", default="*.obj")
    ap.add_argument("--out", default="outputs/demo_town_hunyuan_rank1.obj")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    mesh_paths = sorted(Path(args.mesh_dir).glob(args.pattern))
    buildings = synthetic_scene()
    if len(mesh_paths) < len(buildings):
        raise SystemExit(f"Need {len(buildings)} meshes, found {len(mesh_paths)} in {args.mesh_dir}")

    placed = []
    log = []
    for building, mesh_path in zip(buildings, mesh_paths):
        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh.is_empty or len(mesh.faces) == 0:
            print(f"skip empty mesh: {mesh_path}")
            continue
        placed_mesh = place_mesh(mesh, building["polygon"], building["height"])
        placed.append(placed_mesh)
        log.append({
            "id": building["id"],
            "class": building["class"],
            "source_mesh": str(mesh_path),
            "verts": int(len(placed_mesh.vertices)),
            "faces": int(len(placed_mesh.faces)),
        })
        print(f"{building['id']:3s} <- {mesh_path.name} ({len(placed_mesh.faces):,} faces)")

    if not placed:
        raise SystemExit("No meshes placed.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.util.concatenate(placed)
    scene.export(out_path)
    with out_path.with_suffix(".log.json").open("w") as f:
        json.dump(log, f, indent=2)
    print(f"wrote {out_path} ({len(scene.vertices):,} verts, {len(scene.faces):,} faces)")


if __name__ == "__main__":
    main()
