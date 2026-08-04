"""Re-place generated OSM meshes with an explicit height policy.

This avoids rerunning Hunyuan when the only issue is bad/incomplete OSM height
metadata. It reads an existing pipeline log, loads the already-simplified OBJ
meshes, infers better target heights, exports a recomposed scene, and writes a
new log compatible with scripts/osm_pipeline_map_choices.py.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/osm_recompose_height_policy.py \
        --osm_json outputs/osm_pipeline_smoke/osm_input.json \
        --pipeline_log outputs/osm_pipeline_rerank_smoke/osm_hunyuan_scene.log.json \
        --out_dir outputs/osm_pipeline_heightfix_smoke \
        --height_policy area_aware
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import place_mesh
from scripts.hunyuan_building_mesh_smoke import render_mesh_png


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--pipeline_log", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_pipeline_heightfix_smoke")
    ap.add_argument("--height_policy", choices=["osm", "area_aware"], default="area_aware")
    ap.add_argument("--image_size", type=int, default=384)
    return ap.parse_args()


def class_top(building_class: str) -> str:
    for top in ("RESIDENTIAL", "RELIGIOUS", "COMMERCIAL", "MILITARY", "PUBLIC"):
        if building_class.startswith(top):
            return top
    return "RESIDENTIAL"


def area_default_height(area_m2: float, building_class: str) -> float:
    top = class_top(building_class)
    if top == "RESIDENTIAL":
        if area_m2 < 150:
            return 7.0
        if area_m2 < 400:
            return 8.5
        if area_m2 < 900:
            return 10.5
        if area_m2 < 1800:
            return 14.0
        return 17.5
    if top == "COMMERCIAL":
        if area_m2 < 500:
            return 10.5
        if area_m2 < 1500:
            return 17.5
        if area_m2 < 3000:
            return 24.5
        return 31.5
    if top == "PUBLIC":
        if area_m2 < 500:
            return 10.5
        if area_m2 < 1500:
            return 14.0
        return 21.0
    if top == "RELIGIOUS":
        return max(14.0, min(28.0, 8.0 + 0.25 * np.sqrt(area_m2)))
    if top == "MILITARY":
        return max(10.5, min(24.5, 7.0 + 0.22 * np.sqrt(area_m2)))
    return 8.0


def infer_height(building: dict, policy: str) -> tuple[float, str]:
    osm_height = float(building.get("height", 0.0) or 0.0)
    if policy == "osm":
        return osm_height, "osm"
    area = float(building.get("area", 0.0) or 0.0)
    default = area_default_height(area, str(building.get("class", "")))
    if osm_height <= 0:
        return default, "area_aware_no_osm_height"
    if default > osm_height:
        return default, "area_aware_raised_default"
    return osm_height, "osm_kept"


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base / path
    return candidate if candidate.exists() else path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    payload = json.load(open(args.osm_json))
    source_rows = json.load(open(args.pipeline_log))
    by_id = {str(b["id"]): b for b in payload.get("buildings", [])}

    placed_meshes = []
    out_rows = []
    metrics_rows = []
    repo = Path.cwd()

    for row in source_rows:
        building = by_id[str(row["osm_id"])]
        inferred_height, source = infer_height(building, args.height_policy)
        mesh_path = resolve_path(str(row["simplified_obj"]), repo)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        placed = place_mesh(mesh, building["polygon"], inferred_height)
        placed_meshes.append(placed)

        stem = f"{int(row['index']):02d}_{row['osm_id']}_{row['retrieved_id']}"
        placed_render = render_dir / f"{stem}_placed_{args.height_policy}.png"
        render_mesh_png(placed, image_size=args.image_size).save(placed_render, optimize=True)

        updated = dict(row)
        updated["original_height_m"] = row.get("height_m")
        updated["height_m"] = f"{inferred_height:.2f}"
        updated["height_policy"] = args.height_policy
        updated["height_source"] = source
        updated["placed_render_png"] = str(placed_render)
        out_rows.append(updated)
        metrics_rows.append({
            "index": row["index"],
            "osm_id": row["osm_id"],
            "class": row["class"],
            "retrieved_id": row["retrieved_id"],
            "original_height_m": row.get("height_m"),
            "inferred_height_m": f"{inferred_height:.2f}",
            "height_source": source,
            "simplified_obj": row["simplified_obj"],
            "placed_render_png": str(placed_render),
        })
        print(
            f"{row['osm_id']}: {row.get('height_m')}m -> {inferred_height:.2f}m "
            f"({source})",
            flush=True,
        )

    if not placed_meshes:
        raise SystemExit("No meshes were placed.")

    scene = trimesh.util.concatenate(placed_meshes)
    scene_path = out_dir / "osm_hunyuan_scene.obj"
    scene.export(scene_path)

    log_path = out_dir / "osm_hunyuan_scene.log.json"
    with log_path.open("w") as f:
        json.dump(out_rows, f, indent=2)

    metrics_path = out_dir / "height_policy_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"[height] scene:   {scene_path}", flush=True)
    print(f"[height] log:     {log_path}", flush=True)
    print(f"[height] metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
