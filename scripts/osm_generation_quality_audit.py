"""Audit generated OSM building meshes for collapse and weak outputs.

This pass does not modify the scene. It reads a pipeline log, inspects raw and
simplified/generated meshes, measures basic geometry health, and writes a CSV
plus an annotated sheet. The goal is to identify which Hunyuan outputs should
be regenerated, replaced by retrieved OBJ, or excluded from training data.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/osm_generation_quality_audit.py \
        --osm_json outputs/osm_pipeline_smoke/osm_input.json \
        --pipeline_log outputs/osm_pipeline_heightfix_12/osm_hunyuan_scene.log.json \
        --out_dir outputs/osm_pipeline_quality_12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import place_mesh, rasterize_polygon
from scripts.hunyuan_building_mesh_smoke import title_cell


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--pipeline_log", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_pipeline_quality")
    ap.add_argument("--cell_size", type=int, default=256)
    ap.add_argument("--min_raw_faces", type=int, default=100_000)
    ap.add_argument("--min_simplified_faces", type=int, default=10_000)
    ap.add_argument("--min_height_ratio", type=float, default=0.12)
    ap.add_argument("--min_vertical_extent_ratio", type=float, default=0.18)
    ap.add_argument("--max_flatness_ratio", type=float, default=8.0)
    ap.add_argument("--min_render_ink", type=float, default=0.015)
    return ap.parse_args()


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base / path
    return candidate if candidate.exists() else path


def mesh_metrics(mesh_path: Path) -> dict[str, float | int | str]:
    if not mesh_path.exists():
        return {
            "mesh_exists": 0,
            "verts": 0,
            "faces": 0,
            "extent_x": 0.0,
            "extent_y": 0.0,
            "extent_z": 0.0,
            "xz_max": 0.0,
            "height_ratio": 0.0,
            "flatness_ratio": 999.0,
            "is_watertight": 0,
            "components": 0,
        }
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh is None or len(mesh.vertices) == 0:
        return {
            "mesh_exists": 0,
            "verts": 0,
            "faces": 0,
            "extent_x": 0.0,
            "extent_y": 0.0,
            "extent_z": 0.0,
            "xz_max": 0.0,
            "height_ratio": 0.0,
            "flatness_ratio": 999.0,
            "is_watertight": 0,
            "components": 0,
        }
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    xz_max = float(max(ext[0], ext[2], 1e-9))
    height_ratio = float(ext[1] / xz_max)
    flatness_ratio = float(xz_max / max(ext[1], 1e-9))
    try:
        components = len(mesh.split(only_watertight=False))
    except BaseException:
        components = 0
    return {
        "mesh_exists": 1,
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extent_x": float(ext[0]),
        "extent_y": float(ext[1]),
        "extent_z": float(ext[2]),
        "xz_max": xz_max,
        "height_ratio": height_ratio,
        "flatness_ratio": flatness_ratio,
        "is_watertight": int(bool(mesh.is_watertight)),
        "components": int(components),
    }


def mesh_object_metrics(mesh: trimesh.Trimesh) -> dict[str, float | int | str]:
    if mesh is None or len(mesh.vertices) == 0:
        return {
            "mesh_exists": 0,
            "verts": 0,
            "faces": 0,
            "extent_x": 0.0,
            "extent_y": 0.0,
            "extent_z": 0.0,
            "xz_max": 0.0,
            "height_ratio": 0.0,
            "flatness_ratio": 999.0,
            "is_watertight": 0,
            "components": 0,
        }
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    xz_max = float(max(ext[0], ext[2], 1e-9))
    height_ratio = float(ext[1] / xz_max)
    flatness_ratio = float(xz_max / max(ext[1], 1e-9))
    try:
        components = len(mesh.split(only_watertight=False))
    except BaseException:
        components = 0
    return {
        "mesh_exists": 1,
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extent_x": float(ext[0]),
        "extent_y": float(ext[1]),
        "extent_z": float(ext[2]),
        "xz_max": xz_max,
        "height_ratio": height_ratio,
        "flatness_ratio": flatness_ratio,
        "is_watertight": int(bool(mesh.is_watertight)),
        "components": int(components),
    }


def render_ink(render_path: Path) -> float:
    if not render_path.exists():
        return 0.0
    img = Image.open(render_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return float((arr < 0.96).mean())


def footprint_cell(mask: np.ndarray, title: str, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    mask_img = Image.fromarray(((mask > 0) * 255).astype(np.uint8), "L").resize((size, size), Image.Resampling.NEAREST)
    tint = Image.new("RGB", (size, size), (36, 112, 172))
    img.paste(tint, mask=mask_img)
    ImageDraw.Draw(img).rectangle((0, 0, size - 1, size - 1), outline=(205, 205, 205))
    return title_cell(img, title, size)


def image_cell(path_value: str, title: str, size: int) -> Image.Image:
    path = Path(path_value)
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (size, size), "white")
        ImageDraw.Draw(img).text((8, 8), "missing", fill=(150, 0, 0))
    return title_cell(img, title, size)


def status_cell(row: dict[str, object], size: int) -> Image.Image:
    flags = str(row["flags"])
    status = str(row["status"])
    fill = (222, 245, 229) if status == "pass" else (255, 235, 210) if status == "warn" else (255, 218, 218)
    img = Image.new("RGB", (size, size), fill)
    draw = ImageDraw.Draw(img)
    lines = [
        f"status: {status}",
        f"raw F: {row['raw_faces']}",
        f"simp F: {row['simplified_faces']}",
        f"H/XZ raw: {float(row['raw_height_ratio']):.2f}",
        f"H/XZ placed: {float(row['placed_height_ratio']):.2f}",
        f"ink raw: {float(row['raw_render_ink']):.3f}",
        f"ink placed: {float(row['placed_render_ink']):.3f}",
    ]
    if flags:
        lines.append("flags:")
        lines.extend(flags.split("|")[:5])
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 18
    return title_cell(img, f"quality {row['index']}", size)


def classify(row: dict[str, object], args: argparse.Namespace) -> tuple[str, str]:
    flags = []
    if int(row["raw_faces"]) < args.min_raw_faces:
        flags.append("low_raw_faces")
    if int(row["simplified_faces"]) < args.min_simplified_faces:
        flags.append("low_simplified_faces")
    if float(row["raw_height_ratio"]) < args.min_height_ratio:
        flags.append("raw_flat")
    if float(row["placed_height_ratio"]) < args.min_vertical_extent_ratio:
        flags.append("placed_flat")
    if float(row["raw_flatness_ratio"]) > args.max_flatness_ratio:
        flags.append("raw_high_flatness")
    if float(row["placed_flatness_ratio"]) > args.max_flatness_ratio:
        flags.append("placed_high_flatness")
    if float(row["raw_render_ink"]) < args.min_render_ink:
        flags.append("raw_low_render_ink")
    if float(row["placed_render_ink"]) < args.min_render_ink:
        flags.append("placed_low_render_ink")
    if int(row["raw_components"]) > 64:
        flags.append("many_raw_components")
    if int(row["simplified_components"]) > 64:
        flags.append("many_simplified_components")
    status = "pass" if not flags else "warn"
    hard_flags = {"raw_flat", "placed_flat", "raw_low_render_ink", "placed_low_render_ink"}
    if hard_flags.intersection(flags):
        status = "fail"
    return status, "|".join(flags)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path.cwd()

    payload = json.load(open(args.osm_json))
    log_rows = json.load(open(args.pipeline_log))
    by_id = {str(b["id"]): b for b in payload.get("buildings", [])}

    rows = []
    sheet_rows = []
    for row in log_rows:
        raw_mesh = resolve_path(str(row["hunyuan_raw_glb"]), repo)
        simplified_mesh = resolve_path(str(row["simplified_obj"]), repo)
        placed_render = resolve_path(str(row["placed_render_png"]), repo)
        raw_render = resolve_path(str(row["hunyuan_raw_render_png"]), repo)
        raw = mesh_metrics(raw_mesh)
        simplified = mesh_metrics(simplified_mesh)
        building = by_id[str(row["osm_id"])]
        simplified_obj = trimesh.load(simplified_mesh, force="mesh", process=False)
        placed_scene = mesh_object_metrics(place_mesh(
            simplified_obj,
            building["polygon"],
            float(row["height_m"]),
        ))
        audit = {
            "index": int(row["index"]),
            "osm_id": row["osm_id"],
            "class": row["class"],
            "retrieved_id": row["retrieved_id"],
            "height_m": row["height_m"],
            "area_m2": row.get("area_m2", ""),
            "raw_mesh": str(raw_mesh),
            "simplified_obj": str(simplified_mesh),
            "raw_faces": raw["faces"],
            "raw_verts": raw["verts"],
            "raw_height_ratio": raw["height_ratio"],
            "raw_flatness_ratio": raw["flatness_ratio"],
            "raw_components": raw["components"],
            "raw_watertight": raw["is_watertight"],
            "simplified_faces": simplified["faces"],
            "simplified_verts": simplified["verts"],
            "simplified_height_ratio": simplified["height_ratio"],
            "simplified_flatness_ratio": simplified["flatness_ratio"],
            "simplified_components": simplified["components"],
            "simplified_watertight": simplified["is_watertight"],
            "placed_height_ratio": placed_scene["height_ratio"],
            "placed_flatness_ratio": placed_scene["flatness_ratio"],
            "raw_render_ink": render_ink(raw_render),
            "placed_render_ink": render_ink(placed_render),
            "raw_render_png": str(raw_render),
            "placed_render_png": str(placed_render),
        }
        status, flags = classify(audit, args)
        audit["status"] = status
        audit["flags"] = flags
        rows.append(audit)

        mask = rasterize_polygon(building["polygon"], res=64)
        sheet_rows.append([
            footprint_cell(mask, f"OSM {audit['index']}", args.cell_size),
            image_cell(str(raw_render), "generated", args.cell_size),
            image_cell(str(placed_render), "placed", args.cell_size),
            status_cell(audit, args.cell_size),
        ])
        print(f"{audit['index']:02d} {audit['osm_id']} {status} {flags}", flush=True)

    csv_path = out_dir / "generation_quality_audit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cell_w = args.cell_size
    cell_h = args.cell_size + 28
    sheet = Image.new("RGB", (4 * cell_w, len(sheet_rows) * cell_h), "white")
    for r, cells in enumerate(sheet_rows):
        for c, cell in enumerate(cells):
            sheet.paste(cell, (c * cell_w, r * cell_h))
    sheet_path = out_dir / "generation_quality_audit_sheet.png"
    sheet.save(sheet_path, optimize=True)

    summary = {
        "count": len(rows),
        "pass": sum(1 for r in rows if r["status"] == "pass"),
        "warn": sum(1 for r in rows if r["status"] == "warn"),
        "fail": sum(1 for r in rows if r["status"] == "fail"),
    }
    summary_path = out_dir / "generation_quality_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[quality] csv:     {csv_path}", flush=True)
    print(f"[quality] sheet:   {sheet_path}", flush=True)
    print(f"[quality] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
