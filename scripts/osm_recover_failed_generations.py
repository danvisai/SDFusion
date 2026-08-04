"""Recover failed OSM building generations by retrying candidates or falling back.

This script is intentionally post-processing only. It reads a completed
height-fixed pipeline log plus its quality audit, keeps passing rows unchanged,
and repairs failed rows by:

1. trying the next ranked retrieval candidate through Hunyuan;
2. accepting the first retry that passes the same audit thresholds;
3. falling back to direct retrieved OBJ placement when retries fail.

The output log is compatible with the existing map-choice and quality-audit
scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import place_mesh
from scripts.hunyuan_building_mesh_smoke import load_pipeline, render_mesh_png, title_cell
from scripts.osm_generation_quality_audit import classify, mesh_metrics, mesh_object_metrics, render_ink
from scripts.osm_hunyuan_pipeline_smoke import render_retrieved_obj, safe_stem
from scripts.render_buildingnet_objfiles import make_renderer
from scripts.simplify_hunyuan_meshes import simplify_one


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--pipeline_log", required=True)
    ap.add_argument("--quality_csv", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_recovery")
    ap.add_argument("--recover_status", default="fail", help="Comma-separated statuses to recover.")
    ap.add_argument("--recover_flags", default="raw_flat,placed_flat,raw_low_render_ink,placed_low_render_ink")
    ap.add_argument("--model", choices=["mini", "full"], default="mini")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--octree_resolution", type=int, default=380)
    ap.add_argument("--num_chunks", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--target_faces", type=int, default=50_000)
    ap.add_argument("--max_retry_candidates", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--obj_dir", default="data/BuildingNet_dataset_v0_1/OBJ_MODELS")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min_raw_faces", type=int, default=100_000)
    ap.add_argument("--min_simplified_faces", type=int, default=10_000)
    ap.add_argument("--min_height_ratio", type=float, default=0.12)
    ap.add_argument("--min_vertical_extent_ratio", type=float, default=0.18)
    ap.add_argument("--max_flatness_ratio", type=float, default=8.0)
    ap.add_argument("--min_render_ink", type=float, default=0.015)
    return ap.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base / path
    return candidate if candidate.exists() else path


def should_recover(audit: dict[str, str], recover_status: set[str], recover_flags: set[str]) -> bool:
    status = audit.get("status", "")
    flags = {flag for flag in audit.get("flags", "").split("|") if flag}
    return status in recover_status or bool(flags.intersection(recover_flags))


def candidate_sort_key(row: dict) -> float:
    for key in ("quality_score", "rerank_score", "retrieval_score"):
        if key in row and row[key] not in ("", None):
            return float(row[key])
    return 0.0


def retry_candidates(row: dict, max_count: int) -> list[str]:
    current = str(row["retrieved_id"])
    candidates = [dict(c) for c in row.get("retrieval_candidates", [])]
    candidates.sort(key=candidate_sort_key, reverse=True)
    out = []
    for cand in candidates:
        cand_id = str(cand["candidate_id"])
        if cand_id == current or cand_id in out:
            continue
        out.append(cand_id)
        if len(out) >= max_count:
            break
    return out


def audit_candidate(
    args: argparse.Namespace,
    row: dict,
    building: dict,
    raw_mesh_path: Path,
    simplified_obj: Path,
    raw_render: Path,
    placed_render: Path,
) -> tuple[str, str, dict[str, object]]:
    raw = mesh_metrics(raw_mesh_path)
    simplified = mesh_metrics(simplified_obj)
    simplified_mesh = trimesh.load(simplified_obj, force="mesh", process=False)
    placed = place_mesh(simplified_mesh, building["polygon"], float(row["height_m"]))
    placed_scene = mesh_object_metrics(placed)
    audit = {
        "index": int(row["index"]),
        "osm_id": row["osm_id"],
        "class": row["class"],
        "retrieved_id": row["retrieved_id"],
        "height_m": row["height_m"],
        "area_m2": row.get("area_m2", ""),
        "raw_mesh": str(raw_mesh_path),
        "simplified_obj": str(simplified_obj),
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
    return status, flags, audit


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_missing_cell(label: str, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    draw.text((12, 12), label, fill=(150, 0, 0))
    return title_cell(img, label, size)


def make_recovery_sheet(rows: list[dict[str, object]], out_path: Path, size: int) -> None:
    if not rows:
        img = Image.new("RGB", (size * 2, size + 28), "white")
        ImageDraw.Draw(img).text((12, 12), "No rows required recovery.", fill=(0, 0, 0))
        img.save(out_path, optimize=True)
        return
    cell_w = size
    cell_h = size + 28
    sheet = Image.new("RGB", (4 * cell_w, len(rows) * cell_h), "white")
    for r, row in enumerate(rows):
        before = Path(str(row.get("before_render_png", "")))
        after = Path(str(row.get("after_render_png", "")))
        before_img = Image.open(before).convert("RGB") if before.exists() else render_missing_cell("missing before", size)
        after_img = Image.open(after).convert("RGB") if after.exists() else render_missing_cell("missing after", size)
        info = Image.new("RGB", (size, size), (232, 246, 236) if row.get("final_status") != "fail" else (255, 226, 226))
        draw = ImageDraw.Draw(info)
        lines = [
            str(row.get("osm_id", "")),
            f"before: {row.get('original_candidate', '')}",
            f"after: {row.get('final_candidate', '')}",
            f"method: {row.get('recovery_method', '')}",
            f"status: {row.get('original_status', '')} -> {row.get('final_status', '')}",
            f"flags: {row.get('final_flags', '') or '-'}",
        ]
        y = 10
        for line in lines:
            draw.text((10, y), line[:42], fill=(0, 0, 0))
            y += 20
        sheet.paste(title_cell(info, "recovery", size), (0, r * cell_h))
        sheet.paste(title_cell(before_img, "before placed", size), (cell_w, r * cell_h))
        sheet.paste(title_cell(after_img, "after placed", size), (2 * cell_w, r * cell_h))
        raw = Path(str(row.get("after_raw_render_png", "")))
        raw_img = Image.open(raw).convert("RGB") if raw.exists() else render_missing_cell("fallback/retry raw", size)
        sheet.paste(title_cell(raw_img, "after raw/input", size), (3 * cell_w, r * cell_h))
    sheet.save(out_path, optimize=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    retry_input_dir = out_dir / "retry_inputs"
    raw_dir = out_dir / "hunyuan_raw"
    simplified_dir = out_dir / "hunyuan_simplified"
    render_dir = out_dir / "renders"
    fallback_dir = out_dir / "fallback_meshes"
    for d in (retry_input_dir, raw_dir, simplified_dir, render_dir, fallback_dir):
        d.mkdir(parents=True, exist_ok=True)

    repo = Path.cwd()
    obj_dir = Path(args.obj_dir)
    payload = json.load(open(args.osm_json))
    source_rows = read_json(Path(args.pipeline_log))
    audit_rows = read_csv(Path(args.quality_csv))
    by_id = {str(b["id"]): b for b in payload.get("buildings", [])}
    audit_by_id = {str(row["osm_id"]): row for row in audit_rows}
    recover_status = {v.strip() for v in args.recover_status.split(",") if v.strip()}
    recover_flags = {v.strip() for v in args.recover_flags.split(",") if v.strip()}

    targets = [row for row in source_rows if should_recover(audit_by_id.get(str(row["osm_id"]), {}), recover_status, recover_flags)]
    print(f"[recover] rows={len(source_rows)} targets={len(targets)}", flush=True)

    device = torch.device(args.device)
    renderer = None
    pipe = None
    if targets:
        renderer = make_renderer(device, image_size=args.image_size)
        print(f"[recover] loading Hunyuan {args.model} on {device}", flush=True)
        pipe = load_pipeline(args.model)
        print("[recover] Hunyuan ready", flush=True)

    out_rows = []
    recovery_rows = []
    placed_meshes = []

    for row in source_rows:
        osm_id = str(row["osm_id"])
        building = by_id[osm_id]
        original_audit = audit_by_id.get(osm_id, {})
        new_row = dict(row)
        new_row["recovery_method"] = "none"
        new_row["recovery_from_candidate"] = row["retrieved_id"]
        new_row["recovery_original_status"] = original_audit.get("status", "")
        new_row["recovery_original_flags"] = original_audit.get("flags", "")

        if should_recover(original_audit, recover_status, recover_flags):
            accepted = False
            attempts = retry_candidates(row, args.max_retry_candidates)
            print(f"[recover] {osm_id} retry candidates: {', '.join(attempts) or 'none'}", flush=True)
            for attempt_i, candidate_id in enumerate(attempts, start=1):
                assert renderer is not None and pipe is not None
                obj_path = obj_dir / f"{candidate_id}.obj"
                if not obj_path.exists():
                    continue
                stem = f"{int(row['index']):02d}_{safe_stem(osm_id)}_{candidate_id}_retry{attempt_i}"
                input_image = render_retrieved_obj(obj_path, renderer, device)
                input_path = retry_input_dir / f"{stem}_retrieved_input.png"
                input_image.save(input_path, optimize=True)

                t0 = time.time()
                mesh = pipe(
                    image=input_image.convert("RGBA"),
                    num_inference_steps=args.steps,
                    octree_resolution=args.octree_resolution,
                    num_chunks=args.num_chunks,
                    generator=torch.manual_seed(args.seed + int(row["index"]) * 100 + attempt_i),
                    output_type="trimesh",
                )[0]
                seconds = time.time() - t0

                raw_glb = raw_dir / f"{stem}.glb"
                raw_render = render_dir / f"{stem}_hunyuan_raw.png"
                simplified_obj = simplified_dir / f"{stem}_simplified.obj"
                placed_render = render_dir / f"{stem}_placed.png"
                mesh.export(raw_glb)
                render_mesh_png(mesh, image_size=args.image_size).save(raw_render, optimize=True)
                simp_row = simplify_one(raw_glb, simplified_obj, args.target_faces, args.target_faces)
                simplified_mesh = trimesh.load(simplified_obj, force="mesh", process=False)
                placed = place_mesh(simplified_mesh, building["polygon"], float(row["height_m"]))
                render_mesh_png(placed, image_size=args.image_size).save(placed_render, optimize=True)
                candidate_log = dict(row)
                candidate_log["retrieved_id"] = candidate_id
                status, flags, _audit = audit_candidate(args, candidate_log, building, raw_glb, simplified_obj, raw_render, placed_render)
                print(f"[recover] {osm_id} retry {candidate_id}: {status} {flags}", flush=True)
                if status != "fail":
                    new_row.update({
                        "retrieved_id": candidate_id,
                        "retrieved_input_png": str(input_path),
                        "hunyuan_raw_glb": str(raw_glb),
                        "hunyuan_raw_render_png": str(raw_render),
                        "simplified_obj": str(simplified_obj),
                        "placed_render_png": str(placed_render),
                        "hunyuan_seconds": f"{seconds:.2f}",
                        "raw_verts": int(len(mesh.vertices)),
                        "raw_faces": int(len(mesh.faces)),
                        "simp_verts": simp_row["verts_after"],
                        "simp_faces": simp_row["faces_after"],
                        "recovery_method": "hunyuan_retry",
                        "recovery_retry_index": attempt_i,
                        "recovery_status": status,
                        "recovery_flags": flags,
                    })
                    for cand in new_row.get("retrieval_candidates", []):
                        cand["chosen"] = int(str(cand.get("candidate_id")) == candidate_id)
                    accepted = True
                    break

            if not accepted:
                fallback_id = str(row["retrieved_id"])
                fallback_obj = obj_dir / f"{fallback_id}.obj"
                stem = f"{int(row['index']):02d}_{safe_stem(osm_id)}_{fallback_id}_fallback"
                raw_render = render_dir / f"{stem}_retrieved_raw.png"
                placed_render = render_dir / f"{stem}_placed.png"
                fallback_copy = fallback_dir / f"{stem}.obj"
                mesh = trimesh.load(fallback_obj, force="mesh", process=False)
                mesh.export(fallback_copy)
                render_mesh_png(mesh, image_size=args.image_size).save(raw_render, optimize=True)
                placed = place_mesh(mesh, building["polygon"], float(row["height_m"]))
                render_mesh_png(placed, image_size=args.image_size).save(placed_render, optimize=True)
                candidate_log = dict(row)
                status, flags, _audit = audit_candidate(args, candidate_log, building, fallback_copy, fallback_copy, raw_render, placed_render)
                print(f"[recover] {osm_id} fallback {fallback_id}: {status} {flags}", flush=True)
                new_row.update({
                    "hunyuan_raw_glb": str(fallback_copy),
                    "hunyuan_raw_render_png": str(raw_render),
                    "simplified_obj": str(fallback_copy),
                    "placed_render_png": str(placed_render),
                    "hunyuan_seconds": "0.00",
                    "raw_verts": int(len(mesh.vertices)),
                    "raw_faces": int(len(mesh.faces)),
                    "simp_verts": int(len(mesh.vertices)),
                    "simp_faces": int(len(mesh.faces)),
                    "recovery_method": "retrieved_obj_fallback",
                    "recovery_status": status,
                    "recovery_flags": flags,
                })

            recovery_rows.append({
                "index": row["index"],
                "osm_id": osm_id,
                "original_candidate": row["retrieved_id"],
                "final_candidate": new_row["retrieved_id"],
                "recovery_method": new_row["recovery_method"],
                "original_status": original_audit.get("status", ""),
                "original_flags": original_audit.get("flags", ""),
                "final_status": new_row.get("recovery_status", original_audit.get("status", "")),
                "final_flags": new_row.get("recovery_flags", original_audit.get("flags", "")),
                "before_render_png": row.get("placed_render_png", ""),
                "after_render_png": new_row.get("placed_render_png", ""),
                "after_raw_render_png": new_row.get("hunyuan_raw_render_png", ""),
            })

        mesh_path = resolve_path(str(new_row["simplified_obj"]), repo)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        placed_meshes.append(place_mesh(mesh, building["polygon"], float(new_row["height_m"])))
        out_rows.append(new_row)

    scene = trimesh.util.concatenate(placed_meshes)
    scene_path = out_dir / "osm_hunyuan_scene.obj"
    scene.export(scene_path)
    log_path = out_dir / "osm_hunyuan_scene.log.json"
    log_path.write_text(json.dumps(out_rows, indent=2) + "\n")

    report_csv = out_dir / "recovery_report.csv"
    write_rows_csv(report_csv, recovery_rows)
    summary = {
        "count": len(out_rows),
        "targets": len(targets),
        "hunyuan_retry": sum(1 for r in recovery_rows if r.get("recovery_method") == "hunyuan_retry"),
        "retrieved_obj_fallback": sum(1 for r in recovery_rows if r.get("recovery_method") == "retrieved_obj_fallback"),
        "unchanged": len(out_rows) - len(targets),
    }
    summary_path = out_dir / "recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    sheet_path = out_dir / "recovery_sheet.png"
    make_recovery_sheet(recovery_rows, sheet_path, args.image_size)

    print(f"[recover] scene:   {scene_path}", flush=True)
    print(f"[recover] log:     {log_path}", flush=True)
    print(f"[recover] report:  {report_csv}", flush=True)
    print(f"[recover] summary: {summary_path}", flush=True)
    print(f"[recover] sheet:   {sheet_path}", flush=True)


if __name__ == "__main__":
    main()
