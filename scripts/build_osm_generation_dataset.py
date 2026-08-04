"""Build a JSONL training scaffold from OSM pipeline outputs.

The goal is not to train yet. This creates clean records for a future
conditional/generative model:

    footprint + class + height/context + top-k candidates -> generated asset

It preserves quality labels so later fine-tuning can filter failures or use
them as negative examples.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/build_osm_generation_dataset.py \
        --osm_json outputs/osm_pipeline_smoke/osm_input.json \
        --pipeline_log outputs/osm_pipeline_heightfix_12/osm_hunyuan_scene.log.json \
        --quality_csv outputs/osm_pipeline_quality_12/generation_quality_audit.csv \
        --out_dir outputs/osm_generation_dataset_12 \
        --split smoke12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import rasterize_polygon


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--pipeline_log", required=True)
    ap.add_argument("--quality_csv")
    ap.add_argument("--out_dir", default="outputs/osm_generation_dataset")
    ap.add_argument("--split", default="smoke")
    ap.add_argument("--mask_res", type=int, default=64)
    return ap.parse_args()


def polygon_stats(polygon: list[list[float]] | list[tuple[float, float]]) -> dict[str, float]:
    poly = np.asarray(polygon, dtype=np.float64)
    ext = poly.max(axis=0) - poly.min(axis=0)
    area = 0.5 * abs(sum(
        poly[i, 0] * poly[(i + 1) % len(poly), 1]
        - poly[(i + 1) % len(poly), 0] * poly[i, 1]
        for i in range(len(poly))
    ))
    edges = np.roll(poly, -1, axis=0) - poly
    perimeter = float(np.linalg.norm(edges, axis=1).sum())
    return {
        "bbox_width_m": float(ext[0]),
        "bbox_depth_m": float(ext[1]),
        "bbox_aspect": float(max(ext[0], ext[1]) / max(min(ext[0], ext[1]), 1e-6)),
        "area_m2_from_polygon": float(area),
        "perimeter_m": perimeter,
        "compactness": float((4.0 * np.pi * area) / max(perimeter * perimeter, 1e-6)),
    }


def load_quality(path_value: str | None) -> dict[str, dict[str, str]]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    rows = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rows[str(row["osm_id"])] = row
    return rows


def rel_or_abs(path_value: str | None) -> str:
    if not path_value:
        return ""
    return str(path_value)


def candidate_records(row: dict) -> list[dict[str, object]]:
    candidates = row.get("retrieval_candidates") or []
    out = []
    for cand in candidates:
        out.append({
            "rank": int(cand.get("rank", 0)),
            "candidate_id": cand.get("candidate_id", ""),
            "retrieval_score": float(cand.get("retrieval_score", 0.0)),
            "rerank_score": float(cand.get("rerank_score", cand.get("retrieval_score", 0.0))),
            "chosen": int(cand.get("chosen", 0)),
            "target_aspect": float(cand.get("target_aspect", 0.0)),
            "candidate_aspect": float(cand.get("candidate_aspect", 0.0)),
            "target_height_ratio": float(cand.get("target_height_ratio", 0.0)),
            "candidate_height_ratio": float(cand.get("candidate_height_ratio", 0.0)),
            "aspect_penalty": float(cand.get("aspect_penalty", 0.0)),
            "height_penalty": float(cand.get("height_penalty", 0.0)),
            "candidate_verts": int(cand.get("candidate_verts", 0)),
            "candidate_faces": int(cand.get("candidate_faces", 0)),
        })
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "footprint_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    osm_payload = json.load(open(args.osm_json))
    pipeline_rows = json.load(open(args.pipeline_log))
    quality_by_id = load_quality(args.quality_csv)
    buildings = {str(b["id"]): b for b in osm_payload.get("buildings", [])}

    records = []
    masks = {}
    for row in pipeline_rows:
        osm_id = str(row["osm_id"])
        building = buildings[osm_id]
        mask = rasterize_polygon(building["polygon"], res=args.mask_res).astype(np.uint8)
        mask_key = f"{args.split}_{int(row['index']):04d}_{osm_id}"
        mask_path = mask_dir / f"{mask_key}.npy"
        np.save(mask_path, mask)
        masks[mask_key] = mask

        q = quality_by_id.get(osm_id, {})
        pstats = polygon_stats(building["polygon"])
        height = float(row.get("height_m", building.get("height", 0.0)))
        record = {
            "dataset_version": "osm_generation_scaffold_v1",
            "split": args.split,
            "index": int(row["index"]),
            "osm_id": osm_id,
            "class": row["class"],
            "top_level": next((t for t in ("RESIDENTIAL", "RELIGIOUS", "COMMERCIAL", "MILITARY", "PUBLIC") if row["class"].startswith(t)), "RESIDENTIAL"),
            "polygon_xy_m": building["polygon"],
            "centroid_xy_m": building.get("centroid", []),
            "area_m2": float(building.get("area", row.get("area_m2", 0.0))),
            "height_m": height,
            "height_policy": row.get("height_policy", ""),
            "height_source": row.get("height_source", ""),
            "original_height_m": row.get("original_height_m", ""),
            "footprint_mask_npy": str(mask_path),
            "footprint_mask_res": args.mask_res,
            "geometry_features": {
                **pstats,
                "height_to_sqrt_area": float(height / max(np.sqrt(float(building.get("area", 0.0))), 1e-6)),
                "height_to_bbox_max": float(height / max(pstats["bbox_width_m"], pstats["bbox_depth_m"], 1e-6)),
            },
            "retrieval_policy": row.get("retrieval_policy", ""),
            "retrieval_top_k": int(row.get("retrieval_top_k", 0) or 0),
            "retrieval_candidates": candidate_records(row),
            "selected_candidate_id": row["retrieved_id"],
            "asset_paths": {
                "retrieved_input_png": rel_or_abs(row.get("retrieved_input_png")),
                "hunyuan_raw_glb": rel_or_abs(row.get("hunyuan_raw_glb")),
                "hunyuan_raw_render_png": rel_or_abs(row.get("hunyuan_raw_render_png")),
                "simplified_obj": rel_or_abs(row.get("simplified_obj")),
                "placed_render_png": rel_or_abs(row.get("placed_render_png")),
            },
            "generation_metrics": {
                "hunyuan_seconds": float(row.get("hunyuan_seconds", 0.0) or 0.0),
                "raw_verts": int(row.get("raw_verts", q.get("raw_verts", 0)) or 0),
                "raw_faces": int(row.get("raw_faces", q.get("raw_faces", 0)) or 0),
                "simplified_verts": int(row.get("simp_verts", q.get("simplified_verts", 0)) or 0),
                "simplified_faces": int(row.get("simp_faces", q.get("simplified_faces", 0)) or 0),
            },
            "quality": {
                "status": q.get("status", "unknown"),
                "flags": q.get("flags", ""),
                "raw_height_ratio": float(q.get("raw_height_ratio", 0.0) or 0.0),
                "placed_height_ratio": float(q.get("placed_height_ratio", 0.0) or 0.0),
                "raw_render_ink": float(q.get("raw_render_ink", 0.0) or 0.0),
                "placed_render_ink": float(q.get("placed_render_ink", 0.0) or 0.0),
            },
            "training_use": {
                "include_as_positive": q.get("status", "unknown") == "pass",
                "include_as_negative": q.get("status", "") == "fail",
                "notes": "Scaffold record only; no model training has been run.",
            },
        }
        records.append(record)

    jsonl_path = out_dir / f"{args.split}_records.jsonl"
    with jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    csv_path = out_dir / f"{args.split}_index.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "index", "osm_id", "class", "area_m2", "height_m", "bbox_aspect",
            "selected_candidate_id", "quality_status", "quality_flags",
            "include_as_positive", "include_as_negative", "footprint_mask_npy",
            "simplified_obj", "placed_render_png",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "index": record["index"],
                "osm_id": record["osm_id"],
                "class": record["class"],
                "area_m2": f"{record['area_m2']:.3f}",
                "height_m": f"{record['height_m']:.3f}",
                "bbox_aspect": f"{record['geometry_features']['bbox_aspect']:.6f}",
                "selected_candidate_id": record["selected_candidate_id"],
                "quality_status": record["quality"]["status"],
                "quality_flags": record["quality"]["flags"],
                "include_as_positive": int(record["training_use"]["include_as_positive"]),
                "include_as_negative": int(record["training_use"]["include_as_negative"]),
                "footprint_mask_npy": record["footprint_mask_npy"],
                "simplified_obj": record["asset_paths"]["simplified_obj"],
                "placed_render_png": record["asset_paths"]["placed_render_png"],
            })

    npz_path = out_dir / f"{args.split}_footprint_masks.npz"
    np.savez_compressed(npz_path, **masks)

    summary = {
        "dataset_version": "osm_generation_scaffold_v1",
        "split": args.split,
        "count": len(records),
        "positive_count": sum(1 for r in records if r["training_use"]["include_as_positive"]),
        "negative_count": sum(1 for r in records if r["training_use"]["include_as_negative"]),
        "classes": {},
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "masks_npz": str(npz_path),
    }
    for record in records:
        summary["classes"][record["class"]] = summary["classes"].get(record["class"], 0) + 1
    summary_path = out_dir / f"{args.split}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[dataset] records: {jsonl_path}")
    print(f"[dataset] index:   {csv_path}")
    print(f"[dataset] masks:   {npz_path}")
    print(f"[dataset] summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
