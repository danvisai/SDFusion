"""Run the OSM generation dataset pipeline over multiple tiles.

This is a thin orchestrator around the existing single-step scripts:

    extract_osm -> retrieval/Hunyuan -> heightfix -> quality audit
    -> map choices -> dataset shard -> merged corpus

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/run_osm_generation_batch.py \
        --tile east:40.4234,-86.9050,40.4250,-86.9025 \
        --tile west:40.4234,-86.9100,40.4250,-86.9075 \
        --limit 8 \
        --base_out outputs/osm_batch_lafayette_v2 \
        --existing_dataset outputs/osm_generation_dataset_12/smoke12_records.jsonl \
        --existing_dataset outputs/osm_generation_dataset_tile_north8/north8_records.jsonl \
        --corpus_out outputs/osm_generation_dataset_corpus_v2 \
        --corpus_name campus_lafayette_v2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tile",
        action="append",
        required=True,
        help="Tile as name:south,west,north,east. May be repeated.",
    )
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--base_out", default="outputs/osm_batch")
    ap.add_argument("--model", choices=["mini", "full"], default="mini")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--target_faces", type=int, default=50_000)
    ap.add_argument("--retrieval_top_k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--existing_dataset", action="append", default=[])
    ap.add_argument("--corpus_out", default="outputs/osm_generation_dataset_corpus_batch")
    ap.add_argument("--corpus_name", default="osm_generation_corpus_batch")
    return ap.parse_args()


def parse_tile(tile: str) -> tuple[str, list[str]]:
    if ":" not in tile:
        raise ValueError(f"Tile must be name:south,west,north,east, got {tile!r}")
    name, bbox_text = tile.split(":", 1)
    parts = [p.strip() for p in bbox_text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Tile must have 4 bbox values, got {tile!r}")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")
    return safe_name or "tile", parts


def run(cmd: list[str], cwd: Path = REPO) -> None:
    print("[batch] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def count_buildings(osm_json: Path) -> int:
    if not osm_json.exists():
        return 0
    payload = json.loads(osm_json.read_text())
    return len(payload.get("buildings", []))


def main() -> None:
    args = parse_args()
    base_out = Path(args.base_out)
    base_out.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    dataset_paths = [Path(p) for p in args.existing_dataset]
    tile_summaries = []
    for tile_arg in args.tile:
        tile_name, bbox = parse_tile(tile_arg)
        tile_root = base_out / tile_name
        extract_dir = tile_root / "extract"
        gen_dir = tile_root / "gen"
        height_dir = tile_root / "heightfix"
        quality_dir = tile_root / "quality"
        dataset_dir = tile_root / "dataset"
        extract_dir.mkdir(parents=True, exist_ok=True)
        osm_json = extract_dir / "osm_input.json"

        run([
            python, "scene/extract_osm.py",
            "--bbox", *bbox,
            "-o", str(osm_json),
        ])
        building_count = count_buildings(osm_json)
        if building_count == 0:
            print(f"[batch] skip {tile_name}: no buildings", flush=True)
            continue

        run([
            python, "scripts/osm_hunyuan_pipeline_smoke.py",
            "--osm_json", str(osm_json),
            "--out_dir", str(gen_dir),
            "--limit", str(args.limit),
            "--retrieval_policy", "rerank",
            "--retrieval_top_k", str(args.retrieval_top_k),
            "--model", args.model,
            "--steps", str(args.steps),
            "--target_faces", str(args.target_faces),
            "--device", args.device,
        ])
        run([
            python, "scripts/osm_recompose_height_policy.py",
            "--osm_json", str(osm_json),
            "--pipeline_log", str(gen_dir / "osm_hunyuan_scene.log.json"),
            "--out_dir", str(height_dir),
            "--height_policy", "area_aware",
        ])
        run([
            python, "scripts/osm_generation_quality_audit.py",
            "--osm_json", str(osm_json),
            "--pipeline_log", str(height_dir / "osm_hunyuan_scene.log.json"),
            "--out_dir", str(quality_dir),
        ])
        run([
            python, "scripts/osm_pipeline_map_choices.py",
            "--osm_json", str(osm_json),
            "--pipeline_log", str(height_dir / "osm_hunyuan_scene.log.json"),
            "--out_dir", str(height_dir),
        ])
        split = f"{tile_name}{args.limit}"
        run([
            python, "scripts/build_osm_generation_dataset.py",
            "--osm_json", str(osm_json),
            "--pipeline_log", str(height_dir / "osm_hunyuan_scene.log.json"),
            "--quality_csv", str(quality_dir / "generation_quality_audit.csv"),
            "--out_dir", str(dataset_dir),
            "--split", split,
        ])
        shard = dataset_dir / f"{split}_records.jsonl"
        dataset_paths.append(shard)
        tile_summaries.append({
            "tile": tile_name,
            "bbox": [float(v) for v in bbox],
            "osm_buildings": building_count,
            "dataset_jsonl": str(shard),
        })

    if dataset_paths:
        merge_cmd = [
            python, "scripts/merge_osm_generation_datasets.py",
            "--out_dir", args.corpus_out,
            "--name", args.corpus_name,
        ]
        for dataset_path in dataset_paths:
            merge_cmd.extend(["--dataset", str(dataset_path)])
        run(merge_cmd)

    manifest = {
        "base_out": str(base_out),
        "limit": args.limit,
        "tiles": tile_summaries,
        "existing_datasets": [str(p) for p in args.existing_dataset],
        "corpus_out": args.corpus_out,
        "corpus_name": args.corpus_name,
    }
    manifest_path = base_out / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[batch] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
