"""Merge OSM generation dataset shards into one corpus.

This script expects shards produced by scripts/build_osm_generation_dataset.py.
It concatenates JSONL records, writes a compact index CSV, packs all footprint
masks into one NPZ, and emits a summary for the combined corpus.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/merge_osm_generation_datasets.py \
        --dataset outputs/osm_generation_dataset_12/smoke12_records.jsonl \
        --dataset outputs/osm_generation_dataset_tile_north8/north8_records.jsonl \
        --out_dir outputs/osm_generation_dataset_corpus_v1 \
        --name corpus_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="Input records JSONL. May be repeated.")
    ap.add_argument("--out_dir", default="outputs/osm_generation_dataset_corpus")
    ap.add_argument("--name", default="corpus")
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    mask_arrays = {}
    source_counts = Counter()
    for dataset_path_value in args.dataset:
        dataset_path = Path(dataset_path_value)
        shard_records = read_jsonl(dataset_path)
        for record in shard_records:
            merged_index = len(records)
            merged_id = f"{record.get('split', 'shard')}_{int(record.get('index', merged_index)):04d}_{record['osm_id']}"
            source_counts[record.get("split", "unknown")] += 1
            mask_path = Path(record["footprint_mask_npy"])
            if mask_path.exists():
                mask_arrays[merged_id] = np.load(mask_path)

            merged = dict(record)
            merged["corpus_name"] = args.name
            merged["corpus_index"] = merged_index
            merged["source_records_jsonl"] = str(dataset_path)
            merged["corpus_mask_key"] = merged_id
            records.append(merged)

    jsonl_path = out_dir / f"{args.name}_records.jsonl"
    with jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    csv_path = out_dir / f"{args.name}_index.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "corpus_index", "split", "osm_id", "class", "area_m2", "height_m",
            "selected_candidate_id", "quality_status", "quality_flags",
            "include_as_positive", "include_as_negative", "source_records_jsonl",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "corpus_index": record["corpus_index"],
                "split": record["split"],
                "osm_id": record["osm_id"],
                "class": record["class"],
                "area_m2": f"{float(record['area_m2']):.3f}",
                "height_m": f"{float(record['height_m']):.3f}",
                "selected_candidate_id": record["selected_candidate_id"],
                "quality_status": record["quality"]["status"],
                "quality_flags": record["quality"]["flags"],
                "include_as_positive": int(record["training_use"]["include_as_positive"]),
                "include_as_negative": int(record["training_use"]["include_as_negative"]),
                "source_records_jsonl": record["source_records_jsonl"],
            })

    npz_path = out_dir / f"{args.name}_footprint_masks.npz"
    np.savez_compressed(npz_path, **mask_arrays)

    positive_count = sum(1 for r in records if r["training_use"]["include_as_positive"])
    negative_count = sum(1 for r in records if r["training_use"]["include_as_negative"])
    summary = {
        "corpus_name": args.name,
        "dataset_version": "osm_generation_scaffold_v1",
        "count": len(records),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "classes": dict(Counter(r["class"] for r in records)),
        "splits": dict(source_counts),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "masks_npz": str(npz_path),
    }
    summary_path = out_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[merge] records: {jsonl_path}")
    print(f"[merge] index:   {csv_path}")
    print(f"[merge] masks:   {npz_path}")
    print(f"[merge] summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
