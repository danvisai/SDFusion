"""Build a supervised dataset for footprint-conditioned proposal images.

The target image is currently the successful retrieved-render conditioning PNG
from the OSM/Hunyuan corpus. This gives us a trainable bridge:

    footprint + class + height/context -> building proposal image

Later, the target can be replaced by artist/generated preferred views without
changing the training interface.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CLASS_ORDER = ["RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS", "MILITARY"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_proposal_image_dataset_v1")
    ap.add_argument("--name", default="campus_lafayette_proposal_v1")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--include_warn", action="store_true")
    return ap.parse_args()


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def class_id(top_level: str) -> int:
    return CLASS_ORDER.index(top_level) if top_level in CLASS_ORDER else 0


def make_example(record: dict, source_records: Path) -> dict | None:
    status = record.get("quality", {}).get("status", "")
    positive = bool(record.get("training_use", {}).get("include_as_positive"))
    if not positive and status != "warn":
        return None
    mask = Path(record.get("footprint_mask_npy", ""))
    target = Path(record.get("asset_paths", {}).get("retrieved_input_png", ""))
    if not mask.exists() or not target.exists():
        return None
    geom = record.get("geometry_features", {})
    top_level = record.get("top_level", "RESIDENTIAL")
    return {
        "dataset_version": "osm_proposal_image_v1",
        "source_records": str(source_records),
        "split_source": record.get("split", ""),
        "corpus_index": record.get("corpus_index", record.get("index", 0)),
        "osm_id": record["osm_id"],
        "class": record["class"],
        "top_level": top_level,
        "class_id": class_id(top_level),
        "area_m2": float(record.get("area_m2", 0.0) or 0.0),
        "height_m": float(record.get("height_m", 0.0) or 0.0),
        "bbox_aspect": float(geom.get("bbox_aspect", 1.0) or 1.0),
        "height_to_sqrt_area": float(geom.get("height_to_sqrt_area", 0.0) or 0.0),
        "height_to_bbox_max": float(geom.get("height_to_bbox_max", 0.0) or 0.0),
        "footprint_mask_npy": str(mask),
        "target_image_png": str(target),
        "selected_candidate_id": record.get("selected_candidate_id", ""),
        "quality_status": status,
        "quality_flags": record.get("quality", {}).get("flags", ""),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.records)
    records = load_records(source)

    examples = []
    for record in records:
        status = record.get("quality", {}).get("status", "")
        if status == "warn" and not args.include_warn:
            continue
        ex = make_example(record, source)
        if ex is not None:
            examples.append(ex)

    if not examples:
        raise SystemExit("No usable proposal-image examples found.")

    rng = random.Random(args.seed)
    examples = sorted(examples, key=lambda r: (r["split_source"], int(r["corpus_index"]), r["osm_id"]))
    rng.shuffle(examples)
    val_count = max(1, round(len(examples) * args.val_fraction)) if len(examples) > 1 else 0
    val_rows = examples[:val_count]
    train_rows = examples[val_count:]

    train_path = out_dir / f"{args.name}_train.jsonl"
    val_path = out_dir / f"{args.name}_val.jsonl"
    all_path = out_dir / f"{args.name}_all.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(all_path, examples)

    classes: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in examples:
        classes[row["top_level"]] = classes.get(row["top_level"], 0) + 1
        statuses[row["quality_status"]] = statuses.get(row["quality_status"], 0) + 1
    summary = {
        "dataset_version": "osm_proposal_image_v1",
        "name": args.name,
        "source_records": str(source),
        "count": len(examples),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "classes": classes,
        "quality_statuses": statuses,
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "all_jsonl": str(all_path),
    }
    summary_path = out_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[proposal-dataset] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
