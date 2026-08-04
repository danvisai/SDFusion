"""Summarize an OSM generation corpus JSONL.

The diagnostics are intentionally simple and file-based so they can run after
every batch:

- class and split counts
- positive/negative quality counts
- repeated selected candidate ids
- failure counts by candidate id and flag
- area/height/aspect distributions

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/osm_generation_corpus_diagnostics.py \
        --records outputs/osm_generation_dataset_corpus_v1/campus_lafayette_v1_records.jsonl \
        --out_dir outputs/osm_generation_dataset_corpus_v1/diagnostics
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def write_counter_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records_path = Path(args.records)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(records_path)

    class_counts = Counter(r["class"] for r in records)
    split_counts = Counter(r["split"] for r in records)
    status_counts = Counter(r.get("quality", {}).get("status", "unknown") for r in records)
    candidate_counts = Counter(r["selected_candidate_id"] for r in records)
    candidate_class_counts = Counter((r["class"], r["selected_candidate_id"]) for r in records)
    failure_by_candidate = Counter()
    failure_flag_counts = Counter()
    positive_by_split = defaultdict(int)
    negative_by_split = defaultdict(int)

    for record in records:
        split = record["split"]
        if record["training_use"]["include_as_positive"]:
            positive_by_split[split] += 1
        if record["training_use"]["include_as_negative"]:
            negative_by_split[split] += 1
            candidate = record["selected_candidate_id"]
            failure_by_candidate[candidate] += 1
            flags = record.get("quality", {}).get("flags", "")
            for flag in [f for f in flags.split("|") if f]:
                failure_flag_counts[flag] += 1

    area_values = [float(r["area_m2"]) for r in records]
    height_values = [float(r["height_m"]) for r in records]
    aspect_values = [float(r.get("geometry_features", {}).get("bbox_aspect", 0.0)) for r in records]
    raw_faces = [float(r.get("generation_metrics", {}).get("raw_faces", 0.0)) for r in records]
    simplified_faces = [float(r.get("generation_metrics", {}).get("simplified_faces", 0.0)) for r in records]

    write_counter_csv(
        out_dir / "class_counts.csv",
        ["class", "count"],
        class_counts.most_common(),
    )
    write_counter_csv(
        out_dir / "split_counts.csv",
        ["split", "count", "positive", "negative"],
        [(split, count, positive_by_split[split], negative_by_split[split]) for split, count in split_counts.most_common()],
    )
    write_counter_csv(
        out_dir / "candidate_reuse.csv",
        ["selected_candidate_id", "count"],
        candidate_counts.most_common(),
    )
    write_counter_csv(
        out_dir / "candidate_reuse_by_class.csv",
        ["class", "selected_candidate_id", "count"],
        [(cls, candidate, count) for (cls, candidate), count in candidate_class_counts.most_common()],
    )
    write_counter_csv(
        out_dir / "failure_by_candidate.csv",
        ["selected_candidate_id", "fail_count"],
        failure_by_candidate.most_common(),
    )
    write_counter_csv(
        out_dir / "failure_flags.csv",
        ["flag", "count"],
        failure_flag_counts.most_common(),
    )

    positive_count = sum(1 for r in records if r["training_use"]["include_as_positive"])
    negative_count = sum(1 for r in records if r["training_use"]["include_as_negative"])
    summary = {
        "records": str(records_path),
        "count": len(records),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": float(positive_count / max(len(records), 1)),
        "classes": dict(class_counts),
        "splits": dict(split_counts),
        "quality_status": dict(status_counts),
        "unique_selected_candidates": len(candidate_counts),
        "most_reused_candidates": candidate_counts.most_common(10),
        "failed_candidates": failure_by_candidate.most_common(),
        "failure_flags": dict(failure_flag_counts),
        "area_m2": quantiles(area_values),
        "height_m": quantiles(height_values),
        "bbox_aspect": quantiles(aspect_values),
        "raw_faces": quantiles(raw_faces),
        "simplified_faces": quantiles(simplified_faces),
    }
    summary_path = out_dir / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = out_dir / "diagnostics_report.md"
    lines = [
        "# OSM Generation Corpus Diagnostics",
        "",
        f"Records: {len(records)}",
        f"Positive: {positive_count}",
        f"Negative: {negative_count}",
        f"Positive rate: {summary['positive_rate']:.3f}",
        f"Unique selected candidates: {len(candidate_counts)}",
        "",
        "## Classes",
        *[f"- {k}: {v}" for k, v in class_counts.most_common()],
        "",
        "## Splits",
        *[f"- {k}: {v} records, {positive_by_split[k]} positive, {negative_by_split[k]} negative" for k, v in split_counts.most_common()],
        "",
        "## Most Reused Candidates",
        *[f"- {k}: {v}" for k, v in candidate_counts.most_common(10)],
        "",
        "## Failure Flags",
        *[f"- {k}: {v}" for k, v in failure_flag_counts.most_common()],
    ]
    report_path.write_text("\n".join(lines) + "\n")

    print(f"[diagnostics] summary: {summary_path}")
    print(f"[diagnostics] report:  {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
