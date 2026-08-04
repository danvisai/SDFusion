"""Run quality-rerank A/B tests over multiple OSM tiles and aggregate results."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
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
    ap.add_argument("--out_dir", default="outputs/quality_rerank_ab_sweep")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--model", choices=["mini", "full"], default="mini")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--target_faces", type=int, default=50_000)
    ap.add_argument("--retrieval_top_k", type=int, default=5)
    ap.add_argument("--quality_model", required=True)
    ap.add_argument("--quality_weight", type=float, default=0.20)
    ap.add_argument("--quality_bad_candidate_penalty", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def parse_tile(tile: str) -> tuple[str, list[str]]:
    if ":" not in tile:
        raise ValueError(f"Tile must be name:south,west,north,east, got {tile!r}")
    name, bbox_text = tile.split(":", 1)
    bbox = [part.strip() for part in bbox_text.split(",")]
    if len(bbox) != 4:
        raise ValueError(f"Tile must have 4 bbox values, got {tile!r}")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")
    return safe_name or "tile", bbox


def run(cmd: list[str]) -> None:
    print("[ab-sweep] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def add_counts(total: Counter, summary: dict, prefix: str) -> None:
    arm = summary[prefix]
    total[f"{prefix}_count"] += int(arm.get("count", 0))
    total[f"{prefix}_pass"] += int(arm.get("pass", 0))
    total[f"{prefix}_warn"] += int(arm.get("warn", 0))
    total[f"{prefix}_fail"] += int(arm.get("fail", 0))
    for flag, count in arm.get("flags", {}).items():
        total[f"{prefix}_flag_{flag}"] += int(count)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_rows = []
    comparison_rows = []
    totals: Counter = Counter()

    for tile_arg in args.tile:
        tile_name, bbox = parse_tile(tile_arg)
        tile_dir = out_dir / tile_name
        run([
            sys.executable,
            "scripts/run_quality_rerank_ab_test.py",
            "--bbox",
            *bbox,
            "--out_dir",
            str(tile_dir),
            "--limit",
            str(args.limit),
            "--quality_model",
            args.quality_model,
            "--quality_weight",
            str(args.quality_weight),
            "--quality_bad_candidate_penalty",
            str(args.quality_bad_candidate_penalty),
            "--model",
            args.model,
            "--steps",
            str(args.steps),
            "--target_faces",
            str(args.target_faces),
            "--retrieval_top_k",
            str(args.retrieval_top_k),
            "--device",
            args.device,
        ])

        summary = read_json(tile_dir / "ab_summary.json")
        add_counts(totals, summary, "geometry")
        add_counts(totals, summary, "quality")
        totals["choice_changed_count"] += int(summary.get("choice_changed_count", 0))

        comparison = read_csv(tile_dir / "ab_comparison.csv")
        for row in comparison:
            row = dict(row)
            row["tile"] = tile_name
            comparison_rows.append(row)

        tile_rows.append({
            "tile": tile_name,
            "bbox": ",".join(bbox),
            "geometry_count": summary["geometry"]["count"],
            "geometry_pass": summary["geometry"]["pass"],
            "geometry_warn": summary["geometry"]["warn"],
            "geometry_fail": summary["geometry"]["fail"],
            "quality_count": summary["quality"]["count"],
            "quality_pass": summary["quality"]["pass"],
            "quality_warn": summary["quality"]["warn"],
            "quality_fail": summary["quality"]["fail"],
            "choice_changed_count": summary["choice_changed_count"],
            "tile_dir": str(tile_dir),
        })

    total_count = int(totals["geometry_count"])
    aggregate = {
        "tile_count": len(tile_rows),
        "building_count": total_count,
        "geometry": {
            "pass": int(totals["geometry_pass"]),
            "warn": int(totals["geometry_warn"]),
            "fail": int(totals["geometry_fail"]),
            "pass_rate": float(totals["geometry_pass"] / max(total_count, 1)),
        },
        "quality": {
            "pass": int(totals["quality_pass"]),
            "warn": int(totals["quality_warn"]),
            "fail": int(totals["quality_fail"]),
            "pass_rate": float(totals["quality_pass"] / max(total_count, 1)),
        },
        "choice_changed_count": int(totals["choice_changed_count"]),
        "choice_changed_rate": float(totals["choice_changed_count"] / max(total_count, 1)),
    }

    tile_csv = out_dir / "sweep_tile_summary.csv"
    with tile_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tile_rows)

    comparison_csv = out_dir / "sweep_ab_comparison.csv"
    fieldnames = ["tile"] + [name for name in comparison_rows[0].keys() if name != "tile"]
    with comparison_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2) + "\n")

    changed = [row for row in comparison_rows if str(row.get("choice_changed", "")) == "1"]
    report_lines = [
        "# Quality Rerank A/B Sweep",
        "",
        "## Summary",
        f"- tiles: {aggregate['tile_count']}",
        f"- buildings: {aggregate['building_count']}",
        f"- geometry: {aggregate['geometry']['pass']} pass, {aggregate['geometry']['fail']} fail",
        f"- quality: {aggregate['quality']['pass']} pass, {aggregate['quality']['fail']} fail",
        f"- choice changes: {aggregate['choice_changed_count']} / {aggregate['building_count']}",
        "",
        "## Changed Choices",
    ]
    if changed:
        for row in changed:
            report_lines.append(
                "- {tile} {osm_id}: {geometry_candidate} -> {quality_candidate} "
                "({geometry_status} -> {quality_status})".format(**row)
            )
    else:
        report_lines.append("- none")

    report_path = out_dir / "sweep_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"[ab-sweep] report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
