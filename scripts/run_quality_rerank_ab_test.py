"""Run an A/B test: geometry rerank vs quality-aware rerank.

The script runs both policies on the same OSM tile, applies the same height
policy, audits both outputs, and writes a comparison report.
"""
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
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    ap.add_argument("--osm_json", help="Use an existing OSM JSON instead of extracting bbox")
    ap.add_argument("--out_dir", default="outputs/quality_rerank_ab")
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


def run(cmd: list[str]) -> None:
    print("[ab] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_audit(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def audit_summary(rows: list[dict]) -> dict:
    statuses = Counter(row["status"] for row in rows)
    flags = Counter()
    for row in rows:
        for flag in row.get("flags", "").split("|"):
            if flag:
                flags[flag] += 1
    return {
        "count": len(rows),
        "pass": statuses.get("pass", 0),
        "warn": statuses.get("warn", 0),
        "fail": statuses.get("fail", 0),
        "flags": dict(flags),
    }


def load_choices(log_path: Path) -> dict[str, dict]:
    rows = read_json(log_path)
    return {str(row["osm_id"]): row for row in rows}


def write_comparison(out_dir: Path, geom_dir: Path, quality_dir: Path) -> dict:
    geom_audit = read_audit(geom_dir / "quality" / "generation_quality_audit.csv")
    quality_audit = read_audit(quality_dir / "quality" / "generation_quality_audit.csv")
    geom_log = load_choices(geom_dir / "heightfix" / "osm_hunyuan_scene.log.json")
    quality_log = load_choices(quality_dir / "heightfix" / "osm_hunyuan_scene.log.json")
    geom_audit_by_id = {str(row["osm_id"]): row for row in geom_audit}
    quality_audit_by_id = {str(row["osm_id"]): row for row in quality_audit}

    rows = []
    for osm_id in sorted(set(geom_log) | set(quality_log)):
        g = geom_log.get(osm_id, {})
        q = quality_log.get(osm_id, {})
        ga = geom_audit_by_id.get(osm_id, {})
        qa = quality_audit_by_id.get(osm_id, {})
        rows.append({
            "osm_id": osm_id,
            "class": g.get("class", q.get("class", "")),
            "geometry_candidate": g.get("retrieved_id", ""),
            "quality_candidate": q.get("retrieved_id", ""),
            "choice_changed": int(g.get("retrieved_id", "") != q.get("retrieved_id", "")),
            "geometry_status": ga.get("status", ""),
            "quality_status": qa.get("status", ""),
            "geometry_flags": ga.get("flags", ""),
            "quality_flags": qa.get("flags", ""),
        })

    comparison_csv = out_dir / "ab_comparison.csv"
    with comparison_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "geometry": audit_summary(geom_audit),
        "quality": audit_summary(quality_audit),
        "choice_changed_count": int(sum(row["choice_changed"] for row in rows)),
        "comparison_csv": str(comparison_csv),
        "geometry_dir": str(geom_dir),
        "quality_dir": str(quality_dir),
    }
    summary_path = out_dir / "ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_lines = [
        "# Quality Rerank A/B Test",
        "",
        "## Summary",
        f"- geometry pass/fail: {summary['geometry']['pass']} pass, {summary['geometry']['fail']} fail",
        f"- quality pass/fail: {summary['quality']['pass']} pass, {summary['quality']['fail']} fail",
        f"- choice changes: {summary['choice_changed_count']} / {len(rows)}",
        "",
        "## Geometry Flags",
        *[f"- {k}: {v}" for k, v in summary["geometry"]["flags"].items()],
        "",
        "## Quality Flags",
        *[f"- {k}: {v}" for k, v in summary["quality"]["flags"].items()],
    ]
    report_path = out_dir / "ab_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ab] report: {report_path}", flush=True)
    return summary


def run_arm(args: argparse.Namespace, osm_json: Path, arm_dir: Path, policy: str) -> None:
    gen_dir = arm_dir / "gen"
    height_dir = arm_dir / "heightfix"
    quality_dir = arm_dir / "quality"
    cmd = [
        sys.executable, "scripts/osm_hunyuan_pipeline_smoke.py",
        "--osm_json", str(osm_json),
        "--out_dir", str(gen_dir),
        "--limit", str(args.limit),
        "--retrieval_policy", policy,
        "--retrieval_top_k", str(args.retrieval_top_k),
        "--model", args.model,
        "--steps", str(args.steps),
        "--target_faces", str(args.target_faces),
        "--device", args.device,
    ]
    if policy == "quality":
        cmd.extend([
            "--quality_model", args.quality_model,
            "--quality_weight", str(args.quality_weight),
            "--quality_bad_candidate_penalty", str(args.quality_bad_candidate_penalty),
        ])
    run(cmd)
    run([
        sys.executable, "scripts/osm_recompose_height_policy.py",
        "--osm_json", str(osm_json),
        "--pipeline_log", str(gen_dir / "osm_hunyuan_scene.log.json"),
        "--out_dir", str(height_dir),
        "--height_policy", "area_aware",
    ])
    run([
        sys.executable, "scripts/osm_generation_quality_audit.py",
        "--osm_json", str(osm_json),
        "--pipeline_log", str(height_dir / "osm_hunyuan_scene.log.json"),
        "--out_dir", str(quality_dir),
    ])
    run([
        sys.executable, "scripts/osm_pipeline_map_choices.py",
        "--osm_json", str(osm_json),
        "--pipeline_log", str(height_dir / "osm_hunyuan_scene.log.json"),
        "--out_dir", str(height_dir),
    ])


def main() -> None:
    args = parse_args()
    if not args.osm_json and not args.bbox:
        raise SystemExit("Provide either --osm_json or --bbox.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.osm_json:
        osm_json = Path(args.osm_json)
    else:
        osm_json = out_dir / "osm_input.json"
        run([
            sys.executable, "scene/extract_osm.py",
            "--bbox", *(str(v) for v in args.bbox),
            "-o", str(osm_json),
        ])

    geometry_dir = out_dir / "geometry_rerank"
    quality_dir = out_dir / "quality_rerank"
    run_arm(args, osm_json, geometry_dir, "rerank")
    run_arm(args, osm_json, quality_dir, "quality")
    write_comparison(out_dir, geometry_dir, quality_dir)


if __name__ == "__main__":
    main()
