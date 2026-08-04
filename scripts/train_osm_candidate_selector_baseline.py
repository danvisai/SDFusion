"""Train a lightweight corpus-aware candidate selector baseline.

This baseline is deliberately small. It does not generate geometry. It learns a
scoring model over each OSM footprint's retrieved candidates:

    footprint/class/height features + candidate retrieval/rerank features
        -> probability that this candidate should be selected

The goal is to make the retrieval stage measurable before moving to a true
generative model.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/train_osm_candidate_selector_baseline.py \
        --records outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl \
        --out_dir outputs/osm_candidate_selector_baseline_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


TOP_LEVELS = ["RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS", "MILITARY"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_candidate_selector_baseline")
    ap.add_argument("--val_fraction", type=float, default=0.25)
    ap.add_argument("--min_bad_candidate_count", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260510)
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_float(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def split_records(records: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for record in records:
        key = f"{record.get('split', '')}:{record.get('osm_id', '')}:{record.get('selected_candidate_id', '')}"
        (val if stable_float(key) < val_fraction else train).append(record)
    if not val and len(train) > 1:
        val.append(train.pop())
    if not train and len(val) > 1:
        train.append(val.pop())
    return train, val


def top_level(record: dict) -> str:
    value = record.get("top_level", "")
    if value:
        return str(value)
    cls = str(record.get("class", ""))
    return next((t for t in TOP_LEVELS if cls.startswith(t)), "RESIDENTIAL")


def safe_log(value: float) -> float:
    return float(np.log(max(float(value), 1e-8)))


def candidate_features(record: dict, candidate: dict) -> list[float]:
    geom = record.get("geometry_features", {})
    height = float(record.get("height_m", 0.0) or 0.0)
    area = float(record.get("area_m2", 0.0) or 0.0)
    cand_aspect = float(candidate.get("candidate_aspect", 0.0) or 0.0)
    target_aspect = float(candidate.get("target_aspect", geom.get("bbox_aspect", 0.0)) or 0.0)
    cand_height_ratio = float(candidate.get("candidate_height_ratio", 0.0) or 0.0)
    target_height_ratio = float(candidate.get("target_height_ratio", geom.get("height_to_bbox_max", 0.0)) or 0.0)

    one_hot = [1.0 if top_level(record) == t else 0.0 for t in TOP_LEVELS]
    return [
        safe_log(area),
        height,
        safe_log(height + 1.0),
        float(geom.get("bbox_width_m", 0.0) or 0.0),
        float(geom.get("bbox_depth_m", 0.0) or 0.0),
        float(geom.get("bbox_aspect", 0.0) or 0.0),
        float(geom.get("compactness", 0.0) or 0.0),
        float(geom.get("height_to_sqrt_area", 0.0) or 0.0),
        float(geom.get("height_to_bbox_max", 0.0) or 0.0),
        float(candidate.get("rank", 0.0) or 0.0),
        float(candidate.get("retrieval_score", 0.0) or 0.0),
        float(candidate.get("rerank_score", candidate.get("retrieval_score", 0.0)) or 0.0),
        target_aspect,
        cand_aspect,
        abs(safe_log(cand_aspect) - safe_log(target_aspect)),
        target_height_ratio,
        cand_height_ratio,
        abs(safe_log(cand_height_ratio) - safe_log(target_height_ratio)),
        float(candidate.get("aspect_penalty", 0.0) or 0.0),
        float(candidate.get("height_penalty", 0.0) or 0.0),
        safe_log(float(candidate.get("candidate_verts", 0.0) or 0.0) + 1.0),
        safe_log(float(candidate.get("candidate_faces", 0.0) or 0.0) + 1.0),
        *one_hot,
    ]


FEATURE_NAMES = [
    "log_area_m2",
    "height_m",
    "log_height_plus_1",
    "bbox_width_m",
    "bbox_depth_m",
    "bbox_aspect",
    "compactness",
    "height_to_sqrt_area",
    "height_to_bbox_max",
    "candidate_rank",
    "retrieval_score",
    "rerank_score",
    "target_aspect",
    "candidate_aspect",
    "aspect_log_error",
    "target_height_ratio",
    "candidate_height_ratio",
    "height_ratio_log_error",
    "aspect_penalty",
    "height_penalty",
    "log_candidate_verts",
    "log_candidate_faces",
    *[f"top_{name.lower()}" for name in TOP_LEVELS],
]


def failed_candidate_counts(records: list[dict]) -> Counter:
    counts = Counter()
    for record in records:
        if record.get("training_use", {}).get("include_as_negative"):
            counts[record.get("selected_candidate_id", "")] += 1
    return counts


def usable_positive(record: dict, bad_candidates: set[str]) -> bool:
    return (
        bool(record.get("training_use", {}).get("include_as_positive"))
        and record.get("selected_candidate_id") not in bad_candidates
        and len(record.get("retrieval_candidates", [])) > 0
    )


def build_pair_rows(records: list[dict], bad_candidates: set[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features = []
    labels = []
    meta = []
    for record in records:
        selected = record.get("selected_candidate_id")
        for candidate in record.get("retrieval_candidates", []):
            candidate_id = candidate.get("candidate_id", "")
            if candidate_id in bad_candidates:
                continue
            features.append(candidate_features(record, candidate))
            labels.append(1 if candidate_id == selected else 0)
            meta.append({
                "osm_id": record.get("osm_id", ""),
                "split": record.get("split", ""),
                "class": record.get("class", ""),
                "selected_candidate_id": selected,
                "candidate_id": candidate_id,
            })
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), meta


def score_record(model: RandomForestClassifier, record: dict, bad_candidates: set[str], apply_filter: bool) -> tuple[str, float]:
    candidates = []
    for candidate in record.get("retrieval_candidates", []):
        candidate_id = candidate.get("candidate_id", "")
        if apply_filter and candidate_id in bad_candidates:
            continue
        candidates.append(candidate)
    if not candidates:
        candidates = record.get("retrieval_candidates", [])
    if not candidates:
        return "", 0.0
    x = np.asarray([candidate_features(record, c) for c in candidates], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    best_idx = int(np.argmax(probs))
    return str(candidates[best_idx].get("candidate_id", "")), float(probs[best_idx])


def best_rerank(record: dict, bad_candidates: set[str], apply_filter: bool) -> str:
    candidates = []
    for candidate in record.get("retrieval_candidates", []):
        candidate_id = candidate.get("candidate_id", "")
        if apply_filter and candidate_id in bad_candidates:
            continue
        candidates.append(candidate)
    if not candidates:
        candidates = record.get("retrieval_candidates", [])
    if not candidates:
        return ""
    best = max(candidates, key=lambda c: float(c.get("rerank_score", c.get("retrieval_score", 0.0)) or 0.0))
    return str(best.get("candidate_id", ""))


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def evaluate_records(model: RandomForestClassifier, records: list[dict], bad_candidates: set[str]) -> tuple[dict, list[dict]]:
    rows = []
    for record in records:
        selected = str(record.get("selected_candidate_id", ""))
        learned, learned_score = score_record(model, record, bad_candidates, apply_filter=False)
        filtered, filtered_score = score_record(model, record, bad_candidates, apply_filter=True)
        rerank = best_rerank(record, bad_candidates, apply_filter=False)
        filtered_rerank = best_rerank(record, bad_candidates, apply_filter=True)
        rows.append({
            "split": record.get("split", ""),
            "osm_id": record.get("osm_id", ""),
            "class": record.get("class", ""),
            "selected_candidate_id": selected,
            "rerank_candidate_id": rerank,
            "filtered_rerank_candidate_id": filtered_rerank,
            "learned_candidate_id": learned,
            "learned_score": learned_score,
            "filtered_learned_candidate_id": filtered,
            "filtered_learned_score": filtered_score,
            "rerank_correct": int(rerank == selected),
            "filtered_rerank_correct": int(filtered_rerank == selected),
            "learned_correct": int(learned == selected),
            "filtered_learned_correct": int(filtered == selected),
        })
    metrics = {}
    for key in ["rerank_correct", "filtered_rerank_correct", "learned_correct", "filtered_learned_correct"]:
        metrics[key.replace("_correct", "_top1_accuracy")] = float(np.mean([r[key] for r in rows])) if rows else 0.0
    return metrics, rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(Path(args.records))
    fail_counts = failed_candidate_counts(records)
    bad_candidates = {candidate for candidate, count in fail_counts.items() if count >= args.min_bad_candidate_count}

    positives = [r for r in records if usable_positive(r, bad_candidates)]
    negatives = [r for r in records if r.get("training_use", {}).get("include_as_negative")]
    train_records, val_records = split_records(positives, args.val_fraction)

    x_train, y_train, _ = build_pair_rows(train_records, bad_candidates)
    x_val, y_val, _ = build_pair_rows(val_records, bad_candidates)
    if len(np.unique(y_train)) < 2:
        raise SystemExit("Need at least one positive and one negative candidate row for training.")

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1] if len(x_val) else np.asarray([])
    pair_metrics = {
        "train_candidate_row_count": int(len(y_train)),
        "val_candidate_row_count": int(len(y_val)),
        "train_candidate_positive_rate": float(np.mean(y_train)),
        "val_candidate_positive_rate": float(np.mean(y_val)) if len(y_val) else 0.0,
        "train_candidate_average_precision": float(average_precision_score(y_train, train_prob)),
        "val_candidate_average_precision": float(average_precision_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.0,
        "train_candidate_roc_auc": float(roc_auc_score(y_train, train_prob)) if len(np.unique(y_train)) > 1 else 0.0,
        "val_candidate_roc_auc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.0,
        "train_candidate_accuracy_at_0_5": float(accuracy_score(y_train, train_prob >= 0.5)),
        "val_candidate_accuracy_at_0_5": float(accuracy_score(y_val, val_prob >= 0.5)) if len(y_val) else 0.0,
    }
    train_rank_metrics, train_predictions = evaluate_records(model, train_records, bad_candidates)
    val_rank_metrics, val_predictions = evaluate_records(model, val_records, bad_candidates)

    write_jsonl(out_dir / "filtered_train_records.jsonl", train_records)
    write_jsonl(out_dir / "filtered_val_records.jsonl", val_records)
    write_jsonl(out_dir / "negative_records.jsonl", negatives)
    (out_dir / "bad_candidates.json").write_text(json.dumps({
        "min_bad_candidate_count": args.min_bad_candidate_count,
        "failed_candidate_counts": dict(fail_counts),
        "bad_candidates": sorted(bad_candidates),
    }, indent=2) + "\n")

    with (out_dir / "candidate_selector_model.pkl").open("wb") as f:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES, "bad_candidates": sorted(bad_candidates)}, f)

    with (out_dir / "candidate_selector_predictions.csv").open("w", newline="") as f:
        fieldnames = list((train_predictions + val_predictions)[0].keys()) + ["eval_split"] if (train_predictions or val_predictions) else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in train_predictions:
            writer.writerow({**row, "eval_split": "train"})
        for row in val_predictions:
            writer.writerow({**row, "eval_split": "val"})

    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda item: item[1], reverse=True)
    with (out_dir / "feature_importance.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows((name, f"{value:.8f}") for name, value in importances)

    metrics = {
        "records": str(args.records),
        "positive_records_before_bad_candidate_filter": sum(1 for r in records if r.get("training_use", {}).get("include_as_positive")),
        "positive_records_after_bad_candidate_filter": len(positives),
        "negative_records": len(negatives),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "bad_candidates": sorted(bad_candidates),
        "pair_metrics": pair_metrics,
        "train_rank_metrics": train_rank_metrics,
        "val_rank_metrics": val_rank_metrics,
        "top_feature_importance": importances[:10],
    }
    (out_dir / "candidate_selector_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    report = [
        "# OSM Candidate Selector Baseline",
        "",
        f"Records: {args.records}",
        f"Positive records after filtering: {len(positives)}",
        f"Negative records retained: {len(negatives)}",
        f"Train records: {len(train_records)}",
        f"Validation records: {len(val_records)}",
        "",
        "## Bad Candidates",
        *[f"- {candidate}: {fail_counts[candidate]} failures" for candidate in sorted(bad_candidates)],
        "",
        "## Validation Top-1 Accuracy",
        f"- rerank: {val_rank_metrics.get('rerank_top1_accuracy', 0.0):.3f}",
        f"- filtered rerank: {val_rank_metrics.get('filtered_rerank_top1_accuracy', 0.0):.3f}",
        f"- learned selector: {val_rank_metrics.get('learned_top1_accuracy', 0.0):.3f}",
        f"- filtered learned selector: {val_rank_metrics.get('filtered_learned_top1_accuracy', 0.0):.3f}",
        "",
        "## Pairwise Candidate Metrics",
        f"- train average precision: {pair_metrics['train_candidate_average_precision']:.3f}",
        f"- val average precision: {pair_metrics['val_candidate_average_precision']:.3f}",
        f"- train ROC AUC: {pair_metrics['train_candidate_roc_auc']:.3f}",
        f"- val ROC AUC: {pair_metrics['val_candidate_roc_auc']:.3f}",
        "",
        "## Top Features",
        *[f"- {name}: {value:.4f}" for name, value in importances[:10]],
    ]
    (out_dir / "candidate_selector_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"[selector] report: {out_dir / 'candidate_selector_report.md'}")


if __name__ == "__main__":
    main()
