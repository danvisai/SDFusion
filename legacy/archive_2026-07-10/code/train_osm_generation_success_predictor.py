"""Train a post-generation pass/fail predictor for OSM candidate reranking.

This model learns from audit labels after Hunyuan generation. It is meant to be
used before future Hunyuan calls as a quality-aware reranker over retrieved
top-k candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.osm_candidate_quality_features import (  # noqa: E402
    FEATURE_NAMES,
    candidate_quality_features,
    record_geometry_features,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_generation_success_predictor")
    ap.add_argument("--val_fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--quality_weight", type=float, default=0.20)
    ap.add_argument("--bad_candidate_fail_count", type=int, default=2)
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


def selected_candidate(record: dict) -> dict | None:
    selected = str(record.get("selected_candidate_id", ""))
    for candidate in record.get("retrieval_candidates", []):
        if str(candidate.get("candidate_id", "")) == selected:
            return candidate
    return None


def label(record: dict) -> int:
    return 1 if record.get("quality", {}).get("status") == "pass" else 0


def feature_row(record: dict, candidate: dict) -> list[float]:
    return candidate_quality_features(
        str(record.get("class", "")),
        float(record.get("area_m2", 0.0) or 0.0),
        float(record.get("height_m", 0.0) or 0.0),
        record_geometry_features(record),
        candidate,
    )


def build_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    x, y, meta = [], [], []
    for record in records:
        candidate = selected_candidate(record)
        if candidate is None:
            continue
        x.append(feature_row(record, candidate))
        y.append(label(record))
        meta.append(record)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


def fail_counts(records: list[dict]) -> Counter:
    counts = Counter()
    for record in records:
        if label(record) == 0:
            counts[str(record.get("selected_candidate_id", ""))] += 1
    return counts


def candidate_success_prob(model: RandomForestClassifier, record: dict, candidate: dict) -> float:
    x = np.asarray([feature_row(record, candidate)], dtype=np.float32)
    return float(model.predict_proba(x)[0, 1])


def best_by_quality_aware(
    model: RandomForestClassifier,
    record: dict,
    bad_candidates: set[str],
    quality_weight: float,
) -> tuple[str, float, float]:
    best_id = ""
    best_combined = -1e18
    best_success = 0.0
    fallback = None
    for candidate in record.get("retrieval_candidates", []):
        candidate_id = str(candidate.get("candidate_id", ""))
        rerank_score = float(candidate.get("rerank_score", candidate.get("retrieval_score", 0.0)) or 0.0)
        success = candidate_success_prob(model, record, candidate)
        combined = rerank_score + quality_weight * success
        if candidate_id in bad_candidates:
            combined -= 1.0
        if fallback is None or rerank_score > fallback[1]:
            fallback = (candidate_id, rerank_score, success)
        if combined > best_combined:
            best_id = candidate_id
            best_combined = combined
            best_success = success
    if not best_id and fallback is not None:
        best_id, best_combined, best_success = fallback
    return best_id, best_combined, best_success


def best_by_rerank(record: dict) -> str:
    candidates = record.get("retrieval_candidates", [])
    if not candidates:
        return ""
    best = max(candidates, key=lambda c: float(c.get("rerank_score", c.get("retrieval_score", 0.0)) or 0.0))
    return str(best.get("candidate_id", ""))


def counterfactual_report(
    model: RandomForestClassifier,
    records: list[dict],
    bad_candidates: set[str],
    quality_weight: float,
) -> tuple[dict, list[dict]]:
    rows = []
    for record in records:
        selected = str(record.get("selected_candidate_id", ""))
        qa_id, qa_score, qa_success = best_by_quality_aware(model, record, bad_candidates, quality_weight)
        rerank_id = best_by_rerank(record)
        rows.append({
            "split": record.get("split", ""),
            "osm_id": record.get("osm_id", ""),
            "class": record.get("class", ""),
            "quality_status": record.get("quality", {}).get("status", ""),
            "selected_candidate_id": selected,
            "rerank_candidate_id": rerank_id,
            "quality_candidate_id": qa_id,
            "quality_combined_score": qa_score,
            "quality_predicted_success": qa_success,
            "changed_choice": int(qa_id != rerank_id),
            "selected_was_bad_candidate": int(selected in bad_candidates),
            "quality_choice_bad_candidate": int(qa_id in bad_candidates),
        })
    metrics = {
        "records": len(rows),
        "changed_choice_count": int(sum(r["changed_choice"] for r in rows)),
        "bad_selected_count": int(sum(r["selected_was_bad_candidate"] for r in rows)),
        "bad_quality_choice_count": int(sum(r["quality_choice_bad_candidate"] for r in rows)),
    }
    return metrics, rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [r for r in read_jsonl(Path(args.records)) if selected_candidate(r) is not None]
    train_records, val_records = split_records(records, args.val_fraction)
    x_train, y_train, _ = build_xy(train_records)
    x_val, y_val, _ = build_xy(val_records)
    if len(np.unique(y_train)) < 2:
        raise SystemExit("Need both pass and fail labels in training split.")

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1] if len(x_val) else np.asarray([])
    failures = fail_counts(records)
    bad_candidates = {c for c, n in failures.items() if n >= args.bad_candidate_fail_count}
    cf_metrics, cf_rows = counterfactual_report(model, records, bad_candidates, args.quality_weight)

    metrics = {
        "records": str(args.records),
        "record_count": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "pass_count": int(sum(label(r) for r in records)),
        "fail_count": int(len(records) - sum(label(r) for r in records)),
        "bad_candidates": sorted(bad_candidates),
        "failed_candidate_counts": dict(failures),
        "train_average_precision": float(average_precision_score(y_train, train_prob)),
        "val_average_precision": float(average_precision_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.0,
        "train_roc_auc": float(roc_auc_score(y_train, train_prob)) if len(np.unique(y_train)) > 1 else 0.0,
        "val_roc_auc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.0,
        "train_accuracy_at_0_5": float(accuracy_score(y_train, train_prob >= 0.5)),
        "val_accuracy_at_0_5": float(accuracy_score(y_val, val_prob >= 0.5)) if len(y_val) else 0.0,
        "counterfactual": cf_metrics,
        "top_feature_importance": sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:10],
    }

    with (out_dir / "generation_success_model.pkl").open("wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": FEATURE_NAMES,
            "bad_candidates": sorted(bad_candidates),
            "quality_weight": args.quality_weight,
        }, f)
    (out_dir / "generation_success_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out_dir / "quality_aware_counterfactual.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cf_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cf_rows)
    with (out_dir / "feature_importance.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name, value in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda item: item[1], reverse=True):
            writer.writerow([name, f"{value:.8f}"])

    report = [
        "# OSM Generation Success Predictor",
        "",
        f"Records: {len(records)}",
        f"Pass: {metrics['pass_count']}",
        f"Fail: {metrics['fail_count']}",
        f"Train records: {len(train_records)}",
        f"Validation records: {len(val_records)}",
        "",
        "## Validation",
        f"- average precision: {metrics['val_average_precision']:.3f}",
        f"- ROC AUC: {metrics['val_roc_auc']:.3f}",
        f"- accuracy at 0.5: {metrics['val_accuracy_at_0_5']:.3f}",
        "",
        "## Bad Candidates",
        *[f"- {c}: {failures[c]} failures" for c in sorted(bad_candidates)],
        "",
        "## Counterfactual Quality-Aware Rerank",
        f"- changed choices: {cf_metrics['changed_choice_count']} / {cf_metrics['records']}",
        f"- bad selected candidates in corpus: {cf_metrics['bad_selected_count']}",
        f"- bad candidates selected by quality-aware policy: {cf_metrics['bad_quality_choice_count']}",
        "",
        "## Top Features",
        *[f"- {name}: {value:.4f}" for name, value in metrics["top_feature_importance"]],
    ]
    (out_dir / "generation_success_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"[success] model:  {out_dir / 'generation_success_model.pkl'}")
    print(f"[success] report: {out_dir / 'generation_success_report.md'}")


if __name__ == "__main__":
    main()
