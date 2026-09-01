"""Build #92's final, machine-checkable 2x2 scorecard from four full-heldout artifacts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

ARM_LABELS = {
    "A": "encoded + surface",
    "B": "aligned + surface",
    "C": "encoded + no surface",
    "D": "aligned + no surface",
}
BAR = {
    "vs_input": ("<", 0.98),
    "vol_iou": (">=", 0.876),
    "beats_envelope_rate": (">", 0.05),
}


def _step(artifact: Mapping) -> int | None:
    match = re.search(r"step(\d+)", str(artifact.get("meta", {}).get("a2", "")))
    return int(match.group(1)) if match else None


def _bar(row: Mapping[str, float]) -> tuple[bool, list[str]]:
    failed = []
    for metric, (operator, threshold) in BAR.items():
        value = float(row[metric])
        passed = (value < threshold if operator == "<" else
                  value >= threshold if operator == ">=" else
                  value > threshold)
        if not passed:
            failed.append(f"{metric} {operator} {threshold}")
    return not failed, failed


def _selected(artifact: Mapping) -> dict:
    candidates = []
    for name, row in artifact["summary"].items():
        if not name.startswith("a2_s") or not row:
            continue
        strength = float(name.removeprefix("a2_s"))
        candidates.append((float(row["vol_iou"]), -strength, strength, row))
    if not candidates:
        raise ValueError("artifact has no a2 strength rows")
    _, _, strength, row = max(candidates)
    passed, failed = _bar(row)
    return {
        "checkpoint": artifact.get("meta", {}).get("a2"),
        "step": _step(artifact),
        "strength": strength,
        "evaluated_strengths": sorted(candidate[2] for candidate in candidates),
        **row,
        "bar_met": passed,
        "failed_clauses": failed,
    }


def _delta(left: Mapping, right: Mapping) -> dict:
    metrics = ("fp_iou", "missing", "extra", "vol_iou", "collapse_rate",
               "beats_envelope_rate", "vs_input")
    return {metric: float(left[metric]) - float(right[metric])
            for metric in metrics if metric in left and metric in right}


def _factorial(rows: Mapping[str, Mapping]) -> dict:
    return {
        "surface_marginal": {
            "encoded": _delta(rows["A"], rows["C"]),
            "aligned": _delta(rows["B"], rows["D"]),
        },
        "alignment_effect": {
            "with_surface": _delta(rows["B"], rows["A"]),
            "without_surface": _delta(rows["D"], rows["C"]),
        },
    }


def summarize_matched_curves(curves: Mapping[str, list[Mapping]], strength: float) -> dict:
    """Compute causal 2x2 contrasts only where step and strength are held fixed."""
    if set(curves) != set(ARM_LABELS):
        raise ValueError("the matched 2x2 requires exactly curves A, B, C, and D")
    indexed = {}
    for arm, curve in curves.items():
        by_step = {int(row["step"]): row for row in curve}
        if len(by_step) != len(curve):
            raise ValueError(f"arm {arm} repeats a checkpoint step")
        indexed[arm] = by_step
    step_sets = [set(indexed[arm]) for arm in "ABCD"]
    if any(steps != step_sets[0] for steps in step_sets[1:]):
        raise ValueError("all four curves must contain the same checkpoint steps")
    steps = sorted(step_sets[0])
    if not steps:
        raise ValueError("matched curves are empty")

    by_step = {}
    for step in steps:
        rows = {arm: indexed[arm][step] for arm in "ABCD"}
        populations = {int(row["n"]) for row in rows.values()}
        if len(populations) != 1:
            raise ValueError(f"step {step} was not scored on the same population size")
        by_step[str(step)] = {
            "n": populations.pop(),
            "arms": rows,
            **_factorial(rows),
        }
    endpoint_step = steps[-1]
    return {
        "strength": float(strength),
        "steps": steps,
        "endpoint_step": endpoint_step,
        "by_step": by_step,
        "endpoint": by_step[str(endpoint_step)],
    }


def summarize_arms(arms: Mapping[str, Mapping]) -> dict:
    """Select each arm at its best median-IoU strength and evaluate the fixed AND bar."""
    if set(arms) != set(ARM_LABELS):
        raise ValueError("the 2x2 requires exactly arms A, B, C, and D")
    ids = list(arms["A"]["ids"])
    if any(list(arms[arm]["ids"]) != ids for arm in "BCD"):
        raise ValueError("all four arms must use the same fixed ids in the same order")

    selected = {arm: _selected(arms[arm]) for arm in "ABCD"}
    factorial = _factorial(selected)
    return {
        "n": len(ids),
        "ids": ids,
        "bar": {metric: f"{operator} {threshold}"
                for metric, (operator, threshold) in BAR.items()},
        "candidate_bar_arm": "B",
        "arm_labels": ARM_LABELS,
        "selected": selected,
        "selection_warning": (
            "The optimized contrasts are descriptive deployment comparisons, not pure factorial "
            "effects when arms select different checkpoints or strengths. Use matched_curves for "
            "causal A-C, B-D, B-A, and D-C claims."
        ),
        **factorial,
        "candidate_met_bar": selected["B"]["bar_met"],
        "any_arm_met_bar": any(row["bar_met"] for row in selected.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in "ABCD":
        parser.add_argument(f"--{arm}", required=True, help=f"full-heldout strength artifact for {arm}")
        parser.add_argument(f"--curve-{arm}", dest=f"curve_{arm}",
                            help=f"optional common-strength checkpoint curve for {arm}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    artifacts = {arm: json.loads(Path(getattr(args, arm)).read_text()) for arm in "ABCD"}
    report = summarize_arms(artifacts)
    curve_paths = {arm: getattr(args, f"curve_{arm}") for arm in "ABCD"}
    if any(curve_paths.values()):
        if not all(curve_paths.values()):
            raise SystemExit("provide all four --curve-A/--curve-B/--curve-C/--curve-D paths")
        curves = {arm: json.loads(Path(path).read_text()) for arm, path in curve_paths.items()}
        report["matched_curves"] = summarize_matched_curves(curves, strength=0.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
