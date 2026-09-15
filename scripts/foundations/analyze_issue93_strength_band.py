"""Read #93's strength sweep as a per-footprint editing band, not one aggregate scalar."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

NO_OP_VS_INPUT = 0.98
COLLAPSE_MISSING = 0.15


def _strength(name: str) -> float:
    return float(name.removeprefix("a2_s"))


def _state(candidate: Mapping[str, float], envelope: Mapping[str, float]) -> str:
    if float(candidate["vs_input"]) >= NO_OP_VS_INPUT:
        return "no_op"
    if float(candidate["missing"]) >= COLLAPSE_MISSING:
        return "collapsed"
    if float(candidate["vol_iou"]) > float(envelope["vol_iou"]):
        return "net_positive"
    return "net_negative"


def analyze_strength_band(artifact: Mapping) -> dict:
    """Return the discrete usable-strength range for every pinned building.

    A usable edit must satisfy all three concepts already fixed by map #87: it moves (`vs_input`),
    does not collapse the building (`missing`), and improves quality against that same building's
    footprint envelope.  The sampled strengths remain explicit because a min/max pair alone can hide
    a hole in a non-monotonic band.
    """
    ids = [str(value) for value in artifact["ids"]]
    arms = artifact["per_building"]
    envelope = arms["blockout"]
    strength_arms = sorted(
        ((float(_strength(name)), name, rows) for name, rows in arms.items()
         if name.startswith("a2_s")),
        key=lambda item: item[0],
    )
    if not strength_arms:
        raise ValueError("artifact has no A2 strength arms")
    expected = set(ids)
    for _, name, rows in strength_arms:
        missing = expected - set(rows)
        if missing:
            raise ValueError(f"{name} is missing building ids: {sorted(missing)[:5]}")
    if expected - set(envelope):
        raise ValueError("blockout is missing building ids")

    per_building = {}
    strength_counts = {format(strength, "g"): Counter() for strength, _, _ in strength_arms}
    for building_id in ids:
        states = {}
        usable = []
        for strength, _, rows in strength_arms:
            key = format(strength, "g")
            state = _state(rows[building_id], envelope[building_id])
            states[key] = state
            strength_counts[key][state] += 1
            if state == "net_positive":
                usable.append(strength)
        per_building[building_id] = {
            "usable_strengths": usable,
            "usable_range": [min(usable), max(usable)] if usable else None,
            "state_by_strength": states,
        }

    with_band = sum(bool(row["usable_strengths"]) for row in per_building.values())
    n = len(ids)
    return {
        "n": n,
        "definitions": {
            "no_op": f"vs_input >= {NO_OP_VS_INPUT}",
            "collapsed": f"missing >= {COLLAPSE_MISSING}",
            "net_positive": "acted, did not collapse, and vol_iou > the same building's blockout",
        },
        "strengths": [strength for strength, _, _ in strength_arms],
        "band_exists_count": with_band,
        "band_exists_rate": with_band / n if n else 0.0,
        "by_strength": {
            strength: {
                "states": dict(counts),
                "usable_rate": counts["net_positive"] / n if n else 0.0,
            }
            for strength, counts in strength_counts.items()
        },
        "per_building": per_building,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = analyze_strength_band(json.loads(Path(args.artifact).read_text()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
