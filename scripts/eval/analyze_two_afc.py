"""Ticket 17: analyze responses from the two-AFC study prototype (two_afc_study.py).

Fixed analysis, decided before any collection (PRD story 34: "the two-AFC protocol and analysis
fixed before collecting responses, so that human preference is not analyzed post hoc"): un-blind
each response against the answer key, count how often "decomposition" was picked over "monolith",
and report a Wilson 95% CI against the 50% no-preference null. Human preference is reported
alongside detail FID, not as a replacement for it (PRD: "supports ... does not replace the
automated metric").

Out: execution/artifacts/two_afc_result.json
Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/eval/analyze_two_afc.py --responses <path/to/responses.json>
        [--answer-key execution/artifacts/two_afc_answer_key.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from two_afc_study import two_afc_result  # noqa: E402


def load_responses(path):
    """Participant-exported responses.json: {building_id: "left"|"right"}."""
    data = json.loads(Path(path).read_text())
    for bid, side in data.items():
        if side not in ("left", "right"):
            raise ValueError(f"response for {bid!r} is {side!r}, expected 'left' or 'right'")
    return data


def load_answer_key(path):
    """The NOT-participant-facing manifest two_afc_study.py writes; pulls out just the list
    `two_afc_result` needs."""
    return json.loads(Path(path).read_text())["answer_key"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--answer-key", default=str(REPO / "execution/artifacts/two_afc_answer_key.json"))
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/two_afc_result.json"))
    a = ap.parse_args()

    responses = load_responses(a.responses)
    answer_key = load_answer_key(a.answer_key)
    result = two_afc_result(responses, answer_key)

    print(f"[*] {result['n']} of {len(answer_key)} pairs answered"
          + (f" ({len(result['missing_ids'])} unknown ids ignored)" if result["missing_ids"] else ""))
    if result["n"]:
        lo, hi = result["ci95"]
        print(f"[result] decomposition preferred {result['n_preferred_decomposition']}/{result['n']} "
              f"= {result['proportion']:.2f}  ci95=[{lo:.2f},{hi:.2f}]  "
              f"significant_vs_chance={result['significant_vs_chance']}")
    else:
        print("[result] no usable responses")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
