"""#188: contract for the pre-registered bar and the verdict that reads it.

Synthetic; no GPU, no corpus, no checkpoint. The point of these tests is that the verdict is a pure
function of (measured summary, registration) -- so it cannot be nudged after the numbers are in,
and a clause that goes missing fails loudly instead of quietly passing.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.a2_buildingworld_bar import (  # noqa: E402
    build_registration, clause_value, score_registration,
)

SUMMARY = {
    "blockout": {"n": 900, "fp_iou": 1.0, "missing": 0.0, "extra": 0.080, "vol_iou": 0.925,
                 "collapse_rate": 0.0},
    "codec_ceiling": {"n": 900, "fp_iou": 0.997, "missing": 0.001, "extra": 0.001, "vol_iou": 0.998,
                      "collapse_rate": 0.0},
    "a2_s0.5": {"n": 900, "fp_iou": 0.955, "missing": 0.003, "extra": 0.090, "vol_iou": 0.860,
                "collapse_rate": 0.10, "vs_input": 0.965},
}
EXTRAS = {"generation_failures": 0, "buckets_scored": 6}


def _registration(**over):
    reg = build_registration(
        reference={"blockout": SUMMARY["blockout"], "codec_ceiling": SUMMARY["codec_ceiling"]},
        frozen_ratios={"vol_iou": 0.9240, "extra": 1.2269},
        operating_point={"strength": 0.5, "steps": 20, "guidance": 1.0},
        gate={"n": 900, "digest": "deadbeef", "buckets": 6},
    )
    reg.update(over)
    return reg


class TestClauseValue(unittest.TestCase):
    def test_reads_a_plain_arm_metric(self):
        clause = {"source": "summary", "arm": "a2_s0.5", "metric": "vol_iou"}
        self.assertAlmostEqual(clause_value(clause, SUMMARY, EXTRAS), 0.860)

    def test_reads_a_ratio_against_a_reference_arm(self):
        clause = {"source": "ratio", "arm": "a2_s0.5", "metric": "vol_iou",
                  "relative_to": "blockout"}
        self.assertAlmostEqual(clause_value(clause, SUMMARY, EXTRAS), 0.860 / 0.925)

    def test_reads_a_fact_that_is_not_a_summary_metric(self):
        clause = {"source": "extra", "metric": "generation_failures"}
        self.assertEqual(clause_value(clause, SUMMARY, EXTRAS), 0)

    def test_a_missing_metric_raises_rather_than_scoring_zero(self):
        """A clause that silently evaluates to 0 would pass a `<=` bar it was never measured against."""
        clause = {"source": "summary", "arm": "a2_s0.5", "metric": "not_measured"}
        with self.assertRaises(KeyError):
            clause_value(clause, SUMMARY, EXTRAS)

    def test_a_missing_arm_raises(self):
        clause = {"source": "summary", "arm": "a2_s0.9", "metric": "vol_iou"}
        with self.assertRaises(KeyError):
            clause_value(clause, SUMMARY, EXTRAS)

    def test_a_zero_reference_raises_rather_than_dividing(self):
        summary = dict(SUMMARY, blockout=dict(SUMMARY["blockout"], extra=0.0))
        clause = {"source": "ratio", "arm": "a2_s0.5", "metric": "extra", "relative_to": "blockout"}
        with self.assertRaises(ZeroDivisionError):
            clause_value(clause, summary, EXTRAS)


class TestVerdict(unittest.TestCase):
    def test_a_model_meeting_every_clause_passes(self):
        got = score_registration(SUMMARY, _registration(), arm="a2_s0.5", extras=EXTRAS)
        self.assertEqual(got["verdict"], "PASS", got["clauses"])

    def test_a_failed_kill_clause_kills_even_when_every_bar_passes(self):
        """#183's rule: a KILL is not outvoted by the clauses that passed."""
        extras = dict(EXTRAS, generation_failures=3)
        got = score_registration(SUMMARY, _registration(), arm="a2_s0.5", extras=extras)
        self.assertEqual(got["verdict"], "KILL")
        self.assertFalse(got["clear_kill"])

    def test_a_bucket_that_never_generated_is_a_kill(self):
        extras = dict(EXTRAS, buckets_scored=5)
        got = score_registration(SUMMARY, _registration(), arm="a2_s0.5", extras=extras)
        self.assertEqual(got["verdict"], "KILL")

    def test_a_failed_bar_is_not_met_rather_than_killed(self):
        summary = dict(SUMMARY, **{"a2_s0.5": dict(SUMMARY["a2_s0.5"], vol_iou=0.50)})
        got = score_registration(summary, _registration(), arm="a2_s0.5", extras=EXTRAS)
        self.assertEqual(got["verdict"], "NOT MET")
        self.assertTrue(got["clear_kill"], "no KILL clause failed, so the kill line is still clear")

    def test_collapse_past_the_registered_ceiling_is_a_kill(self):
        """#92's aligned retrain reached 0.4636 collapse; that is the failure this clause names."""
        summary = dict(SUMMARY, **{"a2_s0.5": dict(SUMMARY["a2_s0.5"], collapse_rate=0.46)})
        got = score_registration(summary, _registration(), arm="a2_s0.5", extras=EXTRAS)
        self.assertEqual(got["verdict"], "KILL")

    def test_informational_clauses_never_change_the_verdict(self):
        """#119 ruled that no gate verdict is scored on the legacy pinned 714."""
        reg = _registration()
        reg["clauses"].append({"id": "I9", "kind": "INFORMATIONAL", "source": "summary",
                               "arm": "a2_s0.5", "metric": "fp_iou", "rule": ">=",
                               "threshold": 2.0, "rationale": "impossible on purpose"})
        got = score_registration(SUMMARY, reg, arm="a2_s0.5", extras=EXTRAS)
        self.assertEqual(got["verdict"], "PASS")
        self.assertFalse(next(c for c in got["clauses"] if c["id"] == "I9")["pass"])

    def test_an_unmeasured_informational_metric_does_not_block_the_verdict(self):
        """Informational clauses never decide anything, so their absence must not stop scoring --
        but it has to stay visible rather than being reported as a pass."""
        summary = dict(SUMMARY)
        summary["a2_s0.5"] = {k: v for k, v in SUMMARY["a2_s0.5"].items() if k != "vs_input"}
        got = score_registration(summary, _registration(), arm="a2_s0.5", extras=EXTRAS)
        self.assertEqual(got["verdict"], "PASS")
        i1 = next(c for c in got["clauses"] if c["id"] == "I1")
        self.assertIsNone(i1["value"])
        self.assertFalse(i1["pass"])

    def test_an_unmeasured_scored_metric_still_raises(self):
        """The same leniency must NOT extend to a clause that decides the verdict."""
        summary = dict(SUMMARY)
        summary["a2_s0.5"] = {k: v for k, v in SUMMARY["a2_s0.5"].items() if k != "vol_iou"}
        with self.assertRaises(KeyError):
            score_registration(summary, _registration(), arm="a2_s0.5", extras=EXTRAS)

    def test_every_clause_is_reported_with_its_measured_value(self):
        got = score_registration(SUMMARY, _registration(), arm="a2_s0.5", extras=EXTRAS)
        self.assertEqual(len(got["clauses"]), len(_registration()["clauses"]))
        for clause in got["clauses"]:
            self.assertIn("value", clause)
            self.assertIn("threshold", clause)
            self.assertIn("pass", clause)

    def test_scoring_an_arm_the_run_did_not_produce_raises(self):
        with self.assertRaises(KeyError):
            score_registration(SUMMARY, _registration(), arm="a2_s0.7", extras=EXTRAS)


class TestRegistrationShape(unittest.TestCase):
    def test_records_the_operating_point_119_will_seal(self):
        reg = _registration()
        self.assertEqual(reg["operating_point"],
                         {"strength": 0.5, "steps": 20, "guidance": 1.0})

    def test_carries_a_kill_clause_and_a_bar_clause(self):
        kinds = {c["kind"] for c in _registration()["clauses"]}
        self.assertIn("KILL", kinds)
        self.assertIn("BAR", kinds)

    def test_every_clause_states_why_it_exists(self):
        """A threshold with no recorded rationale is a number someone can argue with later."""
        for clause in _registration()["clauses"]:
            self.assertTrue(clause.get("rationale"), f"{clause['id']} has no rationale")

    def test_clause_ids_are_unique(self):
        ids = [c["id"] for c in _registration()["clauses"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_bar_does_not_require_beating_the_envelope(self):
        """Disclosed on purpose: the source being REPLACED does not beat it either.

        The frozen `vecset_v4_surf` @240k scores vol_iou 0.8625 against its own population's
        blockout at 0.9334, and wins on only 0.56% of rows. A bar demanding the retrain beat the
        envelope would be a bar the sealed source itself fails, which would make #188 a new quality
        claim rather than the substitution #115's amendment describes.
        """
        reg = _registration()
        ratio = next(c for c in reg["clauses"]
                     if c.get("metric") == "vol_iou" and c["kind"] == "BAR")
        self.assertLess(ratio["threshold"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
