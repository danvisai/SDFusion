"""Contract tests for #148's formalized human-evaluation rubric. Synthetic, fast, no GPU.

Pins the three acceptance criteria directly: the rubric is fixed and importable rather than
re-typed ad hoc, a verdict can be recorded against a real #147 visual carving trace (not just an
arbitrary path -- the integration test below produces a genuine trace via `carving_trace` first),
and the rubric/verdict wording keeps a human judgement distinct from any automated metric.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_human_eval_rubric.py
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scene.sdf_edit import footprint_envelope_sdf, layer_program_to_ops, mask_to_rings  # noqa: E402
from scripts.foundations.carving_trace import render_carving_trace, save_carving_trace  # noqa: E402
from scripts.foundations.human_eval_rubric import (  # noqa: E402
    RUBRIC, Verdict, load_verdict, record_verdict,
)

RES = 16


def _real_trace_dir(tmp: Path) -> Path:
    """A genuine #147 visual carving trace on disk -- not a synthetic stand-in -- so tests here
    prove real integration with #147's own output, per #148's acceptance criterion."""
    fp = np.zeros((RES, RES), bool)
    fp[3:12, 2:13] = True
    y0, y1 = 2, 10
    base = footprint_envelope_sdf(fp, y0, y1, res=RES)
    region = np.zeros((RES, RES), bool)
    region[6:10, 4:9] = True
    region &= fp
    program = [dict(op="Layer", height=4, area=int(region.sum()), components=1,
                    region=[r.tolist() for r in mask_to_rings(region)])]
    ops = layer_program_to_ops(program, fp, y0, y1, res=RES)
    trace = render_carving_trace(base, ops, fp, res=RES)
    trace_dir = tmp / "building_42"
    save_carving_trace(trace, trace_dir)
    return trace_dir


class TestRubricIsFixed(unittest.TestCase):
    """#148's first criterion: a fixed, reusable artifact, not re-invented ad hoc."""

    def test_the_rubric_has_exactly_the_three_questions_the_ticket_states(self):
        self.assertEqual([q.text for q in RUBRIC], [
            "Does it look like a building?",
            "Are there visible geometric artifacts?",
            "Does the edit match what was requested?",
        ])

    def test_record_verdicts_keyword_names_track_rubric_exactly(self):
        """A structural check, not just a runtime one: `record_verdict`'s own signature is
        inspected directly against `RUBRIC`'s keys, so the two cannot silently drift apart --
        this fails at test time, not only when someone happens to call the function."""
        import scripts.foundations.human_eval_rubric as mod
        params = inspect.signature(mod.record_verdict).parameters
        keyword_only = [name for name, p in params.items()
                        if p.kind is inspect.Parameter.KEYWORD_ONLY and name != "notes"]
        self.assertEqual(set(keyword_only), {q.key for q in RUBRIC})

    def test_record_verdict_requires_exactly_the_rubric_keys_by_keyword(self):
        """A caller cannot skip a question, misspell one, or invent a fourth -- Python's own
        keyword-argument enforcement is what makes the rubric fixed 'by construction'."""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeError):
                record_verdict(d, "1", "r", looks_like_a_building=True)   # missing two
            with self.assertRaises(TypeError):
                record_verdict(d, "1", "r", looks_like_a_building=True, has_visible_artifacts=True,
                               edit_matches_request=True, an_invented_fourth_question=True)


class TestRecordingAgainstARealTrace(unittest.TestCase):
    """#148's second criterion: a verdict recorded against a specific building's real #147
    visual carving trace output."""

    def test_recording_against_a_directory_with_no_trace_is_refused(self):
        """The verdict and the evidence it answers about must genuinely travel together -- a
        directory holding no rendered #147 frames is refused outright, not silently accepted
        with an empty verdict pointing at nothing."""
        with tempfile.TemporaryDirectory() as d:
            empty_dir = Path(d) / "no_trace_here"
            with self.assertRaises(ValueError):
                record_verdict(empty_dir, building_id="1", rater="r", looks_like_a_building=True,
                               has_visible_artifacts=False, edit_matches_request=True)
            self.assertFalse(empty_dir.exists(), "a refused verdict must not create the directory")

    def test_a_verdict_is_written_inside_the_traces_own_directory(self):
        with tempfile.TemporaryDirectory() as d:
            trace_dir = _real_trace_dir(Path(d))
            before = sorted(p.name for p in trace_dir.iterdir())
            self.assertTrue(any(name.startswith("step0_view") for name in before))

            path = record_verdict(trace_dir, building_id="42", rater="dan",
                                  looks_like_a_building=True, has_visible_artifacts=False,
                                  edit_matches_request=True, notes="clean setback")
            self.assertEqual(path.parent, trace_dir)
            self.assertIn("verdict.json", {p.name for p in trace_dir.iterdir()})
            # the trace frames themselves are untouched by recording a verdict
            after = sorted(p.name for p in trace_dir.iterdir() if p.name != "verdict.json")
            self.assertEqual(before, after)

    def test_no_verdict_is_none_until_one_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            trace_dir = _real_trace_dir(Path(d))
            self.assertIsNone(load_verdict(trace_dir))

    def test_the_recorded_verdict_round_trips_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            trace_dir = _real_trace_dir(Path(d))
            record_verdict(trace_dir, building_id="42", rater="dan", looks_like_a_building=True,
                           has_visible_artifacts=False, edit_matches_request=True,
                           notes="clean setback")
            v = load_verdict(trace_dir)
            self.assertIsInstance(v, Verdict)
            self.assertEqual(v.building_id, "42")
            self.assertEqual(v.rater, "dan")
            self.assertEqual(v.notes, "clean setback")
            self.assertEqual(v.answers, dict(looks_like_a_building=True,
                                             has_visible_artifacts=False,
                                             edit_matches_request=True))

    def test_a_second_verdict_overwrites_the_first(self):
        """One verdict per trace -- matching #127's own single-verdict precedent, not
        aggregating several raters (not asked for here)."""
        with tempfile.TemporaryDirectory() as d:
            trace_dir = _real_trace_dir(Path(d))
            record_verdict(trace_dir, building_id="42", rater="first", looks_like_a_building=True,
                           has_visible_artifacts=False, edit_matches_request=True)
            record_verdict(trace_dir, building_id="42", rater="second", looks_like_a_building=False,
                           has_visible_artifacts=True, edit_matches_request=False)
            v = load_verdict(trace_dir)
            self.assertEqual(v.rater, "second")
            self.assertFalse(v.answers["looks_like_a_building"])

    def test_the_stored_file_is_plain_readable_json(self):
        with tempfile.TemporaryDirectory() as d:
            trace_dir = _real_trace_dir(Path(d))
            path = record_verdict(trace_dir, building_id="42", rater="dan",
                                  looks_like_a_building=True, has_visible_artifacts=False,
                                  edit_matches_request=True)
            raw = json.loads(path.read_text())
            self.assertEqual(raw["building_id"], "42")
            self.assertEqual(set(raw["answers"]), {q.key for q in RUBRIC})


class TestJudgementIsDistinctFromAutomatedMetrics(unittest.TestCase):
    """#148's third criterion: the rubric's own wording distinguishes a human's judgement from
    any automated metric already computed for the same building."""

    def test_the_module_documents_the_distinction_from_automated_metrics(self):
        import scripts.foundations.human_eval_rubric as mod
        text = (mod.__doc__ or "") + (Verdict.__doc__ or "")
        for term in ("judgement", "metric"):
            self.assertIn(term, text.lower(), f"expected {term!r} in the module's own wording")

    def test_a_verdict_carries_no_automated_metric_fields(self):
        """A Verdict's own schema has no room for a metric to sneak in disguised as a judgement --
        only building_id, rater, the fixed answers, and free-text notes."""
        self.assertEqual(set(Verdict.__dataclass_fields__),
                         {"building_id", "rater", "answers", "notes"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
