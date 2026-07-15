"""Contract tests for ticket 17's two-AFC analysis CLI (analyze_two_afc.py).

Fast + data-free: exercises load_responses/main's pure seam over small synthetic JSON files, no
rendering, no GPU.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/eval/test_analyze_two_afc.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/eval
import analyze_two_afc as az  # noqa: E402


class LoadResponsesTest(unittest.TestCase):
    def test_reads_a_plain_building_to_side_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "responses.json"
            p.write_text(json.dumps({"b0": "left", "b1": "right"}))
            out = az.load_responses(p)
            self.assertEqual(out, {"b0": "left", "b1": "right"})

    def test_rejects_a_value_that_is_not_left_or_right(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "responses.json"
            p.write_text(json.dumps({"b0": "sideways"}))
            with self.assertRaises(ValueError):
                az.load_responses(p)


class LoadAnswerKeyTest(unittest.TestCase):
    def test_reads_the_answer_key_list_out_of_the_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "answer_key.json"
            p.write_text(json.dumps({
                "n_pairs": 1, "seed": 0,
                "answer_key": [{"building": "b0", "left": "monolith", "right": "decomposition"}],
            }))
            out = az.load_answer_key(p)
            self.assertEqual(out, [{"building": "b0", "left": "monolith", "right": "decomposition"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
