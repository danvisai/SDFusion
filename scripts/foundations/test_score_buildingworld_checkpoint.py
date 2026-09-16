"""Contract tests for #183's second gate. Synthetic, fast, no corpus, no GPU -- the real
end-to-end path (predict against a live checkpoint) is validated by actually running it, the same
way buildingworld_baseline.py's own `run()` has no direct unit test either.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations import score_buildingworld_checkpoint as gate  # noqa: E402


class TestRunRejectsAWrongObjective(unittest.TestCase):
    def test_a_non_ce_checkpoint_is_rejected_before_any_prediction_runs(self):
        with patch.object(gate, "load_population", return_value={"query": [1, 2],
                          "city": ["Berlin", "Tokyo"], "family": ["flat", "gable"]}), \
             patch.object(gate, "build_held", return_value={}), \
             patch.object(gate, "load_checkpoint", return_value={"objective": "program"}), \
             patch.object(gate, "predict") as mock_predict:
            with self.assertRaisesRegex(ValueError, "'program'"):
                gate.run(Path("whatever.pt"))
            mock_predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
