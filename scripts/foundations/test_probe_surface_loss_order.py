"""Contract tests for #89's recovered surface-loss order probe.

The public seam is the JSON-ready report emitted by the probe.  The expensive measurement uses the
real denoiser and Dora decoder; these tests pin how its observed losses become a ticket decision
without loading either model.
"""
from __future__ import annotations

import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.foundations.probe_surface_loss_order import build_report


class TestSurfaceLossOrderReport(unittest.TestCase):
    def test_relative_spread_is_reported_for_every_timestep(self):
        report = build_report({
            "0.40": {
                "epsilon": [1.00, 1.02, 1.01],
                "surface": [2.000000, 2.000002, 2.000001],
            },
            "0.70": {
                "epsilon": [0.80, 0.81, 0.805],
                "surface": [1.500000, 1.500003, 1.500001],
            },
        })

        self.assertEqual(list(report["by_t"]), ["0.40", "0.70"])
        self.assertAlmostEqual(report["by_t"]["0.40"]["epsilon_spread_pct"],
                               100 * 0.02 / 1.01)
        self.assertAlmostEqual(report["by_t"]["0.40"]["surface_spread_pct"],
                               100 * 0.000002 / 2.000001)
        self.assertEqual(report["by_t"]["0.40"]["n_orderings"], 3)

    def test_report_preserves_raw_observations_and_marks_a_consistent_result(self):
        observed = {
            "0.40": {"epsilon": [1.0, 1.02], "surface": [2.0, 2.000002]},
            "0.55": {"epsilon": [1.0, 1.01], "surface": [2.0, 2.000004]},
        }

        report = build_report(observed)

        self.assertEqual(report["observed"], observed)
        self.assertTrue(report["surface_is_at_least_100x_less_order_sensitive_at_every_t"])
        self.assertGreater(report["by_t"]["0.40"]["sensitivity_ratio"], 100)

    def test_empty_or_misaligned_observations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two orderings"):
            build_report({"0.40": {"epsilon": [1.0], "surface": [1.0]}})
        with self.assertRaisesRegex(ValueError, "same number"):
            build_report({"0.40": {"epsilon": [1.0, 2.0], "surface": [1.0, 2.0, 3.0]}})


class TestSurfaceLossOrderCli(unittest.TestCase):
    def test_an_existing_observation_artifact_can_be_resummarised_without_models(self):
        observed = {"0.40": {"epsilon": [1.0, 1.02], "surface": [2.0, 2.000002]}}
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "observed.json"
            output = Path(td) / "report.json"
            source.write_text(json.dumps({"observed": observed}))

            completed = subprocess.run([
                sys.executable,
                "scripts/foundations/probe_surface_loss_order.py",
                "--measurements", str(source),
                "--out", str(output),
            ], check=False, capture_output=True, text=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["observed"], observed)
            self.assertTrue(report["surface_is_at_least_100x_less_order_sensitive_at_every_t"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
