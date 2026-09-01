"""Contract tests for #94's gradient-conflict probe between the epsilon and surface loss terms.

The public seam is `summarize_conflict`: it turns paired per-row gradient measurements into #94's
candidate-discriminating report without loading the denoiser or the Dora decoder.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.foundations.probe_surface_gradient_conflict import summarize_conflict


def _row(row_id, encoded_cosine, aligned_cosine, eps_norm=1.0, surf_norm=1.0):
    def regime(cosine):
        return {"cosine": cosine, "eps_grad_norm": eps_norm, "surf_grad_norm": surf_norm,
                "eps_loss": 0.1, "surf_loss": 0.2}
    return {"row": row_id, "encoded": regime(encoded_cosine), "aligned": regime(aligned_cosine)}


class TestSummarizeConflict(unittest.TestCase):
    def test_paired_delta_is_computed_per_row_not_from_separate_aggregates(self):
        observed = {"0.40": [_row(1, -0.1, -0.6), _row(2, 0.2, -0.3)]}

        report = summarize_conflict(observed)

        row = report["by_t"]["0.40"]
        self.assertAlmostEqual(row["encoded_cosine_mean"], 0.05)
        self.assertAlmostEqual(row["aligned_cosine_mean"], -0.45)
        self.assertAlmostEqual(row["delta_cosine_mean"], -0.5)
        self.assertEqual(row["n_more_conflicting_aligned"], 2)

    def test_flags_require_every_measured_t_to_agree(self):
        observed = {
            "0.40": [_row(1, -0.1, -0.6)],
            "0.55": [_row(1, -0.1, 0.4)],  # aligned is LESS conflicting here
        }

        report = summarize_conflict(observed)

        self.assertFalse(report["aligned_more_conflicting_at_every_t"])

    def test_gradients_oppose_flag_needs_negative_cosine_in_both_regimes(self):
        self.assertTrue(
            summarize_conflict({"0.40": [_row(1, -0.2, -0.3)]})["gradients_oppose_on_average"])
        self.assertFalse(
            summarize_conflict({"0.40": [_row(1, 0.2, -0.3)]})["gradients_oppose_on_average"])

    def test_norm_ratio_is_averaged_per_regime(self):
        observed = {"0.40": [_row(1, -0.1, -0.1, eps_norm=2.0, surf_norm=1.0)]}

        row = summarize_conflict(observed)["by_t"]["0.40"]

        self.assertAlmostEqual(row["encoded_norm_ratio_mean"], 0.5)
        self.assertAlmostEqual(row["aligned_norm_ratio_mean"], 0.5)

    def test_rejects_rows_missing_a_regime(self):
        observed = {"0.40": [{"row": 1, "encoded": {"cosine": 0.0, "eps_grad_norm": 1.0,
                                                      "surf_grad_norm": 1.0}}]}
        with self.assertRaisesRegex(ValueError, "missing regimes"):
            summarize_conflict(observed)

    def test_rejects_a_timestep_with_no_rows(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            summarize_conflict({"0.40": []})


class TestSurfaceGradientConflictCli(unittest.TestCase):
    def test_an_existing_observation_artifact_can_be_resummarised_without_models(self):
        observed = {"0.40": [_row(1, -0.1, -0.4)]}
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "observed.json"
            output = Path(td) / "report.json"
            source.write_text(json.dumps({"observed": observed}))

            completed = subprocess.run([
                sys.executable,
                "scripts/foundations/probe_surface_gradient_conflict.py",
                "--measurements", str(source),
                "--out", str(output),
            ], check=False, capture_output=True, text=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["observed"], observed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
