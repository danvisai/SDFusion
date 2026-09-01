"""Contract tests for #93's per-footprint strength-band readout."""
from __future__ import annotations

import unittest

from scripts.foundations.analyze_issue93_strength_band import analyze_strength_band


class TestStrengthBandAnalysis(unittest.TestCase):
    def test_a_usable_strength_must_act_survive_and_beat_the_same_buildings_envelope(self):
        artifact = {
            "ids": [1, 2],
            "per_building": {
                "blockout": {
                    "1": {"vol_iou": 0.70},
                    "2": {"vol_iou": 0.80},
                },
                "a2_s0.4": {
                    "1": {"vol_iou": 0.72, "vs_input": 0.99, "missing": 0.01},
                    "2": {"vol_iou": 0.81, "vs_input": 0.90, "missing": 0.20},
                },
                "a2_s0.5": {
                    "1": {"vol_iou": 0.75, "vs_input": 0.95, "missing": 0.02},
                    "2": {"vol_iou": 0.79, "vs_input": 0.90, "missing": 0.02},
                },
                "a2_s0.6": {
                    "1": {"vol_iou": 0.74, "vs_input": 0.90, "missing": 0.03},
                    "2": {"vol_iou": 0.70, "vs_input": 0.70, "missing": 0.20},
                },
            },
        }

        report = analyze_strength_band(artifact)

        self.assertEqual(report["per_building"]["1"]["usable_strengths"], [0.5, 0.6])
        self.assertEqual(report["per_building"]["1"]["usable_range"], [0.5, 0.6])
        self.assertEqual(report["per_building"]["2"]["usable_strengths"], [])
        self.assertIsNone(report["per_building"]["2"]["usable_range"])
        self.assertEqual(report["per_building"]["1"]["state_by_strength"]["0.4"], "no_op")
        self.assertEqual(report["per_building"]["2"]["state_by_strength"]["0.4"], "collapsed")
        self.assertEqual(report["per_building"]["2"]["state_by_strength"]["0.5"], "net_negative")
        self.assertEqual(report["band_exists_rate"], 0.5)

    def test_ids_and_strength_arms_must_be_complete(self):
        artifact = {
            "ids": [1, 2],
            "per_building": {
                "blockout": {"1": {"vol_iou": 0.7}, "2": {"vol_iou": 0.8}},
                "a2_s0.5": {"1": {"vol_iou": 0.8, "vs_input": 0.9, "missing": 0.0}},
            },
        }
        with self.assertRaisesRegex(ValueError, "missing building ids"):
            analyze_strength_band(artifact)


if __name__ == "__main__":
    unittest.main(verbosity=2)
