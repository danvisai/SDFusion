"""Contract tests for the pre-registered four-arm experiment in issue #92."""
from __future__ import annotations

import unittest

from scripts.foundations.run_aligned_retrain import (
    ALIGNMENT_STAMPS, ARM_SPECS, PREREGISTERED, _alignment_stamp, command_for,
)


def _options(command: list[str]) -> dict[str, str]:
    """Return the long-option/value pairs from a train command."""
    return dict(zip(command[2::2], command[3::2]))


class TestArmCommands(unittest.TestCase):
    def setUp(self):
        self.commands = {arm: _options(command_for(arm)) for arm in ARM_SPECS}
        self.preregistered = {a: self.commands[a] for a in PREREGISTERED}

    def test_all_arms_share_the_pre_registered_training_contract(self):
        ignored = {"--blockouts", "--surf_weight", "--out"}
        control = {k: v for k, v in self.commands["A"].items() if k not in ignored}
        for arm, options in self.preregistered.items():
            self.assertEqual({k: v for k, v in options.items() if k not in ignored}, control, arm)
        self.assertEqual(control["--steps"], "240000")
        self.assertEqual(control["--archive_every"], "10000")
        self.assertEqual(control["--seed"], "92")

    def test_alignment_is_the_only_A_B_and_C_D_difference(self):
        for left, right in (("A", "B"), ("C", "D")):
            a, b = self.commands[left].copy(), self.commands[right].copy()
            out_a, out_b = a.pop("--out"), b.pop("--out")
            self.assertNotEqual(out_a, out_b)
            self.assertNotEqual(a.pop("--blockouts"), b.pop("--blockouts"))
            self.assertEqual(a, b)

    def test_surface_loss_is_the_only_A_C_and_B_D_difference(self):
        for with_surface, without_surface in (("A", "C"), ("B", "D")):
            a, b = self.commands[with_surface].copy(), self.commands[without_surface].copy()
            a.pop("--out"); b.pop("--out")
            self.assertEqual(a.pop("--surf_weight"), "1.0")
            self.assertEqual(b.pop("--surf_weight"), "0.0")
            self.assertEqual(a, b)

    def test_the_probe_is_not_one_of_the_pre_registered_arms(self):
        """N must never join the 2x2 by accident -- it would change what the map is judged on."""
        self.assertNotIn("N", PREREGISTERED)
        self.assertEqual(set(PREREGISTERED), {"A", "B", "C", "D"})
        for arm in PREREGISTERED:
            self.assertNotIn("--regions", self.commands[arm], arm)

    def test_the_probe_differs_from_arm_A_by_the_region_filter_alone(self):
        """It is only a diagnostic if arm A is the sole thing it is compared against."""
        probe, control = self.commands["N"].copy(), self.commands["A"].copy()
        self.assertEqual(probe.pop("--regions"), "0,1")
        self.assertNotEqual(probe.pop("--out"), control.pop("--out"))
        self.assertEqual(probe, control)

    def test_shipped_surface_regime_is_explicit(self):
        for arm in ("A", "B"):
            self.assertEqual(self.commands[arm]["--surf_t_center"], "0.0")
            self.assertEqual(self.commands[arm]["--surf_points"], "8192")
            self.assertEqual(self.commands[arm]["--surf_bs"], "1")


class TestAlignmentStamp(unittest.TestCase):
    """#90's greedy@k=256 has been stamped by two builders under two attribute names.

    Both must pass preflight -- rejecting the older spelling stranded the only complete corpus on
    the cluster -- and nothing else may, least of all the methods #90 measured and rejected.
    """

    def test_both_builders_spellings_are_accepted(self):
        self.assertIn(_alignment_stamp({"alignment": "greedy@k=256"}), ALIGNMENT_STAMPS)
        self.assertIn(_alignment_stamp({"method": "greedy_match(candidates=256)"}), ALIGNMENT_STAMPS)

    def test_bytes_attributes_decode(self):
        self.assertEqual(_alignment_stamp({"alignment": b"greedy@k=256"}), "greedy@k=256")

    def test_a_rejected_method_does_not_pass(self):
        for rejected in ("morton@k=256", "hungarian@k=256", "as_encoded", "greedy@k=16"):
            self.assertNotIn(_alignment_stamp({"alignment": rejected}), ALIGNMENT_STAMPS, rejected)

    def test_an_unstamped_cache_does_not_pass(self):
        self.assertIsNone(_alignment_stamp({}))
        self.assertNotIn(_alignment_stamp({}), ALIGNMENT_STAMPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
