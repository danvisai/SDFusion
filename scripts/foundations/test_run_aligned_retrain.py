"""Contract tests for the pre-registered four-arm experiment in issue #92."""
from __future__ import annotations

import unittest

from scripts.foundations.run_aligned_retrain import ARM_SPECS, command_for


def _options(command: list[str]) -> dict[str, str]:
    """Return the long-option/value pairs from a train command."""
    return dict(zip(command[2::2], command[3::2]))


class TestArmCommands(unittest.TestCase):
    def setUp(self):
        self.commands = {arm: _options(command_for(arm)) for arm in ARM_SPECS}

    def test_all_arms_share_the_pre_registered_training_contract(self):
        ignored = {"--blockouts", "--surf_weight", "--out"}
        control = {k: v for k, v in self.commands["A"].items() if k not in ignored}
        for arm, options in self.commands.items():
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

    def test_shipped_surface_regime_is_explicit(self):
        for arm in ("A", "B"):
            self.assertEqual(self.commands[arm]["--surf_t_center"], "0.0")
            self.assertEqual(self.commands[arm]["--surf_points"], "8192")
            self.assertEqual(self.commands[arm]["--surf_bs"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
