"""Contract tests for #179 -- the H1b guided-edit completion proxy.

Synthetic, fast, no GPU, no corpus (matching `test_recover_massing_programs.py`'s own convention).
Fixtures are RES=64, not this repo's usual smaller test RES=32: `fit_program_beam`'s own candidate
generators (`_ramp_candidates` etc.) hardcode the module-level `RES` internally, so a smaller
footprint array shape-mismatches inside it -- see `scene/test_sdf_edit.py::
TestCommitBlockProgramRealFit`'s own docstring for the same constraint.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_guided_edit_completion_proxy.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import mask_to_rings  # noqa: E402
from scripts.foundations.guided_edit_completion_proxy import (  # noqa: E402
    aggregate, op_column_masks, op_signature, run_completion, synthesize_gestures,
)
from scripts.foundations.recover_massing_programs import CARVE_NEEDED, replay_program  # noqa: E402

RES = 64


def _rect_fp() -> np.ndarray:
    m = np.zeros((RES, RES), bool)
    m[8:56, 8:56] = True
    return m


def _layer_entry(mask: np.ndarray, height: int) -> dict:
    return dict(op="Layer", height=int(height), area=int(mask.sum()), components=1,
               region=[r.tolist() for r in mask_to_rings(mask)])


# ==================================================================================================
# op_column_masks -- the height-field analogue of #144's `_contribution`
# ==================================================================================================


class TestOpColumnMasks(unittest.TestCase):
    def setUp(self):
        self.fp = _rect_fp()
        self.y0, self.y1 = 0, 20
        self.full = self.y1 - self.y0 + 1

    def _region(self, z0, z1, x0, x1):
        m = np.zeros_like(self.fp)
        m[z0:z1, x0:x1] = True
        m &= self.fp
        self.assertTrue(m.any())
        return m

    def test_a_single_layer_masks_exactly_its_own_region(self):
        region = self._region(16, 22, 16, 22)
        program = [_layer_entry(region, self.full - 5)]
        masks, h = op_column_masks(self.fp, self.y0, self.y1, program)
        self.assertEqual(len(masks), 1)
        np.testing.assert_array_equal(masks[0], region)
        self.assertTrue((h[region] == self.full - 5).all())
        self.assertTrue((h[self.fp & ~region] == self.full).all())

    def test_two_disjoint_layers_mask_only_their_own_columns_and_agree_with_replay_program(self):
        r1 = self._region(10, 14, 10, 14)
        r2 = self._region(40, 46, 40, 46)
        program = [_layer_entry(r1, self.full - 3), _layer_entry(r2, self.full - 8)]
        masks, h = op_column_masks(self.fp, self.y0, self.y1, program)
        np.testing.assert_array_equal(masks[0], r1)
        np.testing.assert_array_equal(masks[1], r2)
        expect = replay_program(self.fp, self.y0, self.y1, program)
        np.testing.assert_array_equal(h, expect)

    def test_a_later_op_that_cannot_actually_lower_an_overlap_masks_nothing_there(self):
        """Op 2's region overlaps op 1's, but asks for a HEIGHT ABOVE what op 1 already cut to --
        `min(h, v)` keeps op 1's lower value, so op 2's own contribution in the overlap is empty.
        This is exactly why an op's footprint columns must be its own actual delta, not its
        declared region."""
        r1 = self._region(16, 32, 16, 32)
        r2 = self._region(20, 26, 20, 26)              # inside r1
        program = [_layer_entry(r1, self.full - 10), _layer_entry(r2, self.full - 3)]
        masks, _h = op_column_masks(self.fp, self.y0, self.y1, program)
        np.testing.assert_array_equal(masks[0], r1)
        self.assertFalse(masks[1].any(), "op 2 requested a height op 1 had already cut below")


# ==================================================================================================
# op_signature
# ==================================================================================================


class TestOpSignature(unittest.TestCase):
    def setUp(self):
        self.fp = _rect_fp()

    def _region(self, z0, z1, x0, x1):
        m = np.zeros_like(self.fp)
        m[z0:z1, x0:x1] = True
        return m & self.fp

    def test_identical_op_and_mask_produce_equal_signatures(self):
        region = self._region(16, 22, 16, 22)
        op = _layer_entry(region, 10)
        self.assertEqual(op_signature(op, region.copy()), op_signature(dict(op), region.copy()))

    def test_a_different_height_changes_the_signature(self):
        region = self._region(16, 22, 16, 22)
        a = op_signature(_layer_entry(region, 10), region)
        b = op_signature(_layer_entry(region, 11), region)
        self.assertNotEqual(a, b)

    def test_a_different_mask_changes_the_signature(self):
        r1 = self._region(16, 22, 16, 22)
        r2 = self._region(16, 22, 16, 23)
        op = _layer_entry(r1, 10)
        self.assertNotEqual(op_signature(op, r1), op_signature(op, r2))


# ==================================================================================================
# synthesize_gestures
# ==================================================================================================


class TestSynthesizeGestures(unittest.TestCase):
    def setUp(self):
        self.fp = _rect_fp()
        self.full = 21
        # NOT the blockout ceiling everywhere: an "add" gesture on an all-full h_prog would be a
        # no-op and get filtered, which would silently break the "both modes appear" property.
        self.h_prog = np.where(self.fp, np.int16(self.full - 6), 0).astype(np.int16)

    def test_every_gesture_region_is_footprint_local_and_nonempty(self):
        gestures = synthesize_gestures(self.fp, self.h_prog, self.full, seed=0)
        self.assertGreater(len(gestures), 0)
        fp_area = int(self.fp.sum())
        for g in gestures:
            self.assertTrue(g["region"].any())
            self.assertEqual(int((g["region"] & self.fp).sum()), int(g["region"].sum()),
                             "a gesture region must stay inside the footprint")
            self.assertLess(int(g["region"].sum()), fp_area,
                            "footprint-column-LOCAL: never the whole footprint")

    def test_several_distinct_sizes_are_produced(self):
        gestures = synthesize_gestures(self.fp, self.h_prog, self.full, seed=1)
        sizes = {int(g["region"].sum()) for g in gestures}
        self.assertGreater(len(sizes), 1)

    def test_both_modes_appear(self):
        gestures = synthesize_gestures(self.fp, self.h_prog, self.full, seed=2)
        self.assertEqual({g["mode"] for g in gestures}, {"add", "subtract"})

    def test_the_target_changes_only_inside_the_gesture_region(self):
        gestures = synthesize_gestures(self.fp, self.h_prog, self.full, seed=3)
        for g in gestures:
            outside = self.fp & ~g["region"]
            np.testing.assert_array_equal(g["target"][outside], self.h_prog[outside])
            if g["mode"] == "add":
                self.assertTrue((g["target"][g["region"]] >= self.h_prog[g["region"]]).all())
            else:
                self.assertTrue((g["target"][g["region"]] <= self.h_prog[g["region"]]).all())

    def test_clips_to_the_envelope(self):
        gestures = synthesize_gestures(self.fp, self.h_prog, self.full, seed=4)
        for g in gestures:
            self.assertTrue((g["target"][self.fp] >= 1).all())
            self.assertTrue((g["target"][self.fp] <= self.full).all())

    def test_deterministic_given_the_same_seed(self):
        a = synthesize_gestures(self.fp, self.h_prog, self.full, seed=42)
        b = synthesize_gestures(self.fp, self.h_prog, self.full, seed=42)
        self.assertEqual(len(a), len(b))
        for ga, gb in zip(a, b):
            np.testing.assert_array_equal(ga["region"], gb["region"])
            self.assertEqual(ga["mode"], gb["mode"])


# ==================================================================================================
# run_completion -- #7's gate and #144's re-checked locality invariant
# ==================================================================================================


class TestRunCompletion(unittest.TestCase):
    def setUp(self):
        self.fp = _rect_fp()
        self.y0, self.y1 = 0, 20
        self.full = self.y1 - self.y0 + 1

    def _region(self, z0, z1, x0, x1):
        m = np.zeros_like(self.fp)
        m[z0:z1, x0:x1] = True
        return m & self.fp

    def test_gate_passes_on_the_empty_pre_gesture_program(self):
        pre_program: list = []
        pre_masks, _ = op_column_masks(self.fp, self.y0, self.y1, pre_program)
        h_prog = np.where(self.fp, np.int16(self.full), 0).astype(np.int16)
        gestures = synthesize_gestures(self.fp, h_prog, self.full, seed=7)
        g = next(g for g in gestures if g["mode"] == "subtract")
        res = run_completion(self.fp, self.y0, self.y1, pre_program, pre_masks, g,
                             max_ops=4, beam=6, branch=6, allowance=CARVE_NEEDED)
        self.assertTrue(res["gate_pass"], res["gate_problems"])

    def test_locality_holds_when_the_gesture_is_far_from_existing_ops_with_enough_budget(self):
        r1 = self._region(10, 14, 10, 14)
        pre_program = [_layer_entry(r1, self.full - 5)]
        pre_masks, h_prog = op_column_masks(self.fp, self.y0, self.y1, pre_program)
        gestures = synthesize_gestures(self.fp, h_prog, self.full, seed=11)
        g = next(g for g in gestures if not (g["region"] & r1).any())
        # allowance=0.0, not CARVE_NEEDED: on this small fixture the pre-existing carve is tiny
        # relative to the footprint's own bulk volume, so the relative-surplus allowance the real
        # corpus uses would be satisfied with ZERO ops and never exercise the fitter at all.
        res = run_completion(self.fp, self.y0, self.y1, pre_program, pre_masks, g,
                             max_ops=len(pre_program) + 4, beam=12, branch=10, allowance=0.0)
        self.assertTrue(res["gate_pass"], res["gate_problems"])
        self.assertTrue(res["locality_pass"], res["locality_problems"])

    def test_locality_is_flagged_when_the_op_budget_cannot_reconstruct_the_untouched_region(self):
        r1 = self._region(10, 14, 10, 14)
        r2 = self._region(10, 14, 44, 48)
        pre_program = [_layer_entry(r1, self.full - 5), _layer_entry(r2, self.full - 9)]
        pre_masks, h_prog = op_column_masks(self.fp, self.y0, self.y1, pre_program)
        gestures = synthesize_gestures(self.fp, h_prog, self.full, seed=13)
        g = next(g for g in gestures if not (g["region"] & (r1 | r2)).any())
        # max_ops=1 starves the re-fit of the budget it would need to redo BOTH pre-existing
        # layers plus the gesture -- at least one must vanish, a real change outside the gesture.
        res = run_completion(self.fp, self.y0, self.y1, pre_program, pre_masks, g,
                             max_ops=1, beam=6, branch=6, allowance=0.0)
        self.assertFalse(res["locality_pass"])
        self.assertTrue(res["locality_problems"])


# ==================================================================================================
# aggregate -- true N, bootstrap CI, undersampled flag
# ==================================================================================================


class TestAggregate(unittest.TestCase):
    def test_reports_true_n_and_pass_rates(self):
        rows = ([dict(gate_pass=True, locality_pass=True)] * 5
               + [dict(gate_pass=True, locality_pass=False)] * 5)
        summary = aggregate(rows, seed=0)
        self.assertEqual(summary["n"], 10)
        self.assertAlmostEqual(summary["gate_pass_rate"], 1.0)
        self.assertAlmostEqual(summary["locality_pass_rate"], 0.5)
        lo, hi = summary["gate_ci"]
        self.assertLessEqual(lo, summary["gate_pass_rate"])
        self.assertGreaterEqual(hi, summary["gate_pass_rate"])

    def test_flags_undersampled_below_the_threshold(self):
        rows = [dict(gate_pass=True, locality_pass=True)] * 5
        summary = aggregate(rows, seed=0, undersampled_n=30)
        self.assertTrue(summary["undersampled"])

    def test_does_not_flag_undersampled_at_or_above_the_threshold(self):
        rows = [dict(gate_pass=True, locality_pass=True)] * 30
        summary = aggregate(rows, seed=0, undersampled_n=30)
        self.assertFalse(summary["undersampled"])

    def test_empty_rows_is_handled_without_raising(self):
        summary = aggregate([], seed=0)
        self.assertEqual(summary["n"], 0)
        self.assertTrue(summary["undersampled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
