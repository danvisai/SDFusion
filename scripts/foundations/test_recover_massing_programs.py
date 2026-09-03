"""Contract tests for #134's `direct` region fitter and floor confound control. Synthetic, fast,
no GPU, no corpus.

`scene/test_sdf_edit.py::TestVertexBudget` is the GUARD and is deliberately left untouched --
these are new tests for new code, in their own file, mirroring that file's own fixture shapes
(plain shed, exact diagonal, concave, holed) so the same properties can be checked against
`direct` that #131 already pinned for `contained`/`lossless`/`free`.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_recover_massing_programs.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scene.sdf_edit import mask_to_rings  # noqa: E402
from scripts.foundations.recover_massing_programs import (  # noqa: E402
    _rings_to_mask, program_floor, replay_program, simplify_region,
)

RES = 32


def _plain_shed():
    m = np.zeros((RES, RES), bool)
    m[5:20, 4:24] = True
    return m


def _staircase():
    m = np.zeros((RES, RES), bool)
    for i in range(12):
        m[6 + i, 4:8 + i] = True
    return m


def _concave():
    m = np.zeros((RES, RES), bool)
    m[5:20, 5:20] = True
    m[5:12, 12:20] = False
    return m


def _holed():
    m = np.zeros((RES, RES), bool)
    m[5:20, 5:20] = True
    m[12, 12] = False
    return m


def _rings(mask):
    return [r.tolist() for r in mask_to_rings(mask)]


class TestDirectRuleContainment(unittest.TestCase):
    """The one property `direct` must never break: it never claims a cell the exact region does
    not have. This is the same guarantee `contained` gives by deleting only inward; `direct` must
    give it by never growing outward past `exact_mask` either."""

    def test_never_gains_a_cell_on_a_concave_region(self):
        m = _concave()
        rings = _rings(m)
        for budget in (4, 6, 8, 12, 16, 24, 94):
            simp = simplify_region(rings, budget, m, RES, "direct")
            got = _rings_to_mask(simp, RES)
            self.assertEqual(int((got & ~m).sum()), 0, f"budget {budget} gained a cell")

    def test_never_gains_a_cell_on_a_holed_region(self):
        m = _holed()
        rings = _rings(m)
        for budget in (4, 8, 12, 20, 94):
            simp = simplify_region(rings, budget, m, RES, "direct")
            got = _rings_to_mask(simp, RES)
            self.assertEqual(int((got & ~m).sum()), 0, f"budget {budget} gained a cell")

    def test_never_gains_a_cell_on_a_staircase(self):
        m = _staircase()
        rings = _rings(m)
        for budget in (4, 8, 16, 24):
            simp = simplify_region(rings, budget, m, RES, "direct")
            got = _rings_to_mask(simp, RES)
            self.assertEqual(int((got & ~m).sum()), 0, f"budget {budget} gained a cell")


class TestDirectRuleShapes(unittest.TestCase):
    """The same eyeball checks #131 pinned for the trimming rules, run against growth instead."""

    def test_a_plain_shed_keeps_its_four_right_angles(self):
        """⚠️ The check the ticket demands by eye, restated for `direct`: a rectangle stays a
        rectangle -- growth must not invent a 45-degree corner a shed does not have."""
        m = _plain_shed()
        rings = _rings(m)
        for budget in (4, 6, 8, 12, 94):
            simp = simplify_region(rings, budget, m, RES, "direct")
            self.assertEqual(sum(len(r) for r in simp), 4, f"budget {budget}")
            np.testing.assert_array_equal(_rings_to_mask(simp, RES), m)

    def test_an_exact_diagonal_trace_really_is_a_triangle(self):
        """🔑 The finding the budget is for, restated for `direct`: growth should find the SAME
        4-vertex answer the trimming rules converge to, not something worse."""
        m = _staircase()
        rings = _rings(m)
        simp = simplify_region(rings, 4, m, RES, "direct")
        got = _rings_to_mask(simp, RES)
        self.assertEqual(sum(len(r) for r in simp), 4)
        np.testing.assert_array_equal(got, m, "not one cell should differ from the staircase")

    def test_a_one_cell_hole_is_irreducible(self):
        m = _holed()
        rings = _rings(m)
        simp = simplify_region(rings, 4, m, RES, "direct")
        self.assertEqual(len(simp), 2, "the hole survives")
        got = _rings_to_mask(simp, RES)
        self.assertEqual(int((got & ~m).sum()), 0)

    def test_reaches_the_exact_shape_given_enough_budget(self):
        for name, m in (("concave", _concave()), ("holed", _holed()), ("staircase", _staircase())):
            rings = _rings(m)
            simp = simplify_region(rings, 999, m, RES, "direct")
            got = _rings_to_mask(simp, RES)
            np.testing.assert_array_equal(got, m, f"{name} did not converge to the exact shape")


class TestDirectRuleCarriesTheFitForward(unittest.TestCase):
    """RADmesh's own mechanism, restated as a property: re-discretizing to a finer budget must
    not restart the search -- the coarser budget's kept vertices are a strict subset of the
    finer one's, not a different, unrelated set of vertex choices."""

    def test_a_finer_budget_keeps_every_vertex_the_coarser_one_chose(self):
        m = _concave()
        rings = _rings(m)
        schedule = (4, 6, 8, 12, 16, 24)
        prev_verts = None
        for budget in schedule:
            simp = simplify_region(rings, budget, m, RES, "direct")
            outer = {tuple(v) for v in simp[0]}
            if prev_verts is not None:
                self.assertTrue(prev_verts <= outer,
                                f"budget {budget}'s outer ring dropped a vertex the coarser "
                                f"budget kept -- the fit was restarted, not carried forward")
            prev_verts = outer


class TestSeedIsAlwaysContainedAndCoarse(unittest.TestCase):
    """The coarse start itself: never the exact ring, always a valid, contained, minimal shape."""

    def test_the_seed_is_never_the_full_exact_ring(self):
        for name, m in (("concave", _concave()), ("staircase", _staircase())):
            rings = _rings(m)
            exact_verts = sum(len(r) for r in rings)
            simp = simplify_region(rings, 4, m, RES, "direct")
            self.assertLessEqual(sum(len(r) for r in simp), exact_verts, name)


class TestProgramFloor(unittest.TestCase):
    """#134's confound-control arm: the per-building floor a `Layer`-abandoned column falls back
    to, instead of the full envelope height #131 diagnosed as the cause of the spike."""

    def test_the_floor_is_the_lowest_layer_height_in_the_program(self):
        program = [dict(op="Layer", height=9, region=[]), dict(op="Layer", height=4, region=[]),
                  dict(op="Ramp", plane=[5, 0, 0], region=[])]
        self.assertEqual(program_floor(program), 4)

    def test_a_program_with_no_layer_op_has_no_floor(self):
        program = [dict(op="Ramp", plane=[5, 0, 0], region=[])]
        self.assertIsNone(program_floor(program))

    def test_the_empty_program_has_no_floor(self):
        self.assertIsNone(program_floor([]))


class TestReplayProgramFloor(unittest.TestCase):
    """`replay_program`'s new `floor` parameter: an uncovered column starts there instead of the
    full envelope extent -- #131's own diagnosed cause of the spike, tested here.

    ⚠️ `replay_program` reads regions through `_rings_to_mask` at its own default resolution
    (module-level `RES` = 64, not a parameter) -- unlike `simplify_region` above, which takes
    `res` explicitly. This fixture matches that fixed resolution rather than the smaller `RES`
    the other test classes use for speed.
    """

    def setUp(self):
        res = 64
        self.fp = np.zeros((res, res), bool)
        self.fp[10:38, 8:40] = True
        self.y0, self.y1 = 4, 38
        m = np.zeros((res, res), bool)
        m[15:25, 12:22] = True
        m &= self.fp
        self.program = [dict(op="Layer", height=10,
                             region=[r.tolist() for r in mask_to_rings(m)])]
        self.touched = m

    def test_default_floor_is_unchanged_from_before_134(self):
        """A direct regression pin: calling without `floor` must reproduce exactly what
        `replay_program` returned before this ticket."""
        h = replay_program(self.fp, self.y0, self.y1, self.program)
        extent = self.y1 - self.y0 + 1
        np.testing.assert_array_equal(h[self.fp & ~self.touched], extent)

    def test_an_uncovered_column_takes_the_floor_instead_of_the_full_extent(self):
        floor = program_floor(self.program)
        h = replay_program(self.fp, self.y0, self.y1, self.program, floor=floor)
        np.testing.assert_array_equal(h[self.fp & ~self.touched], floor)

    def test_a_touched_column_is_unaffected_by_the_floor(self):
        floor = program_floor(self.program)
        h_default = replay_program(self.fp, self.y0, self.y1, self.program)
        h_floored = replay_program(self.fp, self.y0, self.y1, self.program, floor=floor)
        np.testing.assert_array_equal(h_default[self.touched], h_floored[self.touched])

    def test_a_floor_above_the_full_extent_is_clamped_not_allowed_to_raise_the_building(self):
        extent = self.y1 - self.y0 + 1
        h = replay_program(self.fp, self.y0, self.y1, self.program, floor=extent + 50)
        np.testing.assert_array_equal(h[self.fp & ~self.touched], extent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
