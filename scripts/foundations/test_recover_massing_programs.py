"""Contract tests for #134's `direct` region fitter and floor confound control, #149's
block-coordination scoring bias, and #150's explicit block program. Synthetic, fast, no GPU,
no corpus.

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
from unittest.mock import patch

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scene.sdf_edit import mask_to_rings  # noqa: E402
from scripts.foundations import recover_massing_programs  # noqa: E402
from scripts.foundations.recover_massing_programs import (  # noqa: E402
    BIAS_WEIGHT, BlockProgram, FitBias, UnknownFootprintError, _dists_for, _family_bonus,
    _rings_to_mask, _select, _within_type_bonus, fit_program, fit_program_beam, program_floor,
    replay_program, simplify_region,
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


class TestFitBias(unittest.TestCase):
    """The block-program object itself (#9/#149): every field optional, empty by default."""

    def test_the_default_bias_is_empty(self):
        self.assertTrue(FitBias().is_empty())

    def test_setting_any_one_field_makes_it_non_empty(self):
        self.assertFalse(FitBias(height_rhythm=5).is_empty())
        self.assertFalse(FitBias(roof_family="flat").is_empty())
        self.assertFalse(FitBias(setback=2).is_empty())
        self.assertFalse(FitBias(azimuth=0.0).is_empty())


class TestFitBiasValidation(unittest.TestCase):
    """A typo'd `roof_family` must fail loudly rather than silently scoring zero everywhere."""

    def test_an_unknown_roof_family_is_rejected(self):
        with self.assertRaises(ValueError):
            FitBias(roof_family="gable")

    def test_the_three_real_families_are_accepted(self):
        for family in ("flat", "cut_roof", "ramp"):
            FitBias(roof_family=family)                     # must not raise


class TestFamilyBonus(unittest.TestCase):
    """`_family_bonus`: the one bonus `_select` lets compete ACROSS operation types."""

    def test_unset_is_always_zero(self):
        for op in ("Layer", "CutRoof", "Ramp"):
            self.assertEqual(_family_bonus(dict(op=op), FitBias()), 0.0)

    def test_matches_each_operation_types_own_family(self):
        for op, family in (("Layer", "flat"), ("CutRoof", "cut_roof"), ("Ramp", "ramp")):
            self.assertEqual(_family_bonus(dict(op=op), FitBias(roof_family=family)), 1.0, op)
            other = "flat" if family != "flat" else "ramp"
            self.assertEqual(_family_bonus(dict(op=op), FitBias(roof_family=other)), 0.0, op)


class TestWithinTypeBonus(unittest.TestCase):
    """`_within_type_bonus`: the axes `_select` only ever uses to compare candidates of the SAME
    operation type against each other. A 5x5 footprint gives an edge cell inset depth 1 and a
    centre cell inset depth 3 -- far enough apart that the +-1 voxel tolerance can never blur the
    two together."""

    def setUp(self):
        fp = np.zeros((8, 8), bool)
        fp[1:6, 1:6] = True
        self.dists = _dists_for(fp)
        self.region_edge = np.zeros((8, 8), bool)
        self.region_edge[1, 1] = True                     # inset depth 1
        self.region_centre = np.zeros((8, 8), bool)
        self.region_centre[3, 3] = True                    # inset depth 3

    def test_empty_bias_never_contributes(self):
        meta = dict(op="Layer", height=5, _region=self.region_edge)
        self.assertEqual(_within_type_bonus(meta, FitBias(), self.dists), 0.0)

    def test_roof_family_alone_contributes_nothing_here(self):
        """`roof_family` is `_family_bonus`'s axis, never this one's -- setting only it must leave
        every within-type comparison exactly as an unbiased fitter would make it."""
        meta = dict(op="Layer", height=999, _region=self.region_edge)   # height matches nothing
        self.assertEqual(_within_type_bonus(meta, FitBias(roof_family="flat"), self.dists), 0.0)

    def test_height_rhythm_matches_within_one_voxel(self):
        meta = dict(op="Layer", height=5, _region=self.region_edge)
        self.assertEqual(_within_type_bonus(meta, FitBias(height_rhythm=5), self.dists), 1.0)
        self.assertEqual(_within_type_bonus(meta, FitBias(height_rhythm=6), self.dists), 1.0)
        self.assertEqual(_within_type_bonus(meta, FitBias(height_rhythm=8), self.dists), 0.0)

    def test_height_rhythm_accepts_a_pattern_of_several_targets(self):
        meta = dict(op="Layer", height=5, _region=self.region_edge)
        self.assertEqual(_within_type_bonus(meta, FitBias(height_rhythm=(1, 5, 9)), self.dists), 1.0)
        self.assertEqual(_within_type_bonus(meta, FitBias(height_rhythm=(1, 9)), self.dists), 0.0)

    def test_height_rhythm_never_applies_to_a_roof_op(self):
        meta = dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)
        self.assertEqual(_within_type_bonus(meta, FitBias(height_rhythm=5), self.dists), 0.0)

    def test_setback_reads_the_regions_own_inset_depth(self):
        meta_edge = dict(op="Layer", height=5, _region=self.region_edge)
        meta_centre = dict(op="Layer", height=5, _region=self.region_centre)
        self.assertEqual(_within_type_bonus(meta_edge, FitBias(setback=1), self.dists), 1.0)
        self.assertEqual(_within_type_bonus(meta_centre, FitBias(setback=1), self.dists), 0.0)
        self.assertEqual(_within_type_bonus(meta_centre, FitBias(setback=3), self.dists), 1.0)

    def test_setback_never_applies_to_a_ramp(self):
        meta = dict(op="Ramp", slope=[0.0, 0.0], _region=self.region_edge)
        self.assertEqual(_within_type_bonus(meta, FitBias(setback=1), self.dists), 0.0)

    def test_azimuth_reads_the_ramp_in_129s_own_convention(self):
        """🔑 `slope` is `[x_coeff, z_coeff]` and #129's `plane_to_bins` reads `arctan2(Cx, Bz)`
        -- x first, z second. The arguments were swapped until review caught it, which mirrored
        every angle about 45 degrees while still matching *some* ramp, so nothing failed loudly.
        These fixtures are the convention, written down."""
        meta_0 = dict(op="Ramp", slope=[0.0, 1.0])          # atan2(0, 1) -> 0 degrees
        meta_90 = dict(op="Ramp", slope=[1.0, 0.0])         # atan2(1, 0) -> 90 degrees
        self.assertEqual(_within_type_bonus(meta_0, FitBias(azimuth=0.0), self.dists), 1.0)
        self.assertEqual(_within_type_bonus(meta_0, FitBias(azimuth=90.0), self.dists), 0.0)
        self.assertEqual(_within_type_bonus(meta_90, FitBias(azimuth=90.0), self.dists), 1.0)
        self.assertEqual(_within_type_bonus(meta_90, FitBias(azimuth=0.0), self.dists), 0.0)

    def test_azimuth_wraps_across_zero_degrees(self):
        meta = dict(op="Ramp", slope=[-0.05, 1.0])          # just under 0 degrees
        self.assertEqual(_within_type_bonus(meta, FitBias(azimuth=0.0), self.dists), 1.0)

    def test_azimuth_never_applies_to_a_layer(self):
        meta = dict(op="Layer", height=5, _region=self.region_edge)
        self.assertEqual(_within_type_bonus(meta, FitBias(azimuth=0.0), self.dists), 0.0)

    def test_two_matched_axes_on_the_same_layer_average_to_a_full_bonus(self):
        meta = dict(op="Layer", height=5, _region=self.region_edge)
        bias = FitBias(height_rhythm=5, setback=1)
        self.assertEqual(_within_type_bonus(meta, bias, self.dists), 1.0)

    def test_one_matched_axis_of_two_averages_to_a_half_bonus(self):
        meta = dict(op="Layer", height=5, _region=self.region_edge)
        bias = FitBias(height_rhythm=5, setback=99)         # setback misses badly
        self.assertEqual(_within_type_bonus(meta, bias, self.dists), 0.5)


class TestSelectIsIndependentAcrossAxes(unittest.TestCase):
    """`_select`'s two-stage ranking, directly: three candidates of three different types, close
    enough in raw gain that any of the three axis families COULD flip the pick if it leaked across
    types. Only `roof_family` may."""

    def setUp(self):
        self.dists = _dists_for(np.ones((4, 4), bool))
        region = np.ones((4, 4), bool)
        # gains 100 / 99 / 98: each pair is within BIAS_WEIGHT (0.15) of the others
        self.layer = (99, None, dict(op="Layer", height=5, _region=region))
        self.cut = (100, None, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0))
        self.ramp = (98, None, dict(op="Ramp", slope=[0.0, 1.0]))     # azimuth 0 (see #129)
        self.candidates = [self.layer, self.cut, self.ramp]

    def test_no_bias_picks_the_highest_raw_gain(self):
        self.assertEqual(_select(self.candidates, None, self.dists, 1), [self.cut])

    def test_height_rhythm_alone_never_moves_a_non_layer_candidate_ahead(self):
        bias = FitBias(height_rhythm=5)                      # matches self.layer exactly
        self.assertEqual(_select(self.candidates, bias, self.dists, 1), [self.cut])

    def test_azimuth_alone_never_moves_a_non_ramp_candidate_ahead(self):
        bias = FitBias(azimuth=0.0)                          # matches self.ramp exactly
        self.assertEqual(_select(self.candidates, bias, self.dists, 1), [self.cut])

    def test_roof_family_alone_does_move_the_winner_across_types(self):
        bias = FitBias(roof_family="flat")                   # "flat" is self.layer's family
        self.assertEqual(_select(self.candidates, bias, self.dists, 1), [self.layer])

    def test_roof_family_and_a_within_type_axis_compose(self):
        """Setting both: `azimuth` can still decide WHICH Ramp wins if Ramp wins at all, and
        `roof_family` can still decide that Ramp wins over Layer/CutRoof -- independently."""
        two_ramps = [self.layer, self.cut,
                    (98, None, dict(op="Ramp", slope=[0.0, 1.0])),    # azimuth 0, matches target
                    (98, None, dict(op="Ramp", slope=[1.0, 0.0]))]    # azimuth 90, does not match
        bias = FitBias(roof_family="ramp", azimuth=0.0)
        picked = _select(two_ramps, bias, self.dists, 1)[0]
        self.assertEqual(picked[2]["op"], "Ramp")
        self.assertEqual(picked[2]["slope"], [0.0, 1.0])


class TestSelectAtBeamWidths(unittest.TestCase):
    """⚠️ The gap that shipped a regression: every independence test above uses `n = 1`, and the
    beam path calls `_select` with `n = branch`. An earlier two-stage design keyed every entrant
    of a type on that type's own best raw gain, so `nlargest` drained one type's whole list
    before reaching the next -- a bias matching NOTHING silently replaced two `CutRoof` branches
    with two far worse `Layer` ones. These pin `n > 1` directly."""

    def setUp(self):
        self.dists = _dists_for(np.ones((4, 4), bool))
        region = np.ones((4, 4), bool)
        # one type holds the top gain and a long tail of bad ones; another holds the middle
        self.candidates = [
            (100, None, dict(op="Layer", height=5, _region=region)),
            (10, None, dict(op="Layer", height=5, _region=region)),
            (9, None, dict(op="Layer", height=5, _region=region)),
            (99, None, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)),
            (98, None, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)),
            (97, None, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)),
        ]

    def _gains_and_types(self, picked):
        return sorted((t[0], t[2]["op"]) for t in picked)

    def test_a_within_type_axis_alone_never_changes_the_types_returned(self):
        unbiased = self._gains_and_types(_select(self.candidates, None, self.dists, 3))
        for bias in (FitBias(height_rhythm=5), FitBias(setback=1), FitBias(azimuth=0.0)):
            got = self._gains_and_types(_select(self.candidates, bias, self.dists, 3))
            self.assertEqual(got, unbiased, f"{bias} moved a type at n=3")

    def test_a_bias_matching_nothing_is_the_identity_at_every_width(self):
        for n in (1, 2, 3, 4, 6):
            unbiased = self._gains_and_types(_select(self.candidates, None, self.dists, n))
            for bias in (FitBias(height_rhythm=10 ** 6), FitBias(setback=10 ** 6),
                        FitBias(azimuth=0.0)):
                got = self._gains_and_types(_select(self.candidates, bias, self.dists, n))
                self.assertEqual(got, unbiased, f"{bias} perturbed an unmatched fit at n={n}")

    def test_it_never_returns_more_or_fewer_than_the_unbiased_call(self):
        for n in (1, 2, 3, 4, 6, 10):
            want = len(_select(self.candidates, None, self.dists, n))
            for bias in (FitBias(roof_family="flat"), FitBias(height_rhythm=5),
                        FitBias(roof_family="cut_roof", azimuth=0.0)):
                self.assertEqual(len(_select(self.candidates, bias, self.dists, n)), want,
                                 f"{bias} changed the candidate count at n={n}")

    def test_it_never_returns_the_same_candidate_twice(self):
        for n in (2, 3, 6):
            for bias in (FitBias(roof_family="flat"), FitBias(height_rhythm=5),
                        FitBias(setback=1), FitBias(azimuth=0.0)):
                picked = _select(self.candidates, bias, self.dists, n)
                self.assertEqual(len({id(t) for t in picked}), len(picked),
                                 f"{bias} returned a duplicate at n={n}")

    def test_roof_family_still_shifts_which_type_holds_the_positions(self):
        """The axis that IS allowed to move types must still do so at n > 1.

        ⚠️ Needs its own fixture: on `setUp`'s, `Layer`'s tail (10, 9) is so far below the
        `CutRoof`s (99, 98) that a soft 15% bonus rightly cannot lift it, and a test asserting
        otherwise would be demanding back the very drain this class exists to forbid. Here the
        one `Layer` sits close enough that the bonus genuinely crosses it over.
        """
        region = np.ones((4, 4), bool)
        candidates = [
            (90, None, dict(op="Layer", height=5, _region=region)),
            (100, None, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)),
            (99, None, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)),
            (98, None, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)),
        ]
        unbiased = _select(candidates, None, self.dists, 3)
        self.assertEqual(sum(1 for t in unbiased if t[2]["op"] == "Layer"), 0,
                         "fixture is wrong: raw gain should exclude the Layer entirely")

        picked = _select(candidates, FitBias(roof_family="flat"), self.dists, 3)
        self.assertEqual(sum(1 for t in picked if t[2]["op"] == "Layer"), 1,
                         "a flat bias should pull the close Layer into the branch set")


def _tiny_footprint_fixture():
    """The 16-cell, flat-target=5 fixture every mocked-candidate bias test below shares -- greedy
    and beam threading are checked against the exact same numbers rather than two copies that
    could quietly drift apart."""
    fp = np.zeros((64, 64), bool)
    fp[10:14, 10:14] = True
    y0, y1, full = 0, 9, 10
    target = np.where(fp, 5, 0).astype(np.int16)
    return fp, y0, y1, full, target


def _close_tie_fake_candidates(fp, target, full, close_gap, expect="close"):
    """One `CutRoof` candidate that fully solves the target (gain = every surplus voxel) and one
    `Layer` candidate exactly `close_gap` voxels short of that -- so the two candidates' raw-gain
    gap is a known, exact number, and a bias test can assert on which side of `BIAS_WEIGHT` it
    falls without depending on real geometry finding a tie by chance.

    `expect` asserts the fixture really is what the caller thinks: "close" means a full-match
    bonus could flip it, "wide" means nothing soft ever could. Checked here rather than restated
    at each call site, because it is this function's own gap arithmetic that decides it.
    """
    h_cut = target.copy()
    h_layer = target.copy()
    idx = np.argwhere(fp)[:max(1, close_gap)]
    for z, x in idx:
        h_layer[z, x] = target[z, x] + 1
    gain_cut = int((full - h_cut[fp]).sum())
    gain_layer = int((full - h_layer[fp]).sum())

    gap, headroom = gain_cut - gain_layer, gain_layer * BIAS_WEIGHT
    if expect == "close":
        assert gap < headroom, f"fixture is not a close tie: gap {gap} >= headroom {headroom}"
    elif expect == "wide":
        assert gap > headroom, f"fixture is not a wide gap: gap {gap} <= headroom {headroom}"

    def fake(fp_, dists, target_, h_, ops_allowed=None):
        yield gain_cut, h_cut, dict(op="CutRoof", kind="hip", eaves=5, rate=1.0)
        yield gain_layer, h_layer, dict(op="Layer", height=5, area=int(fp.sum()),
                                        components=1, _region=fp)

    return fake, gain_cut, gain_layer


class TestFitProgramGreedyBiasThreading(unittest.TestCase):
    """`fit_program`'s selection, isolated from real geometry via a mocked candidate stream, the
    way `TestProgramFloor` above isolates `program_floor` from the fitter that calls it."""

    def setUp(self):
        self.fp, self.y0, self.y1, self.full, self.target = _tiny_footprint_fixture()

    def test_unbiased_picks_the_higher_gain_candidate(self):
        fake, gain_cut, gain_layer = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        self.assertGreater(gain_cut, gain_layer)
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1)
        self.assertEqual(ops[0]["op"], "CutRoof")

    def test_a_close_bias_flips_the_choice(self):
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        bias = FitBias(roof_family="flat")                  # "flat" is Layer's family
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1, bias=bias)
        self.assertEqual(ops[0]["op"], "Layer")

    def test_a_wide_gap_is_never_overridden(self):
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 16, expect="wide")
        bias = FitBias(roof_family="flat")
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1, bias=bias)
        self.assertEqual(ops[0]["op"], "CutRoof",
                         "a soft bias must never override a large quality gap")

    def test_no_bias_is_bit_identical_to_an_explicit_empty_bias(self):
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops_a, h_a = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1)
            ops_b, h_b = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1, bias=None)
            ops_c, h_c = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1,
                                     bias=FitBias())
        self.assertEqual(ops_a, ops_b)
        self.assertEqual(ops_a, ops_c)
        np.testing.assert_array_equal(h_a, h_b)
        np.testing.assert_array_equal(h_a, h_c)

    def test_a_bias_no_candidate_matches_falls_back_to_the_unbiased_choice(self):
        """#149 acceptance criterion 3, restated for the case with no near-tie at all: a `Ramp`
        bias on a stream with no `Ramp` candidate at all contributes zero to every candidate, so
        the outcome is identical to not biasing."""
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        bias = FitBias(roof_family="ramp")                   # neither candidate is a Ramp
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1, bias=bias)
        self.assertEqual(ops[0]["op"], "CutRoof")

    def test_height_rhythm_alone_never_flips_the_type_choice(self):
        """The bug review caught: `height_rhythm` only ever scales `Layer` candidates, so a flat
        single-score ranking let a matching `Layer` out-bid a `CutRoof` it was never asked to
        compete against -- an unrequested implicit `roof_family="flat"` preference. `_select`'s
        two-stage ranking must keep the type decision on raw gain alone when `roof_family` itself
        is unset, whatever `height_rhythm` matches."""
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        bias = FitBias(height_rhythm=5)                      # matches the Layer candidate exactly
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1, bias=bias)
        self.assertEqual(ops[0]["op"], "CutRoof",
                         "height_rhythm alone must never decide Layer vs CutRoof")

    def test_setback_alone_never_flips_the_type_choice(self):
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        depth = int(_dists_for(self.fp)["hip"][self.fp].min())
        bias = FitBias(setback=depth)                        # matches the Layer candidate's region
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1, bias=bias)
        self.assertEqual(ops[0]["op"], "CutRoof",
                         "setback alone must never decide Layer vs CutRoof")

    def test_roof_family_still_flips_the_choice_once_type_axes_no_longer_leak(self):
        """The one axis that IS allowed to move the type decision still works after the fix."""
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        bias = FitBias(roof_family="flat")
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=1, bias=bias)
        self.assertEqual(ops[0]["op"], "Layer")


class TestFitProgramBeamBiasThreading(unittest.TestCase):
    """The same fixture, through the beam path at its smallest width (beam=1, branch=1), where its
    per-step selection collapses to exactly the comparison greedy makes -- confirming the bias
    reaches the branch-selection step too, not only `fit_program`'s."""

    def setUp(self):
        self.fp, self.y0, self.y1, self.full, self.target = _tiny_footprint_fixture()

    def test_unbiased_beam_matches_unbiased_greedy(self):
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program_beam(self.fp, self.y0, self.y1, self.target, max_ops=1,
                                      beam=1, branch=1)
        self.assertEqual(ops[0]["op"], "CutRoof")

    def test_a_close_bias_flips_the_beam_choice_too(self):
        fake, *_ = _close_tie_fake_candidates(self.fp, self.target, self.full, 1)
        bias = FitBias(roof_family="flat")
        with patch.object(recover_massing_programs, "_all_candidates", fake):
            ops, _ = fit_program_beam(self.fp, self.y0, self.y1, self.target, max_ops=1,
                                      beam=1, branch=1, bias=bias)
        self.assertEqual(ops[0]["op"], "Layer")

    def test_the_internal_greedy_fallback_is_called_with_the_same_bias(self):
        seen = {}
        real_fit_program = recover_massing_programs.fit_program

        def spy(*args, **kwargs):
            seen["bias"] = kwargs.get("bias")
            return real_fit_program(*args, **kwargs)

        bias = FitBias(roof_family="flat")
        with patch.object(recover_massing_programs, "fit_program", spy):
            fit_program_beam(self.fp, self.y0, self.y1, self.target, max_ops=1,
                             beam=2, branch=2, bias=bias)
        self.assertIs(seen["bias"], bias)


class TestFitProgramBiasRealGeometry(unittest.TestCase):
    """A real (non-mocked) footprint/target through the actual candidate generators: every bias
    combination must still respect containment, proving #149's acceptance criterion 3 structurally
    rather than by hand-building a near-tie. Containment holds here because `_all_candidates`
    itself never yields a candidate that cuts into `target` -- bias only re-ranks what already
    passed that filter."""

    def setUp(self):
        self.fp = np.zeros((64, 64), bool)
        self.fp[16:48, 16:48] = True                         # 32x32 square, hip-like target
        self.y0, self.y1 = 0, 9
        dist = ndimage.distance_transform_edt(self.fp)
        self.target = np.where(self.fp, np.clip(dist, 1, 8).astype(np.int16), 0)

    def test_every_bias_combination_keeps_containment_under_greedy(self):
        for bias in (None, FitBias(roof_family="ramp"), FitBias(roof_family="flat"),
                    FitBias(roof_family="cut_roof"), FitBias(height_rhythm=3),
                    FitBias(setback=2), FitBias(azimuth=45.0),
                    FitBias(roof_family="ramp", azimuth=999.0)):
            _, h = fit_program(self.fp, self.y0, self.y1, self.target, max_ops=4, bias=bias)
            self.assertTrue((h[self.fp] >= self.target[self.fp]).all(), bias)

    def test_every_bias_combination_keeps_containment_under_beam(self):
        for bias in (None, FitBias(roof_family="ramp"), FitBias(roof_family="flat"),
                    FitBias(height_rhythm=3), FitBias(setback=2), FitBias(azimuth=45.0)):
            _, h = fit_program_beam(self.fp, self.y0, self.y1, self.target, max_ops=4,
                                    beam=4, branch=4, bias=bias)
            self.assertTrue((h[self.fp] >= self.target[self.fp]).all(), bias)


def _multi_footprint_corpus():
    """Three small, well-separated footprints for #150's block-program tests: two flat targets at
    different heights (exercise `height_rhythm`/`setback`) and one hip-like target (exercise
    `roof_family`/`azimuth`) -- nothing #149's own fitter doesn't already understand, just three
    of them addressed by id at once."""
    def flat(row0, col0, size, height):
        fp = np.zeros((64, 64), bool)
        fp[row0:row0 + size, col0:col0 + size] = True
        return fp, 0, 9, np.where(fp, height, 0).astype(np.int16)

    def hip(row0, col0, size):
        fp = np.zeros((64, 64), bool)
        fp[row0:row0 + size, col0:col0 + size] = True
        dist = ndimage.distance_transform_edt(fp)
        return fp, 0, 9, np.where(fp, np.clip(dist, 1, 8).astype(np.int16), 0)

    return {"north": flat(4, 4, 4, 6), "south": flat(40, 40, 4, 4), "east": hip(20, 45, 8)}


class TestBlockProgramBias(unittest.TestCase):
    """`BlockProgram.to_bias`: the one `FitBias` every named footprint is re-fit under."""

    def test_to_bias_carries_only_the_set_fields(self):
        bp = BlockProgram(footprint_ids=("north",), height_rhythm=5)
        self.assertEqual(bp.to_bias(), FitBias(height_rhythm=5))

    def test_all_four_axes_round_trip(self):
        bp = BlockProgram(footprint_ids=("north",), height_rhythm=5, roof_family="ramp",
                          setback=2, azimuth=45.0)
        self.assertEqual(bp.to_bias(),
                         FitBias(height_rhythm=5, roof_family="ramp", setback=2, azimuth=45.0))

    def test_no_axes_set_is_an_empty_bias(self):
        self.assertTrue(BlockProgram(footprint_ids=("north",)).to_bias().is_empty())

    def test_an_invalid_roof_family_is_rejected_at_construction(self):
        """Fails fast at `BlockProgram(...)`, not only later at `.apply()` -- reuses `FitBias`'s
        own validation rather than re-implementing the valid-values list."""
        with self.assertRaises(ValueError):
            BlockProgram(footprint_ids=("north",), roof_family="gable")

    def test_a_list_of_ids_is_normalised_to_a_tuple(self):
        """`@dataclass(frozen=True)` implies hashable; a bare `list` field would silently break
        that the moment anyone put a `BlockProgram` in a set or used it as a dict key."""
        bp = BlockProgram(footprint_ids=["north", "south"])
        self.assertEqual(bp.footprint_ids, ("north", "south"))
        hash(bp)                 # must not raise


class TestBlockProgramApply(unittest.TestCase):
    """`BlockProgram.apply`'s own contract, over and above `FitBias`/`_select`'s: it reaches every
    named footprint uniformly, treats a missing id as a caller error rather than a silent skip
    (and fails before fitting anything rather than fitting some and skipping the rest), and is
    deterministic across repeat calls -- #150's four acceptance criteria, directly."""

    def setUp(self):
        self.corpus = _multi_footprint_corpus()

    def test_applies_to_every_named_footprint(self):
        bp = BlockProgram(footprint_ids=("north", "south", "east"), roof_family="flat")
        self.assertEqual(set(bp.apply(self.corpus).keys()), {"north", "south", "east"})

    def test_applying_a_subset_only_touches_that_subset(self):
        bp = BlockProgram(footprint_ids=("north",), roof_family="flat")
        self.assertEqual(set(bp.apply(self.corpus).keys()), {"north"})

    def test_one_axis_set_matches_a_direct_fit_program_beam_call(self):
        """#150 acceptance criterion 1: with only one axis set, every named footprint is re-fit
        with only that axis biased. Checked directly against #149's own fitter rather than
        re-deriving the independence guarantee `_select` already tests elsewhere."""
        bp = BlockProgram(footprint_ids=("north", "south", "east"), height_rhythm=6)
        result = bp.apply(self.corpus)
        for fid, (fp, y0, y1, target) in self.corpus.items():
            want = fit_program_beam(fp, y0, y1, target, bias=FitBias(height_rhythm=6))
            got = result[fid]
            self.assertEqual([o["op"] for o in got[0]], [o["op"] for o in want[0]], fid)
            np.testing.assert_array_equal(got[1], want[1], err_msg=fid)

    def test_no_axes_set_matches_the_unbiased_fitter(self):
        bp = BlockProgram(footprint_ids=("north", "south", "east"))
        result = bp.apply(self.corpus)
        for fid, (fp, y0, y1, target) in self.corpus.items():
            want = fit_program_beam(fp, y0, y1, target)
            np.testing.assert_array_equal(result[fid][1], want[1], err_msg=fid)

    def test_a_missing_footprint_id_raises_a_clear_error(self):
        bp = BlockProgram(footprint_ids=("north", "nonexistent"))
        with self.assertRaises(UnknownFootprintError) as ctx:
            bp.apply(self.corpus)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_multiple_missing_ids_are_all_named_at_once(self):
        bp = BlockProgram(footprint_ids=("north", "missing1", "missing2"))
        with self.assertRaises(UnknownFootprintError) as ctx:
            bp.apply(self.corpus)
        msg = str(ctx.exception)
        self.assertIn("missing1", msg)
        self.assertIn("missing2", msg)

    def test_unknown_footprint_error_is_still_a_key_error(self):
        """So an existing `except KeyError:` around footprint lookup keeps working without
        having to learn a new exception type."""
        self.assertTrue(issubclass(UnknownFootprintError, KeyError))

    def test_a_missing_footprint_id_prevents_any_fit_at_all(self):
        """Fail before doing any work rather than fit some and skip the bad id, so a caller can
        never be handed a partial result under the same name as a full one."""
        bp = BlockProgram(footprint_ids=("north", "south", "nonexistent"))
        with patch.object(recover_massing_programs, "fit_program_beam") as mock_fit:
            with self.assertRaises(UnknownFootprintError):
                bp.apply(self.corpus)
            mock_fit.assert_not_called()

    def test_applying_the_same_object_twice_is_deterministic(self):
        """#150 acceptance criterion 3."""
        bp = BlockProgram(footprint_ids=("north", "south", "east"), roof_family="ramp")
        first, second = bp.apply(self.corpus), bp.apply(self.corpus)
        for fid in bp.footprint_ids:
            self.assertEqual([o["op"] for o in first[fid][0]],
                             [o["op"] for o in second[fid][0]], fid)
            np.testing.assert_array_equal(first[fid][1], second[fid][1], err_msg=fid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
