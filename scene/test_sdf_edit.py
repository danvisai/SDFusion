"""Contract tests for the layer-program bridge into the SDF edit stack (#128), and #151's
group-id tagging / independent per-footprint commit on top of it.

The recovered massing programs of #10 compile to a 64^3 voxel grid through their own deterministic
compiler.  These tests pin the *other* path -- the same program expressed as `EditOp`s and composed
through `scene/sdf_primitives.py` -- and assert the two agree exactly, because a program that only
exists as voxels is geometry evidence, not a recipe.

Pure CPU, no GPU, no corpus. #151's own tests need `scripts.foundations.recover_massing_programs`'
`BlockProgram` (a model-free, CPU-only object -- see that module's own test file) so "no model"
no longer holds file-wide, but nothing here trains or loads a checkpoint.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scene/test_sdf_edit.py
"""
from __future__ import annotations

import itertools
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import (  # noqa: E402
    ALGEBRA, ARCHITECTURAL_VOCABULARY, CORE, PALETTE, PROGRAM_KINDS, VOLUMETRIC,
    EditableBuilding, EditOp, _current_target, canonical_form, commit_block_program, commutes,
    containment_problems, equivalent, finalize_problems, footprint_envelope_sdf,
    is_height_map_representable, layer_program_to_ops, mask_components_rings, mask_to_rings,
    op_problems, program_problems, snap_to_grid,
)
from scene.sdf_primitives import sdf_plane_halfspace  # noqa: E402
from scripts.foundations import recover_massing_programs  # noqa: E402
from scripts.foundations.recover_massing_programs import BlockProgram  # noqa: E402

RES = 32                      # the corpus grid is 64^3; the contract is resolution-agnostic and a
                              # smaller grid keeps these tests a few seconds on CPU.


# ------------------------------------------------------------------------------------------------
# the reference: the voxel compiler the recovered programs already use
# ------------------------------------------------------------------------------------------------

def voxel_occupancy(fp: np.ndarray, y0: int, h: np.ndarray, res: int = RES) -> np.ndarray:
    """Height map -> occupancy [z, y, x]; `recover_massing_programs.occupancy`, restated."""
    yy = np.arange(res)[None, :, None]
    return fp[:, None, :] & (yy >= y0) & (yy < y0 + h[:, None, :].astype(np.int32))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return 1.0 if not union else float((a & b).sum() / union)


def centers(res: int = RES) -> np.ndarray:
    """The (res^3, 3) world points a corpus voxel grid samples, index-order [z, y, x]."""
    ax = np.linspace(-1.0, 1.0, res, dtype=np.float64)
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


class ProgramFixture:
    """A small synthetic building and a layer program over it, in the recovery's own format."""

    def __init__(self):
        self.res = RES
        self.fp = np.zeros((RES, RES), bool)
        self.fp[8:24, 6:26] = True            # an L: a rectangle with a bite taken out
        self.fp[8:14, 20:26] = False
        self.y0, self.y1 = 4, 21
        full = np.int16(self.y1 - self.y0 + 1)
        self.h = np.where(self.fp, full, 0).astype(np.int16)

    def layer(self, mask: np.ndarray, height: int) -> dict:
        self.h = np.where(mask, np.int16(height), self.h).astype(np.int16)
        return dict(op="Layer", height=int(height), area=int(mask.sum()),
                    components=1, region=[r.tolist() for r in mask_to_rings(mask)])

    def ramp(self, mask: np.ndarray, plane) -> dict:
        a, b, c = (float(v) for v in plane)
        zz, xx = np.mgrid[0:RES, 0:RES]
        surf = np.floor(a + b * xx + c * zz)
        self.h = np.where(mask, np.minimum(self.h, surf).astype(np.int16), self.h)
        return dict(op="Ramp", area=int(mask.sum()), plane=[a, b, c],
                    slope=[b, c], region=[r.tolist() for r in mask_to_rings(mask)])


# ------------------------------------------------------------------------------------------------


class TestPlaneHalfspace(unittest.TestCase):
    """The primitive `Ramp` and `CutRoof` are built from."""

    def test_sign_and_unit_gradient(self):
        f = sdf_plane_halfspace((0.0, 3.0, 0.0), -1.5)          # solid is y <= 0.5, un-normalised
        p = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 2.5, 0.0]])
        d = f(p).numpy()
        self.assertLess(d[0], 0.0)
        self.assertAlmostEqual(float(d[1]), 0.0, places=6)
        self.assertGreater(d[2], 0.0)
        self.assertAlmostEqual(float(d[2]), 2.0, places=6)       # normalised: a true distance

    def test_rejects_degenerate_normal(self):
        with self.assertRaises(ValueError):
            sdf_plane_halfspace((0.0, 0.0, 0.0), 1.0)


class TestMaskToRings(unittest.TestCase):
    """A voxel region becomes a polygon whose interior is *exactly* the region's cells.

    ⚠️ No simplification: the ring traces the cell boundary at half-voxel offsets, so voxel centres
    never land on an edge and the point-in-polygon test can never tie. Cutting these down to a
    vertex budget is a separate question, and `TestVertexBudget` below is where it is pinned (#131).
    """

    def _covers_exactly(self, mask):
        from matplotlib.path import Path as MplPath
        rings = mask_to_rings(mask)
        zz, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        pts = np.stack([xx.ravel(), zz.ravel()], 1).astype(float)
        inside = np.zeros(len(pts), bool)
        for i, ring in enumerate(rings):
            hit = MplPath(ring).contains_points(pts)
            inside = hit if i == 0 else (inside & ~hit)
        np.testing.assert_array_equal(inside.reshape(mask.shape), mask)

    def test_rectangle(self):
        m = np.zeros((RES, RES), bool)
        m[5:12, 3:20] = True
        self._covers_exactly(m)

    def test_concave_region(self):
        m = np.zeros((RES, RES), bool)
        m[5:20, 5:20] = True
        m[5:12, 12:20] = False
        self._covers_exactly(m)

    def test_region_with_a_hole(self):
        m = np.zeros((RES, RES), bool)
        m[5:20, 5:20] = True
        m[9:15, 9:15] = False
        rings = mask_to_rings(m)
        self.assertEqual(len(rings), 2, "outer ring plus one hole")
        self._covers_exactly(m)

    def test_region_touching_the_grid_border(self):
        m = np.zeros((RES, RES), bool)
        m[0:6, 0:6] = True
        self._covers_exactly(m)

    def test_orientation_is_counter_clockwise(self):
        m = np.zeros((RES, RES), bool)
        m[5:12, 3:20] = True
        ring = mask_to_rings(m)[0]
        area = 0.5 * np.sum(ring[:, 0] * np.roll(ring[:, 1], -1)
                            - np.roll(ring[:, 0], -1) * ring[:, 1])
        self.assertGreater(area, 0.0, "outer ring must be CCW for an outward prism normal")

    def test_empty_mask_has_no_rings(self):
        self.assertEqual(mask_to_rings(np.zeros((RES, RES), bool)), [])

    def test_a_mask_in_two_pieces_is_refused_and_split(self):
        """A Layer is one polygon; a footprint promises nothing -- 2 of the pinned 714 are two."""
        m = np.zeros((RES, RES), bool)
        m[5:12, 3:9] = True
        m[5:12, 14:20] = True
        with self.assertRaises(ValueError):
            mask_to_rings(m)
        self.assertEqual(len(mask_components_rings(m)), 2)


class TestVertexBudget(unittest.TestCase):
    """#131: a region re-cut to a vertex budget, and the containment guarantee it must not break.

    The exact ring is a raster trace at a median of 94 vertices per region. These pin what the
    budget may and may not do to it -- above all that the **contained** arm never hands an operation
    a cell the exact region did not have, because that is the one property keeping `missing` and
    `collapse_rate` 0 by construction (#10).
    """

    def _cells(self, rings):
        from scripts.foundations.recover_massing_programs import _rings_to_mask
        return _rings_to_mask(rings, RES) if rings else np.zeros((RES, RES), bool)

    def _rings(self, mask):
        return [r.tolist() for r in mask_to_rings(mask)]

    def _simplify(self, mask, budget, rule="contained"):
        from scripts.foundations.recover_massing_programs import simplify_region
        rings = self._rings(mask)
        exact = self._cells(rings)
        got = simplify_region(rings, budget, exact, RES, rule)
        return got, self._cells(got), exact

    def test_simplify_region_keeps_a_plain_shed(self):
        """⚠️ The check the ticket demands by eye: a rectangle keeps its four right angles.

        Marching squares would chamfer each corner diagonally and hand this shed four 45-degree
        eaves it does not have. Deleting existing vertices can never invent one.
        """
        m = np.zeros((RES, RES), bool)
        m[5:20, 4:24] = True
        for budget in (4, 6, 8, 12, 94):
            for rule in ("contained", "lossless", "free"):
                rings, got, exact = self._simplify(m, budget, rule)
                self.assertEqual(sum(len(r) for r in rings), 4, f"budget {budget}")
                np.testing.assert_array_equal(got, exact)

    def test_an_exact_diagonal_trace_is_really_a_triangle(self):
        """🔑 The finding the budget is for: the raster trace carries no information.

        A staircase down a diagonal costs one vertex pair per step, and the same cells are covered
        by a 4-vertex polygon -- so the vertices are the rasterizer's, not the architecture's.
        """
        m = self._staircase()
        self.assertGreater(sum(len(r) for r in self._rings(m)), 20, "the exact ring is a staircase")
        rings, got, exact = self._simplify(m, 4, "contained")
        self.assertEqual(sum(len(r) for r in rings), 4)
        np.testing.assert_array_equal(got, exact, "not one cell changes")

    def test_the_lossless_rule_changes_no_cell_at_all(self):
        """🔑 What "the vertices a region needs" means: run to no budget and lose nothing.

        A deletion is admitted only when its triangle holds no cell centre, so the rasterized
        region is the same set of cells before and after -- the same building to the voxel, at a
        fraction of the vertices.
        """
        for name, m in (("staircase", self._staircase()), ("concave", self._concave()),
                        ("holed", self._holed())):
            rings, got, exact = self._simplify(m, 0, "lossless")
            np.testing.assert_array_equal(got, exact, f"{name} changed a cell")
            self.assertLessEqual(sum(len(r) for r in rings),
                                 sum(len(r) for r in self._rings(m)), name)
        rings, _, _ = self._simplify(self._staircase(), 0, "lossless")
        self.assertEqual(sum(len(r) for r in rings), 4, "a diagonal trace needs four")

    def test_the_contained_arm_never_gains_a_cell(self):
        for name, m in (("concave", self._concave()), ("holed", self._holed())):
            for budget in (4, 6, 8, 12, 16, 24, 94):
                _, got, exact = self._simplify(m, budget, "contained")
                self.assertEqual(int((got & ~exact).sum()), 0,
                                 f"{name} at budget {budget} gained a cell: the region may only "
                                 f"shrink, or the program cuts into GT")

    def test_the_free_arm_does_break_it(self):
        """The constraint is load-bearing rather than decorative: without it the region grows."""
        gained = 0
        for m in (self._concave(), self._holed()):
            for budget in (4, 6):
                _, got, exact = self._simplify(m, budget, "free")
                gained += int((got & ~exact).sum())
        self.assertGreater(gained, 0)

    def test_a_one_cell_hole_is_irreducible_while_contained(self):
        """A speckle hole is 4 vertices that cannot be spent: shrinking it hands back its cell."""
        m = self._holed()
        rings, got, exact = self._simplify(m, 4, "contained")
        self.assertEqual(len(rings), 2, "the hole survives")
        self.assertGreater(sum(len(r) for r in rings), 4, "so the budget is NOT reachable")
        np.testing.assert_array_equal(got & ~exact, np.zeros_like(exact))
        free, got_free, _ = self._simplify(m, 4, "free")
        self.assertEqual(len(free), 1, "the free arm drops the hole")
        self.assertEqual(int((got_free & ~exact).sum()), 1, "and swallows its cell")

    def test_dsl_tokens_counts_what_a_generator_must_emit(self):
        from scripts.foundations.recover_massing_programs import dsl_tokens
        square = [[[0.5, 0.5], [4.5, 0.5], [4.5, 4.5], [0.5, 4.5]]]
        self.assertEqual(dsl_tokens([dict(op="Layer", height=3, region=square)]), 2 + 1 + 8)
        self.assertEqual(dsl_tokens([dict(op="Ramp", plane=[1, 2, 3], region=square)]), 4 + 1 + 8)
        self.assertEqual(dsl_tokens([dict(op="CutRoof", kind="hip", eaves=2, rate=0.5)]), 4)

    def _staircase(self):
        m = np.zeros((RES, RES), bool)
        for i in range(12):
            m[6 + i, 4:8 + i] = True
        return m

    def _concave(self):
        m = np.zeros((RES, RES), bool)
        m[5:20, 5:20] = True
        m[5:12, 12:20] = False
        return m

    def _holed(self):
        m = np.zeros((RES, RES), bool)
        m[5:20, 5:20] = True
        m[12, 12] = False
        return m


class TestEditOpRoundTrip(unittest.TestCase):
    """The new kinds must survive `to_dict` -> JSON -> `from_dict`, or they are not recipe state."""

    def _round_trip(self, op):
        back = EditOp.from_dict(json.loads(json.dumps(op.to_dict())))
        self.assertEqual(back.kind, op.kind)
        self.assertEqual(back.mode, op.mode)
        self.assertEqual(back.id, op.id, "#141: the id must survive the JSON round trip")
        np.testing.assert_allclose(np.asarray(back.size), np.asarray(op.size))
        for a, b in zip(back.polygon or [], op.polygon or []):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(np.asarray(back.planes or []), np.asarray(op.planes or []))
        return back

    def test_layer(self):
        op = EditOp(kind="layer", mode="subtract", size=(0.25, 1.0),
                    polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
        self._round_trip(op)

    def test_ramp_carries_its_plane(self):
        op = EditOp(kind="ramp", mode="subtract", size=(-0.5, 0.5),
                    polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]],
                    planes=[[[0.3, -1.0, 0.1, 0.02]]])
        self._round_trip(op)

    def test_cut_roof_carries_one_clause_per_eave(self):
        op = EditOp(kind="cut_roof", mode="subtract", size=(0.0, 1.0),
                    polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
                    planes=[[[0.0, -1.0, 0.5, 0.1], [0.0, 0.0, 1.0, -0.5]],
                            [[0.0, -1.0, -0.5, 0.1], [0.0, 0.0, -1.0, 0.5]]])
        back = self._round_trip(op)
        self.assertEqual(len(back.planes), 2)
        self.assertEqual(len(back.planes[0]), 2, "a clause is intersected, clauses are unioned")

    def test_unknown_kind_is_refused_by_add(self):
        """#143: `.add()` validates before appending, so this is refused right there -- it never
        reaches the stack in the first place."""
        eb = EditableBuilding(footprint_envelope_sdf(np.ones((4, 4), bool), 0, 1, res=4))
        with self.assertRaises(ValueError):
            eb.add(EditOp(kind="zigzag"))
        self.assertEqual(eb.ops, [], "a refused add must not touch the stack")

    def test_unknown_kind_is_also_refused_by_composed(self):
        """`composed()`'s own defense in depth: a building assembled directly from a malformed op
        list (bypassing `.add()`, e.g. by loading old state) still fails loudly rather than
        compiling garbage."""
        eb = EditableBuilding(footprint_envelope_sdf(np.ones((4, 4), bool), 0, 1, res=4),
                              [EditOp(kind="zigzag")])
        with self.assertRaises(ValueError):
            eb.composed()

    def test_old_state_without_the_new_fields_still_loads(self):
        legacy = {"kind": "box", "center": [0, 0, 0], "size": [1, 1, 1], "mode": "add",
                  "smooth": 0.0, "rot_y": 0.0, "round_r": 0.0, "lib_id": -1, "layer": "annotation"}
        op = EditOp.from_dict(legacy)
        self.assertEqual(op.kind, "box")
        self.assertIsNone(op.polygon)
        self.assertIsNone(op.planes)

    def test_old_state_with_no_id_is_assigned_a_fresh_one(self):
        """#141: `legacy` above predates the id field entirely -- loading it must not fail, and
        the loaded op must come out addressable rather than id-less."""
        legacy = {"kind": "box", "center": [0, 0, 0], "size": [1, 1, 1], "mode": "add",
                  "smooth": 0.0, "rot_y": 0.0, "round_r": 0.0, "lib_id": -1}
        self.assertNotIn("id", legacy)
        op = EditOp.from_dict(legacy)
        self.assertTrue(op.id, "a legacy op must still come out with a usable id")


class TestComposedMatchesVoxelCompiler(unittest.TestCase):
    """The load-bearing claim: the SDF composition and the voxel compiler are the same building."""

    def setUp(self):
        self.fx = ProgramFixture()

    def _occupancy_via_sdf(self, program):
        ops = layer_program_to_ops(program, self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        eb = EditableBuilding(base, ops)
        return eb.to_occupancy(res=RES), ops

    def test_bare_envelope_is_the_extruded_footprint(self):
        occ, ops = self._occupancy_via_sdf([])
        self.assertEqual(ops, [])
        ref = voxel_occupancy(self.fx.fp, self.fx.y0, self.fx.h)
        self.assertEqual(iou(occ, ref), 1.0)

    def test_layer_program(self):
        m = np.zeros((RES, RES), bool)
        m[14:24, 6:26] = True
        m &= self.fx.fp
        program = [self.fx.layer(m, 9)]
        occ, _ = self._occupancy_via_sdf(program)
        ref = voxel_occupancy(self.fx.fp, self.fx.y0, self.fx.h)
        self.assertEqual(iou(occ, ref), 1.0)

    def test_layer_then_ramp(self):
        m = np.zeros((RES, RES), bool)
        m[14:24, 6:26] = True
        m &= self.fx.fp
        r = np.zeros((RES, RES), bool)
        r[8:14, 6:20] = True
        r &= self.fx.fp
        program = [self.fx.layer(m, 9), self.fx.ramp(r, (30.0, -0.75, -0.4))]
        occ, ops = self._occupancy_via_sdf(program)
        self.assertEqual([o.kind for o in ops], ["layer", "ramp"])
        ref = voxel_occupancy(self.fx.fp, self.fx.y0, self.fx.h)
        self.assertEqual(iou(occ, ref), 1.0)

    def test_a_program_never_cuts_below_the_target(self):
        """The containment invariant #10 gets for free must survive the trip through the SDF."""
        m = np.zeros((RES, RES), bool)
        m[14:24, 6:26] = True
        m &= self.fx.fp
        program = [self.fx.layer(m, 9)]
        occ, _ = self._occupancy_via_sdf(program)
        target = voxel_occupancy(self.fx.fp, self.fx.y0, self.fx.h)
        self.assertEqual(int((target & ~occ).sum()), 0)


def run_length(fp: np.ndarray, axis: int) -> np.ndarray:
    """Cells to the nearest non-footprint cell along ONE axis; `_dist_axis`, restated."""
    m = fp if axis == 1 else fp.T
    n = m.shape[1]
    fwd = np.zeros(m.shape, np.int16)
    acc = np.zeros(m.shape[0], np.int16)
    for i in range(n):
        acc = np.where(m[:, i], acc + 1, 0)
        fwd[:, i] = acc
    bwd = np.zeros(m.shape, np.int16)
    acc = np.zeros(m.shape[0], np.int16)
    for i in range(n - 1, -1, -1):
        acc = np.where(m[:, i], acc + 1, 0)
        bwd[:, i] = acc
    out = np.minimum(fwd, bwd)
    return out if axis == 1 else out.T


class TestCutRoof(unittest.TestCase):
    """A `CutRoof` cuts by distance to the wall, and the two roof kinds measure that differently.

    A `hip` measures it in every direction at once, so it compiles to one cap over the region's
    outline. A gable measures it along one axis, staying in its own row, so it compiles to a clause
    per eave -- the slanted plane, clipped to the rows its wall spans and to its own side of it.
    Both forms are tested on a footprint with a reflex corner, which is where the obvious
    compilation (an unclipped inward plane per wall) falls apart.
    """

    def _reference(self, fp, y0, y1, kind, eaves, rate):
        """`_roof_candidates`, restated: height = eaves + (d - 1) * rate, clipped to the blockout."""
        from scipy import ndimage
        d = {"hip": ndimage.distance_transform_edt(fp),
             "gable_x": run_length(fp, 1), "gable_z": run_length(fp, 0)}[kind]
        h = np.where(fp, np.int16(y1 - y0 + 1), 0).astype(np.int16)
        cand = np.minimum(h, np.floor(eaves + (d.astype(np.float32) - 1.0) * rate)).astype(np.int16)
        return np.where(fp, np.maximum(cand, 1), 0).astype(np.int16)

    def _agreement(self, fp, kind, eaves=3, rate=1.0):
        y0, y1 = 4, 21
        entry = dict(op="CutRoof", kind=kind, eaves=eaves, rate=rate)
        ops = layer_program_to_ops([entry], fp, y0, y1, res=RES)
        eb = EditableBuilding(footprint_envelope_sdf(fp, y0, y1, res=RES), ops)
        ref = voxel_occupancy(fp, y0, self._reference(fp, y0, y1, kind, eaves, rate))
        return iou(eb.to_occupancy(res=RES), ref), ops

    def test_convex_footprint_is_exact(self):
        fp = np.zeros((RES, RES), bool)
        fp[8:24, 6:26] = True
        for kind in ("hip", "gable_x", "gable_z"):
            with self.subTest(kind=kind):
                agreement, ops = self._agreement(fp, kind)
                self.assertEqual(len(ops), 1)
                self.assertEqual(agreement, 1.0)

    def test_gable_uses_only_the_eaves_facing_its_axis(self):
        fp = np.zeros((RES, RES), bool)
        fp[8:24, 6:26] = True
        _a, ops = self._agreement(fp, "gable_x")
        self.assertEqual(len(ops[0].planes), 2, "a rectangle has two x-facing eaves")
        self.assertIsNone(ops[0].roof)

    def test_a_hip_is_a_cap_over_the_outline_not_a_set_of_planes(self):
        fp = np.zeros((RES, RES), bool)
        fp[8:24, 6:26] = True
        _a, ops = self._agreement(fp, "hip")
        self.assertIsNone(ops[0].planes)
        self.assertEqual(len(ops[0].roof), 2, "(rate, offset)")

    def test_hip_survives_a_reflex_corner(self):
        """The case that rules out the plane form: at a reflex corner the nearest wall is a vertex."""
        fp = np.zeros((RES, RES), bool)
        fp[8:24, 6:26] = True
        fp[8:14, 20:26] = False
        for rate in (1.0, 2.0):
            with self.subTest(rate=rate):
                agreement, _ops = self._agreement(fp, "hip", rate=rate)
                self.assertGreater(agreement, 0.99)

    def test_a_roof_over_a_footprint_in_two_pieces(self):
        """`CutRoof` carries no region, so it falls back to the footprint -- which may be split."""
        fp = np.zeros((RES, RES), bool)
        fp[8:16, 4:12] = True
        fp[8:16, 18:26] = True
        agreement, ops = self._agreement(fp, "hip")
        self.assertEqual(len(ops), 2, "one cap per piece")
        self.assertGreater(agreement, 0.99)

    def test_gable_survives_a_reflex_corner(self):
        """The case a bare half-space per eave gets wrong, and why each clause is clipped.

        A gable's rule is a *run length along one axis*: how far the wall is if you walk in x and
        stay in your own row. An unclipped plane does neither -- the one raised by the step of an L
        keeps cutting across the rows below it, which measured 0.71 here and as low as 0.04 on a
        real building. Clipping each eave to the rows its wall spans, and to its own side of it,
        restores the run length exactly.
        """
        fp = np.zeros((RES, RES), bool)
        fp[8:24, 6:26] = True
        fp[8:14, 20:26] = False
        for kind in ("gable_x", "gable_z"):
            for rate in (1.0, 2.0):
                with self.subTest(kind=kind, rate=rate):
                    agreement, _ops = self._agreement(fp, kind, rate=rate)
                    self.assertEqual(agreement, 1.0)


class TestEditAndUndo(unittest.TestCase):
    """A recipe you cannot take an operation back off is not reversible."""

    def setUp(self):
        self.fx = ProgramFixture()
        m = np.zeros((RES, RES), bool)
        m[14:24, 6:26] = True
        m &= self.fx.fp
        self.program = [self.fx.layer(m, 9)]
        self.ops = layer_program_to_ops(self.program, self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def test_undo_restores_the_envelope(self):
        eb = EditableBuilding(self.base, list(self.ops))
        carved = eb.to_occupancy(res=RES)
        eb.undo()
        restored = eb.to_occupancy(res=RES)
        envelope = voxel_occupancy(
            self.fx.fp, self.fx.y0,
            np.where(self.fx.fp, np.int16(self.fx.y1 - self.fx.y0 + 1), 0).astype(np.int16))
        self.assertLess(carved.sum(), envelope.sum(), "the layer must actually remove volume")
        self.assertEqual(iou(restored, envelope), 1.0)

    def test_editing_one_layer_leaves_the_rest_of_the_building_alone(self):
        """`CONTEXT.md`: any single decision can be re-rolled "without destroying the rest"."""
        second = np.zeros((RES, RES), bool)
        second[8:14, 6:14] = True
        second &= self.fx.fp
        program = self.program + [self.fx.layer(second, 12)]
        ops = layer_program_to_ops(program, self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        eb = EditableBuilding(self.base, ops)

        heights = lambda occ: occ.sum(axis=1)                   # column heights, [z, x]
        before = heights(eb.to_occupancy(res=RES))
        step = 2.0 / (RES - 1)
        edited = eb.ops[-1]
        edited.size = (edited.size[0] - 3 * step, edited.size[1])   # cut three voxels deeper
        after = heights(eb.to_occupancy(res=RES))

        np.testing.assert_array_equal(after[second], before[second] - 3)
        np.testing.assert_array_equal(after[~second], before[~second])

    def test_state_survives_a_json_round_trip(self):
        eb = EditableBuilding(self.base, list(self.ops))
        state = json.loads(json.dumps(eb.edit_state()))
        again = EditableBuilding.from_state(self.base, state)
        self.assertEqual(iou(eb.to_occupancy(res=RES), again.to_occupancy(res=RES)), 1.0)


# ================================================================================================
# #4 -- the semantic architectural edit algebra
# ================================================================================================


class TestOntology(unittest.TestCase):
    """🔑 #4's first question: what are the operations, and what does each one mean?

    The palette had grown into a flat list mixing raw CSG primitives (box, sphere) with the three
    architectural operations #10 recovered, and nothing declared which was which. `ALGEBRA` is that
    declaration, and it carries the one distinction the rest of the algebra turns on: whether an
    operation is expressible as a **height field**, because on this corpus that decides whether it
    can ever be learned.
    """

    def test_the_palette_is_exactly_the_ontology(self):
        """Both directions. A kind the compiler accepts but the algebra does not describe is a hole
        in the spec; a kind the algebra declares but the palette omits is unreachable. `PALETTE` and
        `PROGRAM_KINDS` are derived from `ALGEBRA` so this holds by construction -- the test is here
        to catch anyone re-literalising them, which is how the three lists drifted before."""
        self.assertEqual(set(PALETTE), set(ALGEBRA))
        self.assertEqual(set(PROGRAM_KINDS), {k for k, v in ALGEBRA.items() if v.tier == CORE})

    def test_the_palette_keeps_the_order_the_host_shows(self):
        """⚠️ `tools/blender_addon/.../bridge.py` returns `PALETTE` straight to a UI enum, so its
        order is user-visible and deriving it must not reshuffle the hand tools."""
        self.assertEqual(PALETTE[:8],
                         ("box", "rounded_box", "sphere", "cylinder", "cone", "gable", "hip",
                          "element"))

    def test_the_three_recovered_operations_are_the_core(self):
        for kind in PROGRAM_KINDS:
            spec = ALGEBRA[kind]
            self.assertEqual(spec.tier, CORE)
            self.assertTrue(spec.height_map, f"{kind} must be height-map representable")

    def test_layer_and_ramp_are_bidirectional_but_cut_roof_stays_subtract_only(self):
        """#140: `layer`/`ramp` gain a real additive mode; `cut_roof`'s additive mirror already
        exists as the volumetric tier's `gable`/`hip`, so it keeps the subtract-only rule #10
        measured (`missing` = 0 on 714/714 -- real massing is only ever cut from its envelope)."""
        self.assertFalse(ALGEBRA["layer"].subtractive_only)
        self.assertFalse(ALGEBRA["ramp"].subtractive_only)
        self.assertTrue(ALGEBRA["cut_roof"].subtractive_only, "#10 measured missing=0 on 714/714")

    def test_the_csg_primitives_are_volumetric_and_not_height_map_representable(self):
        for kind in ("box", "sphere", "cylinder", "cone"):
            self.assertEqual(ALGEBRA[kind].tier, VOLUMETRIC)
            self.assertFalse(ALGEBRA[kind].height_map)

    def test_every_name_the_ticket_lists_is_resolved(self):
        """⚠️ #4 says the algebra must represent nine named things. Each one is either a core
        operation, a spelling of one, or a volumetric operation that CANNOT FIRE on this corpus --
        and the table says which, so none of them is quietly dropped."""
        for name in ("courtyard", "passage", "arcade", "light well", "setback", "terrace",
                     "roof cut", "wing", "roof volume"):
            self.assertIn(name, ARCHITECTURAL_VOCABULARY, f"{name!r} is unaccounted for")
            entry = ARCHITECTURAL_VOCABULARY[name]
            self.assertTrue(entry.note, f"{name!r} has no stated resolution")
            if entry.kind is not None:
                self.assertIn(entry.kind, ALGEBRA)

    def test_a_setback_is_a_layer_and_not_its_own_operation(self):
        """The owner's measurement on this ticket: in a height field a setback IS a Layer whose
        polygon is the inward offset of the footprint, and the fitter finds it as one."""
        self.assertEqual(ARCHITECTURAL_VOCABULARY["setback"].kind, "layer")

    def test_the_void_operations_are_declared_unfireable_on_this_corpus(self):
        """0 voxels of through-void in 4,324,919 of carve. They are representable in the SDF and
        have zero training signal, which is a different claim from 'unsupported'."""
        for name in ("courtyard", "passage", "arcade", "light well"):
            entry = ARCHITECTURAL_VOCABULARY[name]
            self.assertFalse(entry.learnable_here,
                             f"{name} cannot be learned from a corpus with no through-voids")
            self.assertFalse(ALGEBRA[entry.kind].height_map)


class TestOpValidity(unittest.TestCase):
    """#4's 'constrained geometry per type' and 'invalid references', as a checkable predicate.

    Nothing previously stopped a `layer` reaching the compiler with no polygon; it raised somewhere
    inside a prism instead, which is a stack trace rather than a diagnosis.
    """

    def _layer(self, **kw):
        d = dict(kind="layer", mode="subtract", size=(-0.5, 0.5),
                 polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
        d.update(kw)
        return EditOp(**d)

    def test_a_well_formed_layer_has_no_problems(self):
        self.assertEqual(op_problems(self._layer()), [])

    def test_a_layer_without_a_polygon_is_refused(self):
        self.assertTrue(any("polygon" in p for p in op_problems(self._layer(polygon=None))))

    def test_a_ring_of_two_vertices_is_not_a_polygon(self):
        bad = self._layer(polygon=[[[0.0, 0.0], [1.0, 0.0]]])
        self.assertTrue(any("vertices" in p or "ring" in p for p in op_problems(bad)))

    def test_an_unknown_kind_is_refused_by_name(self):
        probs = op_problems(EditOp(kind="buttress"))
        self.assertTrue(any("buttress" in p for p in probs))

    def test_layer_now_accepts_additive_mode(self):
        """#140: raising a column is validated exactly like lowering one."""
        self.assertEqual(op_problems(self._layer(mode="add")), [])

    def test_ramp_now_accepts_additive_mode(self):
        r = dict(kind="ramp", mode="add", size=(-0.5, 0.5),
                 polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]],
                 planes=[[[0.0, 1.0, 0.0, 0.0]]])
        self.assertEqual(op_problems(EditOp(**r)), [])

    def test_cut_roof_may_not_be_additive(self):
        """#140 keeps `cut_roof` subtract-only -- its additive mirror is the volumetric tier's
        `gable`/`hip`, unlike `layer`/`ramp`, which #140 makes bidirectional above."""
        roof = dict(kind="cut_roof", mode="add", size=(-0.5, 0.5),
                    polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
                    roof=[0.5, 0.1])
        self.assertTrue(any("subtract" in p for p in op_problems(EditOp(**roof))))

    def test_a_volumetric_kind_is_unaffected_by_140(self):
        """`box` was never subtract-only and stays that way -- pins that the kind table's other
        entries did not move when `layer`/`ramp` flipped."""
        self.assertFalse(ALGEBRA["box"].subtractive_only)
        self.assertEqual(op_problems(EditOp(kind="box", mode="add", size=(1.0, 1.0, 1.0))), [])

    def test_mode_must_be_add_or_subtract(self):
        self.assertTrue(any("mode" in p for p in op_problems(self._layer(mode="sideways"))))

    def test_a_ramp_needs_exactly_one_plane_clause(self):
        r = dict(kind="ramp", mode="subtract", size=(-0.5, 0.5),
                 polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
        self.assertEqual(op_problems(EditOp(**r, planes=[[[0.0, 1.0, 0.0, 0.0]]])), [])
        self.assertTrue(op_problems(EditOp(**r, planes=None)))

    def test_program_problems_reports_the_offending_index(self):
        ops = [self._layer(), self._layer(polygon=None)]
        probs = program_problems(ops)
        self.assertTrue(probs)
        self.assertTrue(any(p.startswith("op 1") for p in probs), probs)


class TestCommutativity(unittest.TestCase):
    """🔑🔑 #4's 'ordering and commutativity', and the decision the whole algebra turns on.

    Measured on 250 recovered programs before this was written: 78% have two operations whose
    regions overlap, and permuting the operations changed the compiled building on **69.6%** of
    them -- so the serialised algebra was ORDERED, and nothing said so. The cause was entirely that
    the height-map replay applied `Layer` as a SET (`where(region, v, h)`), which can RAISE a column
    a previous operation had lowered.

    Reading `Layer` as a MIN instead changed the result on **0 of 250** recovered programs, and made
    permutation change nothing on **0 of 2,000** permutations. So commutativity was available for
    free, and it is what makes deletion, equivalence and a canonical form well defined at all.
    """

    def setUp(self):
        self.fx = ProgramFixture()
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        a = np.zeros((RES, RES), bool); a[10:22, 8:24] = True; a &= self.fx.fp
        b = np.zeros((RES, RES), bool); b[14:24, 6:20] = True; b &= self.fx.fp
        self.assertTrue((a & b).any(), "the fixture must exercise OVERLAPPING regions")
        prog = [self.fx.layer(a, 12), self.fx.layer(b, 8)]
        self.ops = layer_program_to_ops(prog, self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def test_a_subtractive_program_commutes(self):
        forward = EditableBuilding(self.base, list(self.ops)).to_occupancy(res=RES)
        for perm in itertools.permutations(range(len(self.ops))):
            got = EditableBuilding(self.base, [self.ops[i] for i in perm]).to_occupancy(res=RES)
            self.assertEqual(iou(got, forward), 1.0, f"order {perm} changed the building")

    def test_commutes_is_true_for_an_all_subtractive_program(self):
        self.assertTrue(commutes(self.ops))

    def test_one_additive_operation_breaks_it(self):
        """A union does not commute with a subtraction, so the predicate must refuse."""
        added = list(self.ops) + [EditOp(kind="box", mode="add", size=(0.2, 0.2, 0.2))]
        self.assertFalse(commutes(added))

    def test_the_empty_program_commutes(self):
        self.assertTrue(commutes([]))


class TestCanonicalForm(unittest.TestCase):
    """#4's 'canonical normal form' and 'equivalence'.

    Only meaningful because the core commutes: a normal form for an ORDERED algebra would have to
    preserve order, and then two spellings of the same building would stay distinguishable.
    """

    def setUp(self):
        self.fx = ProgramFixture()
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        a = np.zeros((RES, RES), bool); a[10:22, 8:24] = True; a &= self.fx.fp
        b = np.zeros((RES, RES), bool); b[14:24, 6:20] = True; b &= self.fx.fp
        self.ops = layer_program_to_ops([self.fx.layer(a, 12), self.fx.layer(b, 8)],
                                        self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def test_a_permutation_has_the_same_canonical_form(self):
        self.assertEqual(canonical_form(self.ops), canonical_form(list(reversed(self.ops))))

    def test_it_is_idempotent(self):
        once = canonical_form(self.ops)
        self.assertEqual(once, canonical_form([EditOp.from_dict(d) for d in once]))

    def test_a_different_program_has_a_different_canonical_form(self):
        other = list(self.ops[:1])
        self.assertNotEqual(canonical_form(self.ops), canonical_form(other))

    def test_it_refuses_a_program_that_does_not_commute(self):
        """⚠️ Sorting an ordered program would silently change the building it denotes."""
        with self.assertRaises(ValueError):
            canonical_form(list(self.ops) + [EditOp(kind="box", mode="add")])

    def test_equivalence_is_decided_on_the_geometry_not_the_spelling(self):
        shuffled = list(reversed(self.ops))
        self.assertTrue(equivalent(self.base, self.ops, shuffled, res=RES))
        self.assertFalse(equivalent(self.base, self.ops, self.ops[:1], res=RES))


class TestDeletion(unittest.TestCase):
    """#4's 'deletion'. `undo()` pops the last operation; an edit stack that can only be unwound
    from the top cannot serve #3's edit locality, where a user re-rolls one decision and everything
    unrelated must survive. Commutativity is what makes removing operation *i* well defined."""

    def setUp(self):
        self.fx = ProgramFixture()
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        a = np.zeros((RES, RES), bool); a[10:22, 8:24] = True; a &= self.fx.fp
        b = np.zeros((RES, RES), bool); b[14:24, 6:20] = True; b &= self.fx.fp
        self.ops = layer_program_to_ops([self.fx.layer(a, 12), self.fx.layer(b, 8)],
                                        self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def test_removing_the_first_equals_the_program_without_it(self):
        eb = EditableBuilding(self.base, list(self.ops))
        eb.remove(0)
        want = EditableBuilding(self.base, list(self.ops[1:])).to_occupancy(res=RES)
        self.assertEqual(iou(eb.to_occupancy(res=RES), want), 1.0)

    def test_it_returns_the_operation_it_removed(self):
        eb = EditableBuilding(self.base, list(self.ops))
        self.assertEqual(eb.remove(0).to_dict(), self.ops[0].to_dict())
        self.assertEqual(len(eb.ops), len(self.ops) - 1)

    def test_an_out_of_range_index_is_refused(self):
        eb = EditableBuilding(self.base, list(self.ops))
        with self.assertRaises(IndexError):
            eb.remove(len(self.ops))

    def test_a_negative_index_is_refused_rather_than_wrapping(self):
        """⚠️ Python would happily delete the LAST operation for `remove(-1)`, which is a silent
        wrong answer when the caller meant an id it failed to resolve (#4: invalid references)."""
        eb = EditableBuilding(self.base, list(self.ops))
        with self.assertRaises(IndexError):
            eb.remove(-1)


class TestHeightMapRepresentable(unittest.TestCase):
    """The property that decides which compiler can run a program -- and, on this corpus, whether
    it could ever have been learned."""

    def setUp(self):
        self.fx = ProgramFixture()
        m = np.zeros((RES, RES), bool); m[14:24, 6:26] = True; m &= self.fx.fp
        self.ops = layer_program_to_ops([self.fx.layer(m, 9)],
                                        self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def test_a_recovered_program_is_height_map_representable(self):
        self.assertTrue(is_height_map_representable(self.ops))

    def test_a_subtracted_box_is_not(self):
        """This is how a courtyard would be cut, and why it leaves the 2.5-D world."""
        self.assertFalse(is_height_map_representable(
            list(self.ops) + [EditOp(kind="box", mode="subtract", size=(0.2, 0.9, 0.2))]))

    def test_the_empty_program_is(self):
        self.assertTrue(is_height_map_representable([]))


# ================================================================================================
# #140 -- layer and ramp become bidirectional
# ================================================================================================


class TestBidirectionalCore(unittest.TestCase):
    """#140: `layer`/`ramp` gain a real, learnable `mode="add"` mirroring the lower-only
    `mode="subtract"` they already had. `cut_roof` and the volumetric tier are unaffected --
    `layer_program_to_ops` still hard-codes `mode="subtract"`, so this exercises a hand-built
    additive op, the only way one can exist today (#5 owns the training signal for it).

    Mixing add and subtract on overlapping regions reopens exactly the ordering #4 proved away for
    an all-subtractive program (`union` does not commute with `subtract`) -- #3's grilling decision,
    made knowingly. `commutes`/`canonical_form` needed no code change for this (they already key off
    `op.mode`, not `op.kind`); what follows pins that against a REAL additive `layer`/`ramp`, not
    only the synthetic `box` used elsewhere in this file.
    """

    def setUp(self):
        self.fx = ProgramFixture()
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)
        m = np.zeros((RES, RES), bool)
        m[14:24, 6:26] = True
        m &= self.fx.fp
        self.region = m
        program = [self.fx.layer(m.copy(), 9)]
        self.sub_op = layer_program_to_ops(program, self.fx.fp, self.fx.y0, self.fx.y1, res=RES)[0]
        self.add_op = replace(self.sub_op, mode="add")

    def _heights(self, ops):
        return EditableBuilding(self.base, ops).to_occupancy(res=RES).sum(axis=1)   # [z, x]

    def test_add_and_subtract_both_pass_validation_on_layer(self):
        self.assertEqual(op_problems(self.sub_op), [])
        self.assertEqual(op_problems(self.add_op), [])

    def test_add_then_subtract_differs_from_subtract_then_add(self):
        """🔑 The trade #3 made knowingly, made concrete: subtract-then-add REFILLS the region
        (add is the last word), add-then-subtract CUTS it (subtract is the last word) -- the
        mirror-image outcome of insertion order, not an arbitrary difference."""
        sub_then_add = self._heights([self.sub_op, self.add_op])
        add_then_sub = self._heights([self.add_op, self.sub_op])
        self.assertFalse(np.array_equal(sub_then_add[self.region], add_then_sub[self.region]))
        self.assertTrue(np.all(sub_then_add[self.region] > add_then_sub[self.region]))
        full = self.fx.y1 - self.fx.y0 + 1
        np.testing.assert_array_equal(sub_then_add[self.region], full)

    def test_replay_is_deterministic_and_stable_across_repeated_calls(self):
        ops = [self.sub_op, self.add_op]
        first = self._heights(ops)
        second = self._heights(ops)
        third = self._heights(list(ops))          # a fresh list, same ops in the same order
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, third)

    def test_non_overlapping_add_and_subtract_still_compose_order_free(self):
        """#3's locality invariant: two ops whose regions do not overlap compose identically
        regardless of order -- free, inherited from #4, unchanged by mixing modes."""
        other = np.zeros((RES, RES), bool)
        other[8:14, 6:14] = True
        other &= self.fx.fp
        self.assertFalse((other & self.region).any(), "fixture must be disjoint from self.region")
        other_program = [self.fx.layer(other.copy(), 12)]
        other_sub = layer_program_to_ops(other_program, self.fx.fp, self.fx.y0, self.fx.y1,
                                         res=RES)[0]
        other_add = replace(other_sub, mode="add")
        forward = self._heights([self.sub_op, other_add])
        backward = self._heights([other_add, self.sub_op])
        np.testing.assert_array_equal(forward, backward)

    def test_a_program_with_a_real_additive_layer_does_not_commute(self):
        self.assertFalse(commutes([self.sub_op, self.add_op]))

    def test_commutes_stays_conservative_on_an_all_additive_program(self):
        """`commutes` checks `op.mode == "subtract"` for every op -- #140's own acceptance
        criterion pins that reading unchanged ('false for any program containing an additive
        operation'), even though a pure union is mathematically order-free too. #3's write-up
        notes that case only in passing ('incidentally'); it is not a claim this predicate makes,
        and this test is here so nobody 'fixes' it into a silent behaviour change."""
        other = np.zeros((RES, RES), bool)
        other[8:14, 6:14] = True
        other &= self.fx.fp
        other_program = [self.fx.layer(other.copy(), 12)]
        other_add = replace(
            layer_program_to_ops(other_program, self.fx.fp, self.fx.y0, self.fx.y1, res=RES)[0],
            mode="add")
        self.assertFalse(commutes([self.add_op, other_add]))

    def test_canonical_form_refuses_a_program_with_a_real_additive_ramp(self):
        r = EditOp(kind="ramp", mode="add", size=self.sub_op.size, polygon=self.sub_op.polygon,
                   planes=[[[0.0, 1.0, 0.0, 0.0]]])
        with self.assertRaises(ValueError):
            canonical_form([self.sub_op, r])


class TestRecoveredProgramsStillCommute(unittest.TestCase):
    """#140's regression clause: real recovered programs must still report as commuting now that
    the kind table stops forbidding `add` on `layer`/`ramp`.

    #10's fitter never emits an additive op -- `_op_for` hard-codes `mode="subtract"` for every
    recovered kind, untouched by this ticket -- so this is a property of the DATA that this pins
    against the artifact rather than trusting it stays true as the kind table changes underneath.
    """

    ARTIFACT = REPO / "execution/artifacts/program_recovery_714.json"

    def test_every_recovered_program_still_commutes(self):
        data = json.loads(self.ARTIFACT.read_text())
        rows = data["per_building"]
        res = 64
        # only a CutRoof entry with no region of its own falls back to this mask (11/714 rows);
        # every op's MODE -- what `commutes` checks -- is "subtract" regardless of footprint shape.
        fp = np.ones((res, res), bool)
        tested = 0
        for building_id, row in rows.items():
            program = row.get("program") or []
            if not program:
                continue
            ops = layer_program_to_ops(program, fp, 0, res - 1, res=res)
            self.assertTrue(commutes(ops), f"building {building_id} no longer commutes")
            tested += 1
        self.assertGreater(tested, 400, "the artifact should carry real recovered programs")


# ================================================================================================
# #141 -- every edit operation gets a stable identity
# ================================================================================================


class TestOperationIdentity(unittest.TestCase):
    """#141: an `EditOp` carries a stable id, auto-assigned when the caller doesn't supply one, so
    "reroll this decision" is a lookup rather than a recomputed list position. `TestEditOpRoundTrip`
    above already pins the JSON round trip and the legacy-state (no id) load; this covers the rest
    of #141's contract: uniqueness, an explicit override, and that `canonical_form` -- #4/#140's
    content-only equivalence test -- does not start treating id as content.
    """

    def test_a_new_op_has_a_usable_id_without_being_supplied(self):
        self.assertTrue(EditOp(kind="box").id)

    def test_two_independently_constructed_ops_get_distinct_ids(self):
        self.assertNotEqual(EditOp(kind="box").id, EditOp(kind="box").id)

    def test_an_explicit_id_is_honored_not_overwritten(self):
        self.assertEqual(EditOp(kind="box", id="my-stable-id").id, "my-stable-id")

    def test_canonical_form_ignores_id(self):
        """Two ops built separately, describing the same edit, must still denote the same building
        as far as `canonical_form` is concerned -- it is a proxy for `equivalent()`, which never
        reads `id` because `composed()` never does."""
        x = EditOp(kind="box", mode="subtract", size=(0.2, 0.2, 0.2))
        y = EditOp(kind="box", mode="subtract", size=(0.2, 0.2, 0.2))
        self.assertNotEqual(x.id, y.id)
        self.assertEqual(canonical_form([x]), canonical_form([y]))
        self.assertNotIn("id", canonical_form([x])[0])


class TestRemoveById(unittest.TestCase):
    """#141: an id-addressed removal path alongside the existing index-based `remove`."""

    def test_removes_the_right_operation_regardless_of_position(self):
        ops = [EditOp(kind="box"), EditOp(kind="sphere"), EditOp(kind="cone")]
        eb = EditableBuilding(None, list(ops))
        removed = eb.remove_by_id(ops[1].id)
        self.assertEqual(removed.kind, "sphere")
        self.assertEqual([o.kind for o in eb.ops], ["box", "cone"])

    def test_survives_other_edits_shifting_its_position(self):
        """#141's own motivating case: reroll "the operation I just added" without recomputing its
        current index after later edits inserted around it."""
        target = EditOp(kind="sphere")
        eb = EditableBuilding(None, [target])
        eb.add(EditOp(kind="box"))
        eb.add(EditOp(kind="cone"))
        removed = eb.remove_by_id(target.id)
        self.assertIs(removed, target)
        self.assertEqual([o.kind for o in eb.ops], ["box", "cone"])

    def test_an_unknown_id_is_refused_not_a_silent_no_op(self):
        eb = EditableBuilding(None, [EditOp(kind="box")])
        with self.assertRaises(KeyError):
            eb.remove_by_id("does-not-exist")
        self.assertEqual(len(eb.ops), 1, "a failed removal must not silently drop any operation")

    def test_the_index_based_path_is_unchanged(self):
        """#141 adds a sibling path; it must not alter `remove`'s own behaviour, negative-index
        refusal included."""
        eb = EditableBuilding(None, [EditOp(kind="box"), EditOp(kind="sphere")])
        with self.assertRaises(IndexError):
            eb.remove(-1)
        removed = eb.remove(0)
        self.assertEqual(removed.kind, "box")
        self.assertEqual([o.kind for o in eb.ops], ["sphere"])


# ================================================================================================
# #142 -- snap new operation geometry to the working grid
# ================================================================================================


class TestSnapToGrid(unittest.TestCase):
    """#142: a pure function that rounds a NEW op's region geometry onto the module's own voxel
    pitch, so it meets existing geometry at a clean edge instead of an arbitrary sub-voxel jitter.

    `self.grid_op` is real recovered geometry (via `layer_program_to_ops`) -- already exactly on
    the grid `snap_to_grid` targets -- used as ground truth for what "the same edge" means.
    """

    def setUp(self):
        self.fx = ProgramFixture()
        m = np.zeros((RES, RES), bool)
        m[14:24, 6:26] = True
        m &= self.fx.fp
        program = [self.fx.layer(m, 9)]
        self.grid_op = layer_program_to_ops(program, self.fx.fp, self.fx.y0, self.fx.y1,
                                            res=RES)[0]

    def _jitter(self, op, dx, dz, dy_lo, dy_hi):
        return replace(op, polygon=[[[x + dx, z + dz] for x, z in ring] for ring in op.polygon],
                       size=(op.size[0] + dy_lo, op.size[1] + dy_hi))

    def test_jittered_geometry_snaps_onto_the_grid(self):
        jittered = self._jitter(self.grid_op, 2e-4, -1.5e-4, 1e-4, -3e-5)
        snapped = snap_to_grid(jittered, res=RES)
        np.testing.assert_allclose(np.asarray(snapped.polygon), np.asarray(self.grid_op.polygon))
        np.testing.assert_allclose(np.asarray(snapped.size), np.asarray(self.grid_op.size))

    def test_two_ops_at_different_subvoxel_offsets_meet_at_the_same_edge(self):
        a = snap_to_grid(self._jitter(self.grid_op, 2e-4, -1.5e-4, 1e-4, -3e-5), res=RES)
        b = snap_to_grid(self._jitter(self.grid_op, -1e-4, 2e-4, -2e-4, 1e-4), res=RES)
        np.testing.assert_allclose(np.asarray(a.polygon), np.asarray(b.polygon))
        np.testing.assert_allclose(np.asarray(a.size), np.asarray(b.size))

    def test_an_already_grid_aligned_op_is_unchanged(self):
        """The module's own recovered geometry already sits on this grid (⚠️ 'half-voxel
        offsets', the section header above `_snap_scalar`); snapping it must be a no-op."""
        snapped = snap_to_grid(self.grid_op, res=RES)
        np.testing.assert_allclose(np.asarray(snapped.polygon), np.asarray(self.grid_op.polygon))
        np.testing.assert_allclose(np.asarray(snapped.size), np.asarray(self.grid_op.size))

    def test_smooth_is_untouched_and_keeps_its_default(self):
        self.assertEqual(self.grid_op.smooth, 0.0, "the hard-edge default")
        default = snap_to_grid(self._jitter(self.grid_op, 3e-4, -3e-4, 2e-4, -2e-4), res=RES)
        self.assertEqual(default.smooth, 0.0)
        blended = replace(self.grid_op, smooth=0.4)
        self.assertEqual(snap_to_grid(blended, res=RES).smooth, 0.4)

    def test_the_input_is_not_mutated(self):
        jittered = self._jitter(self.grid_op, 3e-4, -3e-4, 2e-4, -2e-4)
        before_polygon = json.loads(json.dumps(jittered.polygon))
        before_size = tuple(jittered.size)
        snap_to_grid(jittered, res=RES)
        self.assertEqual(jittered.polygon, before_polygon)
        self.assertEqual(jittered.size, before_size)

    def test_id_is_preserved_snapping_does_not_mint_a_new_operation(self):
        jittered = self._jitter(self.grid_op, 3e-4, -3e-4, 2e-4, -2e-4)
        self.assertEqual(snap_to_grid(jittered, res=RES).id, jittered.id)

    def test_an_op_with_no_polygon_passes_through_unchanged(self):
        """Volumetric CSG kinds (`box`, `sphere`, ...) have no region geometry to snap."""
        box = EditOp(kind="box", size=(0.3, 0.3, 0.3))
        self.assertIs(snap_to_grid(box, res=RES), box)

    def test_from_dict_does_not_snap_stored_state(self):
        """#3's opt-in decision, from the load path: `EditOp.from_dict` reproduces stored geometry
        exactly -- jitter and all -- until a caller explicitly calls `snap_to_grid`."""
        jittered = self._jitter(self.grid_op, 3e-4, -3e-4, 2e-4, -2e-4)
        loaded = EditOp.from_dict(json.loads(json.dumps(jittered.to_dict())))
        np.testing.assert_array_equal(np.asarray(loaded.polygon), np.asarray(jittered.polygon))
        self.assertFalse(np.allclose(np.asarray(loaded.polygon), np.asarray(self.grid_op.polygon)),
                         "the fixture's jitter must be large enough to actually move the vertices")

    def test_adding_a_jittered_op_to_a_building_does_not_snap_it(self):
        """#142's own acceptance criterion: snapping is opt-in, not automatic on authorship."""
        jittered = self._jitter(self.grid_op, 3e-4, -3e-4, 2e-4, -2e-4)
        eb = EditableBuilding(None)
        eb.add(jittered)
        np.testing.assert_array_equal(np.asarray(eb.ops[-1].polygon), np.asarray(jittered.polygon))


# ================================================================================================
# #143 -- refuse to commit an invalid operation to a building
# ================================================================================================


class TestAddValidatesBeforeAppending(unittest.TestCase):
    """#143: `EditableBuilding.add` runs the candidate through `op_problems` -- the same
    per-operation predicate `program_problems`/`canonical_form` already use -- BEFORE touching the
    stack, so a malformed op can never silently enter a building's state.
    """

    def setUp(self):
        self.fx = ProgramFixture()
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def test_a_well_formed_operation_is_appended_exactly_as_before(self):
        op = EditOp(kind="box", mode="subtract", size=(0.1, 0.1, 0.1))
        eb = EditableBuilding(self.base)
        result = eb.add(op)
        self.assertIs(result, eb, "add still returns self for chaining")
        self.assertEqual(eb.ops, [op])

    def test_a_missing_required_field_is_refused_with_a_clear_message(self):
        eb = EditableBuilding(self.base)
        with self.assertRaises(ValueError) as ctx:
            eb.add(EditOp(kind="layer", mode="subtract"))          # no polygon
        self.assertIn("polygon", str(ctx.exception))
        self.assertEqual(eb.ops, [])

    def test_the_wrong_mode_for_a_kind_is_refused_with_a_clear_message(self):
        """#140's cut_roof-stays-subtract-only rule is exactly the case the ticket says this
        closes automatically, with no further work needed."""
        eb = EditableBuilding(self.base)
        roof = EditOp(kind="cut_roof", mode="add", size=(-0.5, 0.5),
                      polygon=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
                      roof=[0.5, 0.1])
        with self.assertRaises(ValueError) as ctx:
            eb.add(roof)
        self.assertIn("subtract", str(ctx.exception))

    def test_an_invalid_mode_string_is_refused_with_a_clear_message(self):
        eb = EditableBuilding(self.base)
        with self.assertRaises(ValueError) as ctx:
            eb.add(EditOp(kind="box", mode="sideways"))
        self.assertIn("mode", str(ctx.exception))

    def test_a_rejected_append_leaves_the_stack_byte_for_byte_unchanged(self):
        kept = EditOp(kind="box", mode="subtract", size=(0.2, 0.2, 0.2))
        eb = EditableBuilding(self.base, [kept])
        before = json.dumps([op.to_dict() for op in eb.ops], sort_keys=True)
        with self.assertRaises(ValueError):
            eb.add(EditOp(kind="layer", mode="subtract"))
        after = json.dumps([op.to_dict() for op in eb.ops], sort_keys=True)
        self.assertEqual(before, after)
        self.assertEqual(eb.ops, [kept])


# ================================================================================================
# #144 -- prove the edit-locality invariant on a mixed program
# ================================================================================================


class TestEditLocalityInvariant(unittest.TestCase):
    """#3's actual crux, made checkable rather than left as a stated intention: `composed()` folds
    operations left to right, so one operation's COMPILED CONTRIBUTION -- the set of voxels its own
    application toggles, given whatever the ops before it already built -- can only be changed by
    an operation inserted BEFORE it in the list, never one inserted after, and never anything
    elsewhere on the building. Removing operation X by id (#141) is the case this ticket pins: it
    must leave every operation before X untouched, leave every later operation whose region does
    NOT overlap X's untouched, and it MAY change a later operation whose region does overlap X's --
    on a program that genuinely mixes add and subtract (#140), where #4's free commutativity no
    longer covers the whole space.

    Four regions on one footprint, in insertion order:
      `r_before`   -- BEFORE the removed op: a SUBTRACT that lowers a broad region (containing
                      `r_removed`) first, so the later ADD has real material to raise back up.
                      ⚠️ It DELIBERATELY overlaps the removed op's region too, so "unaffected"
                      here proves the invariant is about POSITION, not about overlap.
      `r_removed`  -- the op that gets removed by id: an ADD, raising `r_before`'s lowered column
                      back up within its own (smaller) region. This is the one real add-mode
                      operation, mixing with the three subtracts around it (#140).
      `r_disjoint` -- AFTER the removed op, region disjoint from it (though it may sit inside
                      `r_before`'s broad lowered area -- irrelevant, since `r_before` is present
                      either way; only overlap with `r_removed` itself can matter here).
      `r_overlap`  -- AFTER the removed op, region overlapping it (and contained in `r_before`'s
                      lowered area outside the intersection, so the ONLY thing that differs
                      with/without `r_removed` is the raised sub-patch this op then cuts back down).
    """

    def setUp(self):
        self.fx = ProgramFixture()
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

        def region(z0, z1, x0, x1):
            m = np.zeros((RES, RES), bool)
            m[z0:z1, x0:x1] = True
            m &= self.fx.fp
            self.assertTrue(m.any(), "fixture region must actually intersect the footprint")
            return m

        self.r_before = region(10, 22, 8, 22)
        self.r_removed = region(12, 18, 10, 18)
        self.r_disjoint = region(8, 14, 6, 10)
        self.r_overlap = region(15, 21, 15, 21)

        self.assertTrue((self.r_before & self.r_removed).any(),
                        "deliberately overlapping, to sharpen the 'before is unaffected' claim")
        self.assertFalse((self.r_disjoint & self.r_removed).any())
        self.assertTrue((self.r_overlap & self.r_removed).any())

        def op_for(mask, height):
            return layer_program_to_ops([self.fx.layer(mask, height)], self.fx.fp, self.fx.y0,
                                        self.fx.y1, res=RES)[0]

        op_before = op_for(self.r_before, 5)                          # SUBTRACT: lower it first
        op_removed = replace(op_for(self.r_removed, 15), mode="add")  # ADD: raise it back, partly
        op_disjoint = op_for(self.r_disjoint, 10)
        op_overlap = op_for(self.r_overlap, 8)

        self.ops = [op_before, op_removed, op_disjoint, op_overlap]
        self.removed_id = op_removed.id

    def _contribution(self, ops, index):
        """The voxels operation `ops[index]`'s own application toggles: the occupancy delta
        between compiling everything through it and everything up to (not including) it."""
        up_to = EditableBuilding(self.base, ops[:index + 1]).to_occupancy(res=RES)
        before = EditableBuilding(self.base, ops[:index]).to_occupancy(res=RES)
        return up_to ^ before

    def _after_removal(self):
        eb = EditableBuilding(self.base, list(self.ops))
        removed = eb.remove_by_id(self.removed_id)
        self.assertEqual(removed.id, self.removed_id)
        return eb.ops                      # [op_before, op_disjoint, op_overlap]

    def test_the_fixture_actually_mixes_modes_with_an_overlapping_pair(self):
        """Sanity check against the ticket's own precondition, not the invariant itself."""
        self.assertEqual({op.mode for op in self.ops}, {"add", "subtract"})
        self.assertFalse(commutes(self.ops))

    def test_every_operation_before_the_removed_one_is_unaffected(self):
        after_ops = self._after_removal()
        before = self._contribution(self.ops, 0)          # op_before, index 0 in both lists
        after = self._contribution(after_ops, 0)
        np.testing.assert_array_equal(before, after)

    def test_a_non_overlapping_later_operation_is_unaffected(self):
        after_ops = self._after_removal()
        before = self._contribution(self.ops, 2)          # op_disjoint: index 2, then 1
        after = self._contribution(after_ops, 1)
        np.testing.assert_array_equal(before, after)

    def test_an_overlapping_later_operation_does_change(self):
        after_ops = self._after_removal()
        before = self._contribution(self.ops, 3)          # op_overlap: index 3, then 2
        after = self._contribution(after_ops, 2)
        self.assertFalse(np.array_equal(before, after),
                         "an overlapping later op's contribution must be free to change")

    def test_deterministic_and_repeatable_across_runs(self):
        def run():
            after_ops = self._after_removal()
            return ([self._contribution(self.ops, i) for i in range(len(self.ops))]
                    + [self._contribution(after_ops, i) for i in range(len(after_ops))])
        first, second = run(), run()
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


# ================================================================================================
# #145 -- bundle the architectural-program gate into a finalize-time check
# ================================================================================================


class TestFinalizeProblems(unittest.TestCase):
    """#145/#7: `finalize_problems` bundles `program_problems`, `commutes`, and
    `is_height_map_representable` -- three checks that existed separately, nothing ran together --
    into one finalize-time report. Unlike #143's `op_problems`-in-`add` gate, it never raises.
    """

    def setUp(self):
        self.fx = ProgramFixture()
        m = np.zeros((RES, RES), bool)
        m[14:24, 6:26] = True
        m &= self.fx.fp
        program = [self.fx.layer(m, 9)]
        self.core_ops = layer_program_to_ops(program, self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def test_a_well_formed_program_produces_an_empty_report(self):
        self.assertEqual(finalize_problems(self.core_ops), [])

    def test_the_empty_program_produces_an_empty_report(self):
        self.assertEqual(finalize_problems([]), [])

    def test_a_syntax_problem_is_named_in_the_report(self):
        bad = [EditOp(kind="layer", mode="subtract")]        # no polygon
        report = finalize_problems(bad)
        self.assertTrue(any("polygon" in p for p in report))

    def test_a_non_commuting_mixed_program_is_named_without_a_height_map_complaint(self):
        """Isolates the commutativity message: `layer`/`ramp` are height-map representable
        regardless of mode, so a mixed add/subtract program built from them only trips this one
        check, not the other."""
        mixed = self.core_ops + [replace(self.core_ops[0], mode="add")]
        report = finalize_problems(mixed)
        self.assertTrue(any("commute" in p for p in report), report)
        self.assertFalse(any("height-map" in p for p in report), report)

    def test_a_non_height_map_representable_program_is_named_without_a_commute_complaint(self):
        """Isolates the height-map message: a subtract-mode volumetric op still commutes with an
        all-subtract core program, so this only trips the other check."""
        volumetric = self.core_ops + [EditOp(kind="box", mode="subtract", size=(0.1, 0.1, 0.1))]
        report = finalize_problems(volumetric)
        self.assertTrue(any("height-map" in p for p in report), report)
        self.assertFalse(any("commute" in p for p in report), report)

    def test_the_gate_never_raises_on_a_severely_malformed_program(self):
        try:
            report = finalize_problems([EditOp(kind="zigzag")])
        except Exception as e:                                # noqa: BLE001
            self.fail(f"finalize_problems raised {e!r} instead of reporting")
        self.assertTrue(report)

    def test_the_gate_is_not_wired_into_add_or_the_preview_path(self):
        """#7's decision: preview/fast-edit stays on #143's per-op gate alone. A program that would
        fail `finalize_problems` (not height-map representable) must still `.add()` cleanly, since
        `add` only runs `op_problems` on the one op being appended."""
        eb = EditableBuilding(None)
        box = EditOp(kind="box", mode="subtract", size=(0.1, 0.1, 0.1))
        eb.add(box)                                            # must not raise
        self.assertEqual(eb.ops, [box])
        self.assertTrue(finalize_problems(eb.ops), "the fixture must genuinely fail the gate")


# ================================================================================================
# #146 -- generalize containment into a footprint-boundary gate
# ================================================================================================


class TestContainmentProblems(unittest.TestCase):
    """#146/#7: containment against the footprint's OWN envelope -- not a GT target (#10/#131's
    guarantee only applies when fitting against a real building). Two rules: stay inside the
    footprint's plan+height bounds, and the exterior must claim the entire boundary at ground
    level. Runs against compiled occupancy, never any one operation's own region.
    """

    def setUp(self):
        self.fx = ProgramFixture()
        self.base = footprint_envelope_sdf(self.fx.fp, self.fx.y0, self.fx.y1, res=RES)

    def _occ(self, ops):
        return EditableBuilding(self.base, ops).to_occupancy(res=RES)

    def test_the_untouched_envelope_passes(self):
        self.assertEqual(containment_problems(self._occ([]), self.fx.fp, self.fx.y0, self.fx.y1),
                         [])

    def test_material_outside_the_footprint_plan_fails(self):
        outside = EditOp(kind="box", mode="add", center=(0.9, 0.0, 0.9), size=(0.05, 0.05, 0.05))
        report = containment_problems(self._occ([outside]), self.fx.fp, self.fx.y0, self.fx.y1)
        self.assertTrue(any("envelope" in p for p in report), report)

    def test_material_above_the_declared_height_range_fails(self):
        step = 2.0 / (RES - 1)
        above = (self.fx.y1 + 5) * step - 1.0                  # 5 voxels above y1, world units
        tall = EditOp(kind="box", mode="add", center=(0.0, above, 0.0), size=(0.3, 0.05, 0.3))
        report = containment_problems(self._occ([tall]), self.fx.fp, self.fx.y0, self.fx.y1)
        self.assertTrue(any("envelope" in p for p in report), report)

    def test_material_below_the_declared_height_range_fails(self):
        step = 2.0 / (RES - 1)
        # 2 voxels below y0 -- still inside the sampled [-1, 1] grid (fixture y0 = 4), unlike a
        # larger offset which would fall off the grid entirely and never register as occupied.
        below = (self.fx.y0 - 2) * step - 1.0
        low = EditOp(kind="box", mode="add", center=(0.0, below, 0.0), size=(0.3, 0.05, 0.3))
        report = containment_problems(self._occ([low]), self.fx.fp, self.fx.y0, self.fx.y1)
        self.assertTrue(any("envelope" in p for p in report), report)

    def test_a_gap_in_the_ground_level_perimeter_fails(self):
        """Carving away a chunk of the footprint's own edge, down through ground level, pulls the
        exterior away from the boundary -- fails even though nothing left the envelope."""
        boundary_chunk = np.zeros((RES, RES), bool)
        boundary_chunk[8:11, 6:10] = True                      # touches the top-left edge
        boundary_chunk &= self.fx.fp
        self.assertTrue(boundary_chunk.any())
        ops = layer_program_to_ops([self.fx.layer(boundary_chunk, 0)], self.fx.fp, self.fx.y0,
                                   self.fx.y1, res=RES)
        report = containment_problems(self._occ(ops), self.fx.fp, self.fx.y0, self.fx.y1)
        self.assertTrue(any("perimeter" in p for p in report), report)

    def test_a_gap_on_the_opposite_edge_of_the_perimeter_also_fails(self):
        """The same check, at the bottom-right edge instead of the top-left -- the underlying rule
        must not be hardcoded to whichever side the first test happened to pick."""
        boundary_chunk = np.zeros((RES, RES), bool)
        boundary_chunk[20:24, 21:26] = True                    # touches the bottom-right edge
        boundary_chunk &= self.fx.fp
        self.assertTrue(boundary_chunk.any())
        ops = layer_program_to_ops([self.fx.layer(boundary_chunk, 0)], self.fx.fp, self.fx.y0,
                                   self.fx.y1, res=RES)
        report = containment_problems(self._occ(ops), self.fx.fp, self.fx.y0, self.fx.y1)
        self.assertTrue(any("perimeter" in p for p in report), report)

    def test_an_interior_void_not_touching_the_boundary_passes(self):
        interior = np.zeros((RES, RES), bool)
        interior[14:18, 14:18] = True                          # well inside, touches no edge
        interior &= self.fx.fp
        self.assertTrue(interior.any())
        ops = layer_program_to_ops([self.fx.layer(interior, 0)], self.fx.fp, self.fx.y0,
                                   self.fx.y1, res=RES)
        report = containment_problems(self._occ(ops), self.fx.fp, self.fx.y0, self.fx.y1)
        self.assertEqual(report, [])

    def test_a_free_standing_interior_element_passes(self):
        """An added interior element, well within the envelope, is exempt from touching the
        boundary -- it only has to obey the outer bound, which it does here."""
        interior_add = EditOp(kind="box", mode="add", center=(0.0, 0.0, 0.0),
                              size=(0.05, 0.05, 0.05))
        report = containment_problems(self._occ([interior_add]), self.fx.fp, self.fx.y0,
                                      self.fx.y1)
        self.assertEqual(report, [])

    def test_the_check_depends_only_on_compiled_occupancy_not_op_history(self):
        """#146's own criterion: the check runs against the FINAL compiled occupancy, not any
        operation's own region. Two different op sequences that compile to the same envelope
        must report identically."""
        empty_report = containment_problems(self._occ([]), self.fx.fp, self.fx.y0, self.fx.y1)
        add_then_subtract_same_spot = [
            EditOp(kind="box", mode="add", center=(0.0, 0.0, 0.0), size=(0.02, 0.02, 0.02)),
            EditOp(kind="box", mode="subtract", center=(0.0, 0.0, 0.0), size=(0.02, 0.02, 0.02)),
        ]
        cancelled_report = containment_problems(self._occ(add_then_subtract_same_spot),
                                                self.fx.fp, self.fx.y0, self.fx.y1)
        self.assertEqual(empty_report, cancelled_report)
        self.assertEqual(empty_report, [])


def _tiny_building(res: int):
    """A minimal `EditableBuilding` with a small square footprint and no ops yet -- the cheapest
    valid prior state `commit_block_program` can be pointed at."""
    fp = np.zeros((res, res), bool)
    fp[2:6, 2:6] = True
    y0, y1 = 0, 3
    return fp, y0, y1, EditableBuilding(footprint_envelope_sdf(fp, y0, y1, res=res))


def _layer_entry(fp: np.ndarray, height: int) -> dict:
    """A recovered-program `Layer` entry over the whole of `fp`, in `recover_massing_programs`'
    own format -- what `BlockProgram.apply` would actually hand back for a trivial flat fit."""
    return dict(op="Layer", height=int(height), area=int(fp.sum()), components=1,
               region=[r.tolist() for r in mask_to_rings(fp)])


class TestCurrentTarget(unittest.TestCase):
    """`_current_target`'s own contract, isolated from `commit_block_program` entirely."""

    def test_a_fully_occupied_column_reports_full_height(self):
        res = 4
        occ = np.zeros((res, res, res), bool)
        occ[1, :, 1] = True
        fp = np.zeros((res, res), bool)
        fp[1, 1] = True
        target = _current_target(occ, fp, 0, res - 1)
        self.assertEqual(int(target[1, 1]), res)

    def test_a_column_with_no_occupancy_at_all_reports_zero_not_full_height(self):
        """The bug review caught: `np.argmax` on an all-`False` slice returns index 0, which the
        raw top-of-column formula reads as the FULL envelope height -- the opposite of empty. A
        `Layer` cut to height 0 (or any edit that empties a column outright) must not come back
        looking like the tallest possible column."""
        res = 4
        occ = np.zeros((res, res, res), bool)          # nothing occupied anywhere
        fp = np.zeros((res, res), bool)
        fp[0, 0] = True
        target = _current_target(occ, fp, 0, res - 1)
        self.assertEqual(int(target[0, 0]), 0)

    def test_off_footprint_columns_are_always_zero_regardless_of_occupancy(self):
        res = 4
        occ = np.zeros((res, res, res), bool)
        occ[2, :, 2] = True                            # occupied, but outside fp below
        fp = np.zeros((res, res), bool)
        target = _current_target(occ, fp, 0, res - 1)
        self.assertEqual(int(target[2, 2]), 0)


def _fake_flat_fit(fp, y0, y1, target, bias=None):
    """Stands in for `recover_massing_programs.fit_program_beam`: whatever footprint it is
    actually handed, it returns one trivial `Layer` program over it -- fast, deterministic, and
    correct for ANY `fp` passed in, so one fake serves any number of footprints without having to
    know in advance which ids `BlockProgram.apply` will ask it to fit.

    Patching `fit_program_beam` itself (the same seam `test_recover_massing_programs.py`'s own
    `TestBlockProgramApply.test_a_missing_footprint_id_prevents_any_fit_at_all` patches) rather
    than `BlockProgram.apply` as a whole means `BlockProgram`'s own real logic -- the missing-id
    check, `to_bias()` -- still runs for real here; only the expensive search underneath is faked.
    """
    return [_layer_entry(fp, 1)], None


class TestCommitBlockProgramGroupId(unittest.TestCase):
    """#151 acceptance criterion 1, isolated from the real fitter's search cost via a mocked
    `fit_program_beam` (see `_fake_flat_fit`)."""

    def setUp(self):
        self.res = 8
        self.fp_a, self.y0_a, self.y1_a, self.building_a = _tiny_building(self.res)
        self.fp_b, self.y0_b, self.y1_b, self.building_b = _tiny_building(self.res)
        self.buildings = {"a": self.building_a, "b": self.building_b}
        self.envelopes = {"a": (self.fp_a, self.y0_a, self.y1_a),
                          "b": (self.fp_b, self.y0_b, self.y1_b)}

    def test_every_op_across_every_footprint_shares_one_group_id(self):
        program = BlockProgram(footprint_ids=("a", "b"), height_rhythm=1)
        with patch.object(recover_massing_programs, "fit_program_beam", _fake_flat_fit):
            reports = commit_block_program(program, self.buildings, self.envelopes, res=self.res)
        self.assertEqual(reports, {"a": [], "b": []})
        all_ops = [op for b in self.buildings.values() for op in b.ops]
        self.assertTrue(all_ops)
        ids = {op.group_id for op in all_ops}
        self.assertEqual(len(ids), 1)
        self.assertIsNotNone(next(iter(ids)))

    def test_a_second_call_gets_a_different_group_id(self):
        program = BlockProgram(footprint_ids=("a", "b"), height_rhythm=1)
        with patch.object(recover_massing_programs, "fit_program_beam", _fake_flat_fit):
            commit_block_program(program, self.buildings, self.envelopes, res=self.res)
            first = self.buildings["a"].ops[0].group_id
            commit_block_program(program, self.buildings, self.envelopes, res=self.res)
            second = self.buildings["a"].ops[0].group_id
        self.assertIsNotNone(first)
        self.assertNotEqual(first, second)

    def test_group_id_does_not_affect_canonical_form(self):
        """#151's own docstring claim on `EditOp.group_id`: stripped by `canonical_form` for the
        same reason `id` is."""
        program = BlockProgram(footprint_ids=("a",), height_rhythm=1)
        with patch.object(recover_massing_programs, "fit_program_beam", _fake_flat_fit):
            commit_block_program(program, self.buildings, self.envelopes, res=self.res)
        tagged = self.buildings["a"].ops
        untagged = [replace(op, group_id=None) for op in tagged]
        self.assertEqual(canonical_form(tagged), canonical_form(untagged))


class TestCommitBlockProgramIndependentCommit(unittest.TestCase):
    """#151 acceptance criteria 2, 3, and 4: one footprint's #7 gate failure is reported against
    that footprint specifically, its prior program is left untouched, and it never blocks the
    others' commit in the same call -- isolated from #7's OWN gate correctness (already #145/#146's
    coverage) via a mocked `finalize_problems`, so this tests only the commit orchestration."""

    def setUp(self):
        self.res = 8
        self.fps, self.buildings, self.envelopes, self.prior_ids = {}, {}, {}, {}
        for fid, height in (("a", 1), ("b", 2), ("c", 3)):
            fp, y0, y1, building = _tiny_building(self.res)
            ops = layer_program_to_ops([_layer_entry(fp, height)], fp, y0, y1, res=self.res)
            building.ops = ops
            self.fps[fid], self.buildings[fid] = fp, building
            self.envelopes[fid] = (fp, y0, y1)
            self.prior_ids[fid] = [op.id for op in ops]

    def test_one_failure_does_not_block_the_others_and_is_reported_by_id(self):
        program = BlockProgram(footprint_ids=("a", "b", "c"), height_rhythm=9)
        with patch.object(recover_massing_programs, "fit_program_beam", _fake_flat_fit), \
             patch("scene.sdf_edit.finalize_problems",
                   side_effect=[[], ["forced failure for b"], []]):
            reports = commit_block_program(program, self.buildings, self.envelopes, res=self.res)

        self.assertEqual(reports["a"], [])
        self.assertEqual(reports["b"], ["forced failure for b"])
        self.assertEqual(reports["c"], [])
        # a and c committed the new program; b kept its exact prior ops (same ids, untouched)
        self.assertNotEqual([op.id for op in self.buildings["a"].ops], self.prior_ids["a"])
        self.assertEqual([op.id for op in self.buildings["b"].ops], self.prior_ids["b"])
        self.assertNotEqual([op.id for op in self.buildings["c"].ops], self.prior_ids["c"])

    def test_the_gate_is_called_once_per_footprint_and_nothing_else_gates(self):
        """#151 acceptance criterion 4: no new block-level validity gate -- #7's own is called
        exactly once per footprint, with no additional gating layer of #151's own."""
        program = BlockProgram(footprint_ids=("a", "b", "c"), height_rhythm=9)
        with patch.object(recover_massing_programs, "fit_program_beam", _fake_flat_fit), \
             patch("scene.sdf_edit.finalize_problems", side_effect=[[], [], []]) as mock_gate:
            commit_block_program(program, self.buildings, self.envelopes, res=self.res)
        self.assertEqual(mock_gate.call_count, 3)


class TestCommitBlockProgramRealFit(unittest.TestCase):
    """One real, unmocked pass through #150's fitter and #7's gate together -- the actual
    pipeline #151 wires, not just its own orchestration logic in isolation.

    ⚠️ Uses `res=64`, not this file's usual `RES=32`: `fit_program_beam` (via `BlockProgram.apply`)
    hardcodes `recover_massing_programs`'s own module-level `RES` internally (e.g. `_ramp_candidates`'
    meshgrid), so a smaller footprint array would shape-mismatch inside it. The envelope (`y1=9`)
    is deliberately taller than either building's initial carved height (6 and 4), so there is real
    surplus for the re-fit to work with -- an envelope equal to the current height would leave
    `_current_target` reporting zero surplus and the bias would have nothing to act on.
    """

    def setUp(self):
        self.res = 64

        def make(row0, col0, size, height):
            fp = np.zeros((self.res, self.res), bool)
            fp[row0:row0 + size, col0:col0 + size] = True
            y0, y1 = 0, 9
            base = footprint_envelope_sdf(fp, y0, y1, res=self.res)
            ops = layer_program_to_ops([_layer_entry(fp, height)], fp, y0, y1, res=self.res)
            return fp, y0, y1, EditableBuilding(base, ops)

        self.fp_a, self.y0_a, self.y1_a, self.building_a = make(4, 4, 6, 6)
        self.fp_b, self.y0_b, self.y1_b, self.building_b = make(40, 40, 6, 4)
        self.buildings = {"a": self.building_a, "b": self.building_b}
        self.envelopes = {"a": (self.fp_a, self.y0_a, self.y1_a),
                          "b": (self.fp_b, self.y0_b, self.y1_b)}

    def test_a_real_coordinated_bias_commits_a_valid_program_to_both(self):
        program = BlockProgram(footprint_ids=("a", "b"), roof_family="flat")
        reports = commit_block_program(program, self.buildings, self.envelopes, res=self.res)

        self.assertEqual(reports, {"a": [], "b": []})
        group_ids = set()
        for fid in ("a", "b"):
            ops = self.buildings[fid].ops
            self.assertTrue(ops)
            self.assertEqual(finalize_problems(ops), [])
            self.assertTrue(all(op.group_id is not None for op in ops))
            group_ids.update(op.group_id for op in ops)
        self.assertEqual(len(group_ids), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
