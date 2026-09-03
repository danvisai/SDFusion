"""Contract tests for the layer-program bridge into the SDF edit stack (#128).

The recovered massing programs of #10 compile to a 64^3 voxel grid through their own deterministic
compiler.  These tests pin the *other* path -- the same program expressed as `EditOp`s and composed
through `scene/sdf_primitives.py` -- and assert the two agree exactly, because a program that only
exists as voxels is geometry evidence, not a recipe.

Pure CPU, no model, no GPU, no corpus.  Same shape as `scene/test_surface_sampling.py`.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scene/test_sdf_edit.py
"""
from __future__ import annotations

import itertools
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import (  # noqa: E402
    ALGEBRA, ARCHITECTURAL_VOCABULARY, CORE, PALETTE, PROGRAM_KINDS, VOLUMETRIC,
    EditableBuilding, EditOp, canonical_form, commutes, equivalent, footprint_envelope_sdf,
    is_height_map_representable, layer_program_to_ops,
    mask_components_rings, mask_to_rings, op_problems, program_problems, snap_to_grid,
)
from scene.sdf_primitives import sdf_plane_halfspace  # noqa: E402

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
