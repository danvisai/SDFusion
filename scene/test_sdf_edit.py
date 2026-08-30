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
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import (  # noqa: E402
    ALGEBRA, ARCHITECTURAL_VOCABULARY, CORE, PALETTE, PROGRAM_KINDS, VOLUMETRIC,
    EditableBuilding, EditOp, canonical_form, commutes, equivalent, footprint_envelope_sdf,
    is_height_map_representable, layer_program_to_ops,
    mask_components_rings, mask_to_rings, op_problems, program_problems,
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
    never land on an edge and the point-in-polygon test can never tie. Turning these into few-vertex
    polygons under a budget is a separate, unstarted question (#4).
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


class TestEditOpRoundTrip(unittest.TestCase):
    """The new kinds must survive `to_dict` -> JSON -> `from_dict`, or they are not recipe state."""

    def _round_trip(self, op):
        back = EditOp.from_dict(json.loads(json.dumps(op.to_dict())))
        self.assertEqual(back.kind, op.kind)
        self.assertEqual(back.mode, op.mode)
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

    def test_unknown_kind_is_refused(self):
        eb = EditableBuilding(footprint_envelope_sdf(np.ones((4, 4), bool), 0, 1, res=4))
        eb.add(EditOp(kind="zigzag"))
        with self.assertRaises(ValueError):
            eb.composed()

    def test_old_state_without_the_new_fields_still_loads(self):
        legacy = {"kind": "box", "center": [0, 0, 0], "size": [1, 1, 1], "mode": "add",
                  "smooth": 0.0, "rot_y": 0.0, "round_r": 0.0, "lib_id": -1, "layer": "annotation"}
        op = EditOp.from_dict(legacy)
        self.assertEqual(op.kind, "box")
        self.assertIsNone(op.polygon)
        self.assertIsNone(op.planes)


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

    def test_every_palette_kind_is_in_the_ontology(self):
        """A kind the compiler accepts but the algebra does not describe is a hole in the spec."""
        for kind in PALETTE:
            self.assertIn(kind, ALGEBRA, f"{kind!r} is compilable but undeclared")

    def test_the_three_recovered_operations_are_the_core(self):
        for kind in PROGRAM_KINDS:
            spec = ALGEBRA[kind]
            self.assertEqual(spec.tier, CORE)
            self.assertTrue(spec.height_map, f"{kind} must be height-map representable")
            self.assertTrue(spec.subtractive_only, "#10 measured missing=0 on 714/714")

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

    def test_a_core_operation_may_not_be_additive(self):
        """The core is subtract-only by measurement, so an additive `layer` is a spec violation and
        not merely unusual -- it would also break commutativity, which everything below relies on."""
        self.assertTrue(any("subtract" in p for p in op_problems(self._layer(mode="add"))))

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
    regions overlap, and permuting the operations changed the compiled building on **68.8%** of
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
