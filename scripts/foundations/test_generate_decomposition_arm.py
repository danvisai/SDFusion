"""Contract tests for the C2 decomposition-arm generator (ticket 12).

Fast + data-free: exercises the pure seams -- which detail types are retrieval-eligible, the
box/sphere-agnostic half-extent used for retrieval scoring, and the retrieval-geometry math
(aspect/y_frac/box_rel_y) -- without touching the GPU model, the element library, or the
planner checkpoint. The massing/detail/compose pipeline itself is verified separately by an
integration run (see the ticket answer), matching this project's established convention for
GPU-dependent code (tickets 05/09/10).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_generate_decomposition_arm.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import generate_decomposition_arm as gda  # noqa: E402


class PoolsForTypeTest(unittest.TestCase):
    def test_tower_pools_over_tower_only(self):
        self.assertEqual(gda.pools_for_type("tower"), ("tower",))

    def test_dome_pools_over_dome_only(self):
        self.assertEqual(gda.pools_for_type("dome"), ("dome",))

    def test_chimney_pools_over_chimney_and_roof_structure(self):
        self.assertEqual(gda.pools_for_type("chimney"), ("chimney", "roof_structure"))

    def test_balcony_pools_over_balcony_only(self):
        self.assertEqual(gda.pools_for_type("balcony"), ("balcony",))

    def test_column_pools_over_column_only(self):
        self.assertEqual(gda.pools_for_type("column"), ("column",))

    def test_types_propose_detail_ops_never_emits_are_not_retrieval_eligible(self):
        # window/door: always procedural by design. roof/stairs/balcony_upper: gained usable
        # library pools (2026-07-14) but propose_detail_ops itself never emits an op for them
        # ("massing already has a roof") -- nothing to ever upgrade, so not in RETRIEVAL_POOLS.
        for det in ("window", "door", "roof", "stairs", "balcony_upper"):
            self.assertIsNone(gda.pools_for_type(det))


class OpHalfExtentTest(unittest.TestCase):
    def test_box_op_returns_its_own_size(self):
        op = dict(kind="box", size=[0.1, 0.2, 0.3])
        self.assertEqual(gda.op_half_extent(op), [0.1, 0.2, 0.3])

    def test_sphere_op_returns_a_symmetric_bounding_box(self):
        op = dict(kind="sphere", size=[0.25])
        self.assertEqual(gda.op_half_extent(op), [0.25, 0.25, 0.25])

    def test_cylinder_op_returns_radius_half_height_radius(self):
        # propose_detail_ops emits "column" as kind="cylinder", size=[radius, height] -- a
        # 2-element size, not a 3-element box half-extent (the bug this test guards against:
        # indexing size[2] on a cylinder op raises IndexError without this branch).
        op = dict(kind="cylinder", size=[0.04, 0.3])
        self.assertEqual(gda.op_half_extent(op), [0.04, 0.15, 0.04])


class YExtentFromOccupancyTest(unittest.TestCase):
    def test_matches_the_occupied_y_range_in_the_grid_linspace(self):
        # Cubic, matching every real massing grid (64^3) and `_occ_frame`'s own R=occ.shape[0]
        # convention -- a non-cubic array isn't a shape this function is meant to support.
        occ = np.zeros((9, 9, 9), dtype=bool)
        occ[2, 2, 2] = True   # low corner
        occ[2, 6, 2] = True   # high corner
        y_ground, y_top = gda.y_extent_from_occupancy(occ)
        g = np.linspace(-1, 1, 9)
        self.assertAlmostEqual(y_ground, g[2])
        self.assertAlmostEqual(y_top, g[6])

    def test_empty_occupancy_raises(self):
        occ = np.zeros((4, 4, 4), dtype=bool)
        with self.assertRaises(ValueError):
            gda.y_extent_from_occupancy(occ)


class RetrievalParamsTest(unittest.TestCase):
    def test_computes_aspect_yfrac_and_box_rel_y(self):
        op = dict(kind="box", center=[0.0, 0.5, 0.0], size=[0.1, 0.2, 0.05])
        params = gda.retrieval_params(op, y_ground=-1.0, y_top=1.0)
        self.assertAlmostEqual(params["aspect"][0], 0.1 / 0.2)
        self.assertAlmostEqual(params["aspect"][1], 0.05 / 0.2)
        self.assertAlmostEqual(params["y_frac"], 1.5 / 2.0)   # (0.5 - (-1.0)) / 2.0
        self.assertAlmostEqual(params["box_rel_y"], 0.4 / 2.0)  # 2*0.2 / 2.0

    def test_sphere_op_uses_symmetric_aspect(self):
        op = dict(kind="sphere", center=[0.0, 0.0, 0.0], size=[0.3])
        params = gda.retrieval_params(op, y_ground=-1.0, y_top=1.0)
        self.assertAlmostEqual(params["aspect"][0], 1.0)
        self.assertAlmostEqual(params["aspect"][1], 1.0)

    def test_cylinder_op_computes_aspect_from_radius_and_half_height(self):
        op = dict(kind="cylinder", center=[0.0, 0.0, 0.0], size=[0.04, 0.3])
        params = gda.retrieval_params(op, y_ground=-1.0, y_top=1.0)
        self.assertAlmostEqual(params["aspect"][0], 0.04 / 0.15)
        self.assertAlmostEqual(params["aspect"][1], 0.04 / 0.15)

    def test_degenerate_y_span_returns_none(self):
        op = dict(kind="box", center=[0.0, 0.0, 0.0], size=[0.1, 0.1, 0.1])
        self.assertIsNone(gda.retrieval_params(op, y_ground=0.3, y_top=0.3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
