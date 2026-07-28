"""Contract tests for the surface sampler (seam 2, spec #68).

Pure CPU, no model and no GPU -- same shape as the shared meshing helper's tests. Asserts external
behaviour (where points land, where normals point, whether edges are favoured), never internals.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scene/test_surface_sampling.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scene.surface_sampling import (  # noqa: E402
    ensure_outward, sample_sharp, sample_streams, sample_uniform,
)


def _box(extent=(1.0, 0.6, 0.8)):
    import trimesh
    return trimesh.creation.box(extents=extent)


def _sphere():
    import trimesh
    return trimesh.creation.icosphere(subdivisions=3, radius=0.7)


class TestOutwardNormals(unittest.TestCase):
    """The guard that a reflection upstream cannot silently invert the encoder's input."""

    def test_inverted_mesh_is_repaired(self):
        m = _box()
        m.invert()
        self.assertLess(m.volume, 0, "precondition: the test mesh is inside-out")
        self.assertGreater(ensure_outward(m).volume, 0)

    def test_sampled_normals_point_away_from_centre(self):
        for mesh in (_box(), _sphere()):
            for inverted in (False, True):
                m = mesh.copy()
                if inverted:
                    m.invert()
                s = sample_uniform(m, 2000, np.random.default_rng(0))
                pts, nrm = s[:, :3], s[:, 3:]
                # on a convex solid centred at the origin, outward means pointing away from it
                self.assertGreater(float((pts * nrm).sum(1).min()), 0.0,
                                   f"inward normal survived (inverted={inverted})")

    def test_normals_are_unit_length(self):
        s = sample_uniform(_box(), 500, np.random.default_rng(0))
        np.testing.assert_allclose(np.linalg.norm(s[:, 3:], axis=1), 1.0, atol=1e-5)


class TestPointsLieOnSurface(unittest.TestCase):
    """Distance to the mesh, via igl (already a repo dependency; trimesh's proximity needs rtree)."""

    @staticmethod
    def _dist_to_mesh(mesh, pts):
        import igl
        fwn = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        d = igl.signed_distance(np.ascontiguousarray(pts, np.float64),
                                np.ascontiguousarray(mesh.vertices, np.float64),
                                np.ascontiguousarray(mesh.faces, np.int32), fwn)[0]
        return np.abs(np.asarray(d))

    def test_uniform_points_are_on_the_mesh(self):
        m = _box()
        s = sample_uniform(m, 1000, np.random.default_rng(0))
        self.assertLess(float(self._dist_to_mesh(m, s[:, :3]).max()), 1e-5)

    def test_sharp_points_are_on_the_mesh(self):
        m = _box()
        s = sample_sharp(m, 1000, np.random.default_rng(0))
        self.assertLess(float(self._dist_to_mesh(m, s[:, :3]).max()), 1e-5)


class TestSharpStreamFavoursEdges(unittest.TestCase):
    """The reason the sharp stream exists: it must actually concentrate on edges."""

    def test_sharp_points_are_nearer_edges_than_uniform_points(self):
        m = _box()
        rng = np.random.default_rng(0)
        sharp = sample_sharp(m, 2000, rng)[:, :3]
        unif = sample_uniform(m, 2000, rng)[:, :3]

        # distance to the nearest box edge = distance to the nearest of the 12 segments; for an
        # axis-aligned box, a point on a face is at least (half-extent - |coord|) from its edges.
        half = np.asarray(m.extents) / 2.0

        def edge_dist(p):
            # two smallest gaps to a face plane; on an edge both are ~0
            gaps = np.sort(half - np.abs(p), axis=1)
            return gaps[:, 1]

        self.assertLess(float(edge_dist(sharp).mean()), float(edge_dist(unif).mean()) / 3.0)


class TestDegenerateInput(unittest.TestCase):
    def test_smooth_mesh_falls_back_instead_of_raising(self):
        # an icosphere has no edge above the threshold -> must still return a usable stream
        s = sample_sharp(_sphere(), 256, np.random.default_rng(0), deg=25.0)
        self.assertEqual(s.shape, (256, 6))
        self.assertTrue(np.isfinite(s).all())

    def test_streams_have_requested_sizes_and_are_finite(self):
        c, s = sample_streams(_box(), n_coarse=128, n_sharp=64, rng=np.random.default_rng(0))
        self.assertEqual(c.shape, (128, 6))
        self.assertEqual(s.shape, (64, 6))
        self.assertTrue(np.isfinite(c).all() and np.isfinite(s).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
