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
    ensure_outward, sample_sharp, sample_streams, sample_uniform, to_array_frame,
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


class TestArrayFrameConversion(unittest.TestCase):
    """The Frame-N -> array-frame swap that cost both A2 runs (#70)."""

    def test_x_and_z_are_exchanged(self):
        v = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        f = np.array([[0, 1, 0]])
        av, _ = to_array_frame(v, f)
        np.testing.assert_allclose(av, [[3.0, 2.0, 1.0], [-6.0, 5.0, -4.0]])

    def test_winding_is_flipped_because_the_swap_is_a_reflection(self):
        # a determinant--1 map reverses orientation; if faces were not reversed too, every normal
        # would point inward and the encoder would silently eat an inside-out surface
        v = np.zeros((3, 3))
        _, af = to_array_frame(v, np.array([[0, 1, 2]]))
        np.testing.assert_array_equal(af, [[2, 1, 0]])

    def test_volume_stays_positive_for_a_solid(self):
        import trimesh
        m = _box()
        av, af = to_array_frame(m.vertices, m.faces)
        got = trimesh.Trimesh(av, af, process=False)
        self.assertGreater(got.volume, 0.0,
                           "swap+reflip must leave the solid outward-wound")
        self.assertAlmostEqual(got.volume, m.volume, places=9)

    def test_applying_it_twice_is_the_identity(self):
        m = _box()
        v1, f1 = to_array_frame(m.vertices, m.faces)
        v2, f2 = to_array_frame(v1, f1)
        np.testing.assert_allclose(v2, np.asarray(m.vertices, np.float64))
        np.testing.assert_array_equal(f2, np.asarray(m.faces))

    def test_output_is_contiguous_so_downstream_encoders_accept_it(self):
        m = _box()
        av, af = to_array_frame(m.vertices, m.faces)
        self.assertTrue(av.flags["C_CONTIGUOUS"])
        self.assertTrue(af.flags["C_CONTIGUOUS"])


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


class TestUniformStreamHonoursItsRng(unittest.TestCase):
    """The coverage the #88 bug did not have: `rng` was accepted and ignored for a year.

    The consequence was not a flaky test, it was a corpus -- the coarse stream is the bulk of what the
    encoder reads, so every latent cached before the fix is a function of numpy's *global* state at
    write time and cannot be reproduced from its own row.
    """

    def test_same_seed_gives_the_same_points(self):
        a = sample_uniform(_box(), 256, np.random.default_rng(0))
        b = sample_uniform(_box(), 256, np.random.default_rng(0))
        np.testing.assert_allclose(a, b)

    def test_different_seeds_give_different_points(self):
        a = sample_uniform(_box(), 256, np.random.default_rng(0))
        b = sample_uniform(_box(), 256, np.random.default_rng(1))
        self.assertFalse(np.allclose(a, b), "the rng is being ignored again")

    def test_a_shared_generator_advances_between_calls(self):
        r = np.random.default_rng(0)
        self.assertFalse(np.allclose(sample_uniform(_box(), 256, r), sample_uniform(_box(), 256, r)))

    def test_global_numpy_state_cannot_perturb_the_draw(self):
        np.random.seed(1234)
        a = sample_uniform(_box(), 256, np.random.default_rng(0))
        np.random.seed(4321)
        _ = np.random.random(97)
        b = sample_uniform(_box(), 256, np.random.default_rng(0))
        np.testing.assert_allclose(a, b)

    def test_streams_are_reproducible_end_to_end(self):
        c1, s1 = sample_streams(_box(), 128, 64, np.random.default_rng(7))
        c2, s2 = sample_streams(_box(), 128, 64, np.random.default_rng(7))
        np.testing.assert_allclose(c1, c2)
        np.testing.assert_allclose(s1, s2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
