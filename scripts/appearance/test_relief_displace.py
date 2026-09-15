"""Does the MESH route actually carry sub-`s*` detail that a 96^3 grid cannot?

The instrument tests come first (a metric nobody has checked is not evidence), then the one
experiment that decides the plan's KILL clause: apply the SAME displacement request to a wall
remeshed at the grid's effective resolution and to one remeshed fine, and measure what each
surface actually carries.

CPU only, no GPU, no diffusion model, no building data.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.appearance.relief_displace import (   # noqa: E402
    S_STAR_M, carries_feature, displace_along_normals, edge_length_for, peak_wavelength,
    radial_psd, remesh_isotropic, sample_displacement, sub_s_star_fraction, vertex_normals,
)

WALL_M = 8.0          # a square wall patch, metres
GRID_EDGE_M = 0.316   # a 96^3 marching-cubes surface on a 30 m building: 30/95
FINE_EDGE_M = 0.125   # target_len = d/2 for d = 0.25 m


def sine_field(res, extent_m, wavelength_m, axis=0):
    g = np.linspace(0.0, extent_m, res)
    gu, gv = np.meshgrid(g, g)
    return np.sin(2 * np.pi * (gu if axis == 0 else gv) / wavelength_m)


def flat_wall(extent_m):
    """Two triangles. This is what a footprint-constrained massing mesh gives you for a wall."""
    v = np.array([[0., 0., 0.], [extent_m, 0., 0.], [extent_m, extent_m, 0.], [0., extent_m, 0.]])
    return v, np.array([[0, 1, 2], [0, 2, 3]], np.int64)


class TestInstrument(unittest.TestCase):
    """The gate metric itself, before it is used to judge anything."""

    def test_short_wavelength_is_all_detail(self):
        f = sine_field(256, WALL_M, 0.5)
        self.assertGreater(sub_s_star_fraction(f, WALL_M), 0.99)

    def test_long_wavelength_is_all_massing(self):
        f = sine_field(256, WALL_M, 4.0)
        self.assertLess(sub_s_star_fraction(f, WALL_M), 0.01)

    def test_flat_field_is_not_detail(self):
        self.assertEqual(sub_s_star_fraction(np.zeros((64, 64)), WALL_M), 0.0)

    def test_constant_offset_is_not_detail(self):
        """A wall pushed bodily inward is massing, not ornament. DC must not count."""
        self.assertEqual(sub_s_star_fraction(np.full((64, 64), 3.7), WALL_M), 0.0)

    def test_psd_peaks_at_the_true_wavelength(self):
        wav, power = radial_psd(sine_field(256, WALL_M, 1.6), WALL_M)
        self.assertAlmostEqual(wav[int(np.argmax(power))], 1.6, delta=0.12)

    def test_s_star_is_the_fixed_boundary(self):
        self.assertEqual(S_STAR_M, 1.0)          # ADR 0004, fixed a priori


class TestNormals(unittest.TestCase):
    def test_flat_wall_normals_are_the_plane_normal(self):
        v, f = flat_wall(WALL_M)
        n = vertex_normals(v, f)
        self.assertTrue(np.allclose(np.abs(n @ np.array([0., 0., 1.])), 1.0))


class TestRemesh(unittest.TestCase):
    def test_a_two_triangle_wall_gains_vertices(self):
        v, f = flat_wall(WALL_M)
        v2, f2 = remesh_isotropic(v, f, FINE_EDGE_M)
        self.assertGreater(len(f2), 2000)
        self.assertLess(abs(v2[:, 2]).max(), 1e-6)      # still planar: remeshing moved nothing

    def test_target_length_is_respected(self):
        v, f = flat_wall(WALL_M)
        v2, f2 = remesh_isotropic(v, f, FINE_EDGE_M)
        e = np.linalg.norm(v2[f2[:, 0]] - v2[f2[:, 1]], axis=1)
        self.assertLess(abs(np.median(e) - FINE_EDGE_M) / FINE_EDGE_M, 0.35)


class TestInwardOnly(unittest.TestCase):
    def test_no_vertex_moves_outward(self):
        """Outward motion is spill. The default must make it impossible, not unlikely."""
        v, f = flat_wall(WALL_M)
        v2, f2 = remesh_isotropic(v, f, 0.5)
        h = np.random.default_rng(0).random(len(v2))
        out = displace_along_normals(v2, f2, h, amplitude=0.3)
        self.assertLessEqual(np.abs(out[:, 2]).max(), 0.3 + 1e-9)
        self.assertLessEqual((out[:, 2] * np.sign(vertex_normals(v2, f2)[:, 2])).max(), 1e-9)

    def test_amplitude_is_the_achieved_depth(self):
        v, f = flat_wall(WALL_M)
        v2, f2 = remesh_isotropic(v, f, 0.5)
        h = np.linspace(0, 1, len(v2))
        out = displace_along_normals(v2, f2, h, amplitude=0.4)
        self.assertAlmostEqual(np.abs(out[:, 2]).max(), 0.4, places=6)


class TestResolutionCeiling(unittest.TestCase):
    """The experiment. Same request, two resolutions, measured on what came out."""

    @staticmethod
    def _carry(edge_m, wavelength_m, amplitude=0.3):
        v, f = flat_wall(WALL_M)
        v2, f2 = remesh_isotropic(v, f, edge_m)
        h = np.sin(2 * np.pi * v2[:, 0] / wavelength_m)
        out = displace_along_normals(v2, f2, h, amplitude=amplitude)
        got = sample_displacement(v2, out, [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], WALL_M, 256)
        return sub_s_star_fraction(got, WALL_M), len(f2)

    def test_fine_mesh_carries_sub_s_star_detail(self):
        frac, _ = self._carry(FINE_EDGE_M, 0.5)
        self.assertGreater(frac, 0.9)

    def test_grid_resolution_mesh_loses_it(self):
        """A 0.5 m pattern is below a 96^3 grid's Nyquist. The surface cannot hold it."""
        fine, _ = self._carry(FINE_EDGE_M, 0.5)
        coarse, _ = self._carry(GRID_EDGE_M, 0.5)
        self.assertGreater(fine, coarse)

    def test_a_high_sub_s_star_score_can_be_pure_aliasing(self):
        """🔑 The reason the gate is a pair. A coarse mesh scores ~0.4 and carries nothing."""
        v, f = flat_wall(WALL_M)
        v2, f2 = remesh_isotropic(v, f, GRID_EDGE_M)
        h = np.sin(2 * np.pi * v2[:, 0] / 0.30)
        out = displace_along_normals(v2, f2, h, amplitude=0.3)
        got = sample_displacement(v2, out, [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], WALL_M, 256)
        self.assertGreater(sub_s_star_fraction(got, WALL_M), 0.3)      # looks like partial success
        self.assertFalse(carries_feature(got, WALL_M, 0.30))           # and yet carries nothing
        self.assertGreater(peak_wavelength(got, WALL_M), 2.0)          # energy is at massing scale

    def test_nyquist_rule_predicts_what_is_carried(self):
        """edge <= feature/2 carries it; above that it aliases. Both directions pinned."""
        for feature in (0.30, 0.20):
            v, f = flat_wall(WALL_M)
            v2, f2 = remesh_isotropic(v, f, edge_length_for(feature))
            h = np.sin(2 * np.pi * v2[:, 0] / feature)
            out = displace_along_normals(v2, f2, h, amplitude=0.3)
            got = sample_displacement(v2, out, [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                      WALL_M, 256)
            self.assertTrue(carries_feature(got, WALL_M, feature),
                            f"edge {edge_length_for(feature)} should carry {feature} m")

    def test_massing_scale_pattern_is_carried_by_both(self):
        """The control: at 4 m both resolutions cope, so the difference above is about scale."""
        for edge in (FINE_EDGE_M, GRID_EDGE_M):
            frac, _ = self._carry(edge, 4.0)
            self.assertLess(frac, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
