"""Contract tests for the fixed detail-scale coincidence measurement (ticket 06).

Fast + mostly data-free: exercises the pure scale/threshold math, plus one synthetic-geometry
integration test that builds a tiny known "building" (a big cube labeled wall + a small cube
labeled window, on disk in a tempdir, parsed through the real OBJ/component-label pipeline) so
the end-to-end extraction is verified against known geometry, not just mocked.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/eval/test_measure_scale_spectrum.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/eval
import measure_scale_spectrum as mss  # noqa: E402


def _write_cube(lines, verts_out, lo, hi):
    """Append one axis-aligned box (8 verts, 6 quad faces) to an in-progress OBJ; returns the
    number of triangles it will fan-triangulate to (always 12: 6 quads x 2)."""
    base = len(verts_out)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    box_verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    verts_out.extend(box_verts)
    for v in box_verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    quads = [(1, 2, 3, 4), (5, 6, 7, 8), (1, 4, 8, 5), (2, 3, 7, 6), (1, 2, 6, 5), (4, 3, 7, 8)]
    for q in quads:
        lines.append("f " + " ".join(str(base + i) for i in q))
    return 12


def _write_test_building(root: Path, name: str, boxes):
    """boxes: list of (lo, hi, component_id, label_id). Writes OBJ + component_labels +
    faceindex_componentID under root/{OBJ_MODELS,component_labels,faceindex_componentID}."""
    obj_dir = root / "OBJ_MODELS"
    clbl_dir = root / "component_labels"
    fidx_dir = root / "faceindex_componentID"
    for d in (obj_dir, clbl_dir, fidx_dir):
        d.mkdir(parents=True, exist_ok=True)

    lines, verts = [], []
    fidx = {}
    comp_label = {}
    tri_cursor = 0
    for lo, hi, comp_id, label_id in boxes:
        n_tri = _write_cube(lines, verts, lo, hi)
        fidx[str(tri_cursor)] = {str(tri_cursor + n_tri - 1): comp_id}
        comp_label[str(comp_id)] = label_id
        tri_cursor += n_tri

    (obj_dir / f"{name}.obj").write_text("\n".join(lines) + "\n")
    json.dump(comp_label, open(clbl_dir / f"{name}_label.json", "w"))
    json.dump(fidx, open(fidx_dir / f"{name}.json", "w"))
    return obj_dir, clbl_dir, fidx_dir


class SStarNormalizedTest(unittest.TestCase):
    def test_matches_adr0004_worked_value(self):
        # ADR 0004: s* = 5 voxels @96^3 -> as a fraction of a building's own max AABB extent,
        # independent of any assumed real-world meters (BuildingNet meshes carry none).
        self.assertAlmostEqual(mss.s_star_normalized(res=96, voxels=5), 5 / 95)

    def test_scales_with_voxel_count_and_resolution(self):
        self.assertAlmostEqual(mss.s_star_normalized(res=64, voxels=3), 3 / 63)
        self.assertGreater(mss.s_star_normalized(res=64), mss.s_star_normalized(res=128))


class InstanceCharScaleTest(unittest.TestCase):
    def test_median_of_three_axes(self):
        self.assertAlmostEqual(mss.instance_char_scale(np.array([10.0, 1.0, 0.1])), 1.0)

    def test_uniform_box(self):
        self.assertAlmostEqual(mss.instance_char_scale(np.array([0.5, 0.5, 0.5])), 0.5)

    def test_uses_absolute_value(self):
        self.assertAlmostEqual(mss.instance_char_scale(np.array([-1.0, 2.0, 3.0])), 2.0)


class ClassifyTest(unittest.TestCase):
    def test_above_threshold(self):
        self.assertEqual(mss.classify(0.1, threshold=0.05), "above_s*")

    def test_below_threshold(self):
        self.assertEqual(mss.classify(0.01, threshold=0.05), "below_s*")

    def test_boundary_counts_as_above(self):
        self.assertEqual(mss.classify(0.05, threshold=0.05), "above_s*")


class AggregateTest(unittest.TestCase):
    def test_known_distribution(self):
        agg = mss.aggregate([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(agg["n"], 5)
        self.assertAlmostEqual(agg["median"], 3.0)
        self.assertAlmostEqual(agg["mean"], 3.0)
        self.assertAlmostEqual(agg["min"], 1.0)
        self.assertAlmostEqual(agg["max"], 5.0)


class ExtractInstancesSyntheticGeometryTest(unittest.TestCase):
    """A big cube labeled wall(1) and a small, disjoint cube labeled window(2), parsed through
    the real OBJ/component-label/faceindex pipeline -- verifies the geometry extraction end to
    end against KNOWN scale, not a mock."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        # A big "wall" (extent 10) far from a small "window" (extent 0.3), so the combined
        # building AABB (max extent ~20.3) makes wall land clearly above s* and window clearly
        # below it (s*_normalized @96^3 ~= 0.0526).
        boxes = [
            ((0, 0, 0), (10, 10, 10), 0, 1),          # wall, label 1
            ((20, 0, 0), (20.3, 0.3, 0.3), 1, 2),     # window, label 2
        ]
        self.obj_dir, self.clbl_dir, self.fidx_dir = _write_test_building(root, "TESTbldg", boxes)

    def tearDown(self):
        self.tmp.cleanup()

    def test_wall_and_window_classify_on_opposite_sides_of_s_star(self):
        instances = list(mss.extract_instances_for_building(
            "TESTbldg", obj_dir=self.obj_dir, clbl_dir=self.clbl_dir, fidx_dir=self.fidx_dir))
        by_label = {}
        for lab, scale in instances:
            by_label.setdefault(lab, []).append(scale)

        self.assertIn(1, by_label)
        self.assertIn(2, by_label)
        wall_scale = by_label[1][0]
        window_scale = by_label[2][0]
        thr = mss.s_star_normalized()
        self.assertGreaterEqual(wall_scale, thr)
        self.assertLess(window_scale, thr)
        # Sanity on the actual numbers (bmax = 20.3, wall extent = 10, window extent = 0.3).
        self.assertAlmostEqual(wall_scale, 10 / 20.3, places=4)
        self.assertAlmostEqual(window_scale, 0.3 / 20.3, places=4)

    def test_disjoint_instances_are_not_merged(self):
        instances = list(mss.extract_instances_for_building(
            "TESTbldg", obj_dir=self.obj_dir, clbl_dir=self.clbl_dir, fidx_dir=self.fidx_dir))
        # Two distinct labels, one instance each -- merge_instances must not fuse the wall and
        # the window (they are far apart) or drop either.
        self.assertEqual(len(instances), 2)

    def test_missing_building_yields_nothing(self):
        instances = list(mss.extract_instances_for_building(
            "NOPE", obj_dir=self.obj_dir, clbl_dir=self.clbl_dir, fidx_dir=self.fidx_dir))
        self.assertEqual(instances, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
