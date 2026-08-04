"""Contract tests for the #71 evaluation harness. Synthetic, fast, no GPU.

The point of the harness is that its numbers are *comparable*, so what is pinned here is exactly the
two things that decide comparability: the missing/extra decomposition means what the map says it
means, and the id set is reproducible.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_eval_massing_arms.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.eval_massing_arms import (  # noqa: E402
    blockout_sdf, pick_ids, summarise, volume_split,
)


def _box(lo, hi, res=16):
    o = np.zeros((res, res, res), bool)
    o[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return o


class TestVolumeSplit(unittest.TestCase):
    """Criterion 3: one aggregate cannot separate 'carved the over-fill' from 'ate the building'."""

    def test_identity_is_perfect(self):
        g = _box((2, 2, 2), (10, 10, 10))
        s = volume_split(g, g)
        self.assertEqual((s["missing"], s["extra"]), (0.0, 0.0))
        self.assertEqual(s["vol_iou"], 1.0)

    def test_over_fill_is_extra_only(self):
        """The blockout case: contains all of GT plus surplus -> 0 missing, extra > 0."""
        g = _box((2, 2, 2), (10, 10, 10))          # 512 voxels
        a = _box((2, 2, 2), (10, 10, 12))          # 640 -> +25% of GT
        s = volume_split(a, g)
        self.assertEqual(s["missing"], 0.0)
        self.assertAlmostEqual(s["extra"], 0.25)

    def test_erosion_is_missing_only(self):
        """The opposite failure: a subset of GT -> 0 extra, missing > 0. Wants the opposite response."""
        g = _box((2, 2, 2), (10, 10, 10))
        a = _box((2, 2, 2), (10, 10, 8))           # 384 -> lost 25% of GT
        s = volume_split(a, g)
        self.assertEqual(s["extra"], 0.0)
        self.assertAlmostEqual(s["missing"], 0.25)

    def test_over_fill_and_erosion_can_share_an_iou(self):
        """The reason the split exists: these two land on the same aggregate and must not be tied."""
        g = _box((0, 0, 0), (8, 8, 8))             # 512 voxels
        over = _box((0, 0, 0), (8, 8, 16))         # superset, doubled  -> IoU 512/1024
        under = _box((0, 0, 0), (8, 8, 4))         # subset,   halved   -> IoU 256/512
        so, su = volume_split(over, g), volume_split(under, g)
        self.assertAlmostEqual(so["vol_iou"], 0.5)
        self.assertAlmostEqual(su["vol_iou"], 0.5)                    # indistinguishable aggregate
        self.assertEqual((so["missing"], so["extra"]), (0.0, 1.0))    # separable once split:
        self.assertEqual((su["missing"], su["extra"]), (0.5, 0.0))    # over-fill vs erosion

    def test_empty_arm_and_empty_gt_do_not_divide_by_zero(self):
        g = _box((2, 2, 2), (6, 6, 6))
        self.assertEqual(volume_split(np.zeros_like(g), g)["missing"], 1.0)
        self.assertEqual(volume_split(g, np.zeros_like(g))["vol_iou"], 0.0)


class TestBlockout(unittest.TestCase):
    """The 'did this beat doing nothing?' arm: a signed field, not a binary mask (#43)."""

    def test_extrusion_is_a_signed_field_matching_the_footprint(self):
        fp = np.zeros((64, 64), np.uint8)
        fp[20:40, 25:45] = 1
        bo = blockout_sdf(fp, 10, 30)
        self.assertIsNotNone(bo)
        occ = bo <= 0
        self.assertTrue((bo > 0).any() and (bo < 0).any())          # a real zero crossing to mesh at 0.0
        np.testing.assert_array_equal(occ.any(axis=1), fp.astype(bool))
        self.assertFalse(occ[:, :9, :].any() or occ[:, 32:, :].any())  # confined to the given slab

    def test_empty_footprint_returns_none_rather_than_crashing(self):
        self.assertIsNone(blockout_sdf(np.zeros((64, 64), np.uint8), 0, 10))


class TestSummarise(unittest.TestCase):
    def test_roughness_keeps_its_guard_prefix(self):
        """It must stay impossible to read the guard as one of the ranked criteria."""
        s = summarise([dict(fp_iou=0.9, missing=0.1, extra=0.2, vol_iou=0.8, guard_roughness=0.005)])
        self.assertIn("guard_roughness", s)
        self.assertNotIn("surface_roughness", s)
        self.assertNotIn("roughness", set(s) - {"guard_roughness"})

    def test_empty_arm_summarises_to_nothing(self):
        self.assertEqual(summarise([]), {})


class TestIdSet(unittest.TestCase):
    """A fixed id set is what makes two runs comparable; ids are global rows of real.h5."""

    def _fake_cache(self, tmp, rows, held):
        import h5py
        p = Path(tmp) / "latents.h5"
        with h5py.File(p, "w") as f:
            f["row"] = np.asarray(rows, np.int32)
            f["held_out"] = np.asarray(held, np.uint8)
        return p

    def test_default_ids_are_the_held_out_rows_ascending(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._fake_cache(tmp, [7, 3, 9, 1], [1, 1, 0, 1])
            ids, lat_of = pick_ids(p, None)
            self.assertEqual(ids, [1, 3, 7])            # deterministic, and independent of cache order
            self.assertEqual(lat_of[7], 0)              # row -> its index in the cache, not its rank

    def test_ids_from_replays_a_previous_run_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._fake_cache(tmp, [7, 3, 9, 1], [1, 1, 0, 1])
            prev = Path(tmp) / "prev.json"
            prev.write_text(json.dumps({"ids": [7, 1]}))
            ids, _ = pick_ids(p, str(prev))
            self.assertEqual(ids, [7, 1])               # order preserved, not re-sorted

    def test_pinned_id_absent_from_the_cache_is_refused_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._fake_cache(tmp, [7, 3], [1, 1])
            prev = Path(tmp) / "prev.json"
            prev.write_text(json.dumps({"ids": [7, 999]}))
            with self.assertRaises(SystemExit):
                pick_ids(p, str(prev))


if __name__ == "__main__":
    unittest.main(verbosity=2)
