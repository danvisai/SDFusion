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
    C2_ALLOWANCE, COLLAPSE_MISSING, S_STAR_VOXELS, blockout_sdf, footprint_split, pick_ids,
    reference_win_rate, summarise, volume_split, vs_input,
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
    @staticmethod
    def _row(missing, vol_iou=0.8):
        return dict(fp_iou=0.9, missing=missing, extra=0.2, vol_iou=vol_iou,
                    guard_roughness=0.005)

    def test_roughness_keeps_its_guard_prefix(self):
        """It must stay impossible to read the guard as one of the ranked criteria."""
        s = summarise([dict(fp_iou=0.9, missing=0.1, extra=0.2, vol_iou=0.8, guard_roughness=0.005)])
        self.assertIn("guard_roughness", s)
        self.assertNotIn("surface_roughness", s)
        self.assertNotIn("roughness", set(s) - {"guard_roughness"})

    def test_empty_arm_summarises_to_nothing(self):
        self.assertEqual(summarise([]), {})

    def test_collapse_rate_uses_the_pre_registered_hollow_boundary(self):
        s = summarise([self._row(COLLAPSE_MISSING - 0.001),
                       self._row(COLLAPSE_MISSING),
                       self._row(0.9)])
        self.assertAlmostEqual(s["collapse_rate"], 2 / 3)

    def test_beats_envelope_is_a_strict_paired_rate(self):
        candidate = {1: self._row(0.0, 0.9), 2: self._row(0.0, 0.5), 3: self._row(0.0, 0.7)}
        envelope = {1: self._row(0.0, 0.8), 2: self._row(0.0, 0.5), 3: self._row(0.0, 0.8)}
        self.assertAlmostEqual(reference_win_rate(candidate, envelope), 1 / 3)

    def test_beats_envelope_refuses_unpaired_rows(self):
        with self.assertRaises(ValueError):
            reference_win_rate({1: self._row(0.0)}, {2: self._row(0.0)})


class TestIdSet(unittest.TestCase):
    """A fixed id set is what makes two runs comparable; ids are global rows of real.h5."""

    def _fake_cache(self, tmp, rows, held, region=None):
        import h5py
        p = Path(tmp) / "latents.h5"
        with h5py.File(p, "w") as f:
            f["row"] = np.asarray(rows, np.int32)
            f["held_out"] = np.asarray(held, np.uint8)
            if region is not None:
                f["region"] = np.asarray(region, np.int32)
        return p

    def test_ids_without_a_region_column_are_the_held_out_rows_ascending(self):
        """Caches written before the region column still load -- they just cannot be stratified."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._fake_cache(tmp, [7, 3, 9, 1], [1, 1, 0, 1])
            ids, lat_of = pick_ids(p, None)
            self.assertEqual(ids, [1, 3, 7])            # deterministic, and independent of cache order
            self.assertEqual(lat_of[7], 0)              # row -> its index in the cache, not its rank

    def test_every_prefix_of_the_id_set_is_region_balanced(self):
        """The #71 defect this guards: ascending row order tracks SOURCE CORPUS.

        `sorted(lat_of)` made the first 48 ids **100% BAG_real (Dutch)** -- zero German, zero
        Japanese -- while the held-out set is ~34/33/32. Every n=48 figure on map #69 was therefore
        single-region, which void-ed the map's "gap to the blockout closes to 0.007" (really 0.071)
        and #80's 11.9% surplus reduction. Interleaving is what makes `--n 48` a sample rather than
        one country, so it is pinned here rather than left to a comment.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # rows are BLOCKED by region, exactly as the real cache is
            rows = list(range(30))
            region = [0] * 10 + [1] * 10 + [2] * 10
            p = self._fake_cache(tmp, rows, [1] * 30, region)
            ids, _ = pick_ids(p, None)
            self.assertEqual(len(ids), 30)
            self.assertEqual(sorted(ids), rows)                    # nothing dropped or duplicated
            of = {r: g for r, g in zip(rows, region)}
            for n in (3, 6, 12, 24):
                seen = [of[i] for i in ids[:n]]
                self.assertEqual({seen.count(g) for g in (0, 1, 2)}, {n // 3},
                                 f"prefix of {n} is not region-balanced: {seen}")

    def test_id_order_is_deterministic_across_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._fake_cache(tmp, list(range(12)), [1] * 12, [0, 1, 2] * 4)
            self.assertEqual(pick_ids(p, None)[0], pick_ids(p, None)[0])

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


class TestFootprintSplit(unittest.TestCase):
    """#85. Criterion 2 is fringe / spill / uncovered, and fringe must never reach the verdict."""

    @staticmethod
    def _occ(mask):
        """A 64^3 occupancy whose vertical projection is `mask`."""
        o = np.zeros((mask.shape[0], 8, mask.shape[1]), bool)
        o[:, 2:6, :] = mask[:, None, :]
        return o

    def test_exact_footprint_scores_zero_everywhere(self):
        fp = np.zeros((32, 32), bool); fp[8:24, 8:24] = True
        s = footprint_split(self._occ(fp), fp)
        self.assertEqual((s["fringe"], s["spill"], s["uncovered"]), (0.0, 0.0, 0.0))

    def test_a_one_voxel_boundary_error_is_FRINGE_not_spill(self):
        """The whole point: a half-voxel rounding of the edge must not count against the model."""
        from scipy import ndimage
        fp = np.zeros((32, 32), bool); fp[8:24, 8:24] = True
        grown = ndimage.binary_dilation(fp, iterations=1)
        s = footprint_split(self._occ(grown), fp)
        self.assertGreater(s["fringe"], 0.0)
        self.assertEqual(s["spill"], 0.0, "a 1-voxel overshoot must be fringe, never spill")
        self.assertEqual(s["uncovered"], 0.0)

    def test_a_detached_mass_outside_the_footprint_is_SPILL(self):
        fp = np.zeros((32, 32), bool); fp[8:16, 8:16] = True
        built = fp.copy(); built[24:28, 24:28] = True          # clear of the footprint and its band
        s = footprint_split(self._occ(built), fp)
        self.assertGreater(s["spill"], 0.0)
        self.assertEqual(s["uncovered"], 0.0)

    def test_a_filled_courtyard_is_SPILL_and_an_unfilled_one_is_UNCOVERED(self):
        """The failure mode only visible off the Dutch-only sample: a ring footprint filled solid."""
        fp = np.zeros((40, 40), bool); fp[6:34, 6:34] = True; fp[16:24, 16:24] = False
        solid = fp.copy(); solid[16:24, 16:24] = True          # courtyard filled in
        self.assertGreater(footprint_split(self._occ(solid), fp)["spill"], 0.0)
        eaten = np.zeros_like(fp); eaten[6:34, 6:20] = True     # half the ring not built
        self.assertGreater(footprint_split(self._occ(eaten), fp)["uncovered"], 0.0)

    def test_tolerance_zero_turns_every_fringe_pixel_into_spill(self):
        from scipy import ndimage
        fp = np.zeros((32, 32), bool); fp[8:24, 8:24] = True
        grown = ndimage.binary_dilation(fp, iterations=1)
        s = footprint_split(self._occ(grown), fp, tol=0)
        self.assertEqual(s["fringe"], 0.0)
        self.assertGreater(s["spill"], 0.0)


class TestVsInput(unittest.TestCase):
    """#75's no-op detector, which the map requires beside every quality number.

    A2's apparent quality came almost entirely from NOT ACTING: at 80k steps it scored 3D IoU 0.857
    while being 99.9% its own input -- the blockout, returned. A score earned that way belongs to the
    blockout, not the generator, so this must be impossible to omit.
    """

    def test_returning_the_input_scores_one(self):
        b = _box((2, 2, 2), (10, 10, 10))
        self.assertEqual(vs_input(b, b), 1.0)

    def test_acting_scores_below_one(self):
        b = _box((2, 2, 2), (10, 10, 10))
        carved = b.copy(); carved[2:4] = False
        self.assertLess(vs_input(carved, b), 1.0)

    def test_a_small_edit_still_reads_as_nearly_a_no_op(self):
        """The case that fooled this map: a 7% edit reads 0.93, not 0.5."""
        b = _box((0, 0, 0), (16, 16, 16))          # 4096 voxels, the whole grid
        edited = b.copy(); edited[:1] = False      # remove ~6% of it
        self.assertGreater(vs_input(edited, b), 0.9)

    def test_disjoint_output_scores_zero(self):
        a = _box((0, 0, 0), (4, 4, 4))
        b = _box((8, 8, 8), (12, 12, 12))
        self.assertEqual(vs_input(a, b), 0.0)


class TestCriterion2Constants(unittest.TestCase):
    """The gate is a decision, not a tuning knob. Pin it so it cannot drift silently."""

    def test_tolerance_is_the_project_detail_scale(self):
        """s* is ADR 0004's massing/detail line (1.0 m ~ 3 voxels @64^3), fixed BEFORE any result.

        If this ever changes to make a number pass, that is a moving goalpost, which #85 was opened
        to prevent. Changing it means changing ADR 0004 and re-stating criterion 2 deliberately.
        """
        self.assertEqual(S_STAR_VOXELS, 3)

    def test_allowance_is_the_value_the_human_chose(self):
        """5%, chosen 2026-08-07 against the FULL held-out set (76.5% [73.4, 79.6] at n=714).

        10% was measured (92.3%) and rejected as a visible fault rather than an approximation.
        """
        self.assertEqual(C2_ALLOWANCE, 0.05)


class TestSharpNormalError(unittest.TestCase):
    """#79. What must hold for SNE to be worth reporting at all.

    The map has been burned three times by a scalar that ranks a melted blob ABOVE a crisp one (#36,
    #63, `deployed-vs-dora`). SNE earns its place only by getting that order right, so the ordering is
    pinned here rather than left to a one-off probe. Rasterisation is real but tiny -- few views, small
    images -- so this stays seconds on CPU.
    """

    @staticmethod
    def _sdf_box(res=48, half=0.42, jitter=0.0, seed=0):
        ax = np.linspace(-1.0, 1.0, res)
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
        d = np.abs(g) - half
        f = (np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)).astype(np.float32)
        if jitter:
            rng = np.random.default_rng(seed)
            # low-frequency wobble == the melt failure mode: the surface moves, edges stop being edges
            n = rng.normal(0, 1, (6, 6, 6))
            from scipy.ndimage import zoom
            f = f + jitter * zoom(n, res / 6, order=3)[:res, :res, :res].astype(np.float32)
        return f

    def test_identical_geometry_scores_zero_and_melt_scores_worse(self):
        import torch
        from scripts.foundations.eval_massing_arms import sharp_normal_error

        gt = self._sdf_box()
        fields = {0: {"gt": gt, "same": gt.copy(), "melted": self._sdf_box(jitter=0.05)}}
        out = sharp_normal_error(fields, ["gt", "same", "melted"], torch.device("cpu"),
                                 views=4, size=96)

        # a metric that does not return 0 for the identical mesh is measuring something else
        self.assertAlmostEqual(out["gt"], 0.0, places=6)
        self.assertAlmostEqual(out["same"], 0.0, places=6)
        # and the ordering the whole ticket turns on
        self.assertGreater(out["melted"], out["same"],
                           "SNE must rank a melted surface WORSE than an identical one -- this is "
                           "exactly what surface_roughness gets backwards")


if __name__ == "__main__":
    unittest.main(verbosity=2)
