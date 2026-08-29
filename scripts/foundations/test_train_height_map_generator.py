"""Contract tests for #127's height-map generator. Synthetic, fast, CPU, no corpus, no GPU.

#127 asks one question -- does a footprint-conditioned height-map generator actually *carve*, or
does it learn identity like every arm before it? A wrong answer is cheap to produce two ways, and
both are pinned here rather than trusted:

  * **Leakage.** The conditioning must be a function of the footprint, the conditioned height and
    the region ONLY. If the target height field reaches the input by any route the answer is
    meaningless, so `condition_channels` is pinned to a signature that cannot see it, and the
    retrieval bank is pinned to exclude the held-out rows it is scored on.
  * **The invariants the output space is *claimed* to give for free.** #127's case rests on
    "footprint-exact, collapse-impossible, valid by construction". Those are properties of
    `apply_depth`, not of the corpus, and they only hold if the clamp is right. `missing` is NOT
    among them -- a generator that over-carves cuts into GT -- and the test that says so is
    deliberate.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_train_height_map_generator.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.eval_massing_arms import RES, volume_split  # noqa: E402
from scripts.foundations.recover_massing_programs import occupancy  # noqa: E402
from scripts.foundations.train_height_map_generator import (  # noqa: E402
    COND_CHANNELS, DEPTH_CLASSES, apply_depth, carve_depth, condition_channels, decode_logits,
    head_channels, height_split, mean_relative_depth, mean_roof_height,
    per_column_loss, decode_prediction, retrieve_nn, roof_description_length,
    roof_shape_stats, summarise, verdict,
)


def _rect(res, z0, z1, x0, x1):
    m = np.zeros((res, res), bool)
    m[z0:z1, x0:x1] = True
    return m


class TestCarveDepthCoding(unittest.TestCase):
    """The label. `depth = extent - top`, which is exactly the per-column carve the blockout needs."""

    def test_round_trip_is_exact(self):
        fp = _rect(16, 2, 10, 3, 11)
        top = np.where(fp, np.int16(7), 0).astype(np.int16)
        top[4:7, 5:9] = 3
        d = carve_depth(top, fp, extent=9)
        np.testing.assert_array_equal(apply_depth(fp, 9, d), top)

    def test_depth_is_zero_off_the_footprint(self):
        fp = _rect(16, 2, 6, 2, 6)
        top = np.where(fp, np.int16(4), 0).astype(np.int16)
        d = carve_depth(top, fp, extent=9)
        self.assertEqual(int(d[~fp].sum()), 0)

    def test_a_flat_full_height_building_is_the_all_zero_label(self):
        """The 42% empty-program majority #10 measured: nothing to carve is the class-0 label."""
        fp = _rect(16, 2, 10, 2, 10)
        top = np.where(fp, np.int16(12), 0).astype(np.int16)
        self.assertEqual(int(np.abs(carve_depth(top, fp, extent=12)).sum()), 0)

    def test_labels_fit_the_class_budget(self):
        """64 classes is the whole range: a column can be carved at most extent-1 below the top."""
        fp = _rect(16, 0, 16, 0, 16)
        top = np.where(fp, np.int16(1), 0).astype(np.int16)
        self.assertLess(int(carve_depth(top, fp, extent=DEPTH_CLASSES).max()), DEPTH_CLASSES)


class TestApplyDepthInvariants(unittest.TestCase):
    """#127's structural claim, made falsifiable: what does the output space give for free?"""

    def test_it_is_footprint_exact_whatever_the_prediction(self):
        fp = _rect(16, 2, 10, 3, 11)
        for d in (np.zeros((16, 16), np.int16), np.full((16, 16), 99, np.int16),
                  np.random.default_rng(0).integers(0, 40, (16, 16)).astype(np.int16)):
            h = apply_depth(fp, 9, d)
            np.testing.assert_array_equal(h > 0, fp)

    def test_every_footprint_column_keeps_at_least_one_voxel(self):
        """Collapse-impossible in the sense #127 means: no hole is ever punched through the plan."""
        fp = _rect(16, 2, 10, 3, 11)
        h = apply_depth(fp, 9, np.full((16, 16), 1000, np.int16))
        self.assertEqual(int(h[fp].min()), 1)

    def test_the_carve_is_purely_subtractive(self):
        """A prediction can never exceed the blockout, so `extra` can never be worse than doing
        nothing. That is the guarantee the depth parameterisation buys, and it is why the arm is
        parameterised as depth rather than as an absolute top."""
        fp = _rect(16, 2, 10, 3, 11)
        rng = np.random.default_rng(1)
        blockout = apply_depth(fp, 9, np.zeros((16, 16), np.int16))
        for _ in range(8):
            d = rng.integers(-5, 40, (16, 16)).astype(np.int16)
            self.assertTrue(bool((apply_depth(fp, 9, d) <= blockout).all()))

    def test_missing_is_NOT_free_and_the_arm_can_still_collapse(self):
        """⚠️ #127 says "`missing` and `collapse_rate` are 0 by clamping". Only the *validity* of
        the solid is free. Over-carving still eats GT, so the collapse rate has to be measured and
        published rather than assumed away."""
        fp = _rect(16, 2, 10, 3, 11)
        gt = occupancy(fp, 0, apply_depth(fp, 9, np.zeros((16, 16), np.int16)))
        over = occupancy(fp, 0, apply_depth(fp, 9, np.full((16, 16), 8, np.int16)))
        self.assertGreater(volume_split(over, gt)["missing"], 0.15)


class TestHeightSplit(unittest.TestCase):
    """The per-epoch validation metric. It only earns its 200x saving if it is the same number."""

    def test_height_split_agrees_with_volume_split(self):
        """Pinned against the voxel path every scored arm actually goes through."""
        rng = np.random.default_rng(3)
        fp = _rect(RES, 8, 40, 11, 52)
        fp[20:28, 30:44] = False                                  # a concave plan, not a rectangle
        for _ in range(6):
            extent = int(rng.integers(6, 60))
            target = apply_depth(fp, extent, rng.integers(0, extent, fp.shape).astype(np.int16))
            pred = apply_depth(fp, extent, rng.integers(-4, extent + 4, fp.shape).astype(np.int16))
            got = height_split(pred, target)
            want = volume_split(occupancy(fp, 2, pred), occupancy(fp, 2, target))
            for k in ("vol_iou", "missing", "extra"):
                self.assertAlmostEqual(got[k], want[k], places=12, msg=k)

    def test_an_exact_prediction_scores_zero_on_both_halves(self):
        fp = _rect(RES, 4, 20, 4, 20)
        t = apply_depth(fp, 12, np.full(fp.shape, 3, np.int16))
        s = height_split(t, t)
        self.assertEqual((s["missing"], s["extra"], s["vol_iou"]), (0.0, 0.0, 1.0))


class TestRoofDescriptionLength(unittest.TestCase):
    """🔑 The form metric, pinned on shapes whose answer is known BY CONSTRUCTION.

    This is the measure that succeeded where the three amplitude statistics failed, so what it
    claims has to be checkable without reference to the corpus: a plane is one operation however
    steep, a gable is two, and a mound is many. If a change to the fitter ever breaks that ordering,
    the metric has stopped meaning what the write-up says it means.
    """

    E = 40                                          # extent in voxels; y0 is arbitrary

    def _surface(self, h):
        fp = np.zeros((RES, RES), bool)
        fp[16:48, 12:52] = True
        return fp, np.where(fp, np.clip(np.rint(h), 1, self.E), 0).astype(np.int16)

    def _ops(self, h):
        fp, surf = self._surface(h)
        return roof_description_length(surf, fp, 8, self.E)

    def test_a_flat_roof_is_one_operation(self):
        self.assertEqual(self._ops(np.full((RES, RES), 30.0))["ops"], 1)

    def test_a_tilted_plane_is_still_one_operation(self):
        """🔑 Slope is not complexity. This is exactly what `roof_relief` got wrong: a shed roof
        steps at every column and is nonetheless the simplest roof after a flat one."""
        _, xx = np.mgrid[0:RES, 0:RES]
        r = self._ops(30 - 0.45 * (xx - 12))
        self.assertEqual(r["ops"], 1)
        self.assertEqual(r["planar_fraction"], 1.0)          # spent on a Ramp, not a Layer

    def test_a_gable_is_two_operations(self):
        _, xx = np.mgrid[0:RES, 0:RES]
        self.assertLessEqual(self._ops(34 - 0.55 * np.abs(xx - 32))["ops"], 2)

    def test_a_dome_costs_far_more_than_a_gable_and_spends_it_on_layers(self):
        """The mound the montages show: no plane explains it, so the fitter stacks flat terraces."""
        zz, xx = np.mgrid[0:RES, 0:RES]
        gable = self._ops(34 - 0.55 * np.abs(xx - 32))
        dome = self._ops(36 - 0.020 * ((xx - 32) ** 2 + (zz - 32) ** 2))
        self.assertGreater(dome["ops"], 3 * gable["ops"])
        self.assertLess(dome["planar_fraction"], 0.5)

    def test_noise_exhausts_the_budget_and_is_not_explained(self):
        zz, xx = np.mgrid[0:RES, 0:RES]
        r = self._ops(30 + 4 * np.sin(xx * 0.9) * np.cos(zz * 0.9))
        self.assertEqual(r["ops"], 16)
        self.assertFalse(r["explained"])

    def test_the_envelope_itself_costs_zero_operations(self):
        """⚠️ The fitter starts FROM the envelope, so an arm that did nothing needs no operations to
        explain. 0 ops means "identical to the input", which reads the same way `vs_input` = 1.000
        does -- and it is why this metric must never be read without `extra` beside it."""
        self.assertEqual(self._ops(np.full((RES, RES), float(self.E)))["ops"], 0)

    def test_a_flat_roof_below_the_envelope_costs_one(self):
        """One `Layer` down from the envelope: the simplest carve there is."""
        self.assertEqual(self._ops(np.full((RES, RES), float(self.E) - 10))["ops"], 1)


class TestRoofShapeStats(unittest.TestCase):
    """The three roof-shape statistics. ⚠️ All three were measured NOT to separate the arms (see
    `roof_shape_stats`); what is pinned here is only that each computes what it claims to, so the
    negative result is a fact about the roofs and not about a broken implementation."""

    def test_a_flat_roof_is_zero_on_all_three(self):
        fp = _rect(16, 2, 12, 2, 12)
        s = roof_shape_stats(apply_depth(fp, 9, np.zeros(fp.shape, np.int16)), fp)
        self.assertEqual((s["relief"], s["curvature"], s["speckle"]), (0.0, 0.0, 0.0))

    def test_a_constant_slope_has_relief_but_no_curvature(self):
        """A shed roof rising one voxel per column: relief 0.5 over both axes, curvature 0 -- which
        is the property that made curvature look like the right discriminator before it was run."""
        fp = _rect(16, 0, 16, 0, 16)
        h = (np.arange(16)[None, :] + 1).astype(np.int16) * np.ones((16, 1), np.int16)
        s = roof_shape_stats(h, fp)
        self.assertAlmostEqual(s["relief"], 0.5, places=6)
        self.assertAlmostEqual(s["curvature"], 0.0, places=6)

    def test_a_single_spike_is_speckle(self):
        fp = _rect(16, 2, 12, 2, 12)
        h = apply_depth(fp, 20, np.zeros(fp.shape, np.int16))
        h[7, 7] = 3
        self.assertGreater(roof_shape_stats(h, fp)["speckle"], 0.0)

    def test_noise_under_the_envelope_costs_shape_but_no_extra(self):
        """🔑 The blind spot these were reaching for: rubble that stays below GT is free on `extra`."""
        fp = _rect(16, 2, 12, 2, 12)
        gt = apply_depth(fp, 20, np.zeros(fp.shape, np.int16))
        noisy = apply_depth(fp, 20, np.indices(fp.shape)[0] % 5)
        self.assertEqual(height_split(noisy, gt)["extra"], 0.0)
        self.assertGreater(roof_shape_stats(noisy, fp)["relief"], roof_shape_stats(gt, fp)["relief"])


class TestConditioningCarriesNoAnswer(unittest.TestCase):
    """The leakage guard. The input is the conditioning #127 names -- footprint, height, region."""

    def test_two_buildings_with_the_same_conditioning_get_identical_input(self):
        """Different roofs, same footprint/height/region => the network cannot tell them apart.
        If this ever fails, the conditioning has grown a channel that saw the target."""
        fp = _rect(16, 2, 10, 3, 11)
        a = condition_channels(fp, extent=9, height_m=12.0, region=1)
        b = condition_channels(fp, extent=9, height_m=12.0, region=1)
        np.testing.assert_array_equal(a, b)

    def test_the_conditioned_height_reaches_the_input(self):
        fp = _rect(16, 2, 10, 3, 11)
        self.assertFalse(np.array_equal(condition_channels(fp, 9, 12.0, 1),
                                        condition_channels(fp, 20, 12.0, 1)))

    def test_the_region_reaches_the_input(self):
        fp = _rect(16, 2, 10, 3, 11)
        self.assertFalse(np.array_equal(condition_channels(fp, 9, 12.0, 0),
                                        condition_channels(fp, 9, 12.0, 2)))

    def test_the_channel_count_matches_the_model_input(self):
        """A channel added here and not there is a silent shape error at the first batch."""
        fp = _rect(16, 2, 10, 3, 11)
        self.assertEqual(condition_channels(fp, 9, 12.0, 1).shape[0], COND_CHANNELS)

    def test_channels_are_finite_and_bounded(self):
        fp = _rect(16, 0, 16, 0, 16)
        c = condition_channels(fp, 64, 300.0, 2)
        self.assertTrue(np.isfinite(c).all())
        self.assertLessEqual(float(np.abs(c).max()), 4.0)


class TestDecode(unittest.TestCase):
    """Argmax, not expectation: the mean of a bimodal roof distribution is a roof nobody built."""

    def test_decode_takes_the_argmax_class(self):
        fp = _rect(8, 1, 7, 1, 7)
        logits = np.zeros((DEPTH_CLASSES, 8, 8), np.float32)
        logits[3] = 1.0
        np.testing.assert_array_equal(decode_logits(logits, fp, extent=9),
                                      apply_depth(fp, 9, np.full((8, 8), 3, np.int16)))

    def _flat_posterior(self, p, shape=(8, 8)):
        z = np.log(np.maximum(np.asarray(p, np.float64), 1e-12))
        return (z[:, None, None] * np.ones((1,) + shape)).astype(np.float32)

    def test_the_median_decode_is_the_ordinal_posterior_quantile(self):
        """cdf = .40, .52, ... so the mode is class 0 and the median is class 1."""
        fp = _rect(8, 1, 7, 1, 7)
        p = np.zeros(DEPTH_CLASSES, np.float64)
        p[0], p[1:6] = 0.40, 0.12
        logits = self._flat_posterior(p)
        self.assertEqual(int(carve_depth(decode_logits(logits, fp, 20), fp, 20)[fp].max()), 0)
        self.assertEqual(int(carve_depth(decode_logits(logits, fp, 20, quantile=0.5), fp, 20)
                             [fp].max()), 1)

    def test_the_mode_can_carve_nothing_where_the_median_carves(self):
        """🔑 The mode-shrinkage this ablation exists to isolate: a column whose posterior puts 45%
        on "do not carve" and 55% spread over depths 6..15 has its mode at 0 and its median at 6."""
        fp = _rect(8, 1, 7, 1, 7)
        p = np.zeros(DEPTH_CLASSES, np.float64)
        p[0], p[6:16] = 0.45, 0.055
        logits = self._flat_posterior(p)
        self.assertEqual(int(decode_logits(logits, fp, 20)[fp].max()), 20)          # carves nothing
        self.assertEqual(int(decode_logits(logits, fp, 20, quantile=0.5)[fp].max()), 20 - 6)

    def test_decode_never_leaves_the_footprint(self):
        fp = _rect(8, 1, 7, 1, 7)
        logits = np.random.default_rng(2).normal(size=(DEPTH_CLASSES, 8, 8)).astype(np.float32)
        np.testing.assert_array_equal(decode_logits(logits, fp, 9) > 0, fp)


class TestMeanRoofBaseline(unittest.TestCase):
    """#127's `mean roof` arm: the unconditional version of the regression-to-the-mean trap."""

    def test_profile_is_the_mean_relative_depth_per_cell(self):
        fp = _rect(8, 0, 8, 0, 8)
        depths = np.stack([np.zeros((8, 8), np.int16), np.full((8, 8), 5, np.int16)])
        prof = mean_relative_depth(depths, np.stack([fp, fp]), np.array([10, 10]))
        np.testing.assert_allclose(prof, 0.25, atol=1e-6)

    def test_cells_no_footprint_covers_are_zero_not_nan(self):
        fp = _rect(8, 0, 4, 0, 4)
        prof = mean_relative_depth(np.zeros((1, 8, 8), np.int16), fp[None], np.array([10]))
        self.assertTrue(np.isfinite(prof).all())

    def test_the_profile_scales_with_the_conditioned_height(self):
        fp = _rect(8, 0, 8, 0, 8)
        prof = np.full((8, 8), 0.25, np.float32)
        self.assertEqual(int(mean_roof_height(prof, fp, 20)[fp].max()), 15)
        self.assertEqual(int(mean_roof_height(prof, fp, 40)[fp].max()), 30)


class TestRetrievalBaseline(unittest.TestCase):
    """1-NN is #127's real bar, so the thing that could flatter it is pinned: seeing the answer."""

    def test_picks_the_footprint_iou_nearest_bank_row(self):
        q = _rect(16, 0, 8, 0, 8)
        bank = np.stack([_rect(16, 0, 2, 0, 2), _rect(16, 0, 8, 0, 7), _rect(16, 8, 16, 8, 16)])
        np.testing.assert_array_equal(retrieve_nn(q[None], bank), [1])

    def test_an_exact_footprint_match_is_preferred(self):
        q = _rect(16, 2, 10, 2, 10)
        bank = np.stack([_rect(16, 2, 10, 2, 9), q.copy(), _rect(16, 2, 11, 2, 10)])
        np.testing.assert_array_equal(retrieve_nn(q[None], bank), [1])

    def test_a_query_absent_from_the_bank_cannot_retrieve_itself(self):
        """The bank is built from TRAINING rows only. This pins the property the caller relies on:
        retrieval returns a bank index, so a held-out row can only be answered by a training row."""
        q = _rect(16, 2, 10, 2, 10)
        bank = np.stack([_rect(16, 0, 3, 0, 3), _rect(16, 12, 16, 12, 16)])
        self.assertIn(int(retrieve_nn(q[None], bank)[0]), (0, 1))


class TestObjectives(unittest.TestCase):
    """The three losses, pinned by the statistic each one's minimiser actually is.

    #127's whole result is that the objective and the decode were naming different statistics, so
    what is tested here is not the algebra but the CLAIM: minimising each loss over a fixed sample
    lands on the mean, the median, or the mode of that sample.
    """

    def _fit_constant(self, sample, objective, quantile=0.5, extent=64):
        """Minimise the loss of ONE constant prediction against a sample of depths, by search.

        A direct numerical check of "what is this loss's Bayes act", using the production function
        rather than a re-derivation of it -- `test_train_vecset_solidity.py`'s docstring records
        what a test that re-implements its subject is worth.
        """
        import torch
        y = torch.tensor(sample, dtype=torch.long).reshape(1, 1, -1)
        ext = torch.tensor([float(extent)])
        best, arg = None, None
        for cand in np.arange(0, extent, 0.25):
            out = torch.full((1, 1, 1, y.shape[-1]), float(cand) / extent)
            loss = float(per_column_loss(out, y, ext, objective, quantile).mean())
            if best is None or loss < best:
                best, arg = loss, float(cand)
        return arg

    # ⚠️ An ODD sample, and deliberately so. The first version used 8 depths, where every value in
    # [0, 12] minimises the pinball loss equally -- the median of an even sample is an interval, not
    # a point -- so the search returned the interval's left end and the test failed on a correct
    # loss. mode 0, median 12, mean 16.89: three statistics far enough apart to tell apart.
    SAMPLE = [0, 0, 0, 0, 12, 12, 20, 48, 60]

    def test_mse_lands_on_the_mean(self):
        """The bland roof #127's design note warns about -- correctly attributed to MSE."""
        self.assertAlmostEqual(self._fit_constant(self.SAMPLE, "mse"), 16.89, delta=0.3)

    def test_the_pinball_loss_at_half_lands_on_the_median(self):
        """🔑 The whole point of the retrain: this objective's minimiser IS what #127 decodes at."""
        self.assertAlmostEqual(self._fit_constant(self.SAMPLE, "quantile", 0.5), 12.0, delta=0.3)

    def test_a_higher_quantile_carves_deeper(self):
        """q trades `missing` against `extra` directly, which is why 0.5 is pre-committed."""
        mid = self._fit_constant(self.SAMPLE, "quantile", 0.5)
        self.assertLess(self._fit_constant(self.SAMPLE, "quantile", 0.25), mid)
        self.assertGreater(self._fit_constant(self.SAMPLE, "quantile", 0.75), mid)

    def test_cross_entropy_lands_on_the_mode(self):
        """🔑 The defect: on a sample whose mode is 0 and whose median is 12, CE targets 0."""
        import torch
        sample = torch.tensor(self.SAMPLE).reshape(1, 1, -1)
        ext = torch.tensor([64.0])
        losses = []
        for cand in (0, 12, 17, 20):
            out = torch.full((1, DEPTH_CLASSES, 1, sample.shape[-1]), -20.0)
            out[:, cand] = 20.0                                     # a posterior peaked at `cand`
            losses.append(float(per_column_loss(out, sample, ext, "ce", 0.5).mean()))
        self.assertEqual(int(np.argmin(losses)), 0, "CE must prefer the modal depth")

    def test_the_heads_differ_and_the_decodes_match_them(self):
        self.assertEqual(head_channels("ce"), DEPTH_CLASSES)
        self.assertEqual(head_channels("mse"), 1)
        self.assertEqual(head_channels("quantile"), 1)
        fp = _rect(8, 1, 7, 1, 7)
        reg = np.full((1, 8, 8), 4.0 / 20.0, np.float32)            # relative depth 0.2 of 20
        for objective in ("mse", "quantile"):
            np.testing.assert_array_equal(
                decode_prediction(reg, fp, 20, objective, 0.5),
                apply_depth(fp, 20, np.full((8, 8), 4, np.int16)))

    def test_a_trained_regression_ignores_the_decode_quantile(self):
        """⚠️ Its statistic was fixed at training time; only `ce` can be re-read at another one."""
        fp = _rect(8, 1, 7, 1, 7)
        reg = np.full((1, 8, 8), 0.25, np.float32)
        np.testing.assert_array_equal(decode_prediction(reg, fp, 20, "quantile", 0.5),
                                      decode_prediction(reg, fp, 20, "quantile", 0.9))


class TestVerdict(unittest.TestCase):
    """The pre-registered bar itself, evaluated mechanically so a write-up cannot soften it.

    This is the most consequential pure function in the module -- it is what turns the medians into
    PASS / NOT MET -- so each clause is pinned separately rather than through one happy path.
    """

    def _arms(self, **arm_extra):
        """A scorecard shaped like the real one: the envelope, the 1-NN bar, then the candidates."""
        def block(extra, collapse, vs_input):
            return {"carve": dict(extra=extra, collapse_rate=collapse, vs_input=vs_input)}
        arms = {"blockout": block(0.2308, 0.0000, 1.0000),
                "nn_retrieval": block(0.1031, 0.1582, 0.8743)}
        arms.update({k: block(*v) for k, v in arm_extra.items()})
        return arms

    def test_beating_the_1nn_bar_within_the_guards_passes(self):
        v = verdict(self._arms(cand=(0.0603, 0.0268, 0.8432)), "carve")["cand"]
        self.assertEqual((v["beats_1nn_extra"], v["collapse_no_worse_than_1nn"], v["moved"]),
                         (True, True, True))
        self.assertTrue(v["pass"])
        self.assertFalse(v["killed_identity"])

    def test_missing_the_1nn_bar_is_not_met_even_though_it_beats_the_envelope(self):
        """The arm this project would previously have called a win: far better than the blockout,
        short of retrieval. #127 fixed the bar at 1-NN precisely so this reads as NOT MET."""
        v = verdict(self._arms(cand=(0.1178, 0.0316, 0.9304)), "carve")["cand"]
        self.assertFalse(v["beats_1nn_extra"])
        self.assertFalse(v["pass"])
        self.assertFalse(v["killed_identity"])

    def test_a_tie_with_the_bar_is_not_a_pass(self):
        v = verdict(self._arms(cand=(0.1031, 0.0000, 0.5)), "carve")["cand"]
        self.assertFalse(v["beats_1nn_extra"])

    def test_collapsing_worse_than_1nn_fails_however_low_the_surplus(self):
        v = verdict(self._arms(cand=(0.0100, 0.4000, 0.5000)), "carve")["cand"]
        self.assertTrue(v["beats_1nn_extra"])
        self.assertFalse(v["collapse_no_worse_than_1nn"])
        self.assertFalse(v["pass"])

    def test_an_arm_that_did_not_move_fails_however_good_it_looks(self):
        """#75: a projection at 0.99 vs-input has not been measured as a generator at all."""
        v = verdict(self._arms(cand=(0.0100, 0.0000, 0.9900)), "carve")["cand"]
        self.assertFalse(v["moved"])
        self.assertFalse(v["pass"])

    def test_identity_is_flagged_as_the_kill(self):
        v = verdict(self._arms(cand=(0.2357, 0.1241, 0.9852)), "carve")["cand"]
        self.assertTrue(v["killed_identity"])
        self.assertFalse(v["pass"])

    def test_the_baselines_are_never_judged_against_themselves(self):
        self.assertEqual(set(verdict(self._arms(cand=(0.05, 0.0, 0.8)), "carve")), {"cand"})


class TestSummarise(unittest.TestCase):
    """Medians, not means: #80's bimodal result is why, and an empty population must not raise."""

    def _row(self, **kw):
        r = dict(missing=0.0, extra=0.0, vs_input=1.0, carved_cols=0.0, gt_carved_cols=0.0,
                 roof_relief=0.0, roof_curvature=0.0, roof_speckle=0.0, gt_roof_relief=0.0,
                 gt_roof_curvature=0.0, gt_roof_speckle=0.0, spill=0.0, fp_iou=1.0, vol_iou=1.0)
        r.update(kw)
        return r

    def test_it_takes_the_median_not_the_mean(self):
        rows = [self._row(extra=e) for e in (0.0, 0.0, 0.9)]
        self.assertEqual(summarise(rows)["extra"], 0.0)

    def test_collapse_rate_is_the_rate_of_buildings_over_the_threshold(self):
        rows = [self._row(missing=m) for m in (0.0, 0.0, 0.20, 0.99)]
        self.assertAlmostEqual(summarise(rows)["collapse_rate"], 0.5)

    def test_an_empty_population_is_nan_and_does_not_raise(self):
        s = summarise([])
        self.assertEqual(s["n"], 0)
        self.assertTrue(np.isnan(s["extra"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
