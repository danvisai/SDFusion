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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.eval_massing_arms import RES, volume_split  # noqa: E402
from scripts.foundations.recover_massing_programs import (  # noqa: E402
    K_OPS, fit_program, occupancy, program_to_slots, replay_program,
)
from scripts.foundations.train_height_map_generator import (  # noqa: E402
    COND_CHANNELS, DEPTH_CLASSES, PLANE_BINS, PROGRAM_TYPES, _d4, _d4_program,
    apply_depth, bins_to_plane, carve_depth, compile_program,
    condition_channels, decode_logits, decode_plane_logits,
    head_channels, height_split, mean_relative_depth, mean_roof_height,
    differentiable_depth, height_rgb, normal_rgb,
    per_column_loss, decode_prediction, PLANE_DECODE, PLANE_QUANTITIES, plane_to_bins,
    plane_to_normalised,
    plane_to_voxel, rebin_planes,
    ASSIGN_DECODE, ASSIGN_TEMPERATURE, assignment_prior, assignment_stats, decode_assignment,
    program_loss, retrieve_nn, roof_description_length, sheet_picks, slot_centroids,
    slope_loss, SLOPE_DECODE_QUANTILE,
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

    def _form_arms(self, **arm_extra):
        """A scorecard carrying the form columns #6's bar is written on."""
        def block(extra, ops, planar, collapse=0.0, vs_input=0.85):
            return {"carve": dict(extra=extra, collapse_rate=collapse, vs_input=vs_input,
                                  dl_ops=ops, dl_planar_fraction=planar)}
        arms = {"blockout": block(0.2308, 0.0, 0.00),
                "nn_retrieval": block(0.1031, 2.0, 0.17)}
        arms.update({k: block(*v) for k, v in arm_extra.items()})
        return arms

    def test_the_program_bar_needs_all_three_clauses(self):
        """#6's bar, evaluated here rather than in prose -- which is this function's whole reason
        for existing. An arm at the compiled label's own form and under the served arm's surplus."""
        v = verdict(self._form_arms(cand=(0.0400, 2.0, 0.50)), "carve")["cand"]
        self.assertEqual((v["form_ops_under_bar"], v["form_planar_over_bar"],
                          v["beats_served_extra"]), (True, True, True))
        self.assertTrue(v["program_pass"])
        self.assertFalse(v["killed_flat"])

    def test_the_guards_are_part_of_the_composite_and_not_only_of_the_prose(self):
        """🔑 #129's endpoint is why. It cleared BOTH form clauses while collapsing 37% of its
        buildings -- more than double 1-NN's 15.8% -- and the composite, which ANDed only the three
        PASS clauses, had nothing to say about it. The written bar always had the guards in it."""
        v = verdict(self._form_arms(cand=(0.0400, 2.0, 0.50, 0.3747)), "carve")["cand"]
        self.assertEqual((v["form_ops_under_bar"], v["form_planar_over_bar"],
                          v["beats_served_extra"]), (True, True, True))
        self.assertFalse(v["collapse_no_worse_than_1nn"])
        self.assertFalse(v["program_pass"])

    def test_an_arm_that_did_not_move_cannot_collect_a_program_pass_either(self):
        """The other guard, and #75's rule: 0.99 vs-input has not been measured as a generator."""
        v = verdict(self._form_arms(cand=(0.0400, 2.0, 0.50, 0.0, 0.9900)), "carve")["cand"]
        self.assertFalse(v["moved"])
        self.assertFalse(v["program_pass"])

    def test_a_terrace_is_killed_however_short_its_description(self):
        """🔑 The trap the bar exists to catch, and it is not hypothetical: #127's plane head scored
        3.0 ops with planar_fraction 0.00, and #6's own arm scored 1.0 ops with 0.00. A
        single-number form metric calls both an improvement."""
        v = verdict(self._form_arms(cand=(0.0400, 1.0, 0.00)), "carve")["cand"]
        self.assertTrue(v["form_ops_under_bar"])          # the ops half looks like a win
        self.assertFalse(v["form_planar_over_bar"])       # and the planar half is the truth
        self.assertFalse(v["program_pass"])
        self.assertTrue(v["killed_flat"])

    def test_good_form_does_not_excuse_leaving_the_surplus(self):
        v = verdict(self._form_arms(cand=(0.1236, 2.0, 0.50)), "carve")["cand"]
        self.assertFalse(v["beats_served_extra"])
        self.assertFalse(v["program_pass"])

    def test_the_kill_boundary_is_the_served_arms_own_planar_fraction(self):
        """`<=`, not `<`: matching the arm you replaced is not an improvement over it."""
        self.assertTrue(verdict(self._form_arms(c=(0.04, 2.0, 0.20)), "carve")["c"]["killed_flat"])
        self.assertFalse(verdict(self._form_arms(c=(0.04, 2.0, 0.21)), "carve")["c"]["killed_flat"])

    def test_an_arm_scored_without_the_form_metric_gets_no_form_verdict(self):
        """`--no_form` is a real run mode, and a missing measurement must read as absent rather
        than as a pass."""
        v = verdict(self._arms(cand=(0.0400, 0.0, 0.85)), "carve")["cand"]
        for k in ("form_ops_under_bar", "form_planar_over_bar", "program_pass", "killed_flat"):
            self.assertNotIn(k, v)

    def test_an_arm_that_sees_gt_is_a_ceiling_and_is_never_given_a_verdict(self):
        """⚠️ #6's compiled-label arm is the fitter's own program with the answer in hand. It scores
        better than anything a generator could and would collect a PASS mechanically, which would be
        this scorecard reporting that the target was hit by looking at it. It is a reference the
        trained arms are read against, so it must not appear in the verdict at all."""
        arms = self._arms(cand=(0.0603, 0.0268, 0.8432))
        arms["program_label (sees GT)"] = {"carve": dict(extra=0.0035, collapse_rate=0.0,
                                                         vs_input=0.83)}
        v = verdict(arms, "carve")
        self.assertNotIn("program_label (sees GT)", v)
        self.assertIn("cand", v)

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


class TestPlanViewMaps(unittest.TestCase):
    """The height and normal maps must show what the montage cannot: whether the roof is planar.

    The claim the sheet is read for is that a normal map separates a roof from a mound -- one flat
    colour per pitch, a seam at a ridge, a continuum on a dome. `roof_shape_stats` is a recorded
    negative for exactly this question, so the replacement is pinned on shapes whose answer is known
    by construction rather than trusted because the pictures look right.
    """

    GRID, EXTENT = 32, 20

    def _fp(self):
        return _rect(self.GRID, 8, 24, 8, 24)

    def _interior_colours(self, h, fp):
        rgb = normal_rgb(h, fp)[10:22, 10:22]      # one ring in from the footprint wall
        return {tuple(c) for c in rgb.reshape(-1, 3)}

    def test_off_the_footprint_both_maps_are_background(self):
        fp = self._fp()
        h = apply_depth(fp, self.EXTENT, np.zeros_like(fp, np.int16))
        for rgb in (height_rgb(h, fp, self.EXTENT), normal_rgb(h, fp)):
            self.assertTrue((rgb[~fp] == 246).all())
            self.assertFalse((rgb[fp] == 246).all())

    def test_a_flat_roof_is_one_normal_and_it_is_up(self):
        fp = self._fp()
        h = apply_depth(fp, self.EXTENT, np.zeros_like(fp, np.int16))
        self.assertEqual(self._interior_colours(h, fp), {(127, 127, 255)})

    def test_one_pitch_is_one_colour_and_a_gable_is_two_plus_a_ridge(self):
        fp = self._fp()
        z, x = np.mgrid[0:self.GRID, 0:self.GRID]
        pitch = apply_depth(fp, self.EXTENT, np.clip(z - 8, 0, None).astype(np.int16))
        gable = apply_depth(fp, self.EXTENT,
                            np.clip(np.minimum(x - 8, 23 - x), 0, None).astype(np.int16))
        self.assertEqual(len(self._interior_colours(pitch, fp)), 1)
        # two pitches and the ridge between them, which is two columns wide on an even-width
        # plan -- four colours in total, and nothing else
        self.assertLessEqual(len(self._interior_colours(gable, fp)), 4)

    def test_a_dome_is_a_continuum_of_normals_and_a_gable_is_not(self):
        fp = self._fp()
        z, x = np.mgrid[0:self.GRID, 0:self.GRID]
        r = np.sqrt((z - 15.5) ** 2 + (x - 15.5) ** 2)
        dome = apply_depth(fp, self.EXTENT, np.clip(r, 0, None).astype(np.int16))
        gable = apply_depth(fp, self.EXTENT,
                            np.clip(np.minimum(x - 8, 23 - x), 0, None).astype(np.int16))
        self.assertGreater(len(self._interior_colours(dome, fp)),
                           4 * len(self._interior_colours(gable, fp)))

    def test_the_height_ramp_is_shared_so_equal_levels_get_equal_colour(self):
        fp = self._fp()
        z, _ = np.mgrid[0:self.GRID, 0:self.GRID]
        a = apply_depth(fp, self.EXTENT, np.clip(z - 8, 0, None).astype(np.int16))
        b = apply_depth(fp, self.EXTENT, np.zeros_like(fp, np.int16))
        ra, rb = (height_rgb(m, fp, self.EXTENT, contour=0, lo=4) for m in (a, b))
        top = a == self.EXTENT                                    # the columns neither arm carved
        self.assertTrue((ra[top] == rb[top]).all())
        self.assertFalse((ra[fp] == rb[fp]).all())


class TestSlopeLoss(unittest.TestCase):
    """The joint term. Cross-entropy judges every column alone, so a mound and a roof that remove
    the same volume cost the same; this one judges the STEP between neighbouring columns, which is
    the quantity the normal map draws and the only thing measured so far that separates them.
    """

    GRID, EXTENT = 24, 20

    def setUp(self):
        import torch
        self.torch = torch
        self.fp = _rect(self.GRID, 4, 20, 4, 20)
        z, x = np.mgrid[0:self.GRID, 0:self.GRID]
        # a gable: two opposing pitches meeting at a ridge, in DEPTH (0 = uncarved)
        self.gable = np.clip(np.minimum(x - 4, 19 - x), 0, None).astype(np.float64) * self.fp
        self.mask = torch.from_numpy(self.fp)[None]
        self.y = torch.from_numpy(self.gable)[None]

    def _slope(self, pred):
        return float(slope_loss(self.torch.from_numpy(np.asarray(pred, np.float64))[None],
                                self.y, self.mask))

    def test_an_exact_prediction_costs_nothing(self):
        self.assertAlmostEqual(self._slope(self.gable), 0.0)

    def test_it_is_blind_to_a_constant_offset_because_it_measures_shape(self):
        """🔑 Why it is an ADDITION to cross-entropy and not a replacement for it: the level is
        CE's job, the arrangement is this term's, and neither can do the other's."""
        self.assertAlmostEqual(self._slope(self.gable + 3.0), 0.0)

    def test_two_predictions_with_the_same_column_error_are_separated_by_their_slope(self):
        """The claim, stated as a test: equal error per column, one planar and one domed."""
        z, x = np.mgrid[0:self.GRID, 0:self.GRID]
        dome = np.sqrt(np.clip(64.0 - (z - 11.5) ** 2 - (x - 11.5) ** 2, 0.0, None))
        dome = dome / dome[self.fp].mean()                    # mean |error| of exactly 1 voxel
        planar, domed = self.gable + 1.0, self.gable + dome
        self.assertAlmostEqual(float(np.abs(planar - self.gable)[self.fp].mean()),
                               float(np.abs(domed - self.gable)[self.fp].mean()), places=6)
        self.assertLess(self._slope(planar) + 0.05, self._slope(domed))

    def test_only_footprint_pairs_count(self):
        off = self.gable.copy()
        off[~self.fp] = 99.0
        self.assertAlmostEqual(self._slope(off), 0.0)

    def test_a_ridge_is_not_punished_for_being_sharp(self):
        """⚠️ The failure mode to avoid: a term that merely smooths would prefer a rounded ridge.
        This one matches GT's steps, so the sharp ridge is free and the rounded one is not."""
        rounded = ndimage.uniform_filter(self.gable, 3) * self.fp
        self.assertLess(self._slope(self.gable), self._slope(rounded))


class TestDifferentiableDepth(unittest.TestCase):
    """Hard forward, soft backward -- the same straight-through the plane head already uses.

    A softmax blend of depths is a smooth field, and a smooth field is the mound being fixed, so
    the slope term must see the depth the arm will actually be decoded at while still passing a
    gradient back to the logits it came from.
    """

    def _logits(self):
        import torch
        # #127's own example: a dominant "do nothing" class whose MEDIAN is six voxels deeper.
        # The masses avoid a cumulative sum that lands exactly on 0.5, where the median is an
        # interval rather than a point and float32 rounding decides which end is returned.
        p = np.full(DEPTH_CLASSES, 1e-6)
        p[0], p[6], p[12] = 0.40, 0.35, 0.25
        return torch.log(torch.tensor(p, dtype=torch.float32)).view(1, DEPTH_CLASSES, 1, 1)

    def test_the_forward_value_is_the_decode_not_a_blend(self):
        import torch
        z = self._logits()
        ext = torch.tensor([20.0])
        self.assertEqual(float(differentiable_depth(z, ext, "ce", None)), 0.0)
        self.assertEqual(float(differentiable_depth(z, ext, "ce", 0.5)), 6.0)

    def test_it_agrees_with_the_decode_the_arm_is_read_at(self):
        import torch
        z = self._logits()
        fp = _rect(1, 0, 1, 0, 1)
        for q in (None, 0.5):
            decoded = decode_logits(z[0].detach().numpy(), fp, 20, q)
            self.assertEqual(int(20 - decoded[0, 0]),
                             int(differentiable_depth(z, torch.tensor([20.0]), "ce", q)))

    def test_the_gradient_reaches_the_logits(self):
        import torch
        z = self._logits().requires_grad_(True)
        differentiable_depth(z, torch.tensor([20.0]), "ce", 0.5).sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.abs().sum()), 0.0)

    def test_the_regressions_pass_their_own_depth_through(self):
        import torch
        out = torch.full((1, 1, 2, 2), 0.25)
        d = differentiable_depth(out, torch.tensor([20.0]), "mse", None)
        self.assertAlmostEqual(float(d.mean()), 5.0)

    def test_the_plane_head_is_read_as_a_height_and_inverted_to_depth(self):
        """`planes` hands the loss a composed height map, not logits -- the one objective whose
        `out` is already in voxels, so the conversion is a subtraction and not a decode."""
        import torch
        heights = torch.full((1, 2, 2), 14.0)
        d = differentiable_depth(heights, torch.tensor([20.0]), "planes", 0.5)
        self.assertAlmostEqual(float(d.mean()), 6.0)


class TestSheetPicks(unittest.TestCase):
    """Which buildings land on a sheet, and ranked by WHICH arm.

    ⚠️ Written because the map sheets shipped ranked by the **blockout**: `write_map_sheets` took
    the first key of an arms dict that starts with the baselines, so "worst" meant worst for the
    envelope -- a figure labelled with a claim it was not showing. The rule is one function now and
    this is the pin.
    """

    def test_it_ranks_by_the_arm_it_is_given_not_by_the_first_one(self):
        rank = [0.9, 0.1, 0.5, 0.7, 0.3]
        picks = sheet_picks(rank, [0, 1, 2, 3, 4], 2)
        self.assertEqual(picks["best"], [1, 4])
        self.assertEqual(picks["worst"], [3, 0])

    def test_only_the_eligible_rows_are_drawn(self):
        rank = [0.9, 0.1, 0.5, 0.7, 0.3]
        self.assertEqual(sheet_picks(rank, [0, 3], 1)["best"], [3])

    def test_the_representative_pick_is_the_middle_of_the_ranking(self):
        picks = sheet_picks([0.1, 0.2, 0.3, 0.4, 0.5], list(range(5)), 1)
        self.assertEqual(picks["representative"], [2])

    def test_a_sheet_wider_than_the_population_does_not_raise(self):
        picks = sheet_picks([0.4, 0.2], [0, 1], 6)
        self.assertEqual(picks["best"], [1, 0])


class TestTrainTimeDecodeMatchesTheServedDecode(unittest.TestCase):
    """⚠️ The slope term shapes whatever surface `differentiable_depth` hands it, so if that surface
    is not the one `decode_logits` serves, the arm is trained on a building nobody ever sees. The
    two run in different precisions (float32 on the GPU, float64 at serve time), which is exactly
    the kind of gap that hides until a posterior lands on the quantile."""

    def test_they_agree_on_two_hundred_random_posteriors(self):
        import torch
        rng = np.random.default_rng(0)
        z = torch.tensor(rng.normal(0, 3, (1, DEPTH_CLASSES, 20, 10)), dtype=torch.float32)
        fp = _rect(20, 0, 20, 0, 10)[:, :10]
        served = 40 - decode_logits(z[0].numpy(), fp, 40, SLOPE_DECODE_QUANTILE)
        trained = differentiable_depth(z, torch.tensor([40.0]), "ce",
                                       SLOPE_DECODE_QUANTILE)[0].numpy().astype(np.int16)
        # ⚠️ The one place they may differ is `apply_depth`'s clamp, which keeps a voxel under every
        # footprint column. The training term deliberately sees the UNCLAMPED depth: on a column the
        # model would carve away entirely, the clamped value is flat and carries no gradient back.
        deep = trained >= 40
        np.testing.assert_array_equal(served[~deep], trained[~deep])
        self.assertTrue((served[deep] == 39).all())
        self.assertGreater(int((~deep).sum()), 100, "the agreement must be tested on real columns")

    def test_the_pre_registered_decode_is_the_median(self):
        self.assertEqual(SLOPE_DECODE_QUANTILE, 0.5)


# ==================================================================================================
# #6 -- the program arm: the label decomposition, and the compiler that is its inverse
# ==================================================================================================

def _fit(fp, target, extent, ops_allowed=("Layer", "Ramp")):
    """Fit a program to a synthetic surface with the vocabulary #6's labels are drawn from."""
    return fit_program(fp, 0, extent - 1, target, max_ops=K_OPS, ops_allowed=ops_allowed)


class TestProgramToSlots(unittest.TestCase):
    """🔑 #6's supervision. A fitted program is a *sequence* of region operations; the generator
    predicts a *fixed set* of typed slots plus one assignment per column. This decomposition is the
    bridge, and it is only supervision at all if it is **lossless**: whatever the fitter found must
    come back out of the slots exactly, or the arm is being trained towards a building the fitter
    never fitted.

    The lossless-ness is not an accident of the vocabulary, it is a property of it. Every operation
    only ever *lowers* the height map, so the final height of a column is the value written by
    whichever operation last touched it -- and recording that owner per column is enough to replay
    the whole cascade in one pass. That is why the generator can predict a set where the fitter
    searched a sequence.
    """

    E = 40

    def _plan(self):
        fp = np.zeros((RES, RES), bool)
        fp[16:48, 12:52] = True
        return fp

    def _surface(self, h):
        fp = self._plan()
        return fp, np.where(fp, np.clip(np.rint(h), 1, self.E), 0).astype(np.int16)

    def test_a_single_flat_roof_is_one_layer_slot_owning_its_columns(self):
        fp, target = self._surface(np.full((RES, RES), 30.0))
        ops, fitted = _fit(fp, target, self.E)
        assign, types, planes = program_to_slots(fp, self.E, ops)
        self.assertEqual(int((types >= 0).sum()), 1)
        self.assertEqual(int(types[0]), PROGRAM_TYPES.index("Layer"))
        np.testing.assert_allclose(planes[0], [30.0, 0.0, 0.0], atol=1e-6)
        self.assertTrue((assign[fp] == 0).all())

    def test_a_shed_roof_is_one_ramp_slot_and_it_keeps_its_slope(self):
        """⚠️ The failure this whole arm exists to avoid: #127's plane head learned six horizontal
        terraces because slope had to survive a straight-through gradient. Here the slope is a
        **label**, so it has to be in the decomposition before any network sees it."""
        _, xx = np.mgrid[0:RES, 0:RES]
        fp, target = self._surface(34 - 0.45 * (xx - 12))
        ops, fitted = _fit(fp, target, self.E)
        assign, types, planes = program_to_slots(fp, self.E, ops)
        self.assertEqual(int(types[0]), PROGRAM_TYPES.index("Ramp"))
        self.assertGreater(abs(float(planes[0][1])), 0.2, "the fall line must survive as a number")

    def test_the_slots_replay_the_fitted_height_map_exactly(self):
        """The contract. Anything less and the arm is trained on a target the fitter did not find."""
        zz, xx = np.mgrid[0:RES, 0:RES]
        for name, h in (("gable", 34 - 0.55 * np.abs(xx - 32)),
                        ("shed", 34 - 0.45 * (xx - 12)),
                        ("setback", np.where(xx < 32, 30.0, 20.0)),
                        ("dome", 36 - 0.020 * ((xx - 32) ** 2 + (zz - 32) ** 2))):
            with self.subTest(name):
                fp, target = self._surface(h)
                ops, fitted = _fit(fp, target, self.E)
                assign, types, planes = program_to_slots(fp, self.E, ops)
                np.testing.assert_array_equal(
                    compile_program(assign, types, planes, fp, self.E), fitted)

    def test_slots_are_ordered_by_descending_area_so_the_labels_cannot_permute(self):
        """🔑 #6 asks about canonicalisation. A set head has no natural slot order, so two runs that
        find the same program in a different order would supervise contradictory labels. Sorting by
        owned area is the cheapest canonical form that always exists, and it removes the need for a
        matching loss entirely."""
        zz, xx = np.mgrid[0:RES, 0:RES]
        fp, target = self._surface(np.where(xx < 40, 30.0, 14.0))
        ops, _ = _fit(fp, target, self.E)
        assign, types, planes = program_to_slots(fp, self.E, ops)
        areas = [int((assign == k).sum()) for k in range(K_OPS) if types[k] >= 0]
        self.assertGreater(len(areas), 1, "this plan needs at least two operations")
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_columns_no_operation_touched_are_the_uncarved_class(self):
        """A program that only carves half the plan leaves the rest at the blockout, and that has to
        be a *class* rather than a slot -- otherwise every building spends an operation on doing
        nothing, which is exactly the no-op #127 spent three tickets ruling out."""
        _, xx = np.mgrid[0:RES, 0:RES]
        fp, target = self._surface(np.where(xx < 32, 24.0, float(self.E)))
        ops, _ = _fit(fp, target, self.E)
        assign, types, planes = program_to_slots(fp, self.E, ops)
        untouched = fp & (xx >= 32)
        self.assertTrue((assign[untouched] == K_OPS).all())

    def test_an_empty_program_assigns_every_column_to_the_uncarved_class(self):
        fp = self._plan()
        assign, types, planes = program_to_slots(fp, self.E, [])
        self.assertTrue((assign[fp] == K_OPS).all())
        self.assertTrue((types < 0).all())

    def test_off_the_footprint_is_never_assigned_to_an_operation(self):
        _, xx = np.mgrid[0:RES, 0:RES]
        fp, target = self._surface(34 - 0.45 * (xx - 12))
        ops, _ = _fit(fp, target, self.E)
        assign, _, _ = program_to_slots(fp, self.E, ops)
        self.assertTrue((assign[~fp] == K_OPS).all())


class TestCompileProgram(unittest.TestCase):
    """The output space itself. #127's case rested on `apply_depth` making validity free; #6's rests
    on the compiler making **planarity and jointness** free on top of it. Both claims are properties
    of this function and neither is a property of the corpus, so they are pinned here."""

    E = 40

    def _plan(self):
        fp = np.zeros((RES, RES), bool)
        fp[16:48, 12:52] = True
        return fp

    def test_a_layer_slot_compiles_flat_however_tilted_its_parameters(self):
        """🔑 The typing gate, and the single mechanical difference from #127's plane head. There, a
        plane's slope was free to decay to zero under L1 and it did, from two initialisations. Here
        `Layer` and `Ramp` are a **discrete decision**: a slot typed flat is flat because the
        compiler ignores its slope, and a slot typed `Ramp` cannot quietly become a terrace."""
        fp = self._plan()
        types = np.array([PROGRAM_TYPES.index("Layer"), -1, -1, -1], np.int8)
        planes = np.zeros((K_OPS, 3), np.float32)
        planes[0] = [26.0, 0.9, -0.4]                    # a slope a Layer must not be allowed
        assign = np.where(fp, 0, K_OPS).astype(np.uint8)
        h = compile_program(assign, types, planes, fp, self.E)
        self.assertEqual(len(np.unique(h[fp])), 1)
        self.assertEqual(int(h[fp][0]), 26)

    def test_a_ramp_slot_compiles_the_plane_it_was_given(self):
        fp = self._plan()
        types = np.array([PROGRAM_TYPES.index("Ramp"), -1, -1, -1], np.int8)
        planes = np.zeros((K_OPS, 3), np.float32)
        planes[0] = [30.0, -0.25, 0.0]                   # height = 30 - 0.25*x
        assign = np.where(fp, 0, K_OPS).astype(np.uint8)
        h = compile_program(assign, types, planes, fp, self.E)
        _, xx = np.mgrid[0:RES, 0:RES]
        want = np.clip(np.floor(30.0 - 0.25 * xx), 1, self.E)
        np.testing.assert_array_equal(h[fp], want[fp].astype(np.int16))

    def test_the_uncarved_class_keeps_the_blockout(self):
        fp = self._plan()
        types = np.full(K_OPS, -1, np.int8)
        h = compile_program(np.full((RES, RES), K_OPS, np.uint8), types,
                            np.zeros((K_OPS, 3), np.float32), fp, self.E)
        self.assertTrue((h[fp] == self.E).all())

    def test_it_is_footprint_exact_and_valid_whatever_the_prediction(self):
        """The same total clamp `apply_depth` gives the per-column arm, kept for the program arm:
        a prediction may be wrong but never invalid, so no run can fail for an unrepresentable
        output rather than for a bad one."""
        rng = np.random.default_rng(0)
        fp = self._plan()
        for _ in range(20):
            assign = rng.integers(0, K_OPS + 1, (RES, RES)).astype(np.uint8)
            types = rng.integers(0, len(PROGRAM_TYPES), K_OPS).astype(np.int8)
            planes = rng.normal(0, 400, (K_OPS, 3)).astype(np.float32)   # wildly out of range
            h = compile_program(assign, types, planes, fp, self.E)
            self.assertTrue((h[~fp] == 0).all())
            self.assertTrue((h[fp] >= 1).all())
            self.assertTrue((h[fp] <= self.E).all())

    def test_a_compiled_two_slot_gable_reads_as_two_operations(self):
        """The form metric, run on the compiler's own output: what #127 could not get out of a
        per-column head must be free here, or the representation has bought nothing."""
        fp = self._plan()
        types = np.array([PROGRAM_TYPES.index("Ramp"), PROGRAM_TYPES.index("Ramp"), -1, -1], np.int8)
        planes = np.array([[18.0, 0.5, 0.0], [50.0, -0.5, 0.0], [0, 0, 0], [0, 0, 0]], np.float32)
        _, xx = np.mgrid[0:RES, 0:RES]
        assign = np.where(fp, np.where(xx < 32, 0, 1), K_OPS).astype(np.uint8)
        h = compile_program(assign, types, planes, fp, self.E)
        r = roof_description_length(h, fp, 0, self.E)
        self.assertLessEqual(r["ops"], 2)
        self.assertEqual(r["planar_fraction"], 1.0)


class TestPlaneConventions(unittest.TestCase):
    """The fitter measures a plane in voxels of the 64-grid; the network predicts one in units of
    the building's own height, because a 6-voxel and a 60-voxel building must not be asked to
    regress the same number (`mean_relative_depth` makes the same argument for the same reason).
    The conversion between them is where a silent factor would hide."""

    def test_the_conversion_round_trips(self):
        rng = np.random.default_rng(1)
        for extent in (6, 23, 60):
            for _ in range(20):
                vox = rng.normal(0, 30, 3).astype(np.float64)
                vox[0] = abs(vox[0])
                back = plane_to_voxel(plane_to_normalised(vox, extent), extent)
                np.testing.assert_allclose(back, vox, atol=1e-4)

    def test_a_flat_plane_is_flat_in_both_conventions(self):
        n = plane_to_normalised(np.array([30.0, 0.0, 0.0]), 40)
        self.assertAlmostEqual(float(n[1]), 0.0, places=6)
        self.assertAlmostEqual(float(n[2]), 0.0, places=6)
        self.assertAlmostEqual(float(n[0]), 30.0 / 40, places=6)

    def test_the_normalised_height_is_scale_free(self):
        """The same roof on a tall and a short building is the same label."""
        a = plane_to_normalised(np.array([20.0, 0.0, 0.0]), 40)
        b = plane_to_normalised(np.array([10.0, 0.0, 0.0]), 20)
        np.testing.assert_allclose(a, b, atol=1e-6)


class TestReplayCommutes(unittest.TestCase):
    """🔑🔑 #4: the height-map compiler must agree with the SDF compiler on ORDER, not just on
    geometry.

    `EditableBuilding` composes every layer-program operation with `sdf_subtract`, and subtracting
    A then B is subtracting their union -- so the SDF path is commutative by construction.
    `replay_program` was not: it applied `Layer` as a SET (`where(region, v, h)`), which can RAISE a
    column an earlier operation had already lowered.

    Measured on 250 recovered programs before this test was written: 78% have two operations whose
    regions overlap, and a permutation changed the compiled building on **69.6%** of them. Reading
    `Layer` as a MIN changed the result on **0 of 250**, and left **0 of 2,000** permutations
    changed. So the two compilers agreed only because #10's fitter never emits an operation that
    would raise a column -- a property of the *search*, which a hand-authored or generated program
    is under no obligation to have.
    """

    E = 20

    def _case(self):
        from scene.sdf_edit import mask_to_rings
        fp = np.zeros((RES, RES), bool)
        fp[8:28, 8:32] = True
        a = np.zeros((RES, RES), bool); a[10:24, 10:28] = True; a &= fp
        b = np.zeros((RES, RES), bool); b[16:28, 8:24] = True; b &= fp
        self.assertTrue((a & b).any(), "the fixture must exercise OVERLAPPING regions")
        prog = [dict(op="Layer", height=14, region=[r.tolist() for r in mask_to_rings(a)]),
                dict(op="Layer", height=6, region=[r.tolist() for r in mask_to_rings(b)])]
        return fp, prog

    def test_a_permutation_does_not_change_the_replayed_building(self):
        fp, prog = self._case()
        first = replay_program(fp, 0, self.E - 1, prog)
        second = replay_program(fp, 0, self.E - 1, list(reversed(prog)))
        np.testing.assert_array_equal(first, second)

    def test_no_operation_ever_raises_a_column(self):
        """The monotone-descent invariant the whole algebra rests on, checked step by step rather
        than assumed from the fitter's good behaviour."""
        fp, prog = self._case()
        h = np.where(fp, np.int16(self.E), 0).astype(np.int16)
        for k in range(len(prog)):
            nxt = replay_program(fp, 0, self.E - 1, prog[:k + 1])
            self.assertFalse((nxt[fp] > h[fp]).any(), f"operation {k} raised a column")
            h = nxt

    def test_the_deeper_layer_wins_wherever_they_overlap(self):
        """What `min` means here, stated as geometry: the overlap takes the LOWER of the two."""
        fp, prog = self._case()
        h = replay_program(fp, 0, self.E - 1, prog)
        from scene.sdf_edit import mask_to_rings                       # noqa: F401
        from scripts.foundations.recover_massing_programs import _rings_to_mask
        a = _rings_to_mask(prog[0]["region"]) & fp
        b = _rings_to_mask(prog[1]["region"]) & fp
        both = a & b
        self.assertTrue(both.any())
        self.assertTrue((h[both] == 6).all(), "the overlap must take the lower of 14 and 6")
        self.assertTrue((h[a & ~b] == 14).all())


class TestProgramAugmentation(unittest.TestCase):
    """The 8 plan symmetries applied to a PROGRAM. Every other arm on #127 trains with them, so this
    one has to as well or its result is confounded by 8x less data.

    ⚠️ The assignment is an image and rotates with the footprint; a plane is not. `height = a + b*x
    + c*z` has to be re-expressed in the rotated frame, and a sign error there would train the arm
    on roofs tilted the wrong way -- silently, because the footprint and the assignment would still
    line up perfectly. So what is compared here is the compiled SURFACE, which is the only thing
    that would show it."""

    E = 40

    def _case(self):
        fp = np.zeros((RES, RES), bool)
        fp[10:40, 18:56] = True
        zz, xx = np.mgrid[0:RES, 0:RES]
        target = np.where(fp, np.clip(np.rint(36 - 0.4 * (xx - 18) - 0.25 * (zz - 10)),
                                      1, self.E), 0).astype(np.int16)
        ops, fitted = _fit(fp, target, self.E)
        return fp, program_to_slots(fp, self.E, ops)

    def test_the_augmented_program_compiles_to_the_augmented_surface(self):
        fp, (assign, types, planes) = self._case()
        base = compile_program(assign, types, planes, fp, self.E)
        for k in range(4):
            for flip in (False, True):
                with self.subTest(k=k, flip=flip):
                    fp2, t2 = _d4(fp, base, k, flip)
                    a2, ty2, p2 = _d4_program(assign, types, planes, k, flip)
                    np.testing.assert_array_equal(
                        compile_program(a2, ty2, p2, fp2, self.E), t2)

    def test_the_identity_symmetry_changes_nothing(self):
        fp, (assign, types, planes) = self._case()
        a2, t2, p2 = _d4_program(assign, types, planes, 0, False)
        np.testing.assert_array_equal(a2, assign)
        np.testing.assert_allclose(p2, planes, atol=1e-6)

    def test_a_flat_slot_stays_flat_under_every_symmetry(self):
        """A `Layer`'s offset is invariant: rotating a flat roof cannot tilt it."""
        types = np.array([PROGRAM_TYPES.index("Layer"), -1, -1, -1], np.int8)
        planes = np.array([[27.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32)
        assign = np.zeros((RES, RES), np.uint8)
        for k in range(4):
            for flip in (False, True):
                _, _, p = _d4_program(assign, types, planes, k, flip)
                np.testing.assert_allclose(p[0], [27.0, 0.0, 0.0], atol=1e-6)


class TestProgramLoss(unittest.TestCase):
    """🔑 #6's training strategy in one function. The arm is supervised on the **program**, never on
    the surface it compiles to -- #127 measured that an L1 on the surface has a flat region as a
    strong optimum and that the plane head fell into it from two initialisations. Only the terms
    that see the labels can put a slope in."""

    def _labels(self, k=K_OPS):
        import torch
        assign = torch.full((2, RES, RES), k, dtype=torch.long)
        assign[:, 16:48, 12:32] = 0
        assign[:, 16:48, 32:52] = 1
        types = torch.tensor([[1, 0, -1, -1], [1, 0, -1, -1]], dtype=torch.long)
        planes = torch.zeros(2, k, 3)
        planes[:, 0] = torch.tensor([0.7, 0.0, -0.3])
        planes[:, 1] = torch.tensor([0.5, 0.0, 0.0])
        return assign, types, planes

    def _perfect_output(self, assign, types, planes, k=K_OPS):
        import torch
        a = torch.zeros(2, k + 1, RES, RES).scatter_(1, assign[:, None], 20.0)
        t = torch.zeros(2, k, len(PROGRAM_TYPES)).scatter_(
            2, types.clamp(min=0)[..., None], 20.0)
        return a, t, planes.clone()

    def test_an_exact_program_costs_nothing(self):
        assign, types, planes = self._labels()
        out = self._perfect_output(assign, types, planes)
        fp = np.zeros((RES, RES), bool)
        fp[16:48, 12:52] = True
        import torch
        m = torch.from_numpy(fp)[None].expand(2, -1, -1)
        loss = program_loss(out, (assign, types, planes), m)
        self.assertLess(float(loss), 1e-3)

    def test_a_wrong_slope_costs_and_the_cost_scales_with_how_flat_it_is(self):
        """🔑 The whole point of the arm. A terraced answer gets every column assigned correctly and
        the slope wrong -- which is exactly #127's plane head, whose failure the form metric only
        caught after 40 epochs had been spent. Here it has to cost *during* training, and cost more
        the flatter it gets, or nothing in the objective prefers a pitch to a step."""
        import torch
        assign, types, planes = self._labels()
        out = self._perfect_output(assign, types, planes)
        m = torch.ones(2, RES, RES, dtype=torch.bool)
        exact = float(program_loss(out, (assign, types, planes), m))

        def cost(frac):
            p = planes.clone()
            p[:, 0, 2] *= frac                               # the ramp flattened towards a terrace
            return float(program_loss((out[0], out[1], p), (assign, types, planes), m)) - exact

        self.assertGreater(cost(0.0), 0.0)
        self.assertGreater(cost(0.0), cost(0.5))
        self.assertAlmostEqual(cost(0.0), 2 * cost(0.5), places=5)

    def test_inactive_slots_do_not_contribute(self):
        """A building needing two operations must not be pushed to invent four."""
        import torch
        assign, types, planes = self._labels()
        out = self._perfect_output(assign, types, planes)
        noise = planes.clone()
        noise[:, 2:] = 7.0                                   # garbage in the unused slots
        loss = program_loss((out[0], out[1], noise), (assign, types, planes),
                            torch.ones(2, RES, RES, dtype=torch.bool))
        self.assertLess(float(loss), 1e-3)


# ==================================================================================================
# #129 -- the plane parameters CLASSIFIED, and the decode that is most of the ticket
# ==================================================================================================

def _shed_program(extent: int = 40):
    """One fitted program over a real shed roof: `(fp, extent, (assign, types, planes))`.

    Shared by the round-trip and the augmentation tests, which need the same non-trivial fit --
    a plane at an arbitrary rotation, so a sign error in either has somewhere to show.
    """
    fp = np.zeros((RES, RES), bool)
    fp[10:40, 18:56] = True
    zz, xx = np.mgrid[0:RES, 0:RES]
    target = np.where(fp, np.clip(np.rint(36 - 0.4 * (xx - 18) - 0.25 * (zz - 10)), 1, extent),
                      0).astype(np.int16)
    ops, _ = _fit(fp, target, extent)
    return fp, extent, program_to_slots(fp, extent, ops)


class TestSlotCentroids(unittest.TestCase):
    """The anchor the offset is measured at. It has to be a function of the ASSIGNMENT alone, so
    that training (label assignment) and inference (predicted assignment) compute it the same way
    from the same code."""

    def test_a_centred_region_is_the_plan_centre(self):
        assign = np.full((RES, RES), K_OPS, np.uint8)
        assign[16:48, 16:48] = 0
        c = slot_centroids(assign, K_OPS)
        np.testing.assert_allclose(c[0], [0.0, 0.0], atol=1e-6)

    def test_an_offset_region_moves_the_anchor_with_it(self):
        assign = np.full((RES, RES), K_OPS, np.uint8)
        assign[0:8, 56:64] = 0                       # top-left in z, far in x
        c = slot_centroids(assign, K_OPS)
        self.assertLess(c[0][0], -0.4)
        self.assertGreater(c[0][1], 0.4)

    def test_a_slot_owning_nothing_falls_back_to_the_plan_centre(self):
        """Total, like everything else the compiler is handed: an untrained head may assign no
        column at all to a slot, and that must not raise."""
        assign = np.full((RES, RES), K_OPS, np.uint8)
        c = slot_centroids(assign, K_OPS)
        np.testing.assert_allclose(c, np.zeros((K_OPS, 2)), atol=1e-9)


class TestPlaneBins(unittest.TestCase):
    """🔑 #129's discretisation. Three quantities, and the reason for each split is the reason the
    decode can work at all:

        offset   the plane's height AT ITS OWN REGION'S CENTROID, in units of the building. Not the
                 height at the plan centre -- a steep plane over an off-centre region extrapolates
                 to a plan-centre height of 4.4 building-heights, and binning that wastes the
                 resolution where the roof actually is.
        pitch    atan of the slope magnitude. Non-negative, and INVARIANT under the mirror that
                 makes the signed slope symmetric, so the symmetry is not in it at all.
        azimuth  the uphill direction. This is where ALL the symmetry went, and it is categorical.
    """

    def _centroid(self, assign=None):
        if assign is None:
            assign = np.full((RES, RES), K_OPS, np.uint8)
            assign[16:48, 12:52] = 0
        return slot_centroids(assign, K_OPS)[0]

    def test_bins_are_in_range_for_any_plane_at_all(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = rng.normal(0, 20, 3)
            b = plane_to_bins(p, self._centroid())
            self.assertEqual(b.shape, (3,))
            self.assertTrue(((b >= 0) & (b < PLANE_BINS)).all(), msg=f"{p} -> {b}")

    def test_a_flat_plane_round_trips_to_exactly_flat(self):
        """🔑 A `Layer`'s label is (h, 0, 0) and the compiler reads only its offset, so the flat bin
        has to decode to EXACTLY zero slope -- otherwise the centroid correction would move the
        offset back by a fraction of a bin and a flat roof would not survive its own encoding."""
        c = self._centroid()
        for h in (0.05, 0.3, 0.5, 0.82, 0.98):
            b = plane_to_bins(np.array([h, 0.0, 0.0]), c)
            p = bins_to_plane(b, c)
            self.assertEqual(float(p[1]), 0.0)
            self.assertEqual(float(p[2]), 0.0)
            self.assertLess(abs(float(p[0]) - h), 1.0 / PLANE_BINS)

    def test_the_offset_bin_is_anchored_at_the_region_not_at_the_plan_centre(self):
        """The property that shrinks the dynamic range: two planes that agree over their own region
        share an offset bin however differently they extrapolate across the rest of the plan."""
        c = self._centroid()
        flat = np.array([0.8, 0.0, 0.0])
        tilted = np.array([0.8, 0.0, 0.0])            # same height at the centroid, big slope
        tilted[1] = 1.5
        tilted[0] = 0.8 - 1.5 * c[0]
        self.assertEqual(int(plane_to_bins(flat, c)[0]), int(plane_to_bins(tilted, c)[0]))

    def test_a_tilted_plane_round_trips_within_the_bin_resolution(self):
        c = self._centroid()
        rng = np.random.default_rng(1)
        for _ in range(200):
            mag = float(rng.uniform(0.05, 2.5))
            ang = float(rng.uniform(0, 2 * np.pi))
            p = np.array([float(rng.uniform(0.1, 0.9)), mag * np.cos(ang), mag * np.sin(ang)])
            p[0] -= p[1] * c[0] + p[2] * c[1]        # so the centroid height is the drawn offset
            q = bins_to_plane(plane_to_bins(p, c), c)
            # the surface, not the parameters: what a bin costs is voxels over the region
            zn, xn = c
            self.assertLess(abs((q[0] + q[1] * zn + q[2] * xn) - (p[0] + p[1] * zn + p[2] * xn)),
                            1.0 / PLANE_BINS)
            self.assertLess(abs(np.hypot(q[1], q[2]) - mag), 0.15 * (1 + mag ** 2))
            d = abs(np.arctan2(q[2], q[1]) - ang) % (2 * np.pi)
            self.assertLess(min(d, 2 * np.pi - d), 2 * np.pi / PLANE_BINS)

    def test_the_binned_label_compiles_to_the_fitted_surface(self):
        """End to end, on a real fit, and through the PRODUCTION round trip (`rebin_planes`) --
        quantising the label must not change what it draws by more than a voxel or two, or the
        ceiling of the classified space is the binning and not the model."""
        fp, e, (assign, types, planes) = _shed_program()
        exact = compile_program(assign, types, planes, fp, e)
        got = compile_program(assign, types, rebin_planes(planes, assign, e), fp, e)
        self.assertLess(float(np.abs(got[fp].astype(float) - exact[fp]).mean()), 2.0)


class TestPlaneDecode(unittest.TestCase):
    """🔑🔑 The ticket. #127's single biggest lever was the decode of exactly such a head, and #129's
    warning is that copying that read here would land back on flat: the posterior median of a
    symmetric bimodal slope is zero. The split into (offset, pitch, azimuth) exists so that each
    quantity gets the read its own posterior deserves, and these tests are the pre-registration of
    which read that is.

        offset   MEDIAN  -- ordinal, and no symmetry: the mirrored roof has the SAME height at its
                            own region's centroid, so the two competing hypotheses agree here.
        pitch    MEDIAN  -- ordinal, non-negative, and invariant under the mirror, so #127's
                            argument applies unchanged: the mode shrinks towards the majority bin.
        azimuth  ARGMAX  -- categorical, and its posterior is antipodally bimodal BY CONSTRUCTION.
                            Any averaging statistic over the circle lands between two roofs, which
                            is #127's mound arriving by a third route.
    """

    def _logits(self, offset=None, pitch=None, azimuth=None):
        lg = np.full((1, 3, PLANE_BINS), -20.0, np.float32)
        for row, spec in enumerate((offset, pitch, azimuth)):
            for b, w in (spec or {0: 1.0}).items():
                lg[0, row, b] = np.log(max(w, 1e-9)) + 10.0
        return lg

    def test_the_azimuth_commits_to_one_mode_and_does_not_average_two(self):
        """The whole ticket in one assertion. Equal mass on two OPPOSITE directions is exactly the
        corpus's own symmetry; a mean or a median returns a direction neither mode holds, and the
        compiled roof is flat or wrong. `argmax` returns one of the two."""
        half = PLANE_BINS // 2
        lg = self._logits(offset={PLANE_BINS // 2: 1.0}, pitch={PLANE_BINS // 2: 1.0},
                          azimuth={8: 1.0, 8 + half: 1.0})
        p = decode_plane_logits(lg, np.zeros((1, 2)))[0]
        got = np.arctan2(p[2], p[1]) % (2 * np.pi)
        want = [(b + 0.5) * 2 * np.pi / PLANE_BINS for b in (8, 8 + half)]
        self.assertTrue(min(abs(got - w) for w in want) < 2 * np.pi / PLANE_BINS,
                        msg=f"azimuth {got:.3f} is neither mode {want}")

    def test_a_bimodal_azimuth_still_decodes_to_a_pitched_plane(self):
        """The failure being avoided, stated as a magnitude: whatever the azimuth read does, the
        decoded plane must still have a slope. #6's arm typed 46% of its slots `Ramp` and drew them
        with a median 0.00-voxel rise."""
        half = PLANE_BINS // 2
        lg = self._logits(offset={PLANE_BINS // 2: 1.0}, pitch={40: 1.0},
                          azimuth={3: 1.0, 3 + half: 1.0})
        p = decode_plane_logits(lg, np.zeros((1, 2)))[0]
        self.assertGreater(float(np.hypot(p[1], p[2])), 0.5)

    def test_the_offset_takes_the_posterior_median_not_the_mode(self):
        """#127's lever, kept where it still applies. A posterior with its mode low and its mass
        high must decode high."""
        lg = self._logits(offset={2: 0.4, 40: 0.3, 44: 0.3}, pitch={0: 1.0}, azimuth={0: 1.0})
        p = decode_plane_logits(lg, np.zeros((1, 2)))[0]
        self.assertGreater(float(p[0]), 0.5)

    def test_the_pitch_takes_a_LOWER_quantile_than_the_mode(self):
        """#132 changed this read, and the reason is geometric rather than statistical.

        `extra` and `missing` are not symmetric in a pitch -- too steep dives below GT over the far
        end of a region and is charged the whole trench, too shallow only leaves surplus -- so the
        pre-registered read sits BELOW the median. On a posterior whose mass is high but whose lower
        quartile is not, that is the difference between a roof and a trench.
        """
        lg = self._logits(offset={PLANE_BINS // 2: 1.0}, pitch={0: 0.4, 40: 0.35, 44: 0.25},
                          azimuth={0: 1.0})
        p = decode_plane_logits(lg, np.zeros((1, 2)))[0]
        med = decode_plane_logits(lg, np.zeros((1, 2)), ("median", "median", "argmax"))[0]
        self.assertLess(float(np.hypot(p[1], p[2])), float(np.hypot(med[1], med[2])),
                        "the pre-registered pitch must read below the median")

    def test_the_pre_registered_pitch_is_the_lower_quartile(self):
        self.assertEqual(PLANE_DECODE, ("median", "q0.25", "argmax"))

    def test_the_decode_of_a_one_hot_posterior_is_the_bin_it_encodes(self):
        """The decode and the encoder are inverses on a confident head, so a perfect classifier
        reaches the quantisation ceiling and nothing else stands between them."""
        c = np.array([[0.1, -0.2]])
        p = np.array([0.62, 0.9, -0.4])
        p[0] -= p[1] * c[0, 0] + p[2] * c[0, 1]
        b = plane_to_bins(p, c[0])
        lg = self._logits(offset={int(b[0]): 1.0}, pitch={int(b[1]): 1.0},
                          azimuth={int(b[2]): 1.0})
        np.testing.assert_allclose(decode_plane_logits(lg, c)[0], bins_to_plane(b, c[0]),
                                   atol=1e-6)

    def test_an_ablation_quantile_reads_the_same_posterior_lower(self):
        """`decode_ablation` sweeps the pitch, so `q0.25` has to be the same cdf read at a lower
        mass -- and never below `q0.5`'s bin, or the sweep would not be measuring what it says."""
        lg = self._logits(offset={PLANE_BINS // 2: 1.0},
                          pitch={4: 0.3, 30: 0.3, 50: 0.4}, azimuth={0: 1.0})
        cen = np.zeros((1, 2))
        lo = np.hypot(*decode_plane_logits(lg, cen, ("median", "q0.25", "argmax"))[0, 1:])
        mid = np.hypot(*decode_plane_logits(lg, cen, ("median", "median", "argmax"))[0, 1:])
        hi = np.hypot(*decode_plane_logits(lg, cen, ("median", "q0.75", "argmax"))[0, 1:])
        self.assertLess(lo, mid)
        self.assertLess(mid, hi)
        # and the pre-registered read IS the low one, which is #132's whole change to this line
        self.assertEqual(np.hypot(*decode_plane_logits(lg, cen, PLANE_DECODE)[0, 1:]), lo)

    def test_median_and_q_half_are_the_same_read(self):
        rng = np.random.default_rng(11)
        lg, cen = rng.normal(0, 3, (K_OPS, 3, PLANE_BINS)), rng.uniform(-.4, .4, (K_OPS, 2))
        np.testing.assert_allclose(
            decode_plane_logits(lg, cen, ("median", "median", "argmax")),
            decode_plane_logits(lg, cen, ("q0.5", "q0.5", "argmax")))

    def test_an_angular_read_is_refused_on_a_quantity_that_is_not_an_angle(self):
        """`circmean` averages ANGLES. Applied to an offset or a pitch it would return a
        plausible-looking bin index with no meaning, so it raises instead."""
        lg, cen = self._logits(), np.zeros((1, 2))
        for q in (0, 1):
            reads = ["median", "median", "argmax"]
            reads[q] = "circmean"
            with self.subTest(quantity=q), self.assertRaises(ValueError):
                decode_plane_logits(lg, cen, tuple(reads))

    def test_an_unknown_read_is_loud(self):
        with self.assertRaises(ValueError):
            decode_plane_logits(self._logits(), np.zeros((1, 2)), ("mean", "median", "argmax"))

    def test_it_is_total_for_an_untrained_head(self):
        rng = np.random.default_rng(3)
        p = decode_plane_logits(rng.normal(0, 5, (K_OPS, 3, PLANE_BINS)),
                                rng.uniform(-0.5, 0.5, (K_OPS, 2)))
        self.assertEqual(p.shape, (K_OPS, 3))
        self.assertTrue(np.isfinite(p).all())


class TestProgramLossClassifiesThePlanes(unittest.TestCase):
    """The same three-term structure as #6, with the one L1 replaced by three cross-entropies. What
    must not change is which slots are supervised at all, and what must change is that a slope now
    has a Bayes act that is not flat."""

    def _case(self):
        """The same two-region, two-slot program `TestProgramLoss` uses -- its labels, re-expressed
        in bins, so the two arms' losses are exercised on one fixture and a change to it cannot
        move one of them without moving the other."""
        import torch
        assign, types, planes = TestProgramLoss()._labels()
        planes[:, 0] = torch.tensor([0.7, 0.0, -0.9])          # a steeper ramp, so a bin is chosen
        cen = np.stack([slot_centroids(assign[i].numpy(), K_OPS) for i in range(2)])
        bins = torch.from_numpy(np.stack([
            np.stack([plane_to_bins(planes[i, k].numpy(), cen[i, k]) for k in range(K_OPS)])
            for i in range(2)])).long()
        return assign, types, bins

    def _perfect(self, assign, types, bins):
        import torch
        a, t, _ = TestProgramLoss()._perfect_output(assign, types, torch.zeros(2, K_OPS, 3))
        p = torch.zeros(2, K_OPS, len(PLANE_QUANTITIES), PLANE_BINS).scatter_(
            3, bins[..., None], 20.0)
        return a, t, p

    def test_an_exact_program_costs_nothing(self):
        import torch
        assign, types, bins = self._case()
        out = self._perfect(assign, types, bins)
        m = torch.ones(2, RES, RES, dtype=torch.bool)
        self.assertLess(float(program_loss(out, (assign, types, bins), m, "class")), 1e-3)

    def test_a_wrong_azimuth_costs(self):
        """🔑 The term that did not exist before. Under the L1 a slope could go to zero for free at
        the population level; here the direction is a class and getting it wrong is a full CE."""
        import torch
        assign, types, bins = self._case()
        a, t, p = self._perfect(assign, types, bins)
        exact = float(program_loss((a, t, p), (assign, types, bins),
                                   torch.ones(2, RES, RES, dtype=torch.bool), "class"))
        p2 = p.clone()
        p2[:, 0, 2] = 0.0
        p2[:, 0, 2, (int(bins[0, 0, 2]) + PLANE_BINS // 2) % PLANE_BINS] = 20.0
        wrong = float(program_loss((a, t, p2), (assign, types, bins),
                                   torch.ones(2, RES, RES, dtype=torch.bool), "class"))
        self.assertGreater(wrong - exact, 1.0)

    def test_a_layers_pitch_and_azimuth_are_not_supervised(self):
        """A flat slot has no direction, and the compiler ignores the one it is handed. Spending
        capacity teaching the head a number nothing reads is #6's own rule, kept."""
        import torch
        assign, types, bins = self._case()
        a, t, p = self._perfect(assign, types, bins)
        p2 = p.clone()
        p2[:, 1, 1:] = 0.0
        p2[:, 1, 1:, 7] = 20.0                       # garbage pitch/azimuth on the Layer slot
        m = torch.ones(2, RES, RES, dtype=torch.bool)
        self.assertLess(float(program_loss((a, t, p2), (assign, types, bins), m, "class")), 1e-3)

    def test_inactive_slots_do_not_contribute(self):
        import torch
        assign, types, bins = self._case()
        a, t, p = self._perfect(assign, types, bins)
        p2 = p.clone()
        p2[:, 2:] = 0.0
        p2[:, 2:, :, 11] = 20.0
        m = torch.ones(2, RES, RES, dtype=torch.bool)
        self.assertLess(float(program_loss((a, t, p2), (assign, types, bins), m, "class")), 1e-3)


class TestAssignmentStats(unittest.TestCase):
    """#132's first question, and it is free: is the assignment head DIFFUSE, or confidently wrong?

    Both #6 and #129 use one slot where the label uses 3.06, and `dl_ops` reads 1.0 because of it.
    This map has turned a decode into the answer twice (#127 argmax -> posterior median, #129
    azimuth argmax over circmean), so the read is checked before the head is blamed. These pin what
    separates the two cases on posteriors whose answer is known by construction.
    """

    K = 4

    def _fp(self, res=16):
        m = np.zeros((res, res), bool)
        m[2:14, 2:14] = True
        return m

    def _label(self, fp):
        """Slot 0 over most of the footprint, slot 1 over a small corner: the area canonicalisation."""
        a = np.full(fp.shape, self.K, np.uint8)
        a[fp] = 0
        a[3:6, 3:6] = 1
        return a

    def _logits(self, fp, label, p_true, p_zero):
        """`p_zero` on slot 0 everywhere, `p_true` on the label's own slot, the rest shared out."""
        z, x = fp.shape
        p = np.zeros((self.K + 1, z, x))
        fixed = np.zeros((self.K + 1, z, x), bool)
        p[0], fixed[0] = p_zero, True
        for k in range(1, self.K + 1):
            sel = label == k
            p[k][sel], fixed[k][sel] = p_true, True
        left = np.clip(1.0 - (p * fixed).sum(0), 1e-9, None)
        free = np.maximum((~fixed).sum(0), 1)
        for k in range(self.K + 1):
            p[k] = np.where(fixed[k], p[k], left / free)
        p /= p.sum(0, keepdims=True)
        return np.log(p)

    def test_a_confident_head_that_is_simply_wrong(self):
        fp = self._fp()
        lab = self._label(fp)
        s = assignment_stats(self._logits(fp, lab, p_true=0.001, p_zero=0.99), lab, fp, self.K)
        self.assertGreater(s["confidence"], 0.9, "the head is sure")
        self.assertLess(s["entropy_norm"], 0.15)
        self.assertLess(s["p_true_minor"], 0.05, "and sure of the WRONG slot")
        self.assertEqual(s["recall_minor"], 0.0)
        self.assertEqual(s["slots_argmax"], 1)
        self.assertEqual(s["slots_label"], 2)

    def test_a_diffuse_head_that_narrowly_loses(self):
        fp = self._fp()
        lab = self._label(fp)
        s = assignment_stats(self._logits(fp, lab, p_true=0.30, p_zero=0.34), lab, fp, self.K)
        self.assertLess(s["confidence"], 0.6, "no class is winning by much")
        self.assertGreater(s["entropy_norm"], 0.5)
        self.assertGreater(s["p_true_minor"], 0.2, "the head DOES know, it just loses")
        self.assertLess(s["p_won_minor"] - s["p_true_minor"], 0.15, "and loses narrowly")
        self.assertEqual(s["slots_argmax"], 1, "argmax still collapses it to one slot")

    def test_the_balanced_read_recovers_a_slot_the_argmax_loses(self):
        """🔑 The mechanism: slots are canonicalised by AREA, so slot 0 owns most columns.

        Dividing the posterior by the slot's own prior is the standard correction for exactly that
        imbalance, and it is the difference between "the head cannot see slot 1" and "the argmax
        cannot hear it".
        """
        fp = self._fp()
        lab = self._label(fp)
        s = assignment_stats(self._logits(fp, lab, p_true=0.30, p_zero=0.34), lab, fp, self.K)
        self.assertEqual(s["slots_argmax"], 1)
        self.assertEqual(s["slots_balanced"], 2, "the prior correction hears slot 1")
        self.assertGreater(s["recall_minor_balanced"], s["recall_minor"])

    def test_it_reads_only_inside_the_footprint(self):
        fp = self._fp()
        lab = self._label(fp)
        lg = self._logits(fp, lab, p_true=0.001, p_zero=0.99)
        noise = lg.copy()
        rng = np.random.default_rng(0)
        noise[:, ~fp] = rng.normal(size=(self.K + 1, int((~fp).sum())))
        a, b = assignment_stats(lg, lab, fp, self.K), assignment_stats(noise, lab, fp, self.K)
        for k in ("confidence", "p_true_minor", "recall_minor", "slots_argmax"):
            self.assertAlmostEqual(a[k], b[k], places=9, msg=k)


class TestAssignmentDecode(unittest.TestCase):
    """#132's assignment read, and the post-hoc correction this ticket REFUTED before running.

    The diagnosis (`assignment_collapse` on both #129 checkpoints) says the head is DIFFUSE rather
    than confidently wrong -- confidence 0.43, normalised entropy 0.80 -- so a decode looked like
    the fix. It is not: dividing by the model's own marginal buys minor-slot recall 0.0000 -> 0.2829
    and pays for it with per-column accuracy 0.4245 -> 0.2203 and the DOMINANT slot 0.8251 ->
    0.1275. It mostly relabels the building. So the served read stays the plain argmax and the
    correction moves into the LOSS, where it can change what is learned instead of reshuffling what
    was not. `balanced` stays reachable because a refutation that cannot be re-run is an anecdote.
    """

    K = 4

    def _fp(self, res=16):
        m = np.zeros((res, res), bool)
        m[2:14, 2:14] = True
        return m

    def _posterior(self):
        p = np.full((self.K + 1, 16, 16), 0.02)
        p[0] = 0.55                                   # slot 0 owns the plan: the area canonicalisation
        p[1, 3:6, 3:6] = 0.40                         # slot 1 is known, and still loses everywhere
        p /= p.sum(0, keepdims=True)
        return np.log(p)

    def test_the_pre_registered_read_is_the_plain_argmax(self):
        self.assertEqual(ASSIGN_DECODE, "argmax")

    def test_the_balanced_read_does_hear_the_slot_the_argmax_never_wins(self):
        """The half of the refuted correction that WORKED, pinned so the refutation is specific."""
        fp, lg = self._fp(), self._posterior()
        arg = decode_assignment(lg, fp, read="argmax")
        bal = decode_assignment(lg, fp, read="balanced")
        self.assertEqual(set(np.unique(arg[fp]).tolist()), {0}, "the argmax hears one slot")
        np.testing.assert_array_equal(bal[3:6, 3:6], 1, "the balanced read hears slot 1")
        self.assertGreater(len(set(np.unique(bal[fp]).tolist())), 1)

    def test_and_the_half_that_did_not_it_neutralises_a_flat_class(self):
        """⚠️ Why it fails on the corpus: a class that is FLAT over the plan has p/prior == 1 by
        construction, so the dominant slot lands in a tie with every other flat class and the
        argmax between them is decided by noise. That is the 0.8251 -> 0.1275 on real weights."""
        fp = self._fp()
        flat = np.log(np.full((self.K + 1, 16, 16), 1.0 / (self.K + 1)))
        p = np.exp(flat)
        ratio = p[:, fp] / p[:, fp].mean(axis=1)[:, None]
        np.testing.assert_allclose(ratio, 1.0, atol=1e-12)

    def test_the_marginal_is_taken_over_the_footprint_only(self):
        fp, lg = self._fp(), self._posterior()
        loud = lg.copy()
        loud[2, ~fp] = 20.0                           # a class that only ever fires OUTSIDE
        np.testing.assert_array_equal(decode_assignment(lg, fp, read="balanced")[fp],
                                      decode_assignment(loud, fp, read="balanced")[fp])

    def test_it_returns_the_dtype_the_compiler_expects(self):
        a = decode_assignment(np.zeros((self.K + 1, 16, 16)), self._fp())
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(a.shape, (16, 16))

    def test_an_empty_footprint_falls_back_to_the_argmax(self):
        lg = np.random.default_rng(1).normal(size=(self.K + 1, 16, 16))
        np.testing.assert_array_equal(decode_assignment(lg, np.zeros((16, 16), bool),
                                                        read="balanced"),
                                      lg.argmax(0).astype(np.uint8))


class TestLogitAdjustedAssignmentLoss(unittest.TestCase):
    """#132's ONE training change: the assignment cross-entropy is logit-adjusted by the label prior.

    Slots are canonicalised by AREA (#6), so slot 0 owns most columns and the per-column
    cross-entropy is imbalanced **by construction of the label**. Adding `tau * log(prior)` to the
    logits during training is the standard adjustment for that, and unlike the post-hoc division it
    changes what the model learns rather than rescaling what it did not. Inference stays a plain
    argmax, which is the pairing that makes the trained scores the balanced ones.
    """

    def _out(self, logits, k=2):
        import torch
        b = logits.shape[0]
        return (logits, torch.zeros(b, k, len(PROGRAM_TYPES)), torch.zeros(b, k, 3))

    def _labels(self, assign, k=2):
        import torch
        return (assign, torch.full((assign.shape[0], k), -1, dtype=torch.long),
                torch.zeros(assign.shape[0], k, 3))

    def test_a_uniform_prior_changes_nothing(self):
        import torch
        rng = np.random.default_rng(0)
        lg = torch.tensor(rng.normal(size=(2, 3, 8, 8)), dtype=torch.float32)
        a = torch.tensor(rng.integers(0, 3, (2, 8, 8)))
        m = torch.ones(2, 8, 8, dtype=torch.bool)
        plain = float(program_loss(self._out(lg), self._labels(a), m))
        flat = torch.full((3,), 1.0 / 3.0)
        adj = float(program_loss(self._out(lg), self._labels(a), m, assign_prior=flat))
        self.assertAlmostEqual(plain, adj, places=5)

    def test_a_skewed_prior_penalises_predicting_the_majority(self):
        """The whole point: the same logits cost MORE when they just reproduce the prior."""
        import torch
        lg = torch.zeros(1, 3, 8, 8)
        lg[:, 0] = 4.0                                   # confidently predict the majority slot
        a = torch.zeros(1, 8, 8, dtype=torch.long)
        a[:, :2] = 1                                     # a sixth of the columns are the rare slot
        m = torch.ones(1, 8, 8, dtype=torch.bool)
        prior = torch.tensor([0.75, 0.25, 1e-6])
        plain = float(program_loss(self._out(lg), self._labels(a), m))
        adj = float(program_loss(self._out(lg), self._labels(a), m, assign_prior=prior))
        self.assertGreater(adj, plain)

    def test_the_adjustment_is_pre_registered_at_one(self):
        self.assertEqual(ASSIGN_TEMPERATURE, 1.0)

    def test_the_prior_is_the_label_frequency_over_footprint_columns(self):
        fp = np.zeros((2, 8, 8), bool)
        fp[:, 2:6, 2:6] = True
        assign = np.full((2, 8, 8), 2, np.int64)
        assign[:, 2:6, 2:4] = 0                          # half the footprint is slot 0
        assign[:, 2:6, 4:6] = 1                          # half is slot 1
        pr = assignment_prior(assign, fp, k_ops=2)
        np.testing.assert_allclose(pr, [0.5, 0.5, 0.0], atol=1e-9)
        self.assertAlmostEqual(float(pr.sum()), 1.0, places=9)


class TestClassPlaneAugmentation(unittest.TestCase):
    """The binned label has to rotate with the plan exactly as the continuous one does, or the arm
    trains on roofs pitched the wrong way -- and the assignment would still line up perfectly, so
    only the compiled surface would show it."""

    def test_the_binned_program_survives_every_plan_symmetry(self):
        fp, e, (assign, types, planes) = _shed_program()

        def binned_surface(a, t, pl, plan):
            return compile_program(a, t, rebin_planes(pl, a, e), plan, e)

        base = binned_surface(assign, types, planes, fp)
        for k in range(4):
            for flip in (False, True):
                with self.subTest(k=k, flip=flip):
                    fp2, want = _d4(fp, base, k, flip)
                    a2, t2, p2 = _d4_program(assign, types, planes, k, flip)
                    got = binned_surface(a2, t2, p2, fp2)
                    self.assertLess(float(np.abs(got[fp2].astype(float)
                                                 - want[fp2].astype(float)).max()), 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
