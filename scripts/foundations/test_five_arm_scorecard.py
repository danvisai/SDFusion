"""Contract tests for #181 -- the five-arm scorecard's OWN new logic (split materialization,
overlap audit, sampling, post-processing). Synthetic and fast except `TestMaterializeSplit`, which
reads real.h5's own `source_id`/`bag_id` columns (metadata only, no SDF volumes -- fast, no GPU).

The scoring pipeline itself (`predict`/`score_arm`/`summarise`/`verdict`/`fit_decode`) is #127's
own, already covered by `test_train_height_map_generator.py`; nothing here re-tests it.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_five_arm_scorecard.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.five_arm_scorecard import (  # noqa: E402
    UNDERSAMPLED_N, arm_ci, audit_overlap, bootstrap_median_ci, build_scorecard, carve_ids,
    filter_to_cache, materialize_split, sample_ids,
)


def _cache(rows, ok, held):
    return dict(row=np.array(rows, np.int32), ok=np.array(ok, np.uint8),
               held=np.array(held, np.uint8))


class TestFilterToCache(unittest.TestCase):
    def test_ids_absent_from_the_cache_are_dropped(self):
        cache = _cache([1, 2, 3], [1, 1, 1], [0, 0, 0])
        np.testing.assert_array_equal(filter_to_cache([1, 2, 5], cache), [1, 2])

    def test_ids_flagged_not_ok_are_dropped(self):
        cache = _cache([1, 2, 3], [1, 0, 1], [0, 0, 0])
        np.testing.assert_array_equal(filter_to_cache([1, 2, 3], cache), [1, 3])

    def test_result_is_sorted_and_deduplication_safe(self):
        cache = _cache([5, 1, 3], [1, 1, 1], [0, 0, 0])
        np.testing.assert_array_equal(filter_to_cache([3, 5, 1], cache), [1, 3, 5])


class TestSampleIds(unittest.TestCase):
    def test_returns_everything_when_n_exceeds_the_population(self):
        ids = [1, 2, 3]
        np.testing.assert_array_equal(sample_ids(ids, 10), [1, 2, 3])

    def test_samples_exactly_n_without_replacement(self):
        ids = list(range(1000))
        got = sample_ids(ids, 100, seed=0)
        self.assertEqual(len(got), 100)
        self.assertEqual(len(set(got.tolist())), 100)
        self.assertTrue(set(got.tolist()).issubset(set(ids)))

    def test_deterministic_given_the_same_seed(self):
        ids = list(range(500))
        a = sample_ids(ids, 50, seed=7)
        b = sample_ids(ids, 50, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_a_different_seed_gives_a_different_sample(self):
        ids = list(range(500))
        a = sample_ids(ids, 50, seed=1)
        b = sample_ids(ids, 50, seed=2)
        self.assertFalse(np.array_equal(a, b))


class TestAuditOverlap(unittest.TestCase):
    def test_full_overlap_when_test_ids_are_the_entire_old_pool(self):
        n_total = 1000
        n_val = max(1, int(0.02 * n_total))
        perm = np.random.default_rng(0).permutation(n_total)
        bag3d_train = perm[2 * n_val:]
        cache = _cache(list(range(n_total)), [1] * n_total, [0] * n_total)   # nothing old-held-out
        result = audit_overlap(bag3d_train, cache, n_total)
        self.assertEqual(result["n_overlap_bag3d_train"], len(bag3d_train))
        self.assertEqual(result["n_overlap_heightmap_pool"], len(bag3d_train))

    def test_zero_overlap_when_test_ids_are_disjoint_from_the_old_pool(self):
        n_total = 1000
        n_val = max(1, int(0.02 * n_total))
        perm = np.random.default_rng(0).permutation(n_total)
        old_val_and_test = perm[:2 * n_val]                # NOT in bag3d_train
        cache = _cache(list(range(n_total)), [1] * n_total,
                       [1 if i in set(old_val_and_test.tolist()) else 0 for i in range(n_total)])
        result = audit_overlap(old_val_and_test, cache, n_total)
        self.assertEqual(result["n_overlap_bag3d_train"], 0)
        self.assertEqual(result["n_overlap_heightmap_pool"], 0)

    def test_heightmap_pool_overlap_is_never_smaller_than_bag3d_train_overlap(self):
        """`held==0` (the height-map pool) is old-train UNION old-val -- a superset of bag3d_train
        alone -- so its overlap with any test set can only be equal or larger."""
        n_total = 2000
        n_val = max(1, int(0.02 * n_total))
        perm = np.random.default_rng(0).permutation(n_total)          # the real Bag3d formula
        bag3d_test = set(perm[:n_val].tolist())                       # held==1 lives here only
        rng = np.random.default_rng(3)
        some_ids = rng.choice(n_total, size=300, replace=False)
        held = [1 if i in bag3d_test else 0 for i in range(n_total)]
        cache = _cache(list(range(n_total)), [1] * n_total, held)
        result = audit_overlap(some_ids, cache, n_total)
        self.assertGreaterEqual(result["n_overlap_heightmap_pool"], result["n_overlap_bag3d_train"])


class TestCarveIds(unittest.TestCase):
    def test_only_ids_above_the_carve_allowance_are_returned(self):
        per_building = {"blockout": [
            dict(id=1, blockout_extra=0.0),
            dict(id=2, blockout_extra=0.5),
            dict(id=3, blockout_extra=0.02),   # exactly CARVE_NEEDED -- included, `>=`
        ]}
        self.assertEqual(carve_ids(per_building), {2, 3})


class TestBootstrapMedianCi(unittest.TestCase):
    """`extra`/`missing` are reported as MEDIANS everywhere else on this map -- this must report
    the same statistic the plain scoring table beside it uses, not silently switch to the mean."""

    def test_point_estimate_is_the_median_not_the_mean(self):
        skewed = [1, 1, 1, 1, 100]              # median 1, mean 20.8 -- far apart on purpose
        pt, lo, hi = bootstrap_median_ci(skewed, seed=0)
        self.assertAlmostEqual(pt, 1.0, places=6)
        self.assertNotAlmostEqual(pt, float(np.mean(skewed)), places=3)

    def test_ci_brackets_the_point_estimate(self):
        pt, lo, hi = bootstrap_median_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=1)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)


class TestArmCi(unittest.TestCase):
    def test_ci_is_computed_only_over_the_named_ids(self):
        per_building = {"blockout": [
            dict(id=1, extra=0.1), dict(id=2, extra=0.9), dict(id=3, extra=0.2)]}
        pt, lo, hi = arm_ci(per_building, "blockout", {1, 3}, "extra")
        self.assertAlmostEqual(pt, 0.15, places=6)         # median of [0.1, 0.2]
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)

    def test_reports_the_median_not_the_mean_on_a_skewed_id_set(self):
        per_building = {"arm": [dict(id=i, extra=v) for i, v in
                                enumerate([1.0, 1.0, 1.0, 1.0, 100.0])]}
        pt, _lo, _hi = arm_ci(per_building, "arm", {0, 1, 2, 3, 4}, "extra")
        self.assertAlmostEqual(pt, 1.0, places=6)


class TestBuildScorecard(unittest.TestCase):
    def _fake_res(self, extras: dict, form=True):
        """`extras`: {arm_key: [per-building extra, ...]}, all arms sharing the same carve-needing
        ids (1..n)."""
        ids = list(range(1, len(next(iter(extras.values()))) + 1))
        per_building = {"blockout": [dict(id=i, blockout_extra=0.5, extra=0.5) for i in ids]}
        arms = {"blockout": {"carve": dict(missing=0.0, extra=0.5, collapse_rate=0.0)}}
        for key, vals in extras.items():
            per_building[key] = [dict(id=i, extra=v, blockout_extra=0.5) for i, v in zip(ids, vals)]
            summ = dict(missing=0.01, extra=float(np.median(vals)), collapse_rate=0.0,
                       vs_input=0.9)
            if form:
                summ.update(dl_ops=2.0, dl_planar_fraction=0.4)
            arms[key] = {"carve": summ}
        # the real `verdict()` never emits an entry for `NOT_GENERATORS` ("blockout",
        # "nn_retrieval", the sees-GT label) at all -- mirrored here, not just excluded downstream.
        verdict = {}
        for key, vals in extras.items():
            if key in ("blockout", "nn_retrieval"):
                continue
            v = dict(pass_=None)
            v["pass"] = bool(np.median(vals) < 0.5)
            if form:
                v["form_planar_over_bar"] = bool(arms[key]["carve"]["dl_planar_fraction"] >= 0.4)
            verdict[key] = v
        return dict(per_building=per_building, arms=arms, verdict=verdict)

    def test_reports_true_carve_n_and_flags_undersampled(self):
        extras = {k: [0.3] * 5 for k in
                 ("nn_retrieval", "ce_median", "ce_median_fit", "ce_median_fit_flat")}
        res = self._fake_res(extras)
        sc = build_scorecard(res)
        self.assertEqual(sc["n_carve"], 5)
        self.assertTrue(sc["undersampled"])
        self.assertLess(5, UNDERSAMPLED_N)

    def test_every_named_arm_gets_a_row_with_a_ci(self):
        extras = {k: [0.1, 0.2, 0.3, 0.4, 0.5] for k in
                 ("nn_retrieval", "ce_median", "ce_median_fit", "ce_median_fit_flat")}
        res = self._fake_res(extras)
        sc = build_scorecard(res)
        self.assertEqual(set(sc["five_arms"]), {
            "blockout", "1-NN retrieval", "raw height-map generator (#127/#155)",
            "fit_decode (#155)", "fit_decode + #9 coordination bias"})
        for row in sc["five_arms"].values():
            self.assertEqual(len(row["extra_ci"]), 2)
            self.assertLessEqual(row["extra_ci"][0], row["extra"])
            self.assertLessEqual(row["extra"], row["extra_ci"][1])

    def test_not_generators_are_excluded_from_the_axis_verdict(self):
        extras = {k: [0.1] * 5 for k in
                 ("nn_retrieval", "ce_median", "ce_median_fit", "ce_median_fit_flat")}
        res = self._fake_res(extras)
        sc = build_scorecard(res)
        self.assertNotIn("blockout", sc["axis_verdict"])
        self.assertNotIn("1-NN retrieval", sc["axis_verdict"])
        for label in ("raw height-map generator (#127/#155)", "fit_decode (#155)",
                      "fit_decode + #9 coordination bias"):
            self.assertIn(label, sc["axis_verdict"])

    def test_axes_are_reported_separately_never_collapsed(self):
        """#181's own bar: an arm can PASS volume/safety while its form axis reads differently --
        the two must never be folded into one word."""
        extras = {"nn_retrieval": [0.1] * 5, "ce_median": [0.1] * 5,
                 "ce_median_fit": [0.01] * 5,       # beats 1-NN -> volume/safety PASS
                 "ce_median_fit_flat": [0.9] * 5}   # loses to 1-NN -> volume/safety KILL
        res = self._fake_res(extras)
        # force the fit arm's form to read KILL despite passing volume/safety
        res["verdict"]["ce_median_fit"]["form_planar_over_bar"] = False
        sc = build_scorecard(res)
        v = sc["axis_verdict"]["fit_decode (#155)"]
        self.assertEqual(v["volume_safety"], "PASS")
        self.assertEqual(v["architectural_form"], "KILL")

    def test_form_reads_not_measured_when_form_metrics_are_absent(self):
        extras = {k: [0.1] * 5 for k in
                 ("nn_retrieval", "ce_median", "ce_median_fit", "ce_median_fit_flat")}
        res = self._fake_res(extras, form=False)
        sc = build_scorecard(res)
        for v in sc["axis_verdict"].values():
            self.assertEqual(v["architectural_form"], "not measured")

    def test_oracle_references_appear_labeled_and_never_in_the_pass_fail_axis_verdict(self):
        """#181's own acceptance criterion: #130's oracle-fed numbers appear in the same table,
        explicitly labeled oracle-fed/ceiling-only -- and #8's H1a section is explicit that oracle
        rows never enter the pass/fail comparison, so they must never key into `axis_verdict`."""
        extras = {k: [0.1] * 5 for k in
                 ("nn_retrieval", "ce_median", "ce_median_fit", "ce_median_fit_flat")}
        res = self._fake_res(extras)
        sc = build_scorecard(res)
        self.assertIn("oracle_reference", sc)
        self.assertGreater(len(sc["oracle_reference"]), 0)
        for label, row in sc["oracle_reference"].items():
            self.assertIn("population", row, f"{label} must disclose which population it's from")
            self.assertNotIn(label, sc["axis_verdict"])
            self.assertNotIn(label, sc["five_arms"])


class TestMaterializeSplit(unittest.TestCase):
    """The one integration point against real.h5's own metadata columns."""

    def test_runs_end_to_end_and_returns_disjoint_val_test(self):
        split = materialize_split()
        self.assertGreater(len(split["test_ids"]), 0)
        self.assertGreater(len(split["val_ids"]), 0)
        self.assertTrue(set(split["test_ids"].tolist()).isdisjoint(set(split["val_ids"].tolist())))
        self.assertEqual(split["n_total"], 35776)


if __name__ == "__main__":
    unittest.main(verbosity=2)
