"""#188: contract for the vecset retrain cohort -- how many rows per region bucket, and which ones.

Synthetic, fast, no GPU, no corpus. Every test calls the production functions; nothing here
re-implements the quota arithmetic it is checking (the mistake `test_train_vecset_solidity.py`
documents, where a local copy passed while the shipped code was a no-op).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.vecset_cohort import (  # noqa: E402
    BUILDINGWORLD_REGIONS, build_manifest, capped_proportional_quota, load_manifest, row_digest,
    select_cohort, select_gate_rows,
)

# The real post-#171 BuildingWorld bucket sizes, so these tests exercise the actual imbalance
# (Canada 640,020 against Oceania 13,092) rather than a tidy synthetic one.
REAL_SIZES = {3: 456_569, 4: 33_315, 5: 243_932, 6: 640_020, 7: 139_850, 8: 13_092}


class TestQuotaArithmetic(unittest.TestCase):
    def test_hits_the_requested_total_exactly(self):
        for total in (600, 6_000, 60_000, 60_001):
            q = capped_proportional_quota(REAL_SIZES, total)
            self.assertEqual(sum(q.values()), total, f"total {total} was not met exactly")

    def test_covers_every_bucket(self):
        q = capped_proportional_quota(REAL_SIZES, 60_000)
        self.assertEqual(sorted(q), sorted(REAL_SIZES))
        self.assertTrue(all(v > 0 for v in q.values()), f"a bucket got nothing: {q}")

    def test_no_bucket_exceeds_the_cap(self):
        """The point of the cap: Canada and Germany must not swamp the cohort."""
        total, cap_frac = 60_000, 0.25
        q = capped_proportional_quota(REAL_SIZES, total, cap_frac=cap_frac)
        for k, v in q.items():
            self.assertLessEqual(v, int(cap_frac * total) + 1, f"bucket {k} broke the cap: {v}")

    def test_no_bucket_falls_below_the_floor(self):
        """The point of the floor: Oceania (13,092 rows, 0.86% of the corpus) stays representable."""
        total, floor_frac = 60_000, 0.05
        q = capped_proportional_quota(REAL_SIZES, total, floor_frac=floor_frac)
        for k, v in q.items():
            self.assertGreaterEqual(v, int(floor_frac * total) - 1, f"bucket {k} fell below: {v}")

    def test_buckets_between_the_bounds_stay_proportional(self):
        """Clipping the extremes must not flatten the middle -- that would be equal-per-bucket.

        The policy is capped-PROPORTIONAL: a bucket that lands strictly inside [floor, cap] keeps
        its true share. Checked on a milder synthetic corpus, because the real BuildingWorld
        imbalance is severe enough that at n=60,000 only one bucket is left free.
        """
        sizes = {3: 300, 4: 200, 5: 1_000, 6: 20}         # 5 hits the cap, 6 hits the floor
        q = capped_proportional_quota(sizes, 400, cap_frac=0.4, floor_frac=0.05)
        self.assertEqual(q[5], 160, "the largest bucket was not capped")
        self.assertEqual(q[6], 20, "the smallest bucket was not floored")
        # 3 and 4 are both strictly inside the bounds, so their 300:200 ratio must survive intact
        self.assertEqual(q[3] / q[4], 300 / 200)

    def test_never_asks_a_bucket_for_more_rows_than_it_has(self):
        small = {3: 10, 4: 1_000_000, 5: 1_000_000, 6: 1_000_000, 7: 1_000_000}
        q = capped_proportional_quota(small, 30_000)
        self.assertLessEqual(q[3], 10, "asked a 10-row bucket for more than it holds")
        self.assertEqual(sum(q.values()), 30_000)

    def test_a_total_larger_than_the_corpus_is_refused(self):
        with self.assertRaises(ValueError):
            capped_proportional_quota({3: 10, 4: 10}, 100)

    def test_an_infeasible_cap_is_refused_rather_than_silently_exceeded(self):
        """cap_frac * k_buckets < 1 cannot reach the total; say so instead of overfilling."""
        with self.assertRaises(ValueError):
            capped_proportional_quota(REAL_SIZES, 60_000, cap_frac=0.10)  # 6 * 0.10 < 1

    def test_is_deterministic(self):
        a = capped_proportional_quota(REAL_SIZES, 60_000)
        b = capped_proportional_quota(dict(reversed(list(REAL_SIZES.items()))), 60_000)
        self.assertEqual(a, b, "quota depends on dict ordering")

    def test_the_buildingworld_region_ids_are_the_post_171_buckets(self):
        self.assertEqual(sorted(BUILDINGWORLD_REGIONS), [3, 4, 5, 6, 7, 8])


def _corpus(n_per_region=2_000, regions=(3, 4, 5, 6, 7, 8), held_out_every=50):
    """A synthetic ledger: rows, their region, and a ~2% held-out flag."""
    rows, regs, held = [], [], []
    r = 0
    for reg in regions:
        for i in range(n_per_region):
            rows.append(r); regs.append(reg); held.append(1 if i % held_out_every == 0 else 0)
            r += 1
    return np.array(rows, np.int64), np.array(regs, np.int32), np.array(held, np.uint8)


class TestSelection(unittest.TestCase):
    def test_meets_the_quota_per_region(self):
        rows, regs, held = _corpus()
        quota = {3: 100, 4: 200, 5: 50, 6: 300, 7: 10, 8: 40}
        picked = select_cohort(rows, regs, held, quota, salt="188")
        got = {int(r): int((regs[np.isin(rows, picked)] == r).sum()) for r in quota}
        self.assertEqual(got, quota)

    def test_never_selects_a_held_out_row(self):
        """The gate population must stay untrained-on. This is the leakage guard."""
        rows, regs, held = _corpus()
        picked = select_cohort(rows, regs, held, {r: 200 for r in (3, 4, 5, 6, 7, 8)}, salt="188")
        held_rows = set(rows[held == 1].tolist())
        self.assertEqual(held_rows & set(picked.tolist()), set())

    def test_is_reproducible_from_the_salt_alone(self):
        rows, regs, held = _corpus()
        q = {r: 50 for r in (3, 4, 5, 6, 7, 8)}
        a = select_cohort(rows, regs, held, q, salt="188")
        b = select_cohort(rows, regs, held, q, salt="188")
        np.testing.assert_array_equal(a, b)

    def test_a_different_salt_draws_a_different_cohort(self):
        rows, regs, held = _corpus()
        q = {r: 50 for r in (3, 4, 5, 6, 7, 8)}
        a = select_cohort(rows, regs, held, q, salt="188")
        b = select_cohort(rows, regs, held, q, salt="different")
        self.assertNotEqual(a.tolist(), b.tolist())

    def test_does_not_depend_on_input_row_order(self):
        """Selection is a property of the row id, not of where it sits in the ledger.

        ⚠️ Ascending row order tracks the SOURCE CORPUS here -- the trap that voided three headline
        figures once (`_stratified_rows`). A selector that is order-sensitive would reintroduce it.
        """
        rows, regs, held = _corpus()
        q = {r: 40 for r in (3, 4, 5, 6, 7, 8)}
        a = select_cohort(rows, regs, held, q, salt="188")
        perm = np.random.default_rng(0).permutation(len(rows))
        b = select_cohort(rows[perm], regs[perm], held[perm], q, salt="188")
        np.testing.assert_array_equal(np.sort(a), np.sort(b))

    def test_growing_the_cohort_keeps_the_smaller_one_inside_it(self):
        """A later top-up must extend the cohort, not reshuffle it.

        Without this, raising the row count would silently invalidate an encode already paid for
        at ~0.2 s/row, and any checkpoint trained on the earlier draw.
        """
        rows, regs, held = _corpus()
        small = select_cohort(rows, regs, held, {r: 25 for r in (3, 4, 5, 6, 7, 8)}, salt="188")
        big = select_cohort(rows, regs, held, {r: 75 for r in (3, 4, 5, 6, 7, 8)}, salt="188")
        self.assertTrue(set(small.tolist()) <= set(big.tolist()),
                        "the smaller cohort is not a subset of the larger one")

    def test_returns_rows_in_ascending_order(self):
        """HDF5 fancy-indexing needs increasing indices, and a sorted cache is readable."""
        rows, regs, held = _corpus()
        picked = select_cohort(rows, regs, held, {r: 30 for r in (3, 4, 5, 6, 7, 8)}, salt="188")
        np.testing.assert_array_equal(picked, np.sort(picked))

    def test_a_quota_larger_than_the_trainable_rows_is_refused(self):
        rows, regs, held = _corpus(n_per_region=100)
        with self.assertRaises(ValueError):
            select_cohort(rows, regs, held, {3: 100_000}, salt="188")

    def test_selection_ignores_every_signal_but_the_row_id(self):
        """Outcome-blind, per #115's surviving method: nothing about the building may enter.

        Re-running with the same ids but a scrambled region column changes which ROWS are eligible
        for each bucket, yet within one bucket the chosen subset is still exactly the hash-ordered
        prefix -- so no property of the geometry can have leaked in.
        """
        rows, regs, held = _corpus()
        one = np.full_like(regs, 3)
        picked = select_cohort(rows, one, held, {3: 60}, salt="188")
        eligible = rows[held == 0]
        expect = select_cohort(eligible, np.full(len(eligible), 3, np.int32),
                               np.zeros(len(eligible), np.uint8), {3: 60}, salt="188")
        np.testing.assert_array_equal(picked, expect)


class TestGateSelection(unittest.TestCase):
    """The gate draw is the mirror image: held-out rows only, never trainable ones."""

    def test_draws_only_held_out_rows(self):
        rows, regs, held = _corpus()
        picked = select_gate_rows(rows, regs, held, {r: 5 for r in (3, 4, 5, 6, 7, 8)}, salt="188")
        trainable = set(rows[held == 0].tolist())
        self.assertEqual(trainable & set(picked.tolist()), set())

    def test_meets_the_quota_per_region(self):
        rows, regs, held = _corpus()
        quota = {3: 7, 4: 3, 5: 10, 6: 12, 7: 2, 8: 6}
        picked = select_gate_rows(rows, regs, held, quota, salt="188")
        got = {int(r): int((regs[np.isin(rows, picked)] == r).sum()) for r in quota}
        self.assertEqual(got, quota)

    def test_can_never_overlap_the_training_cohort(self):
        """The leakage property the whole two-function split exists to make unmissable."""
        rows, regs, held = _corpus()
        q = {r: 20 for r in (3, 4, 5, 6, 7, 8)}
        train = select_cohort(rows, regs, held, q, salt="188")
        gate = select_gate_rows(rows, regs, held, {r: 5 for r in q}, salt="188")
        self.assertEqual(set(train.tolist()) & set(gate.tolist()), set())

    def test_a_quota_larger_than_the_held_out_population_is_refused(self):
        rows, regs, held = _corpus(n_per_region=100, held_out_every=50)   # 2 held out per region
        with self.assertRaises(ValueError):
            select_gate_rows(rows, regs, held, {3: 50}, salt="188")


class TestManifest(unittest.TestCase):
    """The cohort is a COMMITTED artifact, not a rerun of a selector that might have drifted.

    A retrain that costs ~8 GPU-hours to encode and ~27 to train has to be able to say exactly
    which rows it saw, and a reader has to be able to check that claim without a GPU.
    """

    def setUp(self):
        self.rows, self.regs, self.held = _corpus(n_per_region=3_000)
        self.man = build_manifest(self.rows, self.regs, self.held, train_n=600, gate_n=60,
                                  salt="188")

    def test_names_every_row_of_both_roles(self):
        self.assertEqual(len(self.man["train"]["rows"]), 600)
        self.assertEqual(len(self.man["gate"]["rows"]), 60)
        self.assertEqual(self.man["train"]["n"], 600)
        self.assertEqual(self.man["gate"]["n"], 60)

    def test_roles_are_disjoint(self):
        self.assertEqual(set(self.man["train"]["rows"]) & set(self.man["gate"]["rows"]), set())

    def test_records_what_would_be_needed_to_redraw_it(self):
        for key in ("salt", "cap_frac", "floor_frac"):
            self.assertIn(key, self.man["meta"], f"the manifest cannot be reproduced without {key}")

    def test_round_trips_through_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cohort.json"
            p.write_text(json.dumps(self.man))
            got = load_manifest(p, "train")
            np.testing.assert_array_equal(got, np.array(self.man["train"]["rows"]))

    def test_a_tampered_row_list_is_refused(self):
        """The digest is the point: an edited manifest must not silently retrain on other rows."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cohort.json"
            bad = json.loads(json.dumps(self.man))
            bad["train"]["rows"][0] += 1
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_manifest(p, "train")

    def test_an_unknown_role_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cohort.json"
            p.write_text(json.dumps(self.man))
            with self.assertRaises(ValueError):
                load_manifest(p, "confirmation")

    def test_the_digest_is_order_independent_but_content_sensitive(self):
        a = row_digest([3, 1, 2])
        self.assertEqual(a, row_digest([1, 2, 3]))
        self.assertNotEqual(a, row_digest([1, 2, 4]))

    def test_redrawing_from_the_same_salt_reproduces_it(self):
        again = build_manifest(self.rows, self.regs, self.held, train_n=600, gate_n=60, salt="188")
        self.assertEqual(again["train"]["digest"], self.man["train"]["digest"])
        self.assertEqual(again["gate"]["digest"], self.man["gate"]["digest"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
