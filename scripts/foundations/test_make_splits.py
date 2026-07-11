"""Contract tests for the BuildingNet split freezer (ticket 03 / I0.1).

These use a synthetic building universe so they are fast, deterministic, and independent of the
386 GB BuildingNet corpus. Run from the repo root:

  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/foundations/test_make_splits.py
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import make_splits as ms  # noqa: E402


def _synthetic():
    """Imbalanced 4-class universe (like BuildingNet: residential-heavy)."""
    per_class = (("RESIDENTIAL", 80), ("COMMERCIAL", 40), ("PUBLIC", 20), ("RELIGIOUS", 11))
    return [(f"{c}sub_mesh{i:04d}", c) for c, n in per_class for i in range(n)]


class SplitContractTest(unittest.TestCase):
    def setUp(self):
        self.items = _synthetic()
        self.s = ms.make_splits(self.items, seed=0, test_frac=0.15)

    def test_parse_class_strips_subtype(self):
        self.assertEqual(ms.parse_class("RESIDENTIALhouse_mesh1234"), "RESIDENTIAL")
        self.assertEqual(ms.parse_class("COMMERCIALcastle_mesh0365"), "COMMERCIAL")

    def test_deterministic_reproduction(self):
        self.assertEqual(self.s, ms.make_splits(self.items, seed=0, test_frac=0.15))

    def test_seed_changes_the_split(self):
        other = ms.make_splits(self.items, seed=1, test_frac=0.15)
        self.assertNotEqual(set(self.s["test"]), set(other["test"]))

    def test_no_duplicate_ids_in_any_list(self):
        for k, v in self.s.items():
            self.assertEqual(len(v), len(set(v)), f"duplicates in {k}")

    def test_train_fractions_are_nested(self):
        self.assertTrue(set(self.s["train_25"]) <= set(self.s["train_50"]))
        self.assertTrue(set(self.s["train_50"]) <= set(self.s["train_100"]))

    def test_sealed_test_is_disjoint_from_train(self):
        self.assertEqual(set(self.s["test"]) & set(self.s["train_100"]), set())

    def test_full_coverage(self):
        allids = {i for i, _ in self.items}
        self.assertEqual(set(self.s["test"]) | set(self.s["train_100"]), allids)

    def test_fraction_sizes(self):
        n100 = len(self.s["train_100"])
        self.assertAlmostEqual(len(self.s["train_50"]) / n100, 0.5, delta=0.03)
        self.assertAlmostEqual(len(self.s["train_25"]) / n100, 0.25, delta=0.03)

    def test_class_stratified_test_holdout(self):
        test_by = Counter(ms.parse_class(i) for i in self.s["test"])
        all_by = Counter(c for _, c in self.items)
        for cls, tot in all_by.items():
            self.assertAlmostEqual(test_by[cls] / tot, 0.15, delta=0.06,
                                   msg=f"{cls}: {test_by[cls]}/{tot} not ~15%")

    def test_every_class_present_in_each_fraction(self):
        for k in ("train_25", "train_50", "train_100"):
            classes = {ms.parse_class(i) for i in self.s[k]}
            self.assertEqual(classes, {"RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS"}, k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
