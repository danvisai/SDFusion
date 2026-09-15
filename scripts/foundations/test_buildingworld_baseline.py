"""Public contracts for #178: exact retrieval, separate populations, and all-floor verdicts."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations import buildingworld_baseline as baseline
from scripts.foundations.train_height_map_generator import retrieve_nn
from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL


class TestExactRetrieval(unittest.TestCase):
    def test_bounded_bank_keeps_the_first_row_in_cross_batch_ties(self):
        query = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]], bool)
        bank = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]],
                         [[0, 1], [0, 0]], [[0, 0], [1, 0]]], bool)
        np.testing.assert_array_equal(
            retrieve_nn(query, bank, chunk=1, bank_chunk=2), [0, 1])

    def test_duplicate_shortcut_and_search_match_the_original_retrieval(self):
        rng = np.random.default_rng(178)
        bank = rng.random((23, 8, 8)) > 0.6
        bank[20] = bank[3]
        query = np.concatenate((bank[[3, 7]], rng.random((9, 8, 8)) > 0.5))
        expected = retrieve_nn(query, bank)
        self.assertEqual(expected[0], 3)
        np.testing.assert_array_equal(
            baseline.exact_retrieval(query, bank, chunk=3, bank_chunk=5), expected)

    def test_empty_footprint_is_an_error_not_a_silent_bank_filter(self):
        with self.assertRaisesRegex(ValueError, "empty footprint"):
            baseline.exact_retrieval(np.ones((1, 2, 2), bool), np.zeros((1, 2, 2), bool))


class TestPopulationSelection(unittest.TestCase):
    def setUp(self):
        n = FROZEN_SPLIT_N_TOTAL
        # One old train row and one old held-out row, followed by one train and
        # one held-out row in each required city and the excluded Mississauga.
        perm = np.random.default_rng(0).permutation(n)
        cities = (*baseline.BAR_CITIES, "Mississauga")
        self.sources = np.r_[np.zeros(n, np.int32), np.full(2 * len(cities), -1, np.int32)]
        self.keys = np.full(len(self.sources), b"old", dtype="S64")
        for i, city in enumerate(cities):
            self.keys[n + 2 * i:n + 2 * i + 2] = ("bw:" + city.replace(" ", "")).encode()
        self.ledger = dict(row=np.r_[perm[0], perm[int(0.02 * n)], np.arange(n, len(self.sources))],
                           held_out=np.r_[0, 1, np.tile([0, 1], len(cities))].astype(np.uint8))

    def test_full_new_bank_but_only_nine_city_queries(self):
        result = baseline.select_populations(self.ledger, self.sources, self.keys)
        self.assertEqual(len(result["train"]), 11)
        self.assertEqual(len(result["query"]), 9)
        self.assertEqual(set(result["city"]), set(baseline.BAR_CITIES))
        self.assertEqual(result["excluded_cities"], {"Mississauga": 1})
        self.assertFalse(set(result["query"]) & set(result["train"]))
        self.assertTrue(np.all(np.diff(result["train"]) > 0))

    def test_unassigned_buildingworld_row_is_rejected(self):
        self.ledger = {k: v[:-1] for k, v in self.ledger.items()}
        with self.assertRaisesRegex(ValueError, "explicit #177"):
            baseline.select_populations(self.ledger, self.sources, self.keys)

    def test_changed_historical_flag_is_rejected(self):
        self.ledger["held_out"][1] = 0
        with self.assertRaisesRegex(ValueError, "historical"):
            baseline.select_populations(self.ledger, self.sources, self.keys)


def measured_rows():
    records = []
    for i, city in enumerate(baseline.BAR_CITIES):
        records.append(dict(id=100 + i, city=city, family="gable" if i == 0 else "flat",
                            missing=0.0, extra=0.1, vs_input=0.8, dl_ops=2.0,
                            dl_planar_fraction=0.25, carved_cols=0.5, gt_carved_cols=0.5,
                            roof_relief=0.0, roof_curvature=0.0, roof_speckle=0.0,
                            gt_roof_relief=0.0, gt_roof_curvature=0.0, gt_roof_speckle=0.0,
                            dl_residual=0.0, gt_dl_ops=2.0, gt_dl_planar_fraction=0.5,
                            dl_explained=True, fp_iou=1.0, spill=0.0, vol_iou=0.9))
    return records


class TestReportsAndVerdicts(unittest.TestCase):
    def setUp(self):
        self.rows = measured_rows()
        self.report = baseline.population_report(self.rows)
        self.registration = baseline.preregister(self.report)

    def passing_report(self):
        records = copy.deepcopy(self.rows)
        for r in records:
            r.update(extra=0.05, dl_planar_fraction=0.5)
        return baseline.population_report(records)

    def test_full_population_and_pitched_subset_stay_distinct(self):
        self.assertEqual(self.report["overall"]["n"], 9)
        self.assertEqual(self.report["gable_hip"]["n"], 1)
        self.assertEqual(self.report["per_family"]["flat"]["n"], 8)
        self.assertEqual(self.report["per_family"]["hip"], {"n": 0, "status": "empty"})
        json.dumps(self.report, allow_nan=False)

    def test_quotes_each_city_independently_without_rounding(self):
        self.rows[1]["extra"] = 0.0123456789
        registration = baseline.preregister(baseline.population_report(self.rows))
        self.assertEqual(registration["floors"]["city:Boston"]["max_extra"], 0.0123456789)
        self.assertEqual(registration["floors"]["overall"]["max_extra"], 0.1)
        self.assertEqual(registration["status"], "pending_owner_signoff")

    def test_all_floors_must_pass_and_signoff_is_separate(self):
        verdict = baseline.buildingworld_verdict(self.passing_report(), self.registration)
        self.assertTrue(verdict["numeric_pass"])
        self.assertEqual(len(verdict["floors"]), 11)
        self.assertEqual(verdict["registration_status"], "pending_owner_signoff")

    def test_one_failed_city_or_pitched_floor_fails_overall(self):
        for key in ("city", "pitched"):
            report = self.passing_report()
            summary = report["per_city"]["Cambridge"] if key == "city" else report["gable_hip"]
            summary["extra"] = 0.1
            self.assertFalse(baseline.buildingworld_verdict(report, self.registration)["numeric_pass"])

    def test_guard_kill_and_equality_boundaries_cannot_be_bypassed(self):
        for key, value in (("vs_input", 0.98), ("collapse_rate", 0.01),
                           ("dl_ops", 3.0), ("dl_planar_fraction", 0.25), ("extra", 0.1)):
            report = self.passing_report()
            report["overall"][key] = value
            self.assertFalse(baseline.buildingworld_verdict(report, self.registration)["numeric_pass"])

    def test_missing_or_nonfinite_measurement_cannot_pass(self):
        report = self.passing_report()
        report["per_city"].pop("Cambridge")
        self.assertFalse(baseline.buildingworld_verdict(report, self.registration)["numeric_pass"])
        report = self.passing_report()
        report["gable_hip"]["dl_planar_fraction"] = float("nan")
        self.assertFalse(baseline.buildingworld_verdict(report, self.registration)["numeric_pass"])

    def test_missing_required_bar_is_rejected(self):
        self.registration["floors"].pop("gable_hip")
        with self.assertRaisesRegex(ValueError, "eleven"):
            baseline.buildingworld_verdict(self.passing_report(), self.registration)

    def test_zero_extra_marks_infeasibility_without_relaxing_the_rule(self):
        self.rows[1]["extra"] = 0.0
        registration = baseline.preregister(baseline.population_report(self.rows))
        self.assertIn("city:Boston", registration["infeasible_floors"])
        self.assertEqual(registration["floors"]["city:Boston"]["max_extra"], 0.0)

    def test_empty_pitched_population_cannot_calibrate_a_hard_gate(self):
        for r in self.rows:
            r["family"] = "flat"
        with self.assertRaisesRegex(ValueError, "empty population"):
            baseline.preregister(baseline.population_report(self.rows))

    def test_duplicate_row_cannot_reweight_the_aggregate(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            baseline.population_report(self.rows + self.rows[:1])

    def test_signoff_marks_guard_only_floors_without_touching_quoted_numbers(self):
        self.rows[1]["extra"] = 0.0  # city:Boston becomes infeasible
        report = baseline.population_report(self.rows)
        registration = baseline.preregister(report)
        before_bar = copy.deepcopy(registration["floors"]["city:Boston"])
        signed = baseline.apply_signoff(registration, guard_only_floors=["city:Boston"],
                                        note="owner call 2026-09-15")
        self.assertEqual(signed["status"], "signed_off")
        self.assertEqual(signed["guard_only_floors"], ["city:Boston"])
        self.assertEqual(signed["floors"]["city:Boston"], before_bar)
        self.assertEqual(registration["status"], "pending_owner_signoff")  # input untouched

    def test_signoff_rejects_a_floor_outside_infeasible_floors(self):
        with self.assertRaisesRegex(ValueError, "subset of infeasible_floors"):
            baseline.apply_signoff(self.registration, guard_only_floors=["city:Berlin"])

    def test_signoff_cannot_run_twice(self):
        signed = baseline.apply_signoff(self.registration, guard_only_floors=[])
        with self.assertRaisesRegex(ValueError, "already 'signed_off'"):
            baseline.apply_signoff(signed, guard_only_floors=[])

    def test_guard_only_floor_cannot_pass_or_gate_on_its_impossible_clauses(self):
        self.rows[1]["extra"] = 0.0  # city:Boston: max_extra=0.0, beats_extra unreachable
        registration = baseline.preregister(baseline.population_report(self.rows))
        signed = baseline.apply_signoff(registration, guard_only_floors=["city:Boston"])
        report = self.passing_report()  # every city, including Boston, otherwise clears
        verdict = baseline.buildingworld_verdict(report, signed)
        self.assertTrue(verdict["numeric_pass"])
        boston = verdict["floors"]["city:Boston"]
        self.assertTrue(boston["pass_"])
        self.assertFalse(boston["pass_available"])
        self.assertTrue(boston["guard_only"])
        self.assertFalse(boston["beats_extra"])  # reported, but did not block pass_
        other = verdict["floors"]["city:Cape Town"]
        self.assertTrue(other["pass_available"])
        self.assertNotIn("guard_only", other)

    def test_guard_only_floor_still_kills_on_collapse_or_vs_input(self):
        self.rows[1]["extra"] = 0.0
        registration = baseline.preregister(baseline.population_report(self.rows))
        signed = baseline.apply_signoff(registration, guard_only_floors=["city:Boston"])
        report = self.passing_report()
        report["per_city"]["Boston"]["collapse_rate"] = 1.0
        self.assertFalse(baseline.buildingworld_verdict(report, signed)["numeric_pass"])

    def test_historical_control_copies_published_summaries_verbatim(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "control.json"
            historical = dict(meta=dict(n_pinned=714, n_carve=411),
                              arms=dict(nn_retrieval={"carve": {"extra": 0.10310320966625858}},
                                        heightmap_ce_median={"carve": {"extra": 0.06029892111254083}}))
            path.write_text(json.dumps(historical))
            before = path.read_bytes()
            control = baseline.frozen_control(path)
            self.assertEqual(control["nn_retrieval"], historical["arms"]["nn_retrieval"])
            self.assertEqual(control["served_ce_median"], historical["arms"]["heightmap_ce_median"])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
