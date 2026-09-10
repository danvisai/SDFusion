"""Contract tests for #154 -- the void-semantic annotation set's sampling/schema/agreement logic.

Synthetic and fast throughout. This does NOT test `recover_massing_programs.py`'s own fitter or
`carving_trace.py`'s own renderer -- both are #10/#147's, already covered by their own test files;
this only tests the NEW logic #154 adds around them.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_void_semantic_sample.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import mask_to_rings  # noqa: E402
from scripts.foundations.void_semantic_sample import (  # noqa: E402
    LABELS, BUCKET_ORDER, build_annotation_schema, build_operations, cohens_kappa,
    compute_agreement, op_count_bucket, stratified_sample,
)

RES = 64


def _rect(res, z0, z1, x0, x1):
    m = np.zeros((res, res), bool)
    m[z0:z1, x0:x1] = True
    return m


def _layer_entry(mask, height):
    return dict(op="Layer", height=int(height), area=int(mask.sum()), components=1,
               region=[r.tolist() for r in mask_to_rings(mask)])


# ==================================================================================================
# op_count_bucket
# ==================================================================================================


class TestOpCountBucket(unittest.TestCase):
    def test_zero_and_one_are_low(self):
        self.assertEqual(op_count_bucket(0), "low")
        self.assertEqual(op_count_bucket(1), "low")

    def test_two_is_mid(self):
        self.assertEqual(op_count_bucket(2), "mid")

    def test_three_and_up_are_high(self):
        self.assertEqual(op_count_bucket(3), "high")
        self.assertEqual(op_count_bucket(4), "high")
        self.assertEqual(op_count_bucket(10), "high")

    def test_bucket_order_names_every_bucket_op_count_bucket_can_return(self):
        seen = {op_count_bucket(n) for n in range(0, 8)}
        self.assertEqual(seen, set(BUCKET_ORDER))


# ==================================================================================================
# stratified_sample
# ==================================================================================================


class TestStratifiedSample(unittest.TestCase):
    def _candidates(self, per_cell=10):
        out = []
        bid = 0
        for region in ("NL", "DE", "JP"):
            for carve in (True, False):
                for bucket, n_ops in (("low", 1), ("mid", 2), ("high", 4)):
                    for _ in range(per_cell):
                        out.append(dict(id=bid, region=region, carve_needing=carve,
                                        bucket=bucket, n_ops=n_ops, program=[]))
                        bid += 1
        return out

    def test_returns_close_to_n_and_never_more_than_asked(self):
        cands = self._candidates(per_cell=10)
        sample, _comp = stratified_sample(cands, n=60, seed=1)
        self.assertLessEqual(len(sample), 60)
        self.assertGreaterEqual(len(sample), 55)          # rounding may land a few short, not far

    def test_every_id_is_unique_and_drawn_from_the_candidate_pool(self):
        cands = self._candidates(per_cell=10)
        sample, _comp = stratified_sample(cands, n=60, seed=2)
        ids = [c["id"] for c in sample]
        self.assertEqual(len(ids), len(set(ids)))
        pool_ids = {c["id"] for c in cands}
        self.assertTrue(set(ids).issubset(pool_ids))

    def test_the_high_bucket_is_overrepresented_relative_to_its_population_share(self):
        """18 cells, equal population each -> a plain proportional draw gives 1/18 per cell. The
        high bucket (2 carve-status x 3 regions = 6 of 18 cells) must end up with MORE than its
        naive 6/18 share of the sample."""
        cands = self._candidates(per_cell=10)
        sample, comp = stratified_sample(cands, n=60, seed=3, high_bucket_weight=2.0)
        n_high = sum(v for k, v in comp.items() if k.endswith("/high"))
        naive_share = 60 * (6 / 18)
        self.assertGreater(n_high, naive_share)

    def test_a_cell_with_zero_candidates_contributes_zero_and_does_not_crash(self):
        cands = [c for c in self._candidates(per_cell=10) if c["region"] != "JP"]
        sample, comp = stratified_sample(cands, n=60, seed=4)
        self.assertTrue(all(c["region"] != "JP" for c in sample))

    def test_composition_counts_sum_to_the_sample_size(self):
        cands = self._candidates(per_cell=10)
        sample, comp = stratified_sample(cands, n=60, seed=5)
        self.assertEqual(sum(comp.values()), len(sample))

    def test_deterministic_given_the_same_seed(self):
        cands = self._candidates(per_cell=10)
        a, _ = stratified_sample(cands, n=60, seed=42)
        b, _ = stratified_sample(cands, n=60, seed=42)
        self.assertEqual([c["id"] for c in a], [c["id"] for c in b])


# ==================================================================================================
# composite_trace_step -- one 2x2 tile per operation, for the annotation tool's asset uploads
# ==================================================================================================


class TestCompositeTraceStep(unittest.TestCase):
    def test_tiles_four_views_into_one_2x2_image_at_the_original_pixel_size(self):
        from PIL import Image

        from scripts.foundations.void_semantic_sample import composite_trace_step

        views = [Image.new("RGB", (10, 8), c) for c in
                ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))]
        composite = composite_trace_step(views, pad=2)
        self.assertEqual(composite.size, (2 * 10 + 2, 2 * 8 + 2))
        # each quadrant still shows its own view's own color, untouched by the tiling
        self.assertEqual(composite.getpixel((2, 2)), (255, 0, 0))
        self.assertEqual(composite.getpixel((15, 2)), (0, 255, 0))
        self.assertEqual(composite.getpixel((2, 12)), (0, 0, 255))
        self.assertEqual(composite.getpixel((15, 12)), (255, 255, 0))


# ==================================================================================================
# build_operations -- operation_id must survive a rerun of the SAME recovered program, unlike
# EditOp.id's own default (a fresh uuid4 every call)
# ==================================================================================================


class TestBuildOperations(unittest.TestCase):
    def test_produces_one_op_per_program_entry_with_a_stable_id(self):
        fp = _rect(RES, 10, 40, 10, 40)
        y0, y1 = 0, 20
        program = [_layer_entry(fp, 15)]
        ops = build_operations(fp, y0, y1, program)
        self.assertEqual(len(ops), 1)
        self.assertTrue(ops[0].id)                         # non-empty stable id
        self.assertEqual(ops[0].kind, "layer")

    def test_the_same_program_produces_the_same_operation_id_across_reruns(self):
        """The regression this guards: EditOp.id defaults to a fresh uuid4 per call, so without
        build_operations overwriting it, this would fail nearly always."""
        fp = _rect(RES, 10, 40, 10, 40)
        y0, y1 = 0, 20
        program = [_layer_entry(fp, 15)]
        ops_a = build_operations(fp, y0, y1, program)
        ops_b = build_operations(fp, y0, y1, program)
        self.assertEqual(ops_a[0].id, ops_b[0].id)

    def test_different_operations_in_the_same_program_get_different_ids(self):
        fp = _rect(RES, 10, 40, 10, 40)
        y0, y1 = 0, 20
        program = [_layer_entry(fp, 15), _layer_entry(fp, 25)]
        ops = build_operations(fp, y0, y1, program)
        self.assertEqual(len(ops), 2)
        self.assertNotEqual(ops[0].id, ops[1].id)


# ==================================================================================================
# build_annotation_schema
# ==================================================================================================


class TestBuildAnnotationSchema(unittest.TestCase):
    def test_every_operation_gets_both_annotator_slots_empty(self):
        fp = _rect(RES, 10, 40, 10, 40)
        y0, y1 = 0, 20
        ops = build_operations(fp, y0, y1, [_layer_entry(fp, 15)])
        sample = [dict(id=1, region="NL", carve_needing=True, bucket="low", ops=ops)]
        schema = build_annotation_schema(sample, composition={"NL/carve/low": 1}, trace_dirs={})

        self.assertEqual(schema["schema_version"], 1)
        self.assertEqual(set(schema["labels"]), set(LABELS))
        self.assertEqual(len(schema["operations"]), 1)
        op = schema["operations"][0]
        self.assertEqual(op["building_id"], 1)
        self.assertEqual(op["operation_id"], ops[0].id)
        self.assertIsNone(op["annotator_1"]["label"])
        self.assertIsNone(op["annotator_2"]["label"])
        self.assertIsNone(op["adjudication"]["label"])

    def test_a_building_with_no_ops_key_yet_is_skipped_not_crashed_on(self):
        sample = [dict(id=1, region="NL", carve_needing=True, bucket="low")]   # no "ops"
        schema = build_annotation_schema(sample, composition={}, trace_dirs={})
        self.assertEqual(schema["operations"], [])
        self.assertEqual(schema["sample_buildings"][0]["n_ops"], None)

    def test_sample_buildings_records_every_building_including_zero_op_flat_ones(self):
        """A carve_needing=False building's recovered program is empty by construction -- it
        contributes zero operations. sample_buildings must still list it (with n_ops=0), so that
        drop-off is auditable from the artifact rather than silently invisible in `operations`."""
        fp = _rect(RES, 10, 40, 10, 40)
        y0, y1 = 0, 20
        ops = build_operations(fp, y0, y1, [_layer_entry(fp, 15)])
        sample = [
            dict(id=1, region="NL", carve_needing=True, bucket="low", ops=ops),
            dict(id=2, region="JP", carve_needing=False, bucket="low", ops=[]),
        ]
        schema = build_annotation_schema(sample, composition={}, trace_dirs={})
        self.assertEqual(len(schema["sample_buildings"]), 2)
        by_id = {b["building_id"]: b for b in schema["sample_buildings"]}
        self.assertEqual(by_id[1]["n_ops"], 1)
        self.assertEqual(by_id[2]["n_ops"], 0)
        self.assertEqual(len(schema["operations"]), 1)     # building 2 contributes none


# ==================================================================================================
# cohens_kappa / compute_agreement
# ==================================================================================================


class TestCohensKappa(unittest.TestCase):
    def test_perfect_agreement_on_a_mixed_distribution_is_one(self):
        a = ["wing", "roof_cut", "wing", "ambiguous", "roof_cut"]
        self.assertAlmostEqual(cohens_kappa(a, a), 1.0, places=6)

    def test_no_paired_observations_is_none(self):
        self.assertIsNone(cohens_kappa([], []))
        self.assertIsNone(cohens_kappa([None, None], [None, None]))

    def test_only_none_labels_are_dropped_not_miscounted(self):
        a = ["wing", None, "roof_cut"]
        b = ["wing", "roof_cut", "roof_cut"]
        # pair 2 (index 1) is dropped since a[1] is None; remaining pairs agree fully
        self.assertAlmostEqual(cohens_kappa(a, b), 1.0, places=6)

    def test_chance_level_agreement_on_a_single_constant_category_is_none(self):
        """Every pair agrees on the SAME one category -> kappa's own denominator (1 - pe) is 0;
        reported as None, never silently coerced to 1.0."""
        a = ["wing"] * 5
        b = ["wing"] * 5
        self.assertIsNone(cohens_kappa(a, b))

    def test_systematic_disagreement_is_negative(self):
        a = ["wing", "roof_cut", "wing", "roof_cut"]
        b = ["roof_cut", "wing", "roof_cut", "wing"]
        kappa = cohens_kappa(a, b)
        self.assertIsNotNone(kappa)
        self.assertLess(kappa, 0.0)


class TestComputeAgreement(unittest.TestCase):
    def _op(self, a1=None, a2=None):
        return dict(building_id=1, operation_id="x",
                   annotator_1=dict(label=a1), annotator_2=dict(label=a2))

    def test_reports_true_labeled_count_not_total_operation_count(self):
        schema = dict(operations=[self._op("wing", "wing"), self._op(None, None),
                                  self._op("roof_cut", None)])
        agreement = compute_agreement(schema)
        self.assertEqual(agreement["n_total_operations"], 3)
        self.assertEqual(agreement["n_labeled_by_both"], 1)
        self.assertEqual(agreement["percent_agreement"], 1.0)

    def test_disagreements_are_listed_not_hidden(self):
        schema = dict(operations=[self._op("wing", "roof_cut")])
        agreement = compute_agreement(schema)
        self.assertEqual(len(agreement["disagreements"]), 1)
        self.assertEqual(agreement["percent_agreement"], 0.0)

    def test_zero_labeled_operations_reports_none_not_a_divide_by_zero(self):
        schema = dict(operations=[self._op(None, None)])
        agreement = compute_agreement(schema)
        self.assertIsNone(agreement["percent_agreement"])
        self.assertIsNone(agreement["cohens_kappa"])
        self.assertEqual(agreement["disagreements"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
