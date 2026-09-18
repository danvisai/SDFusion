"""#125: contract tests for the whole-volume A2 voxel-correction feasibility experiment.

Runs entirely on CPU with synthetic geometry. It never touches A2, Dora, or a GPU, and it is not
a replacement for the real preregistered 96-row screen -- see `prototype_voxel_editor.py`'s
module docstring. It exists to prove the plumbing #125 requires: no spatial mask, an absolute
dense occupancy endpoint, cache role isolation, digest verification, and the explicit validity
contract, each independently of the real A2/Dora pipeline.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, REAL_CORPUS_PATH  # noqa: E402
from scripts.foundations.prototype_voxel_editor import (  # noqa: E402
    ADD, IN_CHANNELS, KEEP, REMOVE, RES, S_STAR_VOXELS,
    A2_PINNED_714_EXTRA_OF_RECORD, EXTRA_SEED_NOISE_RANGE, GATE_CONFIRMATION_N,
    GATE_SCREEN_ALTERNATIVE_P1, GATE_SCREEN_N,
    RoleIsolationError, action_weight_map, apply_action_to_source, assert_disjoint_rows,
    assert_disjoint_roles, assert_frozen_a2_checkpoint, assay_sensitivity, build_corrector,
    confirmation_gate, connected_components,
    derive_action, envelope_occupancy, error_stratum, fill_sealed_cavities, futility_boundary,
    ground_connected_ok, hollow_shell_voxels, kill_clauses, min_thickness_survival, model_input,
    paired_win_record, probability_of_early_stop, sealed_volume_metrics, wilson_lower_bound,
    occupancy_from_logits, occupancy_iou, occupancy_to_sdf, open_training_cache,
    recover_surface_control, recover_surface_narrowband, row_content_digest, row_list_digest,
    sanitize_footprint, screening_gate, select_screen_rows, sha256_file, signs_consistent,
    summarize_rows, validity_projection_delta, validity_report, verify_cache_integrity,
    volume_metrics, _filter_non_degenerate, _salted_score,
    _select_train_rows_from_handle, _create_cache, _append_cache,
)


def _copy_frozen_identity_with_placeholder_geometry(path: Path) -> None:
    """A corpus file that PASSES `open_real_corpus`'s frozen-identity gate: real `bag_id`/
    `height_m` (~2.4 MB, mirrors `test_frozen_corpus.py`'s `write_corpus`), but tiny placeholder
    geometry -- these tests only need `VoxelCache`'s source-identity plumbing to resolve, never
    real per-row volumes."""
    with h5py.File(REAL_CORPUS_PATH, "r") as f:
        ids = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
        heights = f["height_m"][:FROZEN_SPLIT_N_TOTAL]
    with h5py.File(path, "w") as f:
        f.create_dataset("bag_id", data=ids)
        f.create_dataset("height_m", data=heights)
        f.create_dataset("sdf", shape=(len(ids), 1, 1, 1), dtype="f4")
        f.create_dataset("footprint", shape=(len(ids), 1, 1), dtype="u1")
        f.create_dataset("source_id", shape=(len(ids),), dtype="i4")


def _box(res=16, margin=3, y0=0, y1=10):
    fp = np.zeros((res, res), bool)
    fp[margin : res - margin, margin : res - margin] = True
    occ = np.zeros((res, res, res), bool)
    occ[:, y0 : y1 + 1, :] = fp[:, None, :]
    return fp, occ


# ---------------------------------------------------------------------------------------------
# Whole-volume eligibility / absolute-state reduction
# ---------------------------------------------------------------------------------------------


class TestAbsoluteStateReduction(unittest.TestCase):
    """#125: an auxiliary KEEP/ADD/REMOVE signal "must reduce immediately to absolute occupancy
    and is never the stored state."""

    def test_action_reduction_is_an_identity_for_arbitrary_pairs(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            source = rng.random((5, 6, 7)) < 0.5
            occ = rng.random((5, 6, 7)) < 0.5
            action = derive_action(source, occ)
            np.testing.assert_array_equal(apply_action_to_source(source, action), occ)

    def test_derive_action_labels(self):
        source = np.array([[[True, False, True]]])
        occ = np.array([[[True, True, False]]])
        action = derive_action(source, occ)
        np.testing.assert_array_equal(action[0, 0], [KEEP, ADD, REMOVE])

    def test_action_weight_map_favours_edits_over_keep(self):
        _, source = _box()
        target = source.copy()
        target[0, 2, 0] = not target[0, 2, 0]  # a single edit voxel
        weight = action_weight_map(source, target, keep_weight=0.12, edit_weight=1.0)
        self.assertAlmostEqual(float(weight[0, 2, 0]), 1.0)
        keep_voxel = tuple(np.argwhere(source == target)[0])
        self.assertAlmostEqual(float(weight[keep_voxel]), 0.12)


class TestWholeVolumeEligibility(unittest.TestCase):
    """The representation must express additions AND removals far from any roof/surface band --
    the exact case the old, now-removed, roof/surface `edit_mask` could not represent."""

    def test_interior_add_and_remove_survive_sanitization_untouched(self):
        fp, source = _box()
        target = source.copy()
        target[7:9, 4:6, 7:9] = True     # interior ADD, deep inside the mass
        target[7, 6, 7] = False          # interior REMOVE, deep inside the mass
        sanitized_target = sanitize_footprint(target, fp)
        # Both edits are inside the footprint, so the shared minimal sanitizer must not undo them.
        np.testing.assert_array_equal(sanitized_target, target)
        self.assertTrue((target != source).any())

    def test_model_input_has_no_target_or_envelope_parameter(self):
        """Isolation by construction: `model_input`'s signature has no parameter through which a
        target, identity label, opportunity stratum, or target-derived mask could reach the
        model."""
        params = set(inspect.signature(model_input).parameters)
        self.assertFalse(params & {"target", "envelope", "identity", "opportunity", "mask"})


# ---------------------------------------------------------------------------------------------
# Baseline parity: the shared sanitizer credits nothing to the learned arm
# ---------------------------------------------------------------------------------------------


class TestBaselineParity(unittest.TestCase):
    def test_sanitizer_is_the_same_function_for_baseline_and_candidate(self):
        fp, source = _box()
        spill = source.copy()
        spill[0, 4, 0] = True  # outside the footprint at (z=0, x=0)
        self.assertFalse(fp[0, 0])
        candidate = spill.copy()  # a "candidate" that only differs from source by the spill fix
        candidate[0, 4, 0] = False
        baseline_clean = sanitize_footprint(spill, fp)
        candidate_clean = sanitize_footprint(candidate, fp)
        # Removing spill earns the candidate no credit: after the SAME sanitizer, they agree.
        np.testing.assert_array_equal(baseline_clean, candidate_clean)

    def test_spill_outside_footprint_is_fully_removed(self):
        fp, source = _box()
        source[0, 4, 0] = True
        cleaned = sanitize_footprint(source, fp)
        self.assertFalse((cleaned & ~np.asarray(fp, bool)[:, None, :]).any())


# ---------------------------------------------------------------------------------------------
# Validity contract
# ---------------------------------------------------------------------------------------------


class TestValidity(unittest.TestCase):
    def test_solid_box_is_valid(self):
        fp, occ = _box()
        report = validity_report(occ, fp)
        self.assertTrue(report["valid"])
        self.assertTrue(report["footprint_exact"])
        self.assertTrue(report["ground_connected"])
        self.assertTrue(report["no_hollow_shell"])
        self.assertTrue(report["min_thickness_ok"])

    def test_floating_fragment_fails_ground_connection(self):
        fp, occ = _box()
        floating = occ.copy()
        floating[:] = False
        floating[6:9, 8:11, 6:9] = True  # a solid block with no contact at y=0
        self.assertFalse(ground_connected_ok(floating))
        self.assertFalse(validity_report(floating, fp)["valid"])

    def test_hollow_shell_is_invalid(self):
        fp, occ = _box(res=16, y0=0, y1=14)
        shell = occ.copy()
        shell[5:11, 5:11, 5:11] = False  # scoop out a sealed interior cavity
        shell[6:10, 6:10, 6:10] = False
        # Re-seal the cavity: the scoop above touched the top (y1=14 is open), so close the roof.
        shell[:, 14, :] = occ[:, 14, :]
        report = validity_report(shell, fp)
        self.assertGreater(hollow_shell_voxels(shell).sum(), 0)
        self.assertFalse(report["no_hollow_shell"])
        self.assertFalse(report["valid"])

    def test_courtyard_open_to_sky_is_admissible(self):
        fp, occ = _box(res=16, y0=0, y1=14)
        courtyard = occ.copy()
        courtyard[6:10, 4:15, 6:10] = False  # a full-height shaft open at the top
        self.assertEqual(int(hollow_shell_voxels(courtyard).sum()), 0)
        self.assertTrue(validity_report(courtyard, fp)["no_hollow_shell"])

    def test_passage_through_the_building_is_admissible(self):
        fp, occ = _box(res=16, margin=2)
        passage = occ.copy()
        passage[6:8, 4:8, :] = False  # a horizontal tunnel exiting both side faces
        self.assertEqual(int(hollow_shell_voxels(passage).sum()), 0)
        report = validity_report(passage, fp)
        self.assertTrue(report["no_hollow_shell"])
        self.assertTrue(report["ground_connected"])

    def test_thin_wall_fails_minimum_thickness(self):
        wafer = np.zeros((16, 16, 16), bool)
        wafer[:, 0:8, 7:8] = True  # a single-voxel-thick wall, s* = 3 requires more
        self.assertLess(min_thickness_survival(wafer, S_STAR_VOXELS), 0.5)
        thick = np.zeros((16, 16, 16), bool)
        thick[:, 0:8, 5:11] = True  # a 6-voxel-thick slab
        self.assertGreaterEqual(min_thickness_survival(thick, S_STAR_VOXELS), 0.5)

    def test_multi_blob_footprint_can_still_be_fully_valid(self):
        """Multiple connected components are allowed -- #125's structural voids requirement --
        as long as every component touches the ground. Each blob is comfortably thicker than s*
        in every axis so the minimum-thickness check isn't incidentally what fails here."""
        res = 24
        fp = np.zeros((res, res), bool)
        fp[3:11, 3:19] = True
        fp[15:23, 3:19] = True  # two separate blobs in plan, each 8 voxels thick
        occ = np.zeros((res, res, res), bool)
        occ[:, 0:10, :] = fp[:, None, :]  # touches y=0 (ground)
        _, n = connected_components(occ)
        self.assertGreaterEqual(n, 2)
        self.assertTrue(validity_report(occ, fp)["valid"])

    def test_projection_delta_is_always_populated_including_when_zero(self):
        fp, occ = _box()
        delta = validity_projection_delta(occ, occ)
        self.assertEqual(delta["projection_changed_voxels"], 0)
        self.assertIn("projection_changed_fraction", delta)
        spill = occ.copy()
        spill[0, 4, 0] = True
        delta2 = validity_projection_delta(spill, sanitize_footprint(spill, fp))
        self.assertEqual(delta2["projection_changed_voxels"], 1)


# ---------------------------------------------------------------------------------------------
# Surface realization
# ---------------------------------------------------------------------------------------------


class TestSurfaceRealization(unittest.TestCase):
    def test_control_is_sign_exact(self):
        _, occ = _box()
        self.assertTrue(signs_consistent(recover_surface_control(occ), occ))

    def test_narrowband_is_sign_exact_even_against_disagreeing_source_field(self):
        _, occ = _box()
        rng = np.random.default_rng(0)
        adversarial_field = occupancy_to_sdf(occ) * -1.0  # every sign flipped
        adversarial_field += rng.normal(0, 0.05, occ.shape).astype(np.float32)
        recovered = recover_surface_narrowband(occ, adversarial_field)
        self.assertTrue(signs_consistent(recovered, occ))

    def test_narrowband_uses_source_evidence_when_signs_agree(self):
        _, occ = _box()
        control = occupancy_to_sdf(occ)
        agreeing_field = control * 1.5  # same sign everywhere, different magnitude
        recovered = recover_surface_narrowband(occ, agreeing_field, band_voxels=3.0)
        self.assertTrue(signs_consistent(recovered, occ))
        self.assertFalse(np.array_equal(recovered, control))  # blending actually did something

    def test_sne_wrapper_requires_nonzero_views(self):
        from scripts.foundations.prototype_voxel_editor import sharp_normal_error_for_arms

        with self.assertRaisesRegex(ValueError, "nonzero views"):
            sharp_normal_error_for_arms({}, device="cpu", views=0)


# ---------------------------------------------------------------------------------------------
# Model conditioning and the identity-bias warm start
# ---------------------------------------------------------------------------------------------


class TestModelConditioning(unittest.TestCase):
    def test_model_input_shape_and_channel_zero_is_signed_source(self):
        fp, occ = _box()
        field = occupancy_to_sdf(occ)
        x = model_input(occ, field, fp, height_m=9.0, y0=2, y1=12, region=1)
        self.assertEqual(x.shape, (IN_CHANNELS,) + occ.shape)
        expected_channel0 = occ.astype(np.float32) * 2.0 - 1.0
        np.testing.assert_array_equal(x[0], expected_channel0)

    def test_untrained_corrector_reduces_to_the_identity_bias_of_source(self):
        """With zero-initialised conv weights, the only nonzero path is `identity_bias * x[:,0]`
        -- i.e. before any training the corrector's logits already encode "keep the source"."""
        import torch

        fp, occ = _box()
        field = occupancy_to_sdf(occ)
        x = model_input(occ, field, fp, height_m=9.0, y0=2, y1=12, region=0)
        model = build_corrector(base=4, identity_bias=3.0)
        with torch.no_grad():
            logits = model(torch.from_numpy(x)[None])[0].numpy()
        expected = 3.0 * x[0]
        np.testing.assert_allclose(logits, expected, atol=1e-5)
        np.testing.assert_array_equal(occupancy_from_logits(logits), occ)


# ---------------------------------------------------------------------------------------------
# Metrics and strata
# ---------------------------------------------------------------------------------------------


class TestMetrics(unittest.TestCase):
    def test_volume_metrics_missing_extra_iou(self):
        target = np.zeros((1, 1, 4), bool)
        target[0, 0, :3] = True
        candidate = np.zeros((1, 1, 4), bool)
        candidate[0, 0, 1:] = True  # misses 1, adds 1, agrees on 2
        m = volume_metrics(candidate, target)
        self.assertAlmostEqual(m["missing"], 1 / 3)
        self.assertAlmostEqual(m["extra"], 1 / 3)
        self.assertAlmostEqual(m["vol_iou"], 2 / 4)

    def test_occupancy_iou_symmetry_and_identity(self):
        _, occ = _box()
        self.assertAlmostEqual(occupancy_iou(occ, occ), 1.0)
        other = occ.copy()
        other[0, 4, 0] = not other[0, 4, 0]
        self.assertAlmostEqual(occupancy_iou(occ, other), occupancy_iou(other, occ))

    def test_error_stratum_boundaries(self):
        bounds = (0.05, 0.20)
        self.assertEqual(error_stratum(0.0, bounds), "small")
        self.assertEqual(error_stratum(0.05, bounds), "small")
        self.assertEqual(error_stratum(0.06, bounds), "medium")
        self.assertEqual(error_stratum(0.20, bounds), "medium")
        self.assertEqual(error_stratum(0.21, bounds), "large")

    def _row(self, row, region, sanitized_iou, predicted_iou, envelope_iou, identity,
            e_stratum="small", a_stratum="small", collapsed=False, valid=True):
        return {
            "row": row, "region": region, "region_name": f"r{region}",
            "source": {"vol_iou": sanitized_iou, "missing": 0.0, "extra": 0.0},
            "sanitized_source": {"vol_iou": sanitized_iou, "missing": 0.0, "extra": 0.0},
            "predicted": {"vol_iou": predicted_iou,
                         "missing": 0.20 if collapsed else 0.0, "extra": 0.0},
            "envelope": {"vol_iou": envelope_iou, "missing": 0.0, "extra": 0.0},
            "vs_input": 0.9, "vs_raw_source": 0.9, "identity_envelope": identity,
            "e_stratum": e_stratum, "a_stratum": a_stratum,
            "collapse_source": False, "collapse_sanitized": False,
            "collapse_predicted": collapsed,
            "spill_raw_source": {}, "spill_raw_predicted": {},
            "footprint_c2_pass_predicted": True, "footprint_c2_pass_source": True,
            "validity_predicted": {"valid": valid}, "validity_sanitized_source": {"valid": True},
            "validity_projection": {"projection_changed_voxels": 0},
        }

    def test_summarize_rows_medians_and_splits(self):
        rows = [
            self._row(0, 0, sanitized_iou=0.80, predicted_iou=0.90, envelope_iou=0.5, identity=False),
            self._row(1, 1, sanitized_iou=0.70, predicted_iou=0.60, envelope_iou=0.5, identity=False),
            self._row(2, 2, sanitized_iou=0.99, predicted_iou=0.99, envelope_iou=0.99, identity=True),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["n_identity_envelope"], 1)
        self.assertEqual(summary["n_opportunity"], 2)
        self.assertAlmostEqual(summary["strict_win_rate_vs_sanitized_source"], 1 / 3)
        self.assertAlmostEqual(summary["strict_win_rate_vs_sanitized_source_opportunity"], 0.5)
        self.assertAlmostEqual(summary["identity_predicted_delta_median"], 0.0)
        self.assertIn("r0", summary["by_region"])
        self.assertEqual(summary["by_region"]["r0"]["n"], 1)

    def test_summarize_rows_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            summarize_rows([])


# ---------------------------------------------------------------------------------------------
# Failure accounting: nothing is dropped or retried
# ---------------------------------------------------------------------------------------------


class TestFailureAccounting(unittest.TestCase):
    def test_invalid_and_collapsed_rows_remain_in_the_summary(self):
        rows = [
            TestMetrics()._row(0, 0, 0.8, 0.9, 0.5, False, collapsed=True, valid=False),
            TestMetrics()._row(1, 1, 0.8, 0.9, 0.5, False, collapsed=False, valid=True),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["n"], 2)  # both rows counted, none dropped
        self.assertEqual(summary["n_predicted_invalid"], 1)
        self.assertGreater(summary["collapse_rate_predicted"], 0.0)


# ---------------------------------------------------------------------------------------------
# Screening and full gates
# ---------------------------------------------------------------------------------------------


class TestHollowInvariantScoring(unittest.TestCase):
    """#118: interior fill must not reach any score.

    A building may be modelled hollow or solid; both are the same architecture seen from outside,
    and ISO 19107 treats an interior shell as a first-class feature (#117). Scoring raw volumes
    charges a solid candidate for every voxel of air inside a shell target -- and #117 measured
    ~4.6% of BuildingWorld (~70,600 rows) arriving as 1-3 voxel skins, Calgary ~97% of them.
    Sealing both sides first also puts this arm on the footing of the floors it is graded against:
    #178's 1-NN bar is computed in height-column space, where no interior can exist at all.
    """

    def _shell(self):
        occ = np.zeros((8, 8, 8), bool)
        occ[1:7, 1:7, 1:7] = True
        solid = occ.copy()
        occ[2:6, 2:6, 2:6] = False          # same outside, hollow inside
        return occ, solid

    def test_a_sealed_cavity_is_filled(self):
        hollow, solid = self._shell()
        np.testing.assert_array_equal(fill_sealed_cavities(hollow), solid)

    def test_a_courtyard_open_to_the_sky_is_left_alone(self):
        occ = np.zeros((8, 8, 8), bool)
        occ[1:7, 1:7, 1:7] = True
        occ[2:6, 1:7, 2:6] = False          # a shaft clear through the roof: still architecture
        np.testing.assert_array_equal(fill_sealed_cavities(occ), occ)

    def test_a_hollow_shell_and_its_solid_twin_score_identically(self):
        hollow, solid = self._shell()
        target = np.zeros((8, 8, 8), bool)
        target[1:7, 1:7, 1:6] = True        # one layer shorter: a real, outside-visible difference
        self.assertEqual(sealed_volume_metrics(hollow, target),
                         sealed_volume_metrics(solid, target))

    def test_a_hollow_target_does_not_charge_a_solid_candidate(self):
        hollow, solid = self._shell()
        self.assertEqual(sealed_volume_metrics(solid, hollow),
                         {"vol_iou": 1.0, "missing": 0.0, "extra": 0.0})
        # ...which the raw comparison the prototype shipped with does not say
        self.assertGreater(volume_metrics(solid, hollow)["extra"], 0.0)

    def test_outside_visible_differences_still_score(self):
        _, solid = self._shell()
        shorter = solid.copy()
        shorter[:, 6, :] = False
        self.assertGreater(sealed_volume_metrics(shorter, solid)["missing"], 0.0)


class TestPairedWinAccounting(unittest.TestCase):
    """#126 requires #118 to name the metric AND how ties are counted. Ties are excluded from
    the rate and published, because pooling them into the denominator is what turned a real
    60% into a 46% 'coin flip' on #126's own 72 rows."""

    def test_ties_are_excluded_from_the_rate_and_reported(self):
        record = paired_win_record([(0.1, 0.2), (0.3, 0.2), (0.2, 0.2)])
        self.assertEqual((record["wins"], record["losses"], record["ties"]), (1, 1, 1))
        self.assertEqual(record["decided"], 2)
        self.assertEqual(record["rate"], 0.5)

    def test_direction_is_explicit(self):
        higher = paired_win_record([(0.3, 0.2)], lower_is_better=False)
        self.assertEqual(higher["wins"], 1)
        self.assertEqual(paired_win_record([(0.3, 0.2)])["wins"], 0)

    def test_no_decided_comparisons_is_not_a_pass(self):
        record = paired_win_record([(0.2, 0.2), (0.2, 0.2)])
        self.assertEqual(record["decided"], 0)
        self.assertEqual(record["rate"], 0.0)
        self.assertEqual(record["wilson_lower_bound"], 0.0)

    def test_wilson_lower_bound_matches_the_published_interval(self):
        self.assertAlmostEqual(wilson_lower_bound(60, 100), 0.517807, places=5)
        self.assertAlmostEqual(wilson_lower_bound(30, 57), 0.418825, places=5)

    def test_the_confirmation_population_needs_12361_of_24464(self):
        """At n=24,464 the statistical clause is nearly free -- the 1-NN floors carry the bar.
        On the legacy 411 the same clause needed 0.5426."""
        self.assertLessEqual(wilson_lower_bound(12360, 24464), 0.50)
        self.assertGreater(wilson_lower_bound(12361, 24464), 0.50)


class TestFutilityBoundary(unittest.TestCase):
    """Simon (1989): a two-stage design stops for futility at <= r1 successes of n1."""

    def test_boundary_matches_the_exact_binomial(self):
        self.assertEqual(futility_boundary(GATE_SCREEN_N, GATE_SCREEN_ALTERNATIVE_P1), 136)
        self.assertEqual(futility_boundary(96, 0.60), 49)

    def test_a_bigger_screen_kills_more_useless_arms(self):
        self.assertGreater(probability_of_early_stop(GATE_SCREEN_N, GATE_SCREEN_ALTERNATIVE_P1),
                           probability_of_early_stop(96, GATE_SCREEN_ALTERNATIVE_P1))
        self.assertGreater(probability_of_early_stop(GATE_SCREEN_N, GATE_SCREEN_ALTERNATIVE_P1),
                           0.9)


class TestAssaySensitivity(unittest.TestCase):
    """ICH E10: a comparison is uninterpretable if the control did not behave as expected."""

    def test_a_reproducing_baseline_is_interpretable(self):
        self.assertTrue(assay_sensitivity(0.0930)["baseline_reproduces_arm_of_record"])

    def test_a_drifted_baseline_voids_the_run(self):
        self.assertFalse(assay_sensitivity(0.2000)["baseline_reproduces_arm_of_record"])
        self.assertFalse(assay_sensitivity(0.0100)["baseline_reproduces_arm_of_record"])

    def test_the_tolerance_is_this_projects_own_measured_seed_range(self):
        self.assertEqual(EXTRA_SEED_NOISE_RANGE, 0.04)
        report = assay_sensitivity(A2_PINNED_714_EXTRA_OF_RECORD)
        self.assertEqual(report["reference"], A2_PINNED_714_EXTRA_OF_RECORD)
        self.assertEqual(report["tolerance"], EXTRA_SEED_NOISE_RANGE)


class TestScreeningGate(unittest.TestCase):
    """#118's stage-1 futility boundary. Both surplus axes, both one-time."""

    def _passing_summary(self):
        return {
            "n": GATE_SCREEN_N, "n_scored_for_wins": 250,
            "win_extra": {"decided": 200, "wins": 140, "rate": 0.70},
            "win_missing": {"decided": 200, "wins": 140, "rate": 0.70},
            "vs_input_median_opportunity": 0.90,
            "collapse_rate_predicted": 0.10, "collapse_rate_source": 0.12,
            "identity_missing_delta_median": 0.0, "identity_extra_delta_median": 0.0,
            "footprint_c2_pass_rate_predicted": 0.80, "footprint_c2_pass_rate_source": 0.77,
            "invalid_rate_predicted": 0.01, "invalid_rate_source": 0.01,
            "human_review_pass": True, "sharp_normal_error_views": 22,
            "sharp_normal_error_narrowband_no_worse": True,
        }

    def test_passing_summary_clears_every_clause(self):
        gate = screening_gate(self._passing_summary())
        self.assertTrue(all(gate.values()), gate)

    def test_each_clause_can_fail_independently(self):
        cases = {
            "screen_n_at_least_250": {"n": 96},
            "extra_wins_clear_futility_boundary":
                {"win_extra": {"decided": 200, "wins": 100, "rate": 0.50}},
            "missing_wins_clear_futility_boundary":
                {"win_missing": {"decided": 200, "wins": 100, "rate": 0.50}},
            "enough_decided_comparisons":
                {"win_extra": {"decided": 124, "wins": 124, "rate": 1.0}},
            "moved_on_opportunity_rows": {"vs_input_median_opportunity": 0.99},
            "no_collapse_regression_vs_source": {"collapse_rate_predicted": 0.30},
            "identity_rows_not_degraded": {"identity_missing_delta_median": 0.05},
            "footprint_no_worse_than_source": {"footprint_c2_pass_rate_predicted": 0.50},
            "validity_no_worse_than_source": {"invalid_rate_predicted": 0.50},
            "human_review_passed": {"human_review_pass": False},
            "sharp_normal_error_measured": {"sharp_normal_error_views": 0},
        }
        for key, override in cases.items():
            with self.subTest(key=key):
                self.assertFalse(screening_gate({**self._passing_summary(), **override})[key])

    def test_ties_are_not_scored_as_failures_to_win(self):
        """The boundary is Simon's on the DECIDED comparisons, not on the cohort size. A cohort
        with ties must not be held to the 250-row boundary, which would make every tie a loss."""
        summary = self._passing_summary()
        # 200 decided of 250: 120 wins clears the 200-row boundary (108) but not the 250-row one.
        record = {"decided": 200, "wins": 120, "rate": 0.60}
        summary["win_extra"] = summary["win_missing"] = record
        gate = screening_gate(summary)
        self.assertTrue(gate["extra_wins_clear_futility_boundary"])
        self.assertTrue(gate["enough_decided_comparisons"])
        self.assertGreater(futility_boundary(GATE_SCREEN_N, GATE_SCREEN_ALTERNATIVE_P1), 120)

    def test_the_win_rate_is_scored_on_opportunity_rows(self):
        """#118 asks for an "opportunity-row win rate", and #125's own gate scoped it that way.
        Identity rows belong to the non-degradation clauses; letting them into the win rate
        would score "correctly left it alone" against the arm on the corpus's no-op majority."""
        def row(row_id, identity, predicted_extra):
            return TestMetrics()._row(row_id, 0, 0.8, 0.9, 0.5, identity) | {
                "predicted": {"vol_iou": 0.9, "missing": 0.0, "extra": predicted_extra},
                "sanitized_source": {"vol_iou": 0.8, "missing": 0.0, "extra": 0.10},
            }
        rows = [row(0, False, 0.05), row(1, False, 0.05)] + [row(i, True, 0.10) for i in range(2, 8)]
        summary = summarize_rows(rows)
        self.assertEqual(summary["n_scored_for_wins"], 2)          # the two opportunity rows
        self.assertEqual(summary["win_extra"]["decided"], 2)
        self.assertEqual(summary["win_extra"]["rate"], 1.0)
        # the six identity ties would have swamped the all-rows record
        self.assertEqual(summary["win_extra_all_rows"]["ties"], 6)

    def test_a_thin_decided_slice_cannot_buy_a_cheap_boundary(self):
        summary = self._passing_summary()
        record = {"decided": 40, "wins": 40, "rate": 1.0}   # clears its own tiny boundary
        summary["win_extra"] = summary["win_missing"] = record
        gate = screening_gate(summary)
        self.assertTrue(gate["extra_wins_clear_futility_boundary"])
        self.assertFalse(gate["enough_decided_comparisons"])   # ...but not half the cohort

    def test_a_skipped_human_review_is_absent_not_a_pass(self):
        summary = self._passing_summary()
        del summary["human_review_pass"]
        self.assertFalse(screening_gate(summary)["human_review_passed"])

    def test_a_skipped_sne_run_is_absent_not_a_pass(self):
        summary = self._passing_summary()
        del summary["sharp_normal_error_views"]
        self.assertFalse(screening_gate(summary)["sharp_normal_error_measured"])


class TestConfirmationGate(unittest.TestCase):
    """#118's stage-2 bar on #178's own 24,464-row population."""

    def _passing(self):
        summary = {
            "n": GATE_CONFIRMATION_N,
            "beats_envelope_extra": {"decided": 20000, "wins": 12000, "rate": 0.60},
            "collapse_rate_predicted": 0.10,
            "envelope_extra_median": 0.12, "predicted_extra_median": 0.05,
            "envelope_missing_median": 0.00, "predicted_missing_median": 0.02,
            "footprint_c2_pass_rate_predicted": 0.80, "footprint_c2_pass_rate_source": 0.77,
            "invalid_rate_predicted": 0.01, "invalid_rate_source": 0.01,
            "human_review_pass": True, "sharp_normal_error_views": 22,
            "sharp_normal_error_narrowband_no_worse": True,
            "assay_sensitivity": {"baseline_reproduces_arm_of_record": True},
        }
        screen = {"collapse_rate_predicted": 0.10, "invalid_rate_predicted": 0.01}
        return summary, screen, {"numeric_pass": True}

    def test_passing_run_clears_every_clause(self):
        gate = confirmation_gate(*self._passing())
        self.assertTrue(all(gate.values()), gate)

    def test_each_clause_can_fail_independently(self):
        cases = {
            "confirmation_n_is_the_178_population": {"n": 714},
            "beats_envelope_over_half_of_decided_rows":
                {"beats_envelope_extra": {"decided": 20000, "wins": 9000, "rate": 0.45}},
            "buildingworld_floors_pass": None,
            "no_collapse_regression_vs_screen": {"collapse_rate_predicted": 0.30},
            "validity_no_worse_than_screen": {"invalid_rate_predicted": 0.40},
            "footprint_no_worse_than_source": {"footprint_c2_pass_rate_predicted": 0.10},
            "human_review_passed": {"human_review_pass": False},
            "sharp_normal_error_measured": {"sharp_normal_error_views": 0},
            "assay_sensitivity_holds":
                {"assay_sensitivity": {"baseline_reproduces_arm_of_record": False}},
        }
        for key, override in cases.items():
            summary, screen, verdict = self._passing()
            if override is None:
                verdict = {"numeric_pass": False}
            else:
                summary.update(override)
            with self.subTest(key=key):
                self.assertFalse(confirmation_gate(summary, screen, verdict)[key])

    def test_the_win_rate_clause_needs_its_lower_bound_above_a_coin_flip(self):
        summary, screen, verdict = self._passing()
        summary["beats_envelope_extra"] = {"decided": 24464, "wins": 12360, "rate": 0.5052}
        self.assertFalse(confirmation_gate(summary, screen, verdict)
                         ["beats_envelope_over_half_of_decided_rows"])
        summary["beats_envelope_extra"] = {"decided": 24464, "wins": 12361, "rate": 0.5053}
        self.assertTrue(confirmation_gate(summary, screen, verdict)
                        ["beats_envelope_over_half_of_decided_rows"])


class TestKillClauses(unittest.TestCase):
    """KILL is a disjunction, so its clauses are point-estimate facts and carry no alpha.
    Three tests at 5% ORed together would give a ~14% false-kill rate (Berger 1982's result
    runs the other way for a union); a KILL must be a plainly visible failure, not a p-value."""

    def test_an_arm_the_envelope_dominates_on_both_axes_is_killed(self):
        kill = kill_clauses({"predicted_extra_median": 0.13, "envelope_extra_median": 0.12,
                             "predicted_missing_median": 0.02, "envelope_missing_median": 0.00,
                             "collapse_rate_predicted": 0.05})
        self.assertTrue(kill["killed_not_transforming"])
        self.assertTrue(any(kill.values()))

    def test_beating_the_envelope_on_one_axis_is_not_a_kill(self):
        kill = kill_clauses({"predicted_extra_median": 0.05, "envelope_extra_median": 0.12,
                             "predicted_missing_median": 0.02, "envelope_missing_median": 0.00,
                             "collapse_rate_predicted": 0.05})
        self.assertFalse(kill["killed_not_transforming"])

    def test_destroying_more_buildings_than_naive_retrieval_is_a_kill(self):
        kill = kill_clauses({"predicted_extra_median": 0.05, "envelope_extra_median": 0.12,
                             "predicted_missing_median": 0.02, "envelope_missing_median": 0.00,
                             "collapse_rate_predicted": 0.20})
        self.assertTrue(kill["killed_collapse_over_1nn"])


# ---------------------------------------------------------------------------------------------
# Salted-hash row selection: fixed, order-independent
# ---------------------------------------------------------------------------------------------


class TestSelection(unittest.TestCase):
    def test_salted_score_is_order_independent(self):
        salt = "test-salt"
        rows = list(range(200))
        scores_a = {r: _salted_score(salt, "screen", r) for r in rows}
        shuffled = rows.copy()
        np.random.default_rng(0).shuffle(shuffled)
        scores_b = {r: _salted_score(salt, "screen", r) for r in shuffled}
        self.assertEqual(scores_a, scores_b)

    def test_salted_score_differs_by_purpose_and_salt(self):
        self.assertNotEqual(_salted_score("s", "train", 5), _salted_score("s", "screen", 5))
        self.assertNotEqual(_salted_score("s1", "train", 5), _salted_score("s2", "train", 5))

    def test_select_screen_rows_is_flat_per_region_and_order_independent(self):
        full_rows = list(range(300))
        region_of = {r: r % 3 for r in full_rows}
        a = select_screen_rows(full_rows, region_of, salt="s")
        shuffled = full_rows.copy()
        np.random.default_rng(1).shuffle(shuffled)
        b = select_screen_rows(shuffled, region_of, salt="s")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 96)
        counts = {}
        for row in a:
            counts[region_of[row]] = counts.get(region_of[row], 0) + 1
        self.assertEqual(counts, {0: 32, 1: 32, 2: 32})

    def test_select_screen_rows_fails_loudly_when_a_region_is_short(self):
        full_rows = list(range(10))  # far fewer than 32 per region
        region_of = {r: r % 3 for r in full_rows}
        with self.assertRaisesRegex(RuntimeError, "needs 32"):
            select_screen_rows(full_rows, region_of, salt="s")

    def test_select_screen_rows_is_a_subset_of_full(self):
        full_rows = list(range(300))
        region_of = {r: r % 3 for r in full_rows}
        screen = select_screen_rows(full_rows, region_of, salt="s")
        self.assertTrue(set(screen) <= set(full_rows))


class TestFilterNonDegenerate(unittest.TestCase):
    def test_degenerate_rows_are_excluded(self):
        empty = np.zeros((4, 4, 4), np.float32) + 1.0   # sdf > 0 everywhere: no target at all
        nonempty = np.full((4, 4, 4), -1.0, np.float32)  # sdf <= 0 everywhere: a full target
        handle = {"sdf": {0: empty, 1: nonempty, 2: empty, 3: nonempty}}
        out = _filter_non_degenerate([0, 1, 2, 3], handle)
        self.assertEqual(out, [1, 3])

    @unittest.skipUnless(REAL_CORPUS_PATH.exists(), "real corpus metadata not available")
    def test_full_population_rows_raises_on_unexpected_count(self):
        """Uses a real, identity-passing (but tiny-geometry) corpus so the call reaches the
        count check rather than failing at `open_real_corpus`'s identity gate first."""
        from scripts.foundations.prototype_voxel_editor import full_population_rows

        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.h5"
            _copy_frozen_identity_with_placeholder_geometry(real)
            latents = Path(tmp) / "latents.h5"
            n = 5
            with h5py.File(latents, "w") as h:
                h.create_dataset("row", data=np.arange(n))
                h.create_dataset("held_out", data=np.ones(n, bool))
                h.create_dataset("region", data=np.array([0, 1, 2, 0, 1]))
                h.create_dataset("height_m", data=np.full(n, 9.0))
                h.create_dataset("footprint", data=np.zeros((n, 4, 4), np.uint8))
            with self.assertRaisesRegex(RuntimeError, "expected 999"):
                full_population_rows(latents, real, expected_n=999)


def _uniform_target(fp, res, y0=1, y1=3):
    """A footprint-exact flat slab: its own envelope reproduces it exactly (identity)."""
    occ = np.zeros((res, res, res), bool)
    occ[:, y0 : y1 + 1, :] = fp[:, None, :]
    return occ


def _peaked_target(fp, res, y0=1, y1=3, peak_top=5):
    """A slab with one small raised sub-region: the flat full-footprint extrusion (envelope)
    then covers columns the target does not reach at that height (an opportunity row)."""
    occ = _uniform_target(fp, res, y0, y1)
    zs, xs = np.nonzero(fp)
    zc, xc = zs[len(zs) // 2], xs[len(xs) // 2]
    occ[zc : zc + 2, y1 : peak_top, xc : xc + 2] = True
    return occ


def _sdf_by_row(rows, fp, res, opportunity_rows: set) -> dict:
    return {r: np.where(_peaked_target(fp, res) if r in opportunity_rows
                        else _uniform_target(fp, res), -1.0, 1.0).astype(np.float32)
           for r in rows}


class TestTrainSelectionCore(unittest.TestCase):
    """`is_opportunity = (r % 4) < 2` alternates region (region = r % 2) across both cohorts,
    so both regions have real opportunity AND identity supply -- unlike a naive `r % 4 == 0`
    filter, which would put every opportunity row in a single region."""

    def _setup(self, n=20):
        res = 8
        fp = np.zeros((res, res), bool)
        fp[1:6, 1:6] = True
        rows = list(range(n))
        regions = np.array([r % 2 for r in rows])
        footprints = np.stack([fp] * len(rows))
        index_of = {r: i for i, r in enumerate(rows)}
        opportunity_rows = {r for r in rows if (r % 4) < 2}
        handle = {"sdf": _sdf_by_row(rows, fp, res, opportunity_rows)}
        return rows, index_of, regions, footprints, handle

    def test_opportunity_and_identity_quotas_are_filled(self):
        rows, index_of, regions, footprints, handle = self._setup()
        opportunity_quota, identity_quota = {0: 3, 1: 3}, {0: 2, 1: 2}
        selected = _select_train_rows_from_handle(
            rows, index_of, regions, footprints, handle, salt="t",
            opportunity_quota=opportunity_quota, identity_quota=identity_quota)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(set(selected)), 10)

    def test_train_selection_is_order_independent(self):
        rows, index_of, regions, footprints, handle = self._setup()
        opportunity_quota, identity_quota = {0: 2, 1: 2}, {0: 1, 1: 1}
        a = _select_train_rows_from_handle(rows, index_of, regions, footprints, handle, "t",
                                          opportunity_quota, identity_quota)
        shuffled = rows.copy()
        np.random.default_rng(2).shuffle(shuffled)
        b = _select_train_rows_from_handle(shuffled, index_of, regions, footprints, handle, "t",
                                          opportunity_quota, identity_quota)
        self.assertEqual(a, b)

    def test_raises_with_namespaced_shortfall_when_a_cohort_cannot_be_filled(self):
        rows, index_of, regions, footprints, handle = self._setup()
        with self.assertRaisesRegex(RuntimeError, "opportunity shortfall.*identity shortfall"):
            _select_train_rows_from_handle(rows, index_of, regions, footprints, handle, "t",
                                          {0: 100, 1: 100}, {0: 1, 1: 1})


# ---------------------------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------------------------


class TestDigests(unittest.TestCase):
    def test_row_list_digest_is_order_independent_and_content_sensitive(self):
        self.assertEqual(row_list_digest([3, 1, 2]), row_list_digest([1, 2, 3]))
        self.assertNotEqual(row_list_digest([1, 2, 3]), row_list_digest([1, 2, 4]))

    def test_row_content_digest_changes_with_any_field(self):
        fp, occ = _box()
        base = row_content_digest(1, 0, 9.0, 2, 12, fp, occ, occ)
        self.assertNotEqual(base, row_content_digest(1, 0, 9.0, 2, 12, fp, occ, ~occ))
        self.assertNotEqual(base, row_content_digest(1, 0, 9.5, 2, 12, fp, occ, occ))
        self.assertNotEqual(base, row_content_digest(2, 0, 9.0, 2, 12, fp, occ, occ))

    def test_assert_frozen_a2_checkpoint_rejects_a_wrong_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            f.write(b"not the pinned checkpoint")
            f.flush()
            with self.assertRaisesRegex(ValueError, "does not match the frozen A2 checkpoint"):
                assert_frozen_a2_checkpoint(Path(f.name))

    def test_sha256_file_matches_hashlib_reference(self):
        import hashlib

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"some content")
            f.flush()
            self.assertEqual(sha256_file(Path(f.name)),
                             hashlib.sha256(b"some content").hexdigest())


class TestSharedConstantsMirrorTheFixedIdEvaluator(unittest.TestCase):
    """`S_STAR_VOXELS`/`COLLAPSE_MISSING` are duplicated (not imported) from
    `eval_massing_arms` specifically to avoid a top-level torch/scipy import -- see the comment
    at their definition. This is the drift guard that duplication needs."""

    def test_values_match(self):
        from scripts.foundations.prototype_voxel_editor import COLLAPSE_MISSING, S_STAR_VOXELS
        import scripts.foundations.eval_massing_arms as ema

        self.assertEqual(S_STAR_VOXELS, ema.S_STAR_VOXELS)
        self.assertEqual(COLLAPSE_MISSING, ema.COLLAPSE_MISSING)


class TestAssertDisjoint(unittest.TestCase):
    def test_disjoint_rows_pass(self):
        assert_disjoint_rows([1, 2, 3], [4, 5])  # must not raise

    def test_overlapping_rows_raise(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            assert_disjoint_rows([1, 2, 3], [3, 4])


# ---------------------------------------------------------------------------------------------
# Cache isolation and content-digest verification (synthetic tiny caches, no real corpus needed)
# ---------------------------------------------------------------------------------------------


def _write_tiny_cache(path: Path, role: str, rows: list[int], real_corpus_path: Path,
                      corrupt_row_digest: bool = False):
    fp, occ = _box(res=RES)  # `_create_cache`'s schema is hardcoded to RES=64 per axis
    target = occ.copy()
    _create_cache(path, {"role": role, "real_corpus_path": str(real_corpus_path)})
    for row in rows:
        digest = row_content_digest(row, 0, 9.0, 2, 12, fp, occ, target)
        if corrupt_row_digest:
            digest = "0" * 64
        _append_cache(path, {
            "row": row, "region": 0, "height_m": 9.0, "y0": 2, "y1": 12,
            "row_digest": digest.encode(),
            "footprint": fp.astype(np.uint8),
            "source_field": occupancy_to_sdf(occ).astype(np.float16),
            "source_occ": occ.astype(np.uint8),
            "sanitized_source_occ": sanitize_footprint(occ, fp).astype(np.uint8),
            "target_occ": target.astype(np.uint8),
            "envelope_occ": envelope_occupancy(fp, target).astype(np.uint8),
        })


@unittest.skipUnless(REAL_CORPUS_PATH.exists(), "real corpus metadata not available")
class TestCacheIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with h5py.File(REAL_CORPUS_PATH, "r") as f:
            cls.ids = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
            cls.heights = f["height_m"][:FROZEN_SPLIT_N_TOTAL]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _tiny_real_corpus(self, path: Path) -> None:
        with h5py.File(path, "w") as f:
            f.create_dataset("bag_id", data=self.ids)
            f.create_dataset("height_m", data=self.heights)
            f.create_dataset("sdf", shape=(len(self.ids), 1, 1, 1), dtype="f4")
            f.create_dataset("footprint", shape=(len(self.ids), 1, 1), dtype="u1")
            f.create_dataset("source_id", shape=(len(self.ids),), dtype="i4")

    def test_open_training_cache_rejects_a_screen_role_file(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        screen_path = self.dir / "screen.h5"
        _write_tiny_cache(screen_path, "screen", [1, 2], real)
        with self.assertRaises(RoleIsolationError):
            open_training_cache(screen_path, real=real)

    def test_open_training_cache_rejects_a_full_role_file(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        full_path = self.dir / "full.h5"
        _write_tiny_cache(full_path, "full", [1, 2], real)
        with self.assertRaises(RoleIsolationError):
            open_training_cache(full_path, real=real)

    def test_open_training_cache_accepts_a_train_role_file(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        train_path = self.dir / "train.h5"
        _write_tiny_cache(train_path, "train", [1, 2, 3], real)
        ds = open_training_cache(train_path, real=real)
        self.assertEqual(len(ds), 3)

    def test_finalize_cache_rejects_a_row_list_mismatch(self):
        from scripts.foundations.prototype_voxel_editor import _finalize_cache

        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        path = self.dir / "train.h5"
        _write_tiny_cache(path, "train", [1, 2, 3], real)
        with self.assertRaises(RuntimeError):
            _finalize_cache(path, expected_row_list_digest=row_list_digest([1, 2, 999]))

    def test_verify_cache_integrity_detects_content_digest_corruption(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        path = self.dir / "train.h5"
        _write_tiny_cache(path, "train", [1, 2], real, corrupt_row_digest=True)
        with self.assertRaisesRegex(ValueError, "content-digest mismatch"):
            verify_cache_integrity(path)

    def test_verify_cache_integrity_passes_on_an_uncorrupted_cache(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        path = self.dir / "train.h5"
        _write_tiny_cache(path, "train", [1, 2, 3], real)
        report = verify_cache_integrity(path)
        self.assertEqual(report["n"], 3)
        self.assertEqual(report["mismatched_rows"], [])

    def test_verify_cache_integrity_detects_duplicate_rows(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        path = self.dir / "train.h5"
        _write_tiny_cache(path, "train", [1, 1, 2], real)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_cache_integrity(path)

    def test_assert_disjoint_roles_catches_train_screen_overlap(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        train_path, screen_path, full_path = (self.dir / f"{r}.h5" for r in
                                              ("train", "screen", "full"))
        _write_tiny_cache(train_path, "train", [1, 2, 3], real)
        _write_tiny_cache(screen_path, "screen", [3, 4], real)  # 3 overlaps with train
        _write_tiny_cache(full_path, "full", [3, 4, 5, 6], real)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            assert_disjoint_roles(train_path, screen_path, full_path)

    def test_assert_disjoint_roles_catches_screen_not_subset_of_full(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        train_path, screen_path, full_path = (self.dir / f"{r}.h5" for r in
                                              ("train", "screen", "full"))
        _write_tiny_cache(train_path, "train", [1, 2], real)
        _write_tiny_cache(screen_path, "screen", [3, 99], real)  # 99 not in full
        _write_tiny_cache(full_path, "full", [3, 4, 5], real)
        with self.assertRaisesRegex(ValueError, "not members of full"):
            assert_disjoint_roles(train_path, screen_path, full_path)

    def test_assert_disjoint_roles_passes_for_a_clean_split(self):
        real = self.dir / "real.h5"
        self._tiny_real_corpus(real)
        train_path, screen_path, full_path = (self.dir / f"{r}.h5" for r in
                                              ("train", "screen", "full"))
        _write_tiny_cache(train_path, "train", [1, 2], real)
        _write_tiny_cache(screen_path, "screen", [3, 4], real)
        _write_tiny_cache(full_path, "full", [3, 4, 5, 6], real)
        report = assert_disjoint_roles(train_path, screen_path, full_path)
        self.assertTrue(report["screen_subset_of_full"])


# ---------------------------------------------------------------------------------------------
# The full-population gate: refuses to score anything but a checkpoint that already passed
# its OWN screen. Both guards raise before opening the checkpoint as a model or the full-role
# cache, so they're testable with a plain byte file and a JSON report -- no GPU or corpus needed.
# ---------------------------------------------------------------------------------------------


class TestEvaluateFullRefusesAnUnearnedCheckpoint(unittest.TestCase):
    def _args(self, checkpoint, screen_report):
        import types

        return types.SimpleNamespace(checkpoint=str(checkpoint), screen_report=str(screen_report),
                                    device="cpu", cache="unused", manifest="unused",
                                    real=None, report="unused")

    def test_refuses_a_checkpoint_whose_screen_did_not_pass(self):
        from scripts.foundations.prototype_voxel_editor import evaluate_full_command

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "editor.pth"
            checkpoint.write_bytes(b"checkpoint bytes")
            report = Path(tmp) / "screening.json"
            report.write_text(json.dumps({"summary": {
                "screening_gate_pass": False,
                "editor_checkpoint_sha256": sha256_file(checkpoint)}}))
            with self.assertRaisesRegex(SystemExit, "did not pass its screening gate"):
                evaluate_full_command(self._args(checkpoint, report))

    def test_refuses_a_checkpoint_that_does_not_match_the_screened_one(self):
        from scripts.foundations.prototype_voxel_editor import evaluate_full_command

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "editor.pth"
            checkpoint.write_bytes(b"checkpoint bytes")
            report = Path(tmp) / "screening.json"
            report.write_text(json.dumps({"summary": {
                "screening_gate_pass": True,
                "editor_checkpoint_sha256": "0" * 64}}))  # not this checkpoint's sha256
            with self.assertRaisesRegex(SystemExit, "is not the checkpoint"):
                evaluate_full_command(self._args(checkpoint, report))


if __name__ == "__main__":
    unittest.main(verbosity=2)
