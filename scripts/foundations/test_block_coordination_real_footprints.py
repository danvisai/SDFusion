"""Contract tests for #180 -- multi-footprint block coordination against real footprint sets.

Most of this file is synthetic and fast, matching the repo's usual convention -- but #180's own
point is to source footprints from the product's REAL bundled samples, so `TestLoadRealBlockScenes`
deliberately loads the actual `web/samples/*.png` assets (tiny local files, no GPU, no corpus) rather
than mocking them away; that is the one thing worth checking end to end.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_block_coordination_real_footprints.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import footprint_envelope_sdf  # noqa: E402
from scripts.foundations.block_coordination_real_footprints import (  # noqa: E402
    AXES, CONDITIONS, CUT_FRACTIONS, aggregate, initial_program_for, load_real_block_scenes,
    rasterize_building, run_block_scene, _axis_kwargs, _cluster_quadrants,
)

RES = 64


def _rect_envelope(row0, col0, size, y0=0, y1=20):
    fp = np.zeros((RES, RES), bool)
    fp[row0:row0 + size, col0:col0 + size] = True
    return fp, y0, y1


# ==================================================================================================
# rasterize_building -- the product's own conversion, reused
# ==================================================================================================


class TestRasterizeBuilding(unittest.TestCase):
    def test_a_simple_square_rasterizes_to_a_nonempty_centered_mask(self):
        square = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
        fp, y0, y1 = rasterize_building(square, height=12.0)
        self.assertEqual(fp.shape, (RES, RES))
        self.assertTrue(fp.any())
        self.assertLess(y0, y1)
        # roughly centered: the mask's own centroid should sit near the grid's middle
        zs, xs = np.nonzero(fp)
        self.assertAlmostEqual(zs.mean(), (RES - 1) / 2, delta=RES * 0.15)
        self.assertAlmostEqual(xs.mean(), (RES - 1) / 2, delta=RES * 0.15)

    def test_a_taller_building_gets_a_taller_voxel_range(self):
        square = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
        _fp, y0_short, y1_short = rasterize_building(square, height=6.0)
        _fp2, y0_tall, y1_tall = rasterize_building(square, height=30.0)
        self.assertGreater(y1_tall - y0_tall, y1_short - y0_short)


# ==================================================================================================
# _cluster_quadrants -- deterministic real-adjacency grouping
# ==================================================================================================


class TestClusterQuadrants(unittest.TestCase):
    def test_four_well_separated_groups_become_four_clusters(self):
        pts = []
        for cz in (-50, 50):
            for cx in (-50, 50):
                pts += [[cx + dx, cz + dz] for dx in (-2, 0, 2) for dz in (-2, 0, 2)]
        centroids = np.array(pts, float)
        clusters = _cluster_quadrants(centroids, min_size=3, max_size=9)
        self.assertEqual(len(clusters), 4)
        seen = set()
        for c in clusters:
            self.assertGreaterEqual(len(c), 3)
            seen.update(c)
        self.assertEqual(len(seen), len(centroids))          # every point assigned exactly once

    def test_a_cluster_below_min_size_is_dropped(self):
        centroids = np.array([[0.0, 0.0], [1.0, 1.0]])       # both same quadrant, only 2 points
        self.assertEqual(_cluster_quadrants(centroids, min_size=3), [])

    def test_a_cluster_is_capped_at_max_size(self):
        centroids = np.array([[float(i), float(i)] for i in range(10)])
        clusters = _cluster_quadrants(centroids, min_size=1, max_size=4)
        self.assertTrue(all(len(c) <= 4 for c in clusters))

    def test_empty_input_returns_no_clusters(self):
        self.assertEqual(_cluster_quadrants(np.zeros((0, 2))), [])


# ==================================================================================================
# initial_program_for -- the deterministic placeholder massing
# ==================================================================================================


class TestInitialProgramFor(unittest.TestCase):
    def test_produces_two_layers_at_two_distinct_heights_covering_the_whole_footprint(self):
        fp, y0, y1 = _rect_envelope(10, 10, 20, y0=0, y1=19)     # full = 20, split down the middle
        ops = initial_program_for(fp, y0, y1, cut_fractions=(0.15, 0.35))
        self.assertEqual(len(ops), 2)
        self.assertTrue(all(op.kind == "layer" and op.mode == "subtract" for op in ops))

        from scene.sdf_edit import EditableBuilding
        occ_sdf = footprint_envelope_sdf(fp, y0, y1, res=RES)
        eb = EditableBuilding(occ_sdf, ops)
        h = eb.to_occupancy(res=RES)[:, y0:y1 + 1, :].sum(axis=1)

        cut_a = 20 - max(1, round(20 * 0.15))
        cut_b = 20 - max(1, round(20 * 0.35))
        self.assertNotEqual(cut_a, cut_b, "the two halves must land at genuinely different heights")
        seen = set(np.unique(h[fp]).tolist())
        self.assertEqual(seen, {cut_a, cut_b})
        # every footprint column got EXACTLY one of the two cuts -- the split covers the whole mask
        self.assertTrue((h[fp] > 0).all())

    def test_a_multi_blob_footprint_still_covers_every_cell_exactly_once(self):
        """A concave/disconnected real footprint can leave a half in more than one connected
        piece -- `mask_components_rings` (not the single-component `mask_to_rings`) is what makes
        that safe. Whichever way three disconnected blobs happen to fall across the two halves,
        every one of their cells must still get exactly one op -- none dropped, none doubled."""
        fp = np.zeros((RES, RES), bool)
        fp[5:9, 5:9] = True
        fp[5:9, 20:24] = True
        fp[5:9, 55:59] = True
        y0, y1 = 0, 19
        ops = initial_program_for(fp, y0, y1, cut_fractions=(0.15, 0.35))
        self.assertGreaterEqual(len(ops), 2, "three disconnected blobs need more than one op")

        from scene.sdf_edit import EditableBuilding
        occ_sdf = footprint_envelope_sdf(fp, y0, y1, res=RES)
        h = EditableBuilding(occ_sdf, ops).to_occupancy(res=RES)[:, y0:y1 + 1, :].sum(axis=1)
        self.assertTrue((h[fp] > 0).all(), "every footprint cell must be covered by some op")
        self.assertEqual(int((h > 0).sum()), int(fp.sum()), "no op may add height outside fp")


# ==================================================================================================
# _axis_kwargs -- one representative value per axis, height_rhythm from the scene's own data
# ==================================================================================================


class TestAxisKwargs(unittest.TestCase):
    def setUp(self):
        # full=20 -> cuts 20-3=17, 20-7=13 (CUT_FRACTIONS=(0.15, 0.35));
        # full=40 -> cuts 40-6=34, 40-14=26. Pooled: [13, 17, 26, 34], median 21.5.
        self.envelopes = {
            "a": _rect_envelope(4, 4, 10, y0=0, y1=19),
            "b": _rect_envelope(40, 40, 10, y0=0, y1=39),
        }

    def test_each_single_axis_condition_sets_only_that_field(self):
        for axis in AXES:
            kw = _axis_kwargs(self.envelopes, axis)
            self.assertEqual(set(kw), {axis})

    def test_combined_sets_all_four(self):
        kw = _axis_kwargs(self.envelopes, "combined")
        self.assertEqual(set(kw), {"height_rhythm", "roof_family", "setback", "azimuth"})

    def test_unbiased_sets_nothing(self):
        self.assertEqual(_axis_kwargs(self.envelopes, "unbiased"), {})

    def test_height_rhythm_is_the_median_of_the_scenes_own_placeholder_cuts(self):
        kw = _axis_kwargs(self.envelopes, "height_rhythm")
        self.assertEqual(kw["height_rhythm"], 21.5)


# ==================================================================================================
# run_block_scene -- the full per-footprint, per-condition gate check
# ==================================================================================================


class TestRunBlockScene(unittest.TestCase):
    def setUp(self):
        self.scene = dict(scene_id="synthetic_test", source="synthetic", envelopes={
            "north": _rect_envelope(4, 4, 8, y0=0, y1=19),
            "south": _rect_envelope(40, 40, 8, y0=0, y1=19),
        })

    def test_every_footprint_gets_every_condition_reported(self):
        result = run_block_scene(self.scene)
        self.assertEqual(result["n_footprints"], 2)
        for fid in ("north", "south"):
            conditions = [r["condition"] for r in result["footprint_results"][fid]]
            self.assertEqual(conditions, list(CONDITIONS))

    def test_a_well_formed_coordinated_block_passes_every_condition(self):
        result = run_block_scene(self.scene)
        for fid, rows in result["footprint_results"].items():
            for r in rows:
                self.assertTrue(r["gate_pass"], (fid, r["condition"], r["problems"]))

    def test_containment_is_checked_in_addition_to_finalize_problems(self):
        """#180's own reconciliation note: `commit_block_program` calls `finalize_problems` only.
        Patch it to report clean while the committed occupancy is actually made to violate
        containment, and confirm `run_block_scene` still reports a failure -- proving the
        containment check is genuinely wired in, not just present in the source."""
        import scripts.foundations.block_coordination_real_footprints as mod

        def fake_commit(program, buildings, footprint_envelopes, res=64):
            from scene.sdf_edit import EditOp
            for fid in program.footprint_ids:
                fp, y0, y1 = footprint_envelopes[fid]
                # an op that carves a chunk out of the footprint's own ground-level perimeter --
                # fails containment_problems' rule 2, passes program_problems/commutes/height-map
                # representability trivially (a single subtractive Layer).
                buildings[fid].replace_ops(initial_program_for(fp, y0, y1))
            return {fid: [] for fid in program.footprint_ids}

        with patch.object(mod, "commit_block_program", side_effect=fake_commit), \
             patch.object(mod, "containment_problems", return_value=["forced containment failure"]):
            result = run_block_scene(self.scene)
        for fid, rows in result["footprint_results"].items():
            for r in rows:
                self.assertFalse(r["gate_pass"])
                self.assertIn("forced containment failure", r["problems"])

    def test_conditions_do_not_cascade_into_each_other(self):
        """Each condition must start from the SAME placeholder program, not from whatever the
        previous condition left behind -- verified by checking `_fresh_buildings` is called once
        per condition rather than the buildings dict being reused/mutated across the loop."""
        import scripts.foundations.block_coordination_real_footprints as mod
        calls = []
        real = mod._fresh_buildings

        def spy(envelopes):
            b = real(envelopes)
            calls.append({fid: [{k: v for k, v in op.to_dict().items() if k not in ("id", "group_id")}
                               for op in eb.ops]
                         for fid, eb in b.items()})
            return b

        with patch.object(mod, "_fresh_buildings", side_effect=spy):
            run_block_scene(self.scene)
        self.assertEqual(len(calls), len(CONDITIONS))
        for later in calls[1:]:
            self.assertEqual(later, calls[0])             # every condition starts identically


# ==================================================================================================
# aggregate -- true N, per-condition breakdown, every regression named
# ==================================================================================================


class TestAggregate(unittest.TestCase):
    def test_counts_and_regressions(self):
        scene_results = [
            dict(scene_id="s1", source="x", n_footprints=1, footprint_results={
                "a": [dict(condition="unbiased", gate_pass=True, problems=[]),
                      dict(condition="height_rhythm", gate_pass=True, problems=[]),
                      dict(condition="roof_family", gate_pass=False, problems=["bad"])],
            }),
        ]
        summary = aggregate(scene_results)
        self.assertEqual(summary["n_scenes"], 1)
        self.assertEqual(summary["n_checks"], 3)
        self.assertEqual(summary["n_pass"], 2)
        self.assertEqual(len(summary["regressions"]), 1)
        self.assertEqual(summary["regressions"][0]["footprint_id"], "a")
        self.assertEqual(summary["regressions"][0]["condition"], "roof_family")
        self.assertTrue(summary["regressions"][0]["is_regression"],
                        "unbiased passed for this footprint, so a later failure IS a regression")
        lo, hi = summary["pass_rate_ci"]
        self.assertLessEqual(lo, summary["pass_rate"])
        self.assertGreaterEqual(hi, summary["pass_rate"])

    def test_a_failure_under_unbiased_itself_is_not_flagged_as_a_regression(self):
        scene_results = [
            dict(scene_id="s1", source="x", n_footprints=1, footprint_results={
                "a": [dict(condition="unbiased", gate_pass=False, problems=["pre-existing"]),
                      dict(condition="setback", gate_pass=False, problems=["pre-existing"])],
            }),
        ]
        summary = aggregate(scene_results)
        self.assertEqual(len(summary["regressions"]), 2)
        for r in summary["regressions"]:
            self.assertFalse(r["is_regression"],
                             "already broken before coordination -- not a coordination regression")

    def test_undersampled_flag(self):
        few = [dict(scene_id=f"s{i}", source="x", n_footprints=1,
                    footprint_results={"a": [dict(condition="setback", gate_pass=True, problems=[])]})
              for i in range(2)]
        self.assertTrue(aggregate(few)["undersampled_scenes"])

    def test_empty_is_handled(self):
        summary = aggregate([])
        self.assertEqual(summary["n_scenes"], 0)
        self.assertIsNone(summary["pass_rate"])
        self.assertIsNone(summary["pass_rate_ci"])
        self.assertTrue(summary["undersampled_scenes"])


# ==================================================================================================
# load_real_block_scenes -- the one integration point against the actual bundled assets
# ==================================================================================================


class TestLoadRealBlockScenes(unittest.TestCase):
    def test_returns_a_handful_of_plausible_real_scenes(self):
        scenes = load_real_block_scenes()
        self.assertGreaterEqual(len(scenes), 3, "expected at least a handful of block scenes")
        seen_sources = {s["source"] for s in scenes}
        self.assertTrue(seen_sources.issubset({"munich_oldtown", "lafayette"}))
        self.assertNotIn("synthetic_blocks", seen_sources)
        for s in scenes:
            self.assertGreaterEqual(len(s["envelopes"]), 3)
            self.assertLessEqual(len(s["envelopes"]), 8)
            for fid, (fp, y0, y1) in s["envelopes"].items():
                self.assertEqual(fp.dtype, bool)
                self.assertTrue(fp.any())
                self.assertLess(y0, y1)

    def test_deterministic_across_runs(self):
        a = load_real_block_scenes()
        b = load_real_block_scenes()
        self.assertEqual([s["scene_id"] for s in a], [s["scene_id"] for s in b])
        for sa, sb in zip(a, b):
            self.assertEqual(set(sa["envelopes"]), set(sb["envelopes"]))

    def test_a_footprint_that_rasterizes_to_more_than_one_component_is_excluded(self):
        """A real extracted contour can rasterize to a split mask at RES=64 (a thin neck, a
        self-touching outline) -- caught here rather than surfacing as a crash three calls deeper
        in `initial_program_for`/`mask_to_rings`, which refuses a multi-component mask outright."""
        import scripts.foundations.block_coordination_real_footprints as mod

        split = np.zeros((64, 64), bool)
        split[10:14, 10:14] = True
        split[30:34, 30:34] = True                      # a second, disconnected blob

        def fake_rasterize(points, height):
            return split, 0, 19

        with patch.object(mod, "rasterize_building", side_effect=fake_rasterize):
            scenes = load_real_block_scenes(samples=("munich_oldtown",), min_size=1, max_size=8)
        # every extracted building now rasterizes to the same split (2-component) mask, so every
        # one is excluded and no scene ever reaches even `min_size=1` -- if the exclusion were
        # missing, this would instead return scenes full of un-fittable split footprints.
        self.assertEqual(scenes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
