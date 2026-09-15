"""Contract tests for #177's roof-family-stratified, spatially-blocked BuildingWorld held-out
split. Synthetic, fast, CPU, no network; the ledger-integration tests use temp files, never the
production `corpus_ledger.h5`.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations import corpus_ledger  # noqa: E402
import scripts.foundations.buildingworld_stratified_split as bss  # noqa: E402


class TestTileKey(unittest.TestCase):
    def test_same_cell_maps_to_the_same_key(self):
        self.assertEqual(bss.tile_key("Berlin", 10.0, 10.0, cell_m=150.0),
                         bss.tile_key("Berlin", 140.0, 140.0, cell_m=150.0))

    def test_adjacent_cells_differ(self):
        self.assertNotEqual(bss.tile_key("Berlin", 10.0, 10.0, cell_m=150.0),
                            bss.tile_key("Berlin", 160.0, 10.0, cell_m=150.0))

    def test_negative_coordinates_bucket_correctly(self):
        # floor-division semantics: -1.0 and -149.0 are the same 150m cell, -151.0 is not.
        self.assertEqual(bss.tile_key("Berlin", -1.0, 0.0, cell_m=150.0),
                         bss.tile_key("Berlin", -149.0, 0.0, cell_m=150.0))
        self.assertNotEqual(bss.tile_key("Berlin", -1.0, 0.0, cell_m=150.0),
                            bss.tile_key("Berlin", -151.0, 0.0, cell_m=150.0))

    def test_different_cities_at_the_same_coordinates_never_collide(self):
        """#169's pooled bucket grids four different cities' (independently corrected, unrelated)
        coordinate systems together -- without a city-scoped key, Adelaide's and Perth's own
        unrelated cell (5, 10) would silently merge into one tile."""
        self.assertNotEqual(bss.tile_key("Adelaide", 750.0, 1500.0, cell_m=150.0),
                            bss.tile_key("Perth", 750.0, 1500.0, cell_m=150.0))


class TestGroupOfCity(unittest.TestCase):
    def test_pooled_cities_collapse_to_one_group(self):
        for c in ["Adelaide", "Greater Geelong", "Perth", "Philadelphia"]:
            self.assertEqual(bss.group_of_city(c), bss.POOLED_GROUP_NAME)

    def test_other_cities_are_their_own_group(self):
        for c in ["Berlin", "Tokyo", "Cambridge", "Toronto"]:
            self.assertEqual(bss.group_of_city(c), c)


class TestPlanGroupSplit(unittest.TestCase):
    def _grid(self, nx, ny, cell=150.0, per_cell=10, families=None, seed=0, city="Berlin",
             cell_sizes=None):
        """A synthetic city: nx*ny occupied grid cells, `per_cell` rows each (or, if
        `cell_sizes` is given, a per-cell row count keyed by (ix, iy) -- for tests that need
        deliberately UNEVEN tile sizes), spread across a cycling family sequence."""
        rng = np.random.default_rng(seed)
        rows, cities, xs, ys, fams = [], [], [], [], []
        r = 0
        fam_cycle = families or list(bss.FAMILIES)
        for ix in range(nx):
            for iy in range(ny):
                n = cell_sizes.get((ix, iy), per_cell) if cell_sizes else per_cell
                for k in range(n):
                    rows.append(r)
                    cities.append(city)
                    xs.append(ix * cell + cell / 2 + rng.uniform(-1, 1))
                    ys.append(iy * cell + cell / 2 + rng.uniform(-1, 1))
                    fams.append(fam_cycle[r % len(fam_cycle)])
                    r += 1
        return (np.array(rows), np.array(cities), np.array(xs), np.array(ys), np.array(fams))

    def test_empty_group_returns_nothing_selected(self):
        held, report = bss.plan_group_split(np.array([]), np.array([]), np.array([]),
                                             np.array([]), np.array([]))
        self.assertEqual(held.shape[0], 0)
        self.assertEqual(report["held_out_n"], 0)

    def test_never_splits_a_tile_across_train_and_held_out(self):
        rows, city, x, y, fam = self._grid(6, 6, per_cell=8)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.1, seed=1)
        keys = np.array([bss.tile_key(c, xi, yi) for c, xi, yi in zip(city, x, y)])
        for k in np.unique(keys):
            labels = set(held[keys == k].tolist())
            self.assertEqual(len(labels), 1, f"tile {k!r} spans both train and held-out")

    def test_at_least_one_tile_is_selected_for_a_nonempty_group(self):
        # target_frac tiny enough that 2% of n rounds under a single tile's count.
        rows, city, x, y, fam = self._grid(3, 3, per_cell=50)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.0001, seed=0)
        self.assertGreater(report["held_out_n"], 0)

    def test_achieved_size_is_close_to_the_target(self):
        rows, city, x, y, fam = self._grid(10, 10, per_cell=5)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.1, seed=2)
        self.assertAlmostEqual(report["held_out_frac"], 0.1, delta=0.05)

    def test_family_stratification_tracks_the_groups_own_distribution(self):
        # 80% flat, 20% gable, no hip/complex -- the held-out slice should roughly mirror that,
        # not collapse to one family or force a false even split.
        families = ["flat"] * 8 + ["gable"] * 2
        rows, city, x, y, fam = self._grid(10, 10, per_cell=10, families=families, seed=3)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.2, seed=3)
        achieved = report["held_out_family_fractions"]
        self.assertAlmostEqual(achieved["flat"], 0.8, delta=0.15)
        self.assertAlmostEqual(achieved["gable"], 0.2, delta=0.15)
        self.assertEqual(achieved["hip"], 0.0)
        self.assertEqual(achieved["complex"], 0.0)

    def test_a_family_absent_from_the_group_degrades_gracefully_not_an_error(self):
        rows, city, x, y, fam = self._grid(5, 5, per_cell=10, families=["flat"], seed=4)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.1, seed=4)
        self.assertGreater(report["held_out_n"], 0)
        self.assertEqual(report["held_out_family_fractions"]["gable"], 0.0)

    def test_a_single_tile_group_is_forced_to_take_that_whole_tile(self):
        # every row in one cell -- there is no smaller unit to pick, mirroring #153's NL case.
        rows = np.arange(40)
        city = np.array(["Berlin"] * 40)
        x = np.full(40, 10.0)
        y = np.full(40, 10.0)
        fam = np.array(["flat"] * 40)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.02, seed=0)
        self.assertEqual(report["held_out_n"], 40)
        self.assertTrue(held.all())

    def test_deterministic_for_a_fixed_seed(self):
        rows, city, x, y, fam = self._grid(8, 8, per_cell=6)
        held1, _ = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.15, seed=7)
        held2, _ = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.15, seed=7)
        np.testing.assert_array_equal(held1, held2)

    def test_selection_is_not_biased_toward_the_sparsest_tiles(self):
        """Regression for the bug an earlier 'smallest tile first' heuristic had: it picked
        isolated single-building tiles almost exclusively, which cannot protect against the
        dense-terrace-row leakage #177's own issue text names as the reason spatial blocking
        exists. Half the grid's cells are dense (20 rows, simulating a terrace row); half are
        sparse (1 row, an isolated building). The held-out slice's own mean tile size should
        track the population's mean, not collapse toward the sparse end."""
        cell_sizes = {}
        for ix in range(20):
            for iy in range(20):
                cell_sizes[(ix, iy)] = 20 if (ix + iy) % 2 == 0 else 1
        rows, city, x, y, fam = self._grid(20, 20, per_cell=1, cell_sizes=cell_sizes, seed=5)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.2, seed=5)
        # Population mean tile size is (200*20 + 200*1) / 400 = 10.5. Stopping as soon as the
        # cumulative row count crosses a target is itself a mild "waiting time" bias toward
        # smaller tiles (a run of small tiles delays stopping and all get included) even under
        # pure random order -- 0.3x leaves headroom for that. A sparsest-first bug drives the
        # held-out mean down near 1 (~9.5% of the population mean), far below this bar either way.
        self.assertGreater(report["mean_tile_size_held_out"], report["mean_tile_size_all"] * 0.3)

    def test_a_small_target_family_is_not_crowded_out_by_a_large_one(self):
        """Regression: sequential largest-target-first family processing let one large family's
        own claim run all the way to the OVERALL target before a small-target family ever got a
        dedicated tile. Here 'hip' is a small (10%) minority but has plenty of its own tiles
        available -- round-robin claiming must still give it a nonzero, roughly proportional
        share rather than zero."""
        families = ["flat"] * 45 + ["gable"] * 45 + ["hip"] * 10
        rows, city, x, y, fam = self._grid(15, 15, per_cell=10, families=families, seed=6)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.3, seed=6)
        self.assertGreater(report["held_out_family_fractions"]["hip"], 0.0)

    def test_a_dominant_family_is_not_squeezed_by_equal_turn_taking(self):
        """Regression: an earlier equal-turn round-robin gave every active family exactly one
        claim per round regardless of its own target size, so a small-target family "spent" the
        shared row budget at the same rate as a large one -- squeezing the dominant family's
        achieved share well below its own population share (measured in production: Calgary's
        68%-share family fell to 58% of its held-out slice). 'gable' here is a clear ~70%
        majority; its held-out share must stay close to its population share, not collapse
        toward an even split with the minorities."""
        families = ["gable"] * 7 + ["flat"] * 2 + ["hip"] * 1
        rows, city, x, y, fam = self._grid(15, 15, per_cell=10, families=families, seed=8)
        held, report = bss.plan_group_split(rows, city, x, y, fam, target_frac=0.3, seed=8)
        achieved = report["held_out_family_fractions"]
        self.assertAlmostEqual(achieved["gable"], report["family_fractions"]["gable"], delta=0.1)


class TestBuildAssignment(unittest.TestCase):
    def _inputs(self, rows, cities, x, y, fam, missing_centroid=(), missing_family=()):
        source_key = np.array([f"bw:{bss.CITY_SLUG[c]}".encode("ascii") for c in cities],
                              dtype="S64")
        centroid = {r: (xi, yi) for r, xi, yi in zip(rows, x, y) if r not in missing_centroid}
        family = {r: f for r, f in zip(rows, fam) if r not in missing_family}
        return dict(rows=np.array(rows), source_key=source_key,
                   height_m=np.array([10.0] * len(rows), dtype=np.float32),
                   centroid=centroid, family=family,
                   missing_centroid=list(missing_centroid), missing_family=list(missing_family))

    def test_raises_loudly_when_a_row_is_missing_required_inputs(self):
        inputs = self._inputs([1, 2, 3], ["Berlin"] * 3, [0, 200, 400], [0, 0, 0],
                              ["flat", "gable", "flat"], missing_centroid=[2])
        with self.assertRaises(ValueError):
            bss.build_assignment(inputs)

    def test_every_row_gets_an_explicit_assignment(self):
        rows = list(range(30))
        cities = ["Berlin"] * 15 + ["Tokyo"] * 15
        x = [float(i % 5) * 200 for i in range(30)]
        y = [float(i // 5) * 200 for i in range(30)]
        fam = [bss.FAMILIES[i % 4] for i in range(30)]
        inputs = self._inputs(rows, cities, x, y, fam)
        out_rows, held_out, report = bss.build_assignment(inputs, target_frac=0.2, seed=0)
        self.assertEqual(set(out_rows.tolist()), set(rows))
        self.assertEqual(held_out.shape[0], 30)
        self.assertIn("Berlin", report["groups"])
        self.assertIn("Tokyo", report["groups"])

    def test_pooled_cities_share_one_group_but_report_per_city_diagnostics(self):
        rows = list(range(40))
        cities = (["Adelaide"] * 10 + ["Perth"] * 10 + ["Philadelphia"] * 10
                 + ["Greater Geelong"] * 10)
        x = [float(i % 4) * 200 for i in range(40)]
        y = [float(i // 4) * 200 for i in range(40)]
        fam = [bss.FAMILIES[i % 4] for i in range(40)]
        inputs = self._inputs(rows, cities, x, y, fam)
        out_rows, held_out, report = bss.build_assignment(inputs, target_frac=0.1, seed=0)
        self.assertIn(bss.POOLED_GROUP_NAME, report["groups"])
        self.assertNotIn("Adelaide", report["groups"])  # merged, not its own top-level group
        self.assertEqual(set(report["pooled_per_city"]),
                         {"Adelaide", "Perth", "Philadelphia", "Greater Geelong"})


class TestRunLedgerIntegration(unittest.TestCase):
    """`run()`'s I/O glue against temp files -- never the production ledger."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def _write_real_h5(self, path, n_old=6, extra=None):
        extra = extra or []
        n_new = len(extra)
        n = n_old + n_new
        ids = np.array([f"old#{i}".encode() for i in range(n_old)]
                       + [e["bag_id"] for e in extra], dtype="S64")
        heights = np.array([1.0] * n_old + [e["height_m"] for e in extra], dtype=np.float32)
        source_id = np.array([0] * n_old + [-1] * n_new, dtype=np.int32)
        source_key = np.array([b""] * n_old + [e["source_key"] for e in extra], dtype="S64")
        with h5py.File(path, "w") as f:
            f.create_dataset("sdf", data=np.zeros((n, 2, 2, 2), np.float32))
            f.create_dataset("footprint", data=np.ones((n, 2, 2), np.uint8))
            f.create_dataset("bag_id", data=ids)
            f.create_dataset("height_m", data=heights)
            f.create_dataset("source_id", data=source_id)
            f.create_dataset("source_key", data=source_key)
        return ids, heights

    def _write_centroids_h5(self, path, rows, x, y):
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["committed_rows"] = len(rows)
            f.create_dataset("row", data=np.array(rows, np.int32))
            f.create_dataset("x_m", data=np.array(x, np.float32))
            f.create_dataset("y_m", data=np.array(y, np.float32))
            f.create_dataset("ok", data=np.ones(len(rows), np.uint8))

    def _write_program_labels_h5(self, path, rows, family):
        with h5py.File(path, "w") as f:
            f.create_dataset("row", data=np.array(rows, np.int32))
            f.create_dataset("ok", data=np.ones(len(rows), np.uint8))
            f.create_dataset("family", data=np.array([f.encode() for f in family], dtype="S8"))

    def test_dry_run_writes_nothing_to_the_ledger(self):
        import utils.frozen_corpus as fc

        old_n, old_hash = fc.FROZEN_SPLIT_N_TOTAL, fc.FROZEN_CORPUS_SHA256
        n_old = 6
        extra = [dict(bag_id=f"Berlin#{i}".encode(), height_m=5.0,
                      source_key=b"bw:Berlin") for i in range(20)]
        real_path = self.d / "real.h5"
        ids, heights = self._write_real_h5(real_path, n_old=n_old, extra=extra)
        fc.FROZEN_SPLIT_N_TOTAL = n_old
        fc.FROZEN_CORPUS_SHA256 = fc.row_identity_sha256(ids[:n_old], heights[:n_old])
        try:
            rows = list(range(n_old, n_old + len(extra)))
            centroids_path = self.d / "centroids.h5"
            self._write_centroids_h5(centroids_path, rows,
                                     [float(i * 200) for i in range(len(extra))],
                                     [0.0] * len(extra))
            labels_path = self.d / "labels.h5"
            self._write_program_labels_h5(labels_path, rows,
                                          [bss.FAMILIES[i % 4] for i in range(len(extra))])
            ledger_path = self.d / "ledger.h5"

            report = bss.run(real_path, centroids_path, labels_path, ledger_path,
                             target_frac=0.2, seed=0, dry_run=True,
                             report_out=self.d / "report.json")
            self.assertFalse(ledger_path.exists())
            self.assertFalse((self.d / "report.json").exists())
            self.assertGreater(report["held_out_n_total"], 0)
        finally:
            fc.FROZEN_SPLIT_N_TOTAL, fc.FROZEN_CORPUS_SHA256 = old_n, old_hash

    def test_run_appends_without_disturbing_an_existing_ledger_prefix(self):
        import utils.frozen_corpus as fc

        old_n, old_hash = fc.FROZEN_SPLIT_N_TOTAL, fc.FROZEN_CORPUS_SHA256
        n_old = 6
        extra = [dict(bag_id=f"Berlin#{i}".encode(), height_m=5.0,
                      source_key=b"bw:Berlin") for i in range(20)]
        real_path = self.d / "real.h5"
        ids, heights = self._write_real_h5(real_path, n_old=n_old, extra=extra)
        fc.FROZEN_SPLIT_N_TOTAL = n_old
        fc.FROZEN_CORPUS_SHA256 = fc.row_identity_sha256(ids[:n_old], heights[:n_old])
        try:
            ledger_path = self.d / "ledger.h5"
            corpus_ledger.write_ledger(row=list(range(n_old)), region=[0] * n_old,
                                      held_out=[0, 1, 0, 1, 0, 0], height_m=[1.0] * n_old,
                                      path=ledger_path, source="pre-existing")
            before = corpus_ledger.read_ledger(ledger_path)

            rows = list(range(n_old, n_old + len(extra)))
            centroids_path = self.d / "centroids.h5"
            self._write_centroids_h5(centroids_path, rows,
                                     [float(i * 200) for i in range(len(extra))],
                                     [0.0] * len(extra))
            labels_path = self.d / "labels.h5"
            self._write_program_labels_h5(labels_path, rows,
                                          [bss.FAMILIES[i % 4] for i in range(len(extra))])

            report = bss.run(real_path, centroids_path, labels_path, ledger_path,
                             target_frac=0.2, seed=0, dry_run=False,
                             report_out=self.d / "report.json")

            after = corpus_ledger.read_ledger(ledger_path)
            for col in corpus_ledger.COLUMNS:
                np.testing.assert_array_equal(after[col][:n_old], before[col])
            self.assertEqual(len(after["row"]), n_old + len(extra))
            new_rows = set(int(r) for r in after["row"][n_old:])
            self.assertEqual(new_rows, set(rows))
            region_new = {int(r): int(reg) for r, reg in zip(after["row"][n_old:],
                                                             after["region"][n_old:])}
            self.assertTrue(all(v == bss.LEDGER_REGION for v in region_new.values()))
        finally:
            fc.FROZEN_SPLIT_N_TOTAL, fc.FROZEN_CORPUS_SHA256 = old_n, old_hash

    def test_run_raises_and_appends_nothing_when_a_row_lacks_a_centroid(self):
        import utils.frozen_corpus as fc

        old_n, old_hash = fc.FROZEN_SPLIT_N_TOTAL, fc.FROZEN_CORPUS_SHA256
        n_old = 4
        extra = [dict(bag_id=f"Berlin#{i}".encode(), height_m=5.0,
                      source_key=b"bw:Berlin") for i in range(5)]
        real_path = self.d / "real.h5"
        ids, heights = self._write_real_h5(real_path, n_old=n_old, extra=extra)
        fc.FROZEN_SPLIT_N_TOTAL = n_old
        fc.FROZEN_CORPUS_SHA256 = fc.row_identity_sha256(ids[:n_old], heights[:n_old])
        try:
            rows = list(range(n_old, n_old + len(extra)))
            centroids_path = self.d / "centroids.h5"
            # only 4 of 5 rows get a centroid
            self._write_centroids_h5(centroids_path, rows[:-1],
                                     [0.0, 200.0, 400.0, 600.0], [0.0] * 4)
            labels_path = self.d / "labels.h5"
            self._write_program_labels_h5(labels_path, rows, ["flat"] * 5)
            ledger_path = self.d / "ledger.h5"

            with self.assertRaises(ValueError):
                bss.run(real_path, centroids_path, labels_path, ledger_path, dry_run=False)
            self.assertFalse(ledger_path.exists())
        finally:
            fc.FROZEN_SPLIT_N_TOTAL, fc.FROZEN_CORPUS_SHA256 = old_n, old_hash


if __name__ == "__main__":
    unittest.main()
