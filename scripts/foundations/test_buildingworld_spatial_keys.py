"""Contract tests for #177's BuildingWorld centroid extractor. Synthetic, fast, CPU, no network;
the end-to-end tests need the real corpus's frozen-prefix METADATA (bag_id/height_m only, to pass
#162's hash check) but never its SDF volumes, mirroring `test_ingest_surfaces_buildingworld.py`.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import h5py
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.foundations.buildingworld_spatial_keys as bsk  # noqa: E402
from scripts.foundations.ingest_buildingworld import bag_id_for, source_key_for  # noqa: E402
from scripts.ingest_3dbag import building_to_sdf  # noqa: E402
from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, REAL_CORPUS_PATH, row_identity_sha256


def _box_obj_bytes(size=(4.0, 4.0, 3.0), origin=(0.0, 0.0, 0.0)) -> bytes:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.asarray(origin) + np.asarray(size) / 2.0)
    return trimesh.exchange.obj.export_obj(m).encode("ascii")


class TestSlugToCity(unittest.TestCase):
    def test_reverses_every_ingestable_citys_slug(self):
        from scripts.foundations.ingest_buildingworld import INGESTABLE_CITIES, city_slug

        for city in INGESTABLE_CITIES:
            self.assertEqual(bsk.SLUG_TO_CITY[city_slug(city)], city)


class TestCentroidXY(unittest.TestCase):
    def test_centroid_is_the_bounds_midpoint_not_the_vertex_mean(self):
        # An L-shaped mesh (more vertices on one side) would bias a vertex-mean centroid; the
        # bounds midpoint is invariant to where the extra vertices sit.
        m = trimesh.creation.box(extents=(4.0, 2.0, 3.0))
        m.apply_translation((10.0, -5.0, 1.5))
        x, y = bsk.centroid_xy(m)
        self.assertAlmostEqual(x, 10.0, places=5)
        self.assertAlmostEqual(y, -5.0, places=5)

    def test_reflects_a_city_correction_applied_before_it(self):
        from scripts.foundations.ingest_buildingworld import FEET_TO_M

        m = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        m.apply_translation((3937.0, 3937.0, 0.0))
        m.vertices = m.vertices * FEET_TO_M  # what apply_geometric_correction would do for Boston
        x, y = bsk.centroid_xy(m)
        self.assertAlmostEqual(x, 1200.0, places=3)
        self.assertAlmostEqual(y, 1200.0, places=3)


class TestSimpleTask(unittest.TestCase):
    def test_reads_zip_applies_correction_and_returns_a_centroid(self):
        from scripts.foundations.ingest_buildingworld import FEET_TO_M

        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/x.obj", _box_obj_bytes(size=(4.0, 4.0, 3.0),
                                                          origin=(3937.0, 0.0, 0.0)))
            rec = bsk._simple_task(("Boston", str(zpath), "mesh/x.obj", 42))
        self.assertEqual(rec["row"], 42)
        self.assertEqual(rec["ok"], 1)
        # centroid = (origin + size/2) * FEET_TO_M -- the correction, then the box's own center.
        self.assertAlmostEqual(float(rec["x_m"]), (3937.0 + 2.0) * FEET_TO_M, places=2)

    def test_a_load_error_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/garbage.obj", b"not an obj file at all")
            rec = bsk._simple_task(("Berlin", str(zpath), "mesh/garbage.obj", 7))
        self.assertEqual(rec["row"], 7)
        self.assertEqual(rec["ok"], 0)
        self.assertTrue(np.isnan(rec["x_m"]))
        self.assertIn(b"Error", rec["reason"])


class TestIncrementalRowWriter(unittest.TestCase):
    def test_write_flush_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "centroids.h5"
            with bsk.IncrementalRowWriter(path, flush_every=2) as w:
                for i in range(5):
                    w.add(row=i, x_m=np.float32(i), y_m=np.float32(-i), ok=np.uint8(1),
                         reason=b"")
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 5)
                self.assertEqual(int(f.attrs["committed_rows"]), 5)
                np.testing.assert_array_equal(f["row"][:], np.arange(5))

    def test_resume_skips_already_committed_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "centroids.h5"
            with bsk.IncrementalRowWriter(path, flush_every=100) as w:
                for i in range(3):
                    w.add(row=i, x_m=np.float32(i), y_m=np.float32(i), ok=np.uint8(1),
                         reason=b"")
            with bsk.IncrementalRowWriter(path, resume=True) as w2:
                self.assertTrue(w2.already_done(0))
                self.assertFalse(w2.already_done(3))
                for i in range(3, 6):
                    w2.add(row=i, x_m=np.float32(i), y_m=np.float32(i), ok=np.uint8(1),
                          reason=b"")
            with h5py.File(path, "r") as f:
                self.assertEqual(sorted(int(r) for r in f["row"][:]), list(range(6)))

    def test_no_resume_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "centroids.h5"
            with bsk.IncrementalRowWriter(path) as w:
                w.add(row=0, x_m=np.float32(0), y_m=np.float32(0), ok=np.uint8(1), reason=b"")
            with bsk.IncrementalRowWriter(path, resume=False) as w2:
                w2.add(row=1, x_m=np.float32(1), y_m=np.float32(1), ok=np.uint8(1), reason=b"")
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 1)
                self.assertEqual(int(f["row"][0]), 1)

    def test_schema_mismatch_refuses_to_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "centroids.h5"
            with h5py.File(path, "w") as f:
                f.attrs["schema_version"] = 999
                f.attrs["committed_rows"] = 0
            with self.assertRaises(SystemExit):
                bsk.IncrementalRowWriter(path, resume=True)


@unittest.skipUnless(REAL_CORPUS_PATH.exists(), "real corpus metadata not available")
class TestRunEndToEnd(unittest.TestCase):
    """Mirrors test_ingest_surfaces_buildingworld.TestRunEndToEnd's fixture: `open_real_corpus`
    checks the frozen prefix against a hardcoded hash (#162), so a synthetic `real.h5` can only
    pass by carrying the real corpus's own (bag_id, height_m) for that prefix."""

    R = 8

    @classmethod
    def setUpClass(cls):
        with h5py.File(REAL_CORPUS_PATH, "r") as f:
            cls.ids = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
            cls.heights = f["height_m"][:FROZEN_SPLIT_N_TOTAL]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def _write_synthetic_real_h5(self, path: Path, extra_bag_ids, extra_sdf=None,
                                 city="Berlin") -> None:
        n_old, n_new = FROZEN_SPLIT_N_TOTAL, len(extra_bag_ids)
        n = n_old + n_new
        r = self.R
        sdf = np.zeros((n, r, r, r), np.float32)
        if extra_sdf is not None:
            sdf[n_old:] = extra_sdf
        with h5py.File(path, "w") as f:
            f.create_dataset("sdf", data=sdf)
            f.create_dataset("footprint", data=np.ones((n, r, r), np.uint8))
            f.create_dataset("height_m", data=np.concatenate(
                [self.heights, np.full(n_new, 5.0, np.float32)]))
            f.create_dataset("style_id", data=np.full(n, 8, np.int32))
            f.create_dataset("source_id", data=np.full(n, -1, np.int32))
            f.create_dataset("class_label", data=np.array([b"X"] * n, dtype="S16"))
            f.create_dataset("bag_id", data=np.concatenate(
                [self.ids, np.array(extra_bag_ids, dtype="S64")]))
            f.create_dataset("source_key", data=np.array(
                [b""] * n_old + [source_key_for(city).encode("ascii")] * n_new, dtype="S64"))

    def _zip_with(self, city_dir: Path, members: dict) -> None:
        city_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(city_dir / "mesh.zip", "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)

    def _run(self, real_path: Path, mesh_root: Path, cities, **kw):
        import scripts.foundations.profile_buildingworld_meshes as pbm

        old_root = pbm.MESH_ROOT
        pbm.MESH_ROOT = mesh_root
        try:
            return bsk.run(cities, h5_path=real_path, **kw)
        finally:
            pbm.MESH_ROOT = old_root

    def test_run_writes_a_centroid_per_row_and_preserves_the_frozen_prefix(self):
        real_path = self.d / "real.h5"
        member_names = ["mesh/a.obj", "mesh/b.obj"]
        extra_bag_ids = [bag_id_for("Berlin", n) for n in member_names]
        self._write_synthetic_real_h5(real_path, extra_bag_ids)
        with h5py.File(real_path, "r") as f:
            baseline = row_identity_sha256(f["bag_id"][:FROZEN_SPLIT_N_TOTAL],
                                           f["height_m"][:FROZEN_SPLIT_N_TOTAL])

        mesh_root = self.d / "mesh_root"
        self._zip_with(mesh_root / "Berlin" / "mesh", {
            "mesh/a.obj": _box_obj_bytes(origin=(10.0, 0.0, 0.0)),
            "mesh/b.obj": _box_obj_bytes(origin=(0.0, 20.0, 0.0)),
            "mesh/not_kept.obj": _box_obj_bytes(),
        })
        out_path = self.d / "centroids.h5"
        result = self._run(real_path, mesh_root, ["Berlin"], out_path=out_path)

        with h5py.File(real_path, "r") as f:
            self.assertEqual(row_identity_sha256(f["bag_id"][:FROZEN_SPLIT_N_TOTAL],
                                                 f["height_m"][:FROZEN_SPLIT_N_TOTAL]), baseline)

        self.assertEqual(result["n_total"], 2)
        self.assertEqual(result["n_ok"], 2)
        row_a, row_b = FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 1
        with h5py.File(out_path, "r") as f:
            n = int(f.attrs["committed_rows"])
            by_row = {int(r): i for i, r in enumerate(f["row"][:n])}
            self.assertEqual(set(by_row), {row_a, row_b})
            self.assertAlmostEqual(float(f["x_m"][by_row[row_a]]), 10.0 + 2.0, places=3)
            self.assertAlmostEqual(float(f["y_m"][by_row[row_b]]), 20.0 + 2.0, places=3)

    def test_run_resumes_without_dropping_or_duplicating_rows(self):
        real_path = self.d / "real.h5"
        member_names = ["mesh/a.obj", "mesh/b.obj", "mesh/c.obj"]
        extra_bag_ids = [bag_id_for("Berlin", n) for n in member_names]
        self._write_synthetic_real_h5(real_path, extra_bag_ids)
        mesh_root = self.d / "mesh_root"
        self._zip_with(mesh_root / "Berlin" / "mesh",
                       {n: _box_obj_bytes() for n in member_names})
        out_path = self.d / "centroids.h5"

        first = self._run(real_path, mesh_root, ["Berlin"], out_path=out_path, limit=1)
        self.assertEqual(first["n_total"], 1)
        second = self._run(real_path, mesh_root, ["Berlin"], out_path=out_path)
        self.assertEqual(second["n_total"], 3)
        with h5py.File(out_path, "r") as f:
            self.assertEqual(sorted(int(r) for r in f["row"][:]),
                             list(range(FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 3)))

    def test_run_resolves_a_truncation_collision_end_to_end(self):
        """Two real.h5 rows share one truncated bag_id (a Perth-shaped collision); the centroid
        each row gets must come from ITS OWN mesh, not whichever candidate happened to be
        resolved first -- the slab and the cube are translated to different, checkable centers."""
        long_prefix = "mesh/" + "y" * 60
        member_names = [f"{long_prefix}_part1.obj", f"{long_prefix}_part2.obj"]
        self.assertEqual(bag_id_for("Berlin", member_names[0]),
                         bag_id_for("Berlin", member_names[1]))
        shared_bag_id = bag_id_for("Berlin", member_names[0])

        cube = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
        cube.apply_translation((100.0, 0.0, 1.5))
        slab = trimesh.creation.box(extents=(3.0, 3.0, 9.0))
        slab.apply_translation((0.0, 100.0, 4.5))
        sdf_cube, _, _ = building_to_sdf(cube, self.R)
        sdf_slab, _, _ = building_to_sdf(slab, self.R)

        real_path = self.d / "real.h5"
        # row order (cube, slab) is the REVERSE of member-name order (part1=slab, part2=cube).
        self._write_synthetic_real_h5(real_path, [shared_bag_id, shared_bag_id],
                                      extra_sdf=np.stack([sdf_cube, sdf_slab]))
        mesh_root = self.d / "mesh_root"
        self._zip_with(mesh_root / "Berlin" / "mesh", {
            member_names[0]: trimesh.exchange.obj.export_obj(slab).encode("ascii"),
            member_names[1]: trimesh.exchange.obj.export_obj(cube).encode("ascii"),
        })
        out_path = self.d / "centroids.h5"
        result = self._run(real_path, mesh_root, ["Berlin"], out_path=out_path)
        self.assertEqual(result["n_total"], 2)

        row_cube, row_slab = FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 1
        with h5py.File(out_path, "r") as f:
            by_row = {int(r): i for i, r in enumerate(f["row"][:])}
            self.assertAlmostEqual(float(f["x_m"][by_row[row_cube]]), 100.0, delta=0.5)
            self.assertAlmostEqual(float(f["y_m"][by_row[row_slab]]), 100.0, delta=0.5)

    def test_resuming_a_partially_committed_collision_group_does_not_corrupt_the_committed_row(
            self):
        """Regression: an earlier version passed only the still-PENDING rows of a collision
        group into `_resolve_collision` on resume, leaving the already-committed row's candidate
        mesh free to be re-claimed by the row still being resolved -- silently corrupting an
        already-correct prior result. Simulates "a prior run already committed row_cube" by
        seeding the output file directly (bypassing needing to engineer an actual partial run),
        then confirms a fresh `run()` call leaves that row's centroid untouched, resolves the
        other row correctly, and writes no duplicate."""
        long_prefix = "mesh/" + "y" * 60
        member_names = [f"{long_prefix}_part1.obj", f"{long_prefix}_part2.obj"]
        shared_bag_id = bag_id_for("Berlin", member_names[0])

        cube = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
        cube.apply_translation((100.0, 0.0, 1.5))
        slab = trimesh.creation.box(extents=(3.0, 3.0, 9.0))
        slab.apply_translation((0.0, 100.0, 4.5))
        sdf_cube, _, _ = building_to_sdf(cube, self.R)
        sdf_slab, _, _ = building_to_sdf(slab, self.R)

        real_path = self.d / "real.h5"
        row_cube, row_slab = FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 1
        self._write_synthetic_real_h5(real_path, [shared_bag_id, shared_bag_id],
                                      extra_sdf=np.stack([sdf_cube, sdf_slab]))
        mesh_root = self.d / "mesh_root"
        self._zip_with(mesh_root / "Berlin" / "mesh", {
            member_names[0]: trimesh.exchange.obj.export_obj(slab).encode("ascii"),
            member_names[1]: trimesh.exchange.obj.export_obj(cube).encode("ascii"),
        })

        out_path = self.d / "centroids.h5"
        SENTINEL_X, SENTINEL_Y = 100.0, 0.0  # row_cube's true, already-correct centroid
        with bsk.IncrementalRowWriter(out_path) as w:
            w.add(row=row_cube, x_m=np.float32(SENTINEL_X), y_m=np.float32(SENTINEL_Y),
                 ok=np.uint8(1), reason=b"")

        result = self._run(real_path, mesh_root, ["Berlin"], out_path=out_path)
        self.assertEqual(result["n_total"], 2)  # total committed rows in the file after this run

        with h5py.File(out_path, "r") as f:
            rows = [int(r) for r in f["row"][:]]
            self.assertEqual(sorted(rows), [row_cube, row_slab], "no duplicate row_cube entry")
            by_row = {r: i for i, r in enumerate(rows)}
            self.assertEqual(float(f["x_m"][by_row[row_cube]]), SENTINEL_X,
                            "the already-committed row's mesh must not be re-resolved/corrupted")
            self.assertEqual(float(f["y_m"][by_row[row_cube]]), SENTINEL_Y)
            self.assertAlmostEqual(float(f["y_m"][by_row[row_slab]]), 100.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
