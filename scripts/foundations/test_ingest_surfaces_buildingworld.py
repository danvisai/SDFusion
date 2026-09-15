"""Contract tests for #175's BuildingWorld isosurface extractor. Synthetic, fast, CPU, no
network, and no dependence on the (huge) production `real.h5` or BuildingWorld zips.
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
import scripts.foundations.ingest_surfaces_buildingworld as isb  # noqa: E402
from scripts.foundations.ingest_buildingworld import bag_id_for, source_key_for  # noqa: E402
from scripts.ingest_3dbag import building_to_sdf  # noqa: E402
from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, REAL_CORPUS_PATH, row_identity_sha256  # noqa: E402


def _box_obj_bytes(size=(4.0, 4.0, 3.0), origin=(0.0, 0.0, 0.0)) -> bytes:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.asarray(origin) + np.asarray(size) / 2.0)
    return trimesh.exchange.obj.export_obj(m).encode("ascii")


class TestWantedIds(unittest.TestCase):
    def test_filters_by_source_key_and_maps_bag_id_to_rows(self):
        source_key_col = np.array([b"bw:Berlin", b"bw:Boston", b"bw:Berlin", b""], dtype="S64")
        bag_id_col = np.array([b"Berlin#a.obj", b"Boston#b.obj", b"Berlin#c.obj", b"x"],
                              dtype="S64")
        want = isb._wanted_ids("Berlin", source_key_col, bag_id_col)
        self.assertEqual(want, {b"Berlin#a.obj": [0], b"Berlin#c.obj": [2]})

    def test_no_rows_for_an_unrepresented_city(self):
        source_key_col = np.array([b"bw:Berlin"], dtype="S64")
        bag_id_col = np.array([b"Berlin#a.obj"], dtype="S64")
        self.assertEqual(isb._wanted_ids("Tokyo", source_key_col, bag_id_col), {})

    def test_a_bag_id_shared_by_two_rows_collects_both(self):
        """Truncation collision (#175 postmortem, Perth): two DIFFERENT real.h5 rows can be
        stamped with the identical (truncated) bag_id."""
        source_key_col = np.array([b"bw:Perth", b"bw:Perth"], dtype="S64")
        bag_id_col = np.array([b"Perth#same", b"Perth#same"], dtype="S64")
        want = isb._wanted_ids("Perth", source_key_col, bag_id_col)
        self.assertEqual(want, {b"Perth#same": [0, 1]})


class TestGroupMembersByBagId(unittest.TestCase):
    def test_groups_only_wanted_bag_ids(self):
        want = {bag_id_for("Berlin", "mesh/a.obj"): [0]}
        names = ["mesh/a.obj", "mesh/not_wanted.obj"]
        groups = isb._group_members_by_bag_id("Berlin", names, want)
        self.assertEqual(groups, {bag_id_for("Berlin", "mesh/a.obj"): ["mesh/a.obj"]})

    def test_multiple_names_colliding_on_one_bag_id_are_grouped_together(self):
        long_prefix = "mesh/" + "x" * 60
        want = {bag_id_for("Perth", f"{long_prefix}_part1.obj"): [0]}
        names = [f"{long_prefix}_part1.obj", f"{long_prefix}_part2.obj"]
        # both truncate to the identical 64-byte bag_id
        self.assertEqual(bag_id_for("Perth", names[0]), bag_id_for("Perth", names[1]))
        groups = isb._group_members_by_bag_id("Perth", names, want)
        self.assertEqual(list(groups.values()), [names])


class TestLoadCorrectedMesh(unittest.TestCase):
    def test_applies_per_city_correction(self):
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/x.obj", _box_obj_bytes(size=(4.0, 4.0, 3.0)))
            from scripts.foundations.ingest_buildingworld import FEET_TO_M
            with zipfile.ZipFile(zpath) as zf:
                mesh = isb._load_corrected_mesh("Boston", "mesh/x.obj", zf)
            np.testing.assert_allclose(mesh.extents,
                                       np.array([4.0, 4.0, 3.0]) * FEET_TO_M, atol=1e-6)


class TestResolveCollision(unittest.TestCase):
    def test_matches_each_row_to_its_true_geometry_by_sdf(self):
        """Two candidates share a truncated bag_id; two rows share it too. Each row's STORED sdf
        matches exactly one candidate's recomputed sdf -- `_resolve_collision` must pair them
        correctly even though the name/bag_id gives no signal. Uses a cube vs. a 1:1:3 slab, not
        two differently-SIZED cubes: `building_to_sdf` normalises by each mesh's own extent, so
        same-shaped cubes of any size produce an IDENTICAL normalised sdf and could not be told
        apart by this match at all -- the aspect ratio is the signal that survives."""
        r = 8
        cube = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
        cube.apply_translation((1.5, 1.5, 1.5))
        slab = trimesh.creation.box(extents=(3.0, 3.0, 9.0))
        slab.apply_translation((1.5, 1.5, 4.5))
        sdf_cube, _, _ = building_to_sdf(cube, r)
        sdf_slab, _, _ = building_to_sdf(slab, r)

        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/cube.obj", trimesh.exchange.obj.export_obj(cube)
                           .encode("ascii"))
                zf.writestr("mesh/slab.obj", trimesh.exchange.obj.export_obj(slab).encode("ascii"))

            # rows 10 and 20 hold sdf_slab and sdf_cube respectively -- the OPPOSITE of file
            # order, so a name/order-based guess would get this wrong.
            real_sdf = {10: sdf_slab, 20: sdf_cube}
            with zipfile.ZipFile(zpath) as zf:
                out = isb._resolve_collision("Berlin", b"Berlin#collided", ["mesh/cube.obj",
                                                                            "mesh/slab.obj"],
                                             [10, 20], zf, real_sdf, r)
        self.assertEqual(set(out), {10, 20})
        # row 10 (sdf_slab) must resolve to the SLAB mesh, row 20 (sdf_cube) to the CUBE one.
        self.assertAlmostEqual(float(out[10].extents.max() / out[10].extents.min()), 3.0,
                               places=3)
        self.assertAlmostEqual(float(out[20].extents.max() / out[20].extents.min()), 1.0,
                               places=3)

    def test_a_low_confidence_pairing_is_rejected_not_accepted_as_a_last_resort(self):
        """Code-review finding on #175: the greedy match had no error threshold, so a candidate
        that was merely the LEAST-bad option left would still be accepted and logged as
        `[collision:resolved]`, contradicting this function's own "confident match" docstring
        claim. Here the only candidate in the zip is nothing like the row's stored geometry -- a
        threshold-free match would still "resolve" it (it's the only option); with the IoU floor
        it must be left unresolved instead."""
        r = 8
        cube = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
        cube.apply_translation((1.5, 1.5, 1.5))
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/cube.obj", trimesh.exchange.obj.export_obj(cube)
                           .encode("ascii"))
            # a constant, all-outside sdf: zero occupancy overlap with any real mesh candidate.
            real_sdf = {99: np.full((r, r, r), 5.0, np.float32)}
            with zipfile.ZipFile(zpath) as zf:
                out = isb._resolve_collision("Berlin", b"Berlin#lowconf", ["mesh/cube.obj"],
                                             [99], zf, real_sdf, r)
        self.assertNotIn(99, out)

    def test_min_iou_is_configurable_and_a_lenient_bar_accepts_the_same_pairing(self):
        """Same scenario as the rejection test above, but with `min_iou=0.0` -- confirms the
        rejection is actually the threshold firing, not some other reason the pairing failed."""
        r = 8
        cube = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
        cube.apply_translation((1.5, 1.5, 1.5))
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/cube.obj", trimesh.exchange.obj.export_obj(cube)
                           .encode("ascii"))
            real_sdf = {99: np.full((r, r, r), 5.0, np.float32)}
            with zipfile.ZipFile(zpath) as zf:
                out = isb._resolve_collision("Berlin", b"Berlin#lowconf", ["mesh/cube.obj"],
                                             [99], zf, real_sdf, r, min_iou=0.0)
        self.assertIn(99, out)

    def test_a_row_with_no_candidate_left_is_dropped_not_guessed(self):
        """Two rows share a bag_id but only ONE physical candidate is in the zip -- the second
        row must be reported as unresolved, not silently paired with a mesh that isn't its own."""
        r = 8
        m = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
        m.apply_translation((1.5, 1.5, 1.5))
        sdf_m, _, _ = building_to_sdf(m, r)
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/only.obj", trimesh.exchange.obj.export_obj(m).encode("ascii"))
            real_sdf = {10: sdf_m, 20: np.full((r, r, r), 5.0, np.float32)}
            with zipfile.ZipFile(zpath) as zf:
                out = isb._resolve_collision("Berlin", b"Berlin#x", ["mesh/only.obj"],
                                             [10, 20], zf, real_sdf, r)
        self.assertEqual(set(out), {10})
        self.assertNotIn(20, out)


class TestIncrementalSurfaceWriter(unittest.TestCase):
    """Code-review finding on #175: `run()` used to buffer everything in memory and write once,
    with no resume. Mirrors `test_ingest_buildingworld.py::TestIncrementalCityWriterAndStaging`'s
    coverage shape, adapted for ragged verts/faces."""

    def _row(self, bag_id: bytes, row: int, n_verts: int = 4, n_faces: int = 2):
        verts = np.arange(n_verts * 3, dtype=np.float32).reshape(n_verts, 3)
        faces = np.arange(n_faces * 3, dtype=np.int32).reshape(n_faces, 3) % n_verts
        return dict(bag_id=bag_id, source_key=b"bw:Test", row=row, verts=verts, faces=faces)

    def test_write_flush_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "surfaces.h5"
            with isb.IncrementalSurfaceWriter(path, flush_every=2) as w:
                for i in range(5):
                    r = self._row(f"bag{i}".encode(), i, n_verts=3 + i, n_faces=1 + i)
                    w.add(**r)
                    w.maybe_flush()
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 5)
                self.assertEqual(int(f.attrs["committed_rows"]), 5)
                self.assertEqual(f["vert_offset"][:].tolist(), [0, 3, 7, 12, 18, 25])
                self.assertEqual(f["face_offset"][:].tolist(), [0, 1, 3, 6, 10, 15])
                self.assertEqual(f["verts"].shape[0], 25)
                self.assertEqual(f["faces"].shape[0], 15)

    def test_resume_skips_already_committed_bag_ids_and_appends_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "surfaces.h5"
            with isb.IncrementalSurfaceWriter(path, flush_every=100) as w:
                for i in range(3):
                    w.add(**self._row(f"bag{i}".encode(), i))
                    w.maybe_flush()
            with isb.IncrementalSurfaceWriter(path, resume=True) as w2:
                self.assertTrue(w2.already_done(b"bag0"))
                self.assertFalse(w2.already_done(b"bag3"))
                for i in range(3, 6):
                    w2.add(**self._row(f"bag{i}".encode(), i))
                    w2.maybe_flush()
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 6)
                self.assertEqual(sorted(int(r) for r in f["row"][:]), list(range(6)))
                # offsets stay one contiguous cumulative sequence across the resume boundary
                vo = f["vert_offset"][:]
                self.assertTrue((np.diff(vo) > 0).all())
                self.assertEqual(vo[-1], f["verts"].shape[0])

    def test_no_resume_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "surfaces.h5"
            with isb.IncrementalSurfaceWriter(path) as w:
                w.add(**self._row(b"bag0", 0))
            with isb.IncrementalSurfaceWriter(path, resume=False) as w2:
                w2.add(**self._row(b"bag1", 1))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 1)
                self.assertEqual(int(f["row"][0]), 1)

    def test_schema_mismatch_refuses_to_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "surfaces.h5"
            with h5py.File(path, "w") as f:
                f.attrs["schema_version"] = 999
                f.attrs["committed_rows"] = 0
            with self.assertRaises(SystemExit):
                isb.IncrementalSurfaceWriter(path, resume=True)

    def test_a_flush_mid_call_never_splits_one_bag_ids_rows_across_it(self):
        """The bug this fix closes: `add()` alone used to auto-flush, so a flush landing between
        two `add()` calls for the SAME bag_id (a multi-row collision group) would mark that
        bag_id `already_done` after committing only the first row -- a resume would then skip the
        group entirely and the rest of its rows would be lost forever. `flush_every=1` is the
        worst case: if `add()` still auto-flushed, this would fail immediately."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "surfaces.h5"
            with isb.IncrementalSurfaceWriter(path, flush_every=1) as w:
                w.add(**self._row(b"shared", 10))
                w.add(**self._row(b"shared", 20))   # no maybe_flush() between these two calls
                w.maybe_flush()                      # only now, at the group boundary
            with h5py.File(path, "r") as f:
                self.assertEqual(sorted(int(r) for r in f["row"][:]), [10, 20])


@unittest.skipUnless(REAL_CORPUS_PATH.exists(), "real corpus metadata not available")
class TestRunEndToEnd(unittest.TestCase):
    """Mirrors test_ingest_buildingworld.TestCombineIntoReal's fixture: `open_real_corpus`
    checks the frozen prefix against a hardcoded hash (#162), so a synthetic `real.h5` can only
    pass by carrying the real corpus's own (bag_id, height_m) for that prefix. Only that small
    metadata slice is read; the SDF volumes never are."""

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

    def test_run_writes_one_merged_h5_across_cities(self):
        real_path = self.d / "real.h5"
        member_names = ["mesh/a.obj", "mesh/b.obj"]
        extra_bag_ids = [bag_id_for("Berlin", n) for n in member_names]
        self._write_synthetic_real_h5(real_path, extra_bag_ids)
        with h5py.File(real_path, "r") as f:
            baseline = row_identity_sha256(f["bag_id"][:FROZEN_SPLIT_N_TOTAL],
                                           f["height_m"][:FROZEN_SPLIT_N_TOTAL])

        city_dir = self.d / "mesh_root" / "Berlin" / "mesh"
        city_dir.mkdir(parents=True)
        with zipfile.ZipFile(city_dir / "mesh.zip", "w") as zf:
            for n in member_names:
                zf.writestr(n, _box_obj_bytes())
            zf.writestr("mesh/not_kept.obj", _box_obj_bytes())

        import scripts.foundations.profile_buildingworld_meshes as pbm
        old_root, old_h5 = pbm.MESH_ROOT, isb.H5
        pbm.MESH_ROOT, isb.H5 = self.d / "mesh_root", real_path
        try:
            result = isb.run(["Berlin"], limit=0, out_path=self.d / "surfaces_buildingworld.h5")
        finally:
            pbm.MESH_ROOT, isb.H5 = old_root, old_h5

        with h5py.File(real_path, "r") as f:
            self.assertEqual(row_identity_sha256(f["bag_id"][:FROZEN_SPLIT_N_TOTAL],
                                                 f["height_m"][:FROZEN_SPLIT_N_TOTAL]), baseline)

        self.assertEqual(result["n_total"], 2)
        self.assertEqual(result["per_city"], {"Berlin": 2})
        out = Path(result["out_path"])
        self.assertTrue(out.exists())
        with h5py.File(out, "r") as f:
            self.assertEqual(set(f.keys()),
                             {"verts", "faces", "vert_offset", "face_offset", "row", "bag_id",
                              "source_key"})
            self.assertEqual(f["row"].shape[0], 2)
            self.assertEqual(sorted(int(r) for r in f["row"][:]),
                             [FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 1])
            self.assertEqual(set(bytes(b) for b in f["bag_id"][:]), set(extra_bag_ids))
            self.assertTrue(all(bytes(s) == source_key_for("Berlin").encode("ascii")
                                for s in f["source_key"][:]))
            # A box's Frame-N mesh is a watertight 8-vertex/12-face solid, per mesh.
            self.assertEqual(f["vert_offset"][:].tolist(), [0, 8, 16])
            self.assertEqual(f["face_offset"][:].tolist(), [0, 12, 24])

    def test_run_resolves_a_truncation_collision_end_to_end(self):
        """Two real.h5 rows share one truncated bag_id (a Perth-shaped collision); `run()` must
        still assign each row its own, correctly-matched mesh rather than dropping one or
        duplicating the other. Frame-N normalises out absolute size, so the two candidates use
        different ASPECT RATIOS (a cube vs. a 1:1:3 slab) -- a feature that survives
        normalisation -- rather than different sizes, so a wrong pairing is still detectable in
        the OUTPUT mesh, not just in `_resolve_collision`'s raw return value."""
        long_prefix = "mesh/" + "y" * 60
        member_names = [f"{long_prefix}_part1.obj", f"{long_prefix}_part2.obj"]
        self.assertEqual(bag_id_for("Berlin", member_names[0]),
                         bag_id_for("Berlin", member_names[1]))
        shared_bag_id = bag_id_for("Berlin", member_names[0])

        cube = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
        cube.apply_translation((1.5, 1.5, 1.5))
        slab = trimesh.creation.box(extents=(3.0, 3.0, 9.0))
        slab.apply_translation((1.5, 1.5, 4.5))
        sdf_cube, _, _ = building_to_sdf(cube, self.R)
        sdf_slab, _, _ = building_to_sdf(slab, self.R)

        real_path = self.d / "real.h5"
        # row order (cube, slab) is the REVERSE of member-name order (part1=slab, part2=cube) --
        # a name/order-based guess would mis-pair these.
        self._write_synthetic_real_h5(real_path, [shared_bag_id, shared_bag_id],
                                      extra_sdf=np.stack([sdf_cube, sdf_slab]))

        city_dir = self.d / "mesh_root" / "Berlin" / "mesh"
        city_dir.mkdir(parents=True)
        with zipfile.ZipFile(city_dir / "mesh.zip", "w") as zf:
            zf.writestr(member_names[0], trimesh.exchange.obj.export_obj(slab).encode("ascii"))
            zf.writestr(member_names[1], trimesh.exchange.obj.export_obj(cube).encode("ascii"))

        import scripts.foundations.profile_buildingworld_meshes as pbm
        old_root, old_h5 = pbm.MESH_ROOT, isb.H5
        pbm.MESH_ROOT, isb.H5 = self.d / "mesh_root", real_path
        try:
            result = isb.run(["Berlin"], limit=0, out_path=self.d / "surfaces_buildingworld.h5")
        finally:
            pbm.MESH_ROOT, isb.H5 = old_root, old_h5

        self.assertEqual(result["n_total"], 2)
        row_cube, row_slab = FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 1
        with h5py.File(result["out_path"], "r") as f:
            row_to_idx = {int(r): i for i, r in enumerate(f["row"][:])}
            self.assertEqual(set(row_to_idx), {row_cube, row_slab})

            def aspect_ratio(i):
                a, b = int(f["vert_offset"][i]), int(f["vert_offset"][i + 1])
                ext = np.ptp(np.asarray(f["verts"][a:b]), axis=0)
                return float(ext.max() / ext.min())

            # A slab's aspect ratio (~3:1) survives Frame-N normalisation; a cube's is ~1:1.
            self.assertAlmostEqual(aspect_ratio(row_to_idx[row_cube]), 1.0, delta=0.05)
            self.assertAlmostEqual(aspect_ratio(row_to_idx[row_slab]), 3.0, delta=0.3)

    def test_run_raises_when_nothing_is_recovered(self):
        real_path = self.d / "real.h5"
        self._write_synthetic_real_h5(real_path, [])
        (self.d / "mesh_root" / "Berlin" / "mesh").mkdir(parents=True)
        with zipfile.ZipFile(self.d / "mesh_root" / "Berlin" / "mesh" / "mesh.zip", "w") as zf:
            zf.writestr("mesh/x.obj", _box_obj_bytes())

        import scripts.foundations.profile_buildingworld_meshes as pbm
        old_root, old_h5 = pbm.MESH_ROOT, isb.H5
        pbm.MESH_ROOT, isb.H5 = self.d / "mesh_root", real_path
        try:
            with self.assertRaises(SystemExit):
                isb.run(["Berlin"], limit=0, out_path=self.d / "out.h5")
        finally:
            pbm.MESH_ROOT, isb.H5 = old_root, old_h5

    def test_run_resumes_a_partial_run_without_dropping_or_duplicating_rows(self):
        """Code-review finding on #175: `run()` used to write once at the end with no resume --
        an interrupted run lost everything. `--limit` (a per-CALL cap on newly-processed rows)
        simulates a run that stopped partway; a second call with the same `--out` must pick up
        exactly where it left off."""
        real_path = self.d / "real.h5"
        member_names = ["mesh/a.obj", "mesh/b.obj", "mesh/c.obj"]
        extra_bag_ids = [bag_id_for("Berlin", n) for n in member_names]
        self._write_synthetic_real_h5(real_path, extra_bag_ids)

        city_dir = self.d / "mesh_root" / "Berlin" / "mesh"
        city_dir.mkdir(parents=True)
        with zipfile.ZipFile(city_dir / "mesh.zip", "w") as zf:
            for n in member_names:
                zf.writestr(n, _box_obj_bytes())

        out_path = self.d / "surfaces_buildingworld.h5"
        import scripts.foundations.profile_buildingworld_meshes as pbm
        old_root, old_h5 = pbm.MESH_ROOT, isb.H5
        pbm.MESH_ROOT, isb.H5 = self.d / "mesh_root", real_path
        try:
            first = isb.run(["Berlin"], limit=1, out_path=out_path)
            self.assertEqual(first["n_total"], 1)
            second = isb.run(["Berlin"], limit=0, out_path=out_path, resume=True)
        finally:
            pbm.MESH_ROOT, isb.H5 = old_root, old_h5

        self.assertEqual(second["n_total"], 3)
        with h5py.File(out_path, "r") as f:
            self.assertEqual(f["row"].shape[0], 3)
            self.assertEqual(sorted(int(r) for r in f["row"][:]),
                             list(range(FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 3)))
            self.assertEqual(len(set(bytes(b) for b in f["bag_id"][:])), 3)  # no duplicates
            self.assertEqual(f["vert_offset"][-1], f["verts"].shape[0])
            self.assertEqual(f["face_offset"][-1], f["faces"].shape[0])


class TestDoraFrozenGateRegistration(unittest.TestCase):
    def test_buildingworld_is_registered(self):
        from scripts.foundations.dora_frozen_gate import SOURCES
        self.assertEqual(SOURCES["buildingworld"], "BW")

    def test_load_surfaces_finds_a_registered_buildingworld_file(self):
        import scripts.foundations.dora_frozen_gate as dfg

        with tempfile.TemporaryDirectory() as d:
            old_surf = dfg.SURF
            dfg.SURF = Path(d)
            try:
                m = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
                v = np.ascontiguousarray(m.vertices, np.float32)
                fc = np.ascontiguousarray(m.faces, np.int32)
                with h5py.File(Path(d) / "surfaces_buildingworld.h5", "w") as f:
                    f.create_dataset("verts", data=v)
                    f.create_dataset("faces", data=fc)
                    f.create_dataset("vert_offset", data=np.array([0, len(v)], np.int64))
                    f.create_dataset("face_offset", data=np.array([0, len(fc)], np.int64))
                    f.create_dataset("row", data=np.array([42], np.int32))
                surf = dfg.load_surfaces()
            finally:
                dfg.SURF = old_surf
            self.assertIn(42, surf)
            self.assertEqual(surf[42][2], "buildingworld")


if __name__ == "__main__":
    unittest.main(verbosity=2)
