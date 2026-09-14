"""Contract tests for #174's BuildingWorld ingester. Synthetic, fast, CPU, no network.

Covers the pure per-city/per-row decisions #174 applies (CRS/units correction per #165, the
watertightness gate per #166, provenance construction per #167, the dedup exclusion mechanism per
#160) and the staging-writer/combine-into-real.h5 machinery, against tiny synthetic meshes and a
tiny synthetic real.h5 -- never the production corpus or the real BuildingWorld zips.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import h5py
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.ingest_buildingworld import (  # noqa: E402
    CITY_Z_REFERENCE, IncrementalCityWriter, STAGING_COLUMNS, apply_geometric_correction,
    bag_id_for, city_slug, class_label_for, classify_defect, combine_into_real,
    load_dedup_excluded_ids, process_member, source_key_for, stage_city, z_reference_check,
)
from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, REAL_CORPUS_PATH, row_identity_sha256


def _box_obj_bytes(size=(4.0, 4.0, 3.0), origin=(0.0, 0.0, 0.0)) -> bytes:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.asarray(origin) + np.asarray(size) / 2.0)
    return trimesh.exchange.obj.export_obj(m).encode("ascii")


def _open_box_obj_bytes(size=(4.0, 4.0, 3.0)) -> bytes:
    """A box with its top face removed -- open boundary near z_max, i.e. NOT floor_open."""
    m = trimesh.creation.box(extents=size)
    # top face(s): faces whose vertices are all at the max z
    zmax = m.vertices[:, 2].max()
    keep = ~np.all(np.isclose(m.vertices[m.faces][:, :, 2], zmax), axis=1)
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return trimesh.exchange.obj.export_obj(m).encode("ascii")


class TestCitySlugAndProvenance(unittest.TestCase):
    def test_slug_strips_spaces(self):
        self.assertEqual(city_slug("Cape Town"), "CapeTown")
        self.assertEqual(city_slug("Greater Geelong"), "GreaterGeelong")
        self.assertEqual(city_slug("Berlin"), "Berlin")

    def test_source_key_matches_167s_convention(self):
        self.assertEqual(source_key_for("Berlin"), "bw:Berlin")
        self.assertEqual(source_key_for("Cape Town"), "bw:CapeTown")

    def test_class_label_is_explicitly_truncated_to_16_bytes(self):
        label = class_label_for("Greater Geelong")
        self.assertLessEqual(len(label), 16)
        self.assertEqual(label, b"BW_GreaterGeelon")

    def test_bag_id_encodes_city_and_member_and_is_bounded_to_64_bytes(self):
        bid = bag_id_for("Berlin", "mesh/DEBE00YY10z0001A.obj")
        self.assertLessEqual(len(bid), 64)
        self.assertTrue(bid.startswith(b"Berlin#mesh/"))

    def test_every_ingestable_citys_slug_has_a_z_reference_entry(self):
        from scripts.foundations.ingest_buildingworld import INGESTABLE_CITIES

        for city in INGESTABLE_CITIES:
            self.assertIn(city_slug(city), CITY_Z_REFERENCE, city)


class TestGeometricCorrection(unittest.TestCase):
    def _mesh(self, verts, faces):
        return trimesh.Trimesh(np.asarray(verts, float), np.asarray(faces, int), process=False)

    def test_feet_cities_scale_isotropically(self):
        m = trimesh.creation.box(extents=(3937.0, 3937.0, 3937.0))
        apply_geometric_correction(m, "Boston", "mesh/x.obj")
        np.testing.assert_allclose(m.extents, [1200.0, 1200.0, 1200.0], atol=1e-6)

    def test_philadelphia_only_scales_the_feet_subfolder(self):
        m1 = trimesh.creation.box(extents=(3937.0, 3937.0, 3937.0))
        apply_geometric_correction(m1, "Philadelphia", "mesh/2010_ph_downtown/1.obj")
        np.testing.assert_allclose(m1.extents, [1200.0, 1200.0, 1200.0], atol=1e-6)

        m2 = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        apply_geometric_correction(m2, "Philadelphia", "mesh/2015_scene/1.obj")
        np.testing.assert_allclose(m2.extents, [10.0, 10.0, 10.0], atol=1e-6)

    def test_mississauga_reprojects_xy_only_leaves_z(self):
        import pyproj

        m = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        m.apply_translation((-8_865_940.0, 5_402_057.0, 178.0))
        z_before = m.vertices[:, 2].copy()
        apply_geometric_correction(m, "Mississauga", "mesh/x.obj")
        np.testing.assert_allclose(m.vertices[:, 2], z_before)
        # Web Mercator -> UTM17N at this latitude should land near Mississauga's real easting/
        # northing (hundreds of km, not millions) -- a coarse sanity bound on the transform firing.
        to_utm = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:26917", always_xy=True)
        ex, ey = to_utm.transform(-8_865_940.0, 5_402_057.0)
        self.assertAlmostEqual(float(m.vertices[:, 0].mean()) - 1.0, ex - 1.0, delta=1.0)
        self.assertAlmostEqual(float(m.vertices[:, 1].mean()) - 1.0, ey - 1.0, delta=1.0)

    def test_other_cities_are_left_alone(self):
        m = trimesh.creation.box(extents=(5.0, 5.0, 5.0))
        v_before = m.vertices.copy()
        apply_geometric_correction(m, "Berlin", "mesh/x.obj")
        np.testing.assert_allclose(m.vertices, v_before)

    def test_yarra_richmond_mirrors_y_and_restores_outward_winding(self):
        m = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        vol_before = m.volume
        normal_before = m.face_normals[0].copy()
        apply_geometric_correction(m, "Yarra", "mesh/richmond/richmond_00001.obj")
        # A correct mirror-plus-rewind keeps the mesh a well-formed, outward-facing solid: the
        # (unsigned) volume is preserved and no face silently ends up facing inward.
        self.assertAlmostEqual(abs(m.volume), abs(vol_before), places=6)
        self.assertTrue(m.is_watertight)
        self.assertGreater(np.dot(m.face_normals[0], [normal_before[0], -normal_before[1],
                                                       normal_before[2]]), 0.0)

    def test_yarra_other_suburbs_are_untouched(self):
        m = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        v_before = m.vertices.copy()
        apply_geometric_correction(m, "Yarra", "mesh/burnley/burnley_00001.obj")
        np.testing.assert_allclose(m.vertices, v_before)


class TestDefectClassification(unittest.TestCase):
    def test_watertight_box_is_watertight(self):
        m = trimesh.load(io.BytesIO(_box_obj_bytes()), file_type="obj")
        self.assertEqual(classify_defect(m), "watertight")

    def test_open_top_box_is_floor_open_or_scattered_but_not_watertight(self):
        m = trimesh.load(io.BytesIO(_open_box_obj_bytes()), file_type="obj")
        self.assertIn(classify_defect(m), {"floor_open", "scattered", "mixed"})


class TestZReferenceCheck(unittest.TestCase):
    def test_asl_city_within_band_passes(self):
        rep = z_reference_check("Berlin", [30.0, 35.0, 40.0])
        self.assertTrue(rep["ok"])

    def test_asl_city_far_outside_band_raises(self):
        with self.assertRaises(ValueError):
            z_reference_check("Berlin", [5000.0, 5010.0, 5020.0])

    def test_relative_zero_city_near_zero_passes(self):
        rep = z_reference_check("Melbourne", [0.0, 0.01, -0.01])
        self.assertTrue(rep["ok"])

    def test_relative_zero_city_far_from_zero_raises(self):
        with self.assertRaises(ValueError):
            z_reference_check("Melbourne", [40.0, 41.0])

    def test_unreliable_and_undocumented_cities_never_raise(self):
        for city in ("Adelaide", "Perth", "San Francisco", "Wellington"):
            rep = z_reference_check(city, [-9999.0, 9999.0])
            self.assertTrue(rep["ok"])

    def test_empty_sample_does_not_raise(self):
        rep = z_reference_check("Berlin", [])
        self.assertTrue(rep["ok"])


class TestDedupExclusion(unittest.TestCase):
    def test_loads_candidate_ids_per_pair(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gate.json"
            p.write_text(json.dumps(dict(pairs=dict(
                tokyo_vs_plateau=dict(matches=[dict(candidate_id="bldg_1")]),
                berlin_vs_nrw=dict(matches=[]),
            ))))
            out = load_dedup_excluded_ids(p)
            self.assertEqual(out["Tokyo"], {"bldg_1"})
            self.assertEqual(out["Berlin"], set())

    def test_missing_pair_yields_empty_set(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gate.json"
            p.write_text(json.dumps(dict(pairs={})))
            out = load_dedup_excluded_ids(p)
            self.assertEqual(out["Tokyo"], set())
            self.assertEqual(out["Berlin"], set())


class TestProcessMember(unittest.TestCase):
    def test_keeps_a_plausible_watertight_building(self):
        rec = process_member("Berlin", "mesh/x.obj", _box_obj_bytes(size=(6, 6, 8)), r=16,
                             dedup_excluded=set(), min_h=2.5, max_ext=90.0, min_fp=1)
        self.assertEqual(rec["status"], "kept")
        self.assertEqual(rec["source_id"], -1)
        self.assertEqual(rec["source_key"], b"bw:Berlin")
        self.assertEqual(rec["defect_class"], b"watertight")
        self.assertEqual(rec["sdf"].shape, (16, 16, 16))

    def test_dedup_excluded_id_is_skipped(self):
        rec = process_member("Tokyo", "mesh/bldg_1.obj", _box_obj_bytes(), r=16,
                             dedup_excluded={"bldg_1"}, min_h=2.5, max_ext=90.0, min_fp=1)
        self.assertEqual(rec, dict(status="skip", reason="dedup_excluded"))

    def test_too_short_building_is_skipped_on_extent(self):
        rec = process_member("Berlin", "mesh/x.obj", _box_obj_bytes(size=(4, 4, 0.5)), r=16,
                             dedup_excluded=set(), min_h=2.5, max_ext=90.0, min_fp=1)
        self.assertEqual(rec["status"], "skip")
        self.assertEqual(rec["reason"], "extent")

    def test_oversized_building_is_skipped_on_extent_and_flagged(self):
        rec = process_member("Berlin", "mesh/x.obj", _box_obj_bytes(size=(120, 8, 8)), r=16,
                             dedup_excluded=set(), min_h=2.5, max_ext=90.0, min_fp=1)
        self.assertEqual(rec["status"], "skip")
        self.assertEqual(rec["reason"], "extent")
        self.assertTrue(rec["extent_gt_90"])

    def test_feet_city_extent_filter_applies_to_corrected_units(self):
        # 300 US survey feet ~= 91.4 m: over the 90 m cutoff only AFTER the feet->m correction.
        rec = process_member("Boston", "mesh/x.obj", _box_obj_bytes(size=(300, 20, 20)), r=16,
                             dedup_excluded=set(), min_h=2.5, max_ext=90.0, min_fp=1)
        self.assertEqual(rec["status"], "skip")
        self.assertEqual(rec["reason"], "extent")

    def test_garbage_bytes_are_a_load_error_not_a_crash(self):
        rec = process_member("Berlin", "mesh/x.obj", b"not an obj file", r=16,
                             dedup_excluded=set(), min_h=2.5, max_ext=90.0, min_fp=1)
        self.assertEqual(rec["status"], "skip")
        self.assertEqual(rec["reason"], "load_error")


class TestIncrementalCityWriterAndStaging(unittest.TestCase):
    def _row(self, i, r=8):
        sdf = np.full((r, r, r), -1.0 if i % 2 == 0 else 1.0, np.float32)
        fp = np.ones((r, r), np.uint8)
        return dict(sdf=sdf, footprint=fp, height_m=np.float32(3.0 + i),
                   style_id=np.int32(8), source_id=np.int32(-1),
                   class_label=b"BW_Test", bag_id=f"Test#{i}".encode()[:64],
                   source_key=b"bw:Test", defect_class=b"watertight")

    def test_write_flush_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "Test.h5"
            with IncrementalCityWriter(path, r=8, flush_every=2) as w:
                for i in range(5):
                    w.add(**self._row(i))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["bag_id"].shape[0], 5)
                self.assertEqual(int(f.attrs["committed_rows"]), 5)
                np.testing.assert_array_equal(f["height_m"][:], np.arange(3, 8, dtype=np.float32))

    def test_resume_skips_already_committed_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "Test.h5"
            with IncrementalCityWriter(path, r=8, flush_every=100) as w:
                for i in range(3):
                    w.add(**self._row(i))
            with IncrementalCityWriter(path, r=8, resume=True) as w2:
                self.assertTrue(w2.already_done(b"Test#0"))
                self.assertFalse(w2.already_done(b"Test#3"))
                for i in range(3, 6):
                    w2.add(**self._row(i))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["bag_id"].shape[0], 6)
                np.testing.assert_array_equal(sorted(int(b.decode().split("#")[1])
                                                     for b in f["bag_id"][:]), list(range(6)))

    def test_no_resume_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "Test.h5"
            with IncrementalCityWriter(path, r=8) as w:
                w.add(**self._row(0))
            with IncrementalCityWriter(path, r=8, resume=False) as w2:
                w2.add(**self._row(1))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["bag_id"].shape[0], 1)


@unittest.skipUnless(REAL_CORPUS_PATH.exists(), "real corpus metadata not available")
class TestCombineIntoReal(unittest.TestCase):
    """`assert_frozen_corpus` checks against a HARDCODED baseline hash (#162) -- a synthetic
    corpus can only pass it by carrying the real corpus's own (bag_id, height_m) for the frozen
    prefix, exactly like `test_frozen_corpus.py`'s own acceptance fixtures. Only that 2.4 MB of
    metadata is read; the 32 GB of SDF volumes never is."""

    @classmethod
    def setUpClass(cls):
        with h5py.File(REAL_CORPUS_PATH, "r") as f:
            cls.ids = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
            cls.heights = f["height_m"][:FROZEN_SPLIT_N_TOTAL]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def _write_synthetic_real_h5(self, path: Path, r: int = 2) -> None:
        n = FROZEN_SPLIT_N_TOTAL
        with h5py.File(path, "w") as f:
            f.create_dataset("sdf", data=np.zeros((n, r, r, r), np.float32))
            f.create_dataset("footprint", data=np.ones((n, r, r), np.uint8))
            f.create_dataset("height_m", data=self.heights)
            f.create_dataset("style_id", data=np.full(n, 8, np.int32))
            f.create_dataset("source_id", data=np.zeros(n, np.int32))
            f.create_dataset("class_label", data=np.array([b"X"] * n, dtype="S16"))
            f.create_dataset("bag_id", data=self.ids)

    def test_combine_appends_and_preserves_frozen_prefix(self):
        real_path = self.d / "real.h5"
        self._write_synthetic_real_h5(real_path)
        with h5py.File(real_path, "r") as f:
            baseline = row_identity_sha256(f["bag_id"][:], f["height_m"][:])

        staged_path = self.d / "Berlin.h5"
        with IncrementalCityWriter(staged_path, r=2, flush_every=100) as w:
            w.add(sdf=np.ones((2, 2, 2), np.float32), footprint=np.ones((2, 2), np.uint8),
                 height_m=np.float32(99.0), style_id=np.int32(8), source_id=np.int32(-1),
                 class_label=b"BW_Berlin", bag_id=b"Berlin#mesh/x.obj",
                 source_key=b"bw:Berlin", defect_class=b"watertight")

        out_path = self.d / "real_combined.h5"
        result = combine_into_real(real_path, [staged_path], out_path)
        self.assertEqual(result["n_old"], FROZEN_SPLIT_N_TOTAL)
        self.assertEqual(result["n_new"], 1)
        self.assertEqual(result["n_total"], FROZEN_SPLIT_N_TOTAL + 1)

        with h5py.File(out_path, "r") as f:
            self.assertEqual(row_identity_sha256(f["bag_id"][:FROZEN_SPLIT_N_TOTAL],
                                                 f["height_m"][:FROZEN_SPLIT_N_TOTAL]), baseline)
            self.assertEqual(f["bag_id"][FROZEN_SPLIT_N_TOTAL], b"Berlin#mesh/x.obj")
            self.assertEqual(int(f["source_id"][FROZEN_SPLIT_N_TOTAL]), -1)
            self.assertEqual(f["source_key"][FROZEN_SPLIT_N_TOTAL], b"bw:Berlin")
            self.assertEqual(f["source_key"][0], b"")  # #167: historical rows not retrofitted
            self.assertEqual(f["defect_class"][FROZEN_SPLIT_N_TOTAL], b"watertight")
            self.assertEqual(float(f["height_m"][FROZEN_SPLIT_N_TOTAL]), 99.0)

    def test_combine_tolerates_a_city_that_staged_zero_rows(self):
        """A city whose whole sample fails every gate never calls `IncrementalCityWriter.add`, so
        its staged file has no datasets at all (only the schema/committed_rows attrs) -- this must
        not be treated as an error, and must not contribute any rows."""
        real_path = self.d / "real.h5"
        self._write_synthetic_real_h5(real_path)
        empty_staged = self.d / "Philadelphia.h5"
        with IncrementalCityWriter(empty_staged, r=2, flush_every=100):
            pass  # no .add() calls -- exactly what a zero-kept city produces

        kept_staged = self.d / "Berlin.h5"
        with IncrementalCityWriter(kept_staged, r=2, flush_every=100) as w:
            w.add(sdf=np.ones((2, 2, 2), np.float32), footprint=np.ones((2, 2), np.uint8),
                 height_m=np.float32(5.0), style_id=np.int32(8), source_id=np.int32(-1),
                 class_label=b"BW_Berlin", bag_id=b"Berlin#a", source_key=b"bw:Berlin",
                 defect_class=b"watertight")

        result = combine_into_real(real_path, [empty_staged, kept_staged], self.d / "out.h5")
        self.assertEqual(result["n_new"], 1)
        self.assertEqual(result["n_total"], FROZEN_SPLIT_N_TOTAL + 1)

    def test_combine_never_mutates_the_source_file(self):
        real_path = self.d / "real.h5"
        self._write_synthetic_real_h5(real_path)
        before = real_path.stat().st_size
        staged_path = self.d / "Berlin.h5"
        with IncrementalCityWriter(staged_path, r=2, flush_every=100) as w:
            w.add(sdf=np.ones((2, 2, 2), np.float32), footprint=np.ones((2, 2), np.uint8),
                 height_m=np.float32(1.0), style_id=np.int32(8), source_id=np.int32(-1),
                 class_label=b"BW_Berlin", bag_id=b"Berlin#a", source_key=b"bw:Berlin",
                 defect_class=b"watertight")
        combine_into_real(real_path, [staged_path], self.d / "out.h5")
        self.assertEqual(real_path.stat().st_size, before)

    def test_combine_rejects_a_tampered_frozen_prefix(self):
        real_path = self.d / "real.h5"
        self._write_synthetic_real_h5(real_path)
        with h5py.File(real_path, "a") as f:
            f["bag_id"][0] = b"tampered"
        staged_path = self.d / "Berlin.h5"
        with IncrementalCityWriter(staged_path, r=2, flush_every=100) as w:
            w.add(sdf=np.ones((2, 2, 2), np.float32), footprint=np.ones((2, 2), np.uint8),
                 height_m=np.float32(1.0), style_id=np.int32(8), source_id=np.int32(-1),
                 class_label=b"BW_Berlin", bag_id=b"Berlin#a", source_key=b"bw:Berlin",
                 defect_class=b"watertight")
        with self.assertRaises(ValueError):
            combine_into_real(real_path, [staged_path], self.d / "out.h5")
        self.assertFalse((self.d / "out.h5").exists())


class TestStageCityEndToEnd(unittest.TestCase):
    def test_stage_city_against_a_synthetic_zip(self):
        with tempfile.TemporaryDirectory() as d:
            city_dir = Path(d) / "mesh_root" / "Testville" / "mesh"
            city_dir.mkdir(parents=True)
            zpath = city_dir / "mesh.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("mesh/good_1.obj", _box_obj_bytes(size=(6, 6, 8)))
                zf.writestr("mesh/too_flat.obj", _box_obj_bytes(size=(6, 6, 0.5)))

            import scripts.foundations.ingest_buildingworld as ibw
            import scripts.foundations.profile_buildingworld_meshes as pbm
            ibw.CITY_SLUG["Testville"] = "Testville"
            ibw.CITY_Z_REFERENCE["Testville"] = ("undocumented", None)
            old_root = pbm.MESH_ROOT
            pbm.MESH_ROOT = Path(d) / "mesh_root"
            try:
                out_dir = Path(d) / "staging"
                summary = ibw.stage_city("Testville", r=16, limit=0, min_h=2.5, max_ext=90.0,
                                        min_fp=1, dedup_excluded={}, z_sanity_sample=10,
                                        out_dir=out_dir)
            finally:
                pbm.MESH_ROOT = old_root

            self.assertEqual(summary["counters"].get("kept"), 1)
            self.assertEqual(summary["counters"].get("extent"), 1)
            staged = Path(summary["out_path"])
            self.assertTrue(staged.exists())
            with h5py.File(staged, "r") as f:
                self.assertEqual(f["bag_id"].shape[0], 1)
                self.assertEqual(f["bag_id"][0], b"Testville#mesh/good_1.obj")


if __name__ == "__main__":
    unittest.main(verbosity=2)
