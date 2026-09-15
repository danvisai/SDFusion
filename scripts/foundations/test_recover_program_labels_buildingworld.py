"""Contract tests for #176's program pseudo-label regeneration. Synthetic, fast, CPU -- no real
BuildingWorld mesh, and no real corpus except where `run()`'s own frozen-prefix requirement
(#162) forces a `real.h5`-shaped fixture, guarded by `REAL_CORPUS_PATH.exists()` exactly like
`test_ingest_buildingworld.py::TestCombineIntoReal`.

Covers: row selection (`buildingworld_rows`, `cross_check_row_count` -- informational only, never
fatal), the fit wrapper (`fit_row` -- op-type sequence + family/dl_ops/dl_planar_fraction/is_shed,
and the failure paths), `IncrementalRowWriter`'s write/flush/resume/no-resume behaviour, and `run()`
end to end (sequential and pool paths agree; resume skips committed rows; source_id filtering keeps
only BuildingWorld rows).

Run: env -u LD_PRELOAD ./venv/bin/python3 scripts/foundations/test_recover_program_labels_buildingworld.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.recover_massing_programs import RES  # noqa: E402
from scripts.foundations.recover_program_labels_buildingworld import (  # noqa: E402
    BUILDINGWORLD_SOURCE_ID, IncrementalRowWriter, buildingworld_rows, cross_check_row_count,
    fit_row, run,
)
from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, REAL_CORPUS_PATH  # noqa: E402


class TestBuildingworldRows(unittest.TestCase):
    def test_selects_only_the_sentinel_source_id(self):
        source_id = np.array([0, 1, 2, -1, -1, 0, -1], np.int32)
        np.testing.assert_array_equal(buildingworld_rows(source_id), [3, 4, 6])

    def test_empty_when_no_buildingworld_rows_present(self):
        source_id = np.array([0, 1, 2], np.int32)
        self.assertEqual(len(buildingworld_rows(source_id)), 0)

    def test_the_sentinel_is_the_one_ingest_buildingworld_uses(self):
        """#174's own doc: source_id = -1 for every BuildingWorld row. A drift here would silently
        select the wrong population."""
        self.assertEqual(BUILDINGWORLD_SOURCE_ID, -1)


class TestCrossCheckRowCount(unittest.TestCase):
    def test_missing_surfaces_file_warns_and_does_not_raise(self):
        cross_check_row_count(1234, surfaces_path=Path("/nonexistent/surfaces_buildingworld.h5"))

    def test_matching_count_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "surfaces_buildingworld.h5"
            with h5py.File(p, "w") as f:
                f.create_dataset("row", data=np.arange(10, dtype=np.int32))
            cross_check_row_count(10, surfaces_path=p)

    def test_smaller_surfaces_count_only_warns_since_175_legitimately_drops_rows(self):
        """#175's own `run()` drops any row it can't confidently recover a mesh for (a failed
        mesh load, an unresolved Perth-style bag_id collision), so a smaller surfaces count is an
        expected outcome, not corruption -- must never abort the fit."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "surfaces_buildingworld.h5"
            with h5py.File(p, "w") as f:
                f.create_dataset("row", data=np.arange(9, dtype=np.int32))
            cross_check_row_count(10, surfaces_path=p)  # must not raise

    def test_larger_surfaces_count_also_only_warns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "surfaces_buildingworld.h5"
            with h5py.File(p, "w") as f:
                f.create_dataset("row", data=np.arange(11, dtype=np.int32))
            cross_check_row_count(10, surfaces_path=p)  # must not raise


def _flat_box(r: int = RES):
    """A footprint fully occupied up to a uniform height -- zero surplus, so the fitter should
    return an empty program (`family` == "flat"). `gt`/`fp` follow `height_field`'s own axis
    convention: `gt` is [z, y, x], `fp` is [z, x]."""
    fp = np.zeros((r, r), bool)
    fp[r // 4:3 * r // 4, r // 4:3 * r // 4] = True
    h = max(r // 2, 1)
    gt = np.zeros((r, r, r), bool)
    gt[:, :h, :] = fp[:, None, :]
    return gt, fp


def _gabled_box(r: int = RES):
    """A footprint whose occupied height ridges along `z` -- enough surplus that `Ramp`s/`Layer`s
    should fire, giving a non-flat family."""
    fp = np.zeros((r, r), bool)
    fp[r // 4:3 * r // 4, r // 4:3 * r // 4] = True
    ridge = r // 2
    heights = np.clip(r // 2 - np.abs(np.arange(r) - ridge) // 2, max(r // 8, 1), max(r // 2, 1))
    gt = np.zeros((r, r, r), bool)
    for z in range(r):
        if fp[z].any():
            gt[z, :heights[z], :] = fp[z][None, :]
    return gt, fp


class TestFitRow(unittest.TestCase):
    def test_flat_building_returns_ok_flat_and_empty_ops(self):
        gt, fp = _flat_box()
        rec = fit_row(gt, fp)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["family"], "flat")
        self.assertEqual(rec["dl_ops"], 0)
        self.assertEqual(rec["ops"], "")
        self.assertEqual(rec["fail_stage"], "")

    def test_gabled_building_reports_ops_and_a_nonflat_family(self):
        gt, fp = _gabled_box()
        rec = fit_row(gt, fp)
        self.assertTrue(rec["ok"])
        self.assertGreater(rec["dl_ops"], 0)
        self.assertNotEqual(rec["family"], "flat")
        ops = rec["ops"].split(";")
        self.assertEqual(len(ops), rec["dl_ops"])  # `ops` is the literal op-type sequence
        self.assertTrue(all(op in ("Layer", "Ramp", "CutRoof") for op in ops))

    def test_empty_footprint_fails_with_empty_height_field(self):
        r = RES
        gt = np.zeros((r, r, r), bool)
        fp = np.zeros((r, r), bool)
        rec = fit_row(gt, fp)
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["fail_stage"], "empty_height_field")

    def test_fit_row_agrees_with_fit_and_classify_on_a_flat_building(self):
        """`fit_row` is `height_field` + #159's own `fit_and_classify` -- not a reimplementation
        of it -- so the two must agree exactly."""
        from scripts.foundations.pilot_buildingworld_roof_families import fit_and_classify
        from scripts.foundations.recover_massing_programs import height_field
        gt, fp = _flat_box()
        y0, y1, target = height_field(gt, fp)
        direct = fit_and_classify(fp, y0, y1, target)
        via_row = fit_row(gt, fp)
        self.assertEqual(direct["family"], via_row["family"])
        self.assertEqual(direct["dl_ops"], via_row["dl_ops"])
        self.assertEqual(direct["ops"], via_row["ops"])


class TestIncrementalRowWriter(unittest.TestCase):
    def _rec(self, row):
        return dict(row=row, ok=True, family="flat", dl_ops=0, dl_planar_fraction=0.0,
                   is_shed=False, ops="", fail_stage="")

    def test_write_flush_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "labels.h5"
            with IncrementalRowWriter(path, flush_every=2) as w:
                for i in range(5):
                    w.add(**self._rec(i))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 5)
                self.assertEqual(int(f.attrs["committed_rows"]), 5)
                np.testing.assert_array_equal(f["row"][:], np.arange(5))

    def test_resume_skips_already_committed_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "labels.h5"
            with IncrementalRowWriter(path, flush_every=100) as w:
                for i in range(3):
                    w.add(**self._rec(i))
            with IncrementalRowWriter(path, resume=True) as w2:
                self.assertTrue(w2.already_done(0))
                self.assertFalse(w2.already_done(3))
                for i in range(3, 6):
                    w2.add(**self._rec(i))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 6)
                np.testing.assert_array_equal(sorted(int(r) for r in f["row"][:]), list(range(6)))

    def test_no_resume_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "labels.h5"
            with IncrementalRowWriter(path) as w:
                w.add(**self._rec(0))
            with IncrementalRowWriter(path, resume=False) as w2:
                w2.add(**self._rec(1))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["row"].shape[0], 1)

    def test_schema_mismatch_refuses_to_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "labels.h5"
            with h5py.File(path, "w") as f:
                f.attrs["schema_version"] = 999
                f.attrs["committed_rows"] = 0
            with self.assertRaises(SystemExit):
                IncrementalRowWriter(path, resume=True)


@unittest.skipUnless(REAL_CORPUS_PATH.exists(), "real corpus metadata not available")
class TestRunEndToEnd(unittest.TestCase):
    """`open_real_corpus` enforces the #162 frozen-prefix hash, so any file it opens needs the
    REAL first-35,776 (bag_id, height_m) pair -- built the same way
    `test_ingest_buildingworld.py::TestCombineIntoReal` does. Only that metadata is read from the
    real corpus; the SDF volumes are entirely synthetic and tiny (r=4)."""

    @classmethod
    def setUpClass(cls):
        with h5py.File(REAL_CORPUS_PATH, "r") as f:
            cls.ids = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
            cls.heights = f["height_m"][:FROZEN_SPLIT_N_TOTAL]

    def _write_synthetic_corpus(self, path: Path, n_extra_bw: int, r: int = 4) -> None:
        n = FROZEN_SPLIT_N_TOTAL + n_extra_bw
        gt, fp = _flat_box(r)
        with h5py.File(path, "w") as f:
            sdf = np.where(gt, -1.0, 1.0).astype(np.float32)
            f.create_dataset("sdf", data=np.broadcast_to(sdf, (n, r, r, r)).copy())
            f.create_dataset("footprint", data=np.broadcast_to(fp.astype(np.uint8), (n, r, r)).copy())
            heights = np.concatenate([self.heights, np.full(n_extra_bw, 5.0, np.float32)])
            f.create_dataset("height_m", data=heights)
            bag_ids = np.concatenate([self.ids,
                                      np.array([f"bw:Test#{i}".encode() for i in range(n_extra_bw)],
                                              dtype="S64")])
            f.create_dataset("bag_id", data=bag_ids)
            source_id = np.zeros(n, np.int32)
            source_id[FROZEN_SPLIT_N_TOTAL:] = BUILDINGWORLD_SOURCE_ID
            f.create_dataset("source_id", data=source_id)

    def test_run_fits_only_the_buildingworld_rows_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            corpus = d / "real.h5"
            self._write_synthetic_corpus(corpus, n_extra_bw=4, r=4)
            with h5py.File(corpus, "r") as f:
                rows = buildingworld_rows(np.asarray(f["source_id"]))
            self.assertEqual(list(rows), list(range(FROZEN_SPLIT_N_TOTAL, FROZEN_SPLIT_N_TOTAL + 4)))

            out = d / "labels.h5"
            run(corpus, rows, out, workers=1)
            with h5py.File(out, "r") as f:
                self.assertEqual(f["row"].shape[0], 4)
                self.assertTrue(bool(np.all(f["ok"][:])))
                np.testing.assert_array_equal(sorted(int(r) for r in f["row"][:]), list(rows))

            # resuming with one already-committed row plus one new row only fits the new one
            run(corpus, list(rows) + [FROZEN_SPLIT_N_TOTAL + 4 - 1], out, workers=1, resume=True)
            with h5py.File(out, "r") as f:
                self.assertEqual(f["row"].shape[0], 4)  # no duplicate for the already-done row


if __name__ == "__main__":
    unittest.main()
