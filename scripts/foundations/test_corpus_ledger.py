"""#161: contract tests for the row/region/held_out/height_m ledger split out of vecset_latents.h5."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations import corpus_ledger  # noqa: E402


def _cols(rows):
    """Deterministic (region, held_out, height_m) for a list of row ids, matching no real policy --
    only used so tests do not repeat literal arrays."""
    region = [r % 3 for r in rows]
    held_out = [r % 2 for r in rows]
    height_m = [float(r) + 0.5 for r in rows]
    return list(rows), region, held_out, height_m


class TestWriteReadRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.h5"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trips_every_column_exactly(self):
        row, region, held_out, height_m = _cols([5, 1, 9, 2])
        corpus_ledger.write_ledger(row, region, held_out, height_m, path=self.path, source="test")
        out = corpus_ledger.read_ledger(self.path)
        np.testing.assert_array_equal(out["row"], row)
        np.testing.assert_array_equal(out["region"], region)
        np.testing.assert_array_equal(out["held_out"], held_out)
        np.testing.assert_allclose(out["height_m"], height_m)

    def test_dtypes_match_the_canonical_schema(self):
        row, region, held_out, height_m = _cols([1, 2])
        corpus_ledger.write_ledger(row, region, held_out, height_m, path=self.path)
        out = corpus_ledger.read_ledger(self.path)
        for name, dtype in corpus_ledger.DTYPES.items():
            self.assertEqual(out[name].dtype, dtype)

    def test_write_is_atomic_no_tmp_file_left_behind(self):
        row, region, held_out, height_m = _cols([1, 2])
        corpus_ledger.write_ledger(row, region, held_out, height_m, path=self.path)
        self.assertTrue(self.path.exists())
        self.assertFalse(self.path.with_suffix(self.path.suffix + ".tmp").exists())

    def test_a_second_write_replaces_rather_than_appends(self):
        corpus_ledger.write_ledger(*_cols([1, 2]), path=self.path)
        row, region, held_out, height_m = _cols([7, 8, 9])
        corpus_ledger.write_ledger(row, region, held_out, height_m, path=self.path)
        out = corpus_ledger.read_ledger(self.path)
        np.testing.assert_array_equal(sorted(out["row"].tolist()), [7, 8, 9])

    def test_missing_file_raises_with_guidance(self):
        with self.assertRaisesRegex(FileNotFoundError, "corpus_ledger.write_ledger"):
            corpus_ledger.read_ledger(self.path)

    def test_wrong_schema_version_is_rejected(self):
        with h5py.File(self.path, "w") as f:
            f.attrs["schema_version"] = 999
            for name in corpus_ledger.COLUMNS:
                f.create_dataset(name, data=np.array([0], corpus_ledger.DTYPES[name]))
        with self.assertRaisesRegex(ValueError, "schema 999"):
            corpus_ledger.read_ledger(self.path)

    def test_a_missing_column_is_rejected(self):
        with h5py.File(self.path, "w") as f:
            f.attrs["schema_version"] = corpus_ledger.SCHEMA_VERSION
            f.create_dataset("row", data=np.array([0], np.int32))
        with self.assertRaisesRegex(ValueError, "missing ledger column"):
            corpus_ledger.read_ledger(self.path)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.h5"

    def tearDown(self):
        self.tmp.cleanup()

    def test_mismatched_column_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "1-D with length"):
            corpus_ledger.write_ledger([1, 2, 3], [0, 0], [0, 0, 0], [1.0, 2.0, 3.0], path=self.path)

    def test_duplicate_row_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            corpus_ledger.write_ledger([1, 1], [0, 1], [0, 1], [1.0, 2.0], path=self.path)

    def test_held_out_outside_0_1_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "held_out must be 0 or 1"):
            corpus_ledger.write_ledger([1, 2], [0, 0], [0, 2], [1.0, 2.0], path=self.path)

    def test_non_finite_height_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            corpus_ledger.write_ledger([1, 2], [0, 0], [0, 1], [1.0, float("nan")], path=self.path)


class TestAppendLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.h5"

    def tearDown(self):
        self.tmp.cleanup()

    def test_appends_to_an_existing_ledger(self):
        row, region, held_out, height_m = _cols([1, 2])
        corpus_ledger.write_ledger(row, region, held_out, height_m, path=self.path)
        row2, region2, held_out2, height_m2 = _cols([3, 4])
        corpus_ledger.append_ledger(row2, region2, held_out2, height_m2, path=self.path)
        out = corpus_ledger.read_ledger(self.path)
        np.testing.assert_array_equal(sorted(out["row"].tolist()), [1, 2, 3, 4])

    def test_creates_a_ledger_when_none_exists(self):
        row, region, held_out, height_m = _cols([10])
        corpus_ledger.append_ledger(row, region, held_out, height_m, path=self.path)
        out = corpus_ledger.read_ledger(self.path)
        np.testing.assert_array_equal(out["row"], [10])

    def test_refuses_to_silently_overwrite_an_existing_row(self):
        row, region, held_out, height_m = _cols([1, 2])
        corpus_ledger.write_ledger(row, region, held_out, height_m, path=self.path)
        with self.assertRaisesRegex(ValueError, "already in the ledger"):
            corpus_ledger.append_ledger([2, 3], [0, 0], [0, 0], [1.0, 2.0], path=self.path)
        # the refused append must not have partially applied
        out = corpus_ledger.read_ledger(self.path)
        np.testing.assert_array_equal(sorted(out["row"].tolist()), [1, 2])


class TestIndexByRow(unittest.TestCase):
    def test_maps_row_id_to_array_position(self):
        ledger = {"row": np.array([5, 1, 9], np.int32)}
        self.assertEqual(corpus_ledger.index_by_row(ledger), {5: 0, 1: 1, 9: 2})


class TestExtractFromVecsetLatents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.latents_path = Path(self.tmp.name) / "vecset_latents.h5"

    def tearDown(self):
        self.tmp.cleanup()

    def test_pulls_the_four_ledger_columns_and_ignores_the_latent_arrays(self):
        with h5py.File(self.latents_path, "w") as f:
            f.create_dataset("row", data=np.array([3, 4], np.int32))
            f.create_dataset("region", data=np.array([1, 2], np.int32))
            f.create_dataset("held_out", data=np.array([0, 1], np.uint8))
            f.create_dataset("height_m", data=np.array([10.0, 20.0], np.float32))
            f.create_dataset("latent", data=np.zeros((2, 4, 4), np.float16))
            f.create_dataset("footprint", data=np.zeros((2, 4, 4), np.uint8))
        out = corpus_ledger.extract_from_vecset_latents(self.latents_path)
        np.testing.assert_array_equal(out["row"], [3, 4])
        np.testing.assert_array_equal(out["region"], [1, 2])
        np.testing.assert_array_equal(out["held_out"], [0, 1])
        np.testing.assert_allclose(out["height_m"], [10.0, 20.0])
        self.assertEqual(set(out.keys()), set(corpus_ledger.COLUMNS))

    def test_a_missing_column_is_rejected(self):
        with h5py.File(self.latents_path, "w") as f:
            f.create_dataset("row", data=np.array([1], np.int32))
        with self.assertRaisesRegex(ValueError, "missing column"):
            corpus_ledger.extract_from_vecset_latents(self.latents_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
