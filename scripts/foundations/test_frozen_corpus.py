"""#162: fail closed on corpus identity drift before caches/training/evaluation can run.

Synthetic failure tests run anywhere. Acceptance/append tests copy only the real corpus's
2.4 MB of identity metadata, with tiny placeholder geometry; the production file stays read-only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.frozen_corpus import (  # noqa: E402
    FROZEN_CORPUS_SHA256, FROZEN_SPLIT_N_TOTAL, REAL_CORPUS_PATH,
    open_real_corpus, row_identity_sha256,
)


def write_corpus(path, ids, heights):
    with h5py.File(path, "w") as f:
        f.create_dataset("bag_id", data=ids)
        f.create_dataset("height_m", data=heights)
        # No real geometry is needed to prove the identity boundary.
        f.create_dataset("sdf", shape=(len(ids), 1, 1, 1), dtype="f4")
        f.create_dataset("footprint", shape=(len(ids), 1, 1), dtype="u1")
        f.create_dataset("source_id", shape=(len(ids),), dtype="i4")


class TestIdentityEncoding(unittest.TestCase):
    def test_format_matches_independently_serialized_records(self):
        # 1.0 and 2.0's IEEE-754 representations, each after an S64 ID; no struct padding.
        records = b"a" + b"\0" * 63 + b"\x00\x00\x80\x3f"
        records += b"bc" + b"\0" * 62 + b"\x00\x00\x00\x40"
        self.assertEqual(row_identity_sha256(np.array([b"a", b"bc"], "S64"),
                                             np.array([1, 2], "f4")),
                         hashlib.sha256(records).hexdigest())

    def test_big_endian_heights_have_the_same_identity(self):
        ids = np.array([b"one", b"two"], "S64")
        self.assertEqual(row_identity_sha256(ids, np.array([1.25, 7.5], ">f4")),
                         row_identity_sha256(ids, np.array([1.25, 7.5], "<f4")))

    def test_lossy_or_malformed_identity_columns_are_rejected(self):
        for ids, heights in [
            (np.array([b"id"], "S128"), np.array([1], "f4")),
            (np.array([b"id"], "S64"), np.array([1], "f8")),
            (np.array([b"id"], "S64"), np.array([np.nan], "f4")),
            (np.array([b"id"], "S64"), np.array([[1]], "f4")),
        ]:
            with self.subTest(ids=ids.dtype, heights=heights), self.assertRaises(ValueError):
                row_identity_sha256(ids, heights)


class TestCorpusRejection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "renamed-corpus.h5"

    def test_same_size_different_corpus_is_rejected_even_when_renamed(self):
        write_corpus(self.path, np.full(FROZEN_SPLIT_N_TOTAL, b"other", "S64"),
                     np.ones(FROZEN_SPLIT_N_TOTAL, "f4"))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            open_real_corpus(self.path)
        # Validation failure releases the handle, so the caller can repair/replace the file.
        with h5py.File(self.path, "r+"):
            pass

    def test_missing_or_short_columns_fail_closed(self):
        for missing in ("sdf", "bag_id", "height_m", None):
            write_corpus(self.path, np.array([b"id"], "S64"), np.array([1], "f4"))
            if missing:
                with h5py.File(self.path, "a") as f:
                    del f[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "#162"):
                open_real_corpus(self.path)

    def test_existing_height_cache_does_not_bypass_source_validation(self):
        from scripts.foundations import train_height_map_generator as heightmap

        write_corpus(self.path, np.array([b"id"], "S64"), np.array([1], "f4"))
        cached = Path(self.tmp.name) / "heightmap.npz"
        np.savez(cached, row=np.array([0]))
        with patch.object(heightmap, "H5", self.path):
            with self.assertRaisesRegex(ValueError, "frozen prefix"):
                heightmap.build_cache(cached)

    def test_precompute_rejects_source_before_codec_or_output_creation(self):
        from scripts.foundations import precompute_vecset_latents as precompute

        write_corpus(self.path, np.array([b"id"], "S64"), np.array([1], "f4"))
        out = Path(self.tmp.name) / "latents.h5"
        with patch.object(precompute, "H5", self.path), \
             patch.object(sys, "argv", ["precompute", "--out", str(out)]), \
             patch.object(precompute, "load_dora", side_effect=AssertionError("codec loaded")):
            with self.assertRaisesRegex(ValueError, "frozen prefix"):
                precompute.main()
        self.assertFalse(out.exists())

    def test_legacy_bag_only_dataset_still_loads(self):
        from datasets.bag3d_dataset import Bag3dDataset

        write_corpus(self.path, np.full(100, b"legacy", "S64"), np.ones(100, "f4"))
        with h5py.File(self.path, "a") as f:
            del f["source_id"]
        ds = Bag3dDataset()
        ds.initialize(SimpleNamespace(bag3d_h5=self.path), "test")
        self.assertEqual(len(ds), 2)


@unittest.skipUnless(REAL_CORPUS_PATH.exists(), "real corpus metadata not available")
class TestPinnedCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with h5py.File(REAL_CORPUS_PATH, "r") as f:
            cls.ids = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
            cls.heights = f["height_m"][:FROZEN_SPLIT_N_TOTAL]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "real.h5"
        write_corpus(self.path, self.ids, self.heights)

    def test_live_corpus_matches_recorded_baseline(self):
        with open_real_corpus(REAL_CORPUS_PATH) as f:
            self.assertEqual(row_identity_sha256(f["bag_id"][:FROZEN_SPLIT_N_TOTAL],
                                                 f["height_m"][:FROZEN_SPLIT_N_TOTAL]),
                             FROZEN_CORPUS_SHA256)

    def test_changed_id_height_or_order_is_rejected(self):
        for mutation in ("id", "height", "order"):
            ids, heights = self.ids.copy(), self.heights.copy()
            if mutation == "id":
                ids[-1] = b"replacement-building"
            elif mutation == "height":
                heights[0] += 1
            else:
                ids[[0, 1]], heights[[0, 1]] = ids[[1, 0]], heights[[1, 0]]
            write_corpus(self.path, ids, heights)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "identity mismatch"):
                open_real_corpus(self.path)

    def test_dropped_row_is_rejected(self):
        write_corpus(self.path, self.ids[:-1], self.heights[:-1])
        with self.assertRaisesRegex(ValueError, "frozen prefix"):
            open_real_corpus(self.path)

    def test_truncated_metadata_is_rejected(self):
        with h5py.File(self.path, "a") as f:
            del f["height_m"]
            f.create_dataset("height_m", data=self.heights[:-1])
        with self.assertRaisesRegex(ValueError, "row count"):
            open_real_corpus(self.path)

    def test_appended_rows_preserve_historical_split_and_join_training(self):
        from datasets.bag3d_dataset import Bag3dDataset
        from scripts.foundations.vecset_ceiling_probe import test_indices

        original = Bag3dDataset()
        original.initialize(SimpleNamespace(bag3d_h5=self.path), "test")
        self.assertEqual(len(original), 715)  # 2% of 35,776; one lacks a recovered surface => 714.
        np.testing.assert_array_equal(original.idxs, test_indices(FROZEN_SPLIT_N_TOTAL))
        original_val = Bag3dDataset()
        original_val.initialize(SimpleNamespace(bag3d_h5=self.path), "val")
        write_corpus(self.path, np.concatenate((self.ids, np.array([b"new"], "S64"))),
                     np.concatenate((self.heights, np.array([99], "f4"))))
        for phase, expected in (("test", original.idxs), ("val", original_val.idxs)):
            ds = Bag3dDataset()
            ds.initialize(SimpleNamespace(bag3d_h5=self.path), phase)
            np.testing.assert_array_equal(ds.idxs, expected)
        train = Bag3dDataset()
        train.initialize(SimpleNamespace(bag3d_h5=self.path), "train")
        self.assertIn(FROZEN_SPLIT_N_TOTAL, train.idxs)
        self.assertFalse(np.isin(train.idxs, original.idxs).any())
        self.assertFalse(np.isin(train.idxs, original_val.idxs).any())

    def test_dataset_revalidates_when_worker_opens_the_corpus(self):
        from datasets.bag3d_dataset import Bag3dDataset

        ds = Bag3dDataset()
        ds.initialize(SimpleNamespace(bag3d_h5=self.path), "test")
        with h5py.File(self.path, "a") as f:
            f["bag_id"][0] = b"changed-between-initialization-and-first-read"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ds[0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
