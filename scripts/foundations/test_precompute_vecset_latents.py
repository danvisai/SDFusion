"""CPU contract tests for the resumable vecset-cache writer."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.precompute_vecset_latents import IncrementalCache  # noqa: E402


def _row(row: int) -> dict:
    return {
        "latent": np.full((3, 2), row, np.float16),
        "query_pos": np.full((3, 3), row / 10, np.float16),
        "footprint": np.full((2, 2), row % 2, np.uint8),
        "height_m": float(row + 1),
        "region": row % 3,
        "row": row,
        "held_out": row % 2,
    }


class TestIncrementalCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "cache.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_appends_without_rewriting_committed_rows(self):
        cache = IncrementalCache(self.path, resume=False, flush_every=2)
        cache.add(**_row(10))
        cache.add(**_row(11))
        cache.close({"codec": "test"})

        resumed = IncrementalCache(self.path, resume=True, flush_every=2)
        self.assertEqual(resumed.done, {10, 11})
        resumed.add(**_row(12))
        resumed.close()

        with h5py.File(self.path, "r") as f:
            np.testing.assert_array_equal(f["row"], [10, 11, 12])
            np.testing.assert_array_equal(f["latent"][0], _row(10)["latent"])
            self.assertEqual(int(f.attrs["committed_rows"]), 3)
            self.assertEqual(f.attrs["codec"], "test")

    def test_resume_discards_an_append_past_the_commit_boundary(self):
        cache = IncrementalCache(self.path, resume=False, flush_every=1)
        cache.add(**_row(20))
        cache.close()

        # Simulate a process dying after every dataset was extended but before `committed_rows` was
        # advanced. The next opener must discard this ambiguous tail rather than train on fill data.
        with h5py.File(self.path, "a") as f:
            for name in IncrementalCache.SPECS:
                d = f[name]
                d.resize(2, axis=0)
                d[1] = _row(999)[name]
            self.assertEqual(int(f.attrs["committed_rows"]), 1)

        resumed = IncrementalCache(self.path, resume=True)
        self.assertEqual(resumed.done, {20})
        resumed.close()
        with h5py.File(self.path, "r") as f:
            self.assertTrue(all(f[name].shape[0] == 1 for name in IncrementalCache.SPECS))

    def test_old_fixed_size_cache_is_refused(self):
        with h5py.File(self.path, "w") as f:
            f.create_dataset("row", data=np.array([1], np.int32))
        with self.assertRaisesRegex(SystemExit, "not an incremental cache"):
            IncrementalCache(self.path, resume=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
