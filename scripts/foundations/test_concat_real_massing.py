"""Contract test for #152's provenance-stamp aggregation onto `real.h5`.

Every ingester stamps its own output `.h5` with `attrs["source"]`/`attrs["ingested_at"]`
(docs/DATA_SOURCES.md), but `real.h5` -- the one file everything downstream actually trains on --
is built by `concat_real_massing.py`, which had no attrs handling at all. This pins that the
concat step reads each source's stamp and aggregates it onto `real.h5` itself, keyed by
`source_id`, rather than losing it at the one file that matters.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_concat_real_massing.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.foundations.concat_real_massing as concat_real_massing  # noqa: E402


def _write_source(path: Path, n: int, source: str, ingested_at: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("sdf", data=np.zeros((n, 4, 4, 4), np.float32))
        f.create_dataset("footprint", data=np.zeros((n, 4, 4), np.uint8))
        f.create_dataset("height_m", data=np.zeros(n, np.float32))
        f.attrs["source"] = source
        f.attrs["ingested_at"] = ingested_at


class TestSourceProvenanceStamp(unittest.TestCase):
    def test_real_h5_carries_every_source_s_provenance_stamp(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src_a, src_b = d / "a.h5", d / "b.h5"
            _write_source(src_a, 2, "3dbag", "2024-01-01T00:00:00")
            _write_source(src_b, 3, "nrw", "2024-02-02T00:00:00")
            out = d / "real.h5"

            orig_sources, orig_argv = concat_real_massing.SOURCES, sys.argv
            concat_real_massing.SOURCES = [(src_a, 0), (src_b, 1)]
            try:
                sys.argv = ["concat_real_massing.py", "--out", str(out)]
                concat_real_massing.main()
            finally:
                concat_real_massing.SOURCES, sys.argv = orig_sources, orig_argv

            with h5py.File(out, "r") as f:
                provenance = json.loads(f.attrs["source_provenance"])
            self.assertEqual(provenance["0"], {"input_file": "a.h5", "source": "3dbag",
                                               "ingested_at": "2024-01-01T00:00:00"})
            self.assertEqual(provenance["1"], {"input_file": "b.h5", "source": "nrw",
                                               "ingested_at": "2024-02-02T00:00:00"})

    def test_a_source_missing_its_own_stamp_records_empty_rather_than_raising(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "a.h5"
            with h5py.File(src, "w") as f:
                f.create_dataset("sdf", data=np.zeros((1, 4, 4, 4), np.float32))
                f.create_dataset("footprint", data=np.zeros((1, 4, 4), np.uint8))
                f.create_dataset("height_m", data=np.zeros(1, np.float32))
            out = d / "real.h5"

            orig_sources, orig_argv = concat_real_massing.SOURCES, sys.argv
            concat_real_massing.SOURCES = [(src, 0)]
            try:
                sys.argv = ["concat_real_massing.py", "--out", str(out)]
                concat_real_massing.main()
            finally:
                concat_real_massing.SOURCES, sys.argv = orig_sources, orig_argv

            with h5py.File(out, "r") as f:
                provenance = json.loads(f.attrs["source_provenance"])
            self.assertEqual(provenance["0"]["source"], "")
            self.assertEqual(provenance["0"]["ingested_at"], "")


if __name__ == "__main__":
    unittest.main()
