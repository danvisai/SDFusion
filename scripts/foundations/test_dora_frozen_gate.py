"""Contract test for `load_surfaces`'s `sources` parameter (code-review finding on #175):
registering "buildingworld" in `SOURCES` silently grew this function's default row set by ~1.5M
rows for every caller. `sources` lets a caller opt out; this pins that it actually filters, that
the default still reads every registered source (unchanged behaviour for existing callers), and
that a missing per-source file is skipped (warned, not fatal) either way.

Run: env -u LD_PRELOAD ./venv/bin/python3 scripts/foundations/test_dora_frozen_gate.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations import dora_frozen_gate as gate  # noqa: E402


def _write_source(dir_: Path, src: str, rows: list[int]) -> None:
    """One tetrahedron per row. `faces` uses each building's own LOCAL vertex indices (0..3), not
    a running global offset -- the convention `ingest_surfaces_buildingworld.py`'s own writer uses
    (each mesh's `verts`/`faces` slice is read back independently via `vert_offset`/`face_offset`,
    so the stored face indices are never meant to index into the concatenated global array)."""
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * len(rows),
                np.float64).reshape(len(rows), 4, 3)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int64)
    verts = np.concatenate([v[i] for i in range(len(rows))])
    faces = np.concatenate([f for _ in range(len(rows))])
    vo = np.arange(0, 4 * len(rows) + 1, 4, dtype=np.int64)
    fo = np.arange(0, 4 * len(rows) + 1, 4, dtype=np.int64)
    with h5py.File(dir_ / f"surfaces_{src}.h5", "w") as h:
        h.create_dataset("verts", data=verts)
        h.create_dataset("faces", data=faces)
        h.create_dataset("vert_offset", data=vo)
        h.create_dataset("face_offset", data=fo)
        h.create_dataset("row", data=np.asarray(rows, np.int32))


class TestLoadSurfacesSources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        _write_source(self.d, "bag3d", [0, 1])
        _write_source(self.d, "buildingworld", [2, 3, 4])

    def test_default_reads_every_registered_source(self):
        with patch.object(gate, "SURF", self.d):
            out = gate.load_surfaces()
        self.assertEqual(sorted(out), [0, 1, 2, 3, 4])
        self.assertEqual(out[0][2], "bag3d")
        self.assertEqual(out[2][2], "buildingworld")

    def test_sources_filters_to_only_the_named_ones(self):
        with patch.object(gate, "SURF", self.d):
            out = gate.load_surfaces(sources=("bag3d",))
        self.assertEqual(sorted(out), [0, 1])

    def test_excluding_buildingworld_leaves_bag3d_rows_untouched(self):
        with patch.object(gate, "SURF", self.d):
            out = gate.load_surfaces(sources=[s for s in gate.SOURCES if s != "buildingworld"])
        self.assertNotIn(2, out)
        self.assertIn(0, out)

    def test_a_missing_sources_file_is_skipped_not_fatal(self):
        with patch.object(gate, "SURF", self.d):
            out = gate.load_surfaces(sources=("bag3d", "nrw"))  # nrw file never written
        self.assertEqual(sorted(out), [0, 1])


if __name__ == "__main__":
    unittest.main()
