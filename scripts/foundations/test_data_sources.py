"""Contract tests for #152 -- the corpus license/provenance manifest.

`TestRealH5Coverage` is the one integration point against the actual corpus (`real.h5`'s own
`source_id` column, metadata-scale, no SDF volumes -- fast). Everything else is synthetic.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_data_sources.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.data_sources import DATA_SOURCES, DOC, H5, missing_sources  # noqa: E402

REQUIRED_FIELDS = ("region", "name", "license", "credit", "url", "ingester")


class TestDataSourcesShape(unittest.TestCase):
    """The manifest itself: every entry carries what #152 asks a manifest entry to carry."""

    def test_every_entry_has_every_required_field_nonempty(self):
        for sid, entry in DATA_SOURCES.items():
            for field in REQUIRED_FIELDS:
                self.assertIn(field, entry, f"source_id {sid} missing {field!r}")
                self.assertTrue(str(entry[field]).strip(), f"source_id {sid} has an empty {field!r}")

    def test_every_url_looks_like_a_url(self):
        for sid, entry in DATA_SOURCES.items():
            self.assertTrue(entry["url"].startswith("https://"), f"source_id {sid}'s url")

    def test_source_ids_match_stratified_splits_own_convention(self):
        """`stratified_split.SOURCE_NAMES` is the other place this id space is spelled out --
        pinned here so the two can never silently drift apart."""
        from scripts.foundations.stratified_split import SOURCE_NAMES
        self.assertEqual(set(DATA_SOURCES), set(SOURCE_NAMES))
        for sid, region in SOURCE_NAMES.items():
            self.assertEqual(DATA_SOURCES[sid]["region"], region)


class TestMissingSources(unittest.TestCase):
    """The acceptance-criterion check itself, exercised synthetically first."""

    def _h5(self, tmp_path, source_ids):
        p = Path(tmp_path)
        with h5py.File(p, "w") as f:
            f.create_dataset("source_id", data=np.array(source_ids, np.int32))
        return p

    def test_returns_empty_when_every_present_id_is_covered(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._h5(Path(d) / "t.h5", [0, 1, 2, 0, 1])
            self.assertEqual(missing_sources(p), [])

    def test_reports_an_uncovered_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._h5(Path(d) / "t.h5", [0, 1, 99])
            self.assertEqual(missing_sources(p), [99])

    def test_a_subset_of_covered_ids_is_still_fully_covered(self):
        """A corpus that only ever used ONE of the three sources must not falsely report the
        other two as 'missing' -- missing means present-but-uncovered, not covered-but-absent."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._h5(Path(d) / "t.h5", [1, 1, 1])
            self.assertEqual(missing_sources(p), [])


class TestRealH5Coverage(unittest.TestCase):
    """#152's own acceptance criterion 4, against the actual corpus."""

    def test_every_source_id_in_real_h5_is_covered(self):
        self.assertTrue(H5.exists(), f"{H5} not found")
        self.assertEqual(missing_sources(), [],
                         "a source_id exists in real.h5 with no DATA_SOURCES entry")


class TestDocStaysInSync(unittest.TestCase):
    """`docs/DATA_SOURCES.md` is hand-kept in sync with the manifest, not generated -- this is
    the drift guard #152 asks for ('or equivalent' manifest, still checked against the corpus)."""

    def test_the_doc_exists(self):
        self.assertTrue(DOC.exists(), f"{DOC} not found")

    def test_every_sources_name_license_and_credit_appear_in_the_doc(self):
        text = DOC.read_text()
        for sid, entry in DATA_SOURCES.items():
            self.assertIn(entry["name"], text, f"source_id {sid}'s name missing from {DOC}")
            self.assertIn(entry["license"], text, f"source_id {sid}'s license missing from {DOC}")
            if entry["credit"] != "none required":
                self.assertIn(entry["credit"], text,
                              f"source_id {sid}'s credit string missing from {DOC}")


class TestIngestersStampProvenance(unittest.TestCase):
    """#152 acceptance criterion 2: each ingester writes a snapshot attr to its output .h5.

    Both ingesters make LIVE network calls inside `main()` (the 3D BAG OGC API; NRW/PLATEAU tile
    downloads), so this checks the source directly rather than running them -- a structural check,
    not a behavioral one, but the write call it looks for is a single, simple, non-conditional
    statement inside the `with h5py.File(...) as f:` block, not logic that could pass the text
    check while failing at runtime.
    """

    def _h5_write_block(self, path: Path) -> str:
        """The body of the (last) `with h5py.File(...) as f:` block, found by indentation rather
        than by the next blank line -- neither ingester happens to leave one immediately after the
        block, so a blank-line search would run past it into unrelated code below."""
        lines = path.read_text().splitlines()
        start = next(i for i, ln in enumerate(lines) if "with h5py.File(" in ln)
        indent = len(lines[start]) - len(lines[start].lstrip())
        body = [lines[start]]
        for ln in lines[start + 1:]:
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
                break
            body.append(ln)
        return "\n".join(body)

    def test_ingest_3dbag_stamps_source_and_ingested_at(self):
        block = self._h5_write_block(REPO / "scripts/ingest_3dbag.py")
        self.assertIn('f.attrs["source"]', block)
        self.assertIn('f.attrs["ingested_at"]', block)

    def test_ingest_citygml_lod2_stamps_source_and_ingested_at(self):
        block = self._h5_write_block(REPO / "scripts/foundations/ingest_citygml_lod2.py")
        self.assertIn('f.attrs["source"]', block)
        self.assertIn('f.attrs["ingested_at"]', block)

    def test_ingest_citygml_lod2_docstring_no_longer_names_the_stale_src_key_field(self):
        """The schema docstring named a `src_key` field the code has always actually written as
        `bag_id` (#152 acceptance criterion 3) -- a doc-only fix, pinned so it cannot regress."""
        text = (REPO / "scripts/foundations/ingest_citygml_lod2.py").read_text()
        docstring = text[:text.index('"""', 3)]
        self.assertNotIn("src_key", docstring)
        self.assertIn("bag_id", docstring)


if __name__ == "__main__":
    unittest.main(verbosity=2)
