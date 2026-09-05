"""Contract tests for #147's per-step visual carving trace. Synthetic, fast, no GPU.

Split to match the module's own split: `carving_steps` is pure numpy (what changed, at what
height) and is tested precisely against a hand-built expected mask; `render_carving_trace` turns
that into pixels and is tested at the level PIL supports well -- frame count, canvas identity
across programs, and whether the highlight colour actually shows up where (and only where) it
should.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_carving_trace.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scene.sdf_edit import (  # noqa: E402
    EditOp, footprint_envelope_sdf, layer_program_to_ops, mask_to_rings,
)
from scripts.foundations.carving_trace import (  # noqa: E402
    BASE_COLOR, HIGHLIGHT_COLOR, VIEWS, carving_steps, render_carving_trace, save_carving_trace,
)

RES = 16


def _fixture():
    """A small footprint, its envelope base, and one region mask to carve from it."""
    fp = np.zeros((RES, RES), bool)
    fp[3:12, 2:13] = True
    y0, y1 = 2, 10
    base = footprint_envelope_sdf(fp, y0, y1, res=RES)
    region = np.zeros((RES, RES), bool)
    region[6:10, 4:9] = True
    region &= fp
    return fp, base, y0, y1, region


def _layer_op(fp, y0, y1, region, height):
    program = [dict(op="Layer", height=height, area=int(region.sum()), components=1,
                    region=[r.tolist() for r in mask_to_rings(region)])]
    return layer_program_to_ops(program, fp, y0, y1, res=RES)


class TestCarvingSteps(unittest.TestCase):
    """The pure-numpy half: what changed, exactly, at each step."""

    def test_the_empty_program_has_no_steps(self):
        _fp, base, _y0, _y1, _region = _fixture()
        self.assertEqual(carving_steps(base, [], res=RES), [])

    def test_one_op_produces_one_step(self):
        fp, base, y0, y1, region = _fixture()
        ops = _layer_op(fp, y0, y1, region, 4)
        steps = carving_steps(base, ops, res=RES)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["index"], 0)

    def test_the_changed_mask_matches_the_carved_region_exactly(self):
        """#144's own definition, restated: a column is touched iff a voxel in it toggled -- for
        this height-map layer op, that is exactly its declared region."""
        fp, base, y0, y1, region = _fixture()
        ops = _layer_op(fp, y0, y1, region, 4)
        steps = carving_steps(base, ops, res=RES)
        np.testing.assert_array_equal(steps[0]["changed"], region)

    def test_a_genuine_no_op_touches_no_columns(self):
        _fp, base, _y0, _y1, _region = _fixture()
        noop = EditOp(kind="box", mode="subtract", center=(0.9, 0.9, 0.9), size=(0.02, 0.02, 0.02))
        steps = carving_steps(base, [noop], res=RES)
        self.assertFalse(steps[0]["changed"].any())

    def test_n_ops_produce_n_steps_each_indexed_in_order(self):
        fp, base, y0, y1, region = _fixture()
        other = np.zeros((RES, RES), bool)
        other[3:6, 2:5] = True
        other &= fp
        ops = _layer_op(fp, y0, y1, region, 6) + _layer_op(fp, y0, y1, other, 3)
        steps = carving_steps(base, ops, res=RES)
        self.assertEqual([s["index"] for s in steps], [0, 1])

    def test_it_is_deterministic_across_calls(self):
        fp, base, y0, y1, region = _fixture()
        ops = _layer_op(fp, y0, y1, region, 4)
        a = carving_steps(base, ops, res=RES)
        b = carving_steps(base, ops, res=RES)
        np.testing.assert_array_equal(a[0]["height"], b[0]["height"])
        np.testing.assert_array_equal(a[0]["changed"], b[0]["changed"])


class TestRenderCarvingTrace(unittest.TestCase):
    """The rendering half: PIL frames, one set of 4 fixed views per operation."""

    def test_n_ops_produce_n_entries_of_4_views_each(self):
        fp, base, y0, y1, region = _fixture()
        other = np.zeros((RES, RES), bool)
        other[3:6, 2:5] = True
        other &= fp
        ops = _layer_op(fp, y0, y1, region, 6) + _layer_op(fp, y0, y1, other, 3)
        trace = render_carving_trace(base, ops, fp, res=RES)
        self.assertEqual(len(trace), 2)
        for entry in trace:
            self.assertEqual(len(entry["views"]), 4)
            self.assertEqual(len(entry["views"]), len(VIEWS))
            for img in entry["views"]:
                self.assertEqual(img.mode, "RGB")

    def test_the_empty_program_produces_no_frames(self):
        _fp, base, _y0, _y1, _region = _fixture()
        self.assertEqual(render_carving_trace(base, [], _fp, res=RES), [])

    def test_every_view_of_every_step_shares_one_canvas_size(self):
        """#147's own criterion: no per-mesh rescaling between steps or between programs."""
        fp, base, y0, y1, region = _fixture()
        ops = _layer_op(fp, y0, y1, region, 4)
        trace = render_carving_trace(base, ops, fp, res=RES)
        sizes = {img.size for img in trace[0]["views"]}
        self.assertEqual(len(sizes), 1, "all 4 fixed views must share one canvas size")

        tall_base = footprint_envelope_sdf(fp, 2, 14, res=RES)         # much taller envelope
        tall_ops = _layer_op(fp, 2, 14, region, 3)
        tall_trace = render_carving_trace(tall_base, tall_ops, fp, res=RES)
        self.assertEqual(trace[0]["views"][0].size, tall_trace[0]["views"][0].size,
                         "canvas size must depend on `res` alone, never on content height")

    def test_a_changed_column_renders_in_the_highlight_color(self):
        fp, base, y0, y1, region = _fixture()
        ops = _layer_op(fp, y0, y1, region, 4)
        trace = render_carving_trace(base, ops, fp, res=RES)
        arr = np.asarray(trace[0]["views"][0]).reshape(-1, 3)
        near_highlight = np.abs(arr.astype(int) - np.array(HIGHLIGHT_COLOR)).max(axis=1) < 40
        self.assertTrue(near_highlight.any(), "expected some pixels shaded near HIGHLIGHT_COLOR")

    def test_a_genuine_no_op_step_has_no_highlighted_pixels(self):
        _fp, base, _y0, _y1, _region = _fixture()
        noop = EditOp(kind="box", mode="subtract", center=(0.9, 0.9, 0.9), size=(0.02, 0.02, 0.02))
        trace = render_carving_trace(base, [noop], _fp, res=RES)
        for img in trace[0]["views"]:
            arr = np.asarray(img).reshape(-1, 3)
            near_highlight = np.abs(arr.astype(int) - np.array(HIGHLIGHT_COLOR)).max(axis=1) < 40
            self.assertFalse(near_highlight.any(), "a no-op step must render nothing as changed")

    def test_the_4_views_are_genuinely_different_images(self):
        """Each view rotates which corner faces the fixed camera; a footprint with no rotational
        symmetry must not render identically from every side."""
        fp, base, y0, y1, region = _fixture()
        ops = _layer_op(fp, y0, y1, region, 4)
        trace = render_carving_trace(base, ops, fp, res=RES)
        arrays = [np.asarray(img) for img in trace[0]["views"]]
        # sizes match (same canvas) but pixel content must differ between at least one pair
        self.assertTrue(any(not np.array_equal(arrays[i], arrays[j])
                            for i in range(4) for j in range(i + 1, 4)))

    def test_it_does_not_re_validate_a_program_that_would_fail_the_finalize_gate(self):
        """#147's own criterion: runs against any program already past #145/#146, does not
        re-validate. A mixed add/subtract program (non-commuting, #140) still renders."""
        fp, base, y0, y1, region = _fixture()
        sub_op = _layer_op(fp, y0, y1, region, 4)[0]
        from dataclasses import replace
        add_op = replace(sub_op, mode="add")
        trace = render_carving_trace(base, [sub_op, add_op], fp, res=RES)
        self.assertEqual(len(trace), 2)


class TestSaveCarvingTrace(unittest.TestCase):
    def test_writes_one_file_per_view_per_step(self):
        fp, base, y0, y1, region = _fixture()
        other = np.zeros((RES, RES), bool)
        other[3:6, 2:5] = True
        other &= fp
        ops = _layer_op(fp, y0, y1, region, 6) + _layer_op(fp, y0, y1, other, 3)
        trace = render_carving_trace(base, ops, fp, res=RES)
        with tempfile.TemporaryDirectory() as d:
            paths = save_carving_trace(trace, d)
            self.assertEqual(len(paths), 2 * len(VIEWS))
            for p in paths:
                self.assertTrue(p.exists())
            self.assertTrue((Path(d) / f"step0_view{VIEWS[0]}.png").exists())
            self.assertTrue((Path(d) / f"step1_view{VIEWS[-1]}.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
