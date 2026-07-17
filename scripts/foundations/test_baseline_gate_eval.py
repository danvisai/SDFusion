"""Contract tests for the #27 acceptance-gate scoring (map #24). Synthetic, fast, no GPU/torch.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_baseline_gate_eval.py
"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import baseline_gate_eval as bge  # noqa: E402  (must stay import-light: no torch at module load)


def _row(lcc=0.95, fp=0.70, gen_occ=0.15, region=0):
    return dict(gen_occ=gen_occ, collapsed=bool(gen_occ < 1e-4), lcc=lcc, fp_iou=fp,
                real_fp_self_iou=1.0, region=region)


class TestMetrics(unittest.TestCase):
    def test_lcc_solid_block_is_one(self):
        occ = np.zeros((16, 16, 16), bool); occ[4:12, 4:12, 4:12] = True
        self.assertAlmostEqual(bge.lcc_frac(occ), 1.0, places=6)

    def test_lcc_two_equal_fragments_is_half(self):
        occ = np.zeros((16, 16, 16), bool)
        occ[1:4, 1:4, 1:4] = True          # one cube (27 vox)
        occ[10:13, 10:13, 10:13] = True    # a second, disjoint, equal cube
        self.assertAlmostEqual(bge.lcc_frac(occ), 0.5, places=6)

    def test_lcc_empty_is_zero(self):
        self.assertEqual(bge.lcc_frac(np.zeros((8, 8, 8), bool)), 0.0)

    def test_fp_iou_identical_is_one(self):
        occ = np.zeros((16, 16, 16), bool); occ[4:12, 4:12, 4:12] = True
        real_fp = occ.any(axis=1)
        self.assertAlmostEqual(bge.fp_iou(occ, real_fp), 1.0, places=6)

    def test_fp_iou_disjoint_is_zero(self):
        occ = np.zeros((16, 16, 16), bool); occ[0:4, 4:12, 0:4] = True
        real_fp = np.zeros((16, 16), bool); real_fp[10:14, 10:14] = True
        self.assertEqual(bge.fp_iou(occ, real_fp), 0.0)


class TestScoreGate(unittest.TestCase):
    def test_all_pass(self):
        g = bge.score_gate([_row() for _ in range(20)])
        self.assertTrue(g["collapse_pass"] and g["lcc_pass"] and g["fp_iou_pass"])
        self.assertTrue(g["OVERALL_SCALAR_PASS"])

    def test_collapse_fails_gate(self):
        rows = [_row() for _ in range(19)] + [_row(gen_occ=0.0)]  # 1/20 = 5% collapsed > 1%
        g = bge.score_gate(rows)
        self.assertFalse(g["collapse_pass"]); self.assertFalse(g["OVERALL_SCALAR_PASS"])

    def test_fragmentation_fails_gate(self):
        rows = [_row(lcc=0.5) for _ in range(10)] + [_row(lcc=0.95) for _ in range(10)]  # 50% < 0.90
        g = bge.score_gate(rows)
        self.assertFalse(g["lcc_pass"]); self.assertFalse(g["OVERALL_SCALAR_PASS"])

    def test_footprint_fails_gate(self):
        g = bge.score_gate([_row(fp=0.50) for _ in range(20)])  # median 0.50 < 0.65
        self.assertFalse(g["fp_iou_pass"]); self.assertFalse(g["OVERALL_SCALAR_PASS"])


class TestPerCorpus(unittest.TestCase):
    def test_splits_by_region_and_is_nongating(self):
        rows = [_row(fp=0.7, region=0) for _ in range(3)] + [_row(fp=0.4, region=2) for _ in range(2)]
        d = bge.per_corpus_diagnostics(rows)
        self.assertEqual(set(d), {"0", "2"})           # NL and JP present, DE absent
        self.assertEqual(d["0"]["n"], 3)
        self.assertEqual(d["2"]["n"], 2)
        self.assertAlmostEqual(d["2"]["fp_iou_median"], 0.4, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
