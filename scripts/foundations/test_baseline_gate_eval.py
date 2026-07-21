"""Contract tests for the #27 acceptance-gate scoring (map #24). Synthetic, fast, no GPU/torch.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_baseline_gate_eval.py
"""
from __future__ import annotations
import os, sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import baseline_gate_eval as bge  # noqa: E402  (must stay import-light: no torch at module load)

# GPU behavioral-smoke gating: the map-#24 retrain checkpoint (has ema_df) + an opt-in env flag,
# so the fast default suite skips the heavy 15 GB model loads. Set RUN_GPU_SMOKE=1 to run.
REPO = Path(__file__).resolve().parents[2]
RETRAIN_CKPT = REPO / "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"
_GPU_SMOKE = bool(os.environ.get("RUN_GPU_SMOKE")) and RETRAIN_CKPT.exists()


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


class TestBuildOpt(unittest.TestCase):
    """Phase-1 sampling knobs (map #34): build_opt must thread the EMA toggle so a
    raw-weights config can be scored against the gate for comparison. Pure, no GPU."""

    def test_use_ema_defaults_true(self):
        # Default preserves the deployed behavior (map #24 gate ran with EMA on).
        opt = bge.build_opt("cpu")
        self.assertTrue(opt.use_ema)

    def test_use_ema_override_threads_through(self):
        opt = bge.build_opt("cpu", use_ema=False)
        self.assertFalse(opt.use_ema)

    def test_ddim_default_present(self):
        # The DDIM-step knob is a Phase-1 lever too; the harness sets it after build.
        opt = bge.build_opt("cpu")
        self.assertEqual(opt.ddim_steps, 100)


class TestPerCorpus(unittest.TestCase):
    def test_splits_by_region_and_is_nongating(self):
        rows = [_row(fp=0.7, region=0) for _ in range(3)] + [_row(fp=0.4, region=2) for _ in range(2)]
        d = bge.per_corpus_diagnostics(rows)
        self.assertEqual(set(d), {"0", "2"})           # NL and JP present, DE absent
        self.assertEqual(d["0"]["n"], 3)
        self.assertEqual(d["2"]["n"], 2)
        self.assertAlmostEqual(d["2"]["fp_iou_median"], 0.4, places=6)


@unittest.skipUnless(_GPU_SMOKE, "GPU knob smoke: set RUN_GPU_SMOKE=1 and provide the retrain ckpt")
class TestSamplingKnobsSmoke(unittest.TestCase):
    """Behavioral check (PRD Testing Decisions): the Phase-1 knobs actually take effect on the
    real checkpoint — EMA vs raw swaps weights, and the guidance scale flows to inference. Heavy
    (loads the 15 GB model), so opt-in via RUN_GPU_SMOKE. torch is imported lazily to keep this
    module import-light for the pure tests above."""

    DDIM = 20

    @classmethod
    def setUpClass(cls):
        import torch
        from datasets.bag3d_dataset import Bag3dDataset
        from models.stage3a_model import Stage3aModel
        if not torch.cuda.is_available():
            raise unittest.SkipTest("no CUDA")
        cls.torch, cls._DS, cls._Model = torch, Bag3dDataset, Stage3aModel
        cls.device = "cuda"
        opt = bge.build_opt(cls.device, ckpt=str(RETRAIN_CKPT), use_region=True, use_extra_cond=False)
        ds = Bag3dDataset(); ds.initialize(opt, phase="test")
        cls.item = ds[0]

    def _load(self, use_ema):
        opt = bge.build_opt(self.device, ckpt=str(RETRAIN_CKPT), use_region=True,
                            use_extra_cond=False, use_ema=use_ema)
        opt.ddim_steps = self.DDIM
        m = self._Model(); m.initialize(opt)
        return m

    def _gen(self, model, guidance, seed=0):
        torch = self.torch
        data = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v)
                for k, v in self.item.items() if torch.is_tensor(v)}
        torch.manual_seed(seed); np.random.seed(seed)
        with torch.no_grad():
            sdf = model.inference(data, ddim_steps=self.DDIM, uc_scale=guidance)
        return sdf.detach().cpu().numpy()[0, 0]

    def test_knobs_take_effect(self):
        m_ema = self._load(True)
        ema_g1, ema_g1b, ema_g3 = self._gen(m_ema, 1.0), self._gen(m_ema, 1.0), self._gen(m_ema, 3.0)
        del m_ema; self.torch.cuda.empty_cache()
        m_raw = self._load(False)
        raw_g1 = self._gen(m_raw, 1.0)
        mad = lambda a, b: float(np.abs(a - b).mean())
        self.assertLess(mad(ema_g1, ema_g1b), 1e-6, "same seed+knobs must be deterministic")
        self.assertGreater(mad(ema_g1, ema_g3), 1e-4, "guidance scale must flow to inference")
        self.assertGreater(mad(ema_g1, raw_g1), 1e-4, "use_ema toggle must swap EMA<->raw weights")


if __name__ == "__main__":
    unittest.main(verbosity=2)
