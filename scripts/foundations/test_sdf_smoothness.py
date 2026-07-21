"""Contract tests for the Phase-2 SDF-field smoothness regularizer (map #34).

Pure tensor tests on CPU — no model, no GPU. Verifies the regularizer targets waviness
(curvature) while leaving crisp planar walls free, so it can smooth without erasing edges.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_sdf_smoothness.py
"""
from __future__ import annotations
import os, sys, unittest
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/foundations"))
from models.stage3a_model import _sdf_field_smoothness  # noqa: E402


def _grid(D=12):
    """Coordinate grid centered at 0, unit spacing, shape (D,) of axis indices."""
    return torch.arange(D, dtype=torch.float32) - (D - 1) / 2.0


def _flat(a, D=12):
    """A planar-wall SDF: linear along the D axis, broadcast to (1,1,D,D,D)."""
    return a.view(1, 1, -1, 1, 1).expand(1, 1, D, D, D).contiguous()


class TestSmoothness(unittest.TestCase):
    def test_flat_ramp_grad_tv_near_zero(self):
        # A planar wall = an SDF that is linear along one axis: zero curvature everywhere.
        sdf = _flat(_grid())
        val = _sdf_field_smoothness(sdf, sdf, sigma=0.2, kind="grad_tv").item()
        self.assertLess(val, 1e-5, "a flat/linear wall must be ~free under grad_tv")

    def test_wavy_field_grad_tv_positive_and_larger(self):
        a = _grid()
        flat = _flat(a)
        # add a high-frequency ripple -> nonzero curvature -> penalized
        ripple = 0.3 * torch.sin(1.6 * a).view(1, 1, -1, 1, 1).expand(1, 1, 12, 12, 12)
        wavy = flat + ripple
        v_flat = _sdf_field_smoothness(flat, flat, sigma=0.2, kind="grad_tv").item()
        v_wavy = _sdf_field_smoothness(wavy, wavy, sigma=0.2, kind="grad_tv").item()
        self.assertGreater(v_wavy, 1e-4)
        self.assertGreater(v_wavy, 10 * v_flat + 1e-6, "waviness must cost more than a flat wall")

    def test_sharp_corner_grad_tv_penalizes_but_eikonal_preserves(self):
        # The edge-erosion tradeoff (spec review c): grad_tv is a curvature penalty, so it DOES
        # cost at a genuine crisp corner (min of two planar walls) -- mitigated only by a small
        # weight. eikonal, by contrast, stays ~flat on a true metric corner (|grad|~1 on both
        # faces) -> it is the edge-PRESERVING alternative the PRD names. This test pins that
        # distinction so the choice of `kind` is a deliberate, tested tradeoff, not incidental.
        D = 12
        ax = _grid(D)
        x = ax.view(1, 1, -1, 1, 1)
        z = ax.view(1, 1, 1, 1, -1)
        corner = torch.minimum(x, z).expand(1, 1, D, D, D).contiguous()  # crisp exterior right-angle
        tv = _sdf_field_smoothness(corner, corner, sigma=0.6, kind="grad_tv").item()
        eik = _sdf_field_smoothness(corner, corner, sigma=0.6, kind="eikonal").item()
        self.assertGreater(tv, 1e-3, "grad_tv should register a crisp corner (edge-erosion risk)")
        self.assertLess(eik, tv, "eikonal must be gentler on a true metric corner (edge-preserving)")

    def test_eikonal_zero_on_unit_gradient_ramp(self):
        # |grad|=1 for a unit-slope ramp -> eikonal deviation ~0.
        a = _grid()
        sdf = a.view(1, 1, -1, 1, 1).expand(1, 1, 12, 12, 12).contiguous()
        val = _sdf_field_smoothness(sdf, sdf, sigma=0.2, kind="eikonal").item()
        self.assertLess(val, 1e-4, "unit-gradient SDF must have ~0 eikonal deviation")

    def test_eikonal_positive_on_scaled_gradient(self):
        a = _grid()
        sdf = (2.0 * a).view(1, 1, -1, 1, 1).expand(1, 1, 12, 12, 12).contiguous()  # |grad|=2
        val = _sdf_field_smoothness(sdf, sdf, sigma=0.2, kind="eikonal").item()
        self.assertGreater(val, 0.5, "|grad|=2 must deviate from the unit-norm target")

    def test_finite_on_random_and_differentiable(self):
        torch.manual_seed(0)
        sdf = torch.randn(2, 1, 10, 10, 10, requires_grad=True)
        for kind in ("grad_tv", "eikonal"):
            val = _sdf_field_smoothness(sdf, sdf.detach(), sigma=0.1, kind=kind)
            self.assertTrue(torch.isfinite(val).item())
            self.assertEqual(val.dim(), 0)  # scalar
            val.backward()
            self.assertTrue(torch.isfinite(sdf.grad).all().item())
            sdf.grad = None


VQ_CLEAN = REPO / "logs_building/vqvae_clean_ft/vqvae_clean.pth"
REAL_H5 = REPO / "data/real_massing_v1/real.h5"
_GPU_SMOKE = bool(os.environ.get("RUN_GPU_SMOKE")) and VQ_CLEAN.exists() and REAL_H5.exists()


@unittest.skipUnless(_GPU_SMOKE, "training smoke: set RUN_GPU_SMOKE=1 (needs the clean VQVAE + real.h5)")
class TestSmoothnessInTrainingLoss(unittest.TestCase):
    """PRD Testing Decisions (Phase 2): the regularizer term is finite AND its weight is applied,
    verified via a short training smoke. Gated + heavy (builds the real Stage3a training model).
    Deterministic double-forward: same seed/step, differing ONLY in sm_weight, so total must move
    by exactly (w2-w1) * smooth -- proving the weight flows into the loss, and that gating toggles
    the term. torch/model imports are lazy so the pure tests above stay light."""

    @classmethod
    def setUpClass(cls):
        from types import SimpleNamespace
        import retrain_prior_hybrid as rp
        from datasets.bag3d_dataset import Bag3dDataset
        from models.stage3a_model import Stage3aModel
        if not torch.cuda.is_available():
            raise unittest.SkipTest("no CUDA")
        args = SimpleNamespace(
            device="cuda", vq_ckpt="logs_building/vqvae_clean_ft/vqvae_clean.pth",
            finetune_from=None, use_smooth=1, smooth_weight=0.5, smooth_kind="grad_tv",
            smooth_sigma=0.05, smooth_every=1, lr=1e-4, total_iters=10,
            use_extra_cond=0, use_region=1, p_uncond=0.0, repa=0, repa_weight=0.5,
            repa_stop_frac=0.75, adaln=0, bag3d_h5=str(REAL_H5), bag_ratio=0.5)
        opt = rp.build_opt(args, ckpt_dir="/tmp")
        ds = Bag3dDataset(); ds.initialize(opt, "train")
        from torch.utils.data import DataLoader
        cls.batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
        cls.model = Stage3aModel(); cls.model.initialize(opt); cls.model.switch_train()

    def _forward_total(self, weight, seed=1234):
        m = self.model
        m.set_input(self.batch)
        m.sm_weight = weight
        torch.manual_seed(seed); m._step = 0
        m.forward()
        return float(m.loss_dict["total"]), m.loss_dict

    def test_gating_and_weight_applied(self):
        t0, ld0 = self._forward_total(0.0)
        t1, ld1 = self._forward_total(0.5)
        self.assertIn("smooth", ld1, "the smooth term must be logged when enabled")
        smooth_raw = float(ld1["smooth"])
        self.assertTrue(np.isfinite(smooth_raw) and smooth_raw >= 0.0)
        # deterministic: total moves by exactly weight * smooth -> the weight is applied.
        self.assertAlmostEqual(t1 - t0, 0.5 * smooth_raw, places=4,
                               msg="sm_weight must scale the term into the total loss")

    def test_disabled_drops_term(self):
        self.model.sm_enabled = False
        try:
            _, ld = self._forward_total(0.5)
            self.assertNotIn("smooth", ld, "no smooth term when the regularizer is gated off")
        finally:
            self.model.sm_enabled = True


if __name__ == "__main__":
    unittest.main(verbosity=2)
