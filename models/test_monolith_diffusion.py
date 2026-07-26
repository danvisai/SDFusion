"""Contract tests for the monolith's Gaussian diffusion process (ticket 11).

Fast + CPU-only: pure schedule math plus a tiny randomly-initialized UNet (no real checkpoint,
no GPU) to exercise `p_losses`/`ddim_sample`'s shape and determinism contracts.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     models/test_monolith_diffusion.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "networks"))
import monolith_diffusion as md  # noqa: E402
import monolith_unet as mu  # noqa: E402


class LinearBetaScheduleTest(unittest.TestCase):
    def test_shape_and_bounds(self):
        betas = md.linear_beta_schedule(100, beta_start=1e-4, beta_end=2e-2)
        self.assertEqual(betas.shape, (100,))
        self.assertAlmostEqual(float(betas[0]), 1e-4, places=6)
        self.assertAlmostEqual(float(betas[-1]), 2e-2, places=6)
        self.assertTrue((betas[1:] >= betas[:-1]).all())  # monotonically increasing


class AlphasCumprodTest(unittest.TestCase):
    def test_monotonically_decreasing_and_in_unit_interval(self):
        betas = md.linear_beta_schedule(50)
        ac = md.alphas_cumprod_from_betas(betas)
        self.assertEqual(ac.shape, (50,))
        self.assertTrue((ac[1:] <= ac[:-1]).all())
        self.assertTrue((ac > 0).all() and (ac <= 1).all())


class QSampleTest(unittest.TestCase):
    def test_shape_matches_x0(self):
        betas = md.linear_beta_schedule(10)
        ac = md.alphas_cumprod_from_betas(betas)
        x0 = torch.randn(2, 1, 4, 4, 4)
        noise = torch.randn_like(x0)
        t = torch.tensor([0, 9])
        out = md.q_sample(x0, t, noise, ac)
        self.assertEqual(out.shape, x0.shape)

    def test_near_t_zero_stays_close_to_x0(self):
        betas = md.linear_beta_schedule(1000, beta_start=1e-4, beta_end=2e-2)
        ac = md.alphas_cumprod_from_betas(betas)
        x0 = torch.full((1, 1, 4, 4, 4), 0.5)
        noise = torch.zeros_like(x0)
        out = md.q_sample(x0, torch.tensor([0]), noise, ac)
        self.assertTrue(torch.allclose(out, x0, atol=1e-2))

    def test_at_t_max_dominated_by_noise_not_x0(self):
        betas = md.linear_beta_schedule(1000, beta_start=1e-4, beta_end=2e-2)
        ac = md.alphas_cumprod_from_betas(betas)
        x0 = torch.full((1, 1, 4, 4, 4), 100.0)  # a huge x0 value
        noise = torch.zeros_like(x0)
        out = md.q_sample(x0, torch.tensor([999]), noise, ac)
        # at t=999, sqrt(alphas_cumprod) is small -- x0's huge magnitude should be heavily damped
        self.assertLess(float(out.abs().max()), 100.0)


class GaussianDiffusionTest(unittest.TestCase):
    def _net(self):
        return mu.MonolithUNet(base_channels=4, channel_mults=(1, 2), temb_dim=8)

    def test_p_losses_returns_finite_scalar(self):
        diff = md.GaussianDiffusion(self._net(), timesteps=50)
        x0 = torch.randn(2, 1, 8, 8, 8)
        coarse = torch.randn(2, 1, 8, 8, 8)
        loss = diff.p_losses(x0, coarse)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))

    def test_p_losses_is_nonnegative(self):
        diff = md.GaussianDiffusion(self._net(), timesteps=50)
        x0 = torch.randn(3, 1, 8, 8, 8)
        coarse = torch.randn(3, 1, 8, 8, 8)
        loss = diff.p_losses(x0, coarse)
        self.assertGreaterEqual(float(loss), 0.0)

    def test_zero_surface_weight_matches_plain_mse(self):
        # surface_weight=0 must reduce exactly to the original unweighted objective --
        # a regression guard for the ticket-11 loss-dilution fix.
        torch.manual_seed(0)
        net = self._net()
        x0 = torch.randn(2, 1, 8, 8, 8)
        coarse = torch.randn(2, 1, 8, 8, 8)
        t = torch.tensor([5, 20])
        noise = torch.randn_like(x0)
        diff = md.GaussianDiffusion(net, timesteps=50, surface_weight=0.0)
        weighted = diff.p_losses(x0, coarse, t=t, noise=noise)
        noisy = md.q_sample(x0, t, noise, diff.alphas_cumprod)
        pred = net(noisy, coarse, t)
        import torch.nn.functional as F
        plain = F.mse_loss(pred, noise)
        self.assertAlmostEqual(float(weighted), float(plain), places=5)

    def test_surface_weighting_increases_loss_for_an_error_inside_the_band(self):
        # A constant prediction error INSIDE the surface band (|x0| < surface_band) must be
        # penalized more than the identical error OUTSIDE it -- proves the weighting is
        # actually spatially selective, not a uniform rescale.
        class ConstantErrorModel:
            def __call__(self, x, coarse, t):
                return torch.zeros_like(x)  # predicts eps=0 everywhere

        diff = md.GaussianDiffusion(ConstantErrorModel(), timesteps=50,
                                    surface_band=0.3, surface_weight=4.0)
        noise = torch.ones(1, 1, 4, 4, 4)  # eps=0 prediction -> uniform squared error = 1 everywhere
        t = torch.tensor([0])

        x0_inside = torch.zeros(1, 1, 4, 4, 4)          # |x0| = 0 < 0.3 everywhere: all weighted
        x0_outside = torch.full((1, 1, 4, 4, 4), 0.9)     # |x0| = 0.9 > 0.3 everywhere: unweighted
        coarse = torch.zeros(1, 1, 4, 4, 4)
        loss_inside = diff.p_losses(x0_inside, coarse, t=t, noise=noise)
        loss_outside = diff.p_losses(x0_outside, coarse, t=t, noise=noise)
        self.assertGreater(float(loss_inside), float(loss_outside))

    def test_ddim_sample_output_shape(self):
        diff = md.GaussianDiffusion(self._net(), timesteps=50)
        coarse = torch.randn(1, 1, 8, 8, 8)
        out = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=0)
        self.assertEqual(out.shape, (1, 1, 8, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_ddim_sample_is_reproducible_given_the_same_seed(self):
        diff = md.GaussianDiffusion(self._net(), timesteps=50)
        coarse = torch.randn(1, 1, 8, 8, 8)
        a = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=42)
        b = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=42)
        self.assertTrue(torch.equal(a, b))

    def test_ddim_sample_differs_across_seeds(self):
        diff = md.GaussianDiffusion(self._net(), timesteps=50)
        coarse = torch.randn(1, 1, 8, 8, 8)
        a = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=1)
        b = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=2)
        self.assertFalse(torch.equal(a, b))

    def test_clip_x0_bounds_output_against_an_overconfident_eps_prediction(self):
        # An imperfectly-trained (or adversarially wrong) model predicting a huge, constant
        # epsilon should not blow up x0_pred at high t (dividing by a near-zero
        # sqrt(alphas_cumprod)) into a runaway that compounds over every remaining step --
        # this reproduces the exact failure mode caught against a real early checkpoint
        # (unclamped: outputs outside [-16, 7] on real coarse conditioning).
        class ExplodingEpsModel:
            def __call__(self, x, coarse, t):
                return torch.full_like(x, 100.0)

        diff = md.GaussianDiffusion(ExplodingEpsModel(), timesteps=50)
        coarse = torch.randn(1, 1, 8, 8, 8)
        clamped = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=10, seed=0, clip_x0=1.0)
        unclamped = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=10, seed=0, clip_x0=0.0)
        self.assertLess(float(clamped.abs().max()), float(unclamped.abs().max()))
        self.assertLess(float(clamped.abs().max()), 20.0)


class PredictX0Test(unittest.TestCase):
    """ticket 11's v3 follow-up: an alternative parameterization where the network predicts
    x0 directly instead of noise."""

    def _net(self):
        return mu.MonolithUNet(base_channels=4, channel_mults=(1, 2), temb_dim=8)

    def test_p_losses_targets_x0_not_noise(self):
        # Same shape as test_zero_surface_weight_matches_plain_mse, but for predict_x0=True:
        # the loss must be MSE(pred, x0), not MSE(pred, noise).
        torch.manual_seed(0)
        net = self._net()
        x0 = torch.randn(2, 1, 8, 8, 8)
        coarse = torch.randn(2, 1, 8, 8, 8)
        t = torch.tensor([5, 20])
        noise = torch.randn_like(x0)
        diff = md.GaussianDiffusion(net, timesteps=50, surface_weight=0.0, predict_x0=True)
        loss = diff.p_losses(x0, coarse, t=t, noise=noise)
        noisy = md.q_sample(x0, t, noise, diff.alphas_cumprod)
        pred = net(noisy, coarse, t)
        import torch.nn.functional as F
        expected = F.mse_loss(pred, x0)
        self.assertAlmostEqual(float(loss), float(expected), places=5)

    def test_ddim_sample_zero_init_output_matches_zero_prediction(self):
        # MonolithUNet's output conv is zero-initialized (forward always returns 0 for an
        # untrained model). Under predict_x0, x0_pred is then 0 at every step, so eps is
        # driven entirely by x itself -- confirm this runs and stays finite/bounded rather
        # than assuming eps-prediction's math still applies unchanged.
        diff = md.GaussianDiffusion(self._net(), timesteps=50, predict_x0=True)
        coarse = torch.randn(1, 1, 8, 8, 8)
        out = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=0)
        self.assertEqual(out.shape, (1, 1, 8, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_ddim_sample_reproducible_under_predict_x0(self):
        diff = md.GaussianDiffusion(self._net(), timesteps=50, predict_x0=True)
        coarse = torch.randn(1, 1, 8, 8, 8)
        a = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=7)
        b = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=5, seed=7)
        self.assertTrue(torch.equal(a, b))

    def test_clip_x0_still_bounds_output_under_predict_x0(self):
        # Under predict_x0, the model's output IS x0_pred directly -- an overconfident model
        # predicting a huge constant x0 must still be clamped to the declared data range.
        class ExplodingX0Model:
            def __call__(self, x, coarse, t):
                return torch.full_like(x, 100.0)

        diff = md.GaussianDiffusion(ExplodingX0Model(), timesteps=50, predict_x0=True)
        coarse = torch.randn(1, 1, 8, 8, 8)
        clamped = diff.ddim_sample(coarse, shape=(1, 1, 8, 8, 8), ddim_steps=10, seed=0, clip_x0=1.0)
        self.assertLessEqual(float(clamped.abs().max()), 1.0 + 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
