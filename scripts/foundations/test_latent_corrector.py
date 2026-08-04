"""Contract tests for the #59 latent-space corrector (mirrors
test_baseline_gate_eval.py::TestLoadRefiner, applied to LatentCorrectorUNet3D's
(B,3,16,16,16) raw-VQVAE-latent space instead of RefineUNet3D's (B,1,R,R,R) decoded-SDF
space). Pure CPU, no GPU/model load: LatentCorrectorUNet3D's output conv is zero-init
(models/networks/refine_unet.py), so an UNTRAINED corrector is provably the identity map.

Run: env -u LD_PRELOAD ./sdfusion/bin/python -m unittest scripts.foundations.test_latent_corrector -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

from models.networks.refine_unet import LatentCorrectorUNet3D
from scripts.foundations.train_latent_corrector import save_corrector_ckpt, load_latent_corrector


class TestLatentCorrectorIdentity(unittest.TestCase):
    def test_untrained_corrector_is_identity_and_shape_preserving(self):
        torch.manual_seed(0)
        corrector = LatentCorrectorUNet3D(channels=3, base=8, delta_scale=1.0)
        x = torch.randn(2, 3, 16, 16, 16)  # (B,C,16,16,16), the raw VQVAE latent shape
        with torch.no_grad():
            y = corrector(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x, atol=1e-6),
                        "an untrained (zero-init output layer) corrector must be the identity map")

    def test_identity_holds_at_the_default_base_width(self):
        # same contract at the script's default base=48 (heavier, but still CPU-cheap at 16^3).
        torch.manual_seed(1)
        corrector = LatentCorrectorUNet3D(channels=3, base=48, delta_scale=1.0)
        x = torch.randn(1, 3, 16, 16, 16)
        with torch.no_grad():
            y = corrector(x)
        self.assertTrue(torch.allclose(y, x, atol=1e-6))

    def test_saved_and_loaded_checkpoint_round_trips_architecture(self):
        torch.manual_seed(0)
        corrector = LatentCorrectorUNet3D(channels=3, base=8, delta_scale=0.7)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "corrector.pth"
            save_corrector_ckpt(corrector, path, channels=3, base=8, delta_scale=0.7, step=0)
            loaded = load_latent_corrector(str(path), device="cpu")
        x = torch.randn(1, 3, 16, 16, 16)
        with torch.no_grad():
            y = loaded(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x, atol=1e-6))
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()),
                          "a loaded corrector is a frozen post-process, like load_refiner")


if __name__ == "__main__":
    unittest.main(verbosity=2)
