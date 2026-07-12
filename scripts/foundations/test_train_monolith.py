"""Contract tests for the monolith trainer's checkpoint/resume seam (ticket 11).

Fast + CPU-only + data-free: a tiny model in a temp directory, no GPU, no real BuildingNet
pairs. This is the PRD's own "training smoke test" contract -- "validate loading,
checkpointing, resume... They do not assert that a model converges" -- as an actual automated
test rather than only a manual real-data run. The real-data training loop itself is verified
separately by an integration run (see the ticket answer).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_train_monolith.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "networks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "networks"))
import train_monolith as tm  # noqa: E402
from monolith_unet import MonolithUNet  # noqa: E402


def _tiny_model_and_opt():
    net = MonolithUNet(base_channels=4, channel_mults=(1, 2), temb_dim=8)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    return net, opt


class CheckpointRoundTripTest(unittest.TestCase):
    def test_save_then_load_restores_the_step_count(self):
        net, opt = _tiny_model_and_opt()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pth"
            tm.save_checkpoint(path, net, opt, step=1234, config={"lr": 1e-3})
            net2, opt2 = _tiny_model_and_opt()
            step = tm.load_checkpoint(path, net2, opt2, device="cpu")
            self.assertEqual(step, 1234)

    def test_save_then_load_restores_exact_weights(self):
        net, opt = _tiny_model_and_opt()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p))  # move off zero-init so a mismatch is detectable
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pth"
            tm.save_checkpoint(path, net, opt, step=1, config={})
            net2, opt2 = _tiny_model_and_opt()
            tm.load_checkpoint(path, net2, opt2, device="cpu")
            for p1, p2 in zip(net.parameters(), net2.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_resume_after_further_training_does_not_silently_lose_the_checkpoint(self):
        # Save at step 10, keep training the ORIGINAL model further (weights drift), then load
        # the step-10 checkpoint into a fresh model -- it must reflect step-10 weights, not the
        # drifted ones, proving the file (not in-memory state) is the resume source of truth.
        net, opt = _tiny_model_and_opt()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pth"
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(1.0)
            tm.save_checkpoint(path, net, opt, step=10, config={})
            snapshot = [p.clone() for p in net.parameters()]
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(5.0)  # simulate more training happening
            net2, opt2 = _tiny_model_and_opt()
            step = tm.load_checkpoint(path, net2, opt2, device="cpu")
            self.assertEqual(step, 10)
            for p_snap, p_loaded in zip(snapshot, net2.parameters()):
                self.assertTrue(torch.equal(p_snap, p_loaded))

    def test_atomic_write_leaves_no_tmp_file_behind(self):
        net, opt = _tiny_model_and_opt()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "ckpt.pth"
            tm.save_checkpoint(path, net, opt, step=1, config={})
            leftovers = [p for p in path.parent.iterdir() if p.name != path.name]
            self.assertEqual(leftovers, [])


class CheckpointDigestTest(unittest.TestCase):
    def test_same_content_gives_same_digest(self):
        net, opt = _tiny_model_and_opt()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pth"
            tm.save_checkpoint(path, net, opt, step=1, config={})
            self.assertEqual(tm.checkpoint_digest(path), tm.checkpoint_digest(path))

    def test_different_content_gives_different_digest(self):
        net, opt = _tiny_model_and_opt()
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d) / "a.pth", Path(d) / "b.pth"
            tm.save_checkpoint(p1, net, opt, step=1, config={})
            tm.save_checkpoint(p2, net, opt, step=2, config={})
            self.assertNotEqual(tm.checkpoint_digest(p1), tm.checkpoint_digest(p2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
