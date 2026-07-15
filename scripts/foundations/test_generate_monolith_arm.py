"""Contract tests for the monolith arm's held-out generation (ticket 13).

Fast + data-free: exercises the one pure seam -- unscaling `GaussianDiffusion.ddim_sample`'s
`TRUNC`-normalized output back to metric SDF units -- without touching the GPU model or
checkpoints. The generation pipeline itself is verified separately by an integration run (see
the ticket answer), matching this project's established convention for GPU-dependent code.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_generate_monolith_arm.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import generate_monolith_arm as gma  # noqa: E402


class UnscaleDdimOutputTest(unittest.TestCase):
    def test_multiplies_by_trunc(self):
        self.assertAlmostEqual(gma.unscale_ddim_output(1.0, trunc=0.2), 0.2)
        self.assertAlmostEqual(gma.unscale_ddim_output(-0.5, trunc=0.2), -0.1)

    def test_default_trunc_matches_monolith_pair_dataset(self):
        # frame_n_input's own trunc default, and datasets/monolith_pair_dataset.py's TRUNC --
        # the value the model was actually trained/normalized against.
        self.assertAlmostEqual(gma.unscale_ddim_output(1.0), 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
