"""#188: contract for freezing, hashing and sealing the retrained A2 source for #119.

No GPU. The checkpoints here are tiny synthetic blobs with the same key structure
`train_vecset.py` writes, so the freeze path is exercised for real rather than mocked.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.seal_a2_source import (  # noqa: E402
    DROP_KEYS, build_seal, freeze_checkpoint, sha256_of,
)


def _checkpoint(step=240000, region_free=True):
    blob = {
        "model": {"inp.weight": torch.zeros(4, 4),
                  **({} if region_free else {"region.weight": torch.zeros(3, 4)})},
        "opt": {"state": {"huge": torch.zeros(256, 256)}},
        "step": step,
        "args": {"width": 512, "depth": 8, "heads": 8, "timesteps": 1000,
                 "region_free": region_free},
        "latent_mu": -0.024, "latent_sd": 0.839,
        "latent_channels": 64, "footprint_res": 64,
        "n_regions": 0 if region_free else 3,
    }
    return blob


class TestFreeze(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.src = self.d / "vecset_denoiser.pth"
        torch.save(_checkpoint(), self.src)

    def test_drops_optimizer_state(self):
        """70% of a vecset checkpoint is optimizer state needed only to RESUME training."""
        dest = self.d / "frozen.pth"
        freeze_checkpoint(self.src, dest)
        frozen = torch.load(dest, map_location="cpu", weights_only=False)
        for key in DROP_KEYS:
            self.assertNotIn(key, frozen)

    def test_keeps_everything_a_consumer_needs(self):
        """latent_mu/latent_sd are load-bearing: without them the denoiser decodes to noise."""
        dest = self.d / "frozen.pth"
        freeze_checkpoint(self.src, dest)
        frozen = torch.load(dest, map_location="cpu", weights_only=False)
        for key in ("model", "step", "args", "latent_mu", "latent_sd", "latent_channels",
                    "footprint_res", "n_regions"):
            self.assertIn(key, frozen, f"freezing dropped {key}, which a consumer needs")

    def test_returns_the_digest_of_the_file_it_actually_wrote(self):
        """The digest must be of the FROZEN file, not the source -- they differ by the opt state,
        and #119 seals whatever digest this returns."""
        dest = self.d / "frozen.pth"
        digest = freeze_checkpoint(self.src, dest)
        self.assertEqual(digest, sha256_of(dest))
        self.assertNotEqual(digest, sha256_of(self.src))

    def test_refuses_to_overwrite_an_existing_seal(self):
        """A sealed source is immutable; silently replacing one would invalidate every citation."""
        dest = self.d / "frozen.pth"
        freeze_checkpoint(self.src, dest)
        with self.assertRaises(FileExistsError):
            freeze_checkpoint(self.src, dest)

    def test_re_freezing_does_not_reproduce_the_digest(self):
        """⚠️ Measured, not assumed: `torch.save` is NOT byte-reproducible.

        Two freezes of the identical blob produce different file digests. That is why immutability
        (the test above) is the enforced property and re-derivability is not: a reader cannot
        re-freeze the training checkpoint and expect to recover a sealed digest, so the sealed FILE
        has to be the artifact that is kept.
        """
        a, b = self.d / "a.pth", self.d / "b.pth"
        self.assertNotEqual(freeze_checkpoint(self.src, a), freeze_checkpoint(self.src, b))

    def test_the_frozen_file_still_loads_and_carries_the_same_weights(self):
        """Whatever the container bytes do, the tensors must survive the strip unchanged."""
        import torch
        dest = self.d / "frozen.pth"
        freeze_checkpoint(self.src, dest)
        before = torch.load(self.src, map_location="cpu", weights_only=False)["model"]
        after = torch.load(dest, map_location="cpu", weights_only=False)["model"]
        self.assertEqual(sorted(before), sorted(after))
        for key in before:
            torch.testing.assert_close(before[key], after[key])


class TestSealRecord(unittest.TestCase):
    def _seal(self, verdict="PASS", **over):
        kwargs = dict(
            checkpoint=Path("weights/massing-vecset/a2_bw_regionfree.pth"),
            digest="a" * 64, step=240000, region_free=True,
            operating_point={"strength": 0.5, "steps": 20, "guidance": 1.0},
            cohort={"digest": "b" * 64, "n": 60000},
            verdict={"verdict": verdict, "clear_kill": verdict != "KILL",
                     "numeric_pass": verdict == "PASS", "clauses": []},
            eval_artifact="execution/artifacts/massing_arms_eval_188_candidate.json",
        )
        kwargs.update(over)
        return build_seal(**kwargs)

    def test_records_the_operating_point_119_seals_alongside_the_weights(self):
        """#188: 'record the generation operating point (strength / steps / guidance)'."""
        self.assertEqual(self._seal()["operating_point"],
                         {"strength": 0.5, "steps": 20, "guidance": 1.0})

    def test_records_the_digest_and_the_cohort_it_trained_on(self):
        seal = self._seal()
        self.assertEqual(seal["checkpoint"]["sha256"], "a" * 64)
        self.assertEqual(seal["cohort"]["digest"], "b" * 64)

    def test_records_that_it_is_region_free(self):
        self.assertTrue(self._seal()["checkpoint"]["region_free"])

    def test_names_the_source_it_replaces(self):
        """#115's amendment: only the identity of the frozen checkpoint changes."""
        self.assertIn("643aed08", self._seal()["replaces"]["sha256"])

    def test_refuses_to_seal_a_killed_checkpoint(self):
        with self.assertRaises(ValueError):
            self._seal(verdict="KILL")

    def test_refuses_to_seal_a_checkpoint_that_did_not_meet_the_bar(self):
        with self.assertRaises(ValueError):
            self._seal(verdict="NOT MET")

    def test_an_override_is_permitted_but_must_carry_a_recorded_reason(self):
        """#183 arm 3 ran past a KILL on an explicit owner override, and the record says so.
        The escape hatch exists; an unattributed one does not."""
        with self.assertRaises(ValueError):
            self._seal(verdict="NOT MET", override=True)
        sealed = self._seal(verdict="NOT MET", override=True, override_reason="owner ruling 2026-xx")
        self.assertEqual(sealed["acceptance"]["verdict"], "NOT MET")
        self.assertTrue(sealed["acceptance"]["override"])
        self.assertIn("owner ruling", sealed["acceptance"]["override_reason"])

    def test_a_malformed_digest_is_refused(self):
        with self.assertRaises(ValueError):
            self._seal(digest="not-a-sha")

    def test_round_trips_as_json(self):
        json.loads(json.dumps(self._seal()))


class TestDigest(unittest.TestCase):
    def test_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(b"x" * 100000)
            self.assertEqual(sha256_of(p), hashlib.sha256(b"x" * 100000).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
