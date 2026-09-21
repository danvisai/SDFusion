"""#84: contract for the per-sample surface weighting. Synthetic, fast, no GPU.

⚠️ These tests import `surface_term` from `scripts.train_vecset` and call **the production function**.
The first version of this file re-implemented the reduction locally and asserted against the copy. It
passed while the shipped code was an exact no-op at the flag values actually used, and the copy is why
nobody noticed for two runs. A test that re-implements its subject tests nothing.
"""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.train_vecset import (
    ExperimentRng, LatentSet, build_arg_parser, cache_paths, latent_moments, region_width,
    surface_term,
)


def _mk(n, p=16, err=1.0):
    """(got, tgt, w_t) with identical per-sample error, so only the weighting varies."""
    return torch.full((n, p), err), torch.zeros(n, p), torch.ones(n).reshape(n, 1)


class TestSurfaceTermNoWeighting(unittest.TestCase):
    def test_matches_the_plain_scalar_reduction(self):
        torch.manual_seed(0)
        got, tgt = torch.randn(4, 512), torch.randn(4, 512)
        w_t = torch.rand(4).reshape(4, 1)
        plain = (w_t * (got - tgt) ** 2).mean()               # the pre-#84 expression
        weighted, logged = surface_term(got, tgt, w_t)
        self.assertTrue(torch.allclose(plain, weighted, atol=1e-7))
        self.assertTrue(torch.allclose(plain, logged, atol=1e-7))


class TestSurfaceTermWeighting(unittest.TestCase):
    """The properties #84 asked for, checked on the real function."""

    def test_weighting_still_applies_at_surf_bs_1(self):
        """🔑 The regression this file exists for.

        `--surf_bs` defaults to 1, so exactly one sample carries the term. The original renormaliser
        divided by the mean over that selection -- a 1-element mean -- making the weight identically
        1.0 and the whole flag a no-op. Normalising by the CORPUS mean fixes it, and this is the test
        that would have caught it.
        """
        got, tgt, w_t = _mk(1)
        low, _ = surface_term(got, tgt, w_t, torch.tensor([0.4]), norm=0.8)
        high, _ = surface_term(got, tgt, w_t, torch.tensor([1.2]), norm=0.8)
        self.assertNotAlmostEqual(float(low), float(high),
                                  msg="weighting must still act when only one sample is selected")
        self.assertLess(float(low), float(high))

    def test_a_below_average_weight_exerts_less_pressure(self):
        """#84: 'lower weight on low-solidity / high-complexity footprints'."""
        got, tgt, w_t = _mk(4)
        w = torch.tensor([1.0, 1.0, 0.5, 0.5])
        base, _ = surface_term(got, tgt, w_t, w, norm=0.75)
        lo = got.clone(); lo[2] *= 2.0        # extra error on a LOW-weight sample
        hi = got.clone(); hi[0] *= 2.0        # the same on a HIGH-weight one
        d_lo = surface_term(lo, tgt, w_t, w, norm=0.75)[0] - base
        d_hi = surface_term(hi, tgt, w_t, w, norm=0.75)[0] - base
        self.assertLess(float(d_lo), float(d_hi))
        self.assertAlmostEqual(float(d_lo / d_hi), 0.5, places=5)

    def test_corpus_normalisation_preserves_total_pressure(self):
        """Redistribute, do not reduce.

        Raw per-region weights average 0.58. Applied unnormalised, the flag is also a 42% cut in
        --surf_weight and a gain cannot be attributed to either. Normalising by the corpus mean keeps
        the magnitude and changes only the distribution.
        """
        got, tgt, w_t = _mk(6)
        w = torch.tensor([0.387, 0.387, 0.574, 0.574, 0.779, 0.779])
        corpus_mean = float(w.mean())
        flat, _ = surface_term(got, tgt, w_t)
        raw, _ = surface_term(got, tgt, w_t, w, norm=None)
        norm, _ = surface_term(got, tgt, w_t, w, norm=corpus_mean)
        self.assertLess(float(raw), float(flat) * 0.7)          # unnormalised is a large silent cut
        self.assertAlmostEqual(float(norm), float(flat), places=5)

    def test_normalisation_does_not_depend_on_the_selected_window(self):
        """⚠️ The other half of the original bug: the divisor was the mean over `sel`.

        A window that happened to be all-low-weight got weight 1.0 across the board, so the same
        building was penalised differently depending on who it was batched with. A corpus constant
        cannot do that.
        """
        got, tgt, w_t = _mk(1)
        a, _ = surface_term(got, tgt, w_t, torch.tensor([0.387]), norm=0.58)
        got2, tgt2, w_t2 = _mk(3)
        b, _ = surface_term(got2, tgt2, w_t2, torch.tensor([0.387, 0.387, 0.387]), norm=0.58)
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_logged_magnitude_is_always_the_unweighted_one(self):
        """`surf_hist` must stay comparable to runs without the flag."""
        got, tgt, w_t = _mk(3)
        weighted, logged = surface_term(got, tgt, w_t, torch.full((3,), 0.2), norm=1.0)
        self.assertLess(float(weighted), float(logged))
        self.assertAlmostEqual(float(logged), 1.0, places=5)


class TestExperimentRandomness(unittest.TestCase):
    """#92: the 2x2 must share stochastic training draws without changing inference semantics."""

    def test_seed_is_a_public_training_cli_option(self):
        args = build_arg_parser().parse_args(["--seed", "92"])
        self.assertEqual(args.seed, 92)

    def test_same_seed_replays_pair_batch_and_diffusion_draws(self):
        a = ExperimentRng(92, "cpu")
        b = ExperimentRng(92, "cpu")
        self.assertEqual(a.pair_random(), b.pair_random())
        torch.testing.assert_close(a.randn((2, 3)), b.randn((2, 3)), rtol=0, atol=0)
        torch.testing.assert_close(a.rand((4,)), b.rand((4,)), rtol=0, atol=0)
        torch.testing.assert_close(
            torch.randperm(12, generator=a.loader),
            torch.randperm(12, generator=b.loader),
            rtol=0,
            atol=0,
        )

    def test_surface_queries_do_not_advance_the_training_stream(self):
        with_surface = ExperimentRng(92, "cpu")
        without_surface = ExperimentRng(92, "cpu")
        with_surface.surface_rand((8, 3))
        with_surface.surface_rand((8, 3))
        torch.testing.assert_close(
            with_surface.randn((4, 5)), without_surface.randn((4, 5)), rtol=0, atol=0
        )


class TestLatentMoments(unittest.TestCase):
    def test_matches_float32_moments_without_materialising_the_whole_cache(self):
        source = torch.arange(5 * 3 * 2, dtype=torch.float16).numpy().reshape(5, 3, 2)

        class ChunkOnly:
            shape = source.shape

            def __getitem__(self, item):
                self_slice = item[0] if isinstance(item, tuple) else item
                if not isinstance(self_slice, slice) or self_slice.stop - self_slice.start > 2:
                    raise AssertionError("latent_moments must read bounded row chunks")
                return source[item]

        mean, std = latent_moments(ChunkOnly(), chunk_rows=2)
        expected = source.astype("float32")
        self.assertAlmostEqual(mean, float(expected.mean()), places=6)
        self.assertAlmostEqual(std, float(expected.std()), places=6)


class TestLatentSetStorage(unittest.TestCase):
    @staticmethod
    def _cache(path: Path, rows, values, held, regions=None):
        with h5py.File(path, "w") as cache:
            cache["row"] = np.asarray(rows, np.int32)
            cache["held_out"] = np.asarray(held, np.uint8)
            cache["latent"] = np.asarray(values, np.float16).reshape(len(rows), 2, 1)
            cache["footprint"] = np.ones((len(rows), 4, 4), np.uint8)
            cache["height_m"] = np.arange(len(rows), dtype=np.float32)
            cache["region"] = (np.zeros(len(rows), np.int32) if regions is None
                               else np.asarray(regions, np.int32))

    def test_large_latents_stay_on_disk_and_blockouts_match_by_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            real, block = Path(tmp) / "real.h5", Path(tmp) / "block.h5"
            self._cache(real, [10, 20, 30], [1, 1, 2, 2, 3, 3], [0, 1, 0])
            self._cache(block, [30, 10, 20], [30, 30, 10, 10, 20, 20], [0, 0, 1])
            dataset = LatentSet(real, blockout_path=block)

            self.assertFalse(hasattr(dataset, "z"), "the multi-GB latent cache must remain lazy")
            self.assertEqual(dataset.latent_shape, (2, 1))
            self.assertEqual(len(dataset), 2)
            first, first_block, *_ = dataset[0]
            last, last_block, *_ = dataset[1]
            torch.testing.assert_close(first, torch.full((2, 1), -1.0))
            torch.testing.assert_close(last, torch.full((2, 1), 1.0))
            torch.testing.assert_close(first_block, torch.full((2, 1), 8.0))
            torch.testing.assert_close(last_block, torch.full((2, 1), 28.0))


class TestRegionFilter(unittest.TestCase):
    """#92 follow-up: PLATEAU (region 2) is LoD1, so its pair steps carry a zero target.

    The filter must drop those rows BEFORE the latent moments are taken. If it ran afterwards the
    normalisation would still carry the excluded corpus, and the noise schedule would be posed
    against a distribution the model never sees.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.real, self.block = tmp / "real.h5", tmp / "block.h5"
        # rows 10/20/30 in regions 0/1/2; blockouts stored in a different order on purpose
        TestLatentSetStorage._cache(self.real, [10, 20, 30], [1, 1, 2, 2, 3, 3], [0, 0, 0],
                                    regions=[0, 1, 2])
        TestLatentSetStorage._cache(self.block, [30, 10, 20], [30, 30, 10, 10, 20, 20], [0, 0, 0],
                                    regions=[2, 0, 1])

    def tearDown(self):
        self._tmp.cleanup()

    def test_unfiltered_keeps_every_corpus(self):
        dataset = LatentSet(self.real, blockout_path=self.block)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(sorted(int(x) for x in dataset.r), [0, 1, 2])

    def test_excluding_plateau_drops_only_region_two(self):
        dataset = LatentSet(self.real, blockout_path=self.block, regions=(0, 1))
        self.assertEqual(len(dataset), 2)
        self.assertEqual(sorted(int(x) for x in dataset.r), [0, 1])

    def test_moments_are_taken_after_the_filter(self):
        dataset = LatentSet(self.real, blockout_path=self.block, regions=(0, 1))
        # kept latents are 1,1,2,2 -> mean 1.5, sd 0.5. Region 2's value of 3 must not appear.
        self.assertAlmostEqual(dataset.mu, 1.5, places=5)
        self.assertAlmostEqual(dataset.sd, 0.5, places=5)

    def test_blockout_partners_still_match_by_row_after_filtering(self):
        dataset = LatentSet(self.real, blockout_path=self.block, regions=(0, 1))
        first, first_block, *_ = dataset[0]
        last, last_block, *_ = dataset[1]
        torch.testing.assert_close(first, torch.full((2, 1), -1.0))
        torch.testing.assert_close(last, torch.full((2, 1), 1.0))
        # row 10 -> blockout 10 -> (10 - 1.5) / 0.5 ; row 20 -> blockout 20 -> (20 - 1.5) / 0.5
        torch.testing.assert_close(first_block, torch.full((2, 1), 17.0))
        torch.testing.assert_close(last_block, torch.full((2, 1), 37.0))

    def test_flag_parses_a_comma_list(self):
        args = build_arg_parser().parse_args(["--regions", "0,1"])
        self.assertEqual([int(x) for x in args.regions.split(",")], [0, 1])
        self.assertIsNone(build_arg_parser().parse_args([]).regions)


class TestCachePaths(unittest.TestCase):
    """#188: the retrain trains on the legacy caches AND the new BuildingWorld ones, together.

    The legacy 35,623 rows already have both caches and must enter whole rather than be re-encoded
    at ~0.2 s/row; the BuildingWorld cohort is a separate file. Concatenating at read time keeps one
    source of truth per cache instead of a merged 34 GB copy that can drift from both.
    """

    def test_a_single_path_is_unchanged(self):
        self.assertEqual(cache_paths("a.h5"), ["a.h5"])

    def test_a_comma_separated_list_becomes_several(self):
        self.assertEqual(cache_paths("a.h5,b.h5"), ["a.h5", "b.h5"])

    def test_whitespace_around_a_separator_is_tolerated(self):
        self.assertEqual(cache_paths(" a.h5 , b.h5 "), ["a.h5", "b.h5"])

    def test_none_is_none(self):
        self.assertIsNone(cache_paths(None))

    def test_an_empty_entry_is_refused_rather_than_silently_dropped(self):
        """`a.h5,,b.h5` is a typo; silently training on two caches instead of three is the worst
        possible response to it."""
        with self.assertRaises(ValueError):
            cache_paths("a.h5,,b.h5")


class TestMultiCacheLatentSet(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.legacy, self.legacy_bo = tmp / "legacy.h5", tmp / "legacy_bo.h5"
        self.bw, self.bw_bo = tmp / "bw.h5", tmp / "bw_bo.h5"
        TestLatentSetStorage._cache(self.legacy, [10, 20], [1, 1, 2, 2], [0, 0], regions=[0, 1])
        TestLatentSetStorage._cache(self.legacy_bo, [10, 20], [5, 5, 6, 6], [0, 0], regions=[0, 1])
        TestLatentSetStorage._cache(self.bw, [90, 91], [3, 3, 4, 4], [0, 0], regions=[3, 8])
        TestLatentSetStorage._cache(self.bw_bo, [91, 90], [8, 8, 7, 7], [0, 0], regions=[8, 3])

    def test_reads_every_row_of_every_cache(self):
        ds = LatentSet([self.legacy, self.bw])
        self.assertEqual(len(ds), 4)
        self.assertEqual(sorted(int(r) for r in ds.r), [0, 1, 3, 8])

    def test_pairs_match_by_row_id_across_caches(self):
        """Blockout row order differs per file on purpose -- matching is by corpus id, not position."""
        ds = LatentSet([self.legacy, self.bw], blockout_path=[self.legacy_bo, self.bw_bo])
        self.assertTrue(ds.has_blockouts)
        self.assertEqual(len(ds), 4)
        for i in range(len(ds)):
            real, block, *_ = ds[i]
            # each fixture stores blockout value = real value + 4, so the pairing is checkable
            self.assertAlmostEqual(float(block[0, 0] - real[0, 0]) * ds.sd, 4.0, places=3)

    def test_latent_moments_span_every_cache(self):
        """The normalisation must see the whole training distribution, not just the first file."""
        both = LatentSet([self.legacy, self.bw])
        first = LatentSet([self.legacy])
        self.assertNotAlmostEqual(both.mu, first.mu, places=6)

    def test_a_single_path_still_works_unwrapped(self):
        """Every existing caller passes one path and must keep working untouched."""
        ds = LatentSet(self.legacy)
        self.assertEqual(len(ds), 2)

    def test_mismatched_cache_and_blockout_counts_are_refused(self):
        with self.assertRaises(ValueError):
            LatentSet([self.legacy, self.bw], blockout_path=[self.legacy_bo])


class TestRegionWidth(unittest.TestCase):
    """#188: the conditioning width follows the corpus, and 0 means region-free.

    The frozen A2 source carries `nn.Embedding(3, 512)` purely because nothing ever passed
    `n_regions`. On the new corpus that is an `IndexError` on every BuildingWorld row.
    """

    def test_region_free_is_zero(self):
        self.assertEqual(region_width(np.array([0, 3, 8]), region_free=True), 0)

    def test_otherwise_it_covers_the_largest_id_present(self):
        self.assertEqual(region_width(np.array([0, 1, 2]), region_free=False), 3)
        self.assertEqual(region_width(np.array([0, 3, 8]), region_free=False), 9)

    def test_an_empty_corpus_is_refused(self):
        with self.assertRaises(ValueError):
            region_width(np.array([], np.int32), region_free=False)

    def test_a_negative_region_id_is_refused(self):
        """#174 writes -1 as its pre-backfill sentinel; training on it would index backwards."""
        with self.assertRaises(ValueError):
            region_width(np.array([0, -1, 2]), region_free=False)


class TestRegionFreeFlag(unittest.TestCase):
    def test_the_flag_exists_and_defaults_off(self):
        args = build_arg_parser().parse_args([])
        self.assertFalse(args.region_free)

    def test_it_can_be_turned_on(self):
        args = build_arg_parser().parse_args(["--region_free"])
        self.assertTrue(args.region_free)

    def test_region_free_and_an_explicit_region_filter_are_mutually_exclusive(self):
        """`--regions 0,1` selects rows BY region; asking for that while removing the channel is a
        contradiction worth refusing rather than silently resolving one way."""
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--region_free", "--regions", "0,1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
