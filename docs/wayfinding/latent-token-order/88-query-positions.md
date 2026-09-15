<!-- RECOVERED FROM THE ISSUE TRACKER, 2026-08-14. -->

> **Recovered and re-landed.** The original asset was lost with commit `e20f9af`; this file preserves
> its tracker record. The implementation was reconstructed in `d12cf33`, hardened in `74ba808` and
> `d6fcc8b`, and exercised at corpus scale for #91 in `8101143`. See `RECOVERY.md` for the recovery
> audit and reproduced measurements.


# #88 — Capture the codec's token query positions, which are computed and thrown away today

*State: closed · opened 2026-08-09 · implementation re-landed 2026-08-14*


## Ticket

Part of #87

## Question

The encoder selects each token's 3D query position by farthest-point sampling
(`michelangelo_autoencoder.py:135`) and then discards `idx`. Every alignment method needs those
positions. What is the cheapest way to capture and store them beside the latents, without re-encoding
twice and without changing what the encoder computes?

## Starting point

- The FPS shim the encoder actually uses is the repo's own, at `dora_roundtrip_probe.py:125` — it seeds
  at index 0 rather than randomly, so it is deterministic **given the point cloud**.
- The cloud comes from `sample_streams(mesh, 8192, 8192, rng)` with a **stateful** RNG, so consecutive
  encodes of the same mesh differ. That is the source of the reordering.
- 2048 positions x 3 floats x 35,623 buildings is ~875 MB as float32, ~437 MB as float16. Decide whether
  to store positions or to store a canonical permutation derived from them.
- ⚠️ Whatever is written must be verifiable. #78's `verify_frame` is the precedent: a write-time guard
  that refuses a bad cache rather than trusting the producer.

## Judged on

Positions recoverable for any cached latent, demonstrated by reproducing the measurement in
`docs/research/latent-token-order.md` from the stored data rather than by re-encoding.


---

## Comment — danvisai, 2026-08-09

## Resolved — and the question changed once the precompute was read

Code in `models/shape_codec.py`, `scene/surface_sampling.py`, `precompute_vecset_latents.py`. Writeup
`docs/wayfinding/latent-token-order/88-query-positions.md`, artifact
`execution/artifacts/token_order_from_cache.json` (`f582696`, `e20f9af`).

### 🔑 For the EXISTING caches there is no cheap way — there is no way at all

Two compounding causes, both measured:

1. **`DoraCodec` holds one generator that every `encode` advances**, so a latent's draw depends on every
   building encoded before it. Recovering one building's positions would mean replaying the whole pass
   in order — and the loop has a `[skip]` path, so one failure desynchronises everything after it.
2. **`sample_uniform` accepted an `rng` and ignored it**, calling `trimesh.sample.sample_surface` with no
   seed. The **coarse stream — the bulk of what the encoder sees — came from numpy's GLOBAL generator.**
   Every cached latent is a function of global interpreter state at write time; reseeding the codec
   could not have fixed it, because the codec's generator was never in charge.

So positions are captured *during* encoding, and the draw is made reproducible so this cannot recur.

### The decision the ticket asked for: **positions, not a permutation**

A permutation is derived from one latent's geometry — enough for a canonical sort, useless for matching,
which is a function of **two** position sets and is the direction #89 made the strong candidate. Storing
positions keeps both open; storing a permutation would commit the cache before #90 has chosen.
**fp16**: positions live in `[-1,1]` where fp16 resolves ~1e-3, **30× finer** than the 2/63 voxel pitch.
875 MB → **437 MB**.

### ⚠️ The guard I wrote first was useless, for a reason I had proven myself

Decode a latent at its own positions, require |sdf| ≈ 0 — reads well (0.0349 vs 0.9998 cross-building),
and **cannot catch a within-building permutation**, because #89 measured this decoder as
permutation-invariant at occupancy IoU 1.000000. Token *i* ↔ position *i* is exactly what #90/#91
consume. Caught in review.

`verify_positions` now **re-encodes** sampled rows through `encode_row` — the same function the write
loop uses, so it cannot validate its own copy:

| | smoke cache |
|---|---|
| stored vs re-encoded | **0.0002** |
| same building **shuffled** | **1.0584** ← invisible before |
| another building | 1.1148 |

### Acceptance: reproduced from disk, nothing encoded

`--from_cache`: elementwise **1.1000** · matched **0.0390** · random **1.1009** → **99.9% of the way to
random**, against `latent-token-order.md`'s 100.3% / 0.037 / 1.089. Refuses a cache without `query_pos`.

### 🔑 A correction this forced to #89

#89's matched-order control drew its point cloud separately from the encode, and with `sample_uniform`
ignoring its `rng`, the coarse half **did not correspond to the latent beside it**:

| ordering | #89 as published | corrected |
|---|---|---|
| as encoded | +0.0351 | +0.0480 |
| permuted | +0.0299 | +0.0347 |
| **matched** | **+0.3235** | **+0.7288** |

**The published figure understated the recoverable alignment by more than 2×.** Direction unchanged,
case for #90/#91 strengthened. ⚠️ Across three runs the gain has read 54.6× / 16.8× / 51.2× — direction
stable, magnitude not. Treat 0.73 as "there is a lot to recover", not a target.

### ⚠️ Scope boundary

**The production caches are NOT rewritten.** The shipped `v4_surf` checkpoint and #92's control depend on
them unchanged, and per-row seeding means a rebuild yields *different* latents. Corpus-scale capture
belongs to **#91**, which rebuilds anyway. Until then `--from_cache` works only on caches built after
this commit.

Also fixed a pre-existing test-order flake this exposed (`ContractSuite.setUp` now reseeds), and added
the CPU-only tests for `sample_uniform` honouring its `rng` that the bug had no coverage for.
Reviewed on both axes; 20 findings applied. 133 tests green.


---

## Comment — danvisai, 2026-08-12

Reopening: this ticket's resolution was implemented on a cloud A100 instance that was lost before the resolving commits (cited in the comment above) were pushed. None of those commits exist on any branch or PR in this repo — verified against `git log --all` and `git ls-remote`. The written analysis and decisions above are intact and should be treated as the spec; the implementation, and #91's rebuilt caches specifically, need to be redone from scratch.
