# Why pair training does not carve

*2026-08-21. Measured on the caches arm N is training on, CPU only, no GPU touched.*

The A2 massing model has never removed material. `beats_envelope_rate` is 0.000 at every
checkpoint of every arm ever scored. This note establishes the mechanism, and it is not the
one the training code assumes.

## What the model is actually asked to learn

`scripts/train_vecset.py` spends 80% of its steps (`--pair_frac 0.8`) on an **aligned pair**
step: corrupt from the blockout latent, keep the target as the real latent. The implied
regression target is

    eps_target = sqrt(a)/sqrt(1-a) * (z_blockout - z_real) + eps

Everything the model can learn about carving lives in `z_blockout - z_real`. The second term is
unit-variance noise it cannot predict.

## The measurement

`z_blockout - z_real` is not a carve direction. It is mostly an arbitrary reordering.

A vecset latent is an unordered **set** of 2048 tokens. The two caches were encoded
independently, so token *k* of the blockout has no relation to token *k* of the real building.
On one row, the nearest real-token to each blockout-token sits at the same index **0.0%** of the
time -- yet cosine at the best match is **+0.68**. The information is present; the
correspondence is not.

| | pair distance | random *different* building | cos(pair) | per-token cos at same index |
|---|---|---|---|---|
| `vecset_blockout_latents_v2.h5` (arms A, N) | 1.3837 | 1.3965 | +0.035 | +0.041 |
| `vecset_blockout_latents_v2_aligned.h5` (unused) | **0.9962** | 1.3978 | **+0.502** | **+0.464** |

A building's own blockout is **no closer to it than a randomly chosen other building** -- 1.3837
against 1.3965, both within noise of sqrt(2) = 1.4142, the value for uncorrelated vectors. This
reproduces #90's independent finding (as-encoded +0.048, randomly permuted +0.0347).

## Does the gap track the geometry?

The test that matters: does the latent gap grow with how much actually needs removing?
n=150 training rows, geometric `extra` computed from `real.h5`.

| | corr | **rank corr** | gap on zero-carve rows | gap on carved rows | separation |
|---|---|---|---|---|---|
| unaligned | +0.336 | +0.436 | 1.373 | 1.391 | 0.018 |
| aligned | +0.575 | **+0.710** | 0.904 | 1.018 | **0.114** |

Alignment moves rank correlation 0.44 -> 0.71 and separates carve from no-carve **6x** better.

WARNING: **it does not fully fix it.** On rows where the box and the building are geometrically
*identical*, the aligned latents still sit **0.904** apart. So even with alignment roughly 89% of
the pair target is unrelated to carving -- real signal, but a modulation on a large floor.

## Arm N has already returned its verdict

`run_aligned_retrain.py` pre-registers the decision rule in `ARM_SPECS`:

> if NL+DE-only carves the data was binding, if it still does not then token order is,
> and arm B is what matters.

Arm N ran the full 240000 steps and never carved: `beats_env` is 0.000 on every corpus at
**all six** checkpoints (190k, 200k, 210k, 220k, 230k, 240k).

Japan gives the copy away directly -- its 3D IoU and its `vs_input` are the *same number* at
three checkpoints running (0.9915/0.9915, 0.9593/0.9593, 0.9875/0.9875). The score is the copy.

Germany is the sharpest evidence, because the model moved there and it cost it:

| [de] | 230000 | 240000 |
|---|---|---|
| `vs_input` (1.0 = did nothing) | 0.8373 | **0.9908** |
| 3D IoU | 0.6595 | **0.8158** |

At 230k it departed from its input and scored 0.66. At 240k it went back to handing the input
back and scored 0.82. **Moving less scored better**, and the run's best checkpoint (`vs_input`
0.9857 overall) is the one that does the least. That is the copy incentive measured directly,
not inferred.

**The rule has fired. Token order is binding.**

Only arms A and N have ever been run. **Arm B has never been run.**

## What follows

1. **Arm B** -- `run_aligned_retrain.py --arms B`. Pre-registered in #92, one command, control
   (arm A) already at 240k, single-variable against A by construction (#91 derives the aligned
   cache as a permutation of the same latents). No new code.
2. **Read out `beats_envelope_rate` per corpus.** It has been 0.000 forever; it is the metric
   that decides. Then sweep `--strength`, which is only informative once the target is real.
3. **The fork.** If arm B carves, the latent-transform thesis holds. If it still does not, the
   0.904 floor is the ceiling of this objective, and the supervision has to move to a
   representation where identical geometry means *zero* difference.

## Why voxels are the endpoint, not a detour

The floor exists because a latent token set has no canonical order and no exact identity. On a
voxel grid both problems vanish by construction: identical geometry gives exactly zero
difference, there is no permutation, and "added" and "removed" are labels rather than directions
in an opaque space.

The label is also well-posed. `missing` is 0.00000 on 714 of 714 held-out buildings -- the real
building is always entirely inside its start box. Nothing ever needs adding. Carving is therefore
exactly **binary segmentation of the start box** into keep and cut, with exact labels.

See `voxel-diffusion-fill-mechanisms.md` for DVD and ArchComplete, which supply the machinery.
