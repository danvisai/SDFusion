# A2, first training run — NEGATIVE, and the baseline is the story

**Date:** 2026-07-28 · 6,000 steps, 49.4M params, 34,909 train latents, n=20 held-out.
**Scored on footprint-IoU and 3D IoU** — not roughness, which is anti-correlated with the goal here.

| arm | fp-IoU | 3D IoU |
|---|---|---|
| **blockout (the input)** | **1.000** | **0.837** |
| projected, strength 0.2 | 0.998 | **0.347** |
| projected, strength 0.4 | 0.261 | 0.023 |
| projected, strength 0.6 | 0.197 | 0.042 |
| *map-#24 deployed (generated)* | *0.863* | *0.601* |

**Retention of codec ceiling — dense grid 0.601/0.995 = 0.604; vecset 0.347/0.999 = 0.347.**

## Two findings, and the second matters more

**1. The projection actively degrades its own input.** 3D IoU falls 0.837 → 0.347 at the gentlest
strength and collapses beyond that. This model is worse than doing nothing. Including the blockout as a
scored arm is what makes that legible — 0.347 alone would have read as "a number", not "worse than the
input it was handed".

**2. 🔑 The plain extruded footprint beats the model we ship.** The blockout scores **0.837** 3D IoU
against the deployed generator's **0.601**, at **fp-IoU 1.000 vs 0.863** — and it costs a signed EDT, no
network at all. That was not what this run was built to measure, and it is the most immediately useful
thing it produced.

It also corroborates [#51](https://github.com/danvisai/SDFusion/issues/51) from a different direction:
that prototype found the procedural *base* already crisp and footprint-exact, and recommended fixing it
rather than swapping in map-#24. This says the same thing quantitatively.

## What this does NOT establish

**This is not a verdict on A2.** The run is short and the loss was **still descending** at the end
(0.1381 → 0.1351 over the last 500 steps, from 0.72). Reading a 6,000-step result as an architecture
verdict would repeat exactly the error the frozen gate already taught — drawing a conclusion from a
measurement whose instrument or setup was not yet sound.

Specific suspects, none yet tested:
- **Undertrained.** 6k steps at batch 8 is ~1.4 epochs over 35k latents.
- **Latent normalisation.** A single global mean/std is applied to the vecset latent; if that
  distribution is heavy-tailed the cosine schedule is not well-posed on it, which would hurt most at
  low strength — where the damage is in fact worst.
- **Sampler settings.** 20 DDIM steps from t≈200; an imperfect ε makes the x₀ estimate drift, and the
  ±6 clamp is loose relative to unit-scaled latents.

## Recommendation

Do **not** schedule a long campaign on this evidence. The cheap, decisive next checks, in order:

1. **Sanity-check the latent statistics** — per-channel rather than global normalisation, and confirm
   the round-trip `decode(encode(x))` survives normalise/denormalise. If normalisation is wrong, every
   training step is fighting it and no length of run fixes it.
2. **Train longer only after that** — the loss curve says there is headroom, but headroom is worthless
   if the objective is mis-scaled.
3. **Independently: take the blockout result to [#50](https://github.com/danvisai/SDFusion/issues/50).**
   A signed-EDT extrusion beating the deployed generator, on the criteria that matter, is a
   demo-facing win available today and needs no model at all.

---

# Follow-up (same day): the model works — it is a DISTRIBUTION-SHIFT problem

The "undertrained" hypothesis was wrong, and a cheap diagnostic settled it.

## The model denoises correctly on the distribution it saw

Noise a **training** latent to strength s, denoise, compare to the true latent:

| strength | cos(recovered, true) | cos(noised, true) |
|---|---|---|
| 0.10 | **0.995** | 0.986 |
| 0.20 | **0.989** | 0.949 |
| 0.30 | **0.980** | 0.889 |
| 0.50 | **0.935** | 0.707 |

At every strength it pulls **back** toward truth — 0.707 → 0.935 at s=0.5. This is a working denoiser.

## But blockout latents are off-distribution

Same operator, same strength, on a blockout instead of a noised real latent: 3D IoU **0.290** at s=0.2,
versus cos **0.989** recovery in-distribution. The strength sweep shows a cliff rather than a slope:

| strength | fp-IoU | 3D IoU |
|---|---|---|
| blockout (input) | 1.000 | **0.840** |
| 0.05 / 0.10 | 1.000 | **0.840** (no-op) |
| 0.15 | 1.000 | 0.822 |
| 0.20 | 0.995 | **0.290** (collapse) |

Below the cliff the model does nothing; above it, the latent leaves the codec's manifold and the vecset
decoder returns shredded geometry — the vertical slats visible in `a2-comparison.png`.

**Note:** the "retention 0.841" the eval script prints at s=0.1 is the *blockout's* score passing
through unchanged, not a generator achievement. That line is misleading and should be read with the
blockout arm beside it.

## Diagnosis and fix

The model was trained on **noised real-building latents** and is asked at inference to start from a
**blockout latent** it has never seen. Training and inference distributions do not match.

The fix is the standard one, and this repo already has precedent: train on **aligned pairs** —
blockout latent in, real-building latent out — rather than on Gaussian corruption of real latents
alone. That is the pattern `train_refiner.py` uses (SDEdit/σ-aligned pair corruption), noted on map #58.

This also explains the sampler-stability symptom without needing a sampler fix: the clamp was tightened
to the latent scale (±3σ) and it did not rescue s=0.2, because the problem is *where the trajectory
starts*, not how far each step is allowed to move.
