# #70 — the normalisation is fine. The training cache is not.

**Date:** 2026-07-29 · ticket
[Validate the vecset latent normalisation before any long run](https://github.com/danvisai/SDFusion/issues/70)
· map [#69](https://github.com/danvisai/SDFusion/issues/69)

**Verdict on the ticket: keep the global mean/std. Per-channel normalisation would be actively harmful,
and the handover's step-1 recommendation was wrong.**

**Verdict that matters more: the real-latent training cache is in a frame x↔z-transposed relative to both
the blockout cache and inference. Both A2 runs trained against it.**

Artifacts: `execution/artifacts/vecset_latent_stats.json`,
`execution/artifacts/latent_channel_snr.json`. Scripts:
`scripts/foundations/check_vecset_latent_stats.py`, `scripts/foundations/probe_latent_channel_snr.py`.

---

## 1. The frame bug

`scripts/foundations/precompute_vecset_latents.py` encodes the two caches by different paths:

```python
:83   z = codec.encode(Building(verts=verts_to_world(bv), faces=bf))   # blockout — converted
:85   z = codec.encode(Building(verts=v, faces=fc))                   # real — NOT converted
```

`verts_to_world` maps marching-cubes voxel-index space to the `[-1,1]` world frame. The stored SDF is
indexed **`[z, y, x]`** (handover trap #3), so vertices meshed from it and passed through
`verts_to_world` land in a frame whose axes run `(z, y, x)`. The ingested corpus surfaces are in true
`(x, y, z)` order. The two differ by exactly an **x↔z swap**, and line 85 skips the conversion.

### Measured, n=3 held-out, decoded through the real Dora decoder

| cache | decode vs reference, **identity** | decode vs reference, **x↔z swapped** |
|---|---|---|
| **blockout** (`vecset_blockout_latents.h5`) | **1.0000 / 1.0000 / 1.0000** ✅ | 0.5565 / 0.0963 / 0.4674 |
| **real** (`vecset_latents.h5`) | 0.5555 / 0.0928 / 0.4568 | **0.9976 / 0.9990 / 0.9982** ⚠️ |

The blockout cache is correct. The real cache is correct **only when transposed**, and at 0.998 it is
sitting exactly on the codec's round-trip ceiling — so the latents are *good latents of the wrong
orientation*, not corrupt ones.

Independent confirmations:
- Row keying is **not** the problem: `IoU(cache footprint, GT-implied footprint) = 1.0000` on all three.
- The live inference path is **not** the problem: re-encoding the GT mesh the way inference does and
  decoding gives 3D IoU **1.0000 / 0.9983 / 0.9980**, fp-IoU ~1.000.
- Cached blockout fp-IoU in the grid frame is **1.0000**.

### What this did to training

The two halves of every "aligned pair" were in **different frames**. Pair training therefore taught the
model:

> given a correctly-framed blockout, produce the **x↔z-transposed** building.

A learned axis swap. That accounts for every symptom on the record without needing any other explanation:

| observed | explained by |
|---|---|
| run 1 shredded blockouts into vertical slats at every usable strength | the model learned a transposed manifold; a correctly-framed blockout is off it |
| run 2 "coherent but eroded and pitted", 3D IoU 0.840 → **0.611** | at s=0.5 the learned transpose is *partially* applied — a blend of the building and its transpose |
| razor-thin band: 0.35 and 0.65 collapse, only 0.5 works | low s cannot apply the learned transform (no-op); high s applies it fully (collapse); only mid-s blends to a middling score |
| "the generator replaces the blockout with its own guess" | its guess is the transpose of the answer |

**⚠️ Aligned-pair training made this worse, not better.** Run 1 merely learned a transposed manifold;
pairing a correct input against a transposed target taught the swap *explicitly*. The distribution-shift
diagnosis was right in kind and wrong in cause — it fixed the Gaussian-vs-blockout gap while leaving a
much larger frame gap in place, and the "fix" is what made the frame gap trainable.

### What it voids

**The map-level conclusion "the representation was never the bottleneck" is void.** It rested on A2
landing at 0.611 beside map-#24's 0.601. A2 was trained to reproduce transposed buildings; 0.611 measures
a frame bug, not a representational ceiling. Two of the six negatives (A2 run 1 and run 2) are
**void, not negative**.

**What still stands:** the blockout baseline (**fp-IoU 1.000 / 3D IoU 0.840**) — computed from
`blockout_sdf` directly, never through the cache. The Dora round-trip ceiling (~0.999). The codec-is-not-
the-bottleneck finding. Everything in
[Research: which supervision signals make implicit generators sharp](https://github.com/danvisai/SDFusion/issues/72),
whose §5c point about latent-only supervision this only reinforces.

### Cost to fix

Re-encode `vecset_latents.h5` with `verts_to_world` applied — one line, then one pass over 35,623
buildings on an A100. **[inferred from cache mtimes]** the original passes took roughly 2–3 h each, so
budget that. The blockout cache does **not** need re-encoding. Every checkpoint trained against the old
cache is void.

⚠️ **A cheap guard is warranted, because this class of bug is now two-for-two** (inverted winding in #62,
frame transpose here) and both passed their verification: assert, at cache-write time, that decoding one
sample reproduces its own footprint at fp-IoU ≈ 1. That would have caught this in seconds.

---

## 2. The normalisation question, answered

### Distribution (train split; 600 rows × 2048 tokens per cache)

| | real | blockout |
|---|---|---|
| global mean / std | −0.0251 / **0.8390** | +0.0114 / 0.8405 |
| per-channel std: min / median / max | 0.0218 / 0.7790 / 1.2908 | 0.0204 / 0.7656 / 1.5274 |
| **per-channel std spread (max/min)** | **59.3×** | **74.8×** |
| after global norm, channels outside [0.5, 2.0] | **25.0%** | 26.6% |
| excess kurtosis: min / median / max | −1.40 / +0.25 / **+3.60** | −1.43 / −0.02 / +5.84 |
| tail: fraction \|z\| > 3σ | **0.524%** | 0.390% |
| max \|z\| | **7.92σ** | 6.54σ |

The global std is already **0.839 ≈ 1**, so global normalisation is close to scale-neutral. The blockout
cache normalised by the *real* cache's statistics — what `LatentSet.__getitem__` actually does — comes out
at mean +0.0435, std 1.0017, and the per-channel std profiles of the two caches correlate at **0.965**. So
cross-cache normalisation is sound; that was a reasonable worry and it is not a problem.

### The 59× spread is a real mechanism — on paper

The projection noises to `t_start = strength·(T−1)` and adds unit-variance ε to every channel alike, so
per channel `SNR_c = ᾱ·σ_c² / (1−ᾱ)`:

| strength | t | ᾱ | SNR at σ=1 | channels SNR<1 | <0.1 | <0.01 |
|---|---|---|---|---|---|---|
| 0.15 | 149 | 0.9407 | 15.87 | 15 | 7 | 0 |
| 0.20 | 199 | 0.8987 | 8.87 | 16 | 10 | 1 |
| 0.35 | 349 | 0.7199 | 2.57 | 16 | 15 | 6 |
| **0.50** | 499 | 0.4938 | **0.976** | **39** | **16** | **10** |
| 0.65 | 649 | 0.2692 | 0.368 | 64 | 16 | 14 |

At the only strength that works (s=0.5), a well-scaled channel sits at SNR ≈ 1 and **16 channels are
below 0.1** — the schedule erases them and the denoiser cannot recover what is gone.

### But those channels are dead — so per-channel normalisation would be harmful

A near-zero-variance channel in a KL-regularised latent is usually a *collapsed* dimension. Tested by
ablating through the real decoder, n=8 held-out, medians:

| arm | IoU vs full decode | fp-IoU | 3D IoU |
|---|---|---|---|
| full (reference) | — | 0.436 | 0.433 |
| **16 low-variance channels → their mean** | **1.0000** | 0.436 | 0.433 |
| **16 low-variance channels ← a different building** | **1.0000** | 0.436 | 0.433 |
| **16 highest-variance channels → mean** *(control)* | **0.0000** | 0.004 | 0.000 |

Blanking all 16 low-variance channels changes the decode **not at all**. Substituting *another
building's* values changes it not at all. The control destroys the shape completely, so the ablation is
maximally sensitive and the null result is real.

**Therefore: keep the global mean/std.** Per-channel normalisation would divide 16 dead channels by stds
as small as 0.026, amplifying encoder noise to unit variance and spending **a quarter of the denoiser's
channel capacity** modelling noise the decoder discards. The handover named per-channel normalisation as
step 1; it is the wrong move, and this is the measurement that shows it.

*(The `full` reference arm reads 0.436 / 0.433 rather than ~0.999 — that is the frame bug of §1, measured
here from the other direction.)*

### Two smaller things worth carrying

- **`SetSDEdit.x0_clamp = 3.0`** is justified in-source by *"training normalises latents to unit variance,
  so ~3 sigma is the honest bound."* The global std is 0.839, but the distribution is heavy-tailed —
  **0.52% of entries exceed 3σ and the tail reaches 7.92σ.** The clamp truncates real content. Modest, but
  it is an assumption stated in a comment that the data does not quite support.
- **Element-wise cosine between two independently-encoded latents is meaningless.** Encoding the same
  corpus mesh twice gives cos ≈ 0.06–0.10 against the cached latent, because FPS over a fresh random point
  sample yields a different token *ordering* — a vecset is order-agnostic in meaning but not in storage.
  The in-distribution `cos 0.707 → 0.935` diagnostic is unaffected (it compares noised-then-denoised
  versions of the *same* latent), but no cross-encode cosine should be read as a similarity.

---

## 3. Answer to the ticket

1. **Per-channel statistics:** spread is 59×, and 25% of channels fall outside [0.5, 2.0] after global
   normalisation — but the 16 worst are collapsed dimensions the decoder ignores.
2. **Does the round-trip survive normalisation?** Normalise/denormalise are exact float32 inverses, so the
   literal check is vacuous. The meaningful version — does decoding the cached latent reproduce the
   building — **fails at 0.436 fp-IoU, because of the frame bug, not the normalisation.**
3. **Global or per-channel?** **Global. Keep it.** Evidence in §2.

**The long run must not start until the real latent cache is re-encoded.** That is exactly what this
ticket existed to establish — *"if this is wrong, no length of run fixes it"* — and the thing that was
wrong turned out not to be the normalisation.
