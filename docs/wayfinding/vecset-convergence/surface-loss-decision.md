# Should the diffusion get a decoded-surface loss?

Ticket: [Decide whether the diffusion needs a decoded-surface loss](https://github.com/danvisai/SDFusion/issues/76)
· Map: [#69](https://github.com/danvisai/SDFusion/issues/69)

| what | where |
|---|---|
| probe | `scripts/foundations/probe_surface_loss.py` |
| artifact | `execution/artifacts/surface_loss_probe.json` |

## Decision: **adopt** — and the convergence run should not start on the current objective

Stated precisely, because the distinction matters: the evidence that the current objective is
**inadequate** is strong and measured. The evidence that a surface term **fixes** the melt is not yet
in — that is what the run is for. What is settled is that spending 40 GPU-hours on the objective as it
stands would be spending them on a loss that cannot rank its own candidates.

## 1. The decoupling holds, and it is worse than "decoupled"

A loss is a **ranking device**: it must say which of two candidate outputs is better. So the question
isn't whether latent distance correlates loosely with quality — it's whether it *orders candidates
correctly*. Measured as Spearman rank correlation against decoded 3D IoU, n=12 buildings, 144
candidates drawn from two error families:

| candidate pool | n | L2 | L1 | cosine |
|---|---|---|---|---|
| **pooled (both families)** | 144 | **+0.120** | +0.121 | −0.113 |
| on-manifold only | 24 | +0.050 | +0.102 | −0.188 |
| off-manifold only | 120 | **−0.503** | −0.502 | +0.516 |

L2 and L1 are distances, so a working loss wants a **negative** correlation; cosine is a similarity, so
it wants **positive**. Read that way:

- **Within one error family the latent loss works** (ρ = −0.50). Add more isotropic noise, get a worse
  decode, and latent distance sees it.
- **Across families it is worthless and slightly wrong-signed** (ρ = **+0.12**). Pool on-manifold
  candidates with off-manifold ones — which is the situation any real training run is in, since a
  denoiser's outputs are not drawn from one error family — and the ordering collapses.

This is the quantitative form of [#73](https://github.com/danvisai/SDFusion/issues/73)'s result. #73
showed two individual points (cos 0.083 → IoU 0.999; cos 0.995 → IoU 0.053); this shows the ranking
fails systematically across a pool, which is the property a loss actually needs.

It also confirms [#59](https://github.com/danvisai/SDFusion/issues/59)'s dense-grid finding carries to
the vecset latent, as #72 suspected it would.

## 2. The gradient path is cheap, and the freeze is not automatic

`DoraCodec.query` and `.encode` both run under `torch.no_grad()`
(`models/shape_codec.py:152,160`), so today there is no gradient path at all. Measured with those
bypassed, decoder parameters set `requires_grad_(False)`:

| query points | step time | peak memory | latent grad | decoder params taking grad |
|---|---|---|---|---|
| 1,024 | 0.172 s | 2.50 GB | ✅ | 0 |
| 4,096 | 0.186 s | 3.38 GB | ✅ | 0 |
| 8,192 | 0.205 s | 5.04 GB | ✅ | 0 |
| 16,384 | 0.241 s | 8.37 GB | ✅ | 0 |
| 32,768 | 0.313 s | 15.01 GB | ✅ | 0 |

Three things fall out:

- **Gradients reach the latent and the decoder stays frozen.** Exactly 0 of its parameters take a
  gradient when `requires_grad` is off — the weights are untouched, gradients merely pass through.
- ⚠️ **The freeze is not the default.** Dora's 191.6M parameters ship with `requires_grad=True`; leaving
  them so puts **198 tensors** on the gradient tape for nothing — +16% memory and +19% time (5.86 GB
  and 0.243 s vs 5.04 GB and 0.205 s at 8,192 points).
- 🔑 **Cost is dominated by `decode`, not by query points.** 1,024 points cost 0.172 s and 32× that many
  cost 0.313 s. So the efficient shape is **many query points on few batch elements**, not the reverse.

For scale, a denoiser training step is **305 ms at batch 8 (7.48 GB)**. A surface term on one batch
element at 8,192 points adds ~67% to the step; on the full batch it would be several times over. The
practical recipe is a subset of the batch — or a reduced frequency — with a generous query count.

## 3. Which term

**SDF regression at sampled query points**, against the corpus's own signed field, applied to the
predicted x̂₀ latent. That is the codec's own reconstruction objective, and it is a *reconstruction*
term rather than a *smoothness* term — which the evidence insists on:

- ⚠️ **Not an eikonal or TV term.** StEik's warning is not theoretical here:
  [#60](https://github.com/danvisai/SDFusion/issues/60) measured `grad_tv` **diverging into rubble** at
  w=0.1 in the dense-grid stack, and stable-but-useless at w=0.05.
- 🔑 **Restrict it to low t, or weight it by ᾱ.** ε-prediction recovers x̂₀ = (x_t − √(1−ᾱ)·ε̂)/√ᾱ, and
  the 1/√ᾱ factor amplifies ε-error without bound at high t. That amplification is precisely the
  mechanism #60 identified behind its divergence. A surface term computed on a wildly wrong x̂₀ is noise
  with a large gradient — the worst combination.
- **Split the weighting sharp vs coarse** per #72 §5b, and select checkpoints on the sharp-region loss.
  ⚠️ Temper the expectation with [#74](https://github.com/danvisai/SDFusion/issues/74): the corpus is
  coarse (median 20 faces, and plateau's median is 12), so the sharp set is small — though 100% of
  meshes do have creases above 25°.

## 4. What this means for the convergence run

[Train the aligned-pair generator to convergence](https://github.com/danvisai/SDFusion/issues/75)
should **not** be launched on the objective as it stands. The map's own Notes say confounds get cleared
before a long run, never after, and *"a 40-GPU-hour run on a mis-scaled objective is the one way to
waste"* the compute available. A loss with ρ = +0.12 against the thing we are judging is a mis-scaled
objective by any reading.

Two honest caveats on the adoption:

- This decides the objective is **warranted and affordable**, not that it works. It is a hypothesis
  with a mechanism (#73: the decoder needs on-manifold latents; a surface term penalises decodes
  directly, which is the only signal that points that way) — not a measured fix.
- The unblocking edit is small (drop two `no_grad`s behind a flag, freeze the decoder explicitly) but
  it changes `ShapeCodec`'s contract, which currently promises no-grad. That belongs in the
  implementation, not smuggled in as a side effect.
