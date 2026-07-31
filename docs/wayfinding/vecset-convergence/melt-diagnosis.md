# What the melt is made of

Ticket: [Diagnose the melt: residual noise, decoder intolerance, or weak conditioning?](https://github.com/danvisai/SDFusion/issues/73)
· Map: [#69](https://github.com/danvisai/SDFusion/issues/69)

| what | where |
|---|---|
| diagnostic | `scripts/foundations/diagnose_decoder_tolerance.py` |
| artifact | `execution/artifacts/decoder_tolerance.json` |
| montage | `decoder-tolerance-montage.png` (this folder) |

**Answer: decoder intolerance — to *off-manifold* latent error specifically. And the diagnostic that
motivated the residual-noise hypothesis (`cos 0.707 → 0.935, this is a working denoiser`) measures
nothing about output quality.**

The ticket's experiment needed a trained denoiser and both A2 checkpoints are void (#78). It didn't
need one: *"what does a latent this close to the truth decode to?"* is a property of the decoder alone.

## The measurement — n=16 of the #71 held-out ids, 3 independent noise directions per cosine

| latent | cos to true | 3D IoU vs perfect decode | p10..p90 | missing |
|---|---|---|---|---|
| perfect | 1.000 | 1.000 | — | 0.000 |
| **reencoded** (same mesh, fresh point sample) | **0.083** | **0.999** | 0.998..0.999 | 0.000 |
| iso cos 0.999 | 0.999 | 0.995 | 0.981..0.997 | 0.002 |
| **iso cos 0.995** | 0.995 | **0.053** | 0.035..0.099 | **0.945** |
| iso cos 0.980 | 0.980 | 0.049 | 0.030..0.059 | 0.947 |
| iso cos 0.935 | 0.935 | 0.048 | 0.029..0.070 | 0.940 |
| iso cos 0.900 | 0.900 | 0.051 | 0.032..0.080 | 0.928 |
| iso cos 0.800 | 0.800 | 0.120 | 0.085..0.232 | 0.781 |
| iso cos 0.707 | 0.707 | 0.328 | 0.173..0.672 | 0.463 |

### 🔑 Cosine to the true latent predicts nothing

**A latent at cos 0.083 decodes perfectly. A latent at cos 0.995 is destroyed.** Re-encoding the *same
mesh* with a fresh surface sample makes FPS pick a different token **ordering**, which collapses the
element-wise cosine to 0.083 while the geometry is identical by construction — and the decode is
visually indistinguishable from perfect (IoU 0.999). Meanwhile 0.5% of isotropic noise loses 94.5% of
the volume.

So the tolerated *isotropic* error is under 0.5%, while a 92% "error" that happens to be a re-ordering
is free. The latent is a **set**; element-wise cosine measures ordering agreement, not geometry. #70
had already flagged cross-encode cosine as meaningless — this shows the same defect wrecks the
**in-distribution** `cos 0.707 → 0.935` diagnostic that this map inherited as evidence of a working
denoiser. That number is not evidence about output quality either way.

### The cliff, and the shape of the failure

The cliff sits between **cos 0.999 (IoU 0.995) and cos 0.995 (IoU 0.053)**. Failure progresses
visually as: **vertical slats → the melted-down look → rubble**, at roughly cos 0.995 → 0.935 → 0.707.
The map's melt corresponds to about cos 0.93.

The mechanism is a **global positive shift of the decoded field** (mean +0.586 → +0.959 on the probed
building), so the shape hollows out from the inside rather than roughening locally — which is why
`missing` is ~0.94 while `fp-IoU` stays as high as 0.92. It is not merely a mis-levelled field:
re-thresholding at the volume-matched isolevel recovers IoU 0.03 → 0.54 (measured on one building),
better but nowhere near intact.

⚠️ The non-monotonic partial recovery at cos ≤ 0.80 is real and reproducible across directions, not
noise: once the latent is mostly noise the decoder emits a generic blob whose volume happens to
overlap GT. It is not evidence of tolerance returning.

### Ribbing vs slats — two different things

Per #71 a Dora decode ribs at *every* cosine including 1.000 (fine regular corduroy, a meshing
artifact of a field ~32× too steep to place the surface within a voxel). The **slats** at cos 0.995 are
different — actual through-holes — and the montage shows both side by side, which is what makes them
separable. `a2-first-training-result.md`'s reading that the vertical slats came from an off-manifold
latent is **confirmed**, and now quantified.

## What this rules in and out

- ❌ **Residual noise** — not supported. Training longer moves the denoiser's latent-space accuracy,
  and latent-space accuracy is not the operative variable.
- ✅ **Decoder intolerance** — confirmed, and it is intolerance to *leaving the manifold*, not to
  distance as such. The decoder is perfectly happy with a wholly different-looking latent that is
  on-manifold.
- ❓ **Weak conditioning** — untested. It needs a valid checkpoint, so it waits on
  [#75](https://github.com/danvisai/SDFusion/issues/75).

**It also explains the razor-thin strength band** without appealing to the frame bug: the tolerance
ball is smaller than the smallest edit that changes anything, so there is no useful middle between
"no-op" and "collapse".

**The consequence for training:** the diffusion's entire supervision is `mse_loss(pred, noise)` in
latent space, and this shows that objective is decoupled from decode quality — corroborating #59's
finding by a different route. That is a direct input to
[#76](https://github.com/danvisai/SDFusion/issues/76).
