# Latent-space corrector de-risk — result (#59)

Ticket [#59](https://github.com/danvisai/SDFusion/issues/59) of [map #58](https://github.com/danvisai/SDFusion/issues/58).
**Verdict: NEGATIVE — correcting the diffusion's latent hits the same ~0.0047 ceiling as SDF-space
correction (#54). Post-hoc correction is doubly ruled out.**

## What was built
`scripts/foundations/train_latent_corrector.py` + `LatentCorrectorUNet3D` (residual 3D U-Net on the
16³×3 raw VQVAE latent, zero-init identity, `models/networks/refine_unet.py`) + a CPU identity-contract
test (`scripts/foundations/test_latent_corrector.py`). Reuses `train_refiner.py`'s SDEdit-aligned pair
machinery, moved into latent space: pairs `(z_wavy = encode(SDEdit-corrupted GT), z_clean = encode(GT))`
+ on-the-fly σ-latent-noise pairs. Codec AND diffusion frozen. Eval on the diffusion's ACTUAL held-out
real samples (n=24); "before" = `decode(z)`, "after" = `decode(g(z))` (both round-tripped, isolating the
corrector's own effect). Honest shaded montages.

## Results (roughness, n=24; GT floor 0.0041, #54 SDF-refiner plateau 0.0047)

| variant | roughness before → after | note |
|---|---|---|
| pure latent-L1 (`w_decode 0`) | 0.00502 → **0.00493** | near-identity — latent L1 dropped 4× (0.063→0.015) yet decoded roughness barely moved |
| + decoded-space loss (`w_decode 2`) | 0.00502 → **0.00472** | reaches the #54 plateau, no further; fp-IoU 0.873→0.877 |

Montages: `outputs/latent_corrector/montage.png` (pure), `outputs/latent_corrector_wdec/montage.png`
(w_decode; the committed `latent-corrector-montage.png`). In both, the "corrected" column is visually
indistinguishable from "wavy" — lumpy, nowhere near GT.

## Findings

1. **Minimizing latent L1 is decoupled from decoded crispness** — the corrector drove latent L1 down 4× but
   the decode barely changed. Latent distance ≠ crispness.
2. **A decoded-crispness loss reaches the #54 plateau (0.0047) but no further** — latent-space correction
   lands on the *exact same ceiling* as SDF-space correction (#54), and stays visibly lumpy.
3. ⇒ **Correcting the diffusion's OUTPUT — in either SDF or latent space — cannot recover crispness the
   diffusion never produced.** The cheap post-hoc lever (map #58's premise) is exhausted.

## Escalation
The fix must change what the diffusion *produces*, not correct it afterward. Next: **x0-sharp diffusion
finetune** — finetune the 947M diffusion (warm-start from map-#24) with a decoded-crispness / manifold
objective so its own sampled latent lands crisp. If that also plateaus, the durable **query-based implicit
decoder** ([map #52](https://github.com/danvisai/SDFusion/issues/52) option 2) is the answer.
