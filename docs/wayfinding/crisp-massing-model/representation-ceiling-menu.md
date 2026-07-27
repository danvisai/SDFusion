# Representation-ceiling menu (#55)

Research ticket for [map #52 "Crisp clean massing"](https://github.com/danvisai/SDFusion/issues/52). What
options raise the crispness ceiling **inside our model**, and what does each cost/buy — read in light of
what [#54](https://github.com/danvisai/SDFusion/issues/54) + [#56](https://github.com/danvisai/SDFusion/issues/56)
already proved about *where* the ceiling actually is.

## Current representation (grounded in the configs)

- **VQVAE** (`configs/vqvae_bnet.yaml`, deployed = `vqvae_clean.pth`): 64³ SDF truncated ±0.2 → **16³ × 3**
  latent (`ch_mult [1,2,4]` = 2 downsamples → 64/4=16; `embed_dim/z_channels 3`; codebook `n_embed 8192`;
  inference decodes continuous via `decode_no_quant`). ≈ **21× compression** (262 144 → 12 288 scalars).
- **Diffusion** (`configs/stage3a_sdf_diffusion.yaml`): latent UNet, `image_size 16`, `in_channels 4`
  (3 latent + 1 footprint concat), `model_channels 224`, `context_dim 512`; ~947M params total incl. the
  frozen FootprintEmbedNet. `scale_factor 2.38`.
- A **v2 codec config exists but is untrained** (`vqvae_bnet_v2.yaml`: ch 64→96, res_blocks 1→2, one
  bottleneck attn, trunc 0.2→0.3) — a quality bump at the **same 16³×3 latent / same 21× compression**.

## The reframe: #54 + #56 already located the ceiling

- **#56:** `decode(encode(GT))` = **0.0044 ≈ GT 0.00412** — the VQVAE reconstructs full 64³ SDFs *crisply*.
  **The codec and the 64³ grid are NOT the crispness bottleneck** (a crisp building is representable at 64³).
- **#54:** no post-decode refiner (L1 or eikonal+normal, any weight) beats ~0.0047 — the crisp info is not
  locally recoverable from the wavy decoded field.
- **⇒ The bottleneck is the DIFFUSION producing a latent that decodes wavy.** So every option below is judged
  by one question: **does it make the diffusion produce a crisper result (or move crispness off the diffusion)?**
  Options that only improve the *codec* are near-useless for us.

## The menu

| option | changes | helps OUR bottleneck? | cost |
|---|---|---|---|
| VQVAE **v2** (ch96 / attn, same 16³×3) | codec recon quality | **~No** — codec already crisp (0.0044) | med (retrain codec + diffusion) |
| **Continuous KL-VAE** (drop VQ) | removes codebook quantization | **~No** — inference already `decode_no_quant` & crisp | med |
| **128³ input** (→ 32³ latent) | finer voxel grid | **Low for this** — 64³ GT is already crisp (#56); helps *future fine detail*, not the wavy-wall problem | **High** (≈4× compute/mem + 128³ data rebuild) |
| **Larger latent** (32³×3, or 16³×8) | less compression (2× downsample / more channels) | **Maybe** — more room for a "crisp" mode, but a bigger latent is *harder* to denoise (could wash out) | med-high (retrain codec + diffusion) |
| **Bigger / better diffusion** (more UNet capacity; x0-sharp or latent-manifold objective; latent-space refiner) | the diffusion's accuracy directly | **Yes** — hits the actual bottleneck; cheapest thing that does | **low-med** (retrain/finetune diffusion only; codec untouched) |
| **Query-based implicit / vecset decoder** (diffusion emits a latent *set*/triplane; an MLP decodes SDF at any resolution) | where crispness comes from | **Yes, structurally** — crispness becomes a *decode* property, not something the diffusion must nail on a dense grid; the frontier design, rebuilt in our stack | **High** (new AE + new diffusion head) |

## Recommendation

Two options actually target our bottleneck; the rest improve the codec/grid, which #56 proved is not our limit
(explicitly **deprioritize VQVAE v2, KL-VAE, and 128³** for the crispness goal).

1. **Cheapest-first de-risk — improve the diffusion's latent accuracy** (bigger UNet capacity, and/or an
   x0-sharp objective, and/or a **latent-space** correction that snaps the sampled latent toward the
   `encode(GT)` manifold). Low-med cost (diffusion-only, codec frozen). **Caveat:** #54 showed post-hoc
   correction in *SDF* space plateaus; a *latent*-space correction is untested and has more leverage (the crisp
   codec then does the decode), so it's worth one experiment — but it may hit the same manifold-accuracy wall.
2. **Durable fix — a query-based implicit decoder** (vecset/triplane, built into our stack per the
   no-off-the-shelf-frontier-model constraint). This is the option most likely to actually reach GT crispness,
   because it stops asking the diffusion to place a crisp surface on a coarse dense grid. High cost (a real
   new-AE + new-diffusion effort), but it is *the* representation change the evidence points to.

## Implication for the map

The map's wayfinding is essentially **complete**: the way to crisp clean massing is **not** a composite (#56) or
a post-decode refiner (#54) — it is a **retrain-scale change targeting the diffusion/representation**, cheapest-first
(latent-accuracy) then durable (implicit decoder). Executing either is a fresh **implementation effort**, not more
charting. The crisp target is provably reachable (codec 0.0044); the remaining work is getting the diffusion's
output onto that manifold.
