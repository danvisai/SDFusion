# Diagnosis: the roughness is in the diffusion PRIOR's SDF field — not the codec, resolution, or render

Resolves [Diagnose the source of the generated massing's surface roughness](https://github.com/danvisai/SDFusion/issues/35).

## Method

On the from-scratch LoD2 checkpoint, the A/B/C ladder over 6 held-out buildings (`scripts/foundations/diagnose_surface_roughness.py`): **A** real GT SDF, **B** VQVAE no-quant round-trip of the GT (the decode ceiling), **C** full prior sample. Metrics: IoU(B,A), IoU(C,A), and surf-ratio (boundary/occupied) for each; plus mid-height SDF-field slices to separate field noise from render artifact.

## Results

| measure | value |
|---|---|
| IoU(round-trip B, GT A), median | **0.995** |
| surf-ratio GT (A) / round-trip (B), median | 0.159 / **0.160** (≈ equal) |
| surf-ratio prior sample (C), median | 0.153 (≈ or *below* A — does **not** capture the roughness) |

Visual (`outputs/surface_roughness/ladder_montage.png`, `slice_montage.png`):
- **A ≈ B** — both crisp blocks; the SDF-slice zero-contour is a clean sharp rectangle in both.
- **C** — rough/wavy/blobby geometry, and its SDF **field is genuinely noisy** (ragged zero-contour, mottled interior, spurious exterior blobs).

## Verdict

**The roughness originates in the diffusion prior's sampled SDF field.**
- **Not the VQVAE decode ceiling** — the round-trip reproduces the crisp GT almost perfectly (IoU 0.995, matching surf-ratio, clean field). The codec can represent crisp surfaces.
- **Not the 64³ resolution / VQVAE capacity** — same evidence rules this out; no higher-res pipeline needed.
- **Not a marching-cubes / threshold artifact** — the noise is in the SDF *field* itself (the slice), not introduced at mesh time.
- **surf-ratio (boundary/volume) does not measure the roughness** — confirms the #36 fidelity metric must be **normal/field-based** (normal-consistency), not boundary-count.

Residual note: mild vertical striations appear even in A/B — axis-aligned 64³ voxel walls via marching-cubes; resolution-inherent, present in real data, and distinct from the prior's field noise.

## Implication for the fix (#37)

The fix is **prior-side** — a smoothness regularizer on the sampled field, better/longer prior training, or sampling changes (guidance / more DDIM steps / EMA weights). It is **not** a VQVAE upgrade or a resolution increase; the expensive higher-res path is ruled out.
