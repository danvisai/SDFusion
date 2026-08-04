# Build spec — reference/retrieval-conditioned snap ("give the model a real building to imitate")

**Date:** 2026-06-30 · **Status:** SPEC (plan). · **Picked by user over 4 other context-injection options.**
**Goal:** when the user adds a crude mass to the base, the localized SDEdit snap should make it read as a
coherent BUILDING — by retrieving a SIMILAR REAL building from our corpus and conditioning the snap on it
as a reference, so the output inherits real building-ness (massing/structure/rhythm). Body stays bit-exact
(localized inpainting); only the edit region is regenerated, now guided by the reference.
**Companion:** `reference-context-conditioning-for-buildingness` memory (the full menu), the deployed/finetuned
prior (`project-layer1-data-sourcing-progress`), `COHERENT_ADD_PRIMITIVE_BUILD_SPEC_2026-06-15.md` (the no-image swap).

## 0. Doctrine guardrail
This is a CONDITIONING improvement to the EXISTING snap (massing prior + localized inpainting), NOT a new
model/line and NOT sampling from noise. Reference transfers CHARACTER, not an exact copy (see Dynamic Routing).

## 1. Best paper — Phidias (arXiv 2409.11406, reference-augmented 3D diffusion)
Directly supports reference-augmented **3D completion** + **interactive coarse-guidance** = our add-to-base case.
Three mechanisms we adopt; one we discard:
- ✅ **Reference injection** via a control branch + cross-attention.
- ✅ **Dynamic Reference Routing**: reference coarse (low-res) at high-noise t, finer at low-noise t →
  transfers massing/structure, avoids detail-copy conflicts with imperfect references.
- ✅ **Self-Reference Augmentation + curriculum**: building is its own reference under aug (resize/flip/
  distort/shift) + occasional retrieved ref; start near-identical, ramp difficulty → NO paired data needed.
- ❌ Phidias' multi-view **Canonical Coordinate Maps** — only needed because its backbone is image→3D. We
  are SDF-native: the reference goes straight through our VQVAE encoder. (Same no-image swap as coherent-add.)

## 2. ⭐ The SDF-native adaptation
| Phidias (image→3D) | OUR replacement (in hand) |
|---|---|
| reference → multi-view CCMs | reference building **SDF → VQVAE latent** (16³×3) — direct 3D, richer than views |
| Meta-ControlNet over CCMs | reference latent → tokens → **cross-attention** in the existing Stage3a spatial transformer (context_dim=512) |
| Uni3D image↔pointcloud retrieval | **FootprintEmbedNet** (have) + class/style/height → kNN over `real.h5` (35k) |
| concept image | our symbolic vector (footprint/class/style/height) — already conditioned |

## 3. Mechanism
- **Retrieve:** given the edited building's footprint/class, kNN the corpus for the nearest real building
  (exclude near-duplicates of self at train time). Reference = its SDF.
- **Encode + inject:** VQVAE-encode the reference → latent; project to reference tokens; cross-attend in the
  UNet alongside the existing global-context token. (Alt: ControlNet-style additive branch.)
- **Dynamic routing:** downsample the reference latent by timestep — coarse at high-noise (massing), full at
  low-noise (structure). Implement as a t-dependent avgpool on the reference tokens.
- **Localized blend unchanged:** `out = edited·(1-w) + snapped·w` (body bit-exact).

## 4. Training (warm-start the 6k cross-cultural prior; self-reference, no paired data)
1. Add the reference encoder + cross-attn (gated `use_reference`; default off → old ckpt still loads).
2. Per step: reference = the target building itself with augmentations (resize/flip/grid-distort/shift) OR,
   with ramping prob, a **retrieved** similar real building. **Reference dropout** (~0.1) so it still works
   reference-free (CFG over the reference). Curriculum: start near-identical refs, increase aug + retrieved.
3. Optionally fold in the existing `data/part_edit_pairs/` (input=cruder edited, reference=clean similar,
   target=clean) so it also learns the add-mass scenario explicitly.
4. Finetune from `continue-stage3a-xcultural-warmstart-ft-final` (the 6k prior). Short (Phidias: ~10k steps).

## 5. Reuse
`models/networks/retrieval/footprint_embed.py` (retrieval) · the VQVAE (reference encoder) · Stage3a UNet
cross-attn (`models/stage3a_model.py` `_build_global_context` / spatial transformer) · `real.h5` (corpus) ·
the 6k prior (warm-start) · `scripts/server/refine.py` snap_volume (localized blend, inference).

## 6. Lighter probe FIRST (no retrain — de-risk before the encoder+finetune)
Reference-guided SDEdit without training: retrieve the nearest real building, VQVAE-encode it, and **blend
its latent into the SDEdit start / denoising trajectory in the edit region** (init the masked latent from the
reference instead of pure noise; or a small guidance pull toward the reference latent). Crude, but tells us
whether reference info actually transfers building-ness before we build the encoder + retrain. Quick A/B vs
the current snap on the DE/JP towers.

## 7. Eval
Localized before/after (à la `scripts/sdedit_localized_dejp.py`): does the reference-conditioned snap turn the
placed mass into a coherent element that matches the reference's character, across NL/DE/JP, body bit-exact?
Ablate: no-ref vs self-ref vs retrieved-ref; dynamic-routing on/off.

## 8. Sources
Phidias 2409.11406 · MV-RAG 2508.16577 · ReMoDiffuse 2304.01116 · reference-guided inpainting 2507.23058 ·
RAD 2412.09191 · (retrieval index: our FootprintEmbedNet).
