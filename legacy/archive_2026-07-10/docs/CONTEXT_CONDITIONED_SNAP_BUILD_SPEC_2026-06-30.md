# Build spec — context-conditioned snap ("give the model the right info → right building")

**Date:** 2026-06-30 · **Status:** SPEC. · Chosen after the reference/latent-blend path (#4) was shown dead
(spatial averaging can't transfer character — `reference-context-conditioning-for-buildingness` memory).
**Goal:** condition the localized inpainting snap on rich CONTEXT so a crude added mass becomes a coherent,
type-appropriate, rhythm-aligned BUILDING element. Body stays bit-exact; only the edit region is regenerated.
**Companions:** `COHERENT_ADD_PRIMITIVE_BUILD_SPEC_2026-06-15.md` (the no-image swap, edit-pairs, X-Part
locality, coherence losses — this spec reuses all of it), `RECOHERENCE_IMPROVEMENT_PLAN_2026-06-14.md`.

## 0. Doctrine guardrail
A CONDITIONING improvement to the EXISTING snap (massing prior + localized inpainting) — not a new line, not
sampling from noise. Retrieval-independent (unlike #4): the context is the ACTUAL building, always relevant.
Crisp geometry still procedural; only the DECISION (what element, coherent how) is learned.

## 1. ⭐ The principle: what makes an added mass read as a coherent building element
"Right info → right output." To turn a crude placed mass into the right element, the model must be told FOUR
things. Each maps to a failure we OBSERVED when it was missing:

| The model must know | Why | Failure when MISSING (observed) |
|---|---|---|
| ① **What's already there** — existing building geometry | so the addition attaches, matches scale, doesn't float | global SDEdit re-molds the whole body → blob |
| ② **What's new & where** — placed primitive shape + location | integrate THAT piece; its SHAPE hints the type (tall-thin→tower, flat→balcony, wall-patch→window, @ground→door) | no marker → model "cleans up"/suppresses the addition (the box-ref probe) |
| ③ **What it should become** — class/style/height/culture (+ optional element-type) | the architectural VOCABULARY | wrong-culture/wrong-element output |
| ④ **The grammar to align to** — floor lines, bay spacing, wall planes, existing parts | rhythm/alignment = reads as architecture | floating / misaligned / non-rhythmic elements |

①+② kill the blob/suppression failures; ③ gives the right vocabulary; ④ is what elevates "coherent
blob-free element" → "architecturally rhythmic" (windows snap into rows, balconies on floor lines).

## 2. ⭐ What the context CONTAINS (concrete) — minimal → rich
Two injection modes: **SPATIAL** = concat 64³ (or 16³ latent) channels to the UNet input (geometry/masks/
grids); **GLOBAL** = tokens/FiLM (semantics). Build as ablatable layers A → A+B → A+B+C.

**A. Geometric/spatial context (concat channels) — the load-bearing minimum:**
- `known_body_sdf` — the kept building OUTSIDE the edit region (= what to be coherent with). [①]
- `edit_mask` — where the addition is (so only that region regenerates; the learned analog of the post-hoc
  locality blend). [②]
- `added_primitive_sdf` — the crude placed mass itself: the marker AND the shape signal (shape→element). [②]
- `footprint / wall-plane channel` (have) — exterior surfaces, so the element sits on a wall not in air. [④-lite]

**B. Semantic context (global tokens/FiLM — mostly already present):**
- `class, style, height, region` (have from the cross-cultural work). [③]
- *(optional)* `element_type` token (window/door/tower/balcony/dormer) — explicit intent, derivable from the
  placed-mass shape (the SHAPE→ARCH rule in `facade_shape_to_arch_sheet`) or user-picked. [③]

**C. Structural-grammar context (the architectural layer — richest, biggest lift):**
- `floor/bay grid` (floor pitch, bay spacing) — measured from the building or sampled (`facade_grammar.py`). [④]
- `existing part layout` (positions/types of windows/roofs/towers) — the rhythm to MATCH (= CoherentPartRefiner's
  part-set input via `part_instances.npz` / `derive_part_structure`). [④]
- `alignment lines / height bands` (from `derive_part_structure`). [④]

**Minimal sufficient set hypothesis:** **A + existing B** already fixes blob/suppression + gives right
vocabulary (this is "context-responsive conditional inpainting", doi 10.3390/su18083987). **C** is what adds
true architectural rhythm. The ablation (§5) is how we PROVE which info is actually needed.

## 3. Architecture (how it's fed)
- **Spatial (A):** extend the Stage3a UNet `in_channels` to concat (known_body_sdf, edit_mask, primitive_sdf,
  footprint). Gated `use_context` (default off → old ckpt loads; zero-init the new conv channels so it starts
  as identity). At latent res, encode each via the VQVAE or downsample.
- **Semantic (B):** existing global context vector + optional `element_type` embedding.
- **Grammar (C):** either render the floor/bay grid to a 64³ channel (spatial), or feed part tokens via the
  existing cross-attention (reuse the CoherentPartRefiner set encoder).
- **Learned localized snap:** with `edit_mask` + `primitive` as input, the model LEARNS to regenerate only the
  edit region coherently — superseding the post-hoc mask blend (which we keep as a safety floor).

## 4. Training (warm-start the 6k prior; self-supervised from clean targets; no new data needed)
- **Data:** `data/part_edit_pairs/` (built 2026-06-15: (clean) → (clean + crude added/moved/dup) + marker;
  LoD3 + BuildingNet). Target = the clean building/layout; input = the crude-edited one; context channels =
  known_body + mask + primitive. Plus on-the-fly SDF edit-pairs: ablate a region of a clean real building →
  crude input; clean = target.
- **Self-supervised:** the clean target IS the "right" element → the model learns (crude+context) → coherent.
- **Warm-start** from `continue-stage3a-xcultural-warmstart-ft-final` (the 6k cross-cultural prior). Short.
- Reuse the recohere-plan coherence losses (band-rhythm/spacing/wall-attach) + X-Part neighbour-locality for C.

## 5. ⭐ Methodology = the answer to "what info gives right outputs": ABLATE
Train variants **A**, **A+B**, **A+B+C**; localized before/after eval each (NL/DE/JP + varied edits). The
minimal config that yields coherent, type-correct, rhythm-aligned elements **is** the empirical answer to "what
must the context contain." Expect: A → no more blob/suppression; A+B → right vocabulary; A+B+C → architectural
rhythm. Report the lift per layer.

## 6. Reuse
`models/stage3a_model.py` (conditioning + UNet, extend in_channels) · the 6k prior (warm-start) · the localized
snap `scripts/server/refine.py:snap_volume` (keep as safety floor + inference path) · `data/part_edit_pairs/`,
`derive_part_structure`, `facade_grammar.py`, the CoherentPartRefiner spec (the §0 companion) ·
`sdedit_localized_dejp.py` (eval harness).

## 7. Eval / acceptance
Localized before/after on NL/DE/JP + a pronounced tower, a flat slab (→balcony?), a wall patch (→window?):
does the context-conditioned snap make the added mass a coherent, type-appropriate, rhythm-aligned element,
body bit-exact, beating the current snap? Gates `test_branches` 12/12 + `test_sculpt_flows` 17/17.

## 8. Sources
Context-responsive building inpainting (doi 10.3390/su18083987) · RAD 2412.09191 · OmniPart 2507.06165 /
CoPart 2507.08772 / X-Part 2509.08643 · FacAID 2406.01829 / Pro-DG 2504.01571 · our CoherentPartRefiner spec.
