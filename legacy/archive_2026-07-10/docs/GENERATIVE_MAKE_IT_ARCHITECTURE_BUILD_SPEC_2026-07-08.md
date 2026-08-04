# Build spec — generative "Make it architecture" (retrieval-fit → crop inpainting)

**Date:** 2026-07-08 · **Status:** Phase R IN PROGRESS (started same session).
**Goal (user):** "Make it architecture" should behave like image inpainting guesses pixels —
a box placed on the roof becomes *contextual architecture* (another floor, a water tank, a
stair bulkhead, a dormer…), a subtracted region becomes an *architectural void* (arcade,
loggia, setback) — instead of snapping to one of ~7 hand-coded templates. The user's
diagnosis is correct: the current interpret feels procedural **because it is** (learned
typing → template construction), and the model "has no context."

## 0. Doctrine note (supersedes part of the 2026-06-15 guardrail)
`COHERENT_ADD_PRIMITIVE_BUILD_SPEC_2026-06-15.md` ruled "geometry never leaves the
procedural renderer" after 3 failed detailizers. This spec revises that in two careful
steps, keeping the failure analysis in mind:
- **Phase R (retrieval-fit)** doesn't violate it at all: geometry comes from *real data*
  (BuildingNet component meshes), not from a synthesis net. Decisions stay learned
  (planner typing picks the class; retrieval ranks instances), geometry stays real.
- **Phase G (crop inpainting)** deliberately revisits geometry synthesis WITH the
  diagnosis of why it failed before: the detailizers and the Layer-A/AB context models
  were trained at 64³ *whole-building massing* resolution on LoD2-style data where an
  element is 2–4 voxels and rooftop structures don't exist at all
  (`outputs/layerA_eval/`, 2026-07-08: context conditioning ≈ no effect). Phase G trains
  at *element scale* (local crops, ~0.15–0.25 m voxels) on data that actually contains
  elements. If Phase G fails eval, Phase R remains the shipped behavior.

## 1. Data (all already on disk)
| asset | contents | role |
|---|---|---|
| `data/BuildingNet_dataset_v0_1/model_data/obj/{component_labels, faceindex_componentID}` + `OBJ_MODELS` | 1,849 buildings, per-COMPONENT label ids + face-range→component maps → **exact labeled sub-meshes** | element library (R) + inpainting crops (G) |
| `outputs/part_layouts_full/part_instances.npz` | per-instance (type, centroid, bbox) — planner training fuel | instance statistics / sanity cross-check |
| `data/lod3_tum` (+ `ingest_lod3_citygml.py`) | real LoD3 facades | Phase G crop augmentation (European realism) |
| `data/real_massing_v1/real.h5` | 35k LoD2 massings | context conditioning only — never the element source |
| gap: rooftop clutter (water tanks, HVAC, bulkheads) | thin in BuildingNet | v2: Objaverse-CC tagged models via HF download, or curated like `ingest_ornaments.py` |

## 2. Phase R — retrieval-fit (ship first)
**R1. Element library** (`scripts/foundations/build_element_library.py` →
`data/element_library_v1/`, gitignored, reproducible):
- For each labeled building: OBJ + face→component + component→label; group faces per
  component; merge same-label components whose bboxes touch (a tower = shaft+cap+finial
  components); drop degenerates (< 60 faces or < 1.5% of building height).
- Adopted types (same ids as the planner vocabulary): tower(7), dome(22), chimney(15),
  roof structures(4, only components ABOVE the main roofline), balcony(14/16), column(12),
  window(2), door(6), stairs(17).
- Per element store: normalized sub-mesh (unit-max-extent, y-up), a **48³ SDF crop**
  (surface voxelize + outside flood-fill sign; watertightness not assumed), and metadata:
  type, source building id/class/style-ish prefix, extents relative to source building
  (e/h), centroid height fraction, aspect ratios, face count.
- QA: montage sheet per type (`outputs/element_library_v1/montage_<type>.png`).
**R2. Retrieval** (`scripts/server/element_fit.py`): query = (type from the existing
planner typing, box extents, height-on-building, building class) → rank library by
log-extent-ratio distance + height-fraction distance + class affinity; sample from top-k
with a seed (re-roll = next seed), exactly the re-roll UX interpret already has.
**R3. Fit + compose**: new EditOp kind `element` `{lib_id, center, size, rot_y}` —
`scene/sdf_edit._primitive` gains a sampler that trilinearly samples the element's
precomputed SDF crop, scaled into the op's box (non-uniform scale allowed up to 1.6×
aspect distortion, else pad). Elements are thereby first-class ops: undo / re-roll /
group / town round-trip / bake all reuse existing machinery. Blend into massing with the
op's `smooth` (default 0.08) so the joint reads intentional.
**R4. Live preview**: the raymarch shader can't render arbitrary SDF crops. Reuse the
detail-preview pattern: when `edits` contain `element` ops, sculpt.html swaps the live
volume for a server-composed grid (`/detail_cube_volume`-style round trip) after each
gizmo release, and shows the op as its bounding box while dragging. (Same contract the
sketch-relief preview already uses — no new invariants.)
**R5. interpret_mass integration**: after typing, if the library has ≥ N (=8) instances
of that type, construct via retrieval-fit (element op + the same procedural window/door
regularization where the type calls for it); else fall back to the existing template.
Subtract mode: typed carves (window/door as today) + new `arcade/loggia` handling — a
subtract element op sampled from recessed-type library entries (v1: negated element
crops of balcony/door recesses; honest scope: subtractions stay simpler until Phase G).
**R6. Gates**: B12/F4 extended — a roof-placed box must resolve to an `element` op with
real geometry (mesh vertex count over the box region ↑ vs template), survives snap +
round-trip; determinism per seed.

## 3. Phase G — crop-based context inpainting (the "guess the voxels" model)
Self-supervised pairs from the SAME extraction: for every library element, cut the
building-local crop (element + 1.5× context shell) at 48³/~0.2 m; input = crop with the
element replaced by its crude bbox mass (add case) or smooth-filled (subtract case:
learn to carve the real recess back); target = real crop. Train a small 3D UNet diffusion
(latent-free, direct SDF, ~20–40 M params — element scale doesn't need the 947 M prior)
conditioned on the context shell + type embedding. Inference drops into the SAME element
op: instead of retrieving a library crop, the model *generates* one conditioned on the
actual local context sampled around the placed box. Eval vs Phase R: same-box A/B sheets,
user judgment. Kill criteria: if generated crops are blobbier than retrieval at equal
wall-clock after 2 training iterations, stop — Phase R remains shipped.

## 4. Order of work
1. R1 extractor + QA montages ← **started 2026-07-08**
2. R2 retrieval + R3 element op (server-side, gated by unit sanity script)
3. R4 preview + R5 interpret wiring + R6 gates → ship
4. G data pairs (reuses R1) → G train → A/B eval → ship-or-kill
