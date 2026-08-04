# Build spec — "add a primitive → coherent architectural piece" (no-image, OmniPart-adapted)

**Date:** 2026-06-15 · **Status:** SPEC (plan).
**Goal (user):** in the live SDF sculptor, when the user adds a crude primitive ("a moldy piece"),
it should **adapt/turn into an architectural element that is coherent with the other pieces** on the
building (matched rhythm/scale/attachment, no float/dupe; neighbors re-harmonize). The crisp geometry
stays procedural; only the *decision* (what element, where, coherent how) is learned.
**Companion docs:** `RECOHERENCE_IMPROVEMENT_PLAN_2026-06-14.md` (the losses + X-Part locality + data
prep this reuses), `ARCHITECTURAL_DETAIL_RESEARCH_2026-06-10.md`, `PROGRESS_AND_RESEARCH_2026-06-14.md`.
**Companion memories:** `project_smart_add`, `project_generative_facade_recohere`,
`project_recohere_improvement_plan`, `project_architecture_generation_research`,
`project_part_vocabulary_full`, `project_localized_snap`.

---

## 0. Doctrine guardrail (read first)
Adopt the **structure + cohesion** half of the part-aware papers; **NOT** their geometry synthesis.
A net synthesizing the part *surface* failed 3× here (detailizer L1/GAN/layout — `project_detailizer_v1`).
So the model outputs `type + coherent box + validity`; the crisp geometry is instantiated by the
procedural path we already have (`interpret_mass` smart-add constructions / `DetailParams`). Geometry
never leaves the procedural renderer.

## 1. Best paper(s)
- **OmniPart** (SIGGRAPH Asia 2025, arXiv 2507.06165, **code+weights released**) — the structural
  blueprint: stage-1 autoregressive **part-bbox planner** + stage-2 **joint, spatially-consistent**
  generation. We take the **planner + joint-cohesion**; we **discard stage-2** (image→3D geometry flow).
- **X-Part** (2509.08643) — **edit-locality**: regenerate the edited/added part **+ its neighbors**,
  freeze the rest. This is the "coherent with the other pieces / untouched building stays put" half.
- **SPLICE** (2512.04514) — the joint-refine ancestor; **already built (reduced) = our `PartSetRefiner`**.
  This spec UPGRADES that model, it does not add a new one.

## 2. ⭐ The no-image conditioning swap (the load-bearing adaptation)
OmniPart is **image-conditioned in both stages** (stage-1 reads an image + 2D part masks; stage-2 is
an image-to-3D backbone). Our pipeline has **no visual reference by design** (the research novelty).
This is NOT a gap to patch — the image is a *proxy for 3D shape + part localization, both of which we
already have directly and more richly*. We replace OmniPart's image front-end with our existing
3D/symbolic encoders; the planner+cohesion architecture is **conditioning-agnostic** (only the
encoder changes).

| OmniPart image-derived input | What it stands in for | OUR replacement (already in hand) |
|---|---|---|
| single input image | holistic shape context | **massing SDF / occupancy** — full 3D, not a 2D projection (`MassingEncoderSpatial`, already cross-attends a 64³ SDF in `PartSetRefiner` + the layout planner) |
| 2D part masks | where parts go / decomposition | **the user's added primitive (3D box)** + the **existing part set** — the sculpt action IS the localization, in 3D |
| (implicit) object class/appearance | what kind of thing | **symbolic embedding**: class / style / height (small MLP) |
| stage-2 image→3D flow | part surface synthesis | **N/A — discarded**; procedural instantiation |

**Conditioning vector for the upgraded refiner** = `enc(massing SDF) ⊕ emb(class,style,height) ⊕
part-set slots ⊕ added-primitive marker`. Strictly more information than a single 2D image for this
task. **Data implication:** our training data is already imageless (`part_instances.npz`, recipe
layouts, LoD3 element sets) — the image-swap removes a dependency, adds none.

## 3. Best data
Real **instance-annotated** architectural elements (neither corpus we have supplies this: BuildingNet
labels are sparse/per-point, 3D BAG is facade-less LoD2). Ranked:
1. **City-Facade** (9 classes: windows, doors, **columns, moldings, vaults, arches**) — real element
   vocabulary + spacing/rhythm stats. https://www.sciencedirect.com/science/article/pii/S0924271626000031
2. **BuildingWorld-LoD3** (2511.06337) + **TUM2TWIN** (2505.07396) — real LoD3 buildings, semantic+
   instance facade elements → coherent real layouts.
3. **Our 28k BuildingNet part instances** (`outputs/part_layouts_full/part_instances.npz`) — the base.
4. **Our procedural recipes** — free, perfectly-labeled layouts for pretraining.

## 4. Changes to our DATA
Current `part_instances.npz` = `[building, type, axis-bbox]` — no relationships, no notion of an edit.
1. **Augment vocabulary/cleanliness:** ingest City-Facade / BuildingWorld-LoD3 instances into the same
   `[type, bbox]` frame → richer, cleaner element set than sparse BuildingNet.
2. **Derive relational structure** (`scripts/foundations/derive_part_structure.py` from the recohere
   plan): per building → height-bands, neighbor adjacency, alignment, per-class spacing. This is what
   "coherent with the other pieces" is measured against. (NB: derived + noisy — bands 78% co-planar,
   symmetry only ~10%; use as in the recohere plan.)
3. **⭐ Build EDIT-PAIRS — the key new ingredient.** From each clean layout synthesize
   `(clean) → (clean + a crude "moldy" added/moved/duplicated part)`; target = the clean layout.
   Add an explicit **added-primitive marker channel** (1 bit per slot) so the model knows WHICH piece
   is the user's new mass to integrate. This is what teaches "moldy piece → coherent element".
   Generator: `scripts/foundations/make_part_edit_pairs.py` (new).

## 5. Changes to our TRAINING
Upgrade the existing `PartSetRefiner` (don't replace it):
1. **No-image conditioning** (§2): extend `SetDenoiser` to ingest `enc(SDF) ⊕ emb(class,style,height)
   ⊕ added-primitive marker` alongside the slots. Back-compat default = marker off (gates pin old path).
2. **Condition on the added primitive specifically** — currently it denoises the whole set; the marker
   makes it INTEGRATE this piece, not just tidy.
3. **Edit-pair supervision** — train on §4.3 pairs (we already do a weak corruption version; make it
   core, with the marker + clean target).
4. **Data-grounded coherence losses** — band-rhythm + spacing + wall-attachment (the recohere plan's
   lever B; gate symmetry to the ~10% symmetric buildings).
5. **X-Part neighbor-locality** — per-slot noise mask: regenerate added part + K nearest, freeze the
   rest (mirrors `project_localized_snap`; the recohere plan's lever A).
6. **Procedural-pretrain → LoD3-finetune** — pretrain on free recipe layouts, finetune on real
   City-Facade/BuildingWorld elements.
7. **Keep procedural instantiation** — output `type+box+validity`; `interpret_mass`/`DetailParams`
   render crisp.

## 6. Reuse of existing infra
`models/networks/part_set_refiner.py` (the model to upgrade) · `scripts/server/layout_detail.py`
`interpret_mass`/smart-add (crisp instantiation) · `part_layout_planner` (OmniPart stage-1 analog) ·
`part_instances.npz` (data to augment) · `derive_part_structure` + coherence losses + X-Part locality
(specced in `RECOHERENCE_IMPROVEMENT_PLAN_2026-06-14.md`).

## 7. Sequenced build
1. ✅ **DONE (2026-06-15).** Ingested TUM Ingolstadt **LoD3 CityGML** (savenow/lod3-road-space-models,
   open license) → `scripts/foundations/ingest_lod3_citygml.py` → `data/lod3_tum/lod3_part_instances.npz`
   (**55 bldgs, 1075 windows + 61 doors + 422 roofs**, our `[type,bbox]` frame). **99% of windows row-
   aligned** (vs BuildingNet 78%) — clean coherent-layout fuel.
2. ✅ **DONE (2026-06-15).** `scripts/foundations/make_part_edit_pairs.py` — `(x_corrupt,marker)→x_clean`
   pairs (move/resize/dup/add) + added-primitive marker; LoD3 (318) AND BuildingNet (3528); montages in
   `outputs/part_edit_pairs/`. `scripts/foundations/derive_part_structure.py` ✅ →
   `data/part_structure/{lod3,buildingnet}_structure.npz` (bands ~3-4/bldg, side_id, per-bldg sym_score
   + cleanliness for gating the coherence losses).
3. ✅ **DONE — conditioning (2026-06-15).** `CoherentPartRefiner`/`CoherentSetDenoiser` appended to
   `models/networks/part_set_refiner.py` (existing `PartSetRefiner`+`refiner.pth`+`/recohere_details`
   UNTOUCHED — verified both checkpoints load, 24 routes unchanged). No-image cond (§2): massing-SDF
   (cross-attn) + class (FiLM) + **added-primitive marker** (extra input channel); `refine()` has
   **X-Part neighbour-locality** (freeze slots far from the marked piece). `train_coherent_refiner.py`
   trains on edit-pairs (on-the-fly, SDF+class aligned via build_dataset): val loss 4.06→0.074,
   **moldy added piece integrates** (montage `outputs/part_set_refiner/coherent_add_montage.png`,
   moved ~0.4-1.2 cube-units INTO the building, neighbours frozen). `coherent_refiner.pth` saved.
   ✅ **coherence losses + LoD3 finetune DONE (2026-06-15).** `CoherentPartRefiner.coherence()` =
   band-rhythm + size-uniformity + wall-attachment (`grid_sample` the SDF), self-supervised from the
   clean target in-batch (no cache align needed); GATED to low-noise timesteps + x0 clamp (high-t x0
   estimate explodes otherwise). Trainer: pretrain BuildingNet (4.58→0.079) → **finetune LoD3** (clean
   99%-row data, 0.111→0.061). ✅ **WIRED MESH SHEET** `coherent_add_wired_sheet.py` →
   `outputs/part_set_refiner/coherent_add_wired_sheet.png`: real building meshes ① grid → ② moldy blob
   protruding → ③ refiner integrates it as a flush window (snap-to-wall + carve = the wired
   instantiation). STILL TODO: wire into `/interpret_mass` / sculpt.html + full server gates.
4. Pretrain (recipes) → finetune (LoD3). Eval pre-`regularize_ops` (it masks the delta).
5. Wire into `/interpret_mass` / a new add-primitive flow in `sculpt.html`.
6. GATES (mandate): `test_branches.py` 12/12 + `test_sculpt_flows.py` 17/17 green before+after.

## 8. Acceptance
Add a crude primitive at varied locations → it becomes a typed element whose pose/scale/rhythm match
neighbors (band-σ, spacing-CV, attach metrics improve vs current), neighbors re-harmonize locally,
untouched building bit-stable, gates green. Montage: moldy primitive → coherent element across seeds.

## 9. Sources
OmniPart 2507.06165 (code: github.com/HKU-MMLab/OmniPart) · X-Part 2509.08643 · SPLICE 2512.04514 ·
City-Facade (ScienceDirect S0924271626000031) · BuildingWorld 2511.06337 · TUM2TWIN 2505.07396 ·
FacAID 2406.01829 · Pro-DG 2504.01571.
