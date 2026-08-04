# Track 2 design — learned part-proxy + global-mixing (coherent sculpting)

**Goal:** make "add a part → it integrates *logically* and the result stays a coherent
building" a **learned** behavior — not the deterministic rules of Track 1. Solves the
problems in `memory/project_part_coherence_research.md`: floating/duplicated parts, old
prims not removed, no part logic. Grounded in **SPLICE** (arXiv 2512.04514, the closest),
**SPAGHETTI** (2201.13168), **SALAD** (2303.12236), **StructureNet** (1908.00575).

## Requirement: FULL, EXTENSIBLE element vocabulary (not just towers/dome/roof)
User flag (2026-06-07): we've been over-featuring towers and only use **6 of BuildingNet's 32
part labels** (wall, window, roof, tower, stairs, dome). We ignore common ones — id **6 (in 47%
of buildings)**, 9 (30%), 11 (28%), 12 (27%), 14 (27%) — almost certainly doors, columns,
balconies, railings, beams, chimneys, dormers. The procedural `sdf_detail` route can't scale
(every element needs a hand-coded primitive). **Track 2 must support the WHOLE label set**: the
per-part *learned shape code* makes geometry data-driven, so any labeled part type is covered
without hand-authoring. Near-term prep: identify ids 6/9/11/12/14 (render their point clusters)
and extend `extract_buildingnet_part_layouts.py` + the composer to the full taxonomy. See
`memory/project_training_audit` & `project_part_coherence_research`.

## Why Track 1 isn't enough
Track 1 (flush placement + smooth-union + caps in `sdf_detail.add_landmarks`) stops the worst
floating/duplication, but it's hand-rules: it doesn't *reason* about arrangement, can't
*remove/replace* on edit, and won't generalize to new part types. The principled fix is a
model that represents a building as **parts with relationships** and **re-harmonizes** them.

## Representation (per SPLICE/SPAGHETTI)
A building = an unordered **set of part proxies**, each:
- `type` — one-hot over our label set (wall, window, roof, dome, tower, stairs, …; ids in
  `memory/project_buildingnet_labels_local.md`),
- `pose` — Gaussian ellipsoid (center, scale, orientation) = where/how big the part is,
- `shape_code` — a latent for the part's intrinsic geometry,
- `validity` — present/absent flag (lets the model add/remove parts).

## Two networks
1. **Decoder (set → surface):** a transformer that, for any query point, attends over the
   part-proxy set and predicts occupancy/SDF → one globally-coherent surface (not a union of
   stuck-on prims). Add SPLICE's **attention-guiding loss** (each interior point attends
   mainly to its own part) → prevents parts floating/merging/leaking.
2. **Global mixing diffusion (the coherence engine):** a diffusion over the part-proxy set
   (poses + shape codes + validity). An edit perturbs the set (user adds a crude proxy);
   running the mixing diffusion **denoises the whole set toward a coherent configuration** —
   it attaches the new part, fixes poses, drops/merges redundant proxies (→ *replace, not
   accumulate*), and respects symmetry/adjacency learned from data.

## Sculpt/edit flow (the payoff)
```
user adds a crude primitive  ->  map to a part proxy (type + pose)
   -> insert into the building's proxy set
   -> GLOBAL MIXING diffusion (partial-noise, SDEdit-style on the SET)  -> coherent set
   -> decoder -> occupancy -> marching cubes  -> a building where the part is integrated,
      redundant geometry removed, arrangement sensible.
```
This unifies with our ① SDEdit massing: massing prior gives the coarse solid; the part-mixing
gives coherent *elements*. ① = "looks like a building", Track 2 = "made of parts that make sense".

## Training data (we already have most of it)
- **BuildingNet part labels** (1838 buildings, per-point ids) — `model_data/.../point_labels/`.
  Pipeline: per building, segment points by label → cluster each label into part **instances**
  → fit a Gaussian proxy per instance + a per-part occupancy/shape code. (Extends
  `scripts/extract_buildingnet_part_layouts.py`, which already computes per-class part stats.)
- The **part-composer** (`outputs/part_composer/`) is a starting point for the proxy
  *predictor* (massing → which parts), now upgraded to predict full proxies (pose+code), not
  just counts.
- Reuse the VQVAE/SDF + diffusion infra (`models/networks/recipe_param_diffusion.py`,
  `models/stage3a_model.py`).

## Losses
occupancy BCE/SDF-L1 · Gaussian NLL (proxy fit) · **attention-guiding** (anti-float) ·
diffusion ε-loss (mixing) · KL (proxy latent). Optional StructureNet-style symmetry/adjacency
terms for explicit relationship logic.

## Scope / compute
SPLICE: ~8×3090 ~1 day on PartNet. Scale down: our ~1.8k labeled buildings, ≤8 part types,
64–128³ → feasible on **1 GPU over a few days**. Suggested milestones:
1. Per-part instance extraction → proxy dataset (CPU, ~1 day).
2. Decoder (set→occupancy) + attention-guiding loss; verify reconstruction.
3. Mixing diffusion over proxies; verify add/remove-part edits re-cohere.
4. Wire into the sculpt flow (replaces Track-1 `add_landmarks`) + `/refine_with_edit` (#26).

## Relationship to current code
- Track 1 lives in `scene/sdf_detail.py:add_landmarks` (coherence rules) — keep as fallback.
- Track 2 would be a new `models/networks/part_mixing.py` + `scripts/extract_part_proxies.py`
  + `scripts/train_part_mixing.py`, consumed by `scene/composer_detail.py` (swap the
  heuristic instantiation for the learned decoder+mixing).
- Memories: `project_part_coherence_research`, `project_part_composer`,
  `project_buildingnet_labels_local`, `project_composer_detail_glue`, `project_ai_sculpting_research`.
