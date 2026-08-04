# Architectural Detail + Coherent Mesh Adaptation — research, data, plan

**Date:** 2026-06-10 · **Ask (user):** "as many details as possible to make it look like architecture —
pull research and data required for training, and ways to make a mesh adjust coherently."
**Companion docs:** `TRACK2_part_mixing_design.md` (the learned part-mixing spec),
`TRAINING_GAPS_RESEARCH_PLAN_2026-06-09.md` (themes D/E), `memory/project_part_coherence_research.md`.

**Framing.** Two distinct problems, two distinct fixes:
1. **Detail richness** ("looks like architecture") — windows, doors, balconies, cornices, dormers,
   chimneys, columns, railings… today we hand-code 6-ish elements in `scene/sdf_detail.py`; that
   can't scale. Needs **labeled element data** + a generative/parametric element layer.
2. **Coherent adjustment** ("mesh adjusts coherently when edited") — add/move/remove a part and the
   building re-harmonizes (attach, dedupe, align, replace-not-accumulate). Needs a **part-aware
   representation + a global mixing/cohesion model** — projection priors (our SDEdit) can't do this.

---

## 1. Methods — what to build on

### Coherent part-aware generation/editing (problem 2)
| Paper | What it gives us | Fit |
|---|---|---|
| **OmniPart** (2507.06165) | Two stages: autoregressive **part-bbox layout planner** → spatially-conditioned rectified-flow generating ALL parts *simultaneously & consistently* in the layout. SOTA part-aware gen + compositional editing. | The strongest current blueprint for Track 2: our composer already predicts layouts; replace the heuristic instantiation with layout-conditioned simultaneous part synthesis. |
| **SPLICE** (2512.04514) | Parts = validity + pose embedding + shape code, **denoised jointly toward a coherent configuration**; attention-guided decoding prevents float/leak. Editing-native. | Closest to the sculpt loop: user perturbs the part set → joint refinement re-coheres. The TRACK2 doc is already modeled on this. |
| **X-Part** (2509.08643) | Controllable decomposition of a holistic shape into structure-coherent parts with high fidelity. | The *inverse* direction: turn our massing (or a BuildingNet/BAG mesh) into clean parts → training pairs for free. |
| **PartGen** (2412.18608) | Multi-view diffusion part segmentation + per-part reconstruction; parts reassemble at whole-object quality. | Alternative data engine: part-decompose unlabeled corpora (3D BAG!) via rendering. |
| **DiffFacto** (2305.01921) | Cross-diffusion over part latents w/ controllable per-part edits (point clouds). | Reference for the part-latent cross-attention design. |
| **PartDiffuser** (2511.18801) | Part-wise discrete diffusion for meshes. | Mesh-native option if we move past SDF volumes. |

**Synthesis:** the coherent-adjust recipe the field converged on = **(a) explicit part layout (bboxes/poses)
planned first, (b) all parts generated/refined jointly conditioned on the layout, (c) edits = perturb layout
or one part, then re-run the joint refinement** — never independent unions (exactly the floating-cones bug
we had). This is Track 2; OmniPart/SPLICE make it concrete.

### Detail-richness methods (problem 1)
- **Layout-as-tokens:** OmniPart's variable-length bbox sequence naturally covers MANY element types
  (window grids, dormers, balconies) without per-type hand code.
- **CM2LoD3** (2508.15672) + Grounding-DINO-based facade detection (ISPRS 2024): automatically upgrade
  LoD2 → LoD3 by detecting windows/doors on imagery and carving them into the model — i.e. **a data
  factory for element-annotated buildings from what we already have (3D BAG is LoD2.2)**.
- Our `scene/sdf_detail.py` stays as the *renderer* of elements (crisp CSG instantiation — now also the
  sculptor's detail layer); what must become learned is **which elements, where, and how they relate**.

---

## 2. Data — what to train on (ranked by leverage/cost)

| Dataset | Contents | Use |
|---|---|---|
| **BuildingNet part labels** (LOCAL, free) | 1,838 buildings × per-point labels, **32 classes** — we use only ~6; ids 6/9/11/12/14 appear in 27–47% of buildings (likely doors/columns/balconies/railings/chimneys — must be identified) | The core Track-2 training set: extract part *instances* → (type, pose/bbox, shape code) tuples → layout planner + part generator. First step: identify + adopt the FULL taxonomy (`scripts/identify_buildingnet_labels.py` pattern). |
| **ZAHA** (2025) | 66 facades, **601M annotated points, 15 facade classes** (windows, doors, balconies, moldings…) | Facade-element statistics + train a facade-element detector/segmenter; grounds element placement priors (spacing, alignment, floor rhythm). |
| **City-Facade** (2026) | City-scale facade point clouds, 9 classes incl. columns/moldings/vaults/arches | Same role, city-scale diversity. |
| **TUM-FACADE** | 17 facade classes, TUM campus | Benchmark + extra labels. |
| **3D BAG → LoD3 upgrade** (CM2LoD3 recipe) | Our existing 11,776 (or BuildingWorld's ~5M LoD2) + street imagery → detected windows/doors carved in | The scalable data factory: element-annotated REAL buildings in our exact training frame. Medium effort, big payoff. |
| **Infinigen Indoors** (BSD, procedural) | 100% procedural architecture (walls/windows/doors/stairs) w/ full annotations, constraint-based arrangement DSL | Two uses: (a) generate unlimited *labeled* exteriors-ish elements w/ perfect ground truth; (b) its **constraint-solver design** is the template for our element-arrangement rules (alignment, floor rhythm, symmetry). |
| **BuildingWorld** (2511.06337) | ~5M LoD2 buildings, 44 cities (+ Cyber City unlimited synthetic) | Massing diversity (gap #2/#7) + the substrate for the LoD3-upgrade factory. |
| (Have) recipe corpus + `sdf_detail` | 8 styles + parametric elements | Cheap **synthetic part-layout pairs**: every procedurally-detailed building comes with its exact element layout for free — pretraining data for the layout planner. |

---

## 3. Plan — sequenced (extends TRACK2; Track numbering continued)

1. **Full part-vocabulary adoption (CPU, ~1–2 days).** Identify BuildingNet label ids 6/9/11/12/14 +
   the rest of the 32 (render point clusters per id); extend `extract_buildingnet_part_layouts.py` to the
   full taxonomy; per-class element-instance dataset: (type, bbox/pose, count, adjacency, symmetry).
   → richer `composer` immediately (more element types in ② today, even before any new model).
2. **Element-instance dataset v1 (the Track-2 fuel).** Cluster per-label points into instances → fit
   bbox + pose + coarse shape code per instance; + synthetic layouts from our own detailed recipes
   (free labels). Format mirrors OmniPart: variable-length part-bbox sequence per building.
3. **Layout planner (the OmniPart stage-1 analog, ~days).** Upgrade `PartComposer` from count-prediction
   to **bbox-sequence prediction** conditioned on massing (+class/style). Autoregressive or set
   transformer; trained on (2). This is the "which elements, where" brain.
4. **Joint part refinement = the coherence engine (SPLICE-style, ~1–2 weeks).** Diffusion over the part
   set (validity+pose+shape code) conditioned on massing; **edits = perturb the set → partial-noise
   re-denoise → coherent layout** (attach/dedupe/replace). Decoder: start with `sdf_detail` parametric
   instantiation per type (crisp, cheap); learned per-part decoder later.
5. **Wire into the sculptor.** The detail layer's semantic ops (door/window/balcony/chimney/roof — live
   since 2026-06-10) become *part-set edits*: user drops a part → (4) re-coheres the whole layout → ops
   re-instantiated crisp. "Mesh adjusts coherently" = exactly this loop.
6. **Facade-prior enrichment (parallel, optional).** Train an element-statistics prior on ZAHA/City-Facade
   (window spacing, floor rhythm, balcony placement) as a loss/constraint for (3)–(4); Infinigen-style
   constraint DSL for hard rules (alignment, ground-floor doors).
7. **LoD3 data factory (bigger, later).** CM2LoD3-style imagery-based window/door detection over 3D BAG /
   BuildingWorld → real element-annotated buildings at scale; retrain (3)–(4) on real layouts.

**Relation to current threads:** REPA/conditioning (running) fixes the *massing* prior; this plan fixes
*architecture-ness* — they compose: ① massing (REPA'd prior) → ② parts (this plan) → crisp detail render
(`sdf_detail`). The sculptor's detail layer is already the UI for it.

## 4. Sources
- OmniPart — https://arxiv.org/abs/2507.06165 · SPLICE — https://arxiv.org/abs/2512.04514 · X-Part — https://arxiv.org/abs/2509.08643
- PartGen — https://arxiv.org/abs/2412.18608 · DiffFacto — https://arxiv.org/abs/2305.01921 · PartDiffuser — https://arxiv.org/abs/2511.18801
- CM2LoD3 — https://arxiv.org/abs/2508.15672 · LoD3 upgrade w/ Grounding DINO — https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/471/2024/
- ZAHA / City-Facade — https://www.sciencedirect.com/science/article/pii/S0924271626000031 · TUM-FACADE
- Infinigen / Infinigen Indoors — https://arxiv.org/abs/2406.11824 · BuildingWorld — https://arxiv.org/abs/2511.06337
