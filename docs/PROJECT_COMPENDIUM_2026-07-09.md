# GenerativeTowns: Project Compendium — 2026-07-09

## 1. Project Overview

**GenerativeTowns** generates editable 3D towns from OSM footprint maps, then sculpts, details, ages, and renders individual buildings — every step reversible, every building a compact symbolic recipe rather than a frozen mesh.

### Design Doctrine
**Every decision about what a building looks like is made by a learned model; every realization into geometry is deterministic procedure.** That split is what keeps everything editable: models choose, procedures build, and any choice can be re-rolled without destroying the rest of the building (README.md).

### The Two Web Pages (README.md)
1. **`http://localhost:8099/`** — **Town Page (index.html)**: drop an OSM map / footprint-mask image → get a town; select a building to restyle / re-height / re-roll, age with weathering slider, mount heritage ornament, texture-bake or photoreal-render the whole town, export glTF.
2. **`http://localhost:8099/sculpt.html`** — **SDF Sculptor**: raymarched live editing. Place primitives and let "Make it architecture" interpret them; carve; bake textures; or sketch a rough shape on a wall and sculpt it into bas-relief geometry with stacked reliefs and optional prompts like "a lion head."

Round trips between the two pages are lossless (building opened in Sculptor returns to town with its edits as first-class state) — fixed via commit `9853a61` (HANDOFF_2026-07-06.md).

---

## 2. Status Ledger — Features & Layers

| Feature/Layer | Status | Latest Commit(s) | Source Doc | Notes |
|---|---|---|---|---|
| **Recipe-parameter diffusion (Layer 0)** | SHIPPED | `aee1e97` (README rewrite) | OPTION_B_PLUS_REPORT_2026-06-03.md | footprint+class+style → massing proportions/roof/wings; B+.6 diffusion head, 98.2% retention |
| **Stage 3a SDF diffusion + VQVAE (Layer 1)** | SHIPPED | `b00c558` (upstream branch) | HANDOFF_2026-06-30.md | 947M params; trained on real NL/DE/JP buildings; localized snap prior (`/snap_sdf`) |
| **Cross-cultural data (Layer 1)** | SHIPPED | commit `b00c558` | HANDOFF_2026-06-30.md | real.h5 = 35k LoD2 masses (NL 11.7k + DE 12k + JP 12k); deployed as xcultural warmstart ft-final |
| **Make-it-architecture / Coherent-add (Layer 1.5)** | SHIPPED | `f7fc2d4`, wired `3c561cd`, `d2fc957` | HANDOFF_2026-07-03.md, COHERENT_ADD_PRIMITIVE_BUILD_SPEC_2026-06-15.md | X-Part locality; crude primitives → coherent architectural elements via learned typing + procedural detail |
| **Weathering (Layer 2.5a)** | SHIPPED | `de14e76` | HANDOFF_2026-07-06.md | procedural geometric aging; edge wear, fBm erosion, Worley cracks; wired into export/rebuild and `/neural_render_town` |
| **Ornament retrieval + learned placement (Layer 2.5b)** | SHIPPED | `df0fc64` (v0 retrieval), `b81df48` (learned placement) | HANDOFF_2026-07-06.md | real heritage scans (3 pieces, threedscans.com); placement via PartLayoutPlannerV2 deconflicted by CoherentPartRefiner |
| **Sketch relief v1** | SHIPPED → REMOVED | `20e75d2` (v1 baseline, committed as-is), removed `0c4112f` | (artifact only, not in handoff) | view-space SDXL inpaint; outputs didn't match sketch idea |
| **Sketch relief v2** | SHIPPED | `f8f312a` (wall-space rewrite), `d6fa606` (exact fusion) | HANDOFF_2026-07-07.md | rectify to wall plane, generate bas-relief art, exact closed-form fusion (128³ output); user confirmed working live |
| **Relief stacking** | SHIPPED | `cc52bdc` | DEMO_BUILD_PLAN_2026-07-07.md | /paint_relief prior_sdf_b64 chaining + client reliefChain + F19 gate |
| **Part-layout planner** | INTERNAL ONLY | `0c4112f` | README.md | its user-facing AI-details feature was removed 2026-07-08; the model now serves only Make-it-architecture typing + ornament slot proposal |
| **Part-composer (facade detail)** | SHIPPED | (core model pre-existed) | README.md | always-on statistical facade; class-appropriate (religious→dome, residential→hipped) |
| **Texture bake + photoreal render** | SHIPPED | (SDXL pipeline pre-existed) | README.md, DEPLOYMENT_PLAN.md | SDXL + depth/canny/scribble ControlNets + IP-Adapter; HF cache auto-downloads |
| **Element retrieval-fit (Phase R)** | IN PROGRESS | `aa376a4` (R1 extractor); R2/R3/R5 in working tree 2026-07-09 | GENERATIVE_MAKE_IT_ARCHITECTURE_BUILD_SPEC_2026-07-08.md | 3,204 real BuildingNet elements (48³ SDF crops, 8 types); `element` EditOp kind; interpret_mass fits real geometry for tower/dome/chimney/dormer with procedural fallback; gates running |
| **Element generation (Phase G)** | PLANNED | — | GENERATIVE_MAKE_IT_ARCHITECTURE_BUILD_SPEC_2026-07-08.md | crop inpainting diffusion model (conditional on context); only if Phase R eval shows a gap; else stays unbuilt |
| **Layer-A/AB context snap** | PARKED | `8cabedb` (eval ran; no win over prod) | HANDOFF_2026-07-06.md, DEMO_BUILD_PLAN_2026-07-07.md | context conditioning (known_body + edit_mask + primitive channels) showed no visual advantage vs production xcultural prior; evidence in outputs/layerA_eval/ |
| **AI-detailing feature panel** | REMOVED | `0c4112f` (2026-07-08) | GENERATIVE_MAKE_IT_ARCHITECTURE_BUILD_SPEC_2026-07-08.md | Removed from UI; planner model kept for Make-it-architecture |
| **Disk purge / legacy cleanup** | DEFERRED | — | DEMO_BUILD_PLAN_2026-07-07.md | Phase 4 deferred; document preserved; logs_building/ → keep serving ckpts only; legacy/ (~1.3T) not touched |

**Upstream training code**: moved to `upstream-training` branch at commit `3771d1f` (Phase 2, DEMO_BUILD_PLAN_2026-07-07.md). Main branch is demo/serving only.

---

## 3. Data Map

| Dataset | Location | Contents | Producer | Consumer | Size | Notes |
|---|---|---|---|---|---|---|
| **BuildingNet** | `data/BuildingNet_dataset_v0_1/` | 1,849 buildings; per-component labels; OBJ models + face→component maps | — (external source) | Element library R1 extractor; Phase R retrieval-fit | ~50 GB | Component meshes → normalized SDF crops for retrieval candidates |
| **BuildingNet params** | `outputs/part_layouts_full/part_instances.npz` | per-instance (type, centroid, bbox) × 28k instances | (extracted from BuildingNet labels) | PartLayoutPlannerV2 training; recoherence plan; edit-pair synthesis | ~100 MB | Part vocabulary + spatial statistics |
| **Recipe augmentation v1** | `data/recipe_augmentation_v1/{style}.h5` | 50k synthetic procedural samples (8 files × 6250 each) | procedural recipe generator | recipe_param_diffusion training (B+.6) | ~120 GB | Diverse synthetic massing paired with RNG seed for param extraction |
| **Real massing v1 (cross-cultural)** | `data/real_massing_v1/real.h5` | 35,776 real LoD2 masses (NL 11.7k + DE 12k + JP 12k) | ingest_citygml_lod2.py (CityGML→SDF); 3D BAG; PLATEAU | Stage3a warmstart finetune (Layer 1) | 32.8 GB | Breadth lever for the massing prior; deployed as xcultural-warmstart-ft-final |
| **LoD3 TUM** | `data/lod3_tum/` | real LoD3 facades | (external source) | Phase G crop augmentation | — | European realism for element inpainting training |
| **Ornaments v1** | `data/ornaments_v1/` | normalized real heritage-scan meshes + SDF crops (3 pieces seeded) | ingest_ornaments.py (threedscans.com open) | Ornament retrieval + fit (Layer 2.5b) | ~500 MB | Romanesque relief, Angkor Wat, Bayon casts; gitignored, reproducible |
| **Element library v1** | `data/element_library_v1/` (gitignored) | 3,204 elements: 48³ SDF crops + metadata (type, aspect, height-frac, class) | build_element_library.py (Phase R1) | Phase R2 retrieval (`element_fit.py`); Phase G data-pair synthesis | 709 MB | 8 types: tower 617, column 570, dome 475, balcony 422, roof_structure 358, chimney 313, stairs 251, balcony_upper 198 |

---

## 4. Checkpoint Map

| Checkpoint | Path | Size | Loaded By | Powers | Notes |
|---|---|---|---|---|---|
| **Recipe-parameter diffusion (b6)** | `outputs/recipe_param_diffusion_b6` | 80 MB | `scripts/server/recipe_inference.py` | recipe params sampling (Layer 0 default massing) | Deterministic head + diffusion with jitter for diverse samples |
| **Stage 3a snap prior (final)** | `logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth` | 11.5 GB | `scripts/server/refine.py::_load_sdedit()` | `/snap_sdf` localized inpainting; sculpt page snap button | main prior; fcultural finetune; deployed 2026-07-03 |
| **Stage 3a snap guide** | `logs_building/continue-stage3a-xcultural-warmstart-ft/ckpt/stage3a_steps-1000.pth` | 11.5 GB | `scripts/server/refine.py::_load_sdedit()` | guidance for snap (autoguidance) | weakly-trained guidance variant; same lineage as final |
| **VQVAE** | (embedded in stage3a ckpt, also `vqvae_ckpt` path) | 101 MB | `models/vqvae_model.py` | SDF ↔ latent encoding for stage3a; element library SDF crops | Pre-trained; unchanged |
| **PartLayoutPlannerV2** | `outputs/part_layout_planner_v2/planner.pth` | 18 MB | `scripts/server/layout_detail.py` | AI-details proposals; window/door/balcony/ornament slots; Make-it-architecture typing | Trained on BuildingNet part labels; vocabulary = 27 types |
| **CoherentPartRefiner (X-Part)** | `outputs/part_set_refiner/coherent_refiner.pth` | 54 MB | `scripts/server/layout_detail.py::integrate_new_part()` | coherent-add neighbor-locality (freeze distant parts); ornament deconfliction | Upgraded from SPLICE baseline; takes whole part set + added primitive marker |
| **PartComposer (facade detail)** | `outputs/part_composer/part_composer.pth` | 1.3 MB | `scripts/server/compose_detail()` (all flows) | always-on statistical facade (glazing, roof type, details per class) | Lightweight; both pages share identically |
| **SDXL + ControlNets + IP-Adapter** | `external/hf_cache/` (auto-downloads) | 46 GB | `scripts/appearance/texture_bake.py`, `/paint_relief`, `/neural_render_town` | texture bake, photoreal renders, sketch-relief art generation | Hugging Face cache; first call downloads; no local storage |
| **Depth Anything V2** | `external/hf_cache/` (auto-downloads) | (part of HF cache) | `scripts/appearance/paint_relief.py::height_from_art()` | relief height estimation from SDXL-generated art | Detrended plane-wise to handle scene ramp |
| **Logs_GT footprint embed** | `Logs_GT/retrieval_footprint_full/ckpt_best.pth` | 4.4 MB | `models/stage3a_model.py` (FootprintEmbedNet) | Stage 3a footprint conditioning — LOAD-BEARING for the snap prior; shipped in the demo bundle | easy to forget: lives outside logs_building/ |

---

## 5. Outputs Map — Notable Directories

| Directory | Contents | Produced By | Notes |
|---|---|---|---|
| `outputs/recipe_param_diffusion_b6*` | B+.6 diffusion checkpoint + training logs | `train_recipe_param_diffusion.py` | Layer 0 massing generation |
| `outputs/part_layout_planner_v2` | Planner checkpoint (18 MB) + training setup | (trained prior to project) | Layer 1.5 Make-it-architecture typing |
| `outputs/part_set_refiner` | CoherentPartRefiner checkpoint (54 MB) + legacy `recohere_ops` attempts | (trained prior, upgraded for coherent-add) | X-Part locality (neighbor-aware denoising) |
| `outputs/part_composer` | Facade composer checkpoint (1.3 MB) | (trained prior) | Always-on statistical detail (windows/roof) |
| `outputs/weathering/` | Procedural weathering probe sheets (`weather_probe.png`, `weather_building.png`, `ornament_learned_placement.png`, `ornament_learned_facing.png`) | Layer 2.5a/2.5b demonstration | Verify aging + placement aesthetics |
| `outputs/sketch_relief_verify/` | v1 baseline outputs + v2 wall-space outputs (sheets + live browser shots) | Sketch relief v1 & v2 verification | v1 failed (perspective inpaint re-drew facades); v2 precise with exact fusion |
| `outputs/layerA_eval/` | Context-snap eval before/after sheets (NL/DE/JP) | `sdedit_layerAB_eval.py` (Layer-A/AB context research) | **Context conditioning showed no win over production xcultural prior** → PARKED |
| `outputs/element_library_v1/` (gitignored) | Per-type SDF crop + montage sheets (`montage_tower.png`, etc.); per-element metadata | Phase R1 `build_element_library.py` | **Reproducible; rebuild via same script** |
| `outputs/demo_video/` | Walkthrough clips (VP8 webm, 1400×900) | Playwright headless browser recording | `01_town_from_image` (0:38), `02_edit_weather_ornament_render` (4:51), `03_sculptor_make_architecture` (1:05), `04_sketch_relief` (1:47) |
| `outputs/sdedit_xcultural/` | Localized snap before/after renders (NL/DE/JP buildings) | Layer 1 xcultural finetune demo | Verify snap quality on cross-cultural data |
| `outputs/server_8099.log` | Live server output | `uvicorn` | Captures HTTP requests and model inference timings |

Demo bundle: **`../demo_bundle/`** (8.24 GB dir) + **`../demo_bundle.tar`** (7.86 GB) — assembled code + stripped Stage3a ckpts + serving ckpts + ornaments + requirements + run_demo.sh; gates green (13/13 + 19/19) post-verification (DEMO_BUILD_PLAN_2026-07-07.md, Phase 3).

---

## 6. Test/Gate Reference

### Test Suites
- **`scripts/server/test_branches.py`** — 11 branch/API tests (B6/B13 removed with the AI-detailing feature, `0c4112f`); 11/11 green 2026-07-09 (report_20260709T012921Z)
  - Run: `python scripts/server/test_branches.py`
  - Coverage: massing generation, snap, refine, neural render, ornament placement, weathering, etc.
- **`scripts/server/test_sculpt_flows.py`** — 19 UI-flow tests (incl. F18 sketch-relief and F19 relief-stack gates); 19/19 green 2026-07-09 (report_20260709T014327Z)
  - Run: `python scripts/server/test_sculpt_flows.py`
  - Coverage: sculpt primitives, gizmo controls, Make-it-architecture, relief stacking, textured export, neural render
  - Known: F16 (textured export) flakes under GPU pressure in full-suite runs but reproduces clean in isolation (not a code bug; adopt rerun-once policy per `d530171`)

**Last green run:** 2026-07-09 (11/11 + 19/19, post AI-detailing removal). `/paint_relief` is gated by F18 (single relief) and F19 (relief stack) since `d530171`/`cc52bdc`.

---

## 7. Demo Bundle (Phase 3 DONE, HANDOFF_2026-07-07.md)

**What it is:** a self-contained, runnable archive including:
- Code: main branch checkout
- Checkpoints: recipe_param_diffusion_b6, planner, refiner, composer, VQVAE, Stage3a snap prior (stripped: opt/sched dropped, 11G→3.9G each, inference-identical)
- Data: ornaments_v1
- `requirements.txt` distilled from venv
- `run_demo.sh` (uvicorn one-liner)
- README quickstart (GPU required; first run downloads ~46G HF models)

**Deliverables:**
- `../demo_bundle/` (8.24 GB directory)
- `../demo_bundle.tar` (7.86 GB)

**Verification:**
- Server launched from assembled bundle passed both gates (13/13 + 19/19) with fresh import closure verified
- Video walkthrough recorded headless against bundle server (4 clips, ~8 min total; VP8 webm; requires `--no-sandbox --single-process --disable-gpu --enable-unsafe-swiftshader` on this cluster for chromium)

**Known fix:** global `*.png` gitignore had silently excluded town page sample images (404 on fresh clone) → force-tracked in `c3314cd`, bundle re-tarred.

---

## 8. Open Items & Deferred Work

### Carried over from prior handoffs (still open)
- ~~Non-convex window burial~~ — CLOSED 2026-07-08 (`178c447`): a frame double-transform in test_interior_pull's own measurement; ops were on-wall all along (B13 protected the property until the feature's removal)
- **Weathering live-preview gap (narrowed 2026-07-08, `cc52bdc`)** — now applies on export/rebuild, TEXTURED export, and `/neural_render_town`; still absent only from the sculpt page's live raymarch preview
- **Ornament library size** (3 pieces) — growing blocked by bot-walls on Scan the World / MyMiniFactory; Smithsonian API rate-limited; superseded by sketch-relief ornament generation (HANDOFF_2026-07-06.md)
- **F16 flake** (textured town export under GPU pressure) — adopts rerun-once policy per `d530171`; not a code bug, not investigated further
- ~~Town page planner access~~ — moot since 2026-07-08: the AI-details feature was removed everywhere (`0c4112f`); the planner is internal-only

### New open items (added 2026-07-07 .. 2026-07-09)
- **Demo bundle is stale**: `../demo_bundle{,.tar}` were built BEFORE the AI-detailing removal and Phase R — rebuild via `scripts/make_demo_bundle.py --tar` once Phase R lands (element library dir must be added to the bundle's include list)
- **Relief stacking preview-only contract** (same as texture bake) — second stroke or Bake re-derives from true massing, drops earlier reliefs; fine for demo; revisit if user wants persistent stack (would need reliefs as edits or composited into base_sdf)
- **Relief corner-straddling behavior** — cross-wall strokes clipped to dominant wall (graceful); roofline strokes clipped to face; stroke mostly above wall yields small/odd patch
- **Relief seed randomness** — UI uses random seed per click; art quality varies (no multi-seed picker UI built)
- **Relief lateral detail ceiling** — out_res=128 (~0.16 m voxels in cube frame); mesh GLB download inherits same bound

### Deferred post-demo (research, not blocking)
- **Cross-cultural corpus next steps** (2026-07-07 decision) — region-token retrain, variety eval, scaling NRW → deferred; Layer-1 research agenda; real.h5 (32.8G NL+DE+JP) already feeds deployed xcultural prior
- **Disk purge** (Phase 4, DEMO_BUILD_PLAN_2026-07-07.md) — legacy/ (~1.3T), logs_building/ pruned to serving keep-list, old external/ repos (DiffSplat, Hunyuan3D-2, gaussian-splatting) — documented but not executed (user decision to defer)
- **Phase G crop inpainting** (generative element details) — only if Phase R eval shows a gap vs retrieval; trained 20–40 M params diffusion at element scale; kill criteria after 2 iters if blobbier than retrieval

---

## Timeline & Commit Lineage (Last 40)

**Key milestones (commit messages):**
- `aa376a4` — Phase R1: BuildingNet element-library extractor (component meshes → SDF crops)
- `44ee606` — Spec: generative Make-it-architecture (Phase R retrieval-fit, Phase G inpainting)
- `0c4112f` — Remove AI-detailing feature panel (keep planner model)
- `b81df48` — Layer 2.5b: replace ornament placement heuristic with learned model placement
- `de14e76` — Layer 2.5a: procedural weathering
- `3c561cd` — Wire Make-it-architecture into town page (index.html)
- `d2fc957` — Demo hardening: deploy xcultural snap, fix sphere/cone typing, make sculpt procedural
- `9853a61` — Coherence audit: fix state-loss bugs (round-trip, roof double-apply, orphaned ops)
- `f7fc2d4` — Coherent-add pipeline wired (interpret_mass + CoherentPartRefiner integration)

**Branch structure:**
- `main` — demo/serving code, demo/serving checkpoints (upstream training removed to `upstream-training` at commit `3771d1f`)
- `upstream-training` — full SDFusion research surface (train.py, test.py, datasets/, preprocess/, models/sdfusion_*)

---

**Maintainer:** Danvi Simhadri (danvisai03@gmail.com)  
**Last updated:** 2026-07-09  
**Next session:** finish Phase R (gates + commit + bundle rebuild), then A/B the retrieval-fit look before deciding Phase G.
