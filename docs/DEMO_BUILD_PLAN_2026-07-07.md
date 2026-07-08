# Demo build & repo cleanup plan — decided 2026-07-07

## Progress (updated 2026-07-08)
- **Phase 0 DONE**: branches 12/12, sculpt flows 18/18 incl. new F18 sketch-relief gate;
  F16 rerun-once policy (`d530171`).
- **Phase 1 DONE**: dirty-state checkpoint (`78d3dae`); import closure tracked — 24
  missing modules + web vendor/samples + launch scripts; `.gitignore` `datasets/` rule
  fixed (`3771d1f`).
- **Phase 2 DONE**: `upstream-training` branch at `3771d1f`; main amputated 2,402
  upstream files → 117 tracked (`d7d0cbc`); PROOF: fresh clone + symlinked artifacts
  served both gate suites green (12/12 + 18/18, reports 20260708T163415Z/164620Z).
- **Phase 2.5 NEXT**: window bug → relief persistence → weathering wiring → Layer-A eval.

Decisions (user, 2026-07-07): upstream training code moves to a **branch** (not deleted
outright); demo = **runnable bundle + walkthrough video**; **Stage 3a included** in the
bundle; **no disk purge yet** (git-level cleanup only).

## The generative-vs-procedural map (for the demo narrative / README)

Doctrine: every DECISION about what a building looks like is made by a learned model;
every REALIZATION into geometry is deterministic procedure (that's what keeps everything
editable).

Generative (learned, sampled):
| component | checkpoint (size) | role |
|---|---|---|
| recipe-parameter diffusion | `outputs/recipe_param_diffusion_b6` (80M) | default massing: samples proportions/roof/wings from footprint+class+style |
| Stage 3a SDF diffusion + VQVAE | `logs_building/continue-stage3a-xcultural-warmstart-ft{-final,}/ckpt/*.pth` (11G) + vqvae (101M) | "AI massing (3D BAG)" source, SDEdit, /snap_sdf — trained on real NL/DE/JP buildings |
| PartLayoutPlannerV2 | `outputs/part_layout_planner_v2/planner.pth` (18M) | AI-details proposals, ornament slots, Make-it-architecture typing |
| CoherentPartRefiner | `outputs/part_set_refiner/coherent_refiner.pth` (54M) | coherent-add of sculpted primitives, ornament deconfliction |
| PartComposer | `outputs/part_composer/part_composer.pth` (1.3M) | always-on statistical facade detail |
| SDXL + depth/canny/scribble ControlNets + IP-Adapter | `external/hf_cache` (46G, auto-downloads) | texture bake, photoreal renders, sketch-relief art |
| Depth Anything V2 | hf_cache | relief height from art |

Procedural / classical (NOT learned): footprint extraction from image (Otsu+contours),
recipe→SDF realization, CSG sculpting, Make-it-architecture *construction* (typing is
learned; building is seeded RNG over per-class stats), weathering (fBm/Worley), ornament
retrieval+fit (scans are real data; placement is learned), marching cubes / UV / PBR-map
heuristics, sketch-relief rectification+fusion math.

## Repo findings driving the plan
- Git tracks 2,506 files; 2,362 are `preprocess/` vendored junk. Meanwhile ~half the
  load-bearing demo code is UNTRACKED (recipe_inference.py, neural_appearance.py,
  footprint_image.py, scene/sdf_{edit,primitives,recipes}.py, models/networks/
  {diff_recipe,recipe_param_*,part_layout_planner,part_set_refiner,displacement_field}.py,
  both gate suites, …). A fresh clone cannot run the demo. Fix this FIRST — before any
  restructuring, and never run `git clean` until it's fixed.
- Disk: ~2.3T total; demo needs ~60G (11.5G checkpoints + 46G hf_cache + code). legacy/
  1.3T, logs_building/ 488G, data/ 473G, external/ 56G. Untouched this round by decision.

## Pending-workflows disposition (decided 2026-07-07, same session)
SOLVE before the bundle (user-confirmed):
- **Non-convex footprint window bug** (`propose_detail_ops` buries windows on L/U shapes,
  open since 07-03) → FIX (Phase 2.5).
- **Sketch-relief one-shot behavior** (second stroke/Bake drops earlier reliefs) →
  PERSIST reliefs across strokes within a sculpt session (Phase 2.5).
- **Weathering coverage gap** (only geometry export/rebuild path) → WIRE IN to textured
  export (`export_town_textured`) and `/neural_render_town` (Phase 2.5). User chose this
  over scripting around it.
- **Layer-A/AB context-snap eval** (trained ckpts `continue-stage3a-layerA-context` and
  `continue-stage3a-layerAB-context-elemtype`, never production-faithfully evaluated) →
  RUN the localized before/after eval with real context channels for the report record,
  THEN park. Independent GPU job; can run parallel to any phase (Phase 2.5-parallel).

DROP / formally closed:
- Ornament library growth (API-key/bot-wall blocked) — superseded for demo purposes by
  the sketch reliefer generating wall ornament on demand; carve-library and
  train-a-synthesizer alternatives were already declined 2026-07-06.
- Multi-seed pick-one UI for relief art — "generate again" is the recourse.
- Town page invoking PartLayoutPlannerV2 directly — accepted scope gap (decided 07-06).
- F16 flake — adopt rerun-once policy in the gate script; stop tracking as a bug.

DEFER post-demo (research, not demo-blocking):
- Cross-cultural corpus next steps (region-token retrain, variety eval, scaling NRW) —
  `data/real_massing_v1/real.h5` (32.8G, NL+DE+JP) exists and already fed the deployed
  xcultural prior; the rest is Layer-1 research.

## Phase 0 — safety net (do before anything moves)
1. Run both gates on the current build (`test_branches.py` 12, `test_sculpt_flows.py` 17)
   — they have NOT been re-run since the sketch-relief work; must be green pre-restructure.
2. Add a minimal `/paint_relief` gate test (template: paint_relief_verify.py flow) so the
   new feature is protected through the restructure.

## Phase 1 — make main self-contained (track the missing code)
1. Commit the long-standing working-tree state as a checkpoint: pending deletions
   (preprocess tbb build artifacts, single_mask_outputs) AND the May–June modified
   tracked files (README, models/, datasets/, train.py) — warts and all, so the branch
   point preserves everything.
2. Compute the import closure of `scripts/server/inference_service.py` (+ web/, gates,
   run_*.sh, configs referenced by refine.py/stage3a) and `git add` all of it.
3. Commit: main now runs from a fresh clone (checkpoints/data still external by design).

## Phase 2 — move upstream SDFusion to a branch
1. `git branch upstream-training` at the Phase-1 tip (preserves everything).
2. On main, `git rm`: train.py, test.py, datasets/, preprocess/, demo_*.ipynb,
   models/sdfusion_{txt2shape,mm,*}_model.py and anything else NOT in the demo import
   closure (utils/, options/, launchers/, dataset_info_files/ — verify each against the
   closure before removing; stage3a still needs diffusion_networks, vqvae_networks,
   base_model, model_utils and possibly losses/options pieces).
3. Prove it: clone main into a scratch dir, symlink checkpoints/data/external, launch the
   server, run both gates green.

## Phase 2.5 — product fixes on the cleaned main (each gated before the next)
1. Fix `propose_detail_ops` window burial on non-convex footprints (likely: test window
   candidate positions against the actual footprint polygon / local SDF, not the bbox).
2. Persist sketch reliefs across strokes: keep the sculpt session's relief stack (either
   composite each accepted relief into a session-side working grid the next stroke
   derives from, or record reliefs as first-class edits) — must NOT violate the
   raw-massing invariant for Bake (see sculpt.html's one-shot contract comment).
3. Weathering → textured export + neural render: both flows re-derive detail grids
   server-side; feed them the weathered grid (`b["weather"]`/`b["weather_seed"]` already
   travel with the building). Watch bake seams: weathering displaces the surface the UV
   atlas was unwrapped on.
4. (parallel, independent GPU job) Layer-A/AB context-snap localized eval → report sheet
   in `outputs/`, then park the line.

## Phase 3 — demo bundle + video
1. Bundle = include-list tarball:
   - code: main checkout;
   - checkpoints (~11.5G): recipe_param_diffusion_b6, planner, refiner, composer, vqvae
     ckpt, stage3a snap ckpts (measure whether the demo UI exercises both `-final/latest`
     and `-1000`; ship only what's loaded), data/ornaments_v1;
   - `requirements.txt` distilled from the venv, `run_demo.sh` (uvicorn one-liner),
     README-quickstart rewrite (GPU required; first run downloads ~46G HF models; ports;
     the two pages and what each button demonstrates, using the generative-vs-procedural
     map above as the narrative).
2. Verify the bundle from a clean directory + fresh venv on this cluster.
3. Video: Playwright-recorded walkthrough (chromium needs `--no-sandbox --single-process
   --disable-gpu --enable-unsafe-swiftshader` on this cluster; playwright's video
   recording works headless): image→town, restyle/height/variation, open-in-Sculptor
   round trip, Make-it-architecture + re-roll, AI details, weathering slider, ornament,
   texture bake, neural render, sketch relief (precise + creative lion). Deliver .mp4 +
   key stills.

## Phase 4 — deferred by decision (documented, not executed)
- No disk purge. When revisited: prune logs_building/ to the serving keep-list, then
  legacy/ (1.3T), dead external/ repos (DiffSplat, Hunyuan3D-2, gaussian-splatting),
  training datasets in data/. Keep-lists are in the table above.
