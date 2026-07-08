# GenerativeTowns / SDFusion — Project README

**Last updated:** 2026-06-08
**Audience:** future agents (Claude / Codex / human) picking up this project.
**👉 Start here:** `memory/MEMORY.md` — the live institutional-knowledge index. The current frontier
(foundations: clean VQVAE + footprint+height massing prior) lives in the recent `project_*` memories.
Last narrative handoff: `docs/HANDOFF_2026-06-06.md` (AI-sculpting).
**Maintainer:** Danvi Simhadri (danvisai03@gmail.com).
**Repo root:** `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/`
**Original SDFusion README:** preserved at `README_UPSTREAM.md` (install steps + paper bib).

> ⚠️ The dirty git worktree is **intentional**. Many modified/deleted/untracked files
> are part of ongoing work documented below. Do not run `git restore` blindly —
> ask before touching unstaged changes.

---

> **Latest (2026-06-14):** the AI-sculpting frontier now spans the full three-layer
> hybrid — symbolic state → crisp procedural geometry → **neural appearance** → **Unreal**.
> Shipped: localized generative snap, smart-add (placed mass → typed architecture), live
> detail preview, neural photoreal render, **UV texture bake + PBR + iterative TEXTure +
> town-wide textured glb export for Unreal**, and a geometry **cleanup pass** (floater/sliver
> removal). 17/17 flow + 12/12 branch gates green. Full writeup + research synthesis:
> **`docs/PROGRESS_AND_RESEARCH_2026-06-14.md`**. Open model thread: rhythm-aware re-cohere /
> neuro-symbolic facade-program (`memory/project_architecture_generation_research`).

## 1. Project goal in one paragraph

End deliverable: take an **OSM tile of building footprints** and produce a navigable **3D town**. Each building is generated from `(footprint polygon, class_id, height, style)` — symbolic, categorical inputs (no reference image). The research novelty axis is "truly generative from symbolic input, exploiting category-level structural priors, without a visual reference" — different from every published image-to-3D paper. Output is a composed scene (mesh OR 3DGS) renderable as a 3D world.

---

## 2. Status at a glance — what works, what's running, what's blocked

| Layer | Status | Notes |
|---|---|---|
| OSM → footprint extraction | ✅ shipping | `scene/osm_*.py` |
| Path A — retrieval + Hunyuan refine | ✅ shipping | end-to-end on east4 |
| Path B — baked 3DGS corpus (1849 PLYs, ~317 GB) | ✅ shipping | `data/.../gaussian_splats_v2/` |
| Path C — procedural SDF recipes | ✅ shipping | `scene/sdf_recipes.py`, `demo_sdf_sculpting.ipynb` |
| **Path D — Stage 3a SDF latent diffusion** | ♻️ revived as the SDEdit prior | Was paused for a loss-vs-inference gap (DDIM from pure noise = speckle). **Now repurposed**: it doesn't need to generate from pure noise — SDEdit starts from the user's edit at *partial* noise, an easier task. See AI-Sculpting row + §4. |
| **AI Sculpting ① — SDEdit massing prior** (the frontier) | 🟢 trained + wired | `Stage3aModel.sdedit()` snaps a crude SDF edit onto the building manifold (clean — not the from-noise speckle). Trained on real watertight **3D BAG** massing; **snap-to-plausible wired into the web demo** (`/refine_with_edit` mode=`sdedit` + faithfulness slider). `memory/project_snap_to_plausible_wired.md`, `project_realtime_generative_framing.md`. |
| **AI Sculpting ② — element composition** | 🟢 built | part-composer (`scene/composer_detail.py`) + `sdf_detail` place class-appropriate roof/windows/door/dome/towers. **Style + ornament live HERE, not in the prior** (confirmed empirically). |
| **Foundations** (2026-06-08) | ✅ done | clean VQVAE finetune **fixes the gap#6 surface artifact** (box recon IoU 0.47→1.00); **hybrid prior retrained to 20k** → footprint+height massing prior with CFG dropout + EMA. Key finding: named style can't condition the 64³ footprint-dominated prior (collapses) → belongs in ②. `memory/project_vqvae_clean_finetune.md`, `project_hybrid_prior_retrain.md`. |
| Path E — Stage 3b SDF→Gaussians lifter | ⏸ blocked on Stage 3a | code written, not trained |
| **Option B+ — recipe-param diffusion** | 🟢 working | B+.1-5 + B+.7 done; **B+.6 generative diffusion trained** (8k ep, 98.2% footprint-IoU retention; jitter rescues zero-variance styles). **Stage A inference service + full REAL OSM→generated-town pipeline VALIDATED end-to-end** (62-building Lafayette tile). Hit a **diversity ceiling** (`memory/project_b6_diversity_ceiling.md`) — recipe parameterization bottlenecks quality AND diversity; this is *why* AI-Sculpting/SDEdit (a true shape-latent prior) is the new direction. |
| **Web demo** | 🟢 working (stopped) | `scripts/server/inference_service.py` + `web/index.html` (three.js). Upload a footprint image / OSM-map screenshot → 3D town; per-building **click-to-select, restyle, and ✨ snap-to-plausible (AI sculpt)**. ⚠ launch with the Bash **sandbox disabled** (it kills network-binding servers). `memory/project_web_demo.md`, `project_snap_to_plausible_wired.md`. |
| **Stage 4 — interactive sculpt-and-refine UX** | 🟢 working (headless) | `scene/sdf_edit.py` `EditableBuilding` (~5 ms re-mesh) + `tools/blender_addon/` + `/refine_with_edit` (snap **and** detail-preserving displacement refine). The procedural refine; **SDEdit (AI Sculpting row) is the learned upgrade** that makes the edit *adapt* into a coherent building. |
| Phase 4 — A/B sheets + eval | ⏸ blocked on convergence | depends on Path D OR Option B+ delivering |

**Where we are (2026-06-08).** The architecture is settled as **two layers**: ① a SDEdit **massing
prior** (footprint+height → coherent solid) and ② **element/detail composition** (roof / windows /
door / dome / towers + style / ornament). The OSM→town→click-to-restyle web demo ships and
**snap-to-plausible sculpting is wired into it**. This session's foundations work fixed the VQVAE
surface artifact (gap#6) and retrained the massing prior on the clean VQVAE with CFG dropout + EMA.

**Settled findings** (don't re-derive — see the memories): generate *by editing* (SDEdit at partial
noise), **never from pure noise** (that path is degenerate speckle); the prior is **footprint+height
massing only** — class/style/era collapse inside it because the spatial footprint dominates, so
style/ornament belong in **②**; real-time sculpting is feasible (few-step SDEdit ~100 ms; VQVAE latent
interpolates on-manifold — `project_realtime_probes`); the recipe-param diffusion line (Option B+) hit
a **diversity ceiling** (`project_b6_diversity_ceiling`), which is *why* the shape-latent prior is the
direction. GPU is free; the web demo server is currently stopped.

---

## 3. Data inventory

| Dataset / cache | Location | Size | Contents | Used by |
|---|---|---|---|---|
| **BuildingNet 64³ SDFs** | `data/BuildingNet_dataset_v0_1/resolution_64/<id>/ori_sample_grid.h5` | ~14 GB | 1849 assets. Each h5: `pc_sdf_sample` (262144,1) → reshape (64,64,64) in (D=z, H=y, W=x); `footprint` (1,64,64) top-down silhouette; `sdf_params` (6,) bbox; `norm_params` (4,) | VQVAE training, Stage 3a, B+.7 |
| **BuildingNet OBJ meshes** | `data/BuildingNet_dataset_v0_1/OBJ_MODELS/<id>.obj` | varies | Raw watertight-repaired meshes (non-manifold; many have holes) | retrieval, Hunyuan refine |
| **BuildingNet renders** | `data/BuildingNet_dataset_v0_1/buildingnet_renders/<phase>/<id>.png` | ~3 GB | 512² RGB renders, 3/4-view | CLIP labeling (Phase 1b.2, pending) |
| **Asset dimensions CSV** | `outputs/stage3_metadata/asset_dimensions.csv` | 1850 rows | `id, split, n_occupied, x/y/z extents in normalized frame` | Stage 3a, B+.7 |
| **Baked v2 3DGS corpus** | `data/BuildingNet_dataset_v0_1/gaussian_splats_v2/<id>.ply` + `_preview.png` | ~317 GB | 1849 Inria-format Gaussian Splats baked from BuildingNet meshes, May 2026 | Path B, Stage 3b training data |
| **v1 voxelized GS** | `data/BuildingNet_dataset_v0_1/gsplat_voxelized_32k8/<id>.npz` | ~30 GB | 32³ × 8 slots × 14 attrs per asset | Stage 3b training |
| **Recipe augmentation v1** | `data/recipe_augmentation_v1/<style>_*.npz` | ~50 GB | 50k synthetic procedural buildings via `scene/sdf_recipes.py` (8 styles) | Stage 3a, future B+.4 |
| **3D BAG real corpus** (NEW) | `data/bag3d_v1/bag3d.h5` (⚠ slow chunking) + `/dev/shm/bag3d_fast.h5` (fast RAM copy) | 12 GB | **11,776 real watertight Dutch LoD2.2 buildings** → 64³ signed SDFs (`scripts/ingest_3dbag.py`, igl FAST_WINDING_NUMBER). Real gabled/L-shape/tower massing. | SDEdit massing prior (`dataset_mode=bag3d`) |
| **OSM tiles** | `data/osm_tiles/` (or downloaded on demand) | varies | Lafayette polygon shapefiles (east4, east2, west6, full Lafayette4x4) | OSM pipeline |
| **B+.7 BuildingNet fits** ✅ | `outputs/fit_recipes_buildingnet/best_params.npz` | 227 KB (1556 fits) | Dict `asset_id → {style, params, iou, polygon, bbox}`. Mean IoU 0.656, 68% > 0.5. 79% best-fit to modern (style bias). | B+.5/B+.6/B+.9 training data |
| **B+.4 procedural-param extraction** ✅ | `data/recipe_augmentation_v1/extracted_params/<style>_params.npz` | ~200 KB total (50k samples) | `(params, seed, style_id)` per style; recipe params recovered deterministically from the seed in each h5 sample | B+.5/B+.6/B+.9 training data |
| **B+.5 synthetic conditioning** ✅ | `outputs/recipe_param_dataset/synthetic_cond.npz` | 5.5 MB (50k samples) | `cond (N,46)` scale-invariant + `padded/mask/style_idx/class_idx/seed/height_m/shape_id`. Polygon+height recovered via rng replay (verified: footprint IoU 1.0). Balanced 8×6250. | B+.5/B+.6 training (folded in with real B+.7 fits) |

### Data quality caveats

- **BuildingNet meshes are non-watertight.** The "SDF" is effectively a UDF with a thin (~0.02 unit) negative shell near the surface. `(sdf <= 0)` identifies the *surface band*, NOT the *interior volume*. Use **footprint IoU** (top-down Y-collapse) as the structural quality metric, not iso=0 IoU.
- **Sparsity filter**: ~10% of assets have <100 footprint cells — broken meshes. `scripts/fit_recipes_to_buildingnet.py:fit_asset` has `fp_min_cells=100` and `iso_min_voxels=300` filters; sparsity stats at `outputs/buildingnet_sparsity_audit/sparsity.csv`.
- **Axis convention**: `(D=z, H=y, W=x)`, Y is up. Footprint = Y-collapse onto (D, W). See `memory/project_sdfusion_axes.md`.

---

## 4. Pipelines — the full picture

```
OSM tile (Lafayette / east4 / etc.)
       │
       ▼   per building: (footprint_polygon, class_id, height_m, style_id)
┌─────────────────────────────────────────────────────────────────────────────┐
│  ASSET PATHS (5 shipping/in-progress + 1 new research track)                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Path A: retrieved BuildingNet mesh → Hunyuan3D-2 refine → simplify → mesh  │  ✅ ship
│  Path B: baked 3DGS corpus lookup → GaussianSet                             │  ✅ ship
│  Path C: scene/sdf_recipes.py → marching_cubes(iso=0) → mesh                │  ✅ ship
│  ─────────────────────────────────────────────────────────────────────────  │
│  Path D: Stage 3a latent diffusion → SDF → MC → mesh                        │  ♻️ revived as ① SDEdit massing prior
│  Path E: Stage 3b lifter — SDF → voxelized Gaussians → GaussianSet          │  ⏸ blocked, not trained
│  ─────────────────────────────────────────────────────────────────────────  │
│  Option B+: recipe-param diffusion → params → DiffRecipe → SDF → MC → mesh  │  🟢 works; diversity-ceilinged
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
   place_mesh / place_gsplat (scene/run_demo.py + scene/gsplat_placement.py)
       │
       ▼
   gsplat_compose / trimesh.concatenate
       │
       ▼
   Final 3D town (scene.obj or scene.ply)
```

### Path A — Retrieved BuildingNet mesh → Hunyuan3D refine → place
- **Source files**: `scripts/osm_hunyuan_pipeline_smoke.py`, `scene/run_demo.py:place_mesh`
- **Status**: shipping. Default `--asset_format mesh`
- **Strengths**: best mesh quality so far
- **Weaknesses**: not generative; runs from a fixed retrieval corpus

### Path B — Baked 3DGS corpus lookup → place
- **Source files**: `scripts/osm_hunyuan_pipeline_smoke.py --asset_format gsplat`, `scene/gsplat_placement.py`, `scene/gsplat_compose.py`, `scene/gsplat_guardrail.py`, `scene/gsplat_renderer.py`
- **Status**: shipping. The 1849-asset v2 corpus is the "current production gsplat path".
- **Verification**: `outputs/osm_3dgs_east4_v2_corpus/scene.ply` opens cleanly. East4 verified end-to-end.

### Path C — Procedural SDF recipes
- **Source files**: `scene/sdf_primitives.py`, `scene/sdf_recipes.py`, `scripts/generate_sdf_building.py`, `demo_sdf_sculpting.ipynb`
- **Status**: shipping. `--asset_format sdf_procedural --sdf_style <s>`
- **Styles**: `modern, colonial, victorian, industrial, craftsman, mediterranean, contemporary, public_civic` (8)
- **Strengths**: footprint-faithful by construction (no `placed_flat` issues)
- **Weaknesses**: not generative; outputs are stylized procedural variants

### Path D — Stage 3a conditional SDF latent diffusion → **revived as the ① massing prior**
- **Source files**: `models/stage3a_model.py` (`sdedit()` + `inference()`), `datasets/{bag3d,hybrid}_dataset.py`, `configs/stage3a_sdf_diffusion.yaml`, `train.py`, `scripts/foundations/retrain_prior_hybrid.py`
- **Status**: Path D was paused for a from-pure-noise gap (DDIM from noise = speckle; `memory/project_stage3a_metric_inference_gap.md`), then **repurposed as the SDEdit massing prior** — it never generates from pure noise (SDEdit starts from a *partial*-noise edit). See the AI-Sculpting section below.
- **Current prior**: `logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt/stage3a_steps-latest.pth` (20k, footprint+height, CFG dropout + EMA) on the **clean VQVAE** `logs_building/vqvae_clean_ft/vqvae_clean.pth`. The earlier BAG-only `logs_building/2026-06-05T15-02-24-bag3d-prior-fast/ckpt/stage3a_steps-30000.pth` is what the (pre-foundations) live snap demo loaded.
- **VQVAE**: v1 superseded by the clean finetune (fixes gap#6); v2 retrain failed on aux losses. `memory/project_vqvae_v2_failure.md`, `project_vqvae_clean_finetune.md`.

### Path E — Stage 3b SDF→Gaussians lifter
- **Source files**: `models/stage3b_model.py`, `models/networks/sdf_to_gs_lifter.py`, `datasets/stage3b_dataset.py`, `configs/stage3b_lifter.yaml`, `launchers/train_stage3b.sh`
- **Status**: code written, not trained. Blocked on Stage 3a + GPU contention.
- **Architecture**: 3D UNet with FiLM, takes voxelized SDF + conditioning → `(32, 32, 32, 8 slots × 14 attrs)` Gaussian tensor.

### AI Sculpting — the current frontier (two layers)

The "generative when you *edit*" direction: add a crude primitive and it adapts into a coherent
building (Unbound-style, but with a learned snap-to-plausible layer — Unbound itself uses no ML).
Research basis: `memory/project_ai_sculpting_research.md`, `project_realtime_generative_framing.md`.

**① Massing prior (SDEdit).** `Stage3aModel.sdedit()` encodes the edited SDF → partial-noise →
guided denoise (reuses CFG; **autoguidance** added — guide the strong prior with a weaker ckpt of
itself, `project_autoguidance_sdedit`) → decode → marching cubes. One knob (`strength`) trades
faithfulness↔realism. Trained on real watertight 3D BAG massing (now the clean-VQVAE 20k prior,
footprint+height). **Snap-to-plausible is wired into the web demo**: `scripts/server/refine.py:
refine_sdedit` bridges a recipe edit → Frame-N → sdedit → world mesh, exposed at `/refine_with_edit`
mode=`sdedit` + a faithfulness slider (`project_snap_to_plausible_wired`). Tests:
`scripts/sdedit_bag3d_test.py`, `scripts/server/test_sdedit_refine.py`, `scripts/foundations/verify_snap_new_stack.py`.

**② Element composition.** `scene/composer_detail.py` (part-composer trained on real BuildingNet part
labels) decides class-appropriate elements; `scene/sdf_detail.py` instantiates roof / windows / door /
dome / towers. **Style and ornament live HERE** — the 64³ massing prior is footprint-dominated and
cannot condition on named style (confirmed empirically). `project_composer_detail_glue`, `project_part_composer`.

**Open / next:** the learned **part-mixing** model (`docs/TRACK2_part_mixing_design.md`) for coherent
add/replace of elements; richer real-time interaction (in-loop few-step SDEdit, DualSDF latent handles —
both proven feasible in `project_realtime_probes`).

### Option B+ — Recipe-parameter diffusion (active)

The current most promising direction. Designed 2026-06-01 as an alternative to Path D after Stage 3a's loss-vs-inference gap was diagnosed.

- **Memory**: `memory/project_option_b_plus_phase1.md`, `memory/project_option_b_plus_phase7.md`
- **Source files**:
  - `models/networks/diff_recipe.py` — all 8 differentiable recipes + `build_diff_recipe(style)` factory + `DIFF_RECIPE_REGISTRY`
  - `scripts/test_diff_recipe.py` — B+.1 modern-only verifier (sign_match 0.999 vs procedural)
  - `scripts/test_diff_recipe_all_styles.py` — B+.3 all-styles verifier (8/8 pass)
  - `scripts/test_diff_recipe_diversity.py` — design-space sampler (8 styles × 6 footprints)
  - `scripts/fit_recipes_to_buildingnet.py` — **B+.7 the critical experiment** (fit recipe params to real BuildingNet GT SDFs)
  - `scripts/visualize_recipe_fits.py` — fit quality sheet renderer
  - `scripts/survey_buildingnet_sparsity.py` — GT data quality audit
  - `scripts/debug_fit_single_asset.py` — one-asset diagnostic
- **Phases**:
  - ✅ **B+.1** — differentiable modern recipe, gradients flow, Adam recovers params (97.5% MSE reduction)
  - ✅ **B+.2** — soft clamps + IQ normalized smin variants
  - ✅ **B+.3** — all 8 styles match procedural at sign_match ≥ 0.986
  - ✅ **B+.7** — fit to real BuildingNet. Final 1556/1849 fits: **IoU mean 0.656, median 0.741, 68% > 0.5**. 79% best-fit to modern.
  - ✅ **B+.4** — synthetic param extraction via rng replay → `data/recipe_augmentation_v1/extracted_params/*.npz` (50k samples). Zero variance for victorian/industrial/mediterranean/public_civic.
  - ✅ **B+.5** — deterministic MLP head `(cond) → recipe_params`. **DONE 2026-06-03.** Overfit memorizes (train norm MSE → ~7e-4). **Real + 50k synthetic folded in**: held-out **footprint IoU 0.617 vs 0.629 fit ceiling = 98% retention** (real-only ablation: 0.608 / 96.7% — synthetic *improved* it), and all 8 styles now covered (mediterranean had 0 real fits). Conditioning is scale-invariant (real = Frame-N, synthetic = world meters). Files: `models/networks/recipe_param_space.py`, `models/networks/recipe_param_head.py`, `scripts/train_recipe_param_head.py`, `scripts/recover_synthetic_conditioning.py`. Outputs in `outputs/recipe_param_head_b5/` (+ `_realonly_si`, `_overfit` ablations). See `memory/project_option_b_plus_phase5.md`.
  - 🟡 **B+.6** — diffusion over the recipe-param space — **the "truly generative" head. SCAFFOLDED 2026-06-03.** Conditional DDPM (cosine schedule, ε-pred, masked loss) + DDIM sampling (`eta` stochastic diversity + classifier-free guidance), reusing the B+.5 stack. Canonical run (8000 ep, pool 26.6k; **converges by ~ep 1300**, loss floors ~0.11): **sampled-param footprint IoU 0.620 vs 0.631 fitted = 98.2% retention** — matches the deterministic head while being generative. Longer training confirmed the scaffold was already converged (2500 ep gave the same ~98%); B+.6 is data/config-limited, not training-time-limited. **Jitter** (`fit_param_normalizer_with_jitter`) rescues the zero-variance styles: mediterranean (0 real fits) sampled std 0.000→0.052 with jitter; reload generates 4 different mediterranean buildings from one footprint. Files: `models/networks/recipe_param_diffusion.py`, `scripts/train_recipe_param_diffusion.py`. Outputs in `outputs/recipe_param_diffusion_b6/`. See `memory/project_option_b_plus_phase6.md`. *Scaffold, not final — needs longer training + guidance/eta tuning.*
  - ⏳ **B+.8** — drop in FlexiCubes (`nv-tlabs/FlexiCubes`) for differentiable mesh extraction
  - ⏳ **B+.9** — joint train on procedural ∪ fitted-real with StEik divergence regularizer

---

## 5. Code map — which file does what

### Models
```
models/
├── stage3a_model.py            Path D prior + AI-Sculpting — conditional SDF latent diffusion; has sdedit() (SDEdit sculpt) + inference()
├── stage3b_model.py            Path E — SDF→Gaussians lifter (not trained)
├── vqvae_model.py              SDF VQVAE (v1 in production; v2 failed retrain)
├── sdfusion_model.py           Original SDFusion (unused)
├── sdfusion_model_img2shape.py Image→shape baseline (unused; _build_fp3d_for source)
├── sdfusion_img2shape_model.py Older variant
├── base_model.py               create_model() factory; registers stage3a, stage3b
├── arap_deformer.py            ARAP (abandoned per HANDOFF_2026-05-19.md — non-manifold meshes break libigl)
└── networks/
    ├── diff_recipe.py          Option B+ — all 8 differentiable recipes (~900 LOC, ALL OUR PHASE B+.1-3 WORK)
    ├── recipe_param_space.py   Option B+ — shared param-space contract (styles/dims/mask, featurizer, scalers, jitter-aware normalizer); used by B+.5 + B+.6
    ├── recipe_param_head.py     Option B+ — B+.5 deterministic param-prediction MLP head
    ├── recipe_param_diffusion.py Option B+ — B+.6 conditional diffusion (denoiser + cosine DDPM + DDIM/CFG sampling) — the generative head
    ├── sdf_residual_net.py     SDFCorrectionNet from old Plan X (abandoned)
    ├── sdf_to_gs_lifter.py     Stage 3b network
    ├── vqvae_networks/         VQVAE encoder/decoder
    ├── diffusion_networks/     UNet + DDIM sampler for Stage 3a
    └── retrieval/              FootprintEmbedNet (used as conditioning encoder)
```

### Datasets
```
datasets/
├── stage3a_dataset.py          Mixed BuildingNet + recipe-aug for Stage 3a
├── bag3d_dataset.py            real 3D BAG corpus for the SDEdit prior (dataset_mode=bag3d, style_id=8)
├── hybrid_dataset.py           foundations retrain: BAG + procedural (RAM-preloaded), footprint+height conditioning
├── stage3b_dataset.py          Stage 3b paired SDF + voxelized GS
├── buildingnet_dataset.py      Base BuildingNet loader (augment flag added)
└── base_dataset.py             CreateDataset() factory; registers stage3a, stage3b modes
```

### Scene composition (shipped)
```
scene/
├── sdf_primitives.py           Torch-native SDF primitives (Quilez-style IQ SDFs)
├── sdf_recipes.py              8 procedural style recipes
├── sdf_edit.py                 Stage 4 interactive edit engine (EditableBuilding + palette EditOps)
├── sdf_detail.py               Procedural facade detail (windows/bands/cornice/plinth/roof/landmarks + NEW add_door)
├── composer_detail.py          NEW — the GLUE: part-composer decides elements → sdf_detail instantiates (compose_detail)
├── sdf_vqvae_prior.py          Optional VQVAE smoothing for procedural SDFs
├── run_demo.py                 place_mesh (mesh asset placement)
├── gsplat_common.py            Inria PLY loader/writer
├── gsplat_placement.py         place_gsplat (Gaussian asset placement)
├── gsplat_compose.py           Scene composition for Gaussian paths
├── gsplat_guardrail.py         cull_outside_footprint
└── gsplat_renderer.py          gsplat 1.5.3-based renderer
```

### Scripts (organized by purpose)
```
scripts/
# OSM pipeline drivers
├── osm_hunyuan_pipeline_smoke.py         Main OSM→3D pipeline. --asset_format {mesh, gsplat, sdf_procedural}
├── osm_pipeline_map_choices.py           Map renderer for choices
├── osm_recompose_height_policy.py        Inference-time height policy

# Data prep (one-time)
├── precompute_asset_heights.py           Phase 1b.1 (DONE) — produces asset_dimensions.csv
├── generate_recipe_augmentation.py       Phase 1b.3 (DONE) — produces 50k recipe-aug samples
├── voxelize_gsplats.py                   Phase 1c (DONE) — produces gsplat_voxelized_32k8/
├── compute_scale_factor_v2.py            VQVAE latent std for scale_factor
├── repreview_gsplat_v2.py                Phase 0.1 (DONE) — gsplat previews

# Stage 3a evaluation / debug
├── stage3_generate.py                    Standalone inference CLI (uses Stage 3a + Stage 3b)
├── audit_stage3a_inference.py            ⚠ The script that diagnosed Stage 3a's metric-vs-inference gap
├── inference_variants_test.py            Tests 6 inference strategies on Stage 3a ckpt
├── eval_vqvae_ab.py                      VQVAE v1 vs v2 A/B comparison

# AI Sculpting — SDEdit + composer (NEW, the frontier — see docs/HANDOFF_2026-06-06.md)
├── ingest_3dbag.py                       🔥 3D BAG OGC API → watertight LoD2.2 → 64³ SDF h5 (data/bag3d_v1). ⚠ chunks=(1,64,64,64)
├── sdedit_bag3d_test.py                  🔥 SDEdit on the trained prior — the REAL quality signal (before/edited/sdedit montage)
├── sdedit_sculpt.py                      SDEdit test on procedural-corpus input
├── test_composer_detail.py              🔥 composer→detail glue: per-class element placement (massing vs composed)
├── preview_corpus.py / render_montage.py corpus / mesh montage renderers
# the SDEdit method = models/stage3a_model.py:sdedit() ; the glue = scene/composer_detail.py ;
# the dataset = datasets/bag3d_dataset.py (dataset_mode=bag3d) ; door = scene/sdf_detail.py:add_door()

# Web demo (NEW) — scripts/server/
├── server/inference_service.py           FastAPI: /health /generate_from_image /regenerate_building /refine_with_edit + serves web/index.html  (⚠ launch sandbox-OFF)
├── server/web/index.html                 three.js demo: upload footprint image/OSM → 3D town; click-to-select + restyle a building
├── server/footprint_image.py             extract_footprints (skimage Otsu+contours) → to_meters
├── server/make_sample_footprints.py      builds web/samples/{synthetic_blocks,munich_oldtown,lafayette}.png + samples.json

# Option B+ (NEW, active)
├── test_diff_recipe.py                   B+.1 verifier
├── test_diff_recipe_all_styles.py        B+.3 verifier
├── test_diff_recipe_diversity.py         Diversity sampler
├── fit_recipes_to_buildingnet.py         🔥 B+.7 fitter — the critical experiment
├── extract_recipe_params.py              B+.4 synthetic param extraction (rng replay)
├── recover_synthetic_conditioning.py     B+.5 recover synthetic (polygon,height,class) via rng replay → synthetic_cond.npz
├── train_recipe_param_head.py            🔥 B+.5 deterministic head trainer (real+synthetic, overfit + IoU eval + sheet)
├── train_recipe_param_diffusion.py       🔥 B+.6 diffusion trainer (jitter + sampling IoU + param diversity check + sheet)
├── sweep_recipe_diffusion_sampling.py    B+.6 guidance/eta sweep (guidance = diversity knob; picks service defaults)
├── server/recipe_inference.py            Stage A engine: cond → diffusion params → recipe mesh → glb (embeddable)
├── server/inference_service.py           Stage A FastAPI: /health /params_to_mesh /regenerate_building /generate_tile
├── server/validate_generated_meshes.py   validate 3D mesh form + diversity (matplotlib gallery)
├── server/validate_scene_compose.py      compose many buildings from real BuildingNet footprints into one scene
├── server/demo_osm_town.py               🔥 capstone: real OSM tile (extract_osm) → B+.6 generated 3D town
├── server/demo_munich_town.py            🔥 FULL data-grounded town: B+.6 + facade detail + BuildingNet-occurrence landmarks + OSM roof:shape (Munich)
├── server/test_facade_detail.py          facade detail (windows/bands/cornice/plinth) before/after
├── server/test_landmarks.py              rich landmarks (dome/tower/steps) grounded in BuildingNet labels
├── extract_buildingnet_detail_stats.py   real per-class glazing/roof stats from local BuildingNet labels
├── identify_buildingnet_labels.py        ID the 31 part labels by geometry (window=2 wall=1 roof=4 dome=22 tower=7 stairs=17)
├── visualize_recipe_fits.py              Render fit quality sheets
├── survey_buildingnet_sparsity.py        GT quality audit
├── debug_fit_single_asset.py             Single-asset debug

# Comparison sheets / paper figures
├── build_path_comparison_sheet.py        GT vs Path D ceiling vs Path B side-by-side
```

### Configs
```
configs/
├── vqvae_bnet.yaml             v1 VQVAE (in production)
├── vqvae_bnet_v2.yaml          v2 VQVAE (failed; kept for reference)
├── stage3a_sdf_diffusion.yaml  Path D config
├── stage3b_lifter.yaml         Path E config
└── sdfusion-img2shape.yaml     Original image→shape baseline
```

### Launchers
```
launchers/
├── train_stage3a.sh                       Path D training
├── train_sdfusion_img2shape_smoke.sh      Image baseline smoke
├── train_stage3b.sh                       Path E training (not yet launched)
└── train_vqvae_bnet_v2.sh                 v2 VQVAE retrain (failed; for reference)
```

---

## 6. Output map — which outputs belong to which pipeline

```
outputs/
├── stage3_metadata/                                Phase 1b — metadata caches (heights, dimensions)
│   └── asset_dimensions.csv                        1850 rows; needed by Stage 3a + B+.7
│
├── vqvae_ab_diagnostic_t03/                        VQVAE v1 vs v2 A/B (16 val assets)
│   ├── visual_sheet.png                            cols = GT | v1 | v2-final | v2-best (proves v1 > v2)
│   ├── per_asset.csv
│   └── summary.txt
│
├── audit_stage3a_2026_05_29/                       Stage 3a inference audit (the bug hunt)
│   └── audit_sheet.png                             5 cols: A_GT, B_VQ_ceiling, C_t50, D_t500, E_DDIM
│
├── path_comparison_2026_05_27/                     Path A vs B vs D-ceiling
│   └── path_comparison.png                         16 buildings × 3 paths
│
├── fit_recipes_buildingnet/                        🔥 B+.7 outputs (active)
│   ├── per_asset_fits.csv                          asset_id, ok, best_style, best_iou, per-style IoU/L1
│   ├── best_params.npz                             dict[asset_id → fit dict]; for B+.5/B+.9 training
│   └── visuals/
│       └── fit_quality_sheet.png                   top4 + med4 + bot4 GT vs fitted recipe
│
├── recipe_param_dataset/
│   └── synthetic_cond.npz                          🔥 B+.5 recovered synthetic conditioning (50k × 8 styles; cond+padded+mask+meta)
├── recipe_param_head_b5/                           🔥 B+.5 deterministic head — CANONICAL (real + synthetic combined)
│   ├── head.pth                                    trained MLP + args (reload needs hidden/depth/dropout from args)
│   ├── scalers.npz                                 FeatureScaler + ParamNormalizer stats (needed to use head)
│   ├── metrics.json                                pool composition, norm-MSE history, per-style raw MSE, IoU summary
│   └── head_iou_sheet.png                          GT | fitted-param | predicted-param footprints
├── recipe_param_head_b5_realonly_si/               B+.5 ablation: real-only, scale-invariant (IoU 0.608 / 96.7%)
├── recipe_param_head_b5_overfit/                   B+.5 memorization check (train on all real, no val → ~7e-4)
├── recipe_param_diffusion_b6/                       🔥 B+.6 generative diffusion (scaffold)
│   ├── denoiser.pth                                trained ConditionalDenoiser + args
│   ├── scalers.npz                                 FeatureScaler + (jitter-aware) ParamNormalizer
│   ├── jitter.npz                                  jitter_mask + strength (which (style,dim) cells were jittered)
│   ├── metrics.json                                loss history, sampled-IoU summary, per-style diversity
│   ├── diffusion_iou_sheet.png                     GT | fitted-param | SAMPLED-param footprints
│   ├── sampling_sweep.{csv,json}                   guidance/eta sweep (guidance = diversity knob)
│   ├── mesh_gallery_{styles,diversity}.png         3D mesh-form validation (8 styles; seed/guidance diversity)
│   ├── scene_demo.{png,glb}                        13 buildings from real BuildingNet footprints composed
│   └── osm_town_demo{,_big}.{png,glb}              🔥 real Lafayette OSM tile → generated town (10 / 62 buildings)
│
├── buildingnet_sparsity_audit/
│   └── sparsity.csv                                per-asset {n_iso, n_fp, sdf_min}; informs filter thresholds
│
├── diff_recipe_phase1/                             B+.1
│   └── test1_combined.png                          modern: procedural vs DiffRecipeModern
│
├── diff_recipe_phase3/                             B+.3
│   ├── all_styles_sheet.png                        all 8 styles, default params
│   └── <style>_{ref,diff}.png                      per-style ref vs diff (16 PNGs)
│
├── diff_recipe_diversity/                          Design-space demo
│   ├── design_space_8styles_6variants.png          8 styles × 6 footprint variants
│   └── <style>_var<n>_<footprint>.png              48 individual variant PNGs
│
├── osm_3dgs_east4_v2_corpus/  (731 MB)             Path B output on east4 OSM tile
├── osm_3dgs_baked_east4/ + east2/                  Path B older runs
├── osm_sdf_east4_{colonial,victorian,...}/         Path C output, per-style
├── _artifacts_v2_round/                            Misc procedural SDF artifacts
└── _artifacts_stage3/                              (planned: Phase 4 A/B sheets)

logs_building/
├── 2025-05-19T19-58-28-vqvae-...release/           ✅ v1 VQVAE (in use)
├── 2026-05-24T23-07-13-stage3a-bs32-80k/           Original Stage 3a dead run (iter 6000 ckpt)
├── continue-2026-05-27T13-52-02-stage3a-bs32-...   First resume (iter 9000)
├── continue-2026-05-29T22-24-26-stage3a-...-iter9k/ Latest Stage 3a (iter ~8000 of resume2; PAUSED)
└── _launch_logs/                                   Bash launch logs

# Deleted today (in trash, recoverable until purge):
/scratch/gilbreth/dsimhadr/.trash_2026-06-01/        ~1.6 GB of stale logs/outputs
```

---

## 7. Memory — institutional knowledge for agents

Auto-memory location: `/home/dsimhadr/.claude/projects/-scratch-gilbreth-dsimhadr-GenerativeTowns-SDFusion/memory/`

**`MEMORY.md` is the live index — read it first; this README does not duplicate it.** Highest-value
recent entries:

- `project_realtime_generative_framing` — the frontier framing (① massing prior, ② detail; where the
  generative compute sits relative to the edit loop; the "better-than-Unbound" axes).
- `project_snap_to_plausible_wired` — snap-to-plausible wired into the web demo (the world↔Frame-N bridge).
- `project_vqvae_clean_finetune` — clean VQVAE finetune; **gap#6 fixed** (simple losses + clean data, not aux losses).
- `project_hybrid_prior_retrain` — the 20k footprint+height prior + the **style-collapse root-cause**.
- `project_realtime_probes` — real-time feasibility (few-step SDEdit ~100 ms; latent interpolation on-manifold).
- `project_training_audit` + `project_gap_fixes_research` — what's missing in the prior + papers→fixes.
- `project_b6_diversity_ceiling` — why recipe-param diffusion plateaued (motivates the shape-latent prior).
- `project_sdfusion_axes` — the (D=z,H=y,W=x) convention · `feedback_dont_underweight_existing_infra` — a user pref.

---

## 8. Planning docs — what was thought through, when

| File | Date | What it covers |
|---|---|---|
| `docs/HYBRID_PIPELINE_PLAN.md` | early May 2026 | Original "SDF-as-footprint-correction + ARAP" plan. **Abandoned** because libigl ARAP fails on BuildingNet non-manifold meshes. |
| `docs/HANDOFF.md` | 2026-05-08 | Snapshot of the SDFCorrectionNet line of work (Plan X) before the pivot |
| `docs/PROJECT_STATUS.md` | mid-May | General project status |
| `docs/CODEX_PROGRESS_REPORT_2026-05-08.md` | 2026-05-08 | Codex agent progress reports |
| `docs/CODEX_PROGRESS_REPORT_2026-05-09.md` | 2026-05-09 | "" |
| `docs/CODEX_RESEARCH_NOTES_GENERATIVE_PROPOSALS_2026-05-11.md` | 2026-05-11 | Survey of related papers (PrITTI, GenCAD, GaussianCity, INST-Sculpt, etc.) — **read this if proposing new methods** |
| `docs/CODEX_FULL_PIPELINE_REPORT_2026-05-12.md` | 2026-05-12 | Full pipeline snapshot |
| `docs/HANDOFF_2026-05-19.md` | 2026-05-19 | Pre-Option-B+ formal handoff. Paths A/B/C shipping, ARAP abandonment, related-work survey. |
| `docs/HANDOFF_2026-06-06.md` | 2026-06-06 | The AI-sculpting frontier (last narrative handoff): SDEdit prior + composer/detail glue, 3D BAG corpus, gotchas. Newer foundations work (2026-06-08) lives in the `project_*` memories, not a doc. |
| `~/.claude/plans/proud-waddling-cocoa.md` | 2026-05-24 | The Stage 3 plan (Path D + Path E specification) |
| `~/.claude/plans/cheeky-wiggling-quokka.md` | early May | 3DGS-based OSM-to-Town pipeline plan with categorical style + SDF guardrails |
| `docs/DEPLOYMENT_PLAN.md` | 2026-06-02 | **End-state plan**: Unreal/Blender/Unity plugin strategy for city-scale deployment. Read this when starting plugin work. Stage A marked scaffolded. |
| `docs/OPTION_B_PLUS_REPORT_2026-06-03.md` | 2026-06-03 | **Consolidated B+.5→B+.6→Stage A→OSM-town report**: all numbers, the diversity-ceiling finding + B+.6h height-generation fix. |
| `docs/professor_report/REPORT.md` | 2026-06-06 | Clean external progress report (+ PDF/HTML): goal, two-layer method, 3D BAG data, sculpt/composer results, plateau. Good high-level overview. |
| `docs/TRACK2_part_mixing_design.md` | 2026-06-07 | Design for the learned part-proxy + global-mixing model (coherent add/replace of elements) — the principled ② upgrade. Designed, not built. |

---

## 9. Research papers consulted (recent agent runs in this project)

- **SDFusion** (arxiv 2212.04493, CVPR 2023) — foundation of Path D + the AI-Sculpting SDEdit prior
- **SDEdit** (arxiv 2108.01073, ICLR 2022) — 🔥 **the core AI-Sculpting mechanism**: noise the user's crude edit → denoise with a generative prior → snap onto the manifold; `strength` = faithfulness↔realism. Implemented as `Stage3aModel.sdedit()`.
- **DualSDF** (arxiv 2004.02869, CVPR 2020) — shared coarse-primitive ↔ fine-SDF latent; considered for sculpting (drag a handle, fine surface follows)
- **SPAGHETTI** (arxiv 2201.13168, SIGGRAPH 2022) + **SALAD** (2303.12236, ICCV 2023) — part-latent Gaussians + transformer "mixing network" re-harmonizes parts; the Route-B (part-aware) sculpting option
- **3DShape2VecSet** (arxiv 2301.11445, SIGGRAPH 2023) — vec-set latent for diffusion + native shape completion; the general-representation option if we rebuild the prior
- **SuperFit / SuperFrusta** (arxiv 2512.09201, CVPR 2026) — residual primitive fitting; informed B+.7's fitter design (`memory/project_option_b_plus_phase7.md` for details)
- **INST-Sculpt** (arxiv 2502.02891) — ZBrush-style stroke editing of neural SDFs; relevant for Stage 4 interactive UX
- **PrITTI** — primitive-based controllable generation
- **SALAD** (arxiv 2303.12236, ICCV 2023) — part-level latent diffusion; architecture reference for B+.6
- **GenCAD** (arxiv 2409.16294) — image-conditioned CAD-parameter diffusion; architecture reference for B+.6
- **FlexiCubes** (nv-tlabs, SIGGRAPH 2023) — differentiable mesh extraction; planned for B+.8
- **HF-NeuS / Implicit Displacement Fields** (arxiv 2106.05187) — base SDF + learned displacement; Option D fallback if B+ library is too limited
- **DiffCSG** (arxiv 2409.01421, SIGGRAPH Asia 2024) — differentiable CSG via rasterization
- **UCSG-Net** (arxiv 2006.09102, NeurIPS 2020) — primitive-parameter prediction with soft binarization (annealing trick we may borrow)
- **StEik** (arxiv 2305.18414, NeurIPS 2023) — divergence regularizer for stable SDF optimization; planned for B+.9
- **Inigo Quilez SDF body of work** — `iquilezles.org/articles/distfunctions/` + `iquilezles.org/articles/smin/` — all our primitives are built on this; B+.2 uses IQ normalized quartic smin

---

## 10. Quick-start for agents

### Environment
```bash
# Activate (no conda activate needed; in-repo venv)
PY=/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/sdfusion/bin/python

# Always strip XALT before running python on Gilbreth
env -u LD_PRELOAD -u LD_LIBRARY_PATH $PY ...

# If imports fail with "unknown location" — Gilbreth scratch purged the .py files.
# Recovery: see memory/env_gilbreth_scratch_purge.md
#   pip install --force-reinstall --no-deps -r sdfusion_env_freeze_clean.txt
```

### Common operations
```bash
# Check what training is running
pgrep -af "train.py"

# Retrain the massing prior (clean VQVAE + footprint+height + CFG dropout + EMA)
# ⚠ recipe h5 has slow chunking -> HybridDataset preloads it to RAM; nThreads 0 (h5py+fork hangs)
env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 \
  ./sdfusion/bin/python scripts/foundations/retrain_prior_hybrid.py --total_iters 20000
# status: tail logs_building/<run-stamp>-stage3a-hybrid-clean/loss_log.txt

# Run B+.7 fit on N assets
env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=0 \
  ./sdfusion/bin/python scripts/fit_recipes_to_buildingnet.py \
    --n_assets 100 --steps 600

# Audit Stage 3a (diagnose loss-vs-inference gap)
env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=0 \
  ./sdfusion/bin/python scripts/audit_stage3a_inference.py \
    --ckpt <path_to_ckpt>

# Re-render a fit quality sheet
env -u LD_PRELOAD -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="" \
  ./sdfusion/bin/python scripts/visualize_recipe_fits.py

# AI Sculpting (see docs/HANDOFF_2026-06-06.md). ⚠ GOTCHAS: corpus h5 needs chunks=(1,64,64,64);
# train with --nThreads 0 (h5py+fork hangs) + HDF5_USE_FILE_LOCKING=FALSE; corpus in /dev/shm (RAM).

# Snap-to-plausible on the new stack (clean VQVAE + 20k prior) — the deployed sculpt path
env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 \
  ./sdfusion/bin/python scripts/foundations/verify_snap_new_stack.py
# → outputs/foundations/verify_snap_new_stack.png
# VQVAE recon benchmark (the gap#6 check): scripts/foundations/bench_vqvae_recon.py --vq_ckpt <ckpt>

# Composer → detail glue (per-class element placement)
env -u LD_PRELOAD -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="" \
  ./sdfusion/bin/python scripts/test_composer_detail.py    # → outputs/composer_detail_preview/

# Web demo (⚠ sandbox must be OFF for the server to bind a port)
env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  ./sdfusion/bin/python -m uvicorn scripts.server.inference_service:app --host 0.0.0.0 --port 8099
# then ssh -L 8099:<gpu-node>:8099 ... and open http://localhost:8099/
```

### Where to look for stuck training
1. `loss_log.txt` in the latest log dir — losses by iter
2. `images/test_step*_gen_.png` — actual generative outputs
3. `audit_stage3a_inference.py` — **definitive** check whether DDIM from pure noise produces buildings
4. If `simple` loss is low but `images/` is noise → metric-vs-inference gap (`memory/project_stage3a_metric_inference_gap.md`)

---

## 11. Glossary

- **Path A/B/C/D/E** — five asset-generation paths (see §4)
- **Option B+** — the recipe-parameter diffusion track (B+.1–B+.9). Works, but hit a diversity ceiling; the shape-latent **SDEdit massing prior** (AI-Sculpting) is now the primary direction
- **Frame N** — BuildingNet normalized frame, asset fits in [-1, 1]^3, axes (D=z, H=y, W=x), Y up
- **Frame W** — world frame (meters, OSM coords)
- **Footprint IoU** — top-down Y-collapse silhouette IoU; the **correct** structural metric for non-watertight BuildingNet meshes
- **sign IoU / iso=0 IoU** — voxelwise occupancy IoU; **wrong** for BuildingNet because meshes are non-watertight
- **Recipe params** — for Option B+, the ~5-12 floats per style that parameterize a building. See `models/networks/diff_recipe.py:<Style>ParamLayout`.
- **DiffRecipe** — a torch.nn.Module wrapping a procedural recipe so it's end-to-end differentiable in the parameters

---

## 12. Things future agents should NOT touch (without asking)

- The **clean VQVAE** `logs_building/vqvae_clean_ft/vqvae_clean.pth` and the **20k massing prior** `logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt/` — the current foundation (snap demo / SDEdit depend on these). The old v1 VQVAE (`...2025-05-19T19-58-28-vqvae-...release/`) + 30k BAG prior are kept for the pre-foundations demo.
- The procedural recipe library in `scene/sdf_recipes.py` / `scene/sdf_primitives.py`. Many things import these unchanged. If extending, ADD to them; don't refactor in place.
- The Stage-3a checkpoints under `logs_building/*-stage3a-*/ckpt/` — 11–15 GB each, many GPU-hours apiece.

---

## 13. If you're a new agent — start here

1. Read this README end-to-end (you're doing it).
2. Read `memory/MEMORY.md` index for institutional knowledge.
3. Check what's running: `pgrep -af "train.py\|fit_recipes"`.
4. Check the open tasks via `TaskList` (if Claude Code) or in-conversation context.
5. **Ask the user before**:
   - Killing any background process
   - Running `git restore` / `git reset`
   - Touching v1 VQVAE checkpoint
   - Starting another training run while one is active (will OOM the GPU)
6. Skim `docs/HANDOFF_2026-05-19.md` for the formal pre-Option-B+ snapshot.

---
