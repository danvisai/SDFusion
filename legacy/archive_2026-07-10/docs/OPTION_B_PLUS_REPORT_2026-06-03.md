# Option B+ — B+.5 → B+.6 → Stage A → end-to-end validation

**Date:** 2026-06-03
**Scope:** one working session. Took Option B+ from "B+.4/B+.7 data curated" to a trained
generative head, a deployed inference service, a validated real-OSM→town pipeline, and a
rigorous diversity analysis (with a fix experiment).
**Audience:** future agents / Danvi. Companion to `README.md` and the `memory/` files cited
inline.

---

## TL;DR

- **B+.5 deterministic head** (`(cond)→params` MLP): footprint-IoU **0.617 / 98% retention**
  of the B+.7 fit ceiling on held-out real buildings. All 50k synthetic conditioning
  recovered via rng replay (verified footprint IoU 1.0) and folded in; conditioning made
  scale-invariant so real (Frame-N) and synthetic (meters) share a space.
- **B+.6 diffusion head** (the "truly generative" one): conditional DDPM/DDIM over the 12
  recipe params. Trained to convergence (8k ep). **Samples at 98.2% retention** — i.e. it
  *matches* the deterministic head. Jitter rescues the zero-variance styles (mediterranean
  0.000→0.052). EMA: no change (confirmed training is not the lever).
- **Stage A inference service** (`scripts/server/`): FastAPI wrapping the head; symbolic
  input → diffusion params → recipe mesh → glb. `params_to_mesh` ~20 ms (slider budget).
- **Full pipeline validated end-to-end:** real Lafayette OSM tile (62 buildings, osmnx +
  Overpass) → generated 3D town. This is the literal project deliverable, working.
- **Key finding:** per-footprint **generative diversity is ~0** and the bottleneck is the
  parameterization + one-to-one data, NOT the diffusion. A height-generation experiment
  (B+.6h) recovers a *modest* 6× diversity with class-appropriate heights; a 120-ep smoke
  over-claimed 40× (undertraining — corrected).

---

## 1. What was built (files)

| File | Role |
|---|---|
| `models/networks/recipe_param_space.py` | param-space contract: styles/dims/mask, **scale-invariant featurizer** (COND_DIM=46), `FeatureScaler`+`ParamNormalizer`, `fit_param_normalizer_with_jitter` |
| `models/networks/recipe_param_head.py` | B+.5 deterministic MLP head |
| `models/networks/recipe_param_diffusion.py` | B+.6 `ConditionalDenoiser` + `GaussianDiffusion` (cosine, ε-pred, masked loss, DDIM/CFG) |
| `scripts/recover_synthetic_conditioning.py` | recover 50k synthetic (polygon,height,class) via rng replay → `synthetic_cond.npz` |
| `scripts/train_recipe_param_head.py` | B+.5 trainer (real+synthetic, overfit + IoU eval) |
| `scripts/train_recipe_param_diffusion.py` | B+.6 trainer (jitter + EMA + sampling IoU + diversity) |
| `scripts/sweep_recipe_diffusion_sampling.py` | guidance/eta sweep |
| `scripts/train_recipe_diffusion_genheight.py` | **B+.6h** height-generation experiment |
| `scripts/server/recipe_inference.py` | Stage A engine (embeddable) |
| `scripts/server/inference_service.py` | Stage A FastAPI |
| `scripts/server/validate_generated_meshes.py` | 3D mesh-form gallery |
| `scripts/server/validate_scene_compose.py` | compose buildings from real footprints |
| `scripts/server/demo_osm_town.py` | real OSM tile → generated town |
| `scripts/server/compare_heads_diversity.py` | B+.5 vs B+.6 quality+diversity |
| `scripts/server/render_genheight_diversity.py` | B+.6h class-structure + render |

Outputs live under `outputs/recipe_param_head_b5*/`, `outputs/recipe_param_diffusion_b6*/`,
`outputs/recipe_diffusion_genheight/`, `outputs/recipe_param_dataset/`.

---

## 2. Numbers

### B+.5 deterministic head (footprint IoU, held-out real val)
| config | pool | IoU | retention | styles |
|---|---|---|---|---|
| real-only | 1.3k | 0.608 | 96.7% | 7/8 |
| real + 50k synthetic | 26.6k | **0.617** | **98.1%** | 8/8 |

Scale-invariant featurizer (drop absolute w/d/area/height; keep outline + aspect/fill/
compactness/slenderness) was the prerequisite for mixing the two frames; it *raised*
real-only IoU 96.0→96.7%.

### B+.6 diffusion head
- 8000 ep, pool 26.6k, **converges by ~ep 1300** (loss floors ~0.11).
- Sampled-param footprint IoU **0.620 / 98.2% retention** (matches B+.5).
- EMA (decay 0.999): identical (0.620). Training is not the lever.
- Sampling sweep: **guidance is a diversity knob, not a quality knob** here (g=1 best IoU;
  higher g trades ~0.5% IoU for 2–9× param diversity). Service default g=2, eta=1.

### Stage A service (TestClient)
- `/regenerate_building`, `/generate_tile`, `/params_to_mesh` (~20 ms), `/health`. glb
  decodes + loads in trimesh. Mixed-style batch generates. Bad style → 400.

### End-to-end OSM
- `--bbox 40.4175 -86.8965 40.4215 -86.8915` → 62 Lafayette buildings → one scene
  (541k verts / 1M faces). `osm_town_demo_big.png/glb`.

---

## 3. The diversity finding (and the B+.6h fix) — read this

`compare_heads_diversity.py` measured, for held-out footprints, K samples per footprint:

| guidance | B+.6 footprint IoU | B+.5 footprint IoU | 3D-occupancy diversity |
|---|---|---|---|
| 2.0 | 0.586 | 0.590 | 0.008 |
| 3.0 | 0.615 | 0.616 | 0.001 |
| 8.0 | 0.606 | 0.616 | 0.006 (quality dropping) |

**Per-footprint generative diversity is ~0.** Root cause:
1. footprint AND height are conditioning INPUTS that dominate the building volume;
2. `(footprint-shape → params)` is nearly one-to-one in the data (each B+.7 fit is a single
   deterministic optimum; the procedural recipe is deterministic given a footprint).

A fixed-grid lever analysis: **height swept 0.6–1.4× gives 0.300 diversity** (~60× the
diffusion's actual) while a recipe param gives 0.001 → height is the lever, but it's an input.

**B+.6h experiment** — move `slenderness` (= height/√area) from input to a 13th generated
dim, conditioning only on footprint shape + class + style:
- 120-ep smoke: 0.324 diversity (40×). **This was undertraining — do not cite it.**
- 4000-ep converged: **0.049 diversity (6×), footprint IoU 0.594**, heights CLASS-STRUCTURED
  (craftsman ~8 m, public ~12 m, commercial ~16 m).

**Honest reframe:** within a fixed (footprint, class, style), a building's height/form is
*legitimately* fairly determined, so large intra-conditioning diversity may be the wrong
goal. A town's diversity comes from the **variety of footprints + classes across the tile**
(inter-building), which already works. The generative head's real value is plausible
per-input generation + samplable class-appropriate height/style — not dramatic
same-footprint variation. Bigger expressiveness gains need a richer recipe/displacement
target (Option D / B+.8 FlexiCubes), or discrete roof/massing types added to the recipes.

Full detail: `memory/project_b6_diversity_ceiling.md`.

---

## 4. Recommended next steps

1. **Ship B+.5 or B+.6 behind Stage A as-is** — both deliver ~0.61 footprint IoU; B+.6 adds
   samplable height/style. The service contract is stable either way.
2. **Don't** train longer / tune sampling for quality — it's ceilinged by the recipe library.
3. To raise *quality* (boxiness) and *expressiveness*: extend the recipe library with discrete
   roof/massing types (ADD to `scene/sdf_recipes.py`, don't refactor), or add a HF-NeuS-style
   displacement field (Option D), or B+.8 FlexiCubes for differentiable mesh + joint loss.
4. **Stage A productionization:** real GeoJSON ingestion + CRS→meter projection, `/refine_with_edit`,
   Docker + live uvicorn, batched sampling for big tiles.
5. Then `docs/DEPLOYMENT_PLAN.md` Stage B (Blender) / Stage C (Unreal).

---

## 5. Caveats / gotchas discovered

- Headless rendering: no pyrender/OpenGL on this env; matplotlib 3D only (pyglet needs a display).
- osmnx 2.0.7 + Overpass reachable from the Gilbreth compute node → live OSM works.
- PyTorch 2.6 `torch.load` defaults to `weights_only=True`; our ckpts that pickle argparse
  PosixPaths need `weights_only=False` (trusted) — fixed the savers to stringify Paths.
- Measuring height diversity needs a FIXED query grid: `_query_grid` derives both grid and
  height from bbox, so scaling height scales the grid and occupancy is scale-invariant.
- Smoke runs lie about diversity (undertrained models over-generate). Always confirm on a
  converged run.
