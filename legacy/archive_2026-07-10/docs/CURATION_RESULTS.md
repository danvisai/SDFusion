# Curation Results — Phase B+.4 + B+.7

**Date:** 2026-06-03
**Status:** Complete. Both data sources ready for Phase B+.5 (deterministic head) and B+.6 (recipe-param diffusion).
**Related:** `README.md`, `memory/project_option_b_plus_phase7.md`, `docs/DEPLOYMENT_PLAN.md`

---

## TL;DR

We now have ~51,500 (conditioning → recipe_params) training pairs ready for the generative model:

- **1,556 real-data pairs** from BuildingNet — recipe params extracted via gradient descent (Phase B+.7)
- **50,000 synthetic pairs** from procedural recipe samples — recipe params recovered deterministically from the saved seed (Phase B+.4)

The real-data fit quality is **mean IoU 0.656 / median 0.741**, with **68% above IoU 0.5** and **53% above IoU 0.7**. There's a **strong modern-style bias** (79% of real fits map to modern) that needs addressing during diffusion training.

---

## Phase B+.4 — synthetic param extraction (DONE)

**Script:** `scripts/extract_recipe_params.py`
**Source data:** `data/recipe_augmentation_v1/{style}.h5` (8 files, 6250 samples each, 50k total)
**Output:** `data/recipe_augmentation_v1/extracted_params/{style}_params.npz`

### Method
The procedural generator saves the random `seed` per sample. The recipes in `scene/sdf_recipes.py` are deterministic given that seed. The extraction script **replays** the rng calls per recipe to recover the random decisions (mech_active, chimney offset, etc.) and converts them to the DiffRecipe parameter vector layout in `models/networks/diff_recipe.py`.

No SDF re-computation needed — just replay the rng. The conversion takes seconds.

### Results

| Style | N samples | n_params | Random decisions | Notes |
|---|---|---|---|---|
| modern | 6250 | 9 | mech_active + 2 conditional offsets | most random variation |
| colonial | 6250 | 5 | chimney_active + 1 conditional offset | simple |
| victorian | 6250 | 12 | **none** (all hardcoded) | all samples have identical params |
| industrial | 6250 | 7 | none | identical |
| craftsman | 6250 | 6 | porch_active | binary variation |
| mediterranean | 6250 | 3 | none | identical |
| contemporary | 6250 | 6 | 2 offsets | continuous variation |
| public_civic | 6250 | 8 | none | identical |

**Caveat:** styles with no random decisions (victorian, industrial, mediterranean, public_civic) have **identical synthetic params across all 6250 samples**. This is by design — the procedural recipes use random seed only for those few decisions. For the diffusion model, these styles will look like deterministic point distributions in parameter space. We may need to add jitter at training time to make them learnable beyond memorization.

### DiffRecipe-vs-procedural fidelity
The DiffRecipe forward with extracted params produces an SDF ~99% identical to the procedural version. The ~1% difference comes from:
- Soft-clamp replacing hard `max()` in DiffRecipe
- Sigmoid blending replacing hard union for occupancy gates (mech, chimney, porch)

This is intentional — the diffusion model only needs the param distribution, not exact SDF replication.

---

## Phase B+.7 — real BuildingNet fitting (DONE)

**Script:** `scripts/fit_recipes_to_buildingnet.py`
**Source data:** `data/BuildingNet_dataset_v0_1/resolution_64/{asset_id}/ori_sample_grid.h5` (1849 assets)
**Output:**
- `outputs/fit_recipes_buildingnet/per_asset_fits.csv` (1849 rows; 1556 with valid fits)
- `outputs/fit_recipes_buildingnet/best_params.npz` (1556 successful fits; consumed by B+.6)
- `outputs/fit_recipes_buildingnet/visuals/fit_quality_sheet.png` (12-row diagnostic)

### Method
For each BuildingNet GT SDF:
1. Apply sparsity filter (drop if footprint < 100 cells OR iso=0 voxels < 300)
2. Extract polygon from footprint mask (~16 vertices via skimage contour)
3. For each of 8 styles, run 600 Adam steps minimizing:
   - Surface-band L1 (only voxels where `|GT_SDF| < 0.08` contribute)
   - + 0.1 × anchor L1 (clipped at -0.08, prevents drift)
4. Pick the style with highest footprint IoU
5. Save `(style, params, IoU, polygon, bbox)` per asset

Runtime: **40,310 seconds** (11.2 hours) on A100 = ~26s per asset attempted.

### Aggregate results

```
n_attempted:        1849
n_succeeded:        1556   (15.8% filtered as sparse/broken)
IoU mean:           0.656
IoU median:         0.741
IoU std:            0.272
IoU > 0.85:         ~28%   (high-fidelity fits)
IoU > 0.70:         53.5%
IoU > 0.50:         68.2%
IoU < 0.30:         ~14%   (recipe library can't represent)
```

### Style distribution of best fits

```
modern         1235  (79.4%)   ← heavy bias; most permissive recipe
victorian        99  (6.4%)
industrial       68  (4.4%)
public_civic     50  (3.2%)
contemporary     45  (2.9%)
craftsman        44  (2.8%)
colonial         15  (1.0%)
mediterranean     0  (0.0%)    ← never the best fit
```

### Why modern dominates

The modern style produces a body + parapet ring + optional small mech box. The parameter space (9 dims with continuous offsets) gives the optimizer the most freedom. For any tall flat-top building (hotels, museums, palaces, offices), modern provides the closest match with the most slack to fine-tune.

Other styles have **constrained constructions** (gable roof, dome, hip + spire, etc.) that hurt the fit unless the GT actually looks like that. So unless the GT screams "this is a Victorian with a tower", the optimizer falls back to modern.

### Why mediterranean never wins

Mediterranean only has 3 parameters: roof_h_ratio, eaves_expand_ratio, edge_band_h. With so little flexibility, it can't match any building better than the more-parameter-rich modern can. Effectively subsumed.

### Sparsity filter details

The 293 dropped assets (15.8%) failed the data quality gate:
- `n_fp < 100` (footprint mask has < 100 occupied cells): some assets are essentially just floating fragments
- `n_iso < 300` (SDF iso=0 has < 300 occupied voxels): non-watertight meshes with mostly-positive SDF

These assets' fits would have been meaningless (the optimizer can't fit to fragments). Sparsity stats at `outputs/buildingnet_sparsity_audit/sparsity.csv`.

### Per-class breakdown (rough)

From the CSV, IoU distribution by top-level class label:
- COMMERCIAL: mean IoU ~0.65 (offices, hotels — fit well via modern)
- RESIDENTIAL: mean IoU ~0.70 (houses with simpler geometry, often modern)
- RELIGIOUS: mean IoU ~0.55 (cathedrals + complex churches — recipe library struggles here)
- PUBLIC: mean IoU ~0.60 (mixed museums, schools, factories)

The "religious" and complex-public bucket is where the library has the biggest gap.

---

## Combined training dataset for B+.5/B+.6

| Source | Pairs | Cond → params mapping | Style balance |
|---|---|---|---|
| Real (B+.7) | 1,556 | grounded in BuildingNet GT meshes | 79% modern, rest sparse |
| Synthetic (B+.4) | 50,000 | exact recipe parameter from rng replay | balanced 6250/style |
| **Combined** | **51,556** | mixed | dominated by synthetic balance |

The synthetic data dominates by 32×, so the diffusion will primarily learn the synthetic param distribution. The real-data fits provide grounding — they teach the model "these are the recipe params that approximate real buildings under these conditioning inputs."

For B+.6 training, recommended mixing strategy:
- 70% synthetic, 30% real OR
- Class-conditional sampling: weight by inverse class frequency in the real data

---

## Known limitations from this curation

1. **Style imbalance.** Real fits are 79% modern. The diffusion will inherit this. Mitigations:
   - Sample real data with style-balancing during training
   - Use BuildingNet's class prefix as a **style hint** (religious → bias toward victorian/public_civic)
   - Add a style-classifier auxiliary loss

2. **Mediterranean / no-rng styles have zero variance.** All 6250 synthetic samples have identical params for victorian, industrial, mediterranean, public_civic. The diffusion can't learn diversity for these. Mitigations:
   - Add training-time jitter to no-rng styles' params (e.g., Gaussian noise ~ 5% of param scale)
   - Or just drop these styles from the synthetic set and rely on real fits for them

3. **Recipe library ceiling ~70% IoU.** Buildings with cathedrals, multi-tower churches, ornate residentials, complex industrials don't fit cleanly. Mitigations:
   - Accept the limitation; train on the 68% > 0.5 subset
   - Or extend the library (Stage 4 or later) with more primitives or a HF-NeuS displacement field (Option D)

4. **Sparsity filter dropped 15.8% of corpus.** Those are non-watertight meshes with broken SDFs. Not recoverable without re-preprocessing BuildingNet meshes. Acceptable.

---

## What's ready vs what's next

**Ready for consumption (Phase B+.5/B+.6):**
- `outputs/fit_recipes_buildingnet/best_params.npz` — real-data training pairs
- `data/recipe_augmentation_v1/extracted_params/*.npz` — synthetic-data training pairs
- `outputs/stage3_metadata/asset_dimensions.csv` — per-asset metadata for conditioning lookup

**Done since (2026-06-03):**
- ✅ **B+.5 deterministic head** — see new section below. Real held-out footprint IoU 0.617 vs 0.629 ceiling = 98% retention.
- ✅ **Synthetic conditioning recovered** — the gap noted earlier (synthetic npz only stored `params/seed/style_id`, no footprint/height) is closed. `scripts/recover_synthetic_conditioning.py` replays the worker rng stream to recover `(polygon, height_m, class)` exactly (footprint reproduce IoU 1.0). Output `outputs/recipe_param_dataset/synthetic_cond.npz`.

**Pending:**
- B+.6: recipe-parameter diffusion training (5-7 days on A100)
- B+.7-extension: if time permits, re-run with style-balanced loss or class-conditioned init to boost mediterranean / colonial coverage
- B+.8: FlexiCubes integration for differentiable mesh extraction (if joint mesh-loss training is needed in B+.9)
- B+.9: joint train on procedural ∪ real with StEik regularizer

---

## Phase B+.5 — deterministic head + synthetic folded in (DONE 2026-06-03)

**Scripts:** `scripts/recover_synthetic_conditioning.py`, `scripts/train_recipe_param_head.py`
**Shared modules:** `models/networks/recipe_param_space.py` (param-space contract + scale-invariant featurizer + scalers), `models/networks/recipe_param_head.py` (MLP)

### Scale-invariant conditioning (prerequisite for folding synthetic in)
Real B+.7 fits live in normalized Frame-N (~unit); synthetic recipe-aug lives in world meters (5–35 m). Absolute size features would split the two domains. The featurizer was made scale-invariant — normalized outline (30) + `{aspect, fill_ratio, compactness, slenderness=height/√area}` (4) + class one-hot (4) + style one-hot (8) = **COND_DIM 46**. This is valid because the recipe params are themselves ratios applied to the polygon+height passed to the recipe at forward time. Real-only IoU actually rose 96.0%→96.7% under the change.

### Synthetic conditioning recovery
The B+.4 npz only stored `(params, seed, style_id)`. The recovery replays `np.random.default_rng(seed)` → `sample_polygon(style, rng)` then `sample_height(class, rng)` — the *worker* stream, separate from the recipe-internal param stream B+.4 replayed, but seeded identically so they stay row-aligned. **Verified:** recovered `height_m` matches the stored h5 value to ~1e-6 for all 50k samples, and procedural footprint reproduces at IoU 1.0000 for all 8 styles. Output: `outputs/recipe_param_dataset/synthetic_cond.npz` (50000 rows, balanced 8×6250).

### Results
| Config | Pool | Real held-out footprint IoU | Retention of B+.7 ceiling | Style coverage |
|---|---|---|---|---|
| Real-only (ablation) | 1,323 | 0.608 | 96.7% | 7/8 (no mediterranean) |
| **Real + synthetic (canonical)** | **26,584** (real ×8 + 16k synth) | **0.617** | **98.1%** | **8/8** |
| Overfit memorization check | 1,556 | — (train norm MSE → 7e-4) | — | — |

Folding synthetic in **improved** the grounding metric (regularization) AND gave full 8-style coverage. On the dominant modern style the predicted footprint equals the fitted footprint near-exactly on held-out buildings (see `outputs/recipe_param_head_b5/head_iou_sheet.png`). Param MSE overstates the gap on rare styles (extreme optimizer params + few val samples); footprint IoU is the metric that matters and it holds.

### Still open for B+.6
- Zero-variance synthetic styles (victorian/industrial/mediterranean/public_civic) are delta distributions in param space — need train-time jitter so a generative model learns diversity.
- `train_recipe_param_head.py` mixing knobs: `--synthetic`, `--synth_cap_per_style`, `--real_repeat`.

---

## Quick-access summary numbers

If you read nothing else:

| Question | Answer |
|---|---|
| Can the recipe library represent real BuildingNet? | Yes for 68% of corpus; mean footprint IoU 0.66 |
| Is the real-data dataset big enough for training? | 1,556 real pairs + 50k synthetic, all with recovered conditioning |
| What's the biggest known problem? | Modern dominates real fits 79%; synthetic folding gives style balance |
| Does a deterministic head work? | Yes — B+.5 retains 98% of the fit IoU on held-out buildings |
| What's the next experiment? | Phase B+.6 — recipe-parameter diffusion (the generative head) |
| When will the generative model be ready? | ~2 weeks (B+.6 training) |
