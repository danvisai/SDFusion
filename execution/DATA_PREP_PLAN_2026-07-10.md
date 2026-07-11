# Data preparation plan for the massing-scale-decomposition proof (2026-07-10, rev2)

Feeds `execution/RESEARCH_PROOF_PLAN_2026-07-10.md` Step 0 and `execution/IMPLEMENTATION_PLAN_2026-07-10.md`
M0/M1/M5. Written after a full disk audit of `data/`, `external/`, and a scratch+home sweep.
Vocabulary: `CONTEXT.md`.

## 1. Inventory — what exists, is it USED, by what

| Dataset | Size | Content | Used by | Verdict |
|---|---|---|---|---|
| `data/BuildingNet_dataset_v0_1` | 386 G | 1,849 real labeled buildings; `resolution_64/*/ori_sample_grid.h5` real SDFs | Stage 3a; element library; **real detail pairs** | **USED — core** (only *detail-bearing* real corpus) |
| `data/real_massing_v1/real.h5` | 31 G | 35k LoD2 **massing** (NL+DE+JP) | Stage 3a massing | **USED** (massing only — no facades) |
| ├ `nrw.h5` (DE) / `plateau.h5` (JP) | 11 G / 8.6 G | LoD2 massing ingredients | `concat_real_massing.py` → real.h5 | **USED** (via concat) |
| `data/bag3d_v1/bag3d.h5` (+labels) | 12 G | NL LoD2 massing + labels | massing (NL part) | **USED** |
| `data/recipe_augmentation_v1` | 15 G | 50k synthetic massing+params | recipe-param diffusion | **USED** |
| `data/detail_pairs_v1/pairs*.h5` | 11 G | **SYNTHETIC** (coarse↔**composer**-detailed) pairs, random footprints, **no BuildingNet ids** | old `train_detailizer.py` | **NOT a valid monolith baseline** — it trained a net to imitate the composer. Superseded by REAL pairs (M1). Keep only as a documented negative. |
| `data/part_structure/*.npz` | <1 M | BuildingNet + LoD3 part instances | planner / refiner | **USED** |
| `data/lod3_tum` | 46 M | German **LoD3** — real facade windows/doors | refiner, part pairs | **USED for refiner — NOT in element library** ⇒ the "+real element data" ablation source (M5.2) |
| `data/element_library_v1` | 677 M | 3,204 real element SDF crops | retrieval | **USED but LEAKY** (built from all 1,849 incl. would-be test) → must rebuild (M5.1) |
| `data/ornaments_v1` | 158 M | 3 heritage scans | ornament retrieval | USED (demo wrapper) |
| `data/ShapeNet` | 464 M | legacy SDFusion | — | legacy, not GenerativeTowns |
| `*_smoke.h5` | small | CI subsets | test suites | keep |
| **`external/buildingnet_official_v1`** | **9.3 G** | raw BuildingNet download (`OBJ_MODELS.zip`, `file*.zip`) | **NO code references it** | **UNUSED / redundant** → cleanup candidate |
| `external/buildingnet_official` | 4 K | empty stub | — | delete |

**Correction to the "Japanese elements" memory:** no Japanese *element* data exists on disk;
`plateau.h5` is LoD2 *massing*. The genuinely under-used *element* source is German `lod3_tum`.

## 2. The key structural fact: massing data is plentiful, detail data is scarce
- **Massing regime:** ~35k real LoD2 masses + 50k synthetic + BuildingNet. Plenty → Stage 3a massing
  stays at **full** data for both arms (not the contested variable).
- **Detail regime:** only ~1,849 BuildingNet buildings carry real facade *detail*. `real.h5`/`plateau.h5`
  are facade-less and **cannot** supply detail ground truth or FID-real.

This asymmetry **is the thesis in the data itself**. Two consequences:
1. The **detail** data-scaling fractions (25/50/100 %) are fractions of **BuildingNet** (the headline
   corpus). `lod3_tum` is NOT in the fractions — it enters only as the M5.2 "+real element data" ablation.
2. **FID-real** and the **monolith's detail target** come only from BuildingNet.

## 3. Prep tasks (in order)

**P1 — Frozen held-out real TEST split (blocks everything).**
- From BuildingNet, hold out a fixed ~15 % test set, stratified by class, sealed as
  `data/splits_v1/test_ids.json`. Never enters any training set OR any element library.
- Per test building derive: footprint (ground projection) + height (massing target) + neutral-shaded
  real facade renders (FID-real). Reuse the element-library OBJ parser.

**P2 — Deterministic training fractions 25/50/100 % (BuildingNet detail).**
- Nested (25 ⊂ 50 ⊂ 100), seeded id lists `data/splits_v1/train_{25,50,100}.json` from the train
  remainder. **Both arms consume the SAME list per fraction (D2 equal-data):**
  - **Monolith:** REAL pairs built from `train_X` (see M1 `make_real_detail_pairs.py`) — NOT the
    synthetic `detail_pairs_v1`.
  - **Decomposition:** element library **rebuilt from `train_X` ids** (per-fraction). Stage 3a massing
    stays full.

**P3 — Per-fraction, leakage-safe element libraries.**
- Extend `build_element_library.py` to accept `--include-ids` / `--exclude-ids`. Build
  `element_library_train{25,50,100}` (each from its fraction, all excluding test ids). The shipped
  `element_library_v1` is leaky (built from all 1,849) — do not use it for the experiment.
- (Ablation, M5.2) a `+lod3_tum` variant to isolate the real-non-BuildingNet-detail delta.

**P-C1 — C1 (transform) data — mostly derivable/existing, minimal prep.**
- From-noise samples: none needed — sample the live Stage 3a prior directly.
- Blockout inputs: footprint-extrude of the P1 test footprints (derivable, no new data).
- Sculpt-sweep cases: a small hand-authored set of (real building, crude edit) pairs (~8–12).
- Residual evidence: already trained (`Logs_GT/sdf_residual_full_v4_aug_topk3`, `CorrectionPairDataset`
  over BuildingNet) — no new data; just read the metrics.

**P4 — (ordering marker) FID/render harness + monolith runs** — see IMPLEMENTATION M0.2/M1; needs P1–P3.

## 4. Cleanup (separate from training prep)
- `external/buildingnet_official_v1` (9.3 G) + `external/buildingnet_official` (stub): after P1 verifies
  the extracted `data/BuildingNet_dataset_v0_1` is complete, safe to delete/archive. **Not before.**

## 5. Open decisions (recommended defaults, confirm before running)
1. **Test-split axis:** in-distribution random-stratified (recommended) vs cross-culture holdout
   (train BuildingNet → test `lod3_tum`). → default in-distribution; cross-culture as an optional figure.
2. **`lod3_tum` enrichment timing:** default = BuildingNet-only for the headline, add `lod3_tum` as the
   M5.2 "+real element data" ablation delta (isolates the lever).
3. **Coarse-side derivation for the real pairs (M1.1):** low-pass the real SDF vs footprint-extrude vs
   both as monolith inputs. → default: low-pass (keeps the pair aligned to the same building).
