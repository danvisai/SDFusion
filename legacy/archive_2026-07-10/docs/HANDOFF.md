# GenerativeTowns / SDFusion — Project Handoff

Self-contained reference for resuming this project in a fresh chat session.
Authoritative as of 2026-05-08. Supersedes earlier `PROJECT_STATUS.md`. The
`CODEX_PROGRESS_REPORT_2026-05-08.md` is the deepest blow-by-blow log; this
document is the lean version meant to bootstrap a new conversation without
re-importing the prior context.

---

## 0. End goal (what we are building)

**Input:** a 2D map containing multiple building footprints (vector polygons
from OSM, or eventually a stylized map image). Per-polygon optional class
+ height.

**Output:** a 3D mesh per footprint, scaled and placed at world coordinates,
composing a coherent urban scene. Use case is video-game procedural city
generation. Designer drops a map → gets a 3D layout.

**Strategy:** retrieval-first. Pull the closest real BuildingNet OBJ that
matches the footprint, place it at the polygon's world coords, and apply a
learned SDF-domain *residual* to nudge the building toward the input footprint
shape. Real OBJs preserve architectural detail (windows, gables, ornaments)
that any from-scratch 3D generator at our compute scale cannot match.

---

## 1. Where we are right now (working pipeline, end to end)

```
F0  OSM polygons     -->   {polygon, class, height}
                                 |
F1  FootprintEmbed   -->   256-d L2-normalized embedding  (TRAINED, ckpt_best.pth)
                                 |
                              kNN over BuildingNet train embeddings (INDEX BUILT)
                                 |
F2  retrieved id     -->   load OBJ_MODELS/<id>.obj  (1849 real meshes on disk)
                                 |
                                 +-- bbox-fit to input polygon  (smoke done; bbox-only, no rotation/contour)
                                 |
F3  SDFCorrection    -->   3D U-Net predicts latent residual (TRAINED, ckpt_best.pth - selected by L1)
                                 |
F4  apply residual   -->   corrected SDF (64^3)
                                 |
F5  mesh extract     -->   marching cubes OR MeshUDF on |corrected_SDF|
                                 |
F6  world placement  -->   scale + translate to world coords (NOT yet implemented)
F7  scene compose    -->   trimesh concat / SDF-union (NOT yet implemented)
```

| stage | status | files / artifacts |
|---|---|---|
| F0 OSM extractor | working | `scene/extract_osm.py` (smoke: 129 buildings from Lafayette IN bbox) |
| F1 embedding net | trained | `models/networks/retrieval/footprint_embed.py` + `Logs_GT/retrieval_footprint_full/ckpt_best.pth` |
| F1 kNN index | built | `data/BuildingNet_dataset_v0_1/retrieval_index/{train,val,test}_embeddings.npz` |
| F2 retrieval + bbox align | smoke done | `scripts/retrieval_visual_smoke.py`, `scripts/retrieval_alignment_smoke.py` |
| F3 SDFCorrection model | trained 30 epochs | `models/networks/sdf_residual_net.py` + `Logs_GT/sdf_residual_full/ckpt_best.pth` (and `ckpt_latest.pth`) |
| F4 corrected SDF | working | inline in `train_sdf_residual.py` and `scripts/eval_sdf_residual_meshes.py` |
| F5 mesh extraction | both methods integrated | MC via `skimage`; MeshUDF via `external/MeshUDF/` (compiled) |
| F6 world placement | NOT done | needs `scene/place.py` |
| F7 scene compose | NOT done | needs `scene/compose.py` (trimesh concat + optional SDF union) |
| Renders for all 1849 ids | partial 1361/1849 | `data/.../buildingnet_renders/` — fill-in is pending; previous pass hung |

---

## 2. Empirical results to date (the numbers that matter)

### Retrieval quality (epoch-1 best, val nearest-neighbor)
- top-class accuracy (5 top-level classes): **0.9465**
- top-subtype accuracy (53 sub-classes): **0.9465**
- training continued 30 epochs but val NN-class plateaued early; ckpt_best is from epoch 1.

### Retrieval visual alignment (3 val queries × top-2 neighbors, bbox-only fit)
- footprint IoU mean: **0.30** (range 0.09–0.52)
- raw IoU is a weak signal because BuildingNet footprints are wall-traces of hollow meshes. Visual contact-sheet inspection is the better arbiter (see `outputs/retrieval_visual_smoke/`).

### Correction-pair dataset (top-1 retrieval, full splits)
- train pairs: **1481**, val pairs: **187**, total ~1.2 GB
- retrieval similarity: median 0.999 (very high — most retrievals are same-subtype neighbors)
- residual L1 median: 0.27 (train), 0.25 (val) — non-trivial residuals, model has work to do
- residual outliers exist (max ~7.4); training should use robust losses

### SDF residual training (30 epochs, batch 8, base_channels 16, residual_clip 1.0)
| metric | source baseline | corrected (ckpt_best.pth, epoch 27 by L1) | corrected (ckpt_latest, epoch 30) |
|---|---|---|---|
| Val SDF L1 | 0.301 | **0.097** (3.1× better) | 0.102 |
| Val sign IoU | 0.155 | 0.131 | 0.179 |
| Val footprint IoU | 0.20 | 0.26 | 0.28 |

**Key finding:** residual reliably improves continuous SDF L1 by ~3× and footprint IoU by ~30%, but binary occupancy (sign IoU) is unstable across epochs. Best sign-IoU was at epoch 19 (0.252) — but the L1-best checkpoint missed that peak. Mixed-criterion checkpoint selection is needed.

### Mesh extraction (16 val examples, marching cubes from corrected SDF)
- corrected mesh fragmentation reduced vs aligned-source baseline on 9–10 of 16 examples
- corrected footprint IoU improved on 7–8 of 16
- MeshUDF on `|corrected_SDF|` not yet evaluated against MC on corrected SDF — open experiment

---

## 3. What's been ruled out (do not rebuild)

| approach | reason | evidence |
|---|---|---|
| from-scratch SDFusion training on 1480 buildings | too few samples for diffusion to converge to recognizable buildings; only blobs after 100K+ steps | empirical, prior runs |
| ControlNet finetune on top of frozen SD1.5 | photographic prior dominates; outputs drift to photo-realistic regardless of target style; 4 versions tried (photo / depth / oblique / height) | extensive smoke runs, archived under `legacy/` |
| Marching cubes on hollow 64³ SDFs as the *primary* output | fragments thin-shell into disconnected pieces; silhouette IoU 0.08–0.51 vs OBJ truth | `outputs/sdf_audit/` |
| Watertight repair (voxelize-fill) | erases architectural detail (windows, ornaments); the whole point of using OBJ-direct meshes was to *keep* detail | design decision, supported by audit |
| Full UDF pivot (retrain VQVAE on UDFs) | existing VQVAE already round-trips signed SDFs at 0.83–0.99 IoU; feeding UDFs to the same VQVAE drops to 0.20–0.83 — not worth a 12 h retrain | empirical, see `b4el8ikfw` task output |
| ARAP / biharmonic mesh deformation as the *primary* output path | libigl's `arap_precomputation` and `harmonic` both fail on BuildingNet's non-manifold topology even after `merge_vertices()` + degenerate-face removal | works on clean cube; fails on real meshes |

---

## 4. Open issues and ranked next steps

Listed by impact. Each has a concrete command.

### 4.1 Re-train residual with mixed-criterion checkpoint selection (HIGH impact, ~1 h)

`train_sdf_residual.py` has been updated to also save `ckpt_best_iou.pth` based on validation corrected sign IoU. The current run only has `ckpt_best.pth` (L1-selected). A re-run produces both checkpoints.

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python train_sdf_residual.py \
  --pair_root data/BuildingNet_dataset_v0_1/correction_pairs \
  --out_dir Logs_GT/sdf_residual_full_v2 \
  --epochs 30 \
  --batch_size 8 \
  --num_workers 4 \
  --base_channels 16 \
  --residual_clip 1.0 \
  --device cuda
```

### 4.2 Compare MeshUDF vs MC on corrected SDFs (MEDIUM impact, ~30 min)

The mesh-eval script currently uses MC only. MeshUDF is integrated; need to add a `--use_meshudf` flag or sibling script. Hollow corrected SDFs may extract better via MeshUDF-on-`|corrected|` than MC at iso=0.

Reference implementation pattern (already exists in `scripts/extraction_strategy_smoke.py` for source SDFs).

### 4.3 Top-K correction pairs (MEDIUM impact, ~1 h)

Currently top-1 only — residual model overfits to "near-identity" cases. Top-3 mining gives diverse retrieved-target gaps and acts as data augmentation.

```bash
./sdfusion/bin/python scripts/build_correction_pairs.py \
  --phase train --top_k 3 --out_dir data/.../correction_pairs_topk3
```

Then retrain residual on the larger pair set (3× data).

### 4.4 Improve retrieval alignment beyond bbox (MEDIUM impact, ~3 h)

Current `retrieval_alignment_smoke.py` does X/Z bbox-fit only. Add:
- 4-rotation search ({0°, 90°, 180°, 270°}) maximizing aligned-IoU
- Optionally aspect-ratio-preserving fit instead of axis-independent stretch

This will improve the F2 stage's footprint match before any SDF correction is applied.

### 4.5 Build the F6 world-placement + F7 scene-compose stages (HIGH impact, ~4 h)

Nothing connects retrieved building → multi-polygon scene yet. Need:
- `scene/place.py`: takes (mesh in Frame N, polygon area + centroid in Frame W, target height) → mesh in Frame W
- `scene/compose.py`: takes list of placed meshes → single OBJ via `trimesh.util.concatenate` (default), `trimesh.boolean.union` only on overlapping bboxes

### 4.6 Fill in remaining renders (LOW impact, ~30 min)

488 buildings still don't have OBJ-direct renders (the previous pass hung). Renders are needed for the eval discriminator, not for the inference path. Re-run with timeout/skip-on-failure:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/render_buildingnet_objfiles.py --phase all
```

(may need a per-mesh timeout wrapper if it hangs again — TBD which mesh causes it).

### 4.7 Future: image-based F0 (the designer-facing front-end) (LOW priority, 2–3 days)

Train a small UNet on synthetic (rendered OSM tile, OSM ground-truth mask) pairs to extract polygons + classes from raster maps. Documented in `HYBRID_PIPELINE_PLAN.md` §10.

---

## 5. File index

### Active code (not legacy)
```
data/BuildingNet_dataset_v0_1/                      # 1849-building v1 dataset (24 GB OBJs + 18 GB PLYs + 1.3 GB SDFs + derived)
├── OBJ_MODELS/<id>.obj + .mtl + textures           # real meshes, hollow but detailed
├── POINT_CLOUDS/<id>.ply                           # 100K-point clouds
├── resolution_64/<id>/ori_sample_grid.h5           # SDF + UDF + footprint
├── footprints_png/{train,val,test}/<id>.png        # binary 64×64 silhouettes (axis-correct)
├── buildingnet_heights/{train,val,test}/<id>.png   # top-down height maps
├── buildingnet_depths/{train,val,test}/<id>.png    # front-axis depth
├── buildingnet_renders/{train,val,test}/<id>.png   # OBJ-direct ortho renders (1361/1849, fill-in pending)
├── splits/{train,val,test}_split.txt               # active: 1480/186/180 (v1)
├── splits_v1_official/                             # backup of v1 official (1481/187/181)
├── splits_custom_v0_1_filtered/                    # backup of pre-v1 filter
├── splits_pre_v1promote_bak/                       # backup before v1 promotion
├── retrieval_index/                                # train/val/test embeddings + metadata
├── correction_pairs/{train,val}/pairs/*.npz        # 1481 + 187 (source, target, residual) tuples
└── 3DWarehouse_IDs/                                # provenance metadata

models/
├── networks/retrieval/footprint_embed.py           # F1 — small CNN encoder
├── networks/sdf_residual_net.py                    # F3 — small 3D U-Net residual predictor
├── networks/diffusion_networks/                    # legacy SDFusion UNet (kept for reference)
├── networks/vqvae_networks/                        # frozen VQVAE encoder/decoder
├── sdfusion_model_img2shape.py                     # legacy SDFusion model class (kept)
└── arap_deformer.py                                # ARAP attempt (now demoted, kept for reference)

datasets/
├── buildingnet_dataset.py                          # base loader
├── buildingnet_retrieval_dataset.py                # F1 training
├── correction_pair_dataset.py                      # F3 training
└── (legacy: building_fp2shape_dataset.py, etc.)

train.py                                            # legacy SDFusion trainer
train_retrieval.py                                  # F1
train_sdf_residual.py                               # F3 (saves ckpt_best.pth by L1, ckpt_best_iou.pth by sign-IoU after recent update)

scripts/
├── extract_osm.py                                  # F0
├── retrieval_smoke.py                              # F1 sanity
├── retrieval_visual_smoke.py                       # F2 visual contact sheets
├── retrieval_alignment_smoke.py                    # F2 bbox-fit smoke
├── build_retrieval_index.py                        # F1 → kNN index
├── build_correction_pairs.py                       # F3 training data
├── eval_sdf_residual.py                            # F3 visual eval
├── eval_sdf_residual_meshes.py                     # F3 mesh eval (MC only currently)
├── extraction_strategy_smoke.py                    # MC vs MeshUDF on source SDFs
├── render_buildingnet_objfiles.py                  # OBJ-direct renderer (replaces broken SDF-MC version)
├── render_buildingnet_heightmaps.py                # top-down height maps
├── render_buildingnet_depthviews.py                # front-axis depth maps
├── recompute_footprints_from_sdf.py                # axis-correct footprint computation
├── filter_low_inside_splits.py                     # inside%-filter for splits
└── make_val_split.py                               # original val carve

scene/
└── extract_osm.py                                  # F0 (also a copy)

external/
├── Hunyuan3D-2/                                    # frozen 3B image-to-3D (archived as comparison baseline; not on active path)
├── MeshUDF/                                        # ECCV 2022 implementation, custom_mc compiled
├── hf_cache/                                       # SD1.5 + Hunyuan3D-2 weights (~13 GB)
└── _gdown_home/                                    # gdown's cookies cache (HOME redirect)

logs_building/
└── 2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth   # FROZEN VQVAE

Logs_GT/
├── retrieval_footprint_full/                       # F1 — ckpt_best.pth (epoch 1, val_nn=0.9465)
└── sdf_residual_full/                              # F3 — ckpt_best.pth (L1) + ckpt_latest.pth

outputs/
├── retrieval_visual_smoke/                         # 6 val contact sheets (HUMAN-INSPECT THESE NEXT)
├── retrieval_alignment_smoke/                      # 3 val alignment sheets + fitted OBJs
├── correction_pairs_smoke/val/                     # 3-id correction-pair sanity
├── sdf_residual_eval_best_l1/val/                  # 16-example visual eval
├── sdf_residual_mesh_eval_best_l1/val/             # 16-example mesh eval (MC only)
├── sdf_residual_mesh_eval_latest/val/              # 16-example with latest ckpt
└── extraction_smoke_meshudf/                       # MC vs MeshUDF on 4 source SDFs

legacy/                                             # 1.3 TB of obsolete experiments (CN runs, old SDFusion training, smoke outputs). Do not depend on.

docs/
├── HANDOFF.md                                      # THIS DOCUMENT
├── HYBRID_PIPELINE_PLAN.md                         # original plan (v1, partly outdated)
├── PROJECT_STATUS.md                               # earliest status (very stale)
└── CODEX_PROGRESS_REPORT_2026-05-08.md             # blow-by-blow log
```

---

## 6. Environment quirks (Gilbreth-specific)

- **Always strip XALT preload** for any Python that touches CUDA:
  ```bash
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python ...
  ```
- **Home-quota constraint**: don't write data/cache files to `/home/dsimhadr/`. Redirect cache via:
  ```bash
  env -u LD_PRELOAD -u LD_LIBRARY_PATH HOME="$(pwd)/external/_gdown_home" ./sdfusion/bin/<tool> ...
  ```
- **`/home/.local` is at 15 GB** — probably stale pip --user installs. Worth cleaning periodically (separate task).
- **GPU**: 1× A100 80GB. ~10 GB used by the residual training; plenty of headroom.
- **Splits text files lack trailing newlines** — `wc -l` undercounts by 1. Use Python `[ln.strip() for ln in f if ln.strip()]` for reliable counts (1481/187/181).
- **Bash background tasks may hang silently** — recent OBJ-rendering attempt sat for 20 min producing 0 output. Wrap long batch jobs with per-iteration timeouts.

---

## 7. Decisions log (for new chat: don't relitigate)

| decision | choice | rationale |
|---|---|---|
| Splits in use | v1 official 1480/186/180 | promoted from `splits_v1_official/`, captured in `splits_pre_v1promote_bak/` |
| Latent representation | signed SDF (existing VQVAE) | empirical: VQVAE round-trips SDFs at 0.83–0.99 IoU; UDFs drop to 0.20–0.83 |
| Output extraction | MC primary, MeshUDF as comparison | both integrated; MeshUDF reduces fragmentation but doesn't fix the 64³ detail ceiling |
| Mesh deformation (ARAP/biharmonic) | abandoned as primary | libigl fails on BuildingNet non-manifold meshes |
| Final-output strategy | retrieved OBJ + corrected SDF as guidance | preserves architectural detail, residual aids alignment |
| Watertight repair | NOT done | erases detail; we want hollow buildings |
| VQVAE retrain on UDFs | NOT done | not worth the 12 h cost given current SDF round-trip quality |
| Hunyuan3D-2 | demoted to baseline | not on active path; kept under `external/` for evaluation comparison |
| Class taxonomy | 5 top-level for retrieval filter, 53 subtypes for embedding aux CE | matches the model_id prefix structure of BuildingNet |
| Footprint conditioning convention | `(D=z, H=y, W=x)`, footprint placed on (D, W) | empirically verified via VQVAE encode→decode IoU axis check |
| Frame N (mesh-frame) | unit-sphere centered, scale = max-extent | consistent with `preprocess/create_sdf.py` |
| `bake_footprint` | NOT used; replaced by `recompute_footprints_from_sdf.py` | original projects all faces incl. roofs; broken |

---

## 8. To verify on resume (one-shot health check)

```bash
cd /scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion

# 1) GPU and env
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -c "
import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 2) Data state
echo 'splits:' && for s in train val test; do echo -n "  $s: "; wc -l < data/BuildingNet_dataset_v0_1/splits/${s}_split.txt; done
echo 'OBJs:' && ls data/BuildingNet_dataset_v0_1/OBJ_MODELS/*.obj | wc -l
echo 'pairs train:' && ls data/BuildingNet_dataset_v0_1/correction_pairs/train/pairs/ | wc -l
echo 'pairs val:'   && ls data/BuildingNet_dataset_v0_1/correction_pairs/val/pairs/ | wc -l

# 3) Models
ls -lh Logs_GT/retrieval_footprint_full/ckpt_best.pth Logs_GT/sdf_residual_full/ckpt_best.pth \
       logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth

# 4) Quick retrieval smoke (~30 s)
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/retrieval_smoke.py 2>&1 | head -20
```

If all of the above return the expected values, you're at the same checkpoint
described here and can pick from §4 directly.

---

## 9. The "first move" if starting a new chat right now

Concrete suggestion, in order:

1. **Inspect the 6 retrieval visual contact sheets and 3 alignment sheets** under
   `outputs/retrieval_visual_smoke/` and `outputs/retrieval_alignment_smoke/`.
   These determine whether retrieval is good enough to ship as-is or needs
   improvement.
2. **Re-run residual training to save `ckpt_best_iou.pth`** (§4.1 — 1 h).
3. **Compare MC vs MeshUDF on corrected SDFs** (§4.2 — 30 min).
4. Based on those results, decide whether to:
   - Improve retrieval alignment (§4.4)
   - Build F6/F7 (§4.5) — most impactful for actual demo
   - Build top-k correction pairs (§4.3) for stronger residual training

---

## 10. Bottom-line one-liner

A working retrieval-first pipeline exists end-to-end except for world placement
and scene composition (~1 day's work). Residual model improves SDF L1 by 3× and
footprint IoU by ~30%, but binary occupancy is unstable across epochs — fixed
by saving an IoU-criterion checkpoint on the next training run. The remaining
engineering is integration glue (F6, F7), not new ML research.
