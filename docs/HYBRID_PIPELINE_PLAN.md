# Hybrid Retrieval + SDF-Correction Pipeline — Implementation Plan

## 0. Naming and conventions used throughout

- **Frame A (mesh frame).** Native OBJ coordinates as shipped in `OBJ_MODELS/<id>.obj`. Per-mesh, arbitrary scale.
- **Frame N (normalized).** Centered + isotropically scaled so the mesh fits in the BuildingNet `[-1, 1]^3` unit-sphere-equivalent box used by `preprocess/create_sdf.py`. SDFs at `resolution_64/<id>/ori_sample_grid.h5` live here. Voxel layout `(D=z, H=y, W=x)`, Y up.
- **Frame W (world).** Metric meters in the input map. Y is up in N, but the map coordinates we'll see are XZ in W (footprint plane) plus a `height` scalar.
- All neural inputs use Frame N. Frame W↔Frame N transformation happens only at compose time.
- Class label is parsed from the `<CLASS><subtype>_mesh<id>` filename prefix. 53 distinct prefixes total (5 top-level × variable subtypes). Use the full prefix as a 53-way category id.
- "Retrieved building" means the candidate selected by k-NN over a footprint+class embedding. Always referred to by its model id.
- "Residual" means a 3D tensor in latent space (`(3, 16, 16, 16)`) added to the VQVAE-encoded retrieved latent before VQVAE decode.

---

## 1. System architecture

### 1.0 Map input (F0) — two routes

```
ROUTE A (immediate, vector-based, 1 h to implement):
  OSM tile / GeoJSON / shapefile  ──[osmnx parser]──▶  list of {polygon, class, height}

ROUTE B (future, for raster map images):
  rendered map screenshot (PNG)   ──[map-segmenter]──▶  per-class semantic mask
                                                       (building / road / water / park / bg)
                                  ──[contour fit]──▶  list of {polygon, class, height_estimate}
```

The hybrid pipeline downstream consumes the same `{polygon, class, height}` triple regardless of which route produced it. Route A is the production path for now and the only path needed to demo the retrieval+correction stages. Route B is a real-but-future ML problem (see §10).

### 1.1 Training time

```
                              [offline, one-time per id]
  OBJ_MODELS/<id>.obj  --->  watertight repair  ---> repaired mesh (OBJ)
                                                       |
                                                       v
                                       create_sdf  ---> SDF_64^3  (Frame N)
                                                       |
                                                       v
                                  recompute_footprints --> footprint_64^2 (Frame N, Y-collapsed)
                                                       |
                                                       v
                          render_buildingnet_objfiles --> ortho RGB 512^2 (3/4 view)
                                                       |
                                                       v
              VQVAE.encode(SDF) -> latent_z_target  shape (3, 16, 16, 16)

   For each train id i:
      footprint_i + class_i --[FootprintEmbedNet]--> emb_i  shape (256,)

   Build retrieval index over {emb_j : j in train\{i}}.
   For each i:
      candidate_j = argmin_j  d(emb_i, emb_j)   excluding j==i, optionally restricted to
                                                same top-level class
      latent_z_retrieved_j = VQVAE.encode(SDF_j)
      target_residual_i = latent_z_target_i - latent_z_retrieved_j

   Train SDFCorrectionNet:
      input  = (latent_z_retrieved_j, footprint_i_3D, class_id_i, height_map_i)
      output = predicted_residual_i  shape (3, 16, 16, 16)
      loss   = MSE(predicted_residual_i, target_residual_i)  + voxel-domain consistency loss
```

### 1.2 Inference time (single footprint)

Output strategy: deform the retrieved OBJ guided by the corrected SDF (ARAP). No marching cubes — preserves all architectural detail (windows, gables, ornaments) of the retrieved BuildingNet mesh.

```
input footprint (HxW binary) + class_id + target_height
        |
        +--> FootprintEmbedNet --> query_emb (256,)
        |
        +--> kNN over training_set embeddings, top-K=5 (class-filtered)
        |
        +--> select retrieved_id (best match by composite score)
        |
        +--> retrieved_OBJ = load OBJ_MODELS/<retrieved_id>.obj          # full detail mesh, hollow
        +--> VQVAE.encode(retrieved_SDF) --> z_retrieved (3,16,16,16)
        +--> _build_fp3d_for(input_footprint) --> fp3d (1,16,16,16)
        +--> SDFCorrectionNet(z_retrieved, fp3d, class_id, height_map) --> z_residual
        |
        +--> z_corrected = z_retrieved + alpha * z_residual
        +--> VQVAE.decode_no_quant(z_corrected) --> SDF_corrected (1,1,64,64,64) in Frame N
        |
        +--> ARAPDeformer(retrieved_OBJ, SDF_corrected, input_footprint):
                target_positions = sample SDF iso-surface near each boundary vertex
                solve ARAP energy: minimize Σ||(R_i (v_j - v_i)) - (v'_j - v'_i)||²
                                   s.t. boundary vertices match target_positions
                                   (libigl arap solver, ~10 iters, ~100-500 ms/mesh)
                                   |
                                   v
                deformed_OBJ = retrieved_OBJ with vertices displaced toward
                               corrected SDF surface, interior topology preserved
        |
        +--> Frame N -> Frame W transform:
                         scale = sqrt(footprint_area_world / footprint_area_N)
                         translate = world_centroid
                         lift Y so ground plane sits at world Z=0
        |
        v
     deformed mesh in Frame W (one OBJ per polygon, full BuildingNet detail intact)
```

### 1.3 Inference time (multi-polygon map)

```
map (vector polygons + class + height per polygon)
        |
        +--> for each polygon p: run 1.2 in parallel -> mesh_W_p
        |
        +--> compose:
                Option A (mesh-merge): trimesh union of all mesh_W_p, optional CSG cleanup.
                Option B (SDF-union):  rasterize all per-polygon SDFs into a coarse global
                                       SDF at world resolution K^3 (K~512 for a 200m town);
                                       elementwise min; marching cubes once.
                Default to Option A unless polygons overlap (rare for footprints).
        |
        v
  composite town mesh (single OBJ)
```

### 1.4 Tensor shapes / data types

| Stage                | Shape                | dtype     |
| -------------------- | -------------------- | --------- |
| OBJ vertices         | varies               | float32   |
| Footprint PNG / h5   | (H,W) or (1,64,64)   | uint8     |
| SDF h5 dataset       | (262144, 1)          | float32   |
| SDF reshaped         | (1, 64, 64, 64)      | float32   |
| VQVAE latent z       | (3, 16, 16, 16)      | float32   |
| Footprint vol fp3d   | (1, 16, 16, 16)      | float32   |
| Class id             | scalar long          | int64     |
| Class embed          | (32,)                | float32   |
| Height map           | (1, 64, 64)          | float32   |
| FootprintEmbed out   | (256,)               | float32   |
| Residual prediction  | (3, 16, 16, 16)      | float32   |

---

## 2. Components to train

### 2.1 FootprintEmbedNet (small CNN with class head)

- **Role.** Map (footprint mask, class id) → 256-d embedding for kNN retrieval.
- **Input.** `footprint` `(B, 1, 64, 64)` float in {0,1}; `class_id` `(B,)` long.
- **Output.** `embedding` `(B, 256)` L2-normalized.
- **Architecture.** 5-block 2D conv (32→64→128→256→256, stride-2 each) + global avg pool + FC(256). Class id passed through `nn.Embedding(num_classes=53, dim=32)` and concatenated to pooled features before the final FC. Roughly 1.2 M params.
- **Loss.**
  - Triplet loss with semi-hard mining over (anchor=footprint_i, positive=footprint_j_same_class_with_high_IoU, negative=random_other_class).
  - Auxiliary cross-entropy classifier head on `class_id` to keep the embedding class-aware. Total loss = triplet + 0.1 × CE.
  - Define footprint similarity for positive mining as `IoU(fp_i, best_alignment(fp_j))` where best alignment is over rotations of {0, 90, 180, 270} and a 1-D scale ratio (greedy, on 64×64 masks).
- **Training time.** A100, batch 256, ~30 epochs over 1480 samples × 4 rotation augmentations ≈ 5920 effective samples. Should converge in 40–60 minutes.
- **File path (new).** `models/networks/retrieval/footprint_embed.py`. New trainer at `train_retrieval.py`. Existing `datasets/buildingnet_dataset.py` can be reused with a minimal wrapper that returns `(fp, class_id)` only.
- **Reuse.** The existing CLIP image encoder in `models/networks/clip_networks/network.py` is overkill (110 M params, photo-domain prior). Build a small bespoke CNN as above.

### 2.2 SDFCorrectionNet (3D residual UNet operating in VQVAE latent space)

- **Role.** Given a retrieved building's latent + the desired footprint + class + height, predict the latent residual that morphs the retrieved building into one whose decoded SDF respects the input footprint and matches class statistics.
- **Input.**
  - `z_retrieved` `(B, 3, 16, 16, 16)` from frozen VQVAE.encode of retrieved SDF
  - `fp3d_target` `(B, 1, 16, 16, 16)` from `_build_fp3d_for` on the target footprint
  - `class_id` `(B,)` long → embedded to `(B, 64)` and broadcast into spatial conditioning
  - `height_map` `(B, 1, 64, 64)` resized to `(B, 1, 16, 16)` and broadcast along the depth axis as a second `c_concat` channel
- **Concatenated input volume.** `(B, 5, 16, 16, 16)` = `[z_retrieved (3) | fp3d_target (1) | height_vol (1)]`.
- **Conditioning vector for cross-attn.** class embedding `(B, 64)` projected to 768.
- **Output.** `z_residual` `(B, 3, 16, 16, 16)`.
- **Architecture.** Reuse `models/networks/diffusion_networks/network.py` with `in_channels=5`, `out_channels=3`, `dims=3`. Strip the time-embedding path: this is a deterministic residual predictor, not a diffusion model. Total ≈ 35 M params (same as current SDFusion UNet body).
- **Loss.**
  - `L_latent = MSE(z_residual_pred, z_residual_gt)` where `z_residual_gt = VQVAE.encode(SDF_target) - VQVAE.encode(SDF_retrieved)`. Use the existing `scale_factor=2.380615` consistently inside the residual.
  - `L_decode = MSE(VQVAE.decode_no_quant(z_retrieved + z_residual_pred), SDF_target)` computed every K steps (decode is ~60 ms, do it at every step for smaller batch). Weight 0.5.
  - `L_silh = BCE(silhouette(decoded_SDF), input_footprint_64)` to anchor the footprint constraint. Weight 0.25.
  - Total: `L = L_latent + 0.5 * L_decode + 0.25 * L_silh`.
- **Training time.** A100, batch 8, ~30 k steps. Latent-space MSE converges fast since the residual is small. Expect 4–6 hours including the auxiliary decode loss. Possibly less if `L_decode` is downsampled to every-other step.
- **File path (new).** `models/sdfusion_correction_model.py` (new), wraps the existing UNet. Trainer extension in `train.py` controlled by a new flag `--task correction`. Config at `configs/sdfusion_correction.yaml` (copy of `sdfusion-img2shape.yaml`, set `in_channels=5`, drop diffusion params).
- **Reuse.**
  - Existing VQVAE checkpoint `logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth` is the encoder/decoder. Frozen.
  - Existing UNet body `models/networks/diffusion_networks/network.py` (`DiffusionUNet`).
  - `_build_fp3d_for` from `models/sdfusion_model_img2shape.py:221-237` (axis-correct).
  - `scale_factor` from `configs/sdfusion-img2shape.yaml` (`2.380615`).
  - `models/model_utils.py:load_vqvae`.

### 2.3 ARAPDeformer (mesh-domain morph guided by corrected SDF)

- **Role.** Take a retrieved BuildingNet mesh and the corrected SDF, displace the mesh's exterior vertices to land on the corrected SDF iso-surface while keeping the interior topology rigid. Preserves all architectural detail (windows, ornaments, eaves) that marching cubes would have lost.
- **Inputs.**
  - `retrieved_mesh`: trimesh with V vertices (typically 1k–240k), F faces, hollow exterior shell
  - `corrected_sdf`: float32 `(D, H, W)` = `(64, 64, 64)` in Frame N
- **Output.** `deformed_mesh` with same topology (same V, F count) but displaced positions.
- **Algorithm.** As-Rigid-As-Possible (Sorkine & Alexa 2007), libigl's `igl.arap` reference implementation:
  1. Identify "anchor" vertices = those near the SDF iso-surface (within 1 voxel = `2/64` units in Frame N).
  2. Compute target positions for anchors by gradient-walking each anchor along `∇sdf` toward `sdf=0`.
  3. Solve global ARAP energy: minimize Σᵢ minᵣ Σⱼ∈N(i) ‖Rᵢ(vⱼ-vᵢ) - (v'ⱼ-v'ᵢ)‖² subject to anchor constraints. Alternates rigid-rotation per cell + global Laplacian solve.
  4. Iterate ~10× to convergence.
- **Cost per inference.** ~100 ms for small meshes (5k verts), ~500–1000 ms for large meshes (100k+ verts). Acceptable; can be parallelized across polygons in a multi-building scene.
- **Failure modes & mitigations.**
  - **Tangling** on thin features (e.g. spires): cap displacement magnitude per vertex at 0.3 (Frame N units); skip vertices whose proposed move would create flipped triangles.
  - **Insufficient anchors** (mesh has no vertex near iso-surface): fall back to iterative Laplacian smoothing + repeat anchor identification at coarser threshold.
  - **Very large meshes** (240k verts): subsample anchors to 5k for the ARAP solve, propagate to remaining vertices via barycentric interpolation.
- **Implementation.**
  - Use `libigl` Python bindings: `pip install libigl`.
  - File: `models/arap_deformer.py` (new). ~200 LOC.
  - No training required; classical algorithm with hyperparameters (anchor threshold, displacement cap, iteration count) tuned via the smoke step.
- **Why directly Option 2 (ARAP) instead of Option 1 (closed-form per-vertex):**
  - Closed-form move-along-gradient tangles on thin features (spires, chimneys) and creates flipped triangles.
  - ARAP enforces local rigidity, keeping interior structure coherent.
  - libigl's solver is mature and battle-tested.
  - Cost difference is small (100ms vs 500ms) compared to total per-building inference.

### 2.4 Optional: ClassDiscriminativeHead

- **Role.** A small classifier on rendered output that runs at evaluation time to score "does the corrected mesh still look like its target class?"
- **Input.** Single 3/4 ortho render at 224×224.
- **Output.** logits over 53 classes.
- **Architecture.** ResNet-18 from `models/networks/resnet_v1.py` finetuned on `buildingnet_renders/` for the 53-class task.
- **Loss.** Cross-entropy.
- **Training time.** ~30 minutes.
- **Use.** Eval-only. Not on the inference path. Provides one of the metrics in §6.
- **File path (new).** `eval/class_discriminator.py`.

### 2.5 Out of scope (do not build)

- A graph CNN learned-deformer over retrieved mesh vertices. ARAP (§2.3) is a deterministic algorithm with no learned weights; we keep the *learned* novelty concentrated in SDFCorrectionNet, which is also where the class-conditional intelligence lives.
- A second diffusion model. The deterministic residual predictor in §2.2 is the novelty; adding diffusion noise on top buys nothing for the mesh-quality argument and reintroduces the convergence risk that ruled out from-scratch SDFusion training.
- **Watertight mesh repair.** Originally proposed but dropped: BuildingNet meshes are intentionally hollow (walls + roofs only) and that's actually what we want for high-detail building output. The pipeline now uses ARAP on the original hollow mesh, so watertight repair would only erase architectural detail without any benefit.

---

## 3. Data-side decisions to make BEFORE training

### 3.1 SDF resolution: stay at 64³

- 64³ is what the trained VQVAE expects, latent shape `(3,16,16,16)`.
- Hollow SDFs are fine for our purposes: SDFCorrectionNet operates entirely in latent space, and the corrected SDF is used only as a **deformation guide** (ARAP), never decoded back to a mesh via marching cubes.
- 128³ deferred to stretch goals (§8); only matters if the residual model itself struggles to express fine deformations.

### 3.2 Splits: use the full v1 set (1480 / 186 / 180), not the inside%-filtered 1361

- Promote `splits_v1_official/*.txt` to `splits/*.txt` directly (1480 / 186 / 180 = 1849 ids).
- No further filtering needed: hollow meshes are now first-class inputs (we don't filter for inside%), and every v1 id has a real .obj file with proper exterior detail.

### 3.3 Constructing (retrieved, target) pairs

For each train id `i`:
- Compute `emb_i = FootprintEmbedNet(fp_i, class_i)`.
- Build a precomputed neighbor matrix: for each `i`, find top-K=8 nearest neighbors over the rest of the train split, restricted to **same top-level class** (RESIDENTIAL/RELIGIOUS/COMMERCIAL/MILITARY/PUBLIC).
- During SDFCorrectionNet training, sample one neighbor `j ∈ topK(i)` uniformly at random per epoch as the "retrieved" example. Sampling within top-K (rather than always top-1) acts as data augmentation and prevents overfitting to the closest-neighbor distribution.
- Cache neighbor lists at `data/BuildingNet_dataset_v0_1/retrieval_index/topk_train.npy` (shape `(N_train, K)`).

### 3.4 Class handling: hard filter at inference, soft signal at training

- **Retrieval at inference.** Hard filter to same top-level class (5-way). Inside the class, rank by L2 distance over embeddings.
- **Embedding training.** Auxiliary CE loss (§2.1) keeps embeddings class-aware. Triplet mining uses cross-class negatives.
- **Why hard at inference.** Mixing classes (e.g. retrieving a CHURCH for a HOUSE polygon) defeats the "retrieval inherits clean class-typical geometry" argument. The residual model can adapt small footprint variation but not the "is this a steeple or a chimney" question.
- **Why soft at training.** Some sub-types (e.g. RESIDENTIALhouse vs COMMERCIALhouse with 752 vs 9 instances) are sparse and rigid filtering would starve them. The CE auxiliary keeps the embedding space class-organized without preventing the residual from learning cross-subtype morphs.

### 3.5 Renders coverage

After we re-filter the splits post-watertight, any newly-promoted ids will need rendering via `scripts/render_buildingnet_objfiles.py` (≈ 4 s/id, GPU). Renders are required as the input to the discriminator (§2.3) and any visualization pipeline. They are NOT on the inference path of the hybrid pipeline itself.

---

## 4. Inference pipeline (detailed)

### 4.1 Single-footprint inference

```python
def infer_one(input_footprint, class_id, target_height_m, world_centroid_xyz, world_polygon_area_m2):
    # 1. Embedding + retrieval
    fp64 = resize_to_64(input_footprint)
    fp_norm = normalize_to_unit_box(fp64)
    q_emb = footprint_embed(fp_norm, class_id)

    candidates = topk_index.query(q_emb, k=5, filter=class_id.top_level)
    retrieved_id = max(candidates, key=lambda j: aligned_iou(fp_norm, train_fp[j]))

    # 2. Load retrieved SDF, encode to latent
    sdf_r   = load_h5(f"resolution_64/{retrieved_id}/ori_sample_grid.h5")["pc_sdf_sample"]
    sdf_r   = sdf_r.reshape(1, 1, 64, 64, 64)
    z_r     = vqvae.encode_no_quant(sdf_r) * scale_factor

    # 3. Build conditioning volumes
    fp3d    = build_fp3d(fp_norm, D=16, H=16, W=16)
    h_vol   = build_height_vol(target_height_m, D=16, H=16, W=16)

    # 4. Predict residual
    inp     = torch.cat([z_r, fp3d, h_vol], dim=1)
    z_resid = correction_net(inp, class_id)

    # 5. Add residual, decode
    z_out   = z_r + z_resid
    z_out   = z_out / scale_factor
    sdf_out = vqvae.decode_no_quant(z_out)

    # 6. Marching cubes in Frame N
    verts_N, faces = marching_cubes(sdf_out[0,0].cpu().numpy(), level=0.0)

    # 7. Frame N -> Frame W
    s_world        = sqrt(world_polygon_area_m2 / footprint_area_in_N(fp_norm))
    verts_W        = verts_N * s_world + world_centroid_xyz
    verts_W[:, 1] += target_height_m / 2.0

    return Trimesh(verts_W, faces)
```

### 4.2 kNN details

- Index: brute-force `torch.cdist` over the training embeddings (5 ms per query, no FAISS needed at this scale).
- Class filter: precompute boolean mask per top-level class. Apply mask before the cdist argmin.
- Tie-break: aligned IoU (rotation in {0,90,180,270} + per-axis scale 0.7–1.3 search) on 64×64 footprint masks, top-5 candidates → top-1 winner. Adds ~50 ms but matters for footprint fidelity.
- Empty-class fallback: if class_id is unknown or filtered set is empty, fall back to global top-1 by emb distance.

### 4.3 World-coords placement

- Map-side input: vector polygon coords in meters, target height in meters, optional class.
- Compute polygon area, polygon centroid, and polygon principal axis (PCA on vertices).
- The retrieved SDF's principal axis (computed once at indexing time and cached) is rotated to align with the polygon's principal axis. A single 2D rotation about Y in Frame N before the World transform.
- Scale: `s_world = sqrt(polygon_area_m2 / unit_area_in_N)`. Vertical scale is decoupled — set to `target_height_m / current_height_in_N` per-axis-Y so a tall polygon lifts a tall mesh.
- Anisotropy clamp: refuse to rescale if `s_xz / s_y` exceeds 3.0 (keeps proportions sane); else fall back to isotropic and accept some footprint mismatch.

### 4.4 Multi-building scene composition

- **Default: mesh-merge.** After per-polygon meshes are produced, run `trimesh.boolean.union` over all of them in a single pass. For 50–100 buildings on an A100 host this is on the order of 10–60 seconds. Output: single OBJ.
- **Fallback for overlapping polygons: SDF union.** If any two polygon AABBs overlap, fall back to per-AABB-overlap pair voxelization at K=512^3 over the bounding region of just the overlap, take elementwise min, marching cubes the local fix-up, re-merge. Skip this until you actually encounter overlaps in real input maps.

---

## 5. Step-by-step implementation order

| #  | Step                                       | Build / change                                                                                                         | Verify                                                                                                                                | Duration   | Depends on |
| -- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| 0a | **F0 — OSM vector parser (Route A).** Pipe `osmnx` results into our `{polygon, class, height}` schema. | New `scene/extract_osm.py`. Uses `osmnx.features_from_bbox(...)` for buildings + `osmnx.graph_from_bbox(...)` for roads. Maps OSM `building` tag → our 5 top-level classes (RESIDENTIAL/RELIGIOUS/COMMERCIAL/MILITARY/PUBLIC); falls back to `RESIDENTIALhouse` for unknown. Computes height via `building:levels × 3.5 m` or `height` tag when present. | Output a JSON with at least 5 polygons + class + height for a small bounding box of Lafayette, IN. Manual visual check: open in QGIS / leaflet. | 1 h | — |
| 1  | Promote v1 splits to active                | Replace `splits/*.txt` with `splits_v1_official/*.txt`                                                                 | `wc -l splits/*.txt` shows 1480/186/180; dataset loader smoke loads one batch                                                         | 5 min      | —          |
| 2  | Render OBJ ortho views for any newly-promoted ids | `scripts/render_buildingnet_objfiles.py --phase all` (overwrite false; should fill in ~485 missing) | All 1849 ids have renders                                                                                                              | 30 min     | 1          |
| 3  | **SMOKE 0: end-to-end, random retrieval, zero residual, ARAP-deform.** Build inference script `scripts/hybrid_smoke.py` that picks 4 train footprints, retrieves a random different building, runs `z_corrected = z_r + 0`, then ARAP-deforms the retrieved OBJ to the (zero-residual) iso-surface. Writes deformed OBJs. | `outputs/hybrid_smoke_step3/` contains 4 OBJ files, all visibly buildings with original detail intact. ARAP solve must complete without face flips. | **30 min** | 1 |
| 4  | ARAPDeformer implementation                 | New `models/arap_deformer.py`. libigl-based ARAP with anchor identification, displacement cap, fallback paths.         | Smoke: deform `RESIDENTIALhouse_mesh5954` to a slightly-altered footprint; output OBJ opens in MeshLab; no flipped faces. | 1 h        | 3          |
| 5  | FootprintEmbedNet implementation            | New `models/networks/retrieval/footprint_embed.py`, `train_retrieval.py`                                              | Train on synthetic copy: deliberately corrupt 100 footprints with rot/scale; embedding distance < 0.1 between original and corrupted   | 1 h        | 1          |
| 6  | Train FootprintEmbedNet                     | Run trainer, ~50 epochs                                                                                                | Retrieval top-1 same-class accuracy > 60% on val; visual top-K montage shows similar building shapes                                   | 1 h        | 5          |
| 7  | Build retrieval index                       | `scripts/build_retrieval_index.py` writes `retrieval_index/{embeddings.npy, topk_train.npy, class_mask.npy}`           | k=5 query for 4 val footprints prints sane neighbor IDs; aligned IoU on top-1 > 0.5                                                    | 10 min     | 6          |
| 8  | **SMOKE 1: retrieval + ARAP, no residual**. Pull real top-1 retrieval, ARAP-deform to input footprint via the retrieved building's own SDF (no correction yet). | Outputs class-appropriate; deformed mesh silhouette IoU vs input footprint > 0.6. | 30 min | 7, 4 |
| 9  | SDFCorrectionNet architecture & dataset wrapper | New `models/sdfusion_correction_model.py`, `datasets/correction_pair_dataset.py` (returns retrieved+target pairs)      | Forward pass on a batch of 2: shapes match, no NaN. `print(z_resid.std())` non-zero                                                    | 2 h        | 7          |
| 10 | **SMOKE 2: train SDFCorrectionNet on 16 pairs for 200 steps**, latent-MSE only. | Loss decreases monotonically. Save checkpoint. Sanity: `decoded(z_r + pred_resid)` is closer to target SDF than `decoded(z_r)` is. | 30 min | 9 |
| 11 | Full SDFCorrectionNet training              | 30k steps, batch 8, bf16, lr 1e-4, all train ids. Loss = L_latent + 0.5·L_decode + 0.25·L_silh_deformed (ARAP-deformed retrieved mesh's silhouette vs input footprint).         | Val L_latent plateau; periodic visual: side-by-side of (input fp, retrieved mesh, ARAP-deformed mesh, GT mesh) every 2k steps          | 4–6 h      | 9          |
| 12 | World-coords transform & scene compose      | New `scene/compose.py`. Frame N→W transform plus trimesh concat (boolean union as fallback for overlapping polygons)   | Compose 3 buildings on a synthetic 3-polygon map; OBJ opens in MeshLab without overlap pathology                                       | 2 h        | 11         |
| 13 | **SMOKE 3: full pipeline on a 6-polygon synthetic map**, including OSM Route A input from Step 0a. Single command: GeoJSON in → town OBJ out. | Outputs match human inspection of "this is a residential block" with full BuildingNet detail | 15 min | 12, 0a |
| 14 | Eval suite (§6)                             | `eval/run_eval.py` over the test footprints                                                                            | Metrics computed end-to-end                                                                                                            | 1–2 h      | 11         |

**The "first end-to-end smoke" is Step 6.** It validates the entire dataflow (SDFs → VQVAE → MC → mesh) without committing to any new training. Expected wall time including initial model loads is ~5–10 minutes for the smoke pass itself.

---

## 6. Evaluation plan

### 6.1 Quantitative

Run on the test footprints. For each:
1. Run hybrid pipeline → corrected mesh.
2. Run baseline 1: pure retrieval (top-1, no residual). Same retrieval, but `z_corrected = z_r`.
3. Optional: run frozen Hunyuan3D-2 on a depth-rendered version for comparison.

Metrics:

| Metric                           | Definition                                                                                            | Target                                |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------- |
| Footprint silhouette IoU         | `IoU(top-down silhouette of generated mesh, input footprint)` at 64×64                                | > 0.7 mean (vs ~0.4 for retrieval-only) |
| Class accuracy (discriminator)   | `argmax ClassDiscriminativeHead(render(generated_mesh)) == class_id`                                  | > 75% top-1                           |
| Chamfer distance to closest test mesh | `chamfer(generated_mesh, ground_truth_mesh)` in Frame N                                            | reported per-class median             |
| FID over 3/4-view renders        | FID between renders of generated vs renders of full test set                                          | reported, target < 50                 |
| Inference time per footprint     | wall-clock seconds                                                                                    | < 1.0 s on A100 (excluding compose)   |

### 6.2 Qualitative

- 4×4 panel: rows = {input footprint, retrieved (no residual), corrected, ground truth}, columns = 4 representative classes (house, church, factory, castle). Saved at `outputs/hybrid_eval/qual_panel.png`.
- Multi-building scene: take a real urban OSM tile (provided by user), produce composite OBJ, render top-down at 1k resolution + 4 oblique angles.

### 6.3 Ablations to run after Step 16

- Residual disabled (= Baseline 1) — quantifies the residual's contribution.
- Class filter disabled — quantifies the class-filter contribution to retrieval.
- Top-1 vs top-5-with-IoU-tiebreak retrieval — quantifies the alignment refinement.
- Footprint conditioning replaced with zeros at correction-net input — quantifies whether the residual is actually using the footprint cue.

---

## 7. Risks and mitigations

| Risk                                                                                       | Mitigation                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ARAP tangles on thin features (spires, chimneys)                                            | Cap per-vertex displacement at 0.3 (Frame N units), reject moves that flip triangle normals, and fall back to local Laplacian smoothing for problem regions. If still bad, subsample anchors and propagate via barycentric.            |
| ARAP solve is slow on 240k-vertex meshes                                                    | Subsample anchors to 5k; solve ARAP on a decimated proxy mesh (target 10k verts via quadric error) and propagate displacements back to the original mesh via point-to-surface barycentric mapping. Cost target: <1 s for any building. |
| Residual training is unstable because target is small                                      | Normalize target residual by its empirical std on a 100-id subset; train on normalized residuals; un-normalize at inference. Also clip latent residual during inference if `||z_resid|| > 3 σ` to prevent decode artifacts.            |
| `_build_fp3d_for` axis bug regression                                                      | Run the silhouette-IoU axis sanity test (`encode→decode IoU > 0.8 along Y axis`) as a CI-style assertion at the top of `train_correction.py`.                                                                                          |
| Retrieval kNN is too small a pool and ablating just memorizes the single nearest neighbor  | Force k≥5 retrieval at training time and sample uniformly within k. Augment with rotated/reflected footprints (×4) for retrieval training only.                                                                                       |
| Distribution shift: real input map polygons may have aspect ratios not seen in training    | Augment training footprints with random rectangular padding (±15%) and aspect-ratio jitter (×0.7–1.3). Also evaluate on synthetic rectangular polygons up front (Step 15).                                                              |
| Class label mismatch at inference (user provides "house" but model expects "RESIDENTIALhouse") | Build a label normalization map at `eval/label_map.py` mapping common synonyms to the 53 prefixes. Default to `RESIDENTIALhouse` if unknown.                                                                                          |
| World-coords scale produces meshes that interpenetrate the ground plane                     | Always lift mesh by `bbox.min.y * -1` after Frame N→W; never trust pre-translation Y.                                                                                                                                                  |
| VQVAE checkpoint is from before the watertight repair — encoder may underperform on the new SDFs | Run a quick eval: encode→decode IoU on 20 newly-regenerated SDFs. If IoU < 0.75, retrain VQVAE for 12 h before SDFCorrectionNet training. If IoU > 0.75, proceed and revisit only if correction-net fails.                              |
| Compose-time mesh boolean union is slow or fails on degenerate inputs                       | Default to mesh-concatenation (no boolean) until polygon overlap is detected. Concatenation works for any inputs, looks correct for non-overlapping footprints, and avoids CSG failure modes.                                          |

---

## 10. Future: image-based map input (the designer-facing front-end)

Goal: a designer / modeler drops a map screenshot (Google-Maps-style stylized rendering, an OSM render, or a hand-drawn site plan) and gets the same `{polygon, class, height}` triples that Route A produces from vectors.

### 10.1 Why this is easier than aerial segmentation

Stylized rendered maps have **deterministic visual semantics** — buildings are flat colored polygons, roads are colored lines, classes are encoded by color/style. The model only needs to learn "what does the renderer's rasterization of an OSM building look like in pixels." That's a much narrower distribution than "what does a building look like from above in real photographs."

### 10.2 Synthetic training set is essentially free

Because OSM already has the ground-truth labels for every tile we render:

```
1. Sample N=2000 random bounding boxes worldwide (mix urban / suburban / dense / sparse)
2. For each: render via Mapbox or OSM tile server → input PNG (512×512 or 1024×1024)
3. For each: query OSM features → multi-class mask (building / road / water / park / background)
4. Save (image, mask) pair
```

Generating 2k pairs takes ~3–4 h of API calls (no GPU). No human labeling required.

### 10.3 Model architecture

| component        | choice                                                                  | params  |
| ---------------- | ----------------------------------------------------------------------- | ------- |
| backbone         | SegFormer-B0 or U-Net (ResNet-18 encoder)                               | 5–14 M  |
| classes          | 5 (background, building, road, water, park)                             |         |
| training time    | A100, batch 16, 50 epochs over 2k pairs                                 | ~2 h    |
| post-processing  | argmax → per-class binary mask → `cv2.findContours` → simplified polygons | <1 s/tile |

### 10.4 Outputs of F0 Route B

For each detected building contour → polygon. Class assigned by majority-class of contour interior. Height defaults to a class-conditional median (residential 8 m, commercial 15 m, religious 12 m). Height stretch: a tiny regressor on building width/area + class → height estimate (trained from OSM `building:levels` data).

### 10.5 When to build it

After Steps 1–16 of the main plan are done and we have a working retrieval+correction pipeline. Route B is purely a front-end change — it doesn't touch any of the existing trainable components. Estimated end-to-end build: **2 days** including data scraping, training, and integration.

---

## 8. Stretch ideas (after Steps 1–16 pass)

1. **Diffusion-residual variant.** Replace the deterministic SDFCorrectionNet with a small conditional latent diffusion that samples residuals. This buys variety: same input footprint can produce N different plausible buildings. Reuses the existing diffusion infrastructure in `models/sdfusion_model_img2shape.py`. Cost: ~6 h additional training, ~5× inference time. Defer until single-residual quality is good.
2. **128³ SDFs after VQVAE retrain.** Higher-resolution geometry unlocks finer architectural detail (windows, gables). Requires retraining the VQVAE (~12 h) and re-running create_sdf at 128³ (storage ×8 → ~100 GB). Worth it only if 64³ output looks blocky in the final eval.
3. **Footprint-to-height predictor.** When the input map has no per-polygon height, predict it from `(footprint, class)` with a tiny regressor trained on `buildingnet_heights/` data. Removes a manual input from the pipeline.
4. **Cross-class residuals (relax §3.5 hard filter).** Train a slightly bigger correction net that conditions on (target class) and accepts retrievals from any class. Useful when the target class is data-starved (e.g. RELIGIOUSpalace with 12 samples).

---

## 9. Decisions made / remaining

**Resolved:**
1. ~~Watertight repair~~ — **dropped.** Hollow meshes are kept; ARAP deforms them directly so detail is preserved.
2. ~~VQVAE retrain~~ — **not needed.** Current 64³ SDFs continue to be used; the residual operates in latent space and the corrected SDF is only a deformation guide.
3. **Mesh deformer:** **ARAP (Option 2)**, libigl's reference implementation.

**Still open, but non-blocking:**
4. **Hunyuan3D-2 status:** archive to `legacy/`? It's not on any active path. Keeping it for evaluation comparison is cheap (~9 GB cache).
5. **Class taxonomy:** default to **53 sub-classes for the embedding's auxiliary CE + 5 top-level for the retrieval filter**. Override only if you want a custom collapse.

These two can be answered any time before Step 7 (retrieval index) without delaying earlier steps.

---

## Critical files for implementation

- `models/sdfusion_model_img2shape.py` (source for `_build_fp3d_for`, `scale_factor` plumbing, VQVAE wiring patterns)
- `models/networks/diffusion_networks/network.py` (UNet body to repurpose for SDFCorrectionNet with `in_channels=5`, time-embedding stripped)
- `models/networks/vqvae_networks/network.py` (frozen VQVAE encoder/decoder used at every stage)
- `datasets/buildingnet_dataset.py` (loader to extend for retrieval pairs and correction-pair dataset)
- `preprocess/create_sdf.py` (must consume `OBJ_REPAIRED/` after watertight pass; do not call `bake_footprint` — recompute via `scripts/recompute_footprints_from_sdf.py`)
- `logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth` (frozen VQVAE checkpoint)
