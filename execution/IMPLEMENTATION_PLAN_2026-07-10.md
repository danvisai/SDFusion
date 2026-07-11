# Implementation plan — massing-scale-decomposition proof (2026-07-10, rev2)

File-level tasks for `execution/RESEARCH_PROOF_PLAN` + `execution/DATA_PREP_PLAN`, coherent with the
experiment decisions in `docs/adr/0002-experiment-design.md` (D1–D4). Vocabulary: `CONTEXT.md`.

## Corrections that shaped this plan
- **A. The existing detailizer is NOT a valid baseline.** `make_detail_pairs.py` builds
  `detail_pairs_v1` *synthetically* (detailed = **composer** output), so `train_detailizer.py` learned
  to imitate our procedure. The monolith must train on **REAL** (massing → BuildingNet-SDF) pairs; real
  grids exist at `data/BuildingNet_dataset_v0_1/resolution_64/*/ori_sample_grid.h5`. Old detailizers
  demote to a documented negative.
- **B. Hunyuan3D-2 dropped (D4).** It's image-conditioned, can't take a footprint; "just get more data"
  is answered instead by extrapolating the fractioned monolith curve. No giant integration.

## Global conventions
- **Working resolution ≥ 96³**, identical across all arms (a window is 2–4 voxels at 64³).
- **Equal-data rule (D2):** at fraction *X*, monolith trains on `train_X` real pairs AND the
  decomposition retrieves from a library built from `train_X` ids; Stage 3a massing stays full.
- **Two decomposition arms (D1):** headline = **Stage 3a massing** + retrieval; ablation = recipe-param
  → procedural massing + retrieval.

---

## M0 — Prerequisites (gate everything; mostly no GPU)
- **I0.1 Splits** — new `scripts/foundations/make_splits.py` → `data/splits_v1/{test,train_25,50,100}.json`
  (nested, seeded, class-stratified; test ≈15 % sealed). *Done:* disjoint, `test ∩ train_* = ∅`.
- **I0.2 FID + facade-render harness** — new `scripts/eval/render_facades.py` (reuse
  `scripts/appearance/texture_bake.py::make_cameras` + SDF trace, **neutral gbuffer/normal shader — NOT
  the SDXL bake**) and `scripts/eval/fid.py` (clean-fid/pytorch-fid). *Done:* FID(real,real)≈0 sanity.
- **I0.3 `s*` coincidence TEST** (parallel; no training) — new `scripts/eval/measure_scale_spectrum.py`.
  **`s*` is fixed a priori** (`k` voxels at the working res, ≈0.5 m) — this script does **not** choose it;
  it measures whether each BuildingNet semantic-detail category falls below the fixed `s*` and massing
  above. *Done:* the coincidence figure (pass or fail); the fixed `s*` recorded in `CONTEXT.md`.

## M-C1 — Transform claim (C1) — cheap, high-signal, uses the LIVE prior (no training)
- **I-C1.1 From-noise vs blockout FID** — new `scripts/eval/transform_vs_noise.py`: per test footprint,
  (a) sample the Stage 3a prior **from noise** (the best *honest* unconditional sample, not a strawman);
  (b) **SDEdit from a footprint-extrude blockout**; render both (I0.2 neutral shader) → FID vs real.
  *Done:* FID(noise) ≫ FID(blockout).
- **I-C1.2 Sculpt strength sweep** — new `scripts/eval/sculpt_sweep.py`: a small set of (building, crude
  edit) cases; sweep SDEdit `strength` via `/snap_sdf`; report faithfulness (IoU-to-edit) ↔ realism (FID)
  as one curve + qualitative sheets. *Done:* the faithfulness↔realism curve (one operator spans gen↔edit).
- **I-C1.3 Residual datapoint (already done)** — pull `Logs_GT/sdf_residual_full_v4_aug_topk3`
  (`val_corrected_fp_iou≈0.999` vs `val_corrected_iou≈0.13–0.28`) into the paper as the "transform aligns
  massing, not detail" figure. No new work.

Runs in Step 1 (before the C2 kill-gate): C1 uses the live prior + an existing result, so it is the
cheapest high-signal evidence.

## M1 — Honest monolith baseline (Correction A)
- **I1.1 Real detail pairs per fraction** — new `scripts/foundations/make_real_detail_pairs.py`: for each
  id, detailed = `ori_sample_grid.h5`; coarse = **low-pass** of that SDF (default; footprint-extrude as
  a variant). Write `data/detail_pairs_real_{25,50,100}.h5` filtered by the split lists. *Done:* pair
  montage at ≥96³.
- **I1.2 Train monolith per fraction** — `scripts/foundations/train_detailizer.py
  --pairs data/detail_pairs_real_{frac}.h5 --out outputs/monolith_{frac}`. *Done:* 3 ckpts + val curves.

## M2 — Decomposition eval arm (headline = Stage 3a massing)
- **I2.0 Per-fraction libraries** — build `element_library_train{25,50,100}` (M5.1 tool, `--include-ids
  train_X --exclude-ids test`). Feeds the equal-data rule.
- **I2.1 Batch-generate for test footprints** — over test ids: **Stage 3a SDF massing** (full data) +
  detail **retrieved** from `element_library_train{frac}` (`scripts/server/element_fit.py`) + composer
  regularization where the type calls for it. Skeleton: `scripts/stage3_generate.py`. *Done:* one SDF per
  test footprint per fraction at the working resolution.
- **I2.2 Render + FID both arms** (I0.2) → the **headline number** per fraction. *Done:*
  FID_detail(decomp) < FID_detail(monolith), massing Chamfer ≈ equal.
- **I2.3 KILL-GATE (100 % first):** run I1.2@100 + I2.1@full-library + I2.2 before the 25/50 runs.

## M3 — Data-scaling curve
- **I3.1** monolith@25,50 (I1.2) + decomposition@25,50 (I2.0/I2.1 with fractioned libraries) → the (2×3)
  FID curve. *Done:* monolith slow-improving vs decomposition graceful; extrapolate the monolith slope
  for the "more data" rebuttal.

## M4 — Robustness ablation + user study
- **I4.1 Recipe-massing arm (D1 ablation)** — rerun I2.1 with recipe-param → procedural massing instead
  of Stage 3a. *Done:* factorization win holds across massing source.
- **I4.2 2AFC study** — minimal static web form over paired neutral renders (decomp vs monolith vs real).
  *Done:* preference table.

## M5 — Element library correctness + enrichment
- **I5.1 Leakage-safe, id-filtered builder** — extend `scripts/foundations/build_element_library.py` with
  `--include-ids/--exclude-ids`; this powers both the leakage fix and the per-fraction libraries (I2.0).
  *Done:* no test building contributes a retrieved element.
- **I5.2 `lod3_tum` enrichment (ablation)** — extend the extractor to ingest `data/lod3_tum` LoD3 facade
  components; build a `+lod3_tum` library; re-run I2.2 as a "+real element data" delta. *Done:* Δ-FID.

---

## Dependency graph
```
I0.1 ─┬─ I1.1 ─ I1.2 ───────────────┐
      ├─ I5.1 ─ I2.0 ─ I2.1 ─ I2.2 ─┼─ I2.3 KILL-GATE ─ I3.1 ─ I4.1/I4.2
I0.2 ─┘                              │
I0.3  (parallel, no deps)     I5.2 after the headline
```

## First actions (in order)
1. **I0.1 splits** + **I5.1 id-filtered library builder** (unblocks the leakage fix + per-fraction libs).
2. **I0.2 FID harness** with the real-real≈0 sanity; **I0.3** in parallel.
3. **M-C1** (from-noise-vs-blockout FID + sculpt sweep) — cheap, uses the LIVE prior, establishes the
   **transform** half (C1) before any training.
4. **I1.1 real pairs @100 → I1.2 monolith@100**, and **I2.0 full-library → I2.1 decomp**, then **I2.3
   C2 kill-gate**.
5. Past the gate: **I3.1** curve (fraction pairs + libraries), then **M4/M5**.

## Open decisions still to confirm
- Working resolution for the detail comparison (recommend 96³; 128³ if detail under-reads).
- Coarse-side derivation in I1.1 — low-pass (default) vs footprint-extrude vs both as monolith inputs.
