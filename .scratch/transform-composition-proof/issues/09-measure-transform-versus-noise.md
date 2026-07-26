# Measure SDEdit Transform Versus From-Noise Sampling

Type: task
Status: resolved
Blocked by: 03, 05

## Question

Implement and run the C1a comparison on held-out footprints: the best honest Stage 3a from-noise
sample versus SDEdit from a footprint-extrude blockout, rendered through the same neutral harness.
Record sampling contracts, massing metrics, FID with uncertainty, qualitative failures, and whether
the predicted transform advantage holds.

## Comments

## Answer

**Built:** `scripts/eval/transform_vs_noise.py` (TDD, 7 contract tests for the two pure seams —
footprint-extrude blockout construction and Stage3a-training-leakage classification; model
inference is GPU/checkpoint-dependent and verified by the real run below, matching this project's
established convention for that kind of code).

**LEAKAGE CATCH (before running anything):** `data/splits_v1/test.json` (ticket 03's sealed
research-proof split) is a *different* partition than the original
`data/BuildingNet_dataset_v0_1/splits/{train,val,test}_split.txt` Stage 3a was actually trained
on. Checked the overlap: **224/277** of ticket 03's "held-out" ids were in Stage3a's own gradient
training set, 26 in its validation set, and only **27** were genuinely never seen by the live
prior in any form. The ticket's own wording is "held-out footprints" — using the other 250 would
let memorization pass as evidence of generalization. Evaluated the clean 27 only;
`held_out_population()`/`classify_leakage()` compute and record this partition every run.

**Sampling contract (both arms share it, differ only in diffusion starting point):** conditioning
= real footprint + height (Frame-N contract, `frame_n_input`) + true BuildingNet subtype
`class_id` (recomputed via the *exact* `subtype_to_idx` map `Stage3aDataset` builds at training
time — same input ids, same deterministic sort) + the style-agnostic massing token
(`STYLE_UNKNOWN_ID=8`). From-noise = `Stage3aModel.inference()`, full 50-step DDIM from pure
Gaussian noise. Blockout = `Stage3aModel.sdedit(strength=0.5)` on the footprint-extrude blockout
(the real footprint solid-extruded to the real height — ADR 0004's declared coarse-input
variant). Guidance is **matched plain CFG** (`guide_model=None`, model default `uc_scale=1.0`,
`ddim_steps=None`→50) rather than production's autoguidance for `sdedit` — `.inference()` has no
autoguidance path, so using it only for `sdedit` would confound "better starting point" with
"better guidance," which isn't the claim under test. `strength=0.5` is the codebase's existing
canonical value (`eval_harness.py`'s primary strength, `refine_sdedit`'s own default) — ticket 10
is the dedicated strength sweep, so this ticket fixes one value rather than duplicating that work.
Loaded via `Refiner._load_sdedit(autoguidance=False)` — the actual deployed 2026-07-03
cross-cultural checkpoint, not an arbitrary training snapshot.

**A second data-quality catch, disclosed not filtered:** BuildingNet's native 64³ SDFs vary
continuously in interior occupancy across the 27-building clean population (0.02%–6.4% occupied
voxels, `real_occupancy_frac` recorded per building) — no natural gap separates "broken" from
"valid," so no exclusion threshold was applied (would have been an undisclosed, results-motivated
choice). Footprint IoU is reported as the *primary* massing metric because it is far more robust
to this than full-volume IoU (CONTEXT.md already privileges paired IoU for footprint-determined
massing); full-volume IoU is reported alongside, not hidden.

**Result over the 27 clean buildings, 0 failures — the predicted transform advantage holds:**

| metric | from-noise | blockout SDEdit |
|---|---|---|
| mean footprint IoU | 0.356 | **0.592** |
| median footprint IoU | 0.304 | **0.607** |
| mean volume IoU | 0.065 | **0.090** |
| median volume IoU | 0.032 | **0.054** |
| FID vs real (undersampled, n=27) | 225.2 [222.0, 258.5] | 213.3 [217.0, 248.9] |

Footprint IoU roughly **doubles** at the median (0.30 → 0.61) — a large, paired-metric effect at
n=27. Full-volume IoU moves the same direction but more modestly, expected given several
buildings' thin/sparse real ground truth. FID shows no reliable signal (both arms flagged
`undersampled=true`, overlapping CIs) — exactly the ticket-05 finding (2048-d Inception features
need far more than 27 buildings × 6 views), so it is reported honestly as inconclusive rather than
leaned on.

**Qualitative failure mode (`outputs/transform_vs_noise/montage.png`, real vs from-noise vs
blockout for 8 buildings):** for every building with reasonably solid real geometry, from-noise
samples are visibly lumpy, melted blobs with no flat facades or coherent box structure;
blockout-SDEdit samples are consistently more rectilinear and building-like (starkest for
`PUBLICcity_hall_mesh0451`: a jagged blob vs. a clean rectangular mass). Neither arm reproduces
the real building's exact shape (expected — this measures massing *plausibility*, not detail
fidelity), but the qualitative gap matches the quantitative one.

Confirms C1 on genuinely held-out footprints. Establishes the from-noise-vs-blockout harness for
the paper's headline transform evidence; ticket 10 (strength sweep) and the residual-correction
datapoint (already in hand) are the remaining C1 pieces.
