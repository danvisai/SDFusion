# Build Real Full-Data Monolith Pairs

Type: task
Status: resolved
Blocked by: 01, 03

## Question

Build the 100% monolith dataset from real BuildingNet target SDFs and the locked coarse-input
derivation, excluding held-out ids. Verify resolution, sign/axis conventions, alignment, class
coverage, and representative pair montages so the baseline cannot accidentally train on synthetic
composer outputs or corrupted pairs.

## Comments

## Answer

**Built:** `scripts/foundations/build_monolith_pairs.py` (TDD, 16 contract tests for the pure
seams — `coarse_resolution`, `low_pass_sdf`, `footprint_alignment_iou`, `validate_pair`, and reuse
of ticket 04's `select_building_ids` leakage seam). No dataset is duplicated to disk: only
`train_100`'s low-pass-derivable coarse input is a new artifact; the real target SDF already lives
in `data/BuildingNet_dataset_v0_1/resolution_64/<id>/ori_sample_grid.h5` and is loaded on demand
via `render_facades.load_buildingnet_sdf`, matching this proof's existing pattern of tracking small
manifests/ids rather than committing or duplicating large binaries.

**Coarse input = ADR 0004's locked primary, `low_pass_sdf`:** resample the SAME building's real SDF
down to a grid whose own voxel pitch matches the already-fixed `s*` (`coarse_resolution(96, 5) =
19`), then back up to `working_res=96` via the same trilinear `resample_sdf_grid` every other
ticket already resamples with. No new interpolation code path, no new free hyperparameter beyond
what ADR 0004 already fixed. Source and target stay spatially aligned by construction — the coarse
grid is derived from the loaded target array itself, never a separate re-derivation from footprint
or height.

**Axis/sign verification against independent ground truth:** `footprint_alignment_iou` compares a
footprint DERIVED from 3D occupancy (`occ.any(axis=1)`, the H-up convention every eval script
assumes) against BuildingNet's OWN precomputed footprint field stored in `ori_sample_grid.h5`. Spot
check before the full run: axis=1 gives IoU=1.0 on real data; axis=0/2 give ~0.06 — confirms the
assumed convention is correct rather than assumed. Sign convention (`sdf<=0`=inside) matches
tickets 05/06/09 exactly, so all four arms share one contract.

**Full `train_100` build (1,572 buildings, 0 limit):**

| check | result |
|---|---|
| pairs built / requested | 1572 / 1572 |
| failures (shape/finiteness) | 0 |
| leakage (sealed `test` ids present) | 0 |
| footprint axis-convention IoU | mean 1.0, min 1.0, 0 buildings below 0.5 |
| class balance | COMMERCIAL 105, PUBLIC 48, RELIGIOUS 378, RESIDENTIAL 1041 — **exact match** to ticket 03's frozen `train_100` class balance |
| mean occupancy frac | target 1.43%, coarse 1.22% |

**Qualitative (`outputs/monolith_pairs_v1/montage.png`, 3 buildings/class):** the low-pass primary
visibly does what ADR 0004/CONTEXT.md predict — thin isolated spires, columns, and rooftop
ornament in the real target are smoothed away in the coarse input while the broad platform/roof
mass survives (clearest on `COMMERCIALcastle_mesh2985` and `RELIGIOUStemple_mesh0369`). A handful
of real BuildingNet meshes are themselves very sparse/fragmented at this resolution (already
disclosed by ticket 09 — occupancy is a continuous distribution with no natural "corrupted"
cutoff); the coarse version of those stays correspondingly sparse rather than being forced solid,
which is honest behavior, not a bug — occupancy is recorded per pair in `per_pair.json` so a reader
can judge data quality themselves.

**Out:** `data/monolith_pairs_v1/{manifest.json, pairs.json, per_pair.json}` (ids + provenance +
diagnostics only, no duplicated SDF grids — gitignored like every other `data/` artifact),
`outputs/monolith_pairs_v1/montage.png`.

Unblocks ticket 11 (train the full-data monolith).
