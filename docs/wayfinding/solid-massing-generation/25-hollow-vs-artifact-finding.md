# Finding: BuildingNet massing is a thin-shell reality, but solidify-in-place recovers solid blocks

Resolves ticket [Determine whether BuildingNet massing is genuinely hollow or a surface-vs-filled artifact](https://github.com/danvisai/SDFusion/issues/25).

## Method

Sampled 80 buildings from the sealed held-out set (`data/splits_v1/test.json`, seed 0), loaded each native 64³ field (`ori_sample_grid.h5:pc_sdf_sample`), and compared four measures per building:

- **raw** — `(sdf<=0).mean()`, what the model trains on today.
- **fillholes** — `binary_fill_holes(raw)`, recovers *watertight* enclosed voids.
- **colfill** — ground-anchored footprint-column extrusion along the H-up axis (fill each occupied column floor→top), the solid-massing target we actually want.
- **boundary_frac** — fraction of raw-occupied voxels on the surface (shell indicator).

Plus a check of the independent stored `footprint` field for the near-empty tail. Script: `scripts/foundations/probe_buildingnet_solidity.py`; data: `execution/artifacts/buildingnet_solidity_probe.json`; renders: `montage.png` in this folder.

## Results

| Measure | Value |
|---|---|
| raw occupancy, median | **0.56%** |
| held-out below 0.5% occ (raw) | **47.5%** |
| boundary fraction, median | **0.83** (83% of occupied voxels are surface) |
| `fillholes` gain over raw, median | **0.0000** (no enclosed voids to fill) |
| `colfill` occupancy, median | **4.89%** (≈ **11× raw**) |
| below 0.5% occ after `colfill` | **1.25%** (down from 47.5%) |
| stored footprint field empty | **0 / 80** |
| near-empty tail (raw<0.3%): footprint area vs occupancy | **34×** (mean footprint 4.93% vs occ 0.14%) |

## Verdict

**It is a genuine thin-shell / open-mesh reality — not a mismeasurement — but it is fully recoverable in place, and no new corpora are *required* to get solid massing targets.**

1. The SDF faithfully represents a thin, **open (non-watertight)** shell: 83% of occupied voxels are surface, and watertight hole-filling adds *nothing* (0% gain) because the interiors connect to the outside (open bottoms/gaps). Low occupancy is real thin geometry, not a measurement bug.
2. **Watertight fill (winding-number / `binary_fill_holes`) does not work** on these meshes — ruled out as the solidification method.
3. **Footprint-driven extrusion does work:**
   - Buildings with real occupancy geometry (raw ≳ 0.3%) → occupancy column-extrusion yields coherent solid footprint-matching blocks (11× occupancy gain; `villa_mesh0`, `house_mesh9` in the montage are textbook solid blocks with roof shape).
   - The near-empty tail (raw < 0.3%, ~12–15%) can't be extruded from its own noise, **but the stored `footprint` field is intact (34× the occupancy)** → footprint-extrusion recovers a block. This is the *same* footprint field the empty-input fallback needs.

## Implications for downstream tickets

- **[Target representation](https://github.com/danvisai/SDFusion/issues/28)** (now unblocked): the target should be a **footprint-extruded solid**, occupancy-driven where geometry is rich, stored-footprint-driven for the sparse tail. Method = footprint/column extrusion, **not** watertight fill.
- **[Empty-input fallback](https://github.com/danvisai/SDFusion/issues/29)**: the stored `footprint` field is the rescue signal, available per-building in the same h5 — the fallback and the representation share one mechanism.
- **[Corpora audit](https://github.com/danvisai/SDFusion/issues/26)**: reframed from *necessity* to *diversity/coverage* — solid massing is achievable from BuildingNet alone; extra corpora are for variety, not for basic solidity.
