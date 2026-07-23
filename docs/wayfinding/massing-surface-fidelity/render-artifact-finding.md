# The "step texture" is a diagnostic render artifact, not learned geometry (2026-07-23)

Founding finding for the 2026-07-23 re-chart of [Massing Surface Fidelity (#34)](https://github.com/danvisai/SDFusion/issues/34).
Prompted by the observation that a stair-step texture appears on **both** the ground-truth and generated
meshes in every fidelity montage.

## Claim

The dramatic staircase "step texture" is a **binary-occupancy marching-cubes artifact** introduced by the
*diagnostic* renderers — **not** something the model learned from the GT, and **not** present in the
production mesh path.

## Evidence

**The montage meshes a binarized mask, not the SDF.** `scripts/foundations/baseline_gate_eval.py` computes
`occ = gen <= 0` and `real_occ = item["sdf"] <= 0` (lines 137/139), then meshes them with
`measure.marching_cubes(m.astype(np.float32), 0.5)` (line 162). Isosurfacing a 0/1 grid can only place
vertices at voxel-face midpoints → staircase terracing on every non-axis-aligned face, stamped identically
on the "real" and "generated" columns.

**Meshed correctly, the steps vanish.** Same GT sample, two ways:
- `steptexture_gt_binary_vs_sdf.png` — binary occ @0.5 (current montage) = heavy staircase; continuous
  SDF @0.0 (the true field, range −0.32…1.13, ~1400 unique values) = flat, clean walls.
- `steptexture_gen_binary_vs_sdf.png` — same for the generator's output: binary@0.5 = staircase;
  continuous@0 = steps gone.

**Only 3 scripts use the binary path — all diagnostics.** Of ~25 `marching_cubes` call sites, only
`baseline_gate_eval.py:162`, `diagnose_surface_roughness.py:101`, and `probe_buildingnet_solidity.py:115`
mesh binary@0.5. The **production/demo** path (`scene/run_demo.py`, `scene/sdf_primitives.py`, the AB-snap
server, and every other eval) already meshes `marching_cubes(sdf, 0.0)` — the continuous field — so the
deployed meshes never had this artifact.

## What this does and does not overturn

- **Corrects [#35](https://github.com/danvisai/SDFusion/issues/35)'s "NOT a render artifact."** #35 diagnosed
  roughness through its own binary@0.5 montages, conflating two things. Split correctly:
  - **(a) staircase steps** — a render artifact (removable by meshing continuous SDF@0).
  - **(b) mid-scale field waviness** — real, prior-side; survives correct meshing (see the lumpy right
    column of `steptexture_gen_binary_vs_sdf.png` vs. the GT's clean right column). This is the genuine
    crispness gap.
- **#27 gate metrics are unaffected** — footprint-IoU etc. are computed on the binary occupancy, so they are
  render-independent; map #24's numbers stand.
- **Phase-1 / Phase-2 visual verdicts** (`phase1-result.md`, `phase2-result.md`) were judged on artifact
  renders → their "not crisp" conclusions are superseded by a corrected-lens re-baseline. The gate metrics
  they report still stand.

## Consequence for the effort

"Remove the step texture" ≈ **fix the 3 diagnostic renderers to continuous-SDF@0** (free; production is
already correct). Doing so makes the eval montages honest and isolates the *real* remaining problem — the
residual field waviness (b) — which is what the re-chartered map now carries through to crisp.
