# Experiment operating point (ticket 01)

**Status:** accepted (2026-07-10)

Locks the fixed operating contract every arm (real, monolith, decomposition) uses — chosen **before**
observing results so it can't be tuned to outcomes. Resolves ticket 01.

## Working resolution — 96³
All arms render/realize at **96³**, matching the existing detail realization (composer /
`detail_cube_volume` / bake all run at `res=96`). A single **pre-registered preflight** may promote
*every* arm together to 128³ **iff** a 96³ montage shows the fixed facade-detail categories are
unrepresentable; that promotion is recorded before any headline result. Mixed-resolution headline
comparisons are prohibited.

## `s*` (massing/detail boundary) — 1.0 m = 5 voxels @96³
Fixed a priori at **`s*` = 1.0 m**, expressed as an integer voxel width: **5 voxels on the 96³ eval
grid** (≈ 3 voxels on the 64³ massing grid).

- **Metric basis:** the SDF cube `[-1,1]³` covers `center ± scale` m (`scale` = building
  half-max-extent), so `voxel = 2·scale/(res−1)` — for a ~20 m building ≈ **0.21 m/voxel @96³**,
  ≈ **0.32 m/voxel @64³**.
- **Rationale (mechanism, not outcome):** `s*` is tied to the resolution limit of the *massing
  generator itself* — the 64³ Stage 3a grid cannot reliably represent features below ~3–4 voxels
  ≈ 1 m. "What generation cannot resolve" is thus defined by the generator's own Nyquist-scale limit,
  not by where we want the answer. This **corrects the earlier loose "≈0.5 m"** (below the massing-grid
  limit, which would have mis-classified ~1.5 m window openings as massing). Ticket 06 then genuinely
  tests whether the semantic-detail categories fall below this fixed `s*` (and can fail).

## Monolith coarse input — low-pass primary
Primary coarse input is a **low-pass transform of the same building's real SDF** (keeps source and
target aligned). **Footprint-extrude** is a *declared robustness variant*, reported as a labeled
comparison — never silently swapped into the primary.

## Consequences
Updates the `s*` value in `CONTEXT.md` + ADRs 0001/0002 + the `execution/` plans from ≈0.5 m to
**1.0 m (5 vox @96³)**. Unblocks ticket 05 (FID harness at the locked resolution) and ticket 06
(coincidence test against the fixed `s*`).
