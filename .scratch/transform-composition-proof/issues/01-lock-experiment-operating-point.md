# Lock the Experiment Operating Point

Type: grilling
Status: resolved
Blocked by:

## Question

Which fixed operating contract should every arm use: 96^3 or 128^3 working resolution, what integer
voxel width fixes `s*` at approximately 0.5 m before observing results, and whether the monolith's
coarse input is low-pass real SDF only or a declared low-pass/footprint-extrude variant comparison?
Record the choice and its fairness rationale without tuning it to downstream outcomes.

## Comments

## Answer

Recorded in **`docs/adr/0004-experiment-operating-point.md`**.

- **Working resolution: 96³** for all arms (matches the existing detail realization — composer /
  `detail_cube_volume` / bake all run at `res=96`). A single **pre-registered preflight** may promote
  *every* arm together to 128³ iff a 96³ montage shows the fixed facade categories are unrepresentable;
  mixed-resolution headline comparisons are prohibited.
- **`s*` = 1.0 m = 5 voxels @96³** (≈3 voxels @64³). Metric basis: cube `[-1,1]³` covers `center ± scale`
  m → `voxel = 2·scale/(res−1)` ≈ 0.21 m/vox @96³, 0.32 m/vox @64³ for a ~20 m building. **Fairness
  rationale (mechanism, not outcome):** tied to the 64³ massing generator's own resolution limit
  (features below ~3–4 voxels ≈ 1 m are unrepresentable). This **corrects the earlier loose ≈0.5 m**,
  which sat below the massing-grid limit and would have mis-classified ~1.5 m window openings as massing.
- **Monolith coarse input:** low-pass of the same building's real SDF is **primary**; footprint-extrude
  is a **declared robustness variant**, never silently swapped in.

Propagated the corrected `s*` to `CONTEXT.md`, ADRs 0001/0002, and the execution plans. Unblocks
tickets 05 (FID harness at 96³) and 06 (coincidence test against the fixed `s*`).
