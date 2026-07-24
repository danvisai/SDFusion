# Honest Surface-Fidelity Baseline

Execution record for [Honest surface-fidelity baseline + report](https://github.com/danvisai/SDFusion/issues/44)
(spec #42, map [#34](https://github.com/danvisai/SDFusion/issues/34)). Follows #39 (fixed the diagnostic
renderers to mesh the continuous SDF@0.0, not a binary-occupancy staircase) and #43 (factored that meshing
into the shared `mesh_sdf_surface` helper). This ticket re-runs the map-#24 checkpoint through the corrected
renderers at full statistical size and records what the *honest* montages actually show.

**Target:** the accepted map-#24 checkpoint
`logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth`, deployed config
(`--use_region 1 --use_extra_cond 0 --use_ema 1 --ddim 100`, guidance 1.0).

## 1. Montage helper (task 1)

The #27 gate montage in `scripts/foundations/baseline_gate_eval.py` already paired real-LoD2 vs generated
side by side, meshed via `mesh_sdf_surface` (continuous SDF@0.0, #43) — so this did **not** need a new eval
path. The only change: the inline plotting loop was factored into a standalone function,
`save_fidelity_montage(rows, path)` (`scripts/foundations/baseline_gate_eval.py`), so the same honest
GT-vs-generated montage is reusable from other scripts without re-implementing the loop. `main()` now just
calls it. This is exactly the "thin wrapper" case named in the ticket: the gate montage already fully serves
as the dedicated fidelity montage, so no new flag or rendering mode was added. Two CPU-only tests
(`TestSaveFidelityMontage` in `test_baseline_gate_eval.py`) pin that it writes a file for N rows and doesn't
crash on a row with no zero crossing.

## 2–3. #27 gate at n=60 (task 2, 3)

Run: `env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/baseline_gate_eval.py --ckpt logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth --use_region 1 --use_extra_cond 0 --use_ema 1 --ddim 100 --n 60 --tag honest`

| metric | value | threshold | pass |
|---|---|---|---|
| collapse rate | 0.0% | ≤1% | ✅ |
| LCC ≥ 0.90 fraction | 100% | ≥85% | ✅ |
| footprint-IoU median | 0.881 | ≥0.65 | ✅ |
| footprint-IoU p10 | 0.786 | ≥0.35 | ✅ |
| **OVERALL_SCALAR_PASS** | | | **✅ PASS** |

Per-region footprint-IoU median: NL 0.866 (n=17), DE 0.889 (n=29), JP 0.895 (n=14) — all comfortably above
the 0.65 bar, no region collapse.

**#27 PASSES at n=60**, consistent with the accepted map-#24 result (`baseline_gate_eval_lod2-final.json`:
median 0.888 / p10 0.799, a different random draw of the same checkpoint+config). This reconfirms what
`render-artifact-finding.md` already argued on inspection: the gate metrics are computed on binary occupancy,
not the mesh, so they are unaffected by the #39/#43 render fix — the honest re-run is a fresh sample draw
that lands in the same range, not a different regime.

Artifacts: `execution/artifacts/baseline_gate_eval_honest.json`, montage
`outputs/baseline_gate_eval/montage_honest.png` (6-building excerpt, real LoD2 left / generated right).

## 4. Non-gating diagnostics on the honest renders (task 4)

Run: `env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/diagnose_surface_roughness.py --n 6`

| metric | median (n=6) |
|---|---|
| IoU(VQVAE round-trip B, real GT A) | 0.995 |
| IoU(prior sample C, real GT A) | 0.576 |
| surf-ratio (boundary/occupied voxels), A | 0.159 |
| surf-ratio, B | 0.160 |
| surf-ratio, C | 0.138 |

Note on what "honest" changes here: `iou_*` and `surf_*` are computed on **binary voxel occupancy** directly
from the SDF (`baseline_gate_eval.lcc_frac`-style masks), not the mesh — so, like the #27 gate, they were
already render-independent; the #39/#43 fix only changes the **montage pixels**, not these scalars. The
previous `surface_roughness_diagnosis.json` on disk predated the render fix (written 2026-07-20, before
commit `6c1d551`); this run overwrites it with a fresh draw under the current (honest-render) code path. The
`iou_sample_gt` number moves run to run (0.48 → 0.58 here) because the DDIM sampler's own stochastic draws
aren't seeded in this script (unlike the gate harness or the GPU-smoke test) — expected n=6 variance, not a
render effect.

`surf_sample` (0.138) sits **below** `surf_gt`/`surf_roundtrip` (0.159/0.160) — the rough prior sample has a
*lower* boundary/volume ratio than the crisp GT, the opposite of what a naive "roughness raises surface
area" intuition would predict. This is a second, independent confirmation of #35/#36's finding that simple
scalar surface metrics do not track the visual roughness: #35 flagged this for surf-ratio directly ("does
not measure the roughness"); #36 found the same for two normal-consistency metrics. No new gating scalar was
added, per #36's standing decision.

Artifacts: `execution/artifacts/surface_roughness_diagnosis.json`, montages
`outputs/surface_roughness/ladder_montage.png` (A real GT / B VQVAE round-trip / C prior sample) and
`outputs/surface_roughness/slice_montage.png` (mid-height SDF field slices with the 0-level contour drawn).

## 5. Visual characterization

Viewed `montage_honest.png`, `ladder_montage.png`, `slice_montage.png`:

- **GT (A) is crisp.** Every real-LoD2 sample in both montages is a clean, flat-faced polyhedron — sharp
  right-angle corners, flat walls, straight roof edges. No waviness anywhere in the A column.
- **VQVAE round-trip (B) reproduces A almost exactly** — same flat faces, same sharp corners, visually
  indistinguishable from A at a glance (matches the 0.995 median IoU). This is a second, mesh-level
  confirmation of #35's finding that the codec is not the fidelity ceiling.
- **The prior sample (C) is not crisp — it is a rounded, lumpy, "rock-like" solid** in all 6 honest
  ladder-montage rows and all 6 gate-montage rows. Two distinguishable failure modes, both present together:
  - **Edges/corners are eroded, not sharp.** Every right-angle corner in A/B becomes a rounded shoulder in C;
    no generated sample shows a crisp edge anywhere.
  - **Faces bulge with mid-scale bumps and dimples**, not fine noise and not a single large deformation —
    roughly a handful of protrusions/pockmarks per face, at a scale clearly larger than one voxel and
    clearly smaller than the building itself ("mid-scale," consistent with #36's characterization). Some
    samples (gate-montage rows 1, 4, 6) show pitted, cratered tops; none show a flat roof plane.
- **The waviness is in the field, not the mesh.** The slice montage is the field-level check: A/B's 0-level
  contour is a clean straight-edged polygon; C's contour is visibly rippled along otherwise-straight edges
  and rounded at corners, and the coloring near the boundary is mottled rather than a clean gradient band.
  Because this is a raw 2D SDF slice (no marching cubes involved), this is direct evidence the roughness is
  a property of the sampled field itself, not an artifact of meshing at 0.0.
- **No obvious edges-only or faces-only pattern; it reads as a global field texture** — the same wobble
  character appears on top faces, side faces, and edges alike, rather than being concentrated at one
  geometric feature type.
- **Region / size correlation:** the montages don't show an obvious per-region difference in *roughness
  severity* (too few JP/NL samples land in the 6-building excerpts to judge visually). The only correlation
  found in the numbers is on the **gate** metric, not a crispness metric: footprint-IoU rises with building
  size across all three regions (corr(fp_iou, real_occ) ≈ 0.60 on the n=60 honest run; median fp_iou 0.83 for
  the smallest occupancy tercile vs 0.91 for the largest) and per-region fp_iou tracks each region's median
  size (NL 0.87 / DE 0.89 / JP 0.90, sizes rising in the same order). No scalar for waviness itself exists
  (per #36), so a genuine roughness-vs-height correlation can only be judged visually, and the n=6 excerpt is
  too small and too NL/DE-heavy to support a real claim either way — flagged as an open question, not
  resolved here.

**Visual sign-off (orchestrator, opus):** confirmed by direct viewing of all three montages. GT crisp;
VQVAE round-trip visually indistinguishable from GT (codec is *not* the ceiling); prior sample rounded/lumpy
with eroded edges + mid-scale face bumps; the C-field 0-contour is rippled and its surroundings mottled in
the raw slice — the waviness is field-level, not a mesh artifact. Every verdict above stands.

## Status of prior conclusions under honest rendering

| finding | verdict | why |
|---|---|---|
| #35 — roughness is prior-side (sampled SDF field), not codec/64³/render | **CONFIRMED**, more directly | the slice montage now shows the wavy 0-contour *in the raw field*, not just inferred from a rough mesh; B still reproduces A almost exactly |
| #35/#36 — no scalar geometry metric separates crisp from rough | **CONFIRMED**, extended | surf-ratio (this ticket) again fails to separate C from A/B, in the *same direction* as #35 originally found (C even scores lower/"smoother" than the crisp GT) |
| #27 gate metrics are render-independent, map-#24 numbers stand | **CONFIRMED** | n=60 honest re-run lands in the same range as `lod2-final` (0.881/0.786 vs 0.888/0.799) |
| render-artifact-finding.md — "(b) mid-scale field waviness is real, survives correct meshing" | **CONFIRMED** | this is exactly what the honest n=6/n=60 montages show |
| Phase-1 result — no sampling knob (EMA/DDIM/guidance) reaches crisp | **CONFIRMED in substance, not re-verified pixel-for-pixel** | the root cause (prior-side field noise) is now directly confirmed in the raw field, and no sampling-time knob changes a trained field's content, so the "falls short" verdict stands; but the specific comparative visual claims in `phase1-result.md` (e.g. "guidance amplifies striations") were judged on the pre-#39 binary-render montages and were **not** re-run here — re-verifying them under the honest renderer is future work, not part of #44's scope (the baseline, not a re-sweep) |
| Phase-2 result — grad_tv rounds edges / eikonal distorts off-footprint, both fail | **CONFIRMED via gate metrics (render-independent), visual character not re-checkable** | the numeric failure (fp-IoU 0.74 / 0.45, both computed on binary occupancy) is unaffected by the render fix and stands as-is; the fine-tuned checkpoints that produced those montages were deleted (`phase2-result.md`, ~100 GB reclaimed 2026-07-23), so their visual character cannot be re-rendered honestly even if desired |
| Map closure "accept + defer" (`723f2eb`, then superseded by the re-chart `e88698f`) | **superseded (already noted in the map memory)**, not reopened by this ticket | this ticket only establishes the honest baseline (map-#24 unchanged); whether to pursue a new crispness lever (edge-aware objective / different codec, per phase2-result.md's option 1/2) is a downstream decision, not made here |

## Bottom line

The honest renderer does not change the substance of the diagnosis: **map-#24 is solid and footprint-matching
(#27 PASS at n=60) but its surfaces are genuinely wavy** — rounded edges plus mid-scale bumps on faces, visible
directly in the raw SDF field, not manufactured by the old binary-mesh render bug. The old montages
over-dramatized the defect (fake staircase on top of the real wobble); the underlying crispness gap itself is
real and unchanged. No new gating scalar was introduced. This baseline is the hand-off for the next ticket
(choosing a crispness lever beyond the already-exhausted Phase-1/Phase-2 options) — not attempted here.

## Artifacts produced by this ticket

- `execution/artifacts/baseline_gate_eval_honest.json` — n=60 gate result + per-building rows
- `outputs/baseline_gate_eval/montage_honest.png` — GT-vs-generated fidelity montage (6-building excerpt)
- `execution/artifacts/surface_roughness_diagnosis.json` — n=6 A/B/C ladder metrics (honest re-run)
- `outputs/surface_roughness/ladder_montage.png` — A/B/C mesh montage (honest render)
- `outputs/surface_roughness/slice_montage.png` — A/B/C mid-height SDF field-slice montage

Git-durable copies of the three montages (the repo otherwise gitignores `*.png`) are committed alongside this
report: `honest-gate-montage.png`, `honest-ladder-montage.png`, `honest-slice-montage.png`.
