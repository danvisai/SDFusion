# Prototype the Sculpt Strength Sweep

Type: prototype
Status: artifact built (2026-07-13) — awaiting user review (see Answer)
Blocked by: 01, 05

## Question

Create a small, concrete C1b strength-sweep artifact over representative crude edits using the live
`/snap_sdf` operator, plotting edit faithfulness against neutral-render realism. Review the cases with
the user to decide whether the sweep demonstrates one operator spanning generation and editing or
whether the case contract needs revision before a full run.

## Comments

## Answer

**Built:** `scripts/eval/sculpt_strength_sweep.py` (TDD, 7 contract tests for the pure aggregation
seam — `summarize_by_strength` — plus data-shape checks on `EDIT_CASES`/`STRENGTHS`; the
snap/render/FID pipeline itself is GPU/checkpoint-dependent and verified by the real run below,
matching this project's established convention for that kind of code, e.g. tickets 05/09).

**Method:** three canonical crude edits — `tower` (add box), `dome` (add sphere), `carve`
(subtract box) — byte-identical to `eval_harness.py`'s own `EDITS`, this codebase's established
single-op "representative sculpt edit" vocabulary (`sculpt_regression.py`'s `CASES` compose the
same primitives but pair its dome with an extra wing box in one compound case, so it's the same
style of session, not an identical list). Applied to `BASE_BUILDING_ID = "PUBLICcity_hall_mesh0451"`,
a real, held-out (Stage3a-clean) BuildingNet building loaded at its native 64³ resolution — **not**
the deterministic procedural "modern" recipe box the first version of this sweep used (see the
"Update" note below for why). Snapped via `Refiner.snap_volume` called **in-process** — this is
the exact function `/snap_sdf` calls (`inference_service.py:snap_sdf`), not a re-implementation,
avoiding the need for a separately running server (matches tickets 07/09/11's in-process
convention). Every other parameter is left at production's own default (`steps=8`,
`autoguidance=True`, `auto_scale=2.0`, `local=True`) so the sweep exercises the live sculptor's
actual deployed behavior, not a hand-picked configuration.

Swept `STRENGTHS = [0.1, 0.3, 0.5, 0.7, 0.9]` — wider than `eval_harness.py`'s fixed 3-point
`[0.3, 0.5, 0.7]` regression-tracking convention, since this ticket is the dedicated full-range
sweep. **Caught before the final run:** `Stage3aModel.sdedit`'s only source of randomness is one
unseeded `torch.randn_like` noise draw inside `q_sample` (DDIM reverse steps are themselves
deterministic at `ddim_eta=0.0`) — without controlling it, each `(case, strength)` sample carried
an independent noise draw, confounding strength with per-call noise variance and undermining the
ticket's own "strength is the controlled variable" bar. Fixed by resetting
`torch.manual_seed(seed)` immediately before every `snap_volume` call (same latent shape every
time regardless of case/strength → bit-identical raw noise draw), isolating strength as the sole
swept variable.

Faithfulness = `iou_to_edit`, the IoU `snap_volume` already returns between its output and the
pre-snap edited input (this project's established "did the snap keep what the user placed"
metric). Realism = neutral-facade FID (CONTEXT.md: detail fidelity is measured
distributionally) against a fixed real BuildingNet reference population (ticket 09's held-out
"clean" tier, 8 buildings, reused rather than re-derived), rendered through ticket 05's shared
neutral shader. Failures at any stage (snap, render) are caught, logged, and shown as an explicit
"FAILED" cell in the montage rather than silently omitted; a curve-plot point built from fewer
than the full 3-case set would be labeled `(n/3)` (not triggered this run — 0 failures).

**First result (procedural box base), 15 samples, 0 failures:**

| strength | mean IoU-to-edit (faithfulness) | facade FID vs. real (realism, undersampled) |
|---|---|---|
| 0.1 | 0.995 | 319.5 [292.7, 370.2] |
| 0.3 | 0.973 | 319.7 [292.7, 370.6] |
| 0.5 | 0.937 | 319.6 [292.7, 370.4] |
| 0.7 | 0.903 | 319.1 [292.3, 369.5] |
| 0.9 | 0.886 | 318.6 [291.8, 369.0] |

Faithfulness declined monotonically but only modestly (0.995→0.886), and FID was essentially flat
(every point inside every other point's own CI) — both consistent with the box itself: a plain
rectangular mass is already close to on-manifold, so the sculptor has little to do at any
strength, and there's no facade complexity for a strength change to visibly affect.

**Update (2026-07-13, same day, user-directed): swapped to a real complex base building.** The
box gave a technically-valid but visually uninformative sweep — reviewing the first montage, the
shapes stayed nearly identical across all five strengths. Re-ran with `PUBLICcity_hall_mesh0451`
(a real, asymmetric, notched civic building — already flagged in ticket 09's own qualitative
findings as visually distinctive) as the fixed base instead, excluded from the FID reference
population so it isn't compared against itself. Same seeding, same edits, same 15 samples, 0
failures:

| strength | mean IoU-to-edit (faithfulness) | facade FID vs. real (realism, undersampled) |
|---|---|---|
| 0.1 | 0.961 | 247.8 [219.8, 320.6] |
| 0.3 | 0.877 | 237.5 [208.4, 317.9] |
| 0.5 | 0.795 | 231.2 [204.0, 311.5] |
| 0.7 | 0.712 | 238.8 [213.9, 312.1] |
| 0.9 | 0.653 | 234.1 [214.4, 305.9] |

Faithfulness now spans a much wider, still-monotonic-per-case range (roughly 0.96→0.62-0.70 across
the three cases individually — a 30-35 point spread vs. the box's 11), confirming the flat box was
suppressing the effect, not the strength knob itself. FID also moves more (247.8 at `s=0.1` down
to a minimum around 231 at `s=0.5`, back up slightly at 0.7/0.9) — visually, the montage shows the
snapped tower/dome/carve additions genuinely reshaping and blending into the building's own
massing as strength rises, not just sitting inertly on top of it as with the box. Still
`fid.undersampled=true` at every point (same 18 generated vs. 48 real images, unchanged sample
size), so the non-monotonic FID dip at `s=0.5` should be read as noise, not a discovered optimum,
until this exact harness is rerun at a larger view/case count.

Assets (both correspond to the complex-base result; the box-base montage/curve were overwritten,
not archived separately): `outputs/sculpt_strength_sweep/montage.png` (all 3 cases × 5 strengths,
visual), `outputs/sculpt_strength_sweep/faithfulness_vs_realism.png` (the trade-off curve),
`execution/artifacts/sculpt_strength_sweep.json` (full manifest + git provenance, complex-base
result only).

**Left for the user's review (this ticket's own bar — a prototype to react to, not a decided
result):** with a real, complex base, the faithfulness side clearly behaves as one coherent knob
spanning generation (ticket 09) and editing (here) — strength visibly trades against the placed
edit now, not just on paper. The realism side is genuinely untested at this sample size — the open
question is whether it's worth scaling this exact harness's view/case count past ticket 05's
undersampling floor before treating any strength (e.g. the `s=0.5` dip) as "more realistic," or
whether that investment belongs behind the paused C2 thread instead (see map.md's 2026-07-12
pivot).
