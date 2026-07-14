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
style of session, not an identical list). Applied to real, held-out (Stage3a-clean) BuildingNet
buildings loaded at native 64³ resolution — **not** the deterministic procedural "modern" recipe
box the first version of this sweep used (see the first "Update" note below for why; a second
"Update" further down scales this from one base building to all 27). Snapped via
`Refiner.snap_volume` called **in-process** — this is the exact function `/snap_sdf` calls
(`inference_service.py:snap_sdf`), not a re-implementation, avoiding the need for a separately
running server (matches tickets 07/09/11's in-process convention). Every other parameter is left
at production's own default (`steps=8`, `autoguidance=True`, `auto_scale=2.0`, `local=True`) so
the sweep exercises the live sculptor's actual deployed behavior, not a hand-picked configuration.

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

**Update (2026-07-13, later, user-directed): increased the sample count for a trustworthy FID
reading.** One base building gave only 3 distinct generated shapes (18 pooled images) — nowhere
near ticket 05's established `fid.undersampled` floor (N per arm ≥ 2048-d). Rebuilt the sweep
around two levers instead of one:

- **Generated side:** every one of the 27 Stage3a-clean held-out buildings (`all_base_building_ids()`)
  × the 3 canonical edits = **81 distinct shapes** — the actual ceiling of this project's
  leakage-safe test population crossed with its canonical edit vocabulary; no more of either
  exists without inventing a new selection policy or a new edit vocabulary, neither of which this
  ticket does. `N_VIEWS` raised to 28 so 81 × 28 = 2268 pooled images clears the raw 2048
  threshold.
- **Real reference side:** has no such ceiling — it's ground truth, never fed to the model — so it
  now draws 200 buildings from `data/splits_v1/train_100.json` (ticket 03's frozen, leakage-audited
  TRAIN split, verified 0-overlap with `test.json`) instead of the 8-building test-side tier,
  giving 200 genuinely distinct real shapes (5600 images) rather than more views of few.

Smoke-tested at `--n-bases 2` first (30 samples, 0 failures) before committing to the full run.
Full run: **405 samples (81 shapes × 5 strengths), 0 failures.**

| strength | mean IoU-to-edit (min–max across 81 shapes) | facade FID vs. real | n_generated / n_real | undersampled |
|---|---|---|---|---|
| 0.1 | 0.828 (0.002–0.994) | 66.7 [73.9, 87.8] | 2268 / 5600 | **false** |
| 0.3 | 0.725 (0.0001–0.954) | 64.2 [70.7, 86.5] | 2268 / 5600 | **false** |
| 0.5 | 0.627 (0.0004–0.906) | 69.2 [75.2, 92.4] | 2268 / 5600 | **false** |
| 0.7 | 0.521 (0.0–0.853) | 72.3 [77.1, 96.1] | 2268 / 5600 | **false** |
| 0.9 | 0.454 (0.0–0.826) | 69.4 [73.7, 93.0] | 2268 / 5600 | **false** |

Faithfulness declines monotonically in aggregate across the full 81-shape population (0.828 →
0.454), a cleaner and more defensible trend than either single-building run gave. `undersampled`
now reads **false** for every strength — the raw N > 2048-d check this codebase has used since
ticket 05 finally clears.

**But this is not simply "now trustworthy" — two things surfaced that a smaller run couldn't have
shown, and both are disclosed rather than smoothed over:**

1. **The point estimate falls *outside* its own 95% bootstrap CI at every single strength** (e.g.
   `s=0.1`: point 66.7 vs. CI [73.9, 87.8] — point below the *lower* bound). This exact symptom is
   ticket 05's own documented red flag for small-effective-sample FID bias — except ticket 05 saw
   it while *undersampled*, and this run isn't. The reason: `bootstrap_fid_ci`'s group-aware
   resampling (deliberately built so correlated camera views of one shape aren't double-counted as
   independent samples) resamples at the *shape* level, not the image level — so the bootstrap's
   effective sample size is still 81 groups (generated) / 200 groups (real), regardless of how
   many views were rendered per shape. Raising `N_VIEWS` to clear the raw 2048-image threshold
   changed the image count but not the number of independent groups the bootstrap actually
   resamples over — confirmed by re-checking the smoke test's own 6-shape/10-building run, where
   every point estimate *did* fall inside its CI (no anomaly at the smaller, honestly-undersampled
   scale). **Conclusion: raw image count cleared the codebase's established check, but the true
   bottleneck — independent shape/building count, both still far short of Heusel et al.'s
   recommended ≥10,000 — did not, and the FID point estimates here should be read as directional
   only, not calibrated.**

2. **56% of the entire Stage3a-clean held-out population (15 of 27 buildings) has occupancy below
   0.5%, median 0.31%** — a more systematic characterization than ticket 09's own note that
   occupancy ranges "0.02%–6.4%, no natural gap separates broken from valid" (true, but that
   phrasing doesn't convey that more than half the population sits at the sparse end).
   Visually (`outputs/sculpt_strength_sweep/montage.png`, e.g. `COMMERCIALhotel_building_mesh0162`
   at 0.10% occupancy), these buildings' "base" thumbnail is barely a few scattered fragments —
   for `tower`/`dome`, the snapped output is essentially just the added primitive, disconnected
   from the barely-there base; for `carve`, it degenerates into small fragmented glitch-blobs,
   which is why several `min_iou_to_edit` values above sit at 0.0. Per this project's own
   established policy (ticket 09: disclose, don't filter, since "no natural gap separates broken
   from valid" is itself an a-posteriori judgment call), these are **not excluded** here either —
   but their prevalence (a majority of the population, not a handful of outliers) likely
   contributes to both the wide faithfulness range and, plausibly, some of the FID/bootstrap
   anomaly above (a bimodal generated population — well-formed buildings vs. near-empty
   fragments — is a harder distribution for a 2048-d covariance estimate to summarize than a
   homogeneous one). Whether to separate these two regimes in future analysis, or continue
   reporting them pooled, is a call for the user, not made unilaterally here.

Assets: `outputs/sculpt_strength_sweep/montage.png` (3 of the 27 base buildings × 3 cases × 5
strengths, `--montage-bases 3` — all 27 are swept for the statistics above, only 3 are shown
visually), `outputs/sculpt_strength_sweep/faithfulness_vs_realism.png` (the trade-off curve, now
over the 81-shape aggregate), `execution/artifacts/sculpt_strength_sweep.json` (full manifest:
`base_building_ids`, `n_distinct_shapes=81`, per-sample `base_occ_frac`, git provenance).

**Left for the user's review (this ticket's own bar — a prototype to react to, not a decided
result):** the faithfulness side now behaves as one coherent knob spanning generation (ticket 09)
and editing (here) across a real, defensibly-sized population. The realism side cleared this
codebase's own raw sample-size check but revealed a deeper one (bootstrap effective-N, not just
raw image count) that a real fix would need ~10,000+ independent shapes/buildings to fully
resolve — likely more investment than a prototype ticket should spend unilaterally. The
newly-quantified 56%-sparse characteristic of the shared held-out population is also worth
flagging beyond this ticket, since other tickets (09, and any future one reusing this same
27-building tier) rest on the same population.
