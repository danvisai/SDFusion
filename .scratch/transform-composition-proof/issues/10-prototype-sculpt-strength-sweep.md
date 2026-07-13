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
style of session, not an identical list). Applied to a deterministic procedural "modern" recipe
base building (no recipe-diffusion sampling, so bit-identical across runs) via
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

**Result over 15 samples (3 cases × 5 strengths), 0 failures:**

| strength | mean IoU-to-edit (faithfulness) | facade FID vs. real (realism, undersampled) |
|---|---|---|
| 0.1 | 0.995 | 319.5 [292.7, 370.2] |
| 0.3 | 0.973 | 319.7 [292.7, 370.6] |
| 0.5 | 0.937 | 319.6 [292.7, 370.4] |
| 0.7 | 0.903 | 319.1 [292.3, 369.5] |
| 0.9 | 0.886 | 318.6 [291.8, 369.0] |

Faithfulness declines **monotonically** with strength across all three edit cases individually
(per-case range roughly 0.996→0.87-0.89), confirming the strength knob does trade against the
placed edit as expected. The decline is modest (not down toward 0 even at `s=0.9`) — expected
under the live sculptor's default `local=True` blending, which forces the untouched majority of
the volume back to the crisp pre-snap composed shape regardless of strength (see
`refine.py:snap_volume`'s own docstring), so this ceiling is a property of the deployed operator's
default, not a sweep artifact. FID is **flat** across strength — every point falls inside every
other point's own bootstrap CI — but this is honestly inconclusive rather than a null result: at
3 cases × 6 views = 18 pooled generated images (48 real) in a 2048-d feature space,
`fid.undersampled=true` fires for every point (ticket 05's own established floor), so no realism
signal, present or absent, can be trusted at this scale.

Assets: `outputs/sculpt_strength_sweep/montage.png` (all 3 cases × 5 strengths, visual),
`outputs/sculpt_strength_sweep/faithfulness_vs_realism.png` (the trade-off curve),
`execution/artifacts/sculpt_strength_sweep.json` (full manifest + git provenance).

**Left for the user's review (this ticket's own bar — a prototype to react to, not a decided
result):** the faithfulness side of the operator behaves as one coherent knob spanning generation
(ticket 09) and editing (here). The realism side is genuinely untested at this sample size — the
open question is whether it's worth scaling this exact harness's view/case count past ticket 05's
undersampling floor before treating any strength as "more realistic," or whether that investment
belongs behind the paused C2 thread instead (see map.md's 2026-07-12 pivot).
