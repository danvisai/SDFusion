# Acceptance Gate: Solid, Footprint-Matching Massing

Resolves ticket [Define the acceptance gate for solid, footprint-matching massing](https://github.com/danvisai/SDFusion/issues/27). This is the destination's testable form — every later ticket is built to satisfy it.

**Evaluated on** generated held-out output over the Stage3a-clean population, **aggregated across all corpora**, using the **building-only footprint** (per [#32](https://github.com/danvisai/SDFusion/issues/32)). The scalar floors are *necessary*; the visual montage is the *sufficient / final* arbiter.

## The gate

| # | Criterion | Floor | Basis |
|---|---|---|---|
| 1 | **Collapse** | ≤ **1%** of outputs near-empty (`gen_occ < 1e-4`) | small tolerance band, was 26% |
| 2 | **Anti-fragmentation** | **≥ 85%** of outputs have largest 6-connected component **≥ 0.90** of occupancy | calibrated vs real targets (median LCC 0.945) |
| 3 | **Footprint-IoU** (building-only, footprint-axis) | **median ≥ 0.65** and **p10 ≥ 0.35** | current C1 baseline median 0.607 / p10 0.295 → modest improvement demanded |
| 4 | **Aggregation** | pooled metrics gate; **per-corpus values reported as diagnostics** (non-gating) | mixed-style training preferred; solidity guarantee is corpus-uniform |
| 5 | **Visual** | neutral-render montage sign-off — outputs read as "solid block + roof matching footprint" | project is judged visually; **FID excluded** |

**Dropped:** fill-vs-envelope ratio — it penalizes *shape* (a tower/pitched roof leaves envelope empty even when perfectly solid), scoring real solid targets only 0.23 median. It measures "not a rectangular prism," not "hollow."

**FID excluded** from the massing gate: the C2 kill-gate showed FID scores the *fragmentary* monolith as more realistic than the *building-like* decomposition (visual-contradiction limitation); FID measures detail realism, not massing solidity.

## Calibration (`execution/artifacts/massing_gate_calibration.json`)

Real solidified building-only targets (70 clean held-out): largest-connected-component fraction p10 0.13 / p25 0.45 / **median 0.945**; footprint-IoU (C1 baseline, `transform_vs_noise.json`) median 0.607 / p10 0.295 on the 27 Stage3a-clean buildings.

## Implementation notes (feed the retrain hand-off #30)

- **The ≥85% LCC bar is a target, not a current pass.** Only ~55–60% of *current* real solidified targets clear LCC ≥ 0.90 — the keep-centered solidify (#28) leaves a fragmented tail. The solidify **must be strengthened** so ≥85% of real targets (and then outputs) clear it. This is the gate's main forcing function.
- Footprint-IoU measured against the **building-only footprint** (#32), not the raw stored footprint.
- **Style diversity across corpora** (user preference) is a data-mix input to the retrain recipe (#30).
