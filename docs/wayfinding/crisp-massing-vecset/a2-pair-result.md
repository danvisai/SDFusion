# Aligned-pair training — the sixth negative, and the point to stop

**Date:** 2026-07-29 · 6,000 steps, 49.4M params, 34,909 aligned blockout→real pairs, n=16 held-out.

| arm | fp-IoU | 3D IoU |
|---|---|---|
| **blockout (input)** | **1.000** | **0.840** |
| projected s=0.35 | 0.068 | 0.006 |
| **projected s=0.5** | 0.854 | **0.611** |
| projected s=0.65 | 0.625 | 0.259 |
| *map-#24 deployed* | *0.863* | *0.601* |

**Pre-registered success condition: beat 0.840. Result: 0.611. This fails.**

## What did improve

Pair training fixed what it was meant to fix. The previous model shredded blockouts into vertical slats
at every usable strength; this one produces **coherent building-like masses**. The distribution-shift
diagnosis was correct and the fix worked.

## What did not

It **degrades its input** — 0.840 → 0.611 — and has a razor-thin working band: 0.35 and 0.65 both
collapse, only 0.5 functions. The render (`a2-pair-comparison.png`) shows why the number flatters it:
at s=0.5 the mass is coherent but eroded and pitted; at s=0.65 it falls apart into slabs.

The generator is not *refining* the blockout. It is **replacing it with its own guess** — and that
guess is no better than the model we already ship.

## 🔑 The conclusion this licenses

**Two architectures that could hardly be more different land in the same place.**

| stack | representation | 3D IoU |
|---|---|---|
| map-#24 | 64³ dense grid, 947M params | **0.601** |
| A2 | 2048-token vecset, 49.4M params | **0.611** |
| *blockout* | *no model at all* | ***0.840*** |

A dense voxel grid and a query-based token set, different codecs, different objectives, different
parameter counts — both converge on ~0.60, and a footprint extrusion with no network beats both by a
wide margin.

**The representation was never the bottleneck.** That is the substantive finding of this effort, and it
is worth more than another training run would be. The whole #52 → #58 → #61 chain was premised on the
dense grid being the limiting factor; the vecset rebuild tested that premise directly and it did not
hold.

## Recommendation: stop the vecset thread

Six negatives, and each fix has revealed the next problem rather than closing the gap. The pattern is
not "nearly there" — it is a direction that keeps failing for different reasons, which is what a wrong
premise looks like.

What survives and stays useful:
- **the surface corpus** (35,623 recovered, verified) — reusable by anything
- **the codec contract and sampler seams**, with 23 tests, and both codecs satisfying one suite
- **the measurement discipline** — control arms, the blockout baseline, montages over scalars
- **the blockout result itself**, already carried to [#50](https://github.com/danvisai/SDFusion/issues/50):
  it beats the shipped generator today, with no model

What the evidence points at instead: the gap between blockout (0.840) and GT (1.000) is **entirely
over-fill** — 0% missing, +21.7% extra — caused by extruding one height across a plan that has several.
That is a much narrower problem than "generate a building", and it does not obviously need a 3D
generative model at all.
