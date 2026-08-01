# Convergence-scale training: the result

Ticket: [Train the aligned-pair generator to convergence](https://github.com/danvisai/SDFusion/issues/75)
· Map: [#69](https://github.com/danvisai/SDFusion/issues/69)

| what | where |
|---|---|
| checkpoints | `logs_building/vecset_v2_pair`, `logs_building/vecset_v2_plain` |
| artifact | `execution/artifacts/massing_arms_eval_pair60k.json` |
| montage | `convergence-run-montage.png` (this folder) |
| probe | `scripts/foundations/probe_vecset_checkpoint.py` |

Two arms, 60,000 steps each (**13.8 epochs**, ~10× the void runs' 1.4), batch 8, LR 1e-4 — deliberately
unchanged so the variable under test was *length*, not length-and-everything-else.

## The answer, on 48 pinned held-out ids

| arm | fp-IoU | missing | extra | 3D IoU |
|---|---|---|---|---|
| gt | 1.000 | 0.000 | 0.000 | 1.000 |
| codec ceiling | 0.997 | 0.000 | 0.001 | 0.999 |
| **blockout (do nothing)** | **1.000** | 0.000 | 0.183 | **0.845** |
| **a2 pair, s=0.5** | **0.910** | **0.033** | **0.214** | **0.801** |
| deployed map-#24 | 0.812 | 0.047 | 0.476 | 0.623 |

**It beats the shipped model on every criterion** — footprint 0.910 vs 0.812, over-fill less than half
(0.214 vs 0.476), 3D IoU 0.801 vs 0.623 — at **49M parameters against ~947M**. And per criterion 1, the
montage shows it **reads as a building**: correct massing, correct footprint, box-like form, with thin
spurious flanges on the flanks that are exactly what `extra 0.214` measures. The deployed arm beside it
is an unrecognisable melted lump.

**It still loses to the trivial extrusion** (0.801 vs 0.845, and footprint 0.910 vs a perfect 1.000).
The map's bar is not cleared.

## 🔑 The trajectory is non-monotonic, and that nearly cost us the result

Peak 3D IoU by checkpoint (n=10 probe, matched grids and seeds):

| epochs | 2.3 | 4.6 | 6.9 | 13.8 |
|---|---|---|---|---|
| peak 3D IoU | 0.719 | 0.657 | 0.532 | **0.840** |
| at strength | 0.5 | 0.6 | 0.6 | 0.5 |
| `missing` at peak | 0.039 | 0.234 | 0.346 | 0.029 |

The middle three points are a clean monotonic decline with a legible mechanism — the model learns to
carve the blockout's surplus and progressively **over-carves**, eating the building. On that basis a
stop was recommended at 6.9 epochs. **That would have been wrong.** The dip is transient: the model
then learns to calibrate the carving, and `missing` collapses from 0.346 back to 0.029.

⚠️ **Do not extrapolate this training curve.** Three monotonic points here did not predict the fourth.
The dip spans 2.3→6.9 epochs, which is wider than most patience windows would survive.

(The n=10 probe reads 0.840; the full 48-id harness reads 0.801. The harness is the authority — the
probe is a cheap tracker, not a verdict.)

## Plain vs aligned pairs: pairs, decisively

[#70](https://github.com/danvisai/SDFusion/issues/70) voided the previous comparison, because pairs
only "won" by learning the axis swap. Re-decided here on correctly-framed data: the **plain** arm
projects a blockout into a hollow shell (at 2.3 epochs, `missing` 0.936 at s=0.35 and 0.971 at s=0.5 —
3D IoU 0.062 and 0.028). Pairs are not a marginal preference; plain does not do the task at all.

## What this says about the latent metrics

At 13.8 epochs `cos(recovered, true)` is **0.9403** at s=0.5 — nowhere near
[#73](https://github.com/danvisai/SDFusion/issues/73)'s 0.999 decode tolerance — while 3D IoU is 0.801.
More confirmation that **latent cosine does not predict decode quality**
([#76](https://github.com/danvisai/SDFusion/issues/76): Spearman +0.12 pooled). The pair loss was
likewise flat from 10k on (0.4701 → 0.4586) across the span where task performance fell to 0.532 and
then rose to 0.801. **Neither the loss nor the cosine tracked the thing we care about.** The harness
did.

⚠️ The narrow #76 finding stands — the objective can't rank its own candidates. The *practical*
inference drawn from it, that longer training would not help, is **refuted**.

## Where this leaves the map

- The generator is now the **best we have by a wide margin**, and the first to produce output a human
  reads as a building — but it has not beaten doing nothing.
- The gap is **0.044 in 3D IoU and 0.090 in footprint**, and the last 30k steps moved it +0.269. The
  curve was still climbing steeply at the point we stopped.
- Two live levers, not mutually exclusive: **train further** (cheapest, and the trajectory supports it),
  and the **decoded-surface loss** ([#80](https://github.com/danvisai/SDFusion/issues/80), implemented
  and smoke-green), which targets the flanges directly — they are exactly "a latent that decodes wrong"
  and the ε-loss is blind to them.
