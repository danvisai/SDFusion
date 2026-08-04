# The band fix: it works, on 60% of buildings

Ticket [#80](https://github.com/danvisai/SDFusion/issues/80) · Map
[#69](https://github.com/danvisai/SDFusion/issues/69) · Run `logs_building/vecset_v5_surfband`

Controlled comparison against `vecset_v4_surf`: **same starting checkpoint, same 60,000 steps, same
loss weight, same query count.** One variable — `--surf_t_center` 0.40 → 0.55, moving the
decoded-surface grading from where the model has little to change to where inference actually runs.

## 🔑 The result is bimodal, and the median hides it

Final checkpoint, 48 pinned held-out ids, s=0.5:

| | count | `extra` | 3D IoU |
|---|---|---|---|
| **solid** (`missing` < 0.15) | **29/48** | **0.149** | **0.833** |
| **hollow** | 19/48 | 0.162 | 0.353 |
| *blockout (the input)* | — | *0.183* | *0.845* |
| *surface-loss model (`v4`)* | — | *0.191* | *0.838* |

Median `missing` is **0.051**; the mean is **0.244**. Reported as a median alone this reads as a clean
result.

**On the 29 buildings where it works it does the job this map has been chasing since #75**: `extra`
0.149 against the blockout's 0.183 — **19% of the surplus removed** — while holding 3D IoU at 0.833
against the blockout's 0.845. No previous model has produced a selective carve on any subset.

**On 19 buildings it guts them**, leaving hollow shells at IoU 0.353.

So this is not "the band fix is worse." It is **the band fix works and is unreliable** — a reliability
problem, not a capability one, and far more tractable than "the model refuses to act."

## ⚠️ The trajectory: two transient collapses, both recovered

| checkpoint | fp-IoU | missing | extra | 3D IoU |
|---|---|---|---|---|
| @190k | 0.912 | 0.777 | 0.135 | **0.195** |
| @220k | 0.903 | 0.773 | 0.145 | **0.200** |
| @230k | 0.934 | 0.027 | 0.189 | **0.825** |
| @240k | 0.954 | 0.051 | 0.158 | **0.737** |

At 220k the renders showed shredded cages and a stop was recommended. **The very next checkpoint
recovered to 0.825.** This is the second time in this effort a run was nearly killed during a transient
— the first was #75's 0.719 → 0.657 → 0.532 → 0.840.

🔑 **This model's training passes through deep multi-checkpoint failure phases and comes out of them.**
A 30,000-step window of catastrophic output is not evidence of a dead run here.

## ⚠️ Two measurement lessons

**Medians lie on bimodal outcomes.** `eval_massing_arms.py` reports medians, which is correct for
unimodal noise and actively misleading here. **It should report a collapse rate beside the median.**
The bimodality was caught only because a render showed one building solid and two hollow *in the same
checkpoint*.

**The aggregate can be flat while the geometry degrades.** Between 190k and 220k, 3D IoU went 0.195 →
0.200 — marginally "better" — while row 2 went from a recognisable box to a shredded cage. Numbers
alone would have reported no change.

## Renders

All in this folder, GT · input · surface-loss model · band fix, same buildings and camera throughout:

| file | what it shows |
|---|---|
| `band-fix-comparison.jpg` | @190k — the hollowing first appears |
| `band-fix-220000-comparison.jpg` | @220k — worse; the near-stop point |
| `band-fix-230000-comparison.jpg` | @230k — cages fill back in |
| `band-fix-240000-comparison.jpg` | @240k — **final: row 2 solid, rows 210/383 hollow** |

`band-fix-*-montage.png` are the full harness outputs (all arms) behind each.

## Next

**Find what separates the 29 from the 19.** If it correlates with something legible — footprint
complexity, building size, source corpus — that is a targeted fix rather than a blind weight sweep.
First investigation this map has had that specific.

Secondary: a lower `--surf_weight` at the same band, to see whether the collapses are simply the term
overshooting on the harder cases.
