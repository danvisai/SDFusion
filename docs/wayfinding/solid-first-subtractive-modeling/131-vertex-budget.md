# #131 — What vertex budget does a program need, and what does a program really cost?

*Effort: solid-first semantic architectural carving. Opened 2026-08-30 from
[#4](4-edit-algebra.md) and [#128](https://github.com/danvisai/SDFusion/issues/128), both of which
flagged the polygon budget and neither of which started it. Run and written 2026-08-30. No
training, no GPU.*

> A `Layer` and a `Ramp` each own a polygon, and every polygon in the recovered corpus is an exact
> voxel-boundary ring with a median of 94 vertices. A 94-vertex ring is a raster trace, not an
> architectural region. What budget does a program need before the region stops being a raster and
> starts being a polygon — and what does that cost in fidelity?

**The answer is 58, and it is a floor rather than a dial.** Cutting every polygon back until one
more deletion would move a cell takes the median region from **94 vertices to 58** and a median
program from **578 DSL tokens to 342** — a 41% saving for **no geometry change at all**: `extra`
0.0030, `missing` 0.0000, the same worst column, the same building to the voxel. Every budget below
that is a fidelity trade, and 🔑🔑 **the trade is far worse than the median `extra` says**. At a
24-vertex cap the median `extra` (0.0182) is still inside the project's 0.02 allowance while
**89.5% of buildings grow a column 15 voxels proud of GT** — five times `s*`. The surplus a budget
buys does not spread out over the roof. It stands up as spikes.


## The measurement

`recover_massing_programs.py --vertex_budget [--budget_montage V]`. It reads the **existing**
`execution/artifacts/program_recovery_714.json`, re-runs no fit and never rewrites it, and takes
57 s on 62 cores. Artifact: `..._vertex_budget.json`; montage
`outputs/program_recovery/vertex_budget.png`.

Every row is scored on the pinned **411 carve-needing** buildings (#126) — an already-flat building
has no operation, so no polygon, so nothing a budget could change; pooling the other 303 in would
dilute every column with rows that are identical by construction. The exact-ring control is **not
read back from the recovery artifact**: it goes through the same `replay_program` as every arm, and
it reproduces #10's recorded `extra` 0.0030 exactly, which is one more check that the artifact is a
program and not a result (#128).

### One greedy, three rules, and they differ in one test

Least-area vertex deletion (Visvalingam) run over **every ring of a region at once**, so the budget
is a per-region total and a hole competes with the outer boundary for it. The rules differ only in
which deletions are admitted:

| rule | admits a deletion when | so |
|---|---|---|
| `contained` | every cell it would **add** is already in the exact region | the region can only shrink: `missing` and collapse stay **0 by construction**, and the whole cost lands on `extra` |
| `lossless` | its triangle holds **no cell centre**, ties settled by the rasterizer | the cell set does not change at all — this is "the vertices a region needs" |
| `free` | always | the control that shows what the constraint is worth |

🔑 **The containment test has to be at the CELL level, not the polygon level.** A chord across a
staircase of half-voxel steps bulges outside the exact *polygon* while covering not one new voxel
*centre*: an exact 62-vertex diagonal trace is the same cells as a 4-vertex triangle. Constraining
the polygon refuses that for nothing, and the first version of this measurement did exactly that.

⚠️ **And the tie is the common case, not the rare one.** A 45° shortcut between two half-voxel
vertices passes *exactly* through a cell centre. Counted as a change, the lossless rule stalls at 26
vertices on a diagonal that is really 4. It is settled by rasterizing the candidate and comparing,
which is the only authority that agrees with the compiler.

⚠️ **Marching squares is the obvious tool and is wrong here**, as `mask_to_rings` already says: it
chamfers every corner diagonally and hands a plain rectangular shed four 45° eaves it does not have
(#128 hit this). This deletes *existing* vertices and never invents one, so a rectangle stays a
rectangle at every budget — pinned by `test_simplify_region_keeps_a_plain_shed`, the plain-shed
check the ticket asks for by eye.


## What a region actually costs

Over all **1233** polygon-owning operations (500 `Ramp`, 733 `Layer`; `CutRoof` owns no polygon):

| vertices per region | min | p10 | p25 | **median** | p75 | p90 | p99 | max | ≤4 | ≤8 | ≤16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact ring | 4 | 30 | 50 | **94** | 142 | 206 | 342 | 818 | 0.01 | 0.02 | 0.04 |
| needed (lossless) | 3 | 20 | 35 | **58** | 94 | 139 | 261 | 547 | 0.02 | 0.02 | 0.07 |

⚠️ **The distribution is what the median hides, and the ticket asked for it for a reason.** The
corpus is not uniformly rasterised: a tenth of regions are already at 30 vertices or fewer and a
hundredth are past 342. But **only 2% of regions are lossless at 8 vertices or fewer** — the raster
trace is not a thin veneer over a quadrilateral. Most of those 94 vertices are genuinely load-bearing
if the cell set has to be reproduced.

🔑 **A region is a swiss cheese as often as it is a polygon.** 15.7% of regions carry holes — 1808
of them, of which **56.3% are a single cell** and 92% are four cells or fewer — and one region
carries **156**. A speckle hole is 4 vertices that cannot be spent: shrinking it hands its cell back
to the operation, so the contained rule refuses. That, and not the outer boundary, is what sets the
floor: **84% of regions reach a 4-vertex cap** and the ones that cannot are the holed ones.


## Fidelity against budget

n=411. `spike` is the median worst column surplus in voxels and `spiked` the fraction of buildings
with any column more than `s*` = 3 voxels proud of GT (ADR 0004). `contained` is the fraction of
regions that gained no cell; `met` the fraction that reached the budget.

| budget | arm | verts | tokens | missing | extra | vs_input | collapse | ops | planar | spike | spiked | contained | met |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | — | 284 | 578 | 0.0000 | **0.0030** | 0.8221 | 0.0000 | 2.0 | 0.50 | **1** | **0.397** | 1.0000 | 1.00 |
| **needed** | — | **163** | **342** | 0.0000 | **0.0030** | 0.8221 | 0.0000 | 2.0 | 0.50 | **1** | **0.397** | 1.0000 | — |
| 4 | inner | 16 | 44 | 0.0000 | 0.0856 | 0.8856 | 0.0000 | 2.0 | 0.50 | 18 | 0.920 | 1.0000 | 0.84 |
| 4 | free | 12 | 37 | 0.0004 | 0.0247 | 0.8404 | **0.0024** | 2.0 | 0.50 | 17 | 0.903 | **0.0389** | 1.00 |
| 6 | inner | 24 | 60 | 0.0000 | 0.0591 | 0.8722 | 0.0000 | 2.0 | 0.50 | 17 | 0.920 | 1.0000 | 0.84 |
| 6 | free | 18 | 49 | 0.0006 | 0.0145 | 0.8282 | **0.0024** | 2.0 | 0.50 | 16 | 0.895 | 0.0324 | 1.00 |
| 8 | inner | 32 | 76 | 0.0000 | 0.0442 | 0.8627 | 0.0000 | 2.0 | 0.50 | 17 | 0.917 | 1.0000 | 0.86 |
| 8 | free | 24 | 63 | 0.0006 | 0.0120 | 0.8264 | 0.0000 | 2.0 | 0.50 | 16 | 0.903 | 0.0373 | 1.00 |
| 12 | inner | 47 | 108 | 0.0000 | 0.0315 | 0.8546 | 0.0000 | 2.0 | 0.50 | 17 | 0.917 | 1.0000 | 0.91 |
| 12 | free | 36 | 87 | 0.0005 | 0.0100 | 0.8242 | 0.0000 | 2.0 | 0.50 | 16 | 0.905 | 0.0568 | 1.00 |
| 16 | inner | 55 | 128 | 0.0000 | 0.0243 | 0.8499 | 0.0000 | 2.0 | 0.50 | 17 | 0.912 | 1.0000 | 0.92 |
| 16 | free | 48 | 111 | 0.0004 | 0.0082 | 0.8258 | 0.0000 | 2.0 | 0.50 | 16 | 0.898 | 0.0876 | 1.00 |
| **24** | **inner** | 72 | 161 | 0.0000 | **0.0182** | 0.8463 | 0.0000 | 2.0 | 0.50 | **15** | **0.895** | 1.0000 | 0.94 |
| 24 | free | 72 | 157 | 0.0002 | 0.0066 | 0.8254 | 0.0000 | 2.0 | 0.50 | 14 | 0.891 | 0.1517 | 1.00 |
| 94 | inner | 214 | 442 | 0.0000 | 0.0046 | 0.8253 | 0.0000 | 2.0 | 0.50 | 7 | 0.635 | 1.0000 | 0.99 |
| 94 | free | 214 | 442 | 0.0000 | 0.0034 | 0.8227 | 0.0000 | 2.0 | 0.50 | 6 | 0.603 | 0.6642 | 1.00 |

### The guarantee survives, and the constraint is what makes it survive

✅ **`contained` 1.0000, `missing` 0.0000 and collapse 0.0000 at every budget from 4 to 94.** #10's
containment guarantee is not fragile: a region that may only shrink can only leave surplus, and the
whole cost of the budget lands on `extra`, which is where #126 says it belongs.

⚠️ **Drop the one test and it breaks immediately.** At a 4-vertex cap the free arm keeps only
**3.9%** of regions contained, hands operations **51,367** cells the exact region did not have, cuts
into GT (`missing` 0.0004) and puts **collapse above zero** (0.0024) for the first time on this map.
Even at 94 it is only 66% contained. The constraint is load-bearing, not decorative.

### 🔑🔑 But the budget's real cost is not `extra`, it is spikes

Read the `spike` column against the `extra` column and they tell opposite stories. At a 24-vertex
cap the median `extra` is **0.0182 — inside the 0.02 allowance**, which reads like an affordable
budget. The same row says the median building's **worst column stands 15 voxels proud of GT**, and
**89.5% of buildings have one over `s*`**, against 1 voxel and 39.7% at the exact ring. The
montage shows them plainly: thin full-height fins along the roof edge where the exact ring had a
clean line.

**The cause is structural, and it follows from the construction rather than from tuning.** Under
`contained` a region can only shrink, so every column it gives up is a column no operation covers
any more — and with nothing underneath it in the cascade, that column reverts to the **full envelope
height**. The recovered programs have no floor. So the surplus a budget buys is not spread thinly
over a roof; it is concentrated into whichever columns fell outside the simplified polygon.

The same reading kills the naive per-building criterion too: the fraction of buildings reaching
#10's 0.02 allowance falls **72.5% → 51.3% (V=24) → 32.8% (V=8) → 13.1% (V=4)**, so even the
budget whose *median* passes leaves half the corpus outside the allowance.

⚠️ This is #127's lesson arriving by a new route. A scalar that looked fine (`extra` 0.0182, ops
2.0, planar 0.50 — every form number unchanged at every budget) hid a visible fault, and the
picture is what caught it. Note that `dl_ops` and `dl_planar_fraction` are **flat at 2.0 / 0.50
across every budget**: the form metric cannot see this at all, which is itself a limit worth
recording.


## The chosen budget, and the reason

**58 vertices per region — the lossless floor — adopted as a serialisation, not imposed as a cap.**

1. **It is free.** 578 → **342 tokens** per program, −41%, with `extra`, `missing`, `vs_input`,
   collapse, the spike statistics and the form pair all **identical to the exact ring**. There is no
   argument for keeping vertices that provably do nothing.
2. **Every hard cap below it is a bad trade at 64³ on this corpus**, and the table prices each step.
   A 24-vertex cap is the best of them and still spikes 89.5% of buildings at 15 voxels. By the
   project's own detail scale that is a visible fault, not an approximation, so none of the
   pre-registered budgets is servable as a hard cap.
3. **The way to a small-vertex head is not a smaller cap.** Because the spikes come from abandoned
   columns rather than from the polygon shape, the fix is a program with a **floor** — a base
   `Layer` over the whole footprint under the cascade — or a fitter that searches few-vertex regions
   *directly* instead of trimming exact ones after the fact. That decision belongs to the ticket
   that builds the region head, and it now has a price list rather than a guess.

### What this settles for the DSL-cost claim

🔑 **A program's real token cost is 342, and `dl_ops` counts 2.** The description-length pair the
#6/#127/#129 scorecards are read on ignores a term two orders of magnitude larger than the one it
counts, and that term is **polygons, not operations** — **98.5%** of a program's tokens at the exact ring
and **97.7%** losslessly, over a median of 4 recovered operations. Any claim about program *simplicity* that rests on `dl_ops` alone is measured
with the dominant term missing, and it should be quoted with the token count beside it.

⚠️ It also cuts the other way, in the program route's favour: #6's arm predicts a **per-column
assignment over 64×64 = 4096 columns**. A lossless polygon serialisation of the same program is
**342 tokens**, a 12× smaller output space — so a polygon region head is *cheaper* than the
assignment head already in use, not more expensive.


## Pinned

`scene/test_sdf_edit.py::TestVertexBudget`, seven tests, CPU, under a second:

* a plain shed keeps its four right angles at every budget under every rule — no 45° eaves;
* an exact diagonal trace really is a triangle: 4 vertices, not one cell changed;
* the lossless rule changes no cell on a staircase, a concave region and a holed one;
* the contained rule never gains a cell, at any budget, on either shape;
* the free rule does gain cells, so the constraint is doing work;
* a one-cell hole is irreducible while contained, and the free rule swallows its cell;
* `dsl_tokens` counts what a generator would have to emit.


## What this does not settle

* ⚠️ **Whether a base `Layer` removes the spikes.** It follows from the construction that it should,
  and it is untested here. That is a change to the *fitter*, which is #10's, not a re-measurement of
  the budget.
* ⚠️ **A budget for a region a generator INVENTS rather than trims.** Every number above is the cost
  of trimming an exact ring after the fact. A fitter that searched 8-vertex regions directly would
  place its vertices differently and could land anywhere between these rows and the exact one.
* The measurement is at **64³**. A raster trace is a resolution artifact, and the vertex counts, the
  hole speckle, and `s*` all move with the grid.
* Whether `CutRoof` — the one operation with no polygon at all — should carry more of the budget's
  work. It is 13 of 1246 operations on this corpus, so there was nothing to measure.

See [4-edit-algebra.md](4-edit-algebra.md), [10-program-recovery.md](10-program-recovery.md),
[126-massing-scoring.md](126-massing-scoring.md), [6-program-generator.md](6-program-generator.md).
