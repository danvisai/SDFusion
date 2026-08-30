# #6 — Which program generator, and which training strategy?

*Effort: solid-first semantic architectural carving. Opened 2026-07-15, taken up 2026-08-29 after
[#127](127-height-map-generator.md) named it as the next arm from two independent measurements.*

> Given the resolved program algebra, available supervision, rough-carve contract, real-recovery
> evidence, and block-coordination representation, which learned formulation and training strategy
> should the specification choose? Compare autoregressive constrained program generation, graph/set
> diffusion, program induction with pseudo-labels, shared versus separate task heads,
> canonicalization/equivalence-aware objectives, curriculum, rejection/repair, and baselines from
> ArcPro, Building-Gym, ShapeAssembly/CSG, CoMa, and CityGenAgent.

The ticket asks the question against the literature. It is answered here against this corpus,
because four of the choices it lists turn out to be **settled by facts about our own data** rather
than by which paper is better — and measuring those facts took less time than reading the papers
would have.


## Why this was the next arm at all

#127 closed with the open problem moved from *amount* to *form*: every trained arm removes about the
right volume, and none produces planes and ridges. It then measured the cause from both directions,
and both point here.

| what was asked | measured | what it rules out |
|---|---|---|
| can supervision put planes in? | a joint slope term buys description length (6.0→5.0 ops) and **reverses** planarity like-for-like | the loss |
| can a decode take a roof out? | an oracle quantile chosen **per building with the answer in hand** buys 12% of the symmetric difference and **exactly zero** shape | the decode |

🔑 What neither can supply is a **joint commitment** — one hypothesis chosen across a run of columns
rather than 4,096 independent summaries. A column's posterior is honest and wide (GT lies in its
80% band on 95% of columns, and that band is 13 voxels wide); the pointwise median of a family of
possible roofs is a mound that is none of them. A program picks a ridge line and a few planes and
compiles them, so every column's height comes from the same decision.


## What the arm is

`--objective program` in `scripts/foundations/train_height_map_generator.py`, on the same 3.4M-param
U-Net trunk every other #127 arm used, so the output space is the only thing that changed.

    K = 4 typed slots     each a `Layer` (flat, one height) or a `Ramp` (one plane)
    + one assignment      per column, over the K slots plus an UNCARVED class
    -> compile            height = the assigned slot's surface, clamped to [1, extent]

`compile_program` is total in exactly the way `apply_depth` is: any assignment, any type and any
plane at all still yields a footprint-exact height map with a voxel under every footprint column.
A prediction can be wrong; it cannot be invalid.


## The four choices #6 asks about, and what decided each

**Program induction with pseudo-labels, or exact supervision?** → **Exact.** The literature reaches
for pseudo-labels, RL and differentiable relaxations because exact programs are usually unavailable.
Here they are not. #10's fitter is deterministic, sees GT, and costs **0.2 s per building**, so the
whole 35,623-row corpus labels in **56 s** on 48 cores — measured before the arm was designed, and
it is the fact that chose the formulation. None of that machinery is bought.

**Autoregressive, or a set head?** → **A set head**, and this is a property of the vocabulary rather
than a preference. The fitter *searches a sequence*: each operation applies to the result of the
last. But **every operation only ever lowers the height map**, so the final height of a column is
whatever the last operation to touch it wrote — and recording that *owner* per column replays the
entire cascade in one pass. The sequence collapses into a set, losslessly. That is
`program_to_slots`, and `test_the_slots_replay_the_fitted_height_map_exactly` pins it on four
surface families.

**Canonicalisation, or an equivalence-aware objective?** → **Canonicalise, by owned area.** A set
head has no natural slot order, so without a canonical form two fits that found the same program in
a different order would supervise contradictory labels. Sorting by area removes the permutation
problem outright, rather than paying for a Hungarian matching loss to tolerate it.

**Shared or separate heads?** → **Split the way the vocabulary already splits.** An operation is
*one plane over one region*: the assignment stays spatial (a conv head), and the planes are pooled
to be a property of the whole building. That pooling is the point — it is what keeps a ridge line
straight across the plan instead of drifting, and per-column independence is exactly what #127
diagnosed as the cause of the mound.

**And no surface term at all.** #127's plane head composed K planes with a learned soft assignment
under an L1 on the compiled surface, and its slopes collapsed to **0.25 voxels across a 40-voxel
building from two different initialisations**, because a flat region is a strong local optimum
there. So nothing in this objective reads the compiled surface. Each term sees a piece of the
program: cross-entropy on the assignment, cross-entropy on the slot type, L1 on the plane — and on
the *offset only* for a slot typed `Layer`, since a flat roof's slope is not a quantity the label
has an opinion about.

🔑 The type is the mechanical difference from the plane head. `Layer` and `Ramp` are a **discrete
decision the compiler obeys**: a slot typed flat has its slope ignored, and a slot typed `Ramp`
cannot quietly become a terrace by shrinking a number.


## `CutRoof` is withheld, and the price is measured

A `CutRoof` surface is a distance transform, not a plane, so no `(type, plane)` slot can carry it.
`program_to_slots` **refuses** it rather than least-squares-fitting a plane through it, which would
put a target in the cache the compiler provably cannot reproduce.

It was 13 of 1,246 operations (1.0%) in the committed 714-building recovery, and it is cheap to
lose because `Ramp` — a general plane at arbitrary rotation — already covers the roof forms
`CutRoof` expresses, a gable being two opposing ramps.

| fit of the 411 carve-needing buildings | median `extra` |
|---|---|
| full vocabulary (`Layer`, `CutRoof`, `Ramp`) — the committed recovery | 0.0030 |
| `Layer` + `Ramp` only — what #6 trains on | **0.0035** |

`recover_massing_programs.py --ops_allowed Layer Ramp` re-runs it, so this is re-checkable rather
than asserted.


## Three things measured before the run

Which are what make the formulation worth a run rather than an argument.

**The ceiling.** The compiled label — the fitter's own program, with GT in hand — scored down the
same path as every arm:

| | `missing` | `extra` | 3D IoU | **form (ops)** | **planar** |
|---|---|---|---|---|---|
| the real building | — | — | — | 2.0 | 0.50 |
| **compiled program label (K=4, sees GT)** | **0.0000** | **0.0035** | **0.9965** | **2.0** | **0.50** |
| per-column CE + median *(#127's served arm)* | 0.0385 | 0.0603 | 0.8948 | 6.0 | 0.20 |
| planes K=6 *(#127's plane head)* | 0.0324 | 0.0772 | 0.8901 | 3.0 | **0.00** |

🔑🔑 **The output space reaches a real roof's form exactly** — 2.0 ops at 0.50 planar, the same
numbers GT scores. Neither #127 representation could: the per-column head needed three times a real
roof's description length, and the plane head halved the count while spending *none* of it on
planes. This is the first representation on this map's record whose ceiling is the right shape.

⚠️ It is a **ceiling, not a result**. It sees GT, and it is excluded from the verdict for exactly
that reason — it would collect a mechanical PASS, which would be the scorecard reporting that the
target was hit by looking at it.

**Robustness — the answer to the obvious objection.** The loss is on the *program* and the scorecard
is on the *surface*, so: how much parameter error does the surface absorb? Measured on the 411 by
perturbing the labels, with no network involved:

| perturbation of the label | `extra` | `missing` |
|---|---|---|
| none | 0.0035 | 0.0000 |
| plane noise σ = 0.02 *of the building's own height* | 0.0127 | 0.0074 |
| plane noise σ = 0.10 | **0.0379** | 0.0299 |
| **a quarter of all column assignments randomised** | **0.0325** | 0.0514 |

Both extremes still score below the served per-column arm's 0.0603. **The output space degrades
gracefully**, so the arm has to be roughly right, not exact — which is what makes supervising
parameters against a surface metric a sound trade rather than a brittle one.

**Slot mix.** Of the labels on the 411: 749 `Layer` and 509 `Ramp`, so **40% of slots are pitched
planes**. There is real slope in the supervision to learn, and 64 of the 411 buildings need exactly
one operation, so the arm must also learn *not* to spend its budget.


## The bar, pre-registered before the run

Committed in `fccef61`, before the first training step, and fixed in the module docstring so a
result cannot re-litigate it. Same 411 carve-needing rows as #127.

    PASS   BOTH halves of form at once -- median `dl_ops` <= 3.0 AND median `dl_planar_fraction`
           >= 0.40 -- AND median `extra` strictly below the served CE+median arm's 0.0603.
    GUARD  collapse rate no worse than 1-NN's, and `vs_input` < 0.98.
    KILL   median `dl_planar_fraction` <= 0.20, the served arm's own value.

⚠️ **Both halves of form, together.** The plane head reached 3.0 ops with `planar_fraction` **0.00**
— it swapped a mound for a terrace, and a single-number form metric would have shipped it. That is
the one trap this bar exists to catch.


## Result — the KILL clause fires, and the reason is a number

40 epochs, 3.39M parameters, the same budget and trunk every #127 arm had.
Artifact: `execution/artifacts/height_map_generator_program_714.json`.

| arm (411 carve-needing) | `missing` | `extra` | `vs_input` | collapse | carved cols | **form (ops)** | **planar** | *(3D IoU)* |
|---|---|---|---|---|---|---|---|---|
| the real building | — | — | — | — | 0.967 | **2.0** | **0.50** | — |
| **program label (sees GT)** | 0.0000 | **0.0035** | 0.8226 | 0.0000 | 0.921 | **2.0** | **0.50** | *0.9965* |
| blockout | 0.0000 | 0.2308 | 1.0000 | 0.0000 | 0.000 | 0.0 | 0.00 | *0.8125* |
| 1-NN retrieval | 0.0257 | 0.1031 | 0.8743 | 0.1582 | 0.930 | 2.0 | 0.17 | *0.8355* |
| CE + median *(#127's served arm)* | 0.0385 | **0.0603** | 0.8432 | 0.0268 | 1.000 | 6.0 | 0.20 | *0.8948* |
| **`heightmap_program`** | 0.0218 | **0.1236** | 0.8952 | **0.0073** | **0.953** | **1.0** | **0.00** | *0.8572* |

    PASS   ops <= 3.0            1.0   ✔
           planar >= 0.40        0.00  ✘
           extra < 0.0603      0.1236  ✘
    GUARD  collapse <= 1-NN's   0.0073 ✔ (the best of any acting arm on this map's record)
           vs_input < 0.98      0.8952 ✔
    KILL   planar <= 0.20        0.00  → **FIRED**

**Recorded as a kill on the arm as trained.** Not on the formulation — the two are separable here,
and the rest of this section is why.

### What did work, and it is not nothing

* **Best collapse rate of any acting arm**, 0.0073 against the served arm's 0.0268 and 1-NN's
  0.1582. The output space's validity guarantee is real.
* **It acts almost everywhere GT does** — carves 0.953 of columns against GT's 0.967, where the
  served CE arm at its own decode carves 0.565. The no-op that closed #69–#92 is nowhere near.
* 🔑 **Every surface it produces is clean.** The montage
  (`outputs/height_map_generator/representative.png`) shows it beside the CE arms: no mound, no
  concentric contour banding, no ripple, no speckle anywhere. It draws crisp flat-topped blocks with
  sharp edges. That is a visibly different *class* of output from anything on #127's record, and it
  is what "the description length is 1 operation" looks like from the side.

### It draws a flat roof where GT has a pitched one, and that is the whole failure

The arm uses **1.19 slots** per building where its label uses **3.06** (78% of buildings get exactly
one). Of the slots it does use, 46% are *typed* `Ramp` — the type head works — but the realised
height range inside a used slot's own region has **median 0.00 voxels**, and 67% are dead flat.

Which head is responsible, measured by replacing one predicted head at a time with its label:

| | `extra` | `missing` |
|---|---|---|
| the arm as served (all three predicted) | 0.1236 | 0.0218 |
| label **types**, predicted assignment + plane | 0.1245 | 0.0221 |
| label **assignment**, predicted type + plane | 0.1119 | 0.0153 |
| label **planes**, predicted assignment + type | **0.0716** | 0.0025 |
| all three from the label (the ceiling) | 0.0035 | 0.0000 |

🔑 **The planes carry the gap.** The types are free, the regions are worth 0.012, and the plane
parameters are worth 0.052 — and they also carry *all* of the form failure. The clincher is to take
the **perfect** program and flatten only its ramps:

| | `extra` | **ops** | **planar** |
|---|---|---|---|
| the perfect program, compiled as fitted | 0.0035 | 2.0 | 0.50 |
| **the same program with every `Ramp` flattened** | 0.0528 | **1.0** | **0.00** |
| the trained arm | 0.1236 | **1.0** | **0.00** |

Flattening a perfect program reproduces the trained arm's form signature **exactly**. The arm's
regions and types are good enough to score a real roof's description length; its slopes are flat.

### 🔑🔑 Why an L1 on a slope must return flat, and it is not a training failure

The signed slope of every `Ramp` in the corpus, in units of the building's own height across the
plan (n = 11,876 components):

| mean | median | p25 | p75 | positive | negative | median &#124;slope&#124; |
|---|---|---|---|---|---|---|
| **−0.0003** | **+0.0000** | −0.644 | +0.651 | **0.500** | **0.496** | 0.646 |

**The distribution is exactly symmetric.** An L1 regression returns the conditional median, and the
median of a symmetric-about-zero quantity is zero — so the objective's own Bayes act is a flat roof,
however long it trains and however well it fits. The magnitude is not small and not being averaged
into noise: a median &#124;slope&#124; of 0.646 is about **25 voxels on a 38-voxel building**. The
*sign* is what the conditioning does not determine, and #126 already measured why — footprint plus
height does not determine the roof, two matched real buildings differing by a median 3-D IoU of
0.886. A roof may pitch either way, and buildings sit at arbitrary grid rotations (#10).

⚠️ **This is #127's own argument, and I reproduced the mistake it warns about, in a new place.**
#127 chose classification over MSE for the per-column depth precisely because "MSE returns the
conditional mean, which on a bimodal roof distribution is a roof nobody built". I then applied an
**L1 regression to the plane parameters** — and here it is sharper than #127's case, because a
symmetric distribution defeats the mean *and* the median together, so switching to a quantile loss
would not help either.

The same disease explains the assignment collapse in its categorical form: cross-entropy decoded at
its argmax takes the plurality slot, and a column whose posterior is spread across three small slots
lands on the one large one. #127 measured that exact bias for the depth head and fixed it by reading
the posterior at its median instead of its mode — worth `extra` 0.1178 → 0.0603. That cure does not
transfer to a categorical assignment, which has no ordering to take a median over.


## The answer to #6

**The formulation is chosen and it is supported.** Predict the program: a set of typed slots plus a
per-column assignment, supervised exactly, canonicalised by area, compiled hard.

| what #6 asked to compare | decided by | answer |
|---|---|---|
| program induction with pseudo-labels / RL / relaxation | 0.2 s per building; corpus labels in 56 s | **not needed** — supervise exactly |
| autoregressive vs set / graph diffusion | every op only lowers ⇒ the owner replays the cascade | **set head**, losslessly |
| canonicalisation vs an equivalence-aware objective | matching buys 2.7% of the plane error | **canonicalise by area**; a matching loss is not worth it |
| shared vs separate heads | head-swap ablation | **separate** — spatial assignment, pooled planes; the planes are the lever |
| curriculum, rejection/repair | collapse 0.0073, `missing` 0.0218, fp-IoU 1.0 by construction | **not needed** — the compiler is total, there is nothing to repair |

**The training strategy is not.** Regressing the plane parameters is refuted, and by a mechanism
rather than by a disappointing number: the target is symmetric, so every central statistic is flat.

🔑 **The next arm is named precisely: classify the plane parameters.** Discretise the offset and the
slope and train them as #127 trained the depth — cross-entropy over bins, and then, because #127's
biggest single lever on this whole map was the *decode* of exactly such a head (argmax→median, one
line, `extra` 0.1178 → 0.0603), the decode of the parameter head is the thing to get right rather
than the classifier. The ceiling it is aiming at is already measured and it is the right one: **2.0
ops at 0.50 planar, a real roof's form exactly, at `extra` 0.0035.**

⚠️ And one warning for whoever runs it. This map's record has three near-misses from reading a
training curve as a trend, and this arm's own best checkpoint was **epoch 13 of 40** with the
remaining 27 epochs flat — so the curve is in the artifact in full, and the checkpoint is chosen on
validation geometry rather than on loss.

