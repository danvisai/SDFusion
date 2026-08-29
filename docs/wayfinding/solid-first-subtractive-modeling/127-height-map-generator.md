# #127 — Does a footprint-conditioned height-map generator carve?

*2026-08-28. One A100, ~35 minutes of training across two arms; the corpus pass that precedes it is
CPU only.*

Answers [#127](https://github.com/danvisai/SDFusion/issues/127), which asked whether a small
footprint + height + region → height-map generator actually carves surplus or learns identity like
every arm since [#69](https://github.com/danvisai/SDFusion/issues/69). Unblocked by
[#126](https://github.com/danvisai/SDFusion/issues/126), whose decision fixes the form of the
answer: the `missing`/`extra` split leads, `vs_input` and the collapse rate stand beside it, the
aggregate 3D IoU is a diagnostic, and the population is the **411 carve-needing** buildings of the
pinned 714.

Code `scripts/foundations/train_height_map_generator.py`, contract tests
`scripts/foundations/test_train_height_map_generator.py`, artifact
`execution/artifacts/height_map_generator_714.json`, montages
`outputs/height_map_generator/{best,representative,worst}.png`.

---

## 🔑🔑 It carves. It is the first generator on this project's record that does.

| | `extra` | `vs_input` | verdict |
|---|---|---|---|
| blockout (do nothing) | 0.2308 | 1.0000 | the input |
| a2_s0.5, the shipped 49M model | 0.2357 | 0.9852 | **worse than its own input**, and 99% of it |
| **height map, CE + median decode** | **0.0603** | **0.8432** | a 74% surplus reduction, and it moved |

Every height-map arm makes a large net-positive edit, beats the envelope on **100% of decided
rows**, and is footprint-exact (fp-IoU 1.0000) by construction. The three-year no-op pattern is not
a property of this task. It was a property of the output spaces it was being asked for in.

⚠️ **And the picture says the shape is wrong.** It carves the right *amount* and the wrong *form* —
a rounded mound where the real roof is planes meeting at a ridge. Both halves are below and neither
should be quoted alone.

## The arms — carve-needing subset, n=411

The aggregate sits right of the bar because #126 demoted it. `carved` is the fraction of footprint
columns the arm cut at all; GT cuts **0.967** of them.

| arm | `missing` | **`extra`** | `vs_input` | collapse | >env `extra` | `carved` | *(3D IoU)* |
|---|---|---|---|---|---|---|---|
| blockout | 0.0000 | 0.2308 | 1.0000 | 0.0000 | — | 0.000 | *0.8125* |
| corpus mean roof | 0.0135 | 0.1369 | 0.9070 | 0.0000 | 100% | 1.000 | *0.8640* |
| **1-NN retrieval** (the bar) | 0.0257 | **0.1031** | 0.8743 | **0.1582** | 100% | 0.930 | *0.8355* |
| heightmap CE + argmax *(pre-registered)* | 0.0090 | 0.1178 | 0.9304 | 0.0316 | 100% | **0.565** | *0.8682* |
| heightmap CE + median *(decode ablation)* | 0.0385 | **0.0603** | 0.8432 | **0.0268** | 100% | 1.000 | *0.8948* |
| heightmap MSE | 0.0482 | 0.0638 | 0.8173 | 0.0511 | 100% | 1.000 | *0.8869* |
| heightmap quantile q=0.5 *(the retrain)* | 0.0371 | 0.0685 | 0.8436 | 0.0219 | 100% | 1.000 | *0.8918* |

This project's arms of record, re-summarised on **the same 411 rows** rather than quoted from their
own populations (#126's like-for-like rule, and map #87's 11.8% correction is why):

| arm | `missing` | `extra` | `vs_input` | collapse | *(3D IoU)* |
|---|---|---|---|---|---|
| deployed map-24 | 0.0504 | 0.7152 | — | 0.1630 | *0.5192* |
| a2_s0.5 (shipped, 49M) | 0.0027 | 0.2357 | 0.9852 | 0.1241 | *0.7736* |
| program recovery K=16 *(sees the answer)* | 0.0000 | 0.0030 | 0.8221 | 0.0000 | *0.9970* |
| codec ceiling | 0.0007 | 0.0015 | — | 0.0049 | *0.9977* |

On the 303 **already-flat** buildings every height-map arm returns the blockout exactly — `extra`
0.0000, 3D IoU 1.0000, 0.000 of columns carved. The empty program is recovered without being taught
as a special case, and the 42% no-op majority #10 warned about is passed rather than exploited.
⚠️ The `mean_roof` arm is the counter-example that shows this is not free: it carves 100% of columns
there too and eats 9.4% of GT.

## ⚖️ The bar was reconsidered by the human, and 1-NN demoted to a reference point

*Ruled 2026-08-28, **after** the results below were seen. Recorded that way on purpose — the
original pre-registration is kept intact underneath, because a bar edited after the fact and
presented as the original is worthless.*

The ruling: **1-NN retrieval is a reference point, not a gate.** The reasoning is that the two arms
are not the same kind of object. Retrieval is **non-parametric** — it carries all 34,909 training
roofs to inference time and copies one. The generator carries 3.4M parameters and must invent a
roof. Retrieval is therefore crisp and planar by construction, because it emits a *real* roof; a
generator cannot get that for free. Requiring the generator to beat it treats a compression
constraint as a quality failure.

⚠️ **The premise that motivated the ruling did not hold, and that is worth stating.** The concern
was that retrieval "will always beat it". Measured, it does not — three of the four trained arms
beat it, and the served one beats it by **41.5%**:

| arm | `extra` | against 1-NN's 0.1031 | collapse |
|---|---|---|---|
| CE + argmax *(the pre-registered arm)* | 0.1178 | **−14.3%** — loses | 0.0316 |
| **CE + median** *(served)* | **0.0603** | **+41.5%** — beats | **0.0268** |
| MSE | 0.0638 | +38.1% — beats | 0.0511 |
| quantile q=0.5 | 0.0685 | +33.5% — beats | 0.0219 |

🔑 And on safety the generator is not close to retrieval — it is **six times better**. Retrieval
collapses on **65 of 411** buildings, because a roof borrowed from a look-alike still cuts into this
building. The served arm collapses on **11**. Retrieval buys its crisp shape by sometimes destroying
the building; that is the cost the aggregate hides.

So the standing verdict is: **the approach clears the bar the ticket set; one decode of one arm did
not.** "Failure" was my word for that and it was too broad — the precise statement is that a single
pre-registered decode missed a single threshold by 0.0147, while every other arm cleared it.

### The validity criterion, and why it cannot rank these arms

The human's proposed criterion is that massing counts as architecture only if it is **connected to
the building** — no floating voxels, no islands, no noise off to the side. Measured on the 411
carve-needing buildings, 26-connectivity, every arm:

| arm | connected components | floating voxels |
|---|---|---|
| the real building | 1–2 (median 1) | 0.00000 |
| footprint envelope | 1–2 (median 1) | 0.00000 |
| 1-NN retrieval | 1–2 (median 1) | 0.00000 |
| every height-map arm | 1–2 (median 1) | 0.00000 |

🔑 **Identical across every arm, including GT itself.** The criterion is satisfied *by construction*
in this output space: `apply_depth` keeps at least one voxel on every footprint column, so a solid
run from the base is the only thing representable. The 1–2 components are inherited from footprints
that are themselves two blobs in plan — GT has them too.

So the criterion **cannot discriminate between these arms**. ⚠️ It is not idle, though: it is exactly
what the *earlier* representations failed. #80's dense-SDF failure mode was hollow shells, A2
collapses on 12.4% of these buildings and the deployed map-24 model on 16.3%. Moving to a height map
is what made floating geometry unrepresentable, and that is a result about the representation rather
than a metric for ranking within it.

What the criterion is reaching for and does not yet capture is **roof form** — planes meeting at a
ridge against a rounded mound. That remains open, and three amplitude statistics failed at it.

## The original pre-registration, kept intact: NOT MET by the arm it was registered for

The bar was committed before the first run (`f1f0dcd`): median `extra` strictly below 1-NN's on the
same rows, collapse no worse than 1-NN's, `vs_input` < 0.98.

| arm | beats 1-NN `extra` | collapse ok | moved | |
|---|---|---|---|---|
| corpus mean roof | ✗ 0.1369 | ✓ | ✓ | **not met** |
| **heightmap CE + argmax** — *the pre-registered arm* | **✗ 0.1178** | ✓ | ✓ | **not met** |
| heightmap CE + median — *decode ablation, added after* | ✓ 0.0603 | ✓ | ✓ | *pass* |
| heightmap MSE | ✓ 0.0638 | ✓ | ✓ | *pass* |

🔑 The arm the bar was written for **misses it**, by 0.0147. Two arms clear it and both were run
after the miss was seen, so neither carries the pre-registration. #10's precedent is the format:
report the whole ladder, in order, and let the pass be read as what it is.

⚠️ **The bar was harder than #127 wrote it.** The ticket's 1-NN row was never committed and does not
reproduce; the version measured here is *stronger* (`extra` 0.1031 against the ticket's 0.1099, and
`missing` 0.0257 against 0.0792) because `transplant_height` renders the retrieved roof
footprint-exact and at the conditioned height, which a footprint-conditioned generator would also
get for free. Beating the ticket's own number would have been enough for the argmax arm; beating a
fair one was not.

## Which rows of #127's table reproduce, and which do not

| arm | #127's table | measured here | |
|---|---|---|---|
| blockout | 0.8125 / 0.0000 / 0.2308 | 0.8125 / 0.0000 / 0.2308 | **exact** |
| a2_s0.5 shipped | 0.7736 / 0.0027 / 0.2357 | 0.7736 / 0.0027 / 0.2357 | **exact** |
| program recovery | 0.9826 / 0.0000 / 0.0177 | (K=16) 0.9970 / 0.0000 / 0.0030 | consistent |
| corpus mean roof | 0.7213 / 0.2427 / **0.0208** | 0.8640 / 0.0135 / **0.1369** | **does not reproduce** |
| 1-NN retrieved roof | 0.8040 / 0.0792 / **0.1099** | 0.8355 / 0.0257 / **0.1031** | **does not reproduce** |

🔑 The two rows that came from committed artifacts reproduce to the digit. The two that were
computed ad-hoc are the two that do not — the same pattern #126 found, one ticket earlier, and the
reason both are now committed code. Neither is recorded as *wrong*: the original computations are
gone. The shape of the `mean_roof` gap (`missing` 0.24 against 0.01) is what an **absolute** mean
profile would produce rather than a height-relative one, since an absolute mean under-builds every
tall building; that is inference from the direction of the error, not a demonstration.

Two guards say this reproduction is on the right footing, both against numbers computed by other
code on other days: the blockout's `extra` and 3D IoU on the 411 match #10 exactly, and the
`a2_s0.5` row matches the shipped artifact exactly.

## 🔑 The decode was the defect, not the posterior

Same weights, same forward pass, one line changed. `extra` **0.1178 → 0.0603**, 3D IoU 0.8682 →
0.8948, collapse 3.2% → 2.7%.

Carve depth is **ordinal** and 54% of footprint columns carry depth 0, so the *mode* of a column's
posterior is a biased estimator of its depth. Measured over all 562,534 footprint columns of the
carve-needing 411:

| | GT | argmax | posterior median |
|---|---|---|---|
| mean carve depth | 7.68 | **4.07** (53% of GT) | **6.38** (83% of GT) |
| columns carved at all | 0.893 | **0.527** | 0.955 |
| under-carved / over-carved | — | 0.643 / 0.195 | 0.472 / 0.394 |
| predicted mean where GT is 21–63 deep | 21+ | 9.29 | 12.21 |

The argmax arm under-carves at **every** depth band and touches barely half the columns it should.
That is not a model that failed to learn the roof — the same network's posterior, read at its
median, recovers 83% of the depth. The quantile is fixed a priori at 0.5 (the Bayes act under
absolute error), not tuned.

⚠️ **This generalises past this ticket.** Per-column classification was adopted here specifically to
escape regression-to-the-mean, and it does escape MSE's blur — but the mode of a dominated ordinal
posterior shrinks toward the majority class, which produces a *quieter* version of the same no-op.
Any future arm that classifies an ordinal target with a large "do nothing" class inherits it.

## The ticket's design note is falsified

> *"Prefer per-column classification over the 64 discrete height levels to plain MSE regression: MSE
> gives the conditional mean, i.e. an averaged, bland roof, which is the same regression-to-the-mean
> trap that produced the no-op."*

Measured, MSE (`extra` 0.0638, IoU 0.8869) **beats** classification-with-argmax (0.1178, 0.8682) and
lands within 0.0035 of classification-with-median. The prediction was right about the mechanism and
wrong about which decode carries it: on this target the conditional *mean* is a better carve than
the conditional *mode*. The bland-roof failure the note describes is real and visible — it is what
the `mean_roof` arm looks like, and what MSE's own output looks like — but it costs shape, not
surplus.

⚠️ MSE pays for it in `missing`: 0.0482 against the argmax arm's 0.0090, and a collapse rate of
5.1% against 3.2%. It carves more by carving less carefully.

## ⚠️ The visual check disagrees with the scorecard

`outputs/height_map_generator/{best,representative,worst}.png` render every arm beside the real
building as shaded isometric massing (the CPU height-field renderer from #10 — a height map needs
no marching cubes).

**What the montages show, and it is consistent across all three:**

* Real buildings and the **1-NN** arm are made of **flat planes meeting at ridges** — gables, hips,
  sharp setbacks. 1-NN looks like a building because it *is* one, copied.
* Every **trained** arm returns a **rounded mound with concentric contour rings**. It sits at
  roughly the right height and removes roughly the right volume, and it is not a roof.
* The **argmax** arm additionally shows spiky and holed artefacts on several rows — visibly the
  worst output of the six, on a scorecard where it is mid-table.
* On `best.png` row 1 the argmax arm scores `extra` **0.000** while looking worse than the blockout
  it started from. `extra` only charges volume *above* GT, so rubble that stays underneath is free.

🔑 So the scorecard #126 fixed — which is the right scorecard for *surplus* — is **blind to roof
form**, and this ticket is the demonstration. The project's stated priority is *visual first,
footprint match second*; on the visual criterion the zero-training retrieval arm wins, and it is the
arm that loses the numeric bar.

### 🔑 The form metric: description length in the project's own vocabulary

*Found 2026-08-28, after the three amplitude statistics below failed.*

Ask the question the way #10 already asks it: **how many architectural operations explain this
roof?** Run #10's `Layer` / `Ramp` / `CutRoof` fitter on each arm's *own* surface and count the
operations it needs. A real roof is a handful of planes meeting at ridges; a mound is a continuum of
orientations and has no short description.

Validated on shapes whose answer is known by construction, before it was pointed at any arm:

| surface | ops | program |
|---|---|---|
| flat roof | 1 | `Layer` |
| shed — one tilted plane | 1 | `Ramp` |
| gable — two planes and a ridge | 2 | `CutRoof > Ramp` |
| hip — four planes | 4 | |
| two-step setback | 2 | `Layer > Layer` |
| **a dome** | **9** | and mostly `Layer`s |
| **noise** | **16+** | and still not explained |

🔑 **Slope is not complexity.** A shed roof steps at every column and is the simplest roof after a
flat one. That is precisely what `roof_relief` got wrong, and why an amplitude statistic could never
work here.

### The gap, measured on the 411

| arm | `extra` | **form (ops)** | **planar fraction** |
|---|---|---|---|
| **the real building** | — | **2.0** | **0.50** |
| footprint envelope | 0.2308 | 0.0 | 0.00 |
| 1-NN retrieval | 0.1031 | **2.0** | 0.17 |
| CE + argmax | 0.1178 | 3.0 | 0.00 |
| **CE + median** *(served)* | **0.0603** | **6.0** | 0.20 |
| MSE | 0.0638 | 5.0 | 0.25 |
| quantile q=0.5 | 0.0685 | 5.0 | 0.25 |

**The served arm needs three times the description length of a real roof**, and spends only 20% of
its budget on planes against GT's 50%. 1-NN matches GT at 2 ops, which is what "it copies a real
roof" means quantitatively. That is the form gap, and it is now a number rather than an impression.

The **operation mix** is the mechanism: architecture spends its budget on `Ramp` and `CutRoof`,
which are planes. A mound cannot be explained by planes, so the fitter falls back to stacking flat
`Layer`s — which is exactly the concentric contour banding the montages show.

⚠️ **The metric is not carve-aware, by design.** The envelope scores 0 ops because the fitter starts
*from* the envelope; an arm that did nothing needs nothing to explain it. Whether an arm acted is
`extra` and `vs_input`'s job. Read form beside them and never alone.

### The planar representation: half the description length, still not a roof

*Run 2026-08-28, after the metric existed.* The fix the metric pointed to: make planarity free the
way the clamped height map made validity free. The output became **K=6 planes plus a per-column
region assignment**, so description length is at most K by construction and a mound is not
representable. An untrained head already scored 3 ops against the per-column model's 6.

| arm | `missing` | `extra` | collapse | **form (ops)** | **planar** | *(3D IoU)* |
|---|---|---|---|---|---|---|
| the real building | — | — | — | **2.0** | **0.50** | — |
| per-column CE + median *(served)* | 0.0385 | **0.0603** | 0.0268 | 6.0 | 0.20 | *0.8948* |
| **planes K=6** | 0.0324 | 0.0772 | **0.0195** | **3.0** | **0.00** | *0.8901* |

**Half the description length and the best collapse rate of any acting arm — and it is still not a
roof.** `planar_fraction` went to **0.00**, worse than the per-column model it replaced. The
representation delivered exactly what it promised and the promise was not sufficient.

🔑 **The slopes collapse to flat, from any initialisation.** Measured on the trained model's real
per-building output, a plane tilts by **0.25 voxels across a 40-voxel building**. The first run
initialised every slope at zero and it never left; the second initialised half the planes tilted by
*half an extent* in spread directions, the biases survived at 0.45 — and the network's learned
weights cancel them back to 0.006. Two initialisations, same answer.

So the model uses its six planes as six **horizontal terraces**. That is #10's own name for this
failure — *"a ziggurat of three or four giant steps"* — and the montages show it as concentric flat
rings where the real roof is a hip or a gable. It swapped a mound for a terrace.

⚠️ **The metric earned its keep here.** Description length alone would have called this a win: 6 → 3
ops. Only the `planar_fraction` half showed that the budget moved from `Layer`-plus-mound to
`Layer`-only, with no plane in it at all. A single-number form metric would have shipped this.

**What it means for the route.** Composing planes with a *learned soft assignment* does not teach
slope: a flat region is a strong optimum under L1, and the assignment can always subdivide a pitch
into more flat steps rather than tilt one plane. The remaining route is to predict the **program
itself** — typed operations with explicit parameters, where a `Ramp`'s slope is an output rather
than something that has to survive a straight-through gradient. #10 already built that supervision.

### Three scalar attempts at the visual difference, all negative

| arm | relief | curvature | speckle | the eye |
|---|---|---|---|---|
| gt | 0.46 | 0.634 | 0.000 | planes and ridges |
| nn_retrieval | 0.32 | 0.454 | 0.000 | planes and ridges |
| heightmap CE + argmax | **0.47** | **0.778** | 0.000 | a mound, plus speckle |
| heightmap CE + median | 0.40 | 0.509 | 0.000 | a mound |
| heightmap MSE | 0.28 | 0.492 | 0.000 | a mound |

`relief` (mean height step) ranks the worst-looking arm **closest to GT**; `curvature` (mean second
difference, 0 on any plane at any slope) ranks two of the mounds **smoother than a real building**;
`speckle` (strict local extrema) is 0.000 on every arm at the median. All three order the arms
roughly opposite to the eye.

The cause is that **GT is itself terraced at 64³** — #10 measured exactly this, "roof-slope
terracing, the staircase a sloped roof makes on a 64³ grid" — so an amplitude statistic cannot tell
a discretised plane from a mound. What separates them is the *organisation* of the steps, which is
what the description length above finally captures. These three are kept, computed and published so
the failed attempt stays on the record beside the one that worked.

## What was pinned, and what the output space actually gives for free

#127 claims a height map is *"a valid solid by construction, footprint-exact, collapse-impossible,
and `missing` and `collapse_rate` are 0 by clamping"*. Measured, two of three:

| claim | verdict |
|---|---|
| footprint-exact | **true** — fp-IoU 1.0000 for every arm, every building, whatever is predicted |
| a valid solid, no hollow shell | **true** — every footprint column keeps ≥ 1 voxel |
| `missing` and `collapse_rate` are 0 | **false** — over-carving still eats GT: the median arm's `missing` is 0.0385 and it collapses on 2.7% |

The clamp is total by design (`apply_depth` accepts negative and out-of-range depths and still
returns a valid solid), and the depth parameterisation makes the arm **purely subtractive**, so
`extra` can never come out worse than doing nothing — which is what #10's `missing`=0 on 714/714
says the corpus is. `test_missing_is_NOT_free_and_the_arm_can_still_collapse` exists so the third
claim cannot quietly return.

Two guards protect the answer itself: `condition_channels` has **no argument through which the
target could reach the model**, and the retrieval bank is built from training rows only.

## Against #126's reference point

#126 set one: *"A height-map generator that reaches ≈0.10 `extra` without collapsing has beaten a
real building's disagreement with another real building."* A real building offered footprint-exact
reaches `extra` **0.0974** with a **16.7%** collapse rate.

| | `extra` | collapse |
|---|---|---|
| a real building, footprint-exact (#126 `alt_exact`) | 0.0974 | 0.1667 |
| 1-NN retrieval, here | 0.1031 | 0.1582 |
| **heightmap CE + median** | **0.0603** | **0.0268** |

The 1-NN arm independently lands on #126's number from a different construction — 34,909 training
buildings retrieved at a median footprint IoU of **0.9521**, against #126's 64 hand-matched pairs at
≥ 0.90 — which is a cross-check on both. The trained arm clears the reference on both columns.
⚠️ It clears the *surplus* reference. It does not produce a building of the quality that reference
was drawn from, per the montages above.

## Limits, stated

- **The pre-registered arm did not pass.** Two arms that did were run after seeing it miss. Nothing
  here is a pre-registered pass and it should not be cited as one.
- **The training curve is non-monotonic**, as this project's always are: validation `missing+extra`
  swings 0.1187–0.1895 across 40 CE epochs with the best at epoch 26, and adjacent epochs differ by
  more than the gap between arms. The full curve is in the artifact. ⚠️ Do not read a trend in it.
- **Checkpoint selection is on `missing + extra`, and the first rule was wrong.** Selecting on
  `extra` alone is gameable by carving the building away — it picked the MSE arm's *first* epoch
  (`extra` 0.039, `missing` 0.082). Both arms were retrained under the corrected rule. Selection is
  on a 1,000-building validation split drawn from the training rows; the pinned 714 are never read
  during training.
- **The distance transform is a supplied feature.** It is a deterministic function of the footprint,
  so no information leaks, but a claim about what the architecture learns unaided is not available
  from this run.
- **One seed, one architecture, one training length.** 3.37M parameters, 40 epochs, no sweep. The
  gap between arms (0.0575 between the two decodes) is far larger than what a seed would move, but
  that is an argument, not a measurement.
- **The height is a user input** (#81), given exactly to every arm including the baselines. This
  measures roof shape, not massing end to end.
- **This is the corpus's 2.5-D structure** (#10). Nothing here transfers to massing with genuine
  through-voids, which this corpus does not contain at 64³.

## ✅ The human reviewed the montages and accepted them

*Recorded 2026-08-28, after the montages above.* The human's verdict on the visual criterion is
**yes** — this satisfies the scope they set, which was *"input a shape and get a blockout which
looks like a building"*, and it does so where the other approaches on this project did not.

⚠️ **This is recorded as their judgement, and it is not the same as mine.** My reading of the same
sheets is in the section above: the trained arms return a mound where the real roof is planes
meeting at a ridge. Both readings are on the record because #127 asks for a human review and the
human is the judge of criterion 1, not the analyst — the same posture `docs/SESSION-HANDOVER-
2026-08-03.md` took when the human accepted an earlier model. The scalar record is unchanged: the
pre-registered arm still missed its bar, and the form gap in "What follows" is still open.

## Wired into the demo

`scripts/server/town_generate_service.py` serves the arm everywhere it serves A2. One knob,
`arm`, on `/generate_building`, `/generate_town` and `/compare_arms`, defaulting to `a2` so no
existing caller changed behaviour. `/arms` is a page that carves one drawn footprint with every arm
on a synchronised camera; the town editor gained a model selector.

🔑 **The cost difference is the demo's headline, not the quality difference.** Measured warm on one
A100, in the service:

| arm | per building | a 29-building town (the Munich preset) |
|---|---|---|
| a2 (49M, through the Dora codec) | ~1.1–7 s | ~3.5 min |
| **height map (3.4M, no codec)** | **~0.1–0.25 s** | **~5 s** |

A height map compiles straight to voxels, so the arm needs no codec at all and is offered even on a
box where Dora is absent.

⚠️ **The height-map arm is deterministic, and that shows in a town.** Two identical footprints
produce *bit-identical* buildings — measured, 0.000000 m apart after recentering — where A2 gets its
variety from per-building noise. A `roof_variation` knob jitters the decode quantile per building
(a coherent deeper-or-shallower roof, never per-column noise, which would be rubble). It
**defaults to 0**, so the demo shows the arm that was actually scored; above 0 the output is no
longer the measured model and the editor says so.

Weights are staged at `weights/massing-heightmap/` with a manifest and checksums, beside A2's.

## What this is research-usable for, and what it is not

The user's scope — footprint in, building-shaped mass out — is met. That is a **capability**
milestone. Stated honestly, it is not by itself a **contribution**, and the two should not be
conflated in a write-up.

**Not novel as a method.** `NOVELTY_SURVEY.md` already lists "footprint or site polygon to
procedural 3D building" and "neural architectural DSL prediction plus deterministic compilation"
among *established ingredients*. A convnet predicting a 64×64 height field from a footprint is an
image-to-image regression with a long precedent, and nothing here changes that.

**What is defensible:**

1. **The measurement.** Six arms across two representations converged to no-op (#69–#92). This
   shows the no-op was a property of the **output space**, not of the models or the data — the same
   corpus, the same conditioning, a 15× smaller network, and it carves. That is a result about
   representation choice, and it is the part worth writing down.
2. **A baseline the program route must now beat.** #10's recovered carving program reaches `extra`
   0.0030 *while seeing the answer*. Any learned program predictor is now required to beat **0.0603
   from a 3.4M convnet**, not the envelope's 0.2308. That raises the bar for #1/#4/#6 considerably.
3. **It measures a route as over-parameterised — it does not retire it.** #113 specifies
   whole-volume binary segmentation of the start box. On this corpus the label is a height map, so
   that route carries 64× more parameters than the label needs — #10 found this and #127 is the
   working demonstration.

   ⚠️ **#113 stays open by the owner's decision (2026-08-29).** An earlier draft of this section
   said this result "retires" the voxel route; it does not. The 64× figure is a measurement about
   *this corpus at this resolution* and says nothing about what the voxel generator is for — voids,
   overhangs and anything genuinely 3-D are outside what a height map can represent at all. The
   ticket is to be worked later; no session may close it or fold it into the program route.

**What it cannot support, and must not be claimed:**

- **Anything 3-D.** One height per plan column. Courtyards, passages, arcades, light wells and
  overhangs are *not representable*. #10 measured that this corpus contains none at 64³, so the
  representation is sufficient **for this corpus** — that is a fact about the data, not about
  architecture, and `NOVELTY_SURVEY.md`'s hypothesis 2 ("architectural voids as first-class
  generative objects") is untestable here.
- **The editability claim.** The output is geometry, not a **semantic architectural edit program**.
  `CONTEXT.md`'s thesis is that editability is core rather than a wrapper; a height map does not
  carry a recipe, so this arm is no better placed than A2 on C1's editing half. #128's edit-stack
  bridge is on the *program*, not on this.
- **Detail.** Massing only, above s\* (ADR 0004). Unchanged.
- **Generality.** One corpus, one resolution, one seed, one architecture.

## Retraining: what would and would not help

**Nothing needs retraining for the demo.** The served checkpoint is the best measured, and training
had converged — the training loss is flat from about epoch 10 of 40, and the swing between adjacent
epochs is larger than the gap between the last twenty. More epochs buy nothing, and this project has
twice been wrong reading that curve.

**The one retrain that looked principled was run, and it did not work.** ⚠️ Recorded because the
prediction was mine and it was wrong.

The reasoning was that the decode and the objective disagree: the model is trained with
cross-entropy, whose Bayes act is the *mode*, and then read at the *median* because the mode
under-carves. A **pinball loss at q=0.5** makes the optimised quantity and the decoded quantity the
same. I predicted a modest gain. Measured on the same 411 buildings:

| arm | `missing` | **`extra`** | `vs_input` | collapse | *(3D IoU)* |
|---|---|---|---|---|---|
| CE + post-hoc median | 0.0385 | **0.0603** | 0.8432 | 0.0268 | *0.8948* |
| **quantile q=0.5, trained directly** | 0.0371 | **0.0685** | 0.8436 | **0.0219** | *0.8918* |

**It is worse, and the difference is real, not noise.** Paired on the same buildings, the retrained
arm beats the post-hoc one on only **166 of 411** (Wilcoxon p=0.0044 on `extra`, p=0.027 on 3D IoU).
`missing` is not separable (p=0.21). Its one advantage — a collapse rate of 2.19% against 2.68% — is
**9 buildings against 11**, which is not a difference worth claiming.

🔑 **Hypothesis for why, offered as a mechanism and not as a finding.** Cross-entropy learns the
*whole* 64-class posterior, so its median is computed from the distribution's actual shape. The
pinball head learns **one scalar per column** and discards the distribution. CE-plus-post-hoc-median
therefore has strictly more information than direct median regression, and "make the objective match
the decode" traded that information away to remove a mismatch that was not costing anything. Untested;
a check would be whether a CE model's median beats a pinball model's *at every quantile*, not just at
0.5.

⚠️ **The form is unchanged.** The retrained arm sits in the same mound family as the other two
regressions on the montages. Making the loss name the right statistic was never going to fix a
shape problem caused by per-column independence, and it did not.

**What this leaves.** The served arm does not change: CE with a post-hoc median decode remains the
best measured, and it is already what the demo serves. The `--objective quantile` path stays in the
code because the arm is on the record and the loss is now pinned by tests, not because anything
should be run on it.

**The form gap will not be closed by retraining this model at all.** A mound is what per-column
independence produces. That needs a different output space — a program — not a better fit of this
one.

## Follow-up arm: the slope term — PRE-REGISTERED 2026-08-29, before the run

Everything above says the same thing twice: every arm removes about the right *amount* and none
produces planes. The diagnosis on the record is per-column independence. Before chartering a program
generator, one cheaper hypothesis is worth eliminating — that some of the joint structure can be
recovered by **supervision** alone, without changing what the network emits.

**The change, in one sentence.** Add a second loss term that compares the *step between neighbouring
columns* in the prediction against the step in GT — the quantity the plan-view normal map draws —
and keep cross-entropy exactly as it is. `slope_loss` in `train_height_map_generator.py`.

Two properties are pinned by test rather than argued:

- it is **blind to a constant offset**, so it cannot do CE's job (how deep to carve) and only speaks
  to how the carve is arranged;
- it **matches** GT's steps rather than minimising them, so a sharp ridge is free. A term that
  merely penalised roughness would prefer a rounded ridge, which is the mound again.

The term needs a height to difference, and a CE head predicts a distribution. Collapsing it with the
softmax expectation would measure the slope of a *blended* field, and `compose_planes` already
records that a smooth blend of surfaces is a mound. So `differentiable_depth` takes a **hard
forward** at the decode the arm is actually served at (the posterior median), with the gradient
flowing through the soft probabilities — the same straight-through the plane head uses, for the same
reason.

**The weight is fixed a priori at λ = 1.0** by one rule, stated before the run and not swept: it
gives the slope term a **20% share** of the converged per-column loss. Measured on the served
checkpoint over 256 validation buildings, CE = 1.5552 and the slope term = 0.3090, so
0.2 × 1.5552 / 0.3090 = 1.007. Sweeping λ against the scorecard would be selecting on the answer.

Nothing else moves: same 40 epochs, same seed, same width, same augmentation, same selection rule
(validation `missing + extra`), same post-hoc median decode, same 411 carve-needing buildings, same
harness.

**The bar, on the 411 carve-needing buildings, against the served arm (`extra` 0.0603, collapse
0.0268, form 6.0 ops / 0.20 planar):**

  PASS      median `dl_ops` **strictly below 6.0** AND `planar_fraction` **strictly above 0.20**,
            with `extra` <= 0.0663 (the served arm plus 10%), collapse <= 0.0268, `vs_input` < 0.98.
  NEGATIVE  form unchanged or worse — `dl_ops` >= 6.0 or `planar_fraction` <= 0.20. This is a real
            answer: it would say supervision cannot buy joint structure and the program route is the
            only way to it.
  KILL      `extra` >= 0.2308. That is the blockout, i.e. identity.

⚠️ **Predicted outcome, recorded so it cannot be revised afterwards: NEGATIVE on form, and a small
improvement in `extra`.** The term shapes the loss but not the architecture — at inference the head
still emits one posterior per column independently — so the mechanism #127 blames is untouched. The
probe is worth its 16 minutes because a *positive* result would be much cheaper than a program
generator, and because "supervision cannot fix it" is a claim this map should be able to cite rather
than assume.


### Result — PASS on the bar as written, and the bar as written was too generous

*Run 2026-08-29. 40 epochs, checkpoint selected at epoch 26 by the unchanged rule. Scored on the
same 411 carve-needing buildings, artifact `execution/artifacts/height_map_generator_slope_714.json`.*

| arm | `missing` | `extra` | `vs_input` | collapse | **form (ops)** | **planar** | *(3D IoU)* |
|---|---|---|---|---|---|---|---|
| the real building | — | — | — | — | **2.0** | **0.50** | — |
| CE + median *(served)* | 0.0385 | **0.0603** | 0.8432 | 0.0268 | 6.0 | 0.20 | *0.8948* |
| **CE + slope + median** | 0.0353 | 0.0651 | 0.8435 | **0.0243** | **5.0** | 0.22 | *0.8944* |

Every pre-registered condition is met on the statistic it was written against, so the verdict is
**PASS**. Taken apart, one half is solid and the other is not, and the difference only shows up
under #126's like-for-like rule:

- 🔑 **Description length is a real, robust gain.** Paired on the same buildings, the slope arm
  needs fewer operations on 216 of 278 non-ties (Wilcoxon **p = 3.5e-22**), mean 6.05 → 5.04. It is
  not bought by carving less: `corr(Δops, Δextra) = −0.09`, and on the **229** buildings where the
  two arms removed essentially the same volume (|Δ`extra`| < 0.005) the mean is still 5.65 → 4.86
  with B simpler on 105/136 non-ties (**p = 1.6e-10**). The fitter also explains **100%** of the
  slope arm's surfaces within the budget against 99.0%, at a slightly lower residual — the surfaces
  are genuinely simpler, not merely cheaper to give up on.
- ⚠️ **Planarity is not.** Pooled, `planar_fraction` 0.20 → 0.22 clears the bar but is not separable
  (p = 0.089). On the matched-volume subset it **reverses** — 0.2222 → 0.2000, higher on 66 of 138,
  p = 0.895. Had the planar condition been pre-registered like-for-like, as #126 requires of every
  other comparison on this map, it would read **NOT MET**. Recorded that way.
- **Surplus is unchanged.** `extra` 0.0603 → 0.0651 on the median but p = 0.40 with essentially
  equal means; `missing`, collapse and 3-D IoU are all within noise.

⚠️ **My pre-registered prediction was wrong, and wrong in a specific way worth keeping.** I
predicted no form movement and a small `extra` gain; what happened is the opposite on both counts.
Supervision *can* buy **simplicity** — the same roof explained by one operation fewer — and it
cannot buy **planes**. The operations the slope term saves are still `Layer` terraces; it makes the
mound smoother and cheaper to describe without making any part of it a plane.

**The montage and the map sheets agree with that reading and not with the ops number alone.** The
slope arm's contour rings are fewer and straighter, its normal maps hold larger uniform regions —
and it is still a mound, in the same family as every other trained arm. `maps_*.png` renders the two
arms side by side; `--maps_arms` exists because seven arms is sixteen columns and an unreadable
figure decides nothing.

🔑 **What this settles for the map.** The joint structure a roof needs is not only missing from the
*loss* — it is missing from the *output space*. A term that couples neighbouring columns recovers
description length and stops there, which is the sharpest evidence yet that planes have to be in the
vocabulary by construction rather than encouraged by supervision. That is exactly what the program
route (#6) provides and what this probe was run to find out cheaply. It cost 20 minutes of A100 and
one loss term, and the served arm does not change.


## What follows

- **The question #127 asked is answered: yes, it carves**, decisively and at 1/15th the parameters
  of the shipped model, and the no-op that closed #69–#92 does not reproduce in this output space.
- **The open problem has moved from *amount* to *form*.** Every trained arm now removes about the
  right volume and none produces planes and ridges. That is a different failure from anything on
  this map's record, and the metric suite cannot currently see it.
- 🔑 **The two facts together point at the same place.** A mound is what per-column independence
  produces — each column is predicted from its own marginal, so the ridge line, which is a *joint*
  property of a run of columns, is averaged away. #10's program recovery has the joint structure
  built in (`Layer`, `Ramp`, `CutRoof` are region-level operations) and reaches `extra` 0.0030 with
  planar output. Predicting a **program** rather than a per-column height is the obvious next arm,
  and #10 already built the exact-label supervision it would need.
- **A form metric is a prerequisite, not a nicety.** Three amplitude statistics failed here. Until
  something separates a mound from a roof, an arm can pass this scorecard by getting the volume
  right, and the montage is the only thing that catches it.
