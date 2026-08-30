# #129 — Does classifying the plane parameters recover the pitch that regressing them cannot?

*Effort: solid-first semantic architectural carving. Opened 2026-08-30 by
[#6](6-program-generator.md), which named this arm on the strength of a mechanism rather than a
disappointing number. Run and written 2026-08-30.*

> #6 chose the formulation — K typed slots plus a per-column assignment, supervised exactly,
> compiled hard — and refuted the *training strategy*: regressing a `Ramp`'s plane parameters under
> an L1/quantile loss **must** return flat, because the signed slope of every `Ramp` in the corpus
> is exactly symmetric. Does **classifying** those parameters — discretised, cross-entropy over
> bins — recover the pitch that the regression provably cannot?

**The mechanism answer is yes, and it is the first time on this map's record.** A trained arm draws
planes: `dl_planar_fraction` **0.00 → 0.40**, and the realised rise inside a slot the arm types
`Ramp` goes from #6's **0.00 voxels** to **20.00**. Nothing on #127's or #6's record did that.

**The arm still does not pass, and the pre-registered checkpoint does not even show the finding.**
The selection rule picked **epoch 2 of 40**, at which the arm barely carves and its planar fraction
is 0.00, so the run of record repeats #6's KILL. The endpoint — a disclosed diagnostic, never the
arm — is where the 0.40 is, and it fails the collapse guard at **37.5%**.


## What changed, and it is one thing

`--plane_head class` on `--objective program`. Same 3.4M U-Net trunk, same assignment head, same
type head, same canonicalisation by area, same 8 plan symmetries, same selection rule, same
compiler. Only the last layer of the slot head and the term that supervises it differ, at a cost of
**0.19M parameters (3.39M → 3.58M, 5.7%)** — too little to be an explanation of anything on its own.

### The three quantities are not the three parameters

| classified | what it is | why not the raw parameter |
|---|---|---|
| **offset** | the plane's height **at its own region's centroid**, in units of the building | `A` is the height at the *plan* centre. A steep plane over a corner region extrapolates to 4.4 building-heights: measured over the corpus, `A` runs **−1.34 … +4.38** while the centroid height runs **0.07 … 0.98**. Binning `A` spends its resolution where no roof is. |
| **pitch** | `atan` of the slope magnitude, over [0, π/2) | Non-negative, needs no range constant, and cannot clip — the corpus's steepest fitted plane rises 13.7 building-heights across the plan and still lands inside the last bin. |
| **azimuth** | the uphill direction, over [0, 2π) | — |

🔑🔑 **The symmetry #6 proved is a MIRROR.** A mirror fixes the pitch and fixes the height at the
region's own centroid, and sends the azimuth to its antipode. So this split moves the **entire**
symmetry into **one categorical variable** and leaves the other two free of it. That is the whole
reason a decode is possible at all, and it is why the parametrisation had to change before the loss
could.

### The decode, pre-registered before the first training step

This map's single biggest lever was the decode of exactly such a head — argmax → posterior median on
#127's depth classifier, one line, `extra` 0.1178 → 0.0603. ⚠️ And #129's warning is that copying
that read here lands straight back on flat. So the read is per quantity, fixed in `PLANE_DECODE`:

| quantity | read | because |
|---|---|---|
| offset | **median** | ordinal, and both competing roofs agree on it — a mirror does not move the height at the region's own centroid. #127's argument, where it still holds. |
| pitch | **median** | ordinal, non-negative, and *invariant* under the mirror: the two roofs the conditioning cannot choose between have the same pitch, so the marginal is the conditional. |
| azimuth | **argmax** | categorical and antipodally bimodal by construction. A circular mean over two opposite modes returns a direction **neither** holds — #127's mound arriving by a third route. `argmax` commits to one of the two, which is what #126 says the task is. |

Committed in `207845a`, before the run.

### The bins were priced before the run

A classifier cannot beat its own bins. `plane_quantisation_ceiling` encodes every fitted plane and
decodes it straight back, with no network involved:

| | `extra` | `missing` |
|---|---|---|
| exact — the continuous ceiling | 0.0035 | 0.0000 |
| binned at 32 | 0.0107 | 0.0052 |
| **binned at 64 — `PLANE_BINS`** | **0.0063** | **0.0035** |
| binned at 128 — available, **not taken** | 0.0039 | 0.0026 |

64 leaves the ceiling an order of magnitude below the 0.0603 the arm has to beat, so the binning is
not the limiter and the binding constraint is examples-per-class. Taking 128 *because it scores
better here* would be choosing a design on a number the trained arm cannot cash.


## The bar, unchanged from #6

🔑 `PROGRAM_BAR` is **not touched**. #129 is judged by exactly the bar #6 was judged by, on exactly
the same 411 carve-needing rows, through the same `verdict()` — so the arm cannot be credited by a
bar that moved with it.

    PASS   median `dl_ops` <= 3.0 AND median `dl_planar_fraction` >= 0.40 AND median `extra` < 0.0603
    GUARD  collapse rate no worse than 1-NN's (0.1582), and `vs_input` < 0.98
    KILL   median `dl_planar_fraction` <= 0.20


## Result

40 epochs, 3.58M parameters. Artifacts: `execution/artifacts/height_map_generator_class_714.json`,
`..._class_714_diagnostics.json`, `..._class_last_714_diagnostics.json`. **Every arm in that run is
listed here**, not a chosen subset.

| arm (411 carve-needing) | `missing` | `extra` | `vs_input` | collapse | carved | **ops** | **planar** | *(3D IoU)* |
|---|---|---|---|---|---|---|---|---|
| the real building | — | — | — | — | 0.967 | **2.0** | **0.50** | — |
| program label *(sees GT — the ceiling)* | 0.0000 | 0.0035 | 0.8226 | 0.0000 | 0.921 | **2.0** | **0.50** | *0.9965* |
| blockout | 0.0000 | 0.2308 | 1.0000 | 0.0000 | 0.000 | 0.0 | 0.00 | *0.8125* |
| mean_roof | 0.0135 | 0.1369 | 0.9070 | 0.0000 | 1.000 | 2.0 | 0.00 | *0.8640* |
| 1-NN retrieval *(the bar)* | 0.0257 | 0.1031 | 0.8743 | 0.1582 | 0.930 | 2.0 | 0.17 | *0.8355* |
| CE at argmax | 0.0090 | 0.1178 | 0.9304 | 0.0316 | 0.565 | 3.0 | 0.00 | *0.8682* |
| CE + median *(#127, served)* | 0.0385 | **0.0603** | 0.8432 | 0.0268 | 1.000 | 6.0 | 0.20 | *0.8948* |
| `heightmap_program` *(#6, regressed planes)* | 0.0218 | 0.1236 | 0.8952 | 0.0073 | 0.953 | 1.0 | **0.00** | *0.8572* |
| **`heightmap_program_class`** *(#129, selected — epoch 2)* | 0.0013 | **0.1507** | 0.9714 | 0.1022 | 0.544 | 1.0 | **0.00** | *0.8126* |
| `heightmap_program_class_last` *(epoch 40, **diagnostic, not the arm**)* | 0.1065 | 0.0902 | 0.8028 | **0.3747** | 0.774 | 1.0 | **0.40** | *0.7854* |

### The arm of record fails, and the KILL fires again

    PASS   ops <= 3.0             1.0    ✔
           planar >= 0.40         0.00   ✘
           extra < 0.0603         0.1507 ✘
    GUARD  collapse <= 0.1582     0.1022 ✔
           vs_input < 0.98        0.9714 ✔  (barely — it is nearly its own input)
    KILL   planar <= 0.20         0.00   → **FIRED**

Evaluated by `verdict()`, not by this table — the clauses are in the artifact's `verdict` block as
`form_planar_over_bar` / `beats_served_extra` / `killed_flat` / `program_pass`.

⚠️ **On surplus it is the worst acting arm in the table**: 0.1507 is worse than #6's own 0.1236 and
worse than the envelope-relative position of every trained arm on the record. Nothing below softens
that, and the run of record for #129 is a second KILL.


## But the selection rule and the bar point at different epochs, and that is the finding

The pre-registered rule selects on validation `missing + extra`. The classified head **trades
`missing` for `extra` as it trains**, so the symmetric difference flattens out while `extra` keeps
falling — and the rule reads that as "no improvement since epoch 2".

| validation, carve-needing | epoch 2 *(selected)* | epoch 40 |
|---|---|---|
| `extra` | 0.1496 | **0.0844** |
| `missing` | **0.0114** | 0.1067 |
| symmetric difference | **0.1609** | 0.1911 |

⚠️ This is the *rule working as specified*, not a bug: #127 adopted the symmetric difference
precisely because selecting on `extra` alone picked a collapsed first epoch on the MSE arm. It is
also the first arm on this ticket where it and the bar disagree, so `train()` now writes the final
epoch as `<tag>_last.pt` — **a diagnostic and never the arm**. Reporting it is disclosure; scoring
the run on it would be selecting on the answer.


## 🔑🔑 Classification does put a pitch in — the measurement, three ways

**1. The realised rise inside a slot the arm types `Ramp`.** #129 asked for this beside every number
because 46.4% of the slots #6's arm used were typed `Ramp` and it drew them with a median
0.00-voxel range. (#129 words this as "typed 46% of its slots `Ramp` **correctly**"; the metric is
the rate at which it types `Ramp`, not its accuracy, and the point survives either reading.)

| | slots typed `Ramp` | **median realised rise** | compile flat |
|---|---|---|---|
| `heightmap_program` *(#6, regressed)* | 0.464 | **0.00 vox** | 0.670 |
| `heightmap_program_class` *(selected)* | 0.522 | **6.00 vox** | 0.481 |
| `heightmap_program_class_last` | 0.504 | **4.00 vox** | 0.481 |

and measured over `Ramp`-typed slots only, the classified arms realise **20–22 voxels** of rise.
🔑 **The regression typed a ramp and drew a terrace; the classifier draws the ramp it types.** That
is #129's question answered at the mechanism, and it does not depend on which checkpoint is read.

**2. The form metric.** `dl_planar_fraction` 0.00 → **0.40** at the endpoint, against GT's 0.50 and
the compiled label's 0.50. Every trained arm before it scored 0.00 (#6's program, #127's plane head,
CE at argmax) or 0.20 (CE + median). **0.40 is the highest a trained arm has reached on this map.**

**3. The picture, which is what caught the mound in the first place.**
`outputs/height_map_generator/maps_class_{best,representative,worst}.png` and
`montage_class_*.png`, all from this run. On the pitched-roof rows the classified arms draw **one
uniform normal colour across the whole roof — a single pitched plane** — where `heightmap_program`
draws flat with a seam and `heightmap_ce_median` draws concentric contours. It is a visibly
different class of output.


## And the same head is what eats the building

The head-swap ablation replaces one predicted head with its label at a time. On the endpoint arm:

| one head replaced by its label | `extra` | `missing` |
|---|---|---|
| all three predicted | 0.0902 | **0.1065** |
| label types | 0.0965 | 0.0992 |
| label assignment | 0.0925 | 0.1184 |
| **label planes** | 0.0866 | **0.0006** |
| all three from the label *(the ceiling)* | 0.0035 | 0.0000 |

🔑 **Replacing the planes takes `missing` 0.1065 → 0.0006 and barely moves `extra`.** The regions and
the types are not what over-carves; the planes are. So the head that finally put a pitch in is the
same head that cuts through the building, and that is one problem rather than two.

⚠️ The mechanism is an asymmetry in the output space that #6 had no occasion to meet: a plane a
little **too steep** dives below GT over the far end of its region and is charged the whole trench,
while one a little **too shallow** only leaves surplus above. `extra` and `missing` are not
symmetric in the pitch, and a pitch estimate that is unbiased in the parameter is biased in the
geometry. The worst montage rows show it directly — a thin wall left standing where a slot cut a
trench to the floor.


## The decode ablation — the pre-registration holds, and by the predicted mechanism

Same weights, same forward pass, eleven reads. ⚠️ Read after the fact; it reports, it does not
choose. Endpoint checkpoint, on the 411:

| offset / pitch / azimuth | `extra` | `missing` | sym | ramp rise |
|---|---|---|---|---|
| **median / median / argmax  ← pre-registered** | 0.0902 | 0.1065 | **0.1967** | 20.0 |
| median / median / **circmean** | 0.0748 | 0.1401 | 0.2149 | 22.0 |
| median / argmax / argmax | 0.0940 | 0.1093 | 0.2033 | 22.0 |
| argmax / median / argmax | 0.0967 | 0.0915 | 0.1882 | 19.0 |
| argmax / argmax / argmax | 0.0975 | 0.0848 | 0.1823 | 21.0 |
| argmax / argmax / circmean | 0.0859 | 0.1202 | 0.2061 | 23.0 |
| median / **pitch q0.25** / argmax | 0.0916 | 0.0816 | **0.1732** | 15.0 |
| median / pitch q0.35 / argmax | 0.0912 | 0.0924 | 0.1836 | 18.0 |
| median / pitch q0.75 / argmax | 0.0859 | 0.1370 | 0.2229 | 23.0 |

✅ **The azimuth `argmax` beats `circmean` on the symmetric difference at every offset/pitch pair, on
both checkpoints, and it beats it in the predicted way**: the circular mean has *lower* `extra` and
much *higher* `missing` — it averages two opposite roofs into one direction that cuts across the
building. The pre-registered read was right, and it was right for the stated reason.

❌ **The offset `median` is not.** `argmax` scores a better symmetric difference on both checkpoints
(0.1882 vs 0.1967 here). #127's lever does not transfer to this quantity, and I predicted it would.

🔑 **The pitch is where the next arm should look.** `q0.25` buys `missing` 0.1065 → 0.0816 for
`extra` +0.0014 — the best symmetric difference in the table, from the same weights. That is the
geometric asymmetry above, showing up exactly where it was predicted to. ⚠️ **Not adopted**: it is a
read chosen after seeing the answer, and the honest form of it is a pre-registered lower-quantile
pitch on the next run.


## What this ticket settles, and what it does not

**Settles**, and it is the question #129 asked:
* 🔑🔑 **Classifying the plane parameters recovers the pitch a regression on them provably cannot.**
  0.00 → 0.40 planar, 0.00 → 20 voxels of realised rise, visible in the plan-view normals. #6's
  refutation of the regression is confirmed to have been about the *loss*, not about the program
  route: the same output space, the same supervision, one head swapped, and the roofs are pitched.
* **The mirror decomposition is the enabling move**, not the cross-entropy on its own. Binning
  `(A, Bz, Cx)` would have put the symmetry in two variables at once and given the decode nothing
  to commit to.
* **The pre-registered azimuth `argmax`** is confirmed against the averaging read it was chosen
  over, on the mechanism it was chosen for.

**Does not settle:**
* ⚠️ **The arm does not pass, and the run of record is a second KILL.** The selected checkpoint has
  planar 0.00 and `extra` 0.1507, worse than #6's.
* ⚠️ **The endpoint fails the collapse guard at 37.5%**, more than double 1-NN's 15.8%. A generator
  that destroys a third of its buildings is not servable whatever its roofs look like.
* ⚠️ **The selection rule cannot see this arm's improvement.** Whether the fix is a different rule,
  a pitch read that stops the over-carve, or both, is unresolved — and any rule change must be
  pre-registered before the run that benefits from it, not chosen from this table.
* The named baselines, set diffusion and curriculum remain [#130](https://github.com/danvisai/SDFusion/issues/130)'s, untouched here.


## Re-running any of it

    P="env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/train_height_map_generator.py"
    W=outputs/height_map_generator

    $P --objective program --plane_head class --tag heightmap_program_class --epochs 40 \
       --montage 0 --no_form --out execution/artifacts/height_map_generator_class_train.json

    $P --ckpt heightmap_program_class=$W/heightmap_program_class.pt \
              heightmap_program_class_last=$W/heightmap_program_class_last.pt \
              heightmap_program=$W/heightmap_program.pt heightmap_ce=$W/heightmap_ce.pt \
       --median_decode --montage 6 --maps 4 \
       --maps_arms heightmap_program_class heightmap_program_class_last heightmap_program \
                   heightmap_ce_median \
       --out execution/artifacts/height_map_generator_class_714.json

    $P --diagnose_program $W/heightmap_program_class.pt \
       --out execution/artifacts/height_map_generator_class_714.json
    $P --diagnose_program $W/heightmap_program_class_last.pt \
       --out execution/artifacts/height_map_generator_class_last_714.json

The bins' ceiling, the head swap, the slot usage, the realised rise and the eleven-read decode table
are all in `--diagnose_program`, so every number above is a committed code path rather than a
notebook. `scripts/foundations/test_train_height_map_generator.py` — 126 tests, 23 of them #129's,
and the load-bearing one is that equal mass on two opposite azimuths decodes to one of them rather
than to their average.
