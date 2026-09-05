"""Does a footprint-conditioned height-map generator carve, or does it learn identity too?

#127's question. Every generator on this project's record -- #69 through #92, six arms across two
representations -- converged to returning its own input. This asks whether that was the *model* or
the *output space*, by moving the output space to the one #10 measured the corpus actually to be:
a 64x64 height map.

WHAT IS PREDICTED, AND WHY IT IS THE CARVE AND NOT THE HEIGHT
-------------------------------------------------------------
The label is the per-column **carve depth** `d = extent - top`, classified over 64 levels, not the
absolute top and not a regression. Three reasons, in the order they matter:

  * **Depth makes the arm purely subtractive.** `apply_depth` clamps to `[1, extent]`, so a
    prediction can never exceed the blockout it started from and `extra` can never come out worse
    than doing nothing. #10 measured `missing`=0 on 714/714 -- the real building is always inside
    its own extruded footprint -- so subtractive-only is the corpus's own structure, not a
    convenience.
  * **Classification, not MSE.** MSE returns the conditional mean, which on a bimodal roof
    distribution (flat top / pitched) is a roof nobody built -- the same regression-to-the-mean that
    produced the no-op. `--objective mse` exists to *test* that claim rather than assume it, and is
    scored as its own arm.
  * **The labels are exact integers already.** No quantisation, no codec, no latent. #10's
    reconstruction residual over the pinned 714 is 71 voxels in 4.3M.

WHAT THE OUTPUT SPACE GIVES FOR FREE, STATED PRECISELY
------------------------------------------------------
#127 claims a height map is "footprint-exact, collapse-impossible, and `missing` and `collapse_rate`
are 0 by clamping". Two of those are true and one is not, and the tests pin the difference:

  * footprint-exact  -- TRUE. `apply_depth` writes exactly the footprint mask, so fp-IoU is 1.0000
                        by construction for every prediction, good or bad.
  * a valid solid    -- TRUE. Every footprint column keeps at least one voxel, so no prediction can
                        punch a hole through the plan or return a hollow shell (#80's failure).
  * `missing` = 0    -- FALSE. Over-carving still eats GT. The collapse rate is measured on the
                        model's own output and published beside every number, exactly as #126
                        requires of the alternative-building arm that collapses on 16.7%.

THE BAR, PRE-REGISTERED BEFORE THE FIRST RUN
--------------------------------------------
Fixed here so a result cannot re-litigate it (map #87's discipline, and #10's record of stopping at
a dip and being wrong twice). Scored on the **411 carve-needing** buildings of the pinned 714 --
303 need no carve at all and a 42% no-op majority flatters every aggregate (#126 point 4).

  PASS   median `extra` strictly below the **1-NN retrieval** arm's, measured on the same rows in
         the same run. 1-NN is the bar, not the blockout (#127).
  GUARD  collapse rate no worse than 1-NN's, and `vs_input` < 0.98 -- an arm that did not move has
         not been measured as a generator at all (#75).
  KILL   median `extra` at or above the blockout's. That is identity, and it answers #127 "no".

The aggregate 3D IoU is reported to the right of the bar and is a diagnostic, never a gate: #126
demoted it because its median cannot rank a real building above the envelope.

Run -- one arm at a time, then score them together against the baselines:

    P="env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/train_height_map_generator.py"
    $P --objective ce  --tag heightmap_ce  --epochs 40 --montage 0   # ~16 min on one A100
    $P --objective mse --tag heightmap_mse --epochs 40 --montage 0
    $P --ckpt heightmap_ce=outputs/height_map_generator/heightmap_ce.pt \
              heightmap_mse=outputs/height_map_generator/heightmap_mse.pt \
              --median_decode --montage 6

The 3D montage says whether an arm reads as a building. The plan-view pair says *why* -- the height
map shows where the volume went, the normal map shows whether what is left is made of planes -- and
it scores nothing, so it runs from finished checkpoints in seconds:

    $P --ckpt heightmap_ce=outputs/height_map_generator/heightmap_ce.pt \
              heightmap_planes=outputs/height_map_generator/heightmap_planes.pt \
              --median_decode --maps_only --maps 4          # best/representative/worst sheets
    $P --ckpt ... --median_decode --maps_only --maps_ids 1341 19229 20650   # named buildings
    $P --ckpt ... --maps_only --maps_arms heightmap_ce_median heightmap_ce_slope_median

The joint SLOPE term is an arm like any other and is off by default, so every arm on the record is
unaffected. Its pre-registered weight is 1.0 (`docs/wayfinding/solid-first-subtractive-modeling/
127-height-map-generator.md`):

    $P --objective ce --slope_weight 1.0 --tag heightmap_ce_slope --epochs 40 --montage 0 --no_form

The first invocation builds `outputs/height_map_generator/height_fields.npz` from the corpus, which
takes ~12 minutes and is then reused by everything else.


#6 -- THE PROGRAM ARM, AND WHY IT IS AN ARM OF THIS SCRIPT AND NOT A NEW ONE
============================================================================
#127 closed with the open problem moved from *amount* to *form*: every trained arm removes about the
right volume and none produces planes and ridges. It measured the cause from both directions --
supervision could not put planes in (the slope term bought description length and reversed
planarity) and decoding could not take a roof out (an oracle quantile chosen per building with the
answer in hand buys 12% of the symmetric difference and **exactly zero shape**). What neither can
supply is a **joint commitment**: one hypothesis chosen across a run of columns rather than 4,096
independent summaries, whose pointwise median is a mound that is none of them.

#6 asks which learned formulation makes that commitment. The answer here is chosen by measurement:

  * **Predict the program, not the surface.** K=4 typed slots -- each a `Layer` (flat) or a `Ramp`
    (a plane) -- plus one assignment per column over the slots and an UNCARVED class. Every column
    in a region gets its height from the slot the region shares, so a ridge line is one decision.
  * **Supervise it with #10's fitter, exactly.** Measured before the arm was designed: the fitter is
    deterministic, sees GT, and costs 0.2 s per building, so the whole 35,623-row corpus labels in
    **56 s** on 48 cores. The literature #6 names reaches for pseudo-labels, RL or a differentiable
    relaxation because exact programs are usually unavailable. Here they are not, so none of that
    machinery is bought, and the surface loss whose flat optimum sank the plane head is not used at
    all.
  * **Canonicalise by area, not by matching.** A set head has no natural slot order, so slots are
    sorted by owned area. That removes the permutation problem outright rather than paying for a
    Hungarian loss to tolerate it.
  * **`CutRoof` is withheld from the label vocabulary**, because its surface is a distance transform
    and no (type, plane) slot can carry it. Measured, not assumed: it was 13 of 1,246 operations,
    and dropping it moves the fit's median `extra` on the 411 from **0.0030 to 0.0035**.

Three facts measured before the run, which are what make the formulation worth a run at all:

    compiled label ceiling, on the 411   extra 0.0035   missing 0.0000   3D IoU 0.9965
    its form                             2.0 ops, planar_fraction 0.50 -- EQUAL to the real building
    robustness                           param noise of 0.10 *of the building's own height* still
                                         scores extra 0.0379, and randomising a QUARTER of the
                                         column assignments still scores 0.0325 -- both below the
                                         served per-column arm's 0.0603

That last one is the answer to the obvious objection to supervising parameters while scoring a
surface: this output space degrades gracefully, so the arm has to be roughly right, not exact.

THE BAR FOR #6, PRE-REGISTERED BEFORE THE FIRST RUN
---------------------------------------------------
Same 411 rows, same discipline. The reference numbers are #127's, re-read on those rows.

  PASS   BOTH halves of form at once -- median `dl_ops` <= 3.0 AND median `dl_planar_fraction`
         >= 0.40 -- AND median `extra` strictly below the served CE+median arm's **0.0603**.
         ⚠️ Both halves, because #127's plane head reached 3.0 ops with planar_fraction **0.00**:
         it swapped a mound for a terrace, and a single-number form metric would have shipped it.
  GUARD  collapse rate no worse than 1-NN's, and `vs_input` < 0.98 (#75).
  KILL   median `dl_planar_fraction` <= 0.20, the served per-column arm's own value. That is the
         terrace failure repeating in a third representation, and it answers #6 "not this way".

  for reference, on those same rows:  GT 2.0 ops / 0.50 planar
                                      CE+median (served) extra 0.0603 / 6.0 / 0.20
                                      planes K=6         extra 0.0772 / 3.0 / 0.00
                                      1-NN retrieval     extra 0.1031 / 2.0 / 0.17

    $P --objective program --tag heightmap_program --epochs 40 --montage 0

The first program run builds `outputs/height_map_generator/program_labels.npz` (56 s, 48 cores).


#129 -- CLASSIFY THE PLANE PARAMETERS, AND DECODE THEM THE WAY #127's DEPTH HEAD TAUGHT
=======================================================================================
#6 kept its formulation and killed its training strategy, with a proof rather than a number: the
signed slope of every `Ramp` in the corpus is **exactly symmetric** -- mean +0.0009, median
+0.0000, 50.0% positive / 49.6% negative over 52,792 components -- so an L1 or a quantile on it
returns a flat roof however long it trains. That is the objective's own Bayes act, not a training
failure. #129 asks whether **classifying** those parameters recovers the pitch that regressing them
provably cannot.

`--plane_head class` on `--objective program`. The ONLY difference from #6's arm is the last layer
of the slot head and the term that supervises it; the trunk, the assignment head, the type head,
the canonicalisation, the augmentation, the selection rule and the compiler are all untouched, so
a difference between the two arms is attributable to the plane head.

WHAT IS CLASSIFIED, AND WHY IT IS NOT (A, Bz, Cx)
-------------------------------------------------
Three quantities at `PLANE_BINS`=64 each, and the re-parametrisation is the argument:

  offset   the plane's height **at its own region's centroid**, in units of the building. `A` is the
           height at the *plan* centre, which a steep plane over a corner region extrapolates to
           4.4 building-heights -- measured over the corpus, `A` runs -1.34..+4.38 while the
           centroid height runs 0.07..0.98. Binning `A` spends its resolution where no roof is.
  pitch    `atan` of the slope magnitude, over [0, pi/2). Needs no range constant and cannot clip.
  azimuth  the uphill direction, over [0, 2pi).

🔑🔑 **The symmetry #6 proved is a MIRROR, and a mirror fixes the pitch and the centroid height and
sends the azimuth to its antipode.** So this split moves the entire symmetry into ONE categorical
variable and leaves the other two free of it. That is what makes the decode below possible at all.

THE DECODE, WHICH #129 SAYS IS MOST OF THE TICKET
--------------------------------------------------
This map's single biggest lever was the decode of exactly such a head -- argmax -> posterior median
on #127's depth classifier, one line, `extra` 0.1178 -> 0.0603. ⚠️ And its loudest warning is that
copying that read here lands straight back on flat, because the median of a symmetric bimodal slope
is zero. So the read is chosen per quantity and fixed in `PLANE_DECODE` before the first step:

  offset   MEDIAN  ordinal, and both competing roofs agree on it (a mirror does not move the height
                   at the region's own centroid), so #127's argument applies unchanged.
  pitch    MEDIAN  ordinal, non-negative, and INVARIANT under the mirror -- the two roofs the
                   conditioning cannot choose between have the same pitch -- so its posterior is not
                   the symmetric bimodal one and the median is not defeated by it.
  azimuth  ARGMAX  categorical, and antipodally bimodal by construction. A circular mean over two
                   opposite modes returns a direction NEITHER holds, which is #127's mound arriving
                   by a third route. `argmax` commits to one of the two, which is what #126 says the
                   task is: the conditioning does not determine which roof, so pick one.

The other ten reads are measured after the fact (`decode_ablation`) and reported. They do not
choose; a row that beats the pre-registered one is a finding, not a decode to adopt retroactively.

MEASURED BEFORE THE RUN
------------------------
    the bins' own ceiling, on the 411   binned at 64:  extra 0.0063  missing 0.0035
    against the continuous ceiling                     extra 0.0035  missing 0.0000
    at 128 bins (available, NOT taken)                 extra 0.0039  missing 0.0026

64 bins leave the ceiling an order of magnitude below the 0.0603 the arm must beat, so the binning
is not the limiter and the binding constraint is examples-per-class. `plane_quantisation_ceiling`
is the re-runnable form of that table.

THE BAR FOR #129, PRE-REGISTERED BEFORE THE FIRST TRAINING STEP
---------------------------------------------------------------
🔑 **`PROGRAM_BAR`, unchanged.** #129 is judged by exactly the bar #6 was judged by, on exactly the
same 411 carve-needing rows, evaluated by the same `verdict()` -- so the arm cannot be credited by a
bar that moved with it. Restated for the reader:

  PASS   median `dl_ops` <= 3.0 AND median `dl_planar_fraction` >= 0.40 AND median `extra` < 0.0603.
  GUARD  collapse rate no worse than 1-NN's, and `vs_input` < 0.98 (#75).
  KILL   median `dl_planar_fraction` <= 0.20. The terrace failure repeating in a fourth
         representation, and it would answer #129 "no".

  reported beside every number, never as a gate:
    the REALISED RISE inside each `Ramp`-typed slot's own region. 46.4% of the slots #6's arm used
    were typed `Ramp` and it drew them with a median **0.00-voxel** range, so "it predicted a ramp"
    is not evidence that it drew one, and only the surface says which.
    `vs_input` and the collapse rate (#126), and a montage -- three amplitude statistics failed to
    separate a mound from a roof, and the picture is what caught it.

  for reference, on those same rows:  GT                    2.0 ops / 0.50 planar
                                      compiled label        extra 0.0035 / 2.0 / 0.50
                                      CE+median (served)    extra 0.0603 / 6.0 / 0.20
                                      #6 program (regress)  extra 0.1236 / 1.0 / 0.00
                                      1-NN retrieval        extra 0.1031 / 2.0 / 0.17

⚠️ The checkpoint is chosen on **validation geometry, not loss** -- #6's best was epoch 13 of 40
with the remaining 27 flat, and this map has three near-misses from reading a curve as a trend.

    $P --objective program --plane_head class --tag heightmap_program_class --epochs 40 \\
       --montage 0 --no_form --out execution/artifacts/height_map_generator_class_train.json
    $P --diagnose_program outputs/height_map_generator/heightmap_program_class.pt \\
       --out execution/artifacts/height_map_generator_class_714.json

The result is written up in `docs/wayfinding/solid-first-subtractive-modeling/
129-classified-plane-parameters.md`, with the full scoring command.


#132 -- THE OVER-CARVE AND THE ONE SLOT, PRE-REGISTERED BEFORE THE FIRST TRAINING STEP
=======================================================================================
#129 answered its mechanism question YES and failed its bar anyway. Its two failures sit on
DIFFERENT heads, and #132 changes one thing on each:

  SURPLUS is a PLANE problem.  Swapping in label planes takes `missing` 0.1065 -> 0.0006 while
  barely moving `extra`: the head that put the pitch in is the head that cuts through the building.
  🔑 `extra` and `missing` are NOT symmetric in a pitch. A plane a little too STEEP dives below GT
  over the far end of its region and is charged the whole trench; one a little too SHALLOW only
  leaves surplus above it. So the loss-minimising *parameter* estimate is a biased *geometry*
  estimate, and the correction is to read the pitch BELOW the median.
     -> `PLANE_DECODE` pitch: median -> **q0.25**, the canonical lower quartile.

  FORM is an ASSIGNMENT problem.  `dl_ops` reads 1.0 on both arms because both use ONE region --
  #6 1.19 slots, #129 0.90 -- not because their planes are flat. The output space is not the
  constraint: the compiled label uses 3.06 slots at 2.0 ops / 0.50 planar.
     -> the assignment cross-entropy is **logit-adjusted by the label prior** (`ASSIGN_PRIOR`,
        tau = `ASSIGN_TEMPERATURE` = 1.0), training-side only.

🔑🔑 THE DIAGNOSIS CAME FIRST, AND IT KILLED THE OBVIOUS FIX. `assignment_collapse` on both of
#129's checkpoints says the assignment posterior is DIFFUSE, not confidently wrong -- confidence
0.43, normalised entropy 0.80, argmax recall on non-dominant-slot columns **0.0000**. That looks
like a decode problem, and this map has twice found the decode WAS the answer. It is not this time:

    read            slots seen   minor recall   per-column acc   DOMINANT slot acc
    argmax             0.90         0.0000          0.4245            0.8251
    prior-balanced     3.70         0.2829          0.2203            0.1275   <- REFUTED

The post-hoc correction mostly relabels the building -- `circmean`'s failure from #129 in a second
place -- because a class that is flat over the plan has p/prior == 1 by construction, so the
dominant slot ties with every other flat class. ⚠️ Measured BEFORE the run rather than after it,
which is the whole point of asking the free question first. The adjustment therefore goes into the
LOSS, where it changes what is learned, with inference left as the plain argmax.

⚠️ THE SELECTION RULE IS UNCHANGED, and that is a choice with a falsifiable prediction attached.
#129's rule picked epoch 2 of 40 because the classified head trades `missing` for `extra` as it
trains, so the symmetric difference plateaued while `extra` kept falling. The prediction: the q0.25
pitch is what stops that trade, so the symmetric difference should now improve with training and
the rule should select a late epoch. If it selects epoch 2 again, the pitch read did not work and
that is the finding -- not a licence to change the rule afterwards.

⚠️ PRE-REGISTERED PREDICTION, so the arm is falsifiable in parts: on #129's weights `q0.25` alone
leaves collapse at 0.2409, still above 1-NN's 0.1582. The pitch read is NOT expected to pass on its
own; the assignment change has to do the rest. If the arm passes with only one of the two working,
that is a result about the other.

THE BAR: `PROGRAM_BAR`, unchanged again, on the same 411 rows through the same `verdict()`. Beside
every number, and on EVERY table including the diagnostics: `vs_input`, the collapse rate (#126),
`slot_usage` beside `dl_ops` -- 1.0 ops at 0.90 slots and 1.0 ops at 3.0 slots are different
results -- the realised rise inside `Ramp`-typed slots, and a montage.

    $P --objective program --plane_head class --tag heightmap_program_adj --epochs 40 \\
       --montage 0 --no_form --out execution/artifacts/height_map_generator_adj_train.json
    $P --diagnose_program outputs/height_map_generator/heightmap_program_adj.pt \\
       --out execution/artifacts/height_map_generator_adj_714.json


#130 -- IS THE FAILURE GRADED BY COMPLEXITY? NO TRAINING RUN, ONE FREE MEASUREMENT
==================================================================================
#132's own report line says a minor-slot recall near zero means "a loss OR A CURRICULUM", and #132
turned only the loss dial. `complexity_strata` prices the other one, off saved weights and the
label cache, and the answer is **no** -- with a mechanism rather than a shrug.

  THE FAILURE IS STEEPLY GRADED. Bucketing the 411 by the LABEL's slot count, #132's arm runs
  `extra` 0.0257 -> 0.1360, collapse 0.094 -> 0.343, and `planar` 1.00 -> 0.00 from 1-slot
  buildings to 4-slot ones. Everything that is wrong with the arm is in the >= 3-slot buildings.

  🔑🔑 BUT ITS SLOT COUNT IS A CONSTANT, SO EXPOSURE IS NOT THE SCARCE THING. The label runs
  1.00 / 2.00 / 3.00 / 4.00 across those buckets and the arm answers 1.56 / 1.97 / 2.15 / **2.16**
  -- it over-fragments the simplest buildings and saturates by the third bucket. And 4-slot
  buildings are already **52.6%** of the carve-needing training rows, so it is not starved of them.
  A curriculum reweights exposure; this arm has not learned a function of the building at all.

  🔑 AND AN EASY-FIRST SCHEDULE IS BACKWARDS, DEFINITIONALLY. Slots are canonicalised by owned
  area, so a building whose label uses two slots cannot supervise slots 2 or 3 AT ALL -- their
  prior in the <= 2 bucket is exactly 0.0000, and 62% of training rows are in it. The hard bucket
  is the LESS imbalanced of the two (slot0:slot3 4.7x on 4-slot rows against 11.9x corpus-wide).
  Pinned in `test_a_low_slot_bucket_gives_the_high_slots_ZERO_support`.

  ⚠️ The binding imbalance is INSIDE each building and no ordering over buildings can reach it.
  #132's logit-adjusted loss flattens it to 1x by construction and already did.

⚠️⚠️ EVERY STRATUM IS A POST-HOC SUBGROUP AND NONE OF THEM PASSES ANYTHING. `PROGRAM_BAR` is
pre-registered on the whole carve-needing population and stays there. #6 already carries one
narrowing-after-the-fact; "but it passes on the easy half" would be the same error with a
population instead of a clause. `label_slots` is written into the scorecard artifact so the same
axis can be applied to the comparison arms without a second run, under the same warning.

    $P --diagnose_program outputs/height_map_generator/heightmap_program_adj.pt \\
       --out execution/artifacts/height_map_generator_strata_714.json

The baselines #6 named, the set-diffusion position and this one are written up in
`docs/wayfinding/solid-first-subtractive-modeling/130-baselines-diffusion-curriculum.md`.


#138 -- THE TYPE HEAD, THE OTHER BINDING CONSTRAINT #132 NAMED AND DID NOT TOUCH
===================================================================================
#132 changed the assignment head and the pitch decode, and left the type head alone by design --
"one change per head was the whole design of this arm". Its own KILL points straight at what it
left: 61% of the arm's used slots are typed `Layer` and 59.2% compile flat, `planar` reaches only
0.12, and that is now the binding constraint on the fourth clause of `PROGRAM_BAR`.

THE FREE QUESTION FIRST, THE SAME ORDER #132 ASKED IT OF THE ASSIGNMENT HEAD. `type_prior` shows
the LABEL itself is steeply slot-index-conditional -- slots are canonicalised by AREA, so a
building's biggest region is a pitch more often than not and its smallest is almost always a flat
setback: slot 0 is Ramp 59.4% of the time over the 34,909 training rows, slot 1 52.3%, slot 2
32.3%, slot 3 13.4%. `type_collapse`, run on `heightmap_program_adj.pt` (#132's checkpoint,
BEFORE this fix was written) asks whether the type head is diffuse, confidently wrong, or tracking
that gradient correctly and just being read at the wrong threshold:

    overall            confidence 0.785   accuracy argmax 0.7576
    slot0 (Ramp 55.5%) recall(Ramp) 0.741   recall(Layer) 0.721   p(Ramp|Ramp label) 0.691
    slot1 (Ramp 50.1%) recall(Ramp) 0.626   recall(Layer) 0.757   p(Ramp|Ramp label) 0.611
    slot2 (Ramp 29.9%) recall(Ramp) 0.357   recall(Layer) 0.939   p(Ramp|Ramp label) 0.421
    slot3 (Ramp 10.5%) recall(Ramp) 0.087   recall(Layer) 0.995   p(Ramp|Ramp label) 0.235

🔑 THE HEAD IS NOT BLIND. `p(Ramp | Ramp label)` exceeds `p(Ramp | Layer label)` at every slot
(slot 3: 0.235 vs 0.128) -- the information is there. It is a plain argmax at a fixed 0.5 threshold
that is the wrong decision rule for a base rate that low, which is a calibration failure and not a
representational one.

⚠️ AND THE DECODE-SIDE FIX IS REFUTED, THE SAME WAY AND FOR THE SAME REASON #132 REFUTED IT ON THE
ASSIGNMENT HEAD. Dividing each slot's posterior by its own label prior before the argmax --
`type_collapse`'s `balanced` column -- recovers slot 3's Ramp recall 0.087 -> 0.957, and pays for it
with Layer recall 0.995 -> 0.413 at that slot and overall accuracy 0.7576 -> 0.6757. It mostly
relabels the building's flat regions as pitched, which is `decode_assignment`'s failure mode
arriving at the second head. `TestTypeStats.test_the_balanced_read_can_flip_a_slot_the_argmax_loses`
pins the mechanism; the served decode stays a plain argmax.

🔑 SO THE FIX GOES WHERE #132'S DID: `tau * log(prior[k, c])` added to slot k's class-c TYPE logit
during training (`TYPE_TEMPERATURE`, fixed a priori at 1.0, the full adjustment, not swept -- the
same reason `ASSIGN_TEMPERATURE` is not swept), decode left as the plain argmax. `type_prior` is
computed once from the TRAINING split's labels, exactly like `assignment_prior`, and travels with
the checkpoint (`type_prior`, `type_temperature`) so a re-scored old checkpoint cannot present
itself as having trained under a correction it did not have.

NOTHING ELSE MOVES: same 40 epochs, same seed, same `plane_head class`, same `PLANE_DECODE`
(pitch q0.25), same `assign_prior` at `ASSIGN_TEMPERATURE` 1.0, same selection rule (validation
`missing + extra`), same 411 carve-needing buildings, same `PROGRAM_BAR`, unchanged for the fourth
time.

⚠️ PRE-REGISTERED PREDICTION, so this is falsifiable: the type fix should raise `planar_fraction`
and the Ramp-typed share of used slots without moving `extra`/`missing`/collapse by much, because
the assignment head and the pitch decode are untouched -- mirroring #132's own assignment fix, which
moved `slots_used_by_arm` and left the surplus pair roughly where it was. If `extra` or the collapse
rate move by more than a rounding amount instead, that is a result about the two heads' losses
interacting, not about either head read in isolation.

    $P --objective program --plane_head class --tag heightmap_program_typeadj --epochs 40 \\
       --montage 0 --no_form --out execution/artifacts/height_map_generator_typeadj_train.json
    $P --diagnose_program outputs/height_map_generator/heightmap_program_typeadj.pt \\
       --out execution/artifacts/height_map_generator_typeadj_714.json

The result is written up in `docs/wayfinding/solid-first-subtractive-modeling/
138-type-head-imbalance.md`.


#139 -- HALVE THE ASSIGNMENT CORRECTION BEFORE STACKING ANOTHER HEAD FIX ON TOP OF A DIFFUSE ONE
====================================================================================================
#138 fixed the type head's own imbalance and it worked exactly as diagnosed -- and its own write-up
named the reason the win cost so much surplus: the (unchanged) assignment head is still diffuse,
confidence 0.34 on a 5-way posterior, and every extra region the type fix makes genuinely planar is
a region the diffuse assignment head may have placed on the wrong columns. Fixing the type head a
second time cannot fix that. This ticket asks the question #138's own "what follows" section named:
does the assignment head's diffuseness have a cheaper fix than a type-temperature sweep?

THE FREE QUESTION FIRST. `assignment_prior`'s pooled `slot0:slot3` skew is 11.9x -- the number
`ASSIGN_TEMPERATURE` = 1.0 is calibrated against. Recomputing the SAME prior restricted to buildings
whose label actually uses >= 2 / >= 3 / exactly 4 slots gives 9.13x / 6.42x / 4.71x. The pooled
figure is not a measurement of how rare a real slot 3 is; it is inflated by 1/2/3-slot buildings
that always own slot 0 and structurally never reach slot 3 at all, pooled in as if their absence
were evidence of rarity rather than of low complexity.

🔑 The standard logit-adjustment recipe (`tau * log(prior)`, plain-argmax decode) targets a UNIFORM
decision boundary at tau=1.0, which is the right target when training and deployment class balance
differ. They do not differ here -- the pinned 714 are drawn from the same distribution as training
-- so tau=1.0 asks the head to hit a balance the corpus does not have. That is a candidate mechanism
for why #132's own disclosed cost (confidence 0.43 -> 0.34, dominant-slot accuracy 0.8251 -> 0.2677)
was so much larger than the recall it bought (0.0000 -> 0.28).

THE CHANGE, in one line: `ASSIGN_TEMPERATURE` 1.0 -> 0.5. Nothing else moves -- same `plane_head
class`, same `PLANE_DECODE` (pitch q0.25), same TYPE head as #132 (no `type_prior`; #138's fix is
deliberately NOT stacked on this run, so the two heads' effects stay separable), same selection
rule, same 411 carve-needing buildings, same `PROGRAM_BAR`.

⚠️ 0.5 is the untuned midpoint between no correction (0.0) and #132's full one (1.0), chosen before
the run and not swept -- this ticket runs exactly one new value. Sweeping tau after seeing where
0.5 lands would be the same near-miss this map has made three times already.

⚠️ PRE-REGISTERED PREDICTION, so this is falsifiable: a smaller tau should recover some dominant-
slot confidence and accuracy, at the cost of giving back some of the minor-slot recall #132 bought
-- a smoother point on the same trade, not a free lunch. If collapse and `extra` fall by more than
that trade would predict, or if minor-slot recall falls to zero, that is a different finding this
ticket did not expect.

    $P --objective program --plane_head class --tag heightmap_program_assign_tau05 --epochs 40 \\
       --montage 0 --no_form --out execution/artifacts/height_map_generator_assign_tau05_train.json
    $P --diagnose_program outputs/height_map_generator/heightmap_program_assign_tau05.pt \\
       --out execution/artifacts/height_map_generator_assign_tau05_714.json

⚠️⚠️ CORRECTION, added after the run above was scored and read. "No `type_prior`" was FALSE: at the
time this ran, #138's type-head correction was wired unconditionally into every `--objective
program` run with no way to disable it, so `heightmap_program_assign_tau05` silently trained WITH
#138's fix stacked on, not without it -- the two heads' effects were never separable in that
checkpoint. Caught by reading the saved checkpoint's own `type_prior` key rather than trusting this
comment. `--no_type_prior` now exists so a run can actually claim this; the TRUE isolated arm is
`heightmap_program_assign_tau05_only`, and the original (accidentally combined) checkpoint is
written up on its own terms as #140. Re-run command for the corrected arm:

    $P --objective program --plane_head class --tag heightmap_program_assign_tau05_only --epochs 40 \\
       --no_type_prior --montage 0 --no_form \\
       --out execution/artifacts/height_map_generator_assign_tau05_only_train.json
    $P --diagnose_program outputs/height_map_generator/heightmap_program_assign_tau05_only.pt \\
       --out execution/artifacts/height_map_generator_assign_tau05_only_714.json

The result is written up in `docs/wayfinding/solid-first-subtractive-modeling/
139-assignment-temperature.md`.


#140 -- THE ACCIDENTAL COMBINED ARM, WRITTEN UP RATHER THAN DISCARDED
====================================================================================================
`heightmap_program_assign_tau05` -- assign tau=0.5 AND #138's type_prior together, the checkpoint
the bug above produced -- is a real 2x2 cell (assign tau x type fix) that #138 and #139 both named
as the natural next arm and neither had actually run on purpose yet. `missing` and collapse compound
favourably (collapse 0.1727, the best on this route); `extra` and `planar_fraction` compound
UNFAVOURABLY (planar 0.33, lower than either single fix alone: 0.50 type-only, 0.67 assign-only).
Not pre-registered as a combined-arm hypothesis before running -- reported because the checkpoint
and its numbers are real, not because the experiment was designed to produce them. Written up in
`docs/wayfinding/solid-first-subtractive-modeling/140-combined-assignment-and-type.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.eval_massing_arms import (              # noqa: E402
    COLLAPSE_MISSING, RES, fp_iou, footprint_split, volume_split, vs_input,
)
from scripts.foundations.measure_scoring_optimum import (        # noqa: E402
    compare_to_envelope, transplant_height,
)
from scripts.foundations.recover_massing_programs import (       # noqa: E402
    CARVE_NEEDED, H5, K_OPS, SHIP714, SLOT_TYPES, FitBias, fit_program_beam, height_field,
    occupancy, plane_surface, program_to_slots, render_iso,
)

LATENTS = REPO / "data/real_massing_v1/vecset_latents.h5"
WORK = REPO / "outputs/height_map_generator"
CACHE = WORK / "height_fields.npz"
PROGRAM_CACHE = WORK / "program_labels.npz"

# One class per voxel of carve depth. Measured over all 35,623 corpus rows: the deepest carve is
# 59 voxels of a 60-voxel extent, and no column is ever cut below 1, so depth lies in [0, 63] and 64
# covers the label range exactly with nothing clipped away at training time.
DEPTH_CLASSES = RES

# Which decode the joint slope term takes its hard forward at. NOT a flag: #127 measured that the
# mode under-carves and the arm is served at its posterior median, so the term must see the surface
# the arm is actually read at. Exposing it as a sweepable option would be selecting on the answer.
SLOPE_DECODE_QUANTILE = 0.5

# Buildings held back from training to select the checkpoint. Drawn from the TRAINING rows -- the
# pinned 714 are never seen, not even for early stopping.
VAL_BUILDINGS = 1000

# #6's slot vocabulary, re-exported under this module's name so the arm's own tests and the demo
# read one spelling. `Layer` is flat and `Ramp` is tilted, and which of the two a slot is is a
# DISCRETE prediction -- that is the single mechanical difference from the plane head below, whose
# slopes were free to decay to zero under L1 and did, from two separate initialisations.
PROGRAM_TYPES = SLOT_TYPES

# The three program-supervision terms are summed at equal weight. Fixed a priori and NOT swept:
# every term is already in commensurate units (two cross-entropies in nats, one L1 in units of the
# building's own height), and this ticket's record has two near-misses from reading a curve as a
# trend. A sweep here would be selecting on the answer.
PROGRAM_TERM_WEIGHTS = dict(assign=1.0, type=1.0, param=1.0)

# #129's discretisation of a slot's plane. One number, used for all three quantities, and it is
# DEPTH_CLASSES -- the resolution #127 already showed a per-column ordinal head learns at, so the
# classified plane head is asked to do nothing at a resolution this trunk has not already done.
# Measured before the run (`plane_quantisation_ceiling`, and the `--diagnose_program` table): at 64
# bins the exact labels still compile to `extra` 0.0063 / `missing` 0.0035 on the 411, which is an
# order of magnitude below the 0.0603 the arm has to beat, so the binning is not the limiter.
# 128 bins buys 0.0039/0.0026 and was NOT taken: the ceiling is already far below the bar and the
# binding constraint is how many examples a class gets, not how fine it is.
PLANE_BINS = DEPTH_CLASSES

# The three quantities, in the order the head emits them. NOT (A, Bz, Cx): see `plane_to_bins`.
PLANE_QUANTITIES = ("offset", "pitch", "azimuth")

# ⚠️ Load-bearing, not a convenience. Pitch bin 0 decodes to EXACTLY zero slope, so a `Layer`'s
# label -- which is (h, 0, 0) with the slope exactly zero, checked over the whole corpus -- survives
# its own encoding. Without it the centroid correction would subtract a bin-centre slope from the
# offset and a flat roof would come back a fraction of a bin off for no reason at all.
PITCH_FLAT_BIN = 0

# 🔑🔑 #132's ASSIGNMENT correction, pre-registered here before the first training step -- and it
# is in the LOSS rather than in the decode, because the decode version was measured and REFUTED
# first. #6 and #129 both used ONE slot where the label uses 3.06, and `assignment_collapse` on
# both of #129's checkpoints says the posterior is DIFFUSE rather than confidently wrong:
# confidence 0.43, normalised entropy 0.80, and on the 201,777 columns whose label is a
# non-dominant slot the per-column argmax recall is **0.0000**.
#
# The MECHANISM, and it is why this is a correction rather than a knob: slots are canonicalised by
# AREA (#6), so slot 0 owns most columns of most buildings and the per-column cross-entropy is
# imbalanced **by construction of the label**, not by anything geometric.
#
# ⚠️ THE POST-HOC VERSION IS REFUTED, measured before this run rather than after it. Dividing the
# posterior by the model's own marginal buys minor-slot recall 0.0000 -> 0.2829 and pays with
# per-column accuracy 0.4245 -> 0.2203 and the DOMINANT slot 0.8251 -> 0.1275: it mostly relabels
# the building, which is `circmean`'s failure from #129 in a second place. The reason is structural
# and `test_and_the_half_that_did_not_it_loses_the_dominant_slot` pins it THROUGH `decode_assignment`
# itself -- a class flat over the plan has p/prior == 1 by construction, so a dominant slot that is
# also nearly flat ties with every other flat class and second-order structure breaks the tie.
#
# 🔑 So the adjustment goes where it can change what is LEARNED instead of rescaling what was not:
# `tau * log(prior)` added to the assignment logits during training, with inference left as the
# plain argmax. That pairing is what makes the trained scores the balanced ones. The prior is the
# LABEL's class frequency over training footprint columns -- a fixed quantity computed once from
# the labels, never from the model and never from the pinned 714.
#
# ⚠️ TEMPERATURE 1.0 is the full adjustment and is deliberately NOT swept: sweeping it trades slot
# count against surplus directly, which is selecting on the answer, and this map has three
# near-misses from exactly that.
#
# 🔑🔑 #139 -- TAU 1.0 IS CALIBRATED AGAINST AN INFLATED SKEW, AND THAT IS WHY THE POSTERIOR WENT
# DIFFUSE. #132's own write-up already disclosed the cost of tau=1.0 -- dominant-slot accuracy
# 0.8251 -> 0.2677, confidence 0.43 -> 0.34, entropy 0.80 -> 0.885 -- as an accepted trade for
# minor-slot recall 0.0000 -> 0.28. #130's complexity_strata measured, and did not act on, the fact
# that the corpus-pooled `slot0:slot3` skew (11.9x, what tau=1.0 is calibrated against) is not the
# within-building skew a multi-region building actually presents: restricting the SAME prior
# computation to buildings whose label uses >=2 slots gives 9.13x; >=3 gives 6.42x; exactly 4 gives
# 4.71x (#130's own number, reproduced here as a cross-check). The pooled figure is inflated by
# 1/2/3-slot buildings, which always own slot 0 and structurally never own slot 3 -- they are not
# evidence that slot 3 is rarer than it is, only that most buildings do not need a slot 3 at all.
#
# The standard logit-adjustment recipe (`tau * log(prior)`, decode unchanged) targets a UNIFORM
# decision boundary at tau=1.0 -- correct when training and deployment class balance differ, which
# is the textbook long-tail setting. Here they do not differ: the pinned 714 are drawn from the same
# distribution as training. Full correction is therefore asking the assignment head to hit a target
# balance the corpus does not have, which is one candidate explanation for why the fix bought minor
# recall by spending far more dominant-slot confidence than #132's own diagnosis priced.
#
# 🔑 THE CHANGE: tau 1.0 -> 0.5, the untuned midpoint between no correction (0.0, the diffuse-but-
# usable head #129 shipped) and full correction (1.0, #132's disclosed over-correction) -- not a
# value chosen after seeing this run, and not a sweep, because it is the only new value this ticket
# runs. `assign_temperature` still travels with every checkpoint, so #132's own numbers remain
# exactly reproducible from its saved weights regardless of what this constant reads afterwards.
ASSIGN_DECODE = "argmax"
ASSIGN_TEMPERATURE = 0.5

# 🔑 #138 -- THE TYPE HEAD'S OWN VERSION OF THE LINE ABOVE. #132 named `used_slots_typed_ramp` 0.390 the
# binding constraint on its own KILL without asking why it reads that -- `type_prior` shows the
# label itself is steeply slot-index-conditional (slot 0 59.4% Ramp, slot 1 52.3%, slot 2 32.3%,
# slot 3 13.4%, over the 34,909 training rows), which is the same construction the ASSIGN_
# correction above answers, one head over: slots are canonicalised by AREA, so a small region is
# usually a flat setback and the plain per-slot cross-entropy is imbalanced by the label, not by
# anything geometric. `type_collapse` asks the diffuse-or-wrong question of this head before this
# correction is trusted, the same order #132 asked it of the assignment head.
#
# The pairing is the one #132 already chose for the sibling head: `tau * log(prior)` added to the
# TYPE logits during training, decode left as the plain argmax -- so a training-side correction
# changes what is learned rather than a decode-side one reshuffling a finished posterior, which
# `test_and_the_half_that_did_not_it_loses_the_dominant_slot` found destroys the dominant class
# when tried on the assignment head. Whether it helps here is #138's question, not assumed by
# writing the mechanism down.
#
# 🔑 `type_collapse` measured on `heightmap_program_adj.pt` (#132's checkpoint) BEFORE this fix was
# written, and it is the same story: the `balanced` DECODE-side read recovers slot 3's Ramp recall
# 0.087 -> 0.957 but pays for it with Layer recall 0.995 -> 0.413 at that slot and overall accuracy
# 0.758 -> 0.676 -- a bigger relabelling than a gain, exactly `decode_assignment`'s failure mode.
# The head is not blind at the slots the label starves: `p(Ramp|Ramp label)` exceeds
# `p(Ramp|Layer label)` at every slot (slot 3: 0.235 vs 0.128), so the information is there and a
# plain argmax at threshold 0.5 is the wrong decision rule for a base rate that low -- a loss-side
# correction, not a decode-side one.
TYPE_TEMPERATURE = 1.0

# 🔑🔑 #129's DECODE, pre-registered here before the first training step and read by
# `decode_plane_logits`, in the order of `PLANE_QUANTITIES`. The reasons are in that docstring and
# they are per-quantity, because this map's biggest lever and its loudest warning are both about
# this line: argmax -> median was worth `extra` 0.1178 -> 0.0603 on #127's ordinal depth head, and
# copying that read onto a symmetric bimodal slope would land straight back on flat.
# 🔑 #132 changes ONE of these three -- the pitch, median -> q0.25. The reason is geometric and was
# written down in #129 before this run: `extra` and `missing` are NOT symmetric in a pitch. A plane
# a little too STEEP dives below GT over the far end of its region and is charged the whole trench;
# one a little too SHALLOW only leaves surplus above it. So the loss-minimising *parameter* estimate
# is a biased *geometry* estimate, and the correction is to read the pitch below the median.
# ⚠️ Disclosed, not relied upon: q0.25 is also the best row of #129's after-the-fact decode table.
# The mechanism predicts "below the median" and 0.25 is the canonical lower quartile rather than the
# argmax of a sweep -- and #6's `PROGRAM_BAR` is unchanged, so no choice of read can lower the bar.
# ⚠️ Pre-registered PREDICTION, so this is falsifiable: on #129's weights q0.25 alone leaves collapse
# at 0.2409, still above 1-NN's 0.1582, so the pitch read is NOT expected to pass on its own. The
# assignment read has to do the rest, and if the arm passes with only one of the two working, that
# is a result about the other.
PLANE_DECODE = ("median", "q0.25", "argmax")

# ⚠️ `recover_massing_programs.FLOOR_EPS` is 1e-9, which is the right snap for the float64 plane
# `linprog` returns. A slot's plane is stored in the label cache and predicted by the network in
# **float32**, where a value that is mathematically 30 arrives as 29.9999996 -- and `floor` then
# reads it as 29. Measured: 96 of 1,280 columns of a plain shed roof, every one of them a plane
# touching an integral target exactly, which is precisely the case the fitter's own snap exists to
# protect. float32 resolution at the top of a 64-grid is ~4e-6, so this is ~25x the noise and still
# four thousand times smaller than the half-voxel that would change a genuine geometric decision.
PLANE_FLOOR_EPS = 1e-4

N_REGIONS = 3          # source corpora: 0 NL / 1 DE / 2 JP, the `region` column of the latent cache
# footprint mask, conditioned extent, log height in metres, distance-to-edge, region one-hot.
# Pinned by `test_the_channel_count_matches_the_model_input` so the two cannot drift apart.
COND_CHANNELS = 4 + N_REGIONS


# ==================================================================================================
# the label, and the invariants of the output space
# ==================================================================================================

def carve_depth(top: np.ndarray, fp: np.ndarray, extent: int) -> np.ndarray:
    """Height map -> per-column carve depth below the blockout. 0 off the footprint."""
    m = np.asarray(fp, bool)
    return np.where(m, int(extent) - np.asarray(top, np.int32), 0).astype(np.int16)


def apply_depth(fp: np.ndarray, extent: int, depth: np.ndarray) -> np.ndarray:
    """Carve depth -> height map, clamped so the result is a valid solid whatever was predicted.

    The clamp is the whole structural argument of #127 and it is deliberately total: it accepts any
    array at all, including negative and out-of-range depths, and still returns a height map that is
    footprint-exact and at least one voxel deep on every footprint column, never taller than the
    blockout. A prediction can therefore be *wrong*, but never *invalid*.
    """
    m = np.asarray(fp, bool)
    e = int(extent)
    h = np.clip(e - np.asarray(depth, np.int32), 1, max(e, 1))
    return np.where(m, h, 0).astype(np.int16)


def height_split(pred: np.ndarray, target: np.ndarray) -> dict:
    """`volume_split` computed in column space, for two height maps sharing a base level.

    Exactly equal to voxelising both and calling `volume_split` -- a column is a solid run from the
    same `y0` in both, so the intersection is `min` per column -- and about 200x cheaper, which is
    what makes it affordable once per epoch on the validation split. `test_height_split_agrees_with_
    volume_split` is the pin; the scored arms still go through `volume_split` on real occupancy so
    the reported numbers stay on the same path as every other arm on this project's record.
    """
    p, t = np.asarray(pred, np.int64), np.asarray(target, np.int64)
    inter, av, gv = int(np.minimum(p, t).sum()), int(p.sum()), int(t.sum())
    union = av + gv - inter
    return dict(vol_iou=float(inter / union) if union else 0.0,
                missing=float((gv - inter) / gv) if gv else 0.0,
                extra=float((av - inter) / gv) if gv else 0.0)


def roof_description_length(surface: np.ndarray, fp: np.ndarray, y0: int, extent: int,
                            max_ops: int = 16, allowance: float = CARVE_NEEDED) -> dict:
    """🔑 The form metric. **How many architectural operations explain this roof?**

    Three amplitude statistics failed at this (`roof_shape_stats`), because GT is itself terraced at
    64^3 and no measure of step size can tell a discretised plane from a mound. What separates them
    is not amplitude, it is *organisation*: a real roof is a handful of planes meeting at ridges, and
    a mound is a continuum of orientations. So the question is asked in the project's own vocabulary
    -- #10's `Layer` / `Ramp` / `CutRoof` fitter is run on the arm's OWN surface, and what is
    reported is the **description length**: the number of operations needed to explain it.

    Validated on shapes whose answer is known by construction (`test_roof_description_length`):

        flat roof              1 op    Layer
        shed (one plane)       1 op    Ramp
        gable (two planes)     2 ops   CutRoof > Ramp
        hip (four planes)      4 ops
        two-step setback       2 ops   Layer > Layer
        a dome                 9 ops   and mostly Layers -- contour terraces, not planes
        noise                 16+ ops  and still not explained

    🔑 The **operation mix** is as diagnostic as the count. Architecture spends its budget on `Ramp`
    and `CutRoof`, which are planes; a mound cannot be explained by planes, so the fitter falls back
    to stacking flat `Layer`s, which is exactly the concentric contour banding the montages show.

    ⚠️ This metric is **not carve-aware, by design**. The footprint envelope is one flat plane and
    scores 1 op, which is *correct* -- the envelope genuinely is planar. Form is a separate axis from
    surplus, and `extra` / `missing` / `vs_input` already police whether the arm acted. Read the two
    together and never this one alone.
    """
    from scripts.foundations.recover_massing_programs import fit_program

    m = np.asarray(fp, bool)
    surf = np.asarray(surface, np.int16)
    prog, fitted = fit_program(m, int(y0), int(y0) + int(extent) - 1, surf, max_ops, allowance)
    vox = int(surf[m].sum())
    residual = float((fitted[m] - surf[m]).sum() / vox) if vox else 0.0
    mix = [o["op"] for o in prog]
    planar = sum(1 for o in mix if o in ("Ramp", "CutRoof"))
    return dict(ops=len(prog), residual=residual, explained=bool(residual <= allowance),
                planar_ops=planar,
                planar_fraction=float(planar / len(mix)) if mix else 0.0)


def smooth_heightmap(h: np.ndarray, fp: np.ndarray, sigma: float) -> np.ndarray:
    """Footprint-masked Gaussian blur of a per-column height map, footprint-exact in and out.

    A plain `gaussian_filter(h)` would blend in the height-0 exterior at every boundary column,
    understating height exactly at the wall. This is "normalised convolution": blur `h * mask` and
    `mask` separately and divide, so a boundary column's smoothed value is the average of its
    FOOTPRINT neighbours only, never diluted by the outside. Undefined columns (the blurred mask is
    ~0, meaning no footprint pixel was within reach of `sigma`) cannot occur inside a footprint of
    any real building at the sigmas this is used at, but are floored to the raw value rather than
    left as a division artefact if they ever do.
    """
    m = np.asarray(fp, bool).astype(np.float64)
    num = ndimage.gaussian_filter(np.asarray(h, np.float64) * m, sigma)
    den = ndimage.gaussian_filter(m, sigma)
    safe = den > 1e-6
    out = np.where(safe, num / np.where(safe, den, 1.0), h)
    return np.where(fp, np.clip(np.rint(out), 1, None), 0).astype(h.dtype)


def fit_decode(heights: np.ndarray, held: dict, max_ops: int = K_OPS,
               allowance: float = CARVE_NEEDED, bias: FitBias | None = None,
               smooth_sigma: float = 0.0) -> np.ndarray:
    """#8's fusion arm: SERVE #10's beam-fitter's output instead of only measuring it.

    `roof_description_length` already runs `fit_program_beam` on an arm's own surface, but only to
    report `dl_ops`/`dl_planar_fraction` as a diagnostic -- the fitted height map it computes along
    the way is discarded. This function keeps it, so a generator's raw per-column prediction (a
    mound, per #127's montage) is replaced by the small typed `Layer`/`Ramp`/`CutRoof` program the
    fitter finds to explain it, compiled back to a height map.

    🔑 The fitter's containment invariant (`fitted` may never drop below the `target` it is fit to)
    gives this a provable, one-directional trade: since `target` here is the ARM's own prediction,
    not GT, `fitted >= heights` on every footprint column WHEN `smooth_sigma == 0`. Relative to GT,
    that means `missing` can only fall or hold and `extra` can only rise or hold. Whether the
    `dl_ops`/`dl_planar_fraction` gain is worth that `extra` cost is an empirical question this
    function does not answer -- it only makes the arm exist so `score_arm` can.

    ⚠️ Measured unbiased, unsmoothed on the served CE+median arm (#8): `dl_ops` 6.0 -> 3.0 but
    `dl_planar_fraction` 0.20 -> **0.00** -- the fitter resolved the generator's own noise into MORE
    flat `Layer` terraces, not fewer pitched planes, because a plane must dominate every point in
    its region while a `Layer` only has to beat the local max, so noise favours `Layer` on raw gain
    every round. `bias` (#9's own `FitBias`, reused rather than a new mechanism) was tried first and
    measured to have **no effect** at #9's own bias strength -- the noise dominates the per-round
    raw-gain ranking too strongly for a soft nudge to flip it.

    `smooth_sigma` attacks the noise itself, upstream of the fitter, instead: `target` is
    `smooth_heightmap(heights[i], fp, smooth_sigma)`, not the raw prediction. 🔑 This trades away
    part of the containment guarantee above -- blurring can pull a column BELOW what the raw model
    predicted there, so `fitted` is only guaranteed `>= the SMOOTHED target`, not `>= heights`
    itself, and the monotonic missing/extra argument no longer holds unconditionally. That is the
    real cost of this variant and is measured, not assumed.
    """
    out = np.zeros_like(heights)
    for i in range(len(heights)):
        fp, y0, extent = held["fp"][i], int(held["y0"][i]), int(held["extent"][i])
        target = heights[i].astype(np.int16)
        if smooth_sigma > 0:
            target = smooth_heightmap(target, fp, smooth_sigma)
        _, fitted = fit_program_beam(fp, y0, y0 + extent - 1, target,
                                     max_ops=max_ops, allowance=allowance, bias=bias)
        out[i] = fitted
    return out


def roof_shape_stats(h: np.ndarray, fp: np.ndarray) -> dict:
    """Three attempts at a scalar for "does this roof look like a building", and all three fail.

    ⚠️ Recorded as a **negative result**, not as a scorecard column that works. The montages show a
    clear difference -- real roofs and the retrieved ones are flat planes meeting at ridges, while
    every trained arm returns a rounded mound with concentric contours -- and none of these
    statistics separates them. Measured on the carve-needing 411:

        arm            relief  curvature  speckle      the eye says
        gt               0.46      0.634    0.000      planes and ridges
        nn_retrieval     0.32      0.454    0.000      planes and ridges (it copies one)
        heightmap_ce     0.47      0.778    0.000      a mound, plus visible speckle
        ..._ce_median    0.40      0.509    0.000      a mound
        heightmap_mse    0.28      0.492    0.000      a mound

    `relief` ranks the worst-looking arm closest to GT and `curvature` ranks two of the mounds
    *smoother* than a real building, so both order the arms nearly opposite to the eye. The cause is
    that **GT is itself terraced at 64^3**: a pitched roof discretises to a staircase, so an
    amplitude statistic cannot tell a discretised plane from a mound. What differs is the
    *organisation* of the steps -- parallel runs against closed contours -- which is a directional
    property none of these three measures.

    This is the same wall map #34 hit ("roughness is prior-side; 2 scalar metrics failed") and #71
    ("ribbing is not melt"). They are kept, computed and published so the attempt is on the record
    and re-checkable, and the visual criterion stays the one that decides.

        relief     mean |height step| between adjacent footprint columns
        curvature  mean |second difference| along each axis; 0 on any plane at any slope
        speckle    fraction of interior columns that are a strict local extremum over 4 neighbours
    """
    a, m = np.asarray(h, np.int32), np.asarray(fp, bool)
    step = np.concatenate([np.abs(a[:, :-1] - a[:, 1:])[m[:, :-1] & m[:, 1:]],
                           np.abs(a[:-1, :] - a[1:, :])[m[:-1, :] & m[1:, :]]])
    curv = np.concatenate([
        np.abs(a[:, :-2] - 2 * a[:, 1:-1] + a[:, 2:])[m[:, :-2] & m[:, 1:-1] & m[:, 2:]],
        np.abs(a[:-2, :] - 2 * a[1:-1, :] + a[2:, :])[m[:-2, :] & m[1:-1, :] & m[2:, :]]])
    core = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    nb = np.stack([a[:-2, 1:-1], a[2:, 1:-1], a[1:-1, :-2], a[1:-1, 2:]])
    ext = ((a[1:-1, 1:-1] > nb).all(0) | (a[1:-1, 1:-1] < nb).all(0)) & core
    return dict(relief=float(step.mean()) if len(step) else 0.0,
                curvature=float(curv.mean()) if len(curv) else 0.0,
                speckle=float(ext.sum() / core.sum()) if core.any() else 0.0)


def envelope_depth(fp: np.ndarray) -> np.ndarray:
    """The do-nothing prediction: carve nothing, which `apply_depth` renders as the blockout."""
    return np.zeros(np.shape(fp), np.int16)


# ==================================================================================================
# the conditioning -- footprint, conditioned height, region. Nothing else may enter.
# ==================================================================================================

def condition_channels(fp: np.ndarray, extent: int, height_m: float, region: int) -> np.ndarray:
    """[C, Z, X] network input built from #127's conditioning ONLY.

    The signature is the leakage guard: there is no argument through which the target height field
    could reach the model, and `test_two_buildings_with_the_same_conditioning_get_identical_input`
    pins it. Two real buildings with the same footprint, height and region are genuinely
    indistinguishable inputs -- #126 measured that they still differ by a median 3D IoU of 0.886,
    which is the irreducible ambiguity this arm is working inside.

    The distance transform is a deterministic function of the footprint, not new information. It is
    supplied because #10 found the roof operations are functions of distance-to-edge (a hip erodes
    on all sides, a gable on one), and a small convolutional net would otherwise spend capacity
    rediscovering it.
    """
    m = np.asarray(fp, bool)
    edt = ndimage.distance_transform_edt(m).astype(np.float32) / 8.0
    ch = [m.astype(np.float32),
          np.full(m.shape, float(extent) / RES, np.float32),
          np.full(m.shape, float(np.log1p(max(height_m, 0.0))) / 4.0, np.float32),
          np.clip(edt, 0.0, 4.0)]
    for r in range(N_REGIONS):
        ch.append(np.full(m.shape, 1.0 if int(region) == r else 0.0, np.float32))
    return np.stack(ch).astype(np.float32)


def decode_logits(logits: np.ndarray, fp: np.ndarray, extent: int,
                  quantile: float | None = None) -> np.ndarray:
    """[K, Z, X] logits -> height map. **Argmax by default**, never by expectation.

    Taking the mean of the predicted distribution would reintroduce at decode time exactly the
    regression-to-the-mean the classification objective exists to avoid: a column whose posterior is
    split between "flat at full height" and "cut to the eaves" has a mean at neither. Argmax is what
    the pre-registered arm decodes with.

    `quantile` decodes the ordinal posterior's q-quantile instead -- the smallest depth whose
    cumulative probability reaches q. It exists because depth is **ordinal**, and the mode of an
    ordinal posterior is a biased estimator of it when one class dominates: with 54% of columns
    carrying depth 0, a column whose posterior is genuinely spread over 0..12 can have its mode at 0
    while its median is at 6. The quantile is fixed a priori at 0.5 by decision theory (the Bayes
    act under absolute error), NOT fitted -- and it is reported as a decode ablation beside the
    pre-registered arm, never in place of it.
    """
    z = np.asarray(logits, np.float64)
    if quantile is None:
        return apply_depth(fp, extent, np.argmax(z, axis=0).astype(np.int16))
    p = np.exp(z - z.max(axis=0, keepdims=True))
    cdf = np.cumsum(p / p.sum(axis=0, keepdims=True), axis=0)
    return apply_depth(fp, extent, np.argmax(cdf >= quantile, axis=0).astype(np.int16))


# ==================================================================================================
# the zero-training baselines #127 names
# ==================================================================================================

def mean_relative_depth(depths: np.ndarray, fps: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """The corpus's mean roof, per grid cell, as a fraction of the building's own height.

    This is the *unconditional* conditional-mean -- the arm #127's design note warns an MSE
    objective converges to. Relative rather than absolute because the corpus normalises each
    building into the grid: averaging voxel depths across a 6-voxel and a 60-voxel building would
    measure the height distribution, not the roof.

    Cells no footprint covers get 0 rather than NaN, so the profile is defined everywhere.
    """
    f = np.asarray(fps, bool)
    rel = np.where(f, np.asarray(depths, np.float32) /
                   np.maximum(np.asarray(extents, np.float32), 1)[:, None, None], 0.0)
    cover = f.sum(0).astype(np.float32)
    return np.divide(rel.sum(0), cover, out=np.zeros(rel.shape[1:], np.float32), where=cover > 0)


def mean_roof_height(profile: np.ndarray, fp: np.ndarray, extent: int) -> np.ndarray:
    """The mean profile rendered on this footprint at this conditioned height."""
    return apply_depth(fp, extent, np.rint(np.asarray(profile, np.float32) * int(extent)))


def retrieve_nn(query_fps: np.ndarray, bank_fps: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Index into `bank_fps` of the footprint-IoU-nearest bank row, for each query.

    Hyper-parameter free on purpose. The footprint is the shape half of the conditioning, and the
    height half is supplied exactly by `transplant_height`'s rescale, so a distance that mixed the
    two would need a weight -- and a *baseline* with a tuned weight is not a baseline. The bank is
    built from training rows only, so a held-out building can never retrieve itself.
    """
    q = np.asarray(query_fps, bool).reshape(len(query_fps), -1).astype(np.float32)
    b = np.asarray(bank_fps, bool).reshape(len(bank_fps), -1).astype(np.float32)
    qa, ba = q.sum(1), b.sum(1)
    out = np.zeros(len(q), np.int64)
    for s in range(0, len(q), chunk):
        inter = q[s:s + chunk] @ b.T
        union = qa[s:s + chunk, None] + ba[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        out[s:s + chunk] = np.argmax(iou, axis=1)
    return out


# ==================================================================================================
# the corpus as height fields, cached once
# ==================================================================================================

def build_cache(path: Path = CACHE, force: bool = False) -> dict:
    """Every corpus row as (footprint, base level, extent, target height map) + its conditioning.

    Keyed by the **latent cache**'s rows, because that file carries `held_out` -- the one split all
    of this project's arms have been scored against. Reading the 64^3 SDFs once and keeping only the
    height field turns 37 GB into 165 MB, which is the whole reason this task trains in minutes.
    """
    import h5py

    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    with h5py.File(LATENTS, "r") as f:
        rows = f["row"][:].astype(np.int32)
        held = (f["held_out"][:] == 1).astype(np.uint8)
        region = f["region"][:].astype(np.int8)
        height_m = f["height_m"][:].astype(np.float32)
    n = len(rows)
    fps = np.zeros((n, RES, RES), np.uint8)
    targets = np.zeros((n, RES, RES), np.uint8)
    y0s = np.zeros(n, np.int16)
    extents = np.zeros(n, np.int16)
    ok = np.zeros(n, np.uint8)
    t0 = time.time()
    with h5py.File(H5, "r") as g:
        for k, b in enumerate(rows):
            gt = np.asarray(g["sdf"][int(b)], np.float32) <= 0
            fp = np.asarray(g["footprint"][int(b)]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, target = hf
            fps[k] = fp
            targets[k] = np.clip(target, 0, 255)
            y0s[k], extents[k], ok[k] = y0, y1 - y0 + 1, 1
            if (k + 1) % 5000 == 0:
                print(f"  [cache] {k+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    out = dict(row=rows, held=held, region=region, height_m=height_m,
               fp=fps, target=targets, y0=y0s, extent=extents, ok=ok)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[cache] {path}  n={int(ok.sum())}/{n}  {time.time()-t0:.0f}s", flush=True)
    return out


# ==================================================================================================
# the model
# ==================================================================================================

OBJECTIVES = ("ce", "mse", "quantile", "planes", "program")

# How `--objective program` predicts a slot's plane. `regress` is #6 and stays the default,
# so every checkpoint and command on that ticket's record is unaffected; `class` is #129.
PLANE_HEADS = ("regress", "class")


def head_channels(objective: str) -> int:
    """`ce` predicts a distribution over depths; the regressions predict one number per column.

    ⚠️ Only meaningful for the objectives with a per-column head. `planes` and `program` predict a
    region plus parameters and never reach here -- `make_model` routes them first -- so they raise
    rather than quietly returning 1, which would build a one-channel U-Net nothing would fit.
    """
    if objective in ("planes", "program"):
        raise ValueError(f"'{objective}' has no per-column head; make_model builds it directly")
    return DEPTH_CLASSES if objective == "ce" else 1


def make_model(objective: str, width: int, k_planes: int, plane_head: str = "regress",
              k_hyp: int = 1):
    """The one place an objective chooses an architecture.

    `k_hyp` (#8) only widens the 'ce' head's final 1x1 conv to `k_hyp` independent copies of the
    same `DEPTH_CLASSES`-channel posterior -- the 4M-parameter U-Net backbone is untouched, so a
    `k_hyp=1` model is bit-for-bit what every prior arm on this file already built.
    """
    if k_hyp > 1 and objective != "ce":
        raise ValueError(f"k_hyp > 1 needs a distribution per hypothesis; "
                         f"'{objective}' has no per-column posterior to multiply")
    if objective == "program":
        return build_program_model(K_OPS, width, plane_head)
    if objective == "planes":
        return build_plane_model(k_planes, width)
    if k_hyp > 1:
        return build_model(head_channels(objective) * k_hyp, width)
    return build_model(head_channels(objective), width)


def forward_heights(model, x, ext, objective: str):
    """Any objective -> a height map in voxels, differentiably. `ce` is excluded: it predicts a
    distribution, and collapsing it to a height before the loss is exactly the mistake this ticket
    found (its loss must see the classes, not their summary)."""
    out = model(x)
    if objective == "planes":
        return compose_planes(out[0], out[1], ext)
    return (1.0 - out[:, 0]) * ext[:, None, None]      # relative depth -> height, in voxels


def per_column_loss(out, y, ext, objective: str, quantile: float):
    """The training loss for one objective, per footprint column, in **voxel units**.

    All three live here rather than as an if/else at each call site, because they are read against
    each other: this ticket's result is that the three differ in *which statistic of the posterior*
    they target, and that is only legible with them side by side.

        ce        cross-entropy over 64 depth classes. Bayes act: the **mode**.
        mse       squared error on relative depth, rescaled to voxels. Bayes act: the **mean**.
        quantile  the pinball loss at `quantile`. Bayes act: that **quantile** -- at q=0.5, the
                  **median**, which is what #127 found the CE arm had to be decoded at anyway.

    🔑 At q=0.5 the pinball loss is L1 up to a factor of 2, so this arm is median regression and
    nothing more exotic. The point is not the loss's novelty, it is that the objective and the
    decode finally name the same statistic: the CE arm was trained for its mode and read at its
    median, and that mismatch is worth `extra` 0.1178 against 0.0603.
    """
    import torch
    import torch.nn.functional as F

    if objective == "ce":
        return F.cross_entropy(out, y.clamp(0, DEPTH_CLASSES - 1), reduction="none")
    if objective == "planes":
        # `out` is already a composed height map; the target height is extent - depth
        err = out - (ext[:, None, None] - y.float())
        return torch.maximum(quantile * -err, (quantile - 1.0) * -err)
    err = (out[:, 0] - y.float() / ext[:, None, None]) * ext[:, None, None]   # voxels, signed
    if objective == "mse":
        return err ** 2
    return torch.maximum(quantile * -err, (quantile - 1.0) * -err)


def differentiable_depth(out, ext, objective: str, quantile: float | None):
    """Network output -> per-column carve depth in voxels, differentiably, with a HARD forward.

    The slope term needs a *height* to take differences of, and a cross-entropy head predicts a
    distribution. Collapsing it with the softmax expectation would measure the slope of a BLENDED
    field, and this ticket's own finding (`compose_planes`) is that a smooth blend of surfaces is a
    mound -- the exact failure being fixed. So the forward pass takes the arm's real decode, and the
    gradient flows back through the soft probabilities: the same straight-through the plane head
    already uses, for the same reason.

    `quantile` picks which decode: `None` is the mode (the pre-registered decode) and 0.5 the
    median (the decode the arm is actually served at). The regressions have no distribution to
    collapse, so their own predicted depth passes straight through.

    ⚠️ The depth is returned UNCLAMPED, so it is `apply_depth`'s input rather than its output. On a
    column the model would carve away entirely the clamp returns a flat one-voxel slab, and a term
    reading that would have no gradient exactly where the prediction is worst.
    """
    import torch

    if objective == "planes":
        return ext[:, None, None] - out                     # `out` is already a height map
    if objective != "ce":
        return out[:, 0] * ext[:, None, None]               # relative depth -> voxels
    p = torch.softmax(out, dim=1)
    levels = torch.arange(DEPTH_CLASSES, device=out.device, dtype=p.dtype).view(1, -1, 1, 1)
    with torch.no_grad():
        # the cumulative sum in float64, because `decode_logits` serves this same posterior in
        # float64: a column whose cdf lands on the quantile picks a different class in the two
        # precisions, and then the term would be shaping a surface the arm is never read at. No
        # gradient passes through the index, so the cast costs a temporary and nothing else.
        idx = (p.argmax(1, keepdim=True) if quantile is None else
               (torch.cumsum(p.double(), 1) >= quantile).float().argmax(1, keepdim=True))
    hard = torch.zeros_like(p).scatter_(1, idx, 1.0)
    # grouped so the straight-through residual is EXACTLY zero in the forward pass: `hard + p - p`
    # left to right rounds through an intermediate and returns 5.9999995 where the decode says 6,
    # and this value is compared against `decode_logits` by test, not just used as a gradient path
    return ((hard + (p - p.detach())) * levels).sum(1)


def slope_loss(depth, y, mask):
    """🔑 The joint term: L1 between the prediction's first differences and GT's, per column PAIR.

    Every objective in `per_column_loss` scores each of the 4,096 plan columns on its own, so the
    ridge line -- a property of a *run* of columns -- is not in any of them, and a mound and a hip
    roof that remove the same volume cost the same. This term is the quantity the normal map draws,
    moved from the picture into the objective: a pitched plane is a constant step along a run and a
    hard break at the ridge, a mound is a step that drifts everywhere.

    Two properties make it an addition to the per-column loss rather than a replacement:

      * it is **blind to a constant offset**, so it says nothing about how deep to carve -- that
        stays cross-entropy's job -- and only about how the carve is arranged;
      * it matches GT's steps rather than minimising them, so a sharp ridge is **free**. A term that
        merely penalised roughness would prefer a rounded ridge, which is the mound again.

    ⚠️ It shapes the loss, not the architecture: at inference the head still emits one posterior per
    column independently. #127's diagnosis is that per-column *independence* is what produces a
    mound, so this is a probe of how much of that can be recovered by supervision alone, and a
    negative result is a real answer to that question.

    Only pairs with both columns inside the footprint are counted: off the footprint there is no
    surface, and the footprint wall is a vertical cliff that would otherwise dominate every edge.
    """
    d, t, m = depth.float(), y.float(), mask.bool()
    dz, tz, mz = d[:, 1:] - d[:, :-1], t[:, 1:] - t[:, :-1], m[:, 1:] & m[:, :-1]
    dx, tx, mx = d[:, :, 1:] - d[:, :, :-1], t[:, :, 1:] - t[:, :, :-1], m[:, :, 1:] & m[:, :, :-1]
    num = ((dz - tz).abs() * mz).sum() + ((dx - tx).abs() * mx).sum()
    return num / (mz.sum() + mx.sum()).clamp(min=1)


def wta_ce_loss(out: "torch.Tensor", y: "torch.Tensor", m: "torch.Tensor",
               k_hyp: int, epsilon: float = 0.05) -> "torch.Tensor":
    """Relaxed winner-take-all cross-entropy over `k_hyp` independent 'ce' heads (#8).

    `slope_loss` above already tried to fix the mound by PENALISING an incoherent surface, and
    #127 measured it does not (`heightmap_ce_slope`: `extra` 0.0651 against plain median's 0.0603,
    `dl_planar_fraction` 0.22 against 0.20 -- noise). A penalty cannot fix this because the failure
    is not incoherence a single head could learn away: when two training buildings share almost the
    same conditioning but genuinely differ (a roof tilts left on one, right on the other), ONE
    per-column head minimising average cross-entropy over both is doing the correct thing by
    hedging -- the hedge (a mound) is the Bayes-optimal single answer to a genuinely bimodal
    target, not a bug a sharper penalty can train out of it.

    Winner-take-all instead gives the network `k_hyp` separate candidate answers for the SAME
    input. For each training building, all `k_hyp` candidates are scored against its real height
    map (summed over that building's own footprint columns, never per-column -- picking a winner
    per column would let one served building be stitched together from different hypotheses'
    columns, which is incoherent by construction and defeats the entire point); whichever
    hypothesis is already closest gets most of the gradient, so gradient descent pushes it to
    specialise further on buildings like this one instead of every hypothesis being pulled toward
    the same compromise. The other `k_hyp - 1` hypotheses still get a small `epsilon` share rather
    than zero -- Rupprecht et al. 2017's "relaxed" WTA -- because plain hard WTA is documented to
    let a hypothesis that loses early in training never win again, and so never learn anything at
    all ("hypothesis death").

    `out` is `[B, k_hyp * DEPTH_CLASSES, Z, X]`; `y`/`m` are the ordinary per-column target/footprint
    mask every other 'ce' loss here already takes.
    """
    import torch
    import torch.nn.functional as F

    B, _, Z, X = out.shape
    logits = out.view(B, k_hyp, DEPTH_CLASSES, Z, X)
    yc = y.clamp(0, DEPTH_CLASSES - 1)
    per_hyp = torch.stack([F.cross_entropy(logits[:, k], yc, reduction="none")
                          for k in range(k_hyp)], dim=1)                       # [B, k_hyp, Z, X]
    mf = m.float().unsqueeze(1)
    whole = (per_hyp * mf).sum(dim=(2, 3)) / mf.sum(dim=(2, 3)).clamp(min=1)   # [B, k_hyp]
    winner = whole.argmin(dim=1)
    weight = torch.full_like(whole, epsilon / max(k_hyp - 1, 1))
    weight.scatter_(1, winner.unsqueeze(1), 1.0 - epsilon)
    return (whole * weight).sum(dim=1).mean()


def decode_wta(out_k: np.ndarray, fp: np.ndarray, extent: int, k_hyp: int,
               quantile: float | None, target: np.ndarray | None = None) -> np.ndarray:
    """Decode a `k_hyp`-headed 'ce' prediction into ONE height map (#8).

    Each of the `k_hyp` slices is an ordinary single-hypothesis 'ce' posterior and is decoded by
    the exact same `decode_logits` every other 'ce' arm uses -- a hypothesis is not a new kind of
    output, there are just several of them.

    `target`, when given, picks the ORACLE hypothesis: whichever candidate has the lowest
    `missing + extra` against it. ⚠️ This is legitimate ONLY where the real answer is already known
    -- training-time validation, or #8's own stage-1 gate ("if even an oracle can't find a good
    roof among k_hyp candidates, no real selector could either") -- and it is NEVER a servable
    decode: nothing at generation time has `target` to cheat with. Callers that score this against
    `target` must keep it out of any `verdict()` comparison the way `program_label (sees GT)`
    already is (`NOT_GENERATORS`), for the same reason. Without `target`, hypothesis 0 is returned:
    an arbitrary, clearly-unfinished placeholder until a real selector exists.
    """
    cands = [decode_logits(out_k[k * DEPTH_CLASSES:(k + 1) * DEPTH_CLASSES], fp, extent, quantile)
             for k in range(k_hyp)]
    if target is None:
        return cands[0]
    scores = [height_split(c, target) for c in cands]
    best = min(range(k_hyp), key=lambda k: scores[k]["extra"] + scores[k]["missing"])
    return cands[best]


def decode_prediction(out_k: np.ndarray, fp: np.ndarray, extent: int, objective: str,
                      quantile: float | None, plane_head: str = "regress") -> np.ndarray:
    """One network output -> one height map. The inverse of `per_column_loss`, kept beside it.

    ⚠️ `quantile` means two different things by objective and that is deliberate: for `ce` it picks
    which statistic to read OUT of a distribution the training never committed to, and for the
    regressions the statistic was fixed at training time and the argument is ignored. Reading a
    trained median at some other quantile is not possible, which is exactly the property that makes
    the `quantile` arm honest and the CE arm's post-hoc median a decode ablation.
    """
    if objective == "ce":
        return decode_logits(out_k, fp, extent, quantile)
    if objective == "program":
        # ⚠️ argmax on both discrete heads, never a blend: a softmax mixture of two slots is a
        # surface belonging to neither, which is #127's mound arriving by a third route. The one
        # place the network's scale-free plane is converted back to the fitter's voxel convention.
        a, t, p = out_k
        assign = decode_assignment(a, fp)
        # 🔑 the assignment is decoded FIRST, because #129's offset is anchored at each slot's own
        # region and the region is what the assignment says it is
        if plane_head == "class":
            p = decode_plane_logits(p, slot_centroids(assign, len(p)))
        return compile_program(assign, np.argmax(t, axis=-1).astype(np.int8),
                               np.stack([plane_to_voxel(p[k], extent) for k in range(len(p))]),
                               fp, extent)
    if objective == "planes":
        return apply_depth(fp, extent, extent - np.rint(out_k))     # out_k is a height map
    return apply_depth(fp, extent, np.rint(out_k[0] * extent))


def build_model(out_channels: int, width: int = 64):
    """A small U-Net over the 64x64 plan. ~4M parameters against A2's 49M and map-24's 947M.

    Depth is chosen so the bottleneck is 8x8 -- one cell there sees an eighth of the plan, which is
    the scale a setback or a ridge line lives at. Nothing here is novel and nothing needs to be:
    #127 is a question about the output space, so the network is the cheapest thing that can answer
    it, and a bigger one would confound the answer.
    """
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(COND_CHANNELS, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.bot = block(4 * w, 4 * w)
            self.d3, self.d2, self.d1 = block(8 * w, 2 * w), block(4 * w, w), block(2 * w, w)
            self.head = nn.Conv2d(w, out_channels, 1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

        def forward(self, x):
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            s3 = self.e3(self.pool(s2))
            b = self.bot(self.pool(s3))
            x = self.d3(torch.cat([self.up(b), s3], 1))
            x = self.d2(torch.cat([self.up(x), s2], 1))
            x = self.d1(torch.cat([self.up(x), s1], 1))
            return self.head(x)

    return UNet()


# ==================================================================================================
# the planar head -- #127's form gap attacked in the representation, not in the loss
# ==================================================================================================

def compose_planes(logits, params, extent, hard: bool = True):
    """K planes plus a per-column assignment -> one height map, **piecewise-planar by construction**.

    🔑 The design move that already worked twice on this ticket: put the invariant in the
    representation rather than in the loss. A clamped height map made *validity* free -- no floating
    voxels are representable. This makes *planarity* free: the output is K planes and an assignment,
    so its description length is at most K by construction, and a mound is not representable at all.

    Why an assignment and not just `min` over the planes. A gable IS the min of two opposing planes,
    and so is a hip -- but a **setback** is not: two flat roofs at different heights over different
    parts of the plan have a min that is just the lower one everywhere. #10 measured `Layer` at
    **75.4%** of all removed volume, so setbacks are the majority of the corpus and a min-only
    composition would be unable to express most of it. The assignment is what makes each operation
    a *region*, which is exactly what `Layer` and `Ramp` are.

    ⚠️ Hard assignment forward, soft gradient backward (straight-through). A softmax BLEND of planes
    is smooth, and a smooth blend of planes is a mound -- the exact failure being fixed. So the
    forward pass must be hard even though that is what makes the gradient awkward.
    """
    import torch

    b, k, res, _ = logits.shape
    zz = torch.linspace(-0.5, 0.5, res, device=logits.device).view(1, 1, res, 1)
    xx = torch.linspace(-0.5, 0.5, res, device=logits.device).view(1, 1, 1, res)
    a, bz, cx = params[..., 0:1, None], params[..., 1:2, None], params[..., 2:3, None]
    planes = (a + bz * zz + cx * xx) * extent.view(b, 1, 1, 1)          # [B,K,res,res], in voxels
    soft = torch.softmax(logits, dim=1)
    if not hard:
        return (soft * planes).sum(1)
    onehot = torch.zeros_like(soft).scatter_(1, soft.argmax(1, keepdim=True), 1.0)
    w = onehot + soft - soft.detach()                                   # straight-through
    return (w * planes).sum(1)


def build_plane_model(k_planes: int, width: int = 64):
    """The same U-Net trunk, with two heads: a per-column assignment and K global plane parameters.

    The planes are **global per building** and the assignment is **spatial**, which is the split the
    vocabulary already has: a `Ramp` is one plane over one region. Pooling the bottleneck is what
    makes a plane a property of the whole building rather than of a neighbourhood, so a ridge line
    stays straight across the plan instead of drifting -- the per-column independence that produced
    the mound is exactly what this removes.
    """
    import torch
    import torch.nn as nn

    trunk = build_model(width, width)          # reuse the tested U-Net; its head becomes features

    class PlaneNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.assign = nn.Conv2d(width, k_planes, 1)
            self.params = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                        nn.Linear(4 * width, k_planes * 3))
            # ⚠️ The initialisation is load-bearing, and the first version of it was wrong. It set
            # every slope to exactly 0 and crushed the head's weights by 100x, so the planes began
            # flat and STAYED flat: measured after 40 epochs, a plane tilted a median of 0.21 voxels
            # across the whole plan. The model became six horizontal terraces -- a ziggurat, which is
            # #10's own name for this failure -- and its `planar_fraction` fell to 0.00 while the
            # per-column model managed 0.20. Planarity was free; SLOPE was not, because a flat region
            # is a strong local optimum under L1 and the straight-through gradient never escaped it.
            #
            # So the planes now start DIVERSE in slope as well as in height: half flat, half tilted
            # by half an extent across the plan in evenly spread directions. Buildings sit at
            # arbitrary grid rotations (#10), so the directions must cover the circle rather than the
            # axes, and the corpus is 54% flat columns, so the flat half is not optional either.
            with torch.no_grad():
                self.params[-1].weight.mul_(0.1)
                bias = torch.zeros(k_planes, 3)
                bias[:, 0] = torch.linspace(0.55, 1.0, k_planes)
                tilted = k_planes // 2
                ang = torch.linspace(0, float(np.pi), tilted + 1)[:tilted]
                bias[k_planes - tilted:, 1] = 0.5 * torch.cos(ang)
                bias[k_planes - tilted:, 2] = 0.5 * torch.sin(ang)
                self.params[-1].bias.copy_(bias.reshape(-1))
            self.k = k_planes

        def forward(self, x):
            f = self.trunk(x)
            p = self.params(f.mean(dim=(2, 3))).view(-1, self.k, 3)
            return self.assign(f), p

    return PlaneNet()


# ==================================================================================================
# #6 -- the program arm. The form gap attacked in the OUTPUT VOCABULARY, not the loss and not a
# soft composition of planes.
# ==================================================================================================

def plane_to_normalised(plane, extent) -> np.ndarray:
    """The fitter's voxel plane `a + b*x + c*z` -> the network's scale-free `(A, Bz, Cx)`.

    The network predicts a roof in units of the building's own height, on plan coordinates running
    -0.5..0.5 -- the convention `compose_planes` already uses, so the two heads stay readable
    against each other. It matters for the same reason `mean_relative_depth` is relative: a 6-voxel
    and a 60-voxel building must not be asked to regress the same number, or the parameter loss
    measures the corpus's height distribution instead of its roofs.

        height_voxels(z, x) = (A + Bz*zn + Cx*xn) * extent,   zn, xn = (i - (RES-1)/2) / (RES-1)
    """
    a, b, c = (float(v) for v in np.asarray(plane, np.float64))
    e = max(float(extent), 1.0)
    return np.array([(a + 0.5 * (RES - 1) * (b + c)) / e,
                     c * (RES - 1) / e,
                     b * (RES - 1) / e], np.float64)


def plane_to_voxel(params, extent) -> np.ndarray:
    """The inverse of `plane_to_normalised`, so the compiler only ever speaks one convention."""
    A, Bz, Cx = (float(v) for v in np.asarray(params, np.float64))
    e = float(extent)
    b, c = Cx * e / (RES - 1), Bz * e / (RES - 1)
    return np.array([A * e - 0.5 * (RES - 1) * (b + c), b, c], np.float64)


# --------------------------------------------------------------------------------------------------
# #129 -- the plane parameters as three CLASSES, and the anchor that makes their range sane
# --------------------------------------------------------------------------------------------------

# The normalised plan coordinates `plane_to_normalised` is written against, and the angle each
# azimuth bin represents. Two 64x64 grids and a 64-vector, built once at import: small enough that
# a lazy cache would only add a way for them to go stale against `PLANE_BINS`.
_PLAN_ZN, _PLAN_XN = ((np.meshgrid(np.arange(RES), np.arange(RES), indexing="ij")[i]
                       - (RES - 1) / 2.0) / (RES - 1) for i in (0, 1))
_AZIMUTH_CENTRES = (np.arange(PLANE_BINS) + 0.5) * (2 * np.pi) / PLANE_BINS


def slot_centroids(assign, k_ops: int = K_OPS) -> np.ndarray:
    """Each slot's own region's centre, in normalised plan coordinates. `(k_ops, 2)` as `(zn, xn)`.

    🔑 A function of the ASSIGNMENT alone, and that is the point: training reads it off the label
    assignment and inference off the predicted one, through this same code, so the offset a slot
    was trained to emit is the offset the decode puts back.

    ⚠️ Total. A slot owning no column at all -- which an untrained head emits constantly -- falls
    back to the plan centre, where the offset means exactly what #6's `A` meant. `compile_program`
    skips such a slot anyway, so the fallback only has to not raise.
    """
    a = np.asarray(assign)
    out = np.zeros((int(k_ops), 2), np.float64)
    for k in range(int(k_ops)):
        m = a == k
        if m.any():
            out[k] = (_PLAN_ZN[m].mean(), _PLAN_XN[m].mean())
    return out


def plane_to_bins(params, centroid, n_bins: int = 0) -> np.ndarray:
    """A slot's normalised plane `(A, Bz, Cx)` -> three class indices. #129's supervision.

    🔑 The three quantities are NOT the three parameters, and the re-parametrisation is the whole
    reason a classifier can be asked for them:

      offset   the plane's height **at its own region's centroid**, in units of the building. `A` is
               the height at the *plan* centre, which a steep plane over a corner region
               extrapolates to 4.4 building-heights (measured over the corpus: `A` runs -1.34 to
               +4.38, while the centroid height runs 0.07 to 0.98). Binning `A` would spend most of
               its resolution on heights no roof is ever at. Anchoring at the region makes the
               quantity a *height on the building*, bounded like a `Layer`'s offset already was.
      pitch    `atan` of the slope magnitude, over [0, pi/2). Non-negative, needs no range constant,
               and it does not clip -- the corpus's steepest fitted plane rises 13.7 building-
               heights across the plan and still lands inside the last bin.
      azimuth  the uphill direction over [0, 2pi).

    🔑🔑 **The split is where the ticket's argument lives.** #6 measured that the signed slope of
    every `Ramp` in the corpus is exactly symmetric -- 50.0% positive, 49.6% negative, median
    +0.0000 -- so an L1 or a quantile on it must return flat. That symmetry is a MIRROR, and a
    mirror leaves the pitch and the region-centroid height untouched and sends the azimuth to its
    antipode. Splitting this way therefore moves the entire symmetry into ONE categorical variable
    and leaves the other two free of it, which is what lets each be decoded by the read its own
    posterior deserves (`decode_plane_logits`).

    ⚠️ Total, like every other function on this path: any plane at all, including the wildly
    out-of-range ones an untrained head emits, lands in a bin rather than raising.

    `n_bins` overrides `PLANE_BINS`, and exists only so `plane_quantisation_ceiling` can price the
    resolutions that were not chosen. ⚠️ It is `n_bins` in both directions on purpose: `bins` is the
    name of this function's *result*, and a parameter spelled the same would put a count and a
    triple of indices under one word on the same line of the round trip.
    """
    nb = int(n_bins) or PLANE_BINS
    A, Bz, Cx = (float(v) for v in np.asarray(params, np.float64))
    zc, xc = (float(v) for v in np.asarray(centroid, np.float64))
    h = A + Bz * zc + Cx * xc
    mag = float(np.hypot(Bz, Cx))
    off = int(np.clip(np.floor(h * nb), 0, nb - 1))
    pit = int(np.clip(np.floor(np.arctan(mag) / (np.pi / 2) * nb), 0, nb - 1))
    azi = int(np.floor((np.arctan2(Cx, Bz) % (2 * np.pi)) / (2 * np.pi) * nb)) % nb
    return np.array([off, pit, azi], np.int64)


def bins_to_plane(bins, centroid, n_bins: int = 0) -> np.ndarray:
    """Three class indices -> a normalised plane `(A, Bz, Cx)`. The inverse of `plane_to_bins`.

    Every bin is represented by its centre, with the one documented exception of `PITCH_FLAT_BIN`.
    The offset is un-anchored last, so a slot's plane is reconstructed to pass through the height
    the offset class names **at that slot's own centroid** -- which is only the same anchor the
    label was encoded at if the assignment agrees, and that is the intended coupling: a slot whose
    region moved should carry its roof with it.
    """
    nb = int(n_bins) or PLANE_BINS
    off, pit, azi = (int(v) for v in np.asarray(bins))
    zc, xc = (float(v) for v in np.asarray(centroid, np.float64))
    h = (off + 0.5) / nb
    mag = 0.0 if pit == PITCH_FLAT_BIN else float(np.tan((pit + 0.5) * (np.pi / 2) / nb))
    th = (azi + 0.5) * (2 * np.pi) / nb
    Bz, Cx = mag * np.cos(th), mag * np.sin(th)
    return np.array([h - Bz * zc - Cx * xc, Bz, Cx], np.float64)


def decode_plane_logits(logits, centroids, reads=None) -> np.ndarray:
    """🔑🔑 #129's decode, and most of the ticket. `(K, 3, PLANE_BINS)` logits -> `(K, 3)` planes.

    This map's single biggest lever was the decode of exactly such a head: argmax -> posterior
    median on #127's depth classifier moved `extra` 0.1178 -> 0.0603, one line. ⚠️ And #129's own
    warning is that copying that read wholesale here would land straight back on flat, because the
    median of a symmetric bimodal slope IS zero. So the read is chosen per quantity, from the shape
    of that quantity's posterior, and pre-registered here before the first training step:

      offset   **posterior MEDIAN.** Ordinal, and free of the symmetry: the mirrored roof has the
               same height at its own region's centroid, so both competing hypotheses agree on this
               number and averaging them costs nothing. #127's read, where #127's argument holds.
      pitch    **posterior MEDIAN.** Ordinal, non-negative, and *invariant* under the mirror -- the
               two roofs the conditioning cannot choose between have the SAME pitch -- so its
               posterior is not the symmetric bimodal one and the median is not defeated by it.
               The marginal is the conditional, which is why reading it from a separate head costs
               nothing.
      azimuth  **ARGMAX.** Categorical, and its posterior is antipodally bimodal by construction
               (#6: 50.0% / 49.6%). A circular mean or median over two opposite modes returns a
               direction *neither* holds, and a plane pointing nowhere in particular is #127's mound
               arriving by a third route. `argmax` commits to one of the two roofs, which is what
               #126 says the task is: the conditioning does not determine which, so pick one.

    ⚠️ The alternative reads are measured (`decode_ablation`) but `PLANE_DECODE` above is fixed in
    this docstring beforehand. A decode picked after seeing which scored best would be selecting on
    the answer, which is the mistake this map has three near-misses from. `reads` exists ONLY so
    that ablation can re-read the same weights; it defaults to the pre-registered triple.
    """
    lg = np.asarray(logits, np.float64)
    cen = np.asarray(centroids, np.float64)
    rd = tuple(reads or PLANE_DECODE)
    p = np.exp(lg - lg.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(p, axis=-1)

    def read(q: int) -> np.ndarray:
        r = rd[q]
        if r == "median" or r.startswith("q"):
            # "median" is q0.50, spelled out so the pre-registered read and an ablation quantile go
            # through the same line and cannot drift apart
            return (cdf[:, q] >= (0.5 if r == "median" else float(r[1:]))).argmax(axis=-1)
        if r == "argmax":
            return p[:, q].argmax(axis=-1)
        if r == "circmean":
            # ⚠️ azimuth only, and it raises rather than returning a number for the other two: the
            # bin centres it averages are ANGLES, so applied to an offset or a pitch it would
            # silently produce a plausible-looking index with no meaning at all
            if PLANE_QUANTITIES[q] != "azimuth":
                raise ValueError(f"'circmean' is an angular read; PLANE_QUANTITIES[{q}] is "
                                 f"'{PLANE_QUANTITIES[q]}'")
            th = _AZIMUTH_CENTRES
            return np.floor((np.arctan2((p[:, q] * np.sin(th)).sum(-1),
                                        (p[:, q] * np.cos(th)).sum(-1)) % (2 * np.pi))
                            / (2 * np.pi) * PLANE_BINS).astype(np.int64) % PLANE_BINS
        raise ValueError(f"unknown read '{r}'; expected median, argmax, circmean or q<float>")

    bins = np.stack([read(q) for q in range(len(PLANE_QUANTITIES))], axis=-1)
    return np.stack([bins_to_plane(bins[k], cen[k]) for k in range(len(bins))])


def assignment_prior(assign, fp, k_ops: int) -> np.ndarray:
    """The label's slot frequency over footprint columns: `(K+1,)`, summing to 1.

    🔑 The imbalance #132 corrects, measured from the LABEL rather than from the model. Slots are
    canonicalised by area, so this is steeply skewed by construction and that is the whole point.

    ⚠️ Footprint columns only, and computed on the TRAINING split only. Off-footprint columns are
    compiled away, and a prior that had seen the pinned 714 would be a leak.
    """
    a = np.asarray(assign)
    m = np.asarray(fp, bool)
    counts = np.bincount(a[m].ravel().astype(np.int64), minlength=k_ops + 1)[:k_ops + 1]
    return (counts / max(counts.sum(), 1)).astype(np.float64)


def decode_assignment(logits, fp, read: str | None = None,
                      temperature: float | None = None) -> np.ndarray:
    """`(K+1, Z, X)` assignment logits -> the per-column slot the compiler receives.

    🔑 The ONE place the assignment is decoded, so the served path, the training-time validation
    that selects the checkpoint, and the diagnostics all read it the same way. Two argmaxes in two
    files is how a decode quietly stops being the one that is served.

    `balanced` divides each column's posterior by the model's own per-building marginal before the
    argmax -- the standard logit adjustment, and here it corrects an imbalance the LABEL creates:
    slots are canonicalised by area, so slot 0 owns most columns and a per-column cross-entropy is
    imbalanced by construction. See `ASSIGN_DECODE` for the measurement that chose it and the
    over-fragmentation risk it carries.

    ⚠️ The marginal is taken over the FOOTPRINT only. Off-footprint columns are compiled away, and
    letting them into the prior would let a class that never fires on the building set the scale.
    An empty footprint falls back to the plain argmax rather than dividing by nothing.
    """
    lg = np.asarray(logits, np.float64)
    p = np.exp(lg - lg.max(axis=0, keepdims=True))
    p /= p.sum(axis=0, keepdims=True)
    m = np.asarray(fp, bool)
    if (ASSIGN_DECODE if read is None else read) == "argmax" or not m.any():
        return p.argmax(axis=0).astype(np.uint8)
    # ⚠️ `is None`, not `or`: tau = 0.0 is the identity adjustment and a caller asking for it must
    # get it rather than the pre-registered 1.0
    tau = ASSIGN_TEMPERATURE if temperature is None else temperature
    prior = np.clip(p[:, m].mean(axis=1), 1e-12, None)
    return (p / (prior ** tau)[:, None, None]).argmax(axis=0).astype(np.uint8)


def rebin_planes(planes, assign, extent, n_bins: int = 0) -> np.ndarray:
    """A whole slot set's planes through the bins and back, in the fitter's voxel convention.

    🔑 The round trip in one place. `plane_quantisation_ceiling` prices the bins with it and the
    tests check it against a fitted surface and under the 8 plan symmetries -- so a test cannot pass
    on its own private copy of a path the production code has since changed, which is the only way
    a discretisation quietly drifts away from the ceiling it was chosen on.
    """
    cen = slot_centroids(assign, len(planes))
    return np.stack([
        plane_to_voxel(bins_to_plane(plane_to_bins(plane_to_normalised(planes[k], extent),
                                                   cen[k], n_bins), cen[k], n_bins), extent)
        for k in range(len(planes))])


def compile_program(assign, types, planes, fp, extent) -> np.ndarray:
    """A predicted program -> one height map. The output space of #6's arm.

    🔑 **What this makes free, and it is a different thing from what #127's two representations
    made free.** The clamped height map made *validity* free; the plane head was meant to make
    *planarity* free and did, and it still terraced, because a plane whose slope may drift to zero
    is a flat region wearing a plane's name. Here the slot's **type** is a discrete prediction the
    compiler obeys: `Layer` ignores the slope it was handed and `Ramp` compiles the plane it was
    given, so "flat" and "pitched" are different answers rather than the same answer at different
    magnitudes. A slot cannot quietly become a terrace.

    And it is **joint** by construction, which is the property #127 measured to be missing from both
    ends: the ridge line falls out of one shared plane across a whole region, rather than out of
    4,096 columns that each summarised their own posterior and averaged a family of roofs into a
    mound.

    ⚠️ Total, exactly like `apply_depth`, and for the same reason. It accepts any assignment, any
    type and any plane at all -- including the wildly out-of-range params an untrained head emits --
    and still returns a footprint-exact height map with at least one voxel under every footprint
    column and nothing above the blockout. A run may then fail for a bad answer, never for an
    unrepresentable one.

    `planes` is in the fitter's voxel convention, so a program straight out of `program_to_slots`
    compiles without conversion and a *predicted* one is converted once, in `decode_prediction`.
    """
    m = np.asarray(fp, bool)
    e = int(extent)
    a = np.asarray(assign)
    t = np.asarray(types, np.int32)
    p = np.asarray(planes, np.float64)
    h = np.full(m.shape, float(e), np.float64)
    ramp = SLOT_TYPES.index("Ramp")
    for k in range(len(p)):
        sel = a == k
        if not sel.any():
            continue
        # a slot that is inactive (-1) or typed `Layer` is flat: its slope is not read at all
        flat = (p[k, 0], 0.0, 0.0)
        h = np.where(sel, plane_surface(p[k] if t[k] == ramp else flat, PLANE_FLOOR_EPS), h)
    return np.where(m, np.clip(h, 1, max(e, 1)), 0).astype(np.int16)


def program_loss(out, labels, mask, plane_head: str = "regress", assign_prior=None,
                 type_prior=None):
    """🔑 #6's training strategy, in one function: supervise the **program**, never the surface.

    #127 established the trap this avoids, twice and from both directions. Supervision on the
    surface could not put planes in -- an L1 has a flat region as a strong local optimum, so the
    plane head's slopes collapsed to 0.25 voxels across a 40-voxel building from two different
    initialisations -- and no decode could take a roof out of a per-column posterior. So no term
    here reads the compiled height map at all. Each term sees a piece of the program:

        assign  cross-entropy per footprint column over the K slots plus the UNCARVED class. This
                is where the *regions* are learned, and it is a segmentation, not a height.
        type    cross-entropy per ACTIVE slot over (Layer, Ramp). The discrete flat-or-pitched
                decision that a straight-through slope could never make. `type_prior`, if given,
                logit-adjusts it the same way `assign_prior` adjusts the assignment term: slots are
                canonicalised by AREA, so the label's own Ramp share falls from 59% at slot 0 to
                13% at slot 3, and `type_collapse` measured that a plain argmax under-recalls Ramp
                hardest exactly where that base rate is lowest.
        param   `regress` (#6): L1 on the plane, in units of the building's own height, per active
                slot -- and on the OFFSET ONLY for a slot typed `Layer`, because a flat roof's slope
                is not a quantity the label has an opinion about and regressing it towards zero
                would spend capacity teaching the model something the compiler already ignores.

                `class` (#129): the MEAN of three cross-entropies -- offset, pitch and azimuth --
                over the bins `plane_to_bins` defines, with the same masking. ⚠️ The mean, not the
                sum, so the term keeps the weight and the scale #6 gave it: three nats averaged is
                one nats-valued `param` term, `PROGRAM_TERM_WEIGHTS` is untouched, and nothing here
                was swept. #6 refuted the L1 with a proof rather than a number -- the signed slope
                of every `Ramp` in the corpus is exactly symmetric, so the objective's own Bayes act
                is a flat roof however long it trains -- and a cross-entropy has no such act: its
                Bayes act is the whole posterior, and what is done with it is the decode's problem
                (`decode_plane_logits`), which is where #129 says the difficulty actually is.

    ⚠️ Inactive slots contribute nothing to any term. A building the fitter explained in two
    operations must not be pushed to invent four; #10 measured a median of 4 and a mode of 4 at the
    budget, but 59 of the 411 carve-needing buildings need exactly one.
    """
    import torch
    import torch.nn.functional as F

    assign_logits, type_logits, params = out
    assign, types, planes = labels
    m = mask.bool()

    if assign_prior is not None:
        # 🔑 #132's logit adjustment, training-side only. Adding `tau * log(prior)` to class k's
        # logit makes the model earn the majority slot instead of being handed it, so the plain
        # argmax at inference reads the BALANCED posterior. See `ASSIGN_DECODE` for the measurement
        # that put this here rather than in the decode.
        pri = torch.as_tensor(assign_prior, dtype=assign_logits.dtype,
                              device=assign_logits.device).clamp_min(1e-12)
        assign_logits = assign_logits + ASSIGN_TEMPERATURE * torch.log(pri).view(1, -1, 1, 1)
    ce = F.cross_entropy(assign_logits, assign, reduction="none")
    l_assign = (ce * m).sum() / m.sum().clamp(min=1)

    if type_prior is not None:
        # the type-head twin of the assignment adjustment above: `tau * log(prior[k, c])` added to
        # slot k's class-c logit, so the plain argmax at inference reads a posterior that has
        # already earned the minority type rather than one rescaled after the fact.
        tpri = torch.as_tensor(type_prior, dtype=type_logits.dtype,
                               device=type_logits.device).clamp_min(1e-12)
        type_logits = type_logits + TYPE_TEMPERATURE * torch.log(tpri).unsqueeze(0)
    active = types >= 0
    n_active = active.sum().clamp(min=1)
    l_type = (F.cross_entropy(type_logits[active], types[active], reduction="sum") / n_active
              if bool(active.any()) else assign_logits.sum() * 0.0)

    # a Layer's slope is not in the label, so it is not in the loss: weight the offset everywhere
    # and the two slope quantities only where the slot is a Ramp. One mask, both heads.
    is_ramp = (types == SLOT_TYPES.index("Ramp")) & active
    w = torch.stack([active.float(), is_ramp.float(), is_ramp.float()], dim=-1)
    if plane_head == "class":
        # (B, K, 3, PLANE_BINS) logits against (B, K, 3) bins
        # ⚠️ the clamp is unreachable for labels this repo builds -- `plane_to_bins` already
        # clips -- and it stays because the failure it prevents is not debuggable: an out-of-range
        # class index into `cross_entropy` on CUDA is a device-side assert with no line number,
        # from anywhere in the epoch. A wrong label would be caught by the tests; a wrong *caller*
        # would not, and this makes that a wrong number rather than an opaque crash.
        flat = F.cross_entropy(params.reshape(-1, params.shape[-1]),
                               planes.reshape(-1).clamp(0, params.shape[-1] - 1),
                               reduction="none").view(w.shape)
        l_param = (flat * w).sum() / w.sum().clamp(min=1)
    else:
        l_param = ((params - planes).abs() * w).sum() / w.sum().clamp(min=1)

    return (PROGRAM_TERM_WEIGHTS["assign"] * l_assign +
            PROGRAM_TERM_WEIGHTS["type"] * l_type +
            PROGRAM_TERM_WEIGHTS["param"] * l_param)


def build_program_model(k_ops: int, width: int = 64, plane_head: str = "regress"):
    """The same U-Net trunk, with an assignment head and a slot head. ~3.6M parameters.

    The split is the vocabulary's own: an operation is **one plane over one region**, so the plane
    is pooled to a property of the whole building and the region stays spatial. That is what keeps a
    ridge line straight across the plan instead of drifting -- the per-column independence #127
    diagnosed as the cause of the mound is exactly what pooling removes.

    The slot head emits type logits and plane parameters together, from the same pooled feature, so
    a slot's "am I flat" decision and its slope are read off one representation rather than two.

    🔑 `plane_head` is the ONE mechanical difference between #6's arm and #129's, and it is confined
    to the last layer's width: `regress` emits 3 numbers per slot and `class` emits
    `3 x PLANE_BINS` logits. Same trunk, same assignment head, same type head, same loss structure,
    same decode for everything except the plane -- so a difference between the two arms is
    attributable to the plane head and to nothing else. It costs 0.19M parameters, 3.39M -> 3.58M,
    which is 5.7% and far too little to be an explanation of any gap on its own.
    """
    import torch
    import torch.nn as nn

    trunk = build_model(width, width)             # the tested U-Net; its head becomes features
    n_type = len(SLOT_TYPES)
    n_quant = len(PLANE_QUANTITIES)
    n_plane = n_quant * PLANE_BINS if plane_head == "class" else n_quant

    class ProgramNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.k = k_ops
            self.plane_head = plane_head
            self.assign = nn.Conv2d(width, k_ops + 1, 1)
            self.slots = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                       nn.Linear(4 * width, k_ops * (n_type + n_plane)))
            # ⚠️ #127's plane head recorded that this initialisation is load-bearing: starting every
            # plane flat left them flat after 40 epochs. The labels here supervise the slope
            # directly, so the failure cannot repeat for that reason -- but the slots are still
            # canonicalised by AREA, so slot 0 sees mostly large flat setbacks and the later slots
            # the small pitched pieces, and starting them identical wastes the early epochs
            # separating them. Heights spread over the top of the building, slopes spread around
            # the circle because buildings sit at arbitrary grid rotations (#10).
            with torch.no_grad():
                self.slots[-1].weight.mul_(0.1)
                bias = torch.zeros(k_ops, n_type + n_plane)
                heights = torch.linspace(0.9, 0.5, k_ops)
                ang = torch.linspace(0, float(np.pi), k_ops + 1)[:k_ops]
                if plane_head == "class":
                    # the same spread, expressed in the classified parametrisation: nudge slot k's
                    # offset and azimuth towards the bin its regressed counterpart started at. A
                    # nudge and not a one-hot -- a saturated init would take epochs to unlearn.
                    pl = bias[:, n_type:].view(k_ops, n_quant, PLANE_BINS)
                    for k in range(k_ops):
                        p = np.array([float(heights[k]), 0.25 * float(torch.cos(ang[k])),
                                      0.25 * float(torch.sin(ang[k]))])
                        for q, b in enumerate(plane_to_bins(p, np.zeros(2))):
                            pl[k, q, int(b)] = 1.0
                else:
                    bias[:, n_type] = heights
                    bias[:, n_type + 1] = 0.25 * torch.cos(ang)
                    bias[:, n_type + 2] = 0.25 * torch.sin(ang)
                self.slots[-1].bias.copy_(bias.reshape(-1))

        def forward(self, x):
            f = self.trunk(x)
            s = self.slots(f.mean(dim=(2, 3))).view(-1, self.k, n_type + n_plane)
            planes = s[..., n_type:]
            if self.plane_head == "class":
                planes = planes.view(-1, self.k, n_quant, PLANE_BINS)
            return self.assign(f), s[..., :n_type], planes

    return ProgramNet()


def _d4_program(assign, types, planes, k: int, flip: bool):
    """One plan symmetry applied to a PROGRAM, so #6's arm keeps the 8x augmentation every other
    arm on this ticket trains with.

    The assignment is an image and rotates with the footprint. A plane is not: `height = a + b*x +
    c*z` has to be re-expressed in the rotated frame, and getting that wrong would silently train
    the arm on roofs tilted the wrong way -- an error no shape test on the footprint would catch,
    which is why `test_the_augmented_program_compiles_to_the_augmented_surface` compares the
    compiled surfaces rather than the parameters.

    `np.rot90(a)[z, x] = a[x, n-1-z]`, so height'(z, x) = a + b*(n-1-z) + c*x, and the flip
    `a[:, ::-1]` sends x -> n-1-x. Both are applied in the same order as `_d4`.
    """
    n = RES - 1
    a, b, c = planes[:, 0].copy(), planes[:, 1].copy(), planes[:, 2].copy()
    for _ in range(k % 4):
        a, b, c = a + b * n, c.copy(), -b
    if flip:
        a, b = a + b * n, -b
    out = np.stack([a, b, c], axis=1).astype(np.float32)
    ass = np.rot90(assign, k)
    if flip:
        ass = ass[:, ::-1]
    return np.ascontiguousarray(ass), types.copy(), out


# beam=12, not `fit_program_beam`'s own default of 6: it is what the committed 714-building
# recovery used, so the labels and the `program K=16 (sees GT)` row of REFERENCE are the same fit.
def build_program_cache(cache: dict, path: Path = PROGRAM_CACHE, force: bool = False,
                        workers: int = 0, beam: int = 12, branch: int = 6) -> dict:
    """#10's fitter run over the whole corpus, decomposed into slots. #6's supervision.

    🔑 This is why #6 is a supervised-learning ticket and not a program-induction one. The literature
    #6 names reaches for pseudo-labels, RL or a differentiable relaxation precisely because exact
    programs are usually unavailable -- and here they are not: the fitter is deterministic, sees GT,
    reaches a median `extra` of 0.003, and costs **0.2 s per building**, so the entire 35,623-row
    corpus labels in **56 s** on 48 of this machine's 64 cores. Measured before the arm was
    designed, and it is the fact that chose the formulation.

    ⚠️ Fitted with `CutRoof` withheld, so every operation is a plane and `program_to_slots` is
    lossless. `CutRoof` was 13 of 1,246 operations (1.0%) in the committed 714-building recovery,
    and `--ops_allowed` on the recovery script measures what withholding it costs rather than
    assuming it away.
    """
    import multiprocessing as mp

    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    idx = np.nonzero(cache["ok"] > 0)[0]
    jobs = [(cache["fp"][i] > 0, int(cache["extent"][i]),
             cache["target"][i].astype(np.int16), beam, branch) for i in idx]
    n = len(cache["ok"])
    assign = np.full((n, RES, RES), K_OPS, np.uint8)
    types = np.full((n, K_OPS), -1, np.int8)
    planes = np.zeros((n, K_OPS, 3), np.float32)
    residual = np.zeros(n, np.float32)
    t0 = time.time()
    workers = workers or min(mp.cpu_count(), 48)
    with mp.Pool(workers) as pool:
        for done, (i, out) in enumerate(zip(idx, pool.imap(_fit_one_slots, jobs, chunksize=8))):
            assign[i], types[i], planes[i], residual[i] = out
            if (done + 1) % 5000 == 0:
                print(f"  [program] {done+1}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)
    out = dict(assign=assign, types=types, planes=planes, residual=residual,
               ok=cache["ok"], row=cache["row"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[program] {path}  n={len(idx)}  median fitted extra "
          f"{float(np.median(residual[idx])):.4f}  {time.time()-t0:.0f}s", flush=True)
    return out


def _fit_one_slots(job):
    """One building's program labels. Module level so `multiprocessing` can pickle it."""
    fp, extent, target, beam, branch = job
    ops, fitted = fit_program_beam(fp, 0, extent - 1, target, max_ops=K_OPS,
                                   beam=beam, branch=branch, ops_allowed=SLOT_TYPES)
    assign, types, planes = program_to_slots(fp, extent, ops)
    vox = int(target[fp].sum())
    residual = float((fitted[fp] - target[fp]).sum() / vox) if vox else 0.0
    return assign, types, planes, residual


def _d4(fp, target, k: int, flip: bool):
    """One of the 8 plan symmetries, applied to footprint and label together.

    Buildings sit at arbitrary grid rotations already (#10: an axis-aligned ramp could not fix the
    shed-roof residual), so the symmetry group is a property of the corpus rather than an assumption
    imposed on it. The conditioning is rebuilt from the rotated footprint, so nothing can desync.
    """
    fp, target = np.rot90(fp, k), np.rot90(target, k)
    if flip:
        fp, target = fp[:, ::-1], target[:, ::-1]
    return np.ascontiguousarray(fp), np.ascontiguousarray(target)


class HeightFieldSet:
    """Conditioning + label for one split, materialised on demand so augmentation stays honest.

    `program` adds #6's slot labels alongside the per-column depth rather than in place of it: the
    depth label is still what the validation geometry is measured against, so the program arm is
    selected by exactly the rule every other arm on this ticket was.

    ⚠️ `plane_head="class"` changes what the third program tensor *is* -- integer bins rather than
    float parameters -- and it is built here rather than in the loss because the bins are anchored
    at each slot's own centroid and the centroid has to be taken from the SAME augmented assignment
    the plane was rotated with. Doing it downstream would let the two desync silently.
    """

    def __init__(self, cache: dict, idx: np.ndarray, augment: bool, seed: int = 0,
                 program: dict | None = None, plane_head: str = "regress"):
        self.plane_head = plane_head
        self.fp = cache["fp"][idx] > 0
        self.target = cache["target"][idx].astype(np.int16)
        self.extent = cache["extent"][idx].astype(np.int32)
        self.height_m = cache["height_m"][idx]
        self.region = cache["region"][idx].astype(np.int32)
        self.augment, self.rng = augment, np.random.default_rng(seed)
        self.program = None
        if program is not None:
            self.program = dict(assign=program["assign"][idx], types=program["types"][idx],
                                planes=program["planes"][idx])

    def __len__(self):
        return len(self.fp)

    def batch(self, sel: np.ndarray):
        xs, ys, pa, pt, pp = [], [], [], [], []
        for i in sel:
            fp, target = self.fp[i], self.target[i]
            k, flip = (int(self.rng.integers(4)), bool(self.rng.integers(2))) if self.augment \
                else (0, False)
            if self.augment:
                fp, target = _d4(fp, target, k, flip)
            xs.append(condition_channels(fp, int(self.extent[i]), float(self.height_m[i]),
                                         int(self.region[i])))
            ys.append(carve_depth(target, fp, int(self.extent[i])))
            if self.program is not None:
                # ⚠️ the SAME symmetry as the footprint above, drawn once: a program augmented
                # independently of its own plan would supervise a roof on the wrong building
                a, t, p = _d4_program(self.program["assign"][i], self.program["types"][i],
                                      self.program["planes"][i], k, flip)
                pa.append(a)
                pt.append(t)
                n = np.stack([plane_to_normalised(p[j], int(self.extent[i]))
                              for j in range(len(p))])
                if self.plane_head == "class":
                    cen = slot_centroids(a, len(p))
                    n = np.stack([plane_to_bins(n[j], cen[j]) for j in range(len(p))])
                pp.append(n)
        prog = (np.stack(pa).astype(np.int64), np.stack(pt).astype(np.int64),
                np.stack(pp).astype(np.int64 if self.plane_head == "class" else np.float32)) \
            if self.program is not None else None
        return (np.stack(xs), np.stack(ys).astype(np.int64),
                self.extent[sel].astype(np.float32), prog)


def train(cache: dict, args) -> Path:
    """Train one arm and return its selected checkpoint.

    ⚠️ Selection is on a validation split drawn from the TRAINING rows. The pinned 714 are not read
    here at all. This project's record has two near-misses from reading a training curve as a trend
    (#80, twice), so the checkpoint is chosen by a held-in number and the whole curve is written to
    the artifact rather than summarised.

    🔑 It is chosen on the **geometry**, not on the loss. #75/#76 measured that neither the training
    loss nor latent distance tracked the goal on this project, and #76 found latent distance was
    *wrong-signed* pooled across error families. A cross-entropy is a proxy; the height field it
    decodes to is the thing, and it costs one argmax per validation building per epoch to measure
    directly.

    ⚠️ The criterion is `missing + extra` -- the symmetric difference, normalised by GT volume --
    and NOT `extra` alone. Selecting on `extra` was tried first and is unsound in a way that only
    showed up on the MSE arm: an arm that carves the building away scores `extra` 0, so the rule
    picked that arm's **first epoch** (`extra` 0.039, `missing` 0.082) and would then have failed
    the collapse guard for a reason belonging to the selection rule rather than to the objective.
    The symmetric difference cannot be gamed from either end -- no-op and over-carve are both
    penalised -- and it needs no threshold. Validation loss is recorded for the curve and breaks
    ties.

    🔑 The FINAL epoch is also written, as `<tag>_last.pt`, and it is a **diagnostic and never the
    arm**. The rule above is the only thing that selects. It exists because #129's run is the first
    on this ticket where the rule and the bar point at different epochs -- the classified head
    trades `missing` for `extra` as it trains, so the symmetric difference stops improving while
    `extra` keeps falling -- and "what would the endpoint have scored" is then a question the
    record should be able to answer from disk instead of by argument. Reporting it is disclosure;
    scoring the run on it would be selecting on the answer.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pool = np.nonzero((cache["ok"] > 0) & (cache["held"] == 0))[0]
    perm = np.random.default_rng(args.seed).permutation(len(pool))
    val_idx, tr_idx = pool[perm[:VAL_BUILDINGS]], pool[perm[VAL_BUILDINGS:]]
    prog = (build_program_cache(cache, force=args.rebuild_program_cache)
            if args.objective == "program" else None)
    tr = HeightFieldSet(cache, tr_idx, augment=not args.no_aug, seed=args.seed, program=prog,
                        plane_head=args.plane_head)
    va = HeightFieldSet(cache, val_idx, augment=False, program=prog, plane_head=args.plane_head)
    print(f"[train] {len(tr)} buildings, {len(va)} validation, objective={args.objective}"
          + (f", plane_head={args.plane_head}" if args.objective == "program" else "")
          + f", device={dev}", flush=True)

    # 🔑 #132's logit adjustment, from the TRAINING split's labels only and computed once. It is a
    # property of the label's area canonicalisation, not of the model, so it never updates.
    # ⚠️ K_OPS, not `args.k_planes`: `make_model` ignores that flag for the program objective and
    # builds the assignment head with K_OPS + 1 channels. Passing the flag produced a 7-entry prior
    # against a 5-class head, which `test_the_prior_matches_the_models_assignment_head` now pins.
    a_prior = (assignment_prior(tr.program["assign"], tr.fp, K_OPS)
               if args.objective == "program" and tr.program is not None else None)
    if a_prior is not None:
        # the LAST class is "uncarved", not a slot -- naming it slot K would misread the majority
        names = [f"slot{k}" for k in range(K_OPS)] + ["uncarved"]
        print(f"[train] #132 assignment prior over {len(tr)} training buildings: "
              + "  ".join(f"{n} {v:.4f}" for n, v in zip(names, a_prior))
              + f"   (tau={ASSIGN_TEMPERATURE})", flush=True)

    # the TYPE head's own version: per-slot, because slots are canonicalised by AREA and the
    # label's own Ramp share falls from slot 0 to slot 3 (`type_collapse` measured 0.555/0.501/
    # 0.299/0.105 on the pinned 411; this is the training split's own number, never the pinned one).
    t_prior = (type_prior(tr.program["types"], K_OPS)
              if args.objective == "program" and tr.program is not None
              and not args.no_type_prior else None)
    if args.no_type_prior:
        print("[train] --no_type_prior: #138's type-head correction is OFF for this run", flush=True)
    if t_prior is not None:
        ramp = SLOT_TYPES.index("Ramp")
        print(f"[train] type prior (Ramp share) over {len(tr)} training buildings: "
              + "  ".join(f"slot{k} {v:.4f}" for k, v in enumerate(t_prior[:, ramp]))
              + f"   (tau={TYPE_TEMPERATURE})", flush=True)

    model = make_model(args.objective, args.width, args.k_planes, args.plane_head,
                       args.k_hyp).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(len(tr) // args.batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    print(f"[train] {n_par/1e6:.2f}M parameters, {steps} steps"
          + (f", k_hyp={args.k_hyp} (relaxed WTA, epsilon={args.wta_epsilon})"
             if args.k_hyp > 1 else ""), flush=True)

    def loss_of(x, y, ext, prog_labels=None):
        m = x[:, 0] > 0                                   # footprint columns only
        if args.objective == "program":
            # 🔑 the program arm never sees its own compiled surface during training. No `slope_
            # weight` either: the joint structure is in the output space now, and #127 measured
            # that adding it to the loss buys description length without buying planes.
            return program_loss(model(x), prog_labels, m, args.plane_head, a_prior, t_prior)
        if args.k_hyp > 1:
            return wta_ce_loss(model(x), y, m, args.k_hyp, args.wta_epsilon)
        out = (forward_heights(model, x, ext, args.objective) if args.objective == "planes"
               else model(x))
        per = per_column_loss(out, y, ext, args.objective, args.quantile)
        loss = (per * m).sum() / m.sum().clamp(min=1)
        if args.slope_weight:
            loss = loss + args.slope_weight * slope_loss(
                differentiable_depth(out, ext, args.objective, SLOPE_DECODE_QUANTILE), y, m)
        return loss

    def to_dev(b):
        x, y, e, p = b
        return (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
                torch.from_numpy(e).to(dev),
                tuple(torch.from_numpy(t).to(dev) for t in p) if p is not None else None)

    curve, best, best_path = [], (float("inf"), float("inf")), WORK / f"{args.tag}.pt"
    last_path = WORK / f"{args.tag}_last.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)
    val_carve = np.array([
        height_split(apply_depth(va.fp[i], int(va.extent[i]), envelope_depth(va.fp[i])),
                     va.target[i])["extra"] >= CARVE_NEEDED for i in range(len(va))])
    print(f"[train] selecting on validation missing+extra over "
          f"{int(val_carve.sum())}/{len(va)} carve-needing validation buildings", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        run = 0.0
        for s in range(0, len(order) - args.batch + 1, args.batch):
            x, y, e, p = to_dev(tr.batch(order[s:s + args.batch]))
            loss = loss_of(x, y, e, p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
        run /= max(len(order) // args.batch, 1)
        model.eval()
        vl, ve, vm = _validate(model, va, val_carve, args.objective, args.quantile, dev,
                               args.plane_head, a_prior, t_prior, args.k_hyp)
        curve.append(dict(epoch=ep + 1, train=run, val=vl, val_extra=ve, val_missing=vm,
                          val_symmetric=ve + vm))
        mark = ""
        snap = dict(state=model.state_dict(), objective=args.objective, width=args.width,
                    quantile=args.quantile, k_planes=args.k_planes, k_hyp=args.k_hyp,
                    plane_head=args.plane_head, slope_weight=args.slope_weight,
                    slope_decode_quantile=SLOPE_DECODE_QUANTILE,
                    # ⚠️ the decode travels WITH the weights. #129's checkpoints were trained and
                    # scored under ("median","median","argmax"); #132 changes the pitch and adds
                    # the loss-side prior, and a re-scored old checkpoint must not be able to
                    # present itself as the arm that produced #129's numbers.
                    plane_decode=list(PLANE_DECODE), assign_decode=ASSIGN_DECODE,
                    assign_prior=(None if a_prior is None else a_prior.tolist()),
                    assign_temperature=ASSIGN_TEMPERATURE,
                    type_prior=(None if t_prior is None else t_prior.tolist()),
                    type_temperature=TYPE_TEMPERATURE,
                    epoch=ep + 1, val=vl, val_extra=ve, val_missing=vm,
                    val_symmetric=ve + vm, params=n_par)
        if (ve + vm, vl) < best:
            best, mark = (ve + vm, vl), "  <- best"
            torch.save(snap, best_path)
        if ep + 1 == args.epochs:
            torch.save(dict(snap, selected_by="NOTHING -- the final epoch, a diagnostic only"),
                       last_path)
        print(f"  epoch {ep+1:>3}/{args.epochs}  train {run:.4f}  val {vl:.4f}  "
              f"val extra {ve:.4f}  val miss {vm:.4f}  sym {ve+vm:.4f}  "
              f"{time.time()-t0:.0f}s{mark}", flush=True)
    json.dump(curve, open(WORK / f"{args.tag}_curve.json", "w"), indent=1)
    print(f"[train] best validation missing+extra {best[0]:.4f} (loss {best[1]:.4f}) -> "
          f"{best_path}\n[train] final epoch, a diagnostic and NOT the arm -> {last_path}",
          flush=True)
    return best_path


def _validate(model, va, carve_mask, objective: str, quantile: float, dev,
              plane_head: str = "regress", assign_prior=None, type_prior=None,
              k_hyp: int = 1) -> tuple:
    """Validation loss AND the geometric quantity the ticket is judged on, on held-in buildings.

    ⚠️ `k_hyp > 1` (#8) validates at the ORACLE decode (`decode_wta` given the real `va.target`) --
    legitimate here because checkpoint selection is a training-time decision allowed to know the
    answer, same footing as every other selection rule in this function. It is a ceiling on what a
    real, servable selector could ever reach, not a preview of one; nothing about `_validate`
    picking the oracle hypothesis makes the SERVED arm able to do the same.
    """
    import torch

    # ⚠️ The CE arm is validated at its ARGMAX, which is what it is trained for. Validating it at
    # the median would select a checkpoint for a decode the training never committed to, and the
    # post-hoc median has to stay an ablation of a finished model rather than a training signal.
    decode_q = None if objective == "ce" else quantile
    losses, splits = [], []
    with torch.no_grad():
        for s in range(0, len(va), 128):
            sel = np.arange(s, min(s + 128, len(va)))
            x, y, e, p = va.batch(sel)
            xt, yt = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
            et = torch.from_numpy(e).to(dev)
            m = xt[:, 0] > 0
            if objective == "program":
                out = model(xt)
                losses.append(float(program_loss(
                    out, tuple(torch.from_numpy(t).to(dev) for t in p), m, plane_head,
                    assign_prior, type_prior).detach()))
                o = [tuple(t[k].cpu().numpy() for t in out) for k in range(len(sel))]
            elif k_hyp > 1:
                out = model(xt)
                losses.append(float(wta_ce_loss(out, yt, m, k_hyp).detach()))
                o = out.cpu().numpy()
            else:
                out = (forward_heights(model, xt, et, objective) if objective == "planes"
                       else model(xt))
                per = per_column_loss(out, yt, et, objective, quantile)
                losses.append(float(((per * m).sum() / m.sum().clamp(min=1)).detach()))
                o = out.cpu().numpy()
            for k, i in enumerate(sel):
                ext, fp = int(va.extent[i]), va.fp[i]
                decoded = (decode_wta(o[k], fp, ext, k_hyp, decode_q, va.target[i]) if k_hyp > 1
                          else decode_prediction(o[k], fp, ext, objective, decode_q, plane_head))
                splits.append(height_split(decoded, va.target[i]))
    carve = [d for d, m in zip(splits, carve_mask) if m]
    return (float(np.mean(losses)),
            float(np.median([d["extra"] for d in carve])) if carve else float("nan"),
            float(np.median([d["missing"] for d in carve])) if carve else float("nan"))


def predict(ckpt: Path, held: dict, batch: int = 64, cpu: bool = False,
            quantile: float | None = None):
    """Height maps for the pinned buildings from a trained checkpoint, and how it was selected.

    The provenance travels with the prediction rather than with the command line: a `--ckpt` rerun
    scores a file trained by some earlier invocation, and recording the flags of the *rerun* would
    put a number in the artifact that did not produce the checkpoint beside it.

    ⚠️ A `k_hyp > 1` checkpoint (#8) is decoded here at its ORACLE hypothesis (`decode_wta` given
    `held["target"]`) -- this is stage 1's own gate ("is a good roof even IN the k_hyp candidates"),
    not a servable arm, and the returned meta says `oracle=True` so callers keep it out of any
    generator-vs-generator comparison, the same way `program_label (sees GT)` already is kept out.
    """
    import torch

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    # ⚠️ default "regress": #6's committed checkpoints predate the flag and must still load
    head = d.get("plane_head", "regress")
    k_hyp = d.get("k_hyp", 1)
    model = make_model(d["objective"], d["width"], d.get("k_planes", 6), head, k_hyp).to(dev)
    model.load_state_dict(d["state"])
    model.eval()
    out = np.zeros((len(held["fp"]), RES, RES), np.int16)
    with torch.no_grad():
        for s in range(0, len(out), batch):
            sel = range(s, min(s + batch, len(out)))
            x = np.stack([condition_channels(held["fp"][i], int(held["extent"][i]),
                                             float(held["height_m"][i]), int(held["region"][i]))
                          for i in sel])
            xt = torch.from_numpy(x).to(dev)
            if d["objective"] == "planes":
                et = torch.tensor([float(held["extent"][i]) for i in sel], device=dev)
                y = forward_heights(model, xt, et, "planes").cpu().numpy()
            elif d["objective"] == "program":
                heads = model(xt)
                y = [tuple(t[k].cpu().numpy() for t in heads) for k in range(len(list(sel)))]
            else:
                y = model(xt).cpu().numpy()
            for k, i in enumerate(sel):
                if k_hyp > 1:
                    out[i] = decode_wta(y[k], held["fp"][i], int(held["extent"][i]), k_hyp,
                                        quantile, target=held["target"][i])
                else:
                    out[i] = decode_prediction(y[k], held["fp"][i], int(held["extent"][i]),
                                               d["objective"], quantile, head)
    # the whole training curve travels into the artifact, not a summary of it: this project has
    # twice recommended stopping at a dip that recovered (#80), and a curve nobody can re-read is
    # how that happens a third time.
    curve = ckpt.with_name(ckpt.stem + "_curve.json")
    return out, dict(path=str(ckpt), objective=d["objective"], width=d["width"],
                     k_hyp=k_hyp, oracle=bool(k_hyp > 1),
                     decode=(f"ORACLE best-of-{k_hyp} (sees GT -- stage-1 gate, not a servable arm)"
                             if k_hyp > 1 else
                             "argmax" if d["objective"] == "ce" and quantile is None else
                             f"posterior q={quantile}" if d["objective"] == "ce" else
                             # ⚠️ read from PLANE_DECODE / ASSIGN_DECODE, never spelled out: this
                             # string said "median pitch" for two committed #132 artifacts after
                             # the pre-registered read had become q0.25, which defeats the very
                             # guard the same commit added by storing the decode with the weights
                             f"compiled program ({ASSIGN_DECODE} slot, argmax type, "
                             + (" ".join(f"{r} {q}" for q, r in
                                         zip(PLANE_QUANTITIES, PLANE_DECODE)) + ")"
                                if head == "class" else "regressed plane)")
                             if d["objective"] == "program" else
                             f"regression (trained at q={d.get('quantile')})"
                             if d["objective"] == "quantile" else "regression"),
                     plane_head=head if d["objective"] == "program" else None,
                     trained_quantile=d.get("quantile"),
                     slope_weight=d.get("slope_weight", 0.0),
                     slope_decode_quantile=d.get("slope_decode_quantile"),
                     epoch=d.get("epoch"), val_loss=d.get("val"), val_extra=d.get("val_extra"),
                     val_missing=d.get("val_missing"), val_symmetric=d.get("val_symmetric"),
                     params=d.get("params"), selected_on="validation missing+extra",
                     curve=json.load(open(curve)) if curve.exists() else None)


# ==================================================================================================
# scoring
# ==================================================================================================

def score_arm(heights: np.ndarray, held: dict, form: bool = True) -> list:
    """One row of metrics per pinned building, in the order #126 decided they must be read."""
    rows = []
    for i in range(len(heights)):
        fp, y0, extent = held["fp"][i], int(held["y0"][i]), int(held["extent"][i])
        gt = occupancy(fp, y0, held["target"][i])
        bo = occupancy(fp, y0, apply_depth(fp, extent, envelope_depth(fp)))
        occ = occupancy(fp, y0, heights[i])
        r = dict(id=int(held["row"][i]),
                 # #127's actual question in plan view: of the footprint columns, what fraction did
                 # the arm cut at all, against the fraction GT cuts? `extra` says how much surplus
                 # is left; this says whether the arm ACTED, and on how much of the building.
                 carved_cols=float((heights[i][fp] < extent).mean()),
                 gt_carved_cols=float((held["target"][i][fp] < extent).mean()),
                 **{f"roof_{k}": v for k, v in roof_shape_stats(heights[i], fp).items()},
                 **{f"gt_roof_{k}": v for k, v in
                    roof_shape_stats(held["target"][i], fp).items()},
                 **({f"dl_{k}": v for k, v in roof_description_length(
                        heights[i], fp, y0, extent).items()} if form else {}),
                 **({f"gt_dl_{k}": v for k, v in roof_description_length(
                        held["target"][i], fp, y0, extent).items()} if form else {}))
        r.update(volume_split(occ, gt))
        r.update(footprint_split(occ, fp))
        r["fp_iou"] = fp_iou(occ, fp)
        r["vs_input"] = vs_input(occ, bo)
        r["blockout_extra"] = volume_split(bo, gt)["extra"]
        rows.append(r)
    return rows


def summarise(rows: list) -> dict:
    med = lambda k: float(np.median([r[k] for r in rows])) if rows else float("nan")
    return dict(n=len(rows), missing=med("missing"), extra=med("extra"),
                vs_input=med("vs_input"), carved_cols=med("carved_cols"),
                gt_carved_cols=med("gt_carved_cols"),
                **{k: med(k) for k in ("roof_relief", "roof_curvature", "roof_speckle",
                                       "gt_roof_relief", "gt_roof_curvature", "gt_roof_speckle")},
                **({k: med(k) for k in ("dl_ops", "dl_planar_fraction", "dl_residual",
                                        "gt_dl_ops", "gt_dl_planar_fraction")}
                   if rows and "dl_ops" in rows[0] else {}),
                **(dict(dl_explained=float(np.mean([r["dl_explained"] for r in rows])))
                   if rows and "dl_ops" in rows[0] else {}),
                collapse_rate=float(np.mean([r["missing"] >= COLLAPSE_MISSING for r in rows]))
                if rows else float("nan"),
                fp_iou=med("fp_iou"), spill=med("spill"), vol_iou=med("vol_iou"))


# ⚠️ Load-bearing, not a label. `NOT_GENERATORS` keeps the ceiling out of the verdict by NAME, so
# renaming the arm at its construction site and not here would silently hand it a mechanical PASS.
PROGRAM_LABEL_ARM = "program_label (sees GT)"
NOT_GENERATORS = ("blockout", "nn_retrieval", PROGRAM_LABEL_ARM)

# #6's bar, pre-registered in the module docstring in `fccef61` before the first training step and
# fixed here so it is evaluated mechanically rather than in prose. `max_extra` and `kill_planar` are
# #127's served CE+median arm's own numbers on these same 411 rows, quoted rather than chosen.
PROGRAM_BAR = dict(max_ops=3.0, min_planar=0.40, max_extra=0.0603, kill_planar=0.20)


def verdict(arms: dict, pop: str) -> dict:
    """The pre-registered bar, evaluated mechanically so the write-up cannot soften it.

    ⚠️ `NOT_GENERATORS` are excluded. Two of them are the bar itself; the third is #6's compiled
    label, which is the fitter's program built WITH GT IN HAND. It beats anything a generator can
    reach and would collect a mechanical PASS -- a scorecard reporting that the target was hit by
    looking at it. It is a ceiling the trained arms are read against, never a competitor.

    🔑 #6's FORM clauses are added whenever the form metric was measured, and they are the reason
    this function exists rather than a paragraph. Both halves must clear at once: #127's plane head
    scored 3.0 ops at `planar_fraction` **0.00** and #6's own arm scored 1.0 at 0.00, so a
    single-number form metric calls a terrace an improvement twice over. An arm scored with
    `--no_form` gets no form verdict at all -- a missing measurement must read as absent, never as
    a pass.
    """
    out = {}
    bo, nn = arms["blockout"][pop], arms["nn_retrieval"][pop]
    for name, a in arms.items():
        # ⚠️ `_oracle` (#8) is the same shape of ceiling as `PROGRAM_LABEL_ARM`, by suffix rather
        # than a fixed name: it is `predict()`'s k_hyp>1 decode, which is given the real answer to
        # pick a hypothesis and so would collect a mechanical PASS for the same reason the compiled
        # label would.
        if name in NOT_GENERATORS or "_oracle" in name:
            continue
        s = a[pop]
        out[name] = dict(
            beats_1nn_extra=bool(s["extra"] < nn["extra"]),
            collapse_no_worse_than_1nn=bool(s["collapse_rate"] <= nn["collapse_rate"]),
            moved=bool(s["vs_input"] < 0.98),
            killed_identity=bool(s["extra"] >= bo["extra"]),
        )
        out[name]["pass"] = bool(out[name]["beats_1nn_extra"] and
                                 out[name]["collapse_no_worse_than_1nn"] and out[name]["moved"])
        if "dl_ops" in s:
            out[name].update(
                form_ops_under_bar=bool(s["dl_ops"] <= PROGRAM_BAR["max_ops"]),
                form_planar_over_bar=bool(s["dl_planar_fraction"] >= PROGRAM_BAR["min_planar"]),
                beats_served_extra=bool(s["extra"] < PROGRAM_BAR["max_extra"]),
                # `<=`, not `<`: matching the arm you replaced is not an improvement over it
                killed_flat=bool(s["dl_planar_fraction"] <= PROGRAM_BAR["kill_planar"]),
            )
            # ⚠️ the GUARDS are ANDed in, which the written bar always said and this composite
            # did not do. Corrected after #6's run, and it is safe to correct after the fact only
            # because it can make a PASS into a FAIL and never the reverse: it changes no arm's
            # verdict on #6's or #129's record (checked), and it stops an arm reporting
            # `form_planar_over_bar: true` with nothing machine-checked saying the collapse guard
            # sank it -- which is exactly what #129's endpoint did.
            out[name]["program_pass"] = bool(
                out[name]["form_ops_under_bar"] and out[name]["form_planar_over_bar"] and
                out[name]["beats_served_extra"] and out[name]["pass"])
    return out


REFERENCE = {
    # arm -> (committed artifact, key under `per_building`). Quoted, never recomputed: these are
    # this project's record on the SAME pinned 714, and re-deriving them here would risk a second,
    # silently different number for an arm that already has one.
    "a2_s0.5 (shipped)": ("execution/artifacts/massing_arms_eval_ship714.json", "a2_s0.5"),
    "deployed_map24": ("execution/artifacts/massing_arms_eval_ship714.json", "deployed_map24"),
    "codec_ceiling": ("execution/artifacts/massing_arms_eval_ship714.json", "codec_ceiling"),
    "program K=16 (sees GT)": ("execution/artifacts/program_recovery_714.json", None),
}


def reference_arms(carve_ids: set) -> dict:
    """This project's arms of record, re-summarised on exactly the rows scored here.

    #126's rule, applied to the write-up as well as to the run: an arm quoted from one population
    beside an arm measured on another is how "19% surplus reduction" became 11.8% like-for-like on
    map #87. The medians are recomputed from the committed per-building rows, so the population is
    the same 411 buildings whatever those artifacts summarised themselves over.
    """
    out = {}
    for name, (path, key) in REFERENCE.items():
        f = REPO / path
        if not f.exists():
            continue
        doc = json.load(open(f))
        pb = doc["per_building"]
        pb = pb[key] if key else pb
        rows = [r for b, r in pb.items() if int(b) in carve_ids]
        if not rows:
            continue
        med = lambda k: float(np.median([r[k] for r in rows if k in r]))
        out[name] = dict(n=len(rows), missing=med("missing"), extra=med("extra"),
                         vs_input=med("vs_input") if "vs_input" in rows[0] else None,
                         collapse_rate=float(np.mean([r["missing"] >= COLLAPSE_MISSING
                                                      for r in rows])),
                         vol_iou=med("vol_iou"), source=path)
    return out


# ==================================================================================================
# #6's diagnostics. The arm is one compile of three predictions, and `extra` cannot say which of
# them is wrong -- so each of these replaces or perturbs exactly one thing and re-scores.
# ==================================================================================================

def program_predictions(ckpt: Path, held: dict, cpu: bool = False, raw: bool = False):
    """(assignment, types, plane params) per pinned building, as the compiler would receive them.

    🔑 The planes come back in #6's normalised `(A, Bz, Cx)` whichever head produced them -- a
    classified head is decoded here, once, by the served decode. Every diagnostic downstream
    therefore reads one convention and #6's tables stay directly comparable to #129's.

    `raw=True` returns the classified head's logits instead, which is what `decode_ablation` needs
    to re-read the same weights at a different statistic without a second forward pass.
    """
    al, tl, pr, head = _program_forward(ckpt, held, cpu)
    a = np.stack([decode_assignment(al[i], held["fp"][i]) for i in range(len(al))])
    p = pr
    if head == "class" and not raw:
        p = np.stack([decode_plane_logits(pr[k], slot_centroids(a[k], pr.shape[1]))
                      for k in range(len(a))])
    return a, tl.argmax(-1).astype(np.int8), p


def _program_forward(ckpt: Path, held: dict, cpu: bool = False):
    """One forward pass over the pinned set, returning every head RAW: `(assign, type, plane, head)`.

    🔑 Extracted so the served decode (`program_predictions`) and the diagnostics that have to look
    at a posterior *before* it is collapsed (`assignment_collapse`, `decode_ablation`) read the same
    weights through the same path. A diagnostic with its own private forward pass is how a decode
    quietly stops being the one that is served.
    """
    import torch

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    if d["objective"] != "program":
        raise ValueError(f"{ckpt} is a '{d['objective']}' arm; the program diagnostics need one")
    head = d.get("plane_head", "regress")
    dev = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model = make_model("program", d["width"], d.get("k_planes", 6), head).to(dev)
    model.load_state_dict(d["state"])
    model.eval()
    A, T, P = [], [], []
    with torch.no_grad():
        for s in range(0, len(held["fp"]), 64):
            sel = range(s, min(s + 64, len(held["fp"])))
            x = np.stack([condition_channels(held["fp"][i], int(held["extent"][i]),
                                             float(held["height_m"][i]), int(held["region"][i]))
                          for i in sel])
            al, tl, pr = model(torch.from_numpy(x).to(dev))
            A.append(al.cpu().numpy())
            T.append(tl.cpu().numpy())
            P.append(pr.cpu().numpy())
    return np.concatenate(A), np.concatenate(T), np.concatenate(P), head


def _median_split(heights, held, rows):
    """The medians every diagnostic table reports, in #126's order.

    🔑 `vs_input` and the collapse rate are here rather than only on the scorecard because #126's
    rule is about *every* number, and the diagnostics are where it bites hardest: an ablation row
    can buy `extra` by declining to act (#75) or by eating the building, and neither shows in
    `extra`/`missing` medians alone. The `circmean` rows are exactly that case.

    ⚠️ `vs_input` needs occupancy against the blockout the arm started from, so this is ~30x the
    cost of the two medians on their own. The tables it feeds run once per checkpoint, off a
    finished model, so that is affordable where it would not be inside the epoch loop
    (`height_split`, which is what `_validate` uses, exists precisely for that).
    """
    sp = [height_split(heights[k], held["target"][i]) for k, i in enumerate(rows)]
    vsi = []
    for k, i in enumerate(rows):
        fp, y0, e = held["fp"][i], int(held["y0"][i]), int(held["extent"][i])
        vsi.append(vs_input(occupancy(fp, y0, heights[k]),
                            occupancy(fp, y0, apply_depth(fp, e, envelope_depth(fp)))))
    return dict(extra=float(np.median([s["extra"] for s in sp])),
                missing=float(np.median([s["missing"] for s in sp])),
                vs_input=float(np.median(vsi)),
                collapse_rate=float(np.mean([s["missing"] >= COLLAPSE_MISSING for s in sp])))


def head_ablation(pred, label, held, rows) -> dict:
    """🔑 Replace ONE predicted head with its label at a time, and re-score.

    The arm compiles an assignment, a set of types and a set of planes together, so a single
    `extra` cannot say which of the three is costing what -- and guessing is how #127 spent two
    runs on an initialisation that turned out not to be the cause. This is the measurement that
    replaces the guess, and on #6's first arm it is decisive: the types are free, the regions cost
    0.012, and the planes cost 0.052 of the 0.052 that is there to be had.
    """
    pa, pt, pp = pred
    la, lt, lp = label
    vox = lambda i, arr: np.stack([plane_to_voxel(arr[i][k], int(held["extent"][i]))
                                   for k in range(len(arr[i]))])
    variants = {
        "all three predicted": lambda i: (pa[i], pt[i], vox(i, pp)),
        "label types": lambda i: (pa[i], lt[i], vox(i, pp)),
        "label assignment": lambda i: (la[i], pt[i], vox(i, pp)),
        "label planes": lambda i: (pa[i], pt[i], lp[i]),
        "all three from the label (the ceiling)": lambda i: (la[i], lt[i], lp[i]),
    }
    out = {}
    for name, get in variants.items():
        hs = [compile_program(*get(i), held["fp"][i], int(held["extent"][i])) for i in rows]
        out[name] = _median_split(hs, held, rows)
    return out


def flatten_ramps(label, held, rows, y0=None) -> dict:
    """Take the PERFECT program and flatten only its `Ramp`s. The form control.

    ⚠️ This is the one that turns "the arm's slopes came out flat" from a description into a cause.
    If flattening a program that is otherwise exactly right reproduces the trained arm's form
    signature, then the regions and the types are not what failed.
    """
    la, lt, lp = label
    out = {}
    for name, flat in (("compiled as fitted", False), ("every Ramp flattened", True)):
        hs, fm = [], []
        for i in rows:
            e = int(held["extent"][i])
            p = lp[i].copy()
            if flat:
                for k in range(len(p)):
                    if lt[i][k] == SLOT_TYPES.index("Ramp"):
                        n = plane_to_normalised(lp[i][k], e)
                        n[1] = n[2] = 0.0
                        p[k] = plane_to_voxel(n, e)
            h = compile_program(la[i], lt[i], p, held["fp"][i], e)
            hs.append(h)
            fm.append(roof_description_length(h, held["fp"][i], int(held["y0"][i]), e))
        out[name] = dict(**_median_split(hs, held, rows),
                         ops=float(np.median([f["ops"] for f in fm])),
                         planar_fraction=float(np.median([f["planar_fraction"] for f in fm])))
    return out


def slope_symmetry(program: dict, extents: np.ndarray, ok: np.ndarray) -> dict:
    """🔑🔑 Why an L1 on a slope must return flat, and it is a property of the corpus.

    An L1 returns the conditional median. If the signed slope of a roof is symmetric about zero --
    a roof may pitch either way, and #126 measured that footprint plus height does not determine
    which -- then the median IS zero, and the objective's own Bayes act is a flat roof however long
    it trains. A quantile loss does not escape it either, because the mean and the median coincide.

    This is #127's classification-over-regression argument, arriving in the plane parameters.
    """
    ramp = SLOT_TYPES.index("Ramp")
    v = []
    for i in np.nonzero(ok)[0]:
        e = int(extents[i])
        for k in range(program["types"].shape[1]):
            if program["types"][i, k] == ramp:
                n = plane_to_normalised(program["planes"][i, k], e)
                v.extend([float(n[1]), float(n[2])])
    v = np.asarray(v)
    if not len(v):
        return {}
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                positive=float((v > 0).mean()), negative=float((v < 0).mean()),
                median_abs=float(np.median(np.abs(v))))


def canonicalisation_cost(pred, label, held, rows) -> dict:
    """What an equivalence-aware objective would have bought, which #6 asks about directly.

    Slots are canonicalised by owned area because a set head has no natural order. That is lossy
    exactly when two regions have similar area: the model orders them one way and the label the
    other, and slot 2's parameters are then scored against slot 3's label. The alternative is a
    matching loss, and this measures its ceiling -- the error under the BEST permutation.
    """
    import itertools

    _, _, pp = pred
    _, lt, lp = label
    fixed, matched = [], []
    for i in rows:
        act = np.nonzero(lt[i] >= 0)[0]
        if not len(act):
            continue
        lab = np.stack([plane_to_normalised(lp[i][k], int(held["extent"][i])) for k in act])
        pr = pp[i][:len(act)]
        fixed.append(float(np.abs(pr - lab).mean()))
        matched.append(min(float(np.abs(pr[list(q)] - lab).mean())
                           for q in itertools.permutations(range(len(act)))))
    fixed, matched = np.asarray(fixed), np.asarray(matched)
    gap = float((fixed - matched).mean())
    return dict(n=len(fixed), canonical_order=float(fixed.mean()),
                best_permutation=float(matched.mean()), cost=gap,
                cost_share=float(gap / fixed.mean()) if fixed.mean() else 0.0)


def slot_usage(pred, label, held, rows) -> dict:
    """How many of its K slots the arm actually uses, and whether they compile flat."""
    pa, pt, pp = pred
    la, lt, _ = label
    k_ops = label[2].shape[1]
    used_p, used_l, ramp_typed, rise, ramp_rise = [], [], [], [], []
    for i in rows:
        m, e = held["fp"][i], int(held["extent"][i])
        up = set(np.unique(pa[i][m])) - {k_ops}
        used_p.append(len(up))
        used_l.append(len(set(np.unique(la[i][m])) - {k_ops}))
        ramp_typed += [int(pt[i][k] == SLOT_TYPES.index("Ramp")) for k in up]
        h = compile_program(pa[i], pt[i],
                            np.stack([plane_to_voxel(pp[i][k], e) for k in range(k_ops)]), m, e)
        for k in up:
            r = (pa[i] == k) & m
            if int(r.sum()) > 20:
                rise.append(float(h[r].max() - h[r].min()))
                if int(pt[i][k]) == SLOT_TYPES.index("Ramp"):
                    ramp_rise.append(rise[-1])
    rise = np.asarray(rise)
    ramp_rise = np.asarray(ramp_rise)
    # ⚠️ TWO keys, because one name for both measurements has now misled a reader twice: #129
    # renamed `decode_ablation`'s copy after the first time, and in #132 the all-slots figure read
    # 0.00 while the `Ramp`-typed one was 12.00 and I called the pitch gone. A `Layer` is flat by
    # definition, so the all-slots number falls as soon as an arm uses more slots -- it says
    # something about the TYPE mix, not about whether a pitch was drawn.
    med = lambda a: float(np.median(a)) if len(a) else 0.0
    return dict(slots_used_by_arm=float(np.mean(used_p)),
                slots_used_by_label=float(np.mean(used_l)),
                arm_uses_exactly_one=float(np.mean(np.asarray(used_p) == 1)),
                used_slots_typed_ramp=float(np.mean(ramp_typed)) if ramp_typed else 0.0,
                realised_rise_all_slots_voxels=med(rise),
                realised_rise_ramp_typed_voxels=med(ramp_rise),
                n_ramp_typed_slots=int(len(ramp_rise)),
                # ⚠️ over every used slot, so a `Layer` counts as flat BY DEFINITION. Read it with
                # `used_slots_typed_ramp`, never as evidence on its own that planes were lost.
                used_slots_compiling_flat=float((rise < 1).mean()) if len(rise) else 0.0)


def assignment_stats(logits, label_assign, fp, k_ops: int) -> dict:
    """🔑 #132's free question, asked before the head is blamed: **diffuse, or confidently wrong?**

    Both trained arms use one slot where the label uses 3.06 (#6 1.19, #129 0.90), and `dl_ops`
    reads 1.0 because of that and not because their planes are flat. Two very different faults
    produce that one number, and they want opposite fixes:

      * a **diffuse** posterior that knows about a second region and loses the per-column `argmax`
        to slot 0 -- a DECODE problem, and this map has twice found the decode was the answer
        (#127 argmax -> posterior median, `extra` 0.1178 -> 0.0603; #129 azimuth argmax over
        circmean);
      * a **confident** posterior that has never heard of the second region -- a loss or curriculum
        problem, which is [#130]'s third item.

    `p_true_minor` is what separates them: the mass the head puts on the CORRECT slot, on exactly
    the columns whose label is not that building's dominant slot. Near `p_won_minor` means the head
    knows and narrowly loses; near zero means it does not know.

    🔑 The **balanced** read is the mechanism-matched candidate, not a fished one. Slots are
    canonicalised by AREA (#6), so slot 0 owns most columns of most buildings and a per-column
    cross-entropy is imbalanced by construction. Dividing the posterior by the model's OWN marginal
    -- computed from the prediction, using no label -- is the standard correction for that, and it
    is reported here as a diagnosis. ⚠️ Adopting it would need pre-registering before the run that
    benefits from it, with the imbalance as the reason and not this table.
    """
    p = np.asarray(logits, np.float64)
    p = np.exp(p - p.max(axis=0, keepdims=True))
    p /= p.sum(axis=0, keepdims=True)
    m = np.asarray(fp, bool)
    lab = np.asarray(label_assign)
    if not m.any():
        return {}
    inside = p[:, m]                                             # (K+1, n_columns)
    won = inside.max(axis=0)
    lab_in = lab[m]
    # ⚠️ both reads go through `decode_assignment`, never a local argmax. A private copy here is
    # exactly what that function's docstring forbids, and the copy this replaces had silently
    # dropped its `** tau`, so raising ASSIGN_TEMPERATURE would have left this table describing a
    # read the model is not served by.
    arg = decode_assignment(logits, m, "argmax")[m]
    bal = decode_assignment(logits, m, "balanced")[m]

    acc = lambda v, sel: float((v[sel] == lab_in[sel]).mean()) if sel.any() else 0.0
    slots = lambda v: int(len(set(np.unique(v).tolist()) - {k_ops}))
    used = [k for k in np.unique(lab_in) if k != k_ops]
    dominant = max(used, key=lambda k: int((lab_in == k).sum())) if len(used) else k_ops
    minor = (lab_in != dominant) & (lab_in != k_ops)
    n = int(minor.sum())
    take = lambda v: float(v.mean()) if n else 0.0
    every = np.ones(len(lab_in), bool)
    dom = (lab_in == dominant)
    return dict(
        accuracy_argmax=acc(arg, every), accuracy_balanced=acc(bal, every),
        # ⚠️ the correction is only worth having if it buys minor slots WITHOUT losing the dominant
        # one: a read that fragments the largest region has traded surplus for slot count
        dominant_argmax=acc(arg, dom), dominant_balanced=acc(bal, dom),
        confidence=float(np.median(won)),
        entropy_norm=float(np.mean(-(inside * np.log(np.clip(inside, 1e-12, None))).sum(axis=0))
                           / np.log(len(inside))),
        slots_argmax=slots(arg), slots_label=slots(lab_in), slots_balanced=slots(bal),
        n_minor_columns=n,
        p_true_minor=take(inside[lab_in[minor], np.flatnonzero(minor)]) if n else 0.0,
        p_won_minor=take(won[minor]),
        recall_minor=take(arg[minor] == lab_in[minor]),
        recall_minor_balanced=take(bal[minor] == lab_in[minor]),
    )


def assignment_collapse(ckpt: Path, label, held, rows, cpu: bool = False) -> dict:
    """`assignment_stats` over the pinned rows: the K = 1 collapse, diagnosed rather than named.

    Reported beside `slot_usage`, which counts the slots the argmax ends up using. This says WHY
    that count is 1, and the two rows are meant to be read together.
    """
    al, _, _, _ = _program_forward(ckpt, held, cpu)
    la = label[0]
    k_ops = al.shape[1] - 1
    per = [assignment_stats(al[i], la[i], held["fp"][i], k_ops) for i in rows]
    per = [s for s in per if s]
    if not per:
        return {}
    med = lambda k: float(np.median([s[k] for s in per]))
    avg = lambda k: float(np.mean([s[k] for s in per]))
    wt = lambda k: float(np.average([s[k] for s in per],
                                    weights=[max(s["n_minor_columns"], 0) for s in per])) \
        if sum(s["n_minor_columns"] for s in per) else 0.0
    return dict(n=len(per), confidence_median=med("confidence"),
                accuracy_argmax=avg("accuracy_argmax"), accuracy_balanced=avg("accuracy_balanced"),
                dominant_argmax=avg("dominant_argmax"), dominant_balanced=avg("dominant_balanced"),
                entropy_norm_mean=avg("entropy_norm"),
                slots_argmax_mean=avg("slots_argmax"), slots_label_mean=avg("slots_label"),
                slots_balanced_mean=avg("slots_balanced"),
                minor_columns_total=int(sum(s["n_minor_columns"] for s in per)),
                buildings_with_a_minor_slot=float(np.mean([s["n_minor_columns"] > 0 for s in per])),
                p_true_minor=wt("p_true_minor"), p_won_minor=wt("p_won_minor"),
                recall_minor=wt("recall_minor"),
                recall_minor_balanced=wt("recall_minor_balanced"))


def type_prior(types, k_ops: int) -> np.ndarray:
    """Each slot's label Layer/Ramp split, conditioned on the slot being ACTIVE: `(k_ops, 2)`.

    🔑 The imbalance this asks about is NOT #131's corpus-wide 41%-Ramp/59%-Layer split -- it is a
    steep per-SLOT-INDEX gradient, because slots are canonicalised by AREA (#6): a building's
    biggest region is a pitch more often than not, and its smallest is almost always a flat
    setback. Measured over the 34,909 training rows: slot 0 is Ramp 59.4% of the time, slot 1
    52.3%, slot 2 32.3%, slot 3 13.4%. A single scalar prior would average that gradient away
    exactly where #132's assignment fix landed its second region -- slot 1, whose own label is
    still Ramp roughly half the time.

    ⚠️ Per BUILDING, not per column -- a slot is typed once, not once per pixel it owns -- and on
    the TRAINING split only, the same leakage rule `assignment_prior` follows.
    """
    t = np.asarray(types)
    n_type = len(SLOT_TYPES)
    out = np.full((k_ops, n_type), 1.0 / n_type, np.float64)
    for k in range(k_ops):
        active = t[:, k] >= 0
        if active.any():
            counts = np.bincount(t[active, k].astype(np.int64), minlength=n_type)[:n_type]
            out[k] = counts / max(counts.sum(), 1)
    return out


def type_stats(logits, label_types, prior=None) -> dict:
    """One building's slot TYPES against their labels: the type-head analogue of `assignment_stats`.

    #132 named `used_slots_typed_ramp` 0.390 the binding constraint on its own KILL without asking
    whether that is the head being diffuse, confidently wrong, or simply correct about a label that
    is itself steeply slot-index-conditional (`type_prior`). This asks `assignment_stats`'s "diffuse
    or wrong" question of the type head, with the one asymmetry the compiler gives it:
    `compile_program` reads a slot's plane only when its type says `Ramp`, so mistyping a real
    `Ramp` as `Layer` costs a roof and the reverse costs nothing the compiler can see.

    `prior`, if given, is `type_prior`'s `(k_ops, 2)` split: the BALANCED read divides each slot's
    posterior by its own prior before the argmax, mirroring `decode_assignment`'s correction --
    reported here as a diagnosis, not served, exactly as `assignment_stats` reported `balanced`
    before #132 decided where its own fix belonged.

    Every returned array is `(k_ops,)` and positional, so many buildings stack into one
    `(n, k_ops)` matrix without re-deriving which slot is which.
    """
    lg = np.asarray(logits, np.float64)
    p = np.exp(lg - lg.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    lab = np.asarray(label_types)
    active = lab >= 0
    ramp = SLOT_TYPES.index("Ramp")
    arg = p.argmax(axis=-1)
    bal = (p / np.clip(prior, 1e-12, None)).argmax(axis=-1) if prior is not None else arg
    return dict(
        active=active, is_ramp=(lab == ramp) & active,
        p_ramp=p[..., ramp], confidence=p.max(axis=-1),
        entropy_norm=-(p * np.log(np.clip(p, 1e-12, None))).sum(-1) / np.log(p.shape[-1]),
        correct_argmax=(arg == lab) & active, correct_balanced=(bal == lab) & active,
        pred_ramp_argmax=(arg == ramp) & active, pred_ramp_balanced=(bal == ramp) & active,
    )


def type_collapse(ckpt: Path, label, held, rows, cpu: bool = False) -> dict:
    """`type_stats` over the pinned rows, aggregated overall and by slot INDEX.

    The per-slot breakdown is the point: #132's aggregate `used_slots_typed_ramp` 0.390 sits
    between slot 1's label rate (52%) and slot 3's (13%), which one number cannot tell apart from
    "the head guesses the corpus average everywhere" and "the head tracks the per-slot rate closely
    and slot 3 just IS mostly flat". This is that question, per slot.
    """
    _, tl, _, _ = _program_forward(ckpt, held, cpu)
    _, lt, _ = label
    k_ops = tl.shape[1]
    prior = type_prior(lt, k_ops)
    keys = ("active", "is_ramp", "p_ramp", "confidence", "entropy_norm",
           "correct_argmax", "correct_balanced", "pred_ramp_argmax", "pred_ramp_balanced")
    stacked = {key: np.stack([type_stats(tl[i], lt[i], prior)[key] for i in rows]) for key in keys}
    active, is_ramp = stacked["active"], stacked["is_ramp"]
    is_layer = active & ~is_ramp

    def rate(hit, sel):
        return float(hit[sel].mean()) if sel.any() else None

    per_slot = []
    for k in range(k_ops):
        a, r, l = active[:, k], is_ramp[:, k], is_layer[:, k]
        per_slot.append(dict(
            n=int(a.sum()), label_ramp_share=float(r.sum() / max(int(a.sum()), 1)),
            confidence=float(np.median(stacked["confidence"][a, k])) if a.any() else None,
            entropy_norm=float(np.mean(stacked["entropy_norm"][a, k])) if a.any() else None,
            recall_ramp_argmax=rate(stacked["pred_ramp_argmax"][:, k], r),
            recall_ramp_balanced=rate(stacked["pred_ramp_balanced"][:, k], r),
            recall_layer_argmax=rate(~stacked["pred_ramp_argmax"][:, k], l),
            recall_layer_balanced=rate(~stacked["pred_ramp_balanced"][:, k], l),
            p_ramp_given_ramp=float(stacked["p_ramp"][r, k].mean()) if r.any() else None,
            p_ramp_given_layer=float(stacked["p_ramp"][l, k].mean()) if l.any() else None,
        ))
    return dict(
        n=len(rows), prior_ramp_share=[float(x) for x in prior[:, SLOT_TYPES.index("Ramp")]],
        confidence_median=float(np.median(stacked["confidence"][active])) if active.any() else None,
        entropy_norm_mean=float(np.mean(stacked["entropy_norm"][active])) if active.any() else None,
        accuracy_argmax=float(stacked["correct_argmax"][active].mean()) if active.any() else None,
        accuracy_balanced=(float(stacked["correct_balanced"][active].mean())
                           if active.any() else None),
        recall_ramp_argmax=rate(stacked["pred_ramp_argmax"], is_ramp),
        recall_ramp_balanced=rate(stacked["pred_ramp_balanced"], is_ramp),
        recall_layer_argmax=rate(~stacked["pred_ramp_argmax"], is_layer),
        recall_layer_balanced=rate(~stacked["pred_ramp_balanced"], is_layer),
        per_slot=per_slot,
    )


# #130's buckets, named ONCE. They are simultaneously JSON keys, print labels and prose in the
# write-up, so a literal repeated per loop is a silent artifact rename waiting to happen -- the
# same hazard `slot_usage`'s two rise keys were renamed for in #132, one level up.
# ⚠️ The two aggregate rows are sums of the exact rows above them, not separate measurements. They
# exist because a curriculum schedules a FRONT and a BACK, never one slot count at a time.
COMPLEXITY_BUCKETS = (
    ("<=2  (an easy-first schedule's first phase)", 0, 2),
    (">=3  (its last phase)", 3, K_OPS),
    (">=1  (carve-needing -- the population the bar is set on)", 1, K_OPS),
    ("ALL  (what #132's prior was computed on)", 0, K_OPS),
)


def slot_counts_of(assign, fp, k_ops: int) -> tuple:
    """ONE building's LABEL slot count and its assignment class counts, over footprint columns.

    🔑 The single implementation, because #130 needs this number in three places -- the corpus
    table, the pinned table, and the scorecard's `label_slots` -- and this map's record already has
    one metric that meant two things because it was computed twice (#129/#132's `realised_rise`).

    ⚠️ Footprint columns only. Off-footprint columns are compiled away, so a slot that fires only
    outside the footprint is a slot the compiler never sees and this must not count it.
    """
    m = np.asarray(fp, bool)
    counts = np.zeros(k_ops + 1, np.int64)
    if not m.any():
        return 0, counts
    counts = np.bincount(np.asarray(assign)[m].ravel().astype(np.int64),
                         minlength=k_ops + 1)[:k_ops + 1]
    return int((counts[:k_ops] > 0).sum()), counts


def label_complexity(program: dict, cache: dict) -> tuple:
    """`slot_counts_of` over every corpus row, positionally. `(used[n], counts[n, K+1])`.

    ⚠️ From the LABEL, never from a prediction. Bucketing a population by what the model did would
    let the model pick the populations it is then scored on, which is the shape of every
    selecting-on-the-answer near-miss on this map's record.

    ⚠️ Positional: `program["assign"][i]` is paired with `cache["fp"][i]`, which holds because
    `build_program_cache` allocates `n = len(cache["ok"])` rows and stores `row=cache["row"]`. The
    pairing is checked here rather than assumed, because compacting that 146 MB `assign` array to
    the `ok` rows is a natural future optimisation and it would silently mis-pair every building.
    """
    if not np.array_equal(np.asarray(program["row"]), np.asarray(cache["row"])):
        raise ValueError("the program cache is no longer row-aligned with the height cache; "
                         "join on `row` rather than by position")
    a, fp = program["assign"], cache["fp"] > 0
    k_ops = program["types"].shape[1]
    used = np.zeros(len(a), np.int16)
    counts = np.zeros((len(a), k_ops + 1), np.int64)
    for i in range(len(a)):
        used[i], counts[i] = slot_counts_of(a[i], fp[i], k_ops)
    return used, counts


def complexity_strata(pred, label, held, rows, cache: dict, program: dict) -> dict:
    """🔑 #130's free question: is the failure GRADED by complexity, and which way must a
    curriculum run?

    `report_program_diagnostics` has said since #132 that a minor-slot recall near zero means "a
    loss **or a curriculum**", and #132 turned the loss dial (`tau*log(prior)` on the assignment
    logits) without ever pricing the other one. This is that price, and it is free -- off the label
    cache and the forward pass `diagnose_program` already makes. #132's own lesson was that asking
    the cheap question BEFORE pre-registering a fix is what stopped it shipping a decode correction
    that relabels the building; #130 asks it before anyone proposes a schedule.

    Two tables, answering different halves:

    * **the training population**, bucketed by how many slots its LABEL uses, with the assignment
      class prior recomputed INSIDE each bucket. A curriculum reorders examples, so this is what an
      ordering would actually feed the head. No model appears in it -- it is a property of the
      corpus and of #6's canonicalisation by owned area.
    * **the pinned carve-needing rows**, the same buckets, scored the way every table on this
      ticket is scored: #126's rule that `vs_input` and the collapse rate sit beside every median,
      because a row can buy `extra` by declining to act or by eating the building and neither of
      those shows up in `extra` alone.

    ⚠️⚠️ **A stratum's row is a POST-HOC SUBGROUP and it cannot pass anything.** `PROGRAM_BAR` is
    pre-registered on the whole carve-needing population and stays there. #6's write-up already
    carries one narrowing-after-the-fact and flags it as the post-hoc move it was; "but it passes
    on the easy half" would be the same error committed with a population instead of a clause.
    These rows say WHERE the residual sits (#131: price where it sits, not only how big it is).
    They do not say that any part of the arm passed.
    """
    k_ops = program["types"].shape[1]
    used_all, counts_all = label_complexity(program, cache)

    # -- the corpus half: what an ordering over complexity would actually feed the head ---------
    tr = np.nonzero((cache["ok"] > 0) & (cache["held"] == 0))[0]
    population = {}

    def bucket(name, keep):
        idx = tr[keep]
        c = counts_all[idx].sum(axis=0)
        population[name] = dict(
            n=int(len(idx)), share=float(len(idx) / max(len(tr), 1)),
            prior=[float(x) for x in (c / max(c.sum(), 1))],
            # The widest slot ratio -- slot 0 against the LAST slot -- which is the imbalance #132
            # corrected in the loss, as one number. ⚠️ `None` means the last slot has no support at
            # all in this bucket, i.e. the ratio is infinite; JSON has no infinity to write.
            skew=(float(c[0] / c[k_ops - 1]) if c[k_ops - 1] else None))

    for k in range(k_ops + 1):
        bucket(f"{k} slots", used_all[tr] == k)
    for name, lo, hi in COMPLEXITY_BUCKETS:
        bucket(name, (used_all[tr] >= lo) & (used_all[tr] <= hi))

    # -- the arm half: is its residual graded by that same axis? --------------------------------
    # 🔑 The label slot count here comes from `label`, the ALIGNED label every other diagnostic in
    # this module reads, through the same `slot_counts_of` as the corpus table above -- so the two
    # halves cannot drift and no row-id join is needed.
    pa, pt, pp = pred
    lab_used = np.array([slot_counts_of(label[0][i], held["fp"][i], k_ops)[0] for i in rows])
    heights, form, slots = [], [], []
    for i in rows:
        m, e = held["fp"][i], int(held["extent"][i])
        h = compile_program(pa[i], pt[i],
                            np.stack([plane_to_voxel(pp[i][k], e) for k in range(k_ops)]), m, e)
        heights.append(h)
        form.append(roof_description_length(h, m, int(held["y0"][i]), e))
        slots.append(len(set(np.unique(pa[i][m])) - {k_ops}))

    arm = {}

    def arm_bucket(name, keep):
        sel = np.nonzero(keep)[0]
        if not len(sel):
            return
        arm[name] = dict(
            n=int(len(sel)),
            **_median_split([heights[j] for j in sel], held, [rows[j] for j in sel]),
            ops=float(np.median([form[j]["ops"] for j in sel])),
            planar_fraction=float(np.median([form[j]["planar_fraction"] for j in sel])),
            slots_used_by_arm=float(np.mean([slots[j] for j in sel])),
            slots_used_by_label=float(np.mean([lab_used[j] for j in sel])))

    # ⚠️ from 0, not from 1. No pinned carve-needing row has a 0-slot label today, but a row that
    # did would otherwise appear only inside the aggregates and silently leave the exact rows.
    for k in range(k_ops + 1):
        arm_bucket(f"{k} slots", lab_used == k)
    # ⚠️ only the two SCHEDULE buckets here. `>=1` is a statement about the training pool's
    # empty-program majority and is identical to ALL on a carve-needing population, so printing it
    # beside ALL would be a redundant row in a table that has to be read carefully.
    for name, lo, hi in COMPLEXITY_BUCKETS[:2]:
        arm_bucket(name, (lab_used >= lo) & (lab_used <= hi))
    arm_bucket("ALL  (the pre-registered population)", lab_used >= 0)
    return dict(n=len(rows), k_ops=k_ops, n_train=int(len(tr)),
                population=population, arm=arm)


def label_robustness(label, held, rows, seed: int = 0) -> dict:
    """How much error this output space absorbs before the SURFACE metric moves.

    The loss is on the program and the scorecard is on the surface, so the obvious objection is
    that a small parameter error might be catastrophic. It is not, and this is the measurement
    that says so -- with no network involved, so it is a property of the representation.
    """
    la, lt, lp = label
    rng = np.random.default_rng(seed)
    k_ops = lp.shape[1]
    out = {"plane_noise": {}, "assignment_corrupted": {}}
    for sigma in (0.0, 0.02, 0.05, 0.10):
        hs = []
        for i in rows:
            e = int(held["extent"][i])
            n = np.stack([plane_to_normalised(lp[i][k], e) for k in range(k_ops)])
            n = n + rng.normal(0, sigma, n.shape)
            hs.append(compile_program(la[i], lt[i],
                                      np.stack([plane_to_voxel(n[k], e) for k in range(k_ops)]),
                                      held["fp"][i], e))
        out["plane_noise"][f"{sigma:.2f}"] = _median_split(hs, held, rows)
    for frac in (0.0, 0.05, 0.25):
        hs = []
        for i in rows:
            a = la[i].copy()
            hit = (rng.random(a.shape) < frac) & held["fp"][i]
            a[hit] = rng.integers(0, k_ops + 1, int(hit.sum())).astype(a.dtype)
            hs.append(compile_program(a, lt[i], lp[i], held["fp"][i], int(held["extent"][i])))
        out["assignment_corrupted"][f"{frac:.2f}"] = _median_split(hs, held, rows)
    return out


def plane_quantisation_ceiling(label, held, rows) -> dict:
    """🔑 #129's ceiling, and the measurement that chose `PLANE_BINS`. No network involved.

    A classifier cannot beat its own bins, so the first thing to know about a discretisation is what
    it costs the *exact* labels. This encodes every fitted plane to bins and decodes it straight
    back, then compiles and scores -- so the row below the continuous ceiling is the best any
    classifier over these bins could ever reach, and the gap between them is the price of asking
    the question this way at all.

    ⚠️ Reported at several resolutions on purpose. `PLANE_BINS` was fixed at 64 with the 128 row
    already visible and deliberately not taken: at 64 the ceiling is an order of magnitude below
    the 0.0603 the arm has to beat, so the binding constraint is how many examples a class gets,
    not how fine it is -- and taking the finer grid *because it scores better here* would be
    selecting a design on a number the trained arm cannot cash.
    """
    la, lt, lp = label
    out = {}
    hs = [compile_program(la[i], lt[i], lp[i], held["fp"][i], int(held["extent"][i]))
          for i in rows]
    out["exact (the continuous ceiling)"] = _median_split(hs, held, rows)
    for bins in (32, 64, 128):
        hs = []
        for i in rows:
            e = int(held["extent"][i])
            hs.append(compile_program(la[i], lt[i], rebin_planes(lp[i], la[i], e, bins),
                                      held["fp"][i], e))
        out[f"binned at {bins}" + ("  <- PLANE_BINS" if bins == PLANE_BINS else "")] = \
            _median_split(hs, held, rows)
    return out


def decode_ablation(ckpt: Path, held: dict, rows, cpu: bool = False) -> dict:
    """🔑🔑 #129's own question, measured: how much of the arm is the DECODE?

    Same weights, same forward pass, eleven reads of the posterior. #127's record is that this is
    where the leverage on such a head is -- argmax -> posterior median moved `extra` 0.1178 ->
    0.0603 with no retraining at all -- and #129's warning is that the read has to be chosen per
    quantity rather than copied.

    ⚠️ The pre-registered decode is `PLANE_DECODE`, fixed in `decode_plane_logits` before the first
    training step. This table is read AFTER the fact and reports what the alternatives would have
    scored; it does not choose. A row that beats the pre-registered one is a finding to report, not
    a decode to adopt retroactively.

    The `circmean` azimuth rows are the failure the ticket names: over an antipodal posterior the
    resultant cancels, so the direction returned is held by neither mode.
    """
    a, t, lg = program_predictions(ckpt, held, cpu, raw=True)
    if lg.ndim != 4:
        return {}
    grid = [(o, p, z) for o in ("median", "argmax") for p in ("median", "argmax")
            for z in ("argmax", "circmean")]
    # ⚠️ A pitch SWEEP, at the pre-registered offset and azimuth. Not a candidate decode -- it is
    # here because `missing` and `extra` are asymmetric in the pitch: a plane a little too steep
    # dives below GT over the far end of its region, while one a little too shallow only leaves
    # surplus above it. If that asymmetry is real, a *lower* quantile buys `missing` cheaply, and
    # that is a finding for the next arm to pre-register, never a read to adopt here.
    grid += [("median", f"q{q}", "argmax") for q in (0.25, 0.35, 0.75)]
    out = {}
    for reads in grid:
        hs, rise = [], []
        for i in rows:
            e = int(held["extent"][i])
            cen = slot_centroids(a[i], lg.shape[1])
            n = decode_plane_logits(lg[i], cen, reads)
            hs.append(compile_program(a[i], t[i],
                                      np.stack([plane_to_voxel(n[k], e) for k in range(len(n))]),
                                      held["fp"][i], e))
            rise += _realised_rise(hs[-1], a[i], t[i], held["fp"][i])
        name = " / ".join(f"{q} {r}" for q, r in zip(PLANE_QUANTITIES, reads))
        # ⚠️ `ramp_rise_*`, NOT `realised_rise_*`: `slot_usage` already publishes that name for a
        # different measurement (every used slot, `Layer`s included), and one key meaning two
        # things in one artifact is how a reader compares 6.0 against 22.0 and concludes the arm
        # got worse.
        row = dict(**_median_split(hs, held, rows), n_ramp_slots=len(rise),
                   ramp_rise_median_voxels=float(np.median(rise)) if rise else 0.0)
        out[name + ("   <- PRE-REGISTERED" if reads == tuple(PLANE_DECODE) else "")] = row
    return out


def _realised_rise(height: np.ndarray, assign, types, fp) -> list:
    """The height RANGE the compiled surface actually spans inside each slot typed `Ramp`.

    🔑 #129 asks for this beside every number, and the reason is #6's result: that arm typed 46% of
    its slots `Ramp` correctly and drew them with a median rise of **0.00 voxels**. "It predicted a
    ramp" is not evidence that it drew one, and only this reads the surface rather than the label.

    ⚠️ Returns one entry per SLOT, not a per-building summary, and the caller pools them -- which is
    what `slot_usage` does and the two numbers have to be read against each other. Averaging within
    a building first would score every building that used no `Ramp` at all as a 0-voxel rise and
    drag the median to zero for a reason that is about the type head, not about the surface.

    ⚠️ `Ramp`-typed slots only, which is NARROWER than `slot_usage`'s number over every used slot --
    a flat `Layer` genuinely reads 0 and would dilute this. The 20-column floor is `slot_usage`'s,
    so the two are on the same regions otherwise.
    """
    return [float(height[m].max() - height[m].min())
            for k in range(len(types)) if int(types[k]) == SLOT_TYPES.index("Ramp")
            for m in [(assign == k) & fp] if int(m.sum()) > 20]


def diagnose_program(ckpt: Path, held: dict, program: dict, rows, cache: dict,
                     cpu: bool = False) -> dict:
    """Every #6 diagnostic, run together so the write-up's argument is re-runnable from the repo.

    ⚠️ These carry the whole "the slopes are flat because the target is symmetric, and that is not
    a training failure" case. On this project a number that cannot be re-derived from a committed
    code path is an anecdote, so they live here rather than in a notebook.
    """
    pred = program_predictions(ckpt, held, cpu)
    k = np.array([{int(r): i for i, r in enumerate(program["row"])}[int(r)] for r in held["row"]])
    label = (program["assign"][k], program["types"][k], program["planes"][k])
    return dict(
        n=len(rows), checkpoint=str(ckpt),
        head_ablation=head_ablation(pred, label, held, rows),
        flatten_ramps=flatten_ramps(label, held, rows),
        slope_symmetry=slope_symmetry(program, cache["extent"], cache["ok"] > 0),
        canonicalisation=canonicalisation_cost(pred, label, held, rows),
        slot_usage=slot_usage(pred, label, held, rows),
        # #132: WHY slot_usage reads 1. Free, and it decides which fix the next arm pre-registers.
        assignment_collapse=assignment_collapse(ckpt, label, held, rows, cpu),
        # the type-head analogue of the line above: is `used_slots_typed_ramp` diffuse, confidently
        # wrong, or tracking a label that is itself steeply slot-index-conditional?
        type_collapse=type_collapse(ckpt, label, held, rows, cpu),
        # #130: the OTHER dial that row's report line names -- "a loss or a curriculum". #132 turned
        # the loss one; this prices the schedule one, on the label rather than on an argument.
        complexity_strata=complexity_strata(pred, label, held, rows, cache, program),
        robustness=label_robustness(label, held, rows),
        # #129's two. Both are empty for a `regress` checkpoint, which has no bins and no posterior.
        quantisation_ceiling=plane_quantisation_ceiling(label, held, rows),
        decode_ablation=decode_ablation(ckpt, held, rows, cpu),
        plane_decode=list(PLANE_DECODE), plane_bins=PLANE_BINS,
    )


def report_program_diagnostics(d: dict) -> None:
    print("\n" + "=" * 100)
    print(f"#6 PROGRAM DIAGNOSTICS  n={d['n']}   {d['checkpoint']}")
    print("\n  which head carries the surplus (one predicted head replaced by its label)")
    for name, v in d["head_ablation"].items():
        print(f"    {name:42s} extra {v['extra']:.4f}   missing {v['missing']:.4f}")
    print("\n  the form control: the PERFECT program, with its ramps flattened")
    for name, v in d["flatten_ramps"].items():
        print(f"    {name:42s} extra {v['extra']:.4f}   ops {v['ops']:.1f}   "
              f"planar {v['planar_fraction']:.2f}")
    s = d["slope_symmetry"]
    if s:
        print(f"\n  signed slope of every Ramp in the corpus (n={s['n']})")
        print(f"    mean {s['mean']:+.4f}   median {s['median']:+.4f}   "
              f"p25 {s['p25']:+.3f}   p75 {s['p75']:+.3f}")
        print(f"    positive {s['positive']:.3f}   negative {s['negative']:.3f}   "
              f"median |slope| {s['median_abs']:.3f}")
        print("    -> an L1 returns the conditional MEDIAN, so its Bayes act here is a FLAT roof")
    c = d["canonicalisation"]
    print(f"\n  canonicalisation by area vs the best permutation (what a matching loss would buy)")
    print(f"    canonical {c['canonical_order']:.4f}   best permutation {c['best_permutation']:.4f}"
          f"   cost {c['cost']:.4f} = {100*c['cost_share']:.1f}% of the error")
    u = d["slot_usage"]
    print(f"\n  slots used: arm {u['slots_used_by_arm']:.2f}   label {u['slots_used_by_label']:.2f}"
          f"   arm uses exactly one on {u['arm_uses_exactly_one']:.3f}")
    print(f"    of the slots the arm uses, typed Ramp {u['used_slots_typed_ramp']:.3f}; "
          f"{u['used_slots_compiling_flat']:.3f} compile flat (a Layer is flat BY DEFINITION)")
    print(f"    realised rise: over every used slot {u['realised_rise_all_slots_voxels']:.2f} vox; "
          f"over the {u['n_ramp_typed_slots']} Ramp-TYPED slots "
          f"{u['realised_rise_ramp_typed_voxels']:.2f} vox  <- the one that says a pitch was drawn")
    if d.get("assignment_collapse"):
        a = d["assignment_collapse"]
        print(f"\n  #132: WHY that reads 1 -- is the assignment head DIFFUSE or confidently wrong?")
        print(f"    posterior: confidence (median max p) {a['confidence_median']:.3f}   "
              f"normalised entropy {a['entropy_norm_mean']:.3f}")
        print(f"    per-column accuracy: argmax {a['accuracy_argmax']:.4f} -> balanced "
              f"{a['accuracy_balanced']:.4f}   on the DOMINANT slot "
              f"{a['dominant_argmax']:.4f} -> {a['dominant_balanced']:.4f}")
        print(f"    slots seen: argmax {a['slots_argmax_mean']:.2f}   "
              f"prior-balanced {a['slots_balanced_mean']:.2f}   label {a['slots_label_mean']:.2f}")
        print(f"    on the {a['minor_columns_total']} columns whose label is a NON-dominant slot "
              f"({a['buildings_with_a_minor_slot']:.3f} of buildings have one):")
        print(f"      p(correct slot) {a['p_true_minor']:.4f}   vs p(winner) "
              f"{a['p_won_minor']:.4f}   recall {a['recall_minor']:.4f}"
              f"   balanced {a['recall_minor_balanced']:.4f}")
        print(f"    -> near p(winner) means it KNOWS and loses the argmax (a decode); near zero "
              f"means it does not know (a loss or a curriculum)")
    if d.get("type_collapse"):
        t = d["type_collapse"]
        print(f"\n  the TYPE head's own diffuse-or-wrong question, by slot index "
              f"(label Ramp share: {'  '.join(f'{x:.3f}' for x in t['prior_ramp_share'])})")
        print(f"    overall: confidence {t['confidence_median']:.3f}   "
              f"entropy {t['entropy_norm_mean']:.3f}   accuracy argmax {t['accuracy_argmax']:.4f} "
              f"-> balanced {t['accuracy_balanced']:.4f}")
        print(f"    recall(Ramp) argmax {t['recall_ramp_argmax']:.4f} -> balanced "
              f"{t['recall_ramp_balanced']:.4f}   recall(Layer) argmax "
              f"{t['recall_layer_argmax']:.4f} -> balanced {t['recall_layer_balanced']:.4f}")
        for k, s in enumerate(t["per_slot"]):
            print(f"    slot{k}  n={s['n']:>4}  label Ramp {s['label_ramp_share']:.3f}  "
                  f"conf {s['confidence']:.3f}  recall(Ramp) {s['recall_ramp_argmax']}"
                  f" -> {s['recall_ramp_balanced']}  recall(Layer) {s['recall_layer_argmax']}"
                  f" -> {s['recall_layer_balanced']}  p(Ramp|Ramp) {s['p_ramp_given_ramp']}"
                  f"  p(Ramp|Layer) {s['p_ramp_given_layer']}")
    if d.get("complexity_strata"):
        s = d["complexity_strata"]
        k = s["k_ops"]
        print(f"\n  #130: the OTHER dial -- bucketed by the LABEL's slot count. "
              f"⚠️ POST-HOC SUBGROUPS: they say where the residual sits, they cannot pass.")
        # ⚠️ The column is sized from the longest bucket name rather than fixed. `COMPLEXITY_BUCKETS`
        # carries prose, so a name will get longer, and a header that no longer sits over its column
        # is a table nobody can read -- which on this ticket is how the wrong number gets quoted.
        w = max(len(n) for n in list(s["population"]) + list(s["arm"])) + 2
        print(f"    what an ordering over complexity would feed the head "
              f"(n={s['n_train']} training rows, no model in this table)")
        head = " ".join(f"{f'slot{j}':>8}" for j in range(k))
        print(f"      {'bucket':<{w}}{'n':>7}{'share':>8} {head}{'uncarved':>10}"
              f"{'slot0:last':>11}")
        for name, v in s["population"].items():
            sk = "inf" if v["skew"] is None else f"{v['skew']:.1f}"
            print(f"      {name:<{w}}{v['n']:>7}{v['share']:>8.4f} "
                  + " ".join(f"{x:8.4f}" for x in v["prior"][:k])
                  + f"{v['prior'][k]:10.4f}{sk:>11}")
        print(f"    the arm on the same buckets (n={s['n']} carve-needing)")
        print(f"      {'bucket':<{w}}{'n':>6}{'extra':>9}{'missing':>9}{'vs_input':>10}"
              f"{'collapse':>10}{'ops':>6}{'planar':>8}{'slots':>7}{'label':>7}")
        for name, v in s["arm"].items():
            print(f"      {name:<{w}}{v['n']:>6}{v['extra']:>9.4f}{v['missing']:>9.4f}"
                  f"{v['vs_input']:>10.4f}{v['collapse_rate']:>10.4f}{v['ops']:>6.1f}"
                  f"{v['planar_fraction']:>8.2f}{v['slots_used_by_arm']:>7.2f}"
                  f"{v['slots_used_by_label']:>7.2f}")
    print("\n  what the OUTPUT SPACE absorbs before the surface metric moves (no network)")
    for kind, rowset in (("plane noise sigma", "plane_noise"),
                         ("assignment randomised", "assignment_corrupted")):
        for key, v in d["robustness"][rowset].items():
            print(f"    {kind} {key:>5}  extra {v['extra']:.4f}   missing {v['missing']:.4f}")
    if d.get("quantisation_ceiling"):
        print(f"\n  #129: what the BINS cost the exact labels (no network) -- "
              f"PLANE_BINS={d.get('plane_bins')}")
        for name, v in d["quantisation_ceiling"].items():
            print(f"    {name:42s} extra {v['extra']:.4f}   missing {v['missing']:.4f}")
    if d.get("decode_ablation"):
        print(f"\n  #129: the same weights, eleven reads of the posterior. Pre-registered: "
              f"{' / '.join(f'{q} {r}' for q, r in zip(PLANE_QUANTITIES, d['plane_decode']))}")
        for name, v in d["decode_ablation"].items():
            print(f"    {name:52s} extra {v['extra']:.4f}   missing {v['missing']:.4f}   "
                  f"ramp rise {v['ramp_rise_median_voxels']:5.2f} vox")
    print("=" * 100)


def montage(cases, out: Path, cell: int = 5) -> Path:
    """Real building beside every arm, as shaded massing. The human's criterion, not a number.

    #10 recorded three separate occasions where reading a picture corrected a conclusion the scalar
    metric supported, so the arms are rendered side by side on the same buildings rather than
    summarised.
    """
    from PIL import Image, ImageDraw

    names = list(cases[0]["arms"])
    tiles = [[render_iso(c["target"], c["fp"], cell)] +
             [render_iso(c["arms"][n], c["fp"], cell) for n in names] for c in cases]
    tw = max(t.width for row in tiles for t in row)
    th = max(t.height for row in tiles for t in row)
    head, pad, lab, cols = 26, 8, 34, len(names) + 1
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad,
                              head + len(tiles) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, title in enumerate(["REAL BUILDING"] + [n.upper() for n in names]):
        d.text((pad + j * (tw + pad), 8), title, fill=(0, 0, 0))
    for i, row in enumerate(tiles):
        y = head + i * (th + lab)
        for j, t in enumerate(row):
            sheet.paste(t, (pad + j * (tw + pad) + (tw - t.width) // 2, y + (th - t.height) // 2))
        c = cases[i]
        # ⚠️ both numbers on every caption. `extra` is surplus left behind and `missing` is GT the
        # arm CUT INTO -- opposite failures that look nothing alike, and a sheet captioned with only
        # one of them invites the reader to diagnose the other from a picture of the first.
        d.text((pad, y + th + 4), f"id {c['id']}   " + "   ".join(
            f"{n} extra {c['extra'][n]:.3f} / missing {c['missing'][n]:.3f}" for n in names),
            fill=(40, 40, 40))
        d.line([(0, y + th + lab - 2), (sheet.width, y + th + lab - 2)], fill=(225, 225, 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


# ==================================================================================================
# plan-view maps -- the same surfaces the montage draws in 3D, read as height and as slope
# ==================================================================================================

LEGEND_W = 1000          # the map sheet's legends need this much width whatever the tiles need


def height_rgb(h: np.ndarray, fp: np.ndarray, extent: int, contour: int = 2,
               lo: int = 0) -> np.ndarray:
    """[Z, X, 3] plan view of a height map, coloured by level, with iso-contours every `contour`.

    The ramp is shared by every arm on the row and by the real building, so a colour means the same
    height across the row and the arms can be compared by eye. It spans `lo`..extent rather than
    0..extent: the deepest level any arm on that row reaches is usually well above the base, and
    stretching the ramp over the empty part of the range spends most of the colours on heights no
    arm predicted. `lo` is the deepest level anything on the row reaches, the real building
    included, so every arm is read against one scale and none is flattered by its own.

    Contours are drawn because the open question is **form**: a mound and a hip roof can carve the
    same volume and score the same `extra`, and closed concentric rings against a few straight bands
    separate them at a glance where shading does not.
    """
    from matplotlib import colormaps

    m = np.asarray(fp, bool)
    e = max(int(extent), 1)
    lvl = np.clip(np.asarray(h, np.int32), 0, e)
    t = (lvl - int(lo)) / max(e - int(lo), 1)
    rgb = (np.asarray(colormaps["turbo"](np.where(m, np.clip(t, 0.0, 1.0), 0.0)))[..., :3]
           * 255).astype(np.uint8)
    if contour:
        band = lvl // max(int(contour), 1)
        edge = np.zeros_like(m)
        edge[:, :-1] |= m[:, :-1] & m[:, 1:] & (band[:, :-1] != band[:, 1:])
        edge[:-1, :] |= m[:-1, :] & m[1:, :] & (band[:-1, :] != band[1:, :])
        rgb[edge] = (rgb[edge] * 0.45).astype(np.uint8)
    rgb[~m] = 246
    return rgb


def normal_rgb(h: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """[Z, X, 3] plan view of the top surface's unit normal, R=x-slope, G=z-slope, B=up.

    🔑 This reads the *derivative* of the height field, which is where a roof and a mound differ.
    `roof_shape_stats` failed because GT is itself terraced at 64^3 and no amplitude statistic can
    tell a discretised plane from a dome; slope can, and directly: a pitched plane is one flat
    colour, a ridge is a hard seam between two of them, and a dome is a continuous rainbow. Flat
    tops come out pale blue, which is the familiar normal-map convention.

    The gradient is taken with off-footprint columns filled from the nearest footprint column, so
    the footprint wall -- a vertical cliff carrying no roof information -- does not paint a false
    slope around every building.
    """
    m = np.asarray(fp, bool)
    H = np.asarray(h, np.float64)
    if m.any():
        H = H[tuple(ndimage.distance_transform_edt(~m, return_indices=True)[1])]
    gz, gx = np.gradient(H)
    n = np.stack([-gx, -gz, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=0, keepdims=True)
    rgb = ((n.transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb[~m] = 246
    return rgb


def _normal_key(size: int = 96) -> np.ndarray:
    """The legend for `normal_rgb`: every direction a roof can face, drawn as a hemisphere."""
    v, u = np.mgrid[0:size, 0:size] / (size / 2.0) - 1.0
    up = np.sqrt(np.clip(1.0 - (u ** 2 + v ** 2), 0.0, 1.0))
    rgb = (np.stack([u, v, up], -1) * 0.5 + 0.5) * 255
    rgb[(u ** 2 + v ** 2) > 1.0] = 246
    return rgb.astype(np.uint8)


def map_sheet(cases, out: Path, cell: int = 6, contour: int = 2) -> Path:
    """Height map and normal map, real building beside every arm, one row per building.

    The 3D montage answers "would you take this over the extruded footprint". These two views
    answer *why*: the height map shows where the volume went, and the normal map shows whether what
    is left is made of planes. They are drawn from the same `int16` height maps the arms are scored
    on, so nothing here is a separate rendering path that could disagree with the numbers.
    """
    from PIL import Image, ImageDraw

    names = list(cases[0]["arms"])
    def tile(a, box):
        z0, z1, x0, x1 = box
        return Image.fromarray(a[z0:z1, x0:x1]).resize(((x1 - x0) * cell, (z1 - z0) * cell),
                                                       Image.NEAREST)

    rows, lo = [], []
    for c in cases:
        zs, xs = np.nonzero(c["fp"])
        box = (max(zs.min() - 1, 0), min(zs.max() + 2, c["fp"].shape[0]),
               max(xs.min() - 1, 0), min(xs.max() + 2, c["fp"].shape[1]))
        lo.append(min(int(np.asarray(a)[c["fp"]].min()) for a in c["arms"].values()))
        rows.append([t for n in names
                     for t in (tile(height_rgb(c["arms"][n], c["fp"], c["extent"], contour,
                                               lo[-1]), box),
                               tile(normal_rgb(c["arms"][n], c["fp"]), box))])

    tw = max(t.width for r in rows for t in r)
    th = max(t.height for r in rows for t in r)
    head, pad, lab, foot = 40, 10, 30, 150
    cols = 2 * len(names)
    # LEGEND_W is a floor on the sheet, not a decoration: `--maps_arms` narrows the sheet to one
    # arm, and a canvas sized only by the tiles silently clips the key that says what a colour means
    sheet = Image.new("RGB", (max(cols * (tw + pad) + pad, LEGEND_W),
                              head + len(rows) * (th + lab) + foot), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, n in enumerate(names):
        x = pad + 2 * j * (tw + pad)
        d.text((x, 8), n.upper(), fill=(0, 0, 0))
        d.text((x, 24), "height", fill=(90, 90, 96))
        d.text((x + tw + pad, 24), "normals (slope)", fill=(90, 90, 96))
        if j:
            d.line([(x - pad // 2, 0), (x - pad // 2, head + len(rows) * (th + lab))],
                   fill=(210, 210, 215))
    for i, r in enumerate(rows):
        y = head + i * (th + lab)
        for j, t in enumerate(r):
            sheet.paste(t, (pad + j * (tw + pad) + (tw - t.width) // 2, y + (th - t.height) // 2))
        c = cases[i]
        d.text((pad, y + th + 6), f"id {c['id']}   extent {int(c['extent'])} vx"
               f"   {c['height_m']:.1f} m   ramp {lo[i]}-{int(c['extent'])} vx   " + "   ".join(
                   f"{n} extra {c['extra'][n]:.3f}" for n in names if n in c["extra"]),
               fill=(40, 40, 40))
        d.line([(0, y + th + lab - 2), (sheet.width, y + th + lab - 2)], fill=(228, 228, 232))

    # the two legends: what a colour means on each half of the sheet
    y = head + len(rows) * (th + lab) + 16
    from matplotlib import colormaps
    ramp = (np.asarray(colormaps["turbo"](np.linspace(0, 1, 256)))[None, :, :3]
            * 255).astype(np.uint8)
    sheet.paste(Image.fromarray(np.repeat(ramp, 22, 0)).resize((512, 22)), (pad, y + 16))
    d.text((pad, y), "HEIGHT   shared per row: the deepest level any arm reaches -> the extent",
           fill=(0, 0, 0))
    d.text((pad, y + 42), "deepest carve on the row", fill=(60, 60, 60))
    d.text((pad + 400, y + 42), "uncarved (the blockout)", fill=(60, 60, 60))
    kx = pad + 620
    sheet.paste(Image.fromarray(_normal_key()), (kx, y + 12))
    d.text((kx, y), "NORMALS   which way the roof faces", fill=(0, 0, 0))
    d.text((kx + 108, y + 20), "pale blue = flat   |   one flat colour = one pitched plane",
           fill=(60, 60, 60))
    d.text((kx + 108, y + 38), "hard seam = a ridge   |   smooth rainbow = a mound, not a roof",
           fill=(60, 60, 60))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def sheet_picks(rank, eligible, per_sheet: int) -> dict:
    """best / representative / worst, ranked by ONE arm's surplus over the eligible rows.

    🔑 Both sheet writers rank by the **first trained arm**, which is the pre-registered one, so
    "worst" means worst for the arm the bar was written for and not for whichever arm happens to be
    first in the dict. The rule lived at the montage call site while the map sheets silently ranked
    by the blockout; it is one function now so the two cannot drift apart again.
    """
    by = sorted(eligible, key=lambda i: rank[i])
    h = len(by) // 2
    return dict(best=by[:per_sheet], representative=by[h:h + per_sheet], worst=by[-per_sheet:])


def _pick_arms(heights: dict, names) -> dict:
    """The arms to draw, in the order asked for. Unknown names are an error, not a silent drop:
    a sheet quietly missing the arm it was made to show is worse than no sheet."""
    if not names:
        return heights
    missing = [n for n in names if n not in heights]
    if missing:
        sys.exit(f"--maps_arms: no such arm {missing}; have {list(heights)}")
    return {n: heights[n] for n in names}


def write_map_sheets(held: dict, heights: dict, per_sheet: int, ids=None,
                     rank_by: str | None = None) -> None:
    """Pick the buildings and write the map sheets, ranked by the arm the caller names.

    `extra` here comes from `height_split`, the column-space identity of `volume_split` -- the sheet
    is a picture of the same surfaces the artifact scores, and its captions must not come from a
    second, differently-computed number.
    """
    key = rank_by or next(iter(heights))
    extra = {name: [height_split(h[i], held["target"][i])["extra"] for i in range(len(h))]
             for name, h in heights.items()}
    if ids:
        want = {int(i) for i in ids}
        rows = {int(r): i for i, r in enumerate(held["row"])}
        if not want <= set(rows):
            sys.exit(f"--maps_ids: not in the pinned population: {sorted(want - set(rows))}")
        picks = {"picked": [rows[i] for i in ids]}
    else:
        # the carve-needing subset only: on a building whose envelope is already right, a map sheet
        # shows two flat rectangles and says nothing about form (#126 point 4)
        carve = [i for i in range(len(held["fp"]))
                 if height_split(apply_depth(held["fp"][i], int(held["extent"][i]),
                                             envelope_depth(held["fp"][i])),
                                 held["target"][i])["extra"] >= CARVE_NEEDED]
        picks = sheet_picks(extra[key], carve, per_sheet)
    for tag, sub in picks.items():
        cases = [dict(id=int(held["row"][i]), fp=held["fp"][i], extent=int(held["extent"][i]),
                      height_m=float(held["height_m"][i]),
                      arms={"real building": held["target"][i],
                            **{a: heights[a][i] for a in heights}},
                      extra={a: extra[a][i] for a in heights}) for i in sub]
        if cases:
            print(f"[maps] {map_sheet(cases, WORK / f'maps_{tag}.png')}", flush=True)


def report(res: dict) -> None:
    print("\n" + "=" * 100)
    print("the aggregate is right of the bar: #126 demoted it, so it may not head the row")
    for pop, label in (("carve", "CARVE-NEEDING buildings -- the population the bar is set on"),
                       ("flat", "ALREADY-FLAT buildings -- reported, never pooled"),
                       ("all", "all pinned buildings")):
        print(f"\n== {label} (n={res['arms']['blockout'][pop]['n']}) ==")
        print(f"{'arm':22s} {'miss':>7} {'extra':>7} {'vs_inp':>7} {'collapse':>9} "
              f"{'>env:xtr':>9} {'carved':>7} {'FORM:ops':>9} {'planar':>7} | {'(3D IoU)':>9}   "
              f"(GT: carves {res['arms']['blockout'][pop]['gt_carved_cols']:.3f} of columns, "
              f"form {res['arms']['blockout'][pop].get('gt_dl_ops', float('nan')):.1f} ops, "
              f"planar {res['arms']['blockout'][pop].get('gt_dl_planar_fraction', float('nan')):.2f})")
        for name, a in res["arms"].items():
            s = a[pop]
            w = a["beats_envelope_extra"][pop]["rate_ex_ties"]
            print(f"{name:22s} {s['missing']:>7.4f} {s['extra']:>7.4f} {s['vs_input']:>7.4f} "
                  f"{s['collapse_rate']:>9.4f} {w:>9.3f} {s['carved_cols']:>7.3f} "
                  f"{s.get('dl_ops', float('nan')):>9.1f} "
                  f"{s.get('dl_planar_fraction', float('nan')):>7.2f} | {s['vol_iou']:>9.4f}")
    if res.get("reference"):
        print("\n== this project's arms of record, re-summarised on the SAME carve-needing rows ==")
        for name, a in res["reference"].items():
            vi = "     -" if a["vs_input"] is None else f"{a['vs_input']:>7.4f}"
            print(f"{name:22s} {a['missing']:>7.4f} {a['extra']:>7.4f} {vi} "
                  f"{a['collapse_rate']:>9.4f} {'':>9} {'':>7} {'':>9} {'':>7} | "
                  f"{a['vol_iou']:>9.4f}")

    print("\n== the pre-registered bar, on the carve-needing subset ==")
    for name, v in res["verdict"].items():
        print(f"  {name:22s} beats 1-NN `extra` {str(v['beats_1nn_extra']):>5}   "
              f"collapse ok {str(v['collapse_no_worse_than_1nn']):>5}   "
              f"moved {str(v['moved']):>5}   ->  {'PASS' if v['pass'] else 'NOT MET'}"
              + ("   [KILL: identity]" if v["killed_identity"] else ""))
    print("=" * 100)


# ==================================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_from", default=str(SHIP714))
    ap.add_argument("--objective", default="ce", choices=OBJECTIVES,
                    help="which statistic of the per-column posterior to target: ce -> the mode, "
                         "mse -> the mean, quantile -> --quantile (0.5 = the median)")
    ap.add_argument("--plane_head", default="regress", choices=PLANE_HEADS,
                    help="#129. How --objective program predicts a slot's plane: `regress` is #6's "
                         "L1 on (A, Bz, Cx), which must return flat because the corpus's signed "
                         "slope is exactly symmetric; `class` is cross-entropy over binned "
                         "(offset, pitch, azimuth), which has no such Bayes act. The ONLY "
                         "difference between the two arms, so a gap is attributable to it")
    ap.add_argument("--k_planes", type=int, default=6,
                    help="planes for --objective planes. #10 measured a median of 5 operations to "
                         "explain a real roof and 9 at p75, so 6 is the median-plus with room")
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="the pinball loss's quantile; used by --objective quantile ONLY -- the "
                         "slope term reads its own fixed SLOPE_DECODE_QUANTILE. 0.5 is "
                         "the median and is the value #127 pre-committed to -- sweeping it trades "
                         "`missing` against `extra` directly and would be selecting on the answer")
    ap.add_argument("--slope_weight", type=float, default=0.0,
                    help="weight on the joint SLOPE term, added to the per-column loss and never "
                         "in place of it. 0 disables it, which is every arm on #127's record; the "
                         "pre-registered value is 1.0, fixed a priori as a 20%% share of the "
                         "converged loss (CE 1.5552, slope 0.3090) and deliberately not swept")
    ap.add_argument("--k_hyp", type=int, default=1,
                    help="#8: number of independent 'ce' hypothesis heads, trained with relaxed "
                         "winner-take-all instead of one averaged posterior. 1 (default) is every "
                         "prior arm on this file, bit-for-bit unchanged. Only valid with "
                         "--objective ce")
    ap.add_argument("--wta_epsilon", type=float, default=0.05,
                    help="with --k_hyp > 1: the gradient share given to each LOSING hypothesis "
                         "(Rupprecht et al. 2017's relaxed WTA), so an early-losing hypothesis "
                         "still learns something instead of dying")
    ap.add_argument("--tag", default=None, help="run name; defaults to the objective")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_aug", action="store_true", help="disable the 8 plan symmetries")
    ap.add_argument("--no_type_prior", action="store_true",
                    help="disable #138's type-head logit adjustment for --objective program, so an "
                         "assignment-side change (#139) can be measured without it. #139's first run "
                         "did not have this flag and was, undocumented, already the combined arm --"
                         "see 139-assignment-temperature.md's correction. Assignment's own "
                         "`assign_prior` has no equivalent switch: it has never been run isolated "
                         "from anything, so there is nothing yet for it to be confounded with.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--rebuild_program_cache", action="store_true",
                    help="re-fit #6's slot labels over the whole corpus (56 s on 48 cores)")
    ap.add_argument("--diagnose_program", default=None, metavar="CKPT",
                    help="run #6's head-swap / ramp-flatten / slope-symmetry / canonicalisation "
                         "diagnostics on a program checkpoint and exit. These carry the argument "
                         "that the flat slopes are the objective's Bayes act rather than a "
                         "training failure, so they are a code path and not a notebook")
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="score these checkpoints instead of training (name=path or path)")
    ap.add_argument("--fit_decode", action="store_true",
                    help="#8's fusion: add '<arm>_fit' siblings that post-process each trained "
                         "arm's height map (and its _median sibling, if any) through #10's "
                         "beam-search fitter before scoring")
    ap.add_argument("--fit_decode_roof_family", default=None, choices=("flat", "ramp", "cut_roof"),
                    help="with --fit_decode, also add a '<arm>_fit_<family>' sibling that biases "
                         "the fitter's per-round type choice toward this family via #9's FitBias "
                         "-- tests whether that recovers dl_planar_fraction lost to plain fitting")
    ap.add_argument("--fit_decode_smooth", type=float, nargs="*", default=[],
                    help="with --fit_decode, also add one '<arm>_fit_smXX' sibling per sigma that "
                         "Gaussian-blurs the raw height map before fitting (XX = sigma*10, e.g. "
                         "sigma=1.5 -> '_fit_sm15') -- tests whether denoising upstream of the "
                         "fitter recovers dl_planar_fraction where a bias on the fitter could not")
    ap.add_argument("--no_form", action="store_true",
                    help="skip the description-length form metric. It fits a Layer/Ramp/CutRoof "
                         "program to every arm's own surface, which is the only measure found that "
                         "separates a roof from a mound -- and it costs ~0.07s per building per arm")
    ap.add_argument("--median_decode", action="store_true",
                    help="add a second arm per CE checkpoint decoding the posterior MEDIAN rather "
                         "than the mode. A decode ablation reported beside the pre-registered arm, "
                         "never in place of it")
    ap.add_argument("--montage", type=int, default=6, help="buildings per sheet; 0 disables")
    ap.add_argument("--montage_rank", default="extra", choices=("extra", "missing"),
                    help="which failure the sheet's best/representative/worst rank by. `extra` is "
                         "surplus the arm left behind; `missing` is GT it CUT INTO -- the "
                         "destruction `collapse_rate` counts, which an extra-ranked sheet cannot "
                         "show because a building an arm ate has little surplus left on it")
    ap.add_argument("--maps", type=int, default=0,
                    help="buildings per height/normal MAP sheet -- the plan-view pair that shows "
                         "where the volume went and whether what is left is made of planes")
    ap.add_argument("--maps_ids", type=int, nargs="*", default=None,
                    help="render exactly these corpus row ids, e.g. the ones already on a montage")
    ap.add_argument("--maps_arms", nargs="*", default=None,
                    help="restrict the map sheet to these arms. Seven arms is 16 columns and "
                         "unreadable at any print size, and an unreadable figure decides nothing")
    ap.add_argument("--maps_only", action="store_true",
                    help="write the map sheets from --ckpt and exit, skipping the scored run")
    ap.add_argument("--out", default="execution/artifacts/height_map_generator_714.json")
    args = ap.parse_args()
    args.tag = args.tag or (f"heightmap_{args.objective}"
                            + ("_class" if args.objective == "program"
                               and args.plane_head == "class" else ""))

    cache = build_cache(force=args.rebuild_cache)

    if args.diagnose_program:
        ids = [int(i) for i in json.load(open(args.ids_from))["ids"]]
        r2i = {int(r): i for i, r in enumerate(cache["row"])}
        sel = np.array([r2i[i] for i in ids if i in r2i and cache["ok"][r2i[i]]])
        held = {k: cache[k][sel] for k in ("row", "fp", "target", "y0", "extent",
                                           "region", "height_m")}
        held["fp"] = held["fp"] > 0
        held["target"] = held["target"].astype(np.int16)
        # the same carve-needing split the bar is set on -- #126 point 4: a 42% no-op majority
        # flatters every aggregate, and these diagnostics are about the buildings that need a carve
        rows = [i for i in range(len(sel))
                if height_split(apply_depth(held["fp"][i], int(held["extent"][i]),
                                            envelope_depth(held["fp"][i])),
                                held["target"][i])["extra"] >= CARVE_NEEDED]
        d = diagnose_program(Path(args.diagnose_program), held,
                             build_program_cache(cache), rows, cache, cpu=args.cpu)
        report_program_diagnostics(d)
        out = Path(args.out).with_name(Path(args.out).stem + "_diagnostics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(d, open(out, "w"), indent=1)
        print(f"\n[artifact] {out}")
        return

    ckpts = {}
    if args.ckpt:
        for spec in args.ckpt:
            name, _, path = spec.rpartition("=")
            ckpts[name or Path(path).stem] = Path(path)
    else:
        ckpts[args.tag] = train(cache, args)

    # ---- the pinned population, in the pinned order -------------------------------------------
    ids = [int(i) for i in json.load(open(args.ids_from))["ids"]]
    row_to_idx = {int(r): i for i, r in enumerate(cache["row"])}
    sel = np.array([row_to_idx[i] for i in ids if i in row_to_idx and cache["ok"][row_to_idx[i]]])
    held = {k: cache[k][sel] for k in ("row", "fp", "target", "y0", "extent", "region", "height_m")}
    held["fp"] = held["fp"] > 0
    held["target"] = held["target"].astype(np.int16)
    print(f"[ids] {len(sel)} pinned buildings from {args.ids_from}", flush=True)

    if args.maps_only:
        # The sheet only needs the arms' own height maps, so it skips 1-NN retrieval, the mean roof
        # and the form fitter -- minutes of work that would produce nothing this figure draws.
        if not args.ckpt:
            sys.exit("--maps_only scores nothing, so it needs --ckpt to say what to draw")
        heights = {}
        for name, path in ckpts.items():
            heights[name], meta = predict(path, held, cpu=args.cpu)
            if args.median_decode and meta["objective"] == "ce":
                heights[f"{name}_median"], _ = predict(path, held, cpu=args.cpu, quantile=0.5)
        arms = _pick_arms(heights, args.maps_arms)
        write_map_sheets(held, arms, args.maps or 6, args.maps_ids, rank_by=next(iter(arms)))
        return

    # ---- the arms -------------------------------------------------------------------------------
    train_idx = np.nonzero((cache["ok"] > 0) & (cache["held"] == 0))[0]
    bank_fp = cache["fp"][train_idx] > 0
    bank_target = cache["target"][train_idx].astype(np.int16)
    bank_extent = cache["extent"][train_idx].astype(np.int32)
    print(f"[bank] {len(train_idx)} training buildings for retrieval and for the mean roof",
          flush=True)

    heights = {"blockout": np.stack([apply_depth(held["fp"][i], int(held["extent"][i]),
                                                 envelope_depth(held["fp"][i]))
                                     for i in range(len(sel))])}

    bank_depth = np.stack([carve_depth(bank_target[i], bank_fp[i], int(bank_extent[i]))
                           for i in range(len(train_idx))])
    profile = mean_relative_depth(bank_depth, bank_fp, bank_extent)
    heights["mean_roof"] = np.stack([mean_roof_height(profile, held["fp"][i],
                                                      int(held["extent"][i]))
                                     for i in range(len(sel))])

    t0 = time.time()
    nn = retrieve_nn(held["fp"], bank_fp)
    heights["nn_retrieval"] = np.stack([
        transplant_height(bank_target[j], bank_fp[j], int(bank_extent[j]),
                          held["fp"][i], int(held["extent"][i]))
        for i, j in enumerate(nn)])
    print(f"[1-NN] retrieved in {time.time()-t0:.0f}s  "
          f"(median footprint IoU to the retrieved row reported in the artifact)", flush=True)

    # #6's CEILING, scored down the same path as every other arm so its form is comparable: the
    # program the fitter recovered WITH GT IN HAND, compiled. It is not a generator and never
    # competes -- it is the answer to "how good could a program arm be if it predicted perfectly",
    # and #127's record is that a ceiling nobody measured is how a representation gets adopted on a
    # promise. Included whenever the label cache exists, so it costs nothing to have it.
    label_slots = {}
    if PROGRAM_CACHE.exists():
        pl = np.load(PROGRAM_CACHE)
        pidx = {int(r): i for i, r in enumerate(pl["row"])}
        pk = np.array([pidx[int(r)] for r in held["row"]])
        heights[PROGRAM_LABEL_ARM] = np.stack([
            compile_program(pl["assign"][pk[i]], pl["types"][pk[i]], pl["planes"][pk[i]],
                            held["fp"][i], int(held["extent"][i])) for i in range(len(sel))])
        # #130: the LABEL's slot count per pinned building, so ANY arm's `per_building` rows can be
        # stratified by complexity without a second run. `complexity_strata` does this inside
        # --diagnose_program for the one checkpoint under diagnosis; this makes the same axis
        # available for the comparison arms it never sees (`class129_at_q025`, 1-NN, the served CE
        # arm). Same `slot_counts_of`, so the two cannot drift apart. ⚠️ Post-hoc subgroups either
        # way -- see that docstring: the bar is on the whole carve-needing population and stays
        # there. Joined through `pk` (by row id) rather than by position, and over the 714 rather
        # than the corpus, so it neither assumes the caches stay aligned nor pays for 35k rows.
        label_slots = {int(held["row"][i]):
                       slot_counts_of(pl["assign"][pk[i]], held["fp"][i], pl["types"].shape[1])[0]
                       for i in range(len(sel))}

    ckpt_meta = {}
    for raw_name, path in ckpts.items():
        h0, m0 = predict(path, held, cpu=args.cpu)
        # ⚠️ #8: a k_hyp>1 checkpoint's decode is the ORACLE hypothesis (`predict` says so in
        # `oracle=True`), suffixed by CONTENT rather than by whatever `--tag`/`--ckpt` name was
        # chosen, so `verdict()`'s `_oracle` exclusion can never be bypassed by a forgetful name.
        name = f"{raw_name}_oracle" if m0.get("oracle") else raw_name
        heights[name], ckpt_meta[name] = h0, m0
        if args.median_decode and ckpt_meta[name]["objective"] == "ce":
            alt = f"{name}_median"
            heights[alt], ckpt_meta[alt] = predict(path, held, cpu=args.cpu, quantile=0.5)
        if args.fit_decode:
            bases = [name] + ([f"{name}_median"] if f"{name}_median" in heights else [])
            for base in bases:
                fit_name = f"{base}_fit"
                heights[fit_name] = fit_decode(heights[base], held)
                ckpt_meta[fit_name] = dict(ckpt_meta[base],
                                           decode=ckpt_meta[base]["decode"] + " -> #10 beam fit")
                if args.fit_decode_roof_family:
                    ramp_name = f"{fit_name}_{args.fit_decode_roof_family}"
                    heights[ramp_name] = fit_decode(
                        heights[base], held, bias=FitBias(roof_family=args.fit_decode_roof_family))
                    ckpt_meta[ramp_name] = dict(
                        ckpt_meta[base],
                        decode=ckpt_meta[base]["decode"] +
                        f" -> #10 beam fit (roof_family={args.fit_decode_roof_family} bias)")
                for sigma in args.fit_decode_smooth:
                    sm_name = f"{base}_fit_sm{int(round(sigma * 10)):02d}"
                    heights[sm_name] = fit_decode(heights[base], held, smooth_sigma=sigma)
                    ckpt_meta[sm_name] = dict(
                        ckpt_meta[base],
                        decode=ckpt_meta[base]["decode"] +
                        f" -> gaussian blur sigma={sigma} -> #10 beam fit")

    # ---- score, split by population, never pooled -----------------------------------------------
    rows = {name: score_arm(h, held, form=not args.no_form)
            for name, h in heights.items()}
    carve_mask = np.array([r["blockout_extra"] >= CARVE_NEEDED for r in rows["blockout"]])
    pops = {p: np.nonzero(m)[0] for p, m in
            dict(all=np.ones(len(carve_mask), bool), carve=carve_mask, flat=~carve_mask).items()}

    arms = {}
    for name, rr in rows.items():
        a = {p: summarise([rr[i] for i in idx]) for p, idx in pops.items()}
        a["beats_envelope_extra"], a["beats_envelope_iou"] = {}, {}
        for p, idx in pops.items():
            # paired against the SAME building's envelope, by index -- #126's like-for-like rule
            paired = [dict(arm=dict(extra=rr[i]["extra"], vol_iou=rr[i]["vol_iou"]),
                           blockout=dict(extra=rows["blockout"][i]["extra"],
                                         vol_iou=rows["blockout"][i]["vol_iou"])) for i in idx]
            a["beats_envelope_extra"][p] = compare_to_envelope(paired, "arm", "extra", False)
            a["beats_envelope_iou"][p] = compare_to_envelope(paired, "arm", "vol_iou", True)
        arms[name] = a

    res = dict(
        meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), question="#127",
                  ids_from=args.ids_from, gt_h5=str(H5.relative_to(REPO)),
                  n_pinned=len(sel), n_carve=int(carve_mask.sum()),
                  n_train=len(train_idx), depth_classes=DEPTH_CLASSES,
                  checkpoints=ckpt_meta, run_flags=dict(
                      epochs=args.epochs, batch=args.batch, lr=args.lr, width=args.width,
                      seed=args.seed, augment=not args.no_aug,
                      # ⚠️ None on a --ckpt rerun: this block is the flags of THIS invocation, and
                      # a head recorded here that did not produce the checkpoints beside it is the
                      # same trap `predict` avoids by carrying its own provenance
                      plane_head=args.plane_head if not args.ckpt else None,
                      trained_here=not bool(args.ckpt))),
        arms=arms, verdict=verdict(arms, "carve"),
        reference=reference_arms({int(held["row"][i]) for i in pops["carve"]}),
        nn_footprint_iou=float(np.median([
            float((held["fp"][i] & bank_fp[j]).sum()) / max(float((held["fp"][i] | bank_fp[j]).sum()), 1)
            for i, j in enumerate(nn)])),
        per_building={name: rr for name, rr in rows.items()},
        label_slots=label_slots,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    report(res)
    print(f"\n[artifact] {out}")

    # ranked by the FIRST trained arm, which is the pre-registered one -- so "worst" means worst
    # for the arm the bar was written for, not for whichever arm happens to lead the dict
    model_names = [n for n in heights if n in ckpts]
    key = model_names[0] if model_names else "nn_retrieval"

    if args.maps or args.maps_ids:
        arms = _pick_arms(heights, args.maps_arms)
        write_map_sheets(held, arms, args.maps or 6, args.maps_ids,
                         rank_by=key if key in arms else next(iter(arms)))

    if args.montage:
        # 🔑 rank on the failure being investigated. `extra` is surplus; `missing` is the volume the
        # arm ATE, which is what `collapse_rate` counts -- and an extra-ranked sheet cannot show it,
        # because the buildings an arm destroys are often the ones it leaves least surplus on.
        picks = sheet_picks([r[args.montage_rank] for r in rows[key]],
                            pops["carve"].tolist(), args.montage)
        suffix = "" if args.montage_rank == "extra" else f"_by_{args.montage_rank}"
        for tag, sub in picks.items():
            cases = [dict(id=int(held["row"][i]), fp=held["fp"][i], target=held["target"][i],
                          arms={n: heights[n][i] for n in heights},
                          extra={n: rows[n][i]["extra"] for n in heights},
                          missing={n: rows[n][i]["missing"] for n in heights}) for i in sub]
            if cases:
                print(f"[montage] {montage(cases, WORK / f'{tag}{suffix}.png')}")


if __name__ == "__main__":
    main()
