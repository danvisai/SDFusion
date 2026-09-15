# #138 — Does correcting the type head's own imbalance turn flat slots into planes without costing the surplus?

*Effort: solid-first semantic architectural carving. Opened 2026-09-02 from [#132](132-overcarve-and-assignment.md),
which changed the assignment head and the pitch decode, named the type head as the untouched
binding constraint on its own KILL, and deliberately did not touch it -- "one change per head was
the whole design of this arm." Run and written 2026-09-02. One A100, ~19 minutes of training; the
two diagnostic passes are the same forward pass `assignment_collapse` uses, CPU/GPU as available.*

> ⚠️ **Renumbered on filing.** This write-up was authored as `137-type-head-imbalance.md` against an
> issue #137 that was never created; `PR #137` took that number hours later. The ticket was filed
> retroactively on 2026-09-02 as [#138](https://github.com/danvisai/SDFusion/issues/138) and the file
> renamed to match. The run itself is unchanged — only the number is. ⚠️ The pre-registration quoted
> below lives in the module docstring and was **uncommitted** at filing time, so it is not
> independently verifiable in git history.

> #132 left `planar_fraction` at 0.12 -- 61% of the arm's used slots typed `Layer`, 59.2%
> compiling flat -- and named that the binding constraint on the fourth clause of `PROGRAM_BAR`
> without asking why the type head reads that way. Is it diffuse, confidently wrong, or tracking a
> label that is itself steeply slot-index-conditional? And whichever it is, does the same fix
> #132 used on the assignment head -- a training-side logit adjustment, not a decode-side one --
> work here too?

Code `scripts/foundations/train_height_map_generator.py` (`type_prior`, `type_stats`,
`type_collapse`, `TYPE_TEMPERATURE`, and `program_loss`'s new `type_prior=` argument), contract
tests `scripts/foundations/test_train_height_map_generator.py`, artifacts
`execution/artifacts/height_map_generator_typeadj_train.json` and
`..._typeadj_714_diagnostics.json`. No montage this run (`--montage 0`, matching #132's own
command) -- the scorecard alone already gives a clean verdict, and #127/#132's warning that a
picture can overturn a scalar is a reason to render one before shipping this arm, not before
reading its numbers.


## 🔑🔑 The fix does exactly what it diagnosed, and the scorecard gets worse anyway

**`planar_fraction` reaches 0.50 -- matching the real building's own ratio exactly, and the best
number ever measured on this map's record, on any trained arm.** Every prior program arm scored
0.00 or 0.12 (#132) or 0.17-0.25 (#127's per-column arms). This is the first time a trained arm's
compiled surface, re-fit by #10's own program fitter, spends half its operations on a plane rather
than a flat `Layer`.

⚠️ **And every surplus number moved the wrong way.** Against #132, on the same 411 carve-needing
buildings:

| | `missing` | `extra` | `vs_input` | collapse | ops | **planar** | *(3D IoU)* |
|---|---|---|---|---|---|---|---|
| heightmap_program_adj *(#132)* | 0.0659 | 0.0832 | 0.8470 | 0.2579 | 1.0 | 0.12 | *0.8080* |
| **heightmap_program_typeadj *(#138)*** | **0.0835** | **0.0938** | 0.8163 | **0.2774** | 1.0 | **0.50** | *0.7960* |

`missing`, `extra` and the collapse rate are all worse, and 3D IoU falls too. The fix bought the
one thing it was diagnosed to buy and nothing on this map's record has bought it more cleanly, but
the arm it produced is a worse building.


## The free question first, the same order #132 asked it of the assignment head

`type_prior` shows the label itself is steeply slot-index-conditional -- slots are canonicalised by
AREA (#6), so a building's biggest region is a pitch more often than not and its smallest is almost
always a flat setback. Measured over the 34,909 training rows: slot 0 is Ramp 59.4% of the time,
slot 1 52.3%, slot 2 32.3%, slot 3 13.4%.

`type_collapse`, run on `heightmap_program_adj.pt` (#132's checkpoint, **before** this fix was
written) asks whether the type head is diffuse, confidently wrong, or tracking that gradient
correctly and simply read at the wrong threshold:

| slot | label Ramp share | recall(Ramp) | recall(Layer) | p(Ramp \| Ramp label) | p(Ramp \| Layer label) |
|---|---|---|---|---|---|
| 0 | 0.555 | 0.741 | 0.721 | 0.691 | 0.389 |
| 1 | 0.501 | 0.626 | 0.757 | 0.611 | 0.381 |
| 2 | 0.299 | 0.357 | 0.939 | 0.421 | 0.246 |
| 3 | 0.105 | **0.087** | 0.995 | 0.235 | 0.128 |

🔑 **The head is not blind.** `p(Ramp | Ramp label)` exceeds `p(Ramp | Layer label)` at every slot
-- the information is there. It is a plain argmax at a fixed 0.5 threshold that is the wrong
decision rule for a base rate that low: slot 3 is Ramp only 10.5% of the time, so a moderately
informative posterior (mean 0.235 on true Ramps) almost never crosses 0.5, and recall craters to
8.7%. That is a calibration failure, not a representational one -- the same shape of finding #132
made about the assignment head being "diffuse, not confidently wrong."

⚠️ **And the decode-side fix is refuted, the same way and for the same reason #132 refuted it on
the assignment head.** Dividing each slot's posterior by its own label prior before the argmax --
`type_collapse`'s `balanced` column -- recovers slot 3's Ramp recall 0.087 → 0.957, and pays for it
with Layer recall 0.995 → 0.413 at that slot and overall accuracy 0.7576 → 0.6757. It mostly
relabels the building's flat regions as pitched, which is `decode_assignment`'s failure mode
arriving at the second head.
`TestTypeStats.test_the_balanced_read_can_flip_a_slot_the_argmax_loses` pins the mechanism; the
served decode stays a plain argmax.

**So the fix goes where #132's did:** `tau * log(prior[k, c])` added to slot k's class-c TYPE logit
during training (`TYPE_TEMPERATURE`, fixed a priori at 1.0, the full adjustment, not swept -- the
same reason `ASSIGN_TEMPERATURE` is not swept), decode left as the plain argmax. `type_prior` is
computed once from the TRAINING split's labels, exactly like `assignment_prior`, and travels with
the checkpoint (`type_prior`, `type_temperature`) so a re-scored old checkpoint cannot present
itself as having trained under a correction it did not have.


## The bar, pre-registered before the run

Committed in the module docstring before the first training step. Nothing else moves: same 40
epochs, same seed, same `plane_head class`, same `PLANE_DECODE` (pitch q0.25), same `assign_prior`
at `ASSIGN_TEMPERATURE` 1.0, same selection rule (validation `missing + extra`), same 411
carve-needing buildings, same `PROGRAM_BAR`, unchanged for the fourth time.

    PASS   ops <= 3.0   AND   planar >= 0.40   AND   extra < 0.0603
    GUARD  collapse <= 1-NN's 0.1582            AND  vs_input < 0.98
    KILL   planar <= 0.20

⚠️ **Pre-registered prediction, so this is falsifiable:** the type fix should raise
`planar_fraction` and the Ramp-typed share of used slots without moving `extra`/`missing`/collapse
by much, because the assignment head and the pitch decode are untouched. If `extra` or the collapse
rate moved by more than a rounding amount instead, that would be a result about the two heads'
losses interacting, not about either head read in isolation.

**The prediction was wrong on the second half and right on the first.** `planar_fraction` moved far
more than "raised" -- 0.12 → 0.50 -- and `extra`/`missing`/collapse did not stay put; they got
worse by more than a rounding amount. The interaction the prediction flagged as the alternative
outcome is what happened.


## Result — the KILL clause finally clears, and two other clauses take its place

40 epochs, 3.58M parameters, ~19 min on one A100. The rule selected **epoch 21 of 40** -- the same
epoch #132's own run selected. Artifacts: `execution/artifacts/height_map_generator_typeadj_train
.json` and `..._typeadj_714_diagnostics.json`.

| arm (411 carve-needing) | `missing` | `extra` | `vs_input` | collapse | ops | **planar** | *(3D IoU)* |
|---|---|---|---|---|---|---|---|
| the real building | — | — | — | — | **2.0** | **0.50** | — |
| program label *(sees GT — the ceiling)* | 0.0000 | 0.0035 | 0.8226 | 0.0000 | 2.0 | 0.50 | *0.9965* |
| blockout | 0.0000 | 0.2308 | 1.0000 | 0.0000 | 0.0 | 0.00 | *0.8125* |
| 1-NN retrieval *(the guard)* | 0.0257 | 0.1031 | 0.8743 | **0.1582** | 2.0 | 0.17 | *0.8355* |
| CE + median *(#127, served)* | 0.0385 | **0.0603** | 0.8432 | 0.0268 | 6.0 | 0.20 | *0.8948* |
| `heightmap_program` *(#6)* | 0.0218 | 0.1236 | 0.8952 | 0.0073 | 1.0 | 0.00 | *0.8572* |
| `heightmap_program_class` *(#129)* | 0.0013 | 0.1507 | 0.9714 | 0.1022 | 1.0 | 0.00 | *0.8126* |
| `heightmap_program_adj` *(#132)* | 0.0659 | 0.0832 | 0.8470 | 0.2579 | 1.0 | 0.12 | *0.8080* |
| **`heightmap_program_typeadj` *(#138, selected — epoch 21)*** | 0.0835 | 0.0938 | 0.8163 | **0.2774** | 1.0 | **0.50** | *0.7960* |

    PASS   ops <= 3.0            1.0     ✔
           planar >= 0.40        0.50    ✔
           extra < 0.0603      0.0938    ✘
    GUARD  collapse <= 0.1582  0.2774    ✘
           vs_input < 0.98     0.8163    ✔
    KILL   planar <= 0.20        0.50    -> not fired

Evaluated by `verdict()`, not by this table. **Verdict: NOT MET.** ⚠️ It is NOT MET for a different
reason than #132: #132's `killed_flat` clause fired (planar 0.12 ≤ 0.20); #138 clears the KILL
clause by a wide margin and fails on `beats_served_extra` and the collapse `GUARD` instead. That is
progress on the axis the map has been stuck on since #6, spent on a different failure.


## Why the trade happens: the same correction that recalls a real Ramp also mistypes a real Layer

`type_collapse` on `heightmap_program_typeadj.pt` itself, same population, plain argmax (the served
decode):

| slot | label Ramp share | recall(Ramp) | recall(Layer) | p(Ramp \| Ramp label) | p(Ramp \| Layer label) |
|---|---|---|---|---|---|
| 0 | 0.555 | 0.684 *(was 0.741)* | 0.847 *(was 0.721)* | 0.634 | 0.316 |
| 1 | 0.501 | 0.609 *(was 0.626)* | 0.786 *(was 0.757)* | 0.601 | 0.364 |
| 2 | 0.299 | **0.595** *(was 0.357)* | 0.832 *(was 0.939)* | 0.588 | 0.397 |
| 3 | 0.105 | **0.739** *(was 0.087)* | 0.643 *(was 0.995)* | 0.606 | 0.427 |

🔑 **The mechanism is exactly the one predicted, at the size the prediction did not expect.** Slot
3's Ramp recall moves 0.087 → 0.739 -- most of the way to the decode-side `balanced` read's 0.957,
and at a far better price than that read paid (Layer recall 0.643 against the decode-side
correction's 0.413). Slot 2 moves almost as far, 0.357 → 0.595. Slots 0 and 1, whose label priors
were already near 50/50, barely move -- exactly the shape `type_prior`'s per-slot design predicts,
and the reason a single scalar prior would have missed it.

⚠️ **But every recovered Ramp recall is paid for in Layer recall at the same slot**, and slot 2 and
3 are large fractions of the population (281 and 219 of 411 buildings have an active slot there).
`compile_program` reads a slot's continuous plane parameters only when its type says `Ramp`; a slot
that is genuinely flat but now mistyped `Ramp` gets a real, non-zero pitch compiled into a roof that
should have stayed level. That is where the surplus comes from: the fitted `planar_fraction`
climbing to 0.50 is exactly this -- more slots compiling as tilted planes -- and a fraction of those
newly-tilted slots are wrong. `used_slots_typed_ramp` rose from 0.390 (#132) to **0.570**, and the
per-bucket form table shows where it landed:

| label slot count | ops (#132 → #138) | planar (#132 → #138) | extra (#132 → #138) |
|---|---|---|---|
| 1 slot | 1.0 → 1.0 | 1.00 → 1.00 | 0.0257 → 0.0309 |
| 2 slots | 1.0 → 1.0 | 1.00 → 0.60 | 0.0646 → 0.0696 |
| 3 slots | 1.0 → 1.0 | 0.33 → 0.33 | 0.0861 → 0.0807 |
| 4 slots | 1.0 → 1.0 | **0.00 → 0.40** | 0.1360 → 0.1404 |

The 4-slot bucket is where `planar` moves most (0.00 → 0.40) and it is also the largest bucket
(219 of 411, 53%) and the one already carrying the worst collapse rate before this fix (0.3425 in
#132). Buying planes there is buying them in the population least able to absorb a wrong one.

⚠️ **`ops` is completely flat, in every bucket, at 1.0.** The type fix changes what a slot IS once
one has been drawn; it does nothing to how many slots the (untouched) assignment head draws. #6's
description-length pair still reads a building explained by one operation, half the time now a
plane instead of a flat -- progress on `planar_fraction` with zero progress on `dl_ops`, which is
exactly why `verdict()` requires both clauses at once (the same reason #127's plane head and #6's
own first arm each cleared one form clause with a terrace).


## What this settles, and what it does not

**Settles:**
* 🔑🔑 **The type head's imbalance was exactly what `type_collapse` said it was: a calibration
  problem, fixable in the loss, not a representational gap.** The same `tau * log(prior)` recipe
  #132 used on the assignment head works on the type head by the same mechanism, and the diagnosis
  (diffuse-vs-wrong, decode-side refutation) transfers over unchanged.
* 🔑 **This is the best `planar_fraction` ever measured on a trained arm on this map -- 0.50, tying
  the real building's own ratio exactly.** Every earlier program arm (#6, #129, #132) scored
  0.00-0.17. Fixing the type head's calibration is sufficient to make roughly half of the arm's
  drawn operations genuinely planar.
* **The two heads' losses interact, and not gently.** `extra`, `missing`, the collapse rate and 3D
  IoU are all worse than #132's, concentrated in the 4-slot bucket where more Ramp-typing meets the
  buildings the (unchanged) assignment head already over-carves worst.

**Does not settle:**
* ⚠️ **The arm does not pass `PROGRAM_BAR`.** It clears both form clauses for the first time on
  this map's record and fails on `extra` and the collapse `GUARD` instead of the `killed_flat`
  clause #132 tripped. Progress moved from one clause to two others; it did not close the ticket.
* ⚠️ **Whether a smaller `TYPE_TEMPERATURE` trades some of the planar gain for some of the surplus
  back is untested.** 1.0 is the full, pre-registered adjustment and was deliberately not swept,
  for the same reason `ASSIGN_TEMPERATURE` was not: sweeping either after seeing this result would
  be selecting on the answer. A swept temperature is a different, honest follow-up, not a
  same-ticket revision.
* ⚠️ **No montage was rendered** (`--montage 0`, matching #132's own command). #127 and #132 both
  found a picture disagreeing with a scorecard; this ticket's numbers are decisive enough to write
  up without one, but a visual check of where the new false-Ramp slots land is still open before
  this arm is considered for anything beyond this record.
* The assignment head's own diffuseness (#132's finding: confidence 0.34, entropy 0.88, unchanged
  here) is untouched and is very likely the reason the type fix's errors land where they do --
  slots the assignment head is already unsure about are the ones most exposed to a type error, but
  that interaction is inferred from the bucket table, not measured directly.


## Pinned

`scripts/foundations/test_train_height_map_generator.py` -- 167 tests, 17 of them #138's:
`TestTypePrior` (3), `TestTypeStats` (5), `TestTypeTemperatureIsPreRegistered` (1) and
`TestLogitAdjustedTypeLoss` (4, mirroring `TestLogitAdjustedAssignmentLoss` exactly: a uniform
prior changes nothing, a skewed prior penalises the majority type, inactive slots are unaffected,
and the prior's shape matches the model's own type head regardless of `--k_planes`). The load-
bearing one is `test_the_balanced_read_can_flip_a_slot_the_argmax_loses`, which pins the mechanism
the decode-side refutation rests on.


## What follows

- **The question #138 asked is answered: yes, the same fix works, mechanically, exactly as
  diagnosed** -- and it is not sufficient on its own, because it interacts with a head that #132
  already found and left unfixed.
- 🔑 **The two heads cannot safely be corrected independently a third time.** #132 already found
  the assignment head diffuse; #138 finds the same shape of fix on the type head buys real
  calibration and real cost. The natural next arm is not a bigger type-temperature sweep but
  addressing the assignment head's own diffuseness first -- #132's own "does not settle" list
  named this and #138 is now evidence for why it matters: a confident, well-formed assignment would
  give the type fix fewer wrong slots to mistype in the first place.
- **`planar_fraction` 0.50 is a real capability this map did not have before**, and it is now on
  the record as separable from the surplus cost -- the next arm that wants both will have to hold
  this one's mechanism (recall traded for precision, concentrated in the highest-slot-count
  buildings) in view rather than rediscover it.

See [132-overcarve-and-assignment.md](132-overcarve-and-assignment.md),
[129-classified-plane-parameters.md](129-classified-plane-parameters.md),
[6-program-generator.md](6-program-generator.md),
[130-baselines-diffusion-curriculum.md](130-baselines-diffusion-curriculum.md).
