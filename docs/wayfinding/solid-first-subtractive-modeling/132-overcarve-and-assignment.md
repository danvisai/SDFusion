# #132 — Stop the over-carve, and make the assignment commit to a second region

*Effort: solid-first semantic architectural carving. Opened 2026-08-30 from
[#129](129-classified-plane-parameters.md), which answered its mechanism question YES and failed its
bar anyway. Third arm on the formulation [#6](6-program-generator.md) chose. Run and written
2026-08-31.*

> #129 settled that classifying the plane parameters draws the pitch a regression provably cannot,
> and that the same head cuts through the building. Neither trained arm has ever used more than one
> slot. Can one run stop the over-carve without giving back the pitch, and make the assignment
> commit to a second region?


## The two failures sit on different heads, so the arm changes one thing on each

* **Surplus is a PLANE problem.** Swapping in label planes takes `missing` **0.1065 → 0.0006** while
  barely moving `extra`. 🔑 `extra` and `missing` are not symmetric in a pitch: a plane a little too
  **steep** dives below GT over the far end of its region and is charged the whole trench, while one
  a little too **shallow** only leaves surplus above it. The loss-minimising *parameter* estimate is
  a **biased geometry** estimate, and the correction is to read the pitch below the median.
  → `PLANE_DECODE` pitch: `median` → **`q0.25`**.
* **Form is an ASSIGNMENT problem.** `dl_ops` reads 1.0 on both arms because both use **one region**
  — #6 at 1.19 slots, #129 at 0.90 — not because their planes are flat. The output space is not the
  constraint: the compiled label uses **3.06** slots at 2.0 ops / 0.50 planar.
  → the assignment cross-entropy is **logit-adjusted by the label prior**, training-side only.

Both pre-registered in the module docstring in `2810e0b`, **before the first training step**.


## 🔑🔑 The free question came first, and it refuted the obvious fix

#132 required the assignment collapse to be *diagnosed* before a fix was chosen, because this map
has twice found that the decode was the answer — #127's argmax → posterior median moved `extra`
0.1178 → 0.0603 on one line, and #129's azimuth `argmax` beat `circmean` on the mechanism it was
chosen for. `assignment_collapse`, on both of #129's checkpoints:

| | arm of record | endpoint |
|---|---|---|
| confidence (median max posterior, 5 classes) | **0.431** | 0.425 |
| normalised entropy | **0.799** | 0.798 |
| slots seen by `argmax` | 0.90 | 1.15 |
| slots seen prior-balanced | 3.70 | 4.00 |
| slots in the label | 3.06 | 3.06 |
| recall on non-dominant-slot columns, `argmax` | **0.0000** | 0.0066 |
| the same, prior-balanced | 0.2829 | 0.2412 |

**The head is DIFFUSE, not confidently wrong.** With five classes a decided head would sit near
confidence 1.0 and entropy 0.0; this one is at 0.43 and 0.80, and the per-column `argmax` never once
recovers a non-dominant slot across **201,777** such columns. That reads exactly like a decode
problem, and the mechanism was in hand: slots are canonicalised by **area** (#6), so slot 0 owns
most columns of most buildings and the per-column cross-entropy is imbalanced **by construction of
the label**, not by anything geometric.

⚠️ **And the post-hoc correction is refuted — measured before the run rather than after it.**

| read | slots seen | minor recall | per-column accuracy | **dominant-slot accuracy** |
|---|---|---|---|---|
| `argmax` | 0.90 | 0.0000 | **0.4245** | **0.8251** |
| prior-balanced (τ=1) | 3.70 | 0.2829 | 0.2203 | **0.1275** |

Dividing the posterior by the model's own marginal buys the minor slots by **relabelling the
building**: overall accuracy halves and the dominant slot is destroyed. It is #129's `circmean`
failure in a second place — better on the metric it was chosen for, catastrophic on the population.

🔑 **The cause is structural, and it is pinned by a test rather than asserted.** A class that is
*flat over the plan* has `p / prior == 1` by construction, so the dominant slot lands in a tie with
every other flat class and the argmax between them is decided by noise
(`test_and_the_half_that_did_not_it_neutralises_a_flat_class`).

**So the adjustment moved to where it can change what is LEARNED instead of rescaling what was
not**: `τ · log(prior)` added to the assignment logits during training, with inference left as the
plain `argmax`. That pairing is the standard one, and it is the half of the correction the diagnosis
supports. The prior is the label's own class frequency over training footprint columns, computed
once and never from the model or from the pinned 714:

    slot0 0.2491   slot1 0.1161   slot2 0.0501   slot3 0.0210   uncarved 0.5638

Slot 0 is **12×** slot 3, and `uncarved` is the majority by a wide margin. That is the imbalance,
and it is a property of the label's area canonicalisation rather than of any building.

⚠️ **A bug this caught on the first batch, worth recording because the failure mode was silent-ish.**
`make_model` **ignores `--k_planes`** for the program objective and builds the assignment head with
`K_OPS + 1 = 5` channels. Sizing the prior from the flag gave 7 entries against a 5-class head — and
made the printed diagnostic misread, because what a 7-entry printout labelled `slot4` is in fact the
**uncarved** class. It crashed before the first backward pass, so nothing trained under it, and
`test_the_prior_matches_the_models_assignment_head` now builds the model and compares the prior's
length against the head's own channel count.


## What was pre-registered, and what it predicts

⚠️ **The selection rule is deliberately unchanged**, with a falsifiable prediction attached. #129's
rule picked epoch 2 of 40 because the classified head traded `missing` for `extra` as it trained, so
the symmetric difference plateaued while `extra` kept falling. **The prediction: the `q0.25` pitch is
what stops that trade, so the rule should now select a late epoch. If it picks epoch 2 again, the
pitch read did not work** — and that is the finding, not a licence to change the rule afterwards.

⚠️ **A second pre-registered prediction, so the arm is falsifiable in parts:** on #129's weights
`q0.25` alone leaves collapse at 0.2409, above 1-NN's 0.1582. The pitch read is **not** expected to
pass on its own; the assignment change has to do the rest. If the arm passes with only one of the
two working, that is a result about the other.

**The bar is `PROGRAM_BAR`, unchanged for the third time**, on the same 411 carve-needing rows
through the same `verdict()`.

    PASS   median `dl_ops` <= 3.0 AND median `dl_planar_fraction` >= 0.40 AND median `extra` < 0.0603
    GUARD  collapse rate no worse than 1-NN's (0.1582), and `vs_input` < 0.98
    KILL   median `dl_planar_fraction` <= 0.20


## Result: a third KILL, and the two changes are separately attributable

40 epochs, 3.58M parameters, ~20 min on one A100. The rule selected **epoch 21 of 40**. Artifacts:
`execution/artifacts/height_map_generator_adj_714.json` and `..._714_diagnostics.json`. **Every arm
in the run is listed**, not a chosen subset.

| arm (411 carve-needing) | `missing` | `extra` | `vs_input` | collapse | **ops** | **planar** | *(3D IoU)* |
|---|---|---|---|---|---|---|---|
| the real building | — | — | — | — | **2.0** | **0.50** | — |
| program label *(sees GT — the ceiling)* | 0.0000 | 0.0035 | 0.8226 | 0.0000 | **2.0** | **0.50** | *0.9965* |
| blockout | 0.0000 | 0.2308 | 1.0000 | 0.0000 | 0.0 | 0.00 | *0.8125* |
| mean_roof | 0.0135 | 0.1369 | 0.9070 | 0.0000 | 2.0 | 0.00 | *0.8640* |
| 1-NN retrieval *(the guard)* | 0.0257 | 0.1031 | 0.8743 | **0.1582** | 2.0 | 0.17 | *0.8355* |
| CE + median *(#127, served)* | 0.0385 | **0.0603** | 0.8432 | 0.0268 | 6.0 | 0.20 | *0.8948* |
| `heightmap_program` *(#6)* | 0.0218 | 0.1236 | 0.8952 | 0.0073 | 1.0 | 0.00 | *0.8572* |
| `heightmap_program_class` *(#129)* | 0.0013 | 0.1507 | 0.9714 | 0.1022 | 1.0 | 0.00 | *0.8126* |
| **`heightmap_program_adj` *(#132, selected — epoch 21)*** | 0.0659 | **0.0832** | 0.8470 | **0.2579** | 1.0 | **0.12** | *0.8080* |
| `..._adj_last` *(epoch 40, **diagnostic, not the arm**)* | 0.0997 | 0.0630 | 0.8056 | 0.3601 | 1.0 | 0.20 | *0.7856* |
| `class129_at_q025` *(#129's weights, pitch read ONLY)* | 0.0312 | 0.1217 | 0.8923 | **0.0438** | 1.0 | 0.00 | *0.8392* |

    PASS   ops <= 3.0            1.0     ✔
           planar >= 0.40        0.12    ✘
           extra < 0.0603        0.0832  ✘
    GUARD  collapse <= 0.1582    0.2579  ✘
           vs_input < 0.98       0.8470  ✔
    KILL   planar <= 0.20        0.12    -> **FIRED**

Evaluated by `verdict()`, not by this table.

### ✅ The assignment change worked, and it is the first arm here to use a second region

`slot_usage` is now published for **every** arm in the run, not only the selected one:

| | #6 | #129 | `class129_at_q025` | **#132 arm** | #132 endpoint | label |
|---|---|---|---|---|---|---|
| slots used | 1.19 | 0.90 | 0.90 | **2.03** | 2.12 | 3.06 |
| uses exactly one slot | — | 0.895 | 0.895 | **0.324** | 0.260 | — |
| `Ramp`-typed share of used slots | — | 0.522 | 0.522 | 0.390 | 0.421 | — |
| **`Ramp`-typed slots, absolute** | — | — | 190 | **308** | 333 | — |
| realised rise, `Ramp`-typed | — | 22.0 | 16.0 | **12.0** | 14.0 | — |
| realised rise, every used slot | — | 6.0 | 4.0 | 0.00 | 0.00 | — |
| recall on non-dominant-slot columns | — | 0.0000 | 0.0000 | **0.2801** | — | — |
| p(correct slot) on those columns | — | 0.1654 | 0.1654 | **0.2545** | — | — |

🔑 **The absolute row corrects a reading the share invites.** The `Ramp`-typed *share* falls
0.522 → 0.390, which sounds like fewer ramps; the *count* goes **190 → 308**, +62%. The arm draws
more ramps AND more layers — the share fell because the layers grew faster. So "more slots did not
become more planes" is about the ratio the `planar` metric reads, not about the arm drawing fewer
pitched regions than #129. It draws more.

🔑 **And that recall is under the PLAIN argmax**, which is the whole point: the logit-adjusted loss
*taught* what the refuted post-hoc read could only relabel. It reaches the same 0.28 the balanced
read reached on #129's weights, without the balanced read's 0.8251 → 0.1275 wreckage of the dominant
slot at decode time.

**The montage shows it directly.** On a representative gable the arm draws **two pitched planes
meeting at a ridge**, where #129's weights under the same pitch read (`class129_at_q025`) draw a
single shed plane over the whole roof. That is the K = 1 ceiling broken, visibly, for the first time
on this ticket.

### ✅ The pitch read did exactly what it was designed to do

Isolated on #129's own weights, changing nothing but the read:

| `class129_at_q025` vs #129 as recorded | `missing` | `extra` | collapse |
|---|---|---|---|
| #129, pitch `median` | 0.0013 | 0.1507 | 0.1022 |
| the same weights, pitch `q0.25` | 0.0312 | **0.1217** | **0.0438** |

Surplus down, and the collapse rate cut by more than half. The asymmetry argument holds. On #132's
own weights the same trade appears in the decode table: `q0.25` against `median` is `extra`
0.0909 → 0.0832 and `missing` 0.0832 → 0.0659, bought with realised `Ramp` rise 16 → **12 voxels**.

⚠️ **A correction to my own reading, and it is #129's naming hazard repeating.** `slot_usage`
published `realised_rise_median_voxels` over **every used slot**, `Layer`s included, and it read
**0.00** for this arm. That is not "the pitch is gone" — the `Ramp`-typed rise is **12.00 voxels**.
The arm now uses 2.03 slots of which only 39% are typed `Ramp`, so the flat majority drags the
all-slots median to zero. One name, two measurements, and I misread it once before checking.

🔑 **Fixed in code this time, not in prose.** #129 hit the same hazard and renamed only
`decode_ablation`'s copy; `slot_usage` kept the ambiguous key and it misled the next reader, who was
me. It now publishes **both** — `realised_rise_all_slots_voxels` and
`realised_rise_ramp_typed_voxels`, with the slot count behind the second — so the two measurements
cannot be confused by name again.

### ⚠️ Both predictions, and one imprecision in how I wrote one of them

* **"The rule should select a late epoch."** ✅ **Epoch 21 of 40**, against #129's epoch 2 of 40. The
  `missing`-for-`extra` trade no longer pins the rule to the start of training.
* **"`q0.25` alone will not pass."** ✅ `class129_at_q025` is `NOT MET` — planar 0.00, `extra` 0.1217.
  ⚠️ But the collapse figure I quoted in the pre-registration (0.2409) was #129's **endpoint** row
  from its decode table, and the isolating run above uses the **arm of record**, where `q0.25` gives
  **0.0438**. The prediction's conclusion holds; the number I attached to it was for a different
  checkpoint, and writing "on #129's weights" without saying *which* was sloppy.

### 🔑🔑 So why does it still fail? The two fixes fight each other on the geometry

Each change is right on its own axis and they pull opposite ways on the population:

* the assignment change buys a **second region** — and every extra region is another place to
  over-carve. Dominant-slot accuracy falls **0.8251 → 0.2677**, and that is precisely the collapse
  rate going **0.1022 → 0.2579**. The arm carves regions that should not have been carved.
* the pitch change reduces the over-carve, and on #129's single-region weights it is enough
  (collapse 0.0438). Against 2.03 regions it is not.

🔑 **And the extra slots are mostly flat.** The `Ramp`-typed share of used slots falls 0.522 →
**0.390**, so `planar` reaches only 0.12. The prior adjustment was applied to the **assignment**
head; the **type** head was left alone, and it answers `Layer` for most of the small new regions.
More slots did not become more planes.

⚠️ **`used_slots_compiling_flat` (0.592) is NOT evidence for that**, and quoting it as such was
wrong: a `Layer` is flat *by definition*, so that figure falls out of the type mix rather than
testing it, and it rises automatically whenever an arm uses more slots. The load-bearing number is
the `Ramp`-typed share above. It is now reported with that warning attached to it in
`report_program_diagnostics`.

⚠️ The head is also **more diffuse than #129's**, not less: confidence 0.431 → 0.341, normalised
entropy 0.799 → 0.885, overall per-column accuracy 0.4245 → 0.2739. The adjustment bought coverage
of the rare classes by spreading the posterior, which is the known cost of logit adjustment and is
the mechanism behind the dominant-slot loss above.

### Traps

⚠️ **I walked into the `pgrep` self-match this ticket lists.** `until ! kill -0 $(pgrep -f "tag
heightmap_program_adj" ...)` matches the waiter's own command line and can never exit. Poll a file
for a pattern the waiter does not itself contain. ⚠️ And a related one, new: `grep -q "artifact"`
also matches the string `execution/artifacts/...` in a path the run echoes, so the waiter fired
early — anchor it (`^\[artifact\]`).


## What this settles, and what it does not

**Settles:**
* 🔑🔑 **The K = 1 collapse is a LOSS problem, not a decode problem, and the diagnosis said so before
  the run.** The posterior is diffuse rather than confidently wrong, but the post-hoc prior
  correction destroys the dominant slot (0.8251 → 0.1275) while the same correction applied in the
  loss reaches the same minor-slot recall (0.28) under a plain argmax. **Apply an imbalance
  correction where it can change what is learned, not where it can only reshuffle.**
* ✅ **A trained arm here uses more than one region for the first time** — 2.03 slots, one-slot rate
  0.895 → 0.324 — and draws a visible two-plane gable.
* ✅ **The pitch asymmetry is real and the lower-quantile read prices it correctly**: on #129's own
  weights, `extra` 0.1507 → 0.1217 and collapse 0.1022 → 0.0438 for nothing but a read.

**Does not settle:**
* ⚠️ **The arm does not pass; this is the third KILL on this ticket.** planar 0.12, `extra` 0.0832,
  collapse 0.2579.
* ⚠️ **The type head was not touched and is now the binding constraint.** 61% of used slots are typed
  `Layer` and 59.2% compile flat, so more regions bought fewer planes each. A prior adjustment on the
  **type** head is the obvious next move and was deliberately not made here — one change per head was
  the whole design of this arm.
* ⚠️ **Nothing here prices a per-slot over-carve guard.** The collapse comes from carving regions
  that should not be carved, and neither change addresses whether a slot should fire at all.
* The selection rule survived a run it was predicted to survive, so it stays unchanged — but it has
  now never selected an epoch whose `planar` clears the KILL.
