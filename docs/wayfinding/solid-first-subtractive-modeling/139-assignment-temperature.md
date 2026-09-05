# #139 — Is the assignment head's imbalance correction calibrated against the wrong skew?

*Effort: solid-first semantic architectural carving. Opened 2026-09-02 from [#138](138-type-head-imbalance.md),
whose own "what follows" named the assignment head's diffuseness as the thing a type-temperature
sweep cannot fix. Run and written 2026-09-02. One A100, ~20 minutes of training per arm, two arms.*

> ⚠️ **Renumbered on filing.** This write-up was authored as `138-assignment-temperature.md` against
> an assumed issue number that was never created; 138 went to the type-head arm when both tickets
> were finally filed on 2026-09-02, so this one is
> [#139](https://github.com/danvisai/SDFusion/issues/139) and the file was renamed to match. ⚠️ The
> pre-registration quoted below lives in the module docstring and was **uncommitted** at filing
> time, so it is not independently verifiable in git history.

> ⚠️⚠️ **Second correction, found after the first run and before this write-up was finished being
> read.** The first run, `heightmap_program_assign_tau05`, was NOT isolated as its own pre-
> registration and this doc's first draft both claimed. #138's implementation made the type-head
> correction (`type_prior`) apply **unconditionally** to every `--objective program` run, with no
> flag to disable it -- so the "assignment-only" arm silently inherited #138's type fix too, and
> every number in the first draft described the *combined* arm, mislabelled as isolated. Caught by
> checking the saved checkpoint's own metadata (`type_prior` was present) before trusting the
> write-up's own claim of separability -- the same discipline #126 exists to enforce, applied to
> code instead of to a scorecard. A `--no_type_prior` flag now exists so this cannot recur silently,
> and a true isolated arm, `heightmap_program_assign_tau05_only`, was retrained. **Every number
> below is from the corrected, actually-isolated run.** The mislabelled run is not wasted: it is the
> combined arm, and it is written up on its own terms in
> [#140](140-combined-assignment-and-type.md).

> #132's own write-up disclosed the cost of its assignment fix without asking whether the cost was
> load-bearing: dominant-slot accuracy fell 0.8251 → 0.2677 and confidence fell 0.43 → 0.34 to buy
> minor-slot recall 0.0000 → 0.28. Is `ASSIGN_TEMPERATURE` = 1.0 -- the full logit-adjustment,
> targeting a perfectly uniform decision boundary -- doing more correction than this corpus's
> imbalance actually calls for?

Code `scripts/foundations/train_height_map_generator.py` (`ASSIGN_TEMPERATURE` 1.0 → 0.5; a new
`--no_type_prior` flag so this run and #140 could be told apart), contract tests
`scripts/foundations/test_train_height_map_generator.py`, artifacts
`execution/artifacts/height_map_generator_assign_tau05_only_train.json` and
`..._assign_tau05_only_714_diagnostics.json`. No montage this run.


## 🔑🔑 The prior #132 corrected against was inflated -- and isolated, halving it recovers the dominant slot almost completely

**The free measurement, before any run.** `assignment_prior`'s pooled `slot0:slot3` skew is 11.9x —
the number `ASSIGN_TEMPERATURE` = 1.0 is calibrated against. Recomputing the identical prior
restricted to buildings whose *label* actually uses more slots:

| population | n (train) | slot0:slot3 skew |
|---|---|---|
| all training rows (#132's population) | 34,909 | 11.9x |
| label uses ≥2 slots | 16,779 | 9.1x |
| label uses ≥3 slots | 13,223 | 6.4x |
| label uses exactly 4 slots | 10,172 | 4.7x *(matches #130's own number)* |

🔑 The pooled figure is not a measurement of how rare a real slot 3 is. It is inflated by 1/2/3-slot
buildings, which always own slot 0 by area-canonicalisation and structurally never reach slot 3 at
all — their absence from slot 3 is evidence of low complexity, not of slot 3 being 11.9x rarer.

⚠️ **And the standard logit-adjustment recipe targets a uniform test-time balance**, which is the
right target when training and deployment distributions differ. They do not differ here — the
pinned 714 come from the same distribution as training — so `tau=1.0` asks the head to hit a
balance the corpus itself does not have.

**The change:** `ASSIGN_TEMPERATURE` 1.0 → 0.5, the untuned midpoint between no correction (0.0)
and #132's full one (1.0). One value, chosen before the run, not swept. Nothing else moved: same
`plane_head class`, same pitch `q0.25` — and, corrected, `--no_type_prior` so the type head really
is #132's, untouched, this time.


## Result, on the same 411 carve-needing buildings

| arm | `missing` | `extra` | `vs_input` | collapse | *(3D IoU)* |
|---|---|---|---|---|---|
| `heightmap_program_adj` *(#132, tau=1.0)* | 0.0659 | **0.0832** | 0.8470 | 0.2579 | *0.8080* |
| **`heightmap_program_assign_tau05_only` *(#139, tau=0.5, isolated)*** | 0.0926 | **0.0772** | 0.8039 | 0.2603 | **0.8131** |
| 1-NN retrieval *(the guard)* | 0.0257 | 0.1031 | 0.8743 | 0.1582 | *0.8355* |

Two of five numbers moved the direction the mechanism predicts; two moved the other way; one is
flat. This is a **narrower, more mixed** result than the mislabelled first draft reported:

    beats_1nn_extra            0.0772 vs 0.1031   ✔  (the only arm in this chain to clear it in isolation)
    collapse_no_worse_than_1nn 0.2603 vs 0.1582   ✘  (essentially unchanged from #132's 0.2579)
    moved (vs_input < 0.98)                       ✔

**Verdict: NOT MET** (fails the collapse `GUARD`), but `extra` alone clears the 1-NN bar for the
first time in this chain — the ONE number the original pre-registered bar (#6, `f1f0dcd`) actually
asked for.


## 🔑 Why: the dominant slot comes back, and it is not a wash -- this part of the story survives

`assignment_collapse`, same population, `heightmap_program_adj` (tau=1.0) vs
`heightmap_program_assign_tau05_only` (tau=0.5, isolated):

| | tau=1.0 *(#132)* | tau=0.5 *(#139, isolated)* | #129, pre-#132 *(reference)* |
|---|---|---|---|
| confidence (median max p) | 0.341 | 0.317 | 0.43 |
| overall per-column accuracy (argmax) | 0.2739 | **0.4597** | 0.4245 |
| **dominant-slot accuracy** | **0.2677** | **0.8134** | 0.8251 |
| slots seen (argmax) | 2.03 | 1.77 | 1.19 |
| recall on non-dominant-slot columns | 0.2801 | 0.0727 | 0.0000 |

🔑🔑 **Dominant-slot accuracy recovers almost completely** (0.2677 → 0.8134, against a pre-#132
baseline of 0.8251 — a gap of 0.012, not the 0.56 #132's correction opened) — **and this is the real,
isolated size of that recovery.** The mislabelled first draft measured 0.7560 here, because the
type fix riding along was itself changing what the assignment head saw during training.

⚠️ **Minor-slot recall gives back more than the first draft showed.** 0.2801 → 0.0727 in isolation,
against the mislabelled draft's 0.1101 — the type fix was propping part of that number up too.
**Fewer slots are used overall** (1.77, down from #132's 2.03, not up to 2.28) — a lower tau makes
the assignment head *more* content to answer "slot 0" alone, not less, when nothing else is pulling
it towards using more slots.

⚠️ **Confidence went DOWN, not up** (0.341 → 0.317), while dominant accuracy nearly tripled. The two
still are not the same measurement (confidence is how peaked the posterior is regardless of which
class wins), and the direction here is a reminder that "the network got more certain" was never the
claim -- "the certainty it already had stopped attaching to the wrong class" is, and that is what
the accuracy numbers show directly.


## The "form improved for free" finding is not just intact -- it is much bigger than first measured

Nothing in this run touches `program_loss`'s `type` term, and this time that is actually true.

| | tau=1.0 *(#132)* | tau=0.5 *(#139, isolated)* |
|---|---|---|
| `planar_fraction` (ALL, 411) | 0.12 | **0.67** |
| `dl_ops` (ALL, 411) | 1.0 | 1.0 |
| slots used (ALL, 411) | 2.03 | **1.77** |
| arm uses exactly one slot | — | **0.377** |

🔑🔑 **0.67 is the highest `planar_fraction` measured on ANY arm on this map's record — higher than
the real building's own 0.50.** The mislabelled draft reported 0.33 for this effect and credited it
to the assignment change; the true isolated number is roughly double that.

⚠️⚠️ **And it is not the win it looks like read alone, which is exactly #126's and #132's own warning
about `dl_ops`/`planar_fraction` arriving a third time.** This arm uses FEWER slots (1.77) than
#132's already-too-few 2.03, against a label average of 3.06, and answers "just slot 0" on 37.7% of
buildings. A single, confidently-typed dominant plane scores very well on `planar_fraction` while
representing *less* of a building's real structure than #132's more-fragmented, worse-typed attempt.
High planar fraction from low complexity is not the same claim as high planar fraction from
correctly resolving a complex roof, and this table cannot tell the two apart on its own — the same
sentence #132 already had to write about `dl_ops` reading 1.0 for the wrong reason twice.

**Consistent with that reading:** the type head, still untouched and unadjusted, gets slot 3's Ramp
recall to exactly **0.000** in this run (measured via `type_collapse`) — identical to the
uncorrected behaviour #138 diagnosed and fixed. A lower assignment tau does not rescue slot 3's
calibration; it mostly stops asking slot 3 to answer at all.


## What this settles, and what it does not

**Settles:**
* 🔑🔑 **The assignment correction's cost was substantially a calibration artifact.** In true
  isolation, dominant-slot accuracy recovers to within 0.012 of #129's uncorrected number (0.8134 vs
  0.8251), and `extra` clears the 1-NN bar for the first time in this chain (0.0772 vs 0.1031).
* 🔑 **The `planar_fraction` gain from a better-calibrated assignment head is real and larger than
  first measured (0.67, not 0.33)** — but it now reads as a low-complexity artifact (fewer slots
  used, not better-typed ones) rather than as evidence the two heads' errors are simply coupled.
  That hypothesis is neither confirmed nor refuted by this arm alone; #140 tests it directly.
* **Isolating variables by reading a checkpoint's own saved config, not by trusting a run's
  pre-registered intent, caught a real confound before it shipped in the write-up.**

**Does not settle:**
* ⚠️ **The arm does not pass** — `missing` (0.0926) and collapse (0.2603) are barely different from
  or worse than #132's, and the collapse `GUARD` still fails by a wide margin.
* ⚠️ **Minor-slot recall (0.0727) is closer to #129's zero than to #132's 0.28.** Whatever the
  correction still buys on the columns that most need it, it buys less of it at tau=0.5 than at
  tau=1.0 — the trade this ticket predicted, now sized correctly.
* **Whether combining this with #138's type fix compounds favourably is #140's question, not this
  one's** — and the fact that this isolated run's `planar_fraction` (0.67) beats #140's combined
  number (0.33) is itself evidence the two do not simply add.
* ⚠️ **0.5 is one untested point on a continuum, not a located optimum,** for the same reason it
  was in the mislabelled draft.


## Pinned

`scripts/foundations/test_train_height_map_generator.py` — 167 tests.
`TestLogitAdjustedAssignmentLoss.test_the_adjustment_is_pre_registered_at_one_half` replaces the
prior `..._at_one` test. `TestAssignmentDecode.test_and_the_half_that_did_not_it_loses_the_dominant_
slot` now pins its decode-side refutation at an explicit `temperature=1.0`, so halving the
training-side default could not silently weaken a historical refutation it was never about.
`--no_type_prior` has no dedicated unit test of its own yet — it is exercised by this ticket's run,
not by the suite; that is a gap, not a design choice.


## What follows

- **The question #139 asked is answered, on the corrected run: yes, `tau=1.0` was over-correcting**,
  and roughly recovers the assignment head's dominant-slot calibration in isolation.
- 🔑 **The `planar_fraction` finding needs the same caution #6/#129/#132 already learned about
  `dl_ops` in isolation: high and low-complexity are not the same as high and correct.** #140's
  combined arm is the test of whether pairing this with #138's fix produces a *complex, correct*
  roof rather than a *simple, lucky* one.
- **`ASSIGN_TEMPERATURE` = 0.5 is now the value future arms in this chain inherit**, the same way
  #132 changed `PLANE_DECODE`'s pitch component once and it stuck.
- **`--no_type_prior` exists now specifically so this mistake has a name and cannot repeat silently.**
  Any future arm claiming isolation from #138's fix should be checked against its own saved
  `type_prior` key, not against the command that was intended to produce it.

See [132-overcarve-and-assignment.md](132-overcarve-and-assignment.md),
[138-type-head-imbalance.md](138-type-head-imbalance.md),
[140-combined-assignment-and-type.md](140-combined-assignment-and-type.md),
[130-baselines-diffusion-curriculum.md](130-baselines-diffusion-curriculum.md),
[6-program-generator.md](6-program-generator.md).
