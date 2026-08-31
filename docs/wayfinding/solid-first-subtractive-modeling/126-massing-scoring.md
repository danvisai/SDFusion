# #126 — How generated massing is scored

*2026-08-28. CPU only, no GPU touched, no training. The four #92 arms kept the A100 throughout.*

Answers [#126](https://github.com/danvisai/SDFusion/issues/126), which asked whether the paired
massing metric has its optimum at the footprint envelope rather than at a real building, and which
[#127](https://github.com/danvisai/SDFusion/issues/127) is explicitly blocked on.

Code `scripts/foundations/measure_scoring_optimum.py`, contract tests
`scripts/foundations/test_measure_scoring_optimum.py`, artifact
`execution/artifacts/scoring_optimum_714.json`. Scored on the same pre-registered 714 held-out
buildings as [#10](https://github.com/danvisai/SDFusion/issues/10).

---

## Why this needed measuring again rather than deciding from the ticket

#126's whole case rests on one number — *"250 pairs of real held-out buildings … score a median 3D
IoU of 0.674 against each other"* — which was computed ad-hoc and **never committed**. Nothing in
the repository could reproduce it, check it, or re-run it under a different filter. Since the
ticket proposed to retire the project's primary massing metric on the strength of it, the first
task was to make it reproducible.

Two guards say the reproduction is on the right footing, and both are cross-checks against numbers
computed by *other* code on other days:

| guard | expected | measured |
|---|---|---|
| blockout 3D IoU on the 411 carve-needing buildings | 0.8125 (#10) | **0.8125** |
| height-field reconstruction residual over 714 | 71 voxels (#10's overhang count) | **71** |

## 🔑 The premise is confirmed. The conclusion drawn from it is not.

Both halves matter and neither should be quoted alone.

**Confirmed — massing is *not* determined by footprint + height.** That sentence is `CONTEXT.md`'s
stated justification for paired scoring. Two real held-out buildings whose footprints agree to
IoU ≥ 0.90 and whose heights agree within 5% still differ by a median 3D IoU of **0.886** over all
matched pairs (**0.829** on the carve-needing subset) even when one is re-rendered on the other's
exact footprint at the other's exact height. The conditioning
leaves real architectural freedom, and the held-out row is one valid answer among several.

**Not confirmed — "doing nothing beats generating well".** Measured like-for-like, it does not.

## The ladder, and why the ticket's arm was not a fair one

"A plausible alternative real building" is a stand-in for a generator's output, so it is only
evidence if it is offered the way a generator would have to offer it. Each rung removes one thing a
footprint-conditioned generator is *not free to get wrong*, so the gaps attribute the number to a
cause instead of leaving it as one aggregate:

| rung | what it is | what it is still charged for |
|---|---|---|
| `alt_raw` | building *b* exactly as it sits in the corpus | base placement + footprint + roof |
| `alt_aligned` | *b* moved to *a*'s base level | footprint + roof |
| `alt_exact` | *b*'s roof profile on *a*'s footprint at *a*'s height | **roof shape only** |

Base placement is not architecture and the ticket's filter never constrained it: the pinned
buildings sit at **28 distinct `y0`**, and only 58 of 128 matched offers already agree on it.

### Results — the 72 carve-needing offers, all arms on the same population

| arm | missing | `extra` | vs_input | collapse | beats env `extra` | *(3D IoU)* | beats env IoU |
|---|---|---|---|---|---|---|---|
| blockout (do nothing) | 0.0000 | 0.2055 | 1.0000 | 0.0000 | — | *0.8295* | — |
| `alt_raw` | 0.0767 | 0.1461 | 0.8044 | 0.2917 | 70.8% | *0.7738* | 19.4% |
| `alt_aligned` | 0.0751 | 0.1417 | 0.8179 | 0.2639 | 73.6% | *0.7807* | 26.4% |
| **`alt_exact`** | 0.0400 | **0.0974** | 0.8446 | 0.1667 | **100.0%** | *0.8295* | **60.0%** |

Both "beats env" columns **exclude ties** and the counts are in the artifact. Ties are not losses
and pooling them is how the aggregate got its reputation here: on 17 of the 72 offers the
alternative's roof simply **is** the envelope — the real building genuinely has a flat top at full
height — and folding those into the denominator turns 33 wins against 22 losses (60%) into a 46%
"coin flip". On `extra` the alternative wins **55 of 55 decided offers**.

🔑 A plausible real building, offered footprint-exact, **halves the envelope's surplus** (`extra`
0.0974 against 0.2055, winning every decided offer) while landing on **the same median 3D IoU**
(0.82948 against 0.82951 — a coincidence at this population size, not a law).

So the defect #126 found is real but it is **not** a defect of paired scoring. It is a defect of the
**aggregate**: 3D IoU ranks a real building and the envelope as indistinguishable on the median,
while the `missing`/`extra` split separates them unanimously on the same rows. `CONTEXT.md` already
says this about the aggregate in general — *"The aggregate alone is unreadable"* — and this is that
sentence firing on the metric the map has been ranking on.

⚠️ **The alternative is not strictly better, and the collapse rate is why.** It collapses on
**16.7%** of offers by #80's definition (`missing` ≥ 15%) where the envelope never can. A real
building is a *better-shaped* answer than the envelope, not a *safer* one, and any bar built on this
has to carry the collapse rate beside it.

### The 0.674 did not reproduce, and the sensitivity says why

Under the filter #126 states, the pinned 714 admit **64 unordered pairs**, not 250. Loosening the
footprint threshold both admits more pairs and drives `alt_raw` down, which is the shape of a
looser filter than the one written down:

| footprint threshold | unordered pairs | `alt_raw` `extra` | `alt_exact` `extra` | `alt_raw` IoU | `alt_exact` IoU |
|---|---|---|---|---|---|
| 0.80 | 518 | 0.1909 | 0.1031 | 0.7124 | 0.8295 |
| 0.85 | 190 | 0.1634 | 0.1071 | 0.7478 | 0.8295 |
| **0.90 (as stated)** | **64** | **0.1461** | **0.0974** | **0.7738** | **0.8295** |
| 0.95 | 12 | 0.0897 | 0.0703 | 0.7651 | 0.7923 |

Carve-needing rows; the full sweep, both populations and every arm, is in the artifact under
`corpus.threshold_sweep`.

⚠️ The ticket's 250 sits between the 0.85 and 0.80 rows, so its number was most likely measured at a
looser footprint match than the 0.90 it reports — across 0.90 → 0.85 → 0.80 `alt_raw` falls
monotonically (0.7738 → 0.7478 → 0.7124), which is the right direction for the ticket's lower
number. Recorded as **not reproduced**, not as *wrong*: the original computation is gone, so this is
inference from the sweep, not a demonstration.

🔑 `alt_exact` holds at **0.8295** across 0.80–0.90 while `alt_raw` moves by 0.061 over the same
range. The rung that models a footprint-conditioned generator is insensitive to how the population
is drawn, which is what makes it safe to set a bar against; `alt_raw` is not, which is what made the
ticket's number fragile. ⚠️ The 0.95 row is out of pattern for **both** arms (0.7651 and 0.7923) on
**12 pairs** — too few to read as anything, and reported rather than dropped.

## The decision

1. **Keep paired scoring.** Its stated justification is falsified and is corrected below, but it
   survives on the **C1 transform** reading, which is the one the thesis actually needs: the
   question is whether *this* input — a specific blockout or a user sculpt — was projected
   correctly, and that is well-posed however many valid buildings share the footprint.
2. **Demote aggregate 3D IoU to a diagnostic** in new work. Its **median cannot rank** a real
   building above the envelope (0.82948 against 0.82951), so a threshold on the median cannot mean
   what a threshold is for. ⚠️ Stated precisely, because the first draft overstated it: the
   aggregate is not blind — offer by offer it prefers the real building on **60%** of decided
   comparisons. It is a weak discriminator, not a broken one, and the honest reason to demote it is
   that the split is a **unanimous** one (100%) on the same rows.
3. **The `missing` / `extra` split is the headline**, with `vs_input` and the **collapse rate**
   beside it — the alternative's 16.7% is exactly why the split alone is not enough either. This is
   not a new metric: it is the one this project has rediscovered repeatedly and kept failing to
   promote.
4. **Score on the carve-needing subset** wherever carving is the question, never pooled. 303 of 714
   need no carve at all, and a 42% no-op majority flatters every aggregate (#10, and #80's bimodal
   result before it). ⚠️ Where the question is *regression* rather than carving, the full 714 stays
   correct — that is what #92's floor is measured on, and mixing the two is what the withdrawn
   first draft got wrong.

⚠️ Points 2 and 3 bind **new** work. They do not retroactively re-score #92, whose gates were
pre-registered and are discussed below.

**Rejected: scoring massing distributionally** (#126's second option). It measures whether the
*population* looks right, and C1's claim is about a *specific* input's projection, so it cannot
express the thing the thesis has to defend. It stays right for detail, where `CONTEXT.md` already
prescribes it.

**Rejected: scoring against the best of the plausible alternatives** (#126's third option). It is
the option this data most supports — the alternatives are genuinely valid answers — and it is
rejected on cost and on gaming, not on principle:

* It is **undefined for 618 of the 714** held-out buildings. Only 96 enter a matched pair at all, so
  for 87% of the corpus "the best alternative" is the held-out row itself and the metric silently
  reverts to what it already was.
* **Scoring against a best-of-K set is monotone in K**, so the number moves when the corpus grows,
  and it cannot be compared across the arms already on this project's record.
* The thing it was meant to fix — that a good building is punished for not being the held-out row —
  is **already fixed** by reporting `extra` instead of the aggregate, at no cost: the alternative
  wins 55 of 55 decided offers on `extra` while tying on IoU. The cheaper repair works, so the
  expensive one is not needed.

⚠️ Reopen this if a generator is ever built whose failure is *plausible-but-different* rather than
*over-filling*. On today's evidence that failure mode does not exist — every arm on record fails by
surplus.

## What happens to the pre-registered bars

⚠️ **A first draft of this document withdrew #92's IoU clause, and that was wrong.** It argued the
bar was unreachable by citing 0.8295 on the carve-needing offers — while the bar is defined on the
**full 714**. That is a population mismatch, the exact error this measurement exists to catch and
the one `measure_scoring_optimum.py`'s own docstring warns about. Corrected below and left on the
record rather than quietly fixed.

**[#92](https://github.com/danvisai/SDFusion/issues/92) — the bar stands. Two cautions on reading it.**
`map-87.md` states the clause as *"median 3D IoU ≥ 0.876 — **no quality paid for it** (today:
**0.876**)"*. It is a **no-regression floor on the full 714**, not a quality target: a no-op passing
it is by design, because the clauses that make the gate discriminate are `vs_input < 0.98` and
beats-envelope > 5%. Nothing measured here touches those two.

1. ⚠️ **The headroom is thin.** A plausible real building scores **0.8858** on the matched pairs —
   0.010 above the floor. A generator that finally starts carving moves *toward* a real building and
   therefore *toward* that floor, so the guard can fire on a model doing the right thing. It should
   be read as "did quality collapse", never as "is this good".
2. ⚠️ **It must not be re-read on the carve-needing subset**, where a real building scores 0.8295 —
   below the floor. The floor's calibration belongs to the population it was set on.

**Tension with `map-87.md` gate 4, flagged and not overridden.** That gate reads *"3D IoU split into
missing vs extra — **diagnostic only, never pass/fail**"*, which is the reverse of this decision's
point 3. Map #87 fixed its gates *before* the run precisely so results could not re-litigate them,
and #10's own record shows this project stopping at a dip and being wrong twice. So **#92 is judged
on the gates it pre-registered**, unchanged; this decision binds **new** work (#118, #127) and the
amendment to gate 4 is the human's to make, not this ticket's.

**[#118](https://github.com/danvisai/SDFusion/issues/118) — unblocked, with two constraints.** It is
pre-registering gates on *paired improvement*, *beats-envelope* and *vs input*; all three survive.
It may not express paired improvement as an aggregate-IoU threshold, and its beats-envelope gate
must name **which metric and how ties are counted** — on the same 72 rows the answer is 60% (IoU) or
100% (`extra`) excluding ties, and 46% or 76% with ties pooled into the denominator. A gate that
does not say which of those four numbers it means is not pre-registered.

**[#127](https://github.com/danvisai/SDFusion/issues/127) — unblocked.** Its bar is `extra` on the
carve-needing subset against the 1-NN baseline, which is already the form this decision requires.
🔑 It now also has a **reference point**: a real building offered footprint-exact reaches `extra`
**0.0974** with a **16.7%** collapse rate. A height-map generator that reaches ≈0.10 *without*
collapsing has beaten a real building's disagreement with another real building — and unlike the
alternative it cannot collapse, since a clamped height map is solid by construction.

## Limits, stated

- **64 pairs is a small population**, and 96 of 714 buildings enter one. The quartiles are published
  beside every median in the artifact for that reason. The threshold sweep (up to 518 pairs) is what
  carries the robustness claim, not the n=64 row on its own.
- **`alt_exact` is a constructed arm, not a generator.** It borrows a real roof and rescales it; no
  model produced it, and it says nothing about learnability — the same limit #10 recorded.
- The transplant fills destination cells the source footprint misses from the **nearest** source
  column. On pairs matched at IoU ≥ 0.90 that is a small correction, but it is a choice, and at the
  0.80 threshold it is doing more work.
- **`missing` is not free for `alt_exact`** (0.0400): unlike a program fit, a transplanted roof can
  cut into GT. It is reported and not clamped away.
- This measures **the corpus**, which #10 established is a height field at 64³. A metric conclusion
  drawn here transfers to volumes only as far as that does.
