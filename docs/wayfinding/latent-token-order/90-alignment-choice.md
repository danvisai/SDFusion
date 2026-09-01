<!-- RECOVERED FROM THE ISSUE TRACKER, 2026-08-14. -->

> **Recovered and re-landed.** The original asset was lost with commit `26e19e8`; this file preserves
> its tracker record. The matcher, probes, tests, and regenerated artifacts were reconstructed in
> `c542ef2`, and the chosen greedy k=256 alignment was exercised at corpus scale for #91 in
> `8101143`. See `RECOVERY.md` for the recovery audit and reproduced measurements.


# #90 — Choose the token alignment: canonical sort, or explicit matching

*State: closed · opened 2026-08-09 · implementation re-landed 2026-08-14*


## Ticket

Part of #87

## Question

Which ordering goes into the cache — and what does forcing it to be a **permutation** cost?

## What is already measured (#89, corrected by #88)

Element-wise cosine of the pair, n=5, **unconstrained** nearest-neighbour matching:

| ordering | cosine to `z` |
|---|---|
| as encoded | +0.0480 |
| randomly permuted | +0.0347 |
| **position-matched (NN)** | **+0.7288** |

⚠️ **That 0.7288 is an upper bound, not a deliverable.** `cdist(...).argmin` is **many-to-one**: several
envelope tokens can claim the same real token and some real tokens get none. You cannot reorder a cache
with a mapping — only with a bijection. How much alignment survives that constraint is unmeasured, and
it is the number this ticket actually needs.

⚠️ The gain has read **54.6x / 16.8x / 51.2x** across three runs of this control. Direction stable,
magnitude not. n=5 is too small to choose on.

## What to measure

On **at least 100 buildings**, stratified by region (row order tracks source corpus — the trap that
void-ed three figures on #69), report the element-wise latent cosine for:

1. **as encoded** — the floor;
2. **random** — the no-information control;
3. **Morton sort** by query position — its 40.4% figure is a *position* statistic with **no latent-space
   equivalent yet measured**;
4. **nearest-neighbour** matching — the unconstrained upper bound;
5. **Hungarian / optimal assignment** — the bijection that can actually be written to a cache;
6. **greedy or Sinkhorn** if Hungarian is too slow at 2048x2048 across 35,623 rows.

Plus, for the bijective options:

- **unmatched mass**: the envelope and the real building are *different surfaces* — a box has no gable —
  so some tokens have no honest partner. What fraction, and how far away is the partner they are forced
  onto?
- **stability**: same building, two encodes. Does the matching agree?
- **cost**: seconds per building, x 35,623, x 2 caches.

## The tie-breaker the numbers will not show

The sort has a property matching lacks: it is defined on **one** latent, so it also canonicalises the
*plain* (non-pair) steps and inference, where there is no partner to match against. Matching only works
where a partner exists. If the sort gets within reach of the bijective match, that generality may be
worth more than the gap.

## Judged on

A decision with the table behind it, the rejected options recorded with their numbers, and the chosen
method's cost stated so #91 can size its pass.


---

## Comment — danvisai, 2026-08-10

## Decision: **greedy matching, k=256** — and "optimal" turned out to optimise the wrong thing

Module `models/token_alignment.py`, probe `scripts/foundations/probe_token_alignment.py`, artifact
`execution/artifacts/token_alignment_probe.json`, writeup `90-alignment-choice.md` (`26e19e8`).
**n=102 held-out buildings, 34 per region**, via the harness's own `pick_ids`.

| ordering | cosine | unmatched | forced partner | true partner | permutation? | ms | corpus |
|---|---|---|---|---|---|---|---|
| as encoded | 0.0405 | 99.9% | 1.1358 | 0.0166 | — | 0 | — |
| random | 0.0309 | 100.0% | 1.1407 | 0.0109 | — | 0 | — |
| morton | 0.2112 | 99.1% | 0.4912 | 0.0212 | yes | 0.7 | 23 s |
| **nn** (upper bound) | **0.7079** | 43.5% | 0.0560 | 0.0203 | **NO** | 254 | — |
| **greedy k=256** | **0.5387** | **63.3%** | 0.2783 | 0.0186 | **yes** | 499 | **4.93 h** |
| hungarian | 0.5106 | 78.8% | **0.1734** | 0.0183 | yes | 976 | 9.66 h |

Greedy wins on cosine, cost, and unmatched mass, and ties on stability (|diff| 0.0053 vs 0.0051 across
two independent encodes, n=15).

### 🔑 Why the optimal assignment loses

Hungarian **does** win total position error — it is optimal for exactly that. But total distance was
never the objective, only the proxy for *which token corresponds to which*. Minimising the **sum**
degrades a token that had an excellent partner in order to rescue one that has none.

The far/near split shows it directly: **greedy genuinely matches 36.7% of tokens** (within a voxel, mean
0.0186) against Hungarian's **21.2%** — 73% more real correspondences — paying with a worse
forced-partner distance. Hungarian buys a better *average* by making the hopeless tokens look less bad,
and it costs the pairs that carried signal.

**Optimising the proxy harder made the real objective worse.**

### ⚠️ The bijection costs about a quarter of the headline

#89's **0.73** used `cdist(...).argmin` — many-to-one, so it cannot reorder a cache. A real permutation
reaches **0.5387**. Still **13× the 0.0405** training sees today, but **#92 should expect the smaller
number**. `is_permutation` is now the gate; `test_nearest_neighbour_is_NOT_a_permutation` pins it.

### ⚠️ Unmatched mass, and why it is not a defect

63.3% of tokens land beyond one voxel, at mean **0.2783**, against **0.0186** for the 36.7% that find a
partner — an order of magnitude apart, which is what makes the threshold meaningful. Even the
unconstrained upper bound leaves **43.5%** unmatched. The envelope and the real building are genuinely
different surfaces; a box has no gable. **This is why the ceiling is 0.71 and not 1.0.**

### Morton rejected, with numbers

**0.2112**, the least stable of the three (0.0147), and its one argument — that a canonical order also
applies at inference — was measured by #95 at **−0.0150**, inside the noise. Both arguments gone.

### ⚠️ How I nearly chose wrong, twice, the same way

I measured greedy *unrestricted* (0.5596) and reported it beating Hungarian; then added a k-NN
restriction for speed, re-ran, got **0.4881 — below Hungarian** — and reported the reversal. Both
readings were real; **the algorithm changed between them and I compared across the change.**

| k | cosine | corpus |
|---|---|---|
| 16 | 0.4783 | 2.99 h |
| 64 | 0.5141 | 3.34 h |
| **256** | **0.5329** | 4.95 h |
| 2048 | 0.5392 | 19.70 h |

Default now sits at the knee, sweep in the docstring, direction pinned by
`test_more_candidates_never_hurt_greedy`.

### Review

Both axes ran. **Spec** found three gaps, all fixed: *"× 2 caches"* silently dropped (answer: it is
**one** — only the envelope side is permuted, onto a shared real side); *"how far is the forced
partner"* answered with an all-token mean that diluted the subset asked about; and morton's corpus cost
read 6 s against the artifact's 23.1 s. **Standards** found the `record()` closure's `idx`/`order_ref`
duality undocumented. 143 tests green.

### For #91

Use `greedy_match(candidates=256)`; budget **~4.9 h** for alignment on top of the encode, **one** pass.
Expect cosine ≈ **0.54** and ~63% unmatched — if the rebuilt cache lands far from 0.54, suspect the
rebuild, not the method.


---

## Comment — danvisai, 2026-08-12

Reopening: this ticket's resolution was implemented on a cloud A100 instance that was lost before the resolving commits (cited in the comment above) were pushed. None of those commits exist on any branch or PR in this repo — verified against `git log --all` and `git ls-remote`. The written analysis and decisions above are intact and should be treated as the spec; the implementation, and #91's rebuilt caches specifically, need to be redone from scratch.
