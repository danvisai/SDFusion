<!-- Mirrored from the tracker, 2026-08-14. -->

> **Open ticket, mirrored locally** so this effort can be read without the
> tracker. Nothing was lost for this one — it had no committed asset.

> **Resolved, 2026-09-01.** One candidate is refuted with a direct gradient measurement; the other
> two are narrowed but not separated. See the Resolution section below.


# #94 — Why does weighting up the surface term collapse the model, if the term is the cleaner signal?

*State: resolved 2026-09-01 · opened 2026-08-09*


## Ticket

Part of #87

## Question

[#84](https://github.com/danvisai/SDFusion/issues/84) explained the collapse under increased
decoded-surface pressure as the surface term overshooting: solidity fell **89.9% → 57.6% → 19.2%** as
the term pressed harder, and the reading was "the pressure is the mechanism".

[#89](https://github.com/danvisai/SDFusion/issues/89) complicates that. It measured the surface term as
**order-insensitive at every t** (varies 0.0001–0.0005% against the epsilon loss's 0.31–1.53%, a
1,900–5,300× gap), which supports reading it as the *cleaner* of the two signals. A cleaner signal
collapsing the model when weighted up is not explained by "too much pressure on a good signal".

So: **what actually collapses the model when the surface term is weighted up?**

## Candidates, none tested

- The term is clean but its *gradient path* is not — it reaches the latent through a frozen 191.6M
  decoder, and #73 measured that decoder as intolerant to leaving its manifold. A clean objective can
  still push along a direction the decoder cannot absorb.
- The collapse is not about the surface term at all but about what it *displaces*: weighting it up
  reduces the relative weight of the epsilon term, which is what keeps the latent on-manifold.
- The two terms disagree, and the disagreement grows with weight.

## ⚠️ Why this is not urgent

This is a *diagnosis of a rejected branch* — #84 already ruled the band-fix family out, and the shipped
arm is `vecset_v4_surf@240k`. It matters because the map's central problem is "how do we move the latent
without leaving the manifold", and this is a measured instance of exactly that going wrong.

Blocked in spirit by [#92](https://github.com/danvisai/SDFusion/issues/92): once the epsilon target is
aligned, re-run the surface-weight sweep. If the collapse disappears, it was the corrupted bridge all
along; if it persists, it is decoder-side and #73 owns it.

## Judged on

A written answer that picks between the candidates with a measurement, or records that it cannot and
says what would.


---

## Resolution — not disagreement; a magnitude imbalance that compounds under alignment

`train_vecset.py` sums the two losses at the model's own output — `loss = eps_loss + surf_weight *
surf`, both functions of `pred` — so `scripts/foundations/probe_surface_gradient_conflict.py`
computes `d(eps_loss)/d(pred)` and `d(surf_loss)/d(pred)` directly with autograd, on real training
rows against a real checkpoint, holding the model, the real target, the noise and the query points
fixed and varying only which blockout cache — encoded or #91's aligned — sources the pair corruption.
Same single-variable design #92 used.

**Candidate 3 (the terms disagree) is refuted.** At the #92 arms' shared starting checkpoint
(`vecset_v3_pair_long@180k`), n=24 region-balanced rows × 3 timesteps × 2 regimes, the **mean** cosine
similarity between the two gradients is under 0.005 in magnitude at every measured t, in both regimes
— orthogonal, not opposed:

| t/T | encoded cosine (mean) | aligned cosine (mean) |
|---:|---:|---:|
| 0.40 | −0.00024 | −0.00036 |
| 0.55 | +0.00054 | −0.00146 |
| 0.70 | +0.00017 | +0.00110 |

⚠️ Individual rows range wider — the single largest is 0.0145 (row 23779, aligned, t=0.40) — and the
medians are not always the same sign as the means (at t=0.40 both medians are actually the more
negative of the two: −0.00042 encoded, −0.00121 aligned). None of this changes the reading: every
value at every t, mean or median, is at least two orders of magnitude short of the ±0.3–1.0 range that
would read as real opposition. `gradients_oppose_on_average` (defined on the means) is **false** at
every t, and `aligned_more_conflicting_at_every_t` is also **false** — at t=0.70 the aligned regime is
*less* opposed than encoded on the mean. Whatever collapses the model, the two terms fighting each
other's direction is not it, at any t this probe sampled.

**What is large, from the very first shared step, is the surface term's magnitude.** At the weight
arms A and B actually train with (`--surf_weight 1.0`), `surf_grad_norm / eps_grad_norm` is **49x–128x
at the base checkpoint, in both regimes**:

| t/T | encoded norm ratio | aligned norm ratio |
|---:|---:|---:|
| 0.40 | 49.2 | 61.2 |
| 0.55 | 78.9 | 80.2 |
| 0.70 | 127.6 | 87.9 |

At equal nominal weight the surface term does not add to the epsilon signal, it **overwhelms** it —
the combined gradient at the model's output is 98%+ surface-term by construction, before any training
has happened. ⚠️ This is present in **both** regimes at similar scale, so alone it does not explain
why B collapses harder than A.

**But the imbalance grows specifically under alignment as training proceeds.** Re-measured on each
arm's own diverged step-220000 checkpoint, on that arm's own regime only (A on encoded, B on aligned —
each model has only ever seen its own):

| t/T | arm A (encoded) norm ratio | arm B (aligned) norm ratio | B / A |
|---:|---:|---:|---:|
| 0.40 | 5.35 | 11.46 | 2.14x |
| 0.55 | 19.31 | 26.31 | 1.36x |
| 0.70 | 72.60 | 163.06 | 2.25x |

Cosine stays near zero for both (≤ 0.004 in magnitude) — direction is still not the story. This tracks
that same checkpoint's own measured collapse: at step 220000, strength 0.5, A collapses 10.22% and B
18.21% (`outputs/watch_checkpoints/issue92_full714_{A,B}/curve.json`); by the matched 240k endpoint
that gap widens to 8.96% vs 46.36% (#92). The norm-ratio gap and the collapse gap move together.

⚠️ **This is corroboration, not a second controlled trial.** The step-220000 read uses each arm's own
model after 40k steps of divergent optimisation — exactly the confound the 180k shared-checkpoint
measurement above was built to avoid. It is consistent with the mechanism, not proof of it.

**Against the ticket's own pre-registered rule** ("if the collapse disappears, it was the corrupted
bridge all along; if it persists, it is decoder-side and #73 owns it"): collapse does not disappear
under alignment — it is nonzero in both regimes at every checkpoint measured and, unlike the rule's
two anticipated outcomes, it gets **worse**, not merely persistent, specifically under alignment
(10.22% → 18.21% at step 220000, 8.96% → 46.36% at the matched 240k endpoint). Read literally, "persists"
points the rule at candidate 1 (decoder-side), which is consistent with the magnitude-domination
finding above being present in both regimes from the shared starting checkpoint. But the rule was
written expecting a binary outcome and does not have a branch for "persists, and grows worse under
exactly the intervention that was supposed to help" — that third outcome is what the step-220000
comparison actually adds past what the ticket anticipated.

**Answer.** Candidate 3 is refuted — the gradients are orthogonal at every t this probe measured, not
opposed. Candidates
1 (decoder path intolerant) and 2 (displacement) predict the same observable — a large
`surf_grad_norm / eps_grad_norm` — and this measurement cannot separate them: a term 50–160x larger
*is* a displacement of the epsilon signal, and the most plausible reason it is that large is that the
frozen decoder amplifies a latent movement the model was never trained to make — candidate 1's
mechanism producing candidate 2's effect. What this measurement adds beyond the two original
candidates: alignment does not introduce a new failure mode — the imbalance exists from the shared
starting checkpoint, in both regimes — it makes an existing one compound harder as training proceeds
under the aligned target specifically.

**What would finish the split between 1 and 2:** an ablation that reweights the surface term
per-sample to equalise `surf_grad_norm` against `eps_grad_norm`, removing the magnitude imbalance
directly, then re-measures collapse under both regimes. If collapse stops diverging, the imbalance was
the whole mechanism (2, with 1 as its cause). If B still collapses harder than A at matched gradient
scale, something regime-specific survives normalisation and neither candidate as stated is complete.
That is a new short training run, out of reach of this probe.

**Not urgent, as the ticket says**: B is a rejected branch (#92). This closes the diagnosis; it does
not reopen whether B should ship.

Assets: `scripts/foundations/probe_surface_gradient_conflict.py`,
`execution/artifacts/surface_gradient_conflict_probe.json` (n=24, base checkpoint, both regimes),
`execution/artifacts/surface_gradient_conflict_probe_armA_step220000.json` and
`..._armB_step220000.json` (the step-220000 corroboration, one regime each, from each arm's own
weights).
