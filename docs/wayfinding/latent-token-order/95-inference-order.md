<!-- RECOVERED FROM THE ISSUE TRACKER, 2026-08-14. -->

> **Recovered document.** The original asset `95-inference-order.md` was written and committed on another
> machine (`3c4bac5`) and was never pushed; it does not exist in this repository or on origin.
> This file is its findings reconstructed verbatim from GitHub issue #95 — the ticket body
> and every resolution comment. It is the *record*, not the code: the implementation described
> below is still missing. See `RECOVERY.md`.


# #95 — Does an arbitrary token order at inference break a model trained on aligned pairs?

*State: open · opened 2026-08-09*


## Ticket

Part of #87

## Question

Training pairs will be **aligned**. Inference has nothing to align against — the model is handed a
footprint envelope encoded fresh, tokens in whatever order farthest-point sampling produced. Is that a
train/inference mismatch?

## Why I think it is fine, and why that is not good enough

The denoiser has **no positional encoding on the token axis** (`vecset_denoiser.py:14`) and is
permutation-equivariant (pinned by `test_permutation_equivariance`). So it cannot use a token's *index*,
only its *content*. Aligning the pair should therefore fix the **target** while leaving the learned map
content-conditioned, and an arbitrary inference order should pass through unharmed.

⚠️ That is reasoning, not measurement — and this map has already burned me once for exactly that
substitution (#89 asserted t-independence in a flag's help text instead of sweeping t, and the sweep
then mattered). Cheap to check, expensive to be wrong about: #91 spends ~4 h rebuilding caches and #92
~20 h training on the assumption.

## How to check it, against the EXISTING checkpoint

No new training needed. On `vecset_v4_surf@240k`:

1. Project a footprint envelope, then project the **same** envelope with its tokens permuted. Because
   the model is equivariant the outputs should differ only by that permutation, so the **decoded fields
   must be identical**. If they are not, order already matters at inference today and the whole plan
   needs rethinking.
2. Establish what an aligned run would change by simulating it: reorder the input tokens into a
   canonical (Morton) order and confirm the decode is unchanged.

If both hold, an arbitrary inference order is provably harmless and #91/#92 can proceed. If either
fails, the alignment has to be applied at inference too — which is possible for the sort (single-latent)
and **impossible for matching**, since there is no partner at inference. That would decide #90.

## ⚠️ Runs BEFORE #91

Unblocked, ~1 h, and it can invalidate the ordering choice. Doing it after the cache rebuild would be
doing it in the wrong order.

## Judged on

A measured answer with the decoded fields compared numerically, and an explicit statement of which of
#90's candidate methods survive it.


---

## Comment — danvisai, 2026-08-09

## Resolved: order at inference is **noise, not bias** — #91/#92 proceed, and #90's tie-breaker evaporates

Probe `probe_token_order.py --inference`, artifact `execution/artifacts/token_order_inference.json`,
writeup `docs/wayfinding/latent-token-order/95-inference-order.md` (`8c94c06`, `3c4bac5`).
n=24 held-out buildings, **8 per region**, 5 orderings each, on `vecset_v4_surf@240k` at strength 0.5.

### This ticket's own test, run literally — and its premise was wrong

I predicted the decoded fields "**must be identical**" under a token permutation, and said otherwise
*"the whole plan needs rethinking"*. On the real checkpoint at 2048 tokens:

| | max &#124;field difference&#124; |
|---|---|
| permute tokens **and noise** | **7.36e-04** |
| permute tokens **only** | **2.03** |

The operator **is** equivariant on real weights (7.36e-04 across a 191.6M decoder and 20 DDIM steps is
float32 accumulation). The fields are **not** identical under a token-only permutation — 2.03 is the
full range of a field saturating at ±1.

🔑 **The ticket conflated two claims.** Equivariance is about permuting the token set *and its noise*. At
inference the noise comes from a seed, independent of token order, so a permuted input is a different
**sample**, not a broken symmetry. **The premise needed correcting, not the plan** — and that is now
demonstrated rather than argued, which is the distinction this ticket was written to enforce.

### How much does quality actually move

| | range | SD |
|---|---|---|
| median | **0.0208** | **0.0080** |
| mean | 0.0851 | 0.0353 |
| max | **0.5386** | 0.2370 |

**17 of 24 buildings move under 0.05; three move over 0.20.**

The deciding number is the aggregate: implied **SE on a median over n=714 = 0.00132**, against the
harness's own published noise floor of 0.001 and the effects being chased (+0.029 surface loss, +0.008
tripling training). 🔑 And in #92 both arms see the **same** envelope in the **same** order, so it is
common-mode and cancels in a paired per-building difference — now a **requirement** on #92, not an
implication.

### Which of #90's methods survive: **all of them**

Order at inference is noise, not bias, so *matching* is not disqualified by having no partner at
inference.

⚠️ **But the sort's tie-breaker evaporates.** A canonical Morton order — applicable at inference, which
looked like an argument in its favour — measures **−0.0150** against as-encoded: slightly *worse*, well
inside the noise. **Canonicalising the input at inference stabilises nothing.** #90 must choose on
alignment quality and cost alone.

### 🔑 Unexpected: the collapse can be triggered by token order alone

| row | region | range | `vs_input` | per-ordering 3D IoU |
|---|---|---|---|---|
| 23903 | JP | **0.539** | 0.859 | 0.969 · 0.944 · 0.974 · **0.436** · 0.973 |
| 11912 | DE | 0.401 | 0.810 | 0.626 · 0.528 · **0.302** · 0.621 · 0.703 |
| 11874 | DE | 0.326 | 0.572 | 0.657 · 0.539 · **0.331** · 0.426 · 0.595 |

Row 23903 is fine under four orderings and **collapses under the fifth**, with nothing about the
building changed. The unstable buildings carry the **lowest `vs_input`** (0.57–0.89 vs a 0.986 median) —
they are the ones the model actually acts on, sitting at #73's tolerance boundary.

That gives [#84](https://github.com/danvisai/SDFusion/issues/84) and
[#94](https://github.com/danvisai/SDFusion/issues/94) a testable property their size/solidity/region
analyses did not have. It is **not** evidence that alignment fixes the collapse.

### ⚠️ Two process failures worth recording

1. **This probe fell into the map's most-cited trap.** Its first run took the first 10 held-out rows and
   got **10/10 region 0** — and it changed the answer (Dutch-only: mean spread 0.0363 / max 0.1735,
   against 0.0851 / 0.5386 stratified; the Morton delta flipped sign, +0.0051 → −0.0150). Cause: it
   **reimplemented** `pick_ids` instead of calling it, undoing the fix `4b77f8e` had already made. Now
   it calls `pick_ids`.
2. **Both code-review sub-agents died on a session limit**, so this ticket had a **self-review** rather
   than the two-axis split. That is weaker by design and is recorded, not glossed. Two findings it
   raised are unfixed and named: `inference_order` is a ~90-line do-everything function where peers
   split into phases, and it imports the private `_vertical_extent`.


---

## Comment — danvisai, 2026-08-12

Reopening: this ticket's resolution was implemented on a cloud A100 instance that was lost before the resolving commits (cited in the comment above) were pushed. None of those commits exist on any branch or PR in this repo — verified against `git log --all` and `git ls-remote`. The written analysis and decisions above are intact and should be treated as the spec; the implementation, and #91's rebuilt caches specifically, need to be redone from scratch.
