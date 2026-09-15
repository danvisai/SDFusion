<!-- RECOVERED FROM THE ISSUE TRACKER, 2026-08-14. -->

> **Recovered and re-measured.** The original asset was lost with commit `4dcdccd`; this file preserves
> its tracker record. The actual pair-training-path probe has now been reconstructed as
> `scripts/foundations/probe_surface_loss_order.py`, with its fresh measurement in
> `execution/artifacts/surface_loss_order_probe.json`.


# #89 — Does the decoded-surface loss escape the ordering corruption?

*State: closed · opened 2026-08-09 · implementation re-measured 2026-09-01*


## Recovered measurement — 2026-09-01

The reconstructed probe changes only the blockout token order and applies the same permutation to
its diffusion noise, while the real latent target stays fixed. It evaluates the shipped
`vecset_v4_surf@240k` denoiser and the exact `surface_term` from `train_vecset.py` on six
region-balanced training rows, five orderings, 8,192 shared query points, and three timesteps.

| t/T | epsilon spread | decoded-surface spread | epsilon / surface sensitivity |
|---:|---:|---:|---:|
| 0.40 | 1.406993% | 0.0000618% | 22,751× |
| 0.55 | 1.039246% | 0.0001749% | 5,941× |
| 0.70 | 0.286167% | 0.0000345% | 8,297× |

The lost run reported epsilon spreads of 1.53% / 0.97% / 0.31% and surface spreads of
0.0003% / 0.0005% / 0.0001%. The reconstruction reproduces both the scale and schedule trend, and
strengthens the decision: the decoded-surface term is **5,941–22,751× less order-sensitive at every
measured t**. #89's conclusion survives recovery.


## Ticket

Part of #87

## Question

The decoded-surface loss (#76/#80) decodes x-hat-0 and compares **surfaces**, not tokens. Decoding is
permutation-invariant (measured: max field difference 6.8e-06, occupancy IoU 1.000000). So the surface
term should be **immune** to the ordering corruption that this map says wrecks the epsilon target.

Is it? And if it is, does that explain why it was the biggest single lever #69 ever found (+0.029 in a
sixth of the steps that bought +0.008)?

## Why this is worth a ticket

If confirmed, it says the surface term was the **only uncorrupted geometric signal in the objective**,
which changes how it should be weighted once the epsilon target is fixed. It also predicts something
testable: with the bridge repaired, the surface term's marginal value should **drop**, because it is no
longer carrying the whole load.

⚠️ It cuts the other way too. #84 found that pushing the surface term harder collapses the model
(solidity 89.9% -> 57.6% -> 19.2%). If the term is the only clean signal, that collapse needs
re-explaining rather than being attributed to surface pressure alone.

## Judged on

A written answer with the invariance measured on the actual training path, not argued from the decoder
property alone. Cheap: no training run, no cache rebuild. Unblocked, so it can run in parallel.


---

## Comment — danvisai, 2026-08-09

## Resolved: **yes, the surface term is immune** — and the map's premise survived the control that could have voided it

Probe `scripts/foundations/probe_token_order.py`, artifact `execution/artifacts/token_order_probe.json`,
writeup `docs/wayfinding/latent-token-order/89-surface-loss-immunity.md` (`0a17f91`, `4dcdccd`).

### Asked as a spread, not a before/after

A permuted order is not a corrupted version of a privileged "true" order — every order is an equally
correct description of the same shape. So the question is **how much does each term vary across
orderings that are all equally valid?** A term that swings is measuring the ordering, not the geometry.

### Answer, swept across the schedule

| t/T | epsilon loss varies | surface term varies | ratio |
|---|---|---|---|
| **0.40** | 1.53% | 0.0003% | **5,308×** |
| 0.55 | 0.97% | 0.0005% | **1,927×** |
| 0.70 | 0.31% | 0.0001% | **3,466×** |

🔑 **Order-insensitive at every t**, so the conclusion does not depend on the slice. That mattered: the
shipped `vecset_v4_surf` **predates `--surf_t_center`**, so its own surface term ran at ≈0.40, not the
0.55 I first probed. The first version of this probe measured one t and *asserted* t-independence in a
flag's help text — the exact arguing-instead-of-measuring this ticket forbade. Caught in review, fixed.

Structurally as predicted: the term compares `codec.query(x0)` against `codec.query(z)` — field values
at fixed 3-D points, never tokens against tokens. The decoder's permutation-invariance is now pinned by
`test_query_is_permutation_invariant` instead of resting on a docstring claim.

### The second question, answered with the tension left visible

It **supports without proving** that the surface term was the only geometric signal not measuring an
arbitrary ordering — a candidate explanation for it being #69's biggest lever.

⚠️ But that phrase is in tension with a correction this probe also produced, and the two are **not
reconciled**: **51.1%** of the bridge energy survives averaging over orderings and the epsilon loss moves
only ~1%, so "the only uncorrupted signal" is too strong against a merely *degraded* one. Both readings
are recorded; choosing needs #92.

### ⚠️ The ticket's #84 warning — discharged, not dropped

Filed as **[#94](https://github.com/danvisai/SDFusion/issues/94)** (blocked by #92): if the surface term
is the cleaner signal, #84's collapse under increased surface pressure needs a different explanation
than "too much pressure on a good signal".

### 🔑🔑 A control that nearly voided map #87, and then didn't

Reordering `zb` changes the bridge magnitude by **1.007** and the cosine goes **0.039 → 0.032** — today's
order is worth no more than a random one, which reads as "nothing to align". That is the **wrong
contrast**. Against the matched order (n=5):

| ordering | cosine to `z` | position error |
|---|---|---|
| as encoded | +0.0527 | 1.120 |
| randomly permuted | +0.0366 | — |
| **position-matched** | **+0.3235** | **0.037** |

**~6× higher cosine, 16.8× the identity-vs-random gap.** Both halves are needed: today's order is
worthless *and* matching recovers real alignment. Co-located query positions did **not** guarantee
similar latent vectors — a Dora token is cross-attention over the whole cloud, not a local descriptor —
so this had to be measured.

⚠️ Limits, stated: fresh encodes with a re-derived envelope (not the cache rows), many-to-one `argmin`,
n=5, and a different id set gave 54.6× where this gives 16.8× — direction stable, magnitude not.

### For the rest of the map

- ✅ #90 / #91 are worth running — the premise passed its control.
- ▶️ **This is not a recommendation of a method.** It says *some* alignment exists to recover. #90 still
  owes a Morton-sort latent-space cosine, bijectivity, and stability.
- ⚠️ **Re-weight #92 down**: a ~1% loss swing is not a catastrophe being repaired. The pre-registered bar
  stands as written.
- ▶️ Falsifiable side-prediction for #92: with the bridge repaired the surface term's marginal value
  should **drop**. Run aligned with and without `--surf_weight` and compare to the shipped 0.920 → 0.962.

Reviewed on both axes; 16 findings applied (worst: the t-assertion above). 34 tests green.


---

## Comment — danvisai, 2026-08-09

⚠️ **Corrected by [#88](https://github.com/danvisai/SDFusion/issues/88).** This ticket's matched-order control used a hand-rolled position capture that drew its point cloud separately from the encode; `sample_uniform` was ignoring its `rng` at the time, so the coarse half of every position set came from global RNG state and did **not** correspond to the latent beside it.

Re-run through `DoraCodec.encode_with_positions`:

| ordering | as published | corrected |
|---|---|---|
| as encoded | +0.0351 | +0.0480 |
| permuted | +0.0299 | +0.0347 |
| **matched** | **+0.3235** | **+0.7288** |

**The published figure understated the recoverable alignment by more than 2×.** Direction unchanged, conclusion strengthened. The surface-term immunity result is unaffected — it never used those positions. Writeup corrected in place with both figures on the record.


---

## Comment — danvisai, 2026-08-12

Reopening: this ticket's resolution was implemented on a cloud A100 instance that was lost before the resolving commits (cited in the comment above) were pushed. None of those commits exist on any branch or PR in this repo — verified against `git log --all` and `git ls-remote`. The written analysis and decisions above are intact and should be treated as the spec; the implementation, and #91's rebuilt caches specifically, need to be redone from scratch.
