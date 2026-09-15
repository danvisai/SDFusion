<!-- Mirrored from the tracker, 2026-08-14. -->

> **Resolved ticket, mirrored locally** so this effort can be read without the tracker. The four
> training arms survived the machine transition and the final full-heldout evaluation was completed
> on 2026-09-01.


# #92 — Retrain with the aligned pair target, against v4_surf@240k as the control

*State: resolved 2026-09-01 · opened 2026-08-09*


## Ticket

Part of #87

## Task

The run this map exists for. **Four arms in a 2x2**, so both open questions are answered by one
experiment rather than by a sequence of unattributable single runs.

## The design

|  | epsilon target **as encoded** | epsilon target **aligned** |
|---|---|---|
| **with** `--surf_weight 1.0` | A (reproduces the shipped regime) | **B (the candidate)** |
| **without** the surface term | C | D |

- **B vs A** is the map's question, and it is now a genuine single-variable comparison: #91 derives the
  aligned cache as a *permutation of the same latents*, so the two arms differ in token order and in
  nothing else.
- **(B−D) vs (A−C)** tests #89's falsifiable side-prediction: with the bridge repaired, the surface
  term's **marginal value should drop**, because it is no longer the only signal not measuring an
  arbitrary ordering. The shipped arm's gap is 0.920 → 0.962.

⚠️ **A is the control, not `v4_surf@240k`.** The shipped checkpoint trained on the *old* cache, so
comparing against it would mix alignment with a different surface draw. `v4_surf` stays on the record as
the reference for the old regime, and A is what B is judged against.

## Compute

All four resume from the **same** `vecset_v3_pair_long_step180000.pth` base, so the arms share a starting
point and only the studied variables move. 60k steps each at ~305 ms = ~5 h per arm, **~20 h total** on
the A100. Run them sequentially; the GPU is idle.

▶️ **Follow-on, gated on B beating A:** one **from-scratch** aligned run (240k steps, ~20 h). The shared
base spent 180k steps learning from misaligned pairs, so a fine-tune measures "can alignment help from
here", not "what is alignment worth". Only worth the day if the fine-tune shows signal.

## The bar, fixed in advance

Per this map's Destination, criterion 2, all three at the strength maximising median 3D IoU on n=714:
`vs_input` median **< 0.98**, median 3D IoU **>= 0.876**, beats the envelope on **> 5%** of buildings.

## ⚠️ Rules this run must follow

- 🔑🔑 **Do not stop at a dip and do not extrapolate.** #75 went 0.719 -> 0.657 -> **0.532** -> **0.840**;
  the band-fix run went **0.200 -> 0.825** between adjacent checkpoints. A stop was recommended at the
  dip **twice** and was wrong **twice**. Checkpoint every 10k and score at least four per arm.
- Score with `eval_full_heldout.py` on **all 714**, never a prefix.
- Publish the **collapse rate** beside every median.
- ⚠️ Do not lead the scorecard with `extra`: under-building improves it for the wrong reason, a hazard
  that has fired three times on #69.
- Report `vs_input` beside every quality number.

## Judged on

The 2x2 table on n=714 with `vs_input`, a montage for criterion 1, the marginal-surface-value
comparison, and a plain statement of whether the pre-registered bar was met — including "met only
transiently" if that is what happened.


## Resolution — alignment does not restore a usable band

**Verdict: NOT MET, and not met transiently.** Greedy alignment makes the pair objective much easier
to fit, but it does not produce a strength where B both acts and preserves quality. The from-scratch
aligned follow-on is therefore **not triggered**: B did not beat A under the registered visual/AND bar.

### Execution integrity

All four arms resumed the same 180k checkpoint and reached 240k. Every 10k checkpoint from 190k
through 240k was scored at strength 0.5 on the exact 714 IDs from
`execution/artifacts/massing_arms_eval_ship714.json`; all 24 raw artifacts were checked for identical
IDs and population size. No checkpoint was dropped at a dip. The candidate checkpoint was chosen by
median 3D IoU on that common-strength curve, then B alone received the ticket's fixed eight-strength
sweep.

The best checkpoint observed at the common strength of 0.5 was:

| arm | selected step | 3D IoU | `vs_input` | collapse | beats envelope |
|---|---:|---:|---:|---:|---:|
| A — encoded + surface | 220k | 0.8622 | 0.9721 | 10.22% | 0.28% |
| **B — aligned + surface** | **190k** | **0.8735** | **0.9911** | **11.90%** | **9.38%** |
| C — encoded + no surface | 200k | 0.8715 | 0.9876 | 6.16% | 1.26% |
| D — aligned + no surface | 190k | 0.8730 | 0.9850 | 19.89% | 10.92% |

B's apparent +0.0113 over A is a no-op comparison: B is 99.1% its input and therefore inherits the
envelope. At the matched 240k endpoint the sign reverses: B scores 0.7616 against A's 0.8573 and
collapses 46.36% against 8.96%.

### Candidate strength sweep — full 714

| strength | 3D IoU | `vs_input` | collapse | beats envelope | footprint gate pass |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.8075 | 0.9575 | 32.63% | 1.68% | 94.26% |
| 0.40 | 0.8464 | 0.9843 | 19.05% | 3.64% | 97.06% |
| **0.45** | **0.8753** | **0.9938** | **10.64%** | **5.74%** | **98.60%** |
| 0.50 | 0.8735 | 0.9911 | 11.90% | 9.38% | 98.32% |
| 0.55 | 0.8485 | 0.9817 | 16.81% | 9.38% | 96.22% |
| 0.60 | 0.8143 | 0.9625 | 27.17% | 7.14% | 91.46% |
| 0.70 | 0.1294 | 0.1591 | 91.88% | 0.14% | 53.36% |
| 0.85 | 0.1850 | 0.2197 | 100.00% | 0.00% | 12.32% |

At the strength that maximises median IoU, the registered AND bar reads:

- `vs_input < 0.98`: **FAIL**, 0.9938;
- 3D IoU `>= 0.876`: **FAIL**, 0.8753 (short by 0.0007);
- beats envelope `> 5%`: **PASS**, 5.74%.

The only sampled strengths that clearly act are 0.30, 0.60, 0.70, and 0.85; all lose quality, and
the last two are catastrophic. There is no hidden middle: no observed checkpoint at strength 0.5
met the bar, and no strength on the selected checkpoint met it either.

### Visual and footprint criteria

The shaded sweep tells the same story more clearly than the scalar near-miss. At 0.40–0.55 the model
mostly reproduces the envelope, including its missing roof decisions. At 0.60 it begins deforming and
melting faces; by 0.70–0.85 the outputs are slats, rubble, or rounded blobs. No sampled setting reads
as a net-new real building rather than an extrusion or a damaged extrusion. **Criterion 1 fails.**

At the scalar-selected 0.45, 98.60% pass the 5% spill/uncovered footprint gate, above the old 77.0%
reference. This is not an independent win: `vs_input=0.9938` shows that the near-no-op inherits the
envelope's footprint. The worst-first plan still exposes the small tail—detached masses, filled
courtyards, and spill up to 22.0%—but criterion 2 does not regress in aggregate.

### The matched 2x2 and #89's side-prediction

Pure factorial claims use the same step and strength, not each arm's separately selected operating
point. At the matched 240k endpoint, strength 0.5:

| arm | 3D IoU | `vs_input` | collapse | beats envelope |
|---|---:|---:|---:|---:|
| A | 0.8573 | 0.9624 | 8.96% | 0.42% |
| B | 0.7616 | 0.8910 | 46.36% | 3.50% |
| C | 0.8332 | 0.9682 | 17.93% | 0.70% |
| D | 0.8709 | 0.9894 | 19.05% | 9.10% |

The decoded-surface term's endpoint marginal is **+0.0241 IoU / -9.0 collapse points** in the encoded
pair, but **-0.1093 IoU / +27.3 collapse points** in the aligned pair. Across all six matched
checkpoints its median IoU marginal drops from **-0.0391** encoded to **-0.1156** aligned, while its
median collapse penalty grows from **+18.1** to **+26.2 points**. #89's prediction that the surface
term's marginal value would drop after repairing the bridge is confirmed—more strongly than hoped:
the term becomes actively harmful rather than merely redundant.

### Assets

- Machine-checkable decision: `execution/artifacts/issue92_2x2_summary.json`
- Candidate sweep: `execution/artifacts/massing_arms_eval_issue92_strength_armB_step190000.json`
- Six-checkpoint curves: `outputs/watch_checkpoints/issue92_full714_{A,B,C,D}/curve.json`
- Candidate montage: `docs/wayfinding/latent-token-order/92-arm-b-strength-montage.png`
- Worst-first plan: `docs/wayfinding/latent-token-order/92-arm-b-plan-worst.png`
