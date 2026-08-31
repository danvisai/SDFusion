<!-- Mirrored from the tracker, 2026-08-14. -->

> **Open ticket, mirrored locally** so this effort can be read without the
> tracker. Nothing was lost for this one — it had no committed asset.


# #92 — Retrain with the aligned pair target, against v4_surf@240k as the control

*State: open · opened 2026-08-09*


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
