# The frozen round-trip gate — NEGATIVE

**Date:** 2026-07-28 · **n=24 held-out, stratified NL/DE/JP, real LoD2 surfaces (#62).**

**Verdict: the frozen vecset codec does not preserve our surfaces.** It is 2.4× rougher than ground
truth, worse than the model we currently ship, and worse than the wall that stopped every previous
attempt. A fine-tune is not an optimisation here — it is load-bearing.

## Result

| arm | roughness | |
|---|---|---|
| **GT** (stored field) | **0.00404** | the floor |
| **input surface** — CONTROL, codec not involved | **0.00404** | identical to GT |
| **FROZEN codec** | **0.00984** | the measurement |
| *refiner / corrector wall* | *0.00470* | reference |
| *map-#24 deployed* | *0.00552* | what we ship |

**Codec contribution = frozen − input = +0.00580.**

Occupancy tracks closely throughout (e.g. 0.214 → 0.215, 0.108 → 0.106), so the **shape** survives the
round-trip. What degrades is **surface fidelity** — exactly the quantity this effort exists to fix.

## The control arm is what makes this trustworthy

`input` came out **exactly equal to GT (0.00404)**. The recovered surfaces are perfect — as #62's
alignment check independently established — so the input contributes nothing to the degradation. Every
bit of the +0.00580 is the codec.

Without that arm this number would be unreadable, which is precisely the mistake the earlier smoke made.

## Correction: the earlier smoke was not meaningfully confounded

The n=1 smoke scored **0.00839** on a mesh extracted from our own 64³ field, and I attributed that to the
degraded input, expecting real surfaces to fix it. **That was wrong.** With perfect input surfaces the
result is **0.00984** — if anything slightly *worse*. The confound was real but immaterial; the codec was
always the cause. Recorded so the earlier reasoning is not trusted downstream.

## What this does and does not say

**Does not contradict [#63](https://github.com/danvisai/SDFusion/issues/63).** That measured a
*generation* pipeline — a frontier model producing an idealised building from an image, scoring 0.00328.
This measures *reconstruction fidelity* of a specific frozen codec on our own geometry. A vecset decoder
can evidently **produce** crisp surfaces; this particular frozen codec does not **preserve** ours.
Different quantities, both true.

**Does say** the domain gap #64 predicted is real and large. Dora was trained on ~400,000 Objaverse
meshes; LoD2 massing is a different world.

## A caveat that must be resolved before spending a training campaign

Our buildings are **extremely simple** — often ~12 vertices, essentially a box with a roof. Dora expects
complex Objaverse geometry, so this is out-of-distribution in the *unusual* direction: trivially simple
rather than trivially complex. Two consequences worth testing before concluding the codec is at fault:

1. **The sampler may be mis-tuned for this regime.** We feed 8,192 uniform + 8,192 sharp-edge samples.
   On a 12-edge box that is heavy oversampling of edges, and Dora's own preprocessing uses different
   stream ratios on watertight meshes prepared its own way.
2. **The latent is wildly over-provisioned** — 2,048 tokens for a box — so the posterior may be
   contributing noise where there is almost no geometry to describe.

**Recommendation: run a cheap sampling/latent ablation before committing to a fine-tune.** If a
meaningful part of the +0.00580 is our sampling rather than the codec, a fine-tune would be training
against our own artifact. This is hours of work against a multi-week campaign, and it is the same
cheapest-first discipline that produced every useful result in this effort.

The alternative arm, **TripoSG (MIT, SDF-native)**, is already named as the fallback in
[#65](https://github.com/danvisai/SDFusion/issues/65) and is worth measuring in the same harness — the
gate is now a one-command comparison for any codec satisfying the contract.

## Status against spec [#68](https://github.com/danvisai/SDFusion/issues/68)

The gate was the spec's deliverable and it has been delivered — with a negative answer, which is a
result rather than a failure. It sized the fine-tune, and the size is **large**. Nothing in
[#67](https://github.com/danvisai/SDFusion/issues/67) is invalidated: A2's later stages were always
gated on this number. They are now gated on it saying "adaptation is required."

## Reproduce

```
sdfusion/bin/python scripts/foundations/dora_frozen_gate.py --n 24
```

Artifact: `outputs/dora_frozen_gate/gate.json` (per-building rows, aggregate, codec delta).
