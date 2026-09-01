<!-- Mirrored from the tracker, 2026-08-14. -->

> **Open ticket, mirrored locally** so this effort can be read without the
> tracker. Nothing was lost for this one — it had no committed asset.

> **Resolved, 2026-09-01.** Re-measured on #92's aligned candidate rather than the shipped control —
> see the Resolution section below and `execution/artifacts/issue93_strength_band_armB.json`.


# #93 — Re-measure the strength band on the retrained model

*State: resolved 2026-09-01 · opened 2026-08-09*


## Ticket

Part of #87

## Question

Does the corrected target actually give back the **middle** of the strength dial — the thing a user
touches?

## The measurement to repeat

On the shipped `v4_surf@240k`, `vs_input` by projection strength (1.00 = returned the envelope,
< 0.5 = shredded):

| footprint | 0.30 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.70 | 0.85 |
|---|---|---|---|---|---|---|---|---|
| rect 20x12 | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 | 1.000 | **0.006** | 0.000 |
| L-plan | 0.993 | 0.998 | 1.000 | 1.000 | 0.990 | 0.964 | 0.752 | 0.033 |
| block 40x30 | 0.894 | 0.959 | 1.000 | 0.999 | 0.999 | 0.999 | **0.001** | 0.001 |

For the plain rectangle the usable band is **empty** — every setting is either a no-op or rubble. This
is #73's prediction made interactive, and it is what the human found by hand in the demo.

## Also worth explaining

The **L-plan tolerates more than the box** (0.964 at 0.60, 0.752 at 0.70). A low-solidity footprint
having *more* editing room is the opposite of what #84's solidity finding would suggest. Either it
generalises or it is n=1.

## Judged on

The same sweep on the retrained model, on more than three footprints, with the band stated as a range
per footprint rather than a single number. Feeds straight into the demo, where the strength slider is
already exposed.


---

## Comment — danvisai, 2026-08-14

## Control-arm strength data from an unusual angle (the town demo), offered as input to this ticket

Wiring A2 into the town demo ([#99](https://github.com/danvisai/SDFusion/issues/99)) meant driving
`vecset_v4_surf@240k` — this map's designated control — over a **population it was never evaluated
on**: hand-drawn polygons and OSM-extracted footprints, rather than held-out corpus buildings. That
turns out to say something about the band, so it is recorded here rather than left in the demo.

**Caveat first, because it bounds everything below:** a drawn footprint has **no ground truth**, so
the only thing measurable here is `vs_input` — departure from the blockout, not quality. "Moved a
lot" is not "got better". Vertex counts and the render are the only corroboration, and they are
weak. Everything below is `n=3` at `seed=0`, `steps=20`, `guidance=1.0`; the spread column is real
run-to-run variance (`DoraCodec` seeds one RNG at construction, so repeat encodes resample).

### The band is a cliff, not a dial

| footprint | strength | `vs_input` mean (n=3) | spread | verts |
|---|---|---|---|---|
| 18×12 m rectangle | 0.35 | 1.000000 | — | ~15.2k |
| 18×12 m rectangle | 0.5 | **1.0000** | 0.0000 | ~15.2k |
| 18×12 m rectangle | 0.7 | **0.0108** | 0.0074 | 2.8k |
| 18×12 m rectangle | 0.9 | 0.0028 | — | 1.2k |
| L, 20×18 m | 0.5 | 0.9975 | 0.0048 | ~16.7k |
| L, 20×18 m | 0.7 | 0.6573 | **0.2268** | 30.0k |
| L, 20×18 m | 0.9 | 0.4207 | — | 49.1k |

On a plain convex blockout there is **no usable middle**: 0.5 returns the input essentially
bit-for-bit (spread 0.0000 — it is not even stochastic, it is a no-op), and 0.7 discards it almost
entirely, with the vertex count collapsing 15.2k → 2.8k, which reads as the near-empty failure
rather than a different building.

### Past the cliff it is also unstable, and fails in two different ways

The L at 0.7 swings **0.52–0.75 across identical requests** — a 0.23 spread, ~30x the spread at 0.5.
So the region past the cliff is not merely worse, it is not reproducible.

The two shapes also fail *differently*: the rectangle loses vertices (collapses toward empty) while
the L gains them (15.2k → 30k → 49k, i.e. gets noisier/more fragmented). A single scalar reported
at one strength would hide that these are opposite failures.

### Footprint size and height move it as much as strength does

At a fixed strength of 0.5, on real OSM footprints (n=1 each, so indicative only):

| footprint | height | vertical voxels | `vs_input` |
|---|---|---|---|
| Munich, smallest (26 m span) | 12 m | 28 | 0.928 |
| Munich, smallest (26 m span) | 40 m | 61 | 0.964 |
| Munich, median (84 m span) | 12 m | 10 | 0.861 |
| Munich, median (84 m span) | 40 m | 30 | **0.225** |
| Munich, largest (251 m span) | 12 m | 4 | 0.644 |
| Munich, largest (251 m span) | 40 m | 10 | 0.560 |

The same checkpoint at the same strength ranges from 1.000 (no-op) to 0.225 (rewrote three quarters)
purely by changing the conditioning footprint. Note this is **not** monotone in the vertical voxel
budget — the median footprint at 40 m is well resolved (30 of 64 voxels) and is the *most* rewritten
row in the table — so "pancake aspect ratio" does not explain it, and I did not chase it further.

### What this is and is not evidence for

It is **control-arm** data, on an out-of-corpus population, and it is consistent with this map's
premise that the transform lacks a usable band on this checkpoint. It does **not** measure the
retrained model, and it cannot speak to output *quality* for want of ground truth. If the band
re-measurement here wants a second population to check against, the demo path can produce it
cheaply — `scripts/server/town_generate_service.py`, ~10s/building.

Practical consequence already taken: the demo ships `strength=0.5` fixed with no user control, since
the only alternative on offer is a cliff.


---

## Resolution — a per-building band exists, but it covers under 10% of the corpus

Re-measured on **#92's aligned candidate** (`B`, `issue92_aligned_retrain/B_aligned_surf@190000`)
rather than the shipped `v4_surf@240k` control, since that is the "retrained model" this ticket asks
about. Reuses #92's own eight-strength full-714 sweep
(`execution/artifacts/massing_arms_eval_issue92_strength_armB_step190000.json`) rather than repeating
the run — same data, a different reading. `scripts/foundations/analyze_issue93_strength_band.py`
classifies every building at every strength as `no_op` (`vs_input ≥ 0.98`), `collapsed`
(`missing ≥ 0.15`), `net_positive` (acted, survived, and beat *that same building's own* footprint
envelope), or `net_negative`, then reports each building's own usable-strength set instead of one
aggregate scalar — the "range per footprint" this ticket's Judged-on clause asked for, at n=714
rather than three hand-drawn shapes.

| strength | no_op | net_negative | collapsed | net_positive |
|---:|---:|---:|---:|---:|
| 0.30 | 290 | 189 | 233 | 2 |
| 0.40 | 370 | 201 | 136 | 7 |
| 0.45 | 464 | 159 | 76 | 15 |
| 0.50 | 452 | 157 | 85 | 20 |
| 0.55 | 377 | 185 | 120 | 32 |
| 0.60 | 243 | 235 | 194 | 42 |
| 0.70 | 10 | 48 | 656 | 0 |
| 0.85 | 0 | 0 | 714 | 0 |

**A usable strength exists for 69 of 714 buildings (9.66%).** The answer to this ticket's question —
does the corrected target give back the *middle* of the dial — is mostly no: for 90% of the corpus no
sampled strength both acts and helps. 🔑 Of the 69 with a band, **38 have exactly one working
strength, and 19 of those 38 work only at 0.60** — a setting with a **27.17%** collapse rate, higher
than every other sampled strength short of 0.30 (32.63%) and the 0.70/0.85 wipeout. The population 0.60
serves is not the population 0.45–0.50 serves (dominant single-strength counts: 0.60→19, 0.55→9,
0.40→4, 0.50→4, 0.45→2), so there is no single dial setting that is even the *right choice for the
buildings it helps* — a wider net of strengths recovers more buildings than any one of them, but no
one strength is a serviceable default beyond what #92 already picked.

⚠️ **This qualifies #92's own headline "beats envelope" figure.** Of the 67/714 (9.38%) rows #92
counted as beating the envelope at strength 0.5, **47 are no-ops** (`vs_input ≥ 0.98` — the IoU nudged
up while the building barely moved) and 0 are collapsed, leaving only **20/714 (2.8%)** that both
acted and improved. This is the map's own standing trap — "report `vs_input` beside every quality
number... a near-no-op inherits the envelope's perfect footprint and is scored for it" — firing on
#92's own table rather than the checkpoint comparison it was written to guard.

**Not evidence the band improves elsewhere.** This is the aligned candidate that #92 already found
does not meet the registered bar; #93 was scoped to describe *how* the failure is distributed across
strength and footprint, not to reopen whether it passed. It does not.

Asset: `execution/artifacts/issue93_strength_band_armB.json`.
