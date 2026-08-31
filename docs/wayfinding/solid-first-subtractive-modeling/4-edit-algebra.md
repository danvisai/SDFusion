# #4 — The semantic architectural edit algebra

*Effort: solid-first semantic architectural carving. Opened 2026-07-15, taken up 2026-08-30. One of
the three tickets with no open blockers, and the one #7, #9, #8 and #2 all wait on.*

> What is the minimal semantic architectural edit algebra for this system: operation ontology,
> constrained geometry per type, add/subtract semantics, parent/support/containment relations,
> ordering and commutativity, canonical normal form, equivalence, invalid references, deletion,
> resampling, and local regeneration? It must represent courtyards, passages, arcades, light wells,
> setbacks, terraces, roof cuts, wings, roof volumes, and any smaller vocabulary justified during
> the decision, while serving both autonomous generation and rough-carve interpretation.

The vocabulary half was already decided by the owner's measurement on this ticket (2026-08-27) and
by [#10](10-program-recovery.md). What was **not** decided — and what turned out to be wrong in the
code — is the *ordering*. That is this ticket's result.


## 🔑🔑 The algebra was accidentally ordered, and commutativity was free

The SDF compiler composes every layer-program operation with `sdf_subtract`, and subtracting A then
B is subtracting their union — so `EditableBuilding` has always been **commutative by
construction**. The height-map compiler was not: `replay_program` applied `Layer` as a **set**
(`where(region, v, h)`), which can *raise* a column an earlier operation had lowered.

Measured on 250 recovered programs, before any code changed:

| | |
|---|---|
| programs where two operation regions overlap | **78.0%** |
| …where **permuting the operations changed the compiled building** | **69.6%** |
| …where reading `Layer` as a `min` instead differs from the `set` | **0.0%** |
| …where any operation raises a column | **0.0%** |

Then, after switching to `min`, 250 programs × 8 permutations each:

| | |
|---|---|
| permutations that changed the building | **0 of 2,000** |

Re-runnable from the repo, which is what makes these a measurement rather than a recollection:

    ./sdfusion/bin/python scripts/foundations/recover_massing_programs.py \
      --measure_commutativity 250 --out execution/artifacts/program_recovery_714.json

It reads an existing artifact, re-runs no fit and writes nothing, so it cannot disturb the record it
is checking. `replay_program_ordered` keeps the old `Layer`-as-set reading alive for exactly this
purpose and nothing in the pipeline calls it.

So the algebra was order-dependent by accident, the two compilers agreed only because **#10's
fitter never emits an operation that would raise a column** — a property of the *search*, which a
hand-authored or generated program is under no obligation to have — and the fix cost nothing.

**Decision: the core algebra is subtract-only and commutative.** `Layer` takes a `min`
(`recover_massing_programs.replay_program`). #128's bridge still passes unchanged: serialised
program replays to the fitted height map 12/12, composed SDF == voxel compiler 12/12.

⚠️ This is not a tidy-up. Commutativity is what makes the next four answers *exist at all*: an
ordered algebra has no canonical form that is not just "the order you happened to write it in", no
deletion except from the top, and no cheap equivalence test.


## The ontology: two tiers, and the line between them is a measurement

`scene/sdf_edit.ALGEBRA` declares every kind the compiler accepts. The palette had grown into a flat
list mixing raw CSG (box, sphere) with the three architectural operations, and nothing said which
was which.

| tier | kinds | subtract-only | height-map | learnable **here** |
|---|---|---|---|---|
| **core** | `layer`, `ramp`, `cut_roof` | yes | yes | yes |
| **volumetric** | `box`, `rounded_box`, `sphere`, `cylinder`, `cone`, `gable`, `hip`, `element` | no | **no** | **no** |

🔑 The `height_map` column is load-bearing because of #10's measurement, not a preference: `missing`
= 0 on 714/714, 100% of carve volume above the topmost GT voxel, **0 voxels** of through-void, 71
overhang voxels in 4,324,919. An operation that leaves the height field has **no training signal on
this corpus at all** — it is cuttable and never learnable. `is_height_map_representable(ops)` is
that property as a predicate, and on this corpus it answers both "which compiler can run this" and
"could this ever have been generated".


## The nine names #4 requires, each resolved

Recorded per name in `ARCHITECTURAL_VOCABULARY` so none is quietly dropped, and so a later corpus
with real voids can flip `learnable_here` without anyone rediscovering why it was `False`.

| name | resolves to | learnable here | why |
|---|---|---|---|
| **setback** | `layer` | ✅ | **not a separate operation** — in a height field a setback *is* a Layer whose polygon is the inward offset of the footprint, and the fitter finds it as one |
| **terrace** | `layer` | ✅ | a Layer, or a stack of them |
| **roof cut** | `cut_roof` | ✅ | the core operation, hip or gable |
| **roof volume** | `ramp` | ✅ | a pitched roof is one or more Ramps; a gable is two opposing ones |
| **wing** | `layer` | ✅ | ⚠️ **two senses.** A wing at a *different height* is a `Layer` over that part of the plan — the commonest operation in the corpus. A wing as plan *geometry* (the arm of an L) is not an edit: it is footprint, this system's immutable input |
| **courtyard** | `box` | ❌ | a through-void. **0 voxels in 4,324,919** |
| **passage** | `box` | ❌ | a through-void; 0 voxels |
| **light well** | `cylinder` | ❌ | a through-void; 0 voxels |
| **arcade** | `box` | ❌ | an overhang, not a void — but still outside the height field: 71 voxels in 4.3M, **0.0016%** |

So the ticket's requirement is met in the sense that matters: every name is *expressible* (the four
voids cut in the SDF like any CSG solid) and the table is honest that four of them have **zero
training signal on this corpus**. "Representable" and "learnable" are different claims, and
conflating them is how a vocabulary gets adopted that no generator can ever populate.


## The rest of the list

**Constrained geometry per type / invalid references** — `op_problems(op)` returns everything wrong
with one operation as sentences; `program_problems(ops)` prefixes each with the offending index.
Previously a `layer` with no polygon was discovered by crashing inside a prism, which is a stack
trace rather than a diagnosis. A core operation declared `mode="add"` is now a *spec violation*,
because it would also break commutativity.

**Canonical normal form** — `canonical_form(ops)` sorts the operations by their own serialised
geometry, so two spellings of the same building compare equal without compiling either. It
**refuses** a non-commuting program rather than sorting an ordered stack and silently changing what
it denotes.

**Equivalence** — `equivalent(base, a, b)` decides on the *geometry*: compile both, compare
occupancy. The canonical form is the cheap syntactic test; this is the semantic one, and it catches
what a sort cannot — an operation that removes nothing because another already removed it.

**Deletion** — `EditableBuilding.remove(i)` deletes *any* operation, not just the last. `undo()` can
only unwind from the top, which cannot serve #3's edit locality, where a user re-rolls one decision
and everything unrelated survives. ⚠️ A negative index is **refused** rather than wrapped: Python
would delete the last operation for `remove(-1)`, a silent wrong answer when the caller passed an id
it failed to resolve.

**Parent / support / containment relations** — 🔑 **none are needed, and that is the answer rather
than an omission.** In a subtract-only commuting algebra every operation is independent of every
other: there is no ordering to encode, so no parent-child structure to maintain, and no invalid
reference an operation can hold *to another operation*. Containment within a single operation is
already carried by polygon rings (outer, then holes). If the volumetric tier is ever trained, this
answer expires with it — an additive operation must land on something, and that is when support
relations become real.


## ⚠️ "Wings" means two things, and the distinction is load-bearing

A **height** wing — one part of the plan sitting lower or higher than another — is a `Layer`, and it
is the commonest thing in this corpus: #10's recovered programs average **3.06** of them per
building. Measured on the 411 carve-needing held-out buildings, distinct height plateaus covering
≥5% of the footprint each:

| arm | plateaus / building | ≥2 | ≥3 | plan below top level |
|---|---|---|---|---|
| the real building | **3.34** | 0.642 | 0.472 | 0.768 |
| program label (sees GT) | 4.12 | 0.769 | 0.633 | 0.732 |
| **CE + median** *(the served arm)* | **3.12** | 0.691 | 0.501 | 0.690 |
| **#6 program arm** | **1.91** | 0.635 | **0.251** | 0.488 |
| blockout (the input) | 1.00 | 0.000 | 0.000 | 0.000 |

🔑 So stepped massing is **not** a missing capability — the served arm produces it at close to GT
rate. It is the **#6 program arm** that flattens it, which is the slot collapse that ticket already
recorded (1.19 slots used against 3.06 in its own labels). A reader looking at the demo and seeing
only a roof has almost certainly selected `heightmap_program`.

A **plan** wing — the arm of an L — is a different thing and is not an edit at all: it is footprint,
and the footprint is immutable input.


## What this ticket does NOT deliver

- **Resampling** and **local regeneration** are not addressed. Both are about *re-running a
  generator over part of a building*, which needs #3's gesture/scope contract and the generator
  itself; the algebra only had to make them well posed, which commutativity does.
- **The polygon vertex budget is still unstarted** (the owner's ⚠️ on this ticket, and #128's).
  Regions are recovered as exact voxel-boundary rings, not simplified polygons, so the real DSL
  token cost of a program is unknown. That is the next concrete piece of #4 and it is measurable.
- **The volumetric tier is declared, not exercised.** No test carves a real courtyard, because no
  building in this corpus has one.


## What follows

- #7 (validity gates) and #9 (multi-footprint coordination) are unblocked by this together with #3.
- 🔑 The commutativity result is what #3's *edit locality* invariant will rest on: "preserve
  unrelated state" is a theorem here, not a feature to implement.
- The vertex budget is the next measurable question inside #4.
