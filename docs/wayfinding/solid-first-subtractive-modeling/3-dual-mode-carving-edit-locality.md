# #3 — Dual-mode carving and edit locality

*Effort: solid-first semantic architectural carving. Opened 2026-07-15, grilled 2026-09-02. The ticket
[#4](4-edit-algebra.md), [#7](https://github.com/danvisai/SDFusion/issues/7) and
[#9](https://github.com/danvisai/SDFusion/issues/9) were waiting on together.*

> What precise contract should unite autonomous footprint-to-program generation with user-guided
> rough add/subtract interpretation? Define gesture input, semantic completion, accept/reject
> behavior, confidence and alternatives, rerolling, edit scope for one footprint versus a selected
> block, the edit-locality invariant, preservation of unrelated state, failure recovery, and the
> immediate-preview/asynchronous-finalization boundary.

Resolved by interview (`/grilling`) with the ticket owner, not by measurement — this is an
architecture decision, not an experiment. Facts cited below were checked against the repository
during the session; the decisions themselves are the owner's.


## 🔑 "Add" reopens exactly the ordering #4 just closed

#4 proved the core algebra commutative *because* every core operation could only lower a column
(`Layer` reads as `min`). The owner's intent for this ticket is a real, learnable **add** — raising a
column, the mirror of the existing lower — for the same core kinds. That is a deliberate, informed
trade: mixing add and subtract reopens order-dependence for exactly the reason #4 documents (`union`
does not commute with `subtract`), and the decision here is to accept that rather than avoid it.

**Decision: `layer` and `ramp` become bidirectional** (`mode="add"` alongside the existing
`mode="subtract"`). `cut_roof` stays subtract-only — its additive mirror already exists as the
volumetric tier's `gable`/`hip` kinds, and stays there. A program that mixes add and subtract replays
in **insertion order** where regions overlap; a subtract-only (or, incidentally, an all-additive)
sub-program stays commutative among itself, inheriting #4's result rather than re-proving it. The
existing `commutes`/`canonical_form` predicates already classify a mixed program as non-commuting and
already refuse to compute a normal form for one — no change to that logic is needed, only to the
kind table that currently forbids `add` from ever reaching it.

True volumetric addition — an overhang, a floating volume, anything a height field cannot represent
as one value per column — is explicitly **out of scope**. That is not a training limitation this
ticket defers "for now": it is what the representation cannot do at all, independent of how good the
generator gets.


## The backbone is the height-map generator, not the model that was closed the same day

🔑 Checked live during the session: the model actually training on this branch is the height-map
generator ([#138](https://github.com/danvisai/SDFusion/issues/138)/
[#139](https://github.com/danvisai/SDFusion/issues/139)'s line) — 2.5-D, one height per column, needs
no codec. The commit immediately before it on this branch is **"Close map #87: token-order alignment
did not open a usable band"** — the genuinely voxel/SDF-style model (decodes through a dense field and
marching cubes) was closed negative *that same day* (only ~2.8% of buildings in #92/#93's own
measurement genuinely improved). The owner confirmed the height-map generator, not the closed model,
is the intended target — which is also why true volumetric add is out of scope above: the confirmed
backbone structurally cannot do it, regardless of training.


## The aesthetic is deliberately blocky, and that simplifies the boundary question

The owner's stated goal is a Minecraft-like blocky look — sharp, grid-aligned forms — not smoothed
organic blending. That resolves what "clean edges" means for a newly added or removed block: **grid
snap, not smoothing**.

**Decision:** a new op's region geometry is rounded to the module's existing voxel pitch before it is
stored, so additions and removals land on a consistent grid. `EditOp.smooth` keeps defaulting to 0
(hard CSG) and is untouched by snapping. A **learned** generative snapping/cleanup component is an
explicit, named future enhancement the owner wants ("if we can do a learned generative component it
would be great") — but it needs training data that does not exist yet ([#5](
https://github.com/danvisai/SDFusion/issues/5)'s job), so it is not required here. Deterministic
grid-snap ships now; the learned version is future work, not blocked on this ticket but not delivered
by it either.


## Scope: one footprint. Block coordination is #9's, explicitly

The issue text asks for "edit scope for one footprint versus a selected block." **Decision: #3
defines the single-footprint contract only.** Multi-footprint/block coordination — whether buildings
may even interact — is left entirely to
[#9](https://github.com/danvisai/SDFusion/issues/9), consistent with this map's own standing rule
that selected-block edits "may not silently fuse solids across footprint/property boundaries," and
with no adjacency/party-wall code existing anywhere in the codebase today (checked: every building in
`town_generate_service.py` generates independently, on a decorrelated seed, with no cross-building
term).


## Data sourcing is #5's job, not this ticket's

The owner's own framing for how "add" gets taught ("we need to scrape data from the web to understand
and teach architecture") is, word for word, [#5](https://github.com/danvisai/SDFusion/issues/5)'s
mandate ("Audit Data for Recoverable Architectural Programs" — 3D BAG, OSM/CityGML, IFC void
semantics, BuildingNet, ReLoD3, TUM2TWIN, ArchiSet, procedural generation are its named candidates).
**Decision:** this ticket defines an algebra that is *trainable once #5 supplies positive add/void
supervision* — the current corpus has zero such examples (0 through-void voxels, 0 raised columns,
per #4/#10's measurement) — without itself specifying what to scrape or how.


## One shared contract, not one merged generator

`interpret_mass()` (the existing rough-carve completion path) and the arm-based autonomous generator
are, and remain, two separate code paths. **Decision:** #3 defines a shared contract — one operation
vocabulary, one set of validity rules, one locality invariant, one accept/reject/reroll shape — that
both must satisfy, rather than a literal code-level merge into a single generation function. A code
merge is a real engineering project of its own and is not this ticket's claim.


## Every operation gets a stable identity

Checked during the session: an `EditOp` today has no id, only its position in a list —
`EditableBuilding.remove(index)` shifts every later index, and the one existing rough-carve prototype
(`sculpt.html`) already works around this informally with its own `grp` tag rather than relying on the
backend's index. **Decision:** every operation carries a stable id, auto-assigned when not supplied,
surviving serialization. "Reroll one decision, everything unrelated survives" — #4's own forward
pointer to this ticket — is what that id makes checkable rather than aspirational.


## The edit-locality invariant, precisely

This is the actual crux of the ticket's title, restated now that mixing add and subtract reopens
ordering (#4's free commutativity theorem no longer covers the whole space):

- Every operation carries a stable id.
- Undoing or rerolling operation X removes or replaces **exactly** operation X and nothing else.
- Two operations whose regions do not overlap compose identically regardless of order — free,
  inherited from #4, unchanged by this ticket.
- Two operations whose regions **do** overlap compose in insertion order: an operation's compiled
  contribution can only be changed by an operation inserted **before** it, never one inserted after
  it, and never by anything on an unrelated part of the building.

Deliberately **not** append-only: a user re-rolling "the wing I just added" needs to replace at that
operation's position, not only ever add to the end — append-only would break the exact case #4 named
as the reason locality matters.


## What this ticket explicitly leaves not-yet-specified

The issue asked for gesture input, accept/reject behavior, and confidence/alternatives. **Decision:
none of these are pinned down here.** The one interactive prototype that exists (`sculpt.html`) is
explicitly disowned by the owner ("these features are from a demo i am not working with"); the demo
they *are* running (`town_generate_service.py`, port 8767) has no carving UI at all today. Locking in
literal gesture/accept-reject/confidence mechanics against either would be guessing. #3 commits only
to the abstract shape every future interaction layer must honor: *some* mechanism produces or
discards a candidate operation, and whatever that mechanism is, it must respect the locality invariant
above and support reroll-by-id. A completion that fails validity
(`op_problems`/`program_problems` today; [#7](https://github.com/danvisai/SDFusion/issues/7)'s
gates later) is never committed — building state stays untouched — and exact failure messaging is
left to whoever builds that layer. The preview/finalization boundary is not revisited here; the
project's existing domain-model definition for it stands.


## Tickets this spawned

Broken into five tracer-bullet slices against the one seam this contract lives in (the SDF edit
algebra module — `ALGEBRA`/`EditOp`/`op_problems`/`commutes`/`canonical_form`/`EditableBuilding`):

- [#140](https://github.com/danvisai/SDFusion/issues/140) — Make Layer and Ramp Ops Bidirectional in
  the Core Algebra
- [#141](https://github.com/danvisai/SDFusion/issues/141) — Give Every Edit Operation a Stable
  Identity
- [#142](https://github.com/danvisai/SDFusion/issues/142) — Snap New Operation Geometry to the
  Working Grid
- [#143](https://github.com/danvisai/SDFusion/issues/143) — Refuse to Commit an Invalid Operation to
  a Building
- [#144](https://github.com/danvisai/SDFusion/issues/144) — Prove the Edit-Locality Invariant on a
  Mixed Program (blocked by #140, #141)

All `ready-for-agent`.


## What follows

- [#7](https://github.com/danvisai/SDFusion/issues/7) (validity gates) and
  [#9](https://github.com/danvisai/SDFusion/issues/9) (multi-footprint coordination) are unblocked by
  this ticket directly. Per the owner: #9 feeds
  [#8](https://github.com/danvisai/SDFusion/issues/8) (the falsifiable-proof spec), which feeds
  [#2](https://github.com/danvisai/SDFusion/issues/2) (the integration boundary) — this ticket is the
  single door on that chain.
- True volumetric add (overhangs, floating volumes) remains unaddressed and needs a representation
  effort of its own, not further training of the confirmed backbone.
- A learned (rather than deterministic) boundary-cleanup component remains a named, wanted, unbuilt
  future enhancement, blocked on #5.
