# #9 — Multi-footprint coordination

*Effort: solid-first semantic architectural carving. Opened 2026-07-15, grilled 2026-09-03. Blocked
by [#4](4-edit-algebra.md) and [#3](3-dual-mode-carving-edit-locality.md), both closed; blocks
[#8](https://github.com/danvisai/SDFusion/issues/8) and
[#2](https://github.com/danvisai/SDFusion/issues/2) directly.*

> How should a selected set of immutable footprints coordinate height rhythm, roof family,
> setbacks, courtyard patterns, orientation, and style while preserving separate per-footprint
> programs, edit locality, and independent validity? Decide among an explicit block program,
> relational graph, shared latent, constrained joint decoder, or a staged combination, and define
> what group edits may and may not change.

Resolved by interview (`/grilling`) with the ticket owner, not by measurement — this is an
architecture decision, not an experiment. Facts cited below were checked against the repository
during the session; the decisions themselves are the owner's.


## 🔑 The mechanism is a scoring bias inside an existing fitter, not a new model

Of the ticket's five named options, three were ruled out on a fact checked live during the
session: **shared latent** and **constrained joint decoder** both require a trained joint
architecture over multiple buildings, and no cross-building training data or code path exists
anywhere in the repository — [#3](3-dual-mode-carving-edit-locality.md) already confirmed every
building in `town_generate_service.py` generates independently, on a decorrelated seed, with no
cross-building term. Specifying either now would repeat the exact mistake this map keeps catching
elsewhere ([#6](6-program-generator.md)/[#129](129-classified-plane-parameters.md)/
[#132](132-overcarve-and-assignment.md)): aiming a fix at a layer that cannot currently learn
anything. A general **relational graph** is also more structure than the single-footprint algebra
needed — [#4](4-edit-algebra.md) found "no parent/support/containment relations are needed, and
that is the answer rather than an omission."

**Decision: an explicit block program**, applied now, deterministically, with no new training. A
relational graph is the named future upgrade, layered *on top of* the block program rather than
replacing it — useful specifically for pairwise, adjacency-based relations (e.g. party-wall
alignment between footprints that actually touch), which a block-wide scalar/categorical object
cannot express.

The block program does not reach a footprint's geometry by itself. **It becomes a scoring bias
inside [#10](10-program-recovery.md)'s existing constrained beam-search fitter** — the same
deterministic search that already recovers a typed program from a height map at 3D IoU 0.9970,
0.2 s/building, closed and working today. The learned program generator that would be the "natural"
place to accept this conditioning instead ([#6](6-program-generator.md)'s family) has **not passed
its bar in five attempts** ([#129](129-classified-plane-parameters.md)/
[#132](132-overcarve-and-assignment.md)/[#138](138-type-head-imbalance.md)/
[#139](139-assignment-temperature.md) all NOT MET) — building on it now would specify against a
model that doesn't reliably work. The fitter route costs nothing to ship and the generator can
absorb the same bias later once it clears its bar; #9 is not blocked on that happening.

The same mechanism serves **both** modes: fitting a program to a freshly generated height map
(autonomous generation of a new coordinated block) and re-fitting an existing footprint's program
after a group edit (user-guided re-coordination). One mechanism, not two, per
[#3](3-dual-mode-carving-edit-locality.md)'s own "one shared contract, not one merged generator"
rule.


## The four coordinated axes, mapped onto the existing algebra

| axis | maps to | status |
|---|---|---|
| height rhythm | shared pattern over per-footprint `layer` step heights | live now |
| roof family | shared categorical over `cut_roof` / `ramp` / flat | live now |
| setbacks | shared inset pattern for `layer`-as-setback | live now |
| orientation | shared `ramp` azimuth target ([#129](129-classified-plane-parameters.md)'s
  re-parametrisation) | live now |
| courtyard patterns | volumetric tier | ⚠️ deferred — **0 through-void voxels** in the corpus
  ([#4](4-edit-algebra.md)); needs [#5](https://github.com/danvisai/SDFusion/issues/5)'s data |
| style | not in the project glossary | ⚠️ deferred, undefined — resolve alongside courtyard once
  [#5](https://github.com/danvisai/SDFusion/issues/5) supplies real supervision to define it against |

**Decision: the four live axes ship now; courtyard patterns and style are named future axes**,
deferred exactly the way [#4](4-edit-algebra.md) declared the volumetric tier "unlearnable here"
rather than cutting it silently. Both are blocked on [#5](https://github.com/danvisai/SDFusion/issues/5)'s
data audit, which already covers per-building void/program supervision.

Each of the four live axes is **independently selectable** — a single coordinated edit can touch
just one axis (say, roof family) and leave the other three at each footprint's own value. There is
no monolithic "all four together" mode; that would cut against the locality-first pattern
[#3](3-dual-mode-carving-edit-locality.md)/[#4](4-edit-algebra.md) already established (reroll one
decision, everything unrelated survives).

Each axis is a **soft prior, never a hard match**. It biases the fitter's search toward the shared
value; it never forces a footprint into a program that would fail its own validity gates. A hard
requirement would directly contradict this ticket's own "preserving... independent validity" —
exactly the shape of failure [#129](129-classified-plane-parameters.md)/
[#132](132-overcarve-and-assignment.md)/[#138](138-type-head-imbalance.md) kept hitting when a
forced categorical choice didn't fit the geometry it was applied to.


## Re-derivation: full re-fit, not a nudge

**Decision:** every time a block decision changes, each affected footprint's program is **fully
re-fit** — the constrained beam search re-runs from scratch with the new bias — rather than
directly overwriting a parameter on the footprint's already-decided `EditOp`s. A direct nudge would
need a bespoke, hand-written safety check per axis (four of them) to confirm the overwritten value
still produces a valid building; a full re-fit gets that for free, because it is the same
search-and-validate path every footprint's program already goes through, for any axis, with no
extra code per axis. [#10](10-program-recovery.md)'s fitter is cheap enough (0.2 s/building) that
re-fitting a block's worth of footprints on every coordinated decision is not a real cost.


## What a group edit may never do

Restating four standing rules from [#1](https://github.com/danvisai/SDFusion/issues/1)/
[#3](3-dual-mode-carving-edit-locality.md)/[#4](4-edit-algebra.md)/
[#7](7-validity-gates-and-visual-carving-traces.md) rather than inventing new ones:

1. Never fuse solids across footprint/property boundaries — the map's own standing rule, unchanged.
2. Never alter footprint geometry. Footprints are immutable input, full stop, independent of #9.
3. A block-coordinated update to footprint X's program must still pass X's own
   [#7](7-validity-gates-and-visual-carving-traces.md) finalize-time gate before it commits —
   coordination does not get a side door around validity.
4. A block edit's effect on footprint X must land as ordinary stable-id `EditOp`s in X's own stack
   ([#3](3-dual-mode-carving-edit-locality.md)/[#4](4-edit-algebra.md)'s algebra), never as
   side-channel state living outside it.

Footprints are already guaranteed non-overlapping in plan (site-contextualized parcels, per the
input contract); combined with rule 3's per-footprint containment gate, no new block-level
geometric check is needed to keep separate buildings from touching — it falls out of what already
exists rather than being a new obligation of this ticket.


## Commit semantics: strictly per-footprint

**Decision: independent commit only, no all-or-nothing mode.** If the coordinated re-fit fails
[#7](7-validity-gates-and-visual-carving-traces.md)'s gate for one footprint in the block, that
footprint alone keeps its prior program — reported, not committed — while every other footprint in
the block proceeds. An atomic, all-or-block-nothing rule would let one recalcitrant footprint block
an entire street's edit, and directly contradicts the ticket's own phrasing:
"preserving separate per-footprint programs... and independent validity." This is
[#3](3-dual-mode-carving-edit-locality.md)'s own per-building rule ("a completion that fails
validity is never committed; building state stays untouched"), applied at N buildings instead of
one rather than re-decided.


## Block identity: ephemeral, not a new persisted entity

**Decision: no persisted "block" object.** A selected block is a UI-level, one-shot selection —
consistent with the project glossary's own definition ("an edit selection is one footprint or a
user-selected subset"), which does not imply the selection itself is a durable, storable entity.
Introducing one would be new schema/versioning state that [#2](https://github.com/danvisai/SDFusion/issues/2)
(the integration ticket) hasn't touched yet, and nothing forces it.

Each `EditOp` produced by one coordinated decision, across every footprint it touched, carries a
shared **`group_id`** tag — for traceability and "undo this whole coordinated decision together" —
without inventing a first-class block object whose lifecycle this ticket would then have to define.
Two overlapping block edits over time need no reconciliation logic beyond this: each is its own
full re-fit with its own `group_id`, and the most recent one simply wins for any footprint touched
by both, exactly as re-fitting from scratch already implies.


## What this ticket explicitly does not decide

- **Evaluation of coordination quality** is [#8](https://github.com/danvisai/SDFusion/issues/8)'s
  job — its own question text already names "multi-footprint coordination tests" as part of the
  falsifiable-proof package. #9 defines the mechanism, not its metric.
- **API/schema/service wiring** is [#2](https://github.com/danvisai/SDFusion/issues/2)'s job —
  #9 decides the architecture; #2 decides how it's exposed and versioned.
- **The exact scoring-bias formula** inside the fitter (weights, tie-breaking) is left to the
  spawned tickets below, at the same altitude [#3](3-dual-mode-carving-edit-locality.md) and
  [#7](7-validity-gates-and-visual-carving-traces.md) left their own mechanics to
  [#140](https://github.com/danvisai/SDFusion/issues/140)–[#144](https://github.com/danvisai/SDFusion/issues/144)
  and [#145](https://github.com/danvisai/SDFusion/issues/145)–[#148](https://github.com/danvisai/SDFusion/issues/148).
- **The relational-graph upgrade's data need is real but unscoped.** [#5](https://github.com/danvisai/SDFusion/issues/5)'s
  audit is limited to per-building void/program supervision (3D BAG, OSM/CityGML, IFC, BuildingNet,
  ReLoD3, TUM2TWIN, ArchiSet) — none of it is parcel-adjacency or party-wall metadata, which is
  what a relational graph actually needs. Naming this explicitly rather than assuming #5 covers it
  avoids repeating the "assumed vs. measured" mistake this map has already caught itself making
  ([#126](126-massing-scoring.md) on massing, [#130](130-baselines-diffusion-curriculum.md) on
  curriculum). No ticket is opened for it yet — there is nothing ready-for-agent to specify until a
  concrete adjacency data source is found.


## Tickets this spawned

Broken into three tracer-bullet slices against the one seam this contract lives in
([#10](10-program-recovery.md)'s fitter and the [#4](4-edit-algebra.md)/
[#7](7-validity-gates-and-visual-carving-traces.md) algebra/gate it re-fits through), `ready-for-agent`:

- [#149](https://github.com/danvisai/SDFusion/issues/149) — Add Block-Level Scoring Bias to the
  Program Fitter
- [#150](https://github.com/danvisai/SDFusion/issues/150) — Define the Explicit Block Program and
  Selective Per-Axis Application
- [#151](https://github.com/danvisai/SDFusion/issues/151) — Tag Coordinated Edits with a Shared
  Group Id and Commit Independently


## What follows

- [#8](https://github.com/danvisai/SDFusion/issues/8) (the falsifiable-proof spec) and
  [#2](https://github.com/danvisai/SDFusion/issues/2) (the integration boundary) are unblocked by
  this ticket directly.
- Courtyard patterns and style remain named, deferred axes, both waiting on
  [#5](https://github.com/danvisai/SDFusion/issues/5)'s data audit.
- The relational-graph upgrade remains real future work with a genuinely unscoped data
  prerequisite — not blocked on anything named here, but not ready-for-agent either.
