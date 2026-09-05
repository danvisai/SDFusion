# #7 — Validity gates and visual carving traces

*Effort: solid-first semantic architectural carving. Opened 2026-07-15, grilled 2026-09-02. Blocked
by [#4](4-edit-algebra.md) and [#3](3-dual-mode-carving-edit-locality.md), both closed; blocks
[#8](https://github.com/danvisai/SDFusion/issues/8) and
[#10](https://github.com/danvisai/SDFusion/issues/10) directly.*

> What syntax, architectural-program, and final-geometric validity gates should every generated or
> edited program pass, and how should architectural quality, diversity, control, edit locality, and
> failure rate be evaluated? Define the required validation/final visual carving trace, fixed views
> and overlays, hard geometry checks, human evaluation, direct prior-work baselines, and rules
> preventing appearance from hiding geometric failure.

Resolved by interview (`/grilling`) with the ticket owner, not by measurement — this is an
architecture decision, not an experiment. Facts cited below were checked against the repository
during the session; the decisions themselves are the owner's.


## What already existed, and was adopted rather than re-decided

A survey of the repository during the session found several of #7's named items already settled by
prior tickets, carried forward here unchanged rather than re-litigated:

- **Syntax gates** — `op_problems`/`program_problems` (`scene/sdf_edit.py`), auto-enforced on
  `EditableBuilding.add` since [#143](https://github.com/danvisai/SDFusion/issues/143).
- **Quality and failure-rate metrics** — `missing`/`extra`/`vol_iou`/`fp_iou`/`collapse_rate`,
  established across [#10](https://github.com/danvisai/SDFusion/issues/10)/
  [#126](https://github.com/danvisai/SDFusion/issues/126)/
  [#127](https://github.com/danvisai/SDFusion/issues/127). Eval-time metrics only — never gates.
- **Prior-work baselines** — ArcPro, CoMa, CityGenAgent, Building-Gym, ShapeAssembly/CSG, fully
  specified with stated comparison axes in
  [130-baselines-diffusion-curriculum.md](130-baselines-diffusion-curriculum.md). None expose a
  program-level validity notion to compare against, so validity gates are this project's own
  contribution rather than a baseline-comparison axis.


## 🔑 Validity gates: three layers, only two of which block

**Syntax** stays exactly as [#143](https://github.com/danvisai/SDFusion/issues/143) left it —
`op_problems`, auto-enforced per operation on append.

**Architectural-program**: `commutes`, `is_height_map_representable`, and `program_problems` exist
today as separately-callable utilities, invoked nowhere automatically as a bundle. **Decision:**
combine them into one gate, enforced at **finalize time only** — not on the fast preview path, per
#3's immediate-preview/asynchronous-finalization boundary. On failure it returns a report (a list
of problem strings, mirroring `op_problems`'s own convention) rather than raising: a single
malformed op mid-edit is a bug worth raising on (#143's own case), but a fully-composed program
failing at finalize is more often something a human needs to see and decide about, per #3's "a
completion failing validity is never committed."

**Final-geometric — containment, redefined**: [#10](https://github.com/danvisai/SDFusion/issues/10)/
[#131](https://github.com/danvisai/SDFusion/issues/131)'s containment guarantee — a fitted region
may only shrink toward its GT target, never grow past it — only makes sense when a real building
exists to fit against. A freely authored or generated program has no GT. **Decision:** the
generalized rule is containment against the footprint's own envelope, not any target shape, with
two parts:

1. The compiled program's occupancy — core ops and any volumetric interior additions alike — must
   stay within the footprint's plan outline and declared height range. This is the absolute bound;
   nothing, interior or exterior, may cross it.
2. The compiled building's exterior must claim the **entire** footprint boundary at ground level —
   no gap anywhere around the perimeter. A program that carved the whole perimeter away, leaving an
   island in the middle, fails this even though it never left the footprint.

Interior geometry — a courtyard, a void, a second free-standing architectural element — is exempt
from rule 2 (it never has to touch the edge) but not from rule 1 (it still can't cross the outer
bound). This is explicitly this ticket's territory, not
[#9](https://github.com/danvisai/SDFusion/issues/9)'s: #9 owns whether and how *separate*
footprints coordinate; an interior structure within one footprint's own program is this ticket's
concern regardless. No hard geometry checks beyond containment are needed now — the height-map core
kinds (`layer`/`ramp`/`cut_roof`) cannot produce floating disconnection by construction, and true
volumetric add (the only tier that theoretically could) is already ruled out of scope by #3/#140.


## Evaluation dimensions: two adopted, two deferred, one retired as a metric

- **Quality & failure rate** — already covered (see above), adopted as-is.
- **Diversity** — 🔑 **deferred.** No trained generator arm on this map has passed its bar in five
  attempts ([#6](https://github.com/danvisai/SDFusion/issues/6)/
  [#129](https://github.com/danvisai/SDFusion/issues/129)/
  [#132](https://github.com/danvisai/SDFusion/issues/132)/
  [#138](https://github.com/danvisai/SDFusion/issues/138)/
  [#139](https://github.com/danvisai/SDFusion/issues/139)); specifying a diversity metric against a
  generator that has never produced a passing sample risks specifying the wrong thing. Revisit once
  an arm clears the bar.
- **Control** — **deferred** to whichever future ticket defines the interaction/gesture layer,
  which #3 explicitly left not-yet-specified. Captured only informally here, via the human-eval
  rubric's third question.
- **Edit locality** — ⚠️ **retired as a scored metric.**
  [#144](https://github.com/danvisai/SDFusion/issues/144) already *proves* the invariant
  structurally (a removed operation only changes the compiled contribution of a later operation
  whose region genuinely overlaps it). A structural guarantee is strictly stronger than a
  statistical one; a "% of voxels unaffected" score would only restate what #144 already proves
  exactly, with less rigor. No further metric is needed.


## The visual carving trace

Every montage builder in the repository today (`eval_massing_arms.py`, `recover_massing_programs.py`)
compares *final* results side by side; the operation sequence appears only as a text label
("Layer > Ramp > CutRoof"), never rendered step by step. **Decision:** build a genuinely new
per-step renderer — one frame per operation, highlighting the voxels *that operation's own
application toggled* (the same occupancy-delta concept #144's `_contribution` computes), at 4
fixed views (front/back/left/right oblique, single fixed camera per view, no per-mesh rescaling —
the existing montage convention, carried forward). 🔑 This is produced as a **standard artifact for
every finalized program**, not a debug-only tool: that is what makes it usable as #7's own "final
visual carving trace," and it doubles as the visual input the human-evaluation rubric is judged
against. A trace that only shows the end state can't answer "did this operation do what it claims"
— exactly the failure mode #7's own phrasing (appearance hiding geometric failure) is guarding
against.


## Human evaluation

Precedent ([#127](https://github.com/danvisai/SDFusion/issues/127)) was informal: the owner looking
at a rendered montage and giving a plain yes/no verdict against a stated scope sentence, no written
rubric, no tool. **Decision:** formalize into a fixed, minimal rubric — three yes/no questions per
building, recorded against that building's visual carving trace output:

1. Does it look like a building?
2. Are there visible geometric artifacts?
3. Does the edit match what was requested?

These map one-to-one onto quality, appearance-hides-failure, and control — the three dimensions
that don't already have an automated metric or an explicit defer. The rubric's wording must
distinguish the rater's judgement from any automated metric already computed for the same building,
matching the distinction #127 already drew between the owner's verdict and the analyst's own
reading of the same images.


## ⚠️ Rules preventing appearance from hiding geometric failure

The massing-arm eval already follows an implicit rule: no per-mesh rescaling (so volume loss stays
visible), and surface roughness is a guard only, explicitly excluded from ranking. **Decision:**
state this as a general, standing rule for *all* future visual/eval work on this project, not just
that one script — including the new visual carving trace above. Any evaluation or rendering
pipeline built after this ticket inherits it by default rather than having to rediscover it.


## Tickets this spawned

Broken into four tracer-bullet slices, `ready-for-agent`:

- [#145](https://github.com/danvisai/SDFusion/issues/145) — Bundle the Architectural-Program Gate
  into a Finalize-Time Check
- [#146](https://github.com/danvisai/SDFusion/issues/146) — Generalize Containment into a
  Footprint-Boundary Gate
- [#147](https://github.com/danvisai/SDFusion/issues/147) — Build the Per-Step Visual Carving Trace
  Renderer
- [#148](https://github.com/danvisai/SDFusion/issues/148) — Formalize the Human-Evaluation Rubric
  as a Repeatable Artifact

Not ticketed, by decision: diversity and control metrics (deferred above), and the
appearance-hiding-failure rule (a standing principle, not a build task).


## What follows

- [#8](https://github.com/danvisai/SDFusion/issues/8) (the falsifiable-proof spec) and
  [#10](https://github.com/danvisai/SDFusion/issues/10) (program recovery, already closed but
  listed as blocked by this ticket in the map) are unblocked directly.
- Diversity and control metrics remain open questions, explicitly deferred rather than guessed —
  revisit diversity once a generator arm passes its bar, and control once the interaction layer is
  specified.
