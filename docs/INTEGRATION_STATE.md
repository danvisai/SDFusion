# Integration state: where the massing pieces are, and which ones have never met

*Written 2026-08-31. A standing map of how the massing work wires together, kept because the answer
to "how do we connect these" turned out to be much narrower than it looks from the ticket titles.*

⚠️ **Read this before opening [#2](https://github.com/danvisai/SDFusion/issues/2)** ("Define
Integration with the Existing Recipe and SDF Stack"). Most of what that ticket imagines needs
defining already exists and is wired. The real gap is one function call wide.


## The one-sentence version

**The generator predicts a program, throws it away, and serves the height map it compiled** —
so the demo's editable representation and its generated geometry have never actually met, even
though both halves are built, tested, and running.


## What is already wired

**The service.** `scripts/server/town_generate_service.py` serves seven massing arms
(`ARM_ORDER`): `envelope`, `heightmap_mode`, `heightmap_median`, `heightmap_slope`,
**`heightmap_program`**, `retrieval`, `a2`. Weights resolve through a fallback chain —
`weights/massing-heightmap/` (the published staging tree) then `outputs/height_map_generator/` —
so a retrain is picked up without a copy step, and an absent file simply means the arm is not
offered. 🔑 **The program arm already has a slot in the product**, pointed at #6's checkpoint.

**The edit stack.** [#128](https://github.com/danvisai/SDFusion/issues/128) made a layer program
first-class in the recipe/SDF path, and it is closed:

* `scene/sdf_primitives.py:138` `sdf_polygon_prism`, `:163` `sdf_plane_halfspace` — the primitives
  `Layer` and `Ramp` needed and the palette did not have.
* `scene/sdf_edit.py:753` `layer_program_to_ops` — a recovered program to `EditOp`s.
* `scene/sdf_edit.py:423` `EditableBuilding` — undo, re-roll, delete any operation (not only the
  last, because [#4](wayfinding/solid-first-subtractive-modeling/4-edit-algebra.md) proved the
  algebra commutes).
* Verified: the composed SDF matches the voxel compiler, and a serialised program replays to the
  height map the fitter found.

**The scoring.** `scripts/foundations/eval_massing_arms.py` holds the metric definitions; the bar is
machine-checked in `verdict()`. See `CONTEXT.md` → *Reading the numbers*.


## The gap, precisely

`decode_prediction(out_k, fp, extent, "program", ...)` in
`scripts/foundations/train_height_map_generator.py` does this:

1. decodes the assignment (`decode_assignment`), the types, and the planes — **a program**;
2. calls `compile_program(...)` to turn that program into a height map;
3. **returns the height map.**

The program is constructed and discarded inside step 2. The service calls this function, so even the
`heightmap_program` arm serves a compiled surface with no operations attached to it. `EditableBuilding`
has never been handed a *generated* program — only a **recovered** one, which needs ground truth.

So the two halves are:

| | generated from a footprint alone | editable through the real stack |
|---|---|---|
| height map (#127, served) | ✅ ~0.1 s/building | ❌ no operations exist |
| recovered program (#10/#128) | ❌ needs GT to fit | ✅ full undo / re-roll / delete |
| **generated program** | **the missing cell** | |

⚠️ **The project's load-bearing claim — editable/reversible — is currently satisfied only by the
path that has no generator, and the path that ships has nothing to edit.**


## Every part of the missing link already exists

Nothing new has to be designed. The sequence is:

    program_predictions(ckpt, held)        # (assign, types, planes)   train_height_map_generator.py
      -> per-slot boolean masks            # (assign == k) & fp
      -> mask_to_rings(mask)               # exact boundary rings      scene/sdf_edit.py:615
      -> finalise_program(ops)             # op dicts carrying rings   recover_massing_programs.py
      -> layer_program_to_ops(...)         # EditOps                   scene/sdf_edit.py:753
      -> EditableBuilding                  # undo / re-roll / delete   scene/sdf_edit.py:423

Every function on that list is committed, tested, and in use. **No caller runs them in sequence.**

⚠️ Two things a first attempt will hit:

* `mask_to_rings` **raises** on a mask with more than one connected component, by design — a `Layer`
  is one polygon. A *predicted* assignment carries no such guarantee, so predicted slots need
  splitting (`mask_components_rings`) or rejecting. The recovery fitter splits components during the
  search; a generator does not.
* The rings will be **exact voxel traces at a median 94 vertices**, which
  [#131](wayfinding/solid-first-subtractive-modeling/131-vertex-budget.md) measured: droppable to 58
  for free, and **not** reducible below that by trimming without the surplus standing up as spikes.


## What this does NOT depend on

**A passing generator.** Three arms have now failed the bar
([#6](wayfinding/solid-first-subtractive-modeling/6-program-generator.md),
[#129](wayfinding/solid-first-subtractive-modeling/129-classified-plane-parameters.md),
[#132](wayfinding/solid-first-subtractive-modeling/132-overcarve-and-assignment.md) — three KILLs),
and the service already offers #6's checkpoint anyway. Wiring the program through would make the
demo's program arm **editable at whatever quality it currently has**, which is a separate axis from
whether it is good. It would also force the seam [#2](https://github.com/danvisai/SDFusion/issues/2)
has to specify to exist in code first, which is the cheaper order.

⚠️ It is worth being explicit that this is **not** a quality fix. #132's arm destroys ~26% of
buildings (`outputs/height_map_generator/worst_by_missing.png`); an editable destroyed building is
still destroyed.


## Demoable today, with no model at all

`execution/artifacts/program_recovery_714.json` holds a recovered program for all 714 pinned
buildings, and #128's path already turns any of them into an `EditableBuilding`. So a real building
can be shown decomposed into `Layer > Ramp > Ramp`, with operations toggled, deleted, or re-rolled
live through the SDF stack, using GT as a stand-in for the generator. That demonstrates the
load-bearing claim end to end without waiting on arm four.


## Checked, not assumed

⚠️ #132 changed `PLANE_DECODE` (pitch `median` → `q0.25`) and added `ASSIGN_DECODE`, and the service
imports `decode_prediction` from the training module — so those constants reach the demo. Verified
harmless: `ASSIGN_DECODE` is `"argmax"`, identical to the argmax it replaced, and `PLANE_DECODE` is
read only on a `class` plane head while the served `program` checkpoint is `regress`. **A future
change to either constant will silently change what the demo serves.** The decode now travels inside
each checkpoint (`plane_decode`, `assign_decode`) so a mismatch is at least visible in the artifact.

⚠️ And a standing hazard from
[the demo memo](wayfinding/solid-first-subtractive-modeling/): the demo serves **HTML from disk but
Python from memory**, so a running process can be several commits behind the file it is importing.
Check the process start time against the file mtime before believing a demo.
