# Reversibility options for a learned whole-volume voxel transform

**Research cutoff:** 2026-08-19

**Question:** What can be stored in the symbolic recipe so a learned whole-volume transform remains recoverable and editable rather than becoming a frozen field?

**Scope:** Factual option and constraint matrix for [Find a reversible recipe representation for a learned whole-volume transform](https://github.com/danvisai/SDFusion/issues/116). This note does **not** choose the later recipe posture.

## Bottom line

The three candidate representations preserve three different properties:

1. **Source recipe + immutable execution provenance** can replay, disable, or reroll one opaque learned transform. It does not expose the transform's roof, wing, courtyard, passage, or setback choices as independently editable decisions.
2. **A dense or sparse voxel residual** can reconstruct the exact canonical voxel state without retaining the model. It permits cell-level patching, but it is still baked geometry rather than the compact architectural decision state that `CONTEXT.md` calls a symbolic recipe.
3. **Distillation to a semantic architectural edit program** is the only candidate that can expose individual massing decisions with stable identities, parameters, relations, and deterministic realization. It is not guaranteed to reproduce an arbitrary voxel result exactly; inverse programs are latent, ambiguous, and bounded by the DSL's expressivity.

These are not interchangeable notions of reversibility:

| Property | Meaning in this project |
|---|---|
| **Exact reconstruction** | The same canonical output state can be recovered later. |
| **Undo/reroll** | The entire transform can be removed or sampled again. |
| **Recipe closure** | The stage consumes and emits symbolic decisions from which geometry is realized, rather than merely putting baked geometry inside a recipe-shaped container. |
| **Independent editability** | One decision can be changed or rerolled without destroying unrelated decisions. |

The accepted domain model requires the last two, not just the first two: the building is a compact symbolic recipe, `Editable / Reversible` means changing one decision without destroying the rest, and an arbitrary voxel mask is explicitly excluded from a constrained architectural volume ([`CONTEXT.md`](../../../CONTEXT.md#L59-L76), [`CONTEXT.md`](../../../CONTEXT.md#L218-L223), [`CONTEXT.md`](../../../CONTEXT.md#L246-L260)). Recipe closure is a demo-wrapper property rather than a paper claim, but that does not weaken its stated contract.

## What the current repository actually preserves

The earlier recipe path already demonstrates the intended state shape: Blender stores style, class, height, footprint, exact recipe parameters, and an ordered edit list, then remeshes from those values ([`mesh_sync.py`](../../../tools/blender_addon/generative_towns/mesh_sync.py#L1-L36)). `EditableBuilding` serializes primitive add/subtract operations and deterministically folds them over the base SDF ([`sdf_edit.py`](../../../scene/sdf_edit.py#L41-L62), [`sdf_edit.py`](../../../scene/sdf_edit.py#L101-L144)). Town export similarly reconstructs base recipe, sculpt operations, composer decisions, weather seed, and terminal mesh in a fixed order ([`town_export.py`](../../../scripts/server/town_export.py#L87-L150)). This is the local reference for decision-bearing state.

The current A2 town path does not yet meet that reference:

- It creates a footprint envelope, encodes it with Dora, performs Set-SDEdit, decodes the resulting latent to a complete `64^3` field, and then meshes that field ([`town_generate_service.py`](../../../scripts/server/town_generate_service.py#L220-L258)).
- The response contains vertices, faces, `vs_input`, and elapsed time, but no checkpoint digest, codec digest, source-encoding sample seed, projection seed/noise, strength, steps, guidance, region, frame version, or decoded-field digest ([`town_generate_service.py`](../../../scripts/server/town_generate_service.py#L129-L140), [`town_generate_service.py`](../../../scripts/server/town_generate_service.py#L253-L264)).
- The browser retains each result as a Three.js mesh group attached to a footprint; it does not retain an A2 operation record or field ([`town.html`](../../../scripts/server/web/town.html#L161-L163), [`town.html`](../../../scripts/server/web/town.html#L334-L350), [`town.html`](../../../scripts/server/web/town.html#L474-L480)).
- Most importantly for replay, `DoraCodec` owns a stateful NumPy generator. Its own contract says an encoding otherwise depends on every building encoded before it and provides `reseed()` to make a building encode a function of its inputs ([`shape_codec.py`](../../../models/shape_codec.py#L94-L103)). The town service does not call `reseed()` before `codec.encode` ([`town_generate_service.py`](../../../scripts/server/town_generate_service.py#L227-L236)). When supplied, the request seed therefore fixes the projection noise but not the encoded source tokens.

Consequently, the whole-volume transform cannot become more reproducible than its A2 input. A viable representation must either make A2 itself a replayable recipe operation or explicitly accept an opaque cached A2 field as its parent; the latter merely moves the frozen-field boundary upstream.

## Option matrix

| Candidate recipe payload | Exact canonical output | Compact per building | Survives missing model | Edit granularity | Source-recipe change | Downstream decision independence | Compatibility with current recipe definition |
|---|---|---:|---:|---|---|---|---|
| **Replay record:** source recipe revision + code/model/codec digests + settings + source-sampling state + seed/noise | Conditional: yes only in the pinned execution envelope; verify with an output digest | Seed form: yes; full noise/source latent: no longer especially compact | No | Enable/disable/reroll the entire transform | Re-execute whole transform; unrelated implicit features are not preserved | Decisions applied after massing can be replayed, subject to anchor revalidation | Meets the literal re-derivability clause if all artifacts remain available; does not expose the transform's implicit sub-decisions |
| **Dense residual:** canonical base digest + fixed frame/resolution + packed XOR/action/field delta | Yes for the stored canonical channels | Binary occupancy: small at `64^3`; continuous field: much larger | Yes | Individual cells or manually declared groups, not architectural decisions | Residual is valid only against its exact parent revision; rebasing is a new operation | Downstream stages can replay after the residual, subject to anchor revalidation | Conflicts with “compact decisions, not baked geometry” and the explicit avoidance of arbitrary voxel masks |
| **Sparse residual:** base digest + sorted changed cells/runs/tree + values | Same as dense when the encoding is lossless | Only when changes are sufficiently sparse/clustered | Yes | Same semantic limitation as dense residual | Same exact-parent and rebase problem | Same as dense residual | Same conflict; sparsity changes storage, not semantics |
| **Semantic program:** typed operations with IDs, constrained volumes, relations, parameters, seeds, and provenance | Only if the DSL can represent the result exactly; otherwise approximate | Yes | Yes, once distilled | Operation-level edit/delete/reroll with dependency checks | Re-realize and revalidate named operations; unaffected siblings can remain stable | Strongest: downstream choices can attach to operation IDs or realized support surfaces | Directly matches the accepted semantic architectural edit program and deterministic SDF/CSG realization |
| **Semantic program + residual escape hatch** | Yes if residual stores the remainder | Depends on remainder | Yes | Semantic for fitted operations; opaque for the remainder | Program edits also require residual invalidation/rebase | Partial | Above-`s*` residual massing remains a frozen field; calling it “refinement” does not make it a semantic decision |

## Option 1 — exact re-derivation from source recipe and provenance

### Minimum operation record

A replayable operation needs content identities, not mutable filenames or a human-facing version string. At minimum it must record:

```yaml
kind: whole_volume_transform.v1
source_recipe_revision: <content digest>
source_a2_operation:
  code_commit: <git object id>
  a2_weights_sha256: <digest>
  dora_code_commit: <digest>
  dora_weights_sha256: <digest>
  frame_and_rasterizer_version: <id>
  resolution: 64
  encoder_surface_sample_seed: <integer>
  n_coarse: 8192
  n_sharp: 8192
  region: <id>
  strength: <float>
  steps: <integer>
  guidance: <float>
  projection_noise: <tensor digest + durable blob>  # or weaker seed form
editor:
  code_commit: <git object id>
  weights_sha256: <digest>
  algorithm_and_schedule: <versioned settings>
  conditioning: <versioned values>
  noise: <tensor digest + durable blob>              # absent for deterministic correction
runtime:
  container_image_digest: <digest>
  hardware_backend: <recorded platform>
  deterministic_mode: <settings>
expected_source_field_sha256: <digest>
expected_output_state_sha256: <digest>
```

The source-encoding sample seed is separate from Set-SDEdit's projection seed. `SetSDEdit.project` can accept the actual noise tensor, which is stronger provenance than asking a future RNG implementation to recreate it from an integer ([`vecset_projection.py`](../../../models/networks/vecset_projection.py#L55-L75), [`vecset_projection.py`](../../../models/networks/vecset_projection.py#L78-L117)). The current A2 latent/noise shape is `2048 × 64` ([`precompute_vecset_latents.py`](../../../scripts/foundations/precompute_vecset_latents.py#L550-L562)); at float32 that is about **512 KiB**, versus **32 KiB** for one packed `64^3` binary occupancy grid. Storing the exact noise can therefore cost sixteen times the final binary state before the editor's own stochastic state is counted.

Even this record offers a bounded, not universal, guarantee. PyTorch's official reproducibility note says results are not guaranteed across releases, commits, platforms, or CPU/GPU even with identical seeds, and deterministic algorithms only constrain a specific software/hardware envelope ([PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness)). The expected source/output digests are therefore essential: they distinguish a verified replay from a plausible rerun. If cross-platform bit identity is required, storing the canonical output state is the only unconditional route.

### What remains independently editable

- The whole transform can be disabled, restored to its recorded run, or rerolled by changing its noise/seed.
- Footprint, height, region, or earlier massing decisions can be changed, but doing so invalidates the recorded output digest and reruns the **whole** transform. Replay provenance provides no mechanism to keep an unrelated learned roof while changing a learned courtyard.
- Detail, facade, material, weather, and ornament decisions that are explicitly later stages can remain separate and be replayed after massing. They must be revalidated if they are attached by absolute position or to a surface that moved. The current exporter establishes this stage order, but not a stable attachment/rebinding contract.
- The learned output's internal architectural choices are not independently addressable; the transform is one decision.

Thus this option gives strong auditability and whole-operation undo/reroll. Whether treating the entire learned field transform as one acceptable recipe decision is sufficient is a later human posture decision, not a fact established by reproducibility.

## Option 2 — dense or sparse edit residual

### Lossless occupancy residual

For a binary canonical state and an immutable parent revision, the minimal exact residual is a flip mask:

```text
target_occupancy = base_occupancy XOR flip_mask
```

`ADD` and `REMOVE` can be derived from base and target, so a two-bit ternary action lattice is useful for modeling or inspection but is not required for storage. At `64^3`:

- one packed occupancy or XOR mask is `262,144` bits = **32 KiB**;
- a two-bit `KEEP/ADD/REMOVE` lattice is **64 KiB**;
- sorted changed indices stored as `uint32` beat the 32 KiB dense mask only below **8,192 changed voxels**, or **3.125%** of the grid, before container/index overhead;
- delta coding, run-length encoding, and tree formats move the crossover according to clustering and entropy, so the prototype should measure actual residuals rather than assume “voxel” means “sparse.”

OpenVDB is an official example of the sparse alternative: it stores sparse grid topology in a hierarchical tree, represents uniform regions as tiles/background values, and supports active-value iteration ([OpenVDB overview](https://www.openvdb.org/documentation/doxygen/overview.html)). That can make clustered fields compact, but it does not add architectural identity or semantics.

If the canonical state selected by the neighboring representation decision includes a clipped continuous field/TSDF, the residual must preserve that channel too. A dense `64^3` field costs **512 KiB at float16** or **1 MiB at float32** before compression. An occupancy-only residual cannot claim exact reconstruction of a continuous field or of a mesh extracted from that field.

### What remains independently editable

- Exact undo is trivial: remove the residual and realize the parent. Exact redo is also portable because it no longer depends on model execution.
- Cells can be toggled, but a cell is not one of the project's architectural decisions. A labeled connected component is still an arbitrary mask unless its type has a distinct parameterization, relations, validity rules, and deterministic behavior.
- The residual must carry the exact parent recipe/state digest, frame, axis convention, resolution, and quantization. Applying it after a source footprint, height, normalization, or resolution change is not a valid edit; it is an unverified rebase.
- Later details can remain separate only if the residual is the massing stage and later stages are replayed. An editor applied after detail would bake those decisions into the field.

### Recipe-closure boundary

Putting a bitset, VDB tree, or float grid inside a JSON recipe makes the pipeline serializable but does not make the payload a symbolic decision. Under the current glossary it is baked geometry analogous to a frozen mesh: it preserves exact state while forfeiting independent architectural choices. Sparse and dense forms differ in cost, not in this boundary.

A residual can still be valuable as a **prototype evidence artifact**, replay cache, regression oracle, or terminal export. It becomes an accepted recipe representation only if the later human decision deliberately changes the current meaning of symbolic recipe/editability; this research does not make that decision by renaming the residual.

## Option 3 — distill to a semantic architectural edit program

The distilled result would replace the voxel field as authoritative state:

```text
A2 output field -> inverse program recovery -> canonical typed operation graph
                                           -> deterministic SDF/CSG realization
```

Each retained massing decision needs an operation ID, architectural type, add/subtract mode, constrained volume and parameters, parent/support/containment relations, constraints, seed/provenance, and a canonical ordering/equivalence rule. This is the representation already defined by the sibling [Specify Solid-First Semantic Architectural Carving](https://github.com/danvisai/SDFusion/issues/1) map and `CONTEXT.md`.

Primary-source precedents establish both feasibility and limits:

- ArcPro predicts a hierarchical architectural DSL from points and uses a learning-free interpreter to turn the program into a mesh; this demonstrates an architecture-specific inverse-program/compiler split, not arbitrary exact recovery of every input field ([ArcPro, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_ArcPro_Architectural_Programs_for_Structured_3D_Abstraction_of_Sparse_Points_CVPR_2025_paper.html)).
- ShapeAssembly programs expose a structured, parameterized subset of variability for editing and its official implementation includes differentiable program fitting to a target point cloud ([ShapeAssembly project and official implementation](https://github.com/rkjones4/ShapeAssembly)).
- PLAD treats the generating program as latent because real shapes generally lack paired programs; its pseudo-label and approximate-distribution regimes explicitly trade label match against shape-distribution match ([PLAD, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Jones_PLAD_Learning_To_Infer_Shape_Programs_With_Pseudo-Labels_and_Approximate_CVPR_2022_paper.html)).
- Program compactness is part of editability, not cosmetic compression: SIRI reports that large inferred CSG graphs become hard to interpret/edit and uses pruning, optimization, and grafting to obtain more parsimonious programs ([SIRI, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ganeshan_Improving_Unsupervised_Visual_Program_Inference_with_Code_Rewriting_Families_ICCV_2023_paper.html), [supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Ganeshan_Improving_Unsupervised_Visual_Program_Inference_with_Code_Rewriting_Families_ICCV_2023_supplemental.pdf)).
- BuildingSMART's IFC schema illustrates why identity and relations matter: an opening is an object with a predefined type and a one-to-one `IfcRelVoidsElement` relation to the element it subtracts from, rather than merely anonymous negative cells ([IFC 4.3 `IfcOpeningElement`](https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/IfcOpeningElement.htm), [`IfcRelVoidsElement`](https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/HTML/lexical/IfcRelVoidsElement.htm)).

### What remains independently editable

- A courtyard, passage, setback, roof cut, wing, or roof volume can be deleted, resized, retyped, or rerolled by operation ID, subject to its explicit dependency graph and global validity checks.
- Source footprint and height changes can trigger deterministic re-realization and constraint repair while preserving operations whose parameters/relations remain valid. “Preserve” must be tested; it is not automatic.
- Downstream detail decisions can attach to stable operation IDs, named faces/support relations, or regenerated semantic surfaces instead of raw voxel coordinates.
- The original voxel transform can remain as provenance/evidence, but geometry comes from the recovered program. If exact restoration still requires the field, the field remains an opaque second authority.

### Limits

Final geometry does not reveal a unique authoring history. Equivalent Boolean orderings, redundant operations, and different parameterizations can realize the same occupancy. A canonical normal form or equivalence-aware comparison is therefore required before “same program” is meaningful. Moreover, a DSL can preserve only the variability it represents. A massing-scale residual needed to recover accepted output is evidence of a representation ceiling, not detail below `s*`.

## Independent-editability matrix

Legend: **Y** = directly supported; **C** = conditional; **N** = not represented.

| Decision to change after the transform | Replay record | Dense/sparse residual | Semantic program |
|---|---:|---:|---:|
| Disable/restore the whole learned transform | Y | Y | Y |
| Reroll the whole learned transform | Y, while model/runtime survive | N without running the model again | Y at program or operation scope if the generator is retained |
| Change footprint or height and preserve unrelated learned massing features | N; reruns the opaque transform | N; residual is tied to the parent state | C; requires stable parameterization, dependency rules, and empirical locality |
| Change one learned roof/courtyard/passage/setback | N | N semantically; possible only as raw cell surgery | Y if represented as a distinct operation |
| Change a later facade/detail decision without changing massing | C; replay downstream stage | C; replay downstream stage | Y; replay downstream stage with semantic anchors |
| Change weather/material/appearance without changing geometry | Y if kept downstream | Y if kept downstream | Y |
| Reproduce exact accepted binary occupancy after model retirement | N | Y | C; only if exact representability is demonstrated |
| Explain which architectural decision caused a region | N | N | Y, subject to faithful semantic recovery |

## Evidence that must force rejection rather than a vocabulary exception

The later human decision should treat the following as red-line evidence. None is resolved merely by calling a field, cache, or mask a “recipe operation.”

### Replay-record failure

- A replay under the pinned runtime does not match the recorded source/output digest, including after fixing the current stateful Dora encode.
- Required A2/editor/Dora weights, code, container, or licensed artifacts cannot be retained for the lifetime promised by saved recipes.
- The only reliable replay payload becomes a cached source/output field or latent; that has converted the option into residual/frozen-state storage.
- Changing one source decision necessarily regenerates unrelated implicit massing choices, and downstream decisions cannot be rebound or validated without replacement.

### Residual failure

- The residual is the authoritative above-`s*` massing geometry and exposes no stable architectural operation IDs, parameters, or relations.
- A source-footprint, height, frame, or resolution edit silently applies the old residual instead of invalidating it against its parent digest.
- Exactness depends on storing a large continuous remainder after claimed semantic recovery. The remainder is not made non-destructive by compression or an architectural label.
- Validity can only be repaired by globally rewriting the residual, so a local user decision cannot preserve unrelated state.

### Semantic-distillation failure

- The accepted voxel results cannot be represented within preregistered massing-fidelity and human-visible gates without a substantial above-`s*` residual.
- Near-identical fields recover radically different programs under small perturbations, with no canonical/equivalence-aware stabilization.
- Recovered “types” are labels on generic masks and do not change parameterization, relations, constraints, validity, or downstream behavior.
- Editing/deleting/rerolling one operation changes unrelated realized geometry beyond declared dependencies, or routinely breaks footprint containment, connection, thickness, or watertightness.
- Downstream decisions cannot attach to stable semantic identities and must instead be baked into or resampled with the whole field.

### Universal closure failure

If a saved building can be restored only as a final field/mesh, if changing one named decision requires replacing the entire building, or if a later stage consumes the field and discards the decision record, the stage is a `Leak` under the current glossary ([`CONTEXT.md`](../../../CONTEXT.md#L246-L260)). The honest outcomes are to reject the route or explicitly reconsider the accepted domain model/ADR in a separate decision—not to weaken recipe closure inside an implementation ticket.

## Facts the later posture decision must settle

This research leaves three genuinely different policy choices open:

1. Is whole-operation replay/reroll enough for the voxel transform to count as one symbolic massing decision, even though its internal features are not independently editable?
2. Is an exact residual acceptable only as prototype/cache/export evidence, or may the project's definition of symbolic recipe be changed to admit baked voxel geometry?
3. What approximation ceiling and stability/locality evidence must semantic distillation pass before it becomes authoritative, and is any above-`s*` residual disqualifying?

Those are posture decisions for the blocked human ticket. The factual constraint is that no one representation simultaneously provides compact replay, model independence, exact arbitrary-field recovery, and semantic independent editability. The project must choose which guarantee is authoritative or reject the learned whole-volume route when the guarantees cannot be reconciled.
