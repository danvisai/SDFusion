# Voxelizing generated massing for generative building editing

**Research date:** 2026-08-18

**Question:** Given the current A2/Dora massing generator, is it technically sound to (1)
deterministically voxelize its result and then use a generative voxel editor to make the massing more
building-like, and/or (2) use an AI model for the initial mesh-to-voxel conversion?

**Scope:** representation and architecture research only. This note does not change `CONTEXT.md`, an
ADR, or implementation code.

**Source policy:** external claims use papers, author project pages, official repositories, and
official library documentation. Repository claims are linked to checked-in code and wayfinding
evidence.

## Executive verdict

1. **A deterministic mesh-to-grid step is technically sound, but it is unnecessary for the live A2
   path.** A2 already decodes a signed field `fld` on a `64³` grid, thresholds `fld <= 0` for its
   occupancy metrics, and only then extracts the terminal mesh. The clean seam is therefore
   **field → occupancy editor → surface reconstruction**, not field → mesh → voxel editor. Only an
   external mesh-only generator needs mesh voxelization.
2. **A generative occupancy editor is technically sound as a new, building-trained transform.** It is
   not a free post-process or a reusable TRELLIS checkpoint. The editor needs authentic A2 outputs,
   aligned real-building targets, footprint/height conditioning, an explicit edit/preservation mask,
   and hard validity checks. At `64³`, it can address **massing above `s*`**, not facade/detail below
   `s*`.
3. **Do not use AI merely to voxelize a known clean mesh.** Triangle/cell intersection, signed-distance
   sampling, parity, and winding-number methods provide deterministic targets without training error.
   Use a differentiable analytic voxelizer if gradients must reach mesh vertices. Use a learned model
   only when the intended task is actually **repair/completion of ambiguous, open, or corrupt
   geometry**; that is inference, not conversion.
4. **Do not keep only binary occupancy.** Use filled occupancy for bounded edits and solidity metrics,
   but retain a narrow-band SDF/TSDF or a separate query/surface decoder for meshing. Binary `64³`
   alone discards sub-voxel surface location and can delete thin features or fatten them, depending on
   the voxelization rule.

The smallest defensible experiment is therefore an **occupancy-space massing transform**, not an “AI
voxelizer”: consume the field A2 already produces, edit a masked `64³` solid occupancy conditioned on
the same footprint and height, reconstruct a continuous surface field, and compare against both A2 and
the footprint envelope on the full held-out set.

## What the current architecture actually provides

The live town path is already volumetric before it is a mesh:

1. It rasterizes the footprint and constructs a `64³` footprint-envelope SDF.
2. It meshes that envelope only because Dora's encoder consumes a surface.
3. It applies `SetSDEdit` to the encoded envelope.
4. It queries Dora back onto a `64³` signed field, `fld`.
5. It extracts a mesh from `fld`; evaluation occupancy is directly `fld <= 0`.

See [`town_generate_service.py`](../../scripts/server/town_generate_service.py#L196-L244). The codec
contract also exposes arbitrary point queries and defines `decode_grid` as those queries materialized
on a grid; the mesh and SDF are explicitly two projections of the same building
([`shape_codec.py`](../../models/shape_codec.py#L1-L25),
[`shape_codec.py`](../../models/shape_codec.py#L79-L111)).

This changes the proposed first step:

```text
current A2 path
footprint envelope → A2/Dora → signed field → marching cubes → terminal mesh

recommended editor seam
footprint envelope → A2/Dora → signed field ─┬→ solid occupancy → generative massing editor
                                             └→ narrow-band surface values ─────────────┤
                                                                  edited field → mesh

mesh-only fallback
external/generated mesh → deterministic mesh-to-field/occupancy → same editor
```

Meshing and re-voxelizing A2's own result adds a second sampling pass with no new information. This
repository has already validated that a controlled mesh→SDF path can add essentially no measured
roughness when it is configured correctly, but that result depended on a re-voxelized GT control
([ceiling probe](../wayfinding/crisp-massing-vecset/ceiling-probe-result.md)). Avoiding the round trip is
still strictly simpler and removes axis, sign, winding, and resampling failure modes.

One terminology correction matters: the live A2 service creates geometry for a new footprint, but it
is internally a **C1 projection from an envelope**, not sampling from pure noise. Replacing it with a
global voxel generator would conflict with accepted [ADR 0003](../adr/0003-two-claim-thesis.md). A
voxel editor can fit C1 if the same conditioned transform handles envelope-to-massing and user edits
while preserving untouched regions.

## Exact deterministic mesh-to-grid choices

“Voxelize” is underspecified. The implementation must choose whether a voxel represents surface
contact, a center classified as inside, or fractional solid coverage. These are different targets.

| target | deterministic algorithm | strength | characteristic failure |
|---|---|---|---|
| **surface shell** | For every triangle, visit cells in its bounding box and run a triangle–axis-aligned-box overlap test; mark every intersected cell. Conservative voxelization marks all contacted cells. | Works for open meshes because it makes no interior claim. Exact relative to the chosen cell-intersection convention. | Produces a shell, not a filled building. Conservative contact preserves thin surfaces but fattens them. |
| **solid binary occupancy** | First produce a closed surface shell, then classify cell centers by odd/even ray crossings, scan conversion, flood fill from the exterior, or winding number. | Direct input to binary/discrete editors and IoU/solidity metrics. | Strict parity/flood filling assumes a closed, consistently oriented surface; holes cause leaks or ambiguous interior. |
| **sampled SDF/TSDF** | At each grid point, compute closest triangle distance with an AABB/BVH and sign it with a pseudo-normal test or generalized winding number; optionally clamp to a narrow band. Occupancy is `SDF <= 0`. | Retains sub-voxel zero-crossing location and matches SDFusion's current convention. | A sign is not well-defined for a truly open surface; robust methods make a defensible heuristic, not an exact missing solid. |
| **fractional/anti-aliased occupancy** | Supersample each cell or compute/approximate the mesh–cell volume fraction, then store coverage or threshold it. | Reduces sensitivity to grid origin and retains a training confidence near boundaries. | More compute; thresholding ultimately reintroduces a binary choice. |

The classic triangle/box test derives from the separating-axis theorem
([paper](https://doi.org/10.1080/10867651.2001.10487535),
[author code](https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/)). Schwarz and Seidel
give separate conservative surface and solid GPU algorithms, including a thinner 6-separating surface
and sparse octree storage ([author project and paper](https://michael-schwarz.com/research/publ/2010/vox/)).
Open3D's documented mesh voxelizer is explicitly a triangle-intersection **surface** voxelizer
([official documentation](https://www.open3d.org/docs/release/tutorial/geometry/voxelization.html));
`binvox` exposes both exact surface intersection and its normal solid scan/parity behavior
([author documentation](https://www.patrickmin.com/binvox/wiki/doku.php?id=usage_summary)).

For SDFusion's filled-solid semantics, the most direct mesh-only fallback is:

1. Put the mesh into the same metric/normalized frame as the footprint and record grid origin, pitch,
   axis order, and sign convention.
2. Validate finite vertices, non-degenerate faces, bounds, connected components, orientability,
   watertightness, and whether the surface touches the grid boundary.
3. Query unsigned closest-triangle distance at grid points using a BVH.
4. For a watertight, outward-wound mesh, sign with pseudo-normals or winding number. For dirty triangle
   soup, use generalized/fast winding numbers and record the threshold and ambiguity band.
5. Store the clamped SDF/TSDF and derive `solid_occ = sdf <= 0`; optionally store a separate
   conservative surface mask.
6. Run a round-trip control against the original mesh or field before any learned editing.

Libigl documents both signing choices: pseudo-normal signing is fast but assumes a watertight,
non-self-intersecting manifold; generalized winding numbers are slower but robust to unclean meshes.
For a closed surface, winding is exactly 1 inside and 0 outside; for an open/non-manifold but oriented
surface it changes smoothly and is a heuristic interior score
([official libigl tutorial](https://libigl.github.io/tutorial/#generalized-winding-number),
[robust inside/outside paper and author page](https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.html),
[fast winding numbers project](https://www.dgp.toronto.edu/projects/fast-winding-numbers/)). OpenVDB
also provides official mesh-to-signed/unsigned-distance-field tools and signed flood fill
([API](https://www.openvdb.org/documentation/doxygen/MeshToVolume_8h.html)).

### Failure modes that must be made explicit

- **Watertight versus open.** An open facade/roof sheet has no unique solid interior. Surface
  voxelization is still defined; solid voxelization necessarily repairs, closes, or guesses. Calling
  that guess “voxelization” hides a model decision.
- **Orientation and self-intersection.** Parity may double-count shared or grazing intersections;
  winding depends on reasonably consistent orientation. This repository already lost runs to an
  `x↔z` reflection and has observed negative-volume meshes yielding inverted/empty occupancy
  ([surface-corpus finding](../wayfinding/crisp-massing-vecset/surface-corpus.md)).
- **Grid alignment and aliasing.** Center sampling can erase a feature thinner than one pitch;
  conservative intersection can preserve it as a one-cell shell but widen it. Translating or rotating
  the same mesh relative to the grid can change discrete topology. Supersampling helps, but cannot
  make a fixed grid carry detail below its sampling scale.
- **Binary reconstruction.** Libigl's own comparison shows serious aliasing when a signed-distance
  field is clamped to an indicator before contouring; retaining the sampled SDF produces a better zero
  set ([signed-distance tutorial](https://libigl.github.io/tutorial/#signed-distances)).
- **Cubic scaling.** A dense grid costs `O(R³)` cells: `64³ = 262,144`, `128³ = 2,097,152`, and
  `512³ = 134,217,728`. Sparse surface grids reduce storage, while filled solid grids are much less
  sparse for building volumes.

At this project's operating point, `s* = 1.0 m ≈ 3 voxels @64³`; features below it are detail, not
massing ([ADR 0004](../adr/0004-experiment-operating-point.md)). A `64³` occupancy editor may reshape
roofs, wings, setbacks, courtyards, and other structural-scale volumes. It should not be evaluated on
windows, cornices, thin railings, or facade articulation.

## Should the initial voxelizer itself be learned?

### Clean known mesh: no

The target occupancy of a clean mesh under a declared grid rule is already computable. A learned
mesh→voxel network adds approximation error, training-distribution dependence, non-reproducibility,
and a second opportunity to hallucinate geometry. A neural converter is useful only if “make a clean,
closed building from this uncertain observation” is intentionally part of the task.

This is also what the recent high-end systems do: their learned stages operate **after** deterministic
spatial encoding. TRELLIS.2's O-Voxel conversion deterministically stores active cells, edge
intersections, and sub-voxel dual vertices before learning in that representation
([paper](https://arxiv.org/abs/2512.14692),
[official O-Voxel code/docs](https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/README.md)).
Faithful Contouring similarly identifies surface-intersecting voxels and local contour tokens with a
deterministic GPU encoder, then learns an autoencoder/generator over those tokens
([paper](https://arxiv.org/abs/2511.04029),
[official code](https://github.com/Luo-Yihao/FaithC)). Both preserve open/non-manifold and sharp
surfaces better than binary occupancy, but both are **surface representations**, not drop-in filled
building solids.

### Need gradients: use differentiable geometry, not an AI guess

Differentiable Voxelization and Mesh Morphing (DOPH) derives mesh→occupancy/SDF from solid angles and
winding numbers, runs it on the GPU, and differentiates with respect to mesh vertices
([paper](https://arxiv.org/abs/2407.11272),
[official code](https://github.com/Luo-Yihao/DOPH)). Its paper also exposes the key tradeoff: an
accurate near-binary occupancy has gradients concentrated near the surface, while a softer integral
improves gradient flow at the cost of less accurate occupancy. That is useful if the editor is trained
end-to-end into mesh vertices or the upstream decoder; it provides no advantage for an inference-only
conversion of A2's already materialized field.

### Dirty mesh: name the operation repair/completion

If a mesh is open, fragmented, or self-intersecting, a learned shape prior can plausibly predict a
closed building before or jointly with voxel editing. But it must be evaluated as generative
completion: several closures can explain the same open surface. The original mesh, uncertainty mask,
and unchanged regions must remain available so the model's invention is measurable.

## What recent generative systems actually establish

| system | native representation and coarse→fine relationship | editing/completion evidence | what it does **not** establish for SDFusion |
|---|---|---|---|
| **TRELLIS** (CVPR 2025) | Stage 1 generates a sparse voxel structure; Stage 2 generates structured latent features only at active voxels and decodes them to meshes, radiance fields, or Gaussians. | The paper/repo demonstrate variants and local edits within the model's generated SLAT pipeline. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiang_Structured_3D_Latents_for_Scalable_and_Versatile_3D_Generation_CVPR_2025_paper.pdf), [official repo](https://github.com/microsoft/TRELLIS) | It is image/text-conditioned and its sparse voxels outline coarse structure; it is not a post-hoc solid-building editor for arbitrary meshes. |
| **Seed3D 2.0** (2026) | A coarse DiT produces a coarse mesh. Stage 1 geometry is re-encoded as latents and also GPU-voxelized/dilated into a spatial occupancy prior for a high-resolution second stage. | Strong evidence that deterministic coarse-mesh voxelization can anchor learned refinement. [Paper](https://arxiv.org/abs/2605.13862), [official release](https://seed.bytedance.com/en/blog/seed3d-2-0-released-higher-precision-and-greater-usability) | The voxels are conditioning/position, not a generatively edited building solid. The input is an image, and the available primary sources do not establish arbitrary-mesh buildingization. |
| **Home3D 1.0** (2026) | A geometry VAE plus coarse-to-fine flow-matching DiTs reconstruct a watertight mesh through latent SDF modeling. | Confirms coarse→refine is used in a recent production-oriented image-to-3D system. [Paper](https://arxiv.org/abs/2606.27923) | It is furniture/image reconstruction, not occupancy editing; its primary report does not expose a reusable mesh→voxel editor. |
| **DVD** (2026) | Replaces TRELLIS Stage 1 with discrete diffusion over a pure binary `64³` voxel grid, then leaves TRELLIS Stage 2 to generate per-active-voxel latents and final assets. | Block-structured-perturbation fine-tuning supports voxel inpainting/editing in one standard sampling process. [Paper](https://arxiv.org/abs/2605.07971), [official checkpoints/model card](https://huggingface.co/Zhengrui/dvd) | Closest algorithmic precedent, but its target is a **sparse surface scaffold**, trained on roughly 450K general assets with image/text conditions—not filled LoD2 building occupancy or footprints. Its checkpoint is not directly reusable. |
| **ArchComplete** (2024/2025) | Building-specific dense voxel VQ model and autoregressive transformer generate at `64³`; a hierarchy of patch diffusion upsamplers refines toward `512³`. | Demonstrates architectural shape completion, plan-drawing completion, and coarse-to-fine voxel detailization on fully modeled houses. [Paper](https://arxiv.org/abs/2412.17957), [publisher version](https://doi.org/10.1016/j.cag.2025.104477) | Strongest domain precedent, but it natively generates/completes its own house grids. It does not show that a generic post-hoc editor improves authentic outputs of a different generator, and its interior/exterior house domain differs from LoD2 massing. |
| **DiffComplete** (NeurIPS 2023) | Conditional diffusion completes partial shapes represented by `32³` distance/SDF grids, with spatially aligned hierarchical condition features and occupancy-aware fusion. | Establishes generative voxel/SDF completion from partial observations. [Paper](https://arxiv.org/abs/2306.16329), [official code](https://github.com/dvlab-research/DiffComplete) | It is partial-scan completion, commonly class-specific in the released configuration, not architecture-conditioned correction of a full but implausible generated building. |

The literature therefore supports **native volumetric generation**, **masked completion**, and
**coarse geometry used as a spatial prior**. It does not supply evidence that an arbitrary completed
mesh can be run through a generic model and reliably become a better building while preserving its
footprint and unrelated regions. That last claim needs a SDFusion-specific experiment.

## A technically coherent SDFusion editor

### Representation and contract

Use a dual representation at the editor seam:

- `occ_in`: filled `64³` occupancy from `fld <= 0` for categorical/discrete generation, locality,
  solidity, and overlap;
- `tsdf_in`: A2's existing clipped field or a normalized narrow-band SDF for surface position;
- `condition`: footprint mask, height/vertical span, source region, and available class/style;
- `edit_mask`: cells the model may change; all other cells are clamped to the input at every sampling
  step, not merely penalized by a soft preservation loss;
- `occ_out` plus either a continuous residual/TSDF head or a separate surface decoder.

A DVD-like discrete diffusion model is a reasonable editor hypothesis because occupancy is genuinely
binary and its block-perturbation training matches spatial edits. A smaller conditional 3D U-Net or
masked latent-diffusion prototype would also answer the representation question; the first experiment
does not need a frontier-scale architecture.

The continuous output matters. Recomputing a signed Euclidean distance transform from `occ_out` is a
deterministic baseline, but this repo has observed field-slope and meshing artifacts even at identical
occupancy. A narrow-band SDF residual or query decoder should be evaluated alongside it. Occupancy
quality and rendered surface quality must remain separate measurements.

### Training distribution

The editor must see the inputs it will actually receive:

One correction to the earlier framing is important: issue #91's cache aligns the **footprint-envelope
latent tokens** with the real-building latent tokens. It does not contain authentic A2-generated
fields paired with their targets. The corpus row IDs make those pairs straightforward to create, so
the old alignment blocker is gone at the dataset level, but an authentic `A2 output → real field`
cache still has to be generated.

1. Generate and cache current A2 fields for the training footprints, preferably multiple seeds and
   failure strengths, paired by building ID with the existing real LoD2 target.
2. Train authentic-output→real examples for the global “make this massing more building-like” case.
3. Separately train masked/block perturbations of real buildings for localized completion and exact
   preservation outside the mask. Do not pretend that synthetic masking alone matches A2's failure
   distribution.
4. Stratify or reweight by required edit magnitude. PLATEAU was ingested at LoD1: all 210 held-out JP
   envelopes equal their target at 1.0 IoU, and the source contributes 26.1% of training pair steps
   with a zero target ([dataset code](../../scripts/train_vecset.py#L69-L89),
   [run design](../../scripts/foundations/run_aligned_retrain.py#L48-L52)). Without sampling control,
   the optimal lesson is often “do nothing.” Keep identity examples as a preservation test, but do not
   let them dominate the corrective loss.

This proposal does **not** erase the post-hoc-refiner negatives. It changes two variables those
experiments did not jointly test: authentic current-model inputs and a representation where known
cells can be clamped exactly. It must still beat the existing envelope/A2 controls before claiming the
failure mode was representation rather than supervision or data.

### Hard constraints and evaluation

“Building-like” is too weak as a target. The editor needs explicit validity gates:

- ground contact and at least one connected occupied component;
- no empty/solid collapse;
- footprint split into fringe/spill/uncovered at fixed `s*`;
- optional hard exclusion outside the permitted footprint/roof-overhang band;
- minimum wall/volume thickness appropriate to the `64³` massing scale;
- exact identity outside the edit mask;
- watertight surface extraction and consistent winding.

Use the existing full `n=714` paired gate and always publish `vs_input`, collapse rate, 3D IoU,
beats-envelope rate, and spill/uncovered separately. The current strength sweep shows why: a rectangle
moves from a literal no-op at 0.5 to near-empty rubble at 0.7
([#93](../wayfinding/latent-token-order/93-strength-band.md)). A voxel editor only succeeds if it
creates a measurable middle—not just a better median produced by returning its input.

Also restore the validated surface-quality instrument for the fixed montage subset. The current
checkpoint watcher explicitly invokes the harness with `--sne 0`, which is why its artifacts report
`sharp_normal_error.values = {}` and `views = 0`
([watcher](../../scripts/foundations/watch_checkpoints.py#L50-L56)). That is an intentional speed
choice, not evidence that the metric ran. A final editor gate should run nonzero SNE views alongside
the volumetric scores; otherwise it can improve occupancy while surface quality remains unmeasured.

## Compatibility with recorded decisions

The representation hypothesis is compatible with C1 **only if** the voxel state remains an
intermediate transform and the same operator supports envelope-to-massing and masked user editing.
Two variants conflict with current accepted language and would need an explicit decision before
implementation:

- a pure noise-to-voxel generator replacing the envelope projection conflicts with ADR 0003's
  “transform, not generate” claim;
- treating the final arbitrary voxel mask or baked mesh as the editable building conflicts with the
  current symbolic-recipe / solid-first contract, which explicitly makes meshes terminal derived
  outputs and reserves free-form residual fields for constrained optional refinement
  ([`CONTEXT.md`](../../CONTEXT.md#L179-L222)).

This is not a reason to reject the experiment. It defines the safe scope: test occupancy as a local,
constrained **massing transform** first. If it works, decide separately whether it remains a bounded
realization aid, is distilled into semantic operations, or justifies revising the project's core
representation claim.

## Recommended research decision

Proceed with a bounded prototype only if it is framed as follows:

> Edit A2's existing `64³` solid occupancy, conditioned on the same footprint/height and constrained by
> an explicit preservation mask; retain a continuous surface channel; train on authentic A2 outputs
> plus masked real-building examples; evaluate only massing above `s*` against A2 and the envelope.

Do **not** spend a model on the initial mesh-to-voxel conversion. For the current A2 path, eliminate
that conversion entirely and branch from `fld`. For external meshes, use deterministic SDF/winding
voxelization with a round-trip control. Add DOPH only if an experiment specifically requires gradients
back to mesh vertices. Treat learned repair of non-watertight meshes as a separate generative
completion task with its own uncertainty and preservation metrics.
