# State representation for a whole-volume A2 voxel transform

Ticket: [Choose the state representation for a whole-volume A2 voxel transform](https://github.com/danvisai/SDFusion/issues/114)  
Research date: 2026-08-19

## Decision

Use a **dense, absolute, binary filled-occupancy grid at `64³` as the transform state**.

- `EMPTY=0` and `SOLID=1` are the only persisted/generated geometry states.
- Every cell inside the hard footprint support is present in the lattice and is eligible to change.
  Cells outside that support remain present too, but the validity projection may only force them empty.
- A2's decoded zero-level field is retained as a **read-only source-conditioning channel**, after a
  fixed clipping/normalisation calibration. It is not the edited state and must not be described as a
  metric TSDF: this repository measures materially different near-surface slopes for A2/Dora and the
  real SDF corpus.
- `KEEP / ADD / REMOVE` may remain an auxiliary or throwaway deterministic-head parameterisation for
  balancing rare changes. It is immediately reduced with the source occupancy to an absolute output
  occupancy; it is not stored, diffused, meshed, or treated as the building state.
- Continuous surface recovery is a **realization after the transform**, not a second geometry state in
  the first feasibility model. The occupancy result must survive a separate signed-EDT versus learned
  continuous-contour recovery comparison before the route can pass a visual gate.

The concrete interface is:

```text
Fsrc = DoraCodec.decode_grid(a2_latent, 64)     # negative inside, continuous zero-level field
Osrc = (Fsrc <= 0)                              # bool[64,64,64], filled occupancy
C    = {Osrc, norm_clip(Fsrc), footprint, height, region/class/style if available}

# deterministic correction gate
Oout = model(C)                                 # absolute binary endpoint

# later stochastic arm, only if the map's full gate authorizes it
Xt   = categorical_corrupt(Otarget)             # Xt is binary at every grid position
p0   = model(Xt, t, C)                          # Bernoulli posterior over absolute Otarget
Oout = sample_or_argmax(p0)

Oout = validity_project(Oout, footprint, ground, connectivity, minimum_thickness)
mesh = surface_recover(Oout, Fsrc, p0)           # separate, measured realization contract
```

This chooses binary occupancy as the state without throwing away useful input evidence. The distinction
is load-bearing: **the state answers which volume is solid; the source field helps predict where an A2
surface crossed a cell; the surface recoverer decides where to place terminal mesh vertices.**

## Why this is the right state here

### It is the state already exposed by the A2 seam and scored by the harness

The live service queries Dora directly onto a `64³` field and computes `out_occ = fld <= 0` before
reporting `vs_input`; meshing happens after `decode_grid` and adds no state information
([town service](../../../scripts/server/town_generate_service.py#L220-L244)). The codec contract makes
the same seam explicit: `decode_grid` is arbitrary point queries materialised on a grid and normalises
Dora to negative-inside convention ([shape codec](../../../models/shape_codec.py#L1-L25),
[Dora query](../../../models/shape_codec.py#L216-L226)). The evaluation harness likewise ranks geometry
on `field <= 0` occupancy and keeps field-shape measures only as guards
([evaluation](../../../scripts/foundations/eval_massing_arms.py#L119-L143)). Therefore the conversion is
one threshold on data A2 already produces, not mesh-to-voxel inference or a mesh round trip.

The representation also matches the nearest building-specific precedent. ArchComplete prepares
architectural mass/void data as **dense binary voxel grids**, generates its coarse architectural shape
at `64³`, and introduces a hierarchy only for later high-resolution refinement
([paper, §4.1–4.3](https://arxiv.org/html/2412.17957#S4)). That does not prove this correction task
will work, but it does establish that dense binary mass/void is a coherent native representation for
architectural form at this operating point.

### It makes whole-volume eligibility and validity exact

An absolute dense lattice represents both currently solid and currently empty cells. Adding a wing,
opening a courtyard, removing overfill, or rebuilding a roof uses the same two-state operation at every
location. Footprint containment, filled volume, connectedness, ground contact, minimum thickness,
missing/extra volume, and exact `vs input` are all functions of the state itself; none requires
thresholding an unrelated continuous prediction first.

This property matters more than nominal smoothness in the map's first gate. The shipped full-set
artifact reports median A2 `vol_iou=0.8756`, `vs_input=0.9846`, and field slope `0.8127`, while the real
SDF has slope `0.0312` and Dora's codec ceiling `1.3074`
([full 714 artifact](../../../execution/artifacts/massing_arms_eval_ship714.json)). The signs are
compatible, but the magnitudes are not on one metric-distance scale. A raw continuous-field transform
would have to learn topology and codec-specific calibration simultaneously. Thresholding once makes
the topology explicit while retaining a calibrated source field as evidence.

### It is the native variable for a later discrete-diffusion arm

D3PM defines diffusion directly over categorical values and allows uniform, structured, or absorbing
transition matrices without relaxing the data into continuous space
([Austin et al., 2021](https://arxiv.org/abs/2107.03006)). DVD applies that construction to a binary
`N³` voxel grid: each location is a Bernoulli variable, the forward process reassigns discrete states,
and the reverse model predicts the clean categorical posterior at every grid position
([DVD §3.1](https://arxiv.org/html/2605.07971v3#S3.SS1)). Its released implementation explicitly
generates or edits a pure `64³` grid ([official repository](https://github.com/TeCai/DVD)).

DVD is precedent for the **algorithmic type**, not a reusable model: its occupied voxels are a sparse
surface scaffold for TRELLIS stage 2, whereas this decision uses filled building mass. It also reports
hundreds of neural evaluations as a limitation and leaves joint discrete-continuous diffusion as future
work ([DVD limitations](https://arxiv.org/html/2605.07971v3#Sx1)). That supports the map's ordering:
prove deterministic binary correction first; authorize categorical diffusion only after a full gate.

## Alternatives considered

| candidate | what it gets right | why it is not the transform state |
|---|---|---|
| **Dense binary absolute occupancy** | Exact filled mass/void topology, every empty and solid cell addressable, direct invariants and existing metrics, native Bernoulli diffusion. | Does not itself retain a sub-voxel zero crossing; surface recovery is mandatory and separately gated. **Chosen.** |
| **Occupancy + clipped field/TSDF as a joint state** | Carries topology and continuous zero-crossing information; a scalar field can be contoured with interpolated vertices. | A2's raw field is not calibrated to the corpus SDF; joint corruption must preserve sign/field consistency; exact constraints still act on the occupancy half; mixed discrete-continuous diffusion is not the proven DVD path. Use the normalised source field as conditioning, not as a co-equal generated state. |
| **`KEEP / ADD / REMOVE` lattice** | Makes source-relative changes explicit and can balance an overwhelmingly unchanged training target. | It is not a shape without `Osrc`; `KEEP` changes meaning when the source changes; a noised action lattice is not an intermediate geometry; three-state logits cost more than two-state logits and preserve the same severe `KEEP` imbalance. Permit as an auxiliary deterministic head only. |
| **Sparse surface lattice / SLat** | Efficient when only surface anchors need features; demonstrated at `64³` by TRELLIS/DVD. | This effort needs a **filled solid**, and any currently empty interior/exterior cell may become solid. A sparse active set omits exactly the candidate additions the whole-volume contract promises unless dynamic topology and a dense candidate universe are reintroduced. |
| **Hierarchical/VDB state** | Strong when resolution or spatial extent is large: XCube uses VDB and coarse-to-fine latent diffusion to reach effective `1024³` and large scenes ([official XCube](https://github.com/nv-tlabs/XCube)); OpenVDB compresses uniform tiles and narrow-band level sets ([official overview](https://www.openvdb.org/documentation/doxygen/overview.html)). | At `64³`, the dense universe is only 262,144 cells. A hierarchy adds activation, traversal, supervision, and error-propagation decisions before memory is the bottleneck. Reserve it for a later resolution increase, not this massing gate. |

## Sub-voxel surface information and mesh recovery

Binary occupancy does lose the exact A2 crossing along a grid edge. Continuous signed-distance
representations locate a surface at their zero set, and marching cubes places vertices by linear
interpolation of scalar samples ([DeepSDF representation](https://arxiv.org/abs/1901.05103),
[original Marching Cubes paper](https://graphics.stanford.edu/courses/cs348a-21-winter/Papers/Marching_Cubes.pdf)).
That is why `norm_clip(Fsrc)` remains in the condition and why a cubified/binary mesh cannot be the
final visual artifact.

It is nevertheless wrong to promote A2's field to the edited state merely to save that information:

1. The map's target is massing above `s* ≈ 3` voxels, while the lost crossing is sub-voxel.
2. The repository has already measured field-representation contamination. A blockout with occupancy
   byte-identical to GT receives SNE `0.241` because its signed EDT meshes differently; Dora's field
   has a much steeper slope than the corpus field
   ([SNE validation](../vecset-convergence/sne-validation.md),
   [harness baseline](../vecset-convergence/harness-baseline.md)).
3. An arbitrary edited topology cannot reuse A2's original zero crossing everywhere. Newly added or
   removed surfaces need a new contour even if the source field is retained.

The first prototype should therefore keep two mesh-recovery arms while holding `Oout` fixed:

1. **Deterministic control:** signed Euclidean distance transform of `Oout`, negative inside, contoured
   at zero. This is reproducible but is expected to expose the known faceting/ribbing floor.
2. **Continuous recovery candidate:** predict or fit a narrow-band contour field conditioned on
   `{Oout, norm_clip(Fsrc), p0}`, with an exact sign-consistency check against `Oout` at grid samples.

Volumetric success belongs to `Oout`; surface success belongs to the recovery arm. Report both. A
continuous recoverer that silently changes occupancy has failed the contract, and a binary result that
improves IoU but fails SNE/montage has not passed the map's visible-architecture requirement. A
continuous implicit decoder can recover surfaces beyond a fixed grid's vertex positions—Occupancy
Networks demonstrates the general principle using a continuous decision boundary—but adopting one is
a later evidence-backed surface decision, not part of this state choice
([Mescheder et al., 2019](https://arxiv.org/abs/1812.03828)).

## Compute and storage at `64³`

The dense state is small enough that sparsity is not yet buying the dominant resource:

| payload per building | storage |
|---|---:|
| bit-packed occupancy | 32 KiB |
| `uint8` occupancy | 256 KiB |
| `float16` source field | 512 KiB |
| stored `uint8` occupancy + `float16` source field | 768 KiB |
| two-class `float16` logits | 1.0 MiB |
| three-class `float16` action logits | 1.5 MiB |

These figures are raw arrays; 3D-network activations will dominate training memory. They still show
why a VDB/hierarchy is premature and why keeping one read-only field conditioner is cheap. Persist the
cache as bit-packed/`uint8` occupancy plus `float16` clipped source field; materialise floating channels
only per batch.

## Prototype consequences

The implementation-ready representation contract for the next prototype ticket is:

1. Cache authentic A2 `Fsrc` directly from `DoraCodec.decode_grid`; cache `Osrc = Fsrc <= 0` and real
   `Otarget = GT_sdf <= 0`. Never mesh/re-voxelize A2.
2. Fit one clipping/normalisation rule from the authentic training cache only and record it in the
   cache manifest. Call it `source_field`, not metric `tsdf`, unless a measured calibration earns that
   name.
3. Make the model's primary endpoint absolute two-class occupancy at every one of the `64³` cells.
   An auxiliary change-balanced or action loss is allowed; a persisted action grid is not.
4. Apply hard validity as an explicit projection/rejection contract and report how often/how many
   cells it changes. The learned result cannot claim gains created by deterministic cleanup.
5. Evaluate occupancy against sanitized A2 and the footprint envelope before evaluating either mesh
   recovery arm. Preserve `vs input`, missing/extra, collapse, spill/uncovered, identity rows, and the
   map's source-dependence tests.
6. Run the EDT and continuous-recovery arms on identical `Oout`, with nonzero SNE and fixed montages.
7. If and only if deterministic correction passes the full gate, diffuse **absolute binary occupancy**,
   not actions or a raw continuous field.

## Recorded-decision boundary

This is an experimental transform representation, not permission to redefine the building. A dense
arbitrary occupancy result conflicts with the current `CONTEXT.md` language that makes the symbolic
recipe/semantic architectural edit program authoritative and explicitly avoids arbitrary voxel masks.
It can still test ADR 0003's C1 transform hypothesis because it begins from A2 output rather than pure
noise, but it does not by itself establish recipe closure. The parent map deliberately leaves three
outcomes open: store it reproducibly as a recipe operation, distil it into a semantic architectural
edit program, or reject the route. This ticket changes none of those accepted domain decisions.
