# Hard validity contract for whole-volume massing edits

*Settled 2026-09-17 for [Define the hard validity contract for whole-volume massing
edits](https://github.com/danvisai/SDFusion/issues/117), within the whole-volume voxel-transform
map. This adopts the existing occupancy helpers rather than introducing a second validity
framework.*

## Scope and state

This contract applies to one building's **massing**: the coarse architectural solid above the
detail scale `s*`, including roof form, wings, setbacks, and structural-scale exterior-connected
openings. Rooms, circulation, windows, doors, facade articulation, materials, appearance, semantic
operation recovery, and recipe editability remain outside this decision.

The candidate state is dense absolute binary filled occupancy on the `64^3` massing grid. Array
axes are `[z, y, x]`, `y=0` is the ground plane, and `s* = 3` voxels at `64^3` (1.0 m; ADR 0004).
Every in-footprint voxel is eligible for learned change. Validity does not preserve the source's
roof, height, connected-component count, or vertical extent.

## The contract

The post-projection candidate is valid if and only if every row below passes.

| Axis | Exact decision | Executable authority |
|---|---|---|
| Footprint support | Project with footprint intersection. Every occupied `(z, y, x)` must have an occupied conditioning-footprint `(z, x)` cell. There is no hard fringe allowance and no requirement to occupy every footprint cell. | `sanitize_footprint` |
| Nonempty | At least one solid voxel must remain. | `validity_report` |
| Ground connection | Label solid occupancy with 26-connectivity. Every solid component must include at least one voxel on `y=0`. | `ground_connected_ok` |
| Component count | No fixed count or single-component requirement. Any number of solid components is admissible when each is grounded. | `ground_connected_ok` |
| Solid occupancy and cavities | Label empty occupancy with 6-connectivity from all six grid boundary faces. Any empty voxel not reachable from a boundary is a sealed cavity and fails. This is the filled-occupancy contract; mesh watertightness remains a separate surface-realization check. | `hollow_shell_voxels` |
| Courtyards and passages | Exterior-connected empty space is admissible, including shafts open to the sky, through-passages, undercuts, and recesses. Reachability admits geometry without assigning a semantic subtype. A diagonal-only empty-space contact is not exterior reachability. | `hollow_shell_voxels` |
| Wall and roof thickness | Erode solid occupancy by `floor(s*/2)` voxels, with a minimum radius of one and the array exterior treated as solid. At least 50% of occupied volume must survive. At `64^3`, `s*=3`, so the erosion radius is one voxel. The adopted test is deliberately aggregate: it rejects globally thin massing but is not a local guarantee that every appendage is at least `s*` thick. | `min_thickness_survival` |
| Height | The upper profile is free anywhere inside the grid. Footprint projection does not clamp to source or target `(y0, y1)`. Height changes still have to satisfy grounding, cavity, and aggregate-thickness checks. | `sanitize_footprint` plus the other checks above |

`validity_report` composes these outcomes. Its `footprint_contained` field is the precise term;
`footprint_exact` remains only as a compatibility alias for earlier #125 artifacts and must not be
read as full footprint coverage.

### Fringe is measurement, not hard support

`CONTEXT.md` defines **fringe** as footprint disagreement within `s*` of the boundary that is
reported and ignored by the footprint-fidelity score. That scoring rule does not authorize solid
occupancy outside the conditioning footprint. Raw candidate fringe remains visible in
`spill_raw_predicted`; `sanitize_footprint` projects it away and the projection delta is reported.
This keeps discretisation accounting separate from the hard property-boundary constraint.

### “Connected” does not mean one component

The earlier phrase “connected exterior massing” is resolved as **every solid component is ground
connected**, not “exactly one solid component.” Multi-part footprints occur in the corpus and are
valid when every part reaches `y=0`. A floating component fails even when another component is
grounded.

## Projection, rejection, and failure accounting

These are three different policies:

1. **Projection:** footprint intersection is the only automatic validity projection. It is applied
   once to both the A2 baseline and learned candidate. Raw occupancy is retained, and
   `validity_projection_delta` always records the changed voxel count and fraction, including zero.
   Projection cannot earn learned credit.
2. **Rejection:** a post-projection candidate for which `validity_report["valid"]` is false is
   rejected as usable massing. Grounding, cavities, thickness, emptiness, or any remaining spill
   are not repaired. There is no component deletion, cavity filling, thickening, retry, resampling,
   or best-of-k replacement.
3. **Failure accounting:** rejection never removes an evaluation row. Its metrics, row artifact,
   collapse state, projection delta, and individual validity outcomes remain in the fixed cohort;
   `summarize_rows` counts it in `n_predicted_invalid`. Issue #118 owns the aggregate advancement
   threshold, not this per-sample contract.

Training does not use rejection sampling or a hidden repair target. The corrector learns absolute
occupancy from the fixed cache. The cached A2 source is footprint-sanitized for baseline parity;
validity is assessed after thresholding a prediction during evaluation.

## Evidence and tests

The behavior is exercised through the adopted public seams in
[`test_prototype_voxel_editor.py`](../../../scripts/foundations/test_prototype_voxel_editor.py):
exact projection and height freedom, grounded multi-component massing, floating fragments, sealed
cavities, sky-open courtyards, through-passages, 6-connected empty-space reachability, and thin
walls and roofs relative to `s*`.

The production BuildingWorld verification in
[`119-buildingworld-void-passage-verification.md`](119-buildingworld-void-passage-verification.md)
supports the reachability decision: 724/725 Tokyo gap rows and 137/172 Melbourne
`non_boundary_defect` gap rows contained exterior-reachable gaps, accounting for 99.994% and 96.5%
of measured gap voxels respectively. That evidence justifies admitting exterior-connected
structural openings; it does not relax the sealed-cavity failure.

## Consequences

- This decision formalizes the behavior already present in `sanitize_footprint`,
  `ground_connected_ok`, `hollow_shell_voxels`, and `min_thickness_survival`.
- It does not add a learned validity model, stochastic sampling, GPU work, or production serving.
- It does not define architectural opening semantics or establish symbolic-recipe editability.
- Surface recovery must preserve occupancy signs exactly, but surface-quality and human-visible
  advancement gates remain separate decisions.
