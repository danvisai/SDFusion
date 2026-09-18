# #117 - Hard validity contract for whole-volume massing edits

*Settled 2026-09-18 by interview (`/grilling`) with the ticket owner, as map #113 requires for its
human decisions. Signed off by the owner; supersedes the contract proposed in PR #185, which was
closed. No code has been changed yet.*

Every axis below is either adopted from a published standard, or explicitly marked as a
project-specific addition the standards do not cover. Where a threshold exists in the literature
this contract takes the published number rather than inventing one.

## Scope

One building's **exterior massing**: the coarse architectural solid above `s*`, including roof
form, wings, setbacks, annexes, and structural-scale exterior openings. Rooms, circulation,
windows, doors, facade articulation, materials, appearance, semantic operation recovery, and
recipe editability are all outside this decision.

**Representation.** Dense absolute binary filled occupancy on the `64³` grid, axes `[z, y, x]`,
`y` increasing upward. Blocks assembled into a building, conditioned on a 2D footprint. Every
voxel is eligible for learned change. Validity does not preserve the source's roof, height, part
count, or vertical extent.

**Scale.** `s*` = 1.0 m ≈ 3 voxels at `64³` (ADR 0004, fixed a priori); one voxel ≈ 0.32 m for a
~20 m building.

**LOD band.** In the refined specification of Biljecki et al. (2016) this representation sits at
roughly **LOD1.2 to LOD2.1**. It cannot reach LOD2.3, whose roof-overhang requirement is 0.2 m,
smaller than one voxel. This bounds what the model can be asked to learn and what any evaluation
of it can claim.

## Connectivity is fixed by digital topology, not by preference

Object and background connectivity in `Z³` must be **complementary**, or a closed digital surface
fails to separate inside from outside (Rosenfeld; Kong & Rosenfeld, *Digital topology: introduction
and survey*). The valid pairs are (6,26), (26,6), (6,18), (18,6). Using the same connectivity for
both produces genuine paradoxes.

This contract therefore fixes: **solid occupancy is 26-connected, empty space is 6-connected.**
That is the pair `connected_components` and `flood_fill_exterior` already implement. It was
previously reported as an inconsistency to be fixed; it is not, and must not be "corrected".

## The contract

A candidate is **valid** if and only if both rules pass.

| # | Rule | Authority |
|---|---|---|
| 1 | **Non-degenerate.** At least one building part survives rule 2's size filter. | project |
| 2 | **No floating building parts.** Label solid occupancy 26-connected. Discard components below the minimum building-part size as speckle. Recompute the base plane from the surviving parts only. Every surviving part must reach within the significant-height-difference threshold of that base. Parts are grouped after one 6-connected erosion, so a join thinner than `s*` does not merge two parts. | project, sized by LOD spec |

That is the entire hard set.

### Minimum building-part size

Biljecki et al. state the form explicitly: minimum size is "expressed as a distance, which can be
applied to the length, width, or height of a feature, and as a projected footprint area", and rule
out the alternative outright: "**A 3D requirement (volume) will not be used** because it is not
applicable to all types of features." A volume threshold is therefore the wrong shape of rule and
is not used here.

Adopted at the **LOD2.1** grade, the finest this grid supports, whose published threshold for
smaller building parts and extensions is **> 2 m and > 2 m²**:

| Quantity | Published | This grid |
|---|---|---|
| Minimum part extent | 2 m | **6 voxels** (longest bounding-box dimension) |
| Minimum part plan area | 2 m² | **20 voxel columns** (vertical projection) |

A component failing either test is **speckle**: it is left in the geometry untouched, and simply
is not a building part for the purposes of rule 2. It does not fail validity and it does not get a
vote on where the base plane is.

Recorded for reference, not adopted: LOD2.0 uses > 4 m and 10 m² (~12 voxels, ~98 columns) for
large parts such as garages; LOD1.0 uses > 6 m (~19 voxels) for a building at all.

### Significant height difference

**2 m, which is 6 layers on this grid.** This is LOD1.3's published threshold for when a height
difference in a massing model is significant enough to model separately.

Applied here by analogy rather than by direct citation: LOD1.3 states it for multiple top surfaces,
and this contract uses it for base planes. It is the only published number in the LOD specification
for "when is a height difference significant in a massing model", and the alternative was picking
one by taste.

This is what admits a house and a detached garage whose ground planes differ by tens of
centimetres. Biljecki et al. are explicit that this is one building: the specification "defines
that sizeable building parts, extensions and annexes **such as garages** and alcoves, may be
acquired and treated distinctly... **Such objects are still part of the building**."

### Substantial joins

Group parts after eroding the solid by one voxel, then map the labels back. A join thinner than
3 voxels in cross-section, which is `s*`, does not survive and therefore does not merge two parts.

This is a **project-specific addition**, needed because 26-connectivity counts a single corner
contact as joined. Without it a one-voxel thread can attach a floating mass to a grounded one and
launder it past rule 2. Measured cost on the legacy corpus: one building in 35,776 at any size
floor, and zero at the adopted floor.

## Measured on every candidate, never causing failure

Reported for the learned candidate, the A2 baseline, and the GT target alike, so that no rate is
ever quoted without a control.

| Quantity | Why it is not a failure |
|---|---|
| **Sealed cavities** | ISO 19107 defines a Solid as an exterior shell plus optional **interior** shells. A cavity is a first-class modelled feature; val3dity supports "cavities in solids (also called voids or inner shells)" and errors only when a cavity lies outside the solid (`403 INNER_SHELL_OUTSIDE`). Buildings have insides. |
| **Spill outside the footprint** | Owner decision: footprint agreement is a score. Note the stored `footprint` is the target's own vertical projection (`scripts/ingest_3dbag.py:105`), so spill on ground truth is zero by construction and carries no information; only a generated candidate against a given conditioning footprint is informative. |
| **Uncovered footprint** | Owner decision: a score. `CONTEXT.md`'s fringe/spill/uncovered split continues to govern how it is reported. |
| **Interior depth** (`abs(sdf.min())` in voxels) | Diagnostic for solidification failure, see below. Not a property of massing. |
| **Part count, base-plane spread, speckle count, ceiling contact** | Descriptive. |

## Projection, rejection and failure accounting

The issue asks whether invalid samples are projected, rejected, or counted. The answer is that
**nothing is projected and nothing is rejected**:

1. **No automatic modification of any candidate, anywhere.** Footprint intersection is no longer
   applied, at scoring or when building the training cache. `validity_projection_delta` becomes
   dead code because its answer is now always zero.
2. **No rejection.** An invalid candidate is kept. Nothing is retried, resampled, selected
   best-of-k, or filtered on validity during training.
3. **Counting only.** "Invalid" is a label plus a rate. Rows keep their metrics in every aggregate;
   #118 owns the aggregate bar. *Owner-acknowledged consequence: the headline quality figure and
   the invalid rate are measured over overlapping populations.*

## Deliberately absent

- **No thickness rule.** The previous erosion-survival bar rejected massing built exactly at `s*`
  (a 3-voxel wall survives at 33%, below the 50% bar it was checked against) and had no basis in
  any standard. `s*` now appears only where it belongs: as the minimum join cross-section.
- **No cavity rule.** See above.
- **No footprint containment rule.** A score.
- **No height limit.** The upper profile is free. Whole-building fit is an import-time concern.
- **Mesh watertightness** belongs to surface realization, not to filled occupancy. ISO 19107's
  shell conditions (`302 SHELL_NOT_CLOSED`, `303 NON_MANIFOLD_CASE`, `306 SHELL_SELF_INTERSECTION`,
  `307 POLYGON_WRONG_ORIENTATION`) apply to that stage, with val3dity's default tolerances: snap
  1 mm, planarity distance-to-plane 1 cm, planarity normal deviation 20°.

## Where the standards are silent

Stated plainly rather than dressed up. **ISO 19107 and CityGML contain no ground-connection
condition.** CityGML integrates buildings with terrain through a Terrain Intersection Curve, a
different mechanism entirely, and their solids are boundary representations rather than voxel
occupancy. Rule 2 is therefore a project-specific physical-plausibility requirement, adopted
because a massing block floating in mid-air is not a building. Its *sizing* is taken from the LOD
specification; its *existence* is this project's decision.

## Corpus defects this contract does not own

Roughly **4.6% of BuildingWorld rows (~70,600)** have no interior: median interior depth 2.11
voxels against 12.86 for the legacy corpus, so `sdf <= 0` yields a 1 to 3 voxel surface skin rather
than a solid. On 68% of Calgary's affected rows the skin reaches the grid boundary, which a
correctly normalised mesh cannot do. The cause is winding-number solidification of open-floor
meshes: 99.93% of Calgary rows carry `defect_class = floor_open`.

These are failed imports, not invalid buildings. They must not be repaired by loosening this
contract, and this contract must not be the mechanism that discovers them. **A separate ticket
owns filtering or re-ingesting them**, with interior depth as the diagnostic.

## References

- Ledoux, H. (2018). val3dity: validation of 3D GIS primitives according to the international
  standards. *Open Geospatial Data, Software and Standards* 3:1. doi:10.1186/s40965-018-0043-x
- Biljecki, F., Ledoux, H., Stoter, J. (2016). An improved LOD specification for 3D building
  models. *Computers, Environment and Urban Systems* 59:25-37.
- Kong, T.Y., Rosenfeld, A. (1989). Digital topology: introduction and survey. *Computer Vision,
  Graphics, and Image Processing* 48(3):357-393.
- OGC CityGML 2.0, `bldg:BuildingPart`, `TerrainIntersectionCurve`.
- ADR 0003 (generation is projection), ADR 0004 (`s*` fixed a priori).

## Code consequences, for the implementing pass

Not yet applied.

- `validity_report`: rules reduce to the two above; cavity, spill, uncovered, depth and part
  statistics become reported fields that do not enter `valid`.
- `ground_connected_ok`: replaced. Base plane from surviving parts, 6-layer tolerance, erosion-based
  part grouping, LOD-sized speckle filter. Its current `labeled[:, 0, :]` test is unconditionally
  false on this corpus, which no longer matters once the base plane is derived rather than assumed.
- `min_thickness_survival`, `MIN_THICKNESS_SURVIVAL_THRESHOLD`: removed from `valid`.
- `sanitize_footprint`: no longer applied at `_score_row` or `build_cache`. **Requires a cache
  rebuild** (~70 min at `corpus_scope="all"`).
- `validity_projection_delta`: dead; remove.
- `validity_report` additionally computed for the A2 baseline and the GT target.
- Test fixtures currently build geometry at `y0 = 0`, a position no real row occupies; they need
  rebuilding against a derived base plane.
