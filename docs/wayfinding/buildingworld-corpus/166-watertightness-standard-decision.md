# #166 — Decide the watertightness standard for BuildingWorld meshes

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:grilling`. Decided against
[#158](https://github.com/danvisai/SDFusion/issues/158)'s per-city watertightness/defect profile,
with project-owner sign-off on the standard itself and on how to treat the one bucket #158 flagged
as genuinely unresolved.*

## Decision

**Relaxed standard, not the exact 3D BAG bar.** A BuildingWorld mesh is ingestable if
`is_watertight` **or** its non-watertight defect classifies as `floor_open` (boundary edges
concentrated within 5% of mesh height above z_min — a missing floor cap, the case igl's
fast-winding-number SDF is specifically robust to, per #158's methodology). `scattered` and
`mixed` (a majority — more than 50%, but short of `floor_open`'s ≥90% — of boundary edges near
z_min; correction, code review: previously mis-described here as "majority-scattered", the
opposite of #158's own definition) meshes are rejected outright — real non-manifold damage, not
proven FWN-safe.

**`non_boundary_defect` (is_watertight=False, zero open-boundary edges) is provisionally
accepted under the same rule**, pending the validation below. This bucket is large enough to matter
on its own — 54.5% of ALL sampled Melbourne meshes, 25% of ALL sampled Toronto meshes
(218/219 and 100/395 of their non-watertight subsets, respectively) — and #158 explicitly
flagged it as outside the floor_open/scattered framing entirely, so it needed its own check before
folding it into either side.

## Validation: what `non_boundary_defect` actually looks like (Melbourne)

Melbourne's `non_boundary_defect` meshes have visibly lower median SDF occupancy than its
watertight meshes (0.069 vs. 0.142, n=218 vs. n=181) — worth explaining, not just accepting. A
16-mesh contact-sheet render
([`contact_sheet.png`](158-melbourne-nonboundary-validation/contact_sheet.png), 4 watertight
controls included) shows why: these are overwhelmingly **multi-part adjoined structures** —
rowhouse-style buildings modeled as several touching sub-volumes, or facades exported as
paper-thin duplicate-walled boxes. Multiple adjacent sub-boxes sharing internal walls produce
non-manifold (not open-boundary) edges at the seams, which explains both the classification
(`is_watertight=False`, no boundary edges to find) and the lower occupancy (a multi-wing footprint
fills less of its own bounding cube than a single boxy volume does) — a modeling-style artifact,
not obviously corrupt geometry.

To check FWN actually handles this correctly rather than just looking plausible from the raw mesh,
6 more `non_boundary_defect` meshes were reconstructed via marching cubes on the actual SDF output
of the unmodified `building_to_sdf` (R=48) and compared side-by-side with their raw input
([`recon_compare.png`](158-melbourne-nonboundary-validation/recon_compare.png)). Result: the
reconstructed solids preserve overall footprint, roof form (gable/hip), and multi-wing structure
with **no evidence of sign inversion or degenerate (empty/full) output** — the recovered shape is
recognizably the same building at every sample. The reconstructions do show a systematic ribbed/
corrugated wall texture; this reads as an R=48-grid discretization artifact interacting with
multiple independently-wound wall components (marching cubes at low resolution on a multi-part
input), not a sign error — silhouette and proportions are unaffected. Occupancy values on this
6-mesh sample (0.053–0.217) sit within Melbourne's overall range.

**Verdict: accept `non_boundary_defect` provisionally**, on the strength of a 6-mesh reconstruction
check plus the 218-mesh occupancy/solidity comparison — not on a large validated sample. Per-row
provenance for BuildingWorld must carry a `defect_class` field (`watertight` / `floor_open` /
`non_boundary_defect`) so this bucket stays traceable and any future corpus-quality regression
localized to it doesn't get lost inside an undifferentiated "real data" pool. If a downstream gate
(training-loss outlier, generation-quality regression) ever localizes to `non_boundary_defect`
rows specifically, revisit this acceptance rather than assuming the 6-mesh check still holds at
scale.

## Sampled defect-gate pass fraction (arithmetic corrected 2026-09-09)

Combining `watertight + floor_open` (+ `non_boundary_defect` provisionally) against #158's n=400
profile. These are NOT measured ingestion yields: CRS policy, exclusions, extent filters,
deduplication and sampling caps still apply. In particular, Toronto remains excluded by #165;
correcting this table does not reverse that decision.

| Tier | Cities and sampled pass fraction |
|---|---|
| High (>75%) | Tokyo 100%; Melbourne 99.75%; Cape Town 99%; Berlin 97.50%; Mississauga 97.25%; Montreal 89.50%; Boston 89.25%; Greater Geelong 75.75% |
| Medium (20–75%) | New York 45.50%; Cambridge 44.50%; Calgary 33%; Toronto 30% (excluded); Adelaide 25.25%; Edmonton 20% |
| Low (<20%) | Philadelphia 17.75%; Yarra 15.25%; Perth 15%; San Francisco 7.75%; Wellington 4% |

The poor-yield cities are **not** primarily a CRS story — Adelaide, Perth, San Francisco, Wellington,
Yarra, and Edmonton have no confirmed CRS defect (per #157) and are still dominated by `scattered`
damage. #165's decision reprojects the CRS-affected cities that are also high-yield; it does not
attempt to rescue this low-yield tier, which is a separate, larger data-quality problem outside
this ticket's scope.

## Status: decided (2026-09-06)

Project-owner-approved: relaxed floor_open standard, `non_boundary_defect` accepted provisionally
pending the visual check above. `ingest_buildingworld.py` (#174) should apply this rule directly
and stamp `defect_class` into per-row provenance.
