# #158 — BuildingWorld mesh watertightness, boundary-defect character, and extent/solidity profile

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:task` — this profiler makes no
changes to `real.h5` or any other corpus artifact. Feeds
[#166](https://github.com/danvisai/SDFusion/issues/166) (watertightness standard) directly, and
[#165](https://github.com/danvisai/SDFusion/issues/165) (CRS/units policy) alongside
[#157](https://github.com/danvisai/SDFusion/issues/157)'s CRS audit.*

> Build a reusable profiling script and run it across the full per-city mesh sets (not just the
> 25-sample spot checks already on record) to produce, per city: (1) watertight fraction on a much
> larger sample; (2) for non-watertight meshes, a breakdown of WHY — boundary edges concentrated at
> z=min (missing floor cap, likely benign for igl's fast-winding-number SDF) vs. scattered elsewhere
> (non-manifold / broken, likely NOT benign); (3) bounding-box extent, height, and
> footprint-solidity distributions; (4) the per-city subfolder inconsistency (`mesh` vs `obj`,
> `mesh.zip` vs `obj.zip` with different counts). One per-city table covering all of the above — the
> fact base every downstream BuildingWorld decision reads from.

## Method

`scripts/foundations/profile_buildingworld_meshes.py`. For each of the 19 cities: pick the
canonical zip (`mesh.zip` preferred over `obj.zip` when both exist, matching the choice
`render_meshes.py` already made for Adelaide), sample 400 `.obj` members uniformly at random
(fixed seed 158, all 19 populations exceed 400 so every city gets the same sample size — no
population-size confound), load each with `trimesh`, and record:

- `is_watertight`, `is_winding_consistent` (trimesh's own checks, unmodified).
- For non-watertight meshes: boundary edges (edges appearing exactly once in
  `edges_sorted`) are classified by where their midpoint z sits relative to the mesh's own
  z-range. `frac_near_zmin` = fraction of boundary-edge midpoints within 5% of mesh height above
  `z_min`. **`floor_open`** if ≥90% of boundary edges are near z_min (a missing floor cap — the
  benign case the ticket asks about), **`scattered`** if ≤50% are (broken/non-manifold geometry
  spread through the surface), **`mixed`** otherwise. A fourth bucket, **`non_boundary_defect`**,
  covers meshes where `is_watertight` is `False` but there are *zero* open-boundary edges at all —
  i.e. the defect is something else entirely (duplicate/non-manifold-but-closed geometry,
  inconsistent winding). This bucket is **not** covered by the ticket's own floor-cap-vs-scattered
  framing and is called out separately below.
- Extent, height, and footprint-solidity via `scripts/ingest_3dbag.building_to_sdf`, **unmodified**
  — the same function actual ingestion will call — at `R=32` (a coarser grid than the eventual
  ingest resolution; chosen for probe speed, not as a resolution recommendation). `fp_solidity` =
  fraction of the footprint grid occupied, matching `smoke_test.py`'s existing `fp_frac` metric.
- `raw_height_m` and `max_extent` are reported **in each city's own raw file units** — not
  corrected for the unit/projection defects #157 already found. New York/Philadelphia numbers
  below are in US survey feet; Toronto/Mississauga are in Web-Mercator-distorted metres. This is
  why extent and height are reported per-city rather than pooled.

Run: `env -u LD_PRELOAD ./venv/bin/python3 scripts/foundations/profile_buildingworld_meshes.py`
(147s, all 19 cities × 400 samples = 7,600 meshes). Raw per-mesh records and full summary stats:
[`execution/artifacts/buildingworld_mesh_profile.json`](../../../execution/artifacts/buildingworld_mesh_profile.json).

## Results

| City | Pop. | Watertight | Winding-OK | floor_open | scattered | mixed | non_boundary | ext med | ext p90 | frac>90 | solidity med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adelaide | 4,580 | 6.0% | 68.8% | 0 | 293 | 6 | 77 | 30.3 | 58.8 | 2.2% | 0.319 |
| Berlin | 523,213 | 95.8% | 99.8% | 0 | 10 | 0 | 7 | 12.1 | 30.1 | 1.0% | 0.372 |
| Boston | 167,418 | 75.7% | 99.8% | 20 | 25 | 18 | 34 | 45.3 | 88.8 | 9.5% | 0.355 |
| Calgary | 457,474 | 0.0% | 48.0% | 131 | 252 | 16 | 1 | 16.8 | 23.6 | 0.0% | 0.428 |
| Cambridge | 17,377 | 41.5% | 98.3% | 12 | 215 | 7 | 0 | 10.6 | 26.3 | 0.0% | 0.393 |
| Cape Town | 274,598 | 94.0% | 100% | 0 | 4 | 0 | 20 | 14.1 | 32.2 | 2.0% | 0.390 |
| Edmonton | 370,677 | 78.8% | 100% | 3 | 82 | 0 | 0 | 15.2 | 23.1 | 0.7% | 0.466 |
| Greater Geelong | 886 | 0.5% | 95.8% | 301 | 67 | 30 | 0 | 19.9 | 54.0 | 3.3% | 0.375 |
| Melbourne | 8,934 | 45.3% | 95.3% | 0 | 1 | 0 | 218 | 23.0 | 80.2 | 7.7% | 0.358 |
| Mississauga | 145,081 | 97.0% | 99.8% | 1 | 11 | 0 | 0 | 26.5 | 33.4 | 3.0% | 0.347 |
| Montreal | 60,528 | 77.0% | 99.3% | 49 | 38 | 4 | 1 | 17.8 | 32.5 | 2.2% | 0.354 |
| New York | 47,815 | 45.5% | 99.3% | 0 | 218 | 0 | 0 | **85.9** | 178.5 | **44.8%** | 0.264 |
| Perth | 4,136 | 0.0% | 94.3% | 60 | 279 | 61 | 0 | 30.2 | 77.3 | 7.2% | 0.324 |
| Philadelphia | 2,938 | 1.5% | 100% | 65 | 297 | 32 | 0 | **87.0** | 267.5 | **48.7%** | 0.338 |
| San Francisco | 91,360 | 0.0% | 96.3% | 31 | 324 | 45 | 0 | 19.3 | 48.1 | 2.8% | 0.382 |
| Tokyo | 35,577 | 99.3% | 100% | 3 | 0 | 0 | 0 | 17.0 | 43.9 | 2.2% | 0.300 |
| Toronto | 4,255 | 1.3% | **3.7%** | 15 | 265 | 15 | 100 | 77.9 | 148.0 | 38.0% | 0.474 |
| Wellington | 17,368 | 0.0% | 88.0% | 16 | 335 | 49 | 0 | 15.6 | 28.9 | 1.0% | 0.397 |
| Yarra | 26,050 | 0.0% | 98.8% | 61 | 253 | 86 | 0 | 16.5 | 30.1 | 0.3% | 0.326 |

(`n_sampled=400` for every city. Counts in `floor_open`/`scattered`/`mixed`/`non_boundary` are out
of the `400 - n_watertight` non-watertight meshes. `ext med`/`ext p90` = max-extent, in each city's
own raw file units; `frac>90` = fraction already exceeding the existing 90-unit ingest cutoff.)

Subfolder inconsistency, confirmed at full population (matches the ticket's own cited spot numbers
almost exactly): **Adelaide** — canonical `mesh/mesh.zip` (4,580 `.obj`) vs. `obj.zip` (5,868); not
the same set. **Wellington** — canonical `mesh/mesh.zip` (17,368) vs. `obj.zip` (16,248); not the
same set. No other city has more than one zip.

## Key findings

**The n=25 spot-check was noisy enough to matter.** At n=400 several cities move well outside the
prior estimate's ballpark: Cambridge 64%→41.5% watertight, Montreal 92%→77%, Edmonton 32%→78.8%.
Cities already flagged near-zero stay near-zero (Calgary, Perth, San Francisco, Wellington, Yarra
all still 0%). This is exactly why the ticket asked for a larger sample before any standard gets
decided on the smaller one.

*(Correction, code review: Edmonton's mesh loader originally ran with `trimesh.load(...,
process=False)`, which skips vertex-merging and made topologically-shared edges look unshared —
inflating Edmonton's apparent defect rate to a reported 19.8% watertight. With `process`'s default
restored, the true n=400 figure is 78.8%; no other city's numbers in this table changed, since
Edmonton's raw export happened to have unusually heavy coincident-vertex duplication. See
`profile_buildingworld_meshes.py::profile_one`.)*

**`is_watertight` alone conflates two very different failure populations, and the mix is
city-specific.** Greater Geelong fails `is_watertight` 99.5% of the time, but 75% of those failures
are `floor_open` — a missing floor cap, the case igl's fast-winding-number SDF is specifically
robust to. Applying the existing 3D-BAG-style hard `is_watertight` filter would reject nearly all of
Greater Geelong for what may be a cosmetic defect. Contrast Edmonton, Wellington, San Francisco,
Adelaide, Philadelphia, and Toronto, where `scattered` (not `floor_open`) is the dominant failure —
genuinely broken/non-manifold geometry, the case the ticket calls "plausibly NOT benign." A single
relaxed-vs-strict standard applied uniformly to all 19 cities will be wrong for one side of this
split no matter which way it's set.

**A third, unexamined failure mode exists and is large in two cities.** `non_boundary_defect`
(fails `is_watertight`, but has *zero* open-boundary edges — so it's neither a floor cap nor
scattered surface damage; more likely duplicate/degenerate faces or a non-manifold-but-closed
topology) is 218/400 (54.5%) of all sampled Melbourne meshes and 100/400 (25%) of all sampled
Toronto meshes. Within the non-watertight subsets, those are 218/219 (99.5%) and 100/395 (25.3%),
respectively. Neither this ticket's
z-heuristic nor the "benign vs. not" framing it was built on says anything about whether FWN handles
this case safely. **This is a genuine open question for #166**, not a resolved one — the honest
answer for Melbourne and Toronto specifically is "we don't yet know which bucket most of their
defects belong in."

**Toronto has a second, independent defect this session hadn't caught: winding consistency is
3.7%**, i.e. 96.3% of sampled Toronto meshes have inconsistent face winding — far below every other
city (next-lowest is Calgary at 48%, then everything else ≥88%). This is on top of Toronto's
already-known Web Mercator CRS problem (#157) and is a separate reason to treat Toronto with extra
scrutiny (or exclude it) independent of the projection question.

**The extent-cutoff interaction #165 flagged is confirmed and quantified at full sample size.**
New York and Philadelphia — both confirmed US-survey-feet by #157 — have median max-extent 85.9 and
87.0 "units" respectively, with **44.8%** and **48.7%** of sampled buildings already exceeding the
existing ingester's 90-unit multi-building cutoff. If the units bug isn't fixed before that filter
runs, roughly half of both cities' buildings would be silently dropped as "likely merged blocks."
Dividing by the ~3.28× feet→metres factor brings the medians to ~26.2 m and ~26.5 m — unremarkable
single-building sizes, comfortably under the cutoff. Toronto (Web Mercator, anisotropic ~1.38×) also
runs high (77.9 m median, 38.0% >90); correcting for the projection brings the median to ~56.4 m,
still large — Toronto likely has a genuine surplus of large/merged buildings beyond what the CRS bug
explains, so its >90 m filter rate won't fully normalize even after reprojection.

**Footprint-grid fill fraction is fairly uniform** (0.26–0.47 median across cities). The artifact's
`fp_solidity` is occupied grid fraction, NOT footprint area divided by convex-hull area as in the
domain glossary; those measures must not be substituted for one another. New
York (0.264) and Tokyo (0.300) sit lowest, consistent with dense high-rise urban form (more bounding
volume, less solid mass) rather than a data defect.

## Scope and disclosed limitations

- 400 meshes/city, fixed seed — not full population (largest city has 523k meshes; full-population
  scanning was judged unnecessary for a distributional profile at this sample size, but the seed is
  recorded so this is reproducible and extendable).
- Extent/height/solidity computed at `R=32` via the unmodified `building_to_sdf`, a coarser grid
  than final ingest resolution will use — fine for distributions, not a claim about exact per-row
  values ingestion will produce.
- The z-based `floor_open`/`scattered` classification is a heuristic threshold (5% of mesh height),
  not a mesh-repair proof — it says where boundary edges sit, not whether FWN actually produces a
  correct sign for that mesh. No ground-truth SDF comparison was run.
- `non_boundary_defect` meshes are flagged, not diagnosed — see "Key findings" above.

Pure fact-finding; no changes to `real.h5` or any other corpus data. Leaving open for review per
this repo's convention for consequential research/profiling tickets — @danvisai please have a look.
Feeds directly into #165 and #166's decisions.
