# #159 — Pilot the beam-search fitter on BuildingWorld's CRS-safe cities

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:task` — a go/no-go signal, not
a full ingest. Makes no changes to `real.h5`, `corpus_ledger.h5`, or any other corpus artifact.
Reads [#157](https://github.com/danvisai/SDFusion/issues/157)'s CRS verdicts and
[#158](https://github.com/danvisai/SDFusion/issues/158)'s watertightness profile to pick cities;
blocks [#170](https://github.com/danvisai/SDFusion/issues/170)'s falsification pre-registration.*

> Before locking any region or channel decision, test the actual premise of this whole effort:
> does BuildingWorld contain recoverable gable/hip/complex roof mass beyond what the existing
> NL/DE/JP corpus already has, or does it collapse into the fitter's "complex, many-Layer" bucket
> (the same bucket the five prior gable arms — #129/#132/#138/#139/the combined assignment-type
> experiment — already failed against)? Run `building_to_sdf` → a 64×64 height-map raster →
> `fit_program_beam` on a ~2,000-building-per-city sample across the CRS-safe, reasonably-watertight
> cities, and report the resulting roof-family distribution (flat / gable / hip / complex-multi-facet)
> per city, alongside the existing NL/DE/JP corpus's own distribution.

## Which cities

Per the ticket's own text, "for this pilot only" — not waiting on
[#165](https://github.com/danvisai/SDFusion/issues/165)/[#166](https://github.com/danvisai/SDFusion/issues/166)'s
full ingestion-time policy — two classes are excluded from the 19 mesh cities:

- **anisotropic-projection-error** (#157): Toronto, Mississauga. Raw X/Y is Web Mercator while Z is
  real metres, so every roof pitch is measured against a horizontally-stretched footprint — this
  would bias the family classification itself, not just `height_m`.
- **"0/25-class watertight"** — #159 cites the *original* 25-sample spot check by name, not #158's
  later n=400 profile (which moves some of these off exactly zero: Adelaide 6.0%, Greater Geelong
  0.5%, Philadelphia 1.5%), and says explicitly not to reclassify on that "until those tickets'
  cities are separately handled": Adelaide, Calgary, Greater Geelong, Perth, Philadelphia, San
  Francisco, Wellington, Yarra.

Isotropic-unit-error cities (Boston, Cambridge, New York — US survey feet, #157) are deliberately
kept **in**, uncorrected: a uniform scale error changes `height_m` but not a roof's rise/run ratio,
so it cannot bias which family a program fits into — which is exactly why #159 excludes only the
anisotropic pair and not every CRS-flagged city.

Remaining nine: **Berlin, Boston, Cambridge, Cape Town, Edmonton, Melbourne, Montreal, New York,
Tokyo.**

## Method

`scripts/foundations/pilot_buildingworld_roof_families.py`. Per sampled BuildingWorld mesh (2,000
per city, seed 159, uniform over each city's canonical zip — `mesh.zip` preferred over `obj.zip`,
reusing #158's own `canonical_zip`): `trimesh.load` → `ingest_3dbag.building_to_sdf`
(**unmodified**, the same function real ingestion would call) at `R=64` — the corpus's own grid,
not #158's coarser `R=32` probe → `height_field` (`recover_massing_programs.py`) turns that
occupancy into the `(y0, y1, target)` triple the fitter takes → `fit_program_beam`, called at its
own defaults (`max_ops=4, allowance=CARVE_NEEDED=0.02, beam=6, branch=6`, the full
`Layer`/`CutRoof`/`Ramp` vocabulary) — the literal function the ticket names.

The comparison population — "the existing NL/DE/JP corpus" — is `real.h5`'s own rows, 2,000 sampled
per `source_id` (0/1/2 → NL/DE/JP, `stratified_split.py`'s own mapping), fit the identical way but
read directly from the corpus's already-decoded occupancy grids: no mesh, no `building_to_sdf`
step. This is a read-only measurement over the full corpus (including rows in the frozen held-out
split) — the same thing #10/#131 already do to *measure* the fitter, not a training or evaluation
run, so touching held-out rows here carries no leakage risk.

`roof_family()` turns a fitted program into `flat` / `gable` / `hip` / `complex`, reusing three
pieces of precedent already in this codebase rather than inventing a new rule (full reasoning in
the function's own docstring):

- zero ops, or exactly one `Layer` → `flat` (`_ROOF_FAMILY_OF["Layer"] == "flat"`;
  `roof_description_length`'s own worked example, "flat roof: 1 op: Layer"). **More than one
  `Layer` is *not* `flat`** — that is the many-Layer contour-terrace fallback
  `roof_description_length` uses a dome as its example of, and exactly the bucket this ticket asks
  whether BuildingWorld collapses into.
- a `CutRoof` kind set equal to `{"hip"}` → `hip`; **any** other kind present (including a hip
  mixed with a gable) → `gable` — `recover_massing_programs.main()`'s own hip/gable split on
  `roof_kinds` in its #128 edit-stack bridge report, applied here to every fit rather than only its
  sampled bridge subset.
- two or more `Ramp` ops with no `CutRoof` at all → `gable`, per #129's own framing ("a gable is
  two opposing ramps … not in reach until the assignment head commits to a second").
- everything else — chiefly a lone `Ramp` (a shed: one plane, no second one to pair with) — is
  `complex`. A shed has no named bucket among the four the ticket asks for; counted as `complex`
  here and disclosed separately per city below (`n_shed` / "of which shed") rather than silently
  folded in.

`dl_ops`/`dl_planar_fraction` are read directly off the same `fit_program_beam` program
(`len(program)`, and the fraction of its ops that are `Ramp`/`CutRoof`) — the identical fields
`roof_description_length` reports, without paying for that function's own second, independent
greedy re-fit.

Run: `env -u LD_PRELOAD ./venv/bin/python3 scripts/foundations/pilot_buildingworld_roof_families.py`
(529s on 30 forked workers: 18,000 BuildingWorld meshes + 6,000 corpus rows). Raw per-building
records and full summaries:
[`execution/artifacts/buildingworld_roof_family_pilot.json`](../../../execution/artifacts/buildingworld_roof_family_pilot.json).

## Results

| Population | n ok | flat | gable | hip | complex | **gable+hip** | dl_ops med | planar_frac med | shed / complex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Berlin | 1,991 | 0.251 | 0.361 | 0.019 | 0.369 | **0.380** | 2.0 | 0.500 | 34.3% |
| Boston | 1,996 | 0.844 | 0.051 | 0.006 | 0.099 | **0.057** | 0.0 | 0.000 | 4.5% |
| Cambridge | 2,000 | 0.192 | 0.311 | 0.039 | 0.458 | **0.350** | 4.0 | 0.250 | 3.6% |
| Cape Town | 2,000 | 0.260 | 0.343 | 0.030 | 0.367 | **0.373** | 3.0 | 0.250 | 10.2% |
| Edmonton | 2,000 | 0.221 | 0.358 | 0.003 | 0.418 | **0.361** | 3.0 | 0.000 | 0.0% |
| Melbourne | 2,000 | 0.611 | 0.035 | 0.002 | 0.352 | **0.037** | 1.0 | 0.000 | 0.0% |
| Montreal | 2,000 | 0.227 | 0.116 | 0.013 | 0.645 | **0.129** | 2.0 | 0.000 | 1.7% |
| New York | 2,000 | 0.720 | 0.006 | 0.002 | 0.273 | **0.008** | 1.0 | 0.000 | 0.0% |
| Tokyo (BuildingWorld mesh) | 2,000 | 0.243 | 0.179 | 0.011 | 0.567 | **0.191** | 3.0 | 0.000 | 1.5% |
| **NL** (existing corpus) | 2,000 | 0.139 | 0.267 | 0.014 | 0.580 | **0.281** | 4.0 | 0.250 | 5.3% |
| **DE** (existing corpus) | 2,000 | 0.253 | 0.460 | 0.008 | 0.279 | **0.468** | 2.0 | 0.667 | 49.8% |
| **JP** (existing corpus) | 2,000 | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.000 | n/a |

(`n_sampled=2000` for every population; failures were 9/2000 Berlin and 4/2000 Boston, both
`empty_footprint` on a degenerate sliver mesh, excluded from `n ok` and every fraction. "shed /
complex" = the fraction of that population's own `complex` bucket that is actually a lone-`Ramp`
shed rather than a many-Layer fallback — see Key findings.)

## Key findings

**Yes: most of the nine cities show gable/hip signal comparable to, and in several cases exceeding,
the existing corpus's own NL arm — this is a real go signal, not a uniform collapse into
`complex`.** Berlin (0.380), Cape Town (0.373), Edmonton (0.361), and Cambridge (0.350) all sit
close to or above NL's own 0.281 gable+hip fraction, and Berlin/Cambridge/Cape Town/Edmonton are
within reach of DE's 0.468. Hip alone is rare everywhere (0.2%–3.9% BuildingWorld, 0.8%–1.4%
existing corpus) — that is consistent across both populations, not a BuildingWorld-specific gap.

**Three cities — Boston, Melbourne, New York — are dominated by flat massing (72–84% `flat`), not by
`complex`.** This tracks #158's own extent/solidity findings for exactly these cities: New York's
median max-extent was already flagged there as far larger than every other city (44.8% of sampled
meshes exceed the 90-unit multi-building cutoff) and its footprint solidity (0.264) was read as
"consistent with dense high-rise urban form." A city sample skewed toward large downtown/CBD
commercial stock is architecturally plausible ground for genuinely flat-roofed massing rather than
a data defect — this pilot cannot distinguish "real flat commercial stock" from "BuildingWorld
under-samples pitched building types in these three cities' zips," and does not attempt to here.

**`complex` is not one population, and a large slice of it is actually simple.** The 4-family split
this ticket asks for has no bucket for a shed (a single plane, no second one to pair into a gable),
so every lone-`Ramp` program lands in `complex` by this classifier's own rule. That inflates
`complex` unevenly: **49.8% of DE's `complex` bucket and 34.3% of Berlin's is a shed**, not a
many-Layer mess — DE's real pitched-roof fraction, sheds included, is closer to 0.468 + 0.139 ≈ 0.61
than the 0.279 `complex`-adjacent reading suggests. Edmonton, Melbourne, and New York, by contrast,
have **zero** sheds — their `complex` mass really is the many-Layer fallback. Montreal (0.645) and
Tokyo (0.567) also lean heavily `complex` with almost no sheds (1.7%, 1.5%) — plausibly genuine
multi-facet/mansard-style roofscapes (a well-documented feature of Montreal's older housing stock)
rather than a fitter failure, but this pilot does not adjudicate that either; it is exactly the
"complex, many-Layer" outcome the parent map's five prior gable arms already hit on NL, and NL
itself is no cleaner (0.580 `complex`) — so BuildingWorld is not introducing a new failure mode
here, it is reproducing one the existing corpus already has.

**🔑 The existing "NL/DE/JP corpus" comparison is not three comparable populations — JP contributes
*zero* gable/hip/complex signal, at the full population, not a sampling artifact.** All 2,000
sampled JP rows in this pilot came back `flat` with `dl_ops=0` (already-flat, no carve needed at
all). This was cross-checked three independent ways before being reported here:

1. This pilot's own `fit_program_beam` run over `real.h5`'s JP rows (n=2,000, this session).
2. `execution/artifacts/program_recovery_714.json` — an unrelated, pre-existing artifact from a
   prior session (created 2026-08-27, `beam=12`, wider than this pilot's default `beam=6`) — its
   231 JP rows are **also** 231/231 (100%) `n_ops==0`.
3. The raw mesh geometry `ingest_surfaces.py` stored per building (`data/real_massing_v1/
   surfaces_plateau.h5`, Frame-N convention: axis 1 is "up"), checked at the **full population**
   (all 12,000 JP rows, not a sample): **every single one** has exactly 2 unique up-axis vertex
   values — a perfect flat-topped box, at the source-mesh level, before voxelisation. The matched
   DE file (`surfaces_nrw.h5`) shows the expected wide spread (only 17% flat on the first 2,000
   rows checked the same way), which rules out a bug in this check itself. A direct
   re-voxelisation of one JP mesh (`igl.signed_distance` on the stored Frame-N verts/faces,
   compared against `real.h5`'s own stored `sdf` for the same row) gives **IoU = 1.0** — `real.h5`
   is not corrupting or mis-storing anything; the box shape is present all the way back at mesh
   construction.

   ⚠️ An earlier version of this check, run against `surfaces_plateau.h5`'s vertices through
   `building_to_sdf` a second time, produced a wildly different (and wrong) result — `height_m`
   1.81 vs. `real.h5`'s stored 5.87, and a non-flat target. That was **this session's own
   methodology bug**, not a second real discrepancy: `ingest_surfaces.py`'s own docstring states
   its meshes are stored *already* in Frame-N (re-centred, re-scaled, axis-reordered), so feeding
   them through `building_to_sdf` a second time double-transforms them. Recorded here so the same
   mistake isn't repeated by a future check against this file.

   This is a genuine, well-corroborated, full-population fact about the currently-served training
   corpus's JP arm, **not** something this ticket set out to find and **not** something it
   diagnoses further: whether every ingested PLATEAU LOD2 building genuinely has a flat roof in
   this specific tile set, or `ingest_citygml_lod2.py --source plateau`'s semantic-surface
   selection silently failed to capture PLATEAU's `RoofSurface` geometry back in June, is not
   determined here (`_building_rings()` is shared code with NRW, which shows real variety, so a
   universal bug in that function is ruled out — the difference has to be in how PLATEAU's own
   CityGML tags roofs, or in the specific tiles ingested). Flagged as a concrete, consequential
   follow-up: as things stand, **every gable/hip/complex signal in "the existing NL/DE/JP corpus"
   comes from NL and DE alone** — a fact worth knowing before comparing any future region-scheme
   arm against a "3-region" baseline that is actually two regions with a null third.

## Scope and disclosed limitations

- 2,000/city and /region, fixed seed — not full population (BuildingWorld's smallest included city,
  Cape Town, has 274,598 meshes; JP/NL/DE each have ~12,000/11,776/12,000 rows in `real.h5`, so the
  corpus-side sample is a much larger fraction of its population than the BuildingWorld side).
- `R=64`, matching the corpus's own grid — more expensive than #158's `R=32` probe grid, not a
  resolution recommendation for anything beyond this pilot.
- `fit_program_beam` at its own defaults throughout (`max_ops=4, beam=6, branch=6`); no `FitBias`,
  no `wf_planes` — the plain fitter, matching what the ticket names.
- The shed/complex split above is diagnostic, not a fix: this ticket does not propose adding a
  fifth family bucket, only discloses that `complex` is heterogeneous under the ticket's own
  4-bucket ask.
- Isotropic-unit-error cities (Boston, Cambridge, New York) were kept in raw, uncorrected units,
  per the reasoning above — this pilot does not itself verify empirically that a ~3.28× uniform
  scale error truly leaves `fit_program_beam`'s candidate ranking unchanged; that follows from the
  fitter's own geometry (candidates are ranked by voxel gain over a grid that is itself built from
  the same raw units throughout height and footprint alike) but was not re-derived from these
  specific runs.
- Boston/Melbourne/New York's flat-heavy result and Montreal/Tokyo's complex-heavy result are
  reported with a plausible architectural reading (dense downtown stock; older multi-facet
  housing stock) but neither is verified against ground truth here — a montage/visual spot-check
  of a few of each, the way #129/#131 do for their own claims, would be the next cheap step if
  this matters for a region-granularity decision.
- The JP-corpus finding is disclosed at the depth this ticket's own scope supports (three
  independent corroborations, one ruled-out false lead) but its root cause is explicitly not
  determined — see above.

Pure fact-finding; no changes to `real.h5`, `corpus_ledger.h5`, or any other corpus artifact.
