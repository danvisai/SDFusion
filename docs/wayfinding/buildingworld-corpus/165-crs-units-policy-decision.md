# #165 — Decide the CRS/units policy for BuildingWorld's non-metric cities

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:grilling`. Decided against
[#157](https://github.com/danvisai/SDFusion/issues/157)'s completed CRS/units audit and
[#158](https://github.com/danvisai/SDFusion/issues/158)'s watertightness/extent profile, with
project-owner sign-off on the per-city plan.*

## Scope correction: six cities, not four

This ticket's own text named "at minimum" four suspect cities (New York, Philadelphia, Toronto,
Mississauga) — that was the pre-audit spot-check list. #157's completed systematic audit of all 19
cities found **two more**: Boston and Cambridge are also `isotropic-unit-error` (US survey feet,
Massachusetts State Plane), the same category as New York. The decision below covers all six.

## Decision, per city

| City | #157 defect | #158 effective yield | Decision |
|---|---|---|---|
| New York | isotropic, US feet | 46% | **Reproject** — multiply coordinates by 1200/3937 metres per US survey foot (equivalently divide by ≈3.28083333333 feet per metre) |
| Philadelphia | mixed: `2010_ph_downtown/` feet, `2015_scene/` metres | 18% | **Reproject the feet subfolder only** — per-subfolder fix, not per-city; `2015_scene/` is left as-is |
| Boston | isotropic, US feet | 81% | **Reproject** — same exact-factor fix as New York |
| Cambridge | isotropic, US feet | 45% | **Reproject** — same exact-factor fix as New York |
| Mississauga | anisotropic, Web Mercator | 97% | **Reproject** — inverse Web Mercator → UTM 17N (metric), a standard `pyproj` transform, not a heuristic |
| Toronto | anisotropic, Web Mercator | **5%** | **Drop** — reprojection wouldn't be wasted on a CRS problem so much as on a city whose own mesh quality is independently poor: 96.3% inconsistent winding (a defect #157 never flagged, found in #158) and `scattered` as the dominant failure mode regardless of CRS. Fixing the projection recovers a shrinking share of an already-broken population. |

The isotropic-feet fix (four cities) and the anisotropic-Mercator fix (Mississauga) are both
well-defined, exact corrections — this isn't a "cheap vs. risky" split, it's "worth doing" vs. "not
worth doing on a city that's mostly unusable anyway." Toronto is the only city dropped, and it's
dropped on data-quality grounds independent of the CRS question, not because reprojection itself is
hard.

## Two findings from #157 that are in scope here but need no fix

- **Melbourne and Toronto's z-datum is stripped** (mesh z_min ≡ 0 for every building, not real
  above-sea-level elevation). This does **not** need correcting for this effort: `real.h5` stores
  `height_m` as `building_to_sdf`'s `ext[2]` — the building's own relative height — never absolute
  elevation. If a future ticket wants true ASL (e.g. a region-conditioning scheme keyed to
  elevation — see #171's open question), it will need to re-derive it per-building from each city's
  reference elevation table in #157's audit; that's out of scope here.
- **Yarra's `richmond/` subfolder has a Y-sign bug**, independent of any CRS category. Fixing it is
  cheap and mechanical (flip the sign on ingest for that subfolder specifically) but Yarra's overall
  effective yield is 15% for unrelated reasons (`scattered` damage), so this fix's payoff is capped
  regardless. Apply it if Yarra is ingested at all; it is not itself a reason to include or exclude
  Yarra.

## Acceptance test (this ticket's own question, answered)

Before trusting reprojected pitch/height labels for the gable-recovery arm this whole effort exists
for:

1. **Extent-cutoff sanity**: after correction, each reprojected city's `frac_extent_gt_90`
   (buildings exceeding the existing 90 m multi-building cutoff) should drop to a range comparable
   with already-metric-correct cities of similar urban density (single digits to ~10%, per #158's
   table) — not stay near the 45–49% raw-unit rate New York/Philadelphia show today. A city that
   doesn't move after correction means the fix didn't take.
2. **Cross-city pitch-distribution check (Mississauga specifically, the anisotropic case that
   matters)**: compute the roof dihedral-angle (pitch) distribution on a sample of corrected
   Mississauga buildings and compare it against Calgary, Edmonton, and Montreal — same country,
   already metric-correct per #157, comparable building stock. A residual anisotropy would show up
   as Mississauga's roofs reading systematically flatter than these references; if they don't match
   within a reasonable band, the reprojection is wrong or incomplete.
3. **Spot-check against ground truth**: pick 5–10 corrected Mississauga buildings and sanity-check
   scale against public satellite/street imagery for the same footprints.

This test is `ingest_buildingworld.py`'s (#174) job to run at ingest time, not something to execute
speculatively now — recorded here so #174 has a concrete, pre-registered bar rather than "heights
look plausible."

## Status: decided (2026-09-06)

Project-owner-approved: reproject New York, Philadelphia (feet subfolder only), Boston, Cambridge,
and Mississauga; drop Toronto. `ingest_buildingworld.py` (#174) should apply these five cities'
corrections before the 90-unit extent filter runs, per the finding above that the filter behaves
very differently pre- vs. post-correction.
