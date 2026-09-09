# #157 — BuildingWorld per-city CRS, units, and reference elevation

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:research`, pure fact-finding —
opened and closed by this audit with no code or data changes. Feeds the policy decision at
[#165](https://github.com/danvisai/SDFusion/issues/165), which is blocked on this ticket and on
[#158](https://github.com/danvisai/SDFusion/issues/158)'s watertightness/extent profiling.*

> For each of BuildingWorld's 19 mesh-bearing cities, determine the actual coordinate reference
> system and units the source pipeline used, cross-referencing each city's own official
> open-geodata documentation against the observed coordinates in the downloaded `.obj` files; and
> each city's known ground/reference elevation, to sanity-check z-up and elevation-above-sea-level
> against observed mesh z-origins. Produce a table: city → CRS/units verdict (metric-correct /
> isotropic-unit-error / anisotropic-projection-error / unknown) → confidence → source. Pure
> fact-finding — does not decide whether to reproject or drop affected cities
> ([#165](https://github.com/danvisai/SDFusion/issues/165)'s job).

Four cities were already flagged suspect this session by eyeballing raw origins before this audit
started: New York and Philadelphia (US survey feet), Toronto and Mississauga (Web Mercator). This
audit checks all 19 systematically, with sampled coordinates and cited sources for every verdict.


## Method

For each city, 20 `.obj` files were sampled at random (seed fixed) directly out of whichever zip
actually exists under `data/buildingworld_mesh/<City>/{mesh,obj}/{mesh,obj}.zip` — the layout is
inconsistent per city, enumerated once and resolved per city rather than assumed
(`mesh/mesh.zip`, `mesh/obj.zip`, or `obj/mesh.zip`, city-dependent). Each `.obj`'s `v x y z` lines
were parsed directly from inside the zip (no extraction) to get a per-file bounding box.

Two independent checks, cross-referenced against each other:

1. **Forward-transform check.** For each city, its well-known (lat, lon) center was forward-projected
   through a shortlist of candidate EPSG CRSs (`pyproj`) and compared against the *median* observed
   `x_min`/`y_min` across the 20 samples. A candidate that lands within a few km of the observed
   median (reasonable, since sampled buildings scatter across the whole city, not one point) is
   strong evidence for that CRS; a candidate off by hundreds of km or more is ruled out.
2. **Official source check.** Web search + fetch against each city's own open-geodata program,
   looking for a stated EPSG code, state-plane/provincial-plane zone, or "exported for
   Blender/FBX/CAD with a local offset" convention, then checking whether that stated system predicts
   the observed coordinates.

Sample script and raw per-file bounding boxes are not committed (this ticket makes no code/data
changes); the scripts used are throwaway and the exact commands are reproducible from this
document alone (sample N `.obj` files per city zip, parse `v` lines, compare against the EPSG list
cited per city below).


## Summary table

| City | Verdict | Confidence | CRS identified | Evidence |
|---|---|---|---|---|
| Adelaide | metric-correct (offset origin) | High (units), medium (origin) | GDA94 / MGA Zone 54 (EPSG:28354), locally re-centered | data.sa.gov.au states EPSG:28354; local offset matches "for Blender/FBX" export convention |
| Berlin | metric-correct | High | ETRS89 / UTM zone 32N (EPSG:25832) — official LoD2 source states zone 33N (EPSG:25833) | Tight numeric match to 25832; Berlin's own GDI metadata states 25833 (provenance nuance, not a units defect) |
| Boston | **isotropic-unit-error** | High | NAD83 Massachusetts Mainland State Plane, **US survey feet** (EPSG:2249), documented local offset | BPDA/pbcGIS "3D Smart Model" docs state feet + offset (732200 ft W, 2902900 ft S); tight numeric match |
| Calgary | metric-correct | High | NAD83 Alberta 3TM ref. merid. 114°W (EPSG:3776) | Open Calgary states EPSG:3776 for "3D Buildings"; tight numeric match |
| Cambridge (MA) | **isotropic-unit-error** | High | NAD83 Massachusetts Mainland State Plane, **US survey feet** (EPSG:2249), documented local offset | cambridgema.gov GIS states feet + NAVD88-feet + offset (731100, 2902900); same vendor pipeline as Boston (`CAM-*` vs `BOS-*` filenames) |
| Cape Town | metric-correct (deduced) | Medium-high | Hartebeesthoek94 / Lo19, South African survey grid (EPSG:2048), axis-negated | City of Cape Town LOD2.2 CityGML confirmed to exist; numeric match only after negating both axes of the "South Orientated" Lo19 convention |
| Edmonton | metric-correct | High | NAD83 Alberta 3TM ref. merid. 114°W (EPSG:3776) | City of Edmonton open data publishes elevation/contour products in "3TM"; tight numeric match |
| Greater Geelong | metric-correct (offset origin) | High (units), medium (origin) | GDA94 / MGA Zone 55 (EPSG:28355), locally re-centered | City of Greater Geelong mandates MGA55/GDA94 for all spatial data; near-zero local coordinates consistent with a recentered export |
| Melbourne | metric-correct (units), z-datum stripped | Medium | GDA94/GDA2020 MGA Zone 55 (EPSG:28355/7855), locally re-centered; z is NOT elevation | City of Melbourne's own "3D Textured Mesh" doc states MGA55 + AHD vertical datum, but every sampled file has `z_min == 0.000` exactly — elevation has been replaced by height-above-building-base somewhere upstream |
| Mississauga | **anisotropic-projection-error** | High | Raw coords match **EPSG:3857 Web Mercator**; official base map is UTM Zone 17N NAD83 (metres) | data.mississauga.ca states UTM17N metres for its base map; observed X/Y match 3857 almost exactly instead; observed Z (98–210 m) matches Mississauga's real ~90–270 m ASL range, confirming Z is real metres while X/Y are Mercator-inflated |
| Montreal | metric-correct | High | NAD83(CSRS) MTM Zone 8 (EPSG:2950/32188) | Ville de Montréal open data states "NAD83 SCRS(98), MTM-08"; tight numeric match |
| New York | **isotropic-unit-error** | High | NAD83 New York Long Island, **US survey feet** (EPSG:2263) | NYC's own 3-D Building Model / Building Footprints metadata (GitHub `nyc-geo-metadata`) states EPSG:2263; tight numeric match; issue's own prior spot-check |
| Perth | metric-correct (offset origin) | Medium | GDA94/GDA2020 MGA Zone 50 (EPSG:28350/7850), locally re-centered | WA's Landgate mandates GDA2020 statewide; folder names (`3D_Buildings_Level_2_Dec_2024_WSL*`) look like an official WA release; near-zero local coordinates, but z-origin is negative and inconsistent with Perth's real (~0–40 m) ASL — see notes |
| Philadelphia | **mixed** (see notes) | High | `2010_ph_downtown/` = NAD83 PA South, **US feet** (EPSG:2272); `2015_scene/` = NAD83 PA South, **metres** (EPSG:32129) | Both are literally the PASDA-hosted "Philadelphia Building 3D Models 2010" / "...2015" datasets; PASDA's own `CityPhilly` ArcGIS service confirms EPSG:2272/US feet as Philly's GIS default; exact numeric matches for both subsets |
| San Francisco | metric-correct | High | NAD83 UTM Zone 10N (EPSG:26910/32610) | Standard SF Bay Area GIS convention (UTM10N metres); tight numeric match |
| Tokyo | metric-correct (deduced) | Medium-high | JGD2011 Japan Plane Rectangular CS IX (EPSG:6677) | PLATEAU (Tokyo's national digital-twin program) natively ships geographic + height, not this plane-rectangular metric form; good numeric match suggests a reprojection step upstream, still isotropic/metric |
| Toronto | **anisotropic-projection-error** | High | Raw coords match **EPSG:3857 Web Mercator** | Toronto's official GIS convention is not Web Mercator (its open-data guidance defaults to WGS84 for publishing, its 3D Massing source is UTM/MTM-based); observed X/Y match 3857 almost exactly; z is height-above-base (`z_min == 0` for every sample), so the elevation check does not apply here |
| Wellington | metric-correct | High | NZGD2000 / NZTM2000 (EPSG:2193) | LINZ Data Service states EPSG:2193 for Wellington LiDAR/building products; tight numeric match |
| Yarra | metric-correct | Very high | GDA94 / MGA Zone 55 (EPSG:28355), **not** re-centered | data.gov.au's "City of Yarra 3D buildings by suburb" dataset ships `Richmond.3ds`/`Fitzroy.3ds`/etc. — exact match to the sampled subfolder names — and states MGA55 directly; near-exact numeric match, no offset. **But**: the `richmond/` subfolder specifically has its Y axis sign-flipped relative to every other Yarra suburb — see notes |

Bold verdicts are actionable defects if raw coordinates are ingested as metres without correction.
Non-bold "metric-correct" entries should still be reviewed for the origin-recoverability caveats
noted per city below before #165 decides ingestion policy.


## Per-city detail

### North America — Massachusetts state-plane family (Boston, Cambridge)

Both cities' `.obj` files use small "local project" coordinates (Boston: x∈[14742, 51767],
y∈[9313, 53807]; Cambridge: x∈[3807, 7201], y∈[10449, 13419]) that don't match any UTM/state-plane
value directly. The filename convention gave it away first — Boston's files are named
`BOS-<5>-<5>-<5>_Building.obj`, Cambridge's `CAM-<5>-<5>-<5>_building.obj`, the same scheme, strongly
suggesting the same vendor pipeline (**pbcGIS**, the host of Boston's official 3D data downloads).

- **Boston** — [Analyze Boston / pbcGIS's 3D Data Documentation](http://maps.bostonplans.org/3d/3D_Data_Documentation.pdf)
  states: *"State Plane Massachusetts Mainland FIPS 2001 Feet, NAD1983, with coordinate offsets of
  732,200 feet West and 2,902,900 feet South."* Forward-transforming Boston's center through
  EPSG:2249 (MA Mainland, US feet) gives x=775,383.6, y=2,956,557.9; subtracting the documented
  offset gives (43,183.6, 53,657.9) — squarely inside the observed sample range
  (x median 26,982.1, y median 32,431.4; ranges above). z_min median 66.5 (as **feet** per the same
  doc → 20.3 m) and z_max median 92.0 ft (28.0 m) are consistent with Boston's real relief (near
  sea level downtown up to ~100 m on the city's highest hills).
- **Cambridge** — [cambridgema.gov's "Building Model Collection OBJ" page](https://www.cambridgema.gov/gis/3d/3ddata/buildngmodelobj)
  states explicitly: *"Projected Coordinate System: State Plane Massachusetts Mainland (Feet), North
  American Datum of 1983"*; *"Vertical Datum: North American Vertical Datum, 1988 (NAVD 88) Feet
  (Height)"*; origin offset X=731,100 ft, Y=2,902,900 ft, elevation 0, plus a **0.34° clockwise
  rotation** to true north. Transforming the documented origin (71.223391°W, 42.213379°N) through
  EPSG:2249 and subtracting the stated offset lands within 3.2 ft of (0,0) — an essentially exact
  confirmation of the stated convention. Observed z_min median 3.23 ft (0.98 m), z_max median
  10.61 ft (3.23 m) — plausible for low-lying Cambridge, on the low side of typical (Cambridge
  averages are usually cited nearer 10–40 ft), but not inconsistent given a 20-file sample.

**Verdict for both: isotropic-unit-error.** Both cities' raw coordinates are genuine, correctly
scaled State Plane Massachusetts Mainland positions — just in **US survey feet**, not metres, with a
documented (and in Cambridge's case, slightly rotated) local offset applied for CAD/3D-tool
compatibility. If ingested as metres unmodified, every dimension (footprint, height, extent filter)
is inflated ~3.28×, uniformly across x/y/z — this doesn't warp roof pitch (isotropic), but does
directly corrupt `height_m` and any extent-based filtering exactly as #156 flagged for New York and
Philadelphia.


### New York and Philadelphia — the previously-flagged US survey-feet cities, confirmed

- **New York** — [`CityOfNewYork/nyc-geo-metadata`'s `Metadata_3DBuildingModel.md`](https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/Metadata/Metadata_3DBuildingModel.md)
  and `Metadata_BuildingFootprints.md` both state NAD83 New York State Plane, Long Island Zone, **US
  feet (EPSG:2263)**. Forward-transforming NYC's center through EPSG:2263 gives (982,586.6,
  198,968.8); observed median (992,495.8, 216,216.4) is within ~10–17 km — reasonable for a
  city-wide sample of a metro spanning tens of km. z_min median 22.4 ft (6.8 m) matches NYC's
  mostly-low-lying built terrain well. **Isotropic-unit-error, high confidence** — matches the
  session's prior spot-check exactly.
- **Philadelphia — genuinely mixed within one city's zip.** The sampled files split cleanly into two
  named subfolders with two different coordinate systems:
  - `2010_ph_downtown/*.obj`: x∈[2,685,900, 2,695,815], y∈[233,942, 240,754] — matches
    EPSG:2272 (NAD83 PA South, **US feet**) forward-transform (2,693,060, 236,195) to within ~1–3
    km. [PASDA's `CityPhilly` ArcGIS REST endpoint](https://services.pasda.psu.edu/arcgis/rest/services/pasda/CityPhilly/MapServer/0?f=pjson)
    confirms Philadelphia's GIS default is `wkid 102729`/`latestWkid 2272`, `Foot_US` — the same
    system. z values in this subset (base elevations 8.5–38.6 ft, i.e. 2.6–11.8 m, clustering
    tightly around 35–38 ft = 10.7–11.6 m) match Philadelphia's well-known ~39 ft (12 m) average
    ASL almost exactly.
  - `2015_scene/*.obj`: x∈[819,284, 822,205], y∈[71,666, 73,571] — matches EPSG:32129 (NAD83 PA
    South, **metres**, same zone) forward-transform (820,846.4, 71,992.4) to within ~1 km. z_min
    values (8.5–22.7 m) are also consistent with Philadelphia's ~12 m ASL.
  - Both are literally the PASDA-hosted [*"Philadelphia Building 3D Models 2010"*](https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7145)
    and [*"...2015"*](https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7146) releases —
    the folder names in the zip are the dataset names, unmodified.

**Verdict: mixed, high confidence for both halves.** Philadelphia is not one uniform
units error — it's two real PASDA vintages of the *same* state-plane zone, one in feet (isotropic
unit error) and one already in metres (metric-correct). A per-file (not just per-city) units
decision is required for Philadelphia specifically; #165 should know the `2010_ph_downtown/` vs
`2015_scene/` split is a clean, mechanical way to route the fix rather than guessing per building.


### Canada — Alberta 3TM (Calgary, Edmonton), Quebec MTM (Montreal), Web Mercator (Toronto, Mississauga)

- **Calgary** — [Open Calgary's "3D Buildings" dataset page](https://data.calgary.ca/Base-Maps/3D-Buildings/6jhg-8gyc/data)
  states NAD83 Alberta 3TM 114W (**EPSG:3776**). Forward-transform of Calgary's center gives
  (−5,041.9, 5,656,495.3); observed median (−5,834.2, 5,657,863.1) is within ~0.8–1.4 km.
  z_min median 1,091.9 m vs. Calgary's commonly cited ~1,045 m (downtown/tower base) to ~1,084 m
  (airport) — consistent, on the high side because the sample skews toward the city's upland
  neighborhoods rather than the river valley. **Metric-correct, high confidence.**
- **Edmonton** — City of Edmonton's open data portal publishes its elevation/contour products in
  "3TM" (e.g. *Contour Lines 3TM*, *DEM Points 3TM*), the same Alberta convention Calgary uses.
  Forward-transform through the same EPSG:3776 gives (33,551.0, 5,934,922.3); observed median
  (32,107.2, 5,931,732.0) is within ~1.4–3.2 km. z_min median 680.0 m sits right in Edmonton's
  well-known ~645 m (river valley) to ~723 m (upland plain/airport) range — matching the session's
  own earlier ~674 m spot-check. **Metric-correct, high confidence.**
- **Montreal** — Ville de Montréal's open-data pages for its 3D building releases (2009/2013/2016)
  state *"NAD83 SCRS(98)"* projection with *"MTM-08"* (Modified Transverse Mercator zone 8) and
  *"C-GVD28 (NMM)"* as the vertical reference. Forward-transform through EPSG:2950/32188 gives
  (299,533.0, 5,040,221.0–5,040,222.0); observed median (297,998.1, 5,035,918.8) is within
  ~1.5–4.3 km. z_min median 19.1 m is plausible for Montreal's lower (riverside/downtown)
  neighborhoods versus its Mount-Royal high point (~230 m). **Metric-correct, high confidence.**
- **Toronto** — observed x∈[−8,854,675, −8,834,634], y∈[5,409,410, 5,423,754] matches EPSG:3857
  (Web Mercator) forward-transform of Toronto's center, (−8,836,897.4, 5,411,930.0), to within
  ~2.3–2.8 km — an almost exact match. But Toronto's own GIS conventions are not natively
  Web-Mercator: its open-data guidance for general publishing defaults to WGS84, and its 3D
  Massing source data is UTM/MTM-based, not 3857. z_min is **exactly 0.000** for every one of the
  20 sampled files (including one high-rise reaching z_max=164.6 m) — Toronto's mesh Z axis is a
  per-building height-above-base value, not absolute elevation, so the ASL sanity check does not
  apply here at all; only the X/Y anisotropy claim is being made.
- **Mississauga** — [data.mississauga.ca states its base map coordinates are "in metres based on
  the 6 degree Universal Transverse Mercator system, NAD 1983"](https://data.mississauga.ca) — i.e.
  UTM Zone 17N, NAD83 (metres), not Web Mercator. Yet observed x∈[−8,880,680, −8,863,621],
  y∈[5,390,822, 5,421,573] matches EPSG:3857 forward-transform (−8,865,940.7, 5,402,057.7) to
  within ~0.4–3.9 km — again an almost exact match to 3857, not to the city's own stated UTM17N.
  Critically, Mississauga's z_min (median 178.4 m, range 97.9–210.1 m) is **not** zeroed like
  Toronto's — it lands squarely inside Mississauga's real, well-known ~90–270 m ASL range (Lake
  Ontario at ~75 m, the city sitting on the plateau above it). This is the cleanest possible
  confirmation of the anisotropic mechanism the issue named: **X/Y are Web-Mercator-projected
  (true-distance-inflated by the latitude-dependent Mercator scale factor, ~1.3–1.4× at this
  latitude), while Z is real, undistorted metres** — meaning every roof pitch in this city's meshes
  is measured against a horizontally-stretched footprint, flattening the true pitch.

**Verdict: Toronto and Mississauga are anisotropic-projection-error, high confidence** — for both,
raw X/Y match Web Mercator almost exactly, and for Mississauga that mismatch against the city's own
stated UTM17N convention (plus the ASL-consistent Z) rules out "maybe Web Mercator actually is
official" as an explanation. This distortion is not something the two cities' own government portals
publish — it was very likely introduced by a Web-Mercator-tiled ingestion/basemap step somewhere
between the source and what shipped in BuildingWorld.


### Australia / New Zealand — a Blender/FBX-export family with divergent Z handling

Four cities (Adelaide, Perth, Melbourne, Greater Geelong) show small "local" x/y values (hundreds
to low thousands) that don't match any candidate CRS directly — until each city's real MGA
(Map Grid of Australia) coordinate is forward-computed and found to land *almost exactly at the
dataset's local (0,0)*, meaning a large, real-world-derived offset has been subtracted. This is a
well-documented convention, not a defect: data.sa.gov.au's own "3D Model of the City of Adelaide"
listing offers the model *specifically packaged* for Blender and FBX (3D content-creation tools that
handle small numbers far better than 6-digit UTM/MGA eastings in single precision), alongside a
plain multipatch GIS version. Melbourne and Geelong publish the same underlying MGA-based data
through their own open-data portals but at real-world magnitude; whichever export BuildingWorld
pulled evidently went through the same recentering step.

| City | Local median (x, y) | Real MGA candidate (x, y) | Δ | z_min median | Known ASL | z consistent? |
|---|---|---|---|---|---|---|
| Adelaide | (1,634.8, 1,862.3) | MGA54 (280,847.4, 6,132,257.9) | ~1.6 / 6.1 km | −78.6 m | CBD ~50 m | **No** — negative, inconsistent across files (−92 to +6) |
| Perth | (−280.6, −161.0) | MGA50 (392,307.2, 6,464,484.2) | ~0.3 km both axes | −23.1 m | CBD ~0–40 m | **No** — negative median |
| Melbourne | (−993.9, 2,529.9) | MGA55 (320,704.5, 5,812,911.7) | ~1.0 / 1.0 km | 0.000 m (always) | CBD ~31 m | **N/A** — z is height-above-base, not elevation, for every sample |
| Greater Geelong | (11.3, 97.6) | MGA55 (268,817.1, 5,774,263.8) | ~0.01 / 0.1 km | 21.6 m | coastal, ~10–20 m | **Yes** — plausible real AHD-like elevation |

Data.sa.gov.au confirms Adelaide's dataset CRS as GDA94/MGA Zone 54; City of Greater Geelong
mandates GDA94/MGA Zone 55 for all its spatial data; City of Melbourne's own "3D Textured Mesh
(Photomesh)" documentation states *"Map Projection: MGA Zone 55 (MGA55), Vertical Datum: Australian
Height Datum (AHD)"* for its official product — but what's in the BuildingWorld files has had its
elevation replaced with a per-building relative height (z_min≡0), the same pattern already
observed for Toronto. **The four cities are not a uniform family in Z-handling**: Melbourne strips
elevation entirely, Greater Geelong's Z looks like genuine, untouched AHD, and Adelaide/Perth's Z
is negative and internally inconsistent (Adelaide's own samples range from z≈−92 to z≈+41
depending on the specific building/component file — e.g. a file literally named `PGround_*.obj`
sits at z≈−90 to −99 while a file named `CBAE1_*.obj` sits at z≈0 to +41), suggesting some Adelaide
components carry a different, possibly sub-structure-relative Z reference than others. This was not
resolved further — it needs per-file inspection beyond this ticket's sampling budget, and is flagged
as a concrete follow-up for whoever picks up ingestion.

**Verdict: metric-correct for X/Y units and scale in all four (high confidence, real MGA zones,
documented offset convention)**, but the elevation/reference-elevation half of this ticket's ask is
**inconclusive-to-negative for Adelaide and Perth, inapplicable for Melbourne, and passes for
Greater Geelong** — worth stating plainly rather than forcing one verdict onto all four.

**Yarra** is the exception in this family and the cleanest result of the whole audit: its subfolders
are named `richmond/`, `fitzroy/`, `fitzroy_north/`, `clifton_hill/`, `carlton_north/`, `cremorne/`,
`burnley/` — literal, exact matches to the file names inside
[data.gov.au's "City of Yarra 3D buildings by suburb"](https://data.gov.au/data/dataset/yarra-3d-buildings)
dataset (`Richmond.3ds`, `Fitzroy.3ds`, etc.), which states its CRS directly as **MGA55**
(GDA94, EPSG:28355) with **no offset** — real-world-magnitude coordinates. Forward-transform of
Yarra's center gives (323,799.2, 5,811,935.0); observed median (323,169.9, 5,811,403.5) is within
~0.5–0.6 km, the tightest match in the entire audit. z_min median 15.5 m (range 5.1–37.5 m) is
squarely plausible for inner-Melbourne suburbs at 10–30 m ASL. **Metric-correct, very high
confidence.**

⚠️ **But**: every one of the 6 sampled `richmond/*.obj` files has its **y coordinate sign-flipped**
relative to every other Yarra suburb — e.g. `richmond/richmond_01281.obj` has y∈[−5,812,952.5,
−5,812,949.5], while `burnley/burnley_00269.obj` (a directly adjacent suburb) has y∈[5,811,341.0,
5,811,349.5] — same magnitude, opposite sign. This is internal to Yarra's own corpus, not a
units/projection issue, but it means naive ingestion of the `richmond/` tile specifically would
place every Richmond building ~11.6 million metres away from the rest of the city (wrong
hemisphere numerically) unless the sign is corrected. Flagged as a concrete, file-identifiable
defect for whoever writes `ingest_buildingworld.py`.

**Perth's subfolder names** (`3D_Buildings_Level_2_Dec_2024_WSL1`…`WSL5`) look like an official,
dated Western Australian government release (Landgate mandates GDA2020 statewide for all published
spatial data), but a specific metadata page for this exact dataset was not found — hence "medium"
rather than "high" confidence for Perth, despite the tight numeric MGA50 match.


### Africa — Cape Town

Observed x∈[−60,483, −27,679], y∈[−3,771,800, −3,718,260] doesn't match any standard UTM/Hartebeesthoek
candidate directly (all off by 200,000+ km-equivalent when tried the ordinary way). The City of
Cape Town's own 3D building program is confirmed to model buildings at LOD2.2 via CityGML
([FIG 2026 conference paper](https://fig.net/resources/proceedings/fig_proceedings/fig2026/papers/ts11d/TS11D_rottcher_13989.pdf),
title confirmed via search though the PDF itself 403'd on fetch), but South Africa's own national
survey grid is unusual: **Hartebeesthoek94 / Lo-series** ("Transverse Mercator South Orientated")
zones define their axes as **Y = Westing, X = Southing** — i.e. the opposite sign convention from
ordinary easting/northing. Forward-transforming Cape Town's center through EPSG:2048 (Lo19, the
zone whose 19°E central meridian actually covers Cape Town's 18.42°E) gives (westing=53,251.5,
southing=3,755,480.6). **Negating both** — converting to an ordinary east-positive/north-positive
frame without changing the projection itself — gives (−53,251.5, −3,755,480.6), which matches the
observed median (−44,046.7, −3,763,362.7) to within ~7–9 km, reasonable across Cape Town's large
metro area (the sample spans a ~33 km × 55 km bounding box). z_min median 28.5 m (range 9.1–139.6 m)
is consistent with Cape Town's genuinely wide elevation range (sea-level foreshore up to several
hundred metres on Table Mountain's lower slopes).

**Verdict: metric-correct, medium-high confidence.** This is a deduced result (no single official
page states "Lo19, sign-negated" for this exact building dataset) but the numeric match is precise
and the mechanism — a real, correctly-scaled national survey projection with an axis-sign
convention flip — does not distort shape, scale, or roof pitch at all. It is an isometry, not a
units or projection error.


### Asia — Tokyo

Observed x∈[−16,281, −4,381], y∈[−37,667, −22,748] matches EPSG:6677 (JGD2011 / Japan Plane
Rectangular CS IX — the standard metric plane-rectangular system covering the greater Tokyo region)
forward-transform of Tokyo's center, (−16,568.5, −35,908.5), to within ~0.8–8.8 km. z_min median
3.4 m (range 0.7–39.8 m) is consistent with central Tokyo's mostly-low-lying (shitamachi) terrain
with some higher (yamanote) ground. Parent map [#156](https://github.com/danvisai/SDFusion/issues/156)
already notes that BuildingWorld's Tokyo building-id convention matches **PLATEAU's own** — but
PLATEAU's native CityGML output is geographic (lat/lon) plus height, not this projected
plane-rectangular metric form, so this specific metric form was very likely produced by a
reprojection step somewhere between PLATEAU and what shipped in BuildingWorld (a real, still
isotropic/metric conversion — not a defect on its own, but worth knowing for the
BW-Tokyo-vs-PLATEAU duplicate-risk question #156 already flagged as a separate, live concern).

**Verdict: metric-correct, medium-high confidence** (strong numeric match; no single official page
found stating this exact CRS for this exact BuildingWorld release, hence not "high").


### Europe — Berlin

Observed x∈[779,961, 811,442], y∈[5,817,251, 5,839,648] matches EPSG:25832 (ETRS89/UTM zone 32N)
forward-transform of Berlin's center, (798,812.8, 5,828,000.0), to within ~2.6–4.1 km — a good
match. But [Berlin's own official LoD2 metadata record](https://gdi.berlin.de/geonetwork/srv/api/records/3c7c49af-00a4-3bcd-bc00-20e7f0f1b7bf)
states its reference system as `EPSG:25833` (**UTM zone 33N** — the natural zone for Berlin's
13.4°E longitude), which is a much *worse* numeric fit against what we actually downloaded
(off by ~403 km/11 km — an order of magnitude worse). Both 25832 and 25833 are real, correctly
scaled, isotropic ETRS89 UTM CRSs at this latitude (Germany's federal mapping agencies
(AdV/BKG) use 25832 nationwide, even in the east where 25833 would be the "natural" zone, precisely
to give the whole country one consistent metric CRS) — switching between them shifts absolute
position and applies a small grid-convergence rotation, but does not change scale or distort a
single building's shape. z_min median 37.15 m matches Berlin's well-known ~34 m average elevation
well (already spot-checked this session as plausible at ~43.9 m for one sample).

**Verdict: metric-correct, high confidence** for units/scale; the 25832-vs-25833 mismatch against
Berlin's own stated metadata is a provenance detail (which CRS BuildingWorld's own pipeline actually
used vs. what Berlin publishes) rather than a units or shape defect, and is noted for completeness
per #156's broader interest in same-country/different-pipeline discrepancies.


## Cross-cutting findings

1. **The Z axis is not handled consistently across cities, independent of the X/Y verdict.** Three
   distinct Z conventions were observed: (a) real absolute elevation-above-sea-level (Berlin,
   Boston, Cambridge, Calgary, Edmonton, Montreal, New York, both Philadelphia subsets, San
   Francisco, Tokyo, Wellington, Cape Town, Mississauga, Yarra, Greater Geelong — the majority);
   (b) per-building height-above-base, always starting at z=0, with no absolute elevation at all
   (Melbourne, Toronto); (c) negative and internally inconsistent across different building/component
   files within the same city (Adelaide, Perth). Any ingestion code that assumes "z_min is ground
   elevation" or that tries to use z_min for a cross-city elevation feature will silently produce
   nonsense for Melbourne and Toronto (always 0) and Adelaide/Perth (arbitrary offset, sign not even
   consistent within-city for Adelaide).

2. **At least two of the anisotropic/mixed findings look like they were introduced by a pipeline step,
   not inherited from the named city's own open-data program.** Mississauga's official base map is
   documented UTM17N metres, not Web Mercator; Melbourne's official photomesh doc states MGA55 + AHD,
   not a relative Z. Both mismatches point at BuildingWorld's own construction/tiling pipeline (or an
   intermediate web-map-based capture/export step) rather than at the city governments themselves —
   relevant to #156's broader question about which layer (city source vs. BuildingWorld's own
   reconstruction) owns a given defect.

3. **Philadelphia is not a single-verdict city.** Its zip contains two PASDA vintages
   (`2010_ph_downtown/` in feet, `2015_scene/` in metres) of the same state-plane zone. Any
   per-city units correction must be applied per-subfolder for Philadelphia, not uniformly.

4. **Yarra's `richmond/` subfolder has an internal Y-sign flip** relative to the rest of the same
   city's own corpus — a file-identifiable defect, independent of any CRS/units question, that
   would corrupt ingestion silently if not checked.

5. **The four "recentered for Blender/FBX" Australian cities (Adelaide, Perth, Melbourne, Greater
   Geelong) are metric-correct in scale but cannot be geo-anchored back to an absolute lat/lon from
   the `.obj` alone** — the offset is real and documented at the source (data.sa.gov.au explicitly
   ships this as a Blender/FBX convenience format) but is not recoverable from the file contents
   without the source portal's companion metadata. For this project's actual use (single-building
   metric SDF ingestion, not absolute geopositioning), this is very likely immaterial — but it is
   worth recording explicitly rather than silently assuming "unknown CRS" means "wrong units."


## What this does not decide

Per #156 and #165: this ticket does not decide whether to reproject, drop, or otherwise correct any
city or subfolder — that is #165's job, now unblocked on this half of its two dependencies (the
other being #158's watertightness/extent profiling). It also does not re-litigate BuildingWorld's own
construction methodology (arXiv:2511.06337) beyond what's needed to explain an observed coordinate
pattern — the paper itself states no per-city CRS/projection information; that gap is exactly why
this ticket had to derive it from the raw files and each city's own sources instead.


## Sources

- [City of Boston — 3D Data Documentation (PDF)](http://maps.bostonplans.org/3d/3D_Data_Documentation.pdf) — State Plane MA Mainland feet + offset.
- [Boston 3D Buildings (Existing) — Analyze Boston](https://data.boston.gov/dataset/boston-3d-buildings-existing)
- [City of Cambridge, MA — Building Model Collection OBJ](https://www.cambridgema.gov/gis/3d/3ddata/buildngmodelobj) — feet, NAVD88 feet, offset, rotation.
- [City of Cambridge, MA — 3D Data](https://www.cambridgema.gov/gis/3d/3ddata)
- [Open Calgary — 3D Buildings dataset](https://data.calgary.ca/Base-Maps/3D-Buildings/6jhg-8gyc/data) — EPSG:3776.
- [City of Edmonton — Open Data Portal](https://data.edmonton.ca/) (3TM-convention datasets: Contour Lines 3TM, DEM Points 3TM).
- [Ville de Montréal — données ouvertes, Bâtiment 3D 2016](https://donnees.montreal.ca/dataset/batiment-3d-2016-maquette-citygml-lod2-avec-textures2) — NAD83 SCRS(98) MTM-08, C-GVD28.
- [CityOfNewYork/nyc-geo-metadata — Metadata_3DBuildingModel.md](https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/Metadata/Metadata_3DBuildingModel.md) — EPSG:2263.
- [NYC Open Data Technical Standards Manual](https://cityofnewyork.github.io/opendatatsm/citystandards.html)
- [PASDA — Philadelphia Building 3D Models 2010](https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7145)
- [PASDA — Philadelphia Building 3D Models 2015](https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7146)
- [PASDA `CityPhilly` ArcGIS REST endpoint](https://services.pasda.psu.edu/arcgis/rest/services/pasda/CityPhilly/MapServer/0?f=pjson) — WKID 102729/2272, `Foot_US`.
- [data.mississauga.ca](https://data.mississauga.ca) — UTM Zone 17N NAD83 metres base map convention; [Mississauga LOD2 3D Buildings](https://data.mississauga.ca/maps/6c1846f71d474965831368131ef8e8c0).
- [Toronto Open Data — Step 2: Developing open data](https://open.toronto.ca/docs/staff-guidance/step-2-developing-open-data/) (WGS84 publishing guidance).
- [Toronto 3D Massing Shapefile — ArcGIS](https://www.arcgis.com/home/item.html?id=8db806e13af349389554be4c36fb7437)
- [data.sa.gov.au — 3D Model of the City of Adelaide](https://data.sa.gov.au/data/dataset/3d-model) — GDA94/MGA Zone 54, Blender/FBX/multipatch formats.
- [City of Greater Geelong — 3D Massing Model](https://www.geelongaustralia.com.au/data/item/central-geelong-3d-massing-model.aspx) and [Accepted spatial data formats from contractors](https://www.geelongaustralia.com.au/data/article/item/8d20ce57b4dd26f.aspx) (MGA55/GDA94 mandate).
- [data.melbourne.vic.gov.au — City of Melbourne 3D Textured Mesh (Photomesh) 2020](https://data.melbourne.vic.gov.au/explore/dataset/city-of-melbourne-3d-textured-mesh-photomesh-2020/) — MGA55, AHD vertical datum.
- [data.gov.au — City of Yarra 3D buildings by suburb](https://data.gov.au/data/dataset/yarra-3d-buildings) — MGA55, per-suburb `.3ds` files.
- [City of Cape Town 3D city model (FIG 2026 paper listing)](https://fig.net/resources/proceedings/fig_proceedings/fig2026/papers/ts11d/TS11D_rottcher_13989.pdf); [Cape Town Open Data Portal](https://odp-cctegis.opendata.arcgis.com/).
- [LINZ Data Service — Wellington City LiDAR](https://data.linz.govt.nz/layer/105023-wellington-city-lidar-1m-dem-2019-2020/) — EPSG:2193.
- [MLIT Project PLATEAU](https://www.mlit.go.jp/plateau/en/) — Tokyo digital-twin program (native CityGML, geographic + height).
- [gdi.berlin.de — 3D-Gebäudemodelle im Level of Detail 2 (LoD 2), metadata record](https://gdi.berlin.de/geonetwork/srv/api/records/3c7c49af-00a4-3bcd-bc00-20e7f0f1b7bf) — states EPSG:25833.
- [BuildingWorld: A Structured 3D Building Dataset for Urban Foundation Models (arXiv:2511.06337)](https://arxiv.org/pdf/2511.06337) — the dataset's own paper; confirmed to carry **no** per-city CRS/projection documentation, which is why this audit had to derive it from raw coordinates and each city's own sources instead.
- EPSG codes cross-checked via `pyproj` 3.6.1's bundled EPSG database (forward-transform of each
  city's well-known center coordinate through every candidate CRS named above).
