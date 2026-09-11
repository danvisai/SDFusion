# #160 — Geometric duplicate gate: BuildingWorld Tokyo vs. PLATEAU, Berlin vs. NRW

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:task` — a corpus-safety check,
no changes to `real.h5`, `vecset_latents.h5`, or any ingested row.*

> BuildingWorld Tokyo mesh ids follow PLATEAU's own gml:id convention, and the corpus's existing JP
> rows come from tokyo23ku PLATEAU tiles, so there is a live risk some BuildingWorld Tokyo meshes
> are geometric twins of buildings already in `real.h5`, including some in the FROZEN held-out set.
> Berlin's ids share the same official German LoD2 programme family as NRW's, at lower probability.
> id-based matching will not reliably catch this — `bag_id` is a fixed-width `S64` field and
> PLATEAU ids are already observed truncated mid-uuid. Build a geometric dedup gate instead —
> footprint IoU plus `|height_m|` match, computed directly from geometry — and run it against every
> PLATEAU row (for Tokyo) and every NRW row (for Berlin) in `real.h5`. Report match counts and the
> matched building list. Any match, especially a frozen-held-out one, must be excluded from
> ingestion — just execution and a record of what got excluded and why.

## Method

`scripts/foundations/dedup_buildingworld_geometric.py`. The one fact that shapes everything below,
established *before* writing any matching code: `real.h5`'s own `footprint` column cannot be used
for this check. `building_to_sdf` (`scripts/ingest_3dbag.py`) recentres and rescales every building
into its own `[-1,1]` frame and never stores the discarded centroid/scale — so a geometric duplicate
gate needs a WORLD-frame footprint from both sides, re-derived from geometry, not read off the
packed corpus. (`height_m` *is* kept in true metres in `real.h5` — used below not for matching, but
as an independent consistency check on the re-fetch itself.)

- **BuildingWorld (both cities)**: loaded directly from `data/buildingworld_mesh/{Tokyo,Berlin}/
  mesh/mesh.zip`, **every** mesh (35,577 Tokyo, 523,213 Berlin) — not a sample. Confirmed
  empirically (not assumed) that vertex coordinates are already real-world-magnitude and
  non-recentred: Tokyo lands in EPSG:6677 (Japan Plane Rectangular CS IX), Berlin in EPSG:25832
  (UTM32N) — both metric, both absolute. Footprint = convex hull of XY vertices; height = Z extent.
- **PLATEAU (existing JP rows)**: `real.h5`'s 12,000 JP rows come from exactly 3 tiles —
  `533937_2.zip`, `533954_2.zip`, `533957_2.zip` (verified directly against the live file, not
  assumed from the wider `PLATEAU_TILES` list `ingest_citygml_lod2.py` could draw from). Re-fetched
  those 3 tiles from the same PLATEAU S3 source the original ingester used, re-parsed with the
  ingester's own `_buildings_from_gml`/`_building_rings` (byte-identical id construction, so every
  building is matched back to its `real.h5` row by id, not by proximity), but **skipping** the
  per-building recentring step. Its raw geographic posList (lat/lon/height, EPSG:6668 — the
  horizontal-only component of the compound CRS the ingester's header-sniff calls "6697"; only
  lon/lat are reprojected here, height comes from the posList's own metre column, so the
  vertical-datum difference between the two never enters this computation) is reprojected once via
  `pyproj` straight to EPSG:6677 — the SAME CS BuildingWorld Tokyo's own meshes are already in — so
  both sides land in one shared frame without inventing a reference point. **All 12,000/12,000 JP
  rows were re-located, with 0 height-consistency drift** (see "Two disclosed data-quality gaps"
  below for why this isn't automatic, and why NRW's equivalent numbers are lower).
- **NRW (existing DE rows)**: `bag_id` encodes a `LoD2_32_<E>_<N>_1_NW.gml` 1km UTM32 tile-grid id
  for every row — the exact tile filename, needing no lookup table, unlike PLATEAU's opaque mesh
  codes. Re-fetched all 41 tiles real.h5's 12,000 DE rows draw from, same re-parse/re-locate
  approach as PLATEAU (NRW's posList is already projected EPSG:25832 metres, used directly, no
  reprojection needed).
- **Matching**: footprint IoU (exact polygon intersection/union via `shapely`) AND `|height_m|`
  difference must both clear a threshold (IoU ≥ 0.3, height ≤ 2.0 m — judgement calls, not
  specified numerically by the ticket; see below for why they didn't end up mattering). A spatial
  bucket pre-filter (15 m grid cells, 3×3 neighbourhood) keeps this to
  O(candidates × nearby-existing) rather than a full cross product.
- **Corroborating diagnostic, not just a threshold pass/fail**: nearest-neighbour centroid distance
  from every candidate to the closest existing building, via a KD-tree — this is what actually
  distinguishes "nothing is even close" from "something narrowly missed the thresholds," which a
  bare match count cannot.
- **A cheap city-level bbox pre-check runs first and is logged, but is informative only** — it does
  not gate or shortcut the full run below it, which always checks every candidate against every
  verified existing row regardless of what the pre-check found.

Run: `env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/dedup_buildingworld_geometric.py`
(408s total: 3 PLATEAU tiles + 41 NRW tiles fetched and re-parsed, all 35,577 Tokyo and all 523,213
Berlin meshes loaded and matched against their full existing population). Full output, including the
empty match lists and every disclosed exclusion: [`execution/artifacts/buildingworld_duplicate_gate.json`](../../../execution/artifacts/buildingworld_duplicate_gate.json).

## Results

**Tokyo vs. PLATEAU: 0 matches, 35,577 candidates × 12,000/12,000 fully-verified existing rows.**
The minimum straight-line distance from any BuildingWorld Tokyo mesh centroid to its nearest
existing PLATEAU building is **6,258 m**; the 50th percentile across all 35,577 candidates is
10,964 m. A bulk id-prefix check (the truncated ~20-character uuid surviving in `bag_id`, against
all 35,577 BuildingWorld Tokyo filenames) independently found **0/12,000** matches too — two
independent signals, geometry and id, agree.

**Berlin vs. NRW: 0 matches, 523,213 candidates × 10,554/12,000 fully-verified existing rows.**
Minimum distance from any Berlin mesh to its nearest verified NRW building: **454,075 m** (454 km)
— the bbox pre-check's ≥434 km already made this the expected outcome, now confirmed by the actual
exhaustive per-building computation rather than inferred from it.

Nothing needs to be excluded from ingestion on duplicate grounds for either pair, at this data
snapshot.

## Two disclosed data-quality gaps (NRW only — both investigated, neither is a bug in this module)

Tokyo's re-fetch was clean: 12,000/12,000 rows relocated, 0 height disagreements. NRW's was not,
and both gaps were run down individually before being accepted rather than assumed benign:

**1,446/12,000 NRW rows (12.1%) could not be freshly re-verified**, split two ways:
- **150 rows (1.25%, spread across 23 of 41 tiles)**: the row's bag_id is not in the current tile
  content at all. Confirmed not a fetch bug for the most extreme case — tile `LoD2_32_320_5580_1_
  NW.gml` came back a valid, well-formed, genuinely near-empty 1,553-byte GML with zero `Building`
  elements, not an error page or truncated response.
- **1,296 rows (10.9% of the 11,850 that *were* found)**: the row's freshly re-fetched height
  disagrees with the value `real.h5` already stored for it, by more than 5 cm. Verified this is not
  a computation bug: for one mismatched building (`row=11787`, stored `8.186 m`, recomputed
  `3.828 m`), re-deriving its height through `building_to_sdf`'s own mesh-construction pipeline —
  not this module's simpler raw-posList extent — on the SAME fresh fetch reproduced the identical
  `3.828 m`. The two independent height computations agree with each other and disagree with the
  stored value together, which only makes sense if the SOURCE building's geometry has genuinely
  changed since `real.h5` was ingested.

Both point the same direction: **NRW's open-data portal has been updated since the original
ingestion**, at a materially higher rate than PLATEAU's. Rows in either category are excluded from
the population BuildingWorld candidates are actually matched against — their current geometry
cannot be trusted to represent the row frozen in the corpus, so matching against it would risk
checking the wrong building regardless of the answer. `unrelocated_bag_ids` and `height_drifted`
(with per-row stored-vs-recomputed values) are recorded in full in the output artifact for anyone
who wants to audit or re-verify these 1,446 rows by another method.

## Key findings

**The ticket's stated risk does not currently materialize, for a data-specific reason worth
recording rather than treating as "risk resolved forever."** BuildingWorld Tokyo's mesh set and the
3 PLATEAU tiles `real.h5` actually ingested cover different, non-adjacent parts of the city (a 6+ km
gap, not a few metres) — despite sharing an id convention family. Sharing an id *format* was never
evidence of sharing specific buildings; this is the direct, geometric answer the ticket asked for
instead of inferring from format alone.

**This is a snapshot result, not a standing guarantee.** It answers "does BuildingWorld's current
raw Tokyo/Berlin export duplicate the CURRENT 12,000 PLATEAU / 12,000 NRW rows" — not "will it always
be safe." Both re-fetches already show real drift since ingestion (NRW materially, PLATEAU not at
all this run); if #174 later selects a different BuildingWorld subset, or more PLATEAU/NRW tiles are
ingested, this gate needs re-running against the actual selections in play then — 7 minutes end to
end, not a burden. `PLATEAU_TILES_IN_CORPUS` is pinned to today's 3 known tiles specifically so
drift there would show as a diff, not be assumed away.

**The threshold choices (IoU ≥ 0.3, height ≤ 2.0 m) did not end up deciding either result.** The
nearest anything came, in either pair, was 6,258 m and 454,075 m respectively — thousands of times
past the 15 m bucket radius those thresholds operate inside. A stricter or looser threshold would
not change this run's zero-match outcome; they matter for whichever future run finds a genuine close
pair.

## Scope and disclosed limitations

- Footprint is a convex hull of projected vertices, not the true (possibly concave) outline —
  cheap and robust for a duplicate GATE (a true duplicate's hulls still overlap strongly), but not
  a footprint-shape measurement; do not reuse this polygon for anything beyond match/no-match.
- The IoU/height thresholds are this ticket's judgement call, not a value the issue specified. They
  are moot for this run's actual result (see above) but will matter the next time this gate finds a
  genuinely close pair — re-examine them then, informed by real near-miss data this run didn't
  produce.
- 1,446/12,000 NRW rows (12.1%) were excluded from the checked population for the disclosed,
  investigated reasons above — the Berlin result is "0 matches against the 10,554 rows verifiable
  this run," not literally "against all 12,000." Re-running this gate periodically, or before final
  #174 ingestion, would shrink this gap using whatever NRW tile content exists at that time.
  `min_relocate_frac` (default 0.98) still fails the run loudly if a future re-fetch's *unrelocated*
  gap grows past a small, disclosed rate — height drift has no equivalent hard floor, since
  excluding a drifted row is already the safe response to it.
- Covers only the two pairs the ticket named (Tokyo↔PLATEAU, Berlin↔NRW). PLATEAU Tokyo tiles beyond
  the 3 already in `real.h5`, and any other BuildingWorld city against any other existing source,
  are out of scope here.

## Verification

```bash
env -u LD_PRELOAD ./sdfusion/bin/python \
  scripts/foundations/test_dedup_buildingworld_geometric.py
```

29 tests: the footprint/centroid geometry against points of known shape, polygon IoU against pairs
of known overlap (identical, disjoint, a hand-computed partial fraction), the match decision (both
footprint and height must agree, neither alone), the spatial bucketing pre-filter (a true duplicate
survives it, an unrelated pair hundreds of metres away is never even compared), the real
Berlin-vs-NRW bbox-separation numbers as a regression pin, the nearest-neighbour-distance
diagnostic, NRW tile-filename extraction, and the two disclosed-drift behaviors (a small
unrelocated gap is tolerated and reported, a larger one fails loudly; a height-drifted row is
excluded from matching rather than blocking the run).
