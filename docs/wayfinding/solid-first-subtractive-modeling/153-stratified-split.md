# #153 — Stratify the train/held-out split by source region and tile

*Effort: solid-first semantic architectural carving. Opened 2026-09-05, built 2026-09-09. Part of
[#1](https://github.com/danvisai/SDFusion/issues/1). Implements the split-policy gap
[#5](5-data-audit.md) found; feeds [#181](https://github.com/danvisai/SDFusion/issues/181), which
runs against this split.*

> `datasets/bag3d_dataset.py`'s existing split is a row-random
> `np.random.default_rng(0).permutation(n_total)` with no `source_id` stratification and no
> geographic holdout. Stratify by `source_id` so train/val/test each get proportional NL/DE/JP
> representation, and hold out by tile/bbox rather than individual building row so spatially
> adjacent buildings never land on both sides of the boundary.

## 🔑 Why the obvious per-region-fraction approach fails, and what replaces it

The natural first implementation — target 2% of each region independently for val, 2% for test —
was built, run against `data/real_massing_v1/real.h5` (n=35,776; NL 11,776 / DE 12,000 / JP 12,000),
and **rejected**: NL's `ingest_3dbag.py` drew from exactly 5 named ~1km² city-centre bboxes
(gemeente codes recovered from `bag_id`'s own `NL.IMBAG.Pand.<gemeentecode>...` format — see
below), so NL has only 5 whole-tile groups, and its smallest single tile alone is **858 buildings —
7.3% of NL, 3.6x the 2% target**. Any whole-tile-blocked split for NL must include at least one
full tile or exclude NL from held-out entirely; there is no smaller unit available (the packed
corpus stores no per-building coordinates, only a self-normalized SDF grid per building, so no
finer spatial key can be recovered without re-ingesting NL with coordinates recorded — a real gap,
named here, not fixed here).

Naively holding each region to its own 2% fraction therefore made NL's forced ~858/~2387 tiles
dominate val/test (NL ended up **50–75%** of both), which is exactly — inverted — the region-
imbalance bug [#5](5-data-audit.md)/#127 already found in the *old* split and patched around in
`eval_massing_arms.py`'s `pick_ids`. Shipping that would defeat the point of this ticket.

**The fix:** compute each region's own naive achievable count first (its closest whole-tile subset
to the 2% target), then take the **largest such count across regions as a shared absolute target**,
and have every *other* region match that same absolute count as closely as achievable — an exact
0/1 subset-sum search over its own tiles (`_closest_subset` in `stratified_split.py`), not a
greedy approximation. NL's forced minimum (858 for val, 2387 for test) turned out to be the binding
floor for both; DE (41 tiles) and JP (9 tiles) both have enough small tiles to match it almost
exactly.

**⚠️ One subtlety the first version of this got wrong and the tests now pin down:** "closest to a
small target" including the *empty* subset is numerically closer to a small target than any single
oversized tile — the exact-subset-sum search initially returned **zero** NL tiles for both val and
test, because 0 is closer to 235.5 (2% of 11,776) than 858 is. `_closest_subset` now requires
`min_items>=1` whenever the region has tiles, forcing at least one — the whole reason
stratification exists is to *guarantee* every region is represented, not to minimize a distance
metric that's happy to represent it with nothing.

## Tile-key derivation (no re-ingestion required)

| source | key | recovered from | verified |
|---|---|---|---|
| NL (3D BAG) | gemeente code | `bag_id = "NL.IMBAG.Pand.<4-digit gemeente code><...>"` — official Dutch cadastral id; digits after `Pand.` are the CBS municipality code | exactly 5 distinct codes across all 11,776 NL rows, matching `ingest_3dbag.py`'s 5 named bboxes exactly (counts sum to 11,776) |
| DE (NRW CityGML) | source `.gml` tile filename | `ingest_citygml_lod2.py` already writes `bag_id = f"{gml_key}#{gid}"[:64]`; the tile name is the prefix before `#`, truncation-safe since it comes first | 41 distinct tiles, 1–2457 buildings each |
| JP (PLATEAU CityGML) | `<mesh zip>:<gml filename>` pair | same `bag_id` construction, `gml_key = f"{tz}:{n}"` | 9 distinct tiles, 89–2840 buildings each |

## Achieved split

Pooled: **train 26,064 (72.9%) / val 2,592 (7.2%) / test 7,120 (19.9%)** of 35,776. Per region:

| region | n | tiles | val | val % of region | test | test % of region | train |
|---|---:|---:|---:|---:|---:|---:|---:|
| NL | 11,776 | 5 | 858 | 7.29% | 2,387 | 20.27% | 8,531 |
| DE | 12,000 | 41 | 858 | 7.15% | 2,387 | 19.89% | 8,755 |
| JP | 12,000 | 9 | 876 | 7.30% | 2,346 | 19.55% | 8,778 |

Region shares *within* val (33.1% / 33.1% / 33.8%) and *within* test (33.5% / 33.5% / 33.0%) both
land within 1 point of the corpus's own 32.9/33.5/33.5 split — proportional representation
(acceptance criterion 1) is met to the precision whole-tile blocking allows, not merely
approximately. The cost, fully disclosed rather than hidden: **held-out is ~27% of the corpus, not
~4%**, forced by NL's 5-tile ceiling — there is no whole-tile combination that hits 2%/2% and
proportional representation simultaneously, given NL's actual ingestion granularity.

No two buildings from the same tile appear on opposite sides of the train/held-out boundary —
enforced structurally (whole tiles are the unit of assignment, never split) and checked
programmatically (`assert_no_tile_crosses_boundary`, run over the real corpus in
`test_stratified_split.py`, zero violations).

## Acceptance criteria

- [x] Train/val/test each contain roughly proportional NL/DE/JP representation — see table above;
      `RealCorpusRegressionTest.test_achieved_proportions_match_the_recorded_baseline` pins the
      exact achieved counts as a regression.
- [x] No two buildings from the same tile/bbox appear on both sides of the train/held-out boundary
      — `assert_no_tile_crosses_boundary`, tested against both a synthetic NL/DE/JP-shaped fixture
      and the real corpus.
- [x] Stated explicitly: the existing 714/411 held-out artifact
      (`massing_arms_eval_ship714.json`) is **not** regenerated against this split — see below.
- [x] Noted explicitly: `pick_ids`'s round-robin is **not** redundant — see below.

## 🔑 Decision: this is a new, additional split. The existing frozen 714/411 population is untouched.

`Bag3dDataset`'s existing row-random split and `vecset_ceiling_probe.py`'s `test_indices()` (the
mechanism behind the pinned 714/411 held-out population every closed massing ticket on this map
scores against — #10, #92's four arms, #126, #127, #129 through #140 and onward) are **not
modified by this ticket**. Three reasons, not one:

1. **Blast radius.** Swapping the held-out population would silently invalidate every number
   already recorded against the old 714/411 set — dozens of closed tickets' evidentiary trail —
   for a ticket scoped in its own issue as "a data-generation change, independent of any model."
2. **#162 already claims this territory.** [#162](https://github.com/danvisai/SDFusion/issues/162)
   (open, unimplemented) exists specifically to pin `test_indices(35776)`'s row identity ahead of
   BuildingWorld ingestion — the project's own stated intent is to *protect* that population, not
   replace it, at least until BuildingWorld lands.
3. **#181 already names this split as prospective, not retroactive.** Its own text: runs "against
   the split #153 produces" — a fresh scorecard round, not a swap-in for existing numbers.

`scripts/foundations/stratified_split.py` is therefore standalone: it reads `real.h5` directly and
returns a fresh split, callable by whichever future harness (#181's, most immediately) wants it. It
does not touch, import from, or get imported by `datasets/bag3d_dataset.py`.

## `pick_ids`'s round-robin remains necessary, not redundant

`eval_massing_arms.py`'s `pick_ids` interleaves the *old* split's held-out rows by region so a
small `--n` sample stays region-balanced despite that split's row-random imbalance. Since the old
split is untouched (above), `pick_ids` still operates against exactly the population it was written
for and is still doing real work — removing it would reintroduce the 100%-one-region bug it exists
to prevent. A harness built against *this* ticket's new split doesn't need an equivalent trick,
because region balance is already engineered into val/test membership directly rather than patched
at sampling time — worth building that way when #181 is picked up, but that harness doesn't exist
yet and isn't this ticket's job to write.

## What this doesn't decide

- **Finer NL spatial stratification** would need per-building coordinates, which the packed
  corpus doesn't carry today (each building is stored as a self-normalized SDF grid). Re-ingesting
  NL with coordinates recorded is real, named future work — not started here, matching this map's
  standing rule against guessing at unscoped fixes ([#5](5-data-audit.md), [#9](9-multi-footprint-coordination.md)'s
  own restraint).
- **Whether ~27% held-out is an acceptable training-data cost** for whatever model #181 or a future
  ticket trains against this split. This ticket reports the achieved trade-off; deciding whether
  it's worth paying is that future ticket's call, made with this number in hand rather than
  discovered by surprise.
- **BuildingWorld's own split** (#153's sibling problem, one level up in the epic — see
  [#177](https://github.com/danvisai/SDFusion/issues/177), "Build a roof-family-stratified,
  spatially-blocked held-out split for new BuildingWorld rows") is separate work; this ticket's
  whole-group blocking idea is reusable, but this implementation's shared ABSOLUTE targets balance
  nearly equal-sized NL/DE/JP regions, not arbitrary imbalanced city populations. #177 must choose
  its own balance policy and valid spatial keys; do not reuse the algorithm unchanged by assumption.

## Evaluation boundary (reconciled 2026-09-09)

The split prevents source-group overlap within its own labels. It cannot retroactively remove a
checkpoint's exposure to these rows. Metadata inspection found 6,837/7,120 new test rows in the
old Bag3d training split and 6,937/7,120 in the height-map eligible non-held pool before validation
selection. These are pool-overlap counts, not per-checkpoint training manifests. #181 must disclose
exposure and exclude new test rows from retrieval; its no-new-training rule does not permit claiming
unseen-data generalization from an old-trained checkpoint. The `--out` file is a summary, not a
per-row manifest; record corpus identity, parameters and membership when consuming `make_split`.

## Artifacts

- `scripts/foundations/stratified_split.py` — the split (`tile_key`, `make_split`,
  `assert_no_tile_crosses_boundary`), runnable standalone against `real.h5`.
- `scripts/foundations/test_stratified_split.py` — structural tests against a synthetic
  NL/DE/JP-shaped fixture (fast, data-free) plus one regression test pinning the achieved
  proportions against the real corpus (skipped, not failed, if `real.h5` is absent).
