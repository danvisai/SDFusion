# #177 — Roof-family-stratified, spatially-blocked held-out split for new BuildingWorld rows

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `ready-for-agent` label. Blocked by
[#176](https://github.com/danvisai/SDFusion/issues/176) (program pseudo-labels — the roof-family
axis), [#165](https://github.com/danvisai/SDFusion/issues/165) (CRS/units policy),
[#169](https://github.com/danvisai/SDFusion/issues/169) (small-city pooling policy), and
[#161](https://github.com/danvisai/SDFusion/issues/161) (the ledger this ticket writes into).
Blocks [#178](https://github.com/danvisai/SDFusion/issues/178).*

> Extend — never mutate — the existing frozen held-out mechanism by carving a fresh held-out slice
> from just the newly-ingested BuildingWorld rows, stratified by roof family (#176's pseudo-labels)
> and spatially blocked within each city's own corrected coordinate system, so that every existing
> NL/DE/JP row's `held_out` flag comes out bit-identical and every new row gets an explicit
> assignment.

## What this ticket builds

Two scripts and one ledger append:

1. **`scripts/foundations/buildingworld_spatial_keys.py`** — a per-row `(x_m, y_m)` footprint
   centroid for every BuildingWorld row, in that row's own city's corrected metres. Neither
   `real.h5` (#174) nor `surfaces_buildingworld.h5` (#175) stores this: `real.h5`'s `sdf`/
   `footprint` are `building_to_sdf`'s self-normalized per-building grid (#153 already named this
   exact gap for NL — "the packed corpus stores no per-building coordinates"), and #175's `verts`
   are `to_frame_n`'s per-building-centered Frame-N, translation-invariant by construction. The
   position has to be recovered from the raw mesh, once, before either normalization throws it
   away. Rather than re-solve "which zip member is row R's mesh" — #175's own already-solved,
   already-verified problem (Perth's `bag_id`-truncation collisions) — this script imports
   `_wanted_ids` / `_group_members_by_bag_id` / `_load_corrected_mesh` / `_resolve_collision`
   directly from `ingest_surfaces_buildingworld.py` and adds only what it needs on top: the
   corrected mesh's `bounds` mean, taken before `to_frame_n` would discard it. Output:
   `data/real_massing_v1/buildingworld_centroids.h5` (`row`, `x_m`, `y_m`, `ok`, `reason`),
   resumable.
2. **`scripts/foundations/buildingworld_stratified_split.py`** — grids each city's (or #169's
   pooled bucket's) rows into fixed-size metre cells, selects whole cells (never split) via a
   roof-family-aware greedy search toward a 2% target, and appends the result to
   `corpus_ledger.h5` (#161) via `corpus_ledger.append_ledger`, which refuses to overwrite an
   existing row.
3. A refactor of `ingest_buildingworld.apply_geometric_correction`, splitting out a new pure
   `correct_vertices(city, member_name, vertices)` — the array-only half of #165's CRS policy —
   so `buildingworld_spatial_keys.py` can reuse the exact same correction table (via
   `_load_corrected_mesh`) rather than risk a second, silently-drifting copy of it. Verified
   byte-identical to the old mesh-based path for every city/subfolder case
   (`TestCorrectVertices` in `test_ingest_buildingworld.py`); all 38 pre-existing tests in that
   file still pass unchanged.

## Decisions this ticket had to make that its own issue text didn't fully specify

### Why a metre grid, not #153's own whole-tile key

#153's own "what this doesn't decide" section is explicit: "#177 must choose its own balance
policy and valid spatial keys; do not reuse the algorithm unchanged by assumption." NL/DE/JP had a
natural tiling already baked into `bag_id` (gemeente code, CityGML source filename). BuildingWorld
has none: member-name conventions are inconsistent and not reliably spatial across cities —
sequential integers for Calgary, UUIDs for Tokyo, suburb-name subfolders only for some (Yarra,
parts of Greater Geelong). A fixed-size grid over each group's own corrected `(x, y)` is used
instead. `GRID_CELL_M = 150` (a default, not pinned as final — comfortably larger than a terrace/
row-house frontage, small enough that a dense city like Berlin still has tens of thousands of
occupied cells to select from) is the atomic, never-split held-out unit, playing the same
structural role #153's whole tile did.

### The two-axis selection is a disclosed heuristic, not an exact optimum — and went through a real revision

#153's own `_closest_subset` is an exact 0/1 subset-sum search — but over one axis (total count).
Adding roof-family stratification makes this a joint two-axis problem; an exact joint optimum is
materially harder, and the issue's own text downgrades roof-family-bucket trust anyway ("this is
not a fully independent signal... treat 'gable' bucket membership as 'gables this fitter can
already find', not ground truth"). The **first** version of `plan_group_split` tried "smallest
tile first" (mirroring #153's own `_region_tiles` ordering) combined with sequential
largest-target-family-first processing. A `/code-review` pass (see below) traced this against the
first production run's own numbers and found it wrong in a way that mattered: over this ticket's
fine-grained metre grid (unlike #153's already-coarse, natural tiles), "smallest first" means
"sparsest, most isolated buildings first" — cities like San Francisco and Yarra ended up with a
held-out slice at *exactly one building per tile, every time*, and several cities' family
fractions swung by double digits between the city and its held-out slice (e.g. Edmonton's `flat`
share falling from 20% to 10%). Both defects are structural, not tuning: a scattered sample of
isolated buildings cannot catch the dense-terrace-row leakage risk spatial blocking exists to
prevent, and sequential family processing lets one large-target family consume the whole budget
before a small one gets a turn.

The tile-order fix (**seeded random, never sorted by size**) landed in the first pass and stuck: a
dense cluster now has the same chance of being drawn — and, once drawn, is held out as one
untouched block — as a sparse one.

The family-claiming rule went through a SECOND revision, caught by a follow-up review of the first
fix rather than the original spec/standards pass. The first fix made the four families claim tiles
*round-robin* (one per active family per round), which did stop a large family from crowding a
small one out entirely — but a follow-up review traced its own production numbers and found the
mirror-image problem: giving every ACTIVE family one turn per round, regardless of how much it
actually needs, makes a small-target family "spend" the shared row budget at the same rate as a
large one, so the large family runs out of room before its own target is reached. Measured directly:
Calgary's `gable` share (68% of the city) fell to 58% of the round-robin held-out slice — a
double-digit swing in the OPPOSITE direction from the original bug, and comparable in size. The
**final** rule claims, at every step, from whichever family currently has the largest UNMET NEED
(`target − achieved so far`) among families with an eligible tile left — giving each family roughly
`target / mean_tile_size` turns, proportional to what it actually needs, rather than one turn per
round regardless of size. A family drops out the moment its own target is met or its own eligible
tiles run out (the same graceful degradation as both earlier versions). Every tile's PLURALITY
family (ties broken by `FAMILIES`'s own declaration order — `flat` wins a tie) decides which
family's claim can take it. A fill phase tops up to the overall 2% target from whatever tiles
remain once every family is satisfied or exhausted. At least one tile is always selected for a
non-empty group, mirroring #153's own `_closest_subset(min_items=1)` reasoning.

`plan_group_split`'s own report carries `mean_tile_size_all`/`mean_tile_size_held_out` so the
tile-order property stays checkable going forward, not just asserted. Three regression tests pin
all three fixes: `test_selection_is_not_biased_toward_the_sparsest_tiles` (tile order),
`test_a_small_target_family_is_not_crowded_out_by_a_large_one` (a minority family gets
representation), `test_a_dominant_family_is_not_squeezed_by_equal_turn_taking` (a majority family's
share stays close to its own population share, not squeezed toward an even split).

`tile_key` also gained a `city` argument (`f"{city}:{ix}:{iy}"`), not just `f"{ix}:{iy}"` — the
first version keyed a tile by grid cell alone, so #169's pooled bucket (four different cities in
four different corrected coordinate systems) could silently merge two physically unrelated
locations that happened to land on the same `(ix, iy)` pair onto one "tile."

### Pooling (#169), applied verbatim

Adelaide, Greater Geelong, Perth, and Philadelphia — the four cities #169 named — are merged into
one pooled group (`pooled_small_cities`) before gridding/selection; every other ingested city gets
its own independent slice. This is a literal, scoped application of #169's decision, not a general
small-city rule: Cambridge (182 kept rows, smaller than three of the four pooled cities) is **not**
pooled, because #169's own text scopes the policy to exactly those four named cities and explicitly
declines to codify a general threshold. Cambridge gets its own 182-row group and a correspondingly
small (5-row) held-out slice — a disclosed, known consequence of following #169 literally rather
than silently expanding its scope.

Within the pooled bucket, tile selection is size/family-driven across all four cities' tiles
together, with no per-city floor — #169 asks for one combined score plus diagnostic per-city
sub-scores, not proportional representation of each pooled city. Every pooled city is represented
in the final production run's held-out slice (see the per-city table below) — an earlier pass of
this algorithm (before the second code-review round's deficit-proportional fix) left Greater
Geelong at 0 rows entirely, which turned out to be an artifact of that pass's own bias rather than
a genuine population-size limit; see "Family claiming (fix 3)" below for the direct before/after
comparison. #169 itself calls every individual pooled-city number "diagnostic-only /
too-small-to-trust-individually" regardless — no per-city floor is guaranteed or promised by this
ticket, so a future run landing a pooled city at 0% would still be a valid, if uninformative, thing
to report, not necessarily a bug — but it is worth double-checking against the mean-tile-size and
family-drift diagnostics above before assuming it's just bad luck.

### `region` for the ledger: the same `-1` sentinel `real.h5` already uses

`corpus_ledger.h5`'s `region` column has no BuildingWorld-specific value to draw on: #171 (region-
conditioning granularity) has not landed, and #174 already established `-1` ("BW", registered in
`stratified_split.SOURCE_NAMES`) as the disclosed "not yet assigned" sentinel for exactly this
situation. Reusing it here — rather than inventing a second, ledger-local placeholder — means #171
has exactly one place to update when it lands, not two.

### Every row requires BOTH an `ok=1` centroid and an `ok=1` program label, checked before anything is written

`build_assignment` raises (naming the first few affected rows) if any BuildingWorld row lacks
either input, rather than defaulting it to `held_out=0` or silently dropping it — the issue's own
explicit requirement. At production scale this never fired: centroid extraction and #176's program
labeling both independently reached 100% coverage (see below).

### Disclosed pre-existing gap this ticket inherits, not introduces: 153 historical rows have no ledger entry at all

`corpus_ledger.h5`'s pre-#177 prefix has 35,623 rows, not the frozen split's full 35,776 — 153
rows that `#161`'s own extraction (`extract_from_vecset_latents`) never carried forward, because
they failed `vecset_latents.h5` encoding entirely (a pre-existing, already-documented fact:
`five_arm_scorecard.py`'s own docstring already names "153 of real.h5's 35,776 rows fail
vecset_latents.h5 encoding"). Those 153 rows simply have no ledger `held_out` value, before or
after this ticket — inherited behavior, not something #177 changes or is responsible for fixing,
but worth stating plainly here rather than leaving a reader to notice the row-count gap and wonder
whether this ticket caused it.

### A pre-write backup protects the ledger append, not just a post-hoc detection

`corpus_ledger.write_ledger` already replaces `ledger_path` atomically (a temp file + `Path.replace`,
so a crash mid-write cannot corrupt it) — but a *logic* bug in `append_ledger`'s own concatenation
could still write successfully, just with wrong data, and the first version of this ticket's
post-append prefix check could only detect that after the wholesale rewrite had already landed,
leaving a caller with a corrupted `corpus_ledger.h5` and an error message telling them to go find a
backup themselves. `run()` now copies `ledger_path` to a sibling `.177-backup` file before calling
`append_ledger`, and restores from it automatically (raising a `RuntimeError` that says so
explicitly) if the post-append prefix check ever fires; the backup is deleted on success.

## Code review, before finalizing (this session) — two rounds

**Round 1** (`/code-review`, Standards + Spec sub-agents, this project's own rule) against the
first completed production run found one real Spec-level algorithmic defect and one real
Standards-level correctness bug:

- **The selection algorithm's "smallest tile first" heuristic degenerated to a scattered,
  isolated-building sample** — see "The two-axis selection..." above for the full account. First
  fix: seeded-random tile order plus round-robin family claiming.
- **`tile_key` wasn't city-namespaced**, so the pooled bucket could silently merge two different
  cities' unrelated grid cells onto one key — fixed by keying on `f"{city}:{ix}:{iy}"`.
- **A resumed run of `buildingworld_spatial_keys.py` could corrupt an already-committed collision
  resolution**: the first version passed only the still-pending rows of a `bag_id` truncation-
  collision group into `_resolve_collision`, leaving an already-claimed candidate mesh free to be
  re-matched to a different row on resume. Fixed by mirroring `ingest_surfaces_buildingworld.py`'s
  own resume shape exactly — skip a whole `bag_id` group only once every one of its rows is already
  committed, otherwise always recompute the full group fresh and simply skip re-writing whichever
  rows are already done. Did not affect this session's own production data (that run completed in
  one pass, never resumed), but is a real latent bug for any future re-run; a new regression test
  (`test_resuming_a_partially_committed_collision_group_does_not_corrupt_the_committed_row`) pins
  the fixed behavior directly.
- Smaller fixes: `load_inputs` now respects `program_labels_buildingworld.h5`'s own
  `committed_rows` boundary (mirroring what it already did for centroids) rather than trusting an
  interrupted writer's uncommitted tail; `buildingworld_centroids.h5` gained a `reason` column
  (parity with #176's `fail_stage`) recording why a centroid extraction failed, for the rare
  `ok=0` row; `city_for_row` raises a `#177:`-prefixed error instead of a bare `KeyError`;
  `BUILDINGWORLD_SOURCE_ID` is now imported from `recover_program_labels_buildingworld.py` rather
  than re-declared; `run()` (`buildingworld_spatial_keys.py`) no longer holds `real.h5` open across
  the fork-pool creation, matching #176's own established shape.
- `stratified_split.tile_key` additionally gained a hard, explicit rejection of
  `source_id == BUILDINGWORLD_SOURCE_ID` (see the incidental-fix section below) — the Spec
  reviewer's own point that the frozen-prefix restriction fix should not rely solely on every
  future caller remembering it.

**Round 2** (a targeted follow-up verification, also on Opus): re-checked all eight Round-1 fixes
directly against the current code (not just the doc's own account of them) and confirmed each one
landed correctly, then independently re-examined the rewritten selection algorithm with fresh eyes.
It found the round-robin family fix above was itself biased — see "The two-axis selection..." above
for the mirror-image squeeze-the-dominant-family problem it measured directly in the round-robin
run's own numbers — plus three smaller issues, all fixed: the tie-break docstring claimed
"alphabetical" when the code actually broke ties by `FAMILIES`'s declaration order (the docstring
now says so accurately, rather than the code being changed to match an arbitrary claim); every
group was seeded with the exact same `seed` value, so two groups that happened to share a tile
count would draw an identical random permutation (fixed: each group now derives its own seed via
`hashlib.sha256(f"{seed}:{group_name}")`, deterministic and reproducible, not Python's own
randomized string `hash()`); a family whose own target rounds to 0 at small `target_total` gets no
dedicated claims (left as-is — a small-N rounding artifact, not a tile-availability problem, and
the same limitation #153 already accepts for NL's coarse tiles). Centroid extraction and the split
build were both re-run again after the deficit-proportional fix; the numbers in this doc are from
that final run.

## What was actually run this session

- Full unit-test suites, post-fix: `test_ingest_buildingworld.py` (40 tests, including the new
  `TestCorrectVertices`), `test_buildingworld_spatial_keys.py` (13 tests, including the new
  resume-collision regression), `test_buildingworld_stratified_split.py` (23 tests, including the
  three new selection-bias/family-fairness regressions), `test_stratified_split.py` (13 tests,
  including the new BuildingWorld-rejection test) — all passing, synthetic/fast, no dependence on
  the full production corpus beyond the small frozen-prefix metadata slice #162's guard needs. The
  full `scripts/foundations` discovery suite (923 tests) passes with no failures.
- A real smoke run of centroid extraction against production data (Cambridge, 182 rows; a 3,000-row
  slice of Mississauga to measure throughput) before committing to the full run, both before and
  after the resume/schema fix.
- **Production centroid extraction, run twice**: the first pass (before code review) — all 18
  ingestable cities, `--workers 28`, **1,526,778 rows, 0 failures, 195 s**; re-run from scratch
  after the `reason`-column schema addition and the resume-safety fix (`--no_resume`, to get a
  clean file matching the current schema rather than a resumed one) — **identical per-city counts,
  0 failures, 204 s**. (The Round-2 review's fixes were all inside `buildingworld_stratified_split.py`
  only, so centroid extraction did not need a third run.) Both runs an order of magnitude faster
  than #174's own full ingest (which also classified defects and voxelized every mesh), because
  this pass only loads and corrects each mesh, skipping `building_to_sdf` in the common case (only
  the rare `bag_id`-truncation collision path re-voxelizes, exactly as #175 already does). Perth's
  known collisions (#175's own finding) resolved correctly at full scale both times.
- **Production split build, run three times**: pass 1 (original "smallest first + sequential
  family" algorithm) and pass 2 ("random order + round-robin family") were each inspected, found
  flawed, and discarded; `corpus_ledger.h5` was restored to its pre-#177 35,623-row state from a
  backup taken before pass 1 each time a re-run was needed. Pass 3 (random order + deficit-
  proportional family claiming) is what actually landed — `--dry_run` first (inspected against
  pass 2's own numbers to confirm the squeeze was gone), then for real.
- **Verification, independent of `run()`'s own internal check, after the final pass**: read
  `corpus_ledger.h5` before and after via a separate script, confirmed all four columns
  (`row`/`region`/`held_out`/`height_m`) byte-identical across the pre-existing 35,623-row prefix,
  that #162's `assert_frozen_corpus` (called at `run()`'s start, against `real.h5` itself) passed
  before anything was written, and that no `.177-backup` sibling file was left behind (the
  success-path cleanup ran).

## The production run (2026-09-15, final: random tile order + deficit-proportional family claiming)

`corpus_ledger.h5` grew from 35,623 to **1,562,401 rows** (1,526,778 new BuildingWorld rows
appended, region `-1`). **30,779 held out (2.02% of 1,526,778 new rows)**.

| Group | n | tiles | held out | held-out % | mean tile size (all → held-out) | family (all → held-out) |
|---|---:|---:|---:|---:|---|---|
| Berlin | 456,569 | 26,330 | 9,141 | 2.00% | 17.3→16.6 | flat 22%→22%, gable 40%→40%, hip 2%→2%, complex 35%→35% |
| Boston | 114,494 | 5,093 | 2,294 | 2.00% | 22.5→25.8 | flat 93%→94%, gable 3%→3%, hip 1%→0%, complex 3%→3% |
| Calgary | 165,688 | 14,232 | 3,320 | 2.00% | 11.6→13.1 | flat 5%→5%, gable 68%→69%, hip 6%→6%, complex 21%→21% |
| Cambridge | 182 | 29 | 5 | 2.75% | 6.3→5.0 | flat 39%→80%, gable 30%→20%, hip 4%→0%, complex 27%→0% |
| Cape Town | 243,932 | 13,312 | 4,907 | 2.01% | 18.3→20.5 | flat 20%→20%, gable 39%→39%, hip 3%→3%, complex 38%→38% |
| Edmonton | 281,211 | 17,166 | 5,669 | 2.02% | 16.4→17.5 | flat 20%→20%, gable 38%→38%, hip 0.3%→0.3%, complex 42%→41% |
| Melbourne | 7,742 | 1,136 | 156 | 2.01% | 6.8→12.0 | flat 65%→69%, gable 2%→1%, hip 0.1%→0%, complex 32%→29% |
| Mississauga | 139,063 | 8,406 | 2,783 | 2.00% | 16.5→18.3 | flat 2%→2%, gable 34%→34%, hip 25%→25%, complex 38%→38% |
| Montreal | 54,058 | 2,382 | 1,111 | 2.06% | 22.7→24.2 | flat 25%→25%, gable 10%→10%, hip 2%→3%, complex 63%→63% |
| New York | 20,650 | 2,105 | 413 | 2.00% | 9.8→8.3 | flat 99%→98%, gable 0.3%→0.2%, hip 0.1%→0%, complex 1%→2% |
| San Francisco | 4,248 | 1,745 | 86 | 2.02% | 2.4→1.8 | flat 61%→60%, gable 20%→20%, hip 3%→3%, complex 16%→16% |
| Tokyo | 33,315 | 1,495 | 768 | 2.31% | 22.3→22.6 | flat 22%→23%, gable 17%→18%, hip 2%→2%, complex 59%→56% |
| Wellington | 530 | 218 | 11 | 2.08% | 2.4→2.8 | flat 54%→55%, gable 28%→27%, hip 5%→9%, complex 13%→9% |
| Yarra | 2,694 | 601 | 55 | 2.04% | 4.5→4.6 | flat 52%→49%, gable 25%→25%, hip 3%→4%, complex 20%→22% |
| **pooled_small_cities** | 2,402 | 492 | 60 | 2.50% | 4.9→7.5 | flat 16%→12%, gable 29%→30%, hip 4%→2%, complex 51%→57% |

Pooled bucket, per-city diagnostic (#169's own required breakdown — too small to trust
individually):

| City | n | held out | held-out % |
|---|---:|---:|---:|
| Adelaide | 1,104 | 28 | 2.54% |
| Greater Geelong | 604 | 19 | 3.15% |
| Perth | 418 | 7 | 1.67% |
| Philadelphia | 276 | 6 | 2.17% |

Every standalone group lands close to the 2% target; the smallest (Cambridge, Wellington) overshoot
more, an expected, disclosed consequence of whole-tile blocking at low tile counts (identical in
kind to #153's own NL overshoot, just far smaller in absolute size here — Cambridge's 29 tiles for
a target of ~5 rows leaves little room to land closer, and its own tiny held-out family split is
correspondingly noisy). Two things are direct evidence both fixes from the two code-review rounds
hold at production scale, not just in synthetic tests:

- **Tile order (fix 1)**: the mean-tile-size column tracks the population mean in every city (e.g.
  Cape Town 18.3→20.5, Mississauga 16.5→18.3, Edmonton 16.4→17.5) rather than collapsing toward 1,
  which is what the first ("smallest first") pass produced uniformly.
- **Family claiming (fix 3, the deficit-proportional rule)**: achieved per-family fractions now sit
  within a point or two of each city's own population share almost everywhere — contrast with the
  round-robin pass's Calgary `gable` 68%→58% and Yarra `flat` 52%→33%, both now landing at 68%→69%
  and 52%→49% respectively. The one clear standing exception is Greater Geelong's 0%→0% under
  round-robin, now 19 rows (3.15%) represented under the final algorithm — direct evidence the
  earlier pooled-bucket "0% is fine, it's diagnostic-only" framing was papering over an algorithm
  bug, not a genuine population-size limit.

## Incidental fix: #153's split machinery broke once `real.h5` grew (found running the full suite)

Running the full `scripts/foundations` test suite before finishing this ticket surfaced two
pre-existing failures, unrelated to this session's own changes:
`test_stratified_split.RealCorpusRegressionTest.test_achieved_proportions_match_the_recorded_baseline`
and `test_five_arm_scorecard.TestMaterializeSplit.test_runs_end_to_end_and_returns_disjoint_val_test`,
both asserting a split's total row count equals 35,776. Root cause: `stratified_split.main()`,
`five_arm_scorecard.materialize_split`, and `void_semantic_sample.materialize_test_ids` all read
`source_id`/`bag_id` off the *entire* `real.h5` with no row-count restriction — harmless while
`real.h5` had exactly 35,776 rows, but #174 (an earlier ticket on this same map) grew it to
1,562,554. Worse than the test failure itself: `tile_key` treats any `source_id != 0` row's
`bag_id` prefix before its first `#` as the tile key, and BuildingWorld's `bag_id` is
`'<City>#<member>'` — so every one of one city's ~450k rows would silently collapse onto a single
giant "tile," which `make_split`'s shared-target subset-sum search would then have to place
entirely on one side of the boundary. Fixed by restricting all three call sites to
`real.h5[:FROZEN_SPLIT_N_TOTAL]` (matching #162's own established pattern elsewhere — `Bag3dDataset`,
`vecset_ceiling_probe.test_indices()` — of naming the frozen size explicitly rather than deriving it
from whatever the corpus has grown to). Both tests pass again with no change to their own assertions
beyond the same restriction; the achieved baseline numbers are unchanged, since the frozen prefix's
content never moved.

The Round-1 code review's Spec reviewer made a further point: docstring-only disclosure of "callers
must restrict to the frozen prefix" leaves the next caller free to reintroduce the exact same bug.
`stratified_split.tile_key` now hard-rejects `source_id == BUILDINGWORLD_SOURCE_ID` (`-1`) with a
`#153/#177:`-prefixed `ValueError` naming the fix, so a caller who forgets the restriction fails
loudly at the first BuildingWorld row rather than silently producing a corrupted split -- a
backstop, not the primary mechanism (`test_buildingworld_source_id_is_rejected_not_collapsed_into_
one_giant_tile` in `test_stratified_split.py` pins it).

## What this ticket does not do

- **The fitter-independent spot audit** the issue text raises ("consider a small fitter-independent
  spot audit... rather than trusting the bucket blindly", naming `scripts/foundations/
  human_eval_rubric.py`) is advisory language, not a stated acceptance criterion, and is not run in
  this session — disclosed here as deferred follow-up rather than silently skipped. A natural next
  step: sample a handful of `gable`-bucket held-out rows per city and run them through the existing
  rubric before trusting the family axis for anything higher-stakes than "don't skew flat."
- **A per-city floor within the pooled bucket** (guaranteeing every one of the four pooled cities a
  nonzero held-out slice) is not implemented — #169's own text calls each pooled city's number
  diagnostic-only, and the issue does not ask for intra-pool proportionality.
- **Re-deriving `GRID_CELL_M`/`SPLIT_FRAC` per city** from local building density is not attempted;
  both are fixed, disclosed defaults (150 m, 2%) matching this map's existing 2% convention
  (#153/#169), not per-city-tuned.

## Status: implemented, unit-tested, and run to completion in production (2026-09-15)

`data/real_massing_v1/buildingworld_centroids.h5` (1,526,778 rows, 100% coverage) and
`data/real_massing_v1/corpus_ledger.h5` (now 1,562,401 rows, the historical 35,623-row prefix
verified byte-identical) both exist. `execution/artifacts/buildingworld_stratified_split_177.json`
holds the full per-group report. #178 is unblocked.
