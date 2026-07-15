# Generate the Full-Data Decomposition Arm

Type: task
Status: done (2026-07-13, extended 2026-07-14)
Blocked by: 03, 08

## Question

Generate held-out outputs using full-data Stage 3a massing plus the train_100-only retrieval library,
with deterministic seeds and no test provenance. Verify massing/detail assembly, output resolution,
failure accounting, and exact compatibility with the neutral evaluation harness.

## Comments

## Answer

**Resumed 2026-07-13** after the 2026-07-12 pivot paused it — the project owner was explicitly
asked (post ticket-10) whether to resume C2, pursue Layer 1 massing/shape quality instead, or
pick up the trivial deferred "capture residual transform evidence" ticket, and chose to resume
C2. Research first: despite tickets.md's own "fully scoped and code-verified" status note, the
actual chain (Stage3a SDEdit massing → `propose_detail_ops` → `element_fit.retrieve` against the
leakage-safe library → `EditableBuilding` CSG compose) had never been assembled anywhere in the
codebase — each piece worked in isolation, tested separately, but never chained. This was
genuinely new orchestration code, not a rerun of existing code on a new population.

**Two scope decisions made explicitly by the project owner before implementation:**

1. **Retrieval scope: `tower`/`dome`/`chimney` only.** The leakage-safe library
   (`element_library_train100_v1`) has dead or borderline pools for `balcony` (5), `balcony_upper`
   (0), `stairs` (0), `column` (8 — right at `MIN_POOL=8`) — ticket 08 had already flagged this
   gap. Only tower (122), dome (104), chimney (48) clear the floor comfortably. Those four
   excluded types stay procedural (same as `window`/`door` always are) — matching CONTEXT.md's
   own definition of "Composition" as retrieval + learned placement + procedural instantiation,
   not retrieval alone, so excluding a type from retrieval doesn't remove it from the output.
2. **Population: the full 277-id `test.json`, leakage tiers disclosed per building, not
   restricted to the 27-building Stage3a-clean subset.** The massing step's pretrained Stage3a
   prior has known leakage on 224/277 of these ids (ticket 09's own finding) — the monolith
   (trained fresh, `train_100`-only) has none. Every row records its tier
   (clean/val_leak/train_leak) via `transform_vs_noise.classify_leakage`, disclosed rather than
   hidden, so ticket 13 can decide how to treat the asymmetry rather than have it silently baked
   into a pooled number.

**Built:** `scripts/foundations/generate_decomposition_arm.py` (TDD, 11 contract tests for the
pure seams — `pools_for_type`, `op_half_extent`, `y_extent_from_occupancy`, `retrieval_params`;
the massing/detail/compose pipeline itself is GPU/checkpoint-dependent and verified by the real
run below) + `scripts/foundations/verify_decomposition_harness_compat.py` (the ticket's own
"exact compatibility with the neutral evaluation harness" bullet).

**Pipeline per building:** (1) Stage3a SDEdit massing from a footprint-extrude blockout,
strength=0.5 — byte-identical to ticket 09's own C1 generation contract, reused via
`transform_vs_noise.build_condition`, not re-derived. (2) `propose_detail_ops` types and places
window/door/chimney/dome/tower/balcony/column ops on the massing surface. (3) `tower`/`dome`/
`chimney` `add` ops are attempted against the redirected `element_library_train100_v1` (the
exact cache-reset pattern from `test_element_retrieval_baseline.py`, reused not re-derived); a
retrieval attempt that comes back empty (aspect/scale filtering) falls back to the original
procedural op, never a hard failure. (4) `EditableBuilding` CSG-composes the final ops onto the
massing SDF, sampled to `WORKING_RES=96` (ADR 0004).

**Caught before the full run:** `propose_detail_ops`'s production default (`max_ops=14`, tuned
for interactive per-click UI responsiveness) meant window/door ops filled the op budget before
the planner's rarer tower/dome/chimney predictions ever got a chance — an initial 11-building
smoke sample showed **zero** retrieval attempts anywhere. Confirmed by direct investigation (not
guessed): at `max_ops=100` most buildings still only produced 3-19 total ops (never approaching
the cap), but tower/dome/chimney appeared in ~3/8 sampled buildings once given the room. Per an
explicit project-owner decision, raised to `MAX_DETAIL_OPS=40` for this research-batch generation
only (documented in code as a deliberate deviation from the live sculptor's default, not a
silent change to production behavior).

**Code-review findings, fixed:** (1) Standards review found a docstring line typo, a duplicated
`_montage` helper (already an established pattern across 3 sibling scripts — noted, not
refactored, matching the existing convention), and — in the harness-compat script — a missing
`Out:` docstring line, no `git_provenance()`, no per-building failure-accounting (one bad grid
would have killed the whole run), and a vacuous `shape_ok=True` field (the preceding `assert`
would raise before it could ever be `False`). (2) Spec review found a real gap against tickets.md's
own acceptance bar ("every output traces to allowed massing and element sources"): only aggregate
retrieval counts were recorded, not which specific elements. Fixed by adding a `retrieved_elements`
list per building (`lib_id`, `element_type`, `source_building`) — confirmed populated for all 109
retrieval-hit buildings after a full rerun. Spec review also caught the harness-compat script
computing **paired** IoU against the real target on the detail-inclusive composed grid — CONTEXT.md
is explicit that only massing fidelity is paired, detail fidelity is distributional/never paired
— removed rather than caveated, since `generate_decomposition_arm.py`'s own manifest already
records the correct massing-only paired IoU and the harness-compat script doesn't need a fidelity
metric at all, only proof the render/FID pipeline accepts this output format.

**Result — 277/277 succeeded, 0 failures:**

- Leakage tiers: 27 clean, 26 val_leak, 224 train_leak, 0 unknown (matches ticket 09's own count).
- **109/277 buildings (39%) got at least one retrieved element**, sharply concentrated by class —
  architecturally sensible, not a bug: RELIGIOUS 61/67 (91%), RESIDENTIAL 46/184 (25%, mostly
  chimneys), PUBLIC 2/8 (25%), COMMERCIAL 0/18 (0%). Of 4,146 total detail ops across the
  population: 175 retrieved, 28 fell back to procedural after an empty pool, 3,943 stayed
  procedural by design (window/door/balcony/balcony_upper/stairs/column).
- Massing fidelity (paired IoU vs. real target): mean 0.102, median 0.055 across all 277 —
  consistent with ticket 09's own clean-27-subset numbers (mean 0.090, median 0.054) using the
  identical method, cross-validating that massing generation behaves the same at this larger,
  mixed-leakage scale.
- Visual spot-check (`outputs/decomposition_arm_v1/montage.png` + a supplementary render of the
  three known tower-producing buildings) confirms detail composition visibly changes the output —
  window notches, added slabs — and that a genuinely retrieved element (church roof_structure)
  is visually distinct in character (columned portico detail) from the smooth procedural
  additions, not indistinguishable from them.
- Harness compatibility (`verify_decomposition_harness_compat.py`, 20 buildings): all shapes
  correct (96³), all rendered cleanly through `render_facades.py`, Inception feature extraction
  and FID bootstrap ran without error (undersampled at n=20, expected and disclosed, not a
  headline number).

Assets: `data/decomposition_arm_v1/{manifest.json, grids/<id>.npy}` (277 composed grids, ~1GB,
not committed to git — regenerable via the recorded seeds/provenance),
`outputs/decomposition_arm_v1/montage.png`,
`execution/artifacts/decomposition_harness_compat.json`.

**Explicitly out of scope for this ticket** (per its own Question, and confirmed by the Spec
review as not scope creep): the full head-to-head FID/paired-IoU comparison against
`monolith_v3` — that decision belongs to "Decide the Full-Data C2 Kill-Gate" (ticket 13), which
also still needs to build a render+FID+IoU pass for the monolith itself (`eval_monolith.py` never
computed one — only occupancy stats on a `train_100`-internal validation slice, not `test.json`
at all).

## Follow-up (2026-07-14): fixed the retrieval-pool bottleneck and extended scope

User request: find more ways to grow the retrieval pool for tower/dome/etc. Investigated two
candidate levers before landing on the real one:

- **External data (dead end):** `data/lod3_tum` (CityGML LoD3, TUM) was already flagged unused
  for elements in an earlier data audit. Confirmed why: its semantic vocabulary is only
  `window`/`door`/`roof` — CityGML LoD3 doesn't tag towers/domes/chimneys/balconies as discrete
  objects at all, so there's nothing to extract for this vocabulary regardless of population size.
- **`MIN_SOLIDITY=0.12` (the real bottleneck):** checked the *raw* (pre-filter) solidity
  distribution per type in `element_library_train100_v1` and found a single global threshold was
  systematically starving architecturally thin types — every type loses 77-100% of its raw pool
  at 0.12, and `balcony_upper`/`stairs` are *fully* dead (max solidity 0.117/0.106, just under the
  bar). Visual QA of a sample of the newly-unlockable low-solidity crops
  (`outputs/element_library_v1/qa_per_type_solidity_relax.png`) confirmed they're legitimate thin
  architecture — recognizable staircases, railed balcony decks, colonnades/porticos — not the
  skeletal/broken fragments the filter exists to exclude.

**Fix:** `scripts/server/element_fit.py` now uses `MIN_SOLIDITY_BY_TYPE`, a per-type threshold
table, instead of one global scalar — solid types (tower/dome/chimney) keep the original 0.12
bar since they already cleared it fine; thin types get their own, evidence-backed lower bar.
`build_element_library.py` imports the same table (single source of truth) for its manifest
reporting. No re-extraction needed — crops and solidity were always stored, only the *filter*
was wrong — so `data/element_library_train100_v1/manifest.json`'s `pool_size_above_min_solidity`
was patched in place (old global-0.12 figures preserved under
`pool_size_above_min_solidity_history` for provenance):

| type | old pool (0.12) | new pool | new threshold |
|---|---|---|---|
| tower | 122 | 122 | 0.12 (unchanged) |
| dome | 104 | 104 | 0.12 (unchanged) |
| chimney | 48 | 48 | 0.12 (unchanged) |
| roof_structure | 12 | 51 | 0.08 |
| column | 8 | 85 | 0.05 |
| balcony | 5 | 65 | 0.05 |
| balcony_upper | 0 | 72 | 0.04 |
| stairs | 0 | 32 | 0.03 |

Added 5 new tests to `test_element_retrieval_baseline.py` covering the per-type behavior
(threshold lookup + fallback, a thin type below the old global bar but above its own now
retrieves successfully, and a type below even its *own* lower bar still correctly returns
nothing — confirming relaxation didn't silently remove filtering altogether). 13/13 pass; the
existing 8 (all using `type="tower"`, unaffected since its threshold didn't change) still pass
unmodified.

**Extended ticket 12's own scope to match:** `generate_decomposition_arm.py`'s `RETRIEVAL_POOLS`
was originally tower/dome/chimney only (2026-07-13 decision, when balcony/column had dead-or-
borderline pools). Re-examined which types `propose_detail_ops` can actually emit at all —
window/door are always procedural by design, and `roof`/`stairs`/`balcony_upper` are explicitly
skipped inside `propose_detail_ops` itself ("massing already has a roof"), so no op ever carries
those `det` values regardless of library pool size. That leaves five real ADD candidates:
tower/dome/chimney/balcony/column — added the newly-viable balcony/column to `RETRIEVAL_POOLS`.
`balcony_upper`/`stairs` gained usable pools too but are **not** retrieval targets here: there's
nothing for them to ever upgrade.

**A real bug caught before it could ship:** `propose_detail_ops` emits `column` as
`kind="cylinder"`, `size=[radius, height]` (2 elements) — `op_half_extent` only handled
`sphere`/box-shaped ops and would have raised `IndexError` on `size[2]` the first time a real
column op reached retrieval scoring. Fixed with a cylinder branch
(`half = [radius, height/2, radius]`); added 2 new tests (`op_half_extent` and
`retrieval_params` on a cylinder op) that would have caught this before any integration run.
Smoke-tested end-to-end (20 buildings) before the full rerun specifically to prove this path
works, not just pass unit tests in isolation — confirmed `column`/`balcony` `det_type`s appearing
in `retrieved_elements`, 0 crashes.

**Full rerun result — 277/277 succeeded, 0 failures:**

- **158/277 buildings (57%, up from 39%) got at least one retrieved element.** Of 4,150 ops:
  473 retrieved (up from 175), 127 fell back to procedural after an empty pool, 3,550 stayed
  procedural by design.
- Retrieved `det_type` breakdown: `column` 286 (now the dominant retrieved type — columns/
  porches/colonnades apparently common across every building class, unlike towers which
  concentrate in religious architecture), `tower` 98, `chimney` 69, `dome` 13, `balcony` 7.
- Class breakdown shifted accordingly: COMMERCIAL 0/18 → **7/18**, PUBLIC 2/8 → **4/8**,
  RESIDENTIAL 46/184 → **85/184** (nearly doubled), RELIGIOUS 61/67 → 62/67 (already saturated).
- Massing fidelity unchanged (mean 0.102, median 0.055) — expected, since only the detail
  composition step changed, not massing generation.
- Harness compatibility re-verified clean (20/20, `n_failed=0`).
- Full regression pass: 44/44 tests green across `test_element_retrieval_baseline.py` (13),
  `test_build_element_library.py` (16), `test_generate_decomposition_arm.py` (15).

This directly changes what "the decomposition arm" looks like for ticket 13's eventual
comparison: retrieval is no longer a rare event concentrated in religious buildings, but a
majority-case behavior across the population. Whether that changes the eventual C2 verdict is
still ticket 13's question to answer, not this one's.
