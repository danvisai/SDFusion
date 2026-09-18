# Project state

Reconciled 2026-09-09 against commit `c5787a1`, local source, saved evaluation artifacts, and live
GitHub issue/comment state (checked directly via `gh`, not inferred from doc prose). This is the
entry point [README.md](../README.md) and [CONTEXT.md](../CONTEXT.md) point to for "what is
implemented, measured, served, and still open" — organized by **research** (which map/ticket owns a
question), **pipeline** (what code/data actually runs), and **demo** (what a live server actually
serves). [INTEGRATION_STATE.md](INTEGRATION_STATE.md) covers the code-level wiring between these in
more detail; this file is the index and the cross-check that ticket status, code, and served
behavior agree.

Two disclosed limits on this reconciliation itself: (1) GitHub issue state was checked for every
ticket cited below, but a ticket can be open for reasons other than "unstarted" — several here are
explicitly left open for human review, not because work remains; that distinction is called out
per-item, not assumed from state alone. (2) No server was started and no training/inference was run
to re-verify current behavior — "served"/"working" claims below cite the evidence (a log, a commit,
an issue comment) rather than a fresh live check.


## Research

Two active wayfinder maps carry current research; three older maps are settled or intentionally
paused; two more are separate, still-open product/demo tracks.

### Map #1 — [Specify Solid-First Semantic Architectural Carving](https://github.com/danvisai/SDFusion/issues/1)

`docs/wayfinding/solid-first-subtractive-modeling/`. Planning-only charter (does not implement or
train); its children do. Latest closed decision: [#8](https://github.com/danvisai/SDFusion/issues/8)
("Specify the Minimal Falsifiable Proof," closed 2026-09-06), which turned the `NOVELTY_SURVEY.md`
hypotheses into a concrete five-arm/H1a/H1b/H3 scoring protocol and spawned this map's current
frontier. Also closed: the representation/algebra chain (#4, #6, #9, #10, #126–#134, #138–#140,
#144–#148 — the core-algebra, program-recovery, vertex-budget, and validity-gate results
`CONTEXT.md`'s "Established results and limits" section summarizes), and — this session —
[#5](https://github.com/danvisai/SDFusion/issues/5)'s data audit gained a addendum ruling out an
existing procedural corpus (SYNBUILD-3D) as a shortcut past building the still-unbuilt void-tier
synthesizer; #5's own ruling (procedural generation on `data/real.h5`'s footprint distribution is
the only lever) stands unmodified.

**Frontier**, blockers checked live:

| # | Title | State | Blocked by |
|---|---|---|---|
| [#153](https://github.com/danvisai/SDFusion/issues/153) | Stratify the train/held-out split by source region and tile | OPEN, `ready-for-agent` | none — unblocked |
| [#179](https://github.com/danvisai/SDFusion/issues/179) | Build the guided-edit completion proxy, re-verify locality (H1b) | OPEN, `ready-for-agent` | none — unblocked |
| [#180](https://github.com/danvisai/SDFusion/issues/180) | Test multi-footprint coordination on real product footprint sets (H3) | OPEN, `ready-for-agent` | none — unblocked |
| [#181](https://github.com/danvisai/SDFusion/issues/181) | Run the five-arm autonomous-generation scorecard (H1a) | OPEN, `ready-for-agent` | **#153** (needs its split) |
| [#152](https://github.com/danvisai/SDFusion/issues/152) | Record source license/provenance metadata | OPEN, `ready-for-agent` | none — unblocked |
| [#154](https://github.com/danvisai/SDFusion/issues/154) | Build the small human-audited void-semantic annotation set | OPEN, `ready-for-agent` | re-pointed at #153's split (per #8) |
| [#2](https://github.com/danvisai/SDFusion/issues/2) | Define integration with the existing recipe/SDF stack | OPEN, `wayfinder:grilling` | needs a human `/grilling` session, not agent execution |

### Map #156 — [Fold BuildingWorld Into the Massing Corpus (Arm Six on the Gable Bar)](https://github.com/danvisai/SDFusion/issues/156)

`docs/wayfinding/buildingworld-corpus/`. Closed: [#165](https://github.com/danvisai/SDFusion/issues/165)
(CRS/units policy) and [#166](https://github.com/danvisai/SDFusion/issues/166) (watertightness
standard), both closed 2026-09-06 (commit `e48b9e9`), decided using #157's and #158's findings.

**Open for human review, not for more work:**
- [#158](https://github.com/danvisai/SDFusion/issues/158) — watertightness/extent profiling across
  all 19 cities. Its own write-up ends "leaving open for review... @danvisai please have a look" —
  pure fact-finding, nothing left to execute.
- [#157](https://github.com/danvisai/SDFusion/issues/157) — per-city CRS/units/reference-elevation
  audit. The write-up (`157-crs-units-audit.md`) was finished but sat uncommitted from a session
  that was killed by a SLURM walltime timeout; committed and closed out this session (see
  "Current handoff" below).

**Ingestion is specified, not written.**
[#174](https://github.com/danvisai/SDFusion/issues/174) (`ingest_buildingworld.py`, mirrors
`scripts/foundations/ingest_citygml_lod2.py`) does not exist yet — confirmed, no file matching
`*buildingworld*ingest*`/`ingest*buildingworld*` anywhere in `scripts/`. It is blocked by 6 tickets;
2 are done:

| # | Title | State |
|---|---|---|
| #165 | CRS/units policy | ✅ CLOSED |
| #166 | Watertightness standard | ✅ CLOSED |
| [#167](https://github.com/danvisai/SDFusion/issues/167) | Provenance-authority field for region/stratification | ⬜ OPEN |
| [#168](https://github.com/danvisai/SDFusion/issues/168) | Sampling cap / corpus-balance policy | ⬜ OPEN |
| [#160](https://github.com/danvisai/SDFusion/issues/160) | Geometric duplicate gate vs. PLATEAU/NRW rows | ⬜ OPEN |
| [#162](https://github.com/danvisai/SDFusion/issues/162) | Pin/assert identity of the frozen 35,776-row held-out split | ⬜ OPEN |

Downstream of ingestion — [#175](https://github.com/danvisai/SDFusion/issues/175) (isosurface
extraction), [#176](https://github.com/danvisai/SDFusion/issues/176) (pseudo-label regeneration),
[#177](https://github.com/danvisai/SDFusion/issues/177) (roof-family-stratified split for new rows),
[#178](https://github.com/danvisai/SDFusion/issues/178) (rebuild the 1-NN baseline) — are all OPEN
and not startable before #174. A further band of preparatory/decision tickets
([#159](https://github.com/danvisai/SDFusion/issues/159),
[#161](https://github.com/danvisai/SDFusion/issues/161),
[#163](https://github.com/danvisai/SDFusion/issues/163),
[#164](https://github.com/danvisai/SDFusion/issues/164),
[#169](https://github.com/danvisai/SDFusion/issues/169)–[#173](https://github.com/danvisai/SDFusion/issues/173))
is OPEN and unstarted. No new BuildingWorld training/retrain-arm ticket exists yet.

### Other maps

| Map | Issue | State | Note |
|---|---|---|---|
| Footprint-town demo | [#97](https://github.com/danvisai/SDFusion/issues/97) | OPEN | The standalone town editor (`town_generate_service.py`); #102/#104/#105/#135 open |
| Town experience | [#106](https://github.com/danvisai/SDFusion/issues/106) | OPEN | Identity/history, sharing, staged realization; #107–#112 open |
| Whole-volume voxel transform | [#113](https://github.com/danvisai/SDFusion/issues/113) | OPEN | Competing empirical route beside map #1, **not** a replacement; #114–#117 settled, #118–#125 open. Any recipe-contract change needs this map's explicit gate. |
| Latent token order | [#87](https://github.com/danvisai/SDFusion/issues/87) | CLOSED | Fixed the pair-training token-order corruption (#88–#91); the fix did **not** open a usable strength band — #92's aligned retrain made collapse worse (46.36% vs the encoded control's 8.96%). Do not restart from an old handover. |
| Solid massing | [#24](https://github.com/danvisai/SDFusion/issues/24) | CLOSED | Dense-grid massing generator, shipped |
| Surface fidelity / crisp massing / diffusion-latent accuracy | [#34](https://github.com/danvisai/SDFusion/issues/34), [#52](https://github.com/danvisai/SDFusion/issues/52), [#58](https://github.com/danvisai/SDFusion/issues/58) | CLOSED | Closed negative or ceiling-located; superseded by map #61 |
| Crisp massing via vecset (A2) | [#61](https://github.com/danvisai/SDFusion/issues/61) | OPEN | Planning purpose complete; A2 is trained/published. Implementation spec [#66](https://github.com/danvisai/SDFusion/issues/66)/[#67](https://github.com/danvisai/SDFusion/issues/67) retain unresolved quality/localized-edit questions, not an unbuilt codec |
| Vecset convergence | [#69](https://github.com/danvisai/SDFusion/issues/69) | CLOSED | Evaluation harness, decoded-surface loss, band-fix findings |
| C2 transform/composition proof | [#11](https://github.com/danvisai/SDFusion/issues/11) | CLOSED (stale) | Record kept in `.scratch/transform-composition-proof/`, `tickets.md`. Thesis/ADRs remain recorded hypotheses, not a completed comparative proof. Inactive. |
| Old demo-cleanup / clean structural generation | [#47](https://github.com/danvisai/SDFusion/issues/47) CLOSED, [#50](https://github.com/danvisai/SDFusion/issues/50) **OPEN** | — | #50 is not closed or formally marked superseded — its own record names A2 (#61) as the eventual replacement generation; #97's new town service sidesteps #50's problem with a separate A2 path rather than resolving it. Don't cite #50 as done. |

The `#92` retrain itself — 4 arms (A/B/C/D), aligned-pair objective vs. an encoded control — is
CLOSED. All four trained to the pre-registered target step 240000 (checkpoints exist at every 10k
from 190k–240k for all four, last written 2026-08-27; no gap), all 24 checkpoints evaluated, verdict
"**NOT MET, and not met transiently**" (`docs/wayfinding/latent-token-order/92-aligned-retrain.md`,
commit `21a4876`). Nothing from this line is currently training or running.


## Pipeline

**Real-corpus ingestion (live):**
- `scripts/ingest_3dbag.py` — Netherlands (3D BAG).
- `scripts/foundations/ingest_citygml_lod2.py --source {nrw,plateau}` — Germany, Japan.
- Both write the shared `sdf`/`footprint`/`height_m`/`source_id`/`bag_id` schema at 64³ into
  `data/real.h5` — the entire real-building supervision every massing ticket on map #1 measures
  against (per [#5](https://github.com/danvisai/SDFusion/issues/5)'s audit).
- `scripts/foundations/ingest_buildingworld.py` — **does not exist.** Specified by
  [#174](https://github.com/danvisai/SDFusion/issues/174), blocked as above.
- `scripts/foundations/profile_buildingworld_meshes.py` — exists; this is #158's
  watertightness/extent profiler, not an ingester.
- `scripts/foundations/ingest_surfaces.py` — recovers surface meshes from source data and, with
  `--verify`, re-voxelizes a sample and checks occupancy against the existing `real.h5`. It
  **requires an existing `real.h5`; it does not build one from scratch.** 35,623 of the corpus's
  35,776 rows have a recovered surface; a rebuild that compacts away the missing 153 changes row
  identity and the frozen split — do not do this silently (see `REPRODUCING.md` §4b).

**Program recovery and generation:**
- `scripts/foundations/recover_massing_programs.py` — [#10](https://github.com/danvisai/SDFusion/issues/10)'s
  constrained beam-search fitter; recovers an exact `Layer`/`Ramp`/`CutRoof` program from a real
  building's height field (3D IoU 0.9970 at K=4). Closed, done, the corpus's pseudo-label method.
- `scripts/foundations/train_height_map_generator.py` — the footprint-conditioned height-map
  generator (#127) and the typed-slot program generator (#6 lineage). #127's base CE+median model is
  **done and human-accepted** (`extra` 0.0603, `vs_input` 0.8432 on the 411 carve-needing subset) and
  is what's served today. Every subsequent program-route arm — #129, #132, #138, #139, and the
  locally-named "combined" experiment (checkpoint `heightmap_program_assign_tau05`, **not** GitHub
  #140 — #140 is "Make Layer and Ramp Ops Bidirectional," an unrelated, already-closed ticket; see
  the identifier correction in `140-combined-assignment-and-type.md`) — is **NOT MET** against
  `PROGRAM_BAR`. [#155](https://github.com/danvisai/SDFusion/issues/155)'s `fit_decode` (generator
  fused through #10's fitter) is CLOSED and measured — collapse 0.0268 → 0.0049, `extra` 0.0603 →
  0.0807 — and its own closing comment says explicitly: **ship it as an arm, but wiring it into
  `town_generate_service.py` is separate, not-yet-done work.** Any doc or ticket calling `fit_decode`
  "the arm actually served today" is describing the eval harness, not the live service — see Demo,
  below.
- A2 (`scripts/train_vecset.py`, `models/networks/vecset_*.py`) — the 49M-parameter vecset massing
  diffusion. Trained, published (`weights/massing-vecset/`), and is the town service's **default**
  arm. Token-alignment work on it (map #87 / #92) is closed and did not open a usable edit-strength
  band, as above.

**Currently running:** nothing. No training or inference process for any of the above was found
running this session (`ps` checked); the only recent background job (`outputs/watch_2x2_driver.log`,
a checkpoint-scoring poll loop for the already-finished #92 arms) died when its whole interactive
SLURM session hit a 14-day walltime and was timed out at 2026-09-08T20:09:50 — not a pipeline
failure, and nothing was mid-computation when it happened (all #92 checkpoints had been complete
since 2026-08-27).


## Demo

Three independent serving surfaces exist; see [README.md](../README.md#the-three-serving-surfaces)
for exact run commands and [INTEGRATION_STATE.md](INTEGRATION_STATE.md) for how (little) they share.

| Surface | Entry point | Last direct evidence checked this session |
|---|---|---|
| Original recipe/sculpt demo | `scripts/server/inference_service.py`, port 8099 | `outputs/server_8099.log`/`server_8100_verify.log`: a full model-loading run completed **2026-07-10**. Not re-verified live this session. |
| Footprint-town editor | `scripts/server/town_generate_service.py`, port 8767 | `outputs/town_generate_8767.log` shows healthy `/health` responses as late as **2026-09-01**; `outputs/town_generate_8791.log` (2026-08-29) shows 3 height-map arms plus the 34,909-building retrieval bank loading cleanly. Not re-verified live this session. |
| Image footprint extraction | `scripts/server/footprint_extract_service.py`, port 8766 | Separate classical-CV process; no model loading, lower risk of drift. Not independently re-checked. |

No server process is currently running (checked via `ps`). The three surfaces are independent
processes with independent state — the original demo's index/sculpt page uses `localStorage` and
does **not** connect to the new town editor's generated meshes.

**The one integration gap that matters most:** the town service serves compiled **meshes**, never
the program that produced them. Both the neural program path (assignment/type/plane heads →
`compile_program`) and the fit-on-prediction path (#155's `fit_decode`) construct an internal
program and discard it before returning. `EditableBuilding`'s full undo/re-roll/delete stack has
only ever been exercised on a **recovered** program (needs ground truth), never a **generated** one.
Every function needed to close that gap exists and is tested; nothing currently calls them in
sequence. Full detail, including the exact call chain and the two things a first attempt will hit
(`mask_to_rings` rejecting disconnected regions; the 94-vertex median ring size), is in
[INTEGRATION_STATE.md](INTEGRATION_STATE.md).

A second, standing hazard: **HTML is served from disk, Python from process memory** — a long-running
server process can be several commits behind the file it's importing. Check process start time
against file mtime before trusting a live demo's behavior against these docs.


## Current handoff

As of this reconciliation: map #156's latest closed work is #165/#166 (commit `e48b9e9`); #157/#158
are complete write-ups open for human review, not further agent work (#157 committed and closed out
this session — see its GitHub comment for the summary). Map #1's semantic-carving design (#8) is
closed; #152–#154/#179–#181/#2 are the open research/integration frontier, and #153 is the one
other open items key off. The demo has two independent, unreconciled serving surfaces and no
generated-program integration on either. Nothing is running. Headless tests
(`scene.test_sdf_edit`, `scripts.foundations.test_recover_massing_programs`,
`scripts.foundations.test_train_height_map_generator`, `scripts/server/test_town_generate.py
--geometry-only`) verify individual contracts, not completion of any of the above end-to-end tasks —
treat a passing suite as necessary, not sufficient, evidence for any claim in this file.
