# #5 — Data audit for recoverable architectural programs

*Effort: solid-first semantic architectural carving. Opened 2026-07-15, audited 2026-09-04. Blocks
[#10](10-program-recovery.md) and [#6](6-program-generator.md) by the tracker's own field, though
both closed already by measuring the corpus this ticket was meant to have picked first — see
"Why this is confirmatory, not gating" below.*

> Which real and synthetic data sources can legally and technically support semantic architectural
> edit programs over metric footprint sets, and what split, pseudo-label, provenance, licensing,
> and small human-audited annotation strategy should the specification require? Assess at least
> 3D BAG, OpenStreetMap/CityGML, IFC opening/void semantics, BuildingNet, ReLoD3/Texture2LoD3,
> TUM2TWIN, ArchiSet, and procedural programs grounded in real footprint distributions.

Resolved by auditing what this repo already runs against real primary sources (license pages,
dataset READMEs, the papers themselves), not by re-deriving a preference from scratch — the same
substitution [#6](6-program-generator.md) made: several of the ticket's questions turn out to be
settled by facts about our own data and code, and checking those facts took less time than an
abstract comparison would have.


## Why this is confirmatory, not gating

[#10](10-program-recovery.md) and [#6](6-program-generator.md) both formally list this ticket as a
blocker and both closed anyway, because the corpus they needed — real LoD2 footprints with roof
form — was already sitting in `data/real.h5`. That is not a process error to fix; it means two of
this ticket's named sources (3D BAG, and CityGML more broadly) were never really a choice among
alternatives. They are **already the corpus**, ingested, measured, and the thing every closed
massing ticket in this map ([#10](10-program-recovery.md), [#126](126-massing-scoring.md),
[#127](127-height-map-generator.md), [#129](129-classified-plane-parameters.md) onward) scores
against. This audit's job is therefore to (a) confirm that choice was sound and legal, (b) find what
it structurally cannot supply, and (c) settle the split/provenance/pseudo-label/annotation questions
the earlier tickets explicitly left to it — most concretely [#9](9-multi-footprint-coordination.md)'s
courtyard-patterns and style axes, both named there as "waiting on #5's data audit."


## 🔑🔑 The corpus already in production, and what it structurally cannot supply

`data/real.h5` (built by `scripts/ingest_3dbag.py` for the Netherlands and
`scripts/foundations/ingest_citygml_lod2.py --source {nrw,plateau}` for Germany and Japan) is the
**entire** real-building supervision this map has ever measured against: the 714/411 held-out sets
in `massing_arms_eval_ship714.json`, [#10](10-program-recovery.md)'s fitter, and every number in
`CONTEXT.md`'s reference table. Both ingesters reduce each building to the same schema —
`sdf`/`footprint`/`height_m`/`source_id`/`bag_id` at 64³ — which is exactly the height-field
reduction [#4](4-edit-algebra.md) and [#10](10-program-recovery.md) proved is lossless on this
corpus (`missing` of the blockout against GT = 0.000000 on 714/714).

🔑🔑 **And that reduction is also the ceiling, not just a convenience.** [#4](4-edit-algebra.md)
measured the four named void operations directly against the ingested geometry, before any
height-field reduction: **courtyard, passage, and light well are 0 voxels in 4,324,919**, and arcade
(an overhang, not a through-void) is 71 voxels — 0.0016%. This is not a limitation of the fitter or
the height-map representation; it is a fact about what LoD2 solids *are*. 3D BAG (LoD1.2/1.3/2.2),
NRW's LoD2 CityGML, and PLATEAU's LoD2 CityGML all model a building as an extruded footprint plus a
roof — none of the three carries an interior courtyard, a passage cut through the mass, or a light
well, at any LoD2 tier, by construction of the product itself. **No amount of auditing more LoD2
sources changes this** — the gap [#9](9-multi-footprint-coordination.md) named ("courtyard patterns
... needs #5's data") is not a sourcing problem solvable by finding a fourth country's cadastral
LoD2 set. It needs a source that models real interior/through voids at all, which is a different
product tier (LoD3+) or a different representation (BIM/mesh) entirely — addressed below.

This reframes the ticket's real question from "which of these seven is best" to two separable
questions answered separately: **massing and roof form** (already solved, by what's already
running) and **architectural voids** (not solvable by any of the seven at LoD2, and only partially
by two of them at all).


## Per-source audit

| source | technical fit here | license (checked 2026-09-04) | verdict |
|---|---|---|---|
| **3D BAG** (NL) | ✅ in production. LoD1.2/1.3/2.2 CityJSON via `api.3dbag.nl`; roof form present, voids absent (see above). Proven at 3D IoU 0.9970 on the recovered program. | **CC BY 4.0.** Required credit line is specific: *"© 3DBAG by tudelft3d and 3DGI"*, plus a link to [the copyright page](https://docs.3dbag.nl/en/copyright/) in digital media. | Keep. **Compliance gap below.** |
| **CityGML — NRW** (DE) | ✅ in production via `--source nrw`. Same LoD2 solid-plus-roof shape, same void absence. | **Data License Germany – Zero – Version 2.0** (`dl-de/zero-2-0`) — explicitly *"every use permitted without restrictions or conditions."* No attribution legally required. | Keep. No gap. |
| **CityGML — PLATEAU** (JP) | ✅ in production via `--source plateau`. Same shape, same void absence. | Multi-licensed by MLIT: **CC BY 4.0 or ODbL or ODC-BY**, publisher's choice of which to invoke. This repo does not currently record which one it is claiming. | Keep. **Decision + gap below.** |
| **OpenStreetMap / Simple 3D Buildings** | Already used elsewhere (`scene/extract_osm.py`, `scripts/osm_hunyuan_pipeline_smoke.py`) for a *different* pipeline (footprint → BuildingNet retrieval → placement), not for carving supervision. Simple3DBuildings tags are sparse and don't encode a general void history (per [`NOVELTY_SURVEY.md`](NOVELTY_SURVEY.md#available-code-and-data)); OSM footprint polygons themselves are real and useful as *plan geometry*, not as program supervision. | **ODbL 1.0** on the database (attribution + share-alike on the *data*); a trained model's weights are a "produced work" under OSMF's own guidance and are not share-alike-encumbered, but a redistributed *processed extract* would be. | Not a new source for carving programs. Existing use is a separate, already-running pipeline outside this ticket's scope; flagged only so nobody assumes OSM tags could supply void semantics — they can't. |
| **IFC `IfcOpeningElement`** | The one schema in this list that models real openings/voids as an explicit Boolean-difference relationship — genuinely the right semantic shape for courtyard/passage/light-well. No code path in this repo touches IFC at all. | Per-file license varies by publisher; no blanket answer. | **Schema reference only.** No corpus at comparable scale and coverage was found — the [Open IFC Model Repository](https://openifcmodel.cs.auckland.ac.nz/) and similar collections (`ifc-bench`, buildingSMART sample files) are small, heterogeneous single-building sets assembled for software conformance testing, not a footprint-scale training corpus. Use IFC's *void relationship* as the vocabulary reference for the operation ontology (already effectively what [#4](4-edit-algebra.md)'s `ARCHITECTURAL_VOCABULARY` does); do not plan to train on an IFC corpus. |
| **BuildingNet** | Already used elsewhere in this repo (`datasets/buildingnet_dataset.py`, cited by `docs/adr/0001`, `0002`) for the *separate*, already-shipped detail/appearance generator — pure mesh geometry plus retrieved part labels, no footprint conditioning, no metric scale, no carve-order supervision. Real exterior geometry *could* show real courtyards on some of its 2,000 buildings, but nothing in the dataset marks which. | **Gated, not open.** Access requires submitting [a Google Form](https://forms.gle/jFQpoRzRkrTCaTzX8); neither the GitHub repo (`buildingnet/buildingnet_dataset`) nor the project site publishes a license for the released meshes/labels. The existing in-repo usage carries no license record either. | Wrong shape for this ticket regardless of licensing (no footprints, no programs). The unresolved license on the *existing* usage is a real gap, but it belongs to the detail/appearance effort, not this one — named here so it isn't lost, not fixed here. |
| **ReLoD3 / Texture2LoD3** | LoD3 Munich buildings with real opening masks, CityGML + GeoJSON footprints, and street-level imagery — genuinely goes past LoD2. But its openings are **windows and doors on facades**: detail scale (below *s\**) by this project's own glossary, not the structural-scale courtyard/passage/light-well vocabulary [#4](4-edit-algebra.md) actually needs. Munich-only; small *n*. | **CC BY 4.0**, confirmed directly on [the Zenodo record](https://zenodo.org/records/15178144). | Not a fit for *this* ticket's massing-scale void gap. Real candidate for a future facade/opening detail effort (C2, not C1) — noted, not pursued. |
| **TUM2TWIN** | The umbrella project ReLoD3 is drawn from: ~100,000 m² of Munich campus, LoD2/LoD3 CityGML, point clouds, imagery. Small enough (one campus) that it is a plausible **validation** set, not a training source. | The arXiv listing shows a CC-BY badge, but that is arXiv's license on the **preprint**, not necessarily a stated license on the **dataset** — WebFetch against `tum2t.win` did not return an explicit data-license statement. **Unconfirmed**, do not treat as CC BY 4.0 without checking the project site directly. | Candidate for the small human-audited set below *if and only if* its data license is confirmed permissive before use — flagged as an open check, not assumed. |
| **ArchiSet** (ICCV 2025) | Real dataset — 13,728 building forms as point clouds/voxels/meshes plus window-to-wall-ratio labels and 1.48M sketch/render images. But it is a **single-view image-to-mesh reconstruction and window-ratio editing** benchmark: no footprint conditioning, no massing/void/subtraction program of any kind. Wrong shape at the wrong scale (facade detail, like ReLoD3, not massing). | Not found — no GitHub, download link, or license statement located for the dataset (only the CVF/IEEE paper pages, one of which 403'd on fetch). | Not a fit. Named explicitly so a future detail-scale effort doesn't waste time re-discovering the same mismatch. |
| **Procedural programs on real footprint distributions** | Not a data source — a synthesis method. [`NOVELTY_SURVEY.md`](NOVELTY_SURVEY.md#data-strategy-implied-by-the-literature) had ruled this "superseded" once [#10](10-program-recovery.md)'s fitter could recover exact programs from real LoD2 directly. That ruling is correct **for roof/massing operations** and wrong to generalize: it is the *only* lever in this list that can produce a courtyard, passage, or light well at all, since every real source above either lacks them entirely (LoD2) or models them at the wrong scale (LoD3 facades). | N/A (synthetic). | **Un-superseded, narrowly.** Required for two things nothing real can supply: void-tier training examples, and guided-edit (rough-carve) training pairs (below). |

`NOVELTY_SURVEY.md`'s existing table (lines 216–244) covers the same ground at the literature level;
this table supersedes it wherever they disagree, because this one is checked against what the repo
actually runs and against primary license pages rather than paper text alone.


## 🔑 A real compliance gap exists today, independent of anything this ticket decides

Two of the three sources already in `data/real.h5` are CC BY 4.0 (3D BAG confirmed; PLATEAU
conditionally, depending which of its three license options is claimed). Neither `ingest_3dbag.py`
nor `ingest_citygml_lod2.py` records a license, credit string, or source version anywhere, and no
`LICENSE`/`DATA_SOURCES`/`NOTICE` file exists at the repo root or under `docs/`. This is true **today**,
independent of anything else this ticket decides, and it is the kind of gap an "audit data" ticket
exists to surface.

**Decision:** any publication, demo, or redistribution of this corpus or checkpoints trained
directly on it must carry, at minimum:
- 3D BAG: *"© 3DBAG by tudelft3d and 3DGI"* with a link to `https://docs.3dbag.nl/en/copyright/`.
- NRW: no attribution legally required (`dl-de/zero-2-0`), but source should still be named for
  provenance.
- PLATEAU: pick one of CC BY 4.0 / ODbL / ODC-BY explicitly and record the choice — **CC BY 4.0** is
  recommended, for one consistent attribution regime with 3D BAG rather than three different legal
  bases across three ingesters.

This is scoped to a spawned ticket below rather than fixed in this planning-only pass, per
[#1](https://github.com/danvisai/SDFusion/issues/1)'s own rule.


## Split policy: today's split has a proven confound, patched at the wrong layer

`datasets/bag3d_dataset.py:38-42` splits `real.h5` 96/2/2 by `np.random.default_rng(0).permutation`
over all rows — no stratification by `source_id`, no geographic holdout, no adjacency exclusion.
`scripts/foundations/eval_massing_arms.py:557-574`'s `pick_ids` already had to work around a
consequence of this: naive ascending-row sampling of the *held-out set* returned **100% Dutch
buildings** for any small `--n`, because row order tracks ingestion order and "region is the
strongest variable here" (mean height 11.97/5.90/7.47 m NL/DE/JP; blockout `extra` median
0.223/0.162/0.000) — silently voiding one earlier measurement on this map before the fix. The fix
that shipped is a **round-robin re-sort of an already-fixed held-out sample** — it makes small `--n`
eval slices region-balanced, but it does not touch which buildings are *in* train vs. held-out in
the first place, and it does nothing about spatial adjacency (two buildings from the same city bbox
can land on both sides of the boundary).

**Decision:** the split itself, not just eval-time sampling of it, must be stratified by
`source_id` at minimum, and should hold out by tile/bbox rather than by individual building row —
two buildings ~50m apart in the same Rotterdam block are not independent draws (their roof family
and height rhythm correlate structurally, which is exactly what [#9](9-multi-footprint-coordination.md)'s
block-coordination axes are betting on). This is a data-generation change, not a training change,
and is scoped to a spawned ticket rather than done here.


## Provenance: partially real already, one drift to fix

Provenance is in better shape than the license gap: every ingested building already carries
`bag_id` (`scripts/ingest_3dbag.py:168`, `scripts/foundations/ingest_citygml_lod2.py:220`) — the
source-native identifier (BAG pand id / CityGML `gml:id`) — and `source_id` (NL/DE/JP). That is
enough to trace any building back to its origin record, which is what a takedown or a license audit
actually needs.

⚠️ **One doc/code drift, worth a one-line fix rather than a spawned ticket:**
`scripts/foundations/ingest_citygml_lod2.py:18`'s schema docstring names a `src_key (N,) S64` field
that the code never writes (`bag_id` is what's actually created, at line 220). Not a functional bug
— nothing reads `src_key` — but the schema comment should say what the file actually contains.

**Still missing, and worth requiring going forward:** a per-ingestion-run snapshot stamp (API/tile
version or fetch date). 3D BAG and PLATEAU are both living, periodically-republished datasets;
without a recorded snapshot, "the same `bag_id` was different geometry last quarter" is
undiagnosable. Folded into the same attribution-manifest ticket below, since both are "record
metadata about where a building came from" work on the same two files.


## Pseudo-label strategy: already decided, extends without modification to DE/JP

[#6](6-program-generator.md) already answered the induction-method half of this question against
this exact corpus: **exact supervision, not pseudo-labels-with-RL**, via
[#10](10-program-recovery.md)'s constrained beam-search fitter compiling a canonical `Layer`/
`Ramp`/`CutRoof` program from each real building's height field. The project glossary's own term for
the result — **"recovered carving program"** — is precise: *"an approximate semantic architectural
edit program fitted to a real ... building for supervision. It is a pseudo-label whose
reconstruction error and ambiguities must remain visible."* Nothing about extending this to Germany
and Japan requires new work: `ingest_citygml_lod2.py` was written specifically so DE/JP flow into
"the SAME packed h5 schema as `data/bag3d_v1/bag3d.h5`" (its own docstring), and the fitter operates
on that shared schema, not on anything NL-specific. This audit's contribution is confirming there is
no region-specific blocker to recovering programs for DE/JP the same way NL already has been.

**Still genuinely open, and not this ticket's to close:** [`NOVELTY_SURVEY.md`](NOVELTY_SURVEY.md)'s
item 3 ("keep several near-equivalent candidates when the operation history is ambiguous") is not
built — the fitter returns one beam-search winner, not a retained candidate set. That is a fitter
change, not a data-sourcing one, and is out of scope here.


## Guided-edit pairs and the void tier: what procedural synthesis is actually for now

Two needs in this map cannot be met by any real source audited above, for the same underlying
reason — no real dataset here contains a "before" state for an edit, only finished buildings:

1. **Rough-carve interpretation training pairs** ([#3](3-dual-mode-carving-edit-locality.md)'s
   guided-edit mode needs to learn from *(rough gesture, completed operation)* pairs; no real corpus
   records a user's rough gesture before the fact).
2. **Void-tier examples** (courtyard/passage/light well) — zero real examples exist in the corpus at
   all, per [#4](4-edit-algebra.md)'s measurement above, and no audited source here changes that.

`NOVELTY_SURVEY.md`'s "synthesize guided-edit pairs from canonical programs" (item 5) already names
(1). This audit adds (2) as an equally real, equally unmet need with the same fix: procedural
generation grounded in the real footprint distribution already in `data/real.h5` (courtyard/passage/
light-well operations authored against real footprint polygons and real height distributions, not
synthetic footprints from scratch) is the only source in this audit that can populate the volumetric
tier [#4](4-edit-algebra.md) declared but left unexercised. This does not reopen
[#6](6-program-generator.md)'s "exact supervision, not synthetic" decision for the *core* tier
(`layer`/`ramp`/`cut_roof`) — that stands, real data wins there. It is narrowly the mechanism for the
one tier where real data structurally cannot compete, because none exists.


## Small human-audited annotation strategy

`NOVELTY_SURVEY.md`'s item 4 named this as needed and left it unsized. Concretely:

- **What's audited:** not whole buildings — individual **typed operations** from
  [#10](10-program-recovery.md)'s recovered programs, rendered through
  [#7](7-validity-gates-and-visual-carving-traces.md)'s per-step visual carving trace (closed,
  already built for a different purpose and directly reusable here).
- **Size and stratification:** ~60 buildings, sampled from the 714 held-out set, stratified across
  the three regions (NL/DE/JP) **and** [#10](10-program-recovery.md)'s carve-needing split (411
  carve-needing / 303 no-carve), oversampling the higher-op-count bucket — [#130](130-baselines-diffusion-curriculum.md)
  already measured that failure is graded steeply by slot count, so a flat sample would under-audit
  exactly the buildings most likely to have an ambiguous operation.
- **What's labeled:** each operation gets one of the glossary's named architectural types
  (`courtyard`/`passage`/`arcade`/`terrace or setback`/`roof cut`/`wing`/`roof volume`/`light well`)
  or `ambiguous` / `not architectural` (an artifact of the fitter, not a real feature), plus a
  free-text note. This is a **different question** from
  [#7](7-validity-gates-and-visual-carving-traces.md)'s existing three-question rubric (does it look
  like a building / artifacts / matches the request) — that rubric judges finalize-time output
  quality; this one judges whether a recovered *operation* deserves the semantic name attached to
  it, which is what determines whether the model learned "courtyard" or merely "negative polygon"
  ([`NOVELTY_SURVEY.md`](NOVELTY_SURVEY.md), risk #2).
- **Protocol:** two independent annotators per operation, disagreements adjudicated by the ticket
  owner; inter-annotator agreement reported, not hidden, per this map's standing rule against
  metrics that could flatter a result ([#7](7-validity-gates-and-visual-carving-traces.md)'s "no
  per-mesh rescaling" precedent, same spirit).
- **Role:** this set validates whether `layer`/`ramp`/`cut_roof` operations *deserve* their
  glossary-level architectural names on real buildings (they currently resolve generically — see
  [#4](4-edit-algebra.md)'s "nine names, each resolved" table). It does not replace the 714/411
  mechanical eval set; it is a semantic check the mechanical IoU/`extra`/collapse numbers cannot
  perform, same distinction [#127](127-height-map-generator.md) already drew between a scalar and a
  montage.
- Munich's TUM2TWIN/ReLoD3 data (LoD3, real openings) is a plausible **source of extra held-out
  buildings** for this set specifically, once its dataset-level license is confirmed — not for bulk
  training, where its scale (one campus) and scope (facade openings, not massing voids) don't fit.


## What this ticket explicitly does not decide

- **The exact attribution-manifest format, split-stratification code, and annotation-tool
  mechanics** are left to the spawned tickets below, at the altitude
  [#9](9-multi-footprint-coordination.md) left its own mechanics to
  [#149](https://github.com/danvisai/SDFusion/issues/149)–[#151](https://github.com/danvisai/SDFusion/issues/151).
- **The relational-graph upgrade's adjacency/party-wall data need**, named but explicitly *not*
  covered by [#9](9-multi-footprint-coordination.md), stays unscoped — nothing audited here supplies
  parcel-adjacency metadata either, and no ticket is opened for it, matching
  [#9](9-multi-footprint-coordination.md)'s own restraint.
- **Style**, [#9](9-multi-footprint-coordination.md)'s other deferred axis, remains undefined — no
  source audited here defines what "style" would even mean as a labeled quantity, and inventing a
  definition is not a data-sourcing question.
- **Whether to actually build the volumetric-tier procedural synthesizer** is a design/engineering
  decision for the spawned ticket, not settled here beyond "this is the only lever that can."


## Tickets this spawned

Three tracer-bullet slices, `ready-for-agent`, matching the granularity
[#9](9-multi-footprint-coordination.md) spawned as
[#149](https://github.com/danvisai/SDFusion/issues/149)–[#151](https://github.com/danvisai/SDFusion/issues/151):

- [#152](https://github.com/danvisai/SDFusion/issues/152) — Record Source License and Provenance
  Metadata for the Ingested Real Corpus
- [#153](https://github.com/danvisai/SDFusion/issues/153) — Stratify the Train/Held-Out Split by
  Source Region and Tile
- [#154](https://github.com/danvisai/SDFusion/issues/154) — Build the Small Human-Audited
  Void-Semantic Annotation Set

Not spawned as a ready-for-agent ticket: the volumetric-tier procedural synthesizer. It needs a
design decision (what courtyard/passage/light-well parameter distributions to draw from) that this
audit is not positioned to make unilaterally — named as real, necessary future work rather than
guessed at, the same restraint [#9](9-multi-footprint-coordination.md) applied to the
relational-graph upgrade.


## What follows

- [#9](9-multi-footprint-coordination.md)'s courtyard-patterns and style axes remain deferred:
  courtyard now has a named mechanism (procedural synthesis, not any real source), and style remains
  genuinely undefined by anything audited here.
- [#152](https://github.com/danvisai/SDFusion/issues/152)–[#154](https://github.com/danvisai/SDFusion/issues/154)
  are ready for an agent to pick up independently; none blocks the others.
- The compliance gap (two CC BY 4.0 sources, zero attribution anywhere in the repo) is real today
  and does not wait on any further decision — it is the most time-sensitive item this audit found.
