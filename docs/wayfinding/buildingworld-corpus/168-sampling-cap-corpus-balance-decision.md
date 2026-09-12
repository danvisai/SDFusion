# #168 — Decide the BuildingWorld sampling cap and corpus-balance policy

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:grilling`. Decided with
project-owner sign-off in-session; blocks [#174](https://github.com/danvisai/SDFusion/issues/174)
and [#178](https://github.com/danvisai/SDFusion/issues/178).*

## The question

The existing corpus is deliberately balanced: 11,776 NL / 12,000 DE / 12,000 JP rows (held-out
248/235/231, the pinned 714). BuildingWorld's candidate pool is roughly two orders of magnitude
larger and lopsided per city — Berlin ~523k, Calgary ~457k, Edmonton ~371k, Cape Town ~275k, versus
Greater Geelong ~887, Philadelphia ~2,938 pre-filter (populations per
[#158](158-watertightness-extent-profile.md)'s profile). With no stated cap or
reweighting, ingesting BuildingWorld swamps the existing prior, and the served arm's pinned bar
(`extra`=0.0603 vs. 1-NN 0.1031 — measured on the pinned 714's **411 carve-needing** buildings, per
127-height-map-generator.md; correction, code review: not all 714) was measured under the old,
balanced prior. Four
things needed deciding: (a) a per-city ingestion cap; (b) a corpus-wide total cap; (c) whether to
reweight training sampling to preserve (or deliberately shift) the existing source balance; (d) how
this interacts with the pinned 714 as an unchanged regression control versus a separate,
explicitly new-corpus bar.

## Decision

**(a) No per-city cap.** Every city ingests every row that survives the already-decided gates —
[#165](165-crs-units-policy-decision.md)'s CRS reprojection (and Toronto exclusion),
[#166](166-watertightness-standard-decision.md)'s relaxed watertightness standard, and
[#160](160-geometric-duplicate-gate.md)'s duplicate gate against PLATEAU/NRW. Region/city diversity
is explicitly **not** a design goal for this arm: the stated priority is footprint adherence and
crisp roof architecture (the `extra`/`missing`/`planar_fraction` axes map #1's arm-comparison table
tracks), not a curated geographic mixture. Concretely, this means the largest cities (Berlin,
Calgary, Edmonton, Cape Town) each contribute their full post-gate pool — plausibly hundreds of
thousands of rows apiece — dwarfing every existing NL/DE/JP source (11,776–12,000 rows each) and
every small BuildingWorld city (Greater Geelong ~887, Philadelphia ~2,938 pre-filter).

**(b) No corpus-wide total cap.** Same reasoning as (a): take the full post-gate, post-dedup pool
across all 18 remaining cities (Toronto dropped per #165), whatever size that turns out to be.

**(c) No training-time source reweighting.** Sampling stays raw-proportional to however many rows
each source contributes once ingested — no explicit weights to preserve the old NL/DE/JP/BuildingWorld
mixture. This is a deliberate, accepted trade: the training prior will end up dominated by whichever
few cities are largest, and the model's behavior on NL/DE/JP-style or small-city footprints may
drift as a result. The project owner was informed of this consequence and confirmed sampling should
stay proportional to whatever the (uncapped) ingest produces, consistent with (a)/(b)'s same
diversity-is-not-a-priority stance.

**(d) Bar: confirm #178's existing two-bar split; no change to #178.**
[#162](162-frozen-corpus-identity.md)'s pinned 714 stays completely unchanged and un-recomputed,
exactly as #178 already specifies — a fixed regression control, not something this ticket reopens.
#178 ("Rebuild a BuildingWorld-specific 1-NN baseline and pre-register its bar") builds a second,
separate, honestly-new bar: the rebuilt 1-NN baseline scored only on the new BuildingWorld held-out
set, with a per-roof-family breakdown, exactly as #178's own issue text already lays out. The served
arm's already-measured pinned bar (`extra`=0.0603 vs. 1-NN 0.1031, on the 411 carve-needing
buildings within the pinned 714 — not all 714) keeps its meaning as the old-corpus regression
check; it is not expected to describe the model's behavior on BuildingWorld's own held-out rows,
which is precisely why #178's separate bar exists rather than reusing this one.

## Consequences for #174 and #178

- `ingest_buildingworld.py` (#174) needs no cap or reweighting logic. Pull every row that survives
  the CRS (#165), watertightness (#166), and dedup (#160) gates, for every non-Toronto city, and
  write all of it into the new train bank uncapped.
- Training (`train_height_map_generator.py` or its successor) needs no source-balance reweighting
  added. Leave the sampler proportional to whatever rows land in the bank.
- #178 needs no change: it already specifies keeping the pinned 714 unchanged/un-recomputed and
  building a separate BuildingWorld-only bar with a per-roof-family breakdown. This ticket confirms
  that split rather than asking #178 to build a combined number.

## Status: decided (2026-09-11)

Project-owner-approved: no per-city cap, no total cap, no training-time reweighting — "diversity
does not have preference as long as we have crisp architecture forming." Pinned 714 stays frozen
and un-recomputed; #178's existing separate-bar-plus-breakdown plan is confirmed unchanged.
