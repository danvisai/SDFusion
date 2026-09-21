# Wireframe-guided `Ramp` candidates for #10's fitter

*Effort: solid-first semantic architectural carving / BuildingWorld corpus. Exploratory --
**no ticket opened for this yet**. Extends [#10](10-program-recovery.md)'s beam-search fitter
(`scripts/foundations/recover_massing_programs.py`), motivated by
[#129](129-classified-plane-parameters.md)/[#132](132-overcarve-and-assignment.md)/138/139 all
failing to predict roof pitch from the coarse 64³ SDF grid. Data source is BuildingWorld's
separate wireframe representation
([#156](https://github.com/danvisai/SDFusion/issues/156)'s corpus), most directly relevant to
[#159](https://github.com/danvisai/SDFusion/issues/159) ("Pilot the beam-search fitter on
BuildingWorld's CRS-safe cities") if and when that's picked up.*

## What this is, and what it is not

**Not training.** `recover_massing_programs.py` is a classical LP/beam search, not a neural
network -- there is nothing here to train. It reads a building's ground-truth geometry and
recovers a `Layer`/`Ramp`/`CutRoof` *program*, which is what [#6](6-program-generator.md)'s
actual neural generator later trains ON (as a pseudo-label). This work makes that recovery step
more accurate for BuildingWorld rows specifically; it does not touch #6's model or its training
loop at all.

**Not mesh cleanup.** Nothing here modifies, repairs, or denoises the raw `.obj` mesh geometry.
The mesh is read once (`building_to_sdf`) to get ground truth and is otherwise untouched.

**What it actually is:** a new, additional candidate source for #10's fitter, so that when a
wireframe is available for a building (BuildingWorld only -- 3D BAG/NRW/PLATEAU have no wireframe
equivalent), the recovered `Ramp` program for that building can be more accurate than what the
fitter's own LP alone would find. This can only ever operate at **recovery time**, on data that
already has ground truth -- wireframe data does not exist for a new footprint at generation time,
so this is strictly a data/pseudo-label-quality lever, not a generation-time capability.

## Method

1-3 (`scripts/foundations/wireframe_ramp_probe.py`, committed separately): classify a wireframe's
edges (wall/horizontal/sloped) by angle, cluster sloped edges into roof planes via loop-closure
(a plane is accepted only if its loose ends are bridged by a short chain of horizontal edges --
raw support-edge-count thresholds were tried first and dropped, since no single threshold serves
both simple and complex roofs; see that file's own docstrings for the two real bugs found and
fixed: over-permissive hub-vertex seeding, and single-hop-only loop closure missing multi-segment
eaves), convert each accepted plane to pitch/azimuth in Frame-N.

4-5, two versions:

- **v1 (superseded): post-hoc substitution.** Run #10's fitter unbiased to get a finished
  program, then force-overwrite a `Ramp` op's plane equation with the wireframe-derived one where
  regions overlap >=40%. Measured a real but mixed result on 8 Adelaide buildings: 3 wins, 3
  ties, 2 losses on `vol_iou`, including one outright regression (`27_27a_4`: -0.0126).
- **v2 (current): a new candidate source inside the search itself**
  (`_wireframe_ramp_candidates`, `recover_massing_programs.py:254-291`). A wireframe plane is
  evaluated, at every search step, against the CURRENT surplus exactly like the fitter's own
  `_ramp_candidates` -- it must stay at or above `target` everywhere in its region (the same
  containment guard, never cuts into GT) and must remove positive volume, or it is not offered at
  all. It competes for selection purely on `gain` (how much surplus it removes), composed at the
  call site via a new `_candidates_with_wireframe` wrapper -- deliberately NOT added as a
  parameter to `_all_candidates` itself, so existing test doubles that mock `_all_candidates`
  keep working unmodified (the same arm's-length pattern `FitBias` already established).

**The candidate is locally containment-safe:** it removes positive surplus without cutting below
the target. At the same search state, an unbiased higher-gain choice removes more surplus at that
step. This is NOT a proof that the entire finite-budget guided search dominates a separate unguided
run: new branches can prune old paths, and the greedy fallback also receives `wf_planes`.
Ties and optional `FitBias` also preclude a blanket strictly-higher-raw-gain claim. End-to-end
non-regression is measured for the seven reported cases, not guaranteed for every input.
`27_27a_4`, v1's regression, is a small improvement under v2 on the same building.

## Integration point

`fit_program`/`fit_program_beam` both gained an optional `wf_planes=None` kwarg (default `None`
is a pure no-op, reproducing the fitter exactly as it existed before this was added -- verified,
not assumed: `test_no_wf_planes_is_a_pure_no_op_on_the_composed_stream`). A caller passes:

```python
wf_planes = wireframe_planes_in_voxel_space(mesh_bytes, wireframe_obj_text, fp, y0)
ops, h = fit_program_beam(fp, y0, y1, target, wf_planes=wf_planes)
```

`wf_planes` is a list of `{"plane": (a, b, c), "region": bool mask}` dicts, already in the
fitter's own voxel-index convention (`frame_n_plane_to_voxel_plane` in
`wireframe_ramp_carve.py` handles the Frame-N -> voxel-index conversion, cross-checked against
`building_to_sdf`'s own axis convention rather than assumed).

6 new tests (`TestWireframeRampCandidates` in `test_recover_massing_programs.py`): containment
rejection, region confinement, no-surplus-no-candidate, the no-op default, and one end-to-end
"a wireframe candidate that genuinely dominates is what `fit_program` returns" case. All 77 tests
(71 pre-existing + 6 new) pass.

## Where this plugs into the actual pipeline -- and what's NOT connected yet

**Not yet connected to anything that runs automatically.** `real.h5` does not store recovered
programs at all -- #10's fitter is run on demand, reading `sdf`/`footprint` directly, producing an
artifact of recovered programs (e.g. for a training-data-prep or evaluation pass). So there are
two separate gaps before this can affect anything beyond a standalone script:

1. **BuildingWorld isn't in `real.h5` yet.** Ingestion (#174, `ingest_buildingworld.py`) doesn't
   exist and is still blocked on 4 tickets (#167/#168/#160/#162). Until BuildingWorld rows exist
   in the corpus, there is nothing in the shipped eval/training population for `wf_planes` to
   improve.
2. **No row-level real.h5-id <-> wireframe-file correspondence exists.** This work operates
   directly on raw BuildingWorld mesh/wireframe zip files (matched to each other geometrically --
   see `wireframe_ramp_probe.py`'s `match_wireframes_to_meshes`), not on ingested `real.h5` rows.
   Whatever ingests BuildingWorld (#174) would need to preserve enough identity (e.g. the source
   mesh filename) for a later recovery pass to re-locate each row's wireframe file.

**The direct next investigation is [#159](https://github.com/danvisai/SDFusion/issues/159)**
("Pilot the beam-search fitter on BuildingWorld's CRS-safe cities") -- that ticket already plans
to run #10's fitter on raw BuildingWorld samples, not perform full ingestion. That pilot can run
before #174 using the raw mesh/wireframe matching implemented here; its written blockers are the
CRS audit #157 (closed) and profile #158 (review pending). Production row-level use later needs the
two pipeline gaps above resolved. Including the optional wireframe variant is an explicit scope
choice, not automatic completion of the pilot. Downstream of that, if BuildingWorld's
recovered programs are ever used as pseudo-label training data for #6's generator, more accurate
`Ramp` planes on gable/hip roofs would matter more than usual, since fixing gable-roof diversity
is the literal reason map [#156](https://github.com/danvisai/SDFusion/issues/156) exists ("Arm Six
on the Gable Bar").

## Result (Adelaide, single city)

7 of 2,282 confidently-paired buildings had a wireframe candidate actually win a region in the
coherent search (a ~0.3% hit rate -- most of the time the LP's own fit is already optimal against
the discretized target, which is expected and correct). All 7: `vol_iou` improved or held, local
`extra` (within the won region) dropped in every case, several to exactly 0. Visual confirmation
(`scripts/foundations/render_wireframe_ramp_carve.py`): on two of three rendered examples,
baseline's fit visibly fails to carve a hip roof at all (leaves nearly the whole roof top solid);
wireframe-guided carves it down to close to GT. Full per-building data:
`execution/artifacts/wireframe_ramp_carve_adelaide.json`.

## What this doesn't test or decide

- **Statistical significance.** N=7 wins, one city. Not a claim that this generalizes, only that
  the mechanism works correctly and the cases it does win, it wins cleanly and safely.
- **Region *segmentation* quality**, only plane *equation* accuracy once a region is already
  chosen by the fitter's own surplus-labeling. A wireframe plane can still only compete within
  regions the fitter's own connected-component surplus search already proposes.
- **Whether the wireframe plane's own boundary (a convex hull of support vertices, rasterized and
  intersected with the footprint) is itself accurate** -- an independent source of error from the
  plane equation, not measured separately here.
- **Whether recovered-program quality on BuildingWorld actually changes downstream
  training/eval outcomes** -- unreachable until the two pipeline gaps above close.

## Artifacts

- `scripts/foundations/wireframe_ramp_probe.py` -- steps 1-3 (edge classification, plane
  clustering, pitch/azimuth), plus geometric mesh<->wireframe matching.
- `scripts/foundations/render_wireframe_ramp_probe.py` -- visualizes steps 1-3.
- `scripts/foundations/recover_massing_programs.py` -- `_wireframe_ramp_candidates`,
  `_candidates_with_wireframe`, `wf_planes` threaded through `fit_program`/`fit_program_beam`.
- `scripts/foundations/test_recover_massing_programs.py` -- `TestWireframeRampCandidates`.
- `scripts/foundations/wireframe_ramp_carve.py` -- steps 4-5, the coherent-search comparison.
- `scripts/foundations/render_wireframe_ramp_carve.py` -- GT/baseline/wireframe-guided voxel
  comparison, `extra`-highlighted.
- `execution/artifacts/wireframe_ramp_carve_adelaide.json` -- full per-building result data.
