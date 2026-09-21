# Integration state

Reconciled 2026-09-09 through commit `1a2dbde`, local source, saved evaluation artifacts,
and live GitHub issues/comments. This describes checked-in behavior, not the state of a
running server. See [PROJECT_STATE.md](PROJECT_STATE.md) for the full work map.

## Two demos and a research pipeline

| Path | Input → output | What is connected |
|---|---|---|
| Original demo (`inference_service.py`) | Metric footprint/class/height/style → recipe parameters and GLB; sculpt requests → SDF/edit state | Recipe inference, SDEdit/refinement, detail, appearance and export APIs. The index hides single-building, selected-building and export panels. |
| New town (`town_generate_service.py`) | Metric footprint/height/region → normalized field → world-space vertices/faces | A2, optional height-map models and retrieval; NDJSON town streaming and seven-arm comparison. No generated EditOp payload. |
| Height-map research | Conditioning → per-column depths or typed slot predictions → height map | Training, evaluation, program compilation, fitting and diagnostics. The service imports shared decoding from this module. |
| Semantic edit engine (`scene/sdf_edit.py`) | Base SDF plus serialized EditOps → compiled SDF/occupancy/mesh | Add, undo, deletion by stable ID, replay, bidirectional layer/ramp operations and per-footprint coordinated replacement. |

The original index/sculpt bridge uses local storage. It does not connect the new town editor's
generated meshes to the semantic edit engine. Serving `town.html` through the original server
does not change which APIs that page calls.

## What each generator actually serves

`ARM_ORDER` names `envelope`, `heightmap_mode`, `heightmap_median`,
`heightmap_slope`, `heightmap_program`, `retrieval`, and `a2`.
The default is `a2`, loading `vecset_v4_surf.pth`; published v5 checkpoints are not that default.

- The CE mode and median share a model and differ only in posterior decoding.
- The program arm predicts assignment/type/plane heads, then `decode_prediction` compiles
  them to heights. The service returns only a surface, not those decisions.
- #155's `fit_decode` calls the beam fitter on generated heights and returns its fitted height
  map. It is available through the evaluation CLI, **not called by the town service**.
  The decision to ship it is outstanding integration work, explicitly acknowledged in #155's
  closing comment. #8/#181 previously called it served; that wording was inaccurate.
- Unbiased, unsmoothed fusion reduces recorded collapse from 0.0268 to 0.0049, increases
  `extra` from 0.0603 to 0.0807, and reduces planar fraction from 0.20 to 0.00
  (411 historical carve-needing buildings). A safety improvement is not a roof-form fix.

## Generated geometry still loses its program

The neural program path discards slot decisions at compilation. The fit-on-prediction path
discards the fitter's operation list. Recovered programs have been exercised through
`layer_program_to_ops` and `EditableBuilding`, but the generated town response has no
program, operation IDs, base-envelope descriptor, or recipe-version field.

Existing pieces can form an integration path:

```text
predicted slots OR fitted generated height map
  → connected polygon regions + typed operations
  → layer_program_to_ops
  → EditableBuilding(base envelope, operations)
  → serialize recipe state alongside derived preview geometry
```

This is more than adding a return value: #2 must settle state ownership, frame conversion,
disconnected predicted slots, stable identities, replay semantics and the validation boundary.
`mask_to_rings` refuses disconnected regions; `mask_components_rings` can split them.
Lossless rings remain larger than a small fixed vertex budget (#131/#134).

## Validity is implemented in separate layers

| Check | Existing caller / limitation |
|---|---|
| `op_problems` | Enforced by `EditableBuilding.add`; constructors and `from_state` do not automatically run it. |
| `finalize_problems` | Syntax + commutativity + height-map representability; called by `commit_block_program`. |
| `containment_problems` | Checks compiled occupancy against footprint and height bounds, including the ground perimeter. It is a separate helper, not called by `commit_block_program`. |
| Visual carving trace / human rubric | Implemented utilities and tests; not automatically produced by every serving endpoint. |

**Unresolved contract conflict:** #3/#140 allow ordered mixed add/subtract edits. Those compile
and have locality tests, but the #7/#145 finalize helper rejects noncommuting mixed programs.
#179's completed proxy refits add/subtract gestures into accepted recovered programs, without
weakening the gate. It observed validity on 461/461 pairs but failed locality (31.7% operation-level,
41.9% occupancy-level pass across 40 buildings). Direct mixed editing is not automatically accepted,
and a locality-preserving completion strategy remains unresolved for #2.

**Evaluation versus production:** completed #179/#180 harnesses run both checks. #180 observed
330/330 valid checks on 55 footprints / 8 scenes with two-level placeholder massing. The generic
commit helper still calls only `finalize_problems`; neither experiment integrates the full gate
into town serving. The all-success bootstrap intervals are descriptive, not population certainty.

## Dependencies and reproducibility

- Town startup unconditionally loads A2/Dora before optional height-map arms. The height-map
  model itself is codec-free, but a height-map-only deployment is not currently supported.
  `_generate_one` also reads A2 state before dispatch.
- Decoder helpers are imported from the training/experiment module. Checkpoint metadata records
  some decode choices, but the service does not consistently restore those choices; changing
  module defaults can change served behavior.
- Stored SDF grids use negative-inside values and array order `[z,y,x]`. Recovered meshes
  require outward-winding repair and frame conversion before encoding; earlier frame mistakes
  invalidated two runs. Use the established adapters.
- `ingest_surfaces.py --verify` compares against an existing `real.h5`; it does not rebuild
  that file. A clean-clone full-corpus rebuild command is not currently documented/implemented.
- BuildingWorld #162/#177 preserve the historical split. #153 now implements the separate proof
  split; #181 must select it explicitly. Of its 7,120 test rows, 6,837 are in the old Bag3d training
  split and 6,937 in the height-map eligible non-held pool before validation selection. Existing
  checkpoints cannot be presumed unseen-data controls; the old retrieval bank must also be filtered.
- HTML is served from disk while imported Python lives in process memory. Source inspection
  alone cannot establish the version running on a remote demo.

## Current handoff

Latest commit `1a2dbde` completes the #180 evaluation; #179 is also completed with a negative
locality result. `fcff154` adds optional wireframe-guided Ramp candidates to GT recovery,
not BuildingWorld ingestion or generator training/serving. #157/#153 are closed; #158 awaits review.
#2/#152/#154/#181 remain open. Concurrent uncommitted `bank_eligibility` / `--bank_exclude_ids`
work for #181 excludes named rows from retrieval, but does not erase checkpoint training exposure.
The #159 pilot can run from raw source data without waiting for #174; production pseudo-label use
later requires ingestion and row/source identity. The optional wireframe path is not wired into
the standard corpus-recovery CLI or town service. Its adoption remains a map-scope decision.
Headless tests verify individual contracts, not completion of those end-to-end tasks.
See [RECONCILIATION_AUDIT.md](RECONCILIATION_AUDIT.md) for verified counts and coverage.
