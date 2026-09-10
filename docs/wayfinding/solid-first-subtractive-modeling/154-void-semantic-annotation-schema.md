# The void-semantic annotation schema

Storage format for [#154](https://github.com/danvisai/SDFusion/issues/154)'s small human-audited
void-semantic annotation set. Written so a later ticket can consume it as a semantic typing signal
without re-deriving the format (#154's own acceptance criterion 4).

Produced by `scripts/foundations/void_semantic_sample.py`; filled in by two independent annotators
through the annotation tool (an Artifact, not committed to this repo — see the ticket's own resolution
comment for the live link); read back by `compute_agreement` in the same module.

`execution/artifacts/void_semantic_ai_suggestions.json` holds the `{doc_id: {label, confidence,
reasoning}}` review hints an AI model generated for every operation (see the `ai_suggestion` field
note below) — the input to `--ai_suggestions` when regenerating the schema artifact.

## File shape

```json
{
  "schema_version": 1,
  "created": "2026-09-10T12:00:00",
  "labels": ["courtyard", "passage", "arcade", "terrace_or_setback", "roof_cut", "wing",
             "roof_volume", "light_well", "ambiguous", "not_architectural"],
  "sample_n": 60,
  "sample_composition": {"NL/carve/low": 3, "NL/carve/mid": 3, "NL/carve/high": 5, "...": "..."},
  "sample_buildings": [
    {"building_id": 12345, "region": "NL", "carve_needing": true, "op_count_bucket": "high", "n_ops": 3},
    {"building_id": 67890, "region": "JP", "carve_needing": false, "op_count_bucket": "low", "n_ops": 0}
  ],
  "operations": [
    {
      "building_id": 12345,
      "operation_id": "a1b2c3...",
      "operation_index": 0,
      "kind": "layer",
      "mode": "subtract",
      "region": "NL",
      "carve_needing": true,
      "op_count_bucket": "high",
      "trace_dir": "outputs/void_semantic_traces/12345",
      "composite_trace_path": "outputs/void_semantic_traces/12345/op0_composite.png",
      "ai_suggestion": {"label": "roof_volume", "confidence": "medium", "reasoning": "..."},
      "annotator_1": {"label": null, "note": null, "annotated_at": null, "annotator": null, "used_ai_suggestion": null},
      "annotator_2": {"label": null, "note": null, "annotated_at": null, "annotator": null, "used_ai_suggestion": null},
      "adjudication": {"label": null, "note": null, "adjudicated_at": null, "by": null}
    }
  ]
}
```

## Field notes

- **`operation_id`** lives in #141's own `EditOp.id` slot — the same field the edit stack,
  `commit_block_program`, and every other consumer of a recovered program already use — but
  `EditOp.id`'s own default (a fresh `uuid4` per call) is NOT stable across reruns of the same
  program, so `void_semantic_sample.build_operations` overwrites it with a hash of the op's own
  geometry (kind/mode/polygon/size/planes/roof) plus its position among that building's ops. Two
  runs of this module over the same recovered program reproduce the same `operation_id`s; a
  genuinely different recovered program (a re-fit that changed the decision) produces different
  ones, matching the property below.
- **`(building_id, operation_id)`** together are the record's own key, per the ticket's own text.
  `operation_index` is kept too (the op's position in its building's program) since it's how a human
  annotator would naturally refer to "the second operation," but it is NOT the identity key — a
  program can be re-fit and an op's index can shift while its id (if it's the same decision) does
  not.
- **`kind`/`mode`** are the ALGEBRA-level facts already known about the operation (`layer`/`ramp`/
  `cut_roof`, `add`/`subtract`) — #4's own generic resolution. The whole point of this annotation
  set is that `kind` alone doesn't say whether a given `Layer` carve is a `terrace_or_setback`, a
  `wing`, or `not_architectural` (a raster-noise sliver); that's what `annotator_1`/`annotator_2`
  are for.
- **`region`/`carve_needing`/`op_count_bucket`** are recorded so the sample's own stratification is
  auditable per-row, not just in the top-level `sample_composition` summary.
- **`sample_buildings`** lists every one of the ~60 sampled building ids with its own `n_ops` (how
  many operations it actually contributed to `operations` below) — not just the `region x
  carve_needing x op_count_bucket` counts `sample_composition` reports. A `carve_needing: false`
  building's recovered program is empty by construction (nothing to subtract from the envelope), so
  its `n_ops` is always `0` and it contributes no rows to `operations`; a `carve_needing`-true region
  cell with few or no candidates can end up under-represented at the operation level even though
  `sample_composition` (a building-level count) looks balanced. Reading `operations` alone would hide
  this; `sample_buildings` makes the actual operation-level yield auditable from the artifact itself.
- **`composite_trace_path`** points at one image per operation: #7's own 4 fixed views (front/back/
  left/right oblique, delta-highlighted), tiled into a single 2x2 PNG so the annotation tool needs
  one upload per operation instead of four — a packaging choice, not a different rendering.
  `trace_dir` still points at the individual 4-file-per-step directory `save_carving_trace` always
  writes, for anyone who wants the original files.
- **`annotator_1`/`annotator_2`** are deliberately unordered/anonymous slots, not "annotator A is
  always this specific person" — the annotation tool assigns whichever slot a session hasn't already
  filled for that operation, so either of two people filling in labels independently land in
  different slots without needing to coordinate who is "1" and who is "2".
- **`ai_suggestion`** (`{label, confidence, reasoning}`, `None` when not generated) is a REVIEW HINT,
  never a ground-truth annotation — produced by an AI model reading each operation's own composite
  trace image (`scripts/foundations/void_semantic_sample.py`'s `--ai_suggestions` CLI flag embeds it
  via `build_annotation_schema`'s own optional parameter). The annotation tool shows it ONLY to
  whichever pass fills the FIRST slot for an operation (speeding up that pass); the second,
  independent pass is deliberately shown nothing, so `compute_agreement` still measures real human
  agreement rather than two people converging on one model's opinion. `confidence` ("high"/"medium"/
  "low") reflects genuine uncertainty — a large low-confidence share is expected and honest, not a
  sign the feature is broken; it flags exactly the operations that deserve the closest human look.
- **`used_ai_suggestion`** (inside each annotator slot, `None` until that slot is filled) records
  whether that annotator's submitted label matched the suggestion they were shown — `False` when no
  suggestion existed for that operation, or when this is the unaided second pass. This is how the
  degree of AI influence on the "human-audited" label stays auditable rather than invisible.
- **`adjudication`** is filled ONLY when `annotator_1["label"] != annotator_2["label"]` — the
  ticket's own decision that disagreements are adjudicated by the ticket owner, never silently
  averaged away. `compute_agreement`'s own `disagreements` list is exactly the set of operations
  that need this filled in.

## Reading it back

```python
from scripts.foundations.void_semantic_sample import compute_agreement
import json

schema = json.load(open("execution/artifacts/void_semantic_annotations.json"))
agreement = compute_agreement(schema)
# {n_total_operations, n_labeled_by_both, percent_agreement, cohens_kappa, disagreements}
```

`percent_agreement`/`cohens_kappa` are `None` (not `0.0` or `1.0`) when there are zero
both-annotated operations to compute them over, or when every paired label falls on one constant
category (kappa's own denominator is 0/0 there) — never silently coerced to a number that would
misread as a real result.
