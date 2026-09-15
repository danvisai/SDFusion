"""#154 -- the small human-audited void-semantic annotation set: sampling, trace rendering, the
storage schema, and inter-annotator agreement, re-pointed at #153's split by #8's grilling session.

This module builds everything EXCEPT the annotations themselves. #5's own decision is "two
independent annotators" and #8's own re-sequencing note is explicit that disagreements are
"adjudicated by the ticket owner, not silently averaged away" -- both require real human semantic
judgement this module cannot substitute for. What it builds:

1. **Stratified sampling** (`stratified_sample`) -- ~60 buildings from #153's TEST split, across
   region (NL/DE/JP) x carve-needing (#10's own `CARVE_NEEDED` predicate) x op-count bucket, biased
   toward the higher-op-count bucket (#130 already measured failure is graded steeply by slot/op
   count, so that bucket needs real representation even though it's naturally rarer).
2. **Trace rendering** (`render_building_trace`) -- #7's existing `render_carving_trace`/
   `save_carving_trace`, reused unmodified, one call per sampled building.
3. **The storage schema** (`build_annotation_schema`) -- versioned JSON keyed by
   `(building_id, operation_id)`; `operation_id` lives in #141's own `EditOp.id` slot, but
   `build_operations` overwrites its random default with a hash of the op's own geometry so the
   same recovered program reproduces the same id across reruns (see `build_operations`'s own
   docstring), with two empty annotator slots and an adjudication slot per operation.
4. **Agreement** (`compute_agreement`) -- percent agreement + Cohen's kappa over whichever
   operations both annotators have actually labeled, reported honestly (including "not enough
   labeled yet") rather than silently averaged away.

Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
      scripts/foundations/void_semantic_sample.py --ckpt-free   (see main()'s own --help)
Test: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_void_semantic_sample.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene.sdf_edit import footprint_envelope_sdf, layer_program_to_ops  # noqa: E402
from scripts.foundations.carving_trace import render_carving_trace, save_carving_trace  # noqa: E402
from scripts.foundations.recover_massing_programs import CARVE_NEEDED, H5, height_field  # noqa: E402
from scripts.foundations.stratified_split import SOURCE_NAMES, make_split  # noqa: E402
from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, open_real_corpus  # noqa: E402

ARTIFACTS = REPO / "execution/artifacts"
TRACES_DIR = REPO / "outputs/void_semantic_traces"

RECOVERY_MAX_OPS = 4       # #10's own bar-passing settings, reused verbatim
RECOVERY_BEAM = 12
RECOVERY_BRANCH = 10

SAMPLE_N = 60
SAMPLE_SEED = 154

# #4's glossary, restated as #154's own acceptance text names it, minus `courtyard` -- dropped by
# the ticket owner after the measured result below: real through-void courtyards are better served
# by retrieved/procedural assets and left-empty plan area, not by naming a subtractive height-field
# cut after them (0/129 operations had both annotators agree on it; every occurrence was one side of
# a disagreement -- see `docs/wayfinding/solid-first-subtractive-modeling/
# 154-void-semantic-annotation-schema.md`'s own field notes on `adjudication.valid_labels`).
LABELS: Tuple[str, ...] = (
    "passage", "arcade", "terrace_or_setback", "roof_cut", "wing", "roof_volume",
    "light_well", "ambiguous", "not_architectural",
)
SCHEMA_VERSION = 1


# ------------------------------------------------------------------------------------------------
# op-count bucketing -- oversample the bucket #130 found failure graded steeply against
# ------------------------------------------------------------------------------------------------

def op_count_bucket(n_ops: int) -> str:
    """low: 0-1 ops, mid: 2, high: 3+. `RECOVERY_MAX_OPS=4` caps `high` at 3-4; #130's own finding
    (failure graded steeply by slot count) is what motivates oversampling this bucket below, not a
    generic "more is harder" assumption."""
    if n_ops <= 1:
        return "low"
    if n_ops == 2:
        return "mid"
    return "high"


BUCKET_ORDER = ("low", "mid", "high")


# ------------------------------------------------------------------------------------------------
# 1. materialize #153's test split, recover programs over it (reusing #10's own CLI, unmodified)
# ------------------------------------------------------------------------------------------------

def materialize_test_ids(seed: int = 0) -> np.ndarray:
    """#153's own TEST split -- real.h5 row indices, matching every other consumer of this split
    this session (#181's own `materialize_split`).

    Restricted to the frozen NL/DE/JP prefix -- see `five_arm_scorecard.materialize_split`'s own
    docstring for why #177's appended BuildingWorld rows must not reach `make_split`.
    """
    with open_real_corpus(H5) as f:
        source_id = f["source_id"][:FROZEN_SPLIT_N_TOTAL]
        bag_id = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
    split, _report = make_split(source_id, bag_id, seed=seed)
    return np.sort(np.nonzero(split == "test")[0])


def run_recovery_cli(ids_path: Path, out_path: Path,
                     python: str = str(REPO / "sdfusion/bin/python")) -> None:
    """Shells out to #10's own, unmodified `recover_massing_programs.py` CLI -- the same
    beam=12/branch=10/max_ops=4 settings that met #10's own pre-registered bar, not reinvented."""
    cmd = [python, str(REPO / "scripts/foundations/recover_massing_programs.py"),
          "--ids_from", str(ids_path), "--max_ops", str(RECOVERY_MAX_OPS),
          "--beam", str(RECOVERY_BEAM), "--branch", str(RECOVERY_BRANCH), "--montage", "0",
          "--out", str(out_path)]
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def load_candidates(recovery_artifact_path: Path, h5_path: Path = H5) -> List[dict]:
    """Every building in a `recover_massing_programs.py` artifact, annotated with the region and
    op-count bucket `stratified_sample` needs. `blockout_extra` (already computed per row by that
    artifact) is the SAME carve-needing predicate #10's own harness uses -- not re-derived."""
    art = json.load(open(recovery_artifact_path))
    per_building = art["per_building"]
    ids = [int(b) for b in per_building]
    with open_real_corpus(h5_path) as f:
        source_id = f["source_id"][:]
    out = []
    for bid_str, row in per_building.items():
        bid = int(bid_str)
        region = SOURCE_NAMES[int(source_id[bid])]
        n_ops = int(row["n_ops"])
        out.append(dict(id=bid, region=region, n_ops=n_ops, bucket=op_count_bucket(n_ops),
                        carve_needing=bool(row["blockout_extra"] >= CARVE_NEEDED),
                        program=row["program"]))
    return out


# ------------------------------------------------------------------------------------------------
# 2. stratified sampling, oversampling the high-op-count bucket
# ------------------------------------------------------------------------------------------------

def stratified_sample(candidates: Sequence[dict], n: int = SAMPLE_N, seed: int = SAMPLE_SEED,
                      high_bucket_weight: float = 2.0) -> Tuple[List[dict], dict]:
    """~n buildings stratified by (region, carve_needing, bucket), the `high` op-count bucket
    weighted `high_bucket_weight`x a plain population-proportional share (#130's own finding that
    failure grades steeply by op/slot count is why that bucket needs real representation even
    though it is naturally the rarest). Deterministic given `seed`. Returns `(sample, composition)`
    where `composition` is the realized per-cell count, recorded alongside the output per #154's
    own acceptance criterion 1 ("recorded alongside the output, not just the final labels").
    """
    rng = np.random.default_rng(seed)
    cells: Dict[tuple, List[dict]] = {}
    for c in candidates:
        cells.setdefault((c["region"], c["carve_needing"], c["bucket"]), []).append(c)

    weight = {"low": 1.0, "mid": 1.0, "high": high_bucket_weight}
    raw_share = {key: len(v) * weight[key[2]] for key, v in cells.items()}
    total_share = sum(raw_share.values()) or 1.0
    target = {key: raw_share[key] / total_share * n for key in cells}

    # largest-remainder rounding so per-cell integer targets sum to exactly n (or as close as the
    # cell's own population allows), capped by that cell's own size.
    floors = {key: min(int(target[key]), len(cells[key])) for key in cells}
    remainder = n - sum(floors.values())
    order = sorted(cells, key=lambda k: (target[k] - floors[k]), reverse=True)
    for key in order:
        if remainder <= 0:
            break
        if floors[key] < len(cells[key]):
            floors[key] += 1
            remainder -= 1

    sample = []
    for key, k in floors.items():
        if k <= 0:
            continue
        pool = cells[key]
        idx = rng.choice(len(pool), size=k, replace=False)
        sample.extend(pool[i] for i in idx)
    sample.sort(key=lambda c: c["id"])

    composition = {f"{r}/{'carve' if cn else 'flat'}/{b}": floors.get((r, cn, b), 0)
                   for (r, cn, b) in cells}
    return sample, composition


# ------------------------------------------------------------------------------------------------
# 3. operations + trace rendering (#7's renderer, reused unmodified)
# ------------------------------------------------------------------------------------------------

def build_operations(fp: np.ndarray, y0: int, y1: int, program: list) -> list:
    """The sampled building's recovered program as `EditOp`s. `layer_program_to_ops` leaves each
    op's `.id` at #141's own default -- a fresh `uuid4` minted every call, NOT stable across reruns
    of the same program -- so this overwrites it with a hash of the op's own geometry (`to_dict()`
    minus the identity/bookkeeping fields `canonical_form` already excludes for the same reason)
    plus its position among this building's ops. Two runs over the same recovered program then
    reproduce the same `operation_id`s, which #154's own `(building_id, operation_id)` key needs to
    survive re-running this script without orphaning annotations already collected against the old
    ids."""
    ops = layer_program_to_ops(program, fp, y0, y1, res=fp.shape[0])
    for i, op in enumerate(ops):
        payload = {k: v for k, v in op.to_dict().items() if k not in ("id", "group_id")}
        payload["_index"] = i
        op.id = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
    return ops


def composite_trace_step(views: Sequence, pad: int = 4) -> "PIL.Image.Image":  # noqa: F821
    """#7's own 4 fixed views (front/back/left/right oblique) for ONE step, tiled into a single 2x2
    image -- presentation packaging for the annotation tool (one asset upload per operation instead
    of four), not a new rendering: every pixel is still exactly what `render_carving_trace` drew."""
    from PIL import Image

    w, h = views[0].size
    canvas = Image.new("RGB", (2 * w + pad, 2 * h + pad), (16, 16, 18))
    for img, (x, y) in zip(views, [(0, 0), (w + pad, 0), (0, h + pad), (w + pad, h + pad)]):
        canvas.paste(img, (x, y))
    return canvas


def render_building_trace(fp: np.ndarray, y0: int, y1: int, ops: list,
                          out_dir: Path) -> Tuple[List[Path], List[Path]]:
    """One `render_carving_trace` call (#7's own renderer, reused as-is -- #154's acceptance
    criterion 2) produces BOTH artifacts from the same in-memory frames: the standard 4-file-per-
    step output `save_carving_trace` always writes, and one composited 2x2 image per step for the
    annotation tool. Returns `(step_paths, composite_paths)`.
    """
    base = footprint_envelope_sdf(fp, y0, y1, res=fp.shape[0])
    trace = render_carving_trace(base, ops, fp, res=fp.shape[0])
    step_paths = save_carving_trace(trace, out_dir)
    composite_paths = []
    for step in trace:
        img = composite_trace_step(step["views"])
        p = out_dir / f"op{step['index']}_composite.png"
        img.save(p)
        composite_paths.append(p)
    return step_paths, composite_paths


# ------------------------------------------------------------------------------------------------
# 4. the storage schema
# ------------------------------------------------------------------------------------------------

def _empty_annotation() -> dict:
    return dict(label=None, note=None, annotated_at=None, annotator=None, used_ai_suggestion=None)


def build_annotation_schema(sample: Sequence[dict], composition: dict,
                            trace_dirs: Optional[Dict[int, Path]] = None,
                            composite_paths: Optional[Dict[int, List[Path]]] = None,
                            ai_suggestions: Optional[Dict[str, dict]] = None) -> dict:
    """The versioned JSON `docs/wayfinding/solid-first-subtractive-modeling/
    154-void-semantic-annotation-schema.md` documents. One entry per operation, keyed by
    `(building_id, operation_id)`; both annotator slots and the adjudication
    slot start empty -- filled in later, by real annotators, not by this function.

    A flat (`carve_needing=False`) building's recovered program is empty by construction -- there is
    nothing to subtract, so it contributes zero operations. `sample_buildings` records every sampled
    building's own `n_ops` (0 included) alongside `sample_composition`'s building-level counts, so
    that drop-off is auditable from the artifact itself rather than only visible by re-deriving it.

    `ai_suggestions`, when given, is `{doc_id: {label, confidence, reasoning}}` -- an optional
    review hint the annotation tool shows ONLY to whichever annotator fills the FIRST slot for an
    operation; the second, independent pass is deliberately unaided, so `compute_agreement` still
    measures real human agreement rather than two people converging on one model's opinion. `None`
    (the default) when no suggestions were generated for this run.
    """
    trace_dirs = trace_dirs or {}
    composite_paths = composite_paths or {}
    ai_suggestions = ai_suggestions or {}
    operations = []
    sample_buildings = []
    for building in sample:
        ops = building.get("ops")
        sample_buildings.append(dict(
            building_id=building["id"], region=building["region"],
            carve_needing=building["carve_needing"], op_count_bucket=building["bucket"],
            n_ops=(len(ops) if ops is not None else None),
        ))
        if ops is None:
            continue
        rel_trace_dir = str(trace_dirs[building["id"]].relative_to(REPO)) \
            if building["id"] in trace_dirs else None
        composites = composite_paths.get(building["id"], [])
        for i, op in enumerate(ops):
            rel_composite = str(composites[i].relative_to(REPO)) if i < len(composites) else None
            doc_id = f"{building['id']}_{op.id}"
            operations.append(dict(
                building_id=building["id"], operation_id=op.id, operation_index=i,
                kind=op.kind, mode=op.mode, region=building["region"],
                carve_needing=building["carve_needing"], op_count_bucket=building["bucket"],
                trace_dir=rel_trace_dir, composite_trace_path=rel_composite,
                ai_suggestion=ai_suggestions.get(doc_id),
                annotator_1=_empty_annotation(), annotator_2=_empty_annotation(),
                adjudication=dict(label=None, valid_labels=None, note=None, adjudicated_at=None, by=None),
            ))
    return dict(schema_version=SCHEMA_VERSION, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               labels=list(LABELS), sample_n=len(sample), sample_composition=composition,
               sample_buildings=sample_buildings, operations=operations)


# ------------------------------------------------------------------------------------------------
# 5. inter-annotator agreement -- reported honestly, never silently averaged away
# ------------------------------------------------------------------------------------------------

def cohens_kappa(a: Sequence[Optional[str]], b: Sequence[Optional[str]]) -> Optional[float]:
    """Standard unweighted Cohen's kappa over paired categorical labels. `None` (not computable)
    when there are too few paired observations or every label agrees on one constant category
    (kappa's own denominator is then 0/0 -- reported as `None`, never silently coerced to 1.0)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    mat = np.zeros((k, k))
    for x, y in pairs:
        mat[idx[x], idx[y]] += 1
    po = float(np.trace(mat)) / n
    row_marg = mat.sum(axis=1) / n
    col_marg = mat.sum(axis=0) / n
    pe = float(np.dot(row_marg, col_marg))
    if abs(1.0 - pe) < 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


def compute_agreement(schema: dict) -> dict:
    """Percent agreement + Cohen's kappa over operations BOTH annotators have labeled, plus the
    disagreement list itself (#154's own decision: adjudicated by the ticket owner, never silently
    averaged away)."""
    ops = schema["operations"]
    labeled = [o for o in ops if o["annotator_1"]["label"] is not None
              and o["annotator_2"]["label"] is not None]
    a1 = [o["annotator_1"]["label"] for o in labeled]
    a2 = [o["annotator_2"]["label"] for o in labeled]
    n_agree = sum(1 for x, y in zip(a1, a2) if x == y)
    disagreements = [dict(building_id=o["building_id"], operation_id=o["operation_id"],
                          annotator_1=o["annotator_1"]["label"], annotator_2=o["annotator_2"]["label"])
                     for o in labeled if o["annotator_1"]["label"] != o["annotator_2"]["label"]]
    return dict(n_total_operations=len(ops), n_labeled_by_both=len(labeled),
               percent_agreement=(n_agree / len(labeled) if labeled else None),
               cohens_kappa=cohens_kappa(a1, a2), disagreements=disagreements)


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recovery_out", default=str(ARTIFACTS / "void_semantic_recovery_153.json"))
    ap.add_argument("--schema_out", default=str(ARTIFACTS / "void_semantic_annotations.json"))
    ap.add_argument("--trace_dir", default=str(TRACES_DIR))
    ap.add_argument("--sample_n", type=int, default=SAMPLE_N)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--skip_recovery", action="store_true",
                    help="reuse --recovery_out from a previous run instead of re-running #10's fitter")
    ap.add_argument("--ai_suggestions", default=None,
                    help="optional path to a {doc_id: {label, confidence, reasoning}} JSON file; "
                         "embedded into each operation as a review hint, never as a ground-truth "
                         "annotation (see build_annotation_schema's own docstring)")
    args = ap.parse_args()

    ids = materialize_test_ids()
    ids_path = ARTIFACTS / "void_semantic_test_ids.json"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ids=[int(i) for i in ids]), open(ids_path, "w"))
    print(f"[ids] {len(ids)} test ids from #153's split", flush=True)

    if not args.skip_recovery:
        run_recovery_cli(ids_path, Path(args.recovery_out))
    candidates = load_candidates(Path(args.recovery_out))
    print(f"[candidates] {len(candidates)} recovered buildings available for sampling", flush=True)

    sample, composition = stratified_sample(candidates, n=args.sample_n, seed=args.seed)
    print(f"[sample] {len(sample)} buildings  composition={composition}", flush=True)

    trace_dirs = {}
    composite_paths = {}
    with open_real_corpus(H5) as f:
        for building in sample:
            gt = np.asarray(f["sdf"][building["id"]], np.float32) <= 0
            fp = np.asarray(f["footprint"][building["id"]]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, _target = hf
            ops = build_operations(fp, y0, y1, building["program"])
            building["ops"] = ops
            out_dir = Path(args.trace_dir) / str(building["id"])
            _step_paths, comp_paths = render_building_trace(fp, y0, y1, ops, out_dir)
            trace_dirs[building["id"]] = out_dir
            composite_paths[building["id"]] = comp_paths

    ai_suggestions = json.load(open(args.ai_suggestions)) if args.ai_suggestions else None
    schema = build_annotation_schema(sample, composition, trace_dirs, composite_paths, ai_suggestions)
    Path(args.schema_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(schema, open(args.schema_out, "w"), indent=1)
    print(f"[artifact] {args.schema_out}  ({len(schema['operations'])} operations to annotate)",
         flush=True)


if __name__ == "__main__":
    main()
