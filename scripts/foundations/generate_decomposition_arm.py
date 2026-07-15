"""Ticket 12: generate the full-data C2 decomposition arm's held-out outputs.

The C2 comparison (CONTEXT.md) is monolithic detail-*generation* (ticket 11's `monolith_v3`)
vs. detail-*composition*: retrieval + learned placement + procedural instantiation from real
architectural elements. This script builds the composition side for every building in the
sealed `data/splits_v1/test.json` (277 ids) and verifies its output is resolution/format
-compatible with the neutral evaluation harness (`scripts/eval/{render_facades,fid}.py`). The
full head-to-head FID/IoU comparison against the monolith is ticket 13's ("Decide the Full-Data
C2 Kill-Gate") job, not this one -- this ticket's own Question asks to "verify massing/detail
assembly, output resolution, failure accounting, and exact compatibility with the neutral
evaluation harness," not to render the verdict.

Pipeline per building:
  1. MASSING: Stage3a SDEdit from a footprint-extrude blockout, strength=0.5 -- byte-identical
     contract to ticket 09's C1a generation arm (`transform_vs_noise.build_condition` +
     `model.sdedit`), reused rather than re-derived. Native 64^3.
  2. DETAIL LAYOUT: `propose_detail_ops` (the trained part-layout planner) types and places
     window/door/chimney/dome/tower/balcony/column ops on the massing surface. Procedural by
     construction (CONTEXT.md: "Composition" = retrieval + LEARNED PLACEMENT + procedural
     instantiation -- placement is this step, independent of whether a given op then gets
     upgraded to a retrieved element).
  3. RETRIEVAL UPGRADE: `tower`/`dome`/`chimney`/`balcony`/`column` ops -- every ADD type
     `propose_detail_ops` can emit -- are attempted against `element_library_train100_v1`
     (ticket 08's leakage-safe library, redirected in via `use_train100_library()` -- never the
     live `element_library_v1`, which has no leakage manifest at all). Originally scoped to just
     tower/dome/chimney (2026-07-13): balcony/column had dead-or-borderline pools under a single
     global `MIN_SOLIDITY=0.12`. Fixed 2026-07-14: that threshold systematically starved
     architecturally thin types (visual QA confirmed the newly-unlocked low-solidity crops are
     legitimate thin architecture, not the skeletal fragments the filter exists to exclude), so
     `element_fit.py` now uses a per-type threshold table -- balcony/column now clear `MIN_POOL=8`
     comfortably (65/85 elements) and are included here too. `balcony_upper`/`stairs` also gained
     usable pools but are NOT retrieval targets here: `propose_detail_ops` never emits an op for
     either (explicitly skipped -- "massing already has a roof"), so there would be nothing to
     upgrade. A retrieval attempt that comes back empty (aspect/scale filtering drops the pool
     below `MIN_POOL=8` even for a nonempty type) falls back to the original procedural op, never
     a hard failure.
  4. COMPOSE: `EditableBuilding` CSG-unions the (possibly-upgraded) ops onto the massing SDF,
     sampled to `WORKING_RES=96` (ADR 0004's locked shared resolution -- ticket 05/09/10/11 all
     render/compare at this res, never mixed).

Population: the full 277-id `test.json`, NOT restricted to the 27-building Stage3a-clean tier
(project-owner decision, 2026-07-13) -- bigger sample, but the massing step uses the PRETRAINED
Stage3a prior, which already trained on 224/277 of these ids (ticket 09's own leakage finding).
Every row is tagged with its leakage tier (clean/val_leak/train_leak/unknown via
`transform_vs_noise.classify_leakage`) so a downstream comparison can disclose, not hide, the
asymmetry -- the monolith trained fresh on `train_100` only, so all 277 are genuinely clean for
IT; only the decomposition arm's massing half carries this leakage.

Out: data/decomposition_arm_v1/{manifest.json, grids/<id>.npy},
     outputs/decomposition_arm_v1/montage.png
Run:  TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        ./sdfusion/bin/python scripts/foundations/generate_decomposition_arm.py [--limit N]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations", "scripts/server"):
    sys.path.insert(0, str(REPO / _p))

import render_facades as rf  # noqa: E402
from eval_harness import iou, fp_iou  # noqa: E402
from make_splits import parse_class  # noqa: E402
from transform_vs_noise import held_out_population, build_condition, git_provenance  # noqa: E402

STRENGTH = 0.5           # ticket 09's canonical SDEdit strength for the C1 generation contract
WORKING_RES = 96         # ADR 0004's locked shared resolution
# `propose_detail_ops`'s own default (14) is tuned for interactive per-click UI responsiveness,
# not evaluation diversity -- empirically (project-owner decision, 2026-07-13, after finding
# 11/11 sampled buildings got ZERO tower/dome/chimney ops at the default) window/door ops fill
# that budget before the planner's rarer add-type predictions ever get a chance. Raised for
# this research-batch generation only; no building observed while picking this value produced
# more than ~19 total ops even with room for far more, so this is comfortably non-truncating,
# not just "bigger."
MAX_DETAIL_OPS = 40
TRAIN100_LIB = REPO / "data/element_library_train100_v1"

# Every ADD-type `propose_detail_ops` can actually emit (window/door are always procedural by
# design; `roof`/`stairs`/`balcony_upper` are explicitly skipped inside `propose_detail_ops`
# itself -- "massing already has a roof" -- so no op with those `det` values is ever produced,
# independent of library pool size). Originally scoped to tower/dome/chimney only (2026-07-13):
# balcony/balcony_upper/stairs/column had dead-or-borderline pools in the leakage-safe library
# at the old global MIN_SOLIDITY=0.12. Extended to balcony/column (2026-07-14) after fixing that
# threshold to be per-type (element_fit.py's MIN_SOLIDITY_BY_TYPE) -- balcony_upper/stairs are
# NOT added despite now having usable pools, since propose_detail_ops can never emit ops for them.
RETRIEVAL_POOLS = {
    "tower": ("tower",),
    "dome": ("dome",),
    "chimney": ("chimney", "roof_structure"),
    "balcony": ("balcony",),
    "column": ("column",),
}


def pools_for_type(det_type):
    return RETRIEVAL_POOLS.get(det_type)


def op_half_extent(op):
    """3-vector half-extent for retrieval-aspect scoring, regardless of primitive kind --
    `propose_detail_ops` emits `dome` as a `sphere` (`size=[radius]`) and `column` as a
    `cylinder` (`size=[radius, height]`), neither of which is a 3-element box half-extent on
    its own, so both get a natural bounding-box stand-in (real dome/column elements are
    themselves roughly axis-symmetric in the plane orthogonal to their long axis)."""
    if op["kind"] == "sphere":
        r = float(op["size"][0])
        return [r, r, r]
    if op["kind"] == "cylinder":
        r, h = float(op["size"][0]), float(op["size"][1])
        return [r, h / 2, r]
    size = op["size"]
    return [float(size[0]), float(size[1]), float(size[2])]


def y_extent_from_occupancy(occ):
    """(y_ground, y_top) in the SAME `linspace(-1,1,R)` convention `layout_detail.py`'s own
    `_occ_frame` uses -- `propose_detail_ops`'s op centers already live in this frame, so this
    is what makes an op's `center[1]` and a `y_frac` comparable."""
    occ = np.asarray(occ)
    R = occ.shape[0]
    g = np.linspace(-1, 1, R)
    hi = np.where(occ.any((0, 2)))[0]
    if not len(hi):
        raise ValueError("empty massing")
    return float(g[hi.min()]), float(g[hi.max()])


def retrieval_params(op, y_ground, y_top):
    """(aspect, y_frac, box_rel_y) for `element_fit.retrieve`, adapted from
    `interpret_mass`'s single-op derivation (`layout_detail.py`) to an already-placed,
    already-typed op from `propose_detail_ops`'s output (no "visible span above roofline"
    adjustment -- that heuristic exists for crude user-drawn boxes; the planner's ops are
    already reasonably proportioned). Returns None if the massing has no measurable y-extent
    for this op to be scored against (degenerate/near-empty massing)."""
    y_span = y_top - y_ground
    if y_span <= 1e-6:
        return None
    half = op_half_extent(op)
    aspect = (half[0] / max(half[1], 1e-6), half[2] / max(half[1], 1e-6))
    y_frac = (op["center"][1] - y_ground) / y_span
    box_rel_y = 2.0 * half[1] / y_span
    return dict(aspect=aspect, y_frac=y_frac, box_rel_y=box_rel_y)


def use_train100_library():
    """Redirect `element_lib`/`element_fit` at the leakage-safe train100 library and drop
    every cache that would otherwise still hold data from the live production library --
    the exact pattern `scripts/server/test_element_retrieval_baseline.py` already
    establishes, reused rather than re-derived. Script-scoped (no restore): this process
    never needs the production library."""
    from scene import element_lib
    import element_fit as ef
    element_lib.LIB = TRAIN100_LIB
    element_lib._meta = None
    element_lib._crops = None
    element_lib._cache.clear()
    ef._F = None


def upgrade_ops_with_retrieval(ops, y_ground, y_top, building_class, seed):
    """Upgrades `tower`/`dome`/`chimney` `add` ops to retrieved-element ops where the
    leakage-safe library has a usable pool at that placement; every other op -- including a
    retrieval attempt that comes back empty -- passes through unchanged (procedural). Returns
    (final_ops, stats)."""
    import element_fit as ef
    final_ops = []
    stats = dict(n_retrieved=0, n_fallback=0, n_procedural=0, retrieved_elements=[])
    rng = np.random.default_rng(seed)
    for op in ops:
        pools = pools_for_type(op.get("det")) if op.get("mode") == "add" else None
        params = retrieval_params(op, y_ground, y_top) if pools else None
        if pools and params:
            lid, row = ef.retrieve(pools, params["aspect"], params["y_frac"], building_class,
                                   seed=int(rng.integers(1 << 31)), box_rel_y=params["box_rel_y"])
            if lid is not None:
                final_ops.append(ef.element_op(lid, op["center"], op_half_extent(op), det=op["det"]))
                stats["n_retrieved"] += 1
                # traceability (tickets.md's own bar: "every output traces to allowed massing
                # and element sources") -- which real element, from which source building, was
                # actually composed into this output, not just an aggregate count.
                stats["retrieved_elements"].append(dict(
                    lib_id=lid, det_type=op["det"], element_type=row["type"],
                    source_building=row["building"]))
                continue
            stats["n_fallback"] += 1
        else:
            stats["n_procedural"] += 1
        final_ops.append(op)
    return final_ops, stats


def generate_massing(model, building_id, subtype_to_idx, device):
    """Stage3a SDEdit massing from a footprint-extrude blockout -- ticket 09's own C1
    generation contract, reused exactly. Returns (massing_grid_64 numpy, real_occ, real_sdf)
    for paired massing-fidelity scoring against the real target (CONTEXT.md: massing fidelity
    is measured paired, since massing is determined by the footprint)."""
    data, real_occ, real_sdf = build_condition(building_id, subtype_to_idx, device)
    out = model.sdedit(data, strength=STRENGTH, max_sample=1, guide_model=None)
    massing = out[0, 0].detach().cpu().numpy().astype(np.float32)
    return massing, real_occ, real_sdf


def compose_decomposition(massing_grid, building_class, device, seed):
    """massing (64^3 cube-frame SDF) -> typed detail layout -> retrieval-upgraded ops ->
    CSG-composed, WORKING_RES-sampled SDF grid. Returns (composed_grid_96 numpy, ops_summary)."""
    import torch
    from layout_detail import propose_detail_ops
    from refine import volume_to_sdf
    from scene.sdf_edit import EditableBuilding, EditOp
    from scene.sdf_primitives import sample_grid

    ops = propose_detail_ops(massing_grid, building_class, device=device, seed=seed,
                             max_ops=MAX_DETAIL_OPS)
    y_ground, y_top = y_extent_from_occupancy(np.asarray(massing_grid) <= 0)
    final_ops, stats = upgrade_ops_with_retrieval(ops, y_ground, y_top, building_class, seed)
    stats["n_ops_total"] = len(final_ops)

    base = volume_to_sdf(massing_grid, device)
    composed_fn = EditableBuilding(base, [EditOp.from_dict(o) for o in final_ops]).composed()
    with torch.no_grad():
        composed = sample_grid(composed_fn, WORKING_RES, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                               device=device)
    return composed.detach().cpu().numpy().astype(np.float32), stats


def _montage(rows, out_path: Path, cell=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    n_rows, n_cols = len(rows), len(rows[0][1]) if rows else 0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell / 60 * n_cols, cell / 60 * n_rows),
                             subplot_kw={"projection": "3d"}, squeeze=False)
    for ri, (row_label, cells) in enumerate(rows):
        for ci, (title, sdf) in enumerate(cells):
            ax = axes[ri][ci]
            ax.set_axis_off()
            if sdf is not None and (sdf <= 0).sum() > 8:
                try:
                    v, f, *_ = measure.marching_cubes(sdf, 0.0)
                    ax.plot_trisurf(v[:, 2], v[:, 0], f, v[:, 1], color="#c9b790",
                                    edgecolor="none", shade=True)
                    ax.set_xlim(0, sdf.shape[2]); ax.set_ylim(0, sdf.shape[0]); ax.set_zlim(0, sdf.shape[1])
                except Exception:
                    pass
            ax.view_init(elev=14, azim=-60)
            ax.set_title(f"{row_label}\n{title}" if ci == 0 else title, fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N buildings")
    ap.add_argument("--montage-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "data/decomposition_arm_v1"))
    ap.add_argument("--montage-out",
                    default=str(REPO / "outputs/decomposition_arm_v1/montage.png"))
    a = ap.parse_args()

    import torch
    from refine import Refiner
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_train100_library()
    print("[*] element_lib redirected to data/element_library_train100_v1")

    tiers, subtype_to_idx = held_out_population()
    ids = [(bid, tier) for tier, tbids in tiers.items() for bid in tbids]
    ids.sort(key=lambda x: x[0])
    if a.limit:
        ids = ids[: a.limit]
    tier_counts = {t: len(tiers[t]) for t in tiers}
    print(f"[*] {len(ids)} buildings from data/splits_v1/test.json -- leakage tiers: {tier_counts}")

    print("[*] loading the deployed live prior (Stage3a SDEdit massing)...")
    refiner = Refiner(SimpleNamespace(device=device))
    model, _ = refiner._load_sdedit(autoguidance=False)

    grids_dir = Path(a.out_dir) / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    rows, montage_rows, failures = [], [], []
    for i, (bid, tier) in enumerate(ids):
        try:
            massing, real_occ, real_sdf = generate_massing(model, bid, subtype_to_idx, device)
            building_class = parse_class(bid)
            composed, stats = compose_decomposition(massing, building_class, device, a.seed + i)
        except Exception as ex:  # noqa: BLE001
            failures.append(dict(building=bid, tier=tier, error=f"{type(ex).__name__}: {str(ex)[:160]}"))
            print(f"  [{i+1}/{len(ids)}] {bid} FAILED: {failures[-1]['error']}", flush=True)
            continue

        massing_occ = massing <= 0
        m_iou, m_fp_iou = iou(massing_occ, real_occ), fp_iou(massing_occ, real_occ)
        np.save(grids_dir / f"{bid}.npy", composed.astype(np.float16))
        rows.append(dict(building=bid, tier=tier, building_class=building_class,
                         massing_iou=m_iou, massing_fp_iou=m_fp_iou, **stats))
        print(f"  [{i+1}/{len(ids)}] {bid:38s} tier={tier:10s} massing_iou={m_iou:.3f} "
              f"ops={stats['n_ops_total']} retrieved={stats['n_retrieved']} "
              f"fallback={stats['n_fallback']}", flush=True)

        if len(montage_rows) < a.montage_n:
            real96 = rf.resample_sdf_grid(real_sdf, WORKING_RES, device=device) \
                if real_sdf.shape[0] != WORKING_RES else real_sdf
            massing96 = rf.resample_sdf_grid(massing, WORKING_RES, device=device)
            montage_rows.append((f"{bid}\n{tier}", [("real", real96), ("massing", massing96),
                                                    ("decomposition", composed)]))

    if montage_rows:
        _montage(montage_rows, Path(a.montage_out))
        print(f"[save] {a.montage_out}")

    manifest = dict(
        strength=STRENGTH, working_res=WORKING_RES, retrieval_pools=RETRIEVAL_POOLS,
        train100_library=str(TRAIN100_LIB),
        leakage_tier_counts=tier_counts, n_succeeded=len(rows), n_failed=len(failures),
        failures=failures, per_building=rows, grids_dir=str(grids_dir),
        montage=a.montage_out, **git_provenance(),
    )
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(Path(a.out_dir) / "manifest.json", "w"), indent=2)
    print(f"\n[done] {len(rows)} succeeded, {len(failures)} failed")
    print(f"[save] {Path(a.out_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
