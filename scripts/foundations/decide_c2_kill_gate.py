"""Ticket 13: the C2 kill-gate decision -- compare the full-data monolith (ticket 11,
`monolith_v3`) against the full-data decomposition arm (ticket 12) on the same held-out
`data/splits_v1/test.json` population (277 ids), through one shared neutral render+FID+IoU
harness, and apply the PRD's own preregistered decision rule.

PRD (`.scratch/transform-composition-proof/PRD.md`): "If the decomposition does not win on
detail fidelity while retaining comparable massing fidelity, stop the scaling curve and
diagnose the failure rather than retrofitting the hypothesis."

Metrics (CONTEXT.md's own split, applied identically to both arms' FINAL output so the
comparison is symmetric -- neither arm's number is computed on an intermediate step the other
arm doesn't have):
  - Massing fidelity (paired): footprint IoU (`fp_iou`, primary -- robust to facade-level
    noise since detail additions rarely change the top-down silhouette) and full-volume IoU
    (secondary) against the real target, both via `eval_harness.iou`/`fp_iou`.
  - Detail fidelity (distributional): neutral-facade FID via `render_facades.py`/`fid.py`,
    pooled across the population, bootstrap CI (group-aware, per-building).

KNOWN GAP, disclosed not hidden: the PRD names "paired Chamfer and IoU" for massing fidelity;
no Chamfer-distance implementation exists anywhere in this codebase (confirmed by exhaustive
grep before writing this script). Only IoU is reported -- the metric every other massing-
fidelity claim in this project (tickets 07/09/10/12) has also used exclusively.

Failure localization: the decomposition arm's own manifest (`data/decomposition_arm_v1/
manifest.json`) already carries per-building leakage tier, massing IoU, and retrieval/fallback/
procedural counts -- reused here, not recomputed, to separate massing-step failures from
detail-composition-step failures on that arm. The monolith side gets an equivalent per-building
IoU from `generate_monolith_arm.py`'s own manifest. Renderer-effect isolation has no precedent
anywhere in this codebase and is NOT attempted here (nothing to compare the renderer against).

CAVEATS CARRIED FORWARD FROM A 2026-07-13 CODEBASE AUDIT (disclosed per project-owner decision
to proceed with this comparison anyway rather than block on them): `monolith_v3`'s mean
occupancy matches real data but many individual outputs are near-empty/fragmentary; several
BuildingNet element labels feeding the decomposition arm's retrieval library were inferred
heuristically and may mix unrelated geometry into a claimed type; small-sample FID bootstrap
CIs in this exact pipeline have previously shown the point estimate falling outside its own CI
(ticket 10's own finding) -- checked for and reported here too, not assumed absent.

Out: execution/artifacts/c2_kill_gate_decision.json,
     outputs/c2_kill_gate/montage.png
Run:  TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        ./sdfusion/bin/python scripts/foundations/decide_c2_kill_gate.py [--limit N]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations"):
    sys.path.insert(0, str(REPO / _p))

import fid as fidmod  # noqa: E402
import render_facades as rf  # noqa: E402
from eval_harness import iou, fp_iou  # noqa: E402
from make_splits import parse_class  # noqa: E402
from transform_vs_noise import git_provenance  # noqa: E402

COMPARABLE_MASSING_TOLERANCE = 0.05  # a disclosed judgement call (this script's own choice,
                                     # not derived from the PRD, which only says "comparable"):
                                     # decomposition's mean footprint IoU may sit up to this
                                     # many points below the monolith's own and still count as
                                     # "retaining comparable massing fidelity."


def kill_gate_decision(detail_fid_decomp, detail_fid_monolith, massing_fp_iou_decomp,
                       massing_fp_iou_monolith, comparable_tolerance=COMPARABLE_MASSING_TOLERANCE):
    """PRD's own preregistered rule: the gate passes only if the decomposition arm wins on
    detail fidelity (lower FID) AND its massing fidelity -- footprint IoU specifically, this
    module's primary massing metric (see module docstring); full-volume IoU is reported
    alongside but is NOT what this rule reads -- doesn't drop more than `comparable_tolerance`
    below the monolith's own. Returns a plain dict, not a verdict to treat as final -- this is a
    mechanical application of the rule to point estimates; reading it against the reported
    uncertainty (CIs, undersampling) is left to the ticket answer."""
    wins_detail = detail_fid_decomp < detail_fid_monolith
    massing_gap = massing_fp_iou_decomp - massing_fp_iou_monolith
    comparable_massing = massing_gap >= -comparable_tolerance
    gate = "pass" if (wins_detail and comparable_massing) else "fail"
    return dict(wins_detail=wins_detail, comparable_massing=comparable_massing, gate=gate,
                detail_fid_gap=detail_fid_monolith - detail_fid_decomp, massing_iou_gap=massing_gap)


def bootstrap_mean_ci(values, n_boot=2000, seed=0, ci=0.95):
    """Percentile bootstrap CI for the mean of per-building point estimates -- massing IoU gets
    the same "small differences aren't definitive wins" scrutiny PRD story 19 asks for, matching
    the spirit of fid.py's bootstrap_fid_ci but over a plain per-building scalar rather than
    Inception features."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lo, hi = np.percentile(boot_means, [lo_pct, hi_pct])
    return float(arr.mean()), float(lo), float(hi)


def localize_decomposition_failures(rows):
    """Break decomposition-arm fidelity down by leakage tier and by retrieval activity --
    reuses ticket 12's own per-building diagnostics (never recomputed) to separate a massing-
    step failure (implicates the pretrained Stage3a prior, worse on leaked tiers) from a
    detail-composition-step failure (implicates retrieval, worse when few elements retrieved).

    `composition_iou_drop` = ticket 12's own base-massing-only IoU minus this ticket's final
    composed-shape IoU (both paired against the same real target) -- directly isolates how much
    the compose step (adding retrieved/procedural elements onto the massing) costs, rather than
    only comparing final IoU across buckets, which conflates the two steps."""
    by_tier: dict = {}
    for r in rows:
        by_tier.setdefault(r["decomposition_tier"], []).append(r["decomposition_iou"])
    tier_means = {t: float(np.mean(v)) for t, v in by_tier.items()}

    retrieved = [r["decomposition_iou"] for r in rows if r["decomposition_n_retrieved"] > 0]
    no_retrieval = [r["decomposition_iou"] for r in rows if r["decomposition_n_retrieved"] == 0]

    drops = [r["decomposition_massing_iou"] - r["decomposition_iou"] for r in rows]
    drops_retrieved = [r["decomposition_massing_iou"] - r["decomposition_iou"]
                       for r in rows if r["decomposition_n_retrieved"] > 0]
    drops_no_retrieval = [r["decomposition_massing_iou"] - r["decomposition_iou"]
                          for r in rows if r["decomposition_n_retrieved"] == 0]
    return dict(
        mean_iou_by_leakage_tier=tier_means,
        mean_iou_with_retrieval=float(np.mean(retrieved)) if retrieved else None,
        mean_iou_without_retrieval=float(np.mean(no_retrieval)) if no_retrieval else None,
        n_with_retrieval=len(retrieved), n_without_retrieval=len(no_retrieval),
        mean_composition_iou_drop=float(np.mean(drops)) if drops else None,
        mean_composition_iou_drop_with_retrieval=float(np.mean(drops_retrieved)) if drops_retrieved else None,
        mean_composition_iou_drop_without_retrieval=float(np.mean(drops_no_retrieval)) if drops_no_retrieval else None,
    )


def localize_monolith_failures(rows):
    """Break monolith-arm fidelity down by building class and by whether the generated output
    landed in the near-empty regime the 2026-07-13 audit flagged (gen_occ_frac < 0.01%, an
    order of magnitude below typical BuildingNet occupancy) -- distinguishes "the model
    collapsed to near-nothing" from "the model produced real but inaccurate geometry.\""""
    by_class: dict = {}
    for r in rows:
        by_class.setdefault(r["building_class"], []).append(r["monolith_iou"])
    class_means = {c: float(np.mean(v)) for c, v in by_class.items()}

    near_empty = [r["monolith_iou"] for r in rows if r["monolith_gen_occ_frac"] < 1e-4]
    not_empty = [r["monolith_iou"] for r in rows if r["monolith_gen_occ_frac"] >= 1e-4]
    return dict(
        mean_iou_by_class=class_means,
        n_near_empty_generations=len(near_empty), n_non_empty_generations=len(not_empty),
        mean_iou_near_empty=float(np.mean(near_empty)) if near_empty else None,
        mean_iou_non_empty=float(np.mean(not_empty)) if not_empty else None,
    )


def _montage(rows, out_path: Path, cell=220):
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
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--img-res", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--montage-n", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/c2_kill_gate_decision.json"))
    ap.add_argument("--montage-out", default=str(REPO / "outputs/c2_kill_gate/montage.png"))
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    monolith_manifest = json.load(open(REPO / "data/monolith_arm_v1/manifest.json"))
    decomp_manifest = json.load(open(REPO / "data/decomposition_arm_v1/manifest.json"))
    monolith_by_id = {r["building"]: r for r in monolith_manifest["per_building"]}
    decomp_by_id = {r["building"]: r for r in decomp_manifest["per_building"]}

    # Only buildings BOTH arms actually produced an output for -- a disclosed intersection, not
    # silently padding a failed arm's population with the other's successes.
    ids = sorted(set(monolith_by_id) & set(decomp_by_id))
    monolith_only_failed = sorted(set(monolith_by_id) - set(decomp_by_id))
    decomp_only_failed = sorted(set(decomp_by_id) - set(monolith_by_id))
    if a.limit:
        ids = ids[: a.limit]
    print(f"[*] {len(ids)} buildings with output from both arms "
          f"({len(monolith_only_failed)} monolith-only failures excluded, "
          f"{len(decomp_only_failed)} decomposition-only failures excluded)")

    monolith_grids_dir = Path(monolith_manifest["grids_dir"])
    decomp_grids_dir = Path(decomp_manifest["grids_dir"])
    cams = rf.orbit_cameras(n_views=a.views)
    ext = fidmod.InceptionExtractor(device=device)

    rows, montage_rows, failures = [], [], []
    real_imgs, monolith_imgs, decomp_imgs = [], [], []
    real_groups, monolith_groups, decomp_groups = [], [], []
    for gi, bid in enumerate(ids):
        try:
            real96 = rf.load_buildingnet_sdf(bid, working_res=rf.WORKING_RES, device=device)
            mono_grid = np.load(monolith_grids_dir / f"{bid}.npy").astype(np.float32)
            decomp_grid = np.load(decomp_grids_dir / f"{bid}.npy").astype(np.float32)

            real_occ = real96 <= 0
            mono_occ, decomp_occ = mono_grid <= 0, decomp_grid <= 0
            mono_iou_v, mono_fp_v = iou(mono_occ, real_occ), fp_iou(mono_occ, real_occ)
            decomp_iou_v, decomp_fp_v = iou(decomp_occ, real_occ), fp_iou(decomp_occ, real_occ)

            real_v = rf.render_sdf_neutral(real96, cameras=cams, res=a.img_res, device=device)
            mono_v = rf.render_sdf_neutral(mono_grid, cameras=cams, res=a.img_res, device=device)
            decomp_v = rf.render_sdf_neutral(decomp_grid, cameras=cams, res=a.img_res, device=device)
        except Exception as ex:  # noqa: BLE001
            failures.append(dict(building=bid, error=f"{type(ex).__name__}: {str(ex)[:160]}"))
            print(f"  [{gi+1}/{len(ids)}] {bid} FAILED: {failures[-1]['error']}", flush=True)
            continue

        real_imgs.extend(real_v); real_groups.extend([gi] * len(real_v))
        monolith_imgs.extend(mono_v); monolith_groups.extend([gi] * len(mono_v))
        decomp_imgs.extend(decomp_v); decomp_groups.extend([gi] * len(decomp_v))

        drow = decomp_by_id[bid]
        rows.append(dict(
            building=bid, building_class=parse_class(bid),
            monolith_iou=mono_iou_v, monolith_fp_iou=mono_fp_v,
            monolith_gen_occ_frac=monolith_by_id[bid]["gen_occ_frac"],
            decomposition_iou=decomp_iou_v, decomposition_fp_iou=decomp_fp_v,
            decomposition_massing_iou=drow["massing_iou"], decomposition_tier=drow["tier"],
            decomposition_n_retrieved=drow["n_retrieved"], decomposition_n_fallback=drow["n_fallback"],
        ))
        print(f"  [{gi+1}/{len(ids)}] {bid:38s} monolith_iou={mono_iou_v:.3f} "
              f"decomposition_iou={decomp_iou_v:.3f}", flush=True)

        if len(montage_rows) < a.montage_n:
            montage_rows.append((bid, [("real", real96), ("monolith", mono_grid),
                                       ("decomposition", decomp_grid)]))

    if montage_rows:
        _montage(montage_rows, Path(a.montage_out))
        print(f"[save] {a.montage_out}")

    real_feat = ext.features(np.stack(real_imgs))
    mono_feat = ext.features(np.stack(monolith_imgs))
    decomp_feat = ext.features(np.stack(decomp_imgs))

    fid_mono_pt, fid_mono_lo, fid_mono_hi = fidmod.bootstrap_fid_ci(
        mono_feat, real_feat, n_boot=a.n_boot, seed=a.seed,
        groups_a=np.asarray(monolith_groups), groups_b=np.asarray(real_groups))
    fid_decomp_pt, fid_decomp_lo, fid_decomp_hi = fidmod.bootstrap_fid_ci(
        decomp_feat, real_feat, n_boot=a.n_boot, seed=a.seed,
        groups_a=np.asarray(decomp_groups), groups_b=np.asarray(real_groups))
    mono_undersampled = bool(fidmod.undersampled(mono_feat, real_feat))
    decomp_undersampled = bool(fidmod.undersampled(decomp_feat, real_feat))
    # Ticket 10's own finding: clearing the raw N>2048 threshold doesn't guarantee a
    # well-behaved bootstrap -- check explicitly rather than assume it's fine here too.
    mono_point_outside_ci = not (fid_mono_lo <= fid_mono_pt <= fid_mono_hi)
    decomp_point_outside_ci = not (fid_decomp_lo <= fid_decomp_pt <= fid_decomp_hi)
    print(f"[fid] monolith: point={fid_mono_pt:.1f} ci95=[{fid_mono_lo:.1f},{fid_mono_hi:.1f}] "
          f"undersampled={mono_undersampled} point_outside_ci={mono_point_outside_ci}")
    print(f"[fid] decomposition: point={fid_decomp_pt:.1f} ci95=[{fid_decomp_lo:.1f},{fid_decomp_hi:.1f}] "
          f"undersampled={decomp_undersampled} point_outside_ci={decomp_point_outside_ci}")

    mono_ious = [r["monolith_iou"] for r in rows]
    mono_fps = [r["monolith_fp_iou"] for r in rows]
    decomp_ious = [r["decomposition_iou"] for r in rows]
    decomp_fps = [r["decomposition_fp_iou"] for r in rows]

    # PRD story 19: uncertainty on paired metrics too, not just FID -- a percentile bootstrap
    # over per-building IoU, same seed as the FID bootstrap for reproducibility.
    mono_fp_pt, mono_fp_lo, mono_fp_hi = bootstrap_mean_ci(mono_fps, seed=a.seed)
    decomp_fp_pt, decomp_fp_lo, decomp_fp_hi = bootstrap_mean_ci(decomp_fps, seed=a.seed)
    mono_full_pt, mono_full_lo, mono_full_hi = bootstrap_mean_ci(mono_ious, seed=a.seed)
    decomp_full_pt, decomp_full_lo, decomp_full_hi = bootstrap_mean_ci(decomp_ious, seed=a.seed)

    decision = kill_gate_decision(
        detail_fid_decomp=fid_decomp_pt, detail_fid_monolith=fid_mono_pt,
        massing_fp_iou_decomp=decomp_fp_pt, massing_fp_iou_monolith=mono_fp_pt)
    print(f"[decision] {decision}")

    manifest = dict(
        n_buildings=len(rows), n_failed=len(failures), failures=failures,
        monolith_only_failures=monolith_only_failed, decomposition_only_failures=decomp_only_failed,
        per_building=rows,
        input_provenance=dict(
            monolith_arm={k: monolith_manifest.get(k) for k in
                          ("git_rev", "dirty_digest", "checkpoint", "checkpoint_step", "ddim_steps")},
            decomposition_arm={k: decomp_manifest.get(k) for k in
                               ("git_rev", "dirty_digest")},
        ),
        massing=dict(
            footprint_iou=dict(
                monolith=dict(mean=mono_fp_pt, median=float(np.median(mono_fps)),
                              ci95=[mono_fp_lo, mono_fp_hi]),
                decomposition=dict(mean=decomp_fp_pt, median=float(np.median(decomp_fps)),
                                   ci95=[decomp_fp_lo, decomp_fp_hi])),
            full_iou=dict(
                monolith=dict(mean=mono_full_pt, median=float(np.median(mono_ious)),
                              ci95=[mono_full_lo, mono_full_hi]),
                decomposition=dict(mean=decomp_full_pt, median=float(np.median(decomp_ious)),
                                   ci95=[decomp_full_lo, decomp_full_hi])),
            chamfer="NOT COMPUTED -- no Chamfer-distance implementation exists anywhere in this "
                    "codebase (confirmed by exhaustive grep); PRD names it alongside IoU, IoU "
                    "only is reported, matching every other massing-fidelity claim in this "
                    "project (tickets 07/09/10/12).",
        ),
        detail_fid=dict(
            monolith=dict(point=fid_mono_pt, ci95=[fid_mono_lo, fid_mono_hi],
                          undersampled=mono_undersampled, point_outside_ci=mono_point_outside_ci,
                          n_images=len(monolith_imgs)),
            decomposition=dict(point=fid_decomp_pt, ci95=[fid_decomp_lo, fid_decomp_hi],
                              undersampled=decomp_undersampled, point_outside_ci=decomp_point_outside_ci,
                              n_images=len(decomp_imgs)),
            n_real_images=len(real_imgs),
        ),
        decision=decision, comparable_massing_tolerance=COMPARABLE_MASSING_TOLERANCE,
        failure_localization=dict(
            decomposition=localize_decomposition_failures(rows),
            monolith=localize_monolith_failures(rows)),
        cameras=dict(n_views=a.views), image_res=a.img_res, sdf_working_res=rf.WORKING_RES,
        montage=a.montage_out if montage_rows else None,
        **git_provenance(),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(a.out, "w"), indent=2)
    print(f"\n[done] gate={decision['gate']}  {len(rows)} buildings, {len(failures)} failures")
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
