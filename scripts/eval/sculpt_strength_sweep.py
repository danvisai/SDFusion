"""Ticket 10 (C1b): the sculpt-strength sweep -- edit faithfulness vs. neutral-render realism,
across representative crude edits, through the LIVE `/snap_sdf` operator.

Ticket 09 (C1a) already swept the GENERATION side of C1 (from-noise vs. footprint-blockout
SDEdit) at one fixed strength. This ticket sweeps the EDITING side: the same SDEdit transform,
applied to a user's crude placed primitive (add a tower/dome, subtract a carve -- byte-identical
to `scripts/foundations/eval_harness.py`'s own `EDITS`, this codebase's one canonical single-op
"representative sculpt edit" vocabulary; `scripts/server/sculpt_regression.py`'s `CASES` compose
the same tower/dome/carve primitives but pair its dome with an extra wing box in one compound
case, so it's the same style of session rather than an identical list), at every strength in
STRENGTHS. `Refiner.snap_volume` is called in-process rather than over HTTP -- it IS the
function `/snap_sdf` calls (`scripts/server/inference_service.py:snap_sdf`), so this is the same
code path without needing a separately-running server (matches tickets 07/09/11's in-process
convention).

Faithfulness = `iou_to_edit`, the IoU `snap_volume` already returns between its output and the
pre-snap edited input -- this project's established "did the snap keep what the user placed"
metric (`eval_harness.py`'s own `fp_iou`/`iou` convention). Realism = neutral-facade FID
(CONTEXT.md: detail fidelity is measured DISTRIBUTIONALLY) against a fixed real BuildingNet
reference population (ticket 09's held-out "clean" tier, reused rather than re-derived) --
rendered through ticket 05's shared neutral shader for representation parity.

This is a PROTOTYPE (wayfinder ticket type): small and cheap on purpose, to review with the
user, not a scaled headline result. At 3 edit cases x N_VIEWS renders per strength, the pooled
FID sample is far below ticket 05's own >=2048-d undersampling floor -- `fid.undersampled` WILL
fire; reported, not hidden (tickets 05/06/09's convention).

Out: execution/artifacts/sculpt_strength_sweep.json,
     outputs/sculpt_strength_sweep/{montage.png, faithfulness_vs_realism.png}
Run:  TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        ./sdfusion/bin/python scripts/eval/sculpt_strength_sweep.py [--n-real-ref N]
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

import fid as fidmod  # noqa: E402
import render_facades as rf  # noqa: E402
from transform_vs_noise import held_out_population, git_provenance  # noqa: E402

# The canonical crude-edit primitives (byte-identical to eval_harness.py's own EDITS -- this
# codebase's one established single-op "representative sculpt edit" vocabulary, reused rather
# than re-invented).
EDIT_CASES = [
    ("tower", dict(kind="box", center=[0.45, 0.25, 0.0], size=[0.14, 0.5, 0.14], mode="add", smooth=0.0)),
    ("dome", dict(kind="sphere", center=[0.0, 0.55, 0.0], size=[0.3], mode="add", smooth=0.3)),
    ("carve", dict(kind="box", center=[0.0, -0.2, 0.62], size=[0.18, 0.4, 0.2], mode="subtract", smooth=0.0)),
]
# Wider than eval_harness.py's fixed 3-point [0.3, 0.5, 0.7] regression-tracking convention --
# this ticket IS the dedicated full-range sweep, so it also samples the low/high extremes that
# convention doesn't need to revisit every checkpoint.
STRENGTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_VIEWS = 6          # matches ticket 05's sanity-run convention
N_REAL_REF = 8        # matches diagnose_massing_diversity.py's default footprint count


def summarize_by_strength(rows: list) -> list:
    """rows: one dict per (edit case, strength) sample, each with "strength" and
    "iou_to_edit" -- aggregates faithfulness across the fixed edit cases at each strength into
    one summary row per distinct strength, sorted ascending (the x-axis of the
    faithfulness-vs-realism plot)."""
    by_strength: dict = {}
    for row in rows:
        by_strength.setdefault(row["strength"], []).append(row["iou_to_edit"])
    out = []
    for strength in sorted(by_strength):
        ious = by_strength[strength]
        out.append(dict(
            strength=strength, n_cases=len(ious),
            mean_iou_to_edit=float(np.mean(ious)),
            min_iou_to_edit=float(min(ious)),
            max_iou_to_edit=float(max(ious)),
        ))
    return out


def build_base_grid(refiner, device):
    """A deterministic 64^3 procedural "modern" recipe building (eval_harness.py's own base) --
    no recipe-diffusion sampling, so bit-identical across runs; the edits are applied ON this
    fixed base, not on a re-sampled one."""
    from models.networks.diff_recipe import build_diff_recipe
    _, default_fn, _ = build_diff_recipe("modern")
    params = default_fn(device).detach().cpu().numpy()
    poly = np.array([[-7, -9], [7, -9], [7, 9], [-7, 9]], np.float32)
    grid, _c, _s, _hn = refiner.building_volume(poly, "modern", params, 16.0, margin=1.05)
    return grid


def composed_edit_grid(base_grid, edit, device):
    """The pre-snap edited shape (base + one EditOp, composed locally) -- the montage's "edited"
    reference column, matching `sculpt_regression.py`'s middle column."""
    import torch
    from refine import volume_to_sdf
    from scene.sdf_edit import EditableBuilding, EditOp
    R = int(base_grid.shape[0])
    base = volume_to_sdf(base_grid, device)
    edited = EditableBuilding(base, [EditOp.from_dict(edit)]).composed()
    g1 = torch.linspace(-1, 1, R, device=device)
    Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    with torch.no_grad():
        return edited(pts).reshape(R, R, R).cpu().numpy().astype(np.float32)


def real_reference_images(cams, n_ref, img_res, device):
    """Renders of `n_ref` real, held-out (Stage3a-clean) BuildingNet buildings -- the FIXED
    real-facade population every strength's FID is compared against (ADR 0002/0004:
    representation parity, one real reference set, not re-derived per strength)."""
    tiers, _ = held_out_population()
    ids = tiers["clean"][:n_ref]
    imgs = []
    for bid in ids:
        real96 = rf.load_buildingnet_sdf(bid, working_res=rf.WORKING_RES, device=device)
        imgs.append(rf.render_sdf_neutral(real96, cameras=cams, res=img_res, device=device))
    return ids, imgs


def _montage(rows, out_path: Path, cell=200):
    """rows: [(row_label, [(title, sdf_or_None), ...]), ...] -- one row per edit case."""
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


def _curve_plot(summary, fid_by_strength, out_path: Path):
    """Faithfulness (x) vs. realism (y, FID -- lower = more realistic) across STRENGTHS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [row["mean_iou_to_edit"] for row in summary]
    ys = [fid_by_strength[row["strength"]]["point"] for row in summary]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, "o-", color="#2b6cb0")
    n_total = max((row["n_cases"] for row in summary), default=0)
    for row, x, y in zip(summary, xs, ys):
        label = f"s={row['strength']}"
        if row["n_cases"] < n_total:      # a point built from fewer than the full case set
            label += f" ({row['n_cases']}/{n_total})"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("edit faithfulness (mean IoU to pre-snap edit)")
    ax.set_ylabel("realism (facade FID vs. real, lower = more realistic)")
    ax.set_title("C1b sculpt-strength sweep")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strengths", default=None,
                    help="comma-separated override of the default STRENGTHS list")
    ap.add_argument("--n-real-ref", type=int, default=N_REAL_REF)
    ap.add_argument("--views", type=int, default=N_VIEWS)
    ap.add_argument("--img-res", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/sculpt_strength_sweep.json"))
    ap.add_argument("--montage-out", default=str(REPO / "outputs/sculpt_strength_sweep/montage.png"))
    ap.add_argument("--curve-out",
                    default=str(REPO / "outputs/sculpt_strength_sweep/faithfulness_vs_realism.png"))
    a = ap.parse_args()
    strengths = [float(s) for s in a.strengths.split(",")] if a.strengths else STRENGTHS

    import torch
    from refine import Refiner
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[*] loading the deployed live prior (Refiner.snap_volume -- the exact /snap_sdf code path)...")
    refiner = Refiner(SimpleNamespace(device=device))
    base_grid = build_base_grid(refiner, device)

    cams = rf.orbit_cameras(n_views=a.views)
    print(f"[*] rendering {a.n_real_ref} real held-out reference buildings...")
    real_ids, real_imgs = real_reference_images(cams, a.n_real_ref, a.img_res, device)
    real_stack = np.stack([im for views in real_imgs for im in views])
    real_groups = np.repeat(np.arange(len(real_imgs)), [len(v) for v in real_imgs])
    ext = fidmod.InceptionExtractor(device=device)
    real_feat = ext.features(real_stack)

    rows, montage_rows, failures = [], [], []
    pooled_by_strength = {s: dict(imgs=[], groups=[]) for s in strengths}
    for case_i, (name, edit) in enumerate(EDIT_CASES):
        edited_grid = composed_edit_grid(base_grid, edit, device)
        cells = [("base", base_grid), ("edited (pre-snap)", edited_grid)]
        for s in strengths:
            try:
                # Fixed seed before EVERY call (same shape/device each time -> bit-identical
                # raw noise draw): strength is otherwise confounded with `sdedit`'s one
                # unseeded `torch.randn_like` noise draw (DDIM reverse steps are themselves
                # deterministic at ddim_eta=0.0) -- this isolates strength as the sole
                # controlled variable the ticket asks for.
                torch.manual_seed(a.seed)
                snapped, iou_to_edit = refiner.snap_volume(base_grid, [edit], strength=s)
                snapped96 = rf.resample_sdf_grid(snapped, rf.WORKING_RES, device=device)
                views = rf.render_sdf_neutral(snapped96, cameras=cams, res=a.img_res, device=device)
            except Exception as ex:  # noqa: BLE001
                failures.append(dict(case=name, strength=s, error=f"{type(ex).__name__}: {str(ex)[:120]}"))
                print(f"  {name:8s} s={s} FAILED: {failures[-1]['error']}", flush=True)
                cells.append((f"s={s}\nFAILED", None))
                continue
            rows.append(dict(case=name, strength=s, iou_to_edit=float(iou_to_edit)))
            print(f"  {name:8s} s={s}  iou_to_edit={iou_to_edit:.3f}", flush=True)
            pooled_by_strength[s]["imgs"].extend(views)
            pooled_by_strength[s]["groups"].extend([case_i] * len(views))
            cells.append((f"s={s}\niou={iou_to_edit:.2f}", snapped))
        montage_rows.append((name, cells))

    _montage(montage_rows, Path(a.montage_out))
    print(f"[save] {a.montage_out}")

    fid_by_strength = {}
    for s in strengths:
        pooled = pooled_by_strength[s]
        if not pooled["imgs"]:
            continue
        gen_feat = ext.features(np.stack(pooled["imgs"]))
        point, lo, hi = fidmod.bootstrap_fid_ci(gen_feat, real_feat, n_boot=a.n_boot, seed=a.seed,
                                                groups_a=np.asarray(pooled["groups"]),
                                                groups_b=real_groups)
        fid_by_strength[s] = dict(point=point, ci95=[lo, hi],
                                  undersampled=bool(fidmod.undersampled(gen_feat, real_feat)))
        print(f"[fid] s={s}: {fid_by_strength[s]}")

    summary = summarize_by_strength(rows)
    _curve_plot(summary, fid_by_strength, Path(a.curve_out))
    print(f"[save] {a.curve_out}")

    manifest = dict(
        strengths=strengths, edit_cases=[name for name, _ in EDIT_CASES],
        per_sample=rows, failures=failures, summary=summary,
        fid_by_strength={str(s): v for s, v in fid_by_strength.items()},
        real_reference=dict(n=len(real_ids), ids=real_ids),
        cameras=dict(n_views=a.views), image_res=a.img_res, sdf_working_res=rf.WORKING_RES,
        montage=a.montage_out, curve=a.curve_out,
        **git_provenance(),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(a.out, "w"), indent=2)
    print(f"\n[done] {len(rows)} samples, {len(failures)} failures")
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
