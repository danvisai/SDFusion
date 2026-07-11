"""Ticket 09 (C1a): Stage 3a from-noise sampling vs SDEdit from a footprint-extrude blockout,
on held-out test footprints, rendered through the ticket-05 neutral harness.

Both arms use the SAME per-building conditioning (footprint, height, class subtype, the
style-agnostic massing token) built from `frame_n_input` -- the only thing that differs is the
diffusion STARTING POINT: `Stage3aModel.inference()` starts from pure Gaussian noise;
`Stage3aModel.sdedit(strength=0.5)` starts from a partially-noised encoding of the
footprint-extrude blockout (ADR 0004's declared robustness variant of the coarse input --
"extrude the real footprint from the grid floor to the real building's own height"). Guidance is
matched (plain CFG, `guide_model=None`, the model's own default `uc_scale=1.0`) rather than using
production's autoguidance for `sdedit` -- `.inference()` has no autoguidance parameter at all, so
using it only for `sdedit` would confound "better starting point" with "better guidance
mechanism", the two claims C1 is not trying to make. `strength=0.5` and `ddim_steps=None` (->
the model's own default, 50) are the codebase's existing canonical values (`eval_harness.py`'s
STRENGTHS list treats 0.5 as primary; `refine_sdedit`'s own default is `strength=0.5`) --
ticket 10 is the dedicated strength-sweep, so this ticket fixes one value rather than duplicating
that work.

LEAKAGE CATCH (checked before running anything): `data/splits_v1/test.json` (ticket 03's sealed
research-proof test split) is a DIFFERENT partition than the original
`data/BuildingNet_dataset_v0_1/splits/{train,val,test}_split.txt` Stage 3a was actually TRAINED
on. Checking the overlap: 224/277 of ticket 03's "held-out" ids were in Stage3a's own training
set, 26 in its validation set, and only 27 were genuinely never seen by the live prior in any
form. "held-out footprints" (the ticket's own wording) can only mean that clean 27 -- using the
other 250 would let the model's memorization of those specific buildings pass as evidence for
generalization. `held_out_population()` computes this partition and this script evaluates the
clean tier only, reporting the leak counts for the record.

Massing is the PRIMARY metric (paired IoU/footprint-IoU against the real building -- CONTEXT.md:
massing fidelity is measured paired, because massing is determined by the footprint). FID is
supporting evidence only: ticket 05 already found FID needs far more than 2048 images to be
reliable, and this population (27 buildings) is smaller than ticket 05's own sanity run --
`fid.undersampled` will certainly fire; report it, don't hide it.

Out: execution/artifacts/transform_vs_noise.json, outputs/transform_vs_noise/*.png (qualitative
     from-noise vs blockout vs real montages -- the failure mode this ticket predicts is visual).
Run:  TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        ./sdfusion/bin/python scripts/eval/transform_vs_noise.py [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations", "scripts/server"):
    sys.path.insert(0, str(REPO / _p))
sys.path.insert(0, str(REPO))

import fid as fidmod  # noqa: E402
import render_facades as rf  # noqa: E402
from eval_harness import frame_n_input, iou, fp_iou  # noqa: E402
from datasets.buildingnet_retrieval_dataset import (  # noqa: E402
    build_label_maps, load_split_ids, subtype_label,
)
from datasets.stage3a_dataset import STYLE_UNKNOWN_ID  # noqa: E402

STRENGTH = 0.5  # existing codebase default (eval_harness.py, refine_sdedit) -- not swept here
N_VIEWS = 6      # matches ticket 05's sanity-run convention


def footprint_extrude_blockout(occ: np.ndarray) -> np.ndarray:
    """ADR 0004's footprint-extrude coarse-input variant: the real building's top-down
    footprint (any-y occupied (D,W) mask), solid-extruded from the grid floor to the real
    building's own max occupied Y level. `occ`: (D,H,W) boolean occupancy, H = up axis."""
    fp = occ.any(axis=1)                      # (D, W)
    y_occ = occ.any(axis=(0, 2))              # (H,)
    ys = np.where(y_occ)[0]
    top = int(ys.max()) if len(ys) else occ.shape[1] - 1
    blockout = np.zeros_like(occ)
    blockout[:, : top + 1, :] = fp[:, None, :]
    return blockout


def classify_leakage(building_ids, bn_train_ids, bn_val_ids, bn_test_ids) -> dict:
    """Partition `building_ids` (ticket 03's own sealed split) by whether Stage3a's OWN
    original train/val/test split already saw them. `train_leak` is checked first (most
    severe: gradient-trained on it)."""
    train, val, test = set(bn_train_ids), set(bn_val_ids), set(bn_test_ids)
    tiers = dict(clean=[], val_leak=[], train_leak=[], unknown=[])
    for bid in building_ids:
        if bid in train:
            tiers["train_leak"].append(bid)
        elif bid in val:
            tiers["val_leak"].append(bid)
        elif bid in test:
            tiers["clean"].append(bid)
        else:
            tiers["unknown"].append(bid)
    return tiers


def held_out_population():
    """The genuinely-never-seen-by-Stage3a subset of ticket 03's sealed test split, plus the
    subtype->class_id map built EXACTLY as Stage3aDataset builds it at training time (same
    input ids, same deterministic sort) -- the class conditioning must match training or the
    comparison is conditioned on a scrambled subtype id."""
    bn_root = REPO / "data/BuildingNet_dataset_v0_1"
    bn_train, bn_val, bn_test = (load_split_ids(bn_root, p) for p in ("train", "val", "test"))
    subtype_to_idx, _ = build_label_maps(bn_train + bn_val + bn_test)
    splits_v1_test = json.load(open(REPO / "data/splits_v1/test.json"))
    tiers = classify_leakage(splits_v1_test, bn_train, bn_val, bn_test)
    return tiers, subtype_to_idx


def build_condition(building_id, subtype_to_idx, device):
    """Real occupancy (native 64^3) -> the blockout + Frame-N conditioning contract shared by
    both arms. Returns (data_dict, real_occ, real_sdf)."""
    import torch
    real_sdf = rf.load_buildingnet_sdf(building_id, working_res=64, device=device)
    real_occ = real_sdf <= 0
    blockout_occ = footprint_extrude_blockout(real_occ)
    sdf_t, fp_t, height_n = frame_n_input(blockout_occ, device)
    cid = subtype_to_idx.get(subtype_label(building_id), 0)
    data = dict(
        sdf=sdf_t, fp=fp_t,
        class_id=torch.tensor([cid], dtype=torch.long, device=device),
        style_id=torch.full((1,), STYLE_UNKNOWN_ID, dtype=torch.long, device=device),
        height=torch.tensor([height_n], dtype=torch.float32, device=device),
    )
    return data, real_occ, real_sdf


def sample_both_arms(model, data):
    """`data['sdf']` (the blockout) is only READ by `sdedit` (it encodes it); `inference` never
    touches it -- both arms share one conditioning dict."""
    noise_sdf = model.inference(data, max_sample=1)[0, 0].detach().cpu().numpy()
    blockout_sdf = model.sdedit(data, strength=STRENGTH, max_sample=1,
                                guide_model=None)[0, 0].detach().cpu().numpy()
    return noise_sdf, blockout_sdf


def _git_provenance():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return dict(git_rev=None, dirty_digest=None)
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    digest = hashlib.sha1(status.encode()).hexdigest()[:12] if status.strip() else None
    return dict(git_rev=rev, dirty_digest=digest)


def _montage(rows, out_path: Path, cell=224):
    """rows: list of (label, [(title, (D,H,W) sdf or None), ...]) -- one row per building."""
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
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N clean buildings")
    ap.add_argument("--views", type=int, default=N_VIEWS)
    ap.add_argument("--img-res", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/transform_vs_noise.json"))
    ap.add_argument("--montage-out", default=str(REPO / "outputs/transform_vs_noise/montage.png"))
    ap.add_argument("--montage-n", type=int, default=6)
    a = ap.parse_args()

    import torch
    from refine import Refiner
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tiers, subtype_to_idx = held_out_population()
    ids = tiers["clean"]
    if a.limit:
        ids = ids[: a.limit]
    print(f"[population] clean(never seen by Stage3a)={len(tiers['clean'])} "
          f"val_leak={len(tiers['val_leak'])} train_leak={len(tiers['train_leak'])} "
          f"unknown={len(tiers['unknown'])} -- evaluating {len(ids)} clean ids")

    print("[*] loading the deployed live prior (Refiner._load_sdedit, autoguidance=False -- "
          "matched plain-CFG guidance for a controlled comparison)...")
    refiner = Refiner(SimpleNamespace(device=device))
    model, _ = refiner._load_sdedit(autoguidance=False)

    cams = rf.orbit_cameras(n_views=a.views)
    noise_imgs, blockout_imgs, real_imgs = [], [], []
    montage_rows, per_building, failures = [], [], []
    for i, bid in enumerate(ids):
        try:
            data, real_occ, real_sdf = build_condition(bid, subtype_to_idx, device)
            noise_sdf, blockout_sdf = sample_both_arms(model, data)
            noise_occ, blockout_occ = noise_sdf <= 0, blockout_sdf <= 0

            m_iou_noise, m_iou_block = iou(noise_occ, real_occ), iou(blockout_occ, real_occ)
            fp_noise, fp_block = fp_iou(noise_occ, real_occ), fp_iou(blockout_occ, real_occ)
            # Recorded, not filtered on: BuildingNet's native 64^3 SDFs vary widely in interior
            # depth/occupancy (some raw meshes appear non-watertight), and the distribution across
            # the 27-building clean population is continuous with no natural gap separating
            # "broken" from "valid" -- picking a cutoff would be an undisclosed a-posteriori
            # choice. Reported per building so a reader can judge data quality themselves.
            per_building.append(dict(
                building=bid, iou_noise=m_iou_noise, iou_blockout=m_iou_block,
                fp_iou_noise=fp_noise, fp_iou_blockout=fp_block,
                real_occupancy_frac=float(real_occ.mean()), real_sdf_min=float(real_sdf.min())))
            print(f"  [{i+1}/{len(ids)}] {bid}  iou noise={m_iou_noise:.3f} blockout={m_iou_block:.3f}  "
                  f"fp_iou noise={fp_noise:.3f} blockout={fp_block:.3f}  "
                  f"real_occ={100*real_occ.mean():.2f}%", flush=True)

            real96 = rf.load_buildingnet_sdf(bid, working_res=rf.WORKING_RES, device=device)
            noise96 = rf.resample_sdf_grid(noise_sdf, rf.WORKING_RES, device=device)
            block96 = rf.resample_sdf_grid(blockout_sdf, rf.WORKING_RES, device=device)
            real_imgs.append(rf.render_sdf_neutral(real96, cameras=cams, res=a.img_res, device=device))
            noise_imgs.append(rf.render_sdf_neutral(noise96, cameras=cams, res=a.img_res, device=device))
            blockout_imgs.append(rf.render_sdf_neutral(block96, cameras=cams, res=a.img_res, device=device))

            if len(montage_rows) < a.montage_n:
                row_label = f"{bid}\n(real_occ={100*real_occ.mean():.2f}%)"
                montage_rows.append((row_label, [("real", real_sdf),
                                                 ("from-noise", noise_sdf), ("blockout SDEdit", blockout_sdf)]))
        except Exception as ex:  # noqa: BLE001
            failures.append(dict(building=bid, error=f"{type(ex).__name__}: {str(ex)[:120]}"))
            print(f"  [{i+1}/{len(ids)}] {bid} FAILED: {failures[-1]['error']}", flush=True)

    if montage_rows:
        _montage(montage_rows, Path(a.montage_out))
        print(f"[save] {a.montage_out}")

    def stacked(per_bldg_imgs):
        imgs = np.stack([im for views in per_bldg_imgs for im in views])
        groups = np.repeat(np.arange(len(per_bldg_imgs)), [len(v) for v in per_bldg_imgs])
        return imgs, groups

    fid_block = fid_noise = None
    if len(real_imgs) >= 2:
        ext = fidmod.InceptionExtractor(device=device)
        r_imgs, r_groups = stacked(real_imgs)
        r_feat = ext.features(r_imgs)
        n_imgs, n_groups = stacked(noise_imgs)
        n_feat = ext.features(n_imgs)
        b_imgs, b_groups = stacked(blockout_imgs)
        b_feat = ext.features(b_imgs)
        p_n, lo_n, hi_n = fidmod.bootstrap_fid_ci(n_feat, r_feat, n_boot=a.n_boot, seed=a.seed,
                                                   groups_a=n_groups, groups_b=r_groups)
        p_b, lo_b, hi_b = fidmod.bootstrap_fid_ci(b_feat, r_feat, n_boot=a.n_boot, seed=a.seed,
                                                   groups_a=b_groups, groups_b=r_groups)
        fid_noise = dict(point=p_n, ci95=[lo_n, hi_n],
                         undersampled=bool(fidmod.undersampled(n_feat, r_feat)))
        fid_block = dict(point=p_b, ci95=[lo_b, hi_b],
                         undersampled=bool(fidmod.undersampled(b_feat, r_feat)))
        print(f"[fid] from-noise vs real: {fid_noise}")
        print(f"[fid] blockout  vs real: {fid_block}")

    def _stat(key, fn):
        return float(fn([r[key] for r in per_building])) if per_building else None

    stats = dict(
        mean_iou=dict(from_noise=_stat("iou_noise", np.mean), blockout_sdedit=_stat("iou_blockout", np.mean)),
        median_iou=dict(from_noise=_stat("iou_noise", np.median), blockout_sdedit=_stat("iou_blockout", np.median)),
        mean_fp_iou=dict(from_noise=_stat("fp_iou_noise", np.mean), blockout_sdedit=_stat("fp_iou_blockout", np.mean)),
        median_fp_iou=dict(from_noise=_stat("fp_iou_noise", np.median), blockout_sdedit=_stat("fp_iou_blockout", np.median)),
    )

    manifest = dict(
        strength=STRENGTH, ddim_steps=None, guidance="plain CFG (matched, guide_model=None)",
        population=dict(n_clean=len(tiers["clean"]), n_val_leak=len(tiers["val_leak"]),
                        n_train_leak=len(tiers["train_leak"]), n_unknown=len(tiers["unknown"]),
                        n_evaluated=len(ids)),
        n_succeeded=len(per_building), n_failed=len(failures), failures=failures,
        per_building=per_building,
        **stats,
        fid=dict(from_noise_vs_real=fid_noise, blockout_vs_real=fid_block),
        cameras=dict(n_views=a.views), image_res=a.img_res, sdf_working_res=rf.WORKING_RES,
        montage=a.montage_out if montage_rows else None,
        **_git_provenance(),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(a.out, "w"), indent=2)
    print(f"\n[done] mean IoU  from-noise={stats['mean_iou']['from_noise']}  "
          f"blockout={stats['mean_iou']['blockout_sdedit']}")
    print(f"[done] median IoU  from-noise={stats['median_iou']['from_noise']}  "
          f"blockout={stats['median_iou']['blockout_sdedit']}")
    print(f"[done] mean fp-IoU  from-noise={stats['mean_fp_iou']['from_noise']}  "
          f"blockout={stats['mean_fp_iou']['blockout_sdedit']}")
    print(f"[done] median fp-IoU  from-noise={stats['median_fp_iou']['from_noise']}  "
          f"blockout={stats['median_fp_iou']['blockout_sdedit']}")
    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
