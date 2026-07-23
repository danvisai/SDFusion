"""#30 Checkpoint 0 — baseline gate eval: run the DEPLOYED Stage3a on LoD2 held-out and score
against the #27 acceptance gate. If it passes, the map's destination is reached with no retrain.

Conditioning is built by Bag3dDataset (training-correct for LoD2: class=0, style=unknown, height
from GT, region=source_id). Model loaded exactly as deployed (vqvae_bnet + vqvae_clean, use_extra_cond
True, use_region False; era/floors/region default to unknown at inference).

Held-out split caveat: this uses Bag3dDataset's deterministic phase="test" 2% slice. For the from-scratch
LoD2 retrain (which trains on the SAME real.h5 phase="train") that slice is a genuine held-out set. For
the DEPLOYED baseline it is only approximately held out -- that checkpoint trained on bag3d.h5 (NL) under
a different permutation, so the NL third overlaps its training. The baseline failed the gate regardless,
so the NO-GO stands; a sealed class/source-stratified split (per #30) is a follow-up for the retrain evals.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/baseline_gate_eval.py --n 2   (smoke)
For a retrain checkpoint: --ckpt <path> --use_region 1 --use_extra_cond 0 --tag <label> --n 60
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import ndimage
# torch / Stage3aModel / Bag3dDataset are imported lazily inside main() so the pure gate
# functions (lcc_frac, fp_iou, score_gate) are testable without a GPU or heavy deps.

REPO = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(REPO))

CKPT = "logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt/stage3a_steps-latest.pth"


def build_opt(device, ckpt=CKPT, use_region=False, use_extra_cond=True, use_ema=True):
    """Opt for the deployed model by default; pass use_region=True, use_extra_cond=False to score
    the from-scratch LoD2 retrain checkpoints (which changed the conditioning).

    Phase-1 surface-fidelity knob (map #34): `use_ema` selects the checkpoint's EMA weights
    (ema_df) vs the raw training weights at inference. Default True mirrors the deployed path
    (the map #24 gate ran with EMA on); pass False to score the raw-weights config for comparison.
    """
    return SimpleNamespace(
        isTrain=False, device=device, debug="0", gpu_ids=[0], ckpt_dir="/tmp",
        df_cfg="configs/stage3a_sdf_diffusion.yaml",
        vq_cfg="configs/vqvae_bnet.yaml",
        vq_ckpt="logs_building/vqvae_clean_ft/vqvae_clean.pth",
        ckpt=ckpt, ddim_steps=100, use_ema=use_ema,
        use_region=use_region, num_regions=4, region_emb_dim=16, use_extra_cond=use_extra_cond,
        latent_size_HW=(16, 16), latent_size_D=16,
        bag3d_h5="data/real_massing_v1/real.h5", trunc_thres=0.2, augment=False,
    )


def lcc_frac(occ):
    n = int(occ.sum())
    if n == 0: return 0.0
    lab, k = ndimage.label(occ)
    return float(np.bincount(lab.ravel())[1:].max() / n) if k else 0.0


def mesh_sdf_surface(vol):
    """Marching-cubes surface of a CONTINUOUS SDF at level 0.0 -- the production convention
    (scene/sdf_primitives, scene/run_demo). Returns (verts, faces), or (None, None) when the field
    has no usable zero crossing (all-solid or empty) so callers skip instead of crashing.

    Pass the continuous SDF, never a binary 0/1 occupancy: isosurfacing a mask at 0.5 staircases
    every non-axis-aligned face (the #39 render artifact); the 0.0 crossing of the real field comes
    out crisp. This is the one honest way to mesh an SDF surface in the eval harness (#43)."""
    from skimage import measure  # lazy: keep this module import-light (no skimage at load)
    vol = np.asarray(vol, dtype=np.float32)
    # parity with the pre-extraction inline guard: proceed only with >8 non-positive voxels AND
    # some positive voxel (a zero crossing). Anything less -> skip so callers don't crash.
    if int((vol <= 0.0).sum()) <= 8 or not bool((vol > 0.0).any()):
        return None, None
    try:
        verts, faces, *_ = measure.marching_cubes(vol, 0.0)
    except (ValueError, RuntimeError):
        return None, None
    return verts, faces


def score_gate(rows):
    """Pure #27 acceptance-gate scoring over per-building rows. No GPU/torch."""
    a = {k: np.array([r[k] for r in rows], float) for k in ("gen_occ", "lcc", "fp_iou")}
    collapse_rate = float(np.mean([r["collapsed"] for r in rows]))
    lcc_ge90 = float((a["lcc"] >= 0.90).mean())
    fp_med, fp_p10 = float(np.median(a["fp_iou"])), float(np.percentile(a["fp_iou"], 10))
    gate = dict(
        n=len(rows),
        collapse_rate=collapse_rate, collapse_pass=collapse_rate <= 0.01,
        lcc_ge90_frac=lcc_ge90, lcc_pass=lcc_ge90 >= 0.85,
        fp_iou_median=fp_med, fp_iou_p10=fp_p10,
        fp_iou_pass=(fp_med >= 0.65 and fp_p10 >= 0.35),
        real_fp_self_iou_median=float(np.median([r["real_fp_self_iou"] for r in rows])),
    )
    gate["OVERALL_SCALAR_PASS"] = bool(gate["collapse_pass"] and gate["lcc_pass"] and gate["fp_iou_pass"])
    return gate


def per_corpus_diagnostics(rows):
    """Non-gating per-region breakdown (source_id: 0=NL 1=DE 2=JP), per #27 criterion 4."""
    out = {}
    for reg in sorted({int(r["region"]) for r in rows}):
        sub = [r for r in rows if int(r["region"]) == reg]
        out[str(reg)] = dict(
            n=len(sub),
            fp_iou_median=float(np.median([r["fp_iou"] for r in sub])),
            lcc_ge90_frac=float(np.mean([r["lcc"] >= 0.90 for r in sub])),
            collapse_rate=float(np.mean([r["collapsed"] for r in sub])),
        )
    return out


def fp_iou(gen_occ, real_fp):
    g = gen_occ.any(axis=1)  # footprint-axis (H = axis 1)
    r = np.asarray(real_fp).astype(bool)
    inter = (g & r).sum(); uni = (g | r).sum()
    return float(inter / uni) if uni else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--ddim", type=int, default=100, help="DDIM steps (Phase-1 knob: sweep 100->250/500)")
    ap.add_argument("--ckpt", default=CKPT, help="Stage3a checkpoint to score")
    ap.add_argument("--use_region", type=int, default=0, help="1 for the LoD2-retrain checkpoints")
    ap.add_argument("--use_extra_cond", type=int, default=1, help="0 for the LoD2-retrain checkpoints")
    ap.add_argument("--use_ema", type=int, default=1,
                    help="Phase-1 knob (map #34): 1=EMA weights (deployed default), 0=raw weights")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="Phase-1 knob (map #34): CFG unconditional_guidance_scale (1.0=plain conditional)")
    ap.add_argument("--tag", default="", help="suffix for artifact/montage filenames")
    a = ap.parse_args()

    import torch  # lazy: keep module import-light so score_gate is testable without a GPU
    from datasets.bag3d_dataset import Bag3dDataset
    from models.stage3a_model import Stage3aModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = build_opt(device, ckpt=a.ckpt, use_region=bool(a.use_region),
                    use_extra_cond=bool(a.use_extra_cond), use_ema=bool(a.use_ema)); opt.ddim_steps = a.ddim

    print(f"[load] Stage3a from {a.ckpt}", flush=True)
    print(f"[cfg]  use_ema={bool(a.use_ema)} ddim={a.ddim} guidance={a.guidance} "
          f"use_region={bool(a.use_region)} use_extra_cond={bool(a.use_extra_cond)}", flush=True)
    model = Stage3aModel(); model.initialize(opt)

    ds = Bag3dDataset(); ds.initialize(opt, phase="test")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)

    rows = []; montage = []
    for j, idx in enumerate(pick):
        item = ds[int(idx)]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
                for k, v in item.items() if torch.is_tensor(v)}
        with torch.no_grad():
            sdf = model.inference(data, ddim_steps=opt.ddim_steps, uc_scale=a.guidance)  # (1,1,64,64,64)
        gen = sdf.detach().cpu().numpy()[0, 0]
        occ = gen <= 0
        real_fp = item["fp"].numpy()[0]
        real_sdf = item["sdf"].numpy()[0]
        real_occ = real_sdf <= 0
        gen_occ_frac = float(occ.mean())
        rows.append(dict(idx=int(idx), region=int(item["region_id"]),
                         gen_occ=gen_occ_frac, collapsed=bool(gen_occ_frac < 1e-4),
                         lcc=lcc_frac(occ), fp_iou=fp_iou(occ, real_fp),
                         real_occ=float(real_occ.mean()),
                         real_fp_self_iou=fp_iou(real_occ, real_fp)))  # sanity: should be ~1.0
        if len(montage) < 6:
            # store the CONTINUOUS SDF (not binary occ): meshed at 0.0 like the production path,
            # so the montage shows true surfaces, not a binary marching-cubes staircase (#39).
            montage.append((item["region_id"].item(), real_sdf.copy(), gen.copy()))
        print(f"  [{j+1}/{len(pick)}] region={rows[-1]['region']} gen_occ={gen_occ_frac*100:.2f}% "
              f"lcc={rows[-1]['lcc']:.3f} fp_iou={rows[-1]['fp_iou']:.3f} "
              f"real_self_iou={rows[-1]['real_fp_self_iou']:.3f} real_occ={rows[-1]['real_occ']*100:.1f}%", flush=True)

    # visual montage: real GT vs generated
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 3 * len(montage)))
        for ri, (reg, rocc, gocc) in enumerate(montage):
            for ci, (title, m) in enumerate([("real LoD2", rocc), ("generated", gocc)]):
                ax = fig.add_subplot(len(montage), 2, ri * 2 + ci + 1, projection="3d"); ax.set_axis_off()
                v, f = mesh_sdf_surface(m)  # continuous SDF @0.0 -> crisp faces, not a staircase (#43)
                if v is not None:
                    ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color=(0.72, 0.68, 0.55), lw=0)
                    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
                if ri == 0: ax.set_title(title)
                if ci == 0: ax.text2D(-0.1, 0.5, f"region {reg}", transform=ax.transAxes, fontsize=8)
        suffix = f"_{a.tag}" if a.tag else ""
        mp = REPO / f"outputs/baseline_gate_eval/montage{suffix}.png"; mp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(mp, dpi=90, bbox_inches="tight"); plt.close(fig)
        print(f"montage: {mp}", flush=True)
    except Exception as e:
        print(f"montage skipped: {e}", flush=True)

    gate = score_gate(rows)
    import subprocess
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    meta = dict(git_rev=rev, ckpt=a.ckpt, n=len(rows), ddim=a.ddim, seed=0,
                use_ema=bool(a.use_ema), guidance=a.guidance,
                use_region=bool(a.use_region), use_extra_cond=bool(a.use_extra_cond))
    suffix = f"_{a.tag}" if a.tag else ""
    out = REPO / f"execution/artifacts/baseline_gate_eval{suffix}.json"
    out.write_text(json.dumps(dict(meta=meta, gate=gate,
                                   per_corpus=per_corpus_diagnostics(rows), per_building=rows), indent=2))
    print("\n=== GATE (n=%d) ===" % len(rows), flush=True)
    for k, v in gate.items(): print(f"  {k}: {v}", flush=True)
    print(f"artifact: {out}", flush=True)
    return rows, gate


if __name__ == "__main__":
    main()
