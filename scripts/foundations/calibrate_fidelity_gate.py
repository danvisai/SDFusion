"""Map #34 ticket #36 — calibrate the surface-fidelity gate: local normal-consistency of real LoD2
(positive) vs the current rough prior samples (negative control). Floor -> real's ~10th percentile.

field-gradient normal-consistency: the surface normal field is n = normalize(grad SDF); near the
surface, measure the cosine agreement between n and its 3x3x3-smoothed local mean. Crisp field ->
locally-constant gradient direction -> ~1; noisy field -> jittery gradient -> lower. (An earlier
MESH-vertex version was rejected: the #36 calibration showed it saturates near 1 for crisp AND rough
alike, because marching-cubes smooths per-vertex normals before we ever see the field noise #35 found.)

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/calibrate_fidelity_gate.py --n 20
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "foundations"))
from baseline_gate_eval import build_opt  # reuse tested model-loading opt

CKPT = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"


def normal_consistency(sdf, band=0.06):
    """Field-gradient normal-consistency in [0,1]; higher = crisper. Smoothness of the SDF gradient
    DIRECTION in the near-surface band (|sdf|<band). Measures the field noise #35 located, which the
    marching-cubes mesh version smooths away."""
    from scipy.ndimage import uniform_filter
    sdf = sdf.astype(np.float32)
    g = np.gradient(sdf)
    mag = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) + 1e-6
    n = [g[i] / mag for i in range(3)]
    ns = [uniform_filter(n[i], size=3) for i in range(3)]        # local mean of each normal component
    smag = np.sqrt(ns[0] ** 2 + ns[1] ** 2 + ns[2] ** 2) + 1e-6
    cos = (n[0] * ns[0] + n[1] * ns[1] + n[2] * ns[2]) / smag    # n · local-mean-n
    mask = np.abs(sdf) < band
    if mask.sum() < 8:
        return float("nan")
    return float(np.clip(cos[mask], 0.0, 1.0).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); a = ap.parse_args()
    import torch
    from datasets.bag3d_dataset import Bag3dDataset
    from models.stage3a_model import Stage3aModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    opt = build_opt(dev, ckpt=CKPT, use_region=True, use_extra_cond=False)
    print(f"[load] {CKPT}", flush=True)
    model = Stage3aModel(); model.initialize(opt)
    ds = Bag3dDataset(); ds.initialize(opt, phase="test")
    pick = np.random.default_rng(0).choice(len(ds), size=a.n, replace=False)

    real_nc, samp_nc = [], []
    for j, idx in enumerate(pick):
        item = ds[int(idx)]
        real_nc.append(normal_consistency(item["sdf"].numpy()[0]))
        data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v)
                for k, v in item.items() if torch.is_tensor(v)}
        with torch.no_grad():
            samp = model.inference(data, ddim_steps=100).detach().cpu().numpy()[0, 0]
        samp_nc.append(normal_consistency(samp))
        if (j + 1) % 5 == 0:
            print(f"  [{j+1}/{a.n}] real_nc={real_nc[-1]:.3f} sample_nc={samp_nc[-1]:.3f}", flush=True)

    r = np.array([x for x in real_nc if np.isfinite(x)])
    s = np.array([x for x in samp_nc if np.isfinite(x)])
    def pct(x, p): return float(np.percentile(x, p))
    summary = dict(
        n_real=len(r), n_sample=len(s),
        real_median=pct(r, 50), real_p10=pct(r, 10), real_p25=pct(r, 25),
        sample_median=pct(s, 50), sample_p90=pct(s, 90),
        recommend_floor_real_p10=pct(r, 10),
        separation_ok=bool(pct(s, 90) < pct(r, 10)),   # do the two populations separate at the floor?
    )
    (REPO / "execution/artifacts/fidelity_gate_calibration.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FIDELITY CALIBRATION (local normal-consistency) ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    print(f"\n  => real: median {pct(r,50):.3f} / p10 {pct(r,10):.3f}   |   current sample: median {pct(s,50):.3f}", flush=True)
    print(f"  => proposed floor (real p10): {pct(r,10):.3f}  (current sample median {pct(s,50):.3f} should sit below it)", flush=True)


if __name__ == "__main__":
    main()
