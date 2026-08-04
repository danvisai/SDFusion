"""Relevant-reference DIAGNOSTIC: does a TOWER-BEARING reference transfer tower-ness via the latent
blend? Isolates retrieval-quality (fixable) vs the blend-mechanism-itself.

The previous probe retrieved generic boxes (footprint-only retrieval is degenerate) -> reference had no
structure to transfer. Here we scan the corpus for the most tower-like real building (tall + thin top) and
use IT as the reference for the same ref_alpha sweep on the DE/JP tower edits.

Interpretation:
  - tower-ref TRANSFERS tower-ness via blend  -> retrieval was the bottleneck; full #4 worth building.
  - even a perfect tower-ref FAILS via blend   -> the signal needs the trained cross-attn; #4 is a bigger bet.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. ./sdfusion/bin/python scripts/sdedit_ref_tower_diag.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import h5py, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "server"))
from sdedit_bag3d_test import load_model, mesh, TRUNC
from refine import _edit_locality_mask
from scene.sdf_edit import EditOp, _primitive

REAL = REPO / "data/real_massing_v1/real.h5"
TOWER = {"kind": "box", "center": [0.5, 0.35, 0.5], "size": [0.12, 0.7, 0.12], "mode": "add"}
FT = REPO / "logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth"
ALPHAS = [0.0, 0.3, 0.6]
INNER, BAND = 0.2, 0.5
QUERIES = [("DE (NRW)", 12500), ("JP (PLATEAU)", 25000)]
N_SCAN = 900


def cube_pts(dev):
    g = torch.linspace(-1, 1, 64, device=dev)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    return torch.stack([X, Y, Z], -1).reshape(-1, 3)


def tower_score(sdf):                       # sdf (64,64,64) = (D=z, H=y, W=x)
    occ = sdf <= 0.0
    area = occ.mean(axis=(0, 2))            # (H,) occupied cross-section per height
    ys = np.where(area > 1e-3)[0]
    if ys.size < 10:
        return -1.0
    y0, y1 = int(ys[0]), int(ys[-1]); rng = max(y1 - y0, 1)
    h = rng / 64.0
    top = area[max(y1 - rng // 5, y0): y1 + 1].mean()    # top ~20%
    bot = area[y0: y0 + max(rng // 2, 1)].mean()         # bottom ~50%
    if bot < 1e-4 or top < 2e-3:
        return -1.0
    thinness = 1.0 - min(top / bot, 1.0)                  # thin top vs wide base
    return float(h * thinness)


def find_tower_ref(exclude):
    with h5py.File(REAL, "r") as f:
        ntot = f["sdf"].shape[0]
        rows = np.unique(np.random.default_rng(1).integers(0, ntot, size=N_SCAN))
        best = []
        for r in rows:
            r = int(r)
            if r in exclude:
                continue
            s = f["sdf"][r].astype(np.float32)
            best.append((tower_score(s), r))
    best.sort(reverse=True)
    print("[scan] top tower candidates:", [(round(sc, 3), r) for sc, r in best[:5]])
    return best[0][1]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(str(FT), dev)
    pts = cube_pts(dev)
    box = _primitive(EditOp.from_dict(TOWER))(pts).reshape(64, 64, 64)
    w = _edit_locality_mask([TOWER], pts, INNER, BAND).reshape(64, 64, 64).to(dev)

    ref_row = find_tower_ref(exclude={q for _, q in QUERIES})
    with h5py.File(REAL, "r") as f:
        rsdf = f["sdf"][ref_row].astype(np.float32)
    ref_t = torch.from_numpy(rsdf).view(1, 1, 64, 64, 64).to(dev).clamp(-TRUNC, TRUNC)
    with torch.no_grad():
        z_ref = m.vqvae(ref_t, forward_no_quant=True, encode_only=True).detach() * m.scale_factor
    print(f"[diag] tower reference = row{ref_row}  score={tower_score(rsdf):.3f}")

    nc = 3 + len(ALPHAS); nr = len(QUERIES)
    fig = plt.figure(figsize=(2.9 * nc, 3.2 * nr))
    def panel(r, c, title, sdf):
        ax = fig.add_subplot(nr, nc, r*nc + c + 1, projection="3d")
        ax.set_title(title, fontsize=8); ax.set_axis_off(); ax.view_init(elev=18, azim=-55)
        ax.set_box_aspect((1, 1, 1))
        if sdf is None: return
        mm = mesh(sdf if sdf.dim() == 5 else sdf[None, None])
        if mm is None: return
        v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
        ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#c9b790", edgecolor="none", shade=True)
        lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)

    with h5py.File(REAL, "r") as f:
        for r, (qlabel, qrow) in enumerate(QUERIES):
            qsdf = f["sdf"][qrow].astype(np.float32); qfp = f["footprint"][qrow].astype(np.uint8)
            sdf0 = torch.from_numpy(qsdf).view(1, 1, 64, 64, 64).to(dev).clamp(-TRUNC, TRUNC)
            edited = torch.minimum(sdf0[0, 0], box).clamp(-TRUNC, TRUNC)[None, None]
            edit_data = {"sdf": edited, "fp": torch.from_numpy(qfp).float().view(1, 1, 64, 64).to(dev),
                         "class_id": torch.zeros(1, dtype=torch.long, device=dev),
                         "style_id": torch.full((1,), 8, dtype=torch.long, device=dev),
                         "height": torch.tensor([1.0], dtype=torch.float32, device=dev)}
            panel(r, 0, f"{qlabel}\nbody", sdf0)
            panel(r, 1, "edited (+tower)", edited)
            panel(r, 2, f"TOWER ref (row{ref_row})", ref_t)
            for j, a in enumerate(ALPHAS):
                torch.manual_seed(0)
                out = m.sdedit(edit_data, strength=0.5, ddim_steps=50, uc_scale=1.0,
                               ref_latent=(z_ref if a > 0 else None), ref_alpha=a)
                local = edited[0, 0] * (1 - w) + out[0, 0] * w
                print(f"  {qlabel}  ref_alpha={a}  occ={float((local <= 0).float().mean()):.3f}")
                panel(r, 3 + j, f"snap ref_alpha={a}", local[None, None])

    out_p = REPO / "outputs/sdedit_xcultural/ref_tower_diag.png"
    fig.suptitle("Relevant-reference DIAGNOSTIC — TOWER-bearing reference biases the SDEdit start\n"
                 "cols: body | edited(+tower) | TOWER ref | snap ref_alpha 0 / 0.3 / 0.6", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out_p, dpi=88)
    print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
