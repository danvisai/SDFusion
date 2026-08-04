"""Reference-guided snap PROBE (no retrain): does pulling the SDEdit toward a retrieved real
building's latent make the added mass more building-like?

For DE + JP inputs: retrieve the nearest real building (FootprintEmbedNet kNN over real.h5),
encode it via the VQVAE, and bias the SDEdit start latent toward it (ref_alpha sweep). Localized
blend keeps the body bit-exact. A/B columns: ref_alpha = 0 (current snap) / 0.3 / 0.6, plus the
retrieved reference itself. Crude proxy for the full cross-attn reference conditioning (Phidias);
a positive signal here justifies building the encoder+finetune.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. ./sdfusion/bin/python scripts/sdedit_ref_probe.py
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
N_BANK = 1500


def cube_pts(dev):
    g = torch.linspace(-1, 1, 64, device=dev)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    return torch.stack([X, Y, Z], -1).reshape(-1, 3)


def read_row(row):
    with h5py.File(REAL, "r") as f:
        sdf = f["sdf"][row].astype(np.float32)
        fp = f["footprint"][row].astype(np.uint8)
    return sdf, fp


def fp_embed(m, fp_np, dev):
    fp = torch.from_numpy(fp_np).float().view(-1, 1, 64, 64).to(dev)
    cid = torch.zeros(fp.shape[0], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb, _ = m.fp_encoder(fp, class_id=cid)
    return torch.nn.functional.normalize(emb, dim=1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(str(FT), dev)
    pts = cube_pts(dev)
    box = _primitive(EditOp.from_dict(TOWER))(pts).reshape(64, 64, 64)
    w = _edit_locality_mask([TOWER], pts, INNER, BAND).reshape(64, 64, 64).to(dev)

    # retrieval bank
    with h5py.File(REAL, "r") as f:
        ntot = f["sdf"].shape[0]
    rng = np.random.default_rng(0)
    bank_rows = np.unique(rng.integers(0, ntot, size=N_BANK))
    embs = []
    with h5py.File(REAL, "r") as f:
        for i in range(0, len(bank_rows), 256):
            chunk = bank_rows[i:i+256]
            fps = np.stack([f["footprint"][int(r)].astype(np.uint8) for r in chunk])
            embs.append(fp_embed(m, fps, dev))
    bank = torch.cat(embs, 0)                                          # (B,256) normalized
    print(f"[probe] retrieval bank {bank.shape[0]} buildings")

    nc = 3 + len(ALPHAS); nr = len(QUERIES)
    fig = plt.figure(figsize=(2.9 * nc, 3.2 * nr))
    def panel(r, c, title, sdf):
        ax = fig.add_subplot(nr, nc, r*nc + c + 1, projection="3d")
        ax.set_title(title, fontsize=8); ax.set_axis_off(); ax.view_init(elev=18, azim=-55)
        ax.set_box_aspect((1,1,1))
        if sdf is None: return
        mm = mesh(sdf if sdf.dim()==5 else sdf[None,None])
        if mm is None: return
        v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
        ax.plot_trisurf(v[:,0], v[:,2], fc, v[:,1], color="#c9b790", edgecolor="none", shade=True)
        lim=[v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)

    for r, (qlabel, qrow) in enumerate(QUERIES):
        qsdf, qfp = read_row(qrow)
        qemb = fp_embed(m, qfp[None], dev)                            # (1,256)
        sim = (bank @ qemb.T).squeeze(1)                              # cosine
        order = torch.argsort(sim, descending=True).tolist()
        ref_row = next(int(bank_rows[i]) for i in order if int(bank_rows[i]) != qrow)
        rsdf, _ = read_row(ref_row)
        ref_t = torch.from_numpy(rsdf).view(1,1,64,64,64).to(dev).clamp(-TRUNC, TRUNC)
        with torch.no_grad():
            z_ref = m.vqvae(ref_t, forward_no_quant=True, encode_only=True).detach() * m.scale_factor
        print(f"[{qlabel}] query row{qrow} -> ref row{ref_row} (cos {float(sim[order[0]]):.3f})")

        sdf0 = torch.from_numpy(qsdf).view(1,1,64,64,64).to(dev).clamp(-TRUNC, TRUNC)
        edited = torch.minimum(sdf0[0,0], box).clamp(-TRUNC, TRUNC)[None,None]
        base = {"fp": torch.from_numpy(qfp).float().view(1,1,64,64).to(dev),
                "class_id": torch.zeros(1,dtype=torch.long,device=dev),
                "style_id": torch.full((1,),8,dtype=torch.long,device=dev),
                "height": torch.tensor([1.0],dtype=torch.float32,device=dev)}
        edit_data = dict(base); edit_data["sdf"] = edited

        panel(r, 0, f"{qlabel}\nbody", sdf0)
        panel(r, 1, "edited (+tower)", edited)
        panel(r, 2, f"retrieved REF (row{ref_row})", ref_t)
        for j, a in enumerate(ALPHAS):
            torch.manual_seed(0)
            out = m.sdedit(edit_data, strength=0.5, ddim_steps=50, uc_scale=1.0,
                           ref_latent=(z_ref if a > 0 else None), ref_alpha=a)
            local = edited[0,0]*(1-w) + out[0,0]*w
            print(f"  {qlabel}  ref_alpha={a}  occ={float((local<=0).float().mean()):.3f}")
            panel(r, 3+j, f"snap ref_alpha={a}", local[None,None])

    out_p = REPO / "outputs/sdedit_xcultural/ref_probe.png"; out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Reference-guided snap PROBE (no retrain) — retrieved real building biases the SDEdit start\n"
                 "cols: body | edited(+tower) | retrieved ref | snap ref_alpha 0 / 0.3 / 0.6   (body bit-exact)", fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(out_p, dpi=88)
    print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
