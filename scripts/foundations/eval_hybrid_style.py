"""Eval the hybrid prior's STYLE conditioning (the gap#1-revival test): same footprint, vary
style_id -> do the generated buildings differ by style? Runs from-noise inference per style.
CPU by default so it doesn't disturb the running retrain's GPU.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from models.stage3a_model import Stage3aModel
from scene.sdf_primitives import grid_to_mesh

BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
STYLES = ["modern", "colonial", "victorian", "industrial", "craftsman", "mediterranean", "contemporary", "public_civic"]


def rect_fp(hw=0.5, hd=0.34):
    fp = torch.zeros(1, 1, 64, 64)
    g = torch.linspace(-1, 1, 64)
    D, W = torch.meshgrid(g, g, indexing="ij")          # (D=z, W=x)
    fp[0, 0][(D.abs() < hd) & (W.abs() < hw)] = 1.0
    return fp


def base_box(hw=0.5, hd=0.34, hh=0.6, trunc=0.2):
    """A plain footprint-extruded box (Frame-N) to SDEdit from."""
    g = torch.linspace(-1, 1, 64)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")    # (D=z,H=y,W=x)
    q = torch.stack([Z.abs() - hd, (Y + 1 - hh).abs() - hh, X.abs() - hw], 0)  # base near ground
    d = torch.linalg.vector_norm(q.clamp(min=0), dim=0) + q.amax(0).clamp(max=0)
    return d.clamp(-trunc, trunc).view(1, 1, 64, 64, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--styles", default="0,1,2,3,5,7")
    ap.add_argument("--mode", default="noise", choices=["noise", "sdedit"])
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--neutral_style", type=int, default=None, help="style-isolating guidance vs this style id")
    ap.add_argument("--out", default="outputs/foundations/hybrid_style_eval.png")
    args = ap.parse_args()
    dev = args.device

    opt = SimpleNamespace(isTrain=False, device=dev,
        df_cfg=str(REPO/"configs/stage3a_sdf_diffusion.yaml"), vq_cfg=str(REPO/"configs/vqvae_bnet.yaml"),
        vq_ckpt=str(REPO/"logs_building/vqvae_clean_ft/vqvae_clean.pth"), ckpt=None,
        ddim_steps=args.steps, debug="0", gpu_ids=[0] if dev=="cuda" else [], ckpt_dir="/tmp",
        latent_size_HW=(16,16), latent_size_D=16, use_extra_cond=True)
    m = Stage3aModel(); m.initialize(opt)
    state = torch.load(args.ckpt, map_location="cpu")
    m.load_ckpt(state)
    if "ema_df" in state:
        m.df.load_state_dict(state["ema_df"]); print("[eval] using EMA weights")

    fp = rect_fp().to(dev)
    base = base_box().to(dev)
    sids = [int(x) for x in args.styles.split(",")]
    panels = []
    if args.mode == "sdedit":
        panels.append(("base box (input)", base))
    for s in sids:
        torch.manual_seed(0)
        data = {"sdf": (base.clone() if args.mode == "sdedit" else torch.zeros(1,1,64,64,64,device=dev)), "fp": fp,
                "class_id": torch.zeros(1,dtype=torch.long,device=dev),
                "style_id": torch.full((1,),s,dtype=torch.long,device=dev),
                "height": torch.tensor([1.0],device=dev),
                "era_id": torch.full((1,),5,dtype=torch.long,device=dev),
                "floors_id": torch.full((1,),4,dtype=torch.long,device=dev)}
        if args.mode == "sdedit":
            out = m.sdedit(data, strength=args.strength, ddim_steps=args.steps, uc_scale=args.guidance,
                           max_sample=1, neutral_style=args.neutral_style)
        else:
            out = m.inference(data, ddim_steps=args.steps, uc_scale=args.guidance, max_sample=1)
        occ = float((out<=0).float().mean())
        print(f"  style {s} ({STYLES[s]:13s}) occ={occ:.3f}")
        panels.append((f"{STYLES[s]}\nocc={occ:.2f}", out))

    fig = plt.figure(figsize=(3.0*len(panels), 3.4))
    for i,(t,sd) in enumerate(panels):
        ax=fig.add_subplot(1,len(panels),i+1,projection="3d"); ax.set_title(t,fontsize=9); ax.set_axis_off()
        mm=grid_to_mesh(sd.detach().cpu()[0,0],BBOX,iso=0.0)
        if mm is not None and len(mm.vertices):
            v,fc=np.asarray(mm.vertices),np.asarray(mm.faces)
            ax.plot_trisurf(v[:,0],v[:,2],fc,v[:,1],color="#b9c4cf",edgecolor="none",shade=True)
            lim=[v.min(),v.max()];ax.set_xlim(lim);ax.set_ylim(lim);ax.set_zlim(lim)
        ax.view_init(elev=18,azim=-60);ax.set_box_aspect((1,1,1))
    fig.suptitle(f"Hybrid prior @ {Path(args.ckpt).stem} — SAME footprint, vary STYLE (g={args.guidance})", fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.92))
    outp=REPO/args.out; outp.parent.mkdir(parents=True,exist_ok=True); fig.savefig(outp,dpi=88)
    print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
