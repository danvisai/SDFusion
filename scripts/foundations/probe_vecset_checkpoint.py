"""Early read on a live vecset checkpoint: latent recovery accuracy vs #73's tolerance, and the
projection's footprint/volume against the blockout it starts from."""
import sys, argparse, numpy as np, torch, h5py
sys.path.insert(0, '.')
from models.networks.vecset_denoiser import VecsetDenoiser
from models.networks.vecset_projection import SetSDEdit
from models.shape_codec import Building, DoraCodec
from scripts.foundations.baseline_gate_eval import fp_iou, mesh_sdf_surface
from scripts.foundations.dora_roundtrip_probe import load_dora
from scripts.foundations.eval_massing_arms import (H5, LATENTS, RES, TRUNC, blockout_sdf,
                                                   _vertical_extent, pick_ids, volume_split)
from scripts.foundations.vecset_ceiling_probe import verts_to_world

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--strengths", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5])
ap.add_argument("--proj", type=float, nargs="*", default=[0.35, 0.5])
a = ap.parse_args()

dev = "cuda"
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ca = ck["args"]
net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=ca["width"], depth=ca["depth"],
                     heads=ca["heads"], footprint_res=ck["footprint_res"]).to(dev)
net.load_state_dict(ck["model"]); net.eval()
mu, sd = ck["latent_mu"], ck["latent_sd"]
op = SetSDEdit(net, timesteps=ca["timesteps"])
codec = DoraCodec(load_dora(dev))
print(f"[ckpt] {a.ckpt}  step {ck['step']}", flush=True)

cand, lat_of = pick_ids(__import__('pathlib').Path(LATENTS),
                        "execution/artifacts/massing_arms_eval_baseline.json")
ids = cand[:a.n]
cosr = {s: [] for s in a.strengths}
cosn = {s: [] for s in a.strengths}
proj = {s: {"fp": [], "iou": [], "miss": [], "extra": []} for s in a.proj}
bo_arm = {"fp": [], "iou": [], "miss": [], "extra": []}

with h5py.File(LATENTS, "r") as lf, h5py.File(H5, "r") as gt:
    for bid in ids:
        gocc = np.asarray(gt["sdf"][bid], np.float32) <= 0
        fp = np.asarray(lf["footprint"][lat_of[bid]])
        ht = float(lf["height_m"][lat_of[bid]]); rg = int(lf["region"][lat_of[bid]])
        zn = torch.from_numpy((np.asarray(lf["latent"][lat_of[bid]], np.float32) - mu) / sd)[None].to(dev)
        fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
        htt = torch.tensor([ht], device=dev); rgt = torch.tensor([rg], device=dev)

        # (1) in-distribution recovery: noise a REAL latent, denoise, measure cosine
        for s in a.strengths:
            zt = op.noise_to(zn, strength=s, seed=bid)
            zr = op.project(blockout=zn, footprint=fpt, height=htt, region=rgt,
                            strength=s, steps=20, seed=bid)
            f = lambda x, y: float((x.flatten() @ y.flatten()) / (x.norm() * y.norm()))
            cosn[s].append(f(zn, zt)); cosr[s].append(f(zn, zr))

        # (2) the real task: project FROM the blockout
        bo = blockout_sdf(fp, *_vertical_extent(gocc))
        bv, bf = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
        if bv is None:
            continue
        bocc = bo <= 0
        v = volume_split(bocc, gocc)
        bo_arm["fp"].append(fp_iou(bocc, fp)); bo_arm["iou"].append(v["vol_iou"])
        bo_arm["miss"].append(v["missing"]); bo_arm["extra"].append(v["extra"])
        z0 = (codec.encode(Building(verts=verts_to_world(bv), faces=bf)).float() - mu) / sd
        for s in a.proj:
            zp = op.project(blockout=z0, footprint=fpt, height=htt, region=rgt,
                            strength=s, steps=20, seed=bid)
            with torch.no_grad():
                fld = codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]
            occ = fld <= 0; v = volume_split(occ, gocc)
            proj[s]["fp"].append(fp_iou(occ, fp)); proj[s]["iou"].append(v["vol_iou"])
            proj[s]["miss"].append(v["missing"]); proj[s]["extra"].append(v["extra"])

m = lambda x: float(np.median(x)) if x else float('nan')
print(f"\n--- in-distribution latent recovery (n={len(ids)}) ---")
print(f"{'strength':>9} {'cos(noised)':>12} {'cos(recovered)':>15}   vs #73 tolerance 0.999")
for s in a.strengths:
    r = m(cosr[s])
    print(f"{s:>9.2f} {m(cosn[s]):>12.4f} {r:>15.4f}   {'PASS' if r >= 0.999 else 'below'}")
print(f"\n--- projection from the blockout (n={len(bo_arm['fp'])}) ---")
print(f"{'arm':>16} {'fp-IoU':>8} {'missing':>9} {'extra':>8} {'3D IoU':>8}")
print(f"{'blockout (in)':>16} {m(bo_arm['fp']):>8.3f} {m(bo_arm['miss']):>9.3f} "
      f"{m(bo_arm['extra']):>8.3f} {m(bo_arm['iou']):>8.3f}")
for s in a.proj:
    print(f"{'projected s='+str(s):>16} {m(proj[s]['fp']):>8.3f} {m(proj[s]['miss']):>9.3f} "
          f"{m(proj[s]['extra']):>8.3f} {m(proj[s]['iou']):>8.3f}")
