"""Render what the A2 numbers actually mean: GT vs blockout vs projection vs what we ship.

The eval reported scalars and they are hard to read as geometry. Same buildings, identical shading,
every panel meshed from a continuous field at level 0.0, each labelled with the two metrics that matter
(footprint-IoU and 3D IoU) so the picture and the number sit together.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks.vecset_denoiser import VecsetDenoiser                  # noqa: E402
from models.networks.vecset_projection import SetSDEdit                     # noqa: E402
from models.shape_codec import Building, DoraCodec                          # noqa: E402
from scripts.foundations.baseline_gate_eval import build_opt, fp_iou, mesh_sdf_surface  # noqa: E402
from scripts.foundations.dora_roundtrip_probe import load_dora, H5          # noqa: E402
from scripts.foundations.eval_vecset_projection import blockout_sdf         # noqa: E402
from scripts.foundations.vecset_ceiling_probe import RES, TRUNC, verts_to_world  # noqa: E402

MAP24 = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs_building/vecset_pair_v1/vecset_denoiser.pth")
    ap.add_argument("--latents", default="data/real_massing_v1/vecset_latents.h5")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", default="outputs/vecset_eval")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import h5py, trimesh
    from PIL import Image, ImageDraw
    from datasets.bag3d_dataset import Bag3dDataset
    from models.stage3a_model import Stage3aModel
    from scripts.hunyuan_building_mesh_smoke import render_mesh_png

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=a["width"], depth=a["depth"],
                         heads=a["heads"], footprint_res=ck["footprint_res"]).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    mu, sd = ck["latent_mu"], ck["latent_sd"]
    op = SetSDEdit(net, timesteps=a["timesteps"])
    codec = DoraCodec(load_dora(dev))

    opt = build_opt(dev, ckpt=MAP24, use_region=True, use_extra_cond=False, use_ema=True)
    opt.bag3d_h5 = str(H5)
    s3 = Stage3aModel(); s3.initialize(opt); s3.switch_eval()
    ds = Bag3dDataset(); ds.initialize(opt, phase="test")
    gof = {int(g): i for i, g in enumerate(ds.idxs)}

    with h5py.File(args.latents, "r") as f:
        held = np.nonzero(f["held_out"][:] == 1)[0]
        rows, fps = f["row"][:][held], f["footprint"][:][held]
        regs, hts = f["region"][:][held], f["height_m"][:][held]

    picked, rows_out = 0, []
    with h5py.File(H5, "r") as gt:
        for r, fp, reg, hm in zip(rows, fps, regs, hts):
            if picked >= args.n or int(r) not in gof:
                continue
            g = np.asarray(gt["sdf"][int(r)], np.float32); gocc = g <= 0
            ys = np.nonzero(gocc.any(axis=(0, 2)))[0]
            if len(ys) == 0:
                continue
            bo = blockout_sdf(fp, int(ys.min()), int(ys.max()))
            if bo is None:
                continue
            v, fc = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
            if v is None:
                continue

            z0 = ((codec.encode(Building(verts=verts_to_world(v), faces=fc)).float() - mu) / sd)
            fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
            ht = torch.tensor([float(hm)], device=dev); rg = torch.tensor([int(reg)], device=dev)

            item = ds[gof[int(r)]]
            data = {k: (val.unsqueeze(0).to(dev) if torch.is_tensor(val) else val)
                    for k, val in item.items() if torch.is_tensor(val)}
            with torch.no_grad():
                gen = s3.inference(data, ddim_steps=opt.ddim_steps, uc_scale=1.0).cpu().numpy()[0, 0]
                proj = {}
                for s in (0.5, 0.65):
                    zp = op.project(blockout=z0, footprint=fpt, height=ht, region=rg,
                                    strength=s, steps=20, seed=0)
                    proj[s] = codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]

            arms = [("GT (real LoD2)", g), ("blockout (extruded footprint)", bo),
                    ("A2-pair projected s=0.5", proj[0.5]), ("A2-pair projected s=0.65", proj[0.65]),
                    ("map-#24 deployed (shipped)", gen)]
            panels = []
            for label, fld in arms:
                occ = fld <= 0
                fi = fp_iou(occ, fp)
                vi = (occ & gocc).sum() / max((occ | gocc).sum(), 1)
                mv, mf = mesh_sdf_surface(np.clip(fld, -TRUNC, TRUNC))
                img = None
                if mv is not None:
                    img = render_mesh_png(trimesh.Trimesh(verts_to_world(mv), mf, process=False),
                                          image_size=args.size)
                panels.append((label, img, fi, vi))
            rows_out.append(panels)
            picked += 1
            print(f"  rendered row {r}  " +
                  "  ".join(f"{l.split('(')[0].strip()}={v:.2f}" for l, _, _, v in panels), flush=True)

    S, PAD, HDR = args.size, 8, 26
    W = 5 * S + 6 * PAD
    H = HDR + len(rows_out) * (S + HDR) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 7), "A2 first training run — labels are  fp-IoU / 3D-IoU  (higher is better; "
                     "GT is the target)", fill=(0, 0, 0))
    for i, panels in enumerate(rows_out):
        y = HDR + i * (S + HDR)
        for j, (label, img, fi, vi) in enumerate(panels):
            x0 = PAD + j * (S + PAD)
            if img is not None:
                canvas.paste(img.convert("RGB").resize((S, S)), (x0, y + HDR))
            d.text((x0 + 4, y + 6), f"{label}", fill=(0, 0, 0))
            d.text((x0 + 4, y + 16), f"fp {fi:.3f}   3D {vi:.3f}", fill=(90, 90, 90))
    p = out / "a2_comparison.png"
    canvas.save(p)
    print("->", p)


if __name__ == "__main__":
    main()
