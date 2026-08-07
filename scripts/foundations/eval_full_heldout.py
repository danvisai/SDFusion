"""#85: score the FULL held-out set (714), not the 7% sample.

One pass computes everything so the A2 projection is paid for once:
  * the harness's standard columns (fp-IoU, missing, extra, 3D IoU) for every arm
  * criterion 2's new split -- fringe / spill / uncovered -- at tolerances 0,1,2,3(=s*)
  * pass rates across allowances, with binomial confidence intervals

s* = the project's detail scale, ADR 0004: 1.0 m = ~3 voxels @64^3. Fixed a priori, not fitted.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(ROOT))

from scripts.foundations.eval_massing_arms import (RES, _vertical_extent, blockout_sdf,  # noqa: E402
                                                   pick_ids, score_arm)
from scripts.foundations.vecset_ceiling_probe import TRUNC, verts_to_world               # noqa: E402
from scripts.foundations.baseline_gate_eval import mesh_sdf_surface                      # noqa: E402

H5 = ROOT / "data/real_massing_v1/real.h5"
LAT = ROOT / "data/real_massing_v1/vecset_latents.h5"
CKPT = ROOT / "logs_building/vecset_v4_surf/vecset_denoiser_step240000.pth"
OUT = ROOT / "execution/artifacts/criterion2_full714.json"
TOLS = (0, 1, 2, 3)
S_STAR = 3


def split_fp(ref, proj, N):
    from scipy import ndimage
    A = ref.sum()
    if N == 0:
        band = np.zeros_like(ref)
    else:
        band = (ndimage.binary_dilation(ref, iterations=N)
                & ~ndimage.binary_erosion(ref, iterations=N))
    spill = ((proj & ~ref) & ~band).sum()
    unc = ((ref & ~proj) & ~band).sum()
    fringe = ((proj ^ ref) & band).sum()
    return float(fringe / A), float(spill / A), float(unc / A)


def main():
    import h5py
    from models.shape_codec import Building, DoraCodec
    from models.networks.vecset_denoiser import VecsetDenoiser
    from models.networks.vecset_projection import SetSDEdit
    from scripts.foundations.dora_roundtrip_probe import load_dora

    dev = "cuda"
    codec = DoraCodec(load_dora(dev))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ca = ck["args"]
    net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=ca["width"], depth=ca["depth"],
                         heads=ca["heads"], footprint_res=ck["footprint_res"]).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    op = SetSDEdit(net, timesteps=ca["timesteps"])
    mu, sd = ck["latent_mu"], ck["latent_sd"]

    cand, lat_of = pick_ids(LAT, None)
    print(f"[ids] {len(cand)} held-out candidates -- scoring ALL of them", flush=True)
    with h5py.File(LAT, "r") as lf:
        fp_of = {b: np.asarray(lf["footprint"][lat_of[b]]) for b in cand}
        ht_of = {b: float(lf["height_m"][lat_of[b]]) for b in cand}
        rg_of = {b: int(lf["region"][lat_of[b]]) for b in cand}
        lat_np = {b: np.asarray(lf["latent"][lat_of[b]], np.float32) for b in cand}

    rows, t0 = [], time.time()
    with h5py.File(H5, "r") as g:
        for k, bid in enumerate(cand):
            fp = fp_of[bid]
            if not fp.any():
                continue
            gfld = np.asarray(g["sdf"][bid], np.float32)
            gocc = gfld <= 0
            ext = _vertical_extent(gocc)
            if ext is None:
                continue
            bo = blockout_sdf(fp, *ext)
            if bo is None:
                continue
            bv, bf = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
            if bv is None:
                continue

            z0 = (codec.encode(Building(verts=verts_to_world(bv), faces=bf)).float() - mu) / sd
            zp = op.project(blockout=z0,
                            footprint=torch.from_numpy(fp.astype(np.float32))[None, None].to(dev),
                            height=torch.tensor([ht_of[bid]], device=dev),
                            region=torch.tensor([rg_of[bid]], device=dev),
                            strength=0.5, steps=20, guidance=1.0, seed=bid)
            with torch.no_grad():
                a2f = codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]
                ccf = codec.decode_grid(
                    torch.from_numpy(lat_np[bid])[None].to(dev), RES).cpu().numpy()[0, 0]

            ref = fp.astype(bool)
            r = dict(bid=int(bid), region=rg_of[bid])
            for name, fld in (("blockout", bo), ("codec_ceiling", ccf), ("a2", a2f)):
                s = score_arm(fld, gocc, fp)
                r[f"{name}_fp"] = s["fp_iou"]; r[f"{name}_miss"] = s["missing"]
                r[f"{name}_extra"] = s["extra"]; r[f"{name}_iou"] = s["vol_iou"]
            proj = (a2f <= 0).any(axis=1)
            for N in TOLS:
                fr, sp, un = split_fp(ref, proj, N)
                r[f"fringe{N}"] = fr; r[f"spill{N}"] = sp; r[f"unc{N}"] = un
            rows.append(r)
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(cand)}  ({el:.0f}s, eta {el/(k+1)*(len(cand)-k-1):.0f}s)",
                      flush=True)

    print(f"\n[done] {len(rows)} buildings scored in {time.time()-t0:.0f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(n=len(rows), ckpt=str(CKPT.relative_to(ROOT)), strength=0.5,
                                   s_star_voxels=S_STAR, rows=rows), indent=1))

    def col(k):
        return np.array([r[k] for r in rows], float)

    n = len(rows)
    print(f"\n=== STANDARD COLUMNS on the FULL held-out set (n={n}) ===")
    print(f"{'arm':<16}{'fp-IoU med':>12}{'miss med':>10}{'extra med':>11}{'3D IoU med':>12}")
    for a in ("blockout", "codec_ceiling", "a2"):
        print(f"{a:<16}{np.median(col(a+'_fp')):>12.4f}{np.median(col(a+'_miss')):>10.4f}"
              f"{np.median(col(a+'_extra')):>11.4f}{np.median(col(a+'_iou')):>12.4f}")

    print(f"\n=== CRITERION 2 SPLIT, A2 s=0.5, by boundary tolerance (n={n}) ===")
    print(f"{'tol':>5}{'fringe med':>12}{'spill med':>11}{'spill mean':>12}{'spill p90':>11}"
          f"{'unc med':>9}")
    for N in TOLS:
        star = "  <- s*" if N == S_STAR else ""
        print(f"{N:>5}{np.median(col(f'fringe{N}')):>12.4f}{np.median(col(f'spill{N}')):>11.4f}"
              f"{col(f'spill{N}').mean():>12.4f}{np.percentile(col(f'spill{N}'),90):>11.4f}"
              f"{np.median(col(f'unc{N}')):>9.4f}{star}")

    sp, un = col(f"spill{S_STAR}"), col(f"unc{S_STAR}")
    print(f"\n=== PASS RATE at s*={S_STAR} voxels, with 95% CI (n={n}) ===")
    print(f"{'allowance':>10}{'pass':>8}{'rate':>8}{'95% CI':>16}")
    for a in (0.00, 0.02, 0.03, 0.05, 0.10):
        ok = (sp <= a) & (un <= a)
        p = ok.mean(); se = (p * (1 - p) / n) ** 0.5
        print(f"{a*100:>9.0f}%{int(ok.sum()):>8}{p*100:>7.1f}%"
              f"   [{(p-1.96*se)*100:>4.1f}%, {(p+1.96*se)*100:>4.1f}%]")

    print("\n=== is the n=48 sample representative? region mix ===")
    reg = col("region").astype(int)
    first48 = reg[:48]
    for r in sorted(set(reg.tolist())):
        print(f"  region {r}: full {int((reg==r).sum()):>4}/{n} ({(reg==r).mean()*100:>4.1f}%)   "
              f"first-48 {int((first48==r).sum()):>2}/48 ({(first48==r).mean()*100:>4.1f}%)")


if __name__ == "__main__":
    main()
