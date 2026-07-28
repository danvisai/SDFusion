"""Detailizer deploy-eval — run the learned detailizer on REAL pipeline massing, photo out.

The training-set montages (train_detailizer.py val_*.png) show the easy case (clean recipe
coarse). This evaluates the case that matters at deploy time: the SOFT massing the SDEdit
prior / snap emits. For a few footprints x styles:
    row 1: massing from the running server (plain or sdedit)
    row 2: learned detailizer output (this is the new thing)
    row 3: the procedural ② bake (compose_detail) = the teacher, for reference

Run (server up on :8099, detailizer trained):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/foundations/eval_detailizer_on_prior.py \
      --ckpt outputs/detailizer_v1/detailizer.pth [--sdedit]
Output: outputs/detailizer_v1/deploy_eval_<UTC>.png
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import sys
import urllib.request

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure

from scripts.foundations.train_detailizer import DetailizerUNet, up, R_F   # noqa: E402

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
STYLES = ["modern", "colonial", "victorian", "industrial", "craftsman",
          "mediterranean", "contemporary", "public_civic"]
CLASSES = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]
CASES = [  # (footprint, style, class, height)
    ([[-7, -9], [7, -9], [7, 9], [-7, 9]], "modern", "RESIDENTIAL", 16),
    ([[-9, -7], [9, -7], [9, 7], [-9, 7]], "victorian", "RESIDENTIAL", 12),
    ([[-10, -8], [10, -8], [10, 8], [-10, 8]], "public_civic", "RELIGIOUS", 14),
    ([[-8, -10], [8, -10], [8, 10], [-8, 10]], "industrial", "COMMERCIAL", 18),
]


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def draw(ax, g, title):
    ax.set_title(title, fontsize=7)
    if (g <= 0).sum() > 8:
        try:
            v, fc, _, _ = measure.marching_cubes(g.astype(np.float32), level=0.0)
            v = v[:, [2, 1, 0]]
            ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                            edgecolor="none", antialiased=True, shade=True)
            lo, hi = v.min(), v.max()
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
        except Exception:
            pass
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=20, azim=-60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "outputs/detailizer_v1/detailizer.pth"))
    ap.add_argument("--sdedit", action="store_true",
                    help="use AI massing (sdedit_strength=0.45) instead of plain recipe massing")
    ap.add_argument("--trunc", type=float, default=2.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device)
    cond_dim = int(ck.get("cond_dim", 0))
    G = DetailizerUNet(cond_dim=cond_dim).to(device)
    G.load_state_dict(ck["G"]); G.eval()
    print(f"[deploy-eval] detailizer @ iter {ck.get('iter')} · massing="
          f"{'sdedit' if args.sdedit else 'plain'}")

    from scene.sdf_edit import recipe_base_sdf
    from scene.composer_detail import compose_detail, get_composer
    from scene.sdf_primitives import sample_grid
    composer = get_composer(device)

    n = len(CASES)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8), subplot_kw={"projection": "3d"})
    for j, (fp, style, cls, h) in enumerate(CASES):
        body = {"footprint": fp, "style": style, "building_class": cls, "height": h}
        if args.sdedit:
            body["sdedit_strength"] = 0.45
        r = post("/building_sdf", body)
        g_in = np.frombuffer(base64.b64decode(r["sdf_b64"]), dtype="<f4").reshape(64, 64, 64)
        # The server volume lives in a centered CUBE frame (center, half-extent scale);
        # the detailizer was trained in the footprint-bbox frame of make_detail_pairs.
        # Resample cube -> pair frame and convert SDF values to meters.
        p = np.asarray(fp, np.float32)
        pad = 0.12 * max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) + 1.0
        bbox_in = (p[:, 0].min() - pad, 0.0, p[:, 1].min() - pad,
                   p[:, 0].max() + pad, h * 1.5, p[:, 1].max() + pad)
        x0, y0, z0, x1, y1, z1 = bbox_in
        c0 = np.asarray(r["center"], np.float32)
        gz = torch.linspace(z0, z1, 64); gy = torch.linspace(y0, y1, 64)
        gx = torch.linspace(x0, x1, 64)
        Z, Y, X = torch.meshgrid(gz, gy, gx, indexing="ij")
        # world -> cube coords in [-1,1]; grid_sample wants (x,y,z) on a (D,H,W) volume
        gridc = torch.stack([(X - c0[0]) / r["scale"], (Y - c0[1]) / r["scale"],
                             (Z - c0[2]) / r["scale"]], dim=-1)[None]
        vol = torch.from_numpy(g_in.copy())[None, None]
        g_res = torch.nn.functional.grid_sample(vol, gridc, mode="bilinear",
                                                padding_mode="border",
                                                align_corners=True)[0, 0].numpy()
        g_m = np.clip(g_res * r["scale"], -args.trunc, args.trunc)

        # procedural teacher for reference (composer bake on the same footprint) — also
        # the source of the v2 conditioning (composer decides, detailizer renders)
        dec = None
        try:
            base = recipe_base_sdf(style, r["recipe_params"], np.asarray(fp, np.float32),
                                   h, device=device)
            fine_sdf, _, dec = compose_detail(base, np.asarray(fp, np.float32), h, cls,
                                              style=style, seed=0, composer=composer)
            p = np.asarray(fp); pad = 0.12 * (p.max() - p.min()) + 1.0
            head = h * (1.9 if dec["n_towers"] else 1.5)
            bbox = (p[:, 0].min() - pad, 0.0, p[:, 1].min() - pad,
                    p[:, 0].max() + pad, head, p[:, 1].max() + pad)
            g_proc = sample_grid(fine_sdf, R_F, bbox, device=device).cpu().numpy()
        except Exception as ex:
            print(f"  [case {j}] procedural reference failed: {ex}")
            g_proc = np.ones((R_F,) * 3, np.float32)

        with torch.no_grad():
            c = torch.from_numpy(g_m.copy())[None, None].to(device) / args.trunc
            s = torch.tensor([STYLES.index(style)], device=device)
            k = torch.tensor([CLASSES.index(cls)], device=device)
            cd = None
            if cond_dim:
                from scripts.foundations.make_detail_pairs import encode_cond
                cd = torch.from_numpy(encode_cond(dec))[None].to(device)
            pred = G(up(c), s, k, cd)[0, 0].cpu().numpy()

        draw(axes[0, j], g_m, f"massing in · {style}/{cls[:3]}")
        draw(axes[1, j], pred, "LEARNED detailizer")
        draw(axes[2, j], g_proc, "procedural ② (reference)")
        print(f"  [case {j}] {style}: done")

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(os.path.dirname(args.ckpt),
                       f"deploy_eval_{'sdedit' if args.sdedit else 'plain'}_{stamp}.png")
    fig.suptitle(f"detailizer deploy eval · {'sdedit' if args.sdedit else 'plain'} massing · "
                 f"iter {ck.get('iter')}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=110)
    print(f"[deploy-eval] -> {out}")


if __name__ == "__main__":
    main()
