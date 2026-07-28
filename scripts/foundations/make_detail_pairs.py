"""Detailization pair factory — (coarse massing, detailed building) SDF pairs, free labels.

Fuel for a learned style-conditioned detailizer (DECOR-GAN/DECOLLAGE-style coarse->detailed
refiner, see chat 2026-06-11): every recipe building exists in BOTH forms for free —
  coarse = recipe massing only            (what the SDEdit prior outputs)
  fine   = massing + composer-driven sdf_detail (windows/door/roof/landmarks)
Both sampled over the SAME bbox so voxel grids align across resolutions.

Per sample: coarse 64^3 + fine 96^3 truncated SDFs (meters, trunc ±2.0, f16),
style_id, class_id, height, n_towers, seed. chunks=(1,...) (the h5 GPU-starvation gotcha).

Run (GPU, ~15-25 min for 2400):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/foundations/make_detail_pairs.py --n 2400
Output: data/detail_pairs_v1/pairs.h5 + preview montage outputs/detail_pairs_preview.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from models.networks import recipe_param_space as ps              # noqa: E402
from models.networks.diff_recipe import build_diff_recipe         # noqa: E402
from scene.sdf_edit import recipe_base_sdf                        # noqa: E402
from scene.composer_detail import compose_detail, get_composer    # noqa: E402
from scene.sdf_primitives import sample_grid                      # noqa: E402

TRUNC = 2.0          # meters
R_COARSE, R_FINE = 64, 96


def sample_footprint(rng) -> np.ndarray:
    """Random building footprint polygon (m, centered): rect / L / T / U + jitter + rotation."""
    w = rng.uniform(8, 22)
    d = rng.uniform(8, 24)
    kind = rng.choice(["rect", "rect", "L", "T", "U"])  # rect-heavy like real stock
    if kind == "rect":
        pts = [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]
    elif kind == "L":
        cw, cd = w * rng.uniform(0.35, 0.6), d * rng.uniform(0.35, 0.6)
        pts = [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2 - cd], [w / 2 - cw, d / 2 - cd],
               [w / 2 - cw, d / 2], [-w / 2, d / 2]]
    elif kind == "T":
        aw = w * rng.uniform(0.3, 0.5)
        pts = [[-aw / 2, -d / 2], [aw / 2, -d / 2], [aw / 2, 0], [w / 2, 0], [w / 2, d / 2],
               [-w / 2, d / 2], [-w / 2, 0], [-aw / 2, 0]]
    else:  # U
        cw, cd = w * rng.uniform(0.25, 0.4), d * rng.uniform(0.35, 0.6)
        pts = [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [w / 2 - cw, d / 2],
               [w / 2 - cw, d / 2 - cd], [-w / 2 + cw, d / 2 - cd], [-w / 2 + cw, d / 2],
               [-w / 2, d / 2]]
    p = np.asarray(pts, np.float32)
    p += rng.normal(0, 0.25, p.shape).astype(np.float32)            # vertex jitter
    # NOTE: axis-aligned only (no rotation) — sdf_detail places windows on the
    # axis-aligned bbox faces, so rotated footprints get no carved windows. The
    # proper fix (edge-following windows) is a follow-up; bands/plinth/roof/steps
    # are already polygon-clipped (2026-06-11).
    return p


ROOF_KINDS = ["flat", "gabled", "hipped", "dome"]  # one-hot buckets for dec["roof_shape"]
COND_DIM = 12 + 1 + len(ROOF_KINDS) + 5            # detail_vec + glazing + roof + landmarks


def encode_cond(dec) -> np.ndarray:
    """Composer decisions + final facade params -> the v2 conditioning vector.
    Detail fields normalized to [0,1] by (DETAIL_LO, DETAIL_HI)."""
    from scene.sdf_detail import DETAIL_LO, DETAIL_HI
    v = np.asarray(dec["detail_vec"], np.float32)
    v = (v - DETAIL_LO) / (DETAIL_HI - DETAIL_LO)
    s = str(dec["roof_shape"]).lower()
    roof = np.zeros(len(ROOF_KINDS), np.float32)
    idx = 0
    for i, k in enumerate(ROOF_KINDS[1:], 1):
        if k[:4] in s:
            idx = i
    roof[idx] = 1.0
    extra = np.array([dec["glazing"],
                      float(dec["dome"]), dec["dome_r"],
                      dec["n_towers"] / 4.0, (dec["tower_h_ratio"] - 1.0)], np.float32)
    steps = np.array([float(dec["steps"])], np.float32)
    out = np.concatenate([v.astype(np.float32), extra[:1], roof, extra[1:], steps])
    assert out.shape[0] == COND_DIM, out.shape
    return out


def pair_bbox(poly, height, n_towers):
    p = np.asarray(poly)
    x0, z0, x1, z1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    pad = 0.12 * max(x1 - x0, z1 - z0) + 1.0       # match recipe_inference._detailed_mesh
    head = height * (1.9 if n_towers else 1.5)
    return (x0 - pad, 0.0, z0 - pad, x1 + pad, head, z1 + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--out", default=os.path.join(REPO, "data/detail_pairs_v1/pairs.h5"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed0", type=int, default=0, help="offset sample seeds (corpus extension)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = args.device
    composer = get_composer(dev)
    recipes = {s: build_diff_recipe(s) for s in ps.STYLES}

    f = h5py.File(args.out, "w")
    f.create_dataset("coarse", (args.n, R_COARSE, R_COARSE, R_COARSE),
                     dtype="f2", chunks=(1, R_COARSE, R_COARSE, R_COARSE))
    f.create_dataset("fine", (args.n, R_FINE, R_FINE, R_FINE),
                     dtype="f2", chunks=(1, R_FINE, R_FINE, R_FINE))
    f.create_dataset("cond", (args.n, COND_DIM), dtype="f4")
    f.create_dataset("style_id", (args.n,), dtype="i1")
    f.create_dataset("class_id", (args.n,), dtype="i1")
    f.create_dataset("height", (args.n,), dtype="f4")
    f.create_dataset("n_towers", (args.n,), dtype="i1")
    f.create_dataset("seed", (args.n,), dtype="i4")
    f.attrs["trunc"] = TRUNC
    f.attrs["frame"] = "shared bbox per sample; SDF in meters; (D=z,H=y,W=x); y up"

    t0 = time.time()
    done = 0
    for i in range(args.n):
        seed = args.seed0 + i
        rng = np.random.default_rng(seed)
        style_i = int(rng.integers(0, len(ps.STYLES)))
        class_i = int(rng.integers(0, len(ps.CLASSES)))
        style, cls = ps.STYLES[style_i], ps.CLASSES[class_i]
        poly = sample_footprint(rng)
        height = float(rng.uniform(5.0, 26.0))
        _, default_fn, _ = recipes[style]
        params = default_fn(dev).detach().cpu().numpy()

        try:
            base = recipe_base_sdf(style, params, poly, height, device=dev)
            fine_sdf, _layout, dec = compose_detail(base, poly, height, cls, style=style,
                                                    seed=seed, composer=composer)
            bbox = pair_bbox(poly, height, dec["n_towers"])
            g_c = sample_grid(base, R_COARSE, bbox, device=dev)
            g_f = sample_grid(fine_sdf, R_FINE, bbox, device=dev)
        except Exception as ex:
            print(f"  [{i}] SKIP ({type(ex).__name__}: {str(ex)[:60]})")
            continue
        occ = float((g_f <= 0).float().mean())
        if not 0.02 < occ < 0.7:
            print(f"  [{i}] SKIP degenerate occ={occ:.3f}")
            continue

        f["coarse"][done] = g_c.clamp(-TRUNC, TRUNC).cpu().numpy().astype(np.float16)
        f["fine"][done] = g_f.clamp(-TRUNC, TRUNC).cpu().numpy().astype(np.float16)
        f["cond"][done] = encode_cond(dec)
        f["style_id"][done] = style_i
        f["class_id"][done] = class_i
        f["height"][done] = height
        f["n_towers"][done] = dec["n_towers"]
        f["seed"][done] = seed
        done += 1
        if done % 100 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{args.n}  ({rate:.1f}/s, eta {(args.n - done) / max(rate, 1e-9) / 60:.0f} min)", flush=True)

    f.attrs["n_valid"] = done
    f.close()
    print(f"[pairs] {done} pairs -> {args.out}  ({(time.time() - t0) / 60:.1f} min)")

    # ---- preview montage: coarse vs fine for 4 samples -------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    with h5py.File(args.out, "r") as h:
        idx = np.linspace(0, max(done - 1, 0), 4).astype(int)
        fig = plt.figure(figsize=(12, 6))
        for j, k in enumerate(idx):
            for col, (key, R) in enumerate([("coarse", R_COARSE), ("fine", R_FINE)]):
                g = h[key][k].astype(np.float32)
                ax = fig.add_subplot(2, 4, col * 4 + j + 1, projection="3d")
                sname = ps.STYLES[h["style_id"][k]]
                ax.set_title(f"{key} · {sname}", fontsize=7)
                if (g <= 0).sum() > 8:
                    v, fc, _, _ = measure.marching_cubes(g, level=0.0)
                    v = v[:, [2, 1, 0]]
                    ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                                    edgecolor="none", antialiased=True, shade=True)
                    lo, hi = v.min(), v.max()
                    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
                ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=20, azim=-60)
        png = os.path.join(REPO, "outputs", "detail_pairs_preview.png")
        fig.tight_layout(); fig.savefig(png, dpi=100)
        print(f"[pairs] preview -> {png}")


if __name__ == "__main__":
    main()
