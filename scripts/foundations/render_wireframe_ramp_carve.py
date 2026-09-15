"""Visualize wireframe_ramp_carve.py's coherent-search comparison: GT vs. baseline (unbiased
beam search) vs. wireframe-guided (same search, wireframe Ramp candidates offered), voxel-
rendered side by side, with the region a wireframe candidate actually won outlined in red.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/render_wireframe_ramp_carve.py --city Adelaide \
     --stems 21g_22c_60 26_15 29c_29d_32 \
     --out outputs/wireframe_ramp_probe/carve_compare.png
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wireframe_ramp_carve import (   # noqa: E402
    building_to_sdf,
    fit_program_beam,
    height_field,
    match_wireframes_to_meshes,
    MESH_DIR,
    occupancy,
    RES,
    WIRE_DIR,
    wireframe_planes_in_voxel_space,
)


def crop_bounds(*occs, pad=3):
    any_occ = np.zeros_like(occs[0])
    for o in occs:
        any_occ |= o
    idx = np.argwhere(any_occ)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, np.array(any_occ.shape))
    return tuple(slice(lo[i], hi[i]) for i in range(3))


def render_panel(ax, occ, title, extra_mask=None):
    """`extra_mask`: voxels that are solid here but NOT in GT -- i.e. `extra`, the exact metric
    being compared, colored crimson. This is the actual mistake, not just "where the wireframe
    touched" (a first version highlighted the whole vertical column under the touched footprint,
    which is nearly identical between baseline/guided since only the roofline differs -- useless
    for showing what changed). Everything else solid is plain tan."""
    colors = np.empty(occ.shape, dtype=object)
    colors[occ] = "peru"
    if extra_mask is not None:
        colors[occ & extra_mask] = "crimson"
    ax.voxels(occ, facecolors=colors, edgecolor=(0, 0, 0, 0.15), linewidth=0.15)
    ax.set_title(title, fontsize=8)
    ax.set_box_aspect(occ.shape)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def find_pairing_for_stems(zm, zw, stems: set[str]):
    """Locate the (mname, wname) pair for each requested building stem, by re-running the same
    geometric matching wireframe_ramp_carve.py uses -- stems alone don't say which prefix group
    they came from, so every group with any of the stems is checked."""
    def prefix_of(name):
        return Path(name).stem.rsplit("_", 1)[0]

    wanted_prefixes = {s.rsplit("_", 1)[0] for s in stems}
    mesh_by_prefix, wire_by_prefix = {}, {}
    for n in zm.namelist():
        if n.endswith(".obj") and prefix_of(n) in wanted_prefixes:
            mesh_by_prefix.setdefault(prefix_of(n), []).append(n)
    for n in zw.namelist():
        if n.endswith(".obj") and prefix_of(n) in wanted_prefixes:
            wire_by_prefix.setdefault(prefix_of(n), []).append(n)

    found = {}
    for prefix in wanted_prefixes:
        if prefix not in mesh_by_prefix or prefix not in wire_by_prefix:
            continue
        pairing = match_wireframes_to_meshes(zm, zw, mesh_by_prefix[prefix], wire_by_prefix[prefix])
        for wname, mname in pairing.items():
            stem = Path(mname).stem
            if stem in stems:
                found[stem] = (mname, wname)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Adelaide")
    ap.add_argument("--stems", nargs="+", required=True,
                     help="building stems, e.g. 21g_22c_60 26_15 29c_29d_32 "
                          "(from wireframe_ramp_carve_adelaide.json's keys)")
    ap.add_argument("--out", default="outputs/wireframe_ramp_probe/carve_compare.png")
    args = ap.parse_args()

    mesh_zip_path = MESH_DIR / args.city / "mesh" / "obj.zip"
    if not mesh_zip_path.exists():
        mesh_zip_path = MESH_DIR / args.city / "obj" / "mesh.zip"
    wire_zip_path = WIRE_DIR / args.city / "wireframe" / "wireframe.zip"
    zm = zipfile.ZipFile(mesh_zip_path)
    zw = zipfile.ZipFile(wire_zip_path)

    pairs = find_pairing_for_stems(zm, zw, set(args.stems))
    missing = set(args.stems) - set(pairs)
    if missing:
        print(f"[warn] could not re-locate a confident pairing for: {sorted(missing)}")

    rows = []
    for stem in args.stems:
        if stem not in pairs:
            continue
        mname, wname = pairs[stem]
        m = trimesh.load(io.BytesIO(zm.read(mname)), file_type="obj", process=False)
        sdf, fp_u8, _ = building_to_sdf(m, R=RES)
        gt = sdf <= 0
        fp = fp_u8 > 0
        hf = height_field(gt, fp)
        if hf is None:
            print(f"[skip] {stem}: no height field")
            continue
        y0, y1, target = hf
        wf_planes = wireframe_planes_in_voxel_space(zm.read(mname), zw.read(wname).decode(), fp, y0)

        _, baseline_h = fit_program_beam(fp, y0, y1, target)
        baseline_occ = occupancy(fp, y0, baseline_h)

        guided_ops, guided_h = fit_program_beam(fp, y0, y1, target, wf_planes=wf_planes)
        guided_occ = occupancy(fp, y0, guided_h)

        # occupancy()'s array axis 1 is the TRUE vertical/height axis (see its own `yy =
        # np.arange(RES)[None, :, None]`) -- but ax.voxels() draws an array's LAST axis as
        # screen-vertical by mplot3d's default convention. Fed straight in, that put a
        # horizontal building dimension on screen-vertical and the real height axis sideways --
        # every render so far showed buildings lying on their side, not standing. (0, 2, 1)
        # swaps axes 1/2 so height lands last, i.e. actually vertical on screen.
        UP_LAST = (0, 2, 1)
        gt = np.transpose(gt, UP_LAST)
        baseline_occ = np.transpose(baseline_occ, UP_LAST)
        guided_occ = np.transpose(guided_occ, UP_LAST)

        extra_baseline = baseline_occ & ~gt   # solid here but NOT in GT -- the actual mistake
        extra_guided = guided_occ & ~gt

        sl = crop_bounds(gt, baseline_occ, guided_occ)
        rows.append((stem, gt[sl], baseline_occ[sl], guided_occ[sl],
                    extra_baseline[sl], extra_guided[sl]))
        print(f"[loaded] {stem}  extra voxels: baseline={int(extra_baseline.sum())} "
              f"guided={int(extra_guided.sum())}")

    if not rows:
        print("[done] nothing to render")
        return

    fig = plt.figure(figsize=(10, 3.4 * len(rows)))
    for i, (stem, gt_c, base_c, guide_c, eb_c, eg_c) in enumerate(rows):
        ax1 = fig.add_subplot(len(rows), 3, 3 * i + 1, projection="3d")
        render_panel(ax1, gt_c, f"{args.city}/{stem}\nGT")
        ax2 = fig.add_subplot(len(rows), 3, 3 * i + 2, projection="3d")
        render_panel(ax2, base_c, f"baseline\nred = extra ({int(eb_c.sum())} vox)",
                    extra_mask=eb_c)
        ax3 = fig.add_subplot(len(rows), 3, 3 * i + 3, projection="3d")
        render_panel(ax3, guide_c, f"wireframe-guided\nred = extra ({int(eg_c.sum())} vox)",
                    extra_mask=eg_c)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
