"""
Compute orthographic depth-view targets directly from BuildingNet SDFs.

For each id in <phase>_split.txt, loads the (1, 64, 64, 64) SDF from
ori_sample_grid.h5 and projects it into multiple orthographic views without
extracting a mesh. The output is a 3-channel PNG packing depth from three
viewpoints, intended as a richer ControlNet target than a single oblique
photo render.

SDF axis convention (verified empirically; see project_sdfusion_axes memory):
  axes (z, y, x), Y up, indexes (D=z, H=y, W=x).

Channels in the output PNG:
  R = top-down depth, normalized.       depth = first y where (sdf<=0), shape (D, W) = (z, x)
  G = front depth, normalized.          depth = first z where (sdf<=0), shape (H, W) = (y, x)
  B = side depth, normalized.           depth = first x where (sdf<=0), shape (D, H) = (z, y)

Each channel is upsampled (with anti-aliasing) from the source res to 512x512.
Background = 1.0 (white) so empty space reads as "infinitely far" and the
ControlNet sees a clean white canvas around the building.

Run:
    python scripts/render_buildingnet_depthviews.py --dry_run --limit 4
    python scripts/render_buildingnet_depthviews.py
"""
import argparse
import os
import sys

import h5py
import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def project_depth(inside, axis):
    """Direct depth projection of a boolean (sdf<=0) volume.

    inside: bool array of shape (D, H, W) in (z, y, x) layout.
    axis  : axis to *collapse* (the viewing axis). 0 -> top-down (collapse z?)
            1 -> top-down (collapse y, the up axis), 2 -> side (collapse x).

    Returns a 2D float depth map in [0, 1], where 0 = closest surface to the
    camera and 1 = no building in that ray. Shape is the remaining two axes
    after collapsing `axis`.
    """
    present = inside.any(axis=axis)
    # argmax of a bool along axis gives the first True; for rays with no True
    # it returns 0, so we mask those out below.
    first = inside.argmax(axis=axis).astype(np.float32)
    n = inside.shape[axis]
    depth = first / (n - 1)
    # Empty rays -> background (1.0)
    depth = np.where(present, depth, 1.0)
    return depth.astype(np.float32)


def upsample(arr, size):
    """Bicubic upsample a 2D float array in [0,1] to (size, size). Uses PIL
    so we get proper anti-aliasing on the diagonal edges (which is the whole
    reason we're not using marching cubes)."""
    img = (arr.clip(0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(img, "L").resize((size, size), Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def render_one(sdf, image_size=512, iso=0.0, mode="front",
               elev_deg=20.0, azim_deg=30.0):
    """sdf: (D, H, W) float in (z, y, x). Returns RGB uint8 (size, size, 3).

    mode='front'   : axis-aligned front view orthographic depth (grayscale).
                     No rotation -> no interpolation artifacts. Hunyuan3D-2
                     reads this as a clean single-view depth pass.
    mode='oblique' : single 3/4 view orthographic depth (grayscale). Cleaner
                     conceptually but the volume rotation introduces speckle
                     on non-watertight BuildingNet meshes.
    mode='multi3'  : R/G/B = top/front/side. Channels not spatially
                     co-registered — Hunyuan3D-2 reads this as alien input.
    """
    inside = sdf <= iso

    if mode == "front":
        # Camera looks along +Z (axis 0) at axis-aligned axes. For each (y, x)
        # ray, find the first z where the SDF crosses iso, with sub-voxel
        # interpolation for smooth depth.
        present = inside.any(axis=0)
        first_z = inside.argmax(axis=0).astype(np.float32)
        Dz = sdf.shape[0]
        z_below = np.clip(first_z.astype(int) - 1, 0, Dz - 1)
        yy, xx = np.indices(first_z.shape)
        s_below = sdf[z_below, yy, xx]
        s_at    = sdf[first_z.astype(int), yy, xx]
        denom   = s_below - s_at
        frac    = np.where(np.abs(denom) > 1e-6, (s_below - iso) / denom, 0.0)
        sub_z   = first_z - frac
        sub_z   = np.where(first_z > 0, sub_z, first_z)
        depth   = np.where(present, sub_z / max(Dz - 1, 1), 1.0)
        depth   = np.flipud(depth)        # y-up convention
        g = upsample(depth.astype(np.float32), image_size)
        return np.stack([g, g, g], axis=-1)

    if mode == "multi3":
        top_depth   = project_depth(inside, axis=1)
        front_depth = np.flipud(project_depth(inside, axis=0))
        side_depth  = np.flipud(project_depth(inside, axis=2).T)
        r = upsample(top_depth,   image_size)
        g = upsample(front_depth, image_size)
        b = upsample(side_depth,  image_size)
        return np.stack([r, g, b], axis=-1)

    # mode == "oblique" — rotate the *smooth SDF itself* so a chosen view
    # direction aligns with axis 0, then threshold once. Rotating the binary
    # mask (the previous approach) and re-thresholding double-quantizes and
    # produces speckle/stick artifacts at oblique angles. Rotating the SDF
    # preserves the smooth signed-distance gradient, so the iso surface stays
    # crisp regardless of rotation angle.
    pad_val = float(sdf.max())   # outside = far positive, so padding stays "outside"
    sdf_r = sdf.astype(np.float32)
    # azim around Y (axis 1) -> rotate in the z-x plane (axes 0, 2)
    sdf_r = ndi.rotate(sdf_r, angle=azim_deg, axes=(0, 2),
                       reshape=True, order=3, mode="constant", cval=pad_val)
    # elev around the new X axis -> rotate in the z-y plane (axes 0, 1)
    sdf_r = ndi.rotate(sdf_r, angle=elev_deg, axes=(0, 1),
                       reshape=True, order=3, mode="constant", cval=pad_val)
    # Camera looks along +Z (axis 0); per (y, x) ray, find first z where SDF
    # crosses iso. Use the smooth SDF for sub-voxel depth: interpolate the
    # crossing within the cell.
    inside = sdf_r <= iso
    present = inside.any(axis=0)
    first_z = inside.argmax(axis=0).astype(np.float32)
    Dz = sdf_r.shape[0]
    # Sub-voxel refinement: for cells where we just stepped from outside (>iso)
    # to inside (<=iso), interpolate the iso-crossing.
    z_idx_below = np.clip(first_z.astype(int) - 1, 0, Dz - 1)
    yy, xx = np.indices(first_z.shape)
    s_below = sdf_r[z_idx_below, yy, xx]
    s_at    = sdf_r[first_z.astype(int), yy, xx]
    denom   = s_below - s_at
    frac    = np.where(np.abs(denom) > 1e-6, (s_below - iso) / denom, 0.0)
    sub_z   = first_z - frac
    sub_z   = np.where(first_z > 0, sub_z, first_z)
    depth   = np.where(present, sub_z / max(Dz - 1, 1), 1.0)
    depth   = np.flipud(depth)
    # Optional: tiny morphological closing on the depth-foreground mask to
    # eliminate single-pixel speckle without changing geometry.
    fg = (depth < 1.0)
    fg = ndi.binary_closing(fg, iterations=1)
    depth = np.where(fg, depth, 1.0)
    g = upsample(depth.astype(np.float32), image_size)
    return np.stack([g, g, g], axis=-1)


def load_phase_ids(splits_dir, phase):
    p = os.path.join(splits_dir, f"{phase}_split.txt")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--phase", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--iso", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mode", default="front", choices=["front", "oblique", "multi3"])
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=30.0)
    args = ap.parse_args()

    print(f"[*] image_size={args.image_size}  iso={args.iso}", flush=True)

    res_dir = os.path.join(args.data_root, f"resolution_{args.res}")
    splits_dir = os.path.join(args.data_root, "splits")
    out_root = os.path.join(args.data_root, "buildingnet_depths")

    phases = ["train", "val", "test"] if args.phase == "all" else [args.phase]
    phase_ids = []
    for phase in phases:
        for mid in load_phase_ids(splits_dir, phase):
            phase_ids.append((phase, mid))
    print(f"[*] {len(phase_ids)} ids across {phases}", flush=True)
    if args.limit:
        phase_ids = phase_ids[: args.limit]
        print(f"[*] limiting to first {args.limit}", flush=True)

    n_ok = n_skip = n_fail = 0
    for i, (phase, mid) in enumerate(phase_ids):
        h5p = os.path.join(res_dir, mid, "ori_sample_grid.h5")
        if not os.path.exists(h5p):
            n_skip += 1
            continue

        out_dir = os.path.join(out_root, phase)
        os.makedirs(out_dir, exist_ok=True)
        out_p = os.path.join(out_dir, f"{mid}.png")
        if os.path.exists(out_p) and not args.overwrite and not args.dry_run:
            n_skip += 1
            continue

        try:
            with h5py.File(h5p, "r") as f:
                sdf = f["pc_sdf_sample"][:].reshape(args.res, args.res, args.res)
            rgb = render_one(sdf, image_size=args.image_size, iso=args.iso,
                             mode=args.mode, elev_deg=args.elev, azim_deg=args.azim)
        except Exception as e:
            print(f"  [fail] {mid}: {e}")
            n_fail += 1
            continue

        if not args.dry_run:
            Image.fromarray(rgb, "RGB").save(out_p, optimize=True)

        n_ok += 1
        if (i + 1) % 100 == 0 or args.limit:
            print(f"  [{i+1:5d}/{len(phase_ids)}] {phase} {mid}", flush=True)

    print()
    print("=" * 70)
    print(f"  rendered     : {n_ok}")
    print(f"  skipped      : {n_skip}")
    print(f"  failed       : {n_fail}")
    print(f"  output dir   : {out_root}/<phase>/<id>.png")
    print(f"  channels     : R=top-down  G=front  B=side")
    print("=" * 70)


if __name__ == "__main__":
    main()
