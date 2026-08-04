"""
Compute top-down height maps for BuildingNet, co-registered with the footprints.

For each id in <phase>_split.txt, loads (1, 64, 64, 64) SDF in (z, y, x) layout
and computes:

  height_map[z, x] = max y where (sdf <= 0)    (normalized to [0, 1])
  background      = 1.0 (white) where there is no building

Output is a 512x512 PNG, grayscale, with the building centered and ~10% white
margin so the building isn't crammed against the frame edges. Using bicubic
upsample for smooth gradients (so the model targets aren't pixelated).

This is the *same* (z, x) coordinate frame as the footprint — input and target
are spatially co-registered, so ControlNet just has to learn "fill in heights
given outline" rather than projecting between viewpoints.

Run:
    python scripts/render_buildingnet_heightmaps.py --dry_run --limit 4
    python scripts/render_buildingnet_heightmaps.py
"""
import argparse
import os
import sys

import h5py
import numpy as np
from PIL import Image


def height_map_from_sdf(sdf, iso=0.0):
    """sdf: (D, H, W) in (z, y, x). Returns (D, W) float32 height in [0, 1]
    plus a (D, W) bool mask of "building present here"."""
    inside = sdf <= iso
    present = inside.any(axis=1)                    # (D, W) = (z, x)
    H = sdf.shape[1]
    # Find the max y index where (sdf <= iso). For each (z, x):
    #   reverse along axis 1, argmax of the bool gives the offset from the top.
    flipped = inside[:, ::-1, :]
    last_y_from_top = flipped.argmax(axis=1).astype(np.float32)
    # Convert: top-of-cell index to "height from ground" in [0, 1].
    # If the building is present, height = (H - 1 - last_y_from_top) / (H - 1).
    height = (H - 1 - last_y_from_top) / max(H - 1, 1)
    height = np.where(present, height, 0.0)
    return height.astype(np.float32), present


def pack_with_margin(height, present, image_size=512, margin_frac=0.10):
    """Place the (D, W) height map into a centered region of a square image
    with white margin. Returns (image_size, image_size) uint8 grayscale where:
      - white (255)  = background (no building)
      - dark gray    = ground-level present
      - light gray   = tall building
    Specifically, we map height = 0..1 -> 255..0 (taller = darker = closer
    to camera in the depth-map convention Hunyuan3D-2 expects).
    """
    inner = int(round(image_size * (1.0 - 2 * margin_frac)))
    src_arr = np.where(present, 1.0 - height, 1.0).astype(np.float32)  # invert
    pil_inner = Image.fromarray(
        (src_arr.clip(0, 1) * 255).astype(np.uint8), "L"
    ).resize((inner, inner), Image.BICUBIC)

    canvas = Image.new("L", (image_size, image_size), color=255)
    off = (image_size - inner) // 2
    canvas.paste(pil_inner, (off, off))
    arr = np.asarray(canvas, dtype=np.uint8)
    return arr


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
    ap.add_argument("--margin_frac", type=float, default=0.10,
                    help="white margin on each side, as fraction of image_size")
    ap.add_argument("--phase", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--iso", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"[*] image_size={args.image_size}  margin={args.margin_frac:.2f}  iso={args.iso}",
          flush=True)

    res_dir = os.path.join(args.data_root, f"resolution_{args.res}")
    splits_dir = os.path.join(args.data_root, "splits")
    out_root = os.path.join(args.data_root, "buildingnet_heights")

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
            height, present = height_map_from_sdf(sdf, iso=args.iso)
            arr = pack_with_margin(height, present,
                                   image_size=args.image_size,
                                   margin_frac=args.margin_frac)
        except Exception as e:
            print(f"  [fail] {mid}: {e}")
            n_fail += 1
            continue

        if not args.dry_run:
            Image.fromarray(arr, "L").save(out_p, optimize=True)

        n_ok += 1
        if (i + 1) % 100 == 0 or args.limit:
            print(f"  [{i+1:5d}/{len(phase_ids)}] {phase} {mid}", flush=True)

    print()
    print("=" * 70)
    print(f"  rendered      : {n_ok}")
    print(f"  skipped       : {n_skip}")
    print(f"  failed        : {n_fail}")
    print(f"  output dir    : {out_root}/<phase>/<id>.png")
    print(f"  encoding      : white = no building, darker = taller (depth-map convention)")
    print("=" * 70)


if __name__ == "__main__":
    main()
