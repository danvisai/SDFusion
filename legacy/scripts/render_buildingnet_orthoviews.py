"""
Render orthographic side-views of BuildingNet buildings, derived from SDFs.

For each <id>/ori_sample_grid.h5:
  - load (1, 64, 64, 64) SDF, in (z, y, x) layout (Y up, per recompute_footprints_from_sdf.py docstring)
  - marching cubes at iso=0 -> triangle mesh
  - render at a single fixed orthographic 3/4 view (elev=20°, azim=30°)
  - save 512x512 RGB PNG with white background

Output:
  data/BuildingNet_dataset_v0_1/buildingnet_renders/<phase>/<id>.png

These images will be the ControlNet training target ("what should a building
that *has this footprint and class* look like").

Run:
  python scripts/render_buildingnet_orthoviews.py --dry_run --limit 4   # preview 4 ids
  python scripts/render_buildingnet_orthoviews.py                       # full pass
  python scripts/render_buildingnet_orthoviews.py --phase val           # one split
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from skimage import measure


def sdf_to_mesh(sdf_zyx, iso=0.0):
    """Marching cubes on (D,D,D) SDF in (z,y,x) layout. Returns verts, faces in
    XYZ world coords, scaled so the longest axis is 1.0 and centered at origin
    with Y up."""
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sdf_zyx, level=iso,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return None, None
    if len(verts) == 0:
        return None, None
    # marching_cubes returns axes in the order of the input array, so verts are
    # (z, y, x). Convert to (x, y, z) world coords.
    verts = verts[:, [2, 1, 0]]
    # Center and unit-scale.
    verts = verts - verts.mean(axis=0, keepdims=True)
    extent = float(np.abs(verts).max())
    if extent > 1e-9:
        verts = verts / extent
    return verts.astype(np.float32), faces.astype(np.int64)


def make_renderer(device, image_size=512):
    """Single-view orthographic renderer with a soft phong shader and 3-light
    setup. Camera at elev=20°, azim=30°, looking at origin; 'building up' = +Y."""
    from pytorch3d.renderer import (
        FoVOrthographicCameras,
        MeshRasterizer,
        MeshRenderer,
        SoftPhongShader,
        RasterizationSettings,
        PointLights,
        BlendParams,
        look_at_view_transform,
    )

    R, T = look_at_view_transform(dist=2.5, elev=20, azim=30, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(
        device=device,
        R=R, T=T,
        scale_xyz=((1.0, 1.0, 1.0),),    # ortho box ~ unit
    )
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    lights = PointLights(
        device=device,
        location=((2.0, 2.0, 2.0),),
        ambient_color=((0.45, 0.45, 0.45),),
        diffuse_color=((0.55, 0.55, 0.55),),
        specular_color=((0.05, 0.05, 0.05),),
    )
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))  # white BG
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights,
                               blend_params=blend),
    )
    return renderer


def render_one(verts, faces, renderer, device):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex

    v_t = torch.from_numpy(verts).to(device).unsqueeze(0)
    f_t = torch.from_numpy(faces).to(device).unsqueeze(0)
    # Light-grey diffuse so the lighting actually shows the shape
    col = torch.full_like(v_t, 0.78)
    tex = TexturesVertex(verts_features=col)
    mesh = Meshes(verts=v_t, faces=f_t, textures=tex)
    img = renderer(mesh)             # (1, H, W, 4) RGBA in [0,1]
    rgb = img[0, ..., :3].clamp(0, 1).cpu().numpy()
    return (rgb * 255.0).astype(np.uint8)


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
    ap.add_argument("--limit", type=int, default=0,
                    help="render only the first N ids (smoke test)")
    ap.add_argument("--dry_run", action="store_true",
                    help="do everything but skip writing PNGs")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render even if PNG already exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")
    print(f"[*] image_size: {args.image_size}")

    res_dir = os.path.join(args.data_root, f"resolution_{args.res}")
    splits_dir = os.path.join(args.data_root, "splits")
    out_root = os.path.join(args.data_root, "buildingnet_renders")

    phases = ["train", "val", "test"] if args.phase == "all" else [args.phase]
    phase_ids = []
    for phase in phases:
        ids = load_phase_ids(splits_dir, phase)
        for mid in ids:
            phase_ids.append((phase, mid))
    print(f"[*] {len(phase_ids)} ids across {phases}")
    if args.limit:
        phase_ids = phase_ids[: args.limit]
        print(f"[*] limiting to first {args.limit}")

    renderer = make_renderer(device, image_size=args.image_size)

    n_done = n_skip = n_fail = 0
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

        with h5py.File(h5p, "r") as f:
            sdf = f["pc_sdf_sample"][:].reshape(args.res, args.res, args.res)

        verts, faces = sdf_to_mesh(sdf, iso=0.0)
        if verts is None or faces is None or len(faces) < 4:
            print(f"  [skip-empty] {mid}")
            n_fail += 1
            continue

        try:
            rgb = render_one(verts, faces, renderer, device)
        except Exception as e:
            print(f"  [render-fail] {mid}: {e}")
            n_fail += 1
            continue

        if not args.dry_run:
            Image.fromarray(rgb, "RGB").save(out_p, optimize=True)

        n_done += 1
        if (i + 1) % 100 == 0 or args.limit:
            print(f"  [{i+1:5d}/{len(phase_ids)}] {phase} {mid}  V={len(verts)} F={len(faces)}")

    print()
    print("=" * 70)
    print(f"  rendered  : {n_done}")
    print(f"  skipped   : {n_skip}  (no h5, or already rendered)")
    print(f"  failed    : {n_fail}")
    print(f"  output    : {out_root}/<phase>/<id>.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
