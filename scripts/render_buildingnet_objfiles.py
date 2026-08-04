"""
Render orthographic 3/4 views of BuildingNet buildings DIRECTLY from the
real OBJ_MODELS/<id>.obj files (full-fidelity, not marching-cubes-from-SDF).

The previous renderer did marching cubes on the (hollow, walls-only) SDFs and
got fragments. The real OBJs have full geometry — walls, roofs, windows,
ornaments — so rendering from them gives complete, recognizable buildings.

Output:
  data/BuildingNet_dataset_v0_1/buildingnet_objrenders/<phase>/<id>.png

Run:
  python scripts/render_buildingnet_objfiles.py --limit 4   # smoke
  python scripts/render_buildingnet_objfiles.py             # full pass on splits
"""
import argparse
import os

import numpy as np
import torch
import trimesh
from PIL import Image


def load_obj_as_trimesh(obj_path):
    """trimesh.load can return Scene; concat into a single mesh."""
    loaded = trimesh.load(obj_path, force="mesh", process=False)
    if loaded is None or not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
        return None
    return loaded


def make_renderer(device, image_size=512, scale=0.7):
    """Single-view orthographic renderer at elev=20, azim=30. Bigger scale =
    zoom out (wider FOV). 0.7 gives ~15% margin around a unit-sphere mesh."""
    from pytorch3d.renderer import (
        FoVOrthographicCameras, MeshRasterizer, MeshRenderer, SoftPhongShader,
        RasterizationSettings, PointLights, BlendParams, look_at_view_transform,
    )

    R, T = look_at_view_transform(dist=2.5, elev=20, azim=30, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T,
                                     scale_xyz=((scale, scale, scale),))
    rs = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                               faces_per_pixel=1, bin_size=0)
    lights = PointLights(device=device, location=((2.0, 2.0, 2.0),),
                         ambient_color=((0.45, 0.45, 0.45),),
                         diffuse_color=((0.55, 0.55, 0.55),),
                         specular_color=((0.05, 0.05, 0.05),))
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rs),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights,
                               blend_params=blend),
    )


def render_one(mesh, renderer, device):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    # Center + unit-scale (longest axis -> 1)
    v = v - (v.max(0) + v.min(0)) / 2
    extent = float(np.abs(v).max())
    if extent > 1e-9:
        v = v / extent
    # Y-up convention: BuildingNet's normalize-by-pc_norm centers around origin
    # but Y is up. Rotate so building stands upright in the renderer (Y already up).
    v_t = torch.from_numpy(v).to(device).unsqueeze(0)
    f_t = torch.from_numpy(f).to(device).unsqueeze(0)
    col = torch.full_like(v_t, 0.78)
    pmesh = Meshes(verts=v_t, faces=f_t, textures=TexturesVertex(verts_features=col))
    img = renderer(pmesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def load_phase_ids(splits_dir, phase):
    p = os.path.join(splits_dir, f"{phase}_split.txt")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--phase", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}, image_size={args.image_size}")

    obj_dir = os.path.join(args.data_root, "OBJ_MODELS")
    splits_dir = os.path.join(args.data_root, "splits")
    out_root = os.path.join(args.data_root, "buildingnet_objrenders")

    phases = ["train", "val", "test"] if args.phase == "all" else [args.phase]
    phase_ids = []
    for phase in phases:
        for mid in load_phase_ids(splits_dir, phase):
            phase_ids.append((phase, mid))
    print(f"[*] {len(phase_ids)} ids across {phases}")
    if args.limit:
        phase_ids = phase_ids[: args.limit]

    renderer = make_renderer(device, image_size=args.image_size)

    n_ok = n_skip = n_fail = 0
    for i, (phase, mid) in enumerate(phase_ids):
        obj_p = os.path.join(obj_dir, f"{mid}.obj")
        if not os.path.exists(obj_p):
            n_skip += 1
            continue

        out_dir = os.path.join(out_root, phase)
        os.makedirs(out_dir, exist_ok=True)
        out_p = os.path.join(out_dir, f"{mid}.png")
        if os.path.exists(out_p) and not args.overwrite:
            n_skip += 1
            continue

        mesh = load_obj_as_trimesh(obj_p)
        if mesh is None or len(mesh.faces) < 4:
            print(f"  [skip-empty] {mid}")
            n_fail += 1
            continue

        try:
            rgb = render_one(mesh, renderer, device)
        except Exception as e:
            print(f"  [render-fail] {mid}: {e}")
            n_fail += 1
            continue

        Image.fromarray(rgb, "RGB").save(out_p, optimize=True)
        n_ok += 1
        if (i + 1) % 100 == 0 or args.limit:
            print(f"  [{i+1:5d}/{len(phase_ids)}] {phase} {mid}  V={len(mesh.vertices)} F={len(mesh.faces)}")

    print()
    print("=" * 70)
    print(f"  rendered  : {n_ok}")
    print(f"  skipped   : {n_skip}")
    print(f"  failed    : {n_fail}")
    print(f"  output    : {out_root}/<phase>/<id>.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
