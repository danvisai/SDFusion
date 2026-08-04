"""Re-render the existing path_q_smoke meshes from multiple angles with the
camera zoomed out, so the full building is visible with margin.

Note: Hunyuan3D-2's exported GLBs trip the strict-length check in trimesh, so
we go through pygltflib + reconstruct a Trimesh from raw V/F.
"""
import os, glob
import numpy as np
import torch
import trimesh
import pygltflib
from PIL import Image


def _accessor(gltf, idx):
    acc = gltf.accessors[idx]
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    off = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    type_dtype = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,
                  5125:np.uint32,5126:np.float32}
    type_n = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}
    n = type_n[acc.type]
    arr = np.frombuffer(blob, dtype=type_dtype[acc.componentType],
                        count=acc.count*n, offset=off)
    return arr.reshape(acc.count, n) if n > 1 else arr


def load_glb(path):
    g = pygltflib.GLTF2().load(path)
    p = g.meshes[0].primitives[0]
    v = _accessor(g, p.attributes.POSITION).astype(np.float32)
    f = _accessor(g, p.indices).reshape(-1, 3).astype(np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def render_mesh(mesh, image_size=512, dist=3.5, elev=20, azim=30, scale=1.4):
    from pytorch3d.renderer import (
        FoVOrthographicCameras, MeshRasterizer, MeshRenderer,
        SoftPhongShader, RasterizationSettings, PointLights, BlendParams,
        TexturesVertex, look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    v = v - v.mean(axis=0)
    v = v / max(float(np.abs(v).max()), 1e-9)

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(
        device=device, R=R, T=T,
        scale_xyz=((scale, scale, scale),),  # > 1 zooms OUT in ortho
    )
    rs = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                               faces_per_pixel=1, bin_size=0)
    lights = PointLights(device=device, location=((2., 2., 2.),),
                         ambient_color=((0.45, 0.45, 0.45),),
                         diffuse_color=((0.55, 0.55, 0.55),),
                         specular_color=((0.05, 0.05, 0.05),))
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rs),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights,
                               blend_params=blend),
    )
    v_t = torch.from_numpy(v).to(device).unsqueeze(0)
    f_t = torch.from_numpy(f).to(device).unsqueeze(0)
    col = torch.full_like(v_t, 0.78)
    pmesh = Meshes(verts=v_t, faces=f_t, textures=TexturesVertex(verts_features=col))
    img = renderer(pmesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def main():
    out = "outputs/path_q_smoke"
    glbs = sorted(glob.glob(f"{out}/row*_mesh.glb"))
    print(f"meshes: {glbs}")
    angles = [
        ("front",      0,    20),
        ("3q-right",   30,   20),
        ("side-right", 90,   10),
        ("top",        45,   75),
    ]

    rows = []
    for glb in glbs:
        name = os.path.basename(glb).replace("_mesh.glb", "")
        mesh = load_glb(glb)
        print(f"  {name}: V={mesh.vertices.shape[0]:,}  F={mesh.faces.shape[0]:,}")
        row = []
        for label, azim, elev in angles:
            arr = render_mesh(mesh, image_size=512, dist=3.5, elev=elev, azim=azim, scale=1.5)
            row.append(arr)
            Image.fromarray(arr).save(f"{out}/{name}_view_{label}.png")
        rows.append((name, row))

    # Build contact sheet: each row is one mesh, columns = angles
    tile = 384
    n_cols = len(angles)
    sheet = Image.new("RGB", (n_cols * tile, len(rows) * tile), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for ri, (name, row) in enumerate(rows):
        for ci, arr in enumerate(row):
            im = Image.fromarray(arr).resize((tile, tile), Image.BICUBIC)
            sheet.paste(im, (ci * tile, ri * tile))
            draw.text((ci * tile + 4, ri * tile + 4),
                      f"{name} / {angles[ci][0]}", fill="black")
    sheet_p = f"{out}/path_q_smoke_multiview.png"
    sheet.save(sheet_p, optimize=True)
    print(f"\nmulti-view sheet -> {sheet_p}")


if __name__ == "__main__":
    main()
