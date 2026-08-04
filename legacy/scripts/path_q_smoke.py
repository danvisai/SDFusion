"""End-to-end Path Q+ smoke: ControlNet gen -> Hunyuan3D-2 mesh.

Crops the gen column from the step-15k v2 train grid, saves it, runs
Hunyuan3D-2 on it, exports the mesh, and renders a PNG of the mesh
for inline display.
"""
import os
import sys
import time

import numpy as np
import torch
import trimesh
from PIL import Image


def render_mesh_png(mesh, out_path, image_size=512):
    """Render a trimesh.Trimesh to a single PNG via pytorch3d."""
    from pytorch3d.renderer import (
        FoVOrthographicCameras, MeshRasterizer, MeshRenderer,
        SoftPhongShader, RasterizationSettings, PointLights, BlendParams,
        TexturesVertex, look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    # Center + scale to unit
    v = v - v.mean(axis=0)
    v = v / max(float(np.abs(v).max()), 1e-9)
    # pytorch3d's "Y up" matches Hunyuan3D-2's output so just orient like the
    # ortho renderer we used in step 2.
    R, T = look_at_view_transform(dist=2.5, elev=20, azim=30, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T,
                                     scale_xyz=((1.0, 1.0, 1.0),))
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                            faces_per_pixel=1, bin_size=0)
    lights = PointLights(device=device, location=((2., 2., 2.),),
                         ambient_color=((0.45, 0.45, 0.45),),
                         diffuse_color=((0.55, 0.55, 0.55),),
                         specular_color=((0.05, 0.05, 0.05),))
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights,
                               blend_params=blend),
    )
    v_t = torch.from_numpy(v).to(device).unsqueeze(0)
    f_t = torch.from_numpy(f).to(device).unsqueeze(0)
    col = torch.full_like(v_t, 0.78)
    tex = TexturesVertex(verts_features=col)
    pmesh = Meshes(verts=v_t, faces=f_t, textures=tex)
    img = renderer(pmesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
    Image.fromarray((img * 255).astype(np.uint8), "RGB").save(out_path, optimize=True)


def main():
    out_dir = "outputs/path_q_smoke"
    os.makedirs(out_dir, exist_ok=True)

    grid_p = "Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/samples/step015000_train_grid.png"
    grid = Image.open(grid_p)
    print(f"[smoke] grid: {grid.size}", flush=True)
    W, H = grid.size
    tile = W // 3
    rows_to_test = [1, 2]      # the two strongest gens

    print("[smoke] loading Hunyuan3D-2 ...", flush=True)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    print("[smoke] pipeline loaded.", flush=True)

    rendered = []
    for row_i in rows_to_test:
        fp_box = (0, row_i * tile, tile, (row_i + 1) * tile)
        gen_box = (tile, row_i * tile, 2 * tile, (row_i + 1) * tile)
        fp_im = grid.crop(fp_box)
        gen_im = grid.crop(gen_box)
        fp_p = f"{out_dir}/row{row_i}_input_footprint.png"
        gen_p = f"{out_dir}/row{row_i}_controlnet_gen.png"
        fp_im.save(fp_p)
        gen_im.save(gen_p)

        print(f"\n[smoke] row {row_i}: feeding ControlNet gen into Hunyuan3D-2 ...", flush=True)
        t0 = time.time()
        mesh = pipe(image=gen_im)[0]
        dt = time.time() - t0
        glb_p = f"{out_dir}/row{row_i}_mesh.glb"
        mesh.export(glb_p)
        print(f"[smoke] row {row_i}: V={mesh.vertices.shape[0]:,} F={mesh.faces.shape[0]:,}  "
              f"file={os.path.getsize(glb_p)/1e6:.1f}MB  ({dt:.1f}s)", flush=True)

        # Render the mesh for visualization
        render_p = f"{out_dir}/row{row_i}_mesh_render.png"
        render_mesh_png(mesh, render_p)
        print(f"[smoke] row {row_i}: rendered -> {render_p}", flush=True)
        rendered.append((row_i, fp_p, gen_p, render_p))

    # Build a contact sheet: footprint | ControlNet gen | Hunyuan3D-2 mesh render
    tile = 384
    sheet = Image.new("RGB", (3 * tile, len(rendered) * tile), (255, 255, 255))
    for i, (row_i, fp_p, gen_p, render_p) in enumerate(rendered):
        for j, p in enumerate([fp_p, gen_p, render_p]):
            im = Image.open(p).convert("RGB").resize((tile, tile), Image.BICUBIC)
            sheet.paste(im, (j * tile, i * tile))
    sheet_p = f"{out_dir}/path_q_smoke_summary.png"
    sheet.save(sheet_p, optimize=True)
    print(f"\n[smoke] summary sheet -> {sheet_p}", flush=True)


if __name__ == "__main__":
    main()
