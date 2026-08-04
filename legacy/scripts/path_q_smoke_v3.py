"""Smoke: v3 (depth-target) ControlNet -> Hunyuan3D-2 mesh.

Same 2 footprints as the v2 smoke for direct comparison. Outputs to
outputs/path_q_smoke_v3/.
"""
import os, time
import numpy as np
import torch
import trimesh
import pygltflib
from PIL import Image, ImageDraw

CKPT_DIR = "Logs_GT/CN-2026-05-06T17-56-17-footprint2depth-15k-v3/ckpt/controlnet-015000"
SD_BASE  = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUT_DIR  = "outputs/path_q_smoke_v3"


def _accessor(g, idx):
    a = g.accessors[idx]; bv = g.bufferViews[a.bufferView]
    blob = g.binary_blob()
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    td = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
    tn = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}
    n = tn[a.type]
    arr = np.frombuffer(blob, dtype=td[a.componentType], count=a.count*n, offset=off)
    return arr.reshape(a.count, n) if n > 1 else arr


def load_glb(p):
    g = pygltflib.GLTF2().load(p)
    pr = g.meshes[0].primitives[0]
    v = _accessor(g, pr.attributes.POSITION).astype(np.float32)
    f = _accessor(g, pr.indices).reshape(-1, 3).astype(np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def render_mesh(mesh, image_size=512, dist=2.5, elev=20, azim=30, scale=0.6):
    from pytorch3d.renderer import (FoVOrthographicCameras, MeshRasterizer, MeshRenderer,
        SoftPhongShader, RasterizationSettings, PointLights, BlendParams,
        TexturesVertex, look_at_view_transform)
    from pytorch3d.structures import Meshes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    v = v - v.mean(axis=0); v = v / max(float(np.abs(v).max()), 1e-9)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=((0,0,0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T,
                                     scale_xyz=((scale,)*3,))
    rs = RasterizationSettings(image_size=image_size, blur_radius=0., faces_per_pixel=1, bin_size=0)
    lights = PointLights(device=device, location=((2.,2.,2.),),
                         ambient_color=((0.45,)*3,), diffuse_color=((0.55,)*3,),
                         specular_color=((0.05,)*3,))
    blend = BlendParams(background_color=(1.,1.,1.))
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rs),
                            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights,
                                                   blend_params=blend))
    v_t = torch.from_numpy(v).to(device).unsqueeze(0)
    f_t = torch.from_numpy(f).to(device).unsqueeze(0)
    col = torch.full_like(v_t, 0.78)
    pmesh = Meshes(verts=v_t, faces=f_t, textures=TexturesVertex(verts_features=col))
    img = renderer(pmesh)[0,...,:3].clamp(0,1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def make_pipe():
    from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
        UNet2DConditionModel, AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler)
    from transformers import CLIPTextModel, CLIPTokenizer
    print("[v3-smoke] loading SD1.5 ...", flush=True)
    tok = CLIPTokenizer.from_pretrained(SD_BASE, subfolder="tokenizer")
    te  = CLIPTextModel.from_pretrained(SD_BASE, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKL.from_pretrained(SD_BASE, subfolder="vae", torch_dtype=torch.bfloat16)
    unet= UNet2DConditionModel.from_pretrained(SD_BASE, subfolder="unet", torch_dtype=torch.bfloat16)
    sched = DDPMScheduler.from_pretrained(SD_BASE, subfolder="scheduler")
    print(f"[v3-smoke] loading ControlNet {CKPT_DIR}", flush=True)
    cn = ControlNetModel.from_pretrained(CKPT_DIR, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=te, tokenizer=tok, unet=unet, controlnet=cn,
        scheduler=sched, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Reuse the same 2 footprints as v2 smoke for direct comparison
    grid = Image.open("Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/samples/step015000_train_grid.png")
    W, H = grid.size; tile = W // 3
    cases = []
    base_prompts = [
        "a house building, residential, 3/4 view, white background",
        "a villa building, residential, 3/4 view, white background",
    ]
    for row_i, base in zip([1, 2], base_prompts):
        fp_im = grid.crop((0, row_i*tile, tile, (row_i+1)*tile))
        fp_p = f"{OUT_DIR}/row{row_i}_input_footprint.png"
        fp_im.save(fp_p)
        cases.append((row_i, base, fp_im, fp_p))

    pipe = make_pipe()
    print("\n[v3-smoke] generating depth-style ControlNet outputs ...", flush=True)
    cn_gens = []
    g = torch.Generator(device=pipe.device).manual_seed(0)
    for row_i, base, fp_im, _ in cases:
        out = pipe(
            prompt=base,
            image=fp_im,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=g,
        ).images[0]
        gen_p = f"{OUT_DIR}/row{row_i}_v3_gen.png"
        out.save(gen_p)
        print(f"  row {row_i} -> {gen_p}", flush=True)
        cn_gens.append((row_i, out, gen_p))

    print("\n[v3-smoke] running Hunyuan3D-2 on the v3 gens ...", flush=True)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    h3d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

    rows_out = []
    for (row_i, gen_im, gen_p), (_, _, _, fp_p) in zip(cn_gens, cases):
        t0 = time.time()
        mesh = h3d(image=gen_im)[0]
        dt = time.time() - t0
        glb_p = f"{OUT_DIR}/row{row_i}_v3_mesh.glb"
        mesh.export(glb_p)
        print(f"  row {row_i}: V={mesh.vertices.shape[0]:,} F={mesh.faces.shape[0]:,}  ({dt:.1f}s)", flush=True)

        new_render_p = f"{OUT_DIR}/row{row_i}_v3_mesh_render.png"
        Image.fromarray(render_mesh(mesh)).save(new_render_p)
        # Old (v2 photo path) mesh for comparison
        old_glb = f"outputs/path_q_smoke/row{row_i}_mesh.glb"
        if os.path.exists(old_glb):
            old_mesh = load_glb(old_glb)
            old_render_p = f"{OUT_DIR}/row{row_i}_v2_mesh_render.png"
            Image.fromarray(render_mesh(old_mesh)).save(old_render_p)
        else:
            old_render_p = None

        v2_gen_p = f"outputs/path_q_smoke/row{row_i}_controlnet_gen.png"
        rows_out.append((row_i, fp_p, v2_gen_p, old_render_p, gen_p, new_render_p))

    # Comparison sheet: footprint | v2 gen | v2 mesh | v3 gen | v3 mesh
    tile = 360
    sheet = Image.new("RGB", (5 * tile, len(rows_out) * tile), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    headers = ["footprint", "v2 gen (photo)", "v2 mesh", "v3 gen (depth)", "v3 mesh"]
    for ri, paths in enumerate(rows_out):
        row_i, *imgs = paths
        for ci, p in enumerate(imgs):
            if p is None or not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB").resize((tile, tile), Image.BICUBIC)
            sheet.paste(im, (ci * tile, ri * tile))
            draw.text((ci * tile + 4, ri * tile + 4), headers[ci], fill="black")
    sheet_p = f"{OUT_DIR}/v3_vs_v2_comparison.png"
    sheet.save(sheet_p, optimize=True)
    print(f"\n[v3-smoke] comparison sheet -> {sheet_p}", flush=True)


if __name__ == "__main__":
    main()
