"""Run Hunyuan3D-2 on building images and create a quick mesh comparison.

This is intentionally separate from scene/run_demo.py. It lets us test whether
image-to-3D produces better building meshes before wiring it into town assembly.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/hunyuan_building_mesh_smoke.py --model mini --limit 2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
HUNYUAN_ROOT = REPO / "external" / "Hunyuan3D-2"
SCRATCH_CACHE = REPO / "external" / "hf_cache"
HY3DGEN_MODELS = REPO / "external" / "hy3dgen_models"
os.environ.setdefault("HF_HOME", str(SCRATCH_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(SCRATCH_CACHE / "hub"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO / "external" / "xdg_cache"))
os.environ.setdefault("HY3DGEN_MODELS", str(HY3DGEN_MODELS))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HUNYUAN_ROOT) not in sys.path:
    sys.path.insert(0, str(HUNYUAN_ROOT))


DEFAULT_INPUTS = [
    "legacy/outputs/path_q_smoke/row1_controlnet_gen.png",
    "legacy/outputs/path_q_smoke/row2_controlnet_gen.png",
    "legacy/outputs/path_q_smoke_neg/row1_controlnet_gen_neg.png",
    "legacy/outputs/path_q_smoke_neg/row2_controlnet_gen_neg.png",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    ap.add_argument("--out_dir", default="outputs/hunyuan_building_smoke")
    ap.add_argument("--model", choices=["mini", "full"], default="mini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--octree_resolution", type=int, default=380)
    ap.add_argument("--num_chunks", type=int, default=20000)
    ap.add_argument("--image_size", type=int, default=384)
    return ap.parse_args()


def render_mesh_png(mesh: trimesh.Trimesh, image_size: int = 512) -> Image.Image:
    from pytorch3d.renderer import (
        BlendParams,
        FoVOrthographicCameras,
        MeshRasterizer,
        MeshRenderer,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = verts - verts.mean(axis=0, keepdims=True)
    verts = verts / max(float(np.abs(verts).max()), 1e-9)

    r, t = look_at_view_transform(dist=2.5, elev=20, azim=35, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=r, T=t, scale_xyz=((0.75, 0.75, 0.75),))
    raster = RasterizationSettings(
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
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        ),
    )
    verts_t = torch.from_numpy(verts).to(device).unsqueeze(0)
    faces_t = torch.from_numpy(faces).to(device).unsqueeze(0)
    colors = torch.full_like(verts_t, 0.78)
    mesh_t = Meshes(verts=verts_t, faces=faces_t, textures=TexturesVertex(colors))
    image = renderer(mesh_t)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return Image.fromarray((image * 255).astype(np.uint8), "RGB")


def title_cell(image: Image.Image, title: str, size: int) -> Image.Image:
    body = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size + 28), "white")
    canvas.paste(body, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), title, fill=(0, 0, 0))
    return canvas


def load_pipeline(model_name: str):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if model_name == "mini":
        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini",
            variant="fp16",
        )
    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0",
        variant="fp16",
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [Path(p) for p in args.inputs]
    inputs = [p for p in inputs if p.exists()]
    if args.limit > 0:
        inputs = inputs[:args.limit]
    if not inputs:
        raise SystemExit("No input images found.")

    print(f"[hunyuan] loading {args.model} pipeline", flush=True)
    pipe = load_pipeline(args.model)
    print("[hunyuan] pipeline ready", flush=True)

    rows = []
    cells = []
    for i, image_path in enumerate(inputs):
        image = Image.open(image_path).convert("RGBA")
        stem = f"{i:02d}_{image_path.stem}"
        print(f"[hunyuan] {stem}", flush=True)
        t0 = time.time()
        mesh = pipe(
            image=image,
            num_inference_steps=args.steps,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
            generator=torch.manual_seed(args.seed + i),
            output_type="trimesh",
        )[0]
        elapsed = time.time() - t0

        glb_path = out_dir / f"{stem}.glb"
        render_path = out_dir / f"{stem}_render.png"
        mesh.export(glb_path)
        render = render_mesh_png(mesh, image_size=args.image_size)
        render.save(render_path, optimize=True)

        cells.append((
            title_cell(image, f"input {i}", args.image_size),
            title_cell(render, f"hunyuan {i}", args.image_size),
        ))
        rows.append({
            "index": i,
            "input": str(image_path),
            "glb": str(glb_path),
            "render": str(render_path),
            "verts": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "seconds": f"{elapsed:.2f}",
        })
        print(
            f"  wrote {glb_path} V={len(mesh.vertices):,} F={len(mesh.faces):,} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    sheet = Image.new("RGB", (2 * args.image_size, len(cells) * (args.image_size + 28)), "white")
    y = 0
    for input_cell, render_cell in cells:
        sheet.paste(input_cell, (0, y))
        sheet.paste(render_cell, (args.image_size, y))
        y += args.image_size + 28
    sheet_path = out_dir / "hunyuan_building_smoke_sheet.png"
    sheet.save(sheet_path, optimize=True)

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[hunyuan] sheet: {sheet_path}", flush=True)
    print(f"[hunyuan] metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
