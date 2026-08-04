"""Generate OSM building proposal images with a pretrained image prior.

This is the practical ControlCity-style branch for this project:

    OSM footprint/class/height -> SD image prior + footprint ControlNet -> Hunyuan input image

It uses the cached SD1.5 base model plus the existing footprint-to-view
ControlNet checkpoint from the project. The generated images are written with a
manifest that can be consumed by scripts/osm_hunyuan_pipeline_smoke.py using
--conditioning_source image_prior.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_SD_MODEL = (
    "external/hf_cache/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/"
    "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
DEFAULT_CONTROLNET = (
    "legacy/Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/"
    "ckpt/controlnet-015000"
)

NEGATIVE_PROMPT = (
    "people, cars, street, tree, vegetation, text, watermark, logo, blurry, "
    "cropped, cut off, photorealistic street photo, cluttered background, "
    "interior, aerial map, floor plan"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_image_prior_proposals_v1")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--sd_model", default=DEFAULT_SD_MODEL)
    ap.add_argument("--controlnet", default=DEFAULT_CONTROLNET)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--prompt_style", choices=["clean_mass", "hunyuan"], default="clean_mass")
    ap.add_argument("--fallback_procedural", action="store_true", help="Write procedural images if SD/ControlNet load fails.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def class_words(building_class: str) -> tuple[str, str]:
    order = ["RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS", "MILITARY"]
    top = next((name for name in order if building_class.startswith(name)), "RESIDENTIAL")
    sub = building_class[len(top):].replace("_", " ").strip() or "building"
    return top.lower(), sub


def prompt_for(building: dict, style: str) -> str:
    top, sub = class_words(str(building.get("class", "")))
    area = float(building.get("area", 0.0) or 0.0)
    height = float(building.get("height", 0.0) or 0.0)
    floors = max(1, min(8, round(height / 3.2))) if height > 0 else 2
    mass = "large" if area > 900 else "compact" if area < 250 else "medium sized"
    if style == "hunyuan":
        return (
            f"a {mass} {sub} building, {top}, {floors} floors, 3/4 view, "
            "single isolated object, white background, clean gray architectural model"
        )
    return (
        f"a {mass} {sub} building, {top}, {floors} floors, simple matte 3D massing model, "
        "orthographic 3/4 view, plain white background, centered, complete building, sharp silhouette"
    )


def choose_buildings(payload: dict, limit: int) -> list[dict]:
    buildings = [
        b for b in payload.get("buildings", [])
        if len(b.get("polygon", [])) >= 3 and float(b.get("area", 0.0)) > 1.0
    ]
    buildings.sort(key=lambda b: float(b.get("area", 0.0)), reverse=True)
    return buildings[:limit] if limit > 0 else buildings


def safe_stem(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_")[:96] or "building"


def control_image(mask: np.ndarray, size: int) -> Image.Image:
    arr = (mask > 0).astype(np.uint8) * 255
    img = Image.fromarray(arr, "L").resize((size, size), Image.Resampling.NEAREST)
    return Image.merge("RGB", (img, img, img))


def rasterize_polygon(poly: list[list[float]] | list[tuple[float, float]], res: int = 64) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float64)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    norm = (pts - mn) / span
    pad = 4
    pix = np.column_stack([
        pad + norm[:, 0] * (res - 1 - 2 * pad),
        pad + (1.0 - norm[:, 1]) * (res - 1 - 2 * pad),
    ])
    img = Image.new("L", (res, res), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(p) for p in pix], fill=255)
    return (np.asarray(img) > 0).astype(np.float32)


def procedural_baseline(building: dict, size: int) -> Image.Image:
    top, sub = class_words(str(building.get("class", "")))
    bg = (246, 248, 249)
    wall = {
        "residential": (194, 186, 172),
        "commercial": (176, 184, 190),
        "public": (188, 176, 154),
        "religious": (190, 181, 162),
    }.get(top, (194, 186, 172))
    side = tuple(max(0, c - 36) for c in wall)
    roof = (112, 78, 68) if top == "residential" else (83, 91, 98)
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    area = float(building.get("area", 400.0) or 400.0)
    height = float(building.get("height", 8.0) or 8.0)
    w = int(np.clip(size * (0.34 + min(area, 1400.0) / 7000.0), size * 0.34, size * 0.58))
    d = int(np.clip(size * 0.20, size * 0.16, size * 0.28))
    h = int(np.clip(size * (0.20 + height / 120.0), size * 0.20, size * 0.44))
    cx, base_y = size // 2, int(size * 0.70)
    skew = int(size * 0.12)
    roof_poly = [
        (cx - w // 2, base_y - h - d // 2),
        (cx + w // 2, base_y - h - d // 2),
        (cx + w // 2 + skew, base_y - h),
        (cx - w // 2 + skew, base_y - h),
    ]
    front = [roof_poly[3], roof_poly[2], (roof_poly[2][0], base_y), (roof_poly[3][0], base_y)]
    side_poly = [roof_poly[2], roof_poly[1], (roof_poly[1][0], base_y - d // 2), (roof_poly[2][0], base_y)]
    draw.ellipse((cx - w // 2, base_y - 8, cx + w // 2 + skew, base_y + 28), fill=(218, 222, 224))
    draw.polygon(front, fill=wall, outline=(96, 100, 103))
    draw.polygon(side_poly, fill=side, outline=(96, 100, 103))
    draw.polygon(roof_poly, fill=roof, outline=(82, 72, 68))
    floors = max(1, min(6, round(height / 3.2)))
    cols = max(2, min(6, round(w / 70)))
    for r in range(floors):
        y = base_y - 22 - r * max(18, h // max(floors, 1))
        for c in range(cols):
            x = cx - w // 2 + skew + 18 + c * max(20, (w - 40) // cols)
            draw.rectangle((x, y, x + 10, y + 12), fill=(76, 116, 136))
    return img


def title_cell(img: Image.Image, title: str, size: int) -> Image.Image:
    cell = Image.new("RGB", (size, size + 28), "white")
    cell.paste(img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), (0, 28))
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, 0, size - 1, size + 27), outline=(210, 210, 210))
    draw.text((8, 8), title, fill=(20, 20, 20))
    return cell


def make_pipeline(args: argparse.Namespace):
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        StableDiffusionControlNetPipeline,
        UNet2DConditionModel,
        UniPCMultistepScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[image-prior] loading ControlNet: {args.controlnet}", flush=True)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet,
        torch_dtype=dtype,
        local_files_only=True,
    )
    print(f"[image-prior] loading SD components: {args.sd_model}", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae", torch_dtype=dtype, local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(
        args.sd_model, subfolder="unet", torch_dtype=dtype, local_files_only=True
    )
    scheduler = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler", local_files_only=True)
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    print("[image-prior] configuring scheduler", flush=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(f"[image-prior] moving pipeline to {device}", flush=True)
    pipe.to(device)
    pipe.enable_attention_slicing()
    print("[image-prior] pipeline ready", flush=True)
    return pipe


def save_sheet(path: Path, rows: list[tuple[Image.Image, Image.Image, Image.Image]], size: int) -> None:
    sheet = Image.new("RGB", (3 * size, len(rows) * (size + 28)), "white")
    for r, (ctrl, prior, proc) in enumerate(rows):
        y = r * (size + 28)
        for c, (img, title) in enumerate([
            (ctrl, "footprint control"),
            (prior, "image prior"),
            (proc, "procedural baseline"),
        ]):
            cell = title_cell(img.resize((size, size), Image.Resampling.BICUBIC), title, size)
            sheet.paste(cell, (c * size, y))
    sheet.save(path, optimize=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "image_prior_inputs"
    ctrl_dir = out_dir / "control_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    payload = json.load(open(args.osm_json))
    buildings = choose_buildings(payload, args.limit)
    if not buildings:
        raise SystemExit(f"No usable buildings in {args.osm_json}")

    pipe = None
    if not args.fallback_procedural:
        try:
            pipe = make_pipeline(args)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the pretrained image-prior pipeline. "
                "Use --fallback_procedural only for interface/debug output."
            ) from exc

    rows: list[dict[str, object]] = []
    sheet_rows: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    for i, building in enumerate(buildings):
        osm_id = safe_stem(str(building.get("id", f"osm_{i}")))
        stem = f"{i:02d}_{osm_id}"
        mask = rasterize_polygon(building["polygon"], res=64)
        ctrl = control_image(mask, args.image_size)
        prompt = prompt_for(building, args.prompt_style)
        ctrl_path = ctrl_dir / f"{stem}_control.png"
        prior_path = img_dir / f"{stem}_image_prior.png"
        ctrl.save(ctrl_path, optimize=True)

        if pipe is None:
            prior = procedural_baseline(building, args.image_size)
            backend = "procedural_fallback"
        else:
            print(f"[image-prior] generating {stem}", flush=True)
            generator = torch.Generator(device=pipe.device).manual_seed(args.seed + i)
            prior = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=ctrl,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                generator=generator,
            ).images[0].convert("RGB")
            backend = "sd15_footprint_controlnet"
        prior.save(prior_path, optimize=True)
        proc = procedural_baseline(building, args.image_size)
        sheet_rows.append((ctrl, prior, proc))
        rows.append({
            "index": i,
            "osm_id": building.get("id", f"osm_{i}"),
            "class": building.get("class", ""),
            "area_m2": float(building.get("area", 0.0) or 0.0),
            "height_m": float(building.get("height", 0.0) or 0.0),
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "backend": backend,
            "control_png": str(ctrl_path),
            "image_prior_png": str(prior_path),
        })
        print(f"[image-prior] {stem}: {prior_path}", flush=True)

    manifest = out_dir / "image_prior_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": len(rows),
        "backend": rows[0]["backend"],
        "sd_model": args.sd_model,
        "controlnet": args.controlnet,
        "manifest": str(manifest),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_sheet(out_dir / "image_prior_sheet.png", sheet_rows, min(args.image_size, 384))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[image-prior] sheet: {out_dir / 'image_prior_sheet.png'}", flush=True)


if __name__ == "__main__":
    main()
