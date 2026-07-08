"""Neural appearance for the sculptor — photoreal render of the CURRENT building with an
optional STYLE REFERENCE IMAGE (the per-building style embedding, served live).

Geometry (ours, crisp) -> sphere-traced depth+edge G-buffers -> SDXL + multi-ControlNet
(+ IP-Adapter when a style image is given). Models lazy-load once (~30-60s, cached in
HF_HOME=/tmp/hf) behind a lock; a render is ~10-15s on the A100 with cpu offload.
"""
from __future__ import annotations

import io
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/tmp/hf")

_lock = threading.Lock()
_pipe = None
_inpaint = None
_sketch_inpaint = None
_depth_model = None
_depth_processor = None
RES_IMG = 1024
NEG = "cartoon, painting, illustration, low quality, blurry, deformed, text, watermark"
DEFAULT_PROMPT = ("photo of a {style} {cls} building, street view, natural daylight, "
                  "photorealistic, 35mm architectural photography, high detail")


def get_pipe():
    global _pipe
    with _lock:
        if _pipe is None:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
            cn_d = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
                                                   torch_dtype=torch.float16)
            cn_c = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                                                   torch_dtype=torch.float16)
            p = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", controlnet=[cn_d, cn_c],
                torch_dtype=torch.float16)
            p.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                              weight_name="ip-adapter_sdxl_vit-h.safetensors",
                              image_encoder_folder="models/image_encoder")
            p.enable_model_cpu_offload()
            p.set_progress_bar_config(disable=True)
            _pipe = p
    return _pipe


def get_inpaint_pipe():
    """SDXL ControlNet INPAINT pipeline reusing the base pipe's modules (no extra VRAM) —
    for iterative TEXTure-style texturing (keep textured regions, generate new ones)."""
    global _inpaint
    if _inpaint is None:
        from diffusers import StableDiffusionXLControlNetInpaintPipeline
        base = get_pipe()
        with _lock:
            _inpaint = StableDiffusionXLControlNetInpaintPipeline.from_pipe(base)
            _inpaint.set_progress_bar_config(disable=True)
    return _inpaint


def get_sketch_inpaint_pipe():
    """SDXL ControlNet INPAINT pipeline with a THIRD ControlNet (xinsir scribble) alongside
    depth+canny, so a user's rough drawn shape acts as a real structural signal the model
    resolves into "the right thing" (thin lines) or a looser creative interpretation (thick
    lines) — instead of just being blended in as raw pixel color.

    This is a fully independent `from_pretrained` + `enable_model_cpu_offload()` build (like
    get_pipe(), not get_inpaint_pipe()'s from_pipe() reuse): splicing a freshly-loaded
    ControlNet into `get_pipe()`'s already cpu-offload-hooked pipe via from_pipe() causes a
    "tensors on different devices" error, because that net never joined the original
    enable_model_cpu_offload() hook chain. A second full SDXL load costs ~30s once + a few
    GB VRAM, which this GPU has headroom for; it's the reliable option vs. fighting
    accelerate's per-pipe hook bookkeeping. Kept separate from get_inpaint_pipe() so
    bake_texture's existing 2-ControlNet path is untouched either way."""
    global _sketch_inpaint
    if _sketch_inpaint is None:
        from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
        with _lock:
            cn_d = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
                                                   torch_dtype=torch.float16)
            cn_c = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                                                   torch_dtype=torch.float16)
            cn_s = ControlNetModel.from_pretrained("xinsir/controlnet-scribble-sdxl-1.0",
                                                   torch_dtype=torch.float16)
            p = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", controlnet=[cn_d, cn_c, cn_s],
                torch_dtype=torch.float16)
            p.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                              weight_name="ip-adapter_sdxl_vit-h.safetensors",
                              image_encoder_folder="models/image_encoder")
            p.enable_model_cpu_offload()
            p.set_progress_bar_config(disable=True)
            _sketch_inpaint = p
    return _sketch_inpaint


def get_depth_model():
    """Depth Anything V2 (small) monocular depth estimator, run on the GENERATED ART (not
    the building) to turn it into a real depth/height field for paint_relief's relief
    carving — replaces the earlier brightness-heuristic proxy
    (height-from-luminance/Sobel) with an actual learned depth cue (shading + perspective),
    so e.g. a carved-looking motif's implied protrusions/recesses come from real monocular
    depth estimation rather than raw pixel brightness."""
    global _depth_model, _depth_processor
    if _depth_model is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        with _lock:
            if _depth_model is None:
                name = "depth-anything/Depth-Anything-V2-Small-hf"
                _depth_processor = AutoImageProcessor.from_pretrained(name)
                _depth_model = AutoModelForDepthEstimation.from_pretrained(name).to("cuda").eval()
    return _depth_processor, _depth_model


@torch.no_grad()
def trace_gbuffers(grid, res=RES_IMG, cam_pos=(2.0, 0.9, 2.4), look=(0.0, -0.05, 0.0),
                   fov=38.0, device="cuda"):
    """Cube-frame SDF volume -> (depth PIL, edge PIL, hit coverage)."""
    from PIL import Image
    vol = torch.as_tensor(np.asarray(grid, np.float32), device=device)[None, None]

    def sdf(p):
        out = (p.abs() - 1.0).clamp(min=0.0)
        qg = p.clamp(-1.0, 1.0).view(1, 1, 1, -1, 3)
        return F.grid_sample(vol, qg, mode="bilinear", align_corners=True,
                             padding_mode="border").view(-1) + out.norm(dim=-1)

    cp = torch.tensor(cam_pos, dtype=torch.float32, device=device)
    fwd = F.normalize(torch.tensor(look, device=device) - cp, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=device)), dim=0)
    up = torch.linalg.cross(right, fwd)
    th = float(np.tan(np.radians(fov / 2)))
    ii = torch.linspace(-1, 1, res, device=device)
    vy, vx = torch.meshgrid(ii, ii, indexing="ij")
    dirs = F.normalize(fwd[None, None] + (vx[..., None] * right * th)
                       - (vy[..., None] * up * th), dim=-1).reshape(-1, 3)
    t = torch.full((dirs.shape[0],), 0.05, device=device)
    alive = torch.ones_like(t, dtype=torch.bool)
    for _ in range(220):
        p = cp[None] + dirs * t[:, None]
        d = sdf(p)
        t = torch.where(alive, t + d.clamp(min=1e-4) * 0.9, t)
        alive = alive & (d > 2.5e-3) & (t < 7.0)
        if not alive.any():
            break
    p = (cp[None] + dirs * t[:, None]).clamp(-1.0, 1.0)
    hit = (sdf(p) < 1.5e-2) & (t < 7.0)
    eps = 2.0 / grid.shape[0]
    n = torch.stack([sdf(p + torch.tensor([eps, 0, 0], device=device)) - sdf(p - torch.tensor([eps, 0, 0], device=device)),
                     sdf(p + torch.tensor([0, eps, 0], device=device)) - sdf(p - torch.tensor([0, eps, 0], device=device)),
                     sdf(p + torch.tensor([0, 0, eps], device=device)) - sdf(p - torch.tensor([0, 0, eps], device=device))], -1)
    n = F.normalize(n, dim=-1)
    t_img, hit_img = t.view(res, res), hit.view(res, res)
    inv = torch.zeros_like(t_img)
    if hit_img.any():
        tq = t_img[hit_img]
        lo, hi = tq.min(), tq.max()
        inv[hit_img] = 1.0 - 0.85 * (t_img[hit_img] - lo) / (hi - lo + 1e-6)
    nrm = ((n.view(res, res, 3) * 0.5 + 0.5) * hit_img[..., None]).clamp(0, 1).cpu().numpy()
    gx = np.abs(np.diff(nrm, axis=1, prepend=nrm[:, :1])).sum(-1)
    gy = np.abs(np.diff(nrm, axis=0, prepend=nrm[:1])).sum(-1)
    edge = ((gx + gy) * hit_img.cpu().numpy() > 0.25).astype(np.uint8) * 255
    depth_img = Image.fromarray((np.stack([inv.cpu().numpy()] * 3, -1) * 255).astype(np.uint8))
    edge_img = Image.fromarray(np.stack([edge] * 3, -1))
    return depth_img, edge_img, float(hit_img.float().mean())


@torch.no_grad()
def render_building(grid96, style="modern", building_class="RESIDENTIAL",
                    style_ref=None, prompt=None, seed=7, steps=30):
    """grid96: cube-frame detailed SDF. style_ref: PIL image or None. Returns PIL image."""
    depth_img, edge_img, cov = trace_gbuffers(grid96)
    if cov < 0.02:
        raise ValueError(f"empty trace (coverage {cov:.3f})")
    pipe = get_pipe()
    pr = prompt or DEFAULT_PROMPT.format(style=style, cls=building_class.lower())
    from PIL import Image
    blank = Image.new("RGB", (RES_IMG, RES_IMG), 0)
    with _lock:                                  # one render at a time (VRAM)
        pipe.set_ip_adapter_scale(0.65 if style_ref is not None else 0.0)
        img = pipe(prompt=pr, negative_prompt=NEG, image=[depth_img, edge_img],
                   num_inference_steps=int(steps),
                   controlnet_conditioning_scale=[0.85, 0.45],
                   ip_adapter_image=style_ref if style_ref is not None else blank,
                   generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
    return img


def png_b64(img):
    import base64
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# TOWN-SCALE photoreal with PER-BUILDING style references (instance-masked IP-Adapter)
# ---------------------------------------------------------------------------

@torch.no_grad()
def trace_town(items, res=RES_IMG, fov=42.0, device="cuda"):
    """items: [{vol (1,1,R,R,R) cube SDF, center (3,) t, scale float, pos (3,) t}].
    Auto-frames the scene. Returns (depth PIL, edge PIL, ids (res,res) int, coverage)."""
    from PIL import Image

    def scene_sdf(p):
        d = p[:, 1] + 0.0                                  # ground plane
        for it in items:
            q = (p - it["pos"] - it["center"]) / it["scale"]
            qg = q.clamp(-1.0, 1.0).view(1, 1, 1, -1, 3)
            v = F.grid_sample(it["vol"], qg, mode="bilinear", align_corners=True,
                              padding_mode="border").view(-1) * it["scale"]
            d = torch.minimum(d, v + ((q.abs() - 1.0).clamp(min=0) * it["scale"]).norm(dim=-1))
        return d

    # auto camera: fit the scene bbox
    cs = torch.stack([it["pos"] + it["center"] for it in items])
    rad = max(float(max(it["scale"] for it in items)) * 1.6,
              float((cs[:, [0, 2]].max(0).values - cs[:, [0, 2]].min(0).values).max()) * 0.75 + 12)
    ctr = cs.mean(0)
    look = (float(ctr[0]), 4.0, float(ctr[2]))
    cam = (float(ctr[0]) + rad * 1.15, rad * 0.85, float(ctr[2]) + rad * 1.3)
    tmax = rad * 6 + 50

    cp = torch.tensor(cam, dtype=torch.float32, device=device)
    fwd = F.normalize(torch.tensor(look, device=device) - cp, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=device)), dim=0)
    up = torch.linalg.cross(right, fwd)
    th = float(np.tan(np.radians(fov / 2)))
    ii = torch.linspace(-1, 1, res, device=device)
    vy, vx = torch.meshgrid(ii, ii, indexing="ij")
    dirs = F.normalize(fwd[None, None] + (vx[..., None] * right * th)
                       - (vy[..., None] * up * th), dim=-1).reshape(-1, 3)
    t = torch.full((dirs.shape[0],), 0.5, device=device)
    alive = torch.ones_like(t, dtype=torch.bool)
    for _ in range(300):
        p = cp[None] + dirs * t[:, None]
        d = scene_sdf(p)
        t = torch.where(alive, t + d.clamp(min=1e-3) * 0.9, t)
        alive = alive & (d > 2e-2) & (t < tmax)
        if not alive.any():
            break
    p = cp[None] + dirs * t[:, None]
    hit = (scene_sdf(p) < 0.25) & (t < tmax)

    dists = [p[:, 1] + 0.0]
    for it in items:
        q = (p - it["pos"] - it["center"]) / it["scale"]
        qg = q.clamp(-1.0, 1.0).view(1, 1, 1, -1, 3)
        v = F.grid_sample(it["vol"], qg, mode="bilinear", align_corners=True,
                          padding_mode="border").view(-1) * it["scale"]
        dists.append(v + ((q.abs() - 1.0).clamp(min=0) * it["scale"]).norm(dim=-1))
    ids = torch.stack(dists, 0).argmin(0)
    ids = torch.where(hit, ids, torch.full_like(ids, -1)).view(res, res)

    eps = 0.25
    n = torch.stack([scene_sdf(p + torch.tensor([eps, 0, 0], device=device)) - scene_sdf(p - torch.tensor([eps, 0, 0], device=device)),
                     scene_sdf(p + torch.tensor([0, eps, 0], device=device)) - scene_sdf(p - torch.tensor([0, eps, 0], device=device)),
                     scene_sdf(p + torch.tensor([0, 0, eps], device=device)) - scene_sdf(p - torch.tensor([0, 0, eps], device=device))], -1)
    n = F.normalize(n, dim=-1)
    t_img, hit_img = t.view(res, res), hit.view(res, res)
    inv = torch.zeros_like(t_img)
    if hit_img.any():
        tq = t_img[hit_img]
        lo, hi = tq.min(), torch.quantile(tq, 0.99)
        inv[hit_img] = (1.0 - 0.9 * (t_img[hit_img] - lo) / (hi - lo + 1e-6)).clamp(0.05, 1.0)
    nrm = ((n.view(res, res, 3) * 0.5 + 0.5) * hit_img[..., None]).clamp(0, 1).cpu().numpy()
    gx = np.abs(np.diff(nrm, axis=1, prepend=nrm[:, :1])).sum(-1)
    gy = np.abs(np.diff(nrm, axis=0, prepend=nrm[:1])).sum(-1)
    edge = ((gx + gy) * hit_img.cpu().numpy() > 0.25).astype(np.uint8) * 255
    depth_img = Image.fromarray((np.stack([inv.cpu().numpy()] * 3, -1) * 255).astype(np.uint8))
    edge_img = Image.fromarray(np.stack([edge] * 3, -1))
    return depth_img, edge_img, ids.cpu().numpy(), float(hit_img.float().mean())


TOWN_PROMPT_DEFAULT = ("aerial drone photo of a town, varied buildings, streets, "
                       "photorealistic, 50mm, natural light, high detail")


@torch.no_grad()
def render_town(items, refs_per_building, prompt=None, seed=11, steps=30):
    """refs_per_building: list aligned with items — PIL image or None per building.
    Buildings sharing the same ref object are grouped into one mask. Returns PIL image."""
    from PIL import Image
    depth_img, edge_img, ids, cov = trace_town(items)
    if cov < 0.02:
        raise ValueError(f"empty town trace (coverage {cov:.3f})")
    pipe = get_pipe()
    pr = prompt or TOWN_PROMPT_DEFAULT
    blank = Image.new("RGB", (RES_IMG, RES_IMG), 0)

    groups = {}                                            # ref-id -> (ref, [instance ids])
    for i, ref in enumerate(refs_per_building):
        if ref is None:
            continue
        groups.setdefault(id(ref), (ref, []))[1].append(i + 1)   # instance id = idx+1
    with _lock:
        if not groups:
            pipe.set_ip_adapter_scale(0.0)
            return pipe(prompt=pr, negative_prompt=NEG, image=[depth_img, edge_img],
                        num_inference_steps=int(steps),
                        controlnet_conditioning_scale=[0.85, 0.45],
                        ip_adapter_image=blank,
                        generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
        from diffusers.image_processor import IPAdapterMaskProcessor
        refs, masks = [], []
        for ref, members in list(groups.values())[:4]:     # cap encoder cost at 4 refs
            refs.append(ref)
            masks.append(Image.fromarray(
                (np.isin(ids, members).astype(np.uint8)) * 255))
        proc = IPAdapterMaskProcessor()
        mt = proc.preprocess(masks, height=RES_IMG, width=RES_IMG)
        mt = [mt.reshape(1, mt.shape[0], mt.shape[2], mt.shape[3])]
        pipe.set_ip_adapter_scale([[0.7] * len(refs)])
        return pipe(prompt=pr, negative_prompt=NEG, image=[depth_img, edge_img],
                    num_inference_steps=int(steps),
                    controlnet_conditioning_scale=[0.85, 0.45],
                    ip_adapter_image=[refs],
                    cross_attention_kwargs={"ip_adapter_masks": mt},
                    generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
