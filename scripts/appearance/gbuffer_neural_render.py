"""NEURAL APPEARANCE v0 — the Roblox-hybrid experiment on OUR pipeline (zero training).

Doctrine (ours + Roblox Reality): symbolic state + crisp geometry stay procedural; the
neural model only paints PIXELS, conditioned on geometry. Here:
  1. ask the running server for a detailed building volume (96^3 cube SDF — bake quality)
  2. sphere-trace it on GPU -> depth + normal G-buffers (we own an SDF; no mesh renderer)
  3. SDXL + ControlNet-depth re-renders the EXACT geometry photorealistically,
     one image per style prompt (the "style = words/image" knob)

Run (server on :8099; ~10GB one-time HF download to /tmp/hf):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH HF_HOME=/tmp/hf PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python -u scripts/appearance/gbuffer_neural_render.py
Output: outputs/appearance_v0/neural_render_<UTC>.png  (G-buffers + renders sheet)
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs", "appearance_v0")
RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
RES_IMG = 1024
PROMPTS = [
    ("brick", "photo of a victorian brick residential building, ornate windows, slate roof, "
              "overcast daylight, 35mm architectural photography, street level"),
    ("plaster", "photo of a mediterranean plaster building, terracotta roof tiles, warm "
                "golden hour light, architectural photography"),
    ("glass", "photo of a modern building with glass and steel facade, blue hour, "
              "architectural photography, high detail"),
]
NEG = "cartoon, painting, low quality, blurry, deformed, extra buildings, people, cars"


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def fetch_detailed_volume(style="victorian", cls="RESIDENTIAL", height=14):
    b = post("/building_sdf", {"footprint": RECT, "style": style,
                               "building_class": cls, "height": height})
    pv = post("/detail_volume", {"base_sdf_b64": b["sdf_b64"], "res": 64,
                                 "center": b["center"], "scale": b["scale"],
                                 "building_class": cls, "style": style, "seed": 3})
    g = np.frombuffer(base64.b64decode(pv["sdf_b64"]), dtype="<f4")
    return g.reshape(pv["res"], pv["res"], pv["res"]).copy()


def sphere_trace(grid, res=RES_IMG, cam_pos=(2.0, 0.9, 2.4), look=(0.0, -0.05, 0.0),
                 fov=38.0, device="cuda"):
    """Sphere-trace the cube-frame SDF -> (depth in [0,1] inverse, normals rgb, mask)."""
    vol = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, None]

    def sdf(p):                                   # p (N,3) xyz in [-1,1]
        gp = p.view(1, 1, 1, -1, 3)
        return F.grid_sample(vol, gp, mode="bilinear", align_corners=True,
                             padding_mode="border").view(-1)

    cp = torch.tensor(cam_pos, dtype=torch.float32, device=device)
    fwd = F.normalize(torch.tensor(look, device=device) - cp, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=device)), dim=0)
    up = torch.linalg.cross(right, fwd)
    t_half = np.tan(np.radians(fov / 2))
    ii = torch.linspace(-1, 1, res, device=device)
    vy, vx = torch.meshgrid(ii, ii, indexing="ij")
    dirs = F.normalize(fwd[None, None] + (vx[..., None] * right * t_half)
                       - (vy[..., None] * up * t_half), dim=-1).reshape(-1, 3)

    N = dirs.shape[0]
    t = torch.full((N,), 0.05, device=device)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    for _ in range(220):
        p = cp[None] + dirs * t[:, None]
        # outside the cube grid_sample clamps to border; add the true distance to the cube
        # so marching from a far camera still takes sane steps
        outside = (p.abs() - 1.0).clamp(min=0.0)
        d = sdf(p.clamp(-1.0, 1.0)) + outside.norm(dim=-1)
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

    t_img = t.view(res, res)
    hit_img = hit.view(res, res)
    tmin = t_img[hit_img].min() if hit_img.any() else torch.tensor(0.0)
    tmax = t_img[hit_img].max() if hit_img.any() else torch.tensor(1.0)
    inv = torch.zeros_like(t_img)
    inv[hit_img] = 1.0 - 0.85 * (t_img[hit_img] - tmin) / (tmax - tmin + 1e-6)
    nrm = ((n.view(res, res, 3) * 0.5 + 0.5) * hit_img[..., None]).clamp(0, 1)
    return inv.cpu().numpy(), nrm.cpu().numpy(), hit_img.cpu().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    from PIL import Image
    print("[v0] fetching detailed building volume from the pipeline ...")
    grid = fetch_detailed_volume()
    print("[v0] sphere-tracing G-buffers ...")
    depth, normal, mask = sphere_trace(grid)
    depth_img = Image.fromarray((np.stack([depth] * 3, -1) * 255).astype(np.uint8))
    normal_img = Image.fromarray((normal * 255).astype(np.uint8))
    depth_img.save(os.path.join(OUT, "gbuffer_depth.png"))
    normal_img.save(os.path.join(OUT, "gbuffer_normal.png"))
    print(f"[v0] hit coverage {mask.mean():.2f}")

    print("[v0] loading SDXL + ControlNet-depth (first run downloads ~10GB to $HF_HOME) ...")
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
    cn = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
                                         torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn,
        torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    renders = []
    for name, prompt in PROMPTS:
        print(f"[v0] rendering '{name}' ...")
        img = pipe(prompt=prompt, negative_prompt=NEG, image=depth_img,
                   num_inference_steps=30, controlnet_conditioning_scale=0.9,
                   generator=torch.Generator("cuda").manual_seed(7)).images[0]
        img.save(os.path.join(OUT, f"render_{name}.png"))
        renders.append((name, img))

    # ---- sheet -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 2 + len(renders)
    fig, axes = plt.subplots(1, cols, figsize=(3.4 * cols, 3.8))
    for ax, (title, im) in zip(axes, [("depth G-buffer", depth_img),
                                      ("normal G-buffer", normal_img)] +
                               [(f"SDXL · {n}", im) for n, im in renders]):
        ax.imshow(im); ax.set_title(title, fontsize=9); ax.set_axis_off()
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fig.suptitle("neural appearance v0 — OUR geometry, diffusion pixels (zero training)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, f"neural_render_{stamp}.png")
    fig.savefig(out, dpi=110)
    print(f"[v0] -> {out}")


if __name__ == "__main__":
    main()
