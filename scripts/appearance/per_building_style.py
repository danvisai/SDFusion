"""NEURAL APPEARANCE v3 — PER-BUILDING style embeddings via instance-masked IP-Adapter.

Each building carries its OWN style reference image (stored as an embedding): the SDF
tracer emits an INSTANCE G-buffer (which building each ray hit — free, the scene is a min
over buildings), instance masks gate the IP-Adapter cross-attention per region, so three
different reference styles render in ONE consistent image on our exact geometry.

Run (server on :8099; reuses cached SDXL/CN/IP-Adapter in $HF_HOME):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH HF_HOME=/tmp/hf PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python -u scripts/appearance/per_building_style.py
Output: outputs/appearance_v0/per_building_<UTC>.png
"""
from __future__ import annotations

import datetime
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "outputs", "appearance_v0")
RES_IMG = 1024

from scripts.appearance.town_neural_render import (   # noqa: E402
    fetch_block, town_sdf, street_sdf, edges_from_normals, NEG)

#   style groups: building indices (order of BLOCK) -> one reference style each
STYLE_GROUPS = [
    ("amsterdam", [0, 1, 5], "street photo of amsterdam canal houses, dark red brick "
                             "facades, white window frames, photorealistic"),
    ("santorini", [3, 4], "photo of santorini greek island architecture, whitewashed "
                          "walls, blue domes and shutters, photorealistic"),
    ("glass", [2], "photo of a modern office building, blue glass curtain wall facade, "
                   "steel, photorealistic"),
]
TOWN_PROMPT = ("aerial drone photo of a town block with a tree-lined street, sidewalks, "
               "photorealistic, 50mm, high detail, natural light")


def trace_with_ids(items, res=RES_IMG, cam_pos=(55.0, 38.0, 62.0), look=(0.0, 4.0, -2.0),
                   fov=42.0, tmax=220.0, device="cuda"):
    """Sphere-trace the town and ALSO return the instance id per pixel
    (-1 sky · 0 street/ground · 1..N building index+1)."""
    sdf = town_sdf(items, device)
    cp = torch.tensor(cam_pos, dtype=torch.float32, device=device)
    fwd = F.normalize(torch.tensor(look, device=device) - cp, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=device)), dim=0)
    up = torch.linalg.cross(right, fwd)
    th = np.tan(np.radians(fov / 2))
    ii = torch.linspace(-1, 1, res, device=device)
    vy, vx = torch.meshgrid(ii, ii, indexing="ij")
    dirs = F.normalize(fwd[None, None] + (vx[..., None] * right * th)
                       - (vy[..., None] * up * th), dim=-1).reshape(-1, 3)
    t = torch.full((dirs.shape[0],), 0.5, device=device)
    alive = torch.ones_like(t, dtype=torch.bool)
    for _ in range(300):
        p = cp[None] + dirs * t[:, None]
        d = sdf(p)
        t = torch.where(alive, t + d.clamp(min=1e-3) * 0.9, t)
        alive = alive & (d > 2e-2) & (t < tmax)
        if not alive.any():
            break
    p = cp[None] + dirs * t[:, None]
    hit = (sdf(p) < 0.2) & (t < tmax)

    # instance id = argmin over (street, building_i) at the hit points
    dists = [street_sdf(p)]
    for it in items:
        q = (p - it["pos"] - it["center"]) / it["scale"]
        qg = q.clamp(-1.0, 1.0).view(1, 1, 1, -1, 3)
        v = F.grid_sample(it["vol"], qg, mode="bilinear", align_corners=True,
                          padding_mode="border").view(-1) * it["scale"]
        dists.append(v + ((q.abs() - 1.0).clamp(min=0) * it["scale"]).norm(dim=-1))
    ids = torch.stack(dists, 0).argmin(0)
    ids = torch.where(hit, ids, torch.full_like(ids, -1))

    eps = 0.25
    n = torch.stack([sdf(p + torch.tensor([eps, 0, 0], device=device)) - sdf(p - torch.tensor([eps, 0, 0], device=device)),
                     sdf(p + torch.tensor([0, eps, 0], device=device)) - sdf(p - torch.tensor([0, eps, 0], device=device)),
                     sdf(p + torch.tensor([0, 0, eps], device=device)) - sdf(p - torch.tensor([0, 0, eps], device=device))], -1)
    n = F.normalize(n, dim=-1)
    t_img, hit_img = t.view(res, res), hit.view(res, res)
    tq = t_img[hit_img]
    inv = torch.zeros_like(t_img)
    if hit_img.any():
        lo, hi = tq.min(), torch.quantile(tq, 0.99)
        inv[hit_img] = (1.0 - 0.9 * (t_img[hit_img] - lo) / (hi - lo + 1e-6)).clamp(0.05, 1.0)
    nrm = ((n.view(res, res, 3) * 0.5 + 0.5) * hit_img[..., None]).clamp(0, 1)
    return (inv.cpu().numpy(), nrm.cpu().numpy(), hit_img.cpu().numpy(),
            ids.view(res, res).cpu().numpy())


def main():
    os.makedirs(OUT, exist_ok=True)
    from PIL import Image
    device = "cuda"
    print("[v3] fetching block ...")
    items = fetch_block(device)
    print("[v3] tracing with instance ids ...")
    with torch.no_grad():
        depth, normal, mask, ids = trace_with_ids(items, device=device)
    print(f"[v3] hit {mask.mean():.2f} · instances {sorted(np.unique(ids).tolist())}")
    edge = edges_from_normals(normal, mask)
    depth_img = Image.fromarray((np.stack([depth] * 3, -1) * 255).astype(np.uint8))
    edge_img = Image.fromarray(edge.astype(np.uint8))

    # per-group binary masks (building index i -> instance id i+1); street+sky unmasked
    group_masks = []
    for name, members, _ in STYLE_GROUPS:
        m = np.isin(ids, [i + 1 for i in members]).astype(np.float32)
        group_masks.append(Image.fromarray((m * 255).astype(np.uint8)))

    print("[v3] loading SDXL + CNs + IP-Adapter ...")
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
    from diffusers.image_processor import IPAdapterMaskProcessor
    cn_d = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
                                           torch_dtype=torch.float16)
    cn_c = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0",
                                           torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=[cn_d, cn_c],
        torch_dtype=torch.float16)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter_sdxl_vit-h.safetensors",
                         image_encoder_folder="models/image_encoder")
    pipe.enable_model_cpu_offload()
    blank = Image.new("RGB", (RES_IMG, RES_IMG), 0)

    # style refs: reuse cached ones from v2 where present, generate missing
    refs = []
    for name, _members, ref_prompt in STYLE_GROUPS:
        ref_path = os.path.join(OUT, f"styleref_{name}.png")
        if os.path.exists(ref_path):
            refs.append(Image.open(ref_path).convert("RGB"))
        else:
            print(f"[v3] generating style ref '{name}' ...")
            pipe.set_ip_adapter_scale(0.0)
            r = pipe(prompt=ref_prompt, negative_prompt=NEG, image=[blank, blank],
                     num_inference_steps=30, controlnet_conditioning_scale=[0.0, 0.0],
                     ip_adapter_image=blank,
                     generator=torch.Generator("cuda").manual_seed(23)).images[0]
            r.save(ref_path)
            refs.append(r)

    print("[v3] rendering with PER-BUILDING style masks ...")
    proc = IPAdapterMaskProcessor()
    masks_t = proc.preprocess(group_masks, height=RES_IMG, width=RES_IMG)
    # one adapter, n reference images: masks must be (1, n, H, W) wrapped in a list
    masks_t = [masks_t.reshape(1, masks_t.shape[0], masks_t.shape[2], masks_t.shape[3])]
    pipe.set_ip_adapter_scale([[0.7] * len(refs)])
    img = pipe(prompt=TOWN_PROMPT, negative_prompt=NEG, image=[depth_img, edge_img],
               num_inference_steps=30, controlnet_conditioning_scale=[0.85, 0.45],
               ip_adapter_image=[refs],
               cross_attention_kwargs={"ip_adapter_masks": masks_t},
               generator=torch.Generator("cuda").manual_seed(11)).images[0]
    img.save(os.path.join(OUT, "town_per_building.png"))

    # instance map visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    idv = ids.astype(np.float32)
    idv[idv < 0] = np.nan
    panels = [("instance G-buffer", None)] + \
             [(f"ref · {n} → bldg {m}", r) for (n, m, _), r in zip(STYLE_GROUPS, refs)] + \
             [("PER-BUILDING styles, one render", img)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.0))
    axes[0].imshow(idv, cmap=cm.tab10, vmin=0, vmax=9)
    axes[0].set_title(panels[0][0], fontsize=9); axes[0].set_axis_off()
    for ax, (title, im) in zip(axes[1:], panels[1:]):
        ax.imshow(im); ax.set_title(title, fontsize=9); ax.set_axis_off()
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fig.suptitle("neural appearance v3 — per-building style embeddings "
                 "(instance-masked IP-Adapter, one consistent render)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT, f"per_building_{stamp}.png")
    fig.savefig(out, dpi=110)
    print(f"[v3] -> {out}")


if __name__ == "__main__":
    main()
