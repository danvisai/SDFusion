"""NEURAL APPEARANCE v1 — multi-ControlNet (depth + edges-from-normals) + TOWN-SCALE render.

Composes a small town block from the pipeline (per-building bake-quality detail volumes,
placed on a ground plane), sphere-traces the whole scene as one SDF, and re-renders it
photorealistically with SDXL conditioned on BOTH the depth buffer and an edge map derived
from the traced normals (architectural edges: window reveals, cornices, rooflines).

Run (server on :8099; canny CN ~2.5GB one-time download):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH HF_HOME=/tmp/hf PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python -u scripts/appearance/town_neural_render.py
Output: outputs/appearance_v0/town_render_<UTC>.png
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs", "appearance_v0")
RES_IMG = 1024

# ARBITRARY footprints — the pipeline takes any polygon (L/T/U/rect; OSM, masks, hand-
# drawn). (polygon · style · class · height · world x,z position)
def _L(w, d, cw, cd):
    return [[-w/2, -d/2], [w/2, -d/2], [w/2, d/2-cd], [w/2-cw, d/2-cd],
            [w/2-cw, d/2], [-w/2, d/2]]


def _T(w, d, aw):
    return [[-aw/2, -d/2], [aw/2, -d/2], [aw/2, 0], [w/2, 0], [w/2, d/2],
            [-w/2, d/2], [-w/2, 0], [-aw/2, 0]]


def _U(w, d, cw, cd):
    return [[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [w/2-cw, d/2], [w/2-cw, d/2-cd],
            [-w/2+cw, d/2-cd], [-w/2+cw, d/2], [-w/2, d/2]]


def rect(w, d):
    return [[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [-w/2, d/2]]


BLOCK = [
    (_L(16, 18, 8, 9), "victorian", "RESIDENTIAL", 14, (-17, -14)),
    (rect(12, 12), "colonial", "RESIDENTIAL", 10, (2, -16)),
    (_T(16, 14, 7), "modern", "COMMERCIAL", 18, (19, -12)),
    (_U(14, 16, 4.5, 8), "mediterranean", "RESIDENTIAL", 11, (-15, 9)),
    (rect(18, 14), "public_civic", "PUBLIC", 15, (4, 11)),
    (_L(12, 12, 5, 6), "craftsman", "RESIDENTIAL", 8, (21, 10)),
]
# STYLE EMBEDDING demo: the town prompt stays GENERIC — the look comes from a reference
# image via IP-Adapter (the image is encoded once; the embedding is the stored style)
TOWN_PROMPT = ("aerial drone photo of a town block with a tree-lined street, sidewalks, "
               "photorealistic, 50mm, high detail, natural light")
STYLE_REFS = [
    ("amsterdam", "street photo of amsterdam canal houses, dark red brick facades, white "
                  "window frames, gabled roofs, overcast daylight, photorealistic"),
    ("santorini", "photo of santorini greek island architecture, whitewashed walls, blue "
                  "domes and shutters, bright mediterranean sun, photorealistic"),
]
NEG = "cartoon, painting, illustration, low quality, blurry, deformed, text, watermark"


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def fetch_block(device):
    """Per building: bake-quality detail volume + its cube frame -> world placement."""
    items = []
    for (poly, style, cls, h, (px, pz)) in BLOCK:
        b = post("/building_sdf", {"footprint": poly, "style": style,
                                   "building_class": cls, "height": h})
        pv = post("/detail_volume", {"base_sdf_b64": b["sdf_b64"], "res": 64,
                                     "center": b["center"], "scale": b["scale"],
                                     "building_class": cls, "style": style, "seed": 5})
        g = np.frombuffer(base64.b64decode(pv["sdf_b64"]),
                          dtype="<f4").reshape(pv["res"], pv["res"], pv["res"]).copy()
        items.append({
            "vol": torch.as_tensor(g, device=device)[None, None],
            "center": torch.tensor(b["center"], dtype=torch.float32, device=device),
            "scale": float(b["scale"]),
            "pos": torch.tensor([px, 0.0, pz], dtype=torch.float32, device=device),
        })
        print(f"  fetched {style} ({cls}) at ({px},{pz})")
    return items


def _box(p, c, h):
    q = (p - torch.tensor(c, device=p.device)).abs() - torch.tensor(h, device=p.device)
    return q.clamp(min=0).norm(dim=-1) + q.max(-1).values.clamp(max=0)


def _sphere(p, c, r):
    return (p - torch.tensor(c, device=p.device)).norm(dim=-1) - r


# STREET CONTEXT: sidewalks w/ curb edges + trees (canopy+trunk) along the main street
# (z in [-6.5, 6.5] between the two building rows) — real geometry, so it lands in the
# depth + edge G-buffers and grounds the render (kills the maquette look)
SIDEWALKS = [  # (center xyz, half extents)
    ((0.0, 0.075, -5.2), (36.0, 0.075, 1.4)),
    ((0.0, 0.075, 5.0), (36.0, 0.075, 1.4)),
]
TREES = [(x, z) for z in (-5.2, 5.0) for x in (-26, -8, 1, 12, 27)]


def street_sdf(p):
    d = p[:, 1] + 0.0                                      # ground y=0 (road surface)
    for c, h in SIDEWALKS:
        d = torch.minimum(d, _box(p, c, h))
    for (tx, tz) in TREES:
        d = torch.minimum(d, _box(p, (tx, 1.1, tz), (0.14, 1.1, 0.14)))       # trunk
        d = torch.minimum(d, _sphere(p, (tx, 3.0, tz), 1.5))                  # canopy
    return d


def town_sdf(items, device):
    """World-meters SDF of the whole block: buildings + street context."""
    def f(p):                                              # p (N,3) world meters
        d = street_sdf(p)
        for it in items:
            q = (p - it["pos"] - it["center"]) / it["scale"]
            qg = q.clamp(-1.0, 1.0).view(1, 1, 1, -1, 3)
            v = F.grid_sample(it["vol"], qg, mode="bilinear", align_corners=True,
                              padding_mode="border").view(-1) * it["scale"]
            v = v + ((q.abs() - 1.0).clamp(min=0) * it["scale"]).norm(dim=-1)
            d = torch.minimum(d, v)
        return d
    return f


def sphere_trace(sdf, res=RES_IMG, cam_pos=(55.0, 38.0, 62.0), look=(0.0, 4.0, -2.0),
                 fov=42.0, tmax=220.0, device="cuda"):
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
    t = torch.full((N,), 0.5, device=device)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    for _ in range(300):
        p = cp[None] + dirs * t[:, None]
        d = sdf(p)
        t = torch.where(alive, t + d.clamp(min=1e-3) * 0.9, t)
        alive = alive & (d > 2e-2) & (t < tmax)
        if not alive.any():
            break
    p = cp[None] + dirs * t[:, None]
    hit = (sdf(p) < 0.2) & (t < tmax)
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
    return inv.cpu().numpy(), nrm.cpu().numpy(), hit_img.cpu().numpy()


def edges_from_normals(normal, mask):
    """Architectural edge map: discontinuities of the normal field (window reveals,
    cornices, rooflines) -> 1-channel canny-style conditioning image."""
    import cv2
    n8 = (normal * 255).astype(np.uint8)
    gx = np.abs(np.diff(normal, axis=1, prepend=normal[:, :1])).sum(-1)
    gy = np.abs(np.diff(normal, axis=0, prepend=normal[:1])).sum(-1)
    g = ((gx + gy) * mask > 0.25).astype(np.uint8) * 255
    g = cv2.dilate(g, np.ones((2, 2), np.uint8))
    return np.stack([g] * 3, -1)


def main():
    os.makedirs(OUT, exist_ok=True)
    from PIL import Image
    device = "cuda"
    print("[town] fetching block (6 detailed buildings) ...")
    items = fetch_block(device)
    sdf = town_sdf(items, device)
    print("[town] sphere-tracing the block ...")
    with torch.no_grad():
        depth, normal, mask = sphere_trace(sdf, device=device)
    print(f"[town] hit coverage {mask.mean():.2f}")
    edge = edges_from_normals(normal, mask)
    depth_img = Image.fromarray((np.stack([depth] * 3, -1) * 255).astype(np.uint8))
    normal_img = Image.fromarray((normal * 255).astype(np.uint8))
    edge_img = Image.fromarray(edge.astype(np.uint8))
    depth_img.save(os.path.join(OUT, "town_depth.png"))
    normal_img.save(os.path.join(OUT, "town_normal.png"))
    edge_img.save(os.path.join(OUT, "town_edges.png"))

    print("[town] loading SDXL + ControlNets (depth + canny) + IP-Adapter ...")
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
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

    def gen(prompt, control, scales, ref=None, seed=11):
        pipe.set_ip_adapter_scale(0.65 if ref is not None else 0.0)
        kw = dict(ip_adapter_image=ref) if ref is not None else \
            dict(ip_adapter_image=blank)
        return pipe(prompt=prompt, negative_prompt=NEG, image=control,
                    num_inference_steps=30, controlnet_conditioning_scale=scales,
                    generator=torch.Generator("cuda").manual_seed(seed), **kw).images[0]

    renders = []
    for name, ref_prompt in STYLE_REFS:
        ref_path = os.path.join(OUT, f"styleref_{name}.png")
        if os.path.exists(ref_path):
            ref = Image.open(ref_path)
        else:
            print(f"[town] generating style reference '{name}' ...")
            ref = gen(ref_prompt, [blank, blank], [0.0, 0.0], ref=None, seed=23)
            ref.save(ref_path)
        print(f"[town] rendering town in '{name}' style (IP-Adapter embedding) ...")
        img = gen(TOWN_PROMPT, [depth_img, edge_img], [0.85, 0.45], ref=ref)
        img.save(os.path.join(OUT, f"town_style_{name}.png"))
        renders.append((name, ref, img))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("depth + street context", depth_img), ("edges", edge_img)]
    for n, ref, im in renders:
        panels += [(f"style ref · {n}", ref), (f"town in {n} style", im)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.0))
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im); ax.set_title(title, fontsize=9); ax.set_axis_off()
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fig.suptitle("neural appearance v2 — street context + STYLE FROM A REFERENCE IMAGE "
                 "(IP-Adapter embedding; town prompt is generic)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT, f"town_style_{stamp}.png")
    fig.savefig(out, dpi=110)
    print(f"[town] -> {out}")


if __name__ == "__main__":
    main()
