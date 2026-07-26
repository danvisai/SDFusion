"""v2 TEXTURE BAKE — turn a building's geometry into a TEXTURED asset for Unreal.

Pipeline (per building, cube-frame detailed SDF in):
  1. marching cubes -> mesh (cube coords [-1,1])
  2. xatlas -> UV atlas (per-vertex UVs)
  3. N orbit views: sphere-trace depth+edge G-buffers -> SDXL+CN(+IP-Adapter style) ->
     stylized RGB (shared seed + style embedding for cross-view consistency)
  4. rasterize the mesh in UV space -> per-texel 3D position + normal
  5. back-project each texel into every view (visibility + view-facing weighted) -> albedo
     atlas; dilate to fill seams
  6. trimesh with UVs + albedo texture -> glb (a real Unreal material)

Diffusion pixels only ever touch the texture; geometry stays ours/crisp. ~1 min/building
(6 SDXL renders). Style stays consistent across views via one seed + one style embedding,
but independent per-view generation means seams are softened, not pixel-exact — good
"game asset" tier; iterative TEXTure-style refinement is the v2.1 upgrade.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

RES_VIEW = 1024
RES_ATLAS = 1024
N_VIEWS = 6
NEG = "cartoon, painting, illustration, low quality, blurry, deformed, text, watermark, people"


# --------------------------------------------------------------------------- geometry
def _orient_outward(verts, faces, grid, device="cuda"):
    """Flip face winding if it points inward. The (z,y,x)->(x,y,z) axis reorder is a
    reflection that inverts marching-cubes winding -> a viewer culls the OUTSIDE and you
    see the inside walls. Bulletproof check: face normal should align with the SDF gradient
    (outward, since inside<0/outside>0)."""
    vol = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, None]
    fc = verts[faces].mean(1)
    p = torch.as_tensor(fc, dtype=torch.float32, device=device)
    eps = 2.0 / grid.shape[0]

    def s(q):
        qg = q.clamp(-1, 1).view(1, 1, 1, -1, 3)
        return F.grid_sample(vol, qg, mode="bilinear", align_corners=True,
                             padding_mode="border").view(-1)
    grad = torch.stack([s(p + torch.tensor([eps, 0, 0], device=device)) - s(p - torch.tensor([eps, 0, 0], device=device)),
                        s(p + torch.tensor([0, eps, 0], device=device)) - s(p - torch.tensor([0, eps, 0], device=device)),
                        s(p + torch.tensor([0, 0, eps], device=device)) - s(p - torch.tensor([0, 0, eps], device=device))], -1).cpu().numpy()
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    if np.nanmean((fn * grad).sum(1)) < 0:
        faces = np.ascontiguousarray(faces[:, ::-1])
    return faces


def grid_to_mesh_cube(grid):
    """cube-frame SDF grid -> (verts (V,3) in [-1,1] xyz, faces (F,3)), OUTWARD winding."""
    from skimage import measure
    try:
        from scene.mesh_cleanup import cleanup_sdf_grid
        grid = cleanup_sdf_grid(grid)                    # drop floating debris pre-mesh
    except Exception:
        pass
    g = np.ascontiguousarray(grid.astype(np.float32))
    R = g.shape[0]
    v, f, _, _ = measure.marching_cubes(g, level=0.0)
    v = (v[:, [2, 1, 0]] / (R - 1) * 2.0 - 1.0).astype(np.float32)   # (z,y,x)->(x,y,z)
    f = _orient_outward(v, f.astype(np.int64), grid)                # un-flip the reflection
    return v, f


def uv_unwrap(verts, faces):
    """xatlas unwrap. Returns (verts2, faces2, uvs) where uvs in [0,1]^2 per new vertex."""
    import xatlas
    vmap, indices, uvs = xatlas.parametrize(verts, faces)
    return verts[vmap], indices.astype(np.int64), uvs.astype(np.float32)


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.clip(ln, 1e-8, None)


def sdf_gradient_normals(grid, verts, device="cuda"):
    """OUTWARD vertex normals = normalized SDF gradient (inside<0, outside>0 -> grad points
    outward). Unambiguous, unlike marching-cubes winding which can flip the facing test."""
    vol = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, None]
    p = torch.as_tensor(verts, dtype=torch.float32, device=device)
    eps = 2.0 / grid.shape[0]

    def s(q):
        qg = q.clamp(-1, 1).view(1, 1, 1, -1, 3)
        return F.grid_sample(vol, qg, mode="bilinear", align_corners=True,
                             padding_mode="border").view(-1)
    ex = torch.tensor([eps, 0, 0], device=device)
    ey = torch.tensor([0, eps, 0], device=device)
    ez = torch.tensor([0, 0, eps], device=device)
    g = torch.stack([s(p + ex) - s(p - ex), s(p + ey) - s(p - ey), s(p + ez) - s(p - ez)], -1)
    return F.normalize(g, dim=-1).cpu().numpy()


# --------------------------------------------------------------------------- cameras / trace
def occ_frame(grid):
    occ = grid <= 0
    g = np.linspace(-1, 1, grid.shape[0])
    wi, hi, di = np.where(occ.any((0, 1)))[0], np.where(occ.any((0, 2)))[0], np.where(occ.any((1, 2)))[0]
    c = np.array([(g[wi.min()] + g[wi.max()]) / 2, (g[hi.min()] + g[hi.max()]) / 2,
                  (g[di.min()] + g[di.max()]) / 2], np.float32)
    ext = max(g[wi.max()] - g[wi.min()], g[hi.max()] - g[hi.min()], g[di.max()] - g[di.min()])
    return c, float(ext)


def make_cameras(center, ext, n=N_VIEWS):
    r = ext * 1.5 + 0.6
    cams = []
    for i in range(n):
        a = 2 * np.pi * i / n
        cams.append({"pos": (center[0] + r * np.cos(a), center[1] + ext * 0.45,
                             center[2] + r * np.sin(a)), "look": tuple(center), "fov": 40.0})
    # elevated near-top view so roofs get a head-on render (ring views graze them)
    cams.append({"pos": (center[0] + 0.01, center[1] + ext * 1.7 + 0.6, center[2] + 0.01),
                 "look": tuple(center), "fov": 52.0})
    return cams


def trace_view(grid, cam, res=RES_VIEW, device="cuda", aspect=1.0):
    """Returns depth PIL, edge PIL, basis dict (cam_pos, fwd, right, up, t_half, depth metric).

    `res` is the image HEIGHT; width = round(res*aspect). `aspect=1.0` (the default, used by
    every existing bake_texture caller) reproduces the original square capture exactly. A
    non-1.0 `aspect` matches sculpt.html's raymarch shader's `uTan*uAspect` convention, so a
    capture at the LIVE VIEWPORT's own (non-square) aspect ratio unprojects consistently
    (paint_relief.py paints directly on the current view instead of forcing a square crop)."""
    from PIL import Image
    H, W = res, max(1, round(res * aspect))
    vol = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, None]

    def sdf(p):
        out = (p.abs() - 1.0).clamp(min=0.0)
        qg = p.clamp(-1.0, 1.0).view(1, 1, 1, -1, 3)
        return F.grid_sample(vol, qg, mode="bilinear", align_corners=True,
                             padding_mode="border").view(-1) + out.norm(dim=-1)

    cp = torch.tensor(cam["pos"], dtype=torch.float32, device=device)
    fwd = F.normalize(torch.tensor(cam["look"], device=device) - cp, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=device)), dim=0)
    up = torch.linalg.cross(right, fwd)
    th = float(np.tan(np.radians(cam["fov"] / 2)))
    iy = torch.linspace(-1, 1, H, device=device)
    ix = torch.linspace(-1, 1, W, device=device)
    vy, vx = torch.meshgrid(iy, ix, indexing="ij")
    dirs = F.normalize(fwd[None, None] + vx[..., None] * right * (th * aspect)
                       - vy[..., None] * up * th, dim=-1).reshape(-1, 3)
    t = torch.full((dirs.shape[0],), 0.05, device=device)
    alive = torch.ones_like(t, dtype=torch.bool)
    for _ in range(240):
        p = cp[None] + dirs * t[:, None]
        d = sdf(p)
        t = torch.where(alive, t + d.clamp(min=1e-4) * 0.9, t)
        alive = alive & (d > 2.5e-3) & (t < 8.0)
        if not alive.any():
            break
    p = (cp[None] + dirs * t[:, None])
    hit = (sdf(p.clamp(-1, 1)) < 1.5e-2) & (t < 8.0)
    eps = 2.0 / grid.shape[0]
    pc = p.clamp(-1, 1)
    n = torch.stack([sdf(pc + torch.tensor([eps, 0, 0], device=device)) - sdf(pc - torch.tensor([eps, 0, 0], device=device)),
                     sdf(pc + torch.tensor([0, eps, 0], device=device)) - sdf(pc - torch.tensor([0, eps, 0], device=device)),
                     sdf(pc + torch.tensor([0, 0, eps], device=device)) - sdf(pc - torch.tensor([0, 0, eps], device=device))], -1)
    n = F.normalize(n, dim=-1)
    t_img, hit_img = t.view(H, W), hit.view(H, W)
    inv = torch.zeros_like(t_img)
    if hit_img.any():
        tq = t_img[hit_img]
        lo, hi = tq.min(), tq.max()
        inv[hit_img] = 1.0 - 0.85 * (t_img[hit_img] - lo) / (hi - lo + 1e-6)
    nrm = ((n.view(H, W, 3) * 0.5 + 0.5) * hit_img[..., None]).clamp(0, 1).cpu().numpy()
    gx = np.abs(np.diff(nrm, axis=1, prepend=nrm[:, :1])).sum(-1)
    gy = np.abs(np.diff(nrm, axis=0, prepend=nrm[:1])).sum(-1)
    edge = ((gx + gy) * hit_img.cpu().numpy() > 0.25).astype(np.uint8) * 255
    depth_img = Image.fromarray((np.stack([inv.cpu().numpy()] * 3, -1) * 255).astype(np.uint8))
    edge_img = Image.fromarray(np.stack([edge] * 3, -1))
    basis = {"cam_pos": np.array(cam["pos"], np.float32),
             "fwd": fwd.cpu().numpy(), "right": right.cpu().numpy(), "up": up.cpu().numpy(),
             "th": th, "depth": t_img.cpu().numpy(), "hit": hit_img.cpu().numpy(), "res": res,
             "aspect": aspect, "dirs": dirs.view(H, W, 3).cpu().numpy()}
    return depth_img, edge_img, basis


def stylize(pipe, depth_img, edge_img, prompt, style_ref, seed, steps=28):
    from PIL import Image
    blank = Image.new("RGB", depth_img.size, 0)
    pipe.set_ip_adapter_scale(0.6 if style_ref is not None else 0.0)
    img = pipe(prompt=prompt, negative_prompt=NEG, image=[depth_img, edge_img],
               num_inference_steps=int(steps), controlnet_conditioning_scale=[0.9, 0.5],
               ip_adapter_image=style_ref if style_ref is not None else blank,
               generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
    return np.asarray(img.convert("RGB"), np.float32) / 255.0


# --------------------------------------------------------------------------- UV rasterize
def rasterize_uv(uvs, faces, vert_pos, vert_nrm, res=RES_ATLAS):
    """Render the mesh into UV space -> per-texel 3D position + normal + validity mask."""
    pos = np.zeros((res, res, 3), np.float32)
    nrm = np.zeros((res, res, 3), np.float32)
    mask = np.zeros((res, res), bool)
    P = uvs * (res - 1)                                      # uv -> pixel coords
    for f in faces:
        p0, p1, p2 = P[f]
        x0 = int(max(np.floor(min(p0[0], p1[0], p2[0])), 0))
        x1 = int(min(np.ceil(max(p0[0], p1[0], p2[0])), res - 1))
        y0 = int(max(np.floor(min(p0[1], p1[1], p2[1])), 0))
        y1 = int(min(np.ceil(max(p0[1], p1[1], p2[1])), res - 1))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
        d = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(d) < 1e-9:
            continue
        a = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / d
        b = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / d
        c = 1 - a - b
        inside = (a >= -1e-4) & (b >= -1e-4) & (c >= -1e-4)
        if not inside.any():
            continue
        a, b, c = a[inside], b[inside], c[inside]
        gx, gy = xs.ravel()[inside], ys.ravel()[inside]
        vp = vert_pos[f]; vn = vert_nrm[f]
        pos[gy, gx] = a[:, None] * vp[0] + b[:, None] * vp[1] + c[:, None] * vp[2]
        nrm[gy, gx] = a[:, None] * vn[0] + b[:, None] * vn[1] + c[:, None] * vn[2]
        mask[gy, gx] = True
    ln = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = nrm / np.clip(ln, 1e-8, None)
    return pos, nrm, mask


def backproject(points, normals, views, rgbs, tol=0.06, fpow=3.0):
    """points (M,3), normals (M,3); views list of basis; rgbs list of (res,res,3) in [0,1].
    Returns colors (M,3), covered (M,) bool."""
    M = points.shape[0]
    acc = np.zeros((M, 3), np.float32)
    wsum = np.zeros((M, 1), np.float32)
    for basis, rgb in zip(views, rgbs):
        cp, fwd, right, up = basis["cam_pos"], basis["fwd"], basis["right"], basis["up"]
        th, res, depth = basis["th"], basis["res"], basis["depth"]
        rel = points - cp[None]
        z = rel @ fwd
        front = z > 1e-3
        vx = (rel @ right) / (z * th + 1e-9)
        vy = -(rel @ up) / (z * th + 1e-9)
        col = ((vx + 1) * 0.5 * (res - 1)).round().astype(int)
        row = ((vy + 1) * 0.5 * (res - 1)).round().astype(int)
        on = front & (col >= 0) & (col < res) & (row >= 0) & (row < res)
        idx = np.where(on)[0]
        if not len(idx):
            continue
        rr, cc = row[idx], col[idx]
        dist = np.linalg.norm(rel[idx], axis=1)
        vis = np.abs(dist - depth[rr, cc]) < tol            # frontmost surface in this view
        face = normals[idx] @ (-fwd)                         # texel facing the camera
        w = np.clip(face, 0, 1) ** fpow * vis
        good = w > 1e-4
        ii = idx[good]
        acc[ii] += rgb[rr[good], cc[good]] * w[good, None]
        wsum[ii, 0] += w[good]
    covered = wsum[:, 0] > 1e-5
    colors = np.full((M, 3), 0.5, np.float32)
    colors[covered] = acc[covered] / wsum[covered]
    return colors, covered


# per-style PBR base (roughness, metallic) — architecture is mostly dielectric; modern/
# industrial carry some glass/steel. Roughness is then modulated per-texel by albedo detail.
STYLE_PBR = {
    "modern": (0.40, 0.12), "contemporary": (0.35, 0.15), "colonial": (0.80, 0.0),
    "victorian": (0.82, 0.0), "industrial": (0.55, 0.22), "craftsman": (0.80, 0.0),
    "mediterranean": (0.78, 0.0), "public_civic": (0.62, 0.05),
}


def make_pbr_maps(albedo_u8, style="modern", normal_strength=2.2):
    """Derive PBR maps from the baked albedo (no training):
      - NORMAL: height-from-luminance (brick/mortar/window-frame relief) -> tangent normal
      - METALLIC-ROUGHNESS (glTF: G=roughness, B=metallic): per-style base roughness,
        lowered where the albedo is smooth/uniform (glass/flat), metallic per style.
    Returns (normal_u8, metallic_rough_u8, base_rough, base_metal)."""
    import cv2
    a = albedo_u8.astype(np.float32) / 255.0
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    h = cv2.GaussianBlur(lum, (0, 0), 1.0)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    n = np.stack([-gx * normal_strength, -gy * normal_strength, np.ones_like(h)], -1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    normal_u8 = ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)   # OpenGL +Y normal map
    base_r, base_m = STYLE_PBR.get(style, (0.7, 0.0))
    mlum = cv2.GaussianBlur(lum, (0, 0), 3)
    var = np.clip(cv2.GaussianBlur(lum * lum, (0, 0), 3) - mlum * mlum, 0, None)
    var = var / (var.mean() + 1e-6)
    rough = np.clip(base_r * (0.55 + 0.45 * np.clip(var, 0, 1.3) / 1.3), 0.05, 1.0)
    metal = np.full_like(rough, base_m)
    mr_u8 = (np.stack([np.zeros_like(rough), rough, metal], -1) * 255).clip(0, 255).astype(np.uint8)
    return normal_u8, mr_u8, float(base_r), float(base_m)


def _project(points, basis):
    """points (M,3) cube coords -> (row,col,visible,front) in a traced view's image."""
    cp, fwd, right, up = basis["cam_pos"], basis["fwd"], basis["right"], basis["up"]
    th, res, depth = basis["th"], basis["res"], basis["depth"]
    rel = points - cp[None]
    z = rel @ fwd
    vx = (rel @ right) / (z * th + 1e-9)
    vy = -(rel @ up) / (z * th + 1e-9)
    col = ((vx + 1) * 0.5 * (res - 1)).round().astype(int)
    row = ((vy + 1) * 0.5 * (res - 1)).round().astype(int)
    on = (z > 1e-3) & (col >= 0) & (col < res) & (row >= 0) & (row < res)
    dist = np.linalg.norm(rel, axis=1)
    vis = np.zeros(len(points), bool)
    idx = np.where(on)[0]
    vis[idx] = np.abs(dist[idx] - depth[row[idx], col[idx]]) < 0.06
    return row, col, vis, dist


def splat_atlas_to_view(flat_pos, colors, covered, basis):
    """Forward-render the so-far texture into a view -> (init_rgb, known_mask). Nearest texel
    wins per pixel (z-buffer) so back faces don't bleed through."""
    res = basis["res"]
    row, col, vis, dist = _project(flat_pos, basis)
    sel = vis & covered
    init = np.full((res, res, 3), 0.5, np.float32)
    known = np.zeros((res, res), bool)
    idx = np.where(sel)[0]
    if len(idx):
        order = idx[np.argsort(-dist[idx])]                 # far first, near overwrites
        init[row[order], col[order]] = colors[order]
        known[row[order], col[order]] = True
    return init, known


def dilate_atlas(atlas, mask, iters=8):
    """Fill unbaked texels: spread baked color outward a few px (covers UV seams), then
    inpaint the remaining holes so no gray fallback shows through at chart gaps."""
    import cv2
    a = (np.clip(atlas, 0, 1) * 255).astype(np.uint8)
    m = mask.astype(np.uint8) * 255
    for _ in range(iters):                                   # spread color into nearby holes
        dm = cv2.dilate(m, np.ones((3, 3), np.uint8))
        grow = (m == 0) & (dm > 0)
        if not grow.any():
            break
        d = cv2.dilate(a, np.ones((3, 3), np.uint8))
        a[grow] = d[grow]
        m[grow] = 255
    holes = (m == 0).astype(np.uint8)
    if holes.any():
        a = cv2.inpaint(a, holes, 4, cv2.INPAINT_TELEA)
    return a


# --------------------------------------------------------------------------- top-level bake
def bake_building(grid, pipe, prompt, style_ref=None, seed=7, n_views=N_VIEWS,
                  atlas_res=RES_ATLAS, view_res=RES_VIEW, steps=28, style="modern",
                  return_views=False):
    """grid: cube-frame detailed SDF. Returns dict with textured (PBR) trimesh, atlas, views."""
    import trimesh
    from PIL import Image
    verts, faces = grid_to_mesh_cube(grid)
    verts, faces, uvs = uv_unwrap(verts, faces)
    vn = sdf_gradient_normals(grid, verts)                   # OUTWARD (gradient), not winding
    center, ext = occ_frame(grid)
    cams = make_cameras(center, ext, n_views)

    views, rgbs, sheet = [], [], []
    for ci, cam in enumerate(cams):
        depth_img, edge_img, basis = trace_view(grid, cam, res=view_res)
        rgb = stylize(pipe, depth_img, edge_img, prompt, style_ref, seed, steps=steps)
        views.append(basis); rgbs.append(rgb)
        if return_views:
            sheet.append(rgb)

    pos, nrm, mask = rasterize_uv(uvs, faces, verts, vn, atlas_res)
    flat_pos = pos[mask]; flat_nrm = nrm[mask]
    colors, covered = backproject(flat_pos, flat_nrm, views, rgbs)
    atlas = np.zeros((atlas_res, atlas_res, 3), np.float32)
    atlas[mask] = colors
    atlas_u8 = dilate_atlas(atlas, mask)
    atlas_img = Image.fromarray(atlas_u8)

    # PBR: derive normal + metallic-roughness from the albedo + style
    normal_u8, mr_u8, rf, mf = make_pbr_maps(atlas_u8, style)
    from trimesh.visual.material import PBRMaterial
    flip = lambda im: im.transpose(Image.FLIP_TOP_BOTTOM)        # glTF UV origin = bottom-left
    mat = PBRMaterial(baseColorTexture=flip(atlas_img),
                      normalTexture=flip(Image.fromarray(normal_u8)),
                      metallicRoughnessTexture=flip(Image.fromarray(mr_u8)),
                      roughnessFactor=1.0, metallicFactor=1.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)
    out = {"mesh": mesh, "atlas": atlas_img, "normal": Image.fromarray(normal_u8),
           "mr": Image.fromarray(mr_u8), "coverage": float(covered.mean()),
           "n_verts": len(verts), "pbr": (rf, mf)}
    if return_views:
        out["views"] = sheet
    return out


def _finalize(verts, faces, uvs, mask, colors, atlas_res, style, covered):
    """colors (per-texel) -> textured PBR trimesh + maps."""
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    atlas = np.zeros((atlas_res, atlas_res, 3), np.float32)
    atlas[mask] = colors
    atlas_u8 = dilate_atlas(atlas, mask)
    atlas_img = Image.fromarray(atlas_u8)
    normal_u8, mr_u8, rf, mf = make_pbr_maps(atlas_u8, style)
    flip = lambda im: im.transpose(Image.FLIP_TOP_BOTTOM)
    matl = PBRMaterial(baseColorTexture=flip(atlas_img),
                       normalTexture=flip(Image.fromarray(normal_u8)),
                       metallicRoughnessTexture=flip(Image.fromarray(mr_u8)),
                       roughnessFactor=1.0, metallicFactor=1.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=matl)
    return {"mesh": mesh, "atlas": atlas_img, "normal": Image.fromarray(normal_u8),
            "mr": Image.fromarray(mr_u8), "coverage": float(covered.mean()),
            "n_verts": len(verts), "pbr": (rf, mf)}


def _inpaint_view(ipipe, init_img, mask_img, depth_img, edge_img, prompt, style_ref, seed, steps,
                  strength=1.0, extra_control_images=None, extra_control_scales=None,
                  base_control_scales=None, negative=None):
    """strength=1.0 (bake_texture's default) ignores init_img's CONTENT entirely — SDXL
    inpainting at full strength regenerates from noise, using init_img only for size/masking.
    A lower strength (e.g. ~0.6) anchors the img2img denoising to init_img's actual pixels,
    so a user-painted/uploaded image comes through recognizably instead of being hallucinated
    over (paint_relief's 'blend' mode).

    `extra_control_images`/`extra_control_scales`: additional ControlNet conditioning images
    (e.g. paint_relief's scribble ControlNet) appended after the depth+canny pair — `ipipe`
    must have a matching number of ControlNets loaded (neural_appearance.get_sketch_inpaint_pipe).
    `base_control_scales` overrides the depth+canny pair's weights (default [0.9, 0.5], the
    bake contract); paint_relief's wall-space canvas has no real depth/edge content, so it
    passes these near-zero rather than letting a blank conditioning image fight the scribble.
    `negative` overrides the bake-oriented NEG (paint_relief needs anti-facade terms instead).
    """
    from PIL import Image
    blank = Image.new("RGB", init_img.size, 0)
    ipipe.set_ip_adapter_scale(0.6 if style_ref is not None else 0.0)
    control_images = [depth_img, edge_img] + list(extra_control_images or [])
    control_scales = list(base_control_scales or [0.9, 0.5]) + list(extra_control_scales or [])
    out = ipipe(prompt=prompt, negative_prompt=(negative or NEG), image=init_img, mask_image=mask_img,
                control_image=control_images, controlnet_conditioning_scale=control_scales,
                strength=float(strength), num_inference_steps=int(steps),
                ip_adapter_image=style_ref if style_ref is not None else blank,
                generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
    return np.asarray(out.convert("RGB"), np.float32) / 255.0


def bake_building_iterative(grid, pipe, inpaint_pipe, prompt, style_ref=None, seed=7,
                            n_views=N_VIEWS, atlas_res=RES_ATLAS, view_res=RES_VIEW, steps=28,
                            style="modern", return_views=False):
    """TEXTure-style: view 0 generated fully; each later view INPAINTS only the newly-revealed
    region over the texture-so-far (projected in), so new content continues the existing
    texture -> seam-free, cross-view consistent."""
    import cv2
    from PIL import Image
    verts, faces = grid_to_mesh_cube(grid)
    verts, faces, uvs = uv_unwrap(verts, faces)
    vn = sdf_gradient_normals(grid, verts)
    center, ext = occ_frame(grid)
    cams = make_cameras(center, ext, n_views)
    pos, nrm, mask = rasterize_uv(uvs, faces, verts, vn, atlas_res)
    flat_pos, flat_nrm = pos[mask], nrm[mask]
    M = flat_pos.shape[0]
    acc = np.zeros((M, 3), np.float32)
    wsum = np.zeros((M,), np.float32)
    cur = np.full((M, 3), 0.5, np.float32)
    sheet = []
    for k, cam in enumerate(cams):
        depth_img, edge_img, basis = trace_view(grid, cam, res=view_res)
        covered = wsum > 1e-5
        if k == 0 or covered.sum() < 50:
            rgb = stylize(pipe, depth_img, edge_img, prompt, style_ref, seed, steps=steps)
        else:
            init_np, known = splat_atlas_to_view(flat_pos, cur, covered, basis)
            known = cv2.dilate(known.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            gen = basis["hit"] & ~known
            if gen.sum() < 200:
                rgb = init_np                            # nothing new in this view
            else:
                init_img = Image.fromarray((np.clip(init_np, 0, 1) * 255).astype(np.uint8))
                mask_img = Image.fromarray((gen.astype(np.uint8)) * 255)
                rgb = _inpaint_view(inpaint_pipe, init_img, mask_img, depth_img, edge_img,
                                    prompt, style_ref, seed, steps)
        if return_views:
            sheet.append(rgb)
        row, col, vis, _ = _project(flat_pos, basis)
        w = np.clip(flat_nrm @ (-basis["fwd"]), 0, 1) ** 3 * vis
        ii = np.where(w > 1e-4)[0]
        acc[ii] += rgb[row[ii], col[ii]] * w[ii, None]
        wsum[ii] += w[ii]
        cov = wsum > 1e-5
        cur[cov] = acc[cov] / wsum[cov, None]
    covered = wsum > 1e-5
    colors = np.full((M, 3), 0.5, np.float32)
    colors[covered] = acc[covered] / wsum[covered, None]
    out = _finalize(verts, faces, uvs, mask, colors, atlas_res, style, covered)
    if return_views:
        out["views"] = sheet
    return out


def bake_glb(grid96, pipe, prompt, style_ref=None, seed=7, n_views=N_VIEWS, steps=28,
             world_scale=1.0, unit=100.0, style="modern", inpaint_pipe=None):
    """Bake + return (glb_bytes, coverage, n_verts). cube[-1,1] mesh -> meters*unit, base on
    ground. unit=100 -> centimeters (Unreal); world_scale = the building's world half-extent.
    glb carries a full PBR material (albedo + normal + metallic-roughness). inpaint_pipe set
    -> iterative TEXTure-style (seam-free)."""
    if inpaint_pipe is not None:
        res = bake_building_iterative(grid96, pipe, inpaint_pipe, prompt, style_ref=style_ref,
                                      seed=seed, n_views=n_views, steps=steps, style=style)
    else:
        res = bake_building(grid96, pipe, prompt, style_ref=style_ref, seed=seed,
                            n_views=n_views, steps=steps, style=style)
    mesh = res["mesh"]
    mesh.apply_scale(float(world_scale) * float(unit))      # UVs in [0,1] are unaffected
    v = np.asarray(mesh.vertices, np.float32)
    v[:, 1] -= v[:, 1].min()                                # sit on y=0
    mesh.vertices = v
    return mesh.export(file_type="glb"), res["coverage"], res["n_verts"]


if __name__ == "__main__":
    # single-building offline demo: fetch a detailed volume from the server, bake, save
    import base64
    import json
    import os
    import sys
    import urllib.request

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
    import neural_appearance as na  # noqa: E402

    OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "appearance_v0")
    os.makedirs(OUT, exist_ok=True)
    URL = "http://127.0.0.1:8099"

    def post(p, b):
        r = urllib.request.Request(URL + p, data=json.dumps(b).encode(),
                                   headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=600).read())

    RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
    style, cls, h = "victorian", "RESIDENTIAL", 14
    g = post("/building_sdf", {"footprint": RECT, "style": style, "building_class": cls, "height": h})
    pv = post("/detail_volume", {"base_sdf_b64": g["sdf_b64"], "res": 64,
                                 "center": g["center"], "scale": g["scale"],
                                 "building_class": cls, "style": style, "seed": 3})
    grid = np.frombuffer(base64.b64decode(pv["sdf_b64"]), dtype="<f4").reshape(pv["res"], pv["res"], pv["res"]).copy()

    style_ref = None
    ref_path = os.path.join(OUT, "styleref_amsterdam.png")
    if os.path.exists(ref_path):
        from PIL import Image as _I
        style_ref = _I.open(ref_path).convert("RGB")

    print("[bake] loading SDXL ...")
    pipe = na.get_pipe()
    prompt = "photo of a victorian brick townhouse, ornate windows, slate roof, architectural photography"
    print("[bake] baking texture (6 views) ...")
    res = bake_building(grid, pipe, prompt, style_ref=style_ref, seed=7, return_views=True)
    print(f"[bake] coverage {res['coverage']:.2f} · {res['n_verts']} verts")
    res["atlas"].save(os.path.join(OUT, "bake_atlas.png"))
    glb = res["mesh"].export(file_type="glb")
    open(os.path.join(OUT, "bake_building.glb"), "wb").write(glb)
    print(f"[bake] -> {OUT}/bake_building.glb ({len(glb)//1024} KB), bake_atlas.png")

    # verification: vertex-colored preview (sample atlas at each vertex UV)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    atl = np.asarray(res["atlas"], np.float32) / 255.0
    uv = res["mesh"].visual.uv
    R = atl.shape[0]
    vc = atl[np.clip(((1 - uv[:, 1]) * (R - 1)).astype(int), 0, R - 1),
             np.clip((uv[:, 0] * (R - 1)).astype(int), 0, R - 1)]
    V, Fc = res["mesh"].vertices, res["mesh"].faces
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    fcol = vc[Fc].mean(1)
    tri = ax.plot_trisurf(V[:, 0], V[:, 2], Fc, V[:, 1], linewidth=0, antialiased=False, shade=False)
    tri.set_fc(fcol)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=12, azim=-60)
    ax.set_title("textured building (baked albedo on geometry)", fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(res["atlas"]); ax2.set_axis_off()
    ax2.set_title(f"UV albedo atlas · coverage {res['coverage']:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bake_preview.png"), dpi=110)
    print(f"[bake] preview -> {OUT}/bake_preview.png")
