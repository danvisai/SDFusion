"""
Bake BuildingNet OBJs into per-asset 3D Gaussian Splat PLYs.

Pipeline per id:
  1. Load OBJ (trimesh, geometry only), normalize to centered unit-extent frame (Frame N).
  2. Build N orbit views (default 24 = 3 elevations x 8 azimuths) at fixed dist+FoV.
  3. Render targets with pytorch3d (matches existing scripts/render_buildingnet_objfiles.py
     lighting/look so the splat we fit matches downstream renders).
  4. Initialize 3DGS at mesh vertices (subsampled to a cap).
  5. Optimize with gsplat against the 24 targets via L1 + 0.2 * (1 - SSIM-like).
  6. Save Inria-format PLY at data/BuildingNet_dataset_v0_1/gaussian_splats/<id>.ply.

Output PLY format follows the Inria 3DGS convention (degree-0 SH):
  x,y,z, nx,ny,nz, f_dc_0..2, opacity, scale_0..2, rot_0..3

Camera convention:
  pytorch3d (+x LEFT, +y up, +z into screen)  →  flip [-1,-1,1,1] on viewmat  →
  gsplat / OpenCV (+x right, +y down, +z forward)

Run smoke (12 ids):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
    scripts/bake_buildingnet_to_3dgs.py --limit 12 \
    --out_dir data/BuildingNet_dataset_v0_1/gaussian_splats_smoke
"""
from __future__ import annotations
import argparse
import math
import os
import time

import numpy as np
import torch
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement


# --- mesh utilities -----------------------------------------------------------

def load_obj_as_trimesh(obj_path):
    """trimesh.load with concat-into-single-mesh; geometry only."""
    loaded = trimesh.load(obj_path, force="mesh", process=False)
    if loaded is None or not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
        return None
    return loaded


def normalize_to_unit_extent(vertices: np.ndarray):
    """Center mesh at origin and scale so max-axis half-extent == 1."""
    v = vertices.astype(np.float32, copy=True)
    centre = (v.max(0) + v.min(0)) / 2
    v -= centre
    extent = float(np.abs(v).max())
    if extent > 1e-9:
        v /= extent
    return v


# --- camera setup -------------------------------------------------------------

def orbit_camera_grid(n_elev=3, n_azim=8, elev_range=(10.0, 50.0)):
    """Return (elevs, azims) arrays of length n_elev*n_azim covering an orbit."""
    elevs = np.linspace(elev_range[0], elev_range[1], n_elev)
    azims = np.linspace(0, 360, n_azim, endpoint=False)
    grid_e, grid_a = np.meshgrid(elevs, azims, indexing="ij")
    return grid_e.reshape(-1), grid_a.reshape(-1)


def build_views(elevs, azims, dist, device, fov_deg, image_size):
    """Return (R_p3d, T_p3d, viewmats_gsplat (V,4,4), K_gsplat (V,3,3)).

    pytorch3d coord system: +x LEFT, +y UP, +z INTO scene.
    OpenCV / gsplat:        +x RIGHT, +y DOWN, +z INTO scene.
    Conversion: left-multiply pytorch3d-col viewmat by diag(-1, -1, 1, 1).
    """
    from pytorch3d.renderer import look_at_view_transform

    R_p3d, T_p3d = look_at_view_transform(
        dist=dist,
        elev=torch.tensor(elevs, dtype=torch.float32),
        azim=torch.tensor(azims, dtype=torch.float32),
        at=((0.0, 0.0, 0.0),),
    )
    V = R_p3d.shape[0]
    # pytorch3d row-vec → col-vec viewmat
    viewmat = torch.eye(4, device=device).unsqueeze(0).repeat(V, 1, 1)
    viewmat[:, :3, :3] = R_p3d.transpose(-1, -2).to(device)
    viewmat[:, :3, 3] = T_p3d.to(device)
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device)).unsqueeze(0)
    viewmat_gsplat = flip @ viewmat  # (V, 4, 4)

    fy = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fx = fy
    cx = cy = image_size / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(V, 3, 3).contiguous()
    return R_p3d.to(device), T_p3d.to(device), viewmat_gsplat, K


# --- target render (pytorch3d) -----------------------------------------------

def render_targets_pytorch3d(verts_np, faces_np, R_p3d, T_p3d, fov_deg, image_size, device):
    """Render V target views with pytorch3d. Returns (V, H, W, 3) float in [0,1]."""
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, SoftPhongShader,
        RasterizationSettings, PointLights, BlendParams, TexturesVertex,
    )
    from pytorch3d.structures import Meshes

    V = R_p3d.shape[0]
    cameras = FoVPerspectiveCameras(device=device, R=R_p3d, T=T_p3d, fov=fov_deg)
    rs = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1, bin_size=0,
    )
    lights = PointLights(
        device=device,
        location=((2.0, 2.0, 2.0),),
        ambient_color=((0.55, 0.55, 0.55),),
        diffuse_color=((0.45, 0.45, 0.45),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rs),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend),
    )
    v = torch.from_numpy(verts_np).to(device)
    f = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    col = torch.full_like(v.unsqueeze(0), 0.78).expand(V, -1, -1).contiguous()
    v_b = v.unsqueeze(0).expand(V, -1, -1).contiguous()
    f_b = f.unsqueeze(0).expand(V, -1, -1).contiguous()
    pmesh = Meshes(verts=v_b, faces=f_b, textures=TexturesVertex(verts_features=col))
    rgba = renderer(pmesh)  # (V, H, W, 4)
    return rgba[..., :3].clamp(0, 1).detach()


# --- Gaussian init ------------------------------------------------------------

SH_C0 = 0.28209479177387814  # SH degree-0 normalization

def init_gaussians_from_mesh(verts_np, n_max, init_color, device, knn_scale_k=0):
    """Sample up to n_max points uniformly from mesh vertices for Gaussian centers.

    If knn_scale_k > 0, initial per-point log-scale = log(mean distance to
    knn_scale_k nearest neighbors). Otherwise use a constant log(0.015).
    """
    if len(verts_np) > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(verts_np), n_max, replace=False)
        v = verts_np[idx]
    else:
        v = verts_np
    N = v.shape[0]
    means = torch.from_numpy(v.astype(np.float32)).to(device)

    if knn_scale_k > 0 and N > knn_scale_k + 1:
        # Use scipy KDTree for the kNN query (fast, robust, no extra dep beyond scipy).
        from scipy.spatial import cKDTree
        tree = cKDTree(v.astype(np.float64))
        d, _ = tree.query(v, k=knn_scale_k + 1)  # +1 because each point's own neighbor is itself
        mean_d = d[:, 1:].mean(axis=1)
        mean_d = np.clip(mean_d, 1e-4, 1.0)
        log_scale = torch.from_numpy(np.log(mean_d.astype(np.float32))).to(device)
        raw_scales = log_scale.unsqueeze(-1).expand(-1, 3).contiguous()
    else:
        raw_scales = torch.full((N, 3), math.log(0.015), device=device)

    raw_quats = torch.zeros(N, 4, device=device)
    raw_quats[:, 0] = 1.0
    raw_opacities = torch.full((N,), -2.0, device=device)
    sh_dc = torch.full((N, 3), (init_color - 0.5) / SH_C0, device=device)
    return means, raw_scales, raw_quats, raw_opacities, sh_dc


# --- SSIM ---------------------------------------------------------------------

def _gaussian_window_2d(window_size: int, sigma: float, device, dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)  # (W, W)
    return window


def ssim_loss(pred_hw3: torch.Tensor, target_hw3: torch.Tensor,
              window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1 - mean SSIM between two (H, W, 3) images in [0,1]. Returns a scalar."""
    device = pred_hw3.device
    win = _gaussian_window_2d(window_size, sigma, device, pred_hw3.dtype)
    # (1, 3, kW, kW) — depthwise conv per channel
    win3 = win.unsqueeze(0).unsqueeze(0).expand(3, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    p = pred_hw3.permute(2, 0, 1).unsqueeze(0)
    t = target_hw3.permute(2, 0, 1).unsqueeze(0)
    mu_p = torch.nn.functional.conv2d(p, win3, padding=pad, groups=3)
    mu_t = torch.nn.functional.conv2d(t, win3, padding=pad, groups=3)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sigma_p2 = torch.nn.functional.conv2d(p * p, win3, padding=pad, groups=3) - mu_p2
    sigma_t2 = torch.nn.functional.conv2d(t * t, win3, padding=pad, groups=3) - mu_t2
    sigma_pt = torch.nn.functional.conv2d(p * t, win3, padding=pad, groups=3) - mu_pt
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu_pt + c1) * (2.0 * sigma_pt + c2)) / (
        (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    )
    return 1.0 - ssim_map.mean()


# --- optimization -------------------------------------------------------------

def quat_normalize(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


def train_3dgs_for_asset(
    targets,           # (V, H, W, 3) float in [0,1]
    viewmats,          # (V, 4, 4) world-to-camera (gsplat/OpenCV col-vec)
    K,                 # (V, 3, 3) intrinsics
    width, height,
    init_means, init_scales, init_quats, init_opac, init_sh_dc,
    n_iters=7000,
    lr_means=1.6e-4, lr_scales=5e-3, lr_quats=1e-3, lr_opac=5e-2, lr_sh=2.5e-3,
    bg_white=True, log_every=500, device="cuda",
):
    from gsplat import rasterization

    means = init_means.clone().requires_grad_(True)
    scales = init_scales.clone().requires_grad_(True)
    quats = init_quats.clone().requires_grad_(True)
    opacities = init_opac.clone().requires_grad_(True)
    sh_dc = init_sh_dc.clone().requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [means], "lr": lr_means},
        {"params": [scales], "lr": lr_scales},
        {"params": [quats], "lr": lr_quats},
        {"params": [opacities], "lr": lr_opac},
        {"params": [sh_dc], "lr": lr_sh},
    ], eps=1e-15)

    bg = torch.ones(3, device=device) if bg_white else torch.zeros(3, device=device)
    V = targets.shape[0]

    t0 = time.time()
    for it in range(n_iters):
        # Pick one view per iter
        vidx = it % V
        vm = viewmats[vidx:vidx + 1]  # (1, 4, 4)
        Kv = K[vidx:vidx + 1]         # (1, 3, 3)
        tgt = targets[vidx]           # (H, W, 3)

        # Activated params
        act_scales = torch.exp(scales)
        act_quats = quat_normalize(quats)
        act_opac = torch.sigmoid(opacities)
        # SH degree-0 RGB color
        colors = sh_dc * SH_C0 + 0.5

        out, alphas, _meta = rasterization(
            means=means,
            quats=act_quats,
            scales=act_scales,
            opacities=act_opac,
            colors=colors,
            viewmats=vm,
            Ks=Kv,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB",
            backgrounds=bg.unsqueeze(0),
        )
        # out: (1, H, W, 3) when render_mode=RGB
        pred = out[0]
        # L1 + simple low-pass smoothness via downsampled L1 (cheap stand-in for SSIM)
        l1 = (pred - tgt).abs().mean()
        # Downsampled L1 (4x): captures structure without full SSIM dependency
        pred_ds = torch.nn.functional.avg_pool2d(pred.permute(2, 0, 1).unsqueeze(0), 4)
        tgt_ds = torch.nn.functional.avg_pool2d(tgt.permute(2, 0, 1).unsqueeze(0), 4)
        l1_ds = (pred_ds - tgt_ds).abs().mean()
        loss = l1 + 0.2 * l1_ds

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if log_every and (it % log_every == 0 or it == n_iters - 1):
            with torch.no_grad():
                n_visible = (act_opac > 0.05).sum().item()
            print(f"    iter {it:5d}/{n_iters}  loss={loss.item():.4f}  "
                  f"l1={l1.item():.4f}  visible={n_visible}/{means.shape[0]}  "
                  f"elapsed={time.time()-t0:.1f}s")

    return means.detach(), scales.detach(), quats.detach(), opacities.detach(), sh_dc.detach()


def train_3dgs_for_asset_v2(
    targets,
    viewmats,
    K,
    width, height,
    init_means, init_scales, init_quats, init_opac, init_sh_dc,
    n_iters: int = 15000,
    lr_means: float = 1.6e-4, lr_scales: float = 5e-3, lr_quats: float = 1e-3,
    lr_opac: float = 5e-2, lr_sh: float = 2.5e-3,
    w_ssim: float = 0.2,
    scene_scale: float = 1.0,
    log_every: int = 1000,
    device="cuda",
):
    """v2 trainer: gsplat DefaultStrategy densification + real SSIM loss.

    Densification follows the Inria 3DGS schedule (clone/split/prune every 100
    iters between iter 500 and iter min(n_iters, 15000); reset opacities every
    3000 iters). The Gaussian count grows dynamically from the initial set.
    """
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    # gsplat strategy mutates params; they must be a ParameterDict with the
    # standard keys means/scales/quats/opacities and each must have its own
    # single-group Adam optimizer.
    params = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(init_means.clone()),
        "scales":    torch.nn.Parameter(init_scales.clone()),
        "quats":     torch.nn.Parameter(init_quats.clone()),
        "opacities": torch.nn.Parameter(init_opac.clone()),
        "sh0":       torch.nn.Parameter(init_sh_dc.clone()),
    }).to(device)
    lrs = {
        "means": lr_means, "scales": lr_scales, "quats": lr_quats,
        "opacities": lr_opac, "sh0": lr_sh,
    }
    optimizers = {
        k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params
    }

    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_start_iter=500,
        refine_stop_iter=min(n_iters, 15000),
        reset_every=3000,
        refine_every=100,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    bg = torch.ones(3, device=device)
    V = targets.shape[0]
    t0 = time.time()
    for it in range(n_iters):
        vidx = it % V
        vm = viewmats[vidx:vidx + 1]
        Kv = K[vidx:vidx + 1]
        tgt = targets[vidx]

        act_scales = torch.exp(params["scales"])
        act_quats = quat_normalize(params["quats"])
        act_opac = torch.sigmoid(params["opacities"])
        colors = params["sh0"] * SH_C0 + 0.5

        out, _alpha, info = rasterization(
            means=params["means"],
            quats=act_quats,
            scales=act_scales,
            opacities=act_opac,
            colors=colors,
            viewmats=vm,
            Ks=Kv,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB",
            backgrounds=bg.unsqueeze(0),
        )
        pred = out[0]
        l1 = (pred - tgt).abs().mean()
        ss = ssim_loss(pred, tgt)
        loss = l1 + w_ssim * ss

        strategy.step_pre_backward(
            params=params, optimizers=optimizers, state=state, step=it, info=info,
        )

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers.values():
            opt.step()

        strategy.step_post_backward(
            params=params, optimizers=optimizers, state=state, step=it, info=info,
        )

        if log_every and (it % log_every == 0 or it == n_iters - 1):
            with torch.no_grad():
                n_visible = (act_opac > 0.05).sum().item()
            n_total = params["means"].shape[0]
            print(f"    iter {it:5d}/{n_iters}  loss={loss.item():.4f}  "
                  f"l1={l1.item():.4f}  ssim_loss={ss.item():.4f}  "
                  f"visible={n_visible}/{n_total}  elapsed={time.time()-t0:.1f}s")

    return (
        params["means"].detach(),
        params["scales"].detach(),
        params["quats"].detach(),
        params["opacities"].detach(),
        params["sh0"].detach(),
    )


# --- PLY IO -------------------------------------------------------------------

def save_inria_ply(path, means, raw_scales, raw_quats, raw_opacities, sh_dc):
    """Save 3DGS in Inria PLY format (degree-0 SH only)."""
    means_np = means.cpu().numpy().astype(np.float32)
    scales_np = raw_scales.cpu().numpy().astype(np.float32)
    quats_np = raw_quats.cpu().numpy().astype(np.float32)
    opac_np = raw_opacities.cpu().numpy().reshape(-1, 1).astype(np.float32)
    sh_np = sh_dc.cpu().numpy().astype(np.float32)
    N = means_np.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)

    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    elements = np.empty(N, dtype=dtype_full)
    attrs = np.concatenate(
        [means_np, normals, sh_np, opac_np, scales_np, quats_np], axis=1,
    )
    for i in range(N):
        elements[i] = tuple(attrs[i])
    PlyData([PlyElement.describe(elements, 'vertex')], text=False).write(path)


def render_preview(viewmat, K, width, height, means, raw_scales, raw_quats, raw_opacities, sh_dc, device):
    from gsplat import rasterization
    act_scales = torch.exp(raw_scales)
    act_quats = quat_normalize(raw_quats)
    act_opac = torch.sigmoid(raw_opacities)
    colors = sh_dc * SH_C0 + 0.5
    bg = torch.ones(3, device=device)
    with torch.no_grad():
        out, _, _ = rasterization(
            means=means, quats=act_quats, scales=act_scales, opacities=act_opac,
            colors=colors, viewmats=viewmat, Ks=K, width=width, height=height,
            packed=False, render_mode="RGB", backgrounds=bg.unsqueeze(0),
        )
    img = out[0].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


# --- top-level batch loop ----------------------------------------------------

def load_phase_ids(splits_dir, phase):
    p = os.path.join(splits_dir, f"{phase}_split.txt")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--out_dir", default="data/BuildingNet_dataset_v0_1/gaussian_splats")
    ap.add_argument("--phase", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--ids", nargs="*", help="explicit list of ids; overrides --phase/--limit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--n_elev", type=int, default=3)
    ap.add_argument("--n_azim", type=int, default=8)
    ap.add_argument("--dist", type=float, default=2.5)
    ap.add_argument("--fov_deg", type=float, default=30.0)
    ap.add_argument("--n_iters", type=int, default=7000)
    ap.add_argument("--n_gauss_max", type=int, default=50000)
    ap.add_argument("--init_color", type=float, default=0.78)
    ap.add_argument("--save_preview", action="store_true",
                    help="save a single elev=20,azim=30 preview PNG next to each PLY")
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument(
        "--strategy", choices=["v1", "v2"], default="v1",
        help=(
            "v1 = no densification (current baseline). "
            "v2 = gsplat DefaultStrategy densification + kNN init + real SSIM. "
            "Sharper edges and more Gaussians at the cost of ~2-3x longer per asset."
        ),
    )
    ap.add_argument(
        "--knn_scale_k", type=int, default=0,
        help="If > 0, init each Gaussian's log-scale = log(mean-distance to k nearest neighbors). v2 default = 3.",
    )
    ap.add_argument(
        "--w_ssim", type=float, default=0.2,
        help="Weight on the SSIM loss component (v2 only).",
    )
    ap.add_argument(
        "--scene_scale", type=float, default=1.0,
        help="Scene scale used by DefaultStrategy for normalized scale thresholds (Frame N = 1.0).",
    )
    ap.add_argument(
        "--chunk_idx", type=int, default=0,
        help="0-indexed worker id when running multiple bakers in parallel.",
    )
    ap.add_argument(
        "--num_chunks", type=int, default=1,
        help="Total worker count; each worker takes ids[chunk_idx::num_chunks].",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    # v2 defaults: more iters + kNN init
    if args.strategy == "v2":
        if args.n_iters == 7000:
            args.n_iters = 15000
        if args.knn_scale_k == 0:
            args.knn_scale_k = 3

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device} image_size={args.image_size} n_views={args.n_elev*args.n_azim} n_iters={args.n_iters}")

    obj_dir = os.path.join(args.data_root, "OBJ_MODELS")
    splits_dir = os.path.join(args.data_root, "splits")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.ids:
        ids = list(args.ids)
        print(f"[*] explicit ids: {len(ids)}")
    else:
        phases = ["train", "val", "test"] if args.phase == "all" else [args.phase]
        ids = []
        for ph in phases:
            ids.extend(load_phase_ids(splits_dir, ph))
        print(f"[*] {len(ids)} ids across {phases}")
        if args.limit:
            ids = ids[: args.limit]
            print(f"[*] limited to first {len(ids)}")
    if args.num_chunks > 1:
        ids = ids[args.chunk_idx::args.num_chunks]
        print(f"[*] chunk {args.chunk_idx}/{args.num_chunks}: {len(ids)} ids assigned to this worker")

    elevs, azims = orbit_camera_grid(args.n_elev, args.n_azim)
    R_p3d, T_p3d, viewmats, K = build_views(
        elevs, azims, args.dist, device, args.fov_deg, args.image_size,
    )

    n_ok = n_skip = n_fail = 0
    t_run = time.time()
    for i, mid in enumerate(ids):
        obj_p = os.path.join(obj_dir, f"{mid}.obj")
        if not os.path.exists(obj_p):
            print(f"  [skip-missing] {mid}")
            n_skip += 1
            continue
        out_ply = os.path.join(args.out_dir, f"{mid}.ply")
        if os.path.exists(out_ply) and not args.overwrite:
            print(f"  [skip-exists] {mid}")
            n_skip += 1
            continue

        mesh = load_obj_as_trimesh(obj_p)
        if mesh is None or len(mesh.faces) < 4:
            print(f"  [skip-empty] {mid}")
            n_fail += 1
            continue

        v_norm = normalize_to_unit_extent(np.asarray(mesh.vertices))
        f_idx = np.asarray(mesh.faces, dtype=np.int64)

        print(f"\n[{i+1}/{len(ids)}] {mid}  V={len(v_norm)} F={len(f_idx)}")
        t0 = time.time()
        targets = render_targets_pytorch3d(
            v_norm, f_idx, R_p3d, T_p3d, args.fov_deg, args.image_size, device,
        )  # (V, H, W, 3)

        means, raw_scales, raw_quats, raw_opac, sh_dc = init_gaussians_from_mesh(
            v_norm, args.n_gauss_max, args.init_color, device,
            knn_scale_k=args.knn_scale_k,
        )
        print(f"  init N={means.shape[0]} gaussians (knn_k={args.knn_scale_k}); "
              f"rendered {targets.shape[0]} target views in {time.time()-t0:.1f}s")

        trainer = (
            train_3dgs_for_asset_v2 if args.strategy == "v2"
            else train_3dgs_for_asset
        )
        try:
            train_kwargs = dict(
                targets=targets,
                viewmats=viewmats,
                K=K,
                width=args.image_size, height=args.image_size,
                init_means=means, init_scales=raw_scales, init_quats=raw_quats,
                init_opac=raw_opac, init_sh_dc=sh_dc,
                n_iters=args.n_iters,
                log_every=args.log_every,
                device=device,
            )
            if args.strategy == "v2":
                train_kwargs["w_ssim"] = args.w_ssim
                train_kwargs["scene_scale"] = args.scene_scale
            means, raw_scales, raw_quats, raw_opac, sh_dc = trainer(**train_kwargs)
        except Exception as e:
            print(f"  [train-fail] {mid}: {e}")
            n_fail += 1
            continue

        save_inria_ply(out_ply, means, raw_scales, raw_quats, raw_opac, sh_dc)

        if args.save_preview:
            preview_path = os.path.splitext(out_ply)[0] + "_preview.png"
            preview_img = render_preview(
                viewmats[0:1], K[0:1], args.image_size, args.image_size,
                means, raw_scales, raw_quats, raw_opac, sh_dc, device,
            )
            Image.fromarray(preview_img, "RGB").save(preview_path)

        elapsed = time.time() - t0
        n_ok += 1
        print(f"  [ok] saved {out_ply} ({elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print(f"  baked    : {n_ok}")
    print(f"  skipped  : {n_skip}")
    print(f"  failed   : {n_fail}")
    print(f"  total    : {time.time()-t_run:.1f}s")
    print(f"  output   : {args.out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
