"""AI refine / stylize for the sculpt-and-refine UX (Stage 4, `/refine_with_edit`).

A user blocks out massing with crude primitive edits (scene/sdf_edit). `Refiner` projects
that edit onto a CLEAN recipe building in a target style, keeping the massing (footprint +
height extracted from the edit) and applying coherent style detailing. Optionally
RE-STYLES (target_style != the sculpted style).

Two modes (both reuse existing infra — no fragile new training):
  - "fast"   : the trained B+.6 head IS the stylize model. Extract the edit's footprint +
               height, generate clean recipe params for target_style. Amortized, instant.
  - "quality": optimize target_style recipe params to match the full edited SDF (Adam,
               surface-band loss — B+.7's fitter applied to the edit). Slower, tighter match.

Returns refined recipe_params + mesh + a 3D-IoU of how well the refine preserved the edit's
massing. The amortized fast path is the natural place a *learned* refine lives; the quality
path is the optimization reference.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.networks.diff_recipe import build_diff_recipe
from models.networks import recipe_param_space as ps
from models.networks.displacement_field import fit_displacement, normalizer
from scene.sdf_edit import EditableBuilding, EditOp, recipe_base_sdf
from scene.sdf_primitives import sample_grid, grid_to_mesh


def _grid(bbox, res, device):
    x0, y0, z0, x1, y1, z1 = bbox
    xs = torch.linspace(x0, x1, res, device=device)
    ys = torch.linspace(y0, y1, res, device=device)
    zs = torch.linspace(z0, z1, res, device=device)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")        # (D, H, W)
    return torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)


def _edit_locality_mask(edit_dicts, pts, inner, band):
    """Localized-snap weight: 1 inside any edit primitive (dilated by `inner`), fading to 0
    across `band`. Units of inner/band = units of the ops/pts (cube or meters). Lets the
    generative snap remold ONLY the placed mass + a seam — the untouched massing stays
    bit-exact crisp (user complaint 2026-06-12: global SDEdit remolded the whole building)."""
    from scene.sdf_edit import EditOp, _primitive
    d = None
    with torch.no_grad():
        for ed in edit_dicts:
            v = _primitive(EditOp.from_dict(ed))(pts)
            d = v if d is None else torch.minimum(d, v)
    return torch.clamp(1.0 - (d - inner) / max(band, 1e-6), 0.0, 1.0)


def volume_to_sdf(grid, device):
    """Wrap a (D=z, H=y, W=x) SDF volume defined on the [-1,1]^3 cube as a callable SDF.

    Trilinear-interpolated via grid_sample so a *generated/snapped* 64^3 volume can be the
    sculptable base (same role recipe_base_sdf plays for procedural buildings). Query points
    are in cube coords (x, y, z) in [-1, 1]; grid_sample's last axis maps (x,y,z) -> (W,H,D),
    which matches our (D,H,W) layout."""
    vol = torch.as_tensor(np.asarray(grid, np.float32), device=device).view(1, 1, *grid.shape)

    def f(pts: torch.Tensor) -> torch.Tensor:
        g = pts.view(1, 1, 1, -1, 3)                            # (N,1,1,Q,3); coords (x,y,z)
        s = F.grid_sample(vol, g, mode="bilinear", align_corners=True, padding_mode="border")
        return s.view(-1)
    return f


def _bbox(footprint, height, edits, pad=0.18):
    poly = np.asarray(footprint, float)
    x0, z0 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x1, z1 = float(poly[:, 0].max()), float(poly[:, 1].max())
    ymax = float(height) * 1.3
    for op in edits:
        cx, cy, cz = op["center"]; r = max(op.get("size") or (1.0,))
        x0, x1 = min(x0, cx - r), max(x1, cx + r)
        z0, z1 = min(z0, cz - r), max(z1, cz + r)
        ymax = max(ymax, cy + r)
    px, pz = (x1 - x0) * pad + 1.0, (z1 - z0) * pad + 1.0
    return (x0 - px, 0.0, z0 - pz, x1 + px, ymax * 1.1, z1 + pz)


def _mask_to_polygon(mask, x0, x1, z0, z1, n=16):
    import skimage.measure as skm
    cs = skm.find_contours(mask.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    if len(c) < 3:
        return None
    if len(c) > n:
        c = c[np.linspace(0, len(c) - 1, n, dtype=int)]
    D, W = mask.shape
    xs = c[:, 1] / (W - 1) * (x1 - x0) + x0
    zs = c[:, 0] / (D - 1) * (z1 - z0) + z0
    poly = np.stack([xs, zs], axis=-1)
    area = 0.5 * np.sum((poly[1:, 0] - poly[:-1, 0]) * (poly[1:, 1] + poly[:-1, 1]))
    if area > 0:
        poly = poly[::-1]
    return poly.astype(np.float32)


class Refiner:
    def __init__(self, engine, res: int = 64):
        self.engine = engine
        self.device = engine.device
        self.res = res
        self._sd_lock = threading.Lock()

    # -- read the edited shape --------------------------------------------
    def _edited(self, base_state, edits):
        base = recipe_base_sdf(base_state["style"], base_state["recipe_params"],
                               base_state["footprint"], base_state["height"], device=self.device)
        bldg = EditableBuilding(base, [EditOp.from_dict(d) for d in edits])
        bbox = _bbox(base_state["footprint"], base_state["height"], edits)
        grid = _grid(bbox, self.res, self.device)
        with torch.no_grad():
            sdf = bldg.composed()(grid).reshape(self.res, self.res, self.res)  # (D,H,W)
        return sdf, bbox, grid

    def _footprint_height(self, sdf, bbox):
        occ = (sdf <= 0)
        x0, y0, z0, x1, y1, z1 = bbox
        fp = occ.any(dim=1).cpu().numpy()                       # (D, W) over (z, x)
        poly = _mask_to_polygon(fp, x0, x1, z0, z1)
        ys = occ.any(dim=0).any(dim=1).cpu().numpy()            # (H,) occupied y-levels
        top = np.where(ys)[0]
        height = float((top.max() / (self.res - 1)) * (y1 - y0) + y0) if len(top) else float(y1)
        return poly, max(height, 1.0)

    # -- modes ------------------------------------------------------------
    def refine_fast(self, base_state, edits, target_style, building_class, seed):
        sdf, bbox, _ = self._edited(base_state, edits)
        poly, height = self._footprint_height(sdf, bbox)
        params = self.engine.sample_params(poly, height, building_class, target_style,
                                           seed=seed, guidance=2.0)
        return target_style, params, poly, height, sdf, bbox

    def refine_quality(self, base_state, edits, target_style, steps=150, lr=0.05,
                       fit_res=44):
        # NOTE: empirically ~= refine_fast (the match is dominated by the shared
        # footprint+height extraction, not param-fitting), but much slower. Kept as the
        # optimization reference; the endpoint defaults to "fast". Fit on a COARSE grid.
        sdf64, bbox, _ = self._edited(base_state, edits)
        poly, height = self._footprint_height(sdf64, bbox)
        base = recipe_base_sdf(base_state["style"], base_state["recipe_params"],
                               base_state["footprint"], base_state["height"], device=self.device)
        edited = EditableBuilding(base, [EditOp.from_dict(d) for d in edits]).composed()
        grid = _grid(bbox, fit_res, self.device)
        with torch.no_grad():
            target = edited(grid)
        module, default_fn, _ = build_diff_recipe(target_style)
        module = module.to(self.device)
        params = default_fn(self.device).clone().detach().requires_grad_(True)
        poly_t = torch.tensor(poly, device=self.device)
        h_t = torch.tensor(height, device=self.device)
        band = (target.abs() < 0.12).float()
        bn = band.sum().clamp_min(1.0)
        opt = torch.optim.Adam([params], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = module(params, poly_t, h_t, grid)
            loss = ((pred - target).abs() * band).sum() / bn \
                + 0.1 * (pred.clamp(min=-0.12) - target.clamp(min=-0.12)).abs().mean()
            loss.backward(); opt.step()
        return target_style, ps.unpad_params(ps.pad_params(params.detach().cpu().numpy(), target_style),
                                              target_style), poly, height, sdf64, bbox

    def refine_displacement(self, base_state, edits, target_style=None, steps=700,
                            n_pts=70000):
        """Detail-PRESERVING refine: final(x) = recipe_base(x) + displacement(x).

        base = clean recipe building (the original recipe for cleanup, or a fast-refined
        target-style recipe for re-style); displacement = a per-building MLP fit so
        base + d ~= the edited SDF. `out_scale` is set to the residual range so the field
        can reproduce big user-added masses (towers), not just fine surface detail. Keeps
        the sculpted detail, yields a clean closed implicit surface. Returns a baked mesh.
        """
        sdf64, bbox0, _ = self._edited(base_state, edits)
        poly, height = self._footprint_height(sdf64, bbox0)
        # Pad the bbox so the surface doesn't touch the grid boundary (-> watertight).
        x0, y0, z0, x1, y1, z1 = bbox0
        pad = 0.06 * max(x1 - x0, z1 - z0, y1 - y0)
        bbox = (x0 - pad, y0, z0 - pad, x1 + pad, y1 + pad, z1 + pad)
        ts = target_style or base_state["style"]
        if ts == base_state["style"]:
            b_style, b_params, b_poly, b_h = (base_state["style"], base_state["recipe_params"],
                                              base_state["footprint"], base_state["height"])
        else:
            b_style, b_params, b_poly, b_h, _, _ = self.refine_fast(
                base_state, edits, ts, "RESIDENTIAL", seed=None)
        base_sdf = recipe_base_sdf(b_style, b_params, b_poly, b_h, device=self.device)
        edited_sdf = EditableBuilding(
            recipe_base_sdf(base_state["style"], base_state["recipe_params"],
                            base_state["footprint"], base_state["height"], device=self.device),
            [EditOp.from_dict(d) for d in edits]).composed()

        ex0, ey0, ez0, ex1, ey1, ez1 = bbox
        pts = torch.rand(n_pts, 3, device=self.device)
        pts[:, 0] = pts[:, 0] * (ex1 - ex0) + ex0
        pts[:, 1] = pts[:, 1] * (ey1 - ey0) + ey0
        pts[:, 2] = pts[:, 2] * (ez1 - ez0) + ez0
        with torch.no_grad():
            bv, tv = base_sdf(pts), edited_sdf(pts)
        # out_scale must cover the residual so big added masses can be represented.
        out_scale = float(min(max((tv - bv).abs().max().item() * 1.05, 2.0), 14.0))
        norm = normalizer(bbox)
        field = fit_displacement(bv, tv, norm(pts), steps=steps, device=self.device,
                                 n_freq=8, hidden=192, band=0.2, reg=0.006, out_scale=out_scale)

        def final_sdf(p):
            return base_sdf(p) + field(norm(p))

        grid = sample_grid(final_sdf, self.res, bbox, device=self.device)
        mesh = grid_to_mesh(grid, bbox, iso=0.0)
        if mesh is not None and not mesh.is_watertight:
            try:
                mesh.fill_holes()
            except Exception:
                pass
        occ_final = (grid <= 0).cpu().numpy()
        # recompute edited occ on the padded grid for a fair IoU
        occ_edit = (sample_grid(edited_sdf, self.res, bbox, device=self.device) <= 0).cpu().numpy()
        u = (occ_final | occ_edit).sum()
        iou = float((occ_final & occ_edit).sum() / u) if u else 0.0
        return {"style": b_style, "recipe_params": [float(x) for x in b_params],
                "footprint": np.asarray(b_poly).tolist(), "height": float(b_h),
                "mesh": mesh, "iou_to_edit": iou, "field": field}

    # -- Paint-to-relief: painted patch -> 2D art -> real sculpted SDF detail --
    def refine_paint_relief(self, grid, cam, paint_img, prompt, style_ref=None, seed=7,
                            steps_diff=28, strength=0.6, sketch_thickness=12, sketch_scale=0.85,
                            relief_depth=0.12, band=0.07, fit_steps=400, n_pts=20000,
                            view_res=512, aspect=1.0, return_mesh=True):
        """Paint a rough/vague shape on the CURRENT building -> SDXL art (a scribble
        ControlNet reads the drawn SHAPE as a real structural signal, not just blended
        pixel color — see paint_relief.generate_patch_art) restricted to that patch ->
        fuse the art as a real geometric relief via the same Option D machinery as
        `refine_displacement` (fit_displacement/DisplacementField): final(x) =
        base(x) + w(x) * displacement(x), where `w` is 1 on the painted patch fading to 0
        over `band` (`paint_relief.paint_locality_mask`, the paint-stroke analogue of
        `_edit_locality_mask`) so untouched geometry stays bit-exact. Within the patch, the
        target REPLACES whatever's already there (e.g. an existing carved window) with a
        flat wall reference (`paint_relief.flat_wall_point`) plus the art's height, rather
        than perturbing the old geometry — otherwise a window painted over just gets
        slightly bumped instead of cleanly replaced by the new art.

        `grid`: cube-frame (D,H,W) SDF numpy array — the SAME detailed volume the sculptor
        viewer is showing (e.g. from `detail_cube_volume(..., res_out=96)`), NOT world
        meters and NOT the VQVAE's truncated Frame-N space (TRUNC doesn't apply here).
        `cam`: {"pos","look","fov"} matching texture_bake.trace_view's camera contract —
        the frontend sends the exact camera it painted against. `paint_img`: the user's
        rough painted shape/colors (PIL RGBA, transparent where unpainted) in that same view.
        `sketch_thickness`/`sketch_scale`: how literally the drawn SHAPE is followed — thin
        +strong reads as "adjust it to the right thing", thick+weaker as "a creative art
        piece" (paint_relief.scribble_from_mask). `strength` separately controls how much
        the painted COLORS survive (~0.6 "blend" keeps them recognizable).
        `relief_depth`: max relief magnitude in cube units (~0.05 ≈ 2 voxels at 96^3).
        Returns (out_grid (D,H,W) numpy SDF, mesh|None, art_rgb) so a caller can either
        drop `out_grid`/`out_grid`-derived mesh into the live preview, or re-bake color.
        """
        import paint_relief as pr
        base_sdf = volume_to_sdf(grid, self.device)
        rgb, surf_pts, mask_bool, basis = pr.generate_patch_art(
            grid, cam, paint_img, prompt, style_ref=style_ref, seed=seed, steps=steps_diff,
            strength=strength, sketch_thickness=sketch_thickness, sketch_scale=sketch_scale,
            res=view_res, aspect=aspect, device=self.device)
        h_field = pr.height_from_art(rgb, mask_bool, relief_depth)

        hit_rows, hit_cols = np.where(mask_bool & basis["hit"])
        h_surf = torch.as_tensor(h_field[hit_rows, hit_cols], dtype=torch.float32,
                                 device=self.device)

        lo = surf_pts.min(0).values.cpu().numpy()
        hi = surf_pts.max(0).values.cpu().numpy()
        pad = np.maximum(0.15 * (hi - lo), 0.05)
        bbox = tuple((lo - pad).tolist() + (hi + pad).tolist())

        pts = torch.rand(n_pts, 3, device=self.device)
        lo_t = torch.tensor(bbox[:3], device=self.device)
        hi_t = torch.tensor(bbox[3:], device=self.device)
        pts = pts * (hi_t - lo_t) + lo_t

        from scipy.spatial import cKDTree
        tree = cKDTree(surf_pts.cpu().numpy())
        _, nn = tree.query(pts.cpu().numpy(), k=1)
        h_at_pts = h_surf[torch.as_tensor(nn, device=self.device)]
        w = pr.paint_locality_mask(surf_pts, pts, band=band)
        with torch.no_grad():
            bv = base_sdf(pts)

        # Replace whatever's ALREADY there (e.g. a carved window/sill the user painted
        # over) with a flat wall reference + the art relief, instead of just perturbing
        # the existing recess by a small `h` — otherwise the old window and the new art
        # blend together and the window visually "wins" (2026-07-07 user report).
        flat_p0 = pr.flat_wall_point(mask_bool, basis)
        if flat_p0 is not None:
            p0_t = torch.as_tensor(flat_p0, dtype=torch.float32, device=self.device)
            eps = 2.0 / grid.shape[0]
            ex = torch.tensor([eps, 0, 0], device=self.device)
            ey = torch.tensor([0, eps, 0], device=self.device)
            ez = torch.tensor([0, 0, eps], device=self.device)
            with torch.no_grad():
                grad = torch.stack([
                    base_sdf(p0_t[None] + ex) - base_sdf(p0_t[None] - ex),
                    base_sdf(p0_t[None] + ey) - base_sdf(p0_t[None] - ey),
                    base_sdf(p0_t[None] + ez) - base_sdf(p0_t[None] - ez),
                ], -1)
            n0 = F.normalize(grad, dim=-1)[0]                   # outward wall normal
            flat_val = (pts - p0_t[None]) @ n0                  # signed dist to the wall plane
            tv = bv + w * ((flat_val - h_at_pts) - bv)          # blend: bv outside, flat+art inside
        else:                                                   # no usable surrounding context
            tv = bv - w * h_at_pts                              # fall back to the old perturb-only behavior

        norm = normalizer(bbox)
        out_scale = float(min(max((tv - bv).abs().max().item() * 1.05, 0.02),
                              max(relief_depth * 1.5, 0.5)))
        field = fit_displacement(bv, tv, norm(pts), steps=fit_steps, device=self.device,
                                 n_freq=8, hidden=160, band=band, reg=0.01, out_scale=out_scale)

        def final_sdf(p):
            wp = pr.paint_locality_mask(surf_pts, p, band=band)  # 2nd locality gate: keep the
            return base_sdf(p) + wp * field(norm(p))             # effect local past the fit bbox

        full_bbox = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
        out_grid = sample_grid(final_sdf, grid.shape[0], full_bbox, device=self.device)
        mesh = grid_to_mesh(out_grid, full_bbox, iso=0.0) if return_mesh else None
        return out_grid.cpu().numpy().astype(np.float32), mesh, rgb

    # -- SDEdit mode: the learned massing prior (3D BAG) ------------------
    def _mk_stage3a(self, ckpt, use_extra_cond=False, use_adaln=False):
        from types import SimpleNamespace
        from models.stage3a_model import Stage3aModel
        vq = REPO / "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"
        opt = SimpleNamespace(
            isTrain=False, device=self.device,
            df_cfg=str(REPO / "configs/stage3a_sdf_diffusion.yaml"),
            vq_cfg=str(REPO / "configs/vqvae_bnet.yaml"), vq_ckpt=str(vq),
            ckpt=str(ckpt), ddim_steps=50, debug="0",
            use_extra_cond=use_extra_cond,   # hybrid ckpts have era/floors arch (inactive labels)
            use_adaln=use_adaln,             # adaln ckpts have adaln_proj in the UNet
            gpu_ids=[0] if self.device == "cuda" else [], ckpt_dir="/tmp",
            latent_size_HW=(16, 16), latent_size_D=16)
        m = Stage3aModel(); m.initialize(opt); return m

    def _load_sdedit(self, autoguidance=True):
        """Lazy-load the snap prior (+ a weaker ckpt of the SAME run for autoguidance).

        DEPLOYED 2026-07-03: the cross-cultural warm-start finetune (NL+DE+JP massing breadth),
        validated 2026-06-29/30 to fix exactly the failure the demo was hitting — the old
        2026-06-08 hybrid-clean 20k prior snaps a placed mass into a degenerate blob / erases it
        (outputs/sdedit_xcultural/localized_{ab,de,jp}.png); this ckpt keeps it a coherent,
        type-appropriate element. Same use_extra_cond=True (era/floors) architecture, so
        _mk_stage3a below is unchanged. Old 20k ckpt path kept commented for rollback.
        Guide ckpt is an EARLIER (weaker) checkpoint of the SAME finetune run, for autoguidance."""
        import os
        main_dir = REPO / "logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt"
        guide_dir = REPO / "logs_building/continue-stage3a-xcultural-warmstart-ft/ckpt"
        # main_dir = guide_dir = REPO / "logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt"  # old, pre-2026-07-03

        def _ck(d, name):
            # Prefer a node-local copy (SNAP_CKPT_DIR, e.g. /tmp/hybrid_ckpts) — Lustre stalls on
            # sustained 15GB reads; fall back to scratch when the local copy is absent.
            local = Path(os.environ.get("SNAP_CKPT_DIR", "/tmp/hybrid_ckpts")) / name
            return local if local.exists() else d / name

        with self._sd_lock:  # startup warmup thread may race the first request
            if getattr(self, "_sd_main", None) is None:
                self._sd_main = self._mk_stage3a(_ck(main_dir, "stage3a_steps-latest.pth"), use_extra_cond=True)
            if autoguidance and getattr(self, "_sd_guide", None) is None:
                self._sd_guide = self._mk_stage3a(_ck(guide_dir, "stage3a_steps-1000.pth"), use_extra_cond=True)
        return self._sd_main, (self._sd_guide if autoguidance else None)

    @torch.no_grad()
    def _recipe_to_frame_n(self, edited, sample_bbox, R=64, margin=1.05, trunc=0.2):
        """Bridge a world-meters edited SDF -> the prior's Frame-N input contract.

        Reproduces ingest_3dbag.building_to_sdf normalization (center the occupancy bbox,
        scale by max_extent/2*margin) and rebuilds a TRUE Euclidean SDF from the occupancy via
        an exact distance transform (so the truncated band matches the igl SDFs the VQVAE was
        trained on, not the recipe's non-unit-gradient field). Returns (sdf, fp, height_n, c, s).
        """
        from scipy.ndimage import distance_transform_edt
        x0, y0, z0, x1, y1, z1 = sample_bbox
        xs = torch.linspace(x0, x1, R, device=self.device)
        ys = torch.linspace(y0, y1, R, device=self.device)
        zs = torch.linspace(z0, z1, R, device=self.device)
        Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")          # (D=z,H=y,W=x)
        occ = (edited(torch.stack([X, Y, Z], -1).reshape(-1, 3)).reshape(R, R, R) <= 0).cpu().numpy()
        if not occ.any():
            raise ValueError("edited building is empty")
        xc, yc, zc = xs.cpu().numpy(), ys.cpu().numpy(), zs.cpu().numpy()
        wi, hi, di = np.where(occ.any((0, 1)))[0], np.where(occ.any((0, 2)))[0], np.where(occ.any((1, 2)))[0]
        bx, by, bz = (xc[wi.min()], xc[wi.max()]), (yc[hi.min()], yc[hi.max()]), (zc[di.min()], zc[di.max()])
        c = np.array([(bx[0]+bx[1])/2, (by[0]+by[1])/2, (bz[0]+bz[1])/2], np.float32)
        s = float(max(bx[1]-bx[0], by[1]-by[0], bz[1]-bz[0])) / 2 * margin
        # sample the edit on the Frame-N grid mapped back to world
        g1 = torch.linspace(-1, 1, R, device=self.device)
        ZZ, YY, XX = torch.meshgrid(g1, g1, g1, indexing="ij")       # (D=z,H=y,W=x)
        world = torch.stack([XX, YY, ZZ], -1).reshape(-1, 3) * s + torch.tensor(c, device=self.device)
        occN = (edited(world).reshape(R, R, R) <= 0).cpu().numpy()
        if not occN.any():
            raise ValueError("edited building empty after normalization")
        vox = 2.0 / (R - 1)
        sdfN = np.clip((distance_transform_edt(~occN) - distance_transform_edt(occN)) * vox, -trunc, trunc).astype(np.float32)
        ys2 = np.where(occN.any((0, 2)))[0]
        height_n = float((ys2.max() - ys2.min() + 1) * vox)
        sdf_t = torch.from_numpy(sdfN).view(1, 1, R, R, R).to(self.device)
        fp_t = torch.from_numpy(occN.any(1).astype(np.float32)).view(1, 1, R, R).to(self.device)
        return sdf_t, fp_t, height_n, c, s

    def refine_sdedit(self, base_state, edits, strength=0.5, steps=8,
                      autoguidance=True, auto_scale=2.0, margin=1.5, smooth_iters=12,
                      detail=True, building_class="RESIDENTIAL", seed=None):
        """Snap-to-plausible: project the sculpted massing onto the learned 3D BAG manifold
        via SDEdit (partial-noise diffusion). The prior is style-agnostic (style_id=8), so this
        is a *massing* cleanup; `strength` trades faithfulness<->realism. `smooth_iters` Taubin-
        smooths the output to suppress VQVAE surface artifacts (audit gap #6) on flat recipe faces.

        detail=True re-applies the ② composer detail (windows/roof/door/landmarks) onto the
        SNAPPED massing before meshing — without it the user's detailed building came back as
        a bare soft mass and the demo read as "good building -> blob" (fixed 2026-06-12,
        handoff item #30)."""
        edited = EditableBuilding(
            recipe_base_sdf(base_state["style"], base_state["recipe_params"],
                            base_state["footprint"], base_state["height"], device=self.device),
            [EditOp.from_dict(d) for d in edits]).composed()
        bbox = _bbox(base_state["footprint"], base_state["height"], edits)
        sdf_t, fp_t, height_n, c, s = self._recipe_to_frame_n(edited, bbox, margin=margin)
        main_m, guide_m = self._load_sdedit(autoguidance)
        data = {"sdf": sdf_t, "fp": fp_t,
                "class_id": torch.zeros(1, dtype=torch.long, device=self.device),
                "style_id": torch.full((1,), 8, dtype=torch.long, device=self.device),
                "height": torch.tensor([height_n], dtype=torch.float32, device=self.device)}
        out = main_m.sdedit(data, strength=strength, ddim_steps=steps, uc_scale=1.0,
                            guide_model=guide_m, auto_scale=auto_scale)
        occ_o, occ_i = (out[0, 0] <= 0), (sdf_t[0, 0] <= 0)
        u = (occ_o | occ_i).sum().item()
        iou = float((occ_o & occ_i).sum().item() / u) if u else 0.0

        if edits:
            # LOCALIZED snap: keep the crisp edited massing except inside/near the placed
            # mass (mask in WORLD meters; ops are world-frame here)
            R = out.shape[-1]
            g1 = torch.linspace(-1, 1, R, device=self.device)
            Zg, Yg, Xg = torch.meshgrid(g1, g1, g1, indexing="ij")
            world_pts = (torch.stack([Xg, Yg, Zg], -1).reshape(-1, 3) * float(s)
                         + torch.as_tensor(np.asarray(c, np.float32), device=self.device))
            w = _edit_locality_mask(edits, world_pts, inner=0.6, band=2.0).reshape(R, R, R)
            out = (sdf_t[0, 0] * (1 - w) + out[0, 0] * w)[None, None]

        mesh = None
        if detail:
            try:
                from scene.composer_detail import compose_detail, get_composer
                cube_sdf = volume_to_sdf(out[0, 0].detach().cpu().numpy(), self.device)
                c_t = torch.as_tensor(np.asarray(c, np.float32), device=self.device)
                s_f = float(s)

                def world_sdf(p, _f=cube_sdf, _c=c_t, _s=s_f):
                    return _f((p - _c) / _s) * _s

                poly = np.asarray(base_state["footprint"], np.float32)
                h_w = float(base_state["height"])
                sdf_d, _lay, dec = compose_detail(world_sdf, poly, h_w, building_class,
                                                  style=base_state["style"], seed=seed,
                                                  composer=get_composer(self.device))
                pad = 0.12 * max(np.ptp(poly[:, 0]), np.ptp(poly[:, 1])) + 1.0
                head = h_w * (1.9 if dec["n_towers"] else 1.5)
                dbox = (poly[:, 0].min() - pad, 0.0, poly[:, 1].min() - pad,
                        poly[:, 0].max() + pad, head, poly[:, 1].max() + pad)
                from scene.sdf_primitives import sample_grid
                mesh = grid_to_mesh(sample_grid(sdf_d, 96, dbox, device=self.device),
                                    dbox, 0.0)                          # world coords
            except Exception as ex:
                print(f"[refine_sdedit] detail re-apply failed ({ex}); plain massing")
                mesh = None
        if mesh is None:
            mesh = grid_to_mesh(out[0, 0].detach().cpu(), (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), iso=0.0)
            if mesh is not None and len(mesh.vertices):
                if smooth_iters > 0:
                    try:
                        import trimesh
                        trimesh.smoothing.filter_taubin(mesh, iterations=int(smooth_iters))
                    except Exception:
                        pass
                v = np.asarray(mesh.vertices, np.float32) * s + c      # Frame-N -> world (x,y,z)
                v[:, 1] -= v[:, 1].min()                                # sit on the ground
                mesh.vertices = v
        from scene.mesh_cleanup import cleanup_mesh
        mesh = cleanup_mesh(mesh)                                   # weld + drop fragments
        return {"style": base_state["style"],
                "recipe_params": [float(x) for x in base_state["recipe_params"]],
                "footprint": np.asarray(base_state["footprint"]).tolist(),
                "height": float(base_state["height"]), "mesh": mesh, "iou_to_edit": iou}

    # -- real-house massing: retrieve the closest-footprint 3D BAG building ------------
    _BAG_FAST = "/dev/shm/bag3d_fast.h5"
    _BAG_SLOW = str(REPO / "data/bag3d_v1/bag3d.h5")

    def _bag_index(self):
        """Lazy footprint index over the BAG corpus (11776 x 64 x 64 uint8, ~48MB)."""
        if getattr(self, "_bag_fps", None) is None:
            import h5py
            path = self._BAG_FAST if Path(self._BAG_FAST).exists() else self._BAG_SLOW
            with h5py.File(path, "r") as h:
                self._bag_fps = h["footprint"][:].astype(bool)
            self._bag_path = path
        return self._bag_fps, self._bag_path

    def bag_house_volume(self, footprint, height, res=64, margin=1.5):
        """REAL massing: nearest-footprint 3D BAG house, anisotropically fitted to the user's
        footprint bbox + height — real roofs/setbacks by construction; the prior then only has
        to do light cleanup. Returns (grid cube-frame, center, scale, bag_index)."""
        import h5py
        from scipy.ndimage import distance_transform_edt, zoom
        fps, path = self._bag_index()
        # rasterize the user's footprint into the BAG convention (centered, max-extent square)
        poly = np.asarray(footprint, np.float32)
        c2d = (poly.min(0) + poly.max(0)) / 2
        s2d = max(*(poly.max(0) - poly.min(0))) / 2 * 1.05
        from matplotlib.path import Path as MplPath
        g = np.linspace(-1, 1, 64)
        Z, Xg = np.meshgrid(g, g, indexing="ij")
        pts2 = np.stack([Xg.ravel() * s2d + c2d[0], Z.ravel() * s2d + c2d[1]], -1)
        ufp = MplPath(poly).contains_points(pts2).reshape(64, 64)
        inter = (fps & ufp).sum((1, 2)).astype(np.float32)
        union = (fps | ufp).sum((1, 2)).astype(np.float32)
        best = int(np.argmax(inter / np.maximum(union, 1)))
        with h5py.File(path, "r") as h:
            sdf = h["sdf"][best]                                    # Frame-N margin~1.05
        occ = sdf <= 0
        # anisotropic fit: stretch the house's occupied bbox onto the user's footprint aspect + height
        wi, hi, di = [np.where(occ.any(ax)) [0] for ax in ((0, 1), (0, 2), (1, 2))]
        if not len(wi):
            raise ValueError("empty BAG sample")
        w_m, d_m = (poly.max(0) - poly.min(0))                      # user extents (meters)
        ext = np.array([g[wi.max()] - g[wi.min()], g[hi.max()] - g[hi.min()],
                        g[di.max()] - g[di.min()]])                  # house (x, y, z) extents
        s_world = float(max(w_m, d_m, height) / 2 * margin)          # cube scale (meters/unit)
        tgt = np.array([w_m, height, d_m]) / s_world                 # target extents (cube units)
        zf = np.clip(tgt / np.maximum(ext, 1e-3), 0.3, 3.5)
        occ_t = zoom(occ.astype(np.float32), (zf[2], zf[1], zf[0]), order=1) > 0.5  # (D,H,W)=(z,y,x)
        out = np.zeros((res, res, res), bool)
        sl = [slice(0, 0)] * 3
        for a in range(3):
            n = min(occ_t.shape[a], res)
            o_src = (occ_t.shape[a] - n) // 2
            o_dst = (res - n) // 2
            sl[a] = (slice(o_src, o_src + n), slice(o_dst, o_dst + n))
        out[sl[0][1], sl[1][1], sl[2][1]] = occ_t[sl[0][0], sl[1][0], sl[2][0]]
        # centered in all axes == the occ-bbox-centered cube convention (same as recipe massing)
        vox = 2.0 / (res - 1)
        grid = np.clip((distance_transform_edt(~out) - distance_transform_edt(out)) * vox,
                       -0.2, 0.2).astype(np.float32)
        c3 = np.array([c2d[0], height / 2 * 1.0, c2d[1]], np.float32)
        return grid, c3, s_world, best

    # -- volume in/out: feed the raymarched sculptor (web/sculpt.html) -----
    def building_volume(self, footprint, style, recipe_params, height, res=64, margin=1.5):
        """Generate the BASE building as a 64^3 SDF volume in a normalized [-1,1]^3 cube.

        Returns (grid (D=z,H=y,W=x) float32, center c (world x,y,z), scale s, height_n). margin=1.5
        leaves air around the building so sculpt edits (towers/wings) have room before the cube
        boundary, and reduces VQVAE flat-face artifacts on the snap (gap #6 mitigation)."""
        base = recipe_base_sdf(style, recipe_params, footprint, height, device=self.device)
        bbox = _bbox(footprint, height, [])
        sdf_t, _fp, height_n, c, s = self._recipe_to_frame_n(base, bbox, R=res, margin=margin)
        grid = sdf_t[0, 0].detach().cpu().numpy().astype(np.float32)        # (D,H,W), cube frame
        return grid, c, s, height_n

    @torch.no_grad()
    def _cube_to_frame_n(self, composed, R=64, trunc=0.2, center=None, scale=1.0):
        """Sample a composed cube-frame SDF onto the prior's Frame-N input contract (true truncated
        SDF via EDT + footprint + normalized height). With (center, scale) the Frame-N grid maps to
        cube coords q = n*scale + center — used to RE-normalize an edited shape so the building
        fills the cube like the prior's training data (margin~1.05); sign-only sampling makes the
        base volume's distance units irrelevant."""
        from scipy.ndimage import distance_transform_edt
        g1 = torch.linspace(-1, 1, R, device=self.device)
        Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")                # (D=z,H=y,W=x)
        q = torch.stack([X, Y, Z], -1).reshape(-1, 3)
        if center is not None:
            q = q * float(scale) + center
        occ = (composed(q).reshape(R, R, R) <= 0).cpu().numpy()
        if not occ.any():
            raise ValueError("sculpt is empty")
        vox = 2.0 / (R - 1)
        sdfN = np.clip((distance_transform_edt(~occ) - distance_transform_edt(occ)) * vox,
                       -trunc, trunc).astype(np.float32)
        ys = np.where(occ.any((0, 2)))[0]
        height_n = float((ys.max() - ys.min() + 1) * vox)
        sdf_t = torch.from_numpy(sdfN).view(1, 1, R, R, R).to(self.device)
        fp_t = torch.from_numpy(occ.any(1).astype(np.float32)).view(1, 1, R, R).to(self.device)
        return sdf_t, fp_t, height_n

    def snap_volume(self, base_grid, edits, strength=0.5, steps=8, autoguidance=True,
                    auto_scale=2.0, margin=1.05, smooth_sigma=0.8,
                    local=True, local_inner=0.06, local_band=0.12):
        """Volume-native SDEdit snap (the generative module of the sculpt loop): a 64^3 base SDF
        volume (cube frame) + primitive EditOps (cube coords) -> compose -> SDEdit massing prior ->
        a NEW 64^3 volume in the SAME cube frame (so the viewer reloads it and keeps sculpting).

        The edited shape is RE-normalized to the prior's training frame (occupancy bbox fills the
        cube at margin~1.05) before SDEdit — feeding it at the viewer's sculpt-headroom margin
        (~1.5) is off-distribution and yields blobbier output — then resampled back into the
        original viewer cube. Returns (snapped_grid (D,H,W) float32, iou_to_edit).

        local + NO edits = no-op (base returned bit-exact): with nothing placed there is no
        region for the generative snap to act on, and running it anyway remolded the whole
        building under the user's detail ops (complaint 2026-06-12). Pass local=False for an
        explicit whole-building re-mold."""
        if local and not edits:
            return np.asarray(base_grid, np.float32).copy(), 1.0
        base = volume_to_sdf(base_grid, self.device)
        bldg = EditableBuilding(base, [EditOp.from_dict(d) for d in edits])
        composed = bldg.composed()
        R = int(base_grid.shape[0])
        g1 = torch.linspace(-1, 1, R, device=self.device)
        Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")                 # (D=z,H=y,W=x)
        cube_pts = torch.stack([X, Y, Z], -1)                               # (R,R,R,3) xyz
        with torch.no_grad():
            occ = (composed(cube_pts.reshape(-1, 3)).reshape(R, R, R) <= 0).cpu().numpy()
        if not occ.any():
            raise ValueError("sculpt is empty")
        gc = g1.cpu().numpy()
        wi = np.where(occ.any((0, 1)))[0]; hi = np.where(occ.any((0, 2)))[0]; di = np.where(occ.any((1, 2)))[0]
        bx = (gc[wi.min()], gc[wi.max()]); by = (gc[hi.min()], gc[hi.max()]); bz = (gc[di.min()], gc[di.max()])
        c2 = torch.tensor([(bx[0] + bx[1]) / 2, (by[0] + by[1]) / 2, (bz[0] + bz[1]) / 2],
                          dtype=torch.float32, device=self.device)
        s2 = max(bx[1] - bx[0], by[1] - by[0], bz[1] - bz[0]) / 2 * margin
        sdf_t, fp_t, height_n = self._cube_to_frame_n(composed, R=R, center=c2, scale=s2)
        main_m, guide_m = self._load_sdedit(autoguidance)
        data = {"sdf": sdf_t, "fp": fp_t,
                "class_id": torch.zeros(1, dtype=torch.long, device=self.device),
                "style_id": torch.full((1,), 8, dtype=torch.long, device=self.device),
                "height": torch.tensor([height_n], dtype=torch.float32, device=self.device)}
        out = main_m.sdedit(data, strength=strength, ddim_steps=steps, uc_scale=1.0,
                            guide_model=guide_m, auto_scale=auto_scale)
        occ_o, occ_i = (out[0, 0] <= 0), (sdf_t[0, 0] <= 0)
        u = (occ_o | occ_i).sum().item()
        iou = float((occ_o & occ_i).sum().item() / u) if u else 0.0
        # resample Frame-N output back into the stable viewer cube: q_cube -> (q - c2)/s2
        n = (cube_pts - c2) / s2                                            # (R,R,R,3) xyz for grid_sample
        samp = F.grid_sample(out, n.view(1, R, R, R, 3), mode="bilinear",
                             align_corners=True, padding_mode="border")
        snapped = (samp[0, 0] * s2).detach().cpu().numpy().astype(np.float32)
        if smooth_sigma > 0:
            # the prior's output has high-frequency surface wobble (verified knob-independent —
            # AG on/off × margin all wavy); a light SDF low-pass keeps massing, kills the wobble
            from scipy.ndimage import gaussian_filter
            snapped = gaussian_filter(snapped, sigma=float(smooth_sigma))
        if local and edits:
            # LOCALIZED snap: generative result only inside/near the placed mass; the
            # untouched massing stays the crisp composed SDF (not the prior's remold)
            with torch.no_grad():
                flat = cube_pts.reshape(-1, 3)
                edited_vals = composed(flat).reshape(R, R, R)
                w = _edit_locality_mask(edits, flat, local_inner, local_band).reshape(R, R, R)
                comp = edited_vals * (1 - w) + torch.as_tensor(snapped, device=self.device) * w
            snapped = comp.detach().cpu().numpy().astype(np.float32)
        from scene.mesh_cleanup import cleanup_sdf_grid
        snapped = cleanup_sdf_grid(snapped)            # drop floating debris before it meshes
        return snapped, iou

    @torch.no_grad()
    def volume_to_world_mesh(self, grid, center, scale, building_class="RESIDENTIAL",
                             style="modern", seed=None, detail=False, res=96, detail_edits=None):
        """Bake a cube-frame snapped volume to a WORLD-meters mesh sitting on y=0. If detail, run
        the ② composer/detail (class-appropriate roof/windows/door/landmarks) on the snapped massing
        (README's composer->SDEdit wiring, applied at bake time); robust fallback to plain massing.
        `detail_edits`: the caller's raw edit ops (already CSG-unioned into `grid`) — only used
        to check for a user-placed det:'door'/det:'roof' so compose_detail doesn't double it."""
        from scene.sdf_primitives import sample_grid, grid_to_mesh
        c = np.asarray(center, np.float32)
        s = float(scale)
        dev = self.device
        cube = volume_to_sdf(grid, dev)
        c_t = torch.tensor(c, device=dev)
        occ = (np.asarray(grid) <= 0)
        if not occ.any():
            return None
        g1 = np.linspace(-1, 1, grid.shape[0]).astype(np.float32)
        ys = np.where(occ.any(axis=(0, 2)))[0]                       # occupied H(y) levels
        ymin_w = float(g1[ys.min()] * s + c[1])
        height_w = max(float((g1[ys.max()] - g1[ys.min()]) * s), 1.0)

        def placed(pw):                                              # building base -> world y=0
            q = pw.clone()
            q[..., 1] = q[..., 1] + ymin_w
            return cube((q - c_t) / s) * s

        sdf = placed
        if detail:
            try:
                from scene.composer_detail import compose_detail, get_composer, auto_roof_flag
                fp = occ.any(axis=1)                                 # (D,W) top-down silhouette
                poly = _mask_to_polygon(fp, c[0] - s, c[0] + s, c[2] - s, c[2] + s)
                if poly is not None:
                    sdf, _layout, _dec = compose_detail(placed, poly, height_w, building_class,
                                                        style=style, seed=seed,
                                                        roof=auto_roof_flag(detail_edits),
                                                        composer=get_composer(dev))
            except Exception as exc:
                print(f"[snap bake] composer detail unavailable ({exc}); plain massing")
                sdf = placed
        bbox = (c[0] - s, 0.0, c[2] - s, c[0] + s, height_w * 1.7, c[2] + s)
        from scene.mesh_cleanup import cleanup_mesh
        return cleanup_mesh(grid_to_mesh(sample_grid(sdf, res, bbox, device=dev), bbox, 0.0))

    @torch.no_grad()
    def detail_cube_volume(self, grid, center, scale, building_class="RESIDENTIAL",
                           style="modern", seed=None, res_out=96, detail_edits=None):
        """LIVE DETAIL PREVIEW: compose the ② composer detail (windows/bands/plinth/roof/
        landmarks — the bake-quality treatment) onto a cube-frame massing volume and return
        it as a cube-frame SDF volume the viewer can raymarch directly. Same construction
        as the bake (volume_to_world_mesh detail=True), sampled on the viewer cube instead
        of a world bbox. `detail_edits`: the user's raw edit ops (CUBE-frame [-1,1], the SAME
        ones already CSG-unioned into `grid`) — 'add' ops get routed to compose_detail so the
        ADDED primitive gets its own tower/balcony/etc detail instead of staying a bare box."""
        c = np.asarray(center, np.float32)
        s = float(scale)
        dev = self.device
        cube = volume_to_sdf(grid, dev)
        c_t = torch.tensor(c, device=dev)
        occ = (np.asarray(grid) <= 0)
        if not occ.any():
            return np.asarray(grid, np.float32)
        g1 = np.linspace(-1, 1, grid.shape[0]).astype(np.float32)
        ys = np.where(occ.any(axis=(0, 2)))[0]
        ymin_w = float(g1[ys.min()] * s + c[1])
        height_w = max(float((g1[ys.max()] - g1[ys.min()]) * s), 1.0)

        def placed(pw):
            q = pw.clone()
            q[..., 1] = q[..., 1] + ymin_w
            return cube((q - c_t) / s) * s

        sdf = placed
        try:
            from scene.composer_detail import compose_detail, get_composer
            fp = occ.any(axis=1)
            poly = _mask_to_polygon(fp, c[0] - s, c[0] + s, c[2] - s, c[2] + s)
            if poly is not None:
                # cube-frame [-1,1] (rel. to center/scale) -> the world-meter frame `placed`
                # expects (Y measured from the building's own base): world = cube*s + c,
                # then Y needs the same ymin_w correction `placed` applies internally.
                add_ops = []
                for op in (detail_edits or []):
                    if str(op.get("mode", "add")) != "add":
                        continue
                    cw = np.asarray(op["center"], np.float32) * s + c
                    cw[1] -= ymin_w
                    sw = np.asarray(op["size"][:3], np.float32) * s
                    add_ops.append({**op, "center": cw.tolist(), "size": sw.tolist()})
                from scene.composer_detail import auto_roof_flag
                sdf, _lay, _dec = compose_detail(placed, poly, height_w, building_class,
                                                 style=style, seed=seed,
                                                 roof=auto_roof_flag(detail_edits),
                                                 composer=get_composer(dev), add_ops=add_ops)
        except Exception as exc:
            print(f"[detail preview] composer unavailable ({exc}); plain massing")
        gq = torch.linspace(-1, 1, res_out, device=dev)
        Z, Y, X = torch.meshgrid(gq, gq, gq, indexing="ij")
        pw = torch.stack([X * s + c[0], Y * s + c[1] - ymin_w, Z * s + c[2]], -1).reshape(-1, 3)
        out = torch.empty(pw.shape[0], device=dev)
        chunk = 1 << 19
        for i in range(0, pw.shape[0], chunk):
            out[i:i + chunk] = sdf(pw[i:i + chunk])
        return (out.reshape(res_out, res_out, res_out) / s).float().cpu().numpy()

    # -- public -----------------------------------------------------------
    def refine(self, base_state, edits, target_style=None, mode="fast",
               building_class="RESIDENTIAL", seed=None, steps=150,
               strength=0.5, sdedit_steps=8, autoguidance=True, auto_scale=2.0,
               detail=True):
        ts = target_style or base_state["style"]
        if mode == "sdedit":
            # Learned massing prior (3D BAG) — snap the sculpt onto the building manifold,
            # then re-apply the ② composer detail so the result stays architectural.
            return self.refine_sdedit(base_state, edits, strength=strength, steps=sdedit_steps,
                                      autoguidance=autoguidance, auto_scale=auto_scale,
                                      detail=detail, building_class=building_class, seed=seed)
        if mode == "displacement":
            # Detail-preserving; builds its own mesh from base + displacement field.
            return self.refine_displacement(base_state, edits, target_style=ts, steps=steps * 3)
        if mode == "fast":
            style, params, poly, height, sdf, bbox = self.refine_fast(
                base_state, edits, ts, building_class, seed)
        elif mode == "quality":
            style, params, poly, height, sdf, bbox = self.refine_quality(
                base_state, edits, ts, steps)
        else:
            raise ValueError("mode must be fast|quality|displacement")
        mesh, _ = self.engine.params_to_mesh(params, style, poly, height)
        # how well did the refine keep the edit's massing? (3D occupancy IoU)
        ref_occ = self._refined_occ(style, params, poly, height, bbox)
        edit_occ = (sdf <= 0).cpu().numpy()
        u = (ref_occ | edit_occ).sum()
        iou = float((ref_occ & edit_occ).sum() / u) if u else 0.0
        return {"style": style, "recipe_params": [float(x) for x in params],
                "footprint": np.asarray(poly).tolist(), "height": float(height),
                "mesh": mesh, "iou_to_edit": iou}

    @torch.no_grad()
    def _refined_occ(self, style, params, poly, height, bbox):
        module = build_diff_recipe(style)[0].to(self.device)
        grid = _grid(bbox, self.res, self.device)
        p = torch.tensor(np.asarray(params, np.float32), device=self.device)
        pt = torch.tensor(np.asarray(poly, np.float32), device=self.device)
        h = torch.tensor(float(height), device=self.device)
        sdf = module(p, pt, h, grid).reshape(self.res, self.res, self.res)
        return (sdf <= 0).cpu().numpy()
