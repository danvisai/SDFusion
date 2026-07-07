"""Sketch-to-relief: a user sketches a rough shape on a building, the stroke is RECTIFIED
onto its wall plane, a diffusion model generates bas-relief art for that flat frontal
patch, and the art is fused into the building's SDF as a REAL geometric relief (marching
cubes shows an actual bump/carving, not just a color texture).

v2 (wall-space). v1 inpainted directly into the perspective 3D view, conditioned by that
view's depth/canny maps — which contain the composer's window grid — so SDXL's strongest
prior for "building photo + architecture prompt" was MORE FACADE: verification showed it
redrawing whole miniature window-grid facades inside the padded inpaint box, and in the
worst case the relief carpeted the entire building (outputs/sketch_relief_verify/
free_silhouette_fix/star_calib_*.png, final_4_result_full_BROKEN_widespread.png).
Generating on a flat, frontal, orthographic canvas of just the wall patch under the stroke
removes that attractor entirely, spends the model's full resolution on the motif itself,
gives Depth Anything its ideal input (a frontal close-up), and turns the art->3D mapping
into an exact plane mapping instead of a through-the-camera resample.

Reuses, rather than reinvents:
  - texture_bake.trace_view for the camera model + sphere trace (the per-pixel ray/depth
    `basis` used to unproject the painted stroke into 3D and rectify it onto the wall).
  - texture_bake._inpaint_view + neural_appearance.get_sketch_inpaint_pipe for SDXL
    ControlNet inpaint art generation. On the flat canvas the base depth/canny nets have
    nothing real to say, so their scales are passed near-zero and the xinsir scribble net
    carries the drawn shape (see generate_patch_art).
  - Depth Anything V2 (neural_appearance.get_depth_model) to read the generated relief's
    implied height (height_from_art).

The actual SDF fusion (fit_displacement -> final_sdf -> sample_grid -> grid_to_mesh) lives
in refine.py:Refiner.refine_paint_relief, mirroring refine_displacement's existing pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(Path(__file__).resolve().parent), str(REPO / "scripts" / "appearance")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Anti-facade negative prompt (replaces texture_bake.NEG for relief generation): v1's
# washed-out mini-building failure came from SDXL falling back to its architecture-photo
# prior; these terms push against exactly that. The signage terms are from the live UI
# test (2026-07-07): a wide, shallow stroke band reads as a storefront to SDXL, which
# resolved it into a red logo with lettering.
RELIEF_NEG = ("photo, photograph, building, house, facade, window, door, brick wall, "
              "flat, sticker, painting, cartoon, illustration, colorful, text, watermark, "
              "sign, signage, logo, letters, lettering, typography, inscription, "
              "frame, border, blurry, low quality, deformed")


def relief_prompt(user_text=None):
    """Compose the generation prompt: always steer toward a monochrome bas-relief
    "plaster cast" render under raking light. Two reasons: (1) it kills the photo-facade
    attractor that produced v1's washed-out mini-buildings, and (2) it makes the art's
    LUMINANCE track its intended HEIGHT almost by construction (raking light on a relief =
    lit ridges, shadowed recesses), which is exactly what height_from_art reads — both its
    Depth Anything pass (sculptural close-ups are its sweet spot) and its luminance blend.
    `user_text` (the UI's optional prompt) describes WHAT the motif is; the relief-style
    wrapper is applied either way, so a bare subject like "a lion head" still comes out
    carved rather than photographed."""
    core = (f"a carved stone bas-relief of {user_text}" if user_text else
            "a carved stone bas-relief ornament, resolving the sketched shape into one "
            "coherent architectural motif")
    return (core + ", a single sculpted panel centered on a plain flat stone background, "
            "monochrome plaster cast, strong raking light, deep shadows, crisp sculpted "
            "detail, museum quality")


def _resize(arr, shape_hw):
    from PIL import Image
    if arr.shape[:2] == tuple(shape_hw):
        return arr
    h, w = shape_hw
    return np.asarray(Image.fromarray(arr).resize((w, h)))     # PIL size is (W,H)


def _mask_and_rgb_from_paint(paint_img, shape_hw, bg=(0.5, 0.5, 0.5)):
    """A painted image (PIL, any mode) -> (mask_bool, rgb float01, alpha float01) at
    shape_hw=(H,W).

    The mask is "wherever the user actually painted" — for an RGBA canvas (the browser's
    paintCv: transparent background, opaque colored strokes) that's the ALPHA channel, not
    luminance: thresholding luminance instead would silently drop dark paint colors
    (black/navy strokes) from the mask. Plain RGB/L images (e.g. an old white-on-black mask,
    or a plain uploaded photo with no alpha) are treated as fully painted (alpha=255
    everywhere) so this stays compatible with non-canvas inputs.
    rgb is the painted color composited over a neutral `bg`; alpha is returned separately so
    the rectifier can resample the stroke's true coverage onto the wall canvas.
    """
    from PIL import Image
    if not hasattr(paint_img, "convert"):
        paint_img = Image.fromarray(np.asarray(paint_img))
    if paint_img.mode == "RGBA":
        arr = _resize(np.asarray(paint_img), shape_hw).astype(np.float32) / 255.0
        rgb, a = arr[..., :3], arr[..., 3:4]
    else:
        arr = _resize(np.asarray(paint_img.convert("RGB")), shape_hw).astype(np.float32) / 255.0
        rgb, a = arr, np.ones(arr.shape[:2] + (1,), np.float32)
    mask_bool = (a[..., 0] * 255) > 127
    bg_arr = np.asarray(bg, np.float32).reshape(1, 1, 3)
    composited = rgb * a + bg_arr * (1 - a)                # straight alpha-over on neutral bg
    return mask_bool, composited.astype(np.float32), a[..., 0].astype(np.float32)


def _floor8(x, mult=8):
    """SDXL's VAE downsamples by 8x -- diffusers silently floors any non-multiple-of-8
    image dimension to the nearest multiple internally (e.g. height 420 -> 416), which
    would desync the returned art's shape from our mask/canvas arrays if we didn't match
    its rounding up front."""
    return max(mult, int(x) // mult * mult)


def _grid_sdf_at(grid, q, device="cuda"):
    """Trilinear SDF samples of a (D,H,W) cube-frame grid at (N,3) cube-coord points (the
    same grid_sample convention as texture_bake.trace_view's internal sampler)."""
    vol = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, None]
    qt = torch.as_tensor(np.asarray(q, np.float32), device=device).view(1, 1, 1, -1, 3)
    return F.grid_sample(vol, qt.clamp(-1.0, 1.0), mode="bilinear", align_corners=True,
                         padding_mode="border").view(-1)


def paint_locality_mask(surf_pts, pts, band=0.06, inner=0.0, max_ref=4000, seed=0):
    """0..1 falloff by nearest-distance to the painted surface point cloud: 1 on the
    painted surface, fading to 0 across `band` — same shape/intent as
    refine._edit_locality_mask (1 inside a primitive, fading over a seam band), but keyed
    to a point cloud since a paint stroke has no primitive SDF of its own."""
    device = pts.device
    if surf_pts.shape[0] == 0:
        return torch.zeros(pts.shape[0], device=device)
    from scipy.spatial import cKDTree
    ref = surf_pts.detach().cpu().numpy()
    if len(ref) > max_ref:
        ref = ref[np.random.default_rng(seed).choice(len(ref), max_ref, replace=False)]
    d, _ = cKDTree(ref).query(pts.detach().cpu().numpy(), k=1)
    d = torch.as_tensor(d, dtype=torch.float32, device=device)
    return torch.clamp(1.0 - (d - inner) / max(band, 1e-6), 0.0, 1.0)


def flat_wall_point(mask_bool, basis, ring_px=25):
    """Find a reference point representing the "clean, unrecessed wall" near the painted
    patch — NOT the patch's own surface, which may already be inside an existing carved
    window/recess if the user painted over one. Looks at a ring of pixels just OUTSIDE the
    mask and picks whichever hit point is CLOSEST TO THE CAMERA (smallest depth): a window
    recess is farther from the camera than the surrounding flat wall, so the closest point
    nearby is a robust "flat wall" anchor regardless of what's directly under the paint.

    Returns a (3,) numpy point in cube coords, or None if no suitable ring pixels exist.
    """
    import cv2
    m = mask_bool.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
    ring = (cv2.dilate(m, k) > 0) & basis["hit"] & ~mask_bool
    if not ring.any():
        ring = basis["hit"] & ~mask_bool
    if not ring.any():
        return None
    depth = basis["depth"]
    rows, cols = np.where(ring)
    i = np.argmin(depth[rows, cols])
    r0, c0 = rows[i], cols[i]
    cp = np.asarray(basis["cam_pos"], np.float32)
    d0 = np.asarray(basis["dirs"][r0, c0], np.float32)
    return cp + d0 * float(depth[r0, c0])


def wall_frame(grid, mask_bool, basis, device="cuda"):
    """The dominant wall plane under the painted stroke, as an anchored orthonormal frame:
    `p0` a clean unrecessed wall point near the stroke (flat_wall_point), `n` the outward
    SDF-gradient normal there (flipped toward the camera so "out of the wall" always means
    "toward the painter"), `U`/`V` in-plane axes chosen so that, seen from the painter's
    side, U runs right and V runs up (U = world-up x n for walls; for a roof/floor patch,
    where that cross product degenerates, U is seeded from the camera's own right axis)."""
    p0 = flat_wall_point(mask_bool, basis)
    if p0 is None:
        raise ValueError("cannot find a wall reference near the painted stroke")
    eps = 2.0 / grid.shape[0]
    offs = np.array([[eps, 0, 0], [-eps, 0, 0], [0, eps, 0],
                     [0, -eps, 0], [0, 0, eps], [0, 0, -eps]], np.float32)
    s = _grid_sdf_at(grid, p0[None] + offs, device=device).cpu().numpy()
    n = np.array([s[0] - s[1], s[2] - s[3], s[4] - s[5]], np.float32)
    n /= (np.linalg.norm(n) + 1e-8)
    if float(n @ (basis["cam_pos"] - p0)) < 0:
        n = -n
    up = np.array([0.0, 1.0, 0.0], np.float32)
    if abs(float(n @ up)) > 0.95:               # horizontal surface: no meaningful wall-up
        seed = basis["right"] - float(basis["right"] @ n) * n
        U = seed / (np.linalg.norm(seed) + 1e-8)
    else:
        U = np.cross(up, n)
        U /= (np.linalg.norm(U) + 1e-8)
    V = np.cross(n, U)
    return p0.astype(np.float32), n, U, V


def _project_to_view(q, basis):
    """Cube-coord points (N,3) -> continuous (row, col) pixel coords + camera distance in
    the traced view `basis` — the exact inverse of trace_view's ray construction, used to
    resample the painted overlay onto the rectified wall canvas."""
    cp = np.asarray(basis["cam_pos"], np.float32)
    rel = q - cp[None]
    z = np.maximum(rel @ basis["fwd"], 1e-6)
    vx = (rel @ basis["right"] / z) / (basis["th"] * basis["aspect"])
    vy = -(rel @ basis["up"] / z) / basis["th"]
    H, W = basis["hit"].shape
    col = (vx + 1.0) * 0.5 * (W - 1)
    row = (vy + 1.0) * 0.5 * (H - 1)
    return row, col, np.linalg.norm(rel, axis=-1)


def rectify_stroke_to_wall(grid, cam, paint_img, res=512, aspect=1.0, canvas_px=768,
                           pad_frac=0.7, min_extent=0.18, plane_tol=0.25, occl_tol=0.15,
                           device="cuda"):
    """Trace the painter's view once, find the dominant wall plane under the stroke, and
    resample the stroke onto a frontal orthographic canvas of that wall patch.

    The canvas covers the stroke's in-plane bounding box padded by `pad_frac` per side
    (floored to `min_extent` cube units so a tiny dab still gets context), at `canvas_px`
    on its longer side (multiple of 8, shorter side >= 256 — SDXL's usable range). Each
    canvas pixel corresponds to an exact plane point p0 + u*U + v*V; the stroke's color
    and alpha are pulled from the paint overlay by projecting those points back through
    the painter's camera. Two validity gates on the resampled alpha: the view pixel must
    have HIT the building (strokes over empty sky don't count), and the traced depth there
    must agree with the plane point's own camera distance within `occl_tol` (drops paint
    that actually landed on closer/other geometry occluding this wall). Stroke pixels
    whose 3D hits sit farther than `plane_tol` off the plane (spill onto another wall at a
    grazing corner) are dropped before the bounding box is computed, so the canvas frames
    the dominant wall's stroke only.

    Returns a dict: stroke_rgb/stroke_alpha/stroke_mask (Hc,Wc canvas space), pts3d
    (Hc,Wc,3 plane points, cube coords), p0/n/U/V (the wall frame), px_per_unit, basis.
    """
    import cv2
    import texture_bake as tb
    H = _floor8(res)
    W = _floor8(res * aspect)
    _depth, _edge, basis = tb.trace_view(grid, cam, res=H, aspect=W / H, device=device)
    mask_bool, rgb_view, alpha_view = _mask_and_rgb_from_paint(paint_img, basis["hit"].shape)
    hitmask = mask_bool & basis["hit"]
    if not hitmask.any():
        raise ValueError("painted stroke does not hit the building surface from this camera")
    p0, n, U, V = wall_frame(grid, hitmask, basis, device=device)

    cp = np.asarray(basis["cam_pos"], np.float32)
    pts_view = cp[None] + basis["dirs"][hitmask] * basis["depth"][hitmask][:, None]
    on_wall = np.abs((pts_view - p0[None]) @ n) <= plane_tol
    if on_wall.sum() >= 20:
        pts_view = pts_view[on_wall]
    u, v = (pts_view - p0[None]) @ U, (pts_view - p0[None]) @ V
    du, dv = float(u.max() - u.min()), float(v.max() - v.min())
    pu = max(pad_frac * du, (min_extent - du) / 2, 0.02)
    pv = max(pad_frac * dv, (min_extent - dv) / 2, 0.02)
    u0, u1 = float(u.min()) - pu, float(u.max()) + pu
    v0, v1 = float(v.min()) - pv, float(v.max()) + pv
    eu, ev = u1 - u0, v1 - v0
    if eu >= ev:
        Wc, Hc = _floor8(canvas_px), max(_floor8(canvas_px * ev / eu), 256)
    else:
        Hc, Wc = _floor8(canvas_px), max(_floor8(canvas_px * eu / ev), 256)
    uu = u0 + (np.arange(Wc, dtype=np.float32) + 0.5) / Wc * eu
    vv = v1 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc * ev     # row 0 = top of patch
    pts3d = (p0[None, None] + uu[None, :, None] * U[None, None]
             + vv[:, None, None] * V[None, None]).astype(np.float32)

    row, col, dist = _project_to_view(pts3d.reshape(-1, 3), basis)
    map_x = col.reshape(Hc, Wc).astype(np.float32)
    map_y = row.reshape(Hc, Wc).astype(np.float32)
    view_stack = np.dstack([rgb_view, alpha_view, basis["depth"],
                            basis["hit"].astype(np.float32)]).astype(np.float32)
    samp = cv2.remap(view_stack, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    occl_ok = np.abs(samp[..., 4] - dist.reshape(Hc, Wc)) < occl_tol
    stroke_alpha = samp[..., 3] * samp[..., 5] * occl_ok
    return {"stroke_rgb": samp[..., :3], "stroke_alpha": stroke_alpha.astype(np.float32),
            "stroke_mask": stroke_alpha > 0.5, "pts3d": pts3d, "p0": p0, "n": n, "U": U,
            "V": V, "px_per_unit": Wc / eu, "basis": basis}


def scribble_from_mask(mask_bool, thickness=12):
    """Painted mask -> a black-background/white-line "scribble" conditioning image for the
    xinsir SDXL scribble ControlNet (expects hand-drawn-style line art: 0=background,
    255=line). Extracts the painted region's OUTLINE and dilates it by `thickness` px:

      thin  (~2-6px)  -> a precise silhouette the model closely follows ("adjust it to the
                         right thing" — xinsir's documented "thin line = strong control").
      thick (~20-40px, approaching a filled blob) -> coarse guidance the model interprets
                         more freely alongside the prompt ("a creative art piece" — xinsir's
                         documented "thick line = obeys the prompt more").

    One physical knob spans the whole "precise <-> creative" spectrum the model was
    trained for, rather than needing a separate sketch-interpretation step of our own.
    """
    import cv2
    m = mask_bool.astype(np.uint8) * 255
    edges = cv2.Canny(m, 50, 150)
    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thickness), int(thickness)))
        edges = cv2.dilate(edges, k)
    return np.stack([edges] * 3, -1)


def _height_from_luminance(rgb, mask_bool, depth_scale, blur_sigma=2.0):
    """v0 heuristic (brighter than the patch's own mean -> raised, darker -> recessed) —
    same Sobel/luminance idea as texture_bake.make_pbr_maps's height-from-luminance normal
    map. Kept as a fallback if the depth model (height_from_art) can't load, and blended in
    for crisper internal edges. The relief-style prompt (relief_prompt: raking light on
    monochrome stone) makes luminance~height much closer to true than it was for v1's
    photo-style art."""
    import cv2
    lum = cv2.GaussianBlur(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
                           (0, 0), blur_sigma)
    ref = lum[mask_bool] if mask_bool.any() else lum
    centered = lum - float(ref.mean())
    return (np.clip(centered * 2.0, -1.0, 1.0) * float(depth_scale)).astype(np.float32)


def _mask_bbox(mask_bool, pad_frac=0.25, min_size=64):
    """Painted mask -> (y0,y1,x0,x1) bounding box padded by pad_frac on each side. A tiny
    box (e.g. a small brush dab) is floored to min_size so there's enough context around it."""
    ys, xs = np.where(mask_bool)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = mask_bool.shape
    py, px = int((y1 - y0) * pad_frac) + 1, int((x1 - x0) * pad_frac) + 1
    y0, y1 = max(0, y0 - py), min(h, y1 + py + 1)
    x0, x1 = max(0, x0 - px), min(w, x1 + px + 1)
    if y1 - y0 < min_size:
        cy = (y0 + y1) // 2
        y0, y1 = max(0, cy - min_size // 2), min(h, cy + min_size // 2)
    if x1 - x0 < min_size:
        cx = (x0 + x1) // 2
        x0, x1 = max(0, cx - min_size // 2), min(w, cx + min_size // 2)
    return y0, y1, x0, x1


def height_from_art(rgb, mask_bool, depth_scale, blur_sigma=1.0, luminance_blend=0.3,
                    base_frac=0.35):
    """Generated-art RGB (H,W,3 float in [0,1]) -> a height field in SDF cube units,
    via a REAL monocular depth estimator (Depth Anything V2) reading the art's own
    shading/perspective cues, rather than raw pixel brightness (_height_from_luminance,
    the old v0 heuristic) — e.g. a carved-looking motif's actual implied protrusions and
    recesses drive the relief, not just which pixels happen to be bright.

    Depth Anything is a general PHOTO depth estimator: run on the full (H,W) frame, a small
    painted patch surrounded by mostly-flat gray background reads to it as almost the whole
    background of a real scene, and it responds with a soft, low-detail blob that loses the
    shape's actual silhouette detail (verified 2026-07-07: a crisp 5-lobed star produced a
    blurry, undifferentiated bump). Fix: CROP to the mask's own bounding box (+padding) and
    run depth estimation on that close-up crop instead of the full frame. Two further
    corrections, both from the wall-space verification run (sheet_blob_default.png):

    - DETREND: DA predicts a smooth scene-level depth ramp across the crop (its usual
      floor/perspective prior), which on a relief panel is pure bias — our reference
      surface is FLAT by construction (the rectified wall plane). A least-squares plane
      over the motif's own pixels is subtracted, keeping only local relief structure.
    - BASE STEP: DA's output centered on the motif's mean put half the motif BELOW the
      wall plane, which sculpted as a vague smudge, not a carving. A relief plaque
      PROTRUDES: the motif's height is remapped to [base_frac, 1] * depth_scale (its
      internal 5..95th-percentile modulation filling the remaining range), so the
      silhouette gets a real step at its boundary and every motif pixel stands proud of
      the wall. Pixels outside `mask_bool` stay 0 (downstream only reads masked pixels).

    DA's output is disparity-like (larger = closer = should protrude more), so no sign
    flip. Blended with a bit of the sharper luminance heuristic (`luminance_blend`) for
    crisper internal edges — with relief_prompt's raking-light monochrome art, luminance
    genuinely tracks height, so this blend is better-founded than it was for v1 photo art.
    """
    import cv2
    y0, y1, x0, x1 = _mask_bbox(mask_bool)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask_bool[y0:y1, x0:x1]

    try:
        import neural_appearance as na
        import torch
        from PIL import Image
        processor, model = na.get_depth_model()
        img = Image.fromarray((np.clip(crop_rgb, 0, 1) * 255).astype(np.uint8))
        device = next(model.parameters()).device
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            pred = model(**inputs).predicted_depth        # (1, h', w'), model-resolution
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=crop_rgb.shape[:2], mode="bicubic",
            align_corners=False)[0, 0]
        crop_depth = pred.cpu().numpy()
    except Exception as ex:
        print(f"[paint_relief] depth model unavailable ({ex}); falling back to luminance heuristic")
        return _height_from_luminance(rgb, mask_bool, depth_scale, blur_sigma=2.0)

    crop_depth = cv2.GaussianBlur(crop_depth, (0, 0), blur_sigma)
    ys, xs = np.where(crop_mask)
    if len(ys) >= 50:                                     # plane detrend over the motif
        A = np.stack([xs, ys, np.ones_like(xs)], -1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, crop_depth[ys, xs].astype(np.float64), rcond=None)
        gx = np.arange(crop_depth.shape[1], dtype=np.float64)
        gy = np.arange(crop_depth.shape[0], dtype=np.float64)
        crop_depth = crop_depth - (coef[0] * gx[None, :] + coef[1] * gy[:, None] + coef[2])

    if luminance_blend > 0 and crop_mask.any():
        lum = _height_from_luminance(crop_rgb, crop_mask, 1.0, blur_sigma=1.0)
        ds = float(np.abs(crop_depth[crop_mask]).max()) + 1e-6   # match scales before blending
        crop_depth = (1 - luminance_blend) * crop_depth + luminance_blend * lum * ds

    height = np.zeros(mask_bool.shape, np.float32)
    if crop_mask.any():
        vals = crop_depth[crop_mask]
        lo, hi = np.percentile(vals, [5.0, 95.0])
        inside01 = np.clip((vals - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        stepped = float(depth_scale) * (base_frac + (1.0 - base_frac) * inside01)
        block = height[y0:y1, x0:x1]
        block[crop_mask] = stepped.astype(np.float32)
        height[y0:y1, x0:x1] = block
    return height


def segment_generated_art(rgb, region_bool, stroke_bool, margin_px=14, thresh=0.09):
    """Generated-art RGB -> the shape the model actually drew, by color-distance from a
    CALIBRATED background reference (sampled from the model's own output), not a fixed
    neutral constant.

    This exists because the inpainting mask we hand the pipeline (see generate_patch_art)
    is deliberately a generous region, not the user's literal stroke outline — so the
    model has room to invent its own silhouette instead of being hard-clipped to a shape it
    never got to choose. This function figures out what it actually chose.

    A fixed (0.5,0.5,0.5) reference FAILED in practice (verified 2026-07-07): SDXL's
    inpainting output doesn't reproduce an exact neutral gray even for "no content"
    background — it came out with a systematic warm cast (mean [0.50, 0.49, 0.42] instead
    of [0.5,0.5,0.5]), which alone put 89% of true background pixels over the old fixed
    threshold, collapsing the ENTIRE padded box into "foreground" and producing a
    widespread, uncontrolled relief instead of a small localized shape. Fix: calibrate the
    background reference from a ring of the region a safety margin away from the original
    stroke (`stroke_bool` dilated by `margin_px`) — guaranteed to be background by
    construction, and reflecting whatever tint THIS generation actually used.

    Among the resulting connected components, the one that best OVERLAPS the user's stroke
    wins — not the biggest. The biggest-area rule was v1's widespread-relief catastrophe in
    miniature: when the model textures its own background, that background is the largest
    component by far, and picking it sculpts the entire region. The user's stroke is the
    one signal of WHERE the motif is supposed to be; only if nothing overlaps at all does
    area break the tie.
    """
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px, margin_px))
    stroke_dilated = cv2.dilate(stroke_bool.astype(np.uint8), k) > 0
    bg_sample = region_bool & ~stroke_dilated
    bg_arr = (np.median(rgb[bg_sample], axis=0) if bg_sample.sum() >= 50
              else np.asarray((0.5, 0.5, 0.5), np.float32)).reshape(1, 1, 3)
    dist = np.linalg.norm(rgb - bg_arr, axis=-1)
    fg = (dist > thresh) & region_bool
    if not fg.any():
        return region_bool.copy()          # model left everything ~neutral; fall back
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_u8 = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, k2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if n <= 1:
        return region_bool.copy()
    overlaps = [int((stroke_dilated & (labels == i)).sum()) for i in range(1, n)]
    if max(overlaps) > 0:
        return labels == (1 + int(np.argmax(overlaps)))
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def generate_patch_art(grid, cam, paint_img, prompt, style_ref=None, seed=7, steps=28,
                       strength=0.85, sketch_thickness=6, sketch_scale=0.85,
                       res=512, aspect=1.0, canvas_px=768, device="cuda"):
    """Painted stroke -> rectified wall canvas -> SDXL ControlNet inpaint art -> the
    silhouette the model actually drew, everything in canvas space.

    The stroke is rectified onto its wall plane first (rectify_stroke_to_wall), and art is
    generated on that flat frontal canvas: init = the stroke's colors over neutral gray,
    inpaint mask = the whole canvas minus a thin border (kept as calibration background for
    segment_generated_art), conditioning = the xinsir scribble net carrying the stroke's
    outline (`sketch_thickness`/`sketch_scale`: thin+strong follows the drawn shape,
    thick+weak treats it as loose inspiration) with the base depth/canny nets fed constant
    images at near-zero scale — on a flat patch they have nothing real to condition on, and
    v1 showed that feeding them the BUILDING view's maps invites the model to redraw the
    facade. `strength` controls how much the stroke's actual pixels anchor the img2img
    denoise (1.0 ignores them entirely).

    Downstream gets the model's OWN silhouette (segment_generated_art), further clipped to
    plane pixels that lie near the real building surface (|SDF| < 0.08 cube units), so art
    that runs past a wall edge or roofline can't sculpt free-floating relief in the air.

    Returns a dict: rgb (Hc,Wc,3 canvas art), gen_mask (the AI-drawn shape), stroke_mask,
    pts3d (Hc,Wc,3 plane points), p0/n (wall frame, for the flat-reference fusion in
    refine_paint_relief), plus init/scribble images for debug sheets.
    """
    import cv2
    import neural_appearance as na
    import texture_bake as tb
    from PIL import Image

    rect = rectify_stroke_to_wall(grid, cam, paint_img, res=res, aspect=aspect,
                                  canvas_px=canvas_px, device=device)
    stroke_mask = rect["stroke_mask"]
    if not stroke_mask.any():
        raise ValueError("painted stroke does not land on a visible wall from this camera")
    Hc, Wc = stroke_mask.shape
    a = rect["stroke_alpha"][..., None]
    init_np = rect["stroke_rgb"] * a + np.float32(0.5) * (1 - a)
    init_img = Image.fromarray((np.clip(init_np, 0, 1) * 255).astype(np.uint8))

    # the plane extends past the real wall (beyond a roofline, past a corner) but the
    # relief can't: pixels whose plane point is far from the actual surface are off-limits
    near_surface = (np.abs(_grid_sdf_at(grid, rect["pts3d"].reshape(-1, 3), device=device)
                           .cpu().numpy().reshape(Hc, Wc)) < 0.08)

    border = max(8, int(0.04 * min(Hc, Wc)))
    region = np.zeros((Hc, Wc), bool)
    region[border:-border, border:-border] = True
    # ... and neither should the model's WORKING region: asking it to inpaint air past the
    # roofline hands it a truncated band it resolves as something else entirely (live UI
    # test 2026-07-07: a near-cornice stroke came back as a storefront sign) — keep the
    # whole inpaint region on the wall face so the motif is composed for the space that
    # actually exists.
    region &= cv2.erode(near_surface.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if region.sum() < 400:
        raise ValueError("stroke lands on too small a wall patch to compose a relief")
    mask_pil = Image.fromarray(region.astype(np.uint8) * 255)
    scribble_img = Image.fromarray(scribble_from_mask(stroke_mask, thickness=sketch_thickness))
    flat_depth = Image.fromarray(np.full((Hc, Wc, 3), 128, np.uint8))   # constant = flat wall
    blank_edge = Image.fromarray(np.zeros((Hc, Wc, 3), np.uint8))

    ipipe = na.get_sketch_inpaint_pipe()
    rgb = tb._inpaint_view(ipipe, init_img, mask_pil, flat_depth, blank_edge, prompt,
                           style_ref, seed, steps, strength=strength,
                           extra_control_images=[scribble_img],
                           extra_control_scales=[sketch_scale],
                           base_control_scales=[0.2, 0.0], negative=RELIEF_NEG)

    gen_mask = segment_generated_art(rgb, region, stroke_mask)
    clipped = gen_mask & near_surface
    if clipped.sum() >= 30:
        gen_mask = clipped
    if not gen_mask.any():
        gen_mask = stroke_mask.copy()
    return {"rgb": rgb, "gen_mask": gen_mask, "stroke_mask": stroke_mask,
            "pts3d": rect["pts3d"], "p0": rect["p0"], "n": rect["n"],
            "init": init_np, "scribble": np.asarray(scribble_img)}
