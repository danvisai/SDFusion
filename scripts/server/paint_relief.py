"""Paint-to-relief: a user paints a patch on a building, a diffusion model generates 2D
art restricted to that patch, and the art is fused into the building's SDF as a REAL
geometric relief (marching cubes shows an actual bump/carving, not just a color texture).

Reuses, rather than reinvents:
  - texture_bake.trace_view for the camera model + sphere trace (depth/edge G-buffer, the
    ControlNet conditioning images, plus the per-pixel ray/depth `basis` used here to
    unproject the painted mask into 3D).
  - texture_bake._inpaint_view + neural_appearance.get_inpaint_pipe for mask-restricted
    SDXL ControlNet art generation (the same "TEXTure-style" inpaint call used by the v2.1
    texture bake, just gated to the user's paint stroke instead of a newly-revealed view).
  - The Sobel/luminance "height-from-luminance" idea in texture_bake.make_pbr_maps,
    repurposed here to emit a scalar height field (brighter = raised, darker = recessed)
    instead of a shading normal map.

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


def _resize(arr, shape_hw):
    from PIL import Image
    if arr.shape[:2] == tuple(shape_hw):
        return arr
    h, w = shape_hw
    return np.asarray(Image.fromarray(arr).resize((w, h)))     # PIL size is (W,H)


def _mask_and_rgb_from_paint(paint_img, shape_hw, bg=(0.5, 0.5, 0.5)):
    """A painted image (PIL, any mode) -> (mask_bool, rgb float01) at shape_hw=(H,W).

    The mask is "wherever the user actually painted" — for an RGBA canvas (the browser's
    paintCv: transparent background, opaque colored strokes) that's the ALPHA channel, not
    luminance: thresholding luminance instead would silently drop dark paint colors
    (black/navy strokes) from the mask. Plain RGB/L images (e.g. an old white-on-black mask,
    or a plain uploaded photo with no alpha) are treated as fully painted (alpha=255
    everywhere) so this stays compatible with non-canvas inputs.
    rgb is the painted color composited over a neutral `bg` (used as the img2img init image
    — unpainted texels don't matter since only the masked region is read downstream).
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
    return mask_bool, composited.astype(np.float32)


def _floor8(x, mult=8):
    """SDXL's VAE downsamples by 8x -- diffusers silently floors any non-multiple-of-8
    image dimension to the nearest multiple internally (e.g. height 420 -> 416), which
    would desync the returned art's shape from our depth/edge/mask arrays (all still at
    the untouched requested size) if we didn't match its rounding up front."""
    return max(mult, int(x) // mult * mult)


def unproject_mask(grid, cam, paint_img, device="cuda", res=512, aspect=1.0, bg=(0.5, 0.5, 0.5)):
    """Sphere-trace `cam` (same {"pos","look","fov"} shape texture_bake.trace_view takes)
    against `grid`, then keep only the 3D surface hits under the painted region.

    `res`/`aspect` match trace_view's convention (res=image height, aspect=width/height) —
    pass the LIVE VIEWPORT's own aspect so the paint stroke unprojects against exactly what
    the user saw, not a forced square crop. Both are rounded to a multiple of 8 (see
    `_floor8`) before tracing, so every array downstream (depth/edge/mask/generated art)
    agrees on shape.

    Returns (surf_pts (M,3) cube coords, mask_bool (H,W), paint_rgb (H,W,3) float01,
    depth_img, edge_img, basis). depth_img/edge_img are this exact view's ControlNet
    conditioning images, so a caller generating art for the same patch doesn't need to
    re-trace.
    """
    import texture_bake as tb
    H = _floor8(res)
    W = _floor8(res * aspect)
    depth_img, edge_img, basis = tb.trace_view(grid, cam, res=H, aspect=W / H, device=device)
    mask_bool, paint_rgb = _mask_and_rgb_from_paint(paint_img, basis["hit"].shape, bg=bg)

    cp = torch.as_tensor(basis["cam_pos"], dtype=torch.float32, device=device)
    dirs = torch.as_tensor(basis["dirs"], dtype=torch.float32, device=device)
    t = torch.as_tensor(basis["depth"], dtype=torch.float32, device=device)
    hit = torch.as_tensor(basis["hit"], device=device) & torch.as_tensor(mask_bool, device=device)
    pts = (cp[None, None] + dirs * t[..., None])[hit]
    return pts, mask_bool, paint_rgb, depth_img, edge_img, basis


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
    map. Kept as a fallback if the depth model (height_from_art) can't load."""
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


def height_from_art(rgb, mask_bool, depth_scale, blur_sigma=1.0, luminance_blend=0.3):
    """Generated-art RGB (H,W,3 float in [0,1]) -> a signed height field in SDF cube units,
    via a REAL monocular depth estimator (Depth Anything V2) reading the art's own
    shading/perspective cues, rather than raw pixel brightness (_height_from_luminance,
    the old v0 heuristic) — e.g. a carved-looking motif's actual implied protrusions and
    recesses drive the relief, not just which pixels happen to be bright.

    Depth Anything is a general PHOTO depth estimator: run on the full (H,W) frame, a small
    painted patch surrounded by mostly-flat gray background reads to it as almost the whole
    background of a real scene, and it responds with a soft, low-detail blob that loses the
    shape's actual silhouette detail (verified 2026-07-07: a crisp 5-lobed star produced a
    blurry, undifferentiated bump). Fix: CROP to the mask's own bounding box (+padding) and
    run depth estimation on that close-up crop instead of the full frame, so the model's
    resolution is spent on the shape itself, then paste the result back at (0) elsewhere.

    Depth Anything's output is disparity-like (larger value = closer to the camera), which
    is exactly "should protrude more" for a relief — so no sign flip is needed once
    centered on the patch's own mean and scaled by its own spread. Blended with a bit of
    the sharper luminance heuristic (`luminance_blend`) for crisper internal edges, since
    monocular depth alone tends to smooth over flat, low-shading painted regions.
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
    ref = crop_depth[crop_mask] if crop_mask.any() else crop_depth
    spread = float(ref.std()) * 3.0 + 1e-6                  # ~3 sigma covers most of the range
    crop_norm = np.clip((crop_depth - float(ref.mean())) / spread, -1.0, 1.0)

    if luminance_blend > 0:
        crop_lum = _height_from_luminance(crop_rgb, crop_mask, 1.0, blur_sigma=1.0)
        crop_norm = (1 - luminance_blend) * crop_norm + luminance_blend * crop_lum

    height = np.zeros(mask_bool.shape, np.float32)
    height[y0:y1, x0:x1] = crop_norm
    return (height * float(depth_scale)).astype(np.float32)


def segment_generated_art(rgb, region_bool, stroke_bool, margin_px=14, thresh=0.09):
    """Generated-art RGB -> the shape the model actually drew, by color-distance from a
    CALIBRATED background reference (sampled from the model's own output), not a fixed
    neutral constant.

    This exists because the inpainting mask we hand the pipeline (see generate_patch_art)
    is deliberately a generous padded BOX, not the user's literal stroke outline — so the
    model has room to invent its own silhouette instead of being hard-clipped to a shape it
    never got to choose. This function figures out what it actually chose.

    A fixed (0.5,0.5,0.5) reference FAILED in practice (verified 2026-07-07): SDXL's
    inpainting output doesn't reproduce an exact neutral gray even for "no content"
    background — it came out with a systematic warm cast (mean [0.50, 0.49, 0.42] instead
    of [0.5,0.5,0.5]), which alone put 89% of true background pixels over the old fixed
    threshold, collapsing the ENTIRE padded box into "foreground" and producing a
    widespread, uncontrolled relief instead of a small localized shape. Fix: calibrate the
    background reference from a ring of the padded box a safety margin away from the
    original stroke (`stroke_bool` dilated by `margin_px`) — guaranteed to be background by
    construction, and reflecting whatever tint THIS generation actually used.
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
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def generate_patch_art(grid, cam, paint_img, prompt, style_ref=None, seed=7, steps=28,
                       strength=0.6, sketch_thickness=12, sketch_scale=0.85,
                       res=512, aspect=1.0, device="cuda"):
    """Painted patch (a rough/vague drawn shape, e.g. from a browser color-brush canvas, or
    an uploaded photo) -> SDXL ControlNet img2img/INPAINT art generated FREELY over a
    generous region around the stroke — an Adobe-Generative-Fill-style "sketch -> refined
    art" tool, not a literal paint-bucket: the DRAWN SHAPE only ever reaches the model as a
    soft hint (the scribble ControlNet's edge map, plus the painted colors composited into
    the init image), never as a hard pixel mask, so the model is free to invent a different
    silhouette entirely (a real motif, or a cleaner abstract form) rather than just shading
    in the user's literal outline.

    This matters mechanically: SDXL inpainting hard-preserves every pixel OUTSIDE its
    `mask_image` regardless of ControlNet scale or `strength` (verified 2026-07-07 — sweeping
    the scribble ControlNet's scale from 0.85 down to 0.25 changed only internal texture, the
    silhouette stayed pixel-identical to the input, because the mask boundary IS the literal
    stroke outline). So the fix is upstream of any conditioning weight: inpaint over a padded
    BOX around the stroke (`_mask_bbox`, generous padding) instead of the stroke's exact
    shape, then read back what the model actually drew (`segment_generated_art`) and use
    THAT as the mask for everything downstream (the 3D relief follows the AI's silhouette,
    not the user's rough click-mask).

    `sketch_thickness`/`sketch_scale` control how strongly the scribble hint pulls the
    result toward the drawn shape: thin+strong -> "adjust it to the right thing"; thick
    +weaker -> "a creative art piece" (see scribble_from_mask). `strength` separately
    controls how much the painted COLORS (vs. just the shape) survive — 1.0 discards them
    entirely (full hallucination), ~0.5-0.7 keeps them recognizable.

    `res`/`aspect`: trace_view's convention (res=image height, aspect=width/height) — the
    live viewport's own aspect, so painting happens directly on the current 3D view instead
    of a forced square crop.

    Returns (rgb (H,W,3) float in [0,1], surf_pts (M,3) cube coords of the AI-drawn surface,
    mask_bool (H,W) of the AI-drawn shape (not the click-mask), basis) — everything
    refine_paint_relief needs to turn this into a height field, without re-tracing.
    """
    import neural_appearance as na
    import texture_bake as tb
    from PIL import Image

    surf_pts, mask_bool, paint_rgb, depth_img, edge_img, basis = unproject_mask(
        grid, cam, paint_img, device=device, res=res, aspect=aspect)
    if surf_pts.shape[0] == 0:
        raise ValueError("painted mask does not hit the building surface from this camera")

    y0, y1, x0, x1 = _mask_bbox(mask_bool, pad_frac=0.7, min_size=96)
    inpaint_region = np.zeros_like(mask_bool)
    inpaint_region[y0:y1, x0:x1] = True
    inpaint_region &= basis["hit"]

    init_img = Image.fromarray((np.clip(paint_rgb, 0, 1) * 255).astype(np.uint8))
    mask_pil = Image.fromarray((inpaint_region.astype(np.uint8)) * 255)
    scribble_img = Image.fromarray(scribble_from_mask(mask_bool, thickness=sketch_thickness))
    ipipe = na.get_sketch_inpaint_pipe()
    rgb = tb._inpaint_view(ipipe, init_img, mask_pil, depth_img, edge_img, prompt,
                           style_ref, seed, steps, strength=strength,
                           extra_control_images=[scribble_img],
                           extra_control_scales=[sketch_scale])

    gen_mask = segment_generated_art(rgb, inpaint_region, mask_bool)
    cp = np.asarray(basis["cam_pos"], np.float32)
    dirs = np.asarray(basis["dirs"], np.float32)
    depth = np.asarray(basis["depth"], np.float32)
    hit = gen_mask & basis["hit"]
    surf_pts_gen = torch.as_tensor(cp[None, None] + dirs * depth[..., None], dtype=torch.float32,
                                   device=device)[hit]
    if surf_pts_gen.shape[0] == 0:                     # segmentation failed -> fall back to click-mask
        return rgb, surf_pts, mask_bool, basis
    return rgb, surf_pts_gen, gen_mask, basis
