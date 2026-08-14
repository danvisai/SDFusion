"""Fidelity checks for footprint_image.extract_footprints (map #97).

Fidelity here means one thing: does the polygon we hand the editor actually describe the region it
was traced from? Measured as union IoU between the returned polygons, rasterised back onto the
source grid, and the building mask itself -- so the number is against the extractor's own input,
not against a hand-labelled truth nobody has.

Two properties this pins that regressed silently before:
  * Otsu runs. It used to raise on this skimage/numpy pair and get swallowed by a bare `except`,
    leaving every extraction at a hardcoded 0.5.
  * Corner-preserving simplification beats fixed uniform resampling, and is allowed to spend
    fewer vertices doing it.

Run:  ./venv/bin/python scripts/server/test_footprint_extract.py
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/server"))

from footprint_image import (                                        # noqa: E402
    MAX_POLY_POINTS, _otsu_threshold, _simplify_corners, building_mask, extract_footprints,
)

SAMPLES = REPO / "scripts/server/web/samples"
IMAGES = ["synthetic_blocks.png", "munich_oldtown.png", "lafayette.png"]


def _mask_of(raw: bytes) -> np.ndarray:
    """The mask extract_footprints itself works from -- imported, not reimplemented, so this file
    cannot drift out of step with the thing it is scoring (it already did once)."""
    from PIL import Image
    return building_mask(np.asarray(Image.open(io.BytesIO(raw)).convert("L")))


def _union_iou(polys_xy, mask) -> float:
    """IoU of the polygons (as (x=col, y=row)) against the mask, on the mask's own grid."""
    from skimage import draw
    acc = np.zeros_like(mask, bool)
    for p in polys_xy:
        rr, cc = draw.polygon(p[:, 1], p[:, 0], shape=mask.shape)
        acc[rr, cc] = True
    u = int((acc | mask).sum())
    return float((acc & mask).sum() / u) if u else 0.0


def check_otsu_runs():
    """The regression that hid: a real image must produce a real threshold, not the 0.5 fallback."""
    raw = (SAMPLES / "munich_oldtown.png").read_bytes()
    from PIL import Image
    u8 = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        thr = _otsu_threshold(u8)
    assert not caught, f"Otsu fell back: {caught[0].message}"
    assert 0.0 < thr < 1.0, f"threshold {thr} outside the [0,1] domain callers use"
    assert abs(thr - 0.5) > 1e-9, "threshold is exactly the fallback value -- suspicious"
    print(f"[otsu] real threshold on Munich = {thr:.4f} (the old silent fallback was 0.5)  OK")


def check_simplification_is_bounded():
    """A pathological outline must not turn into thousands of editor handles."""
    t = np.linspace(0, 2 * np.pi, 4000)
    noisy = np.stack([300 + 120 * np.cos(t) + np.random.default_rng(0).normal(0, 3, t.size),
                      300 + 120 * np.sin(t) + np.random.default_rng(1).normal(0, 3, t.size)], axis=1)
    p = _simplify_corners(noisy, 1.0)
    assert len(p) <= MAX_POLY_POINTS, f"{len(p)} points exceeds the {MAX_POLY_POINTS} cap"
    assert len(p) >= 3, "simplification destroyed the polygon"
    print(f"[simplify] 4000-point noisy circle -> {len(p)} points, within the {MAX_POLY_POINTS} cap  OK")

    square = np.array([[0., 0.], [0., 100.], [100., 100.], [100., 0.], [0., 0.]])
    dense = np.concatenate([np.linspace(square[i], square[i + 1], 50) for i in range(4)])
    p = _simplify_corners(dense, 1.0)
    assert len(p) == 4, f"a square should simplify to its 4 corners, got {len(p)}"
    print("[simplify] densely-sampled square -> exactly 4 corners  OK")


def check_fidelity_beats_uniform():
    """Corner-preserving simplification must not be worse than the fixed-16 resample it replaces."""
    print(f"\n{'sample':<22}{'uniform-16':>12}{'dp tol=1.0':>12}{'ceiling':>10}   vertices (uniform -> dp)")
    for name in IMAGES:
        raw = (SAMPLES / name).read_bytes()
        mask = _mask_of(raw)

        legacy, _ = extract_footprints(raw)                       # default: uniform 16
        dp, _ = extract_footprints(raw, simplify_px=1.0)
        full, _ = extract_footprints(raw, simplify_px=0.1)        # near-verbatim: the ceiling

        # Vacuity guard, and it is not hypothetical: an earlier version of this file "passed" every
        # IoU assertion below while extracting ZERO polygons, because 0 == 0 satisfies them all.
        # These samples carry 16, 29 and 32 buildings; anything near empty is a broken extractor.
        assert len(legacy) >= 10, f"{name}: only {len(legacy)} regions -- extraction is broken"
        assert 0.05 < mask.mean() < 0.60, f"{name}: mask covers {mask.mean():.1%} -- thresholding is broken"
        assert len(dp) == len(legacy), f"{name}: {len(dp)} regions vs {len(legacy)} -- simplification changed detection"
        iou_l, iou_d, iou_f = (_union_iou(p, mask) for p in (legacy, dp, full))
        v_l, v_d = sum(len(p) for p in legacy), sum(len(p) for p in dp)

        print(f"{name:<22}{iou_l:>12.4f}{iou_d:>12.4f}{iou_f:>10.4f}   {v_l:>5d} -> {v_d:<5d}")
        assert iou_d >= iou_l - 1e-6, f"{name}: simplification LOST fidelity ({iou_d:.4f} < {iou_l:.4f})"
        assert iou_d <= iou_f + 1e-6, f"{name}: {iou_d:.4f} exceeds the un-simplified ceiling {iou_f:.4f}"
        assert all(len(p) <= MAX_POLY_POINTS for p in dp), f"{name}: a polygon blew the point cap"
    print("[fidelity] corner-preserving simplification >= uniform-16 on every sample  OK")


def check_legacy_default_unchanged():
    """inference_service.py's older flow passes no simplify_px and must keep its exact behaviour."""
    raw = (SAMPLES / "munich_oldtown.png").read_bytes()
    polys, hw = extract_footprints(raw)
    assert len(polys) >= 10, f"only {len(polys)} polygons from the default path -- extraction is broken"
    assert all(len(p) <= 16 for p in polys), "the default path must still cap at 16 points"
    assert hw == _mask_of(raw).shape, "image shape contract changed"
    print(f"[legacy] default path still returns <=16 points/polygon ({len(polys)} polygons)  OK")


def main():
    check_otsu_runs()
    check_simplification_is_bounded()
    check_legacy_default_unchanged()
    check_fidelity_beats_uniform()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
