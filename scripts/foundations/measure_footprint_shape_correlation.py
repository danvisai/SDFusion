"""#164: does the SERVED height-map model's per-building error correlate with footprint-shape
statistics the tentative design proposes adding as new conditioning channels?

The design's justification for solidity, aspect ratio, perimeter^2/area, and vertex count leans on
the same "derivable from the footprint" argument already used for the existing edt channel -- but
the raw mask is also derivable from the footprint, and the network already has it. `make_model`
(train_height_map_generator.py) is a UNet with an 8x8 bottleneck and no global-pooling path, so
plan-global integrals like footprint area or convex-hull area may be uncomputable within its
receptive field, or the net may already extract them fine, in which case new channels would just
dilute the stem for zero gain. This is a one-measurement probe of that question, cheap and with no
training run: join each of the pinned 714's stored per-building errors (`extra`, `missing`,
`vol_iou`, already scored by the currently served `heightmap_ce_median` arm in
`execution/artifacts/height_map_generator_class_714.json`) against shape statistics computed fresh
from that same building's footprint mask, and report Pearson/Spearman correlation plus a scatter
for each (statistic, error metric) pair.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/measure_footprint_shape_correlation.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SERVED_ARTIFACT = REPO / "execution/artifacts/height_map_generator_class_714.json"
SERVED_ARM = "heightmap_ce_median"
# Project convention (eval_massing_arms.py): 3D IoU is always reported split into missing/extra,
# never as a lone number, because they are opposite failure modes (under- vs over-build). vol_iou
# is kept alongside as the single aggregate a reader may still want.
ERROR_METRICS = ("extra", "missing", "vol_iou")
SHAPE_STATS = ("solidity", "aspect_ratio", "perimeter_sq_over_area", "vertex_count")


def footprint_polygon(mask: np.ndarray, tol_px: float = 1.0) -> np.ndarray | None:
    """Boolean footprint mask -> Douglas-Peucker-simplified polygon vertices (row, col).

    Reuses `footprint_image._simplify_corners`, the same corner-preserving simplification the town
    editor traces user-uploaded footprints with, rather than the fixed-count uniform resampling
    `refine.py`'s `_mask_to_polygon` uses for rendering -- vertex count and perimeter need real
    corners, not a rendering budget.
    """
    from skimage import measure

    from scripts.server.footprint_image import _simplify_corners

    m = np.asarray(mask) > 0
    contours = measure.find_contours(m.astype(float), 0.5)
    if not contours:
        return None
    c = max(contours, key=len)
    if len(c) < 3:
        return None
    poly = _simplify_corners(c, tol_px)
    return poly if len(poly) >= 3 else None


def footprint_shape_stats(mask: np.ndarray, tol_px: float = 1.0) -> dict | None:
    """The four candidate conditioning-channel statistics for one building's footprint mask.

    - solidity: mask pixel area / convex-hull area (train_vecset.py's `Bag3dDataset` precedent).
      1.0 = convex; lower = re-entrant (courtyards, L-plans, terraced party walls).
    - aspect_ratio: bounding-box max(h,w)/min(h,w) of the mask's nonzero extent.
    - perimeter_sq_over_area: polygon perimeter^2 / mask pixel area -- an isoperimetric shape-
      complexity statistic (16 for a square, larger for an elongated or jagged plan).
    - vertex_count: vertices in the Douglas-Peucker-simplified polygon.

    Returns None for a mask with no polygon or convex hull to compute (empty, or too few nonzero
    pixels) -- callers must count and report these, not drop them silently.
    """
    from scipy.spatial import ConvexHull

    m = np.asarray(mask) > 0
    area = int(m.sum())
    ys, xs = np.nonzero(m)
    if area == 0 or len(xs) < 3:
        return None
    h, w = float(np.ptp(ys) + 1), float(np.ptp(xs) + 1)
    aspect_ratio = max(h, w) / max(min(h, w), 1.0)
    try:
        hull_area = ConvexHull(np.c_[xs, ys].astype(float)).volume   # 2-D: .volume is area
    except Exception:
        return None
    if hull_area <= 0:
        return None
    solidity = float(np.clip(area / hull_area, 0.0, 1.0))
    poly = footprint_polygon(m, tol_px=tol_px)
    if poly is None:
        return None
    edges = np.diff(np.vstack([poly, poly[:1]]), axis=0)
    perimeter = float(np.hypot(edges[:, 0], edges[:, 1]).sum())
    if perimeter <= 0:
        return None
    return dict(solidity=float(solidity), aspect_ratio=float(aspect_ratio),
               perimeter_sq_over_area=perimeter ** 2 / area, vertex_count=int(len(poly)))


def correlate(stat_values, error_values) -> dict:
    """Pearson and Spearman correlation between one shape statistic and one error metric.

    NaN-guarded below 3 paired points, matching `probe_surface_loss.py`'s `spearman` convention --
    an under-powered pair must read as absent, not raise or silently report a spurious +-1.0.
    """
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(stat_values, float)
    y = np.asarray(error_values, float)
    n = len(x)
    if n < 3:
        return dict(n=n, pearson_r=float("nan"), pearson_p=float("nan"),
                   spearman_r=float("nan"), spearman_p=float("nan"))
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return dict(n=n, pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
               spearman_r=float(sr.statistic), spearman_p=float(sr.pvalue))


def collect_records(cache: dict, per_building: list, tol_px: float = 1.0) -> tuple[list, int]:
    """Join each pinned building's stored error metrics to freshly computed footprint-shape stats.

    Returns (records, n_skipped). A row is skipped, and counted, if its id is not in the height-
    field cache or its mask is too degenerate to form shape stats from -- both must be visible in
    the reported total rather than silently shrinking the correlated population.
    """
    row_to_idx = {int(r): i for i, r in enumerate(cache["row"])}
    records, skipped = [], 0
    for b in per_building:
        idx = row_to_idx.get(int(b["id"]))
        if idx is None:
            skipped += 1
            continue
        stats = footprint_shape_stats(cache["fp"][idx], tol_px=tol_px)
        if stats is None:
            skipped += 1
            continue
        records.append(dict(id=int(b["id"]), **stats,
                            **{metric: float(b[metric]) for metric in ERROR_METRICS}))
    return records, skipped


def _plot_scatters(records: list, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    for stat in SHAPE_STATS:
        xs = [r[stat] for r in records]
        fig, axes = plt.subplots(1, len(ERROR_METRICS), figsize=(4 * len(ERROR_METRICS), 4))
        for ax, err in zip(axes, ERROR_METRICS):
            ax.scatter(xs, [r[err] for r in records], s=6, alpha=0.35)
            ax.set_xlabel(stat)
            ax.set_ylabel(err)
        fig.suptitle(f"#164: {stat} vs served-model error ({len(records)} buildings)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stat}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--served-artifact", type=Path, default=SERVED_ARTIFACT)
    ap.add_argument("--arm", default=SERVED_ARM)
    ap.add_argument("--tol-px", type=float, default=1.0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "execution/artifacts/footprint_shape_error_correlation.json")
    ap.add_argument("--plot-dir", type=Path,
                    default=REPO / "execution/artifacts/footprint_shape_error_correlation_plots")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    from scripts.foundations.train_height_map_generator import build_cache

    served = json.loads(args.served_artifact.read_text())
    per_building = served["per_building"][args.arm]
    cache = build_cache()

    records, skipped = collect_records(cache, per_building, tol_px=args.tol_px)
    print(f"[join] {len(records)}/{len(per_building)} '{args.arm}' buildings matched with "
        f"computable shape stats ({skipped} skipped)", flush=True)

    correlations = {
        stat: {
            err: correlate([r[stat] for r in records], [r[err] for r in records])
            for err in ERROR_METRICS
        }
        for stat in SHAPE_STATS
    }
    for stat in SHAPE_STATS:
        for err in ERROR_METRICS:
            c = correlations[stat][err]
            print(f"  {stat:24s} vs {err:9s}  pearson={c['pearson_r']:+.3f} "
                f"(p={c['pearson_p']:.3g})  spearman={c['spearman_r']:+.3f} "
                f"(p={c['spearman_p']:.3g})", flush=True)

    if not args.no_plots:
        _plot_scatters(records, args.plot_dir)
        print(f"[plots] {args.plot_dir}", flush=True)

    out = dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), question="#164",
                        served_artifact=str(args.served_artifact.relative_to(REPO)),
                        arm=args.arm, tol_px=args.tol_px,
                        n_pinned=len(per_building), n_matched=len(records), n_skipped=skipped),
             correlations=correlations, records=records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
