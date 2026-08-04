"""Generate procedural SDF training data for Stage 3a recipe augmentation.

We supplement BuildingNet's 1480 training assets with synthesized
(footprint, class, height, style, SDF) tuples derived from
`scene/sdf_recipes.py`. This attacks the historical "SDFusion from scratch
never converged" failure mode (see memory: project_sdfusion_convergence_history)
by providing the model with ~50k extra paired samples that span the conditioning
distribution densely.

Per style we sample ~6250 examples; 8 styles -> ~50k samples total. The output
is chunked HDF5 (one file per style) under `data/recipe_augmentation_v1/`:

    /sdf       (N, 64, 64, 64) float32   in Frame-N (sampled on bbox covering polygon)
    /footprint (N, 64, 64)     uint8     top-down silhouette (axis 1 = y) of sdf<=0
    /height_m  (N,)            float32   target body height in world meters
    /class_id  (N,)            int32     per-style natural class id (see CLASS_FOR_STYLE)
    /style_id  (N,)            int32     0..7 (indexes STYLES tuple in sdf_recipes)
    /seed      (N,)            int64     RNG seed for the recipe call
    /shape_id  (N,)            int32     0=rect, 1=L, 2=T, 3=quad, 4=hex

Naming convention matches what Stage 3a's dataset loader expects (see plan,
Phase 2). The SDF is stored *untruncated* — clamp at training time so we can
re-target a different `trunc_thres` without regenerating.

Speed-wise this is CPU-bound (one chunk on a 64^3 grid on CPU is fast: ~20-50ms
per sample with the existing primitives). Multiprocess over N workers; default
16 leaves headroom on the 30-core node so the repreview background run is not
starved.

Usage:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
        scripts/generate_recipe_augmentation.py \\
        --out_dir data/recipe_augmentation_v1 \\
        --per_style 6250 --workers 16

Smoke:
    ... --per_style 32 --workers 4
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


SDF_RES = 64  # match BuildingNet preprocess/create_sdf.py

# A natural primary class per style. Real BuildingNet has class as a per-asset
# fixed property; for recipe augmentation we pin each style to its most-typical
# class so the (class, style) joint never invents combinations the model would
# never see at inference.
CLASS_FOR_STYLE = {
    "modern":         "COMMERCIALoffice_building",
    "colonial":       "RESIDENTIALhouse",
    "victorian":      "RESIDENTIALhouse",
    "industrial":     "COMMERCIALfactory",
    "craftsman":      "RESIDENTIALhouse",
    "mediterranean":  "RESIDENTIALvilla",
    "contemporary":   "COMMERCIALoffice_building",
    "public_civic":   "PUBLICmuseum",
}

# Per-class height priors in meters (mean, std clipped to [low, high]).
# These mirror osm_recompose_height_policy.py:area_default_height shape but
# with more variance to populate the conditioning space.
HEIGHT_PRIORS = {
    "RESIDENTIALhouse":         (7.0, 2.0, 4.0, 12.0),
    "RESIDENTIALvilla":         (8.5, 2.5, 5.0, 14.0),
    "COMMERCIALoffice_building":(15.0, 5.0, 6.0, 35.0),
    "COMMERCIALfactory":        (9.0, 2.5, 5.0, 18.0),
    "PUBLICmuseum":             (12.0, 3.5, 7.0, 22.0),
}

# Polygon size ranges in meters. Sampled per shape kind.
SHAPE_SIZE_RANGES = {
    "rect": (5.0, 25.0),
    "L":    (5.0, 18.0),
    "T":    (5.0, 18.0),
    "quad": (5.0, 22.0),
    "hex":  (5.0, 22.0),
}

SHAPE_IDS = {"rect": 0, "L": 1, "T": 2, "quad": 3, "hex": 4}


# --- footprint samplers -----------------------------------------------------

def _rect(rng: np.random.Generator, name: str = "rect") -> np.ndarray:
    lo, hi = SHAPE_SIZE_RANGES[name]
    w = float(rng.uniform(lo, hi))
    d = float(rng.uniform(lo, hi))
    return np.array([
        [-w / 2, -d / 2], [w / 2, -d / 2],
        [w / 2,  d / 2], [-w / 2,  d / 2],
    ], dtype=np.float32)


def _l_shape(rng: np.random.Generator) -> np.ndarray:
    lo, hi = SHAPE_SIZE_RANGES["L"]
    # Two arms of an L.
    a = float(rng.uniform(lo, hi))
    b = float(rng.uniform(lo, hi))
    t1 = float(rng.uniform(lo * 0.4, hi * 0.6))
    t2 = float(rng.uniform(lo * 0.4, hi * 0.6))
    # L laid in +x +z quadrant, shifted to centroid.
    poly = np.array([
        [0,   0], [a,   0], [a,  t1], [t2, t1], [t2, b], [0, b],
    ], dtype=np.float32)
    poly -= poly.mean(axis=0)
    return poly


def _t_shape(rng: np.random.Generator) -> np.ndarray:
    lo, hi = SHAPE_SIZE_RANGES["T"]
    w_top = float(rng.uniform(lo * 1.2, hi))
    h_top = float(rng.uniform(lo * 0.4, hi * 0.6))
    w_stem = float(rng.uniform(lo * 0.4, hi * 0.7))
    h_stem = float(rng.uniform(lo, hi))
    # T laid with top crossbar at +z.
    poly = np.array([
        [-w_top / 2, h_top],
        [ w_top / 2, h_top],
        [ w_top / 2, 0],
        [ w_stem / 2, 0],
        [ w_stem / 2, -h_stem],
        [-w_stem / 2, -h_stem],
        [-w_stem / 2, 0],
        [-w_top / 2, 0],
    ], dtype=np.float32)
    poly -= poly.mean(axis=0)
    return poly


def _quad(rng: np.random.Generator) -> np.ndarray:
    """Irregular convex quad."""
    lo, hi = SHAPE_SIZE_RANGES["quad"]
    base = float(rng.uniform(lo, hi))
    jitter = base * 0.25
    # Start from a square, perturb each vertex.
    p = np.array([
        [-base / 2, -base / 2], [ base / 2, -base / 2],
        [ base / 2,  base / 2], [-base / 2,  base / 2],
    ], dtype=np.float32)
    p += rng.uniform(-jitter, jitter, size=p.shape).astype(np.float32)
    p -= p.mean(axis=0)
    return p


def _hex(rng: np.random.Generator) -> np.ndarray:
    lo, hi = SHAPE_SIZE_RANGES["hex"]
    r = float(rng.uniform(lo / 2.0, hi / 2.0))
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + rng.uniform(-0.2, 0.2)
    radii = r * rng.uniform(0.7, 1.0, size=6).astype(np.float32)
    p = np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=-1).astype(np.float32)
    p -= p.mean(axis=0)
    return p


SHAPE_FNS = {
    "rect": _rect, "L": _l_shape, "T": _t_shape, "quad": _quad, "hex": _hex,
}
# Per-style shape weights — modern/industrial prefer rectangles; victorian /
# public_civic prefer more complex footprints.
SHAPE_WEIGHTS_PER_STYLE = {
    "modern":         {"rect": 4, "L": 1, "T": 0, "quad": 1, "hex": 0},
    "colonial":       {"rect": 4, "L": 2, "T": 1, "quad": 1, "hex": 0},
    "victorian":      {"rect": 2, "L": 3, "T": 2, "quad": 1, "hex": 1},
    "industrial":     {"rect": 5, "L": 1, "T": 0, "quad": 1, "hex": 0},
    "craftsman":      {"rect": 3, "L": 2, "T": 1, "quad": 1, "hex": 0},
    "mediterranean":  {"rect": 3, "L": 2, "T": 0, "quad": 1, "hex": 1},
    "contemporary":   {"rect": 3, "L": 1, "T": 0, "quad": 2, "hex": 1},
    "public_civic":   {"rect": 2, "L": 1, "T": 1, "quad": 1, "hex": 2},
}


def sample_polygon(style: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    weights = SHAPE_WEIGHTS_PER_STYLE[style]
    kinds, ws = zip(*weights.items())
    probs = np.array(ws, dtype=np.float64)
    probs = probs / probs.sum()
    kind = rng.choice(kinds, p=probs)
    return SHAPE_FNS[kind](rng), kind


def sample_height(class_label: str, rng: np.random.Generator) -> float:
    mean, std, lo, hi = HEIGHT_PRIORS[class_label]
    h = float(rng.normal(mean, std))
    return float(np.clip(h, lo, hi))


# --- worker -----------------------------------------------------------------

def _generate_one(args):
    style, sample_idx, seed_global = args
    # Clamp intra-op threading: spawn-pool workers each run a single SDF eval
    # at a time, and torch/blas otherwise launch their own OMP pools per call
    # which oversubscribes the CPU on multi-worker runs (16 workers x 2 cores
    # each on a 30-core node deadlocked our first attempt).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch  # heavy import; doing it lazy keeps fork() cheap
    torch.set_num_threads(1)
    from scene.sdf_primitives import polygon_bbox_with_pad, sample_grid
    from scene.sdf_recipes import build_styled_sdf, STYLES

    rng = np.random.default_rng(seed_global)
    polygon, kind = sample_polygon(style, rng)
    class_label = CLASS_FOR_STYLE[style]
    height_m = sample_height(class_label, rng)

    # Force CPU for the primitives — we want to coexist with GPU jobs.
    sdf_fn = build_styled_sdf(style, polygon, height_m, seed=seed_global)
    # The bbox needs ~2.5x height on Y to capture roof + ornaments per
    # generate_sdf_building.py:93.
    bbox = polygon_bbox_with_pad(polygon, height_m * 2.5, pad=0.10)
    grid = sample_grid(sdf_fn, SDF_RES, bbox, device="cpu")
    sdf_np = grid.detach().cpu().numpy().astype(np.float32)  # (D, H, W) = (z, y, x)

    # Footprint: top-down silhouette via 'any over Y'.
    footprint = (sdf_np <= 0.0).any(axis=1).astype(np.uint8)  # (D=z, W=x)

    style_id = STYLES.index(style)
    return {
        "sample_idx": sample_idx,
        "sdf": sdf_np,
        "footprint": footprint,
        "height_m": height_m,
        "class_label": class_label,
        "style": style,
        "style_id": style_id,
        "seed": seed_global,
        "shape_id": SHAPE_IDS[kind],
    }


def _global_class_ids(class_labels: list[str]) -> tuple[dict, list[str]]:
    """Stable int id per unique class label. Sorted for determinism."""
    uniq = sorted(set(class_labels))
    return {c: i for i, c in enumerate(uniq)}, uniq


def _write_style_h5(out_path: Path, results: list[dict], class_id_map: dict):
    n = len(results)
    if n == 0:
        return
    sdf_arr = np.stack([r["sdf"] for r in results], axis=0)
    fp_arr = np.stack([r["footprint"] for r in results], axis=0)
    height_arr = np.array([r["height_m"] for r in results], dtype=np.float32)
    style_id_arr = np.array([r["style_id"] for r in results], dtype=np.int32)
    seed_arr = np.array([r["seed"] for r in results], dtype=np.int64)
    shape_arr = np.array([r["shape_id"] for r in results], dtype=np.int32)
    class_id_arr = np.array([class_id_map[r["class_label"]] for r in results], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("sdf", data=sdf_arr, compression="gzip", compression_opts=4)
        f.create_dataset("footprint", data=fp_arr, compression="gzip", compression_opts=4)
        f.create_dataset("height_m", data=height_arr)
        f.create_dataset("class_id", data=class_id_arr)
        f.create_dataset("style_id", data=style_id_arr)
        f.create_dataset("seed", data=seed_arr)
        f.create_dataset("shape_id", data=shape_arr)
        # Class label strings — separate dataset since vlen strings need attention.
        class_labels = [r["class_label"].encode("utf-8") for r in results]
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("class_label", data=class_labels, dtype=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/recipe_augmentation_v1")
    ap.add_argument("--per_style", type=int, default=6250,
                    help="Samples to generate per style (8 styles total).")
    ap.add_argument("--workers", type=int, default=16,
                    help="Process pool size. 0 = single-process (debug).")
    ap.add_argument("--seed_base", type=int, default=10_000_000,
                    help="Global RNG seed offset; each sample uses seed_base + i.")
    ap.add_argument("--styles", nargs="*", default=None,
                    help="Optional subset of styles to generate (default: all 8).")
    args = ap.parse_args()

    from scene.sdf_recipes import STYLES
    styles = args.styles if args.styles else list(STYLES)
    print(f"[recipe-aug] styles={styles} per_style={args.per_style} workers={args.workers}")
    print(f"[recipe-aug] total samples = {len(styles) * args.per_style:,}")

    # Build a stable class-id mapping using the styles in this run.
    class_labels = sorted({CLASS_FOR_STYLE[s] for s in styles})
    class_id_map = {c: i for i, c in enumerate(class_labels)}
    print(f"[recipe-aug] class_id_map={class_id_map}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist mapping next to the corpus.
    with open(out_dir / "class_id_map.csv", "w") as f:
        f.write("class_label,class_id\n")
        for c, i in class_id_map.items():
            f.write(f"{c},{i}\n")

    t0 = time.time()
    seed_cursor = args.seed_base
    for style_idx, style in enumerate(styles):
        out_path = out_dir / f"{style}.h5"
        if out_path.exists():
            print(f"[{style}] {out_path} exists; skipping")
            seed_cursor += args.per_style
            continue
        tasks = [
            (style, i, seed_cursor + i)
            for i in range(args.per_style)
        ]
        seed_cursor += args.per_style

        t_style = time.time()
        results: list[dict] = []
        if args.workers <= 0:
            for arg in tasks:
                results.append(_generate_one(arg))
                if len(results) % 50 == 0:
                    rate = len(results) / max(time.time() - t_style, 1e-6)
                    print(f"  [{style}] {len(results)}/{args.per_style} rate={rate:.2f}/s")
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers) as pool:
                for k, res in enumerate(pool.imap_unordered(_generate_one, tasks,
                                                            chunksize=4)):
                    results.append(res)
                    if (k + 1) % 200 == 0:
                        rate = (k + 1) / max(time.time() - t_style, 1e-6)
                        eta = (args.per_style - (k + 1)) / max(rate, 1e-6)
                        print(f"  [{style}] {k+1}/{args.per_style} "
                              f"rate={rate:.2f}/s eta={eta/60:.1f}min")

        results.sort(key=lambda r: r["sample_idx"])  # deterministic order
        _write_style_h5(out_path, results, class_id_map)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [{style}] -> {out_path} ({size_mb:.0f} MB, "
              f"{(time.time()-t_style)/60:.1f} min)")

    total_min = (time.time() - t0) / 60
    print(f"[recipe-aug] done in {total_min:.1f} min")


if __name__ == "__main__":
    main()
