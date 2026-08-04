"""Phase B+.4 — extract DiffRecipe parameter vectors from existing procedural samples.

The procedural generator (scripts/generate_recipe_augmentation.py) saved the
random seed for each sample. The recipe functions in scene/sdf_recipes.py are
deterministic given that seed. This script REPLAYS the random decisions for
each sample and maps them to the DiffRecipe parameter vector layout defined in
models/networks/diff_recipe.py.

Output: per-style .npz file with:
    params         (N, n_params_for_style)  float32  — the DiffRecipe param vector
    seed           (N,)                      int64   — for cross-ref with h5
    style_id       (N,)                      int32

These get consumed by Phase B+.5 (deterministic head sanity check) and
Phase B+.6 (recipe-param diffusion).

NOTE: the DiffRecipe forward with these extracted params will produce an SDF
~99% identical to the procedural version. The 1% difference comes from:
  - Soft-clamp replacing hard max() in DiffRecipe
  - Sigmoid blending replacing hard union for occupancy gates (mech, chimney)
This is intentional — the diffusion model only needs the param distribution,
not exact SDF replication.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from models.networks.diff_recipe import (
    DIFF_RECIPE_REGISTRY,
    MODERN_N_PARAMS, COLONIAL_N_PARAMS, VICTORIAN_N_PARAMS,
    INDUSTRIAL_N_PARAMS, CRAFTSMAN_N_PARAMS, MEDITERRANEAN_N_PARAMS,
    CONTEMPORARY_N_PARAMS, PUBLIC_CIVIC_N_PARAMS,
)


RECIPE_AUG_DIR = REPO / "data/recipe_augmentation_v1"
OUT_DIR = REPO / "data/recipe_augmentation_v1/extracted_params"


# Logit value to assign to "active" / "inactive" occupancy gates.
# sigmoid(5.0) ≈ 0.993 (effectively ON); sigmoid(-5.0) ≈ 0.007 (effectively OFF).
ON_LOGIT = 5.0
OFF_LOGIT = -5.0


def extract_modern(seed: int) -> np.ndarray:
    """Mirror recipe_modern's rng calls in order, then map to DiffRecipeModern params."""
    rng = np.random.default_rng(seed)
    mech_active = rng.random() < 0.6
    if mech_active:
        # off_x = (rng.random() - 0.5) * w * 0.35
        # diff convention: params[5] = (off_x / (w*0.35)) in [-0.5, 0.5]
        off_x = rng.random() - 0.5
        off_z = rng.random() - 0.5
    else:
        off_x = 0.0
        off_z = 0.0
    return np.array([
        0.05,                                    # 0 PARAPET_H_SCALE
        0.04,                                    # 1 PARAPET_INNER_SHRINK
        ON_LOGIT if mech_active else OFF_LOGIT,  # 2 MECH_ACTIVE_LOGIT
        0.18 if mech_active else 0.0,            # 3 MECH_W_RATIO
        0.07 if mech_active else 0.0,            # 4 MECH_H_RATIO
        off_x,                                   # 5 MECH_OFF_X
        off_z,                                   # 6 MECH_OFF_Z
        0.4,                                     # 7 MECH_Y_LIFT_RATIO
        0.2,                                     # 8 PARAPET_INNER_H_EXTRA
    ], dtype=np.float32)


def extract_colonial(seed: int) -> np.ndarray:
    """recipe_colonial: 1 rng call (chimney_active) + 1 conditional (offset)."""
    rng = np.random.default_rng(seed)
    chimney_active = rng.random() < 0.7
    if chimney_active:
        off_x = rng.random() - 0.5
    else:
        off_x = 0.0
    return np.array([
        0.45,                                       # 0 ROOF_H_RATIO
        ON_LOGIT if chimney_active else OFF_LOGIT,  # 1 CHIMNEY_ACTIVE_LOGIT
        0.07 if chimney_active else 0.0,            # 2 CHIMNEY_W_RATIO
        0.85,                                       # 3 CHIMNEY_H_FRAC
        off_x,                                      # 4 CHIMNEY_OFF_X
    ], dtype=np.float32)


def extract_victorian(seed: int) -> np.ndarray:
    """recipe_victorian: no rng calls — all params hardcoded."""
    # rng is seeded but not consumed
    return np.array([
        0.40,   # 0  ROOF_H_RATIO
        0.16,   # 1  TOWER_R_RATIO
        0.40,   # 2  TOWER_H_FRAC
        0.15,   # 3  TOWER_POS_X_RATIO
        0.15,   # 4  TOWER_POS_Z_RATIO
        0.80,   # 5  SPIRE_H_FRAC
        20.0,   # 6  SPIRE_ANGLE_DEG
        0.20,   # 7  BAY_W_RATIO
        0.55,   # 8  BAY_H_FRAC
        0.12,   # 9  BAY_D_RATIO
        -0.25,  # 10 BAY_OFF_X_RATIO
        0.4,    # 11 BAY_BLEND_K
    ], dtype=np.float32)


def extract_industrial(seed: int) -> np.ndarray:
    """recipe_industrial: no rng calls."""
    return np.array([
        0.30,   # 0 SLAB_H
        0.03,   # 1 EAVES_EXPAND_RATIO
        0.18,   # 2 EAVES_H
        0.05,   # 3 STACK_R_RATIO
        0.20,   # 4 STACK_H_FRAC
        0.18,   # 5 STACK_OFF_X (recipe has + w * 0.18 hard offset)
        0.05,   # 6 STACK_OFF_Z (recipe has + d * 0.05)
    ], dtype=np.float32)


def extract_craftsman(seed: int) -> np.ndarray:
    """recipe_craftsman: 1 rng call (porch_active)."""
    rng = np.random.default_rng(seed)
    porch_active = rng.random() < 0.5
    return np.array([
        0.20,                                      # 0 ROOF_H_RATIO
        0.03,                                      # 1 EAVES_EXPAND_RATIO
        ON_LOGIT if porch_active else OFF_LOGIT,   # 2 PORCH_ACTIVE_LOGIT
        0.55 if porch_active else 0.0,             # 3 PORCH_W_RATIO
        0.20 if porch_active else 0.0,             # 4 PORCH_D_RATIO
        0.03 if porch_active else 0.0,             # 5 PORCH_H_FRAC
    ], dtype=np.float32)


def extract_mediterranean(seed: int) -> np.ndarray:
    """recipe_mediterranean: no rng calls."""
    return np.array([
        0.14,   # 0 ROOF_H_RATIO
        0.04,   # 1 EAVES_EXPAND_RATIO
        0.25,   # 2 EDGE_BAND_H
    ], dtype=np.float32)


def extract_contemporary(seed: int) -> np.ndarray:
    """recipe_contemporary: 2 rng calls (off_x, off_z)."""
    rng = np.random.default_rng(seed)
    off_x = rng.random() - 0.5   # normalized to diff convention [-0.5, 0.5]
    off_z = rng.random() - 0.5
    return np.array([
        0.45,    # 0 UPPER_H_RATIO
        0.65,    # 1 UPPER_W_RATIO
        0.70,    # 2 UPPER_D_RATIO
        off_x,   # 3 UPPER_OFF_X
        off_z,   # 4 UPPER_OFF_Z
        0.40,    # 5 BLEND_K
    ], dtype=np.float32)


def extract_public_civic(seed: int) -> np.ndarray:
    """recipe_public_civic: no rng calls."""
    return np.array([
        0.28,   # 0 DOME_R_RATIO
        0.55,   # 1 DOME_Y_OFFSET_FRAC
        0.60,   # 2 DRUM_H_FRAC
        0.95,   # 3 DRUM_R_FRAC
        0.18,   # 4 FLANK_W_RATIO
        0.25,   # 5 FLANK_H_FRAC
        0.50,   # 6 FLANK_D_RATIO
        0.35,   # 7 FLANK_OFF_X_RATIO
    ], dtype=np.float32)


EXTRACTORS = {
    "modern":         extract_modern,
    "colonial":       extract_colonial,
    "victorian":      extract_victorian,
    "industrial":     extract_industrial,
    "craftsman":      extract_craftsman,
    "mediterranean":  extract_mediterranean,
    "contemporary":   extract_contemporary,
    "public_civic":   extract_public_civic,
}


def process_style(style: str, src_h5: Path, out_dir: Path) -> dict:
    extractor = EXTRACTORS[style]
    with h5py.File(src_h5, "r") as f:
        seeds = f["seed"][:]
        style_ids = f["style_id"][:]
        n = len(seeds)
    params = np.stack([extractor(int(s)) for s in seeds], axis=0)
    out_path = out_dir / f"{style}_params.npz"
    np.savez_compressed(out_path, params=params, seed=seeds,
                         style_id=style_ids)
    return {"style": style, "n": n, "params_shape": params.shape,
            "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=str(RECIPE_AUG_DIR))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--styles", default=None,
                    help="Comma-separated; default = all 8")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    styles = args.styles.split(",") if args.styles else list(EXTRACTORS.keys())
    results = []
    for style in styles:
        src = src_dir / f"{style}.h5"
        if not src.exists():
            print(f"  [SKIP] {style}: no h5 at {src}")
            continue
        r = process_style(style, src, out_dir)
        results.append(r)
        print(f"  {style:14s}  n={r['n']:6d}  params={r['params_shape']}  → {Path(r['out']).name}")

    total = sum(r["n"] for r in results)
    print(f"\nTotal samples extracted: {total}")


if __name__ == "__main__":
    main()
