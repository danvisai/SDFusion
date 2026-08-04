from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_h5_frame(data_root: Path, model_id: str, res: int) -> dict[str, np.ndarray]:
    h5_path = data_root / f"resolution_{res}" / model_id / "ori_sample_grid.h5"
    with h5py.File(h5_path, "r") as f:
        sdf = np.asarray(f["pc_sdf_sample"], dtype=np.float32).reshape(res, res, res)
        footprint = np.asarray(f["footprint"], dtype=np.uint8)[0] > 0
        sdf_params = np.asarray(f["sdf_params"], dtype=np.float32)
    return {
        "sdf": sdf,
        "footprint": footprint,
        "sdf_params": sdf_params,
    }


def world_bbox_from_mask(mask: np.ndarray, sdf_params: np.ndarray) -> dict[str, float]:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("empty footprint mask")
    res = mask.shape[0]
    xmin, ymin, zmin, xmax, ymax, zmax = sdf_params.tolist()
    return {
        "xmin": xmin + (float(cols.min()) / res) * (xmax - xmin),
        "xmax": xmin + (float(cols.max() + 1) / res) * (xmax - xmin),
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin + (float(rows.min()) / res) * (zmax - zmin),
        "zmax": zmin + (float(rows.max() + 1) / res) * (zmax - zmin),
    }


def world_bbox_from_sdf(sdf: np.ndarray, sdf_params: np.ndarray, iso: float) -> dict[str, float]:
    inside = sdf <= iso
    zz, yy, xx = np.where(inside)
    if len(xx) == 0:
        return world_bbox_from_mask(inside.any(axis=1), sdf_params)
    d, h, w = sdf.shape
    xmin, ymin, zmin, xmax, ymax, zmax = sdf_params.tolist()
    return {
        "xmin": xmin + (float(xx.min()) / max(w - 1, 1)) * (xmax - xmin),
        "xmax": xmin + (float(xx.max()) / max(w - 1, 1)) * (xmax - xmin),
        "ymin": ymin + (float(yy.min()) / max(h - 1, 1)) * (ymax - ymin),
        "ymax": ymin + (float(yy.max()) / max(h - 1, 1)) * (ymax - ymin),
        "zmin": zmin + (float(zz.min()) / max(d - 1, 1)) * (zmax - zmin),
        "zmax": zmin + (float(zz.max()) / max(d - 1, 1)) * (zmax - zmin),
    }


def center(bbox: dict[str, float], axis: str) -> float:
    return 0.5 * (bbox[f"{axis}min"] + bbox[f"{axis}max"])


def extent(bbox: dict[str, float], axis: str) -> float:
    return max(bbox[f"{axis}max"] - bbox[f"{axis}min"], 1e-8)


def alignment_transform(
    source_bbox: dict[str, float],
    target_fp_bbox: dict[str, float],
    target_sdf_bbox: dict[str, float],
    scale_mode: str,
) -> dict[str, float]:
    sx = extent(target_fp_bbox, "x") / extent(source_bbox, "x")
    sz = extent(target_fp_bbox, "z") / extent(source_bbox, "z")
    if scale_mode == "uniform":
        sx = sz = min(sx, sz)
    sy = float(np.sqrt(sx * sz))
    return {
        "scale_x": float(sx),
        "scale_y": float(sy),
        "scale_z": float(sz),
        "source_cx": float(center(source_bbox, "x")),
        "source_cz": float(center(source_bbox, "z")),
        "source_ymin": float(source_bbox["ymin"]),
        "target_cx": float(center(target_fp_bbox, "x")),
        "target_cz": float(center(target_fp_bbox, "z")),
        "target_ymin": float(target_sdf_bbox["ymin"]),
    }


def target_world_grid(sdf_params: np.ndarray, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, zmin, xmax, ymax, zmax = sdf_params.tolist()
    xs = np.linspace(xmin, xmax, res, dtype=np.float32)
    ys = np.linspace(ymin, ymax, res, dtype=np.float32)
    zs = np.linspace(zmin, zmax, res, dtype=np.float32)
    z_grid, y_grid, x_grid = np.meshgrid(zs, ys, xs, indexing="ij")
    return x_grid, y_grid, z_grid


def world_to_source_indices(
    x_world: np.ndarray,
    y_world: np.ndarray,
    z_world: np.ndarray,
    source_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, zmin, xmax, ymax, zmax = source_params.tolist()
    res = x_world.shape[0]
    ix = (x_world - xmin) / max(xmax - xmin, 1e-8) * (res - 1)
    iy = (y_world - ymin) / max(ymax - ymin, 1e-8) * (res - 1)
    iz = (z_world - zmin) / max(zmax - zmin, 1e-8) * (res - 1)
    return iz, iy, ix


def warp_source_sdf_to_target(
    source_sdf: np.ndarray,
    source_params: np.ndarray,
    target_params: np.ndarray,
    transform: dict[str, float],
) -> np.ndarray:
    res = source_sdf.shape[0]
    x_t, y_t, z_t = target_world_grid(target_params, res)
    x_s = (x_t - transform["target_cx"]) / transform["scale_x"] + transform["source_cx"]
    y_s = (y_t - transform["target_ymin"]) / transform["scale_y"] + transform["source_ymin"]
    z_s = (z_t - transform["target_cz"]) / transform["scale_z"] + transform["source_cz"]
    iz, iy, ix = world_to_source_indices(x_s, y_s, z_s, source_params)
    outside = float(np.max(source_sdf))
    warped = ndimage.map_coordinates(
        source_sdf,
        [iz, iy, ix],
        order=1,
        mode="constant",
        cval=outside,
        prefilter=False,
    )
    return warped.astype(np.float32)


def footprint_from_sdf(sdf: np.ndarray, iso: float) -> np.ndarray:
    return (sdf <= iso).any(axis=1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def filled_iou(a: np.ndarray, b: np.ndarray) -> float:
    return iou(ndimage.binary_fill_holes(a), ndimage.binary_fill_holes(b))


def nearest_neighbors(
    train: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    query_i: int,
    top_k: int,
    phase: str,
    same_top_only: bool,
) -> list[tuple[int, int, str, float]]:
    sims = query["embeddings"][query_i] @ train["embeddings"].T
    masked = sims.copy()
    if same_top_only:
        masked[train["top_ids"] != query["top_ids"][query_i]] = -1e9
    if phase == "train":
        masked[train["ids"] == query["ids"][query_i]] = -1e9
    nn = np.argsort(-masked)[:top_k]
    return [(rank, int(j), str(train["ids"][j]), float(masked[j])) for rank, j in enumerate(nn, 1)]


def save_pair(
    out_path: Path,
    source_aligned_sdf: np.ndarray,
    target_sdf: np.ndarray,
    source_footprint: np.ndarray,
    target_footprint: np.ndarray,
    transform: dict[str, float],
) -> None:
    residual = target_sdf - source_aligned_sdf
    np.savez_compressed(
        out_path,
        source_aligned_sdf=source_aligned_sdf.astype(np.float16),
        target_sdf=target_sdf.astype(np.float16),
        residual_sdf=residual.astype(np.float16),
        source_footprint=source_footprint.astype(np.uint8),
        target_footprint=target_footprint.astype(np.uint8),
        transform=np.array([transform[k] for k in sorted(transform)], dtype=np.float32),
        transform_keys=np.array(sorted(transform), dtype=object),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--index_dir", default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--out_dir", default="data/BuildingNet_dataset_v0_1/correction_pairs")
    ap.add_argument("--phase", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 means all queries")
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--iso", type=float, default=0.0)
    ap.add_argument("--scale_mode", default="anisotropic", choices=["anisotropic", "uniform"])
    ap.add_argument("--same_top_only", action="store_true", default=True)
    ap.add_argument("--metadata_only", action="store_true",
                    help="write CSV only; do not save per-pair SDF arrays")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    pair_dir = out_dir / args.phase / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    train = load_npz(index_dir / "train_embeddings.npz")
    query = load_npz(index_dir / f"{args.phase}_embeddings.npz")

    n_query = len(query["ids"]) if args.limit <= 0 else min(args.limit, len(query["ids"]))
    rows = []
    frame_cache: dict[str, dict[str, np.ndarray]] = {}

    def get_frame(model_id: str) -> dict[str, np.ndarray]:
        if model_id not in frame_cache:
            frame_cache[model_id] = load_h5_frame(data_root, model_id, args.res)
        return frame_cache[model_id]

    for i in range(n_query):
        query_id = str(query["ids"][i])
        q_frame = get_frame(query_id)
        target_fp_bbox = world_bbox_from_mask(q_frame["footprint"], q_frame["sdf_params"])
        target_sdf_bbox = world_bbox_from_sdf(q_frame["sdf"], q_frame["sdf_params"], args.iso)
        neighbors = nearest_neighbors(
            train, query, i, args.top_k, args.phase, args.same_top_only
        )

        for rank, source_index, source_id, sim in neighbors:
            s_frame = get_frame(source_id)
            source_bbox = world_bbox_from_sdf(s_frame["sdf"], s_frame["sdf_params"], args.iso)
            transform = alignment_transform(
                source_bbox,
                target_fp_bbox,
                target_sdf_bbox,
                args.scale_mode,
            )
            aligned_sdf = warp_source_sdf_to_target(
                s_frame["sdf"],
                s_frame["sdf_params"],
                q_frame["sdf_params"],
                transform,
            )
            aligned_fp = footprint_from_sdf(aligned_sdf, args.iso)
            raw_score = iou(q_frame["footprint"], aligned_fp)
            filled_score = filled_iou(q_frame["footprint"], aligned_fp)
            residual = q_frame["sdf"] - aligned_sdf
            pair_name = f"{query_id}__rank{rank:02d}__{source_id}.npz"
            pair_path = pair_dir / pair_name

            if not args.metadata_only:
                save_pair(
                    pair_path,
                    aligned_sdf,
                    q_frame["sdf"],
                    aligned_fp,
                    q_frame["footprint"],
                    transform,
                )

            rows.append({
                "phase": args.phase,
                "query_id": query_id,
                "rank": rank,
                "source_id": source_id,
                "source_index": source_index,
                "similarity": f"{sim:.6f}",
                "footprint_iou": f"{raw_score:.6f}",
                "filled_footprint_iou": f"{filled_score:.6f}",
                "residual_l1": f"{float(np.mean(np.abs(residual))):.6f}",
                "residual_l2": f"{float(np.sqrt(np.mean(residual ** 2))):.6f}",
                "scale_x": f"{transform['scale_x']:.6f}",
                "scale_y": f"{transform['scale_y']:.6f}",
                "scale_z": f"{transform['scale_z']:.6f}",
                "pair_path": "" if args.metadata_only else str(pair_path),
            })
        print(f"[{i + 1:04d}/{n_query:04d}] {query_id} -> {len(neighbors)} pairs", flush=True)

    csv_path = out_dir / args.phase / "pair_metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "phase", "query_id", "rank", "source_id", "source_index", "similarity",
            "footprint_iou", "filled_footprint_iou", "residual_l1", "residual_l2",
            "scale_x", "scale_y", "scale_z", "pair_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")
    if not args.metadata_only:
        print(f"wrote pair arrays under {pair_dir}")


if __name__ == "__main__":
    main()
