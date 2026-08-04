"""Simplify Hunyuan GLB meshes for scene assembly.

Hunyuan3D meshes are visually useful but usually too dense for town-scale
composition. This script uses pymeshlab quadric decimation and writes a metrics
CSV so we can compare source and simplified face counts.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/simplify_hunyuan_meshes.py \
        --input_dir outputs/hunyuan_retrieval_rank1_mini \
        --out_dir outputs/hunyuan_retrieval_rank1_mini_simplified \
        --target_faces 50000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pymeshlab


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="outputs/hunyuan_retrieval_rank1_mini")
    ap.add_argument("--out_dir", default="outputs/hunyuan_retrieval_rank1_mini_simplified")
    ap.add_argument("--pattern", default="*.glb")
    ap.add_argument("--out_ext", default=".obj",
                    help="Output extension supported by pymeshlab, e.g. .obj or .ply.")
    ap.add_argument("--target_faces", type=int, default=50_000)
    ap.add_argument("--min_faces", type=int, default=0,
                    help="Skip decimation when a mesh already has this many faces or fewer. 0 uses target_faces.")
    return ap.parse_args()


def simplify_one(src: Path, dst: Path, target_faces: int, min_faces: int) -> dict[str, object]:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    before_v = int(ms.current_mesh().vertex_number())
    before_f = int(ms.current_mesh().face_number())
    threshold = min_faces if min_faces > 0 else target_faces
    simplified = before_f > threshold
    if simplified:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservenormal=True,
            preserveboundary=True,
            optimalplacement=True,
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(dst))
    after_v = int(ms.current_mesh().vertex_number())
    after_f = int(ms.current_mesh().face_number())
    return {
        "input": str(src),
        "output": str(dst),
        "simplified": int(simplified),
        "verts_before": before_v,
        "faces_before": before_f,
        "verts_after": after_v,
        "faces_after": after_f,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meshes = sorted(input_dir.glob(args.pattern))
    if not meshes:
        raise SystemExit(f"No meshes found: {input_dir}/{args.pattern}")

    rows = []
    for src in meshes:
        dst = out_dir / f"{src.stem}{args.out_ext}"
        row = simplify_one(src, dst, args.target_faces, args.min_faces)
        rows.append(row)
        print(
            f"{src.name}: {row['faces_before']:,} -> {row['faces_after']:,} faces "
            f"{'(kept)' if not row['simplified'] else ''}",
            flush=True,
        )

    metrics_path = out_dir / "simplify_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
