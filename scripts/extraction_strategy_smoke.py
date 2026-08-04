"""
Compare output strategies for the BuildingNet SDF/OBJ pipeline.

This is a read-only smoke test over existing data. For each model id it writes:

  - original normalized OBJ
  - marching-cubes mesh from signed SDF at level 0
  - UDF-style mesh from abs(SDF) at a small positive level
  - optional ARAP-deformed OBJ guided by the SDF
  - footprint/silhouette contact sheet and metrics CSV

The goal is to decide the output path before training retrieval/correction.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_IDS = [
    "RESIDENTIALhouse_mesh8443",
    "RESIDENTIALhouse_mesh4208",
    "RESIDENTIALvilla_mesh3202",
    "RESIDENTIALvilla_mesh5927",
]


def load_h5(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        sdf = np.asarray(f["pc_sdf_sample"], dtype=np.float32).reshape(64, 64, 64)
        footprint = np.asarray(f["footprint"], dtype=np.uint8)[0] > 0
        sdf_params = np.asarray(f["sdf_params"], dtype=np.float32)
        norm_params = np.asarray(f["norm_params"], dtype=np.float32)
    return sdf, footprint, sdf_params, norm_params


def load_obj_normalized(obj_path: Path, norm_params: np.ndarray) -> trimesh.Trimesh:
    loaded = trimesh.load(obj_path, force="mesh", process=False)
    if loaded is None or not hasattr(loaded, "vertices"):
        raise ValueError(f"empty or invalid OBJ: {obj_path}")
    mesh = loaded.copy()
    centroid = norm_params[:3]
    scale = float(norm_params[3])
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - centroid) / scale
    return mesh


def mesh_from_grid(grid: np.ndarray, sdf_params: np.ndarray, level: float) -> trimesh.Trimesh | None:
    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(grid, level=level)
    except Exception as exc:
        print(f"  [mc-fail] level={level}: {exc}")
        return None

    if len(verts_zyx) == 0 or len(faces) == 0:
        return None

    d, h, w = grid.shape
    xmin, ymin, zmin, xmax, ymax, zmax = sdf_params.tolist()
    verts = np.empty((len(verts_zyx), 3), dtype=np.float32)
    verts[:, 0] = xmin + (verts_zyx[:, 2] / max(w - 1, 1)) * (xmax - xmin)
    verts[:, 1] = ymin + (verts_zyx[:, 1] / max(h - 1, 1)) * (ymax - ymin)
    verts[:, 2] = zmin + (verts_zyx[:, 0] / max(d - 1, 1)) * (zmax - zmin)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.process(validate=False)
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        try:
            mesh.update_faces(mesh.nondegenerate_faces())
        except Exception:
            pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)


def mesh_from_meshudf(abs_sdf: np.ndarray, sdf_params: np.ndarray) -> tuple[trimesh.Trimesh | None, str]:
    meshudf_dir = REPO_ROOT / "external" / "MeshUDF" / "custom_mc"
    if not (meshudf_dir / "_marching_cubes_lewiner.py").exists():
        return None, f"missing MeshUDF at {meshudf_dir}"
    if str(meshudf_dir) not in sys.path:
        sys.path.insert(0, str(meshudf_dir))

    try:
        from _marching_cubes_lewiner import udf_mc_lewiner
    except Exception as exc:
        return None, f"import failed: {type(exc).__name__}: {exc}"

    abs_sdf = np.ascontiguousarray(abs_sdf.astype(np.float32))
    grad_zyx = np.stack(np.gradient(abs_sdf), axis=-1).astype(np.float32)
    # MeshUDF expects gradients in the same grid coordinate basis as its volume.
    # It returns vertices in z-y-x grid order after its own internal flip.
    try:
        verts_zyx, faces, _, _ = udf_mc_lewiner(abs_sdf, grad_zyx, spacing=(1.0, 1.0, 1.0))
    except Exception as exc:
        return None, f"udf_mc failed: {type(exc).__name__}: {exc}"

    if len(verts_zyx) == 0 or len(faces) == 0:
        return None, "empty"

    d, h, w = abs_sdf.shape
    xmin, ymin, zmin, xmax, ymax, zmax = sdf_params.tolist()
    verts = np.empty((len(verts_zyx), 3), dtype=np.float32)
    verts[:, 0] = xmin + (verts_zyx[:, 2] / max(w - 1, 1)) * (xmax - xmin)
    verts[:, 1] = ymin + (verts_zyx[:, 1] / max(h - 1, 1)) * (ymax - ymin)
    verts[:, 2] = zmin + (verts_zyx[:, 0] / max(d - 1, 1)) * (zmax - zmin)
    return clean_mesh(trimesh.Trimesh(vertices=verts, faces=faces, process=False)), "ok"


def mesh_components(mesh: trimesh.Trimesh | None) -> tuple[int, int, int, float]:
    if mesh is None or len(mesh.faces) == 0:
        return 0, 0, 0, 0.0
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        comps = []
    if not comps:
        return len(mesh.vertices), len(mesh.faces), 1, 1.0
    largest = max(len(c.faces) for c in comps)
    frac = largest / max(len(mesh.faces), 1)
    return len(mesh.vertices), len(mesh.faces), len(comps), float(frac)


def footprint_from_mesh(
    mesh: trimesh.Trimesh | None,
    sdf_params: np.ndarray,
    res: int = 64,
    samples: int = 120_000,
) -> np.ndarray:
    mask = np.zeros((res, res), dtype=bool)
    if mesh is None or len(mesh.vertices) == 0:
        return mask

    if len(mesh.faces) > 0:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, min(samples, max(5000, len(mesh.faces) * 8)))
        except Exception:
            pts = np.asarray(mesh.vertices)
    else:
        pts = np.asarray(mesh.vertices)

    xmin, _, zmin, xmax, _, zmax = sdf_params.tolist()
    x = pts[:, 0]
    z = pts[:, 2]
    col = np.floor((x - xmin) / max(xmax - xmin, 1e-9) * res).astype(np.int32)
    row = np.floor((z - zmin) / max(zmax - zmin, 1e-9) * res).astype(np.int32)
    ok = (row >= 0) & (row < res) & (col >= 0) & (col < res)
    mask[row[ok], col[ok]] = True
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def mask_image(mask: np.ndarray, title: str, scale: int = 4) -> Image.Image:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
        (mask.shape[1] * scale, mask.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + 24), "white")
    canvas.paste(img, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), title, fill=(0, 0, 0))
    return canvas


def save_contact_sheet(mid: str, masks: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    cells = [mask_image(mask, title) for title, mask in masks]
    w = max(c.width for c in cells)
    h = max(c.height for c in cells)
    sheet = Image.new("RGB", (w * len(cells), h), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, (i * w, 0))
    ImageDraw.Draw(sheet).text((4, h - 18), mid, fill=(200, 0, 0))
    sheet.save(out_path)


def export_mesh(mesh: trimesh.Trimesh | None, path: Path) -> None:
    if mesh is not None and len(mesh.vertices) and len(mesh.faces):
        mesh.export(path)


def maybe_arap(
    obj_mesh: trimesh.Trimesh,
    sdf: np.ndarray,
    max_faces: int,
) -> tuple[trimesh.Trimesh | None, str, float]:
    if len(obj_mesh.faces) > max_faces:
        return None, f"skipped: {len(obj_mesh.faces)} faces > max {max_faces}", 0.0
    try:
        from models.arap_deformer import arap_deform

        start = time.time()
        deformed = arap_deform(
            obj_mesh,
            sdf,
            anchor_threshold=2.0 / 64,
            max_displacement=0.3,
            n_iters=5,
            max_anchors=3000,
            project_iters=3,
            normalize=False,
        )
        return deformed, "ok", time.time() - start
    except Exception as exc:
        return None, f"failed: {type(exc).__name__}: {exc}", 0.0


def run_one(mid: str, args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) / mid
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_root / "resolution_64" / mid / "ori_sample_grid.h5"
    obj_path = data_root / "OBJ_MODELS" / f"{mid}.obj"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    sdf, fp_gt, sdf_params, norm_params = load_h5(h5_path)
    obj_mesh = load_obj_normalized(obj_path, norm_params)
    if args.export_normalized_obj:
        obj_mesh.export(out_dir / f"{mid}_obj_norm.obj")

    abs_sdf = np.abs(sdf)
    mc_mesh = mesh_from_grid(sdf, sdf_params, level=args.sdf_level)
    udf_mesh = mesh_from_grid(abs_sdf, sdf_params, level=args.udf_level)
    meshudf_mesh, meshudf_status = mesh_from_meshudf(abs_sdf, sdf_params)
    arap_mesh, arap_status, arap_seconds = maybe_arap(obj_mesh, sdf, args.max_arap_faces)

    export_mesh(mc_mesh, out_dir / f"{mid}_mc_sdf.obj")
    export_mesh(udf_mesh, out_dir / f"{mid}_udf_abs.obj")
    export_mesh(meshudf_mesh, out_dir / f"{mid}_meshudf.obj")
    export_mesh(arap_mesh, out_dir / f"{mid}_arap.obj")

    masks = {
        "gt_fp": fp_gt,
        "obj": footprint_from_mesh(obj_mesh, sdf_params),
        "mc_sdf": footprint_from_mesh(mc_mesh, sdf_params),
        "udf_abs": footprint_from_mesh(udf_mesh, sdf_params),
        "meshudf": footprint_from_mesh(meshudf_mesh, sdf_params),
        "arap": footprint_from_mesh(arap_mesh, sdf_params),
    }
    save_contact_sheet(mid, list(masks.items()), out_dir / f"{mid}_contact.png")

    row: dict[str, object] = {
        "id": mid,
        "inside_pct": float((sdf <= 0).mean() * 100),
        "obj_faces": int(len(obj_mesh.faces)),
        "meshudf_status": meshudf_status,
        "arap_status": arap_status,
        "arap_seconds": round(arap_seconds, 3),
    }
    for name, mesh in [("mc_sdf", mc_mesh), ("udf_abs", udf_mesh), ("meshudf", meshudf_mesh), ("arap", arap_mesh)]:
        verts, faces, comps, largest_frac = mesh_components(mesh)
        row[f"{name}_verts"] = verts
        row[f"{name}_faces"] = faces
        row[f"{name}_components"] = comps
        row[f"{name}_largest_face_frac"] = round(largest_frac, 4)
        row[f"{name}_fp_iou"] = round(iou(fp_gt, masks[name]), 4)

    print(
        f"  {mid}: inside={row['inside_pct']:.3f}% "
        f"mc comps={row['mc_sdf_components']} iou={row['mc_sdf_fp_iou']} "
        f"udf comps={row['udf_abs_components']} iou={row['udf_abs_fp_iou']} "
        f"meshudf={meshudf_status} comps={row['meshudf_components']} iou={row['meshudf_fp_iou']} "
        f"arap={arap_status}"
    )
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--out_dir", default="outputs/extraction_smoke")
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--sdf_level", type=float, default=0.0)
    ap.add_argument("--udf_level", type=float, default=0.03)
    ap.add_argument("--max_arap_faces", type=int, default=20_000)
    ap.add_argument("--export_normalized_obj", action="store_true",
                    help="also export normalized source OBJ; can be slow for huge meshes")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mid in args.ids:
        try:
            rows.append(run_one(mid, args))
        except Exception as exc:
            print(f"  [fail] {mid}: {type(exc).__name__}: {exc}")
            rows.append({"id": mid, "error": f"{type(exc).__name__}: {exc}"})

    csv_path = out_dir / "metrics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"wrote {csv_path}")
    print(f"artifacts under {out_dir}/<model_id>/")


if __name__ == "__main__":
    main()
