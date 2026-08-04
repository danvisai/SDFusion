"""scene/run_demo.py — Step 5 MVP: F0 -> F1 -> F6 -> F7.

Synthesizes 6 polygons in a local meter frame, retrieves the nearest BuildingNet
OBJ for each (top-level class-filtered), places it at world coords with a
uniform XZ scale to fit the polygon and Y scale to match target height, and
concatenates everything into one OBJ.

Skips the residual stage: the retrieved OBJs already carry full architectural
detail. Residual integration is additive and can plug into the same place_mesh
helper later.

Run:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scene/run_demo.py --out outputs/demo_town.obj
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks.retrieval.footprint_embed import FootprintEmbedNet
from models.networks.sdf_residual_net import SDFResidualUNet
from scripts.build_correction_pairs import (
    alignment_transform,
    warp_source_sdf_to_target,
    world_bbox_from_sdf,
    world_bbox_from_mask,
)


_TOPLEVELS = ("RESIDENTIAL", "RELIGIOUS", "COMMERCIAL", "MILITARY", "PUBLIC")


def synthetic_scene():
    """Six polygons on a 100m x 100m site (XZ ground plane, meters)."""
    return [
        {"id": "B0", "polygon": [(10, 10), (30, 10), (30, 30), (10, 30)],
         "class": "RESIDENTIALhouse", "height": 8.0},
        {"id": "B1", "polygon": [(40, 10), (60, 10), (60, 25), (40, 25)],
         "class": "RESIDENTIALhouse", "height": 8.0},
        {"id": "B2", "polygon": [(70, 10), (90, 10), (90, 35), (70, 35)],
         "class": "COMMERCIALoffice_building", "height": 18.0},
        {"id": "B3", "polygon": [(10, 50), (35, 50), (35, 80), (10, 80)],
         "class": "RELIGIOUSchurch", "height": 14.0},
        {"id": "B4", "polygon": [(45, 55), (60, 55), (60, 75), (45, 75)],
         "class": "RESIDENTIALhouse", "height": 8.0},
        {"id": "B5", "polygon": [(70, 50), (95, 50), (95, 90), (70, 90)],
         "class": "PUBLICcity_hall", "height": 16.0},
    ]


def rasterize_polygon(polygon_xz, res: int = 64) -> np.ndarray:
    """Polygon (list of (x,z) verts) -> res x res float mask in [0, 1].

    Polygon's bbox is normalized to fill the mask with 1px margin, preserving
    aspect ratio.
    """
    poly = np.asarray(polygon_xz, dtype=np.float64)
    xmin, zmin = poly.min(axis=0)
    xmax, zmax = poly.max(axis=0)
    w = max(xmax - xmin, 1e-6)
    h = max(zmax - zmin, 1e-6)
    s = (res - 2) / max(w, h)
    ox = (res - w * s) / 2.0
    oz = (res - h * s) / 2.0
    pixels = [(ox + (x - xmin) * s, oz + (z - zmin) * s) for x, z in poly]
    img = Image.new("L", (res, res), 0)
    ImageDraw.Draw(img).polygon(pixels, fill=255)
    return (np.asarray(img) > 0).astype(np.float32)


def load_retrieval(index_dir: Path, ckpt_path: str, device: torch.device):
    train = np.load(index_dir / "train_embeddings.npz", allow_pickle=True)
    meta = json.load(open(index_dir / "metadata.json"))
    subtype_to_idx = {k: int(v) for k, v in meta["subtype_to_idx"].items()}
    top_to_idx = {k: int(v) for k, v in meta["top_to_idx"].items()}
    model = FootprintEmbedNet(num_classes=len(subtype_to_idx)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "train_emb": train["embeddings"],
        "train_ids": train["ids"],
        "train_top_ids": train["top_ids"],
        "subtype_to_idx": subtype_to_idx,
        "top_to_idx": top_to_idx,
    }


def _resolve_top(full_class: str) -> str:
    for tl in _TOPLEVELS:
        if full_class.startswith(tl):
            return tl
    return "RESIDENTIAL"


def _resolve_subtype_idx(full_class: str, subtype_to_idx: dict[str, int]) -> int:
    if full_class in subtype_to_idx:
        return subtype_to_idx[full_class]
    top = _resolve_top(full_class)
    # Fallback: any subtype with the same top-level prefix
    for k, v in subtype_to_idx.items():
        if k.startswith(top):
            return v
    return 0


@torch.no_grad()
def retrieve(building: dict, fp_mask: np.ndarray, R: dict, device) -> str:
    full_class = building["class"]
    sub_idx = _resolve_subtype_idx(full_class, R["subtype_to_idx"])
    top = _resolve_top(full_class)
    top_idx = R["top_to_idx"].get(top, 0)
    fp = torch.from_numpy(fp_mask)[None, None].to(device).float()
    cls = torch.tensor([sub_idx], dtype=torch.long, device=device)
    emb, _ = R["model"](fp, cls)
    emb = emb.cpu().numpy()[0]
    sims = R["train_emb"] @ emb
    same_top = R["train_top_ids"] == top_idx
    sims = np.where(same_top, sims, -1e9)
    j = int(np.argmax(sims))
    return str(R["train_ids"][j])


def load_residual_model(ckpt_path: str, device: torch.device,
                        residual_clip: float = 1.0,
                        base_channels: int = 16):
    model = SDFResidualUNet(in_channels=2, base_channels=base_channels,
                            residual_clip=residual_clip).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_source_h5(model_id: str, data_root: Path, res: int = 64):
    h5_path = data_root / f"resolution_{res}" / model_id / "ori_sample_grid.h5"
    with h5py.File(h5_path, "r") as f:
        sdf = np.asarray(f["pc_sdf_sample"], dtype=np.float32).reshape(res, res, res)
        sdf_params = np.asarray(f["sdf_params"], dtype=np.float32)
    return sdf, sdf_params


def align_source_to_polygon(source_sdf: np.ndarray, source_params: np.ndarray,
                            polygon_mask: np.ndarray) -> np.ndarray:
    """Warp source SDF to a synthetic unit-cube target frame whose footprint
    bbox matches the input polygon mask's bbox."""
    target_params = np.array([-1, -1, -1, 1, 1, 1], dtype=np.float32)
    src_bbox = world_bbox_from_sdf(source_sdf, source_params, iso=0.0)
    tgt_fp_bbox = world_bbox_from_mask(polygon_mask > 0, target_params)
    tgt_sdf_bbox = {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0,
                    "zmin": -1.0, "zmax": 1.0}
    tx = alignment_transform(src_bbox, tgt_fp_bbox, tgt_sdf_bbox, "anisotropic")
    return warp_source_sdf_to_target(source_sdf, source_params, target_params, tx)


@torch.no_grad()
def predict_corrected_sdf(model, aligned_source: np.ndarray,
                          polygon_mask: np.ndarray, device) -> np.ndarray:
    """Build the (2, D, H, W) input tensor and apply the residual model."""
    fp = polygon_mask.astype(np.float32)
    fp_vol = np.repeat(fp[:, None, :], aligned_source.shape[1], axis=1)
    x = np.stack([aligned_source, fp_vol], axis=0)
    x_t = torch.from_numpy(x)[None].to(device)
    src_t = torch.from_numpy(aligned_source[None, None]).to(device)
    pred = model(x_t)
    corrected = (src_t + pred)[0, 0].cpu().numpy()
    return corrected.astype(np.float32)


def mesh_from_corrected(corrected_sdf: np.ndarray,
                        keep_largest: bool = True) -> trimesh.Trimesh | None:
    """Marching cubes at iso=0 in (z,y,x), output mesh in [-1,1]^3 as (x,y,z).

    Optionally keeps only the largest connected component.
    """
    if (corrected_sdf <= 0).sum() < 8:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(corrected_sdf, level=0.0)
    except (ValueError, RuntimeError):
        return None
    res = corrected_sdf.shape[0]
    # verts come out as (z, y, x) voxel indices. Convert to (x, y, z) in [-1,1].
    verts = verts / max(res - 1, 1) * 2.0 - 1.0
    verts = verts[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if not keep_largest:
        return mesh
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh
    return max(parts, key=lambda m: len(m.vertices))


def place_mesh(mesh: trimesh.Trimesh, polygon_xz, target_height_m: float,
               ground_y: float = 0.0, aspect_preserve: bool = False) -> trimesh.Trimesh:
    """Frame-N mesh -> Frame-W polygon position.

    Default (aspect_preserve=False): non-uniform scale [s_xz, s_y, s_xz] that
    forces the mesh's XZ extent to fill the polygon and its Y extent to match
    target_height_m. Can flatten buildings whose natural height/XZ ratio
    doesn't match the OSM defaults (the "placed_flat" failure mode).

    With aspect_preserve=True: pick a single uniform scale s = min(s_xz, s_y)
    so the mesh keeps its native proportions. The building may under-fill the
    polygon OR be shorter than target_height_m, but it won't be squashed.

    XZ centroid translates to polygon centroid; base lifts to ground_y.
    """
    m = mesh.copy()
    poly = np.asarray(polygon_xz, dtype=np.float64)
    px_min, pz_min = poly.min(axis=0)
    px_max, pz_max = poly.max(axis=0)
    pw, pd = px_max - px_min, pz_max - pz_min
    pcx, pcz = (px_min + px_max) / 2.0, (pz_min + pz_max) / 2.0

    bb = m.bounds
    ext = bb[1] - bb[0]
    s_xz = max(pw, pd) / max(max(ext[0], ext[2]), 1e-6)
    s_y = target_height_m / max(ext[1], 1e-6)
    if aspect_preserve:
        s = min(s_xz, s_y)
        m.apply_scale([s, s, s])
    else:
        m.apply_scale([s_xz, s_y, s_xz])

    bb = m.bounds
    cx = (bb[0][0] + bb[1][0]) / 2.0
    cz = (bb[0][2] + bb[1][2]) / 2.0
    dy = ground_y - bb[0][1]
    m.apply_translation([pcx - cx, dy, pcz - cz])
    return m


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir",
                    default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--retrieval_ckpt",
                    default="Logs_GT/retrieval_footprint_full/ckpt_best.pth")
    ap.add_argument("--obj_dir",
                    default="data/BuildingNet_dataset_v0_1/OBJ_MODELS")
    ap.add_argument("--out", default="outputs/demo_town.obj")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_residual", action="store_true",
                    help="Use corrected SDF + marching cubes instead of raw OBJ.")
    ap.add_argument("--residual_ckpt",
                    default="Logs_GT/sdf_residual_full_v4_aug_topk3/ckpt_best_geom.pth")
    ap.add_argument("--data_root",
                    default="data/BuildingNet_dataset_v0_1")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    R = load_retrieval(Path(args.index_dir), args.retrieval_ckpt, device)
    res_model = None
    if args.use_residual:
        res_model = load_residual_model(args.residual_ckpt, device)
    data_root = Path(args.data_root)

    placed = []
    log = []
    for b in synthetic_scene():
        fp = rasterize_polygon(b["polygon"])
        retrieved_id = retrieve(b, fp, R, device)
        if args.use_residual:
            try:
                src_sdf, src_params = load_source_h5(retrieved_id, data_root)
            except FileNotFoundError as e:
                print(f"  MISSING h5 for {retrieved_id}: {e}")
                continue
            aligned = align_source_to_polygon(src_sdf, src_params, fp)
            corrected = predict_corrected_sdf(res_model, aligned, fp, device)
            mesh = mesh_from_corrected(corrected, keep_largest=True)
            if mesh is None:
                print(f"  MC FAILED for {retrieved_id}")
                continue
            source_label = "corrected_sdf"
        else:
            obj_path = Path(args.obj_dir) / f"{retrieved_id}.obj"
            if not obj_path.exists():
                print(f"  MISSING: {obj_path}")
                continue
            mesh = trimesh.load(obj_path, force="mesh")
            source_label = "obj"

        placed_mesh = place_mesh(mesh, b["polygon"], b["height"])
        placed.append(placed_mesh)
        log.append({"id": b["id"], "class": b["class"],
                    "retrieved": retrieved_id, "source": source_label,
                    "verts": int(len(placed_mesh.vertices)),
                    "faces": int(len(placed_mesh.faces))})
        print(f"  {b['id']:3s} {b['class']:30s} <- {retrieved_id}  [{source_label}]")

    if not placed:
        sys.exit("No meshes were placed.")
    composed = trimesh.util.concatenate(placed)
    composed.export(out_path)
    json.dump(log, open(out_path.with_suffix(".log.json"), "w"), indent=2)
    print(f"\nwrote {out_path}  "
          f"({len(composed.vertices)} verts, {len(composed.faces)} faces)")


if __name__ == "__main__":
    main()
