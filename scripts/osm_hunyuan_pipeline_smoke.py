"""Run an OSM -> retrieval -> Hunyuan -> placed town smoke pipeline.

This is a deliberately small end-to-end test. It consumes the JSON emitted by
scene/extract_osm.py, retrieves BuildingNet exemplars for the largest OSM
footprints, renders those exemplars as image inputs, generates Hunyuan meshes,
simplifies them, places them on the original OSM polygons, and writes a contact
sheet plus CSV metrics.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/osm_hunyuan_pipeline_smoke.py \
        --osm_json outputs/osm_pipeline_smoke/osm_input.json \
        --out_dir outputs/osm_pipeline_smoke --limit 4
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import (
    _resolve_subtype_idx,
    _resolve_top,
    load_retrieval,
    place_mesh,
    rasterize_polygon,
)
from scene.gsplat_common import load_inria_ply, save_inria_ply
from scene.gsplat_compose import compose as gsplat_compose
from scene.gsplat_guardrail import cull_outside_footprint
from scene.gsplat_placement import place_gsplat
from scene.gsplat_renderer import render_gsplat_view
from scene.sdf_primitives import grid_to_mesh, polygon_bbox_with_pad, sample_grid
from scene.sdf_recipes import STYLES as SDF_STYLES, build_styled_sdf
from scene.sdf_vqvae_prior import load_buildingnet_vqvae, procedural_to_mesh_via_vqvae
from scripts.hunyuan_building_mesh_smoke import load_pipeline, render_mesh_png, title_cell
from scripts.osm_candidate_quality_features import building_geometry_features, candidate_quality_features
from scripts.osm_footprint_proposal_images import proposal_image
from scripts.osm_recompose_height_policy import infer_height as infer_height_from_policy
from scripts.render_buildingnet_objfiles import load_obj_as_trimesh, make_renderer, render_one
from scripts.simplify_hunyuan_meshes import simplify_one
from scripts.train_osm_proposal_image_generator import CLASS_COUNT, ProposalUNet


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_pipeline_smoke")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--model", choices=["mini", "full"], default="mini")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--octree_resolution", type=int, default=380)
    ap.add_argument("--num_chunks", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260510)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--target_faces", type=int, default=50_000)
    ap.add_argument("--retrieval_top_k", type=int, default=5)
    ap.add_argument("--retrieval_policy", choices=["top1", "rerank", "quality"], default="rerank")
    ap.add_argument(
        "--conditioning_source",
        choices=["retrieved", "proposal", "learned_proposal", "image_prior"],
        default="retrieved",
        help="Image input for Hunyuan: retrieved OBJ render, procedural proposal, learned proposal, or pretrained image-prior proposal.",
    )
    ap.add_argument("--proposal_detail", choices=["clean", "detailed"], default="clean")
    ap.add_argument("--proposal_footprint_inset", action="store_true")
    ap.add_argument("--learned_proposal_ckpt", help="Checkpoint from train_osm_proposal_image_generator.py")
    ap.add_argument("--image_prior_manifest", help="CSV from generate_osm_image_prior_proposals.py")
    ap.add_argument("--aspect_weight", type=float, default=0.08)
    ap.add_argument("--height_weight", type=float, default=0.02)
    ap.add_argument("--quality_model", help="Optional generation success model .pkl for quality-aware reranking")
    ap.add_argument("--quality_weight", type=float, default=0.20)
    ap.add_argument("--quality_bad_candidate_penalty", type=float, default=1.0)
    ap.add_argument("--index_dir", default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--retrieval_ckpt", default="Logs_GT/retrieval_footprint_full/ckpt_best.pth")
    ap.add_argument("--obj_dir", default="data/BuildingNet_dataset_v0_1/OBJ_MODELS")
    ap.add_argument(
        "--asset_format", choices=["mesh", "gsplat", "sdf_procedural"], default="mesh",
        help=(
            "mesh = current Hunyuan path. "
            "gsplat = skip Hunyuan, use pre-baked 3DGS PLYs from --gsplat_dir. "
            "sdf_procedural = skip retrieval+Hunyuan, generate the building from "
            "scene/sdf_recipes.py given --sdf_style (footprint is preserved exactly)."
        ),
    )
    ap.add_argument(
        "--gsplat_dir",
        default="data/BuildingNet_dataset_v0_1/gaussian_splats",
        help="Directory containing <retrieved_id>.ply baked 3DGS assets (used when --asset_format=gsplat).",
    )
    ap.add_argument(
        "--gsplat_cull", action="store_true",
        help="Apply footprint-mask Gaussian culling at placement (SDF/footprint guardrail).",
    )
    ap.add_argument(
        "--sdf_style", default="colonial", choices=list(SDF_STYLES),
        help="Style recipe used when --asset_format=sdf_procedural.",
    )
    ap.add_argument(
        "--sdf_resolution", type=int, default=96,
        help="Voxel grid resolution per side for SDF sampling + marching cubes.",
    )
    ap.add_argument(
        "--sdf_seed_base", type=int, default=20260514,
        help="Per-building SDF seed = sdf_seed_base + building index.",
    )
    ap.add_argument(
        "--sdf_vqvae_prior", action="store_true",
        help=(
            "When --asset_format=sdf_procedural, additionally pass the procedural "
            "SDF through the frozen BuildingNet VQVAE (encode+decode) to smooth and "
            "regularize the field with a neural prior."
        ),
    )
    ap.add_argument(
        "--height_policy", choices=["osm", "area_aware"], default="osm",
        help=(
            "osm = use building['height'] from the OSM JSON (default). "
            "area_aware = if OSM height is missing or unrealistically low for "
            "the polygon area, raise to a class-aware default (mirrors "
            "scripts/osm_recompose_height_policy.py)."
        ),
    )
    ap.add_argument(
        "--placement_mode", choices=["fit", "aspect_preserve"], default="fit",
        help=(
            "fit = scale per-axis to exactly fit polygon XZ and target Y (current "
            "behavior; can flatten when OSM height is too low). "
            "aspect_preserve = use single uniform scale = min(s_xz, s_y) so "
            "buildings keep their native proportions."
        ),
    )
    ap.add_argument(
        "--cull_mask_resolution", type=int, default=256,
        help="Resolution of the rasterized polygon mask used by --gsplat_cull.",
    )
    ap.add_argument(
        "--cull_dilate_px", type=int, default=2,
        help="Boundary dilation (pixels) of the cull mask. Smaller = tighter cull.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_quality_model(path_value: str | None) -> dict | None:
    if not path_value:
        return None
    with open(path_value, "rb") as f:
        payload = pickle.load(f)
    if "model" not in payload:
        raise ValueError(f"Quality model payload has no 'model': {path_value}")
    payload.setdefault("bad_candidates", [])
    payload.setdefault("quality_weight", 0.20)
    return payload


def choose_buildings(payload: dict, limit: int) -> list[dict]:
    buildings = [
        b for b in payload.get("buildings", [])
        if len(b.get("polygon", [])) >= 3 and float(b.get("area", 0.0)) > 1.0
    ]
    buildings.sort(key=lambda b: float(b.get("area", 0.0)), reverse=True)
    return buildings[:limit] if limit > 0 else buildings


def footprint_image(mask: np.ndarray, title: str, size: int) -> Image.Image:
    body = Image.new("RGB", (size, size), "white")
    arr = (mask > 0).astype(np.uint8) * 255
    fp = Image.fromarray(arr, "L").resize((size, size), Image.Resampling.NEAREST)
    tint = Image.new("RGB", (size, size), (36, 112, 172))
    body.paste(tint, mask=fp)
    draw = ImageDraw.Draw(body)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(210, 210, 210))
    return title_cell(body, title, size)


def render_retrieved_obj(obj_path: Path, renderer, device: torch.device) -> Image.Image:
    mesh = load_obj_as_trimesh(str(obj_path))
    if mesh is None or len(mesh.faces) < 4:
        raise RuntimeError(f"Cannot load/render OBJ: {obj_path}")
    rgb = render_one(mesh, renderer, device)
    return Image.fromarray(rgb, "RGB")


def footprint_aspect(polygon: list[list[float]] | list[tuple[float, float]]) -> float:
    poly = np.asarray(polygon, dtype=np.float64)
    ext = poly.max(axis=0) - poly.min(axis=0)
    return float(max(ext[0], ext[1]) / max(min(ext[0], ext[1]), 1e-6))


def mesh_aspect_and_height_ratio(obj_path: Path) -> tuple[float, float, int, int]:
    mesh = load_obj_as_trimesh(str(obj_path))
    if mesh is None or len(mesh.faces) < 4:
        raise RuntimeError(f"Cannot load candidate OBJ: {obj_path}")
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    xz_min = max(min(ext[0], ext[2]), 1e-6)
    xz_max = max(ext[0], ext[2])
    aspect = float(xz_max / xz_min)
    height_ratio = float(ext[1] / max(xz_max, 1e-6))
    return aspect, height_ratio, int(len(mesh.vertices)), int(len(mesh.faces))


@torch.no_grad()
def retrieve_topk(building: dict, fp_mask: np.ndarray, retrieval: dict, device, top_k: int) -> list[tuple[str, float]]:
    full_class = building["class"]
    sub_idx = _resolve_subtype_idx(full_class, retrieval["subtype_to_idx"])
    top = _resolve_top(full_class)
    top_idx = retrieval["top_to_idx"].get(top, 0)
    fp = torch.from_numpy(fp_mask)[None, None].to(device).float()
    cls = torch.tensor([sub_idx], dtype=torch.long, device=device)
    emb, _ = retrieval["model"](fp, cls)
    emb_np = emb.cpu().numpy()[0]
    sims = retrieval["train_emb"] @ emb_np
    same_top = retrieval["train_top_ids"] == top_idx
    sims = np.where(same_top, sims, -1e9)
    order = np.argsort(-sims)[:top_k]
    return [(str(retrieval["train_ids"][j]), float(sims[j])) for j in order]


def choose_candidate(
    building: dict,
    candidates: list[tuple[str, float]],
    obj_dir: Path,
    policy: str,
    aspect_weight: float,
    height_weight: float,
    quality_payload: dict | None,
    quality_weight: float,
    bad_candidate_penalty: float,
) -> tuple[str, list[dict[str, object]]]:
    target_aspect = footprint_aspect(building["polygon"])
    target_height = float(building.get("height", 8.0))
    target_extent = max(np.ptp(np.asarray(building["polygon"], dtype=np.float64), axis=0))
    target_height_ratio = target_height / max(float(target_extent), 1e-6)
    rows = []
    for rank, (obj_id, sim) in enumerate(candidates, start=1):
        obj_path = obj_dir / f"{obj_id}.obj"
        cand_aspect, cand_height_ratio, verts, faces = mesh_aspect_and_height_ratio(obj_path)
        aspect_penalty = abs(np.log(max(cand_aspect, 1e-6) / max(target_aspect, 1e-6)))
        height_penalty = abs(np.log(max(cand_height_ratio, 1e-6) / max(target_height_ratio, 1e-6)))
        rerank_score = sim - aspect_weight * aspect_penalty - height_weight * height_penalty
        rows.append({
            "rank": rank,
            "candidate_id": obj_id,
            "retrieval_score": float(sim),
            "rerank_score": float(rerank_score),
            "target_aspect": float(target_aspect),
            "candidate_aspect": float(cand_aspect),
            "target_height_ratio": float(target_height_ratio),
            "candidate_height_ratio": float(cand_height_ratio),
            "aspect_penalty": float(aspect_penalty),
            "height_penalty": float(height_penalty),
            "candidate_verts": verts,
            "candidate_faces": faces,
        })
    if policy == "top1":
        chosen = rows[0]
    elif policy == "quality":
        if quality_payload is None:
            raise ValueError("--retrieval_policy quality requires --quality_model")
        model = quality_payload["model"]
        bad_candidates = set(quality_payload.get("bad_candidates", []))
        geom = building_geometry_features(building, target_height)
        area_m2 = float(building.get("area", geom["area_m2_from_polygon"]) or 0.0)
        for row in rows:
            features = candidate_quality_features(
                str(building.get("class", "")),
                area_m2,
                target_height,
                geom,
                row,
            )
            success = float(model.predict_proba(np.asarray([features], dtype=np.float32))[0, 1])
            penalty = bad_candidate_penalty if row["candidate_id"] in bad_candidates else 0.0
            row["predicted_success"] = success
            row["quality_bad_candidate"] = int(row["candidate_id"] in bad_candidates)
            row["quality_score"] = float(row["rerank_score"]) + quality_weight * success - penalty
        chosen = max(rows, key=lambda r: float(r["quality_score"]))
    else:
        chosen = max(rows, key=lambda r: float(r["rerank_score"]))
        if quality_payload is not None:
            model = quality_payload["model"]
            bad_candidates = set(quality_payload.get("bad_candidates", []))
            geom = building_geometry_features(building, target_height)
            area_m2 = float(building.get("area", geom["area_m2_from_polygon"]) or 0.0)
            for row in rows:
                features = candidate_quality_features(
                    str(building.get("class", "")),
                    area_m2,
                    target_height,
                    geom,
                    row,
                )
                success = float(model.predict_proba(np.asarray([features], dtype=np.float32))[0, 1])
                row["predicted_success"] = success
                row["quality_bad_candidate"] = int(row["candidate_id"] in bad_candidates)
                row["quality_score"] = float(row["rerank_score"]) + quality_weight * success
    for row in rows:
        row["chosen"] = int(row["candidate_id"] == chosen["candidate_id"])
    return str(chosen["candidate_id"]), rows


def safe_stem(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_")[:96] or "building"


def class_id(building_class: str) -> int:
    order = ["RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS", "MILITARY"]
    for i, top in enumerate(order):
        if building_class.startswith(top):
            return i
    return 0


def polygon_feature_maps(building: dict, mask: np.ndarray, size: int) -> torch.Tensor:
    poly = np.asarray(building["polygon"], dtype=np.float64)
    ext = poly.max(axis=0) - poly.min(axis=0)
    bbox_aspect = float(max(ext[0], ext[1]) / max(min(ext[0], ext[1]), 1e-6))
    area = float(building.get("area", 0.0) or 0.0)
    height = float(building.get("height", 0.0) or 0.0)
    height_norm = min(height / 40.0, 2.0)
    area_norm = min(np.log1p(area) / np.log1p(8000.0), 1.5)
    aspect_norm = min(np.log(max(bbox_aspect, 1e-6)) / np.log(5.0), 1.5)
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
    mask_t = torch.nn.functional.interpolate(mask_t, size=(size, size), mode="nearest")[0]
    maps = [
        mask_t,
        torch.full((1, size, size), float(height_norm)),
        torch.full((1, size, size), float(area_norm)),
        torch.full((1, size, size), float(aspect_norm)),
    ]
    cls = torch.zeros((CLASS_COUNT, size, size), dtype=torch.float32)
    cls[class_id(str(building.get("class", "")))].fill_(1.0)
    maps.append(cls)
    return torch.cat(maps, dim=0)


def load_learned_proposal(path_value: str | None, device: torch.device) -> tuple[ProposalUNet, int] | None:
    if not path_value:
        return None
    ckpt = torch.load(path_value, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    image_size = int(ckpt_args.get("image_size", 128))
    base = int(ckpt_args.get("base_channels", 32))
    model = ProposalUNet(base=base)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, image_size


def load_image_prior_manifest(path_value: str | None) -> dict[str, str]:
    if not path_value:
        return {}
    rows: dict[str, str] = {}
    with open(path_value, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            osm_id = str(row.get("osm_id", ""))
            image_path = str(row.get("image_prior_png", ""))
            if osm_id and image_path:
                rows[osm_id] = image_path
    return rows


@torch.no_grad()
def learned_proposal_image(
    learned: tuple[ProposalUNet, int],
    building: dict,
    mask: np.ndarray,
    device: torch.device,
    out_size: int,
) -> Image.Image:
    model, image_size = learned
    cond = polygon_feature_maps(building, mask, image_size)[None].to(device)
    pred = model(cond)[0].detach().cpu().clamp(0, 1)
    arr = (pred.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    if image_size != out_size:
        img = img.resize((out_size, out_size), Image.Resampling.BICUBIC)
    return img


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    input_dir = out_dir / "hunyuan_inputs"
    raw_dir = out_dir / "hunyuan_raw"
    simplified_dir = out_dir / "hunyuan_simplified"
    render_dir = out_dir / "renders"
    for d in (input_dir, raw_dir, simplified_dir, render_dir):
        d.mkdir(parents=True, exist_ok=True)

    payload = json.load(open(args.osm_json))
    buildings = choose_buildings(payload, args.limit)
    if not buildings:
        raise SystemExit(f"No usable buildings in {args.osm_json}")

    device = torch.device(args.device)
    print(f"[pipeline] device={device} buildings={len(buildings)}", flush=True)
    quality_payload = load_quality_model(args.quality_model)
    learned_proposal = load_learned_proposal(args.learned_proposal_ckpt, device)
    if args.conditioning_source == "learned_proposal" and learned_proposal is None:
        raise ValueError("--conditioning_source learned_proposal requires --learned_proposal_ckpt")
    image_prior_manifest = load_image_prior_manifest(args.image_prior_manifest)
    if args.conditioning_source == "image_prior" and not image_prior_manifest:
        raise ValueError("--conditioning_source image_prior requires --image_prior_manifest")
    if learned_proposal is not None:
        print(f"[pipeline] learned_proposal_ckpt={args.learned_proposal_ckpt}", flush=True)
    if image_prior_manifest:
        print(f"[pipeline] image_prior_manifest={args.image_prior_manifest}", flush=True)
    if quality_payload is not None:
        print(
            f"[pipeline] quality_model={args.quality_model} "
            f"bad_candidates={len(quality_payload.get('bad_candidates', []))}",
            flush=True,
        )
    # Run retrieval model on CPU to save precious GPU memory for gsplat
    retrieval = load_retrieval(Path(args.index_dir), args.retrieval_ckpt, torch.device("cpu"))
    obj_dir = Path(args.obj_dir)
    if args.asset_format == "mesh":
        obj_renderer = make_renderer(device, image_size=args.image_size)
    else:
        obj_renderer = None
    gsplat_dir = Path(args.gsplat_dir)
    if args.asset_format == "gsplat":
        print(f"[pipeline] asset_format=gsplat, gsplat_dir={gsplat_dir} cull={args.gsplat_cull}", flush=True)
        if not gsplat_dir.is_dir():
            raise FileNotFoundError(f"--gsplat_dir does not exist: {gsplat_dir}")
        pipe = None
    elif args.asset_format == "sdf_procedural":
        print(f"[pipeline] asset_format=sdf_procedural, style={args.sdf_style} "
              f"res={args.sdf_resolution} vqvae_prior={args.sdf_vqvae_prior} "
              f"(retrieval still runs for record-keeping)", flush=True)
        pipe = None
        vqvae_model = load_buildingnet_vqvae(device=str(device)) if args.sdf_vqvae_prior else None
    else:
        print(f"[pipeline] loading Hunyuan {args.model}", flush=True)
        pipe = load_pipeline(args.model)
        print("[pipeline] Hunyuan ready", flush=True)

    rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    sheet_rows: list[list[Image.Image]] = []
    placed_meshes = []
    placed_gaussians = []

    for i, building in enumerate(buildings):
        building_id = safe_stem(str(building.get("id", f"osm_{i}")))
        mask = rasterize_polygon(building["polygon"], res=64)
        candidates = retrieve_topk(building, mask, retrieval, torch.device("cpu"), args.retrieval_top_k)
        retrieved_id, candidate_rows = choose_candidate(
            building,
            candidates,
            obj_dir,
            args.retrieval_policy,
            args.aspect_weight,
            args.height_weight,
            quality_payload,
            args.quality_weight,
            args.quality_bad_candidate_penalty,
        )
        for candidate_row in candidate_rows:
            choice_rows.append({
                "index": i,
                "osm_id": building.get("id"),
                "class": building.get("class"),
                "retrieval_policy": args.retrieval_policy,
                **candidate_row,
            })
        obj_path = obj_dir / f"{retrieved_id}.obj"
        if not obj_path.exists():
            print(f"[pipeline] skip missing OBJ {obj_path}", flush=True)
            continue

        stem = f"{i:02d}_{building_id}_{retrieved_id}"
        print(f"[pipeline] {stem}", flush=True)

        if args.asset_format == "gsplat":
            ply_path = gsplat_dir / f"{retrieved_id}.ply"
            if not ply_path.exists():
                print(f"[pipeline] skip missing baked PLY {ply_path}", flush=True)
                continue
            target_height, height_source = infer_height_from_policy(building, args.height_policy)
            target_height = float(target_height)
            t0 = time.time()
            g = load_inria_ply(str(ply_path), device=str(device))
            placed_g = place_gsplat(
                g, building["polygon"], target_height,
                aspect_preserve=(args.placement_mode == "aspect_preserve"),
            )
            if args.gsplat_cull:
                placed_g = cull_outside_footprint(
                    placed_g, building["polygon"],
                    target_height=target_height,
                    mask_resolution=int(args.cull_mask_resolution),
                    dilate_px=int(args.cull_dilate_px),
                )
            seconds = time.time() - t0
            placed_gaussians.append(placed_g)

            # Camera at unit-frame canonical view; placed renders use a world-aware look-at.
            from pytorch3d.renderer import look_at_view_transform
            poly_arr = np.asarray(building["polygon"], dtype=np.float64)
            cx_w = float((poly_arr[:, 0].min() + poly_arr[:, 0].max()) / 2.0)
            cz_w = float((poly_arr[:, 1].min() + poly_arr[:, 1].max()) / 2.0)
            poly_diag = float(np.linalg.norm(poly_arr.max(0) - poly_arr.min(0)))
            view_dist = max(poly_diag * 1.6, target_height * 2.5, 6.0)

            R_can, T_can = look_at_view_transform(dist=2.5, elev=20.0, azim=30.0, at=((0.0, 0.0, 0.0),))
            R_pf, T_pf = look_at_view_transform(dist=view_dist, elev=20.0, azim=30.0,
                                                at=((cx_w, target_height / 2.0, cz_w),))
            R_ps, T_ps = look_at_view_transform(dist=view_dist, elev=20.0, azim=120.0,
                                                at=((cx_w, target_height / 2.0, cz_w),))
            canon_img = render_gsplat_view(g, R_can, T_can, fov_deg=30.0, image_size=args.image_size)
            placed_front = render_gsplat_view(placed_g, R_pf, T_pf, fov_deg=30.0, image_size=args.image_size)
            placed_side = render_gsplat_view(placed_g, R_ps, T_ps, fov_deg=30.0, image_size=args.image_size)

            canon_path = input_dir / f"{stem}_gsplat_canonical.png"
            placed_front_path = render_dir / f"{stem}_gsplat_placed_front.png"
            placed_side_path = render_dir / f"{stem}_gsplat_placed_side.png"
            canon_img.save(canon_path, optimize=True)
            placed_front.save(placed_front_path, optimize=True)
            placed_side.save(placed_side_path, optimize=True)

            placed_ply = simplified_dir / f"{stem}_placed.ply"
            save_inria_ply(str(placed_ply), placed_g)

            row = {
                "index": i,
                "osm_id": building.get("id"),
                "class": building.get("class"),
                "area_m2": f"{float(building.get('area', 0.0)):.2f}",
                "height_m": f"{target_height:.2f}",
                "retrieved_id": retrieved_id,
                "retrieval_policy": args.retrieval_policy,
                "retrieval_top_k": args.retrieval_top_k,
                "asset_format": "gsplat",
                "conditioning_source": args.conditioning_source,
                "height_policy": args.height_policy,
                "height_source": height_source,
                "placement_mode": args.placement_mode,
                "retrieval_candidates": candidate_rows,
                "gsplat_baked_ply": str(ply_path),
                "gsplat_placed_ply": str(placed_ply),
                "canonical_view_png": str(canon_path),
                "placed_front_png": str(placed_front_path),
                "placed_side_png": str(placed_side_path),
                "gsplat_cull": bool(args.gsplat_cull),
                "n_gaussians_baked": int(g.n),
                "n_gaussians_placed": int(placed_g.n),
                "gsplat_seconds": f"{seconds:.2f}",
            }
            rows.append(row)
            sheet_rows.append([
                footprint_image(mask, f"OSM {i}", args.image_size),
                title_cell(canon_img, f"baked 3DGS", args.image_size),
                title_cell(placed_front, f"placed (front)", args.image_size),
                title_cell(placed_side, f"placed (side)", args.image_size),
            ])
            print(
                f"  {building.get('class')} <- {retrieved_id} [{args.retrieval_policy}]; "
                f"format=gsplat; G_baked={g.n}; G_placed={placed_g.n}; "
                f"cull={args.gsplat_cull}; {seconds:.1f}s",
                flush=True,
            )
            continue

        if args.asset_format == "sdf_procedural":
            target_height, height_source = infer_height_from_policy(building, args.height_policy)
            target_height = float(target_height)
            poly_arr = np.asarray(building["polygon"], dtype=np.float32)
            seed = args.sdf_seed_base + i
            t0 = time.time()
            sdf_fn = build_styled_sdf(args.sdf_style, poly_arr, target_height, seed=seed)
            if args.sdf_vqvae_prior:
                mesh, _bbox = procedural_to_mesh_via_vqvae(
                    sdf_fn, poly_arr, target_height, vqvae_model,
                    res=64, device=str(device),
                )
            else:
                bbox = polygon_bbox_with_pad(poly_arr, target_height * 2.5, pad=0.10)
                grid = sample_grid(sdf_fn, args.sdf_resolution, bbox, device=str(device))
                mesh = grid_to_mesh(grid, bbox)
            seconds = time.time() - t0
            if mesh is None or len(mesh.faces) < 4:
                print(f"[pipeline] skip empty SDF mesh for {stem}", flush=True)
                continue

            placed_meshes.append(mesh)  # already in world frame
            sdf_obj_path = simplified_dir / f"{stem}_sdf_{args.sdf_style}.obj"
            mesh.export(sdf_obj_path)

            canonical_render = render_mesh_png(mesh, image_size=args.image_size)
            placed_render = canonical_render  # SDF mesh is born in world frame
            canonical_path = input_dir / f"{stem}_sdf_canonical.png"
            placed_path = render_dir / f"{stem}_sdf_placed.png"
            canonical_render.save(canonical_path, optimize=True)
            placed_render.save(placed_path, optimize=True)

            row = {
                "index": i,
                "osm_id": building.get("id"),
                "class": building.get("class"),
                "area_m2": f"{float(building.get('area', 0.0)):.2f}",
                "height_m": f"{target_height:.2f}",
                "retrieved_id": retrieved_id,
                "retrieval_policy": args.retrieval_policy,
                "retrieval_top_k": args.retrieval_top_k,
                "asset_format": "sdf_procedural",
                "sdf_style": args.sdf_style,
                "sdf_resolution": int(args.sdf_resolution),
                "sdf_seed": int(seed),
                "sdf_vqvae_prior": bool(args.sdf_vqvae_prior),
                "height_policy": args.height_policy,
                "height_source": height_source,
                "conditioning_source": args.conditioning_source,
                "retrieval_candidates": candidate_rows,
                "sdf_obj": str(sdf_obj_path),
                "canonical_view_png": str(canonical_path),
                "placed_render_png": str(placed_path),
                "n_verts": int(len(mesh.vertices)),
                "n_faces": int(len(mesh.faces)),
                "sdf_seconds": f"{seconds:.2f}",
            }
            rows.append(row)
            sheet_rows.append([
                footprint_image(mask, f"OSM {i}", args.image_size),
                title_cell(canonical_render, f"SDF {args.sdf_style}", args.image_size),
                title_cell(placed_render, "placed (world)", args.image_size),
                title_cell(canonical_render, f"seed={seed}", args.image_size),
            ])
            print(
                f"  {building.get('class')} (would-pick {retrieved_id}); "
                f"format=sdf_procedural style={args.sdf_style} seed={seed} "
                f"V={len(mesh.vertices):,} F={len(mesh.faces):,} {seconds:.1f}s",
                flush=True,
            )
            continue

        if args.conditioning_source == "proposal":
            input_image = proposal_image(
                building,
                args.image_size,
                detail=args.proposal_detail,
                include_footprint_inset=args.proposal_footprint_inset,
            )
            input_path = input_dir / f"{stem}_proposal_input.png"
        elif args.conditioning_source == "learned_proposal":
            assert learned_proposal is not None
            input_image = learned_proposal_image(
                learned_proposal,
                building,
                mask,
                device,
                args.image_size,
            )
            input_path = input_dir / f"{stem}_learned_proposal_input.png"
        elif args.conditioning_source == "image_prior":
            prior_path = Path(image_prior_manifest.get(str(building.get("id", "")), ""))
            if not prior_path.exists():
                raise FileNotFoundError(f"No image prior for OSM id {building.get('id')}: {prior_path}")
            input_image = Image.open(prior_path).convert("RGB").resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            input_path = input_dir / f"{stem}_image_prior_input.png"
        else:
            input_image = render_retrieved_obj(obj_path, obj_renderer, device)
            input_path = input_dir / f"{stem}_retrieved_input.png"
        input_image.save(input_path, optimize=True)

        t0 = time.time()
        mesh = pipe(
            image=input_image.convert("RGBA"),
            num_inference_steps=args.steps,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
            generator=torch.manual_seed(args.seed + i),
            output_type="trimesh",
        )[0]
        seconds = time.time() - t0

        raw_glb = raw_dir / f"{stem}.glb"
        raw_render = render_dir / f"{stem}_hunyuan_raw.png"
        mesh.export(raw_glb)
        raw_preview = render_mesh_png(mesh, image_size=args.image_size)
        raw_preview.save(raw_render, optimize=True)

        simplified_obj = simplified_dir / f"{stem}_simplified.obj"
        simp_row = simplify_one(raw_glb, simplified_obj, args.target_faces, args.target_faces)
        simplified_mesh = trimesh.load(simplified_obj, force="mesh", process=False)
        mesh_target_height, mesh_height_source = infer_height_from_policy(building, args.height_policy)
        placed = place_mesh(
            simplified_mesh,
            building["polygon"],
            float(mesh_target_height),
            aspect_preserve=(args.placement_mode == "aspect_preserve"),
        )
        placed_meshes.append(placed)
        placed_render = render_dir / f"{stem}_placed.png"
        placed_preview = render_mesh_png(placed, image_size=args.image_size)
        placed_preview.save(placed_render, optimize=True)

        row = {
            "index": i,
            "osm_id": building.get("id"),
            "class": building.get("class"),
            "area_m2": f"{float(building.get('area', 0.0)):.2f}",
            "height_m": f"{float(building.get('height', 0.0)):.2f}",
            "retrieved_id": retrieved_id,
            "retrieval_policy": args.retrieval_policy,
            "retrieval_top_k": args.retrieval_top_k,
            "conditioning_source": args.conditioning_source,
            "retrieval_candidates": candidate_rows,
            "retrieved_input_png": str(input_path),
            "hunyuan_raw_glb": str(raw_glb),
            "hunyuan_raw_render_png": str(raw_render),
            "simplified_obj": str(simplified_obj),
            "placed_render_png": str(placed_render),
            "hunyuan_seconds": f"{seconds:.2f}",
            "raw_verts": int(len(mesh.vertices)),
            "raw_faces": int(len(mesh.faces)),
            "simp_verts": simp_row["verts_after"],
            "simp_faces": simp_row["faces_after"],
        }
        rows.append(row)
        sheet_rows.append([
            footprint_image(mask, f"OSM {i}", args.image_size),
            title_cell(input_image, f"{args.conditioning_source} input", args.image_size),
            title_cell(raw_preview, f"hunyuan {i}", args.image_size),
            title_cell(placed_preview, f"placed {i}", args.image_size),
        ])
        print(
            f"  {building.get('class')} <- {retrieved_id} [{args.retrieval_policy}]; "
            f"input={args.conditioning_source}; "
            f"Hunyuan F={len(mesh.faces):,}; simplified F={simp_row['faces_after']:,}",
            flush=True,
        )

    if not rows:
        raise SystemExit("No output assets were generated.")

    if args.asset_format == "gsplat":
        if not placed_gaussians:
            raise SystemExit("No placed Gaussians produced.")
        scene_g = gsplat_compose(placed_gaussians)
        scene_path = out_dir / "osm_3dgs_scene.ply"
        save_inria_ply(str(scene_path), scene_g)
        print(f"[pipeline] composed scene: {scene_g.n:,} gaussians -> {scene_path}", flush=True)
    elif args.asset_format == "sdf_procedural":
        if not placed_meshes:
            raise SystemExit("No SDF meshes were generated.")
        scene = trimesh.util.concatenate(placed_meshes)
        scene_path = out_dir / f"osm_sdf_scene_{args.sdf_style}.obj"
        scene.export(scene_path)
        print(f"[pipeline] composed scene: {len(scene.faces):,} faces -> {scene_path}", flush=True)
    else:
        if not placed_meshes:
            raise SystemExit("No output meshes were generated.")
        scene = trimesh.util.concatenate(placed_meshes)
        scene_path = out_dir / "osm_hunyuan_scene.obj"
        scene.export(scene_path)

    csv_path = out_dir / "osm_hunyuan_pipeline_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    choices_path = out_dir / "osm_retrieval_rerank_choices.csv"
    with choices_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(choice_rows[0].keys()))
        writer.writeheader()
        writer.writerows(choice_rows)

    cell_w = args.image_size
    cell_h = args.image_size + 28
    sheet = Image.new("RGB", (4 * cell_w, len(sheet_rows) * cell_h), "white")
    for r, cells in enumerate(sheet_rows):
        for c, cell in enumerate(cells):
            sheet.paste(cell, (c * cell_w, r * cell_h))
    sheet_path = out_dir / "osm_hunyuan_pipeline_sheet.png"
    sheet.save(sheet_path, optimize=True)

    log_path = scene_path.with_suffix(".log.json")
    with log_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"[pipeline] scene:   {scene_path}", flush=True)
    print(f"[pipeline] sheet:   {sheet_path}", flush=True)
    print(f"[pipeline] metrics: {csv_path}", flush=True)
    print(f"[pipeline] choices: {choices_path}", flush=True)
    print(f"[pipeline] log:     {log_path}", flush=True)


if __name__ == "__main__":
    main()
