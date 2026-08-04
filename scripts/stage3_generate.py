"""Stage 3 generation CLI — produces an SDF, mesh, and optionally Gaussians
from `(footprint, class, height, style)` conditioning.

Inputs (one of):
    --polygon       inline JSON list of [x, z] points in world meters
    --polygon_json  OSM-style JSON (see scripts/osm_hunyuan_pipeline_smoke.py)
    --polygon_npy   .npy file with (P, 2) polygon vertices in XZ
    --footprint_png direct path to a 64x64 binary mask PNG

Conditioning:
    --class       BuildingNet subtype string (e.g. RESIDENTIALhouse)
    --height_m    metric height (meters); also accepts --height_n in Frame-N units
    --style       one of [modern, colonial, victorian, industrial, craftsman,
                  mediterranean, contemporary, public_civic, unknown]

Outputs (all optional, named via --out):
    {out}_sdf.npy        decoded Stage 3a SDF (64^3, float32, Frame-N)
    {out}_mesh.obj       marching-cubes mesh in Frame-N
    {out}_mesh_world.obj mesh placed at the polygon in world coords (if --place)
    {out}_gs.ply         Stage 3b Gaussians in Frame-N (if --stage3b_ckpt given)
    {out}_gs_world.ply   placed Gaussians in world coords (if --place)

Example (single rectangle, victorian, generate both SDF mesh and Gaussians):

    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
        scripts/stage3_generate.py \\
        --polygon '[[-5,-3],[5,-3],[5,3],[-5,3]]' \\
        --class RESIDENTIALhouse \\
        --height_m 8.0 \\
        --style victorian \\
        --stage3a_ckpt logs_building/<...>/ckpt/stage3a_steps-latest.pth \\
        --stage3a_cfg  configs/stage3a_sdf_diffusion.yaml \\
        --vq_cfg       configs/vqvae_bnet_v2.yaml \\
        --vq_ckpt      logs_building/<...>/ckpt/vqvae_steps-latest.pth \\
        --stage3b_ckpt logs_building/<...>/ckpt/stage3b_steps-latest.pth \\
        --stage3b_cfg  configs/stage3b_lifter.yaml \\
        --place \\
        --out outputs/stage3_demo/victorian_rect
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf

from datasets.buildingnet_retrieval_dataset import build_label_maps, load_split_ids
from datasets.stage3a_dataset import STYLE_UNKNOWN_ID, _RECIPE_STYLE_ORDER
from models.stage3a_model import Stage3aModel
from models.networks.sdf_to_gs_lifter import unpack_slots_to_gaussians
from scene.gsplat_common import GaussianSet, save_inria_ply
from scene.gsplat_placement import place_gsplat
from scene.run_demo import place_mesh, rasterize_polygon
from scene.sdf_primitives import grid_to_mesh


def _build_label_map() -> dict[str, int]:
    bn_root = REPO / "data" / "BuildingNet_dataset_v0_1"
    all_ids = (
        load_split_ids(bn_root, "train")
        + load_split_ids(bn_root, "val")
        + load_split_ids(bn_root, "test")
    )
    subtype_to_idx, _ = build_label_maps(all_ids)
    return subtype_to_idx


def _style_to_id(style: str) -> int:
    if style == "unknown":
        return STYLE_UNKNOWN_ID
    if style in _RECIPE_STYLE_ORDER:
        return _RECIPE_STYLE_ORDER.index(style)
    raise SystemExit(f"Unknown style: {style}")


def _load_polygon(args) -> np.ndarray:
    if args.polygon:
        return np.asarray(json.loads(args.polygon), dtype=np.float32)
    if args.polygon_npy:
        return np.load(args.polygon_npy).astype(np.float32)
    if args.polygon_json:
        with open(args.polygon_json) as f:
            payload = json.load(f)
        buildings = payload.get("buildings", [])
        if not buildings:
            raise SystemExit(f"No buildings in {args.polygon_json}")
        return np.asarray(buildings[args.polygon_index]["polygon"], dtype=np.float32)
    return None


def _make_stage3a_opt(args) -> SimpleNamespace:
    """Build the minimal opt namespace Stage3aModel.initialize expects."""
    return SimpleNamespace(
        isTrain=False,
        device=args.device,
        df_cfg=args.stage3a_cfg,
        vq_cfg=args.vq_cfg,
        vq_ckpt=args.vq_ckpt,
        ckpt=args.stage3a_ckpt,
        ddim_steps=args.ddim_steps,
        debug="0",
        # Placeholders for fields BaseModel.initialize touches:
        gpu_ids=[0],
        ckpt_dir="/tmp",
        # Stage 3a-specific:
        latent_size_HW=(16, 16),
        latent_size_D=16,
    )


def _make_stage3b_opt(args) -> SimpleNamespace:
    return SimpleNamespace(
        isTrain=False,
        device=args.device,
        df_cfg=args.stage3b_cfg,
        ckpt=args.stage3b_ckpt,
        debug="0",
        gpu_ids=[0],
        ckpt_dir="/tmp",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polygon", help="Inline polygon JSON: [[x,z], ...]")
    ap.add_argument("--polygon_json", help="OSM-style JSON; uses buildings[polygon_index]['polygon']")
    ap.add_argument("--polygon_index", type=int, default=0)
    ap.add_argument("--polygon_npy", help=".npy file with (P, 2) XZ vertices")
    ap.add_argument("--footprint_png", help="64x64 binary mask PNG (skips polygon rasterization)")

    ap.add_argument("--class", dest="class_label", required=True,
                    help="BuildingNet subtype label, e.g. RESIDENTIALhouse")
    ap.add_argument("--height_m", type=float, default=None,
                    help="Metric height in meters. Either this OR --height_n must be set.")
    ap.add_argument("--height_n", type=float, default=None,
                    help="Frame-N Y extent (~0..2). Overrides --height_m if both set.")
    ap.add_argument("--style", default="unknown",
                    choices=list(_RECIPE_STYLE_ORDER) + ["unknown"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--stage3a_ckpt", required=True)
    ap.add_argument("--stage3a_cfg", default="configs/stage3a_sdf_diffusion.yaml")
    ap.add_argument("--vq_cfg", required=True)
    ap.add_argument("--vq_ckpt", required=True)

    ap.add_argument("--stage3b_ckpt", default=None,
                    help="If given, also run Stage 3b to emit Gaussians.")
    ap.add_argument("--stage3b_cfg", default="configs/stage3b_lifter.yaml")

    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--place", action="store_true",
                    help="Also save world-frame placed mesh / Gaussians using the polygon.")
    ap.add_argument("--aspect_preserve", action="store_true", default=True)
    ap.add_argument("--out", required=True, help="Output path prefix.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1) Resolve conditioning ----------------------------------------------
    label_map = _build_label_map()
    if args.class_label not in label_map:
        raise SystemExit(f"Unknown class label '{args.class_label}'. "
                         f"First 5 known: {list(label_map)[:5]}")
    class_id = int(label_map[args.class_label])
    style_id = _style_to_id(args.style)

    polygon = _load_polygon(args)
    if args.footprint_png:
        from PIL import Image
        fp_img = np.asarray(Image.open(args.footprint_png).convert("L"), dtype=np.float32)
        if fp_img.shape != (64, 64):
            fp_img = np.asarray(
                Image.fromarray(fp_img.astype(np.uint8)).resize((64, 64))
            ).astype(np.float32)
        fp_np = (fp_img > 127).astype(np.float32)
    elif polygon is not None:
        fp_np = rasterize_polygon(polygon, res=64).astype(np.float32)
    else:
        raise SystemExit("Provide one of --polygon, --polygon_json, --polygon_npy, --footprint_png")

    if args.height_n is not None:
        height_cond = float(args.height_n)
    elif args.height_m is not None:
        # Heuristic: a typical real BN asset spans ~0.5 Frame-N Y per 7-10 m
        # of metric height. We'll use 10 m -> 1.0 as a default mapping; users
        # who care about precision should pass --height_n directly.
        height_cond = float(args.height_m / 10.0)
    else:
        raise SystemExit("Provide one of --height_m or --height_n")

    print(f"[stage3] class={args.class_label} (id={class_id}) style={args.style} (id={style_id}) "
          f"height_cond={height_cond:.3f}  fp_occupancy={fp_np.mean():.3f}")

    # --- 2) Build input batch (B=1) ------------------------------------------
    fp_t = torch.from_numpy(fp_np)[None, None, :, :].to(args.device)
    cls_t = torch.tensor([class_id], dtype=torch.long, device=args.device)
    sty_t = torch.tensor([style_id], dtype=torch.long, device=args.device)
    hgt_t = torch.tensor([height_cond], dtype=torch.float32, device=args.device)
    # GT SDF placeholder; only consumed by set_input (.to(device)); not actually
    # used at inference because Stage3aModel.inference samples from random noise.
    dummy_sdf = torch.zeros(1, 1, 64, 64, 64, device=args.device)
    data = {
        "sdf": dummy_sdf,
        "fp": fp_t,
        "class_id": cls_t,
        "style_id": sty_t,
        "height": hgt_t,
    }

    # --- 3) Stage 3a inference -----------------------------------------------
    print(f"[stage3a] loading model from {args.stage3a_ckpt}")
    stage3a_opt = _make_stage3a_opt(args)
    stage3a = Stage3aModel()
    stage3a.initialize(stage3a_opt)
    sdf_t = stage3a.inference(data, ddim_steps=args.ddim_steps)  # (1, 1, 64, 64, 64)
    sdf_np_out = sdf_t.detach().cpu().numpy()[0, 0]              # (64, 64, 64)
    print(f"[stage3a] sdf range=[{sdf_np_out.min():.3f}, {sdf_np_out.max():.3f}]")

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(out_prefix) + "_sdf.npy", sdf_np_out)
    print(f"[stage3a] sdf -> {out_prefix}_sdf.npy")

    # --- 4) Marching cubes -> Frame-N mesh -----------------------------------
    # Frame-N bbox is [-1, 1]^3 for the SDF voxel grid.
    bbox_n = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    sdf_torch = torch.from_numpy(sdf_np_out)
    mesh = grid_to_mesh(sdf_torch, bbox_n, iso=0.0)
    if mesh is None:
        print("[stage3a] marching cubes found no surface; skipping mesh export.")
    else:
        mesh_path = str(out_prefix) + "_mesh.obj"
        mesh.export(mesh_path)
        print(f"[stage3a] mesh V={len(mesh.vertices):,} F={len(mesh.faces):,} -> {mesh_path}")

        if args.place and polygon is not None and args.height_m is not None:
            placed = place_mesh(
                mesh, polygon, args.height_m,
                aspect_preserve=args.aspect_preserve,
            )
            placed_path = str(out_prefix) + "_mesh_world.obj"
            placed.export(placed_path)
            print(f"[stage3a] placed mesh -> {placed_path}")

    # --- 5) Optional Stage 3b -> Gaussians -----------------------------------
    if args.stage3b_ckpt:
        from models.stage3b_model import Stage3bModel
        print(f"[stage3b] loading model from {args.stage3b_ckpt}")
        stage3b_opt = _make_stage3b_opt(args)
        stage3b = Stage3bModel()
        stage3b.initialize(stage3b_opt)
        # Stage 3b consumes Stage 3a's SDF, plus the same conditioning batch.
        data_3b = {
            "sdf": sdf_t,
            "fp": fp_t,
            "class_id": cls_t,
            "style_id": sty_t,
            "height": hgt_t,
        }
        slots_pred, occ_logits = stage3b.inference(data_3b)
        # Frame-N bbox for the Gaussians = [-1, 1]^3 (same convention as SDF).
        bbox_t = torch.tensor([[[-1.0, -1.0, -1.0],
                                [1.0, 1.0, 1.0]]],
                              device=slots_pred.device, dtype=torch.float32)
        sets = unpack_slots_to_gaussians(slots_pred, occ_logits, bbox_t,
                                         occ_threshold=stage3b.occ_threshold)
        attrs = sets[0]
        g = GaussianSet(
            means=attrs["means"].cpu(),
            raw_scales=attrs["raw_scales"].cpu(),
            raw_quats=attrs["raw_quats"].cpu(),
            raw_opac=attrs["raw_opac"].cpu(),
            sh_dc=attrs["sh_dc"].cpu(),
        )
        gs_path = str(out_prefix) + "_gs.ply"
        save_inria_ply(gs_path, g)
        print(f"[stage3b] gaussians N={g.n:,} -> {gs_path}")

        if args.place and polygon is not None and args.height_m is not None:
            g_world = place_gsplat(
                g, polygon, args.height_m,
                aspect_preserve=args.aspect_preserve,
            )
            gs_world_path = str(out_prefix) + "_gs_world.ply"
            save_inria_ply(gs_world_path, g_world)
            print(f"[stage3b] placed gaussians -> {gs_world_path}")

    print("[stage3] done.")


if __name__ == "__main__":
    main()
