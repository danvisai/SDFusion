"""Stage A — FastAPI inference service for the Option B+ generative head.

Stateless HTTP wrapper around RecipeInferenceEngine, implementing the JSON contract from
docs/DEPLOYMENT_PLAN.md so Blender / Unity / Unreal plugins all talk to one API:

    GET  /health
    POST /params_to_mesh        FAST, no model — for host slider edits (<200 ms)
    POST /regenerate_building   generative — sample recipe params + mesh one building
    POST /generate_tile         generative — a batch of footprints -> buildings

Meshes are returned as base64-encoded binary glTF (.glb). Footprints/heights are in
METERS (world frame); the host engine places each building at its returned `position_xz`.

Run (dev):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python -m uvicorn scripts.server.inference_service:app --port 8000
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts" / "appearance"))   # texture_bake (v2)

from recipe_inference import RecipeInferenceEngine, DIVERSITY_PRESETS  # noqa: E402
from refine import Refiner                                             # noqa: E402
from models.networks import recipe_param_space as ps                   # noqa: E402

app = FastAPI(title="GenerativeTowns Inference (Option B+)", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
WEB = Path(__file__).resolve().parent / "web"
_engine: Optional[RecipeInferenceEngine] = None
_refiner: Optional[Refiner] = None


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/sculpt.html")
def sculpt_page():
    """The unbound.io-style raymarched SDF sculptor (single building)."""
    return FileResponse(WEB / "sculpt.html", headers={"Cache-Control": "no-store"})


if (WEB / "samples").exists():
    app.mount("/samples", StaticFiles(directory=str(WEB / "samples")), name="samples")


def engine() -> RecipeInferenceEngine:
    global _engine
    if _engine is None:
        _engine = RecipeInferenceEngine()  # loads B+.6 ckpt once
    return _engine


def refiner() -> Refiner:
    global _refiner
    if _refiner is None:
        _refiner = Refiner(engine())
    return _refiner


@app.on_event("startup")
def _warm():
    engine()
    # Pre-warm the snap prior off the request path: the two ~15 GB ckpts take minutes to
    # load under node contention, which once serialized behind first-click requests
    # (branch-suite B3/B4 timeouts, 2026-06-11).
    import threading
    threading.Thread(target=lambda: refiner()._load_sdedit(), daemon=True).start()


def _guidance(diversity: Optional[str], guidance: Optional[float]) -> float:
    if guidance is not None:
        return float(guidance)
    return DIVERSITY_PRESETS.get((diversity or "medium").lower(), 2.0)


def _b64(glb: bytes) -> str:
    return base64.b64encode(glb).decode("ascii")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ParamsToMeshReq(BaseModel):
    recipe_params: List[float]
    style: str
    footprint: List[List[float]] = Field(..., description="polygon [[x,z],...] in meters")
    height: float


class BuildingReq(BaseModel):
    footprint: List[List[float]]
    style: str
    building_class: str = "RESIDENTIAL"
    height: float = 10.0
    seed: Optional[int] = None
    diversity: Optional[str] = None          # low | medium | high
    guidance: Optional[float] = None         # overrides diversity preset
    eta: float = 1.0
    steps: int = 50
    detail: bool = True                      # compose facade detail + roof + landmarks


class Footprint(BaseModel):
    footprint: List[List[float]]
    style: Optional[str] = None
    building_class: str = "RESIDENTIAL"
    height: float = 10.0


class TileReq(BaseModel):
    buildings: List[Footprint]
    default_style: str = "modern"
    seed: Optional[int] = None
    diversity: Optional[str] = None
    guidance: Optional[float] = None


class MeshResp(BaseModel):
    style: str
    recipe_params: List[float]
    mesh_glb_b64: str
    n_vertices: int
    n_faces: int
    position_xz: List[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "device": e.device, "styles": ps.STYLES,
            "classes": ps.CLASSES, "diversity_presets": DIVERSITY_PRESETS,
            "sdedit_ready": bool(_refiner is not None
                                 and getattr(_refiner, "_sd_main", None) is not None)}


@app.post("/params_to_mesh", response_model=MeshResp)
def params_to_mesh(req: ParamsToMeshReq):
    e = engine()
    if req.style not in ps.STYLE_TO_IDX:
        raise HTTPException(400, f"unknown style '{req.style}'")
    try:
        mesh, _ = e.params_to_mesh(req.recipe_params, req.style, req.footprint, req.height)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    glb = e.mesh_to_glb(mesh)
    nv = 0 if mesh is None else len(mesh.vertices)
    nf = 0 if mesh is None else len(mesh.faces)
    return MeshResp(style=req.style, recipe_params=list(map(float, req.recipe_params)),
                    mesh_glb_b64=_b64(glb), n_vertices=nv, n_faces=nf, position_xz=[0.0, 0.0])


@app.post("/regenerate_building", response_model=MeshResp)
def regenerate_building(req: BuildingReq):
    e = engine()
    if req.style not in ps.STYLE_TO_IDX:
        raise HTTPException(400, f"unknown style '{req.style}'")
    b = e.generate_building(req.footprint, req.building_class, req.height, req.style,
                            seed=req.seed, guidance=_guidance(req.diversity, req.guidance),
                            eta=req.eta, steps=req.steps, detail=req.detail)
    return MeshResp(style=b.style, recipe_params=list(map(float, b.recipe_params)),
                    mesh_glb_b64=_b64(b.glb), n_vertices=b.n_vertices, n_faces=b.n_faces,
                    position_xz=list(b.position_xz))


class RefineReq(BaseModel):
    base_style: str
    base_recipe_params: List[float]
    footprint: List[List[float]]
    height: float
    edits: List[dict] = Field(default_factory=list)   # scene/sdf_edit EditOp dicts
    target_style: Optional[str] = None                # None -> keep base style (cleanup); else re-style
    building_class: str = "RESIDENTIAL"
    mode: str = "fast"                                # fast | quality | displacement | sdedit (learned prior)
    seed: Optional[int] = None
    strength: float = 0.5                             # sdedit: faithfulness<->realism (0=keep, 1=regen)
    sdedit_steps: int = 8                             # sdedit: DDIM steps (8 ~= 25 in quality, far faster)
    autoguidance: bool = True                         # sdedit: guide 30k prior with 10k ckpt of itself
    auto_scale: float = 2.0
    detail: bool = True                               # sdedit: re-apply composer detail on the snapped massing


@app.post("/refine_with_edit", response_model=MeshResp)
def refine_with_edit(req: RefineReq):
    """Project a crude sculpt (base building + primitive edits) onto a clean recipe
    building in `target_style` (cleanup / re-style), keeping the sculpted massing."""
    if req.base_style not in ps.STYLE_TO_IDX:
        raise HTTPException(400, f"unknown base_style '{req.base_style}'")
    ts = req.target_style or req.base_style
    if ts not in ps.STYLE_TO_IDX:
        raise HTTPException(400, f"unknown target_style '{ts}'")
    base_state = {"style": req.base_style, "recipe_params": req.base_recipe_params,
                  "footprint": req.footprint, "height": req.height}
    try:
        out = refiner().refine(base_state, req.edits, target_style=ts, mode=req.mode,
                               building_class=req.building_class, seed=req.seed,
                               strength=req.strength, sdedit_steps=req.sdedit_steps,
                               autoguidance=req.autoguidance, auto_scale=req.auto_scale,
                               detail=req.detail)
    except Exception as ex:
        raise HTTPException(400, f"refine failed: {ex}")
    glb = engine().mesh_to_glb(out["mesh"])
    nv = 0 if out["mesh"] is None else len(out["mesh"].vertices)
    nf = 0 if out["mesh"] is None else len(out["mesh"].faces)
    return MeshResp(style=out["style"], recipe_params=out["recipe_params"],
                    mesh_glb_b64=_b64(glb), n_vertices=nv, n_faces=nf, position_xz=[0.0, 0.0])


class ImageTownReq(BaseModel):
    image_b64: str                       # data URL or raw base64 of a footprint image / OSM map
    meters_across: float = 200.0         # real-world width the image spans
    max_buildings: int = 40
    invert: bool = False                 # flip if buildings/background detected backwards
    seed: Optional[int] = None


class TownBuilding(BaseModel):
    glb_b64: str
    footprint: List[List[float]]         # local, centered (m)
    position: List[float]                # [x, z] world placement (m)
    style: str
    building_class: str
    height: float
    recipe_params: List[float]


class ImageTownResp(BaseModel):
    n_buildings: int
    buildings: List[TownBuilding]        # per-building -> selectable + editable in the UI


_IMG_STYLE = {"RESIDENTIAL": ["colonial", "craftsman", "victorian", "modern"],
              "COMMERCIAL": ["modern", "contemporary", "industrial"],
              "PUBLIC": ["public_civic", "modern"], "RELIGIOUS": ["victorian", "public_civic"]}


@app.post("/generate_from_image", response_model=ImageTownResp)
def generate_from_image(req: ImageTownReq):
    """Extract building footprints from an uploaded image (OSM map / footprint mask) and
    generate a 3D town from them — the core OSM-footprints -> town goal."""
    import io as _io
    import numpy as np
    import trimesh
    from footprint_image import extract_footprints, to_meters
    raw = base64.b64decode(req.image_b64.split(",")[-1])
    polys, hw = extract_footprints(raw, req.max_buildings, invert=req.invert)
    if not polys:
        raise HTTPException(400, "no footprints detected — try the Invert toggle or a clearer image")
    fps = to_meters(polys, hw, req.meters_across)
    areas = [float(abs(np.cross(np.diff(np.vstack([p, p[:1]]), axis=0),
             np.vstack([p, p[:1]])[:-1]).sum()) * 0.5) for p, _ in fps]
    a_sorted = sorted(areas, reverse=True)
    big_cut = a_sorted[max(1, len(areas) // 12)] if areas else 0  # largest ~8% -> civic/religious

    e = engine()
    out = []
    for i, ((local, cen), area) in enumerate(zip(fps, areas)):
        cls = ("RELIGIOUS" if area >= big_cut and i % 3 == 0 else
               "PUBLIC" if area >= big_cut else
               "COMMERCIAL" if area > np.median(areas) else "RESIDENTIAL")
        style = _IMG_STYLE[cls][i % len(_IMG_STYLE[cls])]
        span = float(np.sqrt(max(area, 1.0)))
        height = float(np.clip(span * (3.0 if cls in ("COMMERCIAL", "PUBLIC") else 1.6)
                               + (i * 37 % 9), 5, 70))
        seed = i if req.seed is None else req.seed + i
        b = e.generate_building(local, cls, height, style, seed=seed, detail=True)
        if not b.glb:
            continue
        out.append(TownBuilding(glb_b64=_b64(b.glb), footprint=local.tolist(),
                                position=[float(cen[0]), float(cen[1])], style=style,
                                building_class=cls, height=height,
                                recipe_params=[float(x) for x in b.recipe_params]))
    if not out:
        raise HTTPException(400, "footprints found but generation produced no meshes")
    return ImageTownResp(n_buildings=len(out), buildings=out)


@app.post("/generate_tile", response_model=List[MeshResp])
def generate_tile(req: TileReq):
    e = engine()
    g = _guidance(req.diversity, req.guidance)
    out = []
    for i, f in enumerate(req.buildings):
        style = f.style or req.default_style
        if style not in ps.STYLE_TO_IDX:
            raise HTTPException(400, f"building {i}: unknown style '{style}'")
        seed = None if req.seed is None else req.seed + i
        b = e.generate_building(f.footprint, f.building_class, f.height, style,
                                seed=seed, guidance=g)
        out.append(MeshResp(style=b.style, recipe_params=list(map(float, b.recipe_params)),
                            mesh_glb_b64=_b64(b.glb), n_vertices=b.n_vertices,
                            n_faces=b.n_faces, position_xz=list(b.position_xz)))
    return out


# ---------------------------------------------------------------------------
# SDF-volume endpoints — feed the unbound.io-style raymarched sculptor
# (web/sculpt.html). Volumes are float32 little-endian, layout (D=z,H=y,W=x)
# = x-fastest, directly uploadable as THREE.Data3DTexture(res,res,res).
# ---------------------------------------------------------------------------

class BuildingSdfReq(BaseModel):
    footprint: List[List[float]]
    style: str
    building_class: str = "RESIDENTIAL"
    height: float = 10.0
    seed: Optional[int] = None
    res: int = 64
    sdedit_strength: float = 0.0   # >0: pull the recipe massing onto the real-building manifold
                                   # (hybrid 3D-BAG prior) right at generation
    massing_source: str = "recipe" # recipe | bag (retrieve the closest-footprint REAL 3D BAG
                                   # house and fit it to the user's footprint + height)


class SdfResp(BaseModel):
    sdf_b64: str                          # float32 LE, res^3, (D=z,H=y,W=x)
    res: int
    center: List[float]                   # world center c (x,y,z) of the cube
    scale: float                          # world half-extent s of the cube
    style: str
    recipe_params: List[float]
    footprint: List[List[float]]
    height: float


@app.post("/building_sdf", response_model=SdfResp)
def building_sdf(req: BuildingSdfReq):
    """Generate ONE building and return it as a 64^3 SDF volume in a normalized [-1,1]^3 cube,
    for the raymarched sculptor. (center, scale) map the cube back to world meters for placement."""
    import numpy as np
    e = engine()
    r = refiner()
    if req.style not in ps.STYLE_TO_IDX:
        raise HTTPException(400, f"unknown style '{req.style}'")
    poly = np.asarray(req.footprint, np.float32)
    local = poly - poly.mean(axis=0)
    params = e.sample_params(local, req.height, req.building_class, req.style, seed=req.seed)
    try:
        if req.massing_source == "bag":
            grid, c, s, idx = r.bag_house_volume(local, req.height, res=req.res)
        else:
            grid, c, s, _hn = r.building_volume(local, req.style, params, req.height, res=req.res)
        if req.sdedit_strength > 0:    # generate-time realism: snap the fresh massing
            grid, _ = r.snap_volume(grid, [], strength=float(req.sdedit_strength))
    except Exception as ex:
        raise HTTPException(400, f"building_sdf failed: {ex}")
    return SdfResp(sdf_b64=_b64(grid.astype("<f4").tobytes()), res=int(grid.shape[0]),
                   center=[float(x) for x in c], scale=float(s), style=req.style,
                   recipe_params=[float(x) for x in params], footprint=local.tolist(),
                   height=float(req.height))


class SnapSdfReq(BaseModel):
    base_sdf_b64: str                                 # float32 LE res^3 cube-frame volume
    res: int = 64
    edits: List[dict] = Field(default_factory=list)   # scene/sdf_edit EditOp dicts, CUBE coords [-1,1]
    detail_edits: List[dict] = Field(default_factory=list)  # crisp semantic ops (door/window/...)
                                                      # applied AFTER the snap at high res — never
                                                      # melted by the prior (two-layer doctrine)
    resnap_detail_ops: List[dict] = Field(default_factory=list)  # detail ops to RE-PROJECT onto
                                                      # the snapped surface (walls move under them)
    adjust: bool = False                              # default keeps elements where placed
                                                      # (geometric resnap to the EXTERIOR wall only);
                                                      # opt-in True adds row regularization. The
                                                      # learned interior re-coherence was REMOVED.
    strength: float = 0.5
    sdedit_steps: int = 8
    autoguidance: bool = True
    auto_scale: float = 2.0
    local: bool = True                                # localized snap: generative only at the
                                                      # placed mass + seam; base stays crisp
    return_mesh: bool = False
    center: Optional[List[float]] = None              # cube->world (from /building_sdf) for bake
    scale: Optional[float] = None
    detail: bool = False                              # run ② composer/detail at bake
    building_class: str = "RESIDENTIAL"
    style: str = "modern"
    seed: Optional[int] = None


class SnapResp(BaseModel):
    sdf_b64: str
    res: int
    iou_to_edit: float
    mesh_glb_b64: Optional[str] = None
    resnapped_ops: Optional[List[dict]] = None   # detail ops re-projected onto the new surface


@app.post("/snap_sdf", response_model=SnapResp)
def snap_sdf(req: SnapSdfReq):
    """Volume-native generative snap: base 64^3 SDF volume + primitive EditOps (cube coords) ->
    SDEdit massing prior -> a NEW 64^3 volume in the SAME cube frame (reload + keep sculpting)."""
    import base64 as _b
    import numpy as np
    import torch
    r = refiner()
    try:
        grid = np.frombuffer(_b.b64decode(req.base_sdf_b64.split(",")[-1]),
                             dtype="<f4").reshape(req.res, req.res, req.res).copy()
    except Exception as ex:
        raise HTTPException(400, f"bad base_sdf_b64: {ex}")
    try:
        snapped, iou = r.snap_volume(grid, req.edits, strength=req.strength,
                                     steps=req.sdedit_steps, autoguidance=req.autoguidance,
                                     auto_scale=req.auto_scale, local=req.local)
    except Exception as ex:
        raise HTTPException(400, f"snap failed: {ex}")
    mesh_b64 = None
    if req.return_mesh:
        try:
            grid_out = snapped
            if req.detail_edits:   # re-apply the crisp semantic ops on the snapped massing @96^3
                from refine import volume_to_sdf
                from scene.sdf_edit import EditableBuilding, EditOp
                from scene.sdf_primitives import sample_grid
                comp = EditableBuilding(volume_to_sdf(snapped, r.device),
                                        [EditOp.from_dict(d) for d in req.detail_edits]).composed()
                grid_out = sample_grid(comp, 96, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                                       device=r.device).cpu().numpy()
            if req.center is not None and req.scale is not None:
                mesh = r.volume_to_world_mesh(grid_out, req.center, req.scale,
                                              building_class=req.building_class, style=req.style,
                                              seed=req.seed, detail=req.detail)
            else:
                from scene.sdf_primitives import grid_to_mesh
                mesh = grid_to_mesh(torch.from_numpy(snapped), (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), iso=0.0)
            if mesh is not None and len(mesh.faces):
                mesh_b64 = _b64(engine().mesh_to_glb(mesh))
        except Exception as ex:
            raise HTTPException(400, f"bake mesh failed: {ex}")
    resnapped = None
    if req.resnap_detail_ops:
        from layout_detail import resnap_ops_to_surface, adjust_ops_after_snap
        massing_changed = bool(req.edits) or not req.local
        try:
            if req.adjust and massing_changed:
                # walls moved: geometric re-seat + row regularization (NO learned recohere)
                resnapped, _dropped = adjust_ops_after_snap(snapped, req.resnap_detail_ops,
                                                            device=r.device)
            elif massing_changed:
                resnapped = resnap_ops_to_surface(snapped, req.resnap_detail_ops)
            else:                                      # no-op snap: details must not move
                resnapped = req.resnap_detail_ops
        except Exception:
            resnapped = req.resnap_detail_ops          # fall back to unmoved ops
    return SnapResp(sdf_b64=_b64(snapped.astype("<f4").tobytes()), res=int(snapped.shape[0]),
                    iou_to_edit=float(iou), mesh_glb_b64=mesh_b64, resnapped_ops=resnapped)


class DetailVolumeReq(BaseModel):
    base_sdf_b64: str                   # cube-frame massing volume
    res: int = 64
    center: List[float]
    scale: float
    building_class: str = "RESIDENTIAL"
    style: str = "modern"
    seed: Optional[int] = None
    detail_edits: List[dict] = Field(default_factory=list)  # user's crisp detail ops
    res_out: int = 96


@app.post("/detail_volume")
def detail_volume(req: DetailVolumeReq):
    """LIVE DETAIL PREVIEW: the bake-quality ② treatment (windows/bands/plinth/roof/
    landmarks + the user's detail ops) composed on the current massing, returned as a
    cube-frame volume the viewer raymarches directly — what you'd get from Bake, live."""
    import base64 as _b
    import numpy as np
    r = refiner()
    try:
        grid = np.frombuffer(_b.b64decode(req.base_sdf_b64.split(",")[-1]),
                             dtype="<f4").reshape(req.res, req.res, req.res).copy()
    except Exception as ex:
        raise HTTPException(400, f"bad base_sdf_b64: {ex}")
    try:
        if req.detail_edits:   # user ops first (same order as the bake)
            from refine import volume_to_sdf
            from scene.sdf_edit import EditableBuilding, EditOp
            from scene.sdf_primitives import sample_grid
            import torch as _t
            comp = EditableBuilding(volume_to_sdf(grid, r.device),
                                    [EditOp.from_dict(d) for d in req.detail_edits]).composed()
            grid = sample_grid(comp, req.res, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                               device=r.device).cpu().numpy()
        out = r.detail_cube_volume(grid, req.center, req.scale,
                                   building_class=req.building_class, style=req.style,
                                   seed=req.seed, res_out=req.res_out,
                                   detail_edits=req.detail_edits)
    except Exception as ex:
        raise HTTPException(400, f"detail volume failed: {ex}")
    return {"sdf_b64": _b64(out.astype("<f4").tobytes()), "res": int(out.shape[0])}


class NeuralRenderReq(BaseModel):
    base_sdf_b64: str                   # cube-frame massing volume
    res: int = 64
    center: List[float]
    scale: float
    building_class: str = "RESIDENTIAL"
    style: str = "modern"
    style_ref_b64: Optional[str] = None  # reference IMAGE (png/jpg b64) — the per-building
                                         # style embedding source; None = prompt-only
    prompt: Optional[str] = None
    seed: int = 7
    steps: int = 30
    detail_edits: List[dict] = Field(default_factory=list)


@app.post("/neural_render")
def neural_render(req: NeuralRenderReq):
    """PHOTOREAL render of the current building: bake-quality detail volume -> G-buffers ->
    SDXL multi-ControlNet (+ IP-Adapter when a style reference image is given). Geometry
    stays ours and crisp; the diffusion model only paints pixels. First call loads SDXL
    (~30-60s); after that ~10-15s per render."""
    import base64 as _b
    import io as _io
    import numpy as np
    r = refiner()
    try:
        grid = np.frombuffer(_b.b64decode(req.base_sdf_b64.split(",")[-1]),
                             dtype="<f4").reshape(req.res, req.res, req.res).copy()
    except Exception as ex:
        raise HTTPException(400, f"bad base_sdf_b64: {ex}")
    try:
        if req.detail_edits:
            from refine import volume_to_sdf
            from scene.sdf_edit import EditableBuilding, EditOp
            from scene.sdf_primitives import sample_grid
            comp = EditableBuilding(volume_to_sdf(grid, r.device),
                                    [EditOp.from_dict(d) for d in req.detail_edits]).composed()
            grid = sample_grid(comp, req.res, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                               device=r.device).cpu().numpy()
        grid96 = r.detail_cube_volume(grid, req.center, req.scale,
                                      building_class=req.building_class, style=req.style,
                                      res_out=96, detail_edits=req.detail_edits)
        import neural_appearance as na
        ref = None
        if req.style_ref_b64:
            from PIL import Image
            ref = Image.open(_io.BytesIO(
                _b.b64decode(req.style_ref_b64.split(",")[-1]))).convert("RGB")
        img = na.render_building(grid96, style=req.style, building_class=req.building_class,
                                 style_ref=ref, prompt=req.prompt, seed=req.seed,
                                 steps=req.steps)
        return {"image_b64": na.png_b64(img)}
    except Exception as ex:
        raise HTTPException(400, f"neural render failed: {ex}")


class TownRenderBuilding(BaseModel):
    footprint: List[List[float]]         # local, centered (m)
    position: List[float]                # [x, z] world (m)
    style: str = "modern"
    building_class: str = "RESIDENTIAL"
    height: float = 10.0
    recipe_params: List[float]
    edits: List[dict] = Field(default_factory=list)        # sculpt ops (world frame)
    style_ref_b64: Optional[str] = None                    # per-building style image


class NeuralRenderTownReq(BaseModel):
    buildings: List[TownRenderBuilding]
    prompt: Optional[str] = None
    seed: int = 11
    steps: int = 30


@app.post("/neural_render_town")
def neural_render_town(req: NeuralRenderTownReq):
    """PHOTOREAL TOWN render with PER-BUILDING style references: each building is rebuilt
    exactly (recipe params + sculpt edits + composer detail), the scene is traced as one
    SDF, and instance masks apply each building's reference image only to ITS pixels."""
    import base64 as _b
    import io as _io
    import numpy as np
    import torch as _t
    r = refiner()
    try:
        import neural_appearance as na
        from refine import _bbox
        from scene.sdf_edit import recipe_base_sdf, EditableBuilding, EditOp
        from PIL import Image
        items, refs = [], []
        ref_cache = {}
        for b in req.buildings[:12]:                       # latency cap
            base = recipe_base_sdf(b.style, b.recipe_params, b.footprint, b.height,
                                   device=r.device)
            if b.edits:
                base = EditableBuilding(base, [EditOp.from_dict(d) for d in b.edits]).composed()
            bbox = _bbox(b.footprint, b.height, b.edits)
            sdf_t, _fp, _hn, c, s = r._recipe_to_frame_n(base, bbox, margin=1.3)
            grid = sdf_t[0, 0].detach().cpu().numpy().astype(np.float32)
            g96 = r.detail_cube_volume(grid, c, s, building_class=b.building_class,
                                       style=b.style, res_out=96)
            items.append({"vol": _t.as_tensor(g96, device=r.device)[None, None],
                          "center": _t.as_tensor(np.asarray(c, np.float32), device=r.device),
                          "scale": float(s),
                          "pos": _t.tensor([float(b.position[0]), 0.0,
                                            float(b.position[1])], device=r.device)})
            ref = None
            if b.style_ref_b64:
                key = b.style_ref_b64[:64]
                if key not in ref_cache:
                    ref_cache[key] = Image.open(_io.BytesIO(
                        _b.b64decode(b.style_ref_b64.split(",")[-1]))).convert("RGB")
                ref = ref_cache[key]
            refs.append(ref)
        img = na.render_town(items, refs, prompt=req.prompt, seed=req.seed,
                             steps=req.steps)
        return {"image_b64": na.png_b64(img), "n_buildings": len(items)}
    except Exception as ex:
        raise HTTPException(400, f"town render failed: {ex}")


class BakeTextureReq(BaseModel):
    base_sdf_b64: str                   # cube-frame massing volume (sculptor st.base_b64)
    res: int = 64
    center: List[float]
    scale: float                        # world half-extent (m) of the cube frame
    building_class: str = "RESIDENTIAL"
    style: str = "modern"
    style_ref_b64: Optional[str] = None
    prompt: Optional[str] = None
    detail_edits: List[dict] = Field(default_factory=list)
    seed: int = 7
    n_views: int = 6
    steps: int = 28
    unit: float = 100.0                  # 100 = cm (Unreal), 1 = meters
    iterative: bool = True               # TEXTure-style seam-free (inpaint each new view)


@app.post("/bake_texture")
def bake_texture(req: BakeTextureReq):
    """v2: bake a TEXTURED glb of the current building — UV unwrap + multi-view diffusion
    (depth+edge ControlNet, optional style image) back-projected into an albedo atlas. The
    building imports into Unreal already wearing its style as a material. ~1 min."""
    import base64 as _b
    import io as _io
    import numpy as np
    r = refiner()
    try:
        grid = np.frombuffer(_b.b64decode(req.base_sdf_b64.split(",")[-1]),
                             dtype="<f4").reshape(req.res, req.res, req.res).copy()
        if req.detail_edits:
            from refine import volume_to_sdf
            from scene.sdf_edit import EditableBuilding, EditOp
            from scene.sdf_primitives import sample_grid
            comp = EditableBuilding(volume_to_sdf(grid, r.device),
                                    [EditOp.from_dict(d) for d in req.detail_edits]).composed()
            grid = sample_grid(comp, req.res, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                               device=r.device).cpu().numpy()
        grid96 = r.detail_cube_volume(grid, req.center, req.scale,
                                      building_class=req.building_class, style=req.style,
                                      res_out=96, detail_edits=req.detail_edits)
        import neural_appearance as na
        import texture_bake as tb
        ref = None
        if req.style_ref_b64:
            from PIL import Image
            ref = Image.open(_io.BytesIO(
                _b.b64decode(req.style_ref_b64.split(",")[-1]))).convert("RGB")
        pr = req.prompt or (f"photo of a {req.style} {req.building_class.lower()} building, "
                            "architectural photography, high detail")
        ipipe = na.get_inpaint_pipe() if req.iterative else None
        glb, cov, nv = tb.bake_glb(grid96, na.get_pipe(), pr, style_ref=ref, seed=req.seed,
                                   n_views=req.n_views, steps=req.steps,
                                   world_scale=req.scale, unit=req.unit, style=req.style,
                                   inpaint_pipe=ipipe)
        return {"glb_b64": _b.b64encode(glb).decode(), "coverage": round(cov, 3),
                "n_vertices": nv}
    except Exception as ex:
        raise HTTPException(400, f"bake texture failed: {ex}")


class ExportBuilding(BaseModel):
    footprint: List[List[float]]
    position: List[float]
    style: str = "modern"
    building_class: str = "RESIDENTIAL"
    height: float = 10.0
    recipe_params: List[float]
    edits: List[dict] = Field(default_factory=list)
    weather: float = 0.0                 # Layer 2.5a procedural aging (geometry export path)
    weather_seed: Optional[int] = None
    ornaments: List[dict] = Field(default_factory=list)  # Layer 2.5b relief instances
    style_ref_b64: Optional[str] = None  # per-building style image (textured export only)
    prompt: Optional[str] = None


class ExportTownReq(BaseModel):
    buildings: List[ExportBuilding]
    scale: float = 100.0                 # 100 = centimeters (Unreal), 1 = meters (Blender)
    ground: bool = True
    textures: bool = False               # v2: bake a per-building albedo texture (SLOW)
    n_views: int = 5
    steps: int = 22
    iterative: bool = True               # TEXTure-style seam-free bake


@app.post("/export_town")
def export_town_ep(req: ExportTownReq):
    """Export the whole town as ONE glTF scene for Unreal / Blender — each building rebuilt
    exactly (params + sculpt edits + composer detail), placed at its world position as a
    named node + a ground plane. textures=False -> v1 geometry (gray); textures=True -> v2
    per-building albedo bake (multi-view diffusion, SLOW). Returns glb + manifest."""
    import base64 as _b
    r = refiner()
    try:
        from town_export import export_town, export_town_textured
        if req.textures:
            import neural_appearance as na
            bs = [b.model_dump() for b in req.buildings[:12]]   # bake cost cap
            ipipe = na.get_inpaint_pipe() if req.iterative else None
            glb, manifest, nv = export_town_textured(
                r, bs, na.get_pipe(), unit=req.scale, ground=req.ground,
                n_views=req.n_views, steps=req.steps, inpaint_pipe=ipipe)
        else:
            bs = [b.model_dump() for b in req.buildings[:64]]
            glb, manifest, nv = export_town(r, bs, scale=req.scale, ground=req.ground)
        return {"glb_b64": _b.b64encode(glb).decode(), "manifest": manifest,
                "n_buildings": manifest["n_buildings"], "n_vertices": nv}
    except Exception as ex:
        raise HTTPException(400, f"export failed: {ex}")


class OrnamentBuildingReq(BaseModel):
    footprint: List[List[float]]
    style: str = "modern"
    height: float = 10.0
    seed: Optional[int] = None


@app.post("/ornament_building")
def ornament_building(req: OrnamentBuildingReq):
    """Layer 2.5b: RETRIEVE a culturally-matched heritage-scan relief from
    data/ornaments_v1 and FIT it to the building's main wall (scale/yaw/sink). Returns the
    SYMBOLIC instance ({id, edge, t, y, w}) — append to the building's `ornaments` and
    rebuild via /rebuild_building; the mesh instance is merged procedurally, the diffusion
    prior never touches it."""
    from ornaments import propose
    try:
        inst = propose(req.footprint, req.height, req.style, seed=req.seed)
    except Exception as ex:
        raise HTTPException(400, f"ornament failed: {ex}")
    return {"ornament": inst}


class InterpretMassReq(BaseModel):
    base_sdf_b64: str                   # cube-frame massing volume
    res: int = 64
    op: dict                            # the raw placed EditOp (cube coords)
    building_class: str = "RESIDENTIAL"
    style: str = "modern"
    seed: Optional[int] = None          # sampled construction: same box, different seed ->
                                        # different plausible architecture
    temperature: float = 0.9            # planner typing temperature
    existing_ops: List[dict] = Field(default_factory=list)  # current detail layout, optional:
                                        # when given, a WALL-RHYTHM result (window/door/balcony)
                                        # gets an extra CoherentPartRefiner pass so it aligns
                                        # with same-type neighbors (2026-07-02, layout_detail.
                                        # integrate_new_part); omit/empty -> unchanged behavior.


def _interpret_and_integrate(grid, op, existing_ops, building_class, style, seed, temperature):
    """Shared classify -> typed-construction (+ CoherentPartRefiner alignment) core of
    /interpret_mass and /interpret_mass_world. Cube frame in, cube frame out."""
    from layout_detail import interpret_mass, integrate_new_part
    out = interpret_mass(grid, op, building_class=building_class, style=style,
                         seed=seed, temperature=temperature)
    out["coherent"] = False
    if existing_ops and out.get("ops"):
        n_new = len(out["ops"])
        merged, used = integrate_new_part(grid, existing_ops, out["ops"],
                                          building_class=building_class,
                                          device=refiner().device)
        if used:
            # integrate_new_part appends the new construction's group LAST (existing groups
            # keep their input order first) -> the trailing n_new ops are the refined result.
            out["ops"], out["coherent"] = merged[-n_new:], True
    return out


@app.post("/interpret_mass")
def interpret_mass_ep(req: InterpretMassReq):
    """SMART ADD: make sense of a placed mass — classify it as an architectural part
    (tower/dormer/chimney/balcony/bay/wing/window/door) from its shape + position on the
    massing, and return a crisp typed construction (EditOps) to apply instead. If existing_ops
    is given and the result is a wall-rhythm type, CoherentPartRefiner additionally aligns it
    with same-type neighbors (row/spacing/wall-attach) before returning."""
    import base64 as _b
    import numpy as np
    try:
        grid = np.frombuffer(_b.b64decode(req.base_sdf_b64.split(",")[-1]),
                             dtype="<f4").reshape(req.res, req.res, req.res).copy()
    except Exception as ex:
        raise HTTPException(400, f"bad base_sdf_b64: {ex}")
    try:
        out = _interpret_and_integrate(grid, req.op, req.existing_ops, req.building_class,
                                       req.style, req.seed, req.temperature)
    except Exception as ex:
        raise HTTPException(400, f"interpret failed: {ex}")
    return {"kind": out["kind"], "n": len(out["ops"]), "ops": out["ops"],
            "source": out.get("source", "rules"), "p_types": out.get("p_types", {}),
            "coherent": out["coherent"]}


# -- world-meter frame bridge for the town page (index.html) ----------------
# index.html buildings are symbolic state (recipe params + a world-meter edit list), not a
# cached cube SDF like sculpt.html's — ops must be converted through the SAME (center, scale)
# the massing volume is built with, or they land outside the sampled cube (see
# test_coherent_add_bake.world_box_op, where this bridge was first proven).

def _op_scale_size(kind, size, f):
    """Scale an EditOp size by factor f. Size layouts are kind-specific (scene/sdf_edit
    ._primitive): box/rounded_box=[hx,hy,hz], sphere=[r], cylinder=[r,h],
    gable/hip=[w,d,body_h,roof_h] — all lengths; cone=[angle_deg, height] — the ANGLE is
    scale-invariant, only the height converts."""
    vals = [float(v) for v in size]
    if kind == "cone":
        return [vals[0]] + [v * f for v in vals[1:]]
    return [v * f for v in vals]


def _op_world_to_cube(op, c, s):
    o = dict(op)
    o["center"] = [(float(v) - float(cv)) / s for v, cv in zip(op["center"], c)]
    o["size"] = _op_scale_size(op.get("kind", "box"), op.get("size", [1, 1, 1]), 1.0 / s)
    for k in ("smooth", "round_r"):
        if op.get(k):
            o[k] = float(op[k]) / s
    return o


def _op_cube_to_world(op, c, s):
    o = dict(op)
    o["center"] = [float(v) * s + float(cv) for v, cv in zip(op["center"], c)]
    o["size"] = _op_scale_size(op.get("kind", "box"), op.get("size", [1, 1, 1]), s)
    for k in ("smooth", "round_r"):
        if op.get(k):
            o[k] = float(op[k]) * s
    return o


class InterpretMassWorldReq(BaseModel):
    footprint: List[List[float]]        # local building meters (the town page's b.footprint)
    style: str = "modern"
    building_class: str = "RESIDENTIAL"
    height: float = 10.0
    recipe_params: List[float]
    op: dict                            # the raw placed EditOp, WORLD meters (same frame as
                                        # footprint; y=0 is the ground)
    existing_ops: List[dict] = Field(default_factory=list)  # the building's current edit list
                                        # (world meters): mass ops (no det tag) are composed
                                        # into the massing the new op is classified against;
                                        # det-tagged ops feed the coherence pass.
    seed: Optional[int] = None
    temperature: float = 0.9


@app.post("/interpret_mass_world")
def interpret_mass_world(req: InterpretMassWorldReq):
    """SMART ADD for the town page: same classify -> typed-construction (+ coherent align)
    as /interpret_mass, but in world meters. Rebuilds the base massing volume from the
    building's recipe state, bridges the op through the volume's (center, scale), and returns
    the construction back in world meters — ready to append to the building's edit list and
    rebuild procedurally via /rebuild_building (no diffusion anywhere on this path)."""
    if req.style not in ps.STYLE_TO_IDX:
        raise HTTPException(400, f"unknown style '{req.style}'")
    r = refiner()
    try:
        grid, c, s, _hn = r.building_volume(req.footprint, req.style, req.recipe_params,
                                            req.height, res=64)
    except Exception as ex:
        raise HTTPException(400, f"base massing failed: {ex}")
    s = float(s)
    try:
        cube_exist = [_op_world_to_cube(o, c, s) for o in req.existing_ops]
        mass_ops = [o for o in cube_exist if not o.get("det")]
        detail_ops = [o for o in cube_exist if o.get("det")]
        if mass_ops:   # earlier adds are part of the massing the new op is typed against
            from refine import volume_to_sdf
            from scene.sdf_edit import EditableBuilding, EditOp
            from scene.sdf_primitives import sample_grid
            comp = EditableBuilding(volume_to_sdf(grid, r.device),
                                    [EditOp.from_dict(d) for d in mass_ops]).composed()
            grid = sample_grid(comp, 64, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                               device=r.device).cpu().numpy()
        out = _interpret_and_integrate(grid, _op_world_to_cube(req.op, c, s), detail_ops,
                                       req.building_class, req.style, req.seed,
                                       req.temperature)
        world_ops = [_op_cube_to_world(o, c, s) for o in out["ops"]]
    except Exception as ex:
        raise HTTPException(400, f"interpret failed: {ex}")
    return {"kind": out["kind"], "n": len(world_ops), "ops": world_ops,
            "source": out.get("source", "rules"), "p_types": out.get("p_types", {}),
            "coherent": out["coherent"]}


class RebuildBuildingReq(BaseModel):
    footprint: List[List[float]]
    style: str = "modern"
    building_class: str = "RESIDENTIAL"
    height: float = 10.0
    recipe_params: List[float]
    edits: List[dict] = Field(default_factory=list)   # world-meter EditOps (typed
                                                      # constructions from /interpret_mass_world
                                                      # or crude sculpt ops — both are pure CSG)
    weather: float = 0.0                              # Layer 2.5a procedural aging, 0..1
    weather_seed: Optional[int] = None
    ornaments: List[dict] = Field(default_factory=list)  # Layer 2.5b heritage-relief
                                                      # instances ({id, edge, t, y, w})
    res: int = 96


@app.post("/rebuild_building", response_model=MeshResp)
def rebuild_building(req: RebuildBuildingReq):
    """PROCEDURAL-ONLY rebuild of one town building: recipe base + CSG edits + composer
    detail -> mesh, the exact per-building path /export_town uses (town_export.
    build_building_mesh). The town page's Make-it-architecture and Undo regenerate through
    this, so placed constructions can never be remolded by the diffusion prior."""
    if req.style not in ps.STYLE_TO_IDX:
        raise HTTPException(400, f"unknown style '{req.style}'")
    from town_export import build_building_mesh
    try:
        mesh = build_building_mesh(refiner(), {
            "footprint": req.footprint, "style": req.style, "height": req.height,
            "building_class": req.building_class, "recipe_params": req.recipe_params,
            "edits": req.edits, "weather": req.weather,
            "weather_seed": req.weather_seed, "ornaments": req.ornaments}, res=req.res)
    except Exception as ex:
        raise HTTPException(400, f"rebuild failed: {ex}")
    if mesh is None or not len(mesh.faces):
        raise HTTPException(400, "rebuild produced an empty mesh")
    return MeshResp(style=req.style, recipe_params=list(map(float, req.recipe_params)),
                    mesh_glb_b64=_b64(engine().mesh_to_glb(mesh)),
                    n_vertices=len(mesh.vertices), n_faces=len(mesh.faces),
                    position_xz=[0.0, 0.0])


class ProposeDetailsReq(BaseModel):
    base_sdf_b64: str                   # cube-frame massing volume (from /building_sdf or a snap)
    res: int = 64
    building_class: str = "RESIDENTIAL"
    temperature: float = 0.7
    max_ops: int = 14
    seed: Optional[int] = None          # deterministic proposals (tests)


@app.post("/propose_details")
def propose_details(req: ProposeDetailsReq):
    """LEARNED detail proposal: the part-layout planner samples typed part boxes for this
    massing, snap-to-surface projects them, returns them as sculptor detail EditOps."""
    import base64 as _b
    import numpy as np
    from layout_detail import propose_detail_ops
    try:
        grid = np.frombuffer(_b.b64decode(req.base_sdf_b64.split(",")[-1]),
                             dtype="<f4").reshape(req.res, req.res, req.res).copy()
        ops = propose_detail_ops(grid, building_class=req.building_class,
                                 device=refiner().device, temperature=req.temperature,
                                 max_ops=req.max_ops, seed=req.seed)
    except Exception as ex:
        raise HTTPException(400, f"propose_details failed: {ex}")
    return {"ops": ops, "n": len(ops)}


class RecohereReq(BaseModel):
    base_sdf_b64: str
    res: int = 64
    ops: List[dict] = Field(default_factory=list)
    strength: float = 0.2


@app.post("/recohere_details")
def recohere_details(req: RecohereReq):
    """LEARNED re-coherence (set refiner): jointly denoise the current detail layout against
    the massing — drops implausible parts, adjusts poses; surface-snap as the safety net."""
    import base64 as _b
    import numpy as np
    from layout_detail import recohere_ops
    try:
        grid = np.frombuffer(_b.b64decode(req.base_sdf_b64.split(",")[-1]),
                             dtype="<f4").reshape(req.res, req.res, req.res).copy()
        ops, dropped = recohere_ops(grid, req.ops, device=refiner().device,
                                    strength=req.strength)
    except Exception as ex:
        raise HTTPException(400, f"recohere failed: {ex}")
    return {"ops": ops, "n": len(ops), "dropped": dropped}


class RecohereFacadeReq(BaseModel):
    style: str = "modern"
    building_class: Optional[str] = None
    params: Optional[List[float]] = None   # current 12-dim DetailParams vec; None -> fresh sample
    strength: float = 0.6                  # 0 = keep current facade, 1 = fresh coherent program
    seed: Optional[int] = None


@app.post("/recohere_facade")
def recohere_facade_ep(req: RecohereFacadeReq):
    """GENERATIVE facade re-coherence (program-space). Samples/corrects a DetailParams facade
    PROGRAM with the trained detail head via SDEdit at `strength`; every output is
    aligned-by-construction (domain-repetition windows). Returns the corrected 12-dim param
    vector + named fields; the caller renders it crisp with scene.sdf_detail.add_facade_detail."""
    import facade_recohere as fr
    from scene import sdf_detail as det
    try:
        p = fr.recohere_facade(req.params, style=req.style, strength=req.strength,
                               seed=req.seed, device=refiner().device,
                               building_class=req.building_class)
        vec = fr.params_to_vec(p).tolist()
    except Exception as ex:
        raise HTTPException(400, f"recohere_facade failed: {ex}")
    return {"params": vec, "fields": det.DETAIL_FIELDS,
            "named": {f: float(v) for f, v in zip(det.DETAIL_FIELDS, vec)}}
