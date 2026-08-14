"""#99 -- direct A2/DoraCodec generation service for the town editor (map #97).

Bypasses recipe_inference.generate_building entirely: this calls the A2 vecset denoiser
(scripts/train_vecset.py) through DoraCodec directly, the same way scripts/foundations/
eval_massing_arms.py does for evaluation -- that script is the only known-working reference
for this pipeline, so every conditioning/decode step below reuses its functions rather than
reimplementing them. This repo has a documented history (see models/shape_codec.py's Building
docstring, issue #70) of frame-convention mistakes here silently corrupting two whole training
runs, so the array-frame axis order established by blockout_sdf / mesh_sdf_surface /
verts_to_world is treated as load-bearing, not decorative -- new code composes with it, never
guesses an "equivalent" reordering.

Housekeeping: the footprint editor's polygon lives in real-world meters, the model works in a
per-building-normalized frame keyed on max(footprint extent, height) -- see
scripts/ingest_3dbag.py:building_to_sdf, replicated here for a footprint+height pair since
there's no real mesh to normalize from for a hand-drawn/imported building.

Two endpoints, because a town is not one building N times:

  /generate_building  one building, one JSON response.
  /generate_town      N buildings in one call, **streamed** as NDJSON, one record per building as
                      it finishes. Generation is per-building at ~10s each (measured, see #99), so
                      a 29-building import -- the Munich Altstadt preset -- is ~5 minutes. Behind a
                      single blocking response that is an indistinguishable-from-hung demo; streamed,
                      the town builds up in front of the user and the client can show real progress.
                      Per this map's Notes there is no cross-building conditioning: the batch is a
                      loop, and is only an endpoint (rather than N client calls) so the contract
                      says so, and so the town's total cost is measured in one place.

Run (dev):
  ./venv/bin/uvicorn scripts.server.town_generate_service:app --port 8767
Then open http://localhost:8767/ -- this service also serves the editor page itself, so the demo
does not need inference_service.py (whose startup loads the whole legacy engine, ~8 min) at all.
Image import is a third service (footprint_extract_service.py, port 8766), still separate.
"""

from __future__ import annotations

import json
import inspect
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.vecset_ceiling_probe import RES, TRUNC, verts_to_world       # noqa: E402
from scripts.foundations.eval_massing_arms import blockout_sdf, vs_input as _vs_input  # noqa: E402
from scripts.foundations.baseline_gate_eval import mesh_sdf_surface                    # noqa: E402
from scripts.foundations.dora_roundtrip_probe import load_dora                         # noqa: E402
from models.shape_codec import Building, DoraCodec                                     # noqa: E402
from models.networks.vecset_denoiser import VecsetDenoiser                             # noqa: E402
from models.networks.vecset_projection import SetSDEdit                                # noqa: E402
from scripts.ingest_3dbag import building_to_sdf                                       # noqa: E402
from scripts.foundations.ingest_citygml_lod2 import SOURCE_ID                          # noqa: E402

A2_CKPT = REPO / "weights/massing-vecset/vecset_v4_surf.pth"
WEB = Path(__file__).resolve().parent / "web"
SPACING = 2.0 / (RES - 1)
# Read the corpus margin FROM the function that built the corpus rather than restating 1.05 here.
# A drifted copy would rescale every generated building by a few percent against the frame the model
# was trained in, and would present as a model problem rather than as the transcription error it is.
MARGIN = inspect.signature(building_to_sdf).parameters["margin"].default
# Measured on the AMD box (#99, re-measured after #101's decode-chunk change): ~7.0s/building warm,
# of which the Dora grid decode is ~3.2s and A2's own 20 denoise steps ~3.1s. Generation is
# per-building, so a town costs this times N.
SECS_PER_BUILDING = 7
# How many grid points to push through the frozen decoder at once. ShapeCodec's default of 32768
# takes 5.84s for a 64^3 grid here; 131072 takes 3.24s for a **bit-identical** field (occupancy IoU
# 1.000000 -- chunking only groups point queries that are independent of one another, so this buys
# time and nothing else). 262144 measured no better, so this is the knee, not the maximum.
# Set on the codec INSTANCE rather than the class: the eval harnesses share ShapeCodec and should
# keep their own memory profile. Lower it if a smaller GPU runs out of memory. (#101)
DECODE_CHUNK = 131072
# A guard rail, not a capability claim: an accidental 500-footprint import would occupy the GPU for
# over an hour with no way to call it back. The Munich Altstadt preset is 29.
MAX_BUILDINGS = 120

app = FastAPI(title="Town generation via A2 (map #97, #99)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state = {}


@app.on_event("startup")
def _load_models():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    codec = DoraCodec(load_dora(dev))
    codec.query_chunk = DECODE_CHUNK
    ck = torch.load(A2_CKPT, map_location="cpu", weights_only=False)
    ca = ck["args"]
    net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=ca["width"],
                         depth=ca["depth"], heads=ca["heads"],
                         footprint_res=ck["footprint_res"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    op = SetSDEdit(net, timesteps=ca["timesteps"])
    _state.update(dev=dev, codec=codec, op=op, mu=ck["latent_mu"], sd=ck["latent_sd"],
                  step=int(ck["step"]))
    print(f"[town_generate] A2 step {_state['step']} + DoraCodec loaded on {dev} "
          f"({time.time()-t0:.0f}s)", flush=True)


class ProjectionKnobs(BaseModel):
    """The C1 projection's settings, shared by both endpoints so they cannot drift apart."""

    region: int = SOURCE_ID["nl"]      # the conditioning corpus a building claims to come from. A
                                        # hand-drawn footprint carries no such signal, so it defaults
                                        # to the majority training source (see ingest_citygml_lod2).
    strength: float = 0.5              # matches map-#86's evidence number; <0.35 is out-of-distribution
                                        # for this checkpoint (trained with pair_t_min=0.35)
    steps: int = 20
    guidance: float = 1.0
    seed: Optional[int] = None


class GenerateReq(ProjectionKnobs):
    points: List[List[float]]          # [[x,z], ...] world meters, the drawn/imported footprint
    height: float                      # world meters


class GenerateResp(BaseModel):
    vertices: List[List[float]]        # world meters, y-up, ready for a three.js BufferGeometry
    faces: List[List[int]]
    vs_input: float                    # overlap with the blockout this projected from; near 1.0
                                        # means A2 barely edited it -- report this, don't hide it
    gen_seconds: float


class TownBuilding(BaseModel):
    points: List[List[float]]          # [[x,z], ...] world meters
    height: Optional[float] = None     # None -> the town's default_height (map #97's stated
                                        # fallback: height is a required input, defaulted, never
                                        # inferred -- model-based inference is #82's problem)


class TownReq(ProjectionKnobs):
    buildings: List[TownBuilding]
    default_height: float = 12.0
    # `seed` (inherited) is decorrelated per building as `seed * 1000003 + index`, matching
    # eval_massing_arms.py's convention. One shared draw across a whole town would correlate every
    # building's noise, which is visible as repetition once there are more than a few of them.


def _footprint_normalization(points: np.ndarray, height: float):
    """Per-building frame, replicating scripts/ingest_3dbag.py:building_to_sdf's margin/scale
    convention for a footprint+height pair (no real mesh to normalize from here)."""
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    z0, z1 = points[:, 1].min(), points[:, 1].max()
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    s = max(x1 - x0, z1 - z0, height) / 2 * MARGIN
    if s <= 0:
        raise HTTPException(400, "degenerate footprint/height (zero extent)")
    return s, cx, cz


def _rasterize_footprint(points: np.ndarray, s: float, cx: float, cz: float) -> np.ndarray:
    """World-meter polygon -> (RES,RES) uint8 mask in blockout_sdf's [D,W] = [z,x] axis order."""
    from skimage.draw import polygon2mask
    nx = (points[:, 0] - cx) / s
    nz = (points[:, 1] - cz) / s
    row = np.clip(np.round((nz + 1) / SPACING), 0, RES - 1).astype(int)   # D axis
    col = np.clip(np.round((nx + 1) / SPACING), 0, RES - 1).astype(int)   # W axis
    mask = polygon2mask((RES, RES), np.stack([row, col], axis=1))
    return mask.astype(np.uint8)


def _height_voxel_range(height: float, s: float) -> tuple[int, int]:
    """The footprint sits on the ground (world z=0..height); building_to_sdf centers the full
    bbox, so after centering the vertical extent is symmetric around 0."""
    half = height / (2 * s)
    y0 = int(np.clip(round((-half + 1) / SPACING), 0, RES - 1))
    y1 = int(np.clip(round((half + 1) / SPACING), 0, RES - 1))
    if y1 <= y0:
        y1 = min(y0 + 1, RES - 1)
    return y0, y1


def _require_models():
    if "codec" not in _state:
        raise HTTPException(503, "models still loading")


def _generate_one(points: Sequence[Sequence[float]], height: float, knobs: ProjectionKnobs,
                  seed: Optional[int]) -> dict:
    """One building, world-meter polygon in, world-meter mesh out. Raises HTTPException on refusal.

    Shared verbatim by both endpoints, so the batch path cannot drift from the single path -- the
    town is exactly its buildings generated one at a time. `seed` is passed separately from `knobs`
    because the batch path derives a per-building seed from the request's one.
    """
    if len(points) < 3:
        raise HTTPException(400, "need at least 3 footprint points")
    if height <= 0:
        raise HTTPException(400, "height must be positive")

    t0 = time.time()
    dev, codec, op = _state["dev"], _state["codec"], _state["op"]
    mu, sd = _state["mu"], _state["sd"]

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise HTTPException(400, "footprint points must be [[x, z], ...]")
    s, cx, cz = _footprint_normalization(pts, height)
    fp = _rasterize_footprint(pts, s, cx, cz)
    y0, y1 = _height_voxel_range(height, s)

    bo = blockout_sdf(fp, y0, y1)
    if bo is None:
        raise HTTPException(400, "footprint rasterized to nothing (too small / too thin)")
    bv, bf = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
    if bv is None:
        raise HTTPException(400, "blockout produced no surface")

    with torch.no_grad():
        z0 = (codec.encode(Building(verts=verts_to_world(bv), faces=bf)).float()
              - mu) / sd
        fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
        ht = torch.tensor([height], device=dev)
        rg = torch.tensor([knobs.region], device=dev)
        zp = op.project(blockout=z0, footprint=fpt, height=ht, region=rg,
                        strength=knobs.strength, steps=knobs.steps, guidance=knobs.guidance,
                        seed=seed)
        fld = codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]

    verts, faces = mesh_sdf_surface(fld)
    if verts is None:
        raise HTTPException(500, "A2 output produced no surface (collapsed to empty/solid)")

    bo_occ = bo <= 0
    out_occ = fld <= 0
    vsi = _vs_input(out_occ, bo_occ)

    world = verts_to_world(verts)                       # (N,3) columns = [D,H,W] = [z,y,x], Frame-N
    out_verts = np.stack([
        world[:, 2] * s + cx,           # W(x) -> real world x
        world[:, 1] * s + height / 2,   # H(up) -> real world y (ground at 0)
        world[:, 0] * s + cz,           # D(z) -> real world z
    ], axis=1)

    return dict(
        vertices=out_verts.tolist(),
        faces=np.asarray(faces, dtype=int).tolist(),
        vs_input=vsi,
        gen_seconds=time.time() - t0,
    )


@app.post("/generate_building", response_model=GenerateResp)
def generate_building(req: GenerateReq):
    _require_models()
    return GenerateResp(**_generate_one(req.points, req.height, req, req.seed))


@app.post("/generate_town")
def generate_town(req: TownReq):
    """N buildings, one call, streamed as NDJSON -- one record per building, in request order.

    Records are `{"kind": "building", "index", "vertices", "faces", "vs_input", "gen_seconds"}`,
    `{"kind": "error", "index", "detail"}` for a footprint this service refuses (one bad polygon
    must not cost the user the other 28), and a final `{"kind": "done", ...}` carrying the totals.

    Errors are per-record rather than a failed response because the stream's status code is
    committed the moment the first byte leaves; anything that must fail the whole request has to be
    checked here, before the StreamingResponse is returned.
    """
    _require_models()
    if not req.buildings:
        raise HTTPException(400, "no buildings to generate")
    if len(req.buildings) > MAX_BUILDINGS:
        mins = len(req.buildings) * SECS_PER_BUILDING // 60
        raise HTTPException(400, f"{len(req.buildings)} buildings exceeds the {MAX_BUILDINGS} limit "
                                 f"(~{mins} min of GPU time)")
    if req.default_height <= 0:
        raise HTTPException(400, "default_height must be positive")

    def stream() -> Iterator[str]:
        t0, ok, failed = time.time(), 0, 0
        for i, b in enumerate(req.buildings):
            height = b.height if b.height is not None else req.default_height
            seed = None if req.seed is None else req.seed * 1000003 + i
            try:
                rec = _generate_one(b.points, height, req, seed)
                rec.update(kind="building", index=i, height=height)
                ok += 1
            except HTTPException as e:
                rec = dict(kind="error", index=i, detail=str(e.detail))
                failed += 1
            except Exception as e:                      # never let one building kill the town
                rec = dict(kind="error", index=i, detail=f"{type(e).__name__}: {e}")
                failed += 1
            yield json.dumps(rec) + "\n"
        yield json.dumps(dict(kind="done", count=len(req.buildings), ok=ok, failed=failed,
                              total_seconds=time.time() - t0)) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/health")
def health():
    return {"ok": "codec" in _state, "step": _state.get("step")}


# ---- the editor page itself, so the demo is one service and not the legacy engine's 8-min boot ----
# Guarded the way inference_service.py guards the same mount: a missing samples directory should
# cost the demo its preset buttons, not refuse to start the generator.
if (WEB / "samples").exists():
    app.mount("/samples", StaticFiles(directory=str(WEB / "samples")), name="samples")


@app.get("/")
def index():
    return FileResponse(WEB / "town.html", headers={"Cache-Control": "no-store"})
