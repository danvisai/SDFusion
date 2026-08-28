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

Since #127 it also serves a third endpoint and a second page:

  /compare_arms       one footprint carved by EVERY arm -- the footprint envelope, #127's height-map
                      generator read two ways, the zero-training retrieval baseline, and A2 -- in one
                      response, for a side-by-side visual judgement. ⚠️ A hand-drawn footprint has no
                      ground truth, so `missing`/`extra` cannot be computed there and are not
                      reported; the endpoint returns `vs_input` and the geometry.
  /arms               the page that drives it.
The height-map arms need no codec (a height map compiles straight to voxels), run in ~0.08s against
A2's ~1.1s, and are simply absent from `available` if their checkpoint or the corpus cache is not on
this box.

Run (dev):
  ./venv/bin/uvicorn scripts.server.town_generate_service:app --port 8767
Then open http://localhost:8767/ for the town editor, or http://localhost:8767/arms for #127's
comparison. This service also serves the editor pages itself, so the demo does not need
inference_service.py (whose startup loads the whole legacy engine, ~8 min) at all.
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
from scripts.foundations.recover_massing_programs import occ_to_field, occupancy       # noqa: E402
from scripts.foundations.measure_scoring_optimum import transplant_height              # noqa: E402
from scripts.foundations.train_height_map_generator import (                           # noqa: E402
    CACHE, DEPTH_CLASSES, apply_depth, build_model, condition_channels, decode_logits,
    envelope_depth, retrieve_nn,
)

A2_CKPT = REPO / "weights/massing-vecset/vecset_v4_surf.pth"
# #127's height-map generator. Weights live beside A2's under weights/, which is the staging tree
# that gets published; the training output is kept as a fallback so a fresh retrain is picked up
# without a copy step. Checkpoints are gitignored (*.pt), so the arm is simply absent when neither
# path exists, and `/health` says so.
HEIGHTMAP_CKPTS = (REPO / "weights/massing-heightmap/heightmap_ce.pt",
                   REPO / "outputs/height_map_generator/heightmap_ce.pt")
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

# Every massing generator this service can run, in reading order: what you start from, then what
# each arm does to it. `a2` is last because it is the incumbent being compared against, not the
# subject. `_arm_field` validates against this list, so an unknown arm is a 400 and never a silent
# fallback to some default -- a demo that quietly shows you a different model than you asked for is
# worse than one that refuses.
ARM_ORDER = ("envelope", "heightmap_mode", "heightmap_median", "retrieval", "a2")

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
    _load_heightmap_arms(dev)


def _load_heightmap_arms(dev) -> None:
    """#127's height-map generator and its retrieval baseline, loaded beside A2.

    Both are optional and neither touches the codec: a height map compiles straight to voxels, so
    this path works whether or not Dora is present. If the checkpoint or the corpus cache is
    missing the service still starts and simply offers fewer arms -- the demo is a comparison, and
    a comparison that refuses to start because one arm is unavailable is worse than a short one.
    """
    t0 = time.time()
    path = next((p for p in HEIGHTMAP_CKPTS if p.exists()), None)
    if path is not None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = build_model(DEPTH_CLASSES if ck["objective"] == "ce" else 1, ck["width"]).to(dev)
        net.load_state_dict(ck["state"])
        net.eval()
        _state["hm"] = net
        _state["hm_meta"] = dict(epoch=ck.get("epoch"), objective=ck["objective"],
                                 params=ck.get("params"), path=str(path.relative_to(REPO)))
        print(f"[town_generate] #127 height map ({ck['objective']}, epoch {ck.get('epoch')}, "
              f"{ck.get('params', 0)/1e6:.2f}M) from {path.relative_to(REPO)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"[town_generate] no height-map checkpoint in "
              f"{[str(p.relative_to(REPO)) for p in HEIGHTMAP_CKPTS]} -- arm unavailable",
              flush=True)
    if CACHE.exists():
        d = np.load(CACHE)
        keep = (d["ok"] > 0) & (d["held"] == 0)          # TRAINING rows only, never the pinned 714
        _state["bank"] = dict(fp=d["fp"][keep] > 0, target=d["target"][keep].astype(np.int16),
                              extent=d["extent"][keep].astype(np.int32))
        print(f"[town_generate] retrieval bank: {int(keep.sum())} training buildings "
              f"({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"[town_generate] no corpus cache at {CACHE} -- retrieval arm unavailable", flush=True)


class ProjectionKnobs(BaseModel):
    """The C1 projection's settings, shared by both endpoints so they cannot drift apart."""

    arm: str = "a2"                    # which massing generator to run; see ARM_ORDER. Defaults to
                                        # a2 so every existing caller of /generate_building and
                                        # /generate_town keeps the behaviour it had before #127.
    region: int = SOURCE_ID["nl"]      # the conditioning corpus a building claims to come from. A
                                        # hand-drawn footprint carries no such signal, so it defaults
                                        # to the majority training source (see ingest_citygml_lod2).
    strength: float = 0.5              # matches map-#86's evidence number; <0.35 is out-of-distribution
                                        # for this checkpoint (trained with pair_t_min=0.35)
    steps: int = 20
    guidance: float = 1.0
    seed: Optional[int] = None
    # ⚠️ Height-map arms are DETERMINISTIC: identical footprints give bit-identical buildings, which
    # in a town reads as obvious cloning (A2 gets its variety from per-building noise instead). This
    # jitters the decode quantile per building -- a coherent "deeper or shallower roof", not
    # per-column noise, which would be rubble. **Defaults to 0**, so the demo shows the arm that was
    # actually measured; quality away from q=0.5 is unmeasured and this is a demo affordance, not a
    # research knob. Ignored by every other arm.
    roof_variation: float = 0.0


class GenerateReq(ProjectionKnobs):
    points: List[List[float]]          # [[x,z], ...] world meters, the drawn/imported footprint
    height: float                      # world meters


class GenerateResp(BaseModel):
    arm: str                           # which generator produced this, echoed back so a client
                                        # cannot mislabel a mesh it asked for by name
    vertices: List[List[float]]        # world meters, y-up, ready for a three.js BufferGeometry
    faces: List[List[int]]
    vs_input: float                    # overlap with the footprint envelope this started from; near
                                        # 1.0 means the arm barely edited it -- report, don't hide
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


# Warm, measured on one A100: A2 is 20 denoise steps through a 191M codec; the height-map arms are
# a forward pass of a 3.4M convnet plus a marching-cubes surface. Only used to size the refusal
# message and the editor's ETA, never to decide anything.
ARM_SECONDS = {"a2": 7.0, "heightmap_mode": 0.3, "heightmap_median": 0.3,
               "retrieval": 0.5, "envelope": 0.2}


def _require_models(arm: str = "a2"):
    """Refuse early, and only for what this arm actually needs.

    The height-map arms compile straight to voxels, so a blanket "codec not loaded" guard would
    refuse them for a dependency they do not have -- which is the same claim `_load_heightmap_arms`
    is written around, and it has to hold at the endpoint too or it is not true of the service.
    """
    # A name nobody implements is the caller's mistake (400); a name this box cannot serve today is
    # the service's state (503). Collapsing them tells a client to retry a request that can never
    # succeed, and hides a typo behind what looks like a transient outage.
    if arm not in ARM_ORDER:
        raise HTTPException(400, f"unknown arm {arm!r}; expected one of {list(ARM_ORDER)}")
    if arm == "a2" and "codec" not in _state:
        raise HTTPException(503, "models still loading")
    if arm not in _available_arms():
        raise HTTPException(503, f"arm {arm!r} is not available on this box "
                                 f"(have: {_available_arms()})")


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

    fld = _arm_field(knobs.arm, fp, y0, y1, height, knobs, seed, bo)
    verts, faces = mesh_sdf_surface(np.clip(fld, -TRUNC, TRUNC))
    if verts is None:
        raise HTTPException(500, f"{knobs.arm} produced no surface (collapsed to empty/solid)")

    return dict(
        arm=knobs.arm,
        **_to_world_mesh(verts, faces, s, cx, cz, height),
        vs_input=_vs_input(fld <= 0, bo <= 0),
        gen_seconds=time.time() - t0,
    )


def _arm_field(arm: str, fp: np.ndarray, y0: int, y1: int, height: float,
               knobs: "ProjectionKnobs", seed: Optional[int], envelope: np.ndarray) -> np.ndarray:
    """One arm, one SDF on the corpus grid. The single place an arm name turns into geometry.

    Every endpoint dispatches here, so `/generate_building`, `/generate_town` and `/compare_arms`
    cannot drift apart in what an arm *means* -- the same failure `_generate_one` was already shared
    to prevent between the single and batch paths.
    """
    if arm not in ARM_ORDER:
        raise HTTPException(400, f"unknown arm {arm!r}; expected one of {list(ARM_ORDER)}")
    if arm not in _available_arms():
        raise HTTPException(503, f"arm {arm!r} is not available on this box "
                                 f"(have: {_available_arms()})")
    if arm == "envelope":
        return envelope
    if arm in ("heightmap_mode", "heightmap_median"):
        q = None if arm == "heightmap_mode" else _roof_quantile(knobs.roof_variation, seed)
        return _heightmap_field(fp, y0, y1, height, knobs.region, q)
    if arm == "retrieval":
        return _retrieval_field(fp, y0, y1)

    dev, codec, op = _state["dev"], _state["codec"], _state["op"]
    mu, sd = _state["mu"], _state["sd"]
    bv, bf = mesh_sdf_surface(np.clip(envelope, -TRUNC, TRUNC))
    if bv is None:
        raise HTTPException(400, "blockout produced no surface")
    with torch.no_grad():
        z0 = (codec.encode(Building(verts=verts_to_world(bv), faces=bf)).float() - mu) / sd
        fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
        zp = op.project(blockout=z0, footprint=fpt,
                        height=torch.tensor([height], device=dev),
                        region=torch.tensor([knobs.region], device=dev),
                        strength=knobs.strength, steps=knobs.steps, guidance=knobs.guidance,
                        seed=seed)
        return codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]


def _to_world_mesh(verts, faces, s: float, cx: float, cz: float, height: float) -> dict:
    """Frame-N surface -> world-meter mesh. One copy, shared by every arm.

    Split out when #127's arms were added rather than duplicated into them: this repo has a
    documented history (issue #70) of a frame convention drifting between two copies and silently
    corrupting a whole training run, and two spellings of this transform is exactly how that starts.
    """
    world = verts_to_world(verts)                       # (N,3) columns = [D,H,W] = [z,y,x], Frame-N
    out_verts = np.stack([
        world[:, 2] * s + cx,           # W(x) -> real world x
        world[:, 1] * s + height / 2,   # H(up) -> real world y (ground at 0)
        world[:, 0] * s + cz,           # D(z) -> real world z
    ], axis=1)
    return dict(vertices=out_verts.tolist(), faces=np.asarray(faces, dtype=int).tolist())


# ==================================================================================================
# #127's arms -- the same footprint carved four ways, for a side-by-side visual judgement
# ==================================================================================================

class CompareReq(BaseModel):
    """One footprint, every arm. `region` and `height` are the same conditioning A2 gets."""

    points: List[List[float]]
    height: float
    region: int = SOURCE_ID["nl"]
    arms: Optional[List[str]] = None     # None -> every arm this service can currently offer


def _roof_quantile(variation: float, seed: Optional[int]) -> float:
    """The decode quantile for one building: 0.5 unless the caller asked for variety.

    Clamped to [0.2, 0.8] because the tails are not roofs -- q -> 0 is the mode's under-carve and
    q -> 1 carves the building away -- and the clamp is on the QUANTILE rather than on the result,
    so it cannot interact with the height map's own guarantees.
    """
    if variation <= 0 or seed is None:
        return 0.5
    u = np.random.default_rng(seed).random()
    return float(np.clip(0.5 + (u - 0.5) * 2.0 * float(variation), 0.2, 0.8))


def _heightmap_field(fp: np.ndarray, y0: int, y1: int, height: float, region: int,
                     quantile: Optional[float]) -> np.ndarray:
    """#127's generator on a hand-drawn footprint. Conditioning only -- no ground truth exists here."""
    extent = int(y1 - y0 + 1)
    mask = fp.astype(bool)
    x = condition_channels(mask, extent, float(height), int(region))
    with torch.no_grad():
        logits = _state["hm"](torch.from_numpy(x)[None].to(_state["dev"])).cpu().numpy()[0]
    return occ_to_field(occupancy(mask, y0, decode_logits(logits, mask, extent, quantile)))


def _retrieval_field(fp: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """The zero-training baseline: the nearest training footprint's roof, rendered on this one."""
    bank, mask, extent = _state["bank"], fp.astype(bool), int(y1 - y0 + 1)
    j = int(retrieve_nn(mask[None], bank["fp"])[0])
    h = transplant_height(bank["target"][j], bank["fp"][j], int(bank["extent"][j]), mask, extent)
    return occ_to_field(occupancy(mask, y0, h))


def _available_arms() -> List[str]:
    have = {"envelope"}
    if "hm" in _state:
        have |= {"heightmap_mode", "heightmap_median"}
    if "bank" in _state:
        have.add("retrieval")
    if "codec" in _state:
        have.add("a2")
    return [a for a in ARM_ORDER if a in have]


@app.post("/compare_arms")
def compare_arms(req: CompareReq):
    """Carve one footprint with every arm and return them together, for a visual judgement.

    A loop over the same `_generate_one` that `/generate_building` calls, so an arm cannot look one
    way here and another way in the town editor.

    ⚠️ There is **no ground truth for a footprint you drew**, so `missing` / `extra` -- the numbers
    #126 made the headline and #127 is judged on -- cannot be computed here and are not reported.
    What this endpoint gives is `vs_input` (did the arm act on the footprint envelope at all) and
    the geometry. The judgement it supports is the visual one, which is this project's stated first
    criterion and the one #127's scorecard was measured to be blind to.
    """
    _require_models("envelope")
    have = _available_arms()
    wanted = [a for a in (req.arms or have) if a in have]
    out, notes = [], []
    for arm in wanted:
        try:
            out.append(_generate_one(req.points, req.height,
                                     ProjectionKnobs(arm=arm, region=req.region), None))
        except HTTPException as exc:
            notes.append(f"{arm}: {exc.detail}")
    if not out and notes:
        raise HTTPException(400, "; ".join(notes))
    pts = np.asarray(req.points, dtype=np.float64)
    s, cx, cz = _footprint_normalization(pts, req.height)
    y0, y1 = _height_voxel_range(req.height, s)
    return dict(arms=out, notes=notes, available=have,
                footprint_voxels=int(_rasterize_footprint(pts, s, cx, cz).sum()),
                extent_voxels=int(y1 - y0 + 1), heightmap=_state.get("hm_meta"))


@app.get("/arms")
def arms_page():
    return FileResponse(WEB / "arms.html")


@app.post("/generate_building", response_model=GenerateResp)
def generate_building(req: GenerateReq):
    _require_models(req.arm)
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
    _require_models(req.arm)
    if not req.buildings:
        raise HTTPException(400, "no buildings to generate")
    if len(req.buildings) > MAX_BUILDINGS:
        mins = len(req.buildings) * ARM_SECONDS.get(req.arm, SECS_PER_BUILDING) / 60
        raise HTTPException(400, f"{len(req.buildings)} buildings exceeds the {MAX_BUILDINGS} limit "
                                 f"(~{mins:.0f} min of GPU time on {req.arm})")
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
    return {"ok": "codec" in _state, "step": _state.get("step"), "arms": _available_arms()}


# ---- the editor page itself, so the demo is one service and not the legacy engine's 8-min boot ----
# Guarded the way inference_service.py guards the same mount: a missing samples directory should
# cost the demo its preset buttons, not refuse to start the generator.
if (WEB / "samples").exists():
    app.mount("/samples", StaticFiles(directory=str(WEB / "samples")), name="samples")
if (WEB / "vendor").exists():
    app.mount("/vendor", StaticFiles(directory=str(WEB / "vendor")), name="vendor")


@app.get("/")
def index():
    return FileResponse(WEB / "town.html", headers={"Cache-Control": "no-store"})
