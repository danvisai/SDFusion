"""Standalone image-footprint-extraction service for the town editor (map #97, #98).

Wraps footprint_image.extract_footprints, which is pure image processing (PIL/numpy/
skimage) with no model dependency -- deliberately kept separate from inference_service.py
so testing the editor's image-import doesn't require booting the heavy massing engine
(inference_service.py's startup hook loads it unconditionally, ~8 min).

Run (dev):
  ./venv/bin/uvicorn scripts.server.footprint_extract_service:app --port 8766
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from footprint_image import extract_footprints, to_meters  # noqa: E402

app = FastAPI(title="Footprint image extraction (map #97)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractReq(BaseModel):
    image_b64: str
    meters_across: float = 200.0
    invert: bool = False
    simplify_px: float = 1.0   # corner-preserving simplification tolerance, in source pixels.
                                # The town editor opts in (footprint_image defaults to the older
                                # fixed-16-point resample for inference_service.py's sake); 1.0 is
                                # the measured pick -- see extract_footprints' docstring.


class ExtractResp(BaseModel):
    footprints: List[List[List[float]]]  # each: [[x,z], ...] world meters, closed-ish polygon


@app.post("/extract_footprints", response_model=ExtractResp)
def extract(req: ExtractReq):
    try:
        raw = base64.b64decode(req.image_b64.split(",")[-1])
    except Exception as e:
        raise HTTPException(400, f"bad image_b64: {e}")
    polys_px, hw = extract_footprints(raw, invert=req.invert, simplify_px=req.simplify_px)
    scaled = to_meters(polys_px, hw, req.meters_across)
    footprints = [(local_poly + centroid).tolist() for local_poly, centroid in scaled]
    return ExtractResp(footprints=footprints)


@app.get("/health")
def health():
    return {"ok": True}
