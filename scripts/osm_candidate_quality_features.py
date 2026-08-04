"""Shared feature helpers for OSM candidate quality/rerank models."""
from __future__ import annotations

import numpy as np


TOP_LEVELS = ["RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS", "MILITARY"]


def safe_log(value: float) -> float:
    return float(np.log(max(float(value), 1e-8)))


def polygon_stats(polygon: list[list[float]] | list[tuple[float, float]]) -> dict[str, float]:
    poly = np.asarray(polygon, dtype=np.float64)
    ext = poly.max(axis=0) - poly.min(axis=0)
    area = 0.5 * abs(sum(
        poly[i, 0] * poly[(i + 1) % len(poly), 1]
        - poly[(i + 1) % len(poly), 0] * poly[i, 1]
        for i in range(len(poly))
    ))
    edges = np.roll(poly, -1, axis=0) - poly
    perimeter = float(np.linalg.norm(edges, axis=1).sum())
    return {
        "bbox_width_m": float(ext[0]),
        "bbox_depth_m": float(ext[1]),
        "bbox_aspect": float(max(ext[0], ext[1]) / max(min(ext[0], ext[1]), 1e-6)),
        "area_m2_from_polygon": float(area),
        "perimeter_m": perimeter,
        "compactness": float((4.0 * np.pi * area) / max(perimeter * perimeter, 1e-6)),
    }


def top_level_from_class(class_name: str) -> str:
    return next((t for t in TOP_LEVELS if str(class_name).startswith(t)), "RESIDENTIAL")


def candidate_quality_features(
    class_name: str,
    area_m2: float,
    height_m: float,
    geom: dict[str, float],
    candidate: dict,
) -> list[float]:
    cand_aspect = float(candidate.get("candidate_aspect", 0.0) or 0.0)
    target_aspect = float(candidate.get("target_aspect", geom.get("bbox_aspect", 0.0)) or 0.0)
    cand_height_ratio = float(candidate.get("candidate_height_ratio", 0.0) or 0.0)
    target_height_ratio = float(candidate.get("target_height_ratio", geom.get("height_to_bbox_max", 0.0)) or 0.0)
    one_hot = [1.0 if top_level_from_class(class_name) == t else 0.0 for t in TOP_LEVELS]
    return [
        safe_log(area_m2),
        float(height_m),
        safe_log(float(height_m) + 1.0),
        float(geom.get("bbox_width_m", 0.0) or 0.0),
        float(geom.get("bbox_depth_m", 0.0) or 0.0),
        float(geom.get("bbox_aspect", 0.0) or 0.0),
        float(geom.get("compactness", 0.0) or 0.0),
        float(geom.get("height_to_sqrt_area", 0.0) or 0.0),
        float(geom.get("height_to_bbox_max", 0.0) or 0.0),
        float(candidate.get("rank", 0.0) or 0.0),
        float(candidate.get("retrieval_score", 0.0) or 0.0),
        float(candidate.get("rerank_score", candidate.get("retrieval_score", 0.0)) or 0.0),
        target_aspect,
        cand_aspect,
        abs(safe_log(cand_aspect) - safe_log(target_aspect)),
        target_height_ratio,
        cand_height_ratio,
        abs(safe_log(cand_height_ratio) - safe_log(target_height_ratio)),
        float(candidate.get("aspect_penalty", 0.0) or 0.0),
        float(candidate.get("height_penalty", 0.0) or 0.0),
        safe_log(float(candidate.get("candidate_verts", 0.0) or 0.0) + 1.0),
        safe_log(float(candidate.get("candidate_faces", 0.0) or 0.0) + 1.0),
        *one_hot,
    ]


FEATURE_NAMES = [
    "log_area_m2",
    "height_m",
    "log_height_plus_1",
    "bbox_width_m",
    "bbox_depth_m",
    "bbox_aspect",
    "compactness",
    "height_to_sqrt_area",
    "height_to_bbox_max",
    "candidate_rank",
    "retrieval_score",
    "rerank_score",
    "target_aspect",
    "candidate_aspect",
    "aspect_log_error",
    "target_height_ratio",
    "candidate_height_ratio",
    "height_ratio_log_error",
    "aspect_penalty",
    "height_penalty",
    "log_candidate_verts",
    "log_candidate_faces",
    *[f"top_{name.lower()}" for name in TOP_LEVELS],
]


def record_geometry_features(record: dict) -> dict[str, float]:
    geom = dict(record.get("geometry_features", {}))
    area = float(record.get("area_m2", 0.0) or 0.0)
    height = float(record.get("height_m", 0.0) or 0.0)
    if "bbox_aspect" not in geom and record.get("polygon_xy_m"):
        geom.update(polygon_stats(record["polygon_xy_m"]))
    geom.setdefault("height_to_sqrt_area", float(height / max(np.sqrt(area), 1e-6)))
    geom.setdefault("height_to_bbox_max", float(height / max(geom.get("bbox_width_m", 0.0), geom.get("bbox_depth_m", 0.0), 1e-6)))
    return geom


def building_geometry_features(building: dict, height_m: float) -> dict[str, float]:
    geom = polygon_stats(building["polygon"])
    area = float(building.get("area", geom["area_m2_from_polygon"]) or 0.0)
    geom["height_to_sqrt_area"] = float(height_m / max(np.sqrt(area), 1e-6))
    geom["height_to_bbox_max"] = float(height_m / max(geom["bbox_width_m"], geom["bbox_depth_m"], 1e-6))
    return geom
