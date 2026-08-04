"""Generate sample footprint images for the web demo (so there's something to upload).

Makes a synthetic block layout + REAL OSM footprint rasters (Munich old town, Lafayette)
as black-buildings-on-white PNGs into scripts/server/web/samples/, plus a samples.json
listing each with its real metric width (meters_across) for the demo.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts/server/web/samples"
OUT.mkdir(parents=True, exist_ok=True)


def synthetic():
    from PIL import Image, ImageDraw
    im = Image.new("L", (700, 520), 245); d = ImageDraw.Draw(im)
    rng = np.random.default_rng(0)
    for gx in range(5):
        for gy in range(4):
            if rng.random() < 0.15:
                continue
            x0 = 40 + gx * 130 + rng.integers(0, 20); y0 = 40 + gy * 120 + rng.integers(0, 20)
            w = rng.integers(50, 100); h = rng.integers(45, 85)
            d.rectangle([x0, y0, x0 + w, y0 + h], fill=70)
    im.save(OUT / "synthetic_blocks.png")
    return {"file": "synthetic_blocks.png", "label": "Synthetic blocks", "meters_across": 220}


def osm_raster(name, label, bbox, meters_across, px=820):
    import osmnx as ox
    n, s, e, w = bbox
    try:
        gdf = ox.features_from_bbox(bbox=(w, s, e, n), tags={"building": True})
    except Exception:
        gdf = ox.features_from_bbox(n, s, e, w, tags={"building": True})
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = ox.projection.project_gdf(gdf)
    polys = []
    for _, r in gdf.iterrows():
        g = r.geometry
        for gg in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            polys.append(np.asarray(gg.exterior.coords)[:, :2])
    if not polys:
        return None
    allp = np.concatenate(polys); mn = allp.min(0); mx = allp.max(0)
    span = (mx - mn).max()
    from PIL import Image, ImageDraw
    im = Image.new("L", (px, px), 245); d = ImageDraw.Draw(im)
    for p in polys:
        q = (p - mn) / span * (px - 20) + 10
        q[:, 1] = px - q[:, 1]   # flip y for image coords
        d.polygon([tuple(v) for v in q], fill=70)
    im.save(OUT / f"{name}.png")
    return {"file": f"{name}.png", "label": label, "meters_across": float(span)}


def main():
    samples = [synthetic()]
    for nm, lab, bb, ma in [
        ("munich_oldtown", "Munich Altstadt (real OSM)", [48.1400, 48.1362, 11.5785, 11.5725], 500),
        ("lafayette", "Lafayette IN (real OSM)", [40.4205, 40.4175, -86.8915, -86.8965], 400),
    ]:
        try:
            r = osm_raster(nm, lab, bb, ma)
            if r:
                samples.append(r); print(f"  {nm}: {r['meters_across']:.0f} m span")
        except Exception as ex:
            print(f"  {nm}: failed ({ex})")
    json.dump(samples, open(OUT / "samples.json", "w"), indent=2)
    print("[samples]", [s["file"] for s in samples], "->", OUT)


if __name__ == "__main__":
    main()
