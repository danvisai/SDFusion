"""
Step 0a — F0 Route A: OSM vector parser.

Pulls buildings + roads from OpenStreetMap for a given place or bounding box,
normalizes to the schema the downstream pipeline expects:
  {
    "buildings": [
      {"id": "OSM_<id>", "polygon": [[x, y], ...],   # local meters, origin at bbox SW
       "class": "RESIDENTIALhouse",                  # one of our 53 prefixes
       "height": 10.5,                               # meters
       "centroid": [cx, cy], "area": 200.5},
      ...
    ],
    "roads": [
      {"polyline": [[x, y], ...], "type": "secondary", "width": 6.0},
      ...
    ],
    "bbox": {"south": ..., "west": ..., "north": ..., "east": ...},
    "origin_latlng": [lat, lng],
    "meters_per_degree_lat": 111320.0,
    "meters_per_degree_lng": ...,
  }

Run:
    python scene/extract_osm.py --place "Lafayette, IN" --radius 200 -o /tmp/lafayette.json
    python scene/extract_osm.py --bbox 40.4234 -86.9075 40.4250 -86.9050 -o /tmp/area.json
"""
import argparse
import json
import math
import os
from typing import Tuple

import osmnx as ox
import shapely.geometry as sg


# ---- OSM tag → our 53-prefix taxonomy mapping ----------------------------------

# Top-level: RESIDENTIAL / RELIGIOUS / COMMERCIAL / MILITARY / PUBLIC
# We map OSM `building` tag values into these. The full prefix is
# <TOPLEVEL><subtype>; if no subtype is known we use a sensible default.
_OSM_BUILDING_TO_CLASS = {
    # residential
    "residential": "RESIDENTIALhouse", "house": "RESIDENTIALhouse",
    "detached": "RESIDENTIALhouse", "semidetached_house": "RESIDENTIALhouse",
    "terrace": "RESIDENTIALhouse", "bungalow": "RESIDENTIALhouse",
    "cabin": "RESIDENTIALhouse", "static_caravan": "RESIDENTIALhouse",
    "apartments": "RESIDENTIALhouse", "dormitory": "RESIDENTIALhouse",
    "hotel": "RESIDENTIALhotel_building",
    "villa": "RESIDENTIALvilla", "mansion": "RESIDENTIALvilla",
    # religious
    "church": "RELIGIOUSchurch", "chapel": "RELIGIOUSchurch",
    "cathedral": "RELIGIOUScathedral",
    "mosque": "RELIGIOUSmosque",
    "temple": "RELIGIOUStemple",
    "synagogue": "RELIGIOUStemple",
    "monastery": "RELIGIOUSmonastery",
    "shrine": "RELIGIOUStemple",
    "religious": "RELIGIOUSchurch",
    # commercial
    "commercial": "COMMERCIALoffice_building",
    "office": "COMMERCIALoffice_building",
    "retail": "COMMERCIALoffice_building", "shop": "COMMERCIALoffice_building",
    "supermarket": "COMMERCIALoffice_building",
    "kiosk": "COMMERCIALoffice_building",
    "industrial": "COMMERCIALfactory", "warehouse": "COMMERCIALfactory",
    "factory": "COMMERCIALfactory", "manufacture": "COMMERCIALfactory",
    "hangar": "COMMERCIALfactory",
    "museum": "COMMERCIALmuseum",
    # public / civic
    "public": "PUBLICoffice_building",
    "government": "PUBLICoffice_building", "civic": "PUBLICoffice_building",
    "school": "PUBLICschool_building", "university": "PUBLICschool_building",
    "college": "PUBLICschool_building", "kindergarten": "PUBLICschool_building",
    "hospital": "PUBLICoffice_building", "clinic": "PUBLICoffice_building",
    "library": "PUBLICmuseum",
    "fire_station": "PUBLICoffice_building",
    "train_station": "PUBLICoffice_building",
    "transportation": "PUBLICoffice_building",
    "stadium": "PUBLICoffice_building", "sports_hall": "PUBLICoffice_building",
    "city_hall": "RELIGIOUScity_hall",
    "monument": "PUBLICmuseum",
    # military
    "military": "MILITARYcastle", "barracks": "MILITARYcastle",
    "bunker": "MILITARYcastle",
    "castle": "MILITARYcastle", "fortress": "MILITARYcastle",
    "tower": "MILITARYcastle",
    "palace": "MILITARYpalace",
    # generic / unknown — default to residential house
    "yes": "RESIDENTIALhouse",
}

_DEFAULT_FLOOR_HEIGHT_M = 3.5
_DEFAULT_LEVELS_BY_TOPLEVEL = {
    "RESIDENTIAL": 2, "RELIGIOUS": 4, "COMMERCIAL": 4,
    "PUBLIC": 3, "MILITARY": 3,
}


def osm_to_class(building_tag: str) -> str:
    if not building_tag or not isinstance(building_tag, str):
        return "RESIDENTIALhouse"
    return _OSM_BUILDING_TO_CLASS.get(building_tag.lower(), "RESIDENTIALhouse")


def height_from_row(row, top_level: str) -> float:
    """Extract building height in meters from an OSM feature row."""
    # explicit `height` tag in meters
    h = row.get("height")
    if h is not None and not _isnan(h):
        try:
            v = float(str(h).rstrip(" mM"))
            if v > 0:
                return v
        except (ValueError, TypeError):
            pass
    # `building:levels`
    levels = row.get("building:levels")
    if levels is not None and not _isnan(levels):
        try:
            v = float(str(levels))
            if v > 0:
                return v * _DEFAULT_FLOOR_HEIGHT_M
        except (ValueError, TypeError):
            pass
    # fall back to a class-conditional default
    n_levels = _DEFAULT_LEVELS_BY_TOPLEVEL.get(top_level, 2)
    return n_levels * _DEFAULT_FLOOR_HEIGHT_M


def _isnan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """Approximate meters per degree at given latitude (WGS84)."""
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_per_deg_lng = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_per_deg_lat, m_per_deg_lng


def latlng_to_local_xy(lat: float, lng: float, origin_lat: float, origin_lng: float,
                       m_per_lat: float, m_per_lng: float) -> Tuple[float, float]:
    return ((lng - origin_lng) * m_per_lng, (lat - origin_lat) * m_per_lat)


def extract_buildings(buildings_gdf, origin_lat, origin_lng, m_per_lat, m_per_lng):
    out = []
    for idx, row in buildings_gdf.iterrows():
        geom = row.geometry
        # take outer ring of a Polygon, or first polygon of a MultiPolygon
        if isinstance(geom, sg.Polygon):
            ring = geom.exterior
        elif isinstance(geom, sg.MultiPolygon):
            ring = max(geom.geoms, key=lambda g: g.area).exterior
        else:
            continue
        coords_latlng = list(ring.coords)
        polygon_xy = [
            latlng_to_local_xy(lat, lng, origin_lat, origin_lng, m_per_lat, m_per_lng)
            for lng, lat in coords_latlng
        ]
        # area in square meters via shoelace
        n = len(polygon_xy)
        area = 0.5 * abs(sum(
            polygon_xy[i][0] * polygon_xy[(i + 1) % n][1]
            - polygon_xy[(i + 1) % n][0] * polygon_xy[i][1]
            for i in range(n)
        ))
        cx = sum(p[0] for p in polygon_xy) / max(n, 1)
        cy = sum(p[1] for p in polygon_xy) / max(n, 1)

        cls = osm_to_class(row.get("building"))
        top_level = next((t for t in ("RESIDENTIAL", "RELIGIOUS", "COMMERCIAL", "MILITARY", "PUBLIC") if cls.startswith(t)), "RESIDENTIAL")
        height_m = height_from_row(row, top_level)

        osm_id = idx if isinstance(idx, str) else f"OSM_{idx[1] if isinstance(idx, tuple) else idx}"
        out.append({
            "id": str(osm_id),
            "polygon": polygon_xy,
            "class": cls,
            "height": float(height_m),
            "centroid": [cx, cy],
            "area": float(area),
        })
    return out


def extract_roads(roads_graph, origin_lat, origin_lng, m_per_lat, m_per_lng):
    """Each edge in osmnx graph_from_bbox -> a polyline."""
    if roads_graph is None:
        return []
    out = []
    for u, v, k, data in roads_graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            # synthesize from node positions
            uy, ux = roads_graph.nodes[u]["y"], roads_graph.nodes[u]["x"]
            vy, vx = roads_graph.nodes[v]["y"], roads_graph.nodes[v]["x"]
            coords_latlng = [(ux, uy), (vx, vy)]
        else:
            coords_latlng = list(geom.coords)
        polyline_xy = [
            latlng_to_local_xy(lat, lng, origin_lat, origin_lng, m_per_lat, m_per_lng)
            for lng, lat in coords_latlng
        ]
        out.append({
            "polyline": polyline_xy,
            "type": data.get("highway", "unclassified"),
            "width": float(data.get("width", _default_road_width(data.get("highway", "")))),
            "name": data.get("name", None),
        })
    return out


def _default_road_width(highway: str) -> float:
    return {
        "motorway": 12.0, "trunk": 10.0, "primary": 8.0, "secondary": 7.0,
        "tertiary": 6.0, "residential": 5.0, "service": 4.0, "footway": 2.0,
        "path": 2.0, "cycleway": 2.5, "unclassified": 5.0,
    }.get(highway, 5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", help="OSM-recognizable place name, e.g. 'Lafayette, IN'")
    ap.add_argument("--radius", type=float, default=200.0,
                    help="meters around place centroid (used with --place)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                    help="explicit bbox (south west north east, in degrees)")
    ap.add_argument("--out", "-o", required=True, help="output JSON path")
    ap.add_argument("--no-roads", action="store_true", help="skip road extraction")
    args = ap.parse_args()

    if not args.place and not args.bbox:
        raise SystemExit("Specify either --place or --bbox.")

    # Resolve bbox (S, W, N, E)
    if args.bbox:
        south, west, north, east = args.bbox
    else:
        gdf = ox.geocode_to_gdf(args.place)
        cy, cx = float(gdf.centroid.y.iloc[0]), float(gdf.centroid.x.iloc[0])
        m_per_lat, m_per_lng = _meters_per_degree(cy)
        d_lat = args.radius / m_per_lat
        d_lng = args.radius / m_per_lng
        south, west, north, east = cy - d_lat, cx - d_lng, cy + d_lat, cx + d_lng

    origin_lat, origin_lng = south, west
    m_per_lat, m_per_lng = _meters_per_degree(0.5 * (south + north))

    print(f"[osm] bbox  south={south:.6f} west={west:.6f} north={north:.6f} east={east:.6f}")
    print(f"[osm] origin lat/lng = ({origin_lat:.6f}, {origin_lng:.6f})")
    print(f"[osm] meters/deg  lat={m_per_lat:.1f}  lng={m_per_lng:.1f}")

    print("[osm] fetching buildings ...")
    buildings_gdf = ox.features_from_bbox(
        bbox=(west, south, east, north), tags={"building": True}
    )
    if buildings_gdf is None or len(buildings_gdf) == 0:
        buildings = []
    else:
        # only Polygon / MultiPolygon
        buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        buildings = extract_buildings(
            buildings_gdf, origin_lat, origin_lng, m_per_lat, m_per_lng
        )
    print(f"[osm]  -> {len(buildings)} buildings")

    if args.no_roads:
        roads = []
    else:
        print("[osm] fetching roads ...")
        try:
            roads_g = ox.graph_from_bbox(
                bbox=(west, south, east, north), network_type="drive", simplify=True
            )
            roads = extract_roads(roads_g, origin_lat, origin_lng, m_per_lat, m_per_lng)
        except Exception as e:
            print(f"[osm] road fetch failed: {e}")
            roads = []
        print(f"[osm]  -> {len(roads)} road segments")

    payload = {
        "buildings": buildings,
        "roads": roads,
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "origin_latlng": [origin_lat, origin_lng],
        "meters_per_degree_lat": m_per_lat,
        "meters_per_degree_lng": m_per_lng,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[osm] wrote {args.out}")

    # Stats summary
    if buildings:
        cls_counts = {}
        for b in buildings:
            cls_counts[b["class"]] = cls_counts.get(b["class"], 0) + 1
        print("[osm] class breakdown:")
        for c, n in sorted(cls_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {c:30s} {n}")
        heights = [b["height"] for b in buildings]
        print(f"[osm] height stats: min={min(heights):.1f}  median={sorted(heights)[len(heights)//2]:.1f}  max={max(heights):.1f}")
        areas = [b["area"] for b in buildings]
        print(f"[osm] area stats:   min={min(areas):.1f}  median={sorted(areas)[len(areas)//2]:.1f}  max={max(areas):.1f}")


if __name__ == "__main__":
    main()
