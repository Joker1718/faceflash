"""Zone geometry helpers shared by UI, config, and stream tooling."""

from __future__ import annotations

from typing import Iterable, Sequence


def rect_to_polygon(x1, y1, x2, y2):
    return [(float(x1), float(y1)), (float(x2), float(y1)),
            (float(x2), float(y2)), (float(x1), float(y2))]


def normalize_polygon(poly: Iterable[Sequence[float]], frame_w: int, frame_h: int):
    fw = max(float(frame_w), 1.0)
    fh = max(float(frame_h), 1.0)
    return [(float(x) / fw, float(y) / fh) for x, y in poly]


def polygon_for_frame(zone: dict, frame_w: int, frame_h: int):
    """Project a zone dict onto a target frame size.

    Supports modern normalized zones (`polygon_norm`) and legacy pixel zones
    with a stored draw-time `frame_size`.
    """
    if zone.get("polygon_norm"):
        return [(float(x) * frame_w, float(y) * frame_h)
                for x, y in zone["polygon_norm"]]
    base_w, base_h = zone.get("frame_size", (frame_w, frame_h))
    sx = frame_w / max(float(base_w), 1.0)
    sy = frame_h / max(float(base_h), 1.0)
    return [(float(x) * sx, float(y) * sy) for x, y in zone["polygon"]]


def zone_payload(name: str, polygon, secure: bool, frame_w: int, frame_h: int):
    return {
        "name": str(name),
        "polygon": [(float(x), float(y)) for x, y in polygon],
        "polygon_norm": normalize_polygon(polygon, frame_w, frame_h),
        "frame_size": [int(frame_w), int(frame_h)],
        "secure": bool(secure),
    }


def point_in_polygon(x: float, y: float, poly) -> bool:
    """Ray-casting point-in-polygon. poly = [(x, y), ...]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def zone_counts(detections, zones, frame_w: int, frame_h: int):
    counts = {z["name"]: 0 for z in zones}
    for x1, y1, x2, y2, *_ in detections:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for z in zones:
            if point_in_polygon(cx, cy, polygon_for_frame(z, frame_w, frame_h)):
                counts[z["name"]] += 1
    return counts
