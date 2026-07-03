"""Grounded operator AI helpers for FaceFlash events.

This module intentionally starts as deterministic "small AI": it turns an
operator question into a structured event query, filters real persisted events,
and summarizes only those rows. A cloud/local LLM can be added later behind the
same contract, but security products need the data boundary first.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional


_EVENT_ALIASES = {
    "unknown": {"visitor_new", "enrollment_suggested"},
    "visitor": {"visitor_new", "visitor_returned", "enrollment_suggested"},
    "watchlist": {"watchlist_match"},
    "tailgating": {"tailgating_risk"},
    "capacity": {"capacity_exceeded"},
    "secure": {"zone_entered", "tailgating_risk"},
    "zone": {"zone_entered", "zone_exited"},
    "entered": {"zone_entered", "person_appeared"},
    "left": {"zone_exited", "person_left"},
    "liveness": {"liveness_failed"},
    "spoof": {"liveness_failed"},
    "alert": {"watchlist_match", "tailgating_risk", "capacity_exceeded",
              "liveness_failed"},
}


def _day_bounds(dt: datetime) -> tuple[float, float]:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def _time_window(text: str, now: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    now_ts = float(time.time() if now is None else now)
    now_dt = datetime.fromtimestamp(now_ts)

    day_start = None
    day_end = None
    if "yesterday" in text:
        day_start, day_end = _day_bounds(now_dt - timedelta(days=1))
    elif "today" in text:
        day_start, day_end = _day_bounds(now_dt)
    elif "this week" in text:
        start = (now_dt - timedelta(days=now_dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        day_start = start.timestamp()

    m = re.search(r"last\s+(\d+)\s*(hour|hours|hr|hrs|day|days)", text)
    if m:
        qty = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(hours=qty) if unit.startswith(("hour", "hr")) else timedelta(days=qty)
        return now_ts - delta.total_seconds(), None

    m = re.search(r"after\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridian = m.group(3)
        if meridian == "pm" and hour != 12:
            hour += 12
        if meridian == "am" and hour == 12:
            hour = 0
        base = datetime.fromtimestamp(day_start) if day_start is not None else now_dt
        start = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return start.timestamp(), day_end

    if day_start is not None:
        return day_start, day_end

    return None, None


def _extract_named(text: str, keys: Iterable[str]) -> Optional[str]:
    key_expr = "|".join(re.escape(k) for k in keys)
    m = re.search(rf"\b(?:{key_expr})\s+([a-z0-9][a-z0-9_\- ]{{0,40}})", text)
    if not m:
        return None
    value = m.group(1)
    value = re.split(r"\b(?:after|before|yesterday|today|last|this week|with|and)\b", value)[0]
    return value.strip(" .,:;") or None


def parse_event_question(question: str, now: Optional[float] = None) -> Dict[str, Any]:
    """Convert an operator question into a conservative structured query."""
    text = " ".join((question or "").lower().split())
    event_types = set()
    for word, aliases in _EVENT_ALIASES.items():
        if word == "visitor" and "unknown" in text:
            continue
        if word in text:
            event_types.update(aliases)

    since, until = _time_window(text, now)
    return {
        "question": question,
        "event_types": sorted(event_types),
        "since": since,
        "until": until,
        "camera": _extract_named(text, ["camera", "cam"]),
        "zone": (_extract_named(text, ["zone", "area"]) or
                 _extract_named(text, ["near", "in", "at"])),
        "person": _extract_named(text, ["person", "name"]),
        "visitor_id": _extract_named(text, ["visitor"]),
    }


def _event_matches(event: Dict[str, Any], query: Dict[str, Any]) -> bool:
    data = event.get("data") or {}
    if query.get("event_types") and event.get("event_type") not in query["event_types"]:
        return False
    ts = float(event.get("timestamp", 0))
    if query.get("since") is not None and ts < float(query["since"]):
        return False
    if query.get("until") is not None and ts >= float(query["until"]):
        return False
    if query.get("camera") and str(data.get("camera", "")).lower() != query["camera"]:
        return False
    zone = query.get("zone")
    if zone:
        values = [data.get("zone"), data.get("zone_name")]
        values.extend(data.get("zones") or [])
        if zone not in {str(v).lower() for v in values if v is not None}:
            return False
    person = query.get("person")
    if person and person not in str(data.get("name") or data.get("person") or "").lower():
        return False
    visitor_id = query.get("visitor_id")
    if visitor_id and visitor_id not in str(data.get("visitor_id") or "").lower():
        return False
    return True


def filter_events(events: Iterable[Dict[str, Any]], query: Dict[str, Any],
                  limit: int = 100) -> List[Dict[str, Any]]:
    """Return newest-first events matching a parsed AI query."""
    limit = max(1, int(limit))
    matched = [e for e in events if _event_matches(e, query)]
    matched.sort(key=lambda e: float(e.get("timestamp", 0)), reverse=True)
    return matched[:limit]


def summarize_events(events: List[Dict[str, Any]], query: Optional[Dict[str, Any]] = None) -> str:
    """Summarize only supplied events; never invent missing details."""
    if not events:
        return "No matching FaceFlash events were found."

    types = Counter(e.get("event_type", "unknown") for e in events)
    cameras = Counter((e.get("data") or {}).get("camera") for e in events
                      if (e.get("data") or {}).get("camera"))
    zones = Counter((e.get("data") or {}).get("zone") for e in events
                    if (e.get("data") or {}).get("zone"))

    newest = max(float(e.get("timestamp", 0)) for e in events)
    oldest = min(float(e.get("timestamp", 0)) for e in events)
    summary = [
        f"Found {len(events)} matching event{'s' if len(events) != 1 else ''}.",
        "Event mix: " + ", ".join(f"{k}={v}" for k, v in types.most_common()),
        "Time span: " + datetime.fromtimestamp(oldest).strftime("%Y-%m-%d %H:%M:%S")
        + " to " + datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M:%S"),
    ]
    if cameras:
        summary.append("Top cameras: " + ", ".join(f"{k}={v}" for k, v in cameras.most_common(3)))
    if zones:
        summary.append("Top zones: " + ", ".join(f"{k}={v}" for k, v in zones.most_common(3)))

    notable = []
    for event in sorted(events, key=lambda e: float(e.get("timestamp", 0)))[:5]:
        data = event.get("data") or {}
        parts = [datetime.fromtimestamp(float(event.get("timestamp", 0))).strftime("%H:%M:%S"),
                 event.get("event_type", "event")]
        for key in ("camera", "zone", "name", "visitor_id", "label"):
            if data.get(key):
                parts.append(f"{key}={data[key]}")
        notable.append(" ".join(parts))
    if notable:
        summary.append("First events: " + "; ".join(notable))
    return " ".join(summary)


def answer_event_question(events: Iterable[Dict[str, Any]], question: str,
                          now: Optional[float] = None,
                          limit: int = 100) -> Dict[str, Any]:
    """Parse, filter, and summarize real events for an operator question."""
    query = parse_event_question(question, now=now)
    matched = filter_events(events, query, limit=limit)
    return {
        "query": query,
        "count": len(matched),
        "events": matched,
        "summary": summarize_events(matched, query=query),
    }
