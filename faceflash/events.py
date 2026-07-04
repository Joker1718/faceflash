"""Event persistence and querying for FaceFlash operator workflows."""

from __future__ import annotations

import json
import threading
from contextlib import nullcontext
from typing import Iterable, List, Optional


class EventStore:
    """Small SQLite-backed event store around the existing stream_events table."""

    def __init__(self, db, lock=None):
        self.db = db
        self.lock = lock or threading.RLock()
        self.ensure_schema()

    def _locked(self):
        return self.lock if self.lock is not None else nullcontext()

    def ensure_schema(self):
        with self._locked():
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS stream_events ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " ts REAL NOT NULL,"
                " event_type TEXT NOT NULL,"
                " data TEXT NOT NULL DEFAULT '{}')")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stream_events_ts "
                "ON stream_events(ts DESC)")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stream_events_type_ts "
                "ON stream_events(event_type, ts DESC)")
            self.db.commit()

    def insert(self, timestamp: float, event_type: str, data: dict):
        with self._locked():
            self.db.execute(
                "INSERT INTO stream_events(ts, event_type, data) VALUES (?, ?, ?)",
                (float(timestamp), str(event_type), json.dumps(data or {})))
            self.db.commit()

    def query(self, event_type: Optional[str] = None,
              event_types: Optional[Iterable[str]] = None,
              since: Optional[float] = None,
              until: Optional[float] = None,
              camera: Optional[str] = None,
              zone: Optional[str] = None,
              data_value: Optional[str] = None,
              limit: int = 1000,
              offset: int = 0) -> List[dict]:
        """Query events newest first.

        event_type/time filters are applied in SQL. camera/zone live in the JSON
        payload today, so those are filtered after parsing without truncating
        the requested time range first.
        """
        limit = max(1, min(int(limit), 10000))
        offset = max(0, int(offset))
        types = list(event_types or ([] if event_type is None else [event_type]))

        q = "SELECT ts, event_type, data FROM stream_events"
        cond, args = [], []
        if types:
            cond.append("event_type IN (%s)" % ",".join("?" for _ in types))
            args.extend(types)
        if since is not None:
            cond.append("ts >= ?")
            args.append(float(since))
        if until is not None:
            cond.append("ts < ?")
            args.append(float(until))
        if data_value is not None:
            cond.append("data LIKE ?")
            args.append(f'%"{data_value}"%')
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY ts DESC"

        with self._locked():
            rows = self.db.execute(q, args).fetchall()

        camera_l = camera.lower() if camera else None
        zone_l = zone.lower() if zone else None
        out = []
        for ts, typ, raw in rows:
            try:
                data = json.loads(raw or "{}")
            except Exception:
                data = {}
            if camera_l and str(data.get("camera", "")).lower() != camera_l:
                continue
            if zone_l:
                values = [data.get("zone"), data.get("zone_name")]
                values.extend(data.get("zones") or [])
                if zone_l not in {str(v).lower() for v in values if v is not None}:
                    continue
            if data_value is not None and data_value not in {
                    str(v) for v in data.values()}:
                continue
            out.append({"timestamp": ts, "event_type": typ, "data": data})

        return out[offset:offset + limit]

    def delete_before(self, timestamp: float) -> int:
        with self._locked():
            cur = self.db.execute("DELETE FROM stream_events WHERE ts < ?",
                                  (float(timestamp),))
            self.db.commit()
            return int(cur.rowcount or 0)

    def stats(self) -> dict:
        with self._locked():
            total = self.db.execute("SELECT COUNT(*) FROM stream_events").fetchone()[0]
            by_type = self.db.execute(
                "SELECT event_type, COUNT(*) FROM stream_events "
                "GROUP BY event_type ORDER BY COUNT(*) DESC").fetchall()
        return {
            "events": int(total),
            "by_type": {str(k): int(v) for k, v in by_type},
        }
