"""Minimal FastAPI operator API for camera/zone/event management."""

from typing import Optional
from urllib.parse import urlsplit, urlunsplit


def create_app(db_path: str):
    """Create a FastAPI app backed by a FaceFlash gallery DB.

    Install with: pip install fastapi uvicorn
    Run with: uvicorn 'faceflash.operator_api:create_app("faces.db")'
    """
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise ImportError(
            "FastAPI operator API requires: pip install fastapi uvicorn"
        ) from e

    from faceflash import FaceFlash
    from faceflash.events import EventStore

    ff = FaceFlash(index_path=db_path)
    event_store = None
    if getattr(ff.index, "_db", None) is not None:
        event_store = EventStore(ff.index._db, lock=ff.index._db_lock)
    app = FastAPI(title="FaceFlash Operator API", version="0.1")

    class CameraIn(BaseModel):
        name: str
        source: str = "0"
        settings: dict = Field(default_factory=dict)

    class ZoneIn(BaseModel):
        name: str
        polygon: list
        secure: bool = False
        settings: dict = Field(default_factory=dict)

    class AiEventSearchIn(BaseModel):
        question: str
        limit: int = 100
        now: Optional[float] = None

    def redact_source(source: str) -> str:
        value = str(source)
        try:
            parsed = urlsplit(value)
        except Exception:
            return value
        if not parsed.scheme or not parsed.netloc:
            return value
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, f"***:***@{host}",
                               parsed.path, parsed.query, parsed.fragment))
        return value

    def public_camera(cfg: dict) -> dict:
        out = dict(cfg)
        if "source" in out:
            out["source"] = redact_source(out["source"])
        return out

    @app.get("/status")
    def status():
        return {"ok": True, "stats": ff.stats(),
                "cameras": len(ff.index.list_camera_configs())}

    @app.get("/cameras")
    def list_cameras():
        return {"cameras": [public_camera(c) for c in ff.index.list_camera_configs()]}

    @app.put("/cameras/{name}")
    def save_camera(name: str, body: CameraIn):
        settings = dict(body.settings)
        cfg = ff.index.save_camera_config(name, body.source, **settings)
        return {"camera": public_camera(cfg)}

    @app.delete("/cameras/{name}")
    def delete_camera(name: str):
        if not ff.index.delete_camera_config(name):
            raise HTTPException(status_code=404, detail="camera not found")
        return {"deleted": True}

    @app.get("/cameras/{camera}/zones")
    def list_zones(camera: str):
        return {"zones": ff.index.list_zone_configs(camera)}

    @app.put("/cameras/{camera}/zones/{name}")
    def save_zone(camera: str, name: str, body: ZoneIn):
        cfg = ff.index.save_zone_config(
            camera, name, body.polygon, secure=body.secure, **body.settings)
        return {"zone": cfg}

    @app.delete("/cameras/{camera}/zones/{name}")
    def delete_zone(camera: str, name: str):
        if not ff.index.delete_zone_config(camera, name):
            raise HTTPException(status_code=404, detail="zone not found")
        return {"deleted": True}

    @app.get("/events")
    def events(event_type: Optional[str] = None,
               since: Optional[float] = None,
               until: Optional[float] = None,
               camera: Optional[str] = None,
               zone: Optional[str] = None,
               limit: int = 1000,
               offset: int = 0):
        if event_store is None:
            return {"events": []}
        return {"events": event_store.query(
            event_type=event_type, since=since, until=until, camera=camera,
            zone=zone, limit=limit, offset=offset)}

    @app.post("/ai/search-events")
    def ai_search_events(body: AiEventSearchIn = Body(...)):
        from faceflash.ai import answer_event_question, parse_event_question

        if event_store is None:
            return answer_event_question([], body.question, now=body.now,
                                         limit=body.limit)
        query = parse_event_question(body.question, now=body.now)
        events = event_store.query(
            event_types=query.get("event_types"),
            since=query.get("since"),
            until=query.get("until"),
            camera=query.get("camera"),
            zone=query.get("zone"),
            limit=max(body.limit, 1000),
        )
        return answer_event_question(
            events, body.question, now=body.now, limit=body.limit)

    @app.post("/ai/summarize-incident")
    def ai_summarize_incident(body: AiEventSearchIn = Body(...)):
        from faceflash.ai import answer_event_question, parse_event_question

        if event_store is None:
            result = answer_event_question([], body.question, now=body.now,
                                           limit=body.limit)
            return {"summary": result["summary"], "count": 0,
                    "query": result["query"]}
        query = parse_event_question(body.question, now=body.now)
        events = event_store.query(
            event_types=query.get("event_types"),
            since=query.get("since"),
            until=query.get("until"),
            camera=query.get("camera"),
            zone=query.get("zone"),
            limit=max(body.limit, 1000),
        )
        result = answer_event_question(
            events, body.question, now=body.now, limit=body.limit)
        return {
            "summary": result["summary"],
            "count": result["count"],
            "query": result["query"],
        }

    return app
