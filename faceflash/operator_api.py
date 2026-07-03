"""Minimal FastAPI operator API for camera/zone/event management."""

from typing import Optional


def create_app(db_path: str):
    """Create a FastAPI app backed by a FaceFlash gallery DB.

    Install with: pip install fastapi uvicorn
    Run with: uvicorn 'faceflash.operator_api:create_app("faces.db")'
    """
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as e:
        raise ImportError(
            "FastAPI operator API requires: pip install fastapi uvicorn"
        ) from e

    from faceflash import FaceFlash

    ff = FaceFlash(index_path=db_path)
    app = FastAPI(title="FaceFlash Operator API", version="0.1")

    class CameraIn(BaseModel):
        name: str
        source: str = "0"
        settings: dict = {}

    class ZoneIn(BaseModel):
        name: str
        polygon: list
        secure: bool = False
        settings: dict = {}

    class AiEventSearchIn(BaseModel):
        question: str
        limit: int = 100
        now: Optional[float] = None

    @app.get("/status")
    def status():
        return {"ok": True, "stats": ff.stats(),
                "cameras": len(ff.index.list_camera_configs())}

    @app.get("/cameras")
    def list_cameras():
        return {"cameras": ff.index.list_camera_configs()}

    @app.put("/cameras/{name}")
    def save_camera(name: str, body: CameraIn):
        settings = dict(body.settings)
        cfg = ff.index.save_camera_config(name, body.source, **settings)
        return {"camera": cfg}

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
               limit: int = 1000):
        # Use a monitor only as the event query facade; persisted events live
        # in the same gallery DB.
        from faceflash.stream import StreamMonitor
        mon = StreamMonitor(ff, persist_events=True)
        return {"events": mon.events(event_type=event_type,
                                     since=since,
                                     limit=limit)}

    @app.post("/ai/search-events")
    def ai_search_events(body: AiEventSearchIn = Body(...)):
        from faceflash.ai import answer_event_question
        from faceflash.stream import StreamMonitor

        mon = StreamMonitor(ff, persist_events=True)
        events = mon.events(limit=max(1000, body.limit * 5))
        return answer_event_question(
            events, body.question, now=body.now, limit=body.limit)

    @app.post("/ai/summarize-incident")
    def ai_summarize_incident(body: AiEventSearchIn = Body(...)):
        from faceflash.ai import answer_event_question
        from faceflash.stream import StreamMonitor

        mon = StreamMonitor(ff, persist_events=True)
        events = mon.events(limit=max(1000, body.limit * 5))
        result = answer_event_question(
            events, body.question, now=body.now, limit=body.limit)
        return {
            "summary": result["summary"],
            "count": result["count"],
            "query": result["query"],
        }

    return app
