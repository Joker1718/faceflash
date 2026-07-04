from datetime import datetime

import pytest

from faceflash.ai import answer_event_question, parse_event_question


def _ts(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def test_parse_event_question_handles_yesterday_after_time():
    now = _ts("2026-07-03T12:00:00")
    q = parse_event_question(
        "Show unknown visitors near lab-door yesterday after 6pm",
        now=now,
    )

    assert q["event_types"] == ["enrollment_suggested", "visitor_new"]
    assert q["zone"] == "lab-door"
    assert q["since"] == _ts("2026-07-02T18:00:00")
    assert q["until"] == _ts("2026-07-03T00:00:00")


def test_answer_event_question_filters_and_summarizes_real_events():
    events = [
        {
            "timestamp": _ts("2026-07-02T18:15:00"),
            "event_type": "visitor_new",
            "data": {"camera": "front", "zone": "lab-door", "visitor_id": "visitor_1"},
        },
        {
            "timestamp": _ts("2026-07-02T19:00:00"),
            "event_type": "capacity_exceeded",
            "data": {"camera": "front", "zone": "lab-door"},
        },
        {
            "timestamp": _ts("2026-07-01T18:30:00"),
            "event_type": "visitor_new",
            "data": {"camera": "front", "zone": "lab-door", "visitor_id": "visitor_old"},
        },
    ]

    result = answer_event_question(
        events,
        "unknown visitors in lab-door yesterday after 6pm",
        now=_ts("2026-07-03T12:00:00"),
    )

    assert result["count"] == 1
    assert result["events"][0]["data"]["visitor_id"] == "visitor_1"
    assert "Found 1 matching event" in result["summary"]
    assert "visitor_new=1" in result["summary"]


def test_operator_api_ai_search_events(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from faceflash import FaceFlash
    from faceflash.operator_api import create_app
    from faceflash.stream import StreamEvent, StreamMonitor

    db = str(tmp_path / "faces.db")
    ff = FaceFlash(index_path=db)
    monitor = StreamMonitor(ff, persist_events=True)
    monitor._emit(StreamEvent(
        "tailgating_risk",
        _ts("2026-07-03T10:00:00"),
        {"camera": "front", "zone": "entry"},
    ))
    ff.index.close()

    app = create_app(db)
    client = TestClient(app)
    response = client.post("/ai/search-events", json={
        "question": "tailgating in entry today",
        "now": _ts("2026-07-03T12:00:00"),
        "limit": 10,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["event_type"] == "tailgating_risk"
    assert "tailgating_risk=1" in payload["summary"]


def test_operator_api_redacts_camera_source(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from faceflash.operator_api import create_app

    client = TestClient(create_app(str(tmp_path / "faces.db")))
    response = client.put("/cameras/front", json={
        "name": "front",
        "source": "rtsp://user:secret@example.test:554/live",
        "settings": {"mode": "anonymous"},
    })

    assert response.status_code == 200
    camera = response.json()["camera"]
    assert "secret" not in camera["source"]
    assert camera["source"] == "rtsp://***:***@example.test:554/live"

    response = client.get("/cameras")
    assert "secret" not in response.text


def test_operator_api_events_query_uses_event_store(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from faceflash import FaceFlash
    from faceflash.events import EventStore
    from faceflash.operator_api import create_app

    db = str(tmp_path / "faces.db")
    ff = FaceFlash(index_path=db)
    store = EventStore(ff.index._db, lock=ff.index._db_lock)
    store.insert(_ts("2026-07-03T09:00:00"), "visitor_new",
                 {"camera": "front", "zone": "lobby"})
    store.insert(_ts("2026-07-03T10:00:00"), "tailgating_risk",
                 {"camera": "front", "zone": "entry"})
    ff.index.close()

    client = TestClient(create_app(db))
    response = client.get("/events", params={
        "event_type": "tailgating_risk",
        "zone": "entry",
        "limit": 5,
    })

    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["event_type"] == "tailgating_risk"
