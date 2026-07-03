"""Stream/tracker/analytics tests — no camera or models needed."""

import numpy as np

from faceflash.stream import IOUTracker, Analytics, StreamMonitor, _iou, _nms


def test_iou_basics():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.1 < _iou((0, 0, 10, 10), (5, 0, 15, 10)) < 0.5


def test_nms_suppresses_overlaps():
    boxes = [(0, 0, 100, 100), (5, 5, 105, 105), (200, 200, 300, 300)]
    kept = _nms(boxes, iou_threshold=0.4)
    assert len(kept) == 2
    assert (200, 200, 300, 300) in kept


def test_tracker_matches_and_ages():
    tr = IOUTracker(iou_threshold=0.3, max_missed=2)

    tracks = tr.update([(0, 0, 100, 100)], timestamp=1.0)
    assert len(tracks) == 1
    tid = tracks[0].track_id

    # moved slightly -> same track
    tracks = tr.update([(10, 0, 110, 100)], timestamp=2.0)
    assert len(tracks) == 1 and tracks[0].track_id == tid
    assert tracks[0].frames_seen == 2

    # new face far away -> new track
    tracks = tr.update([(10, 0, 110, 100), (500, 500, 600, 600)], timestamp=3.0)
    assert len(tracks) == 2

    # both vanish -> aged out after max_missed frames
    tr.update([], timestamp=4.0)
    tr.update([], timestamp=5.0)
    tracks = tr.update([], timestamp=6.0)
    assert len(tracks) == 0


def test_tracker_greedy_matching_prefers_best_iou():
    tr = IOUTracker(iou_threshold=0.1, max_missed=5, box_smoothing=0)
    tr.update([(0, 0, 100, 100), (80, 0, 180, 100)], timestamp=1.0)
    a, b = tr.tracks
    # detection overlapping A strongly and B weakly must go to A
    tr.update([(5, 0, 105, 100), (85, 0, 185, 100)], timestamp=2.0)
    assert a.box == (5, 0, 105, 100)
    assert b.box == (85, 0, 185, 100)


def test_tracker_box_smoothing_damps_jitter():
    tr = IOUTracker(box_smoothing=0.5)
    tr.update([(100, 100, 200, 200)], timestamp=1.0)
    tr.update([(110, 100, 210, 200)], timestamp=2.0)   # jitter +10px
    x1 = tr.tracks[0].box[0]
    assert 100 < x1 < 110          # blended, not snapped


def test_display_grace_keeps_track_visible_through_missed_frames():
    ff_stub = type("E", (), {"n_candidates": 100})()
    mon = StreamMonitor(ff_stub, display_grace=3)
    mon.tracker.update([(0, 0, 50, 50)], timestamp=1.0)
    assert len(mon._visible_tracks()) == 1
    mon.tracker.update([], timestamp=2.0)   # detector blinks
    mon.tracker.update([], timestamp=3.0)
    assert len(mon._visible_tracks()) == 1  # still drawn (grace)
    for ts in (4.0, 5.0):
        mon.tracker.update([], timestamp=ts)
    assert len(mon._visible_tracks()) == 0  # grace exhausted


def test_analytics_visits_and_dwell():
    an = Analytics(visit_gap_seconds=60.0)
    an.person_appeared("alice", 100.0)
    an.person_update("alice", 130.0)
    assert an.get_visit_count("alice") == 1
    assert an.get_dwell_time("alice") == 30.0

    # within the gap -> same visit
    an.person_appeared("alice", 150.0)
    assert an.get_visit_count("alice") == 1

    # beyond the gap -> new visit
    an.person_appeared("alice", 300.0)
    assert an.get_visit_count("alice") == 2


def test_anomaly_alerts_emitted_once_per_track():
    ff_stub = type("E", (), {"n_candidates": 100})()
    mon = StreamMonitor(ff_stub, anomaly_threshold=2)
    events = []
    mon.on_event(events.append)

    # simulate what run() does after the threshold trips on track 7
    mon.analytics.unknown_sightings[7] = 5
    for _ in range(3):  # three frames
        for a in mon.analytics.get_anomalies():
            if a["track_id"] not in mon._alerted_tracks:
                mon._alerted_tracks.add(a["track_id"])
                mon._emit_anomaly = True
                from faceflash.stream import StreamEvent
                mon._emit(StreamEvent("anomaly", 0.0, a))
    assert len(events) == 1  # not three


def test_detect_returns_empty_on_blank_frame():
    ff_stub = type("E", (), {"n_candidates": 100})()
    mon = StreamMonitor(ff_stub)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    boxes, crops = mon._detect(blank)
    assert boxes == [] and crops == {}


# ── v0.2 stream upgrades: reader, re-ID, persistence, webhooks ──────────────

def test_latest_frame_reader_returns_newest_and_ends():
    from faceflash.stream import _LatestFrameReader
    import time as _t

    class FakeCap:
        def __init__(self, n):
            self.i, self.n = 0, n
        def read(self):
            if self.i >= self.n:
                return False, None
            self.i += 1
            f = np.full((4, 4, 3), self.i, dtype=np.uint8)
            _t.sleep(0.002)
            return True, f

    r = _LatestFrameReader(FakeCap(50))
    _t.sleep(0.05)
    alive, frame = r.read()
    assert alive and frame is not None
    first_seen = int(frame[0, 0, 0])
    assert first_seen > 1          # older frames were dropped, not queued
    # drains to the end and reports closed
    deadline = _t.time() + 2.0
    while _t.time() < deadline:
        alive, frame = r.read()
        if not alive and frame is None:
            break
        _t.sleep(0.005)
    assert not alive
    r.stop()


def test_tracker_reid_recovers_lost_identity():
    rng = np.random.default_rng(3)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    tr = IOUTracker(iou_threshold=0.3, max_missed=1)
    tracks = tr.update([(0, 0, 100, 100)], timestamp=1.0)
    tracks[0].identity, tracks[0].confidence = "alice", 0.9
    tracks[0].embedding = emb

    # face disappears -> track dies into re-ID memory
    tr.update([], timestamp=2.0)
    tr.update([], timestamp=3.0)
    assert len(tr.tracks) == 0 and len(tr.lost) == 1

    # same face reappears elsewhere
    m = tr.reid_match(emb, threshold=0.65)
    assert m is not None and m.identity == "alice"
    assert len(tr.lost) == 0       # consumed

    # a different face does NOT match
    tr2 = IOUTracker(max_missed=1)
    t = tr2.update([(0, 0, 50, 50)], timestamp=1.0)[0]
    t.embedding = emb
    tr2.update([], 2.0); tr2.update([], 3.0)
    other = rng.standard_normal(512).astype(np.float32)
    other /= np.linalg.norm(other)
    assert tr2.reid_match(other, threshold=0.65) is None


def test_events_persist_to_gallery_db(tmp_path):
    from faceflash.engine import FaceFlash
    from faceflash.stream import StreamMonitor, StreamEvent

    rng = np.random.default_rng(5)
    embs = rng.standard_normal((1200, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    ff = FaceFlash(index_path=str(tmp_path / "g.db"))
    ff.index.add_batch(embs, [f"p{i % 200}" for i in range(1200)])

    mon = StreamMonitor(ff)
    mon._emit(StreamEvent("person_appeared", 100.0, {"name": "p1"}))
    mon._emit(StreamEvent("person_left", 160.0, {"name": "p1", "dwell_seconds": 60.0}))
    mon._emit(StreamEvent("anomaly", 170.0, {"track_id": 9, "sightings": 3}))

    evs = mon.events()
    assert len(evs) == 3
    assert evs[0]["event_type"] == "anomaly"           # newest first
    assert mon.events(event_type="person_left")[0]["data"]["dwell_seconds"] == 60.0
    assert len(mon.events(since=150.0)) == 2
    ff.index.close()

    # events survive restart (same .db)
    ff2 = FaceFlash(index_path=str(tmp_path / "g.db"))
    mon2 = StreamMonitor(ff2)
    assert len(mon2.events()) == 3
    ff2.index.close()


def test_webhook_delivery():
    import json as _json
    import threading as _th
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from faceflash.stream import StreamMonitor, StreamEvent

    received = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(_json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200); self.end_headers()
        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    _th.Thread(target=srv.serve_forever, daemon=True).start()

    ff_stub = type("E", (), {"n_candidates": 100})()
    mon = StreamMonitor(ff_stub)
    mon.on_event(f"http://127.0.0.1:{srv.server_port}/hook")
    mon._emit(StreamEvent("person_appeared", 1.0, {"name": "bob"}))

    import time as _t
    deadline = _t.time() + 3.0
    while not received and _t.time() < deadline:
        _t.sleep(0.02)
    srv.shutdown()
    assert received and received[0]["event_type"] == "person_appeared"
    assert received[0]["data"]["name"] == "bob"


# ── anonymous visitor counting + attendance ─────────────────────────────────

def test_visitor_registry_counts_unique_people():
    from faceflash.stream import _VisitorRegistry
    rng = np.random.default_rng(9)
    a = rng.standard_normal(512).astype(np.float32); a /= np.linalg.norm(a)
    b = rng.standard_normal(512).astype(np.float32); b /= np.linalg.norm(b)

    reg = _VisitorRegistry(threshold=0.55)
    v1, new1 = reg.assign(a)
    v1_again, new2 = reg.assign(a * 0.99 + 0.01 * b / np.linalg.norm(a * 0.99 + 0.01 * b))
    v2, new3 = reg.assign(b)

    assert v1 == "visitor_1" and new1
    assert v1_again == "visitor_1" and not new2   # same person, same ID
    assert v2 == "visitor_2" and new3             # different person, new ID
    assert reg.count == 2


def test_attendance_summary():
    an = Analytics(visit_gap_seconds=60.0)
    an.person_appeared("alice", 100.0)
    an.person_update("alice", 150.0)
    an.person_appeared("alice", 400.0)     # second visit (gap > 60s)
    an.person_update("alice", 430.0)

    att = an.attendance()
    assert att["alice"]["visits"] == 2
    assert att["alice"]["first_seen"] == 100.0
    assert att["alice"]["total_dwell_seconds"] == 80.0


def test_monitor_report_includes_visitor_count():
    ff_stub = type("E", (), {"n_candidates": 100})()
    mon = StreamMonitor(ff_stub)
    rng = np.random.default_rng(11)
    e = rng.standard_normal(512).astype(np.float32); e /= np.linalg.norm(e)
    mon.visitors.assign(e)
    r = mon.report()
    assert r["unique_unregistered_visitors"] == 1
    assert "attendance" in r
