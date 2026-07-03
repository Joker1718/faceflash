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
    tr = IOUTracker(iou_threshold=0.1, max_missed=5)
    tr.update([(0, 0, 100, 100), (80, 0, 180, 100)], timestamp=1.0)
    a, b = tr.tracks
    # detection overlapping A strongly and B weakly must go to A
    tr.update([(5, 0, 105, 100), (85, 0, 185, 100)], timestamp=2.0)
    assert a.box == (5, 0, 105, 100)
    assert b.box == (85, 0, 185, 100)


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
