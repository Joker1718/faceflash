"""
Real-time face identification from camera/RTSP streams.

Connects to a webcam or RTSP URL, samples frames, detects + tracks + identifies
faces continuously, and provides analytics (crowd count, dwell time, visit
frequency, anomaly alerts).

Status: a working single-camera prototype — good for demos, personal
projects, and small deployments. It is NOT a multi-camera production
pipeline: no GPU batching, no failover, no certified anti-spoofing. If you
need those, run a dedicated NVR (e.g. Frigate) and use FaceFlash as the
recognition backend.

Requires: opencv-python (cv2)

Usage:
    from faceflash import FaceFlash
    from faceflash.stream import StreamMonitor

    ff = FaceFlash("my_index.db")
    monitor = StreamMonitor(ff)

    # Callback for events
    monitor.on_event(lambda e: print(e))

    # Start processing (blocking)
    monitor.run("rtsp://camera_ip:554/stream")
    # Or webcam: monitor.run(0)
"""

import time
import json
import queue
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Union

import numpy as np

logger = logging.getLogger(__name__)


class _WebhookSender:
    """Posts events as JSON to a URL from a worker thread.

    A slow or dead endpoint must never stall the video loop, so events go
    through a bounded queue; overflow drops the oldest event and logs.
    """

    def __init__(self, url: str, timeout: float = 3.0, max_queue: int = 256):
        self.url = url
        self.timeout = timeout
        self._q: "queue.Queue" = queue.Queue(maxsize=max_queue)
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def __call__(self, event: "StreamEvent"):
        payload = {"event_type": event.event_type,
                   "timestamp": event.timestamp,
                   "data": event.data}
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()          # drop oldest
                self._q.put_nowait(payload)
            except queue.Empty:
                pass
            logger.warning("webhook queue full — dropping oldest event")

    def _worker(self):
        import urllib.request
        while True:
            payload = self._q.get()
            try:
                req = urllib.request.Request(
                    self.url, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=self.timeout).read()
            except Exception as e:
                logger.warning("webhook delivery failed: %s", e)


def _point_in_polygon(x: float, y: float, poly) -> bool:
    """Ray-casting point-in-polygon. poly = [(x, y), ...]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and                 x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _sharpness(crop: "np.ndarray") -> float:
    """Blur score: variance of the Laplacian on the aligned crop.
    Sharp faces score in the hundreds; motion blur drops below ~30."""
    try:
        import cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 1e9   # can't measure -> don't block identification


class _VisitorRegistry:
    """Counts UNIQUE unregistered people by clustering their embeddings.

    Every unknown face gets matched against the visitors seen so far
    (cosine against a running-mean embedding); a match reuses that
    pseudonymous ID, a miss mints visitor_N. So five appearances of the
    same stranger count as ONE visitor, not five unknowns.

    Privacy: embeddings live in memory for this run only — nothing about
    unregistered people is written to disk (events carry only the
    pseudonymous ID).
    """

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self._ids: List[str] = []
        self._centroids: List[np.ndarray] = []
        self._counts: List[int] = []

    def assign(self, embedding: np.ndarray) -> tuple:
        """Returns (visitor_id, is_new)."""
        if self._centroids:
            sims = np.array([float(np.dot(c, embedding)) for c in self._centroids])
            best = int(sims.argmax())
            if sims[best] >= self.threshold:
                # fold into the running mean (keeps the centroid stable)
                n = self._counts[best]
                c = (self._centroids[best] * n + embedding) / (n + 1)
                self._centroids[best] = c / (np.linalg.norm(c) + 1e-12)
                self._counts[best] += 1
                return self._ids[best], False
        vid = f"visitor_{len(self._ids) + 1}"
        self._ids.append(vid)
        self._centroids.append(embedding / (np.linalg.norm(embedding) + 1e-12))
        self._counts.append(1)
        return vid, True

    @property
    def count(self) -> int:
        return len(self._ids)


class _LatestFrameReader:
    """Capture thread that always holds only the newest frame.

    On live sources, if processing runs slower than the stream, OpenCV
    buffers frames and the monitor falls further and further behind live.
    This thread drains the source continuously so the main loop always
    processes the most recent frame. Only used for live sources — draining
    a FILE this way would skip most of it nondeterministically.
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._alive = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._alive:
            ret, frame = self._cap.read()
            if not ret:
                self._alive = False
                return
            with self._lock:
                self._frame = frame

    def read(self):
        """(source_still_open, latest_frame_or_None). The frame is consumed:
        the same frame is never returned twice."""
        with self._lock:
            frame, self._frame = self._frame, None
        return (self._alive or frame is not None), frame

    def stop(self):
        self._alive = False


# ─────────────────────────────────────────────────────────────────────────────
# IOU Tracker — lightweight multi-face tracking without deep features
# ─────────────────────────────────────────────────────────────────────────────

def _nms(boxes, iou_threshold: float = 0.4):
    """Greedy non-maximum suppression on (x1,y1,x2,y2) boxes, largest first."""
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                   reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_threshold for k in kept):
            kept.append(b)
    return kept


def _iou(box_a, box_b):
    """Intersection over union of two (x1,y1,x2,y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


@dataclass
class Track:
    """A tracked face across frames."""
    track_id: int
    box: tuple  # (x1, y1, x2, y2)
    identity: Optional[str] = None
    visitor_id: Optional[str] = None   # pseudonymous ID for unregistered faces
    confidence: float = 0.0
    quality: float = 0.0               # sharpness of the best crop seen
    zones: frozenset = frozenset()     # zone names the face is currently in
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_seen: int = 0
    frames_missed: int = 0
    embedding: Optional[np.ndarray] = None


class IOUTracker:
    """Simple IOU-based multi-object tracker.

    Matches detections to existing tracks by spatial overlap.
    No deep features needed — fast and sufficient for stationary cameras.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15,
                 reid_retention_seconds: float = 120.0,
                 box_smoothing: float = 0.5):
        """
        iou_threshold: minimum IOU to associate a detection with a track.
        max_missed: drop a track after this many consecutive frames without a match.
        reid_retention_seconds: keep dropped tracks that carry a face embedding
            around for this long, so a person who reappears can be re-matched
            by appearance instead of getting a fresh track (see reid_match()).
        """
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.reid_retention = reid_retention_seconds
        self.box_smoothing = box_smoothing
        self.tracks: List[Track] = []
        self.lost: List[Track] = []       # dropped-but-remembered (have embeddings)
        self._next_id = 0

    def _retire(self, dead: List[Track], timestamp: float):
        """Move dead tracks with embeddings into the re-ID memory."""
        self.lost.extend(t for t in dead if t.embedding is not None)
        self.lost = [t for t in self.lost
                     if timestamp - t.last_seen <= self.reid_retention][-64:]

    def reid_match(self, embedding: np.ndarray, threshold: float = 0.65):
        """Best recently-lost track whose face embedding matches, or None.

        The face embedding IS an appearance feature — this is what DeepSORT
        adds a whole second network for, and we get it free from the
        identification step. Matching removes the track from re-ID memory.
        """
        if not self.lost:
            return None
        best_i, best_sim = -1, threshold
        for i, t in enumerate(self.lost):
            sim = float(np.dot(t.embedding, embedding))
            if sim >= best_sim:
                best_i, best_sim = i, sim
        if best_i < 0:
            return None
        return self.lost.pop(best_i)

    def update(self, detections: List[tuple], timestamp: float) -> List[Track]:
        """Update tracks with new frame detections.

        Args:
            detections: list of (x1, y1, x2, y2) bounding boxes
            timestamp: frame timestamp (seconds)

        Returns:
            Current active tracks (matched + new)
        """
        if not detections:
            # Age out all tracks
            for t in self.tracks:
                t.frames_missed += 1
            dead = [t for t in self.tracks if t.frames_missed > self.max_missed]
            self.tracks = [t for t in self.tracks if t.frames_missed <= self.max_missed]
            self._retire(dead, timestamp)
            return self.tracks

        # Match detections to existing tracks by IOU
        matched_tracks = set()
        matched_dets = set()

        # Greedy matching: best IOU first
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = _iou(track.box, det)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        for score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            # Update track. The raw detection box jitters a few pixels every
            # frame; blend it with the previous box so the drawn rectangle is
            # steady. (Matching already happened, so smoothing can't break it.)
            if self.box_smoothing > 0:
                a = self.box_smoothing
                old_box = self.tracks[ti].box
                self.tracks[ti].box = tuple(
                    int(round(a * o + (1 - a) * n))
                    for o, n in zip(old_box, detections[di]))
            else:
                self.tracks[ti].box = detections[di]
            self.tracks[ti].last_seen = timestamp
            self.tracks[ti].frames_seen += 1
            self.tracks[ti].frames_missed = 0
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Age unmatched tracks
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.frames_missed += 1

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(Track(
                    track_id=self._next_id,
                    box=det,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frames_seen=1,
                ))
                self._next_id += 1

        # Remove dead tracks (embedded ones go to re-ID memory)
        dead = [t for t in self.tracks if t.frames_missed > self.max_missed]
        self.tracks = [t for t in self.tracks if t.frames_missed <= self.max_missed]
        self._retire(dead, timestamp)

        return self.tracks


# ─────────────────────────────────────────────────────────────────────────────
# Analytics Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisitRecord:
    """A single visit by a person."""
    identity: str
    first_seen: float
    last_seen: float
    dwell_seconds: float = 0.0


class Analytics:
    """Tracks crowd metrics, dwell time, visit frequency, and anomalies.

    Fed by the stream monitor with track updates every frame.
    """

    def __init__(self, visit_gap_seconds: float = 300.0,
                 anomaly_threshold: int = 3):
        """
        visit_gap_seconds: gap between sightings to count as a new visit (default 5 min).
        anomaly_threshold: unknown face seen this many visits triggers alert.
        """
        self.visit_gap = visit_gap_seconds
        self.anomaly_threshold = anomaly_threshold

        # Current state
        self._active_tracks: Dict[int, Track] = {}
        self._last_seen: Dict[str, float] = {}  # identity → last timestamp

        # Persistent counters
        self.visits: Dict[str, List[VisitRecord]] = defaultdict(list)
        self.unknown_sightings: Dict[int, int] = defaultdict(int)  # track_id → count

    @property
    def crowd_count(self) -> int:
        """Number of distinct faces currently visible."""
        return len(self._active_tracks)

    def unique_people_count(self) -> int:
        """Number of distinct identified people currently visible."""
        identities = {t.identity for t in self._active_tracks.values() if t.identity}
        return len(identities)

    def update(self, tracks: List[Track], timestamp: float):
        """Called every frame with current tracks."""
        self._active_tracks = {t.track_id: t for t in tracks}

    def person_appeared(self, identity: str, timestamp: float):
        """Called when a track gets identified."""
        last = self._last_seen.get(identity, 0)
        if timestamp - last > self.visit_gap:
            # New visit
            self.visits[identity].append(VisitRecord(
                identity=identity,
                first_seen=timestamp,
                last_seen=timestamp,
            ))
        self._last_seen[identity] = timestamp

    def person_update(self, identity: str, timestamp: float):
        """Called on each frame a known person is still visible."""
        self._last_seen[identity] = timestamp
        if self.visits[identity]:
            v = self.visits[identity][-1]
            v.last_seen = timestamp
            v.dwell_seconds = timestamp - v.first_seen

    def get_dwell_time(self, identity: str) -> float:
        """Current dwell time (seconds) for a person's latest visit."""
        if self.visits[identity]:
            return self.visits[identity][-1].dwell_seconds
        return 0.0

    def get_visit_count(self, identity: str) -> int:
        """Total number of distinct visits by this person."""
        return len(self.visits[identity])

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Faces seen frequently that are NOT in the gallery."""
        alerts = []
        for track_id, count in self.unknown_sightings.items():
            if count >= self.anomaly_threshold:
                alerts.append({
                    "track_id": track_id,
                    "sightings": count,
                    "alert": "unknown_frequent_visitor",
                })
        return alerts

    def report(self) -> Dict[str, Any]:
        """Current analytics snapshot."""
        return {
            "crowd_count": self.crowd_count,
            "unique_identified": self.unique_people_count(),
            "total_visits_today": sum(len(v) for v in self.visits.values()),
            "anomalies": self.get_anomalies(),
            "top_visitors": sorted(
                [(k, len(v)) for k, v in self.visits.items()],
                key=lambda x: -x[1]
            )[:10],
        }

    def attendance(self) -> Dict[str, Dict[str, Any]]:
        """Per-person attendance summary for this run: first/last seen,
        visit count, total dwell. The raw material for a timesheet."""
        out = {}
        for identity, vlist in self.visits.items():
            if not vlist:
                continue
            out[identity] = {
                "first_seen": vlist[0].first_seen,
                "last_seen": vlist[-1].last_seen,
                "visits": len(vlist),
                "total_dwell_seconds": round(sum(v.dwell_seconds for v in vlist), 1),
            }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Stream Monitor — ties camera + detection + tracking + identification + analytics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamEvent:
    """An event emitted by the stream monitor."""
    event_type: str  # "person_appeared", "person_left", "anomaly", "frame"
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class StreamMonitor:
    """Real-time face identification + analytics from a video stream.

    Connects to a webcam index (0, 1, ...) or RTSP/HTTP URL.
    Detects faces, tracks them across frames, identifies against the gallery,
    and maintains analytics (crowd count, dwell time, visits, anomalies).

    Usage:
        ff = FaceFlash("gallery.db")
        monitor = StreamMonitor(ff, sample_fps=2.0)
        monitor.on_event(my_callback)
        monitor.run(0)  # webcam
    """

    def __init__(self, engine, sample_fps: float = 2.0,
                 identify_interval: int = 5,
                 threshold: float = 0.4,
                 max_far: Optional[float] = None,
                 iou_threshold: float = 0.3,
                 max_missed_frames: int = 30,
                 visit_gap_seconds: float = 300.0,
                 anomaly_threshold: int = 3,
                 reid_threshold: float = 0.65,
                 persist_events: bool = True,
                 det_size: int = 480,
                 det_model: str = "10g",
                 display_grace: int = 4,
                 min_track_frames: int = 2,
                 min_det_score: float = 0.55,
                 idle_fps: float = 1.0,
                 min_sharpness: float = 20.0,
                 swap_threshold: float = 0.35,
                 camera_name: Optional[str] = None,
                 snapshot_dir: Optional[str] = None):
        """
        engine: a FaceFlash instance with a loaded gallery.
        sample_fps: how many frames/sec to process (lower = less CPU).
        identify_interval: re-identify a track every N frames it's seen.
        threshold: cosine similarity threshold for a match.
        max_far: if set, uses FAR-calibrated threshold instead.
        iou_threshold: IOU for tracker association.
        max_missed_frames: frames before a lost track is dropped.
        visit_gap_seconds: gap to count as a new visit.
        anomaly_threshold: unknown-face sighting count that triggers alert.
        reid_threshold: cosine similarity to re-match a reappearing face to a
            recently-lost track (appearance re-ID; uses the face embedding).
        persist_events: write events to a stream_events table in the gallery's
            SQLite file (no-op for the legacy .npy backend).
        det_size: detector input resolution. Detection dominates the frame
            cost and scales with this (measured on CPU: 640 -> 68ms,
            480 -> 42ms, 320 -> 21ms per frame). 480 (default) is right for
            typical webcam/door-cam faces; use 640 when faces are small in
            the frame, 320 when CPU is tight.
        det_model: "10g" (default) or "500m" — the light SCRFD is ~5x faster
            still (7.9ms @480 on CPU) at some accuracy cost on small/hard
            faces. det_model="500m", det_size=320 runs a full pipeline in
            ~10ms/frame on plain CPU (Raspberry-Pi territory).
        display_grace: keep drawing a track (and counting its identity as
            present) for this many processed frames after the detector last
            saw it. Detectors blink on single frames constantly; without
            grace, boxes/names flash on and off and person_left/appeared
            events churn. 0 restores the strict old behavior.
        min_track_frames: a new track must be detected this many times before
            it's drawn or counted — kills ghost boxes from single-frame
            false positives.
        min_det_score: don't spend embedding+search on detections below this
            confidence; they're usually junk crops that mis-identify.
        idle_fps: processing rate while NO faces are tracked (live sources).
            An attendance camera is idle most of the day — no point burning
            sample_fps worth of detection on empty frames. Ramps back to
            sample_fps the moment a face appears.
        min_sharpness: skip identification on crops blurrier than this
            (Laplacian variance; sharp faces score 100+, motion blur <30).
            The track retries on the next, hopefully sharper, frame. The
            stored track embedding always comes from the sharpest crop seen.
        swap_threshold: if a track's new embedding disagrees with its stored
            one below this cosine, a DIFFERENT person now owns the box (two
            people crossed paths) — identity resets and re-resolves instead
            of sticking with the swapped name.
        camera_name: tag for every event this monitor emits ("door", "lobby").
        snapshot_dir: OPT-IN event replay — save a face crop JPEG alongside
            person_appeared / visitor_new events (path lands in the event
            data, queryable via events()). Off by default: this stores images
            of everyone, including strangers — check your consent obligations.
        """
        self.engine = engine
        self.sample_fps = sample_fps
        self.identify_interval = identify_interval
        self.threshold = threshold
        self.max_far = max_far
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold

        self.tracker = IOUTracker(iou_threshold=iou_threshold,
                                  max_missed=max_missed_frames)
        self.analytics = Analytics(visit_gap_seconds=visit_gap_seconds,
                                   anomaly_threshold=anomaly_threshold)

        self._callbacks: List[Callable[[StreamEvent], None]] = []
        self._running = False
        self._frame_count = 0
        self._alerted_tracks = set()   # anomaly alerts already emitted
        self._prev_identities = set()
        self.det_size = det_size
        self.det_model = det_model
        self.display_grace = display_grace
        self.min_track_frames = min_track_frames
        self.min_det_score = min_det_score
        self.idle_fps = idle_fps
        self.visitors = _VisitorRegistry()
        self.min_sharpness = min_sharpness
        self.swap_threshold = swap_threshold
        self.camera_name = camera_name
        self.snapshot_dir = snapshot_dir
        self.zones: Dict[str, list] = {}
        self._engine_lock = threading.Lock()   # shared when multi-camera
        self._detector = None          # lazy; False = tried and failed

        # Persist events into the SAME faces.db the gallery lives in, so
        # analytics history survives restarts and is queryable with SQL.
        self._events_db = None
        if persist_events:
            db = getattr(getattr(engine, "index", None), "_db", None)
            if db is not None:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS stream_events ("
                    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    " ts REAL NOT NULL,"
                    " event_type TEXT NOT NULL,"
                    " data TEXT NOT NULL DEFAULT '{}')")
                db.commit()
                self._events_db = db

    def on_event(self, callback):
        """Register an event callback, or a webhook URL string.

        A URL gets events POSTed as JSON from a background worker
        ({"event_type", "timestamp", "data"}) — delivery never blocks the
        video loop, failures are logged.
        """
        if isinstance(callback, str):
            callback = _WebhookSender(callback)
        self._callbacks.append(callback)

    def add_zone(self, name: str, polygon):
        """Define a named zone as a polygon of (x, y) pixel coords.
        Tracks entering/leaving fire zone_entered / zone_exited events, and
        report() includes live per-zone counts."""
        if len(polygon) < 3:
            raise ValueError("A zone needs at least 3 points")
        self.zones[name] = [(float(x), float(y)) for x, y in polygon]

    def _emit(self, event: StreamEvent):
        if self.camera_name is not None:
            event.data.setdefault("camera", self.camera_name)
        if self._events_db is not None:
            try:
                self._events_db.execute(
                    "INSERT INTO stream_events(ts, event_type, data) VALUES (?, ?, ?)",
                    (event.timestamp, event.event_type, json.dumps(event.data)))
                self._events_db.commit()
            except Exception as e:
                logger.warning("event persistence failed: %s", e)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning("Event callback error: %s", e)

    def events(self, event_type: Optional[str] = None,
               since: Optional[float] = None, limit: int = 1000) -> List[dict]:
        """Query persisted events (newest first) from the gallery's .db."""
        if self._events_db is None:
            return []
        q = "SELECT ts, event_type, data FROM stream_events"
        cond, args = [], []
        if event_type is not None:
            cond.append("event_type = ?"); args.append(event_type)
        if since is not None:
            cond.append("ts >= ?"); args.append(since)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY ts DESC LIMIT ?"; args.append(limit)
        return [{"timestamp": r[0], "event_type": r[1], "data": json.loads(r[2])}
                for r in self._events_db.execute(q, args).fetchall()]

    def stop(self):
        """Stop the stream processing loop."""
        self._running = False

    def run(self, source: Union[int, str], show: bool = False,
            max_frames: Optional[int] = None):
        """Process a video stream (blocking).

        Args:
            source: webcam index (0) or RTSP/HTTP URL string.
            show: if True, display annotated frames (requires GUI).
            max_frames: stop after this many processed frames (for testing).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required: pip install opencv-python")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        is_live = isinstance(source, int) or str(source).lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://"))

        self._running = True
        self._frame_count = 0
        self._prev_identities = set()

        logger.info("Stream started: source=%s, live=%s, sample_fps=%.1f",
                    source, is_live, self.sample_fps)
        try:
            if is_live:
                self._run_live(cap, cv2, show, max_frames)
            else:
                self._run_file(cap, cv2, show, max_frames)
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
            self._running = False
            logger.info("Stream stopped. Frames processed: %d", self._frame_count)

    def _run_live(self, cap, cv2, show, max_frames):
        """Live sources: a capture thread drains the stream so processing
        never falls behind real time; frames are processed at sample_fps."""
        reader = _LatestFrameReader(cap)
        last_proc = 0.0
        try:
            while self._running:
                alive, frame = reader.read()
                if not alive:
                    break
                if frame is None:
                    time.sleep(0.005)
                    continue

                # Idle scenes don't deserve full detection rate: drop to
                # idle_fps when nothing is tracked, snap back on first face.
                rate = self.sample_fps if self.tracker.tracks else                     min(self.idle_fps, self.sample_fps)
                min_gap = 1.0 / rate
                now = time.time()
                if now - last_proc < min_gap:
                    # Not due for processing — keep the display smooth
                    if show:
                        self._draw_frame(frame, self._visible_tracks())
                        cv2.imshow("FaceFlash Stream", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        time.sleep(0.005)
                    continue
                last_proc = now

                self._frame_count += 1
                if max_frames and self._frame_count > max_frames:
                    break
                active = self._process_frame(frame, cv2)
                if show:
                    self._draw_frame(frame, active)
                    cv2.imshow("FaceFlash Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            reader.stop()

    def _run_file(self, cap, cv2, show, max_frames):
        """File sources: synchronous sequential reads (draining a file the
        live way would skip most of it), sampled by frame index."""
        stream_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(stream_fps / self.sample_fps))
        frame_idx = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Show every frame (smooth video) but only process some
            if frame_idx % frame_interval != 0:
                if show:
                    self._draw_frame(frame, self._visible_tracks())
                    cv2.imshow("FaceFlash Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            self._frame_count += 1
            if max_frames and self._frame_count > max_frames:
                break
            active = self._process_frame(frame, cv2)
            if show:
                self._draw_frame(frame, active)
                cv2.imshow("FaceFlash Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def _process_frame(self, frame, cv2) -> List[Track]:
        """Detect → track → identify → analytics → events, for one frame.
        Returns the active tracks (for drawing)."""
        timestamp = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ONE detector pass: boxes for the tracker + aligned
        # crops for identification (no re-detection per track)
        boxes, crop_by_box, score_by_box = self._detect(frame_rgb)

        # Update tracker (fast — just IOU matching)
        tracks = self.tracker.update(boxes, timestamp)
        self.analytics.update(tracks, timestamp)

        # Only identify tracks actually detected THIS frame (their aligned
        # crops exist). All due tracks embed in ONE batched ONNX call.
        detected_now = [t for t in tracks if t.frames_missed == 0]
        due = [t for t in detected_now
               if score_by_box.get(tuple(t.box), 1.0) >= self.min_det_score
               and ((t.identity is None and t.frames_seen >= 3)
                    or (t.identity is not None and
                        t.frames_seen % self.identify_interval == 0))]
        if due:
            self._identify_tracks(due, frame_rgb, crop_by_box)

        # Display + presence use a grace window: a single missed detection
        # must not blank the overlay or fire person_left/appeared churn.
        active_tracks = self._visible_tracks()

        # Events
        current_identities = {t.identity for t in active_tracks if t.identity}
        appeared = current_identities - self._prev_identities
        left = self._prev_identities - current_identities

        for name in appeared:
            self.analytics.person_appeared(name, timestamp)
            self._emit(StreamEvent("person_appeared", timestamp,
                                   {"name": name}))
        for name in left:
            dwell = self.analytics.get_dwell_time(name)
            self._emit(StreamEvent("person_left", timestamp,
                                   {"name": name, "dwell_seconds": round(dwell, 1)}))
        for name in current_identities:
            self.analytics.person_update(name, timestamp)

        # Zones: membership by box center; fire enter/exit on change
        if self.zones:
            for track in active_tracks:
                cx = (track.box[0] + track.box[2]) / 2.0
                cy = (track.box[1] + track.box[3]) / 2.0
                now_in = frozenset(z for z, poly in self.zones.items()
                                   if _point_in_polygon(cx, cy, poly))
                who = track.identity or track.visitor_id or f"track_{track.track_id}"
                for z in now_in - track.zones:
                    self._emit(StreamEvent("zone_entered", timestamp,
                                           {"zone": z, "name": who}))
                for z in track.zones - now_in:
                    self._emit(StreamEvent("zone_exited", timestamp,
                                           {"zone": z, "name": who}))
                track.zones = now_in

        # Opt-in event replay: snapshot the face behind person_appeared
        if self.snapshot_dir and appeared:
            self._save_snapshots(appeared, active_tracks, crop_by_box, timestamp)

        # Anomaly check (counts only frames the face was actually detected)
        for track in detected_now:
            if track.identity is None and track.frames_seen > 10:
                self.analytics.unknown_sightings[track.track_id] += 1
        for a in self.analytics.get_anomalies():
            if a["track_id"] not in self._alerted_tracks:
                self._alerted_tracks.add(a["track_id"])
                self._emit(StreamEvent("anomaly", timestamp, a))

        self._prev_identities = current_identities
        return active_tracks

    def report(self) -> Dict[str, Any]:
        """Analytics snapshot incl. the unique-unregistered-visitor count."""
        r = self.analytics.report()
        r["unique_unregistered_visitors"] = self.visitors.count
        r["attendance"] = self.analytics.attendance()
        if self.zones:
            counts = {z: 0 for z in self.zones}
            for t in self._visible_tracks():
                for z in t.zones:
                    counts[z] += 1
            r["zones"] = counts
        return r

    def _save_snapshots(self, appeared, active_tracks, crop_by_box, timestamp):
        """Write a JPEG of each newly-appeared person's crop (opt-in replay)."""
        try:
            from PIL import Image
            from pathlib import Path as _P
            out = _P(self.snapshot_dir)
            out.mkdir(parents=True, exist_ok=True)
            for name in appeared:
                track = next((t for t in active_tracks if t.identity == name), None)
                crop = crop_by_box.get(tuple(track.box)) if track else None
                if crop is None:
                    continue
                fname = f"{int(timestamp)}_{name}.jpg"
                Image.fromarray(crop).save(out / fname, quality=85)
                self._emit(StreamEvent("snapshot", timestamp,
                                       {"name": name, "path": str(out / fname)}))
        except Exception as e:
            logger.warning("snapshot save failed: %s", e)

    def _visible_tracks(self) -> List[Track]:
        """Tracks to draw / count as present: seen enough times to be real
        (no single-frame ghost boxes), detected recently enough (grace)."""
        return [t for t in self.tracker.tracks
                if t.frames_seen >= self.min_track_frames
                and t.frames_missed <= self.display_grace]

    def _get_detector(self):
        """Stream-owned SCRFD at det_size (the shared one is fixed at 640)."""
        if self._detector is None:
            try:
                from faceflash.align import FaceDetector
                self._detector = FaceDetector(input_size=self.det_size,
                                              model=self.det_model)
            except Exception as e:
                logger.warning("SCRFD unavailable (%s) — using Haar fallback", e)
                self._detector = False
        return self._detector or None

    def _detect(self, frame_rgb: np.ndarray):
        """One detector pass → (boxes, {box: aligned_crop}).

        SCRFD gives boxes AND landmarks in a single inference, so the aligned
        112x112 crops come for free — identification then never re-runs the
        detector on track crops. When SCRFD ran and found nothing, that IS
        the answer — Haar only fills in when SCRFD can't run at all.
        """
        det = self._get_detector()
        if det is not None:
            try:
                from faceflash.align import align_face
                raw_boxes, kps = det.detect(frame_rgb)
                boxes, crops, scores = [], {}, {}
                for i in range(len(raw_boxes)):
                    crop = align_face(frame_rgb, kps[i])
                    if crop is None:
                        continue
                    b = tuple(int(round(v)) for v in raw_boxes[i][:4])
                    boxes.append(b)
                    crops[b] = crop
                    scores[b] = float(raw_boxes[i][4])
                return boxes, crops, scores
            except Exception as e:
                logger.debug("SCRFD detect failed: %s", e)

        # Haar fallback with NMS (multi-face; detect_faces_boxes' fallback
        # only reports the largest)
        h, w = frame_rgb.shape[:2]
        min_face = max(40, min(h, w) // 10)
        max_face = min(h, w) * 2 // 3
        try:
            import cv2
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=6,
                minSize=(min_face, min_face),
                maxSize=(max_face, max_face),
            )
            if len(faces):
                boxes = [(int(x), int(y), int(x + fw), int(y + fh))
                         for (x, y, fw, fh) in faces]
                boxes = _nms(boxes, iou_threshold=0.4)
                return boxes, {}, {b: 1.0 for b in boxes}   # haar has no score
        except Exception:
            pass

        return [], {}, {}

    def _face_for_track(self, track: Track, frame_rgb: np.ndarray,
                        crop_by_box: dict):
        """Aligned crop for a track: free from this frame's detector pass,
        else crop the track region and re-detect (Haar path)."""
        aligned = crop_by_box.get(tuple(track.box))
        if aligned is not None:
            return aligned
        from faceflash.detect import detect_and_align
        x1, y1, x2, y2 = track.box
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return detect_and_align(crop)

    def _identify_tracks(self, due: List[Track], frame_rgb: np.ndarray,
                         crop_by_box: dict):
        """Identify a batch of tracks against the gallery (one embed call)."""
        try:
            pairs = []
            for track in due:
                face = self._face_for_track(track, frame_rgb, crop_by_box)
                if face is None:
                    continue
                q = _sharpness(face)
                if q < self.min_sharpness:
                    continue          # blurry crop — retry on a sharper frame
                pairs.append((track, face, q))
            if not pairs:
                return
            with self._engine_lock:
                embeddings = self.engine.embedder.embed_batch(
                    [f for _, f, _ in pairs])
        except Exception as e:
            logger.debug("Batch identify failed: %s", e)
            return

        threshold = self.threshold
        if self.max_far is not None:
            threshold = self.engine._threshold_for_far(self.max_far)

        for (track, _, q), embedding in zip(pairs, embeddings):
            # Swap guard: the tracker matches by position, so when two people
            # cross paths a track can end up on the WRONG face. If the new
            # embedding clearly isn't the same person as the stored one,
            # reset identity and let it re-resolve below.
            if track.embedding is not None and                     float(np.dot(track.embedding, embedding)) < self.swap_threshold:
                track.identity = None
                track.visitor_id = None
                track.confidence = 0.0
                track.quality = 0.0
            # keep the embedding from the sharpest crop seen (re-ID quality)
            if q >= track.quality:
                track.embedding = embedding
                track.quality = q

            # Appearance re-ID: did this face belong to a recently-lost track?
            # (Occlusions / brief exits otherwise reset identity, dwell
            # continuity and the unknown-visitor counter.)
            if track.identity is None:
                old_track = self.tracker.reid_match(embedding, self.reid_threshold)
                if old_track is not None:
                    track.identity = old_track.identity
                    track.visitor_id = old_track.visitor_id
                    track.confidence = old_track.confidence
                    track.first_seen = old_track.first_seen
                    # carry the anomaly bookkeeping across the track change
                    prev = self.analytics.unknown_sightings.pop(old_track.track_id, 0)
                    if prev:
                        self.analytics.unknown_sightings[track.track_id] += prev
                    if old_track.track_id in self._alerted_tracks:
                        self._alerted_tracks.add(track.track_id)
                    if track.identity is not None:
                        continue        # identity restored, no gallery search needed

            with self._engine_lock:
                results = self.engine.index.search(
                    embedding, k=1,
                    n_candidates=self.engine.n_candidates)
            if results and results[0][1] >= threshold:
                track.identity = results[0][0]
                track.confidence = results[0][1]
            elif track.visitor_id is None:
                # Unregistered: count them as a unique anonymous visitor
                vid, is_new = self.visitors.assign(embedding)
                track.visitor_id = vid
                if is_new:
                    self._emit(StreamEvent("visitor_new", time.time(),
                                           {"visitor_id": vid,
                                            "unique_visitors": self.visitors.count}))

    def _draw_frame(self, frame, tracks: List[Track]):
        """Annotate frame with track boxes and names (for show=True)."""
        import cv2
        for track in tracks:
            x1, y1, x2, y2 = track.box
            name = track.identity or track.visitor_id or f"Unknown #{track.track_id}"
            color = (0, 255, 0) if track.identity else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({track.confidence:.2f})" if track.identity else name
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-camera — N sources, one gallery, one process
# ─────────────────────────────────────────────────────────────────────────────

class MultiCameraMonitor:
    """Run several cameras against ONE gallery in one process.

    Each camera gets its own StreamMonitor (own tracker, own zones); they
    share the engine (serialized behind one lock — identification is the
    only contended resource) and one visitor registry, so an unregistered
    person seen at the door and later in the lobby counts as ONE visitor
    (cross-camera re-ID by embedding). Every event carries its camera name.

    This is deliberately NOT a distributed system: one process, one box.
    For camera fleets, failover, and hardware acceleration, run a real NVR
    (e.g. Frigate) and use FaceFlash as the recognition backend.

    Usage:
        mc = MultiCameraMonitor(ff, {"door": 0, "lobby": "rtsp://..."})
        mc.on_event(print)
        mc.monitors["door"].add_zone("entrance", [(0, 0), (400, 0), (400, 700), (0, 700)])
        mc.run()          # blocks; Ctrl-C or mc.stop() to end
    """

    def __init__(self, engine, cameras: Dict[str, Union[int, str]], **monitor_kwargs):
        self.engine = engine
        shared_lock = threading.Lock()
        shared_visitors = _VisitorRegistry()
        self.monitors: Dict[str, StreamMonitor] = {}
        for name, source in cameras.items():
            m = StreamMonitor(engine, camera_name=name, **monitor_kwargs)
            m._engine_lock = shared_lock       # one identification at a time
            m.visitors = shared_visitors       # cross-camera visitor identity
            self.monitors[name] = m
        self._sources = dict(cameras)
        self._threads: List[threading.Thread] = []

    def on_event(self, callback):
        """Register a callback (or webhook URL) on every camera."""
        for m in self.monitors.values():
            m.on_event(callback)

    def run(self, block: bool = True):
        """Start all cameras (one thread each)."""
        for name, m in self.monitors.items():
            t = threading.Thread(target=m.run, args=(self._sources[name],),
                                 daemon=True, name=f"faceflash-cam-{name}")
            t.start()
            self._threads.append(t)
        if block:
            try:
                while any(t.is_alive() for t in self._threads):
                    time.sleep(0.5)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        for m in self.monitors.values():
            m.stop()
        for t in self._threads:
            t.join(timeout=5.0)

    def report(self) -> Dict[str, Any]:
        """Combined snapshot: per-camera analytics + global visitor count."""
        per_cam = {name: m.report() for name, m in self.monitors.items()}
        visitors = next(iter(self.monitors.values())).visitors.count \
            if self.monitors else 0
        return {"cameras": per_cam, "unique_unregistered_visitors": visitors}
