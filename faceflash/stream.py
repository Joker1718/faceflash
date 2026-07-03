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
    confidence: float = 0.0
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
                 reid_retention_seconds: float = 120.0):
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
            # Update track
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
                 persist_events: bool = True):
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

    def _emit(self, event: StreamEvent):
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
        min_gap = 1.0 / self.sample_fps
        last_proc = 0.0
        try:
            while self._running:
                alive, frame = reader.read()
                if not alive:
                    break
                if frame is None:
                    time.sleep(0.005)
                    continue

                now = time.time()
                if now - last_proc < min_gap:
                    # Not due for processing — keep the display smooth
                    if show:
                        self._draw_frame(frame, [t for t in self.tracker.tracks
                                                 if t.frames_missed == 0])
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
                    self._draw_frame(frame, [t for t in self.tracker.tracks
                                             if t.frames_missed == 0])
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
        boxes, crop_by_box = self._detect(frame_rgb)

        # Update tracker (fast — just IOU matching)
        tracks = self.tracker.update(boxes, timestamp)
        self.analytics.update(tracks, timestamp)

        # Only identify tracks that need it (expensive — do sparingly)
        active_tracks = [t for t in tracks if t.frames_missed == 0]
        for track in active_tracks:
            should_identify = (
                track.identity is None and track.frames_seen >= 3
            ) or (
                track.identity is not None and
                track.frames_seen % self.identify_interval == 0
            )
            if should_identify:
                self._identify_track(track, frame_rgb, timestamp,
                                     aligned=crop_by_box.get(tuple(track.box)))

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

        # Anomaly check
        for track in active_tracks:
            if track.identity is None and track.frames_seen > 10:
                self.analytics.unknown_sightings[track.track_id] += 1
        for a in self.analytics.get_anomalies():
            if a["track_id"] not in self._alerted_tracks:
                self._alerted_tracks.add(a["track_id"])
                self._emit(StreamEvent("anomaly", timestamp, a))

        self._prev_identities = current_identities
        return active_tracks

    def _detect(self, frame_rgb: np.ndarray):
        """One detector pass → (boxes, {box: aligned_crop}).

        SCRFD gives boxes AND landmarks in a single inference, so the aligned
        112x112 crops come for free — identification then never re-runs the
        detector on track crops. Haar fallback (no landmarks) returns boxes
        only; those tracks identify via the per-crop path.
        """
        try:
            from faceflash.detect import detect_faces_boxes
            dets = [d for d in detect_faces_boxes(frame_rgb)
                    if d["method"] == "scrfd"]
            if dets:
                boxes = [tuple(d["box"]) for d in dets]
                return boxes, {tuple(d["box"]): d["crop"] for d in dets}
        except Exception:
            pass

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
                return _nms(boxes, iou_threshold=0.4), {}
        except Exception:
            pass

        return [], {}

    def _identify_track(self, track: Track, frame_rgb: np.ndarray,
                        timestamp: float, aligned: Optional[np.ndarray] = None):
        """Try to identify a tracked face against the gallery.

        `aligned` is the pre-aligned crop from this frame's detector pass
        (SCRFD path) — used directly. Without it (Haar path), the track
        region is cropped and re-detected.
        """
        try:
            if aligned is not None:
                face = aligned
            else:
                from faceflash.detect import detect_and_align

                x1, y1, x2, y2 = track.box
                h, w = frame_rgb.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    return
                face = detect_and_align(crop)
            embedding = self.engine.embedder.embed(face)
            track.embedding = embedding

            # Appearance re-ID: did this face belong to a recently-lost track?
            # (Occlusions / brief exits otherwise reset identity, dwell
            # continuity and the unknown-visitor counter.)
            if track.identity is None:
                old_track = self.tracker.reid_match(embedding, self.reid_threshold)
                if old_track is not None:
                    track.identity = old_track.identity
                    track.confidence = old_track.confidence
                    track.first_seen = old_track.first_seen
                    # carry the anomaly bookkeeping across the track change
                    prev = self.analytics.unknown_sightings.pop(old_track.track_id, 0)
                    if prev:
                        self.analytics.unknown_sightings[track.track_id] += prev
                    if old_track.track_id in self._alerted_tracks:
                        self._alerted_tracks.add(track.track_id)
                    if track.identity is not None:
                        return          # identity restored, no gallery search needed

            # Search
            threshold = self.threshold
            if self.max_far is not None:
                threshold = self.engine._threshold_for_far(self.max_far)

            results = self.engine.index.search(
                embedding, k=1,
                n_candidates=self.engine.n_candidates)

            if results and results[0][1] >= threshold:
                track.identity = results[0][0]
                track.confidence = results[0][1]

        except Exception as e:
            logger.debug("Identify track %d failed: %s", track.track_id, e)

    def _draw_frame(self, frame, tracks: List[Track]):
        """Annotate frame with track boxes and names (for show=True)."""
        import cv2
        for track in tracks:
            x1, y1, x2, y2 = track.box
            name = track.identity or f"Unknown #{track.track_id}"
            color = (0, 255, 0) if track.identity else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({track.confidence:.2f})" if track.identity else name
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
