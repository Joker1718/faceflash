# FaceFlash Usage Guide

Everything beyond `register` / `search` / `verify`. Each section is self-contained — jump to what you need.

---

## Table of Contents

- [Streaming (webcam / RTSP)](#streaming)
- [Zones](#zones)
- [Privacy modes (anonymous, watchlist)](#privacy-modes)
- [Consent & scheduling](#consent--scheduling)
- [Multi-camera](#multi-camera)
- [Light profile (7x faster)](#light-profile)
- [Liveness detection](#liveness-detection)
- [Progressive enrollment](#progressive-enrollment)
- [Webhooks](#webhooks)
- [Deploying on Raspberry Pi](#raspberry-pi)
- [CLI reference](#cli)
- [REST API](#rest-api)

---

## Streaming

Connect to a webcam or RTSP camera. Faces are detected, tracked across frames, and identified against your gallery continuously.

```python
from faceflash import FaceFlash
from faceflash.stream import StreamMonitor

ff = FaceFlash("office.db")
monitor = StreamMonitor(ff, sample_fps=2.0)

# Print events as they happen
monitor.on_event(lambda e: print(f"{e.event_type}: {e.data}"))

# Webcam
monitor.run(0, show=True)

# Or RTSP camera
monitor.run("rtsp://admin:pass@192.168.1.100:554/stream1")
```

Press `q` to quit when `show=True`.

**Key parameters:**

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `sample_fps` | 2.0 | Frames processed per second. Lower = less CPU. |
| `det_size` | 480 | Detection input resolution. 320 = faster, 640 = better for small faces. |
| `min_sharpness` | 20 | Skip blurry faces (Laplacian variance). |
| `max_missed_frames` | 15 | Drop a track after this many frames without a detection. |
| `identify_interval` | 5 | Re-identify confirmed tracks every N frames. |

---

## Zones

Define polygonal regions in the camera view. Get events when people enter/exit zones, plus live per-zone counts.

```python
monitor = StreamMonitor(ff)

# Define zones as pixel coordinates (at least 3 points)
monitor.add_zone("entrance", [(0, 0), (300, 0), (300, 500), (0, 500)])
monitor.add_zone("checkout", [(400, 200), (640, 200), (640, 480), (400, 480)])

# Events fire automatically:
# StreamEvent("zone_entered", data={"name": "Alice", "zone": "entrance"})
# StreamEvent("zone_exited",  data={"name": "Alice", "zone": "entrance"})
```

**Capacity enforcement:**

```python
monitor.set_capacity(max_people=20)          # whole frame
monitor.set_capacity(max_people=5, zone="checkout")  # per zone
# Fires "capacity_exceeded" event when breached, re-arms when count drops.
```

**Anti-tailgating:**

```python
monitor.set_secure_zone("entrance")
# Fires "tailgating_risk" when an identified + unidentified face
# appear together in the secure zone.
```

---

## Privacy Modes

Three modes control what the stream does with unregistered faces:

```python
# Default: identify everyone, track unknown visitors
monitor = StreamMonitor(ff, mode="identify")

# Anonymous: count people, dwell time, zones — but NEVER identify anyone
monitor = StreamMonitor(ff, mode="anonymous")
# No gallery queries, no identity stored. Legally different from face recognition.

# Watchlist: only alert on specific people, ignore everyone else
monitor = StreamMonitor(ff, mode="watchlist", watchlist=["Banned_Person"])
# Non-watchlist faces leave no trace whatsoever.
```

---

## Consent & Scheduling

**Consent lifecycle:**

```python
ff = FaceFlash("company.db")

# Register with consent + expiry
ff.register("Alice", "alice.jpg", consent="employee_access", consent_expires="2026-12-31")

# Update consent later
ff.set_consent("Alice", consent="renewed", expires="2027-12-31")

# After expiry: Alice silently disappears from search results
# (still stored — re-consent to restore)
```

**Access scheduling:**

```python
# Bob can only be recognized Mon-Fri 9am-6pm
ff.set_schedule("Bob", "Mon-Fri 9:00-18:00")

# Search with access enforcement
result = ff.search("query.jpg", check_access=True)
# If Bob matches outside his hours:
# {"name": "Bob", "access": "denied", "reason": "outside scheduled hours (Mon-Fri 9:00-18:00)"}
```

**Scoped search:**

```python
# Only match within a subset (family, department, VIP tier)
result = ff.search("query.jpg", within=["Alice", "Bob", "Charlie"])
```

**Audit (GDPR data subject access):**

```python
report = ff.audit("Alice")
# Returns: enrollment info, consent state, schedule, every event mentioning Alice
```

---

## Multi-Camera

One process, multiple cameras, shared gallery:

```python
from faceflash.stream import MultiCameraMonitor

ff = FaceFlash("building.db")
mc = MultiCameraMonitor(ff, {
    "front_door": 0,                          # webcam
    "lobby": "rtsp://192.168.1.101:554/stream",
    "parking": "rtsp://192.168.1.102:554/stream",
})

# Events from all cameras, tagged with camera name
mc.on_event(lambda e: print(f"[{e.data.get('camera')}] {e.event_type}: {e.data}"))

# Per-camera zones
mc.monitors["front_door"].add_zone("entrance", [(0,0), (400,0), (400,700), (0,700)])
mc.monitors["front_door"].set_secure_zone("entrance")

# Start all cameras (blocks, Ctrl-C to stop)
mc.run()

# Combined report
print(mc.report())
```

**How it works:**
- Each camera has its own tracker and zones
- They share ONE gallery (thread-safe, serialized search)
- They share ONE visitor registry (same stranger at door + lobby = one visitor)
- Embedding runs concurrently across cameras (no lock needed)
- Every event carries its camera name

---

## Light Profile

7x faster on CPU, slightly lower accuracy. Ideal for streaming and edge devices.

```python
# Default: ArcFace r50 (accurate, ~50ms/face on CPU)
ff = FaceFlash("gallery.db")

# Light: MobileFaceNet (~7ms/face on CPU)
ff = FaceFlash("gallery_light.db", profile="light")
```

**Tradeoffs:**

| | Accurate (default) | Light |
|--|-------------------|-------|
| Embedding model | ArcFace w600k_r50 | MobileFaceNet |
| Inference time | ~50ms/face | ~7ms/face |
| Model size | 166 MB | 13 MB |
| Accuracy | Higher (validated on MS1MV2) | Lower on hard cases (pose, occlusion) |
| Best for | Enrollment, offline search, benchmarks | Streaming, real-time, Pi |

**Important:** The two profiles use different embedding spaces. A gallery enrolled with `profile="accurate"` cannot be searched with `profile="light"` (and vice versa). The system enforces this automatically.

**Stream with light profile:**

```python
ff = FaceFlash("fast.db", profile="light")
monitor = StreamMonitor(ff, det_model="500m", det_size=320)
# ~10ms/frame total on CPU → real-time on a Pi
monitor.run(0, show=True)
```

---

## Liveness Detection

**Passive (single frame, fast):**

```python
ff = FaceFlash("gallery.db")
result = ff.check_liveness("photo.jpg")
# {"is_live": True, "score": 0.94, "label": "live"}
# or: {"is_live": False, "score": 0.12, "label": "print_attack"}

# Integrate with search — reject spoofs before identifying
result = ff.search("query.jpg", check_liveness=True)
# If spoofed: {"matches": [], "spoofed": True, "liveness": {...}}
```

**Active (Face Flashing — flash colors, analyze reflection):**

```python
from faceflash.face_flashing import FaceFlashingChallenge

challenge = FaceFlashingChallenge(n_flashes=8)
result = challenge.run(camera_index=0)
# Flashes 8 random colors, analyzes how your face reflects them
# {"is_live": True, "confidence": 0.87, "n_frames_captured": 8}
```

Active is harder to spoof (attacker can't predict the flash pattern) but requires a screen + camera setup (kiosk/terminal use).

---

## Progressive Enrollment

The system tracks unknown repeat visitors. After N sightings, it suggests enrollment:

```python
monitor = StreamMonitor(ff, suggest_enroll_after=5)

def on_event(e):
    if e.event_type == "enrollment_suggested":
        print(f"Seen {e.data['sightings']} times — who is visitor {e.data['visitor_id']}?")

monitor.on_event(on_event)

# Later, enroll them by their visitor ID:
monitor.enroll_visitor("visitor_3", "New Employee Bob")
```

---

## Webhooks

Send events to any URL as JSON POST:

```python
# Pass a URL string instead of a callback
monitor.on_event("https://hooks.slack.com/services/XXX/YYY/ZZZ")
monitor.on_event("http://localhost:5000/faceflash-events")

# Events POST as:
# {"event_type": "person_appeared", "timestamp": 1719950000.0, "data": {"name": "Alice", "camera": "door"}}
```

Delivery is non-blocking (background thread with bounded queue). A dead endpoint won't stall the video loop.

---

## Raspberry Pi

FaceFlash runs on a Pi 5 with the light profile. Here's how:

**Hardware:**
- Raspberry Pi 5 (4GB+ RAM)
- USB webcam or Pi Camera Module
- MicroSD card (32GB+)

**Setup:**

```bash
# On the Pi
sudo apt update && sudo apt install -y python3-venv python3-dev cmake
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

git clone https://github.com/raghavenderreddygrudhanti/faceflash
cd faceflash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu]"
maturin develop --release

# Register faces
faceflash register --name "You" --image your_photo.jpg --db home.db

# Run
faceflash stream --db home.db --source 0 --profile light --det-model 500m --det-size 320
```

**Expected performance on Pi 5:**
- Detection (SCRFD 500m @ 320): ~8-12ms
- Embedding (MobileFaceNet): ~6-8ms
- Search (Rust SIMD NEON): <1ms
- **Total: ~15-20ms/frame → 3-5 fps processing, smooth display**

Memory: ~80 MB total (model + 6 MB index for 100K faces + overhead).

---

## CLI

```bash
# Register
faceflash register --name Alice --image alice.jpg --db office.db

# Search
faceflash search --image query.jpg --db office.db

# Verify
faceflash verify --image1 a.jpg --image2 b.jpg

# Gallery management
faceflash names --db office.db
faceflash remove --name Bob --db office.db
faceflash stats --db office.db

# Calibrate FAR thresholds on your gallery
faceflash calibrate --db office.db

# REST API server
faceflash serve --db office.db --port 8080

# Stream (live identification)
faceflash stream --db office.db --source 0 --show
```

---

## REST API

Start with `faceflash serve --db gallery.db --port 8080`, then:

```bash
# Search (POST image bytes)
curl -X POST http://localhost:8080/search \
  -H "Content-Type: image/jpeg" \
  --data-binary @query.jpg

# Register
curl -X POST "http://localhost:8080/register?name=Alice" \
  -H "Content-Type: image/jpeg" \
  --data-binary @alice.jpg

# List enrolled people
curl http://localhost:8080/names

# Remove
curl -X DELETE http://localhost:8080/person/Bob

# Stats
curl http://localhost:8080/stats
```

All responses are JSON.
