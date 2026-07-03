"""
Smart Doorbell — complete end-to-end deployment with FaceFlash.

Setup:
    pip install faceflash[cpu] opencv-python
    faceflash register --name "Alice" --image alice.jpg --db doorbell.db
    python examples/smart_doorbell.py

With RTSP camera:
    python examples/smart_doorbell.py --source "rtsp://admin:pass@192.168.1.100:554/stream"

Set DOORBELL_WEBHOOK_URL env var for push notifications (Slack, Discord, ntfy).
"""
import os, sys, time, json, argparse, logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from faceflash import FaceFlash
from faceflash.stream import StreamMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("doorbell")

class DoorbellHandler:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url
        self._last_alert = {}

    def __call__(self, event):
        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        if event.event_type == "person_appeared":
            name = event.data["name"]
            if self._should_alert(name):
                self._log_and_send(f"[{ts}] Welcome home, {name}!", event.data)
        elif event.event_type == "visitor_new":
            self._log_and_send(f"[{ts}] Unknown visitor at the door ({event.data['visitor_id']})", event.data)
        elif event.event_type == "enrollment_suggested":
            logger.info(f"[{ts}] Repeat visitor {event.data['visitor_id']} seen {event.data['sightings']} times")
        elif event.event_type == "tailgating_risk":
            self._log_and_send(f"[{ts}] TAILGATING detected!", event.data)

    def _should_alert(self, name, cooldown=60):
        now = time.time()
        if now - self._last_alert.get(name, 0) < cooldown:
            return False
        self._last_alert[name] = now
        return True

    def _log_and_send(self, msg, data):
        logger.warning(msg)
        if not self.webhook_url:
            return
        import urllib.request
        payload = json.dumps({"text": msg, "content": msg, "data": data}).encode()
        try:
            req = urllib.request.Request(self.webhook_url, data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug(f"Webhook failed: {e}")

def main():
    p = argparse.ArgumentParser(description="Smart doorbell with FaceFlash")
    p.add_argument("--source", default=0, help="0 for webcam, or RTSP URL")
    p.add_argument("--db", default="doorbell.db")
    p.add_argument("--webhook", default=os.environ.get("DOORBELL_WEBHOOK_URL"))
    p.add_argument("--snapshots", default="doorbell_snapshots")
    p.add_argument("--profile", default="light", choices=["accurate", "light"])
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    ff = FaceFlash(args.db, profile=args.profile)
    logger.info(f"Gallery: {len(ff)} faces, {len(ff.names())} people")
    if len(ff) == 0:
        logger.warning(f"No faces registered! Run: faceflash register --name 'You' --image photo.jpg --db {args.db}")

    os.makedirs(args.snapshots, exist_ok=True)

    monitor = StreamMonitor(
        ff, sample_fps=2.0,
        det_model="500m" if args.profile == "light" else "10g",
        det_size=320 if args.profile == "light" else 480,
        snapshot_dir=args.snapshots,
        suggest_enroll_after=5,
        max_missed_frames=10,
        persist_events=True,
    )
    monitor.add_zone("door", [(0,0),(640,0),(640,480),(0,480)])
    monitor.set_secure_zone("door")

    handler = DoorbellHandler(webhook_url=args.webhook)
    monitor.on_event(handler)
    if args.webhook:
        monitor.on_event(args.webhook)
        logger.info(f"Webhooks -> {args.webhook}")

    logger.info(f"Doorbell starting: source={source}, profile={args.profile}")
    logger.info("Press Ctrl-C to stop")

    try:
        monitor.run(source, show=args.show)
    except KeyboardInterrupt:
        pass

    report = monitor.report()
    logger.info(f"\nSession: {report.get('unique_unregistered_visitors', 0)} unique visitors")
    for name, info in report.get("attendance", {}).items():
        logger.info(f"  {name}: {info.get('visits',0)} visit(s)")

if __name__ == "__main__":
    main()
