"""
Demo: Real-time face identification + analytics from webcam.

Run:
    python examples/demo_stream.py

What it does:
    1. Opens your webcam
    2. Detects and tracks faces in real-time
    3. Identifies anyone registered in the gallery
    4. Shows crowd count, dwell time, visit frequency
    5. Alerts on unknown frequent visitors (anomaly)

Press 'q' to quit. Press 'r' to print a report.

Prerequisites:
    pip install opencv-python
    # Register some faces first:
    # faceflash register --name "Your Name" --image your_photo.jpg --db demo.db
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceflash import FaceFlash
from faceflash.stream import StreamMonitor


def on_event(event):
    """Print events as they happen."""
    if event.event_type == "person_appeared":
        print(f"  >> {event.data['name']} appeared")
    elif event.event_type == "person_left":
        print(f"  << {event.data['name']} left (was here {event.data['dwell_seconds']}s)")
    elif event.event_type == "anomaly":
        print(f"  !! ANOMALY: unknown face (track #{event.data['track_id']}) "
              f"seen {event.data['sightings']} times")


def main():
    db_path = "demo.db"

    # Create engine
    ff = FaceFlash(db_path)
    print(f"Gallery: {len(ff)} faces, {len(ff.names())} people")

    if len(ff) == 0:
        print("\nNo faces registered. Register some first:")
        print("  faceflash register --name 'Your Name' --image photo.jpg --db demo.db")
        print("\nStarting anyway — will show 'Unknown' for all faces.\n")

    # Create stream monitor
    monitor = StreamMonitor(
        ff,
        sample_fps=2.0,           # process 2 frames/sec
        identify_interval=10,     # re-identify every 10 frames
        threshold=0.4,
        iou_threshold=0.3,
        max_missed_frames=30,     # keep track alive for 15 sec at 2fps
        visit_gap_seconds=60,     # 1 min gap = new visit (short for demo)
        anomaly_threshold=5,      # alert after 5 sightings
    )
    monitor.on_event(on_event)

    print("Starting webcam stream (press 'q' in the window to quit)...")
    print("Events will print here as they happen.\n")

    try:
        monitor.run(source=0, show=True)
    except KeyboardInterrupt:
        pass

    # Final report
    print("\n" + "=" * 50)
    print("  SESSION REPORT")
    print("=" * 50)
    report = monitor.analytics.report()
    print(f"  Peak crowd count: {report['crowd_count']}")
    print(f"  Unique identified people: {report['unique_identified']}")
    print(f"  Total visits: {report['total_visits_today']}")
    if report['top_visitors']:
        print(f"  Top visitors:")
        for name, count in report['top_visitors']:
            dwell = monitor.analytics.get_dwell_time(name)
            print(f"    {name}: {count} visit(s), last dwell {dwell:.1f}s")
    if report['anomalies']:
        print(f"  Anomalies: {len(report['anomalies'])} unknown frequent faces")
    print("=" * 50)


if __name__ == "__main__":
    main()
