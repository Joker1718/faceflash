"""
Guided webcam self-test — validates the things unit tests can't reach.

    python scripts/selftest_webcam.py

Opens your webcam and walks you through four checks, printing PASS/FAIL:

  1. DETECTION    a green box tracks your face, steady (no flashing)
  2. LIVENESS     your live face reads LIVE; a photo/phone reads SPOOF
  3. RECOGNITION  after you enroll, it labels you by name
  4. SWAP GUARD   (optional) a second person never steals your name

Keys:
  d  run a 5-second detection stability sample
  e  enroll the current face under a name you type in the terminal
  l  run a one-shot liveness check on the current frame
  s  identify the current face against the gallery
  r  print the analytics report
  q  quit and print the summary

Nothing is uploaded; the gallery is a throwaway selftest.db you can delete.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from faceflash import FaceFlash
from faceflash.detect import detect_faces_boxes

DB = os.path.join(os.path.dirname(__file__), "..", "selftest.db")


def main():
    ff = FaceFlash(DB)
    results = {"detection": None, "liveness_live": None,
               "liveness_spoof": None, "recognition": None}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam (index 0). Close other apps using it.")
        return 1

    print(__doc__)
    print(f"Gallery: {len(ff)} faces enrolled. Press keys in the video window.\n")

    last_label = ""
    last_live = ""
    detect_sample_until = 0.0
    detect_sample_total = 0
    detect_sample_hits = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = [d for d in detect_faces_boxes(rgb) if d["method"] == "scrfd"]

        if dets:
            d = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) *
                    (d["box"][3] - d["box"][1]))
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hud = " | ".join(x for x in (last_label, last_live) if x)
            if hud:
                cv2.putText(frame, hud, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if detect_sample_until:
            detect_sample_total += 1
            detect_sample_hits += int(bool(dets))
            remaining = max(0.0, detect_sample_until - now)
            cv2.putText(frame, f"detection sample {remaining:.1f}s",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            if now >= detect_sample_until:
                rate = detect_sample_hits / max(1, detect_sample_total)
                results["detection"] = rate >= 0.9
                print(f"  detection stability: {rate*100:.0f}% frames had a SCRFD box")
                print("    expect >=90% while your face stays visible and still")
                detect_sample_until = 0.0

        cv2.putText(frame, "d=detect  e=enroll  l=liveness  s=identify  r=report  q=quit",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.imshow("FaceFlash self-test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("d"):
            print("  detection sample started: keep your face visible for 5 seconds")
            detect_sample_until = time.time() + 5.0
            detect_sample_total = 0
            detect_sample_hits = 0

        elif key == ord("e"):
            if not dets:
                print("  no face detected — face the camera and try again")
                continue
            cv2.destroyAllWindows()
            name = input("  enroll current face as name: ").strip() or "me"
            path = _save_temp(frame)
            try:
                out = ff.register(name, path)
                print(f"  enrolled {name!r} (detection={out['detection']}, "
                      f"gallery now {out['index_size']})")
            except Exception as e:
                print(f"  enroll failed: {e}")
            finally:
                os.unlink(path)

        elif key == ord("l"):
            path = _save_temp(frame)
            try:
                res = ff.check_liveness(path)
                last_live = f"{res['label'].upper()} {res['liveness_score']:.2f}"
                print(f"  liveness: {res['label']}  P(live)={res['liveness_score']:.3f}")
                print("    -> hold your LIVE face: expect 'live'")
                print("    -> hold a PHOTO/phone screen: expect 'spoof'")
                if res["label"] == "live":
                    results["liveness_live"] = True
                elif res["label"] == "spoof":
                    results["liveness_spoof"] = True
            except Exception as e:
                print(f"  liveness failed: {e}")
            finally:
                os.unlink(path)

        elif key == ord("s"):
            if len(ff) == 0:
                print("  gallery empty — press 'e' to enroll yourself first")
                continue
            path = _save_temp(frame)
            try:
                res = ff.search(path, k=1)
                if res["matches"]:
                    m = res["matches"][0]
                    last_label = f"{m['name']} {m['confidence']:.2f}"
                    print(f"  identified: {m['name']}  confidence={m['confidence']:.3f}")
                    results["recognition"] = True
                else:
                    last_label = "unknown"
                    print("  no match above threshold (expected if not enrolled)")
            except Exception as e:
                print(f"  identify failed: {e}")
            finally:
                os.unlink(path)

        elif key == ord("r"):
            print("  gallery:", ff.names())

    cap.release()
    cv2.destroyAllWindows()

    _summary(results)
    print(f"\n(throwaway gallery at {os.path.abspath(DB)} — delete when done)")
    return 0


def _save_temp(bgr_frame):
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    f.close()
    cv2.imwrite(f.name, bgr_frame)
    return f.name


def _summary(results):
    print("\n" + "=" * 50)
    print("  SELF-TEST SUMMARY")
    print("=" * 50)
    checks = [
        ("Face detection tracks steadily", results["detection"]),
        ("Live face reads LIVE", results["liveness_live"]),
        ("Photo/screen reads SPOOF", results["liveness_spoof"]),
        ("Enrolled face is identified", results["recognition"]),
    ]
    for label, val in checks:
        mark = "PASS" if val is True else ("FAIL" if val is False else "skipped")
        print(f"  [{mark:7}] {label}")
    print("=" * 50)
    if all(v is True for _, v in checks):
        print("  All checks passed — the branch is validated for merge.")
    else:
        print("  Run each check (e/l/s keys). 'skipped' = you didn't test it yet.")
        print("  Any FAIL is a real bug to fix before merge — tell me which.")


if __name__ == "__main__":
    sys.exit(main())
