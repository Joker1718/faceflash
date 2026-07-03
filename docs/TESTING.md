# FaceFlash Testing Gate

Use this before merging a feature branch, tagging a release, or publishing a
benchmark post.

## 1. Fast Local Gate

```bash
python3 -m pytest tests -q
python3 -m compileall -q faceflash scripts tests
```

Expected result: all tests pass. A localhost webhook test may skip in locked
sandbox environments that disallow binding `127.0.0.1`; it should run normally
on GitHub Actions and a regular developer machine.

## 2. Webcam Acceptance Gate

Run the guided test:

```bash
python3 scripts/selftest_webcam.py
```

Pass criteria:

- Press `d`: SCRFD detection stays visible for at least 90% of the 5-second
  sample while your face is in frame.
- Press `l` on your real face: liveness prints `live`.
- Press `l` while holding a phone/photo: liveness prints `spoof`.
- Press `e` to enroll yourself, then `s`: the top match is your enrolled name.
- With two people in frame, labels should not swap while people cross paths.

If the Streamlit app flashes names or boxes but this script is stable, the bug
is in the app rendering/state layer, not the tracker/search core.

## 3. Stream Regression Gate

For stream features, run the unit tests that avoid camera/model downloads:

```bash
python3 -m pytest tests/test_stream.py tests/test_liveness.py -q
```

These cover tracker aging, display grace, appearance re-ID, anonymous/watchlist
modes, zone events, snapshots, event persistence, webhook delivery, and the
MiniFASNet liveness class-order/crop regressions.

## 4. Benchmark Sanity Gate

Before changing README numbers, run a small public-data sanity check:

```bash
bash scripts/reproduce_public.sh
```

This is not the 1M benchmark. It proves the public reproduction path still
installs, downloads data, and runs the same benchmark code.

## 5. Full Benchmark Gate

Only update headline numbers after a clean RunPod run:

```bash
bash scripts/runpod_ms1m.sh
```

Save the generated `results/runpod/*.json` files with the commit that updates
the README. Do not mix numbers from different machines or dates in one table.

## 6. Manual Release Checklist

- Fresh clone installs with `pip install -e ".[cpu,dev]"`.
- `FaceFlash("test.db")` can register, close, reopen, and search.
- `FaceFlash(profile="light")` refuses to open an accurate-profile gallery.
- Stream events persist to SQLite and survive restart.
- Webcam liveness does not label a real face as spoof in normal lighting.
- README benchmark tables point to matching result JSON files.
