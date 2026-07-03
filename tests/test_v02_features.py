"""v0.2 feature tests: enrollment guards, FAR calibration, CLI.
All model-free (no ONNX download) so they run anywhere CI does."""

import json

import numpy as np
import pytest

from faceflash.detect import detect_and_align_info
from faceflash.engine import (FaceFlash, NoFaceFoundError,
                              _threshold_from_curve, _LFW_FAR_CURVE)
from faceflash.index import FaceIndex


def _gallery(n=1500, n_people=300, seed=7):
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs, [f"person_{i % n_people}" for i in range(n)]


# ── enrollment guards ────────────────────────────────────────────────────────

def test_detect_info_reports_center_crop_on_blank_image():
    blank = np.zeros((200, 300, 3), dtype=np.uint8)
    face, method = detect_and_align_info(blank)
    assert method == "center_crop"
    assert face.shape[0] == face.shape[1] == 200  # center square


def test_register_refuses_center_crop(tmp_path, monkeypatch):
    """A photo with no face raises NoFaceFoundError unless allow_fallback."""
    from PIL import Image
    img_path = tmp_path / "noface.jpg"
    Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(img_path)

    ff = FaceFlash()
    with pytest.raises(NoFaceFoundError):
        ff.register("Alice", str(img_path))
    assert len(ff) == 0  # nothing got enrolled

    # allow_fallback enrolls it (embedder stubbed — no model download in CI)
    class FakeEmbedder:
        def embed(self, face):
            v = np.ones(512, dtype=np.float32)
            return v / np.linalg.norm(v)
    monkeypatch.setattr(ff, "_embedder", FakeEmbedder())
    out = ff.register("Alice", str(img_path), allow_fallback=True)
    assert out["registered"] and out["detection"] == "center_crop"
    assert len(ff) == 1


def test_register_flags_duplicates(tmp_path, monkeypatch):
    """Registering the same face under a new name reports duplicate_of."""
    from PIL import Image
    img_path = tmp_path / "x.jpg"
    Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(img_path)

    rng = np.random.default_rng(0)
    same_face = rng.standard_normal(512).astype(np.float32)
    same_face /= np.linalg.norm(same_face)

    ff = FaceFlash()

    class FakeEmbedder:
        def embed(self, face):
            return same_face
    monkeypatch.setattr(ff, "_embedder", FakeEmbedder())

    ff.register("Alice", str(img_path), allow_fallback=True)
    out = ff.register("Bob", str(img_path), allow_fallback=True)
    assert out["duplicate_of"]["name"] == "Alice"
    assert out["duplicate_of"]["confidence"] > 0.99


# ── FAR calibration ──────────────────────────────────────────────────────────

def test_threshold_from_curve_interpolates_and_clamps():
    t_1pct = _threshold_from_curve(_LFW_FAR_CURVE, 0.01)
    t_01pct = _threshold_from_curve(_LFW_FAR_CURVE, 0.001)
    t_tiny = _threshold_from_curve(_LFW_FAR_CURVE, 1e-9)
    assert t_1pct < t_01pct < t_tiny        # stricter FAR -> higher threshold
    assert t_tiny == 0.35                    # clamps to most conservative
    assert _threshold_from_curve(_LFW_FAR_CURVE, 0.5) == 0.169  # loose clamp
    mid = _threshold_from_curve(_LFW_FAR_CURVE, 0.002)
    assert 0.20 < mid < 0.25                 # interpolated between anchors


def test_calibrate_on_gallery_and_persist(tmp_path):
    embs, labels = _gallery()
    ff = FaceFlash(index_path=str(tmp_path / "faces.db"))
    ff.index.add_batch(embs, labels)

    cal = ff.calibrate(n_pairs=20_000)
    assert cal["n_pairs"] == 20_000
    fars = [f for f, _ in cal["curve"]]
    thrs = [t for _, t in cal["curve"]]
    assert sorted(fars, reverse=True) == fars     # descending FAR
    assert sorted(thrs) == thrs                   # ascending threshold

    # max_far -> threshold uses the gallery curve, monotonic
    t_loose = ff._threshold_for_far(0.03)
    t_strict = ff._threshold_for_far(0.001)
    assert t_loose < t_strict

    # persists with the .db across reopen
    ff.index.close()
    ff2 = FaceFlash(index_path=str(tmp_path / "faces.db"))
    assert ff2.index.get_calibration() is not None
    assert abs(ff2._threshold_for_far(0.001) - t_strict) < 1e-9
    ff2.index.close()


def test_calibrate_requires_two_people():
    ff = FaceFlash()
    embs, _ = _gallery(n=50)
    ff.index.add_batch(embs, ["only_one_person"] * 50)
    with pytest.raises(RuntimeError):
        ff.calibrate()


def test_uncalibrated_falls_back_to_lfw_curve():
    ff = FaceFlash()
    assert ff._threshold_for_far(0.001) == pytest.approx(
        _threshold_from_curve(_LFW_FAR_CURVE, 0.001))


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_names_stats_remove_calibrate(tmp_path, capsys):
    from faceflash.cli import main

    embs, labels = _gallery()
    idx = FaceIndex(db_path=str(tmp_path / "faces.db"))
    idx.add_batch(embs, labels)
    idx.close()

    db = ["--db", str(tmp_path / "faces.db")]

    assert main(["names"] + db) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["count"] == 1500 and "person_0" in out["names"]

    assert main(["stats"] + db) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["backend"] == "sqlite" and out["count"] == 1500

    assert main(["calibrate", "--pairs", "20000"] + db) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["calibrated"] and len(out["curve"]) >= 3

    assert main(["remove", "--name", "person_3"] + db) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["removed"] == 5 and out["index_size"] == 1495


# ── streaming across sessions + bounding boxes ──────────────────────────────

def test_streaming_pca_fit_across_sessions(tmp_path):
    """Enroll below the PCA threshold, close, reopen, cross the threshold:
    the fit must trigger and re-encode the session-1 rows on disk."""
    rng = np.random.default_rng(21)
    embs = rng.standard_normal((1300, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    db = str(tmp_path / "stream.db")

    s1 = FaceIndex(db_path=db)
    s1.add_batch(embs[:500], [f"p{i}" for i in range(500)])
    assert not s1._pca_fitted
    s1.close()

    s2 = FaceIndex(db_path=db)
    for i in range(500, 1300):
        s2.add(embs[i], f"p{i}")
    s2.search(embs[0], k=1)  # force flush
    assert s2._pca_fitted
    assert s2.search(embs[42], k=1)[0][0] == "p42"      # session-1 face
    assert s2.search(embs[1200], k=1)[0][0] == "p1200"  # session-2 face
    s2.close()

    s3 = FaceIndex(db_path=db)
    assert s3._pca_fitted and len(s3) == 1300
    assert s3.search(embs[42], k=1)[0][0] == "p42"
    s3.close()


def test_detect_faces_boxes_fallback_shape():
    from faceflash.detect import detect_faces_boxes
    blank = np.zeros((200, 300, 3), dtype=np.uint8)
    out = detect_faces_boxes(blank)
    assert len(out) == 1
    d = out[0]
    assert d["method"] == "center_crop"
    assert d["box"] == [50, 0, 250, 200]      # centered square in a 300x200 image
    assert d["crop"].shape == (200, 200, 3)
    assert d["det_score"] == 0.0


# ── model profiles (CPU-first) ───────────────────────────────────────────────

def test_profile_persists_and_guards_mismatch(tmp_path):
    """A gallery remembers which embedding model enrolled it; mixing raises."""
    embs, labels = _gallery()
    db = str(tmp_path / "g.db")

    ff = FaceFlash(index_path=db, profile="light")
    ff.index.add_batch(embs, labels)
    assert ff.profile == "light"
    ff.index.close()

    # reopening without a profile adopts the stored one
    ff2 = FaceFlash(index_path=db)
    assert ff2.profile == "light"
    ff2.index.close()

    # explicit mismatch is an error, not silent garbage results
    with pytest.raises(ValueError):
        FaceFlash(index_path=db, profile="accurate")


def test_default_profile_is_accurate(tmp_path):
    ff = FaceFlash(index_path=str(tmp_path / "d.db"))
    assert ff.profile == "accurate"
    assert ff.index.get_meta("embed_profile") == "accurate"
    ff.index.close()


def test_index_meta_roundtrip_legacy(tmp_path):
    embs, labels = _gallery(n=100)
    idx = FaceIndex()
    idx.add_batch(embs, labels)
    idx.set_meta("embed_profile", "light")
    idx.save(str(tmp_path / "dir"))

    idx2 = FaceIndex()
    idx2.load(str(tmp_path / "dir"))
    assert idx2.get_meta("embed_profile") == "light"


def test_unknown_profile_rejected():
    from faceflash.embed import ensure_model
    with pytest.raises(ValueError):
        ensure_model("turbo")


# ── access control, consent, scoping, audit, dedup ──────────────────────────

def _stub_engine_with(monkeypatch, ff, emb):
    class FakeEmbedder:
        def embed(self, face): return emb
        def embed_batch(self, faces): return __import__("numpy").stack([emb]*len(faces))
    monkeypatch.setattr(ff, "_embedder", FakeEmbedder())


def test_schedule_parsing():
    import datetime as dt
    from faceflash.engine import _schedule_allows
    wed_noon = dt.datetime(2026, 1, 7, 12, 0)   # Wednesday
    sun_noon = dt.datetime(2026, 1, 11, 12, 0)  # Sunday
    assert _schedule_allows("Mon-Fri 9:00-18:00", wed_noon)
    assert not _schedule_allows("Mon-Fri 9:00-18:00", dt.datetime(2026, 1, 7, 20, 0))
    assert not _schedule_allows("Mon-Fri 9:00-18:00", sun_noon)
    assert _schedule_allows("Sat,Sun 10:00-14:00", sun_noon)


def test_consent_expiry_excludes_from_search(tmp_path, monkeypatch):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    rng = np.random.default_rng(0)
    e = rng.standard_normal(512).astype(np.float32); e /= np.linalg.norm(e)

    ff = FaceFlash(index_path=str(tmp_path / "g.db"))
    _stub_engine_with(monkeypatch, ff, e)
    ff.register("Alice", str(img), allow_fallback=True, check_duplicate=False)

    # valid consent -> matches
    ff.set_consent("Alice", "granted", expires="2999-01-01")
    assert ff.search(str(img))["matches"][0]["name"] == "Alice"
    # expired -> silently gone, but still stored
    ff.set_consent("Alice", "granted", expires="2020-01-01")
    assert ff.search(str(img))["matches"] == []
    assert "Alice" in ff.index.names()
    ff.index.close()


def test_scoped_search_within(tmp_path, monkeypatch):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    rng = np.random.default_rng(1)
    shared = rng.standard_normal(512).astype(np.float32); shared /= np.linalg.norm(shared)

    ff = FaceFlash(index_path=str(tmp_path / "g.db"))
    _stub_engine_with(monkeypatch, ff, shared)
    for nm in ("Alice", "Bob", "Carol"):
        ff.register(nm, str(img), allow_fallback=True, check_duplicate=False)

    names = {m["name"] for m in ff.search(str(img), k=5, within=["Alice", "Bob"])["matches"]}
    assert names <= {"Alice", "Bob"} and "Carol" not in names
    ff.index.close()


def test_access_schedule_denies_outside_hours(tmp_path, monkeypatch):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    rng = np.random.default_rng(2)
    e = rng.standard_normal(512).astype(np.float32); e /= np.linalg.norm(e)

    ff = FaceFlash(index_path=str(tmp_path / "g.db"))
    _stub_engine_with(monkeypatch, ff, e)
    ff.register("Bob", str(img), allow_fallback=True, check_duplicate=False)
    ff.set_schedule("Bob", "Mon-Fri 9:00-9:01")   # essentially never now
    res = ff.search(str(img), check_access=True)
    assert res["matches"] == []                   # denied is not grantable
    assert res["access_denied"][0]["name"] == "Bob"
    ff.index.close()


def test_audit_report(tmp_path, monkeypatch):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    rng = np.random.default_rng(3)
    e = rng.standard_normal(512).astype(np.float32); e /= np.linalg.norm(e)

    ff = FaceFlash(index_path=str(tmp_path / "g.db"))
    _stub_engine_with(monkeypatch, ff, e)
    ff.register("Alice", str(img), allow_fallback=True, check_duplicate=False)
    ff.set_consent("Alice", "granted", expires="2999-01-01")
    ff.set_schedule("Alice", "Mon-Fri 9:00-18:00")

    a = ff.audit("Alice")
    assert a["enrolled_faces"] == 1
    assert a["consent"] == "granted" and not a["consent_expired"]
    assert a["schedule"] == "Mon-Fri 9:00-18:00"
    ff.index.close()


def test_gallery_dedup_finds_same_face_different_names(tmp_path, monkeypatch):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    rng = np.random.default_rng(4)
    same = rng.standard_normal(512).astype(np.float32); same /= np.linalg.norm(same)

    ff = FaceFlash(index_path=str(tmp_path / "g.db"))
    _stub_engine_with(monkeypatch, ff, same)
    ff.register("Alice", str(img), allow_fallback=True, check_duplicate=False)
    ff.register("Alice_dup", str(img), allow_fallback=True, check_duplicate=False)

    dups = ff.find_duplicates(threshold=0.85)
    assert dups and {dups[0]["a"], dups[0]["b"]} == {"Alice", "Alice_dup"}
    ff.index.close()


# ── review-fix regressions ──────────────────────────────────────────────────

def test_access_denied_not_in_matches(tmp_path, monkeypatch):
    """Denied faces must NOT be actionable — matches[0] can't open a door."""
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    rng = np.random.default_rng(5)
    e = rng.standard_normal(512).astype(np.float32); e /= np.linalg.norm(e)

    ff = FaceFlash(index_path=str(tmp_path / "g.db"))
    _stub_engine_with(monkeypatch, ff, e)
    ff.register("Bob", str(img), allow_fallback=True, check_duplicate=False)
    ff.set_schedule("Bob", "Mon-Fri 9:00-9:01")   # never, right now

    res = ff.search(str(img), check_access=True)
    assert res["matches"] == []                    # nothing grantable
    assert res["access_denied"][0]["name"] == "Bob"
    assert res["access_denied"][0]["access"] == "denied"
    ff.index.close()


def test_consent_datetime_precision():
    from faceflash.engine import _consent_expired
    import datetime as dt
    future = (dt.datetime.now() + dt.timedelta(hours=1)).isoformat(timespec="seconds")
    past = (dt.datetime.now() - dt.timedelta(hours=1)).isoformat(timespec="seconds")
    assert not _consent_expired(future)
    assert _consent_expired(past)
    assert not _consent_expired("2999-01-01")      # bare date still works


def test_legacy_remove_keeps_person_meta_and_calibration(tmp_path):
    """#7: remove() in legacy mode must not drop calibration / person meta."""
    from faceflash.index import FaceIndex
    embs, labels = _gallery(n=200)
    idx = FaceIndex()
    idx.add_batch(embs, labels)
    idx.set_person_meta("person_0", "consent", "granted")
    idx.set_calibration({"curve": [[0.001, 0.3]], "n_pairs": 100})
    idx.save(str(tmp_path / "dir"))

    idx2 = FaceIndex(); idx2.load(str(tmp_path / "dir"))
    # force the mmap-compaction path
    idx2.float_vectors = np.load(str(tmp_path / "dir" / "float_vectors.npy"), mmap_mode="r")
    idx2._float_is_mmap = True
    idx2.remove("person_5")

    idx3 = FaceIndex(); idx3.load(str(tmp_path / "dir"))
    assert idx3.get_person_meta("person_0", "consent") == "granted"
    assert idx3.get_calibration() is not None


# ── execution provider selection ─────────────────────────────────────────────

def test_provider_selection_excludes_static_only_for_detector(monkeypatch):
    import faceflash.runtime as rt

    class FakeORT:
        @staticmethod
        def get_available_providers():
            return ["CoreMLExecutionProvider", "OpenVINOExecutionProvider",
                    "CPUExecutionProvider"]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeORT)
    monkeypatch.delenv("FACEFLASH_PROVIDERS", raising=False)

    # embedder (static shapes) may use accelerators (OpenVINO ranks first)
    assert rt.select_providers(dynamic_shapes=False)[0] == "OpenVINOExecutionProvider"
    # detector (dynamic shapes) must fall through to CPU only
    assert rt.select_providers(dynamic_shapes=True) == ["CPUExecutionProvider"]


def test_provider_env_override(monkeypatch):
    import faceflash.runtime as rt

    class FakeORT:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeORT)

    monkeypatch.setenv("FACEFLASH_PROVIDERS", "CPUExecutionProvider")
    assert rt.select_providers() == ["CPUExecutionProvider"]
    # unavailable forced provider is dropped, CPU still appended
    monkeypatch.setenv("FACEFLASH_PROVIDERS", "TensorRTExecutionProvider")
    assert rt.select_providers() == ["CPUExecutionProvider"]
