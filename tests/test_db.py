"""SQLite backend tests: write-through persistence, streaming insertion,
parity with the legacy in-memory index, removal semantics, and migration."""

import numpy as np
import pytest

from faceflash.index import FaceIndex


N_FIT = 1024  # PCA fit threshold


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(7)
    embs = rng.standard_normal((1500, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    labels = [f"person_{i % 300}" for i in range(len(embs))]
    return embs, labels


def test_db_create_add_search(tmp_path, embeddings):
    embs, labels = embeddings
    db = tmp_path / "faces.db"
    idx = FaceIndex(db_path=str(db))
    idx.add_batch(embs, labels)

    assert len(idx) == 1500
    assert idx._pca_fitted  # above threshold
    r = idx.search(embs[42], k=1)
    assert r[0][0] == "person_42"
    assert r[0][1] > 0.999
    idx.close()
    assert db.exists()


def test_db_persistence_across_reopen(tmp_path, embeddings):
    embs, labels = embeddings
    db = tmp_path / "faces.db"
    idx = FaceIndex(db_path=str(db))
    idx.add_batch(embs, labels)
    idx.close()

    idx2 = FaceIndex(db_path=str(db))
    assert len(idx2) == 1500
    assert idx2._pca_fitted  # quantizer restored from meta
    r = idx2.search(embs[7], k=1)
    assert r[0][0] == "person_7" and r[0][1] > 0.999
    idx2.close()


def test_db_streaming_insertion_after_reopen(tmp_path, embeddings):
    """The headline: add faces to an existing index with no refit/rewrite."""
    embs, labels = embeddings
    db = tmp_path / "faces.db"
    idx = FaceIndex(db_path=str(db))
    idx.add_batch(embs, labels)
    idx.close()

    idx2 = FaceIndex(db_path=str(db))
    rng = np.random.default_rng(9)
    new = rng.standard_normal(512).astype(np.float32)
    new /= np.linalg.norm(new)
    idx2.add(new, "brand_new_person")
    r = idx2.search(new, k=1)
    assert r[0][0] == "brand_new_person"
    idx2.close()

    # ...and it persisted without any explicit save()
    idx3 = FaceIndex(db_path=str(db))
    assert len(idx3) == 1501
    assert "brand_new_person" in idx3
    r = idx3.search(new, k=1)
    assert r[0][0] == "brand_new_person"
    idx3.close()


def test_db_matches_legacy_results(tmp_path, embeddings):
    """Same data, same queries -> same answers from both backends."""
    embs, labels = embeddings
    legacy = FaceIndex()
    legacy.add_batch(embs, labels)
    dbidx = FaceIndex(db_path=str(tmp_path / "faces.db"))
    dbidx.add_batch(embs, labels)

    queries = embs[::97]
    for q in queries:
        rl = legacy.search(q, k=3)
        rd = dbidx.search(q, k=3)
        assert [x[0] for x in rl] == [x[0] for x in rd]
        np.testing.assert_allclose([x[1] for x in rl], [x[1] for x in rd], atol=1e-5)

    bl = legacy.search_batch(queries, k=1)
    bd = dbidx.search_batch(queries, k=1)
    assert [r[0][0] for r in bl] == [r[0][0] for r in bd]
    dbidx.close()


def test_db_remove_deletes_from_disk(tmp_path, embeddings):
    embs, labels = embeddings
    db = tmp_path / "faces.db"
    idx = FaceIndex(db_path=str(db))
    idx.add_batch(embs, labels)

    n = idx.remove("person_5")
    assert n == 5
    assert len(idx) == 1495
    assert "person_5" not in idx
    r = idx.search(embs[5], k=1)     # embs[5] was person_5
    assert r[0][0] != "person_5"
    idx.close()

    # gone from the file itself, not just from RAM
    import sqlite3
    con = sqlite3.connect(str(db))
    assert con.execute(
        "SELECT COUNT(*) FROM faces WHERE name='person_5'").fetchone()[0] == 0
    con.close()

    idx2 = FaceIndex(db_path=str(db))
    assert len(idx2) == 1495
    assert idx2.search(embs[6], k=1)[0][0] == "person_6"
    idx2.close()


def test_db_floats_stay_on_disk(tmp_path, embeddings):
    embs, labels = embeddings
    idx = FaceIndex(db_path=str(tmp_path / "faces.db"))
    idx.add_batch(embs, labels)
    s = idx.stats()
    assert s["backend"] == "sqlite"
    assert "sqlite" in s["float_mode"]
    # resident = binary codes only (64 B/face), floats are NOT counted
    assert s["resident_memory_mb"] == s["binary_memory_mb"]
    assert s["binary_memory_mb"] < 0.2   # 1500 * 64B ≈ 0.09 MB
    idx.close()


def test_migration_npy_to_db_and_back(tmp_path, embeddings):
    embs, labels = embeddings
    legacy = FaceIndex()
    legacy.add_batch(embs, labels)
    legacy.save(str(tmp_path / "legacy_dir"))

    # npy dir -> .db
    loaded = FaceIndex()
    loaded.load(str(tmp_path / "legacy_dir"))
    loaded.save(str(tmp_path / "migrated.db"))

    dbidx = FaceIndex(db_path=str(tmp_path / "migrated.db"))
    assert len(dbidx) == 1500
    assert dbidx.search(embs[3], k=1)[0][0] == "person_3"

    # .db -> npy dir (round trip)
    dbidx.save(str(tmp_path / "roundtrip_dir"))
    back = FaceIndex()
    back.load(str(tmp_path / "roundtrip_dir"))
    assert len(back) == 1500
    assert back.search(embs[3], k=1)[0][0] == "person_3"
    dbidx.close()


def test_db_below_pca_threshold(tmp_path):
    """Small galleries (random-projection codes) survive a reopen."""
    rng = np.random.default_rng(11)
    embs = rng.standard_normal((50, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    db = tmp_path / "small.db"

    idx = FaceIndex(db_path=str(db))
    idx.add_batch(embs, [f"p{i}" for i in range(50)])
    assert not idx._pca_fitted
    assert idx.search(embs[10], k=1)[0][0] == "p10"
    idx.close()

    idx2 = FaceIndex(db_path=str(db))
    assert len(idx2) == 50
    assert not idx2._pca_fitted
    # seeded random projection is identical across processes, so the codes
    # stored pre-fit still match freshly encoded queries
    assert idx2.search(embs[10], k=1)[0][0] == "p10"
    idx2.close()


def test_db_pca_fit_crosses_threshold_incrementally(tmp_path):
    """Adding one-by-one across the PCA threshold re-encodes the DB rows."""
    rng = np.random.default_rng(13)
    embs = rng.standard_normal((N_FIT + 50, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    db = tmp_path / "grow.db"

    idx = FaceIndex(db_path=str(db))
    for i in range(len(embs)):
        idx.add(embs[i], f"p{i}")
    assert idx._pca_fitted
    assert idx.search(embs[0], k=1)[0][0] == "p0"
    idx.close()

    idx2 = FaceIndex(db_path=str(db))
    assert idx2._pca_fitted
    assert idx2.search(embs[0], k=1)[0][0] == "p0"
    assert idx2.search(embs[-1], k=1)[0][0] == f"p{len(embs)-1}"
    idx2.close()


def test_legacy_npy_backend_unchanged(tmp_path, embeddings):
    """The old format still round-trips (mmap'd floats, remove keeps mmap)."""
    embs, labels = embeddings
    idx = FaceIndex()
    idx.add_batch(embs, labels)
    idx.save(str(tmp_path / "idx_dir"))

    idx2 = FaceIndex()
    idx2.load(str(tmp_path / "idx_dir"))
    assert idx2.stats()["backend"] == "npy"
    assert "mmap" in idx2.stats()["float_mode"]
    idx2.remove("person_9")
    assert "mmap" in idx2.stats()["float_mode"]   # still mmap after remove
    assert idx2.search(embs[10], k=1)[0][0] == "person_10"


def test_operator_camera_zone_config_roundtrip(tmp_path):
    db = tmp_path / "ops.db"
    idx = FaceIndex(db_path=str(db))
    idx.save_camera_config(
        "front-door",
        source="0",
        mode="watchlist",
        threshold=0.45,
        watchlist=["Alice"],
        capacity={"limit": 2, "zone": "entry"},
    )
    idx.save_zone_config(
        "front-door",
        "entry",
        [(0.1, 0.1), (0.7, 0.1), (0.7, 0.9), (0.1, 0.9)],
        secure=True,
    )
    idx.close()

    reopened = FaceIndex(db_path=str(db))
    cam = reopened.get_camera_config("front-door")
    assert cam["source"] == "0"
    assert cam["mode"] == "watchlist"
    assert cam["threshold"] == 0.45
    assert cam["watchlist"] == ["Alice"]
    zones = reopened.list_zone_configs("front-door")
    assert len(zones) == 1
    assert zones[0]["name"] == "entry"
    assert zones[0]["secure"] is True
    assert zones[0]["polygon"][0] == (0.1, 0.1)
    reopened.close()


def test_delete_camera_config_removes_zones(tmp_path):
    idx = FaceIndex(db_path=str(tmp_path / "ops.db"))
    idx.save_camera_config("cam-a")
    idx.save_zone_config("cam-a", "zone-a", [(0, 0), (1, 0), (1, 1)])
    assert idx.delete_camera_config("cam-a")
    assert idx.get_camera_config("cam-a") is None
    assert idx.list_zone_configs("cam-a") == []
    idx.close()


def test_operator_config_export_import_replace(tmp_path):
    src = FaceIndex(db_path=str(tmp_path / "src.db"))
    src.save_camera_config("cam1", source="rtsp://example/stream", mode="anonymous")
    src.save_zone_config("cam1", "lobby", [(0, 0), (1, 0), (1, 1)], secure=False)
    data = src.export_operator_config()
    src.close()

    dst = FaceIndex(db_path=str(tmp_path / "dst.db"))
    result = dst.import_operator_config(data, replace=True)
    assert result == {"cameras": 1, "zones": 1}
    assert dst.get_camera_config("cam1")["source"] == "rtsp://example/stream"
    assert dst.list_zone_configs("cam1")[0]["name"] == "lobby"
    dst.close()
