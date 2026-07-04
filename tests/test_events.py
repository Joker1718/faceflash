import sqlite3

from faceflash.events import EventStore


def test_event_store_filters_without_truncating_newer_rows():
    db = sqlite3.connect(":memory:")
    store = EventStore(db)

    # Newer noisy rows should not prevent an older filtered match from being
    # found. This is the bug we avoid by querying before applying the limit.
    for i in range(50):
        store.insert(2000 + i, "person_appeared", {"camera": "front", "zone": "lobby"})
    store.insert(1000, "tailgating_risk", {"camera": "front", "zone": "entry"})

    rows = store.query(event_types=["tailgating_risk"], zone="entry", limit=1)
    assert len(rows) == 1
    assert rows[0]["timestamp"] == 1000
    assert rows[0]["event_type"] == "tailgating_risk"


def test_event_store_retention_and_stats():
    db = sqlite3.connect(":memory:")
    store = EventStore(db)
    store.insert(10, "visitor_new", {"camera": "door"})
    store.insert(20, "visitor_new", {"camera": "door"})
    store.insert(30, "capacity_exceeded", {"camera": "door"})

    assert store.stats()["by_type"]["visitor_new"] == 2
    assert store.delete_before(25) == 2
    assert store.stats()["events"] == 1
