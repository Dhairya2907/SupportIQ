"""The in-database vector store, exercised with hand-made vectors (no embedding API needed)."""
from app.db import SessionLocal
from app.services import vectors


def test_upsert_query_filter_exclude(client):  # client fixture creates the tables
    with SessionLocal() as db:
        vectors.upsert(
            db,
            "unit",
            [
                {"ref": "a", "text": "north", "vector": [1, 0, 0], "source": "s1", "title": "A"},
                {"ref": "b", "text": "north-east", "vector": [1, 1, 0], "source": "s2", "title": "B"},
                {"ref": "c", "text": "south", "vector": [-1, 0, 0], "source": "s1", "title": "C"},
            ],
        )
        db.commit()

    hits = vectors.query("unit", [1, 0, 0], k=3)
    assert [h["ref"] for h in hits] == ["a", "b", "c"]
    assert hits[0]["score"] == 1.0 and hits[2]["score"] == -1.0

    only_s1 = vectors.query("unit", [1, 0, 0], k=3, source="s1")
    assert [h["ref"] for h in only_s1] == ["a", "c"]

    without_a = vectors.query("unit", [1, 0, 0], k=1, exclude_ref="a")
    assert [h["ref"] for h in without_a] == ["b"]

    # upsert replaces by ref and the cache refreshes
    with SessionLocal() as db:
        vectors.upsert(db, "unit", [{"ref": "a", "text": "moved south", "vector": [-1, 0, 0]}])
        db.commit()
    assert vectors.query("unit", [1, 0, 0], k=1)[0]["ref"] == "b"
    assert vectors.refs("unit") == {"a", "b", "c"}

    with SessionLocal() as db:
        vectors.remove(db, "unit", ["b", "c"])
        db.commit()
    assert vectors.refs("unit") == {"a"}


def test_empty_kind_returns_nothing(client):
    assert vectors.query("nothing-here", [1, 2, 3], k=5) == []
    assert vectors.count("nothing-here") == 0
