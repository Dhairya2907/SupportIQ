"""A tiny vector store inside the main database (SQLite locally, PostgreSQL in production).

Rows live in the `embeddings` table; search is brute-force cosine with NumPy. At this scale
(dozens of policy chunks, hundreds to low thousands of tickets) that is faster than a vector
database round trip and needs no extra service or persistent disk.

Writes take the caller's session so they commit together with the ticket. Reads use a short
session of their own and an in-memory cache per kind, refreshed after any write.
"""
from __future__ import annotations

import json
import logging
import threading

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal

log = logging.getLogger(__name__)
_lock = threading.Lock()
_cache: dict[str, tuple[list[dict], np.ndarray]] = {}  # kind -> (rows, unit-normalised matrix)
_stale: set[str] = set()


def _model_cls():
    from app.models import Embedding

    return Embedding


def invalidate(kind: str) -> None:
    with _lock:
        _stale.add(kind)


def upsert(db: Session, kind: str, items: list[dict]) -> None:
    """items: [{"ref": str, "text": str, "vector": list[float], "source": str|None, "title": str|None}]"""
    if not items:
        return
    Embedding = _model_cls()
    refs = [it["ref"] for it in items]
    db.execute(delete(Embedding).where(Embedding.kind == kind, Embedding.model == settings.embedding_model, Embedding.ref.in_(refs)))
    db.add_all(
        Embedding(
            kind=kind,
            ref=it["ref"],
            source=it.get("source"),
            title=it.get("title"),
            text=it["text"][:4000],
            model=settings.embedding_model,
            vector=json.dumps([round(float(x), 6) for x in it["vector"]]),
        )
        for it in items
    )
    db.flush()
    invalidate(kind)


def remove(db: Session, kind: str, refs: list[str]) -> None:
    Embedding = _model_cls()
    db.execute(delete(Embedding).where(Embedding.kind == kind, Embedding.ref.in_(refs)))
    invalidate(kind)


def refs(kind: str, db: Session | None = None) -> set[str]:
    Embedding = _model_cls()
    stmt = select(Embedding.ref).where(Embedding.kind == kind, Embedding.model == settings.embedding_model)
    if db is not None:
        return set(db.scalars(stmt).all())
    with SessionLocal() as s:
        return set(s.scalars(stmt).all())


def count(kind: str) -> int:
    return len(_load(kind)[0])


def _load(kind: str) -> tuple[list[dict], np.ndarray]:
    with _lock:
        if kind in _cache and kind not in _stale:
            return _cache[kind]
    Embedding = _model_cls()
    with SessionLocal() as s:
        rows = s.scalars(select(Embedding).where(Embedding.kind == kind, Embedding.model == settings.embedding_model)).all()
        meta = [{"ref": r.ref, "source": r.source, "title": r.title, "text": r.text} for r in rows]
        vecs = [json.loads(r.vector) for r in rows]
    if vecs:
        M = np.asarray(vecs, dtype=np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    else:
        M = np.zeros((0, settings.embedding_dim), dtype=np.float32)
    with _lock:
        _cache[kind] = (meta, M)
        _stale.discard(kind)
    return meta, M


def query(kind: str, vector: list[float], k: int = 5, source: str | None = None, exclude_ref: str | None = None) -> list[dict]:
    """Top-k rows by cosine similarity: [{"ref", "source", "title", "text", "score"}]."""
    meta, M = _load(kind)
    if M.shape[0] == 0:
        return []
    q = np.asarray(vector, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    scores = M @ q
    order = np.argsort(-scores)
    out: list[dict] = []
    for i in order:
        m = meta[int(i)]
        if source is not None and m["source"] != source:
            continue
        if exclude_ref is not None and m["ref"] == exclude_ref:
            continue
        out.append({**m, "score": round(float(scores[i]), 3)})
        if len(out) >= k:
            break
    return out
