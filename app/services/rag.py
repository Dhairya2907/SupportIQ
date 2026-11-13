"""Retrieval over the policy knowledge base and similar-ticket search.

Embeddings come from services/embeddings.py (Gemini by default) and are stored in the main
database by services/vectors.py. Two kinds of rows share the same embedding space:

  kb      policy chunks from data/knowledge_base/*.md, split on H2 headings
  ticket  one row per ticket, for similar-ticket detection

Rebuild the knowledge base with `python -m app.services.rag`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.services import embeddings, vectors

log = logging.getLogger(__name__)

PRIMARY_DOC_BY_CATEGORY = {
    "billing": "billing_payments",
    "delivery": "shipping_policy",
    "technical": "technical_support",
    "returns": "returns_refunds",
    "account": "account_access",
}


# ------------------------------------------------------------------ knowledge base
def _chunk_markdown(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    source = path.stem
    parts = re.split(r"(?m)^##\s+", text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i == 0:
            lines = [ln for ln in part.splitlines() if ln.strip() and not ln.startswith("#")]
            if not lines:
                continue
        title = part.splitlines()[0].lstrip("# ").strip()
        chunks.append({"ref": f"{source}-{i}", "source": source, "title": title, "text": part})
    return chunks


def build_index(db: Session | None = None) -> int:
    """(Re)embed every policy chunk. Returns the number of chunks indexed."""
    if not settings.enable_rag or not embeddings.available():
        log.warning("RAG disabled or embeddings unavailable; knowledge base not indexed")
        return 0
    chunks = [c for p in sorted(settings.knowledge_base_dir.glob("*.md")) for c in _chunk_markdown(p)]
    if not chunks:
        log.warning("No knowledge base documents in %s", settings.knowledge_base_dir)
        return 0
    vecs = embeddings.embed_documents([c["text"] for c in chunks])
    items = [{**c, "vector": v} for c, v in zip(chunks, vecs)]
    if db is not None:
        vectors.upsert(db, "kb", items)
    else:
        from app.db import SessionLocal

        with SessionLocal() as s:
            vectors.upsert(s, "kb", items)
            s.commit()
    log.info("Indexed %d KB chunks from %d documents with %s", len(chunks), len({c["source"] for c in chunks}), settings.embedding_model)
    return len(chunks)


def ensure_index() -> None:
    """Build the knowledge base index once if the table is empty (called at startup)."""
    if settings.enable_rag and vectors.count("kb") == 0:
        try:
            build_index()
        except Exception as exc:
            log.warning("Knowledge base indexing failed: %s", exc)


def embed_for_ticket(text: str) -> list[float] | None:
    """One embedding per ticket, reused for retrieval and similarity. None when unavailable."""
    if not settings.enable_rag or not embeddings.available():
        return None
    return embeddings.embed_query(text)


def retrieve(query: str | None = None, k: int = 3, category: str | None = None, query_vector: list[float] | None = None) -> list[dict]:
    """Top-k policy chunks. If a category is known, at least two chunks come from that
    category's primary policy document, which keeps retrieval on topic for short or
    non-English tickets where pure similarity drifts."""
    if not settings.enable_rag:
        return []
    vec = query_vector if query_vector is not None else (embed_for_ticket(query) if query else None)
    if vec is None or vectors.count("kb") == 0:
        return []
    hits: list[dict] = []
    primary = PRIMARY_DOC_BY_CATEGORY.get(category or "")
    if primary:
        hits += vectors.query("kb", vec, k=2, source=primary)
    seen = {h["ref"] for h in hits}
    for h in vectors.query("kb", vec, k=k + 2):
        if h["ref"] not in seen:
            hits.append(h)
            seen.add(h["ref"])
        if len(hits) >= k:
            break
    return hits[:k]


# ------------------------------------------------------------------ similar tickets
def index_tickets(items: list[tuple[int, str]], db: Session, vecs: list[list[float]] | None = None) -> None:
    """items = [(ticket_id, text), ...]. Embeds in batches unless vectors are supplied."""
    if not settings.enable_rag or not items or not embeddings.available():
        return
    if vecs is None:
        vecs = embeddings.embed_queries([t for _, t in items])
    vectors.upsert(db, "ticket", [{"ref": str(tid), "text": text, "vector": v} for (tid, text), v in zip(items, vecs)])


def index_ticket(ticket_id: int, text: str, db: Session, vector: list[float] | None = None) -> None:
    index_tickets([(ticket_id, text)], db, [vector] if vector is not None else None)


def indexed_ticket_ids(db: Session | None = None) -> set[int]:
    if not settings.enable_rag:
        return set()
    return {int(r) for r in vectors.refs("ticket", db)}


def remove_tickets(ticket_ids: list[int], db: Session) -> None:
    vectors.remove(db, "ticket", [str(i) for i in ticket_ids])


def similar_tickets(text: str | None = None, k: int = 5, exclude_id: int | None = None, vector: list[float] | None = None) -> list[dict]:
    """[{"ticket_id": int, "score": float}] most similar first; never includes exclude_id."""
    if not settings.enable_rag:
        return []
    vec = vector if vector is not None else (embed_for_ticket(text) if text else None)
    if vec is None:
        return []
    hits = vectors.query("ticket", vec, k=k, exclude_ref=str(exclude_id) if exclude_id is not None else None)
    return [{"ticket_id": int(h["ref"]), "score": h["score"]} for h in hits]


# ------------------------------------------------------------------ groundedness
_STOP = set(
    "the a an is are was were be been to of and or in on for with your you we our this that it its as at by from will can has have not if".split()
)


def groundedness(response: str, chunks: list[dict]) -> float:
    """Cheap lexical groundedness: share of response sentences whose content words mostly
    appear in the retrieved context. Good enough for a trend line; swap for an LLM judge later."""
    if not response or not chunks:
        return 0.0
    context_words = set(re.findall(r"\w+", " ".join(c["text"] for c in chunks).lower())) - _STOP
    sentences = [s for s in re.split(r"(?<=[.!?।])\s+", response) if len(s.split()) >= 4]
    if not sentences:
        return 0.0
    supported = 0
    for s in sentences:
        words = set(re.findall(r"\w+", s.lower())) - _STOP
        if words and len(words & context_words) / len(words) >= 0.5:
            supported += 1
    return round(supported / len(sentences), 3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from app.db import init_db

    init_db()
    n = build_index()
    print(f"Indexed {n} chunks with {settings.embedding_model} ({settings.embedding_provider}).")
    for q in ["my order is late and I have no tracking update", "मेरा ऑर्डर अभी तक नहीं आया, ट्रैकिंग अपडेट नहीं हो रही"]:
        print(f"\nQuery: {q}")
        for hit in retrieve(q):
            print(f"- {hit['source']} / {hit['title']} (score {hit['score']})")
