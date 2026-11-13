"""Vector search with Chroma + Sentence Transformers. Two collections, same embedder:

  kb__<model>       policy documents from data/knowledge_base/*.md, split on H2 headings
  tickets__<model>  one entry per ticket, for similar-ticket detection

Collection names carry the embedding model, so switching models (e.g. to the multilingual
one for Hindi) never mixes vector spaces. Rebuild the KB with `python -m app.services.rag`.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)
_MODEL_SLUG = re.sub(r"[^a-z0-9]+", "-", settings.embedding_model.lower()).strip("-")
KB_COLLECTION = f"kb__{_MODEL_SLUG}"
TICKET_COLLECTION = f"tickets__{_MODEL_SLUG}"


# ------------------------------------------------------------------ infrastructure
@lru_cache(maxsize=1)
def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(settings.chroma_dir))


@lru_cache(maxsize=1)
def _embedder():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)


@lru_cache(maxsize=4)
def _collection(name: str):
    return _client().get_or_create_collection(name, embedding_function=_embedder(), metadata={"hnsw:space": "cosine"})


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
        chunks.append({"id": f"{source}-{i}", "source": source, "title": title, "text": part})
    return chunks


def build_index() -> int:
    col = _collection(KB_COLLECTION)
    chunks = [c for p in sorted(settings.knowledge_base_dir.glob("*.md")) for c in _chunk_markdown(p)]
    if not chunks:
        log.warning("No knowledge base documents in %s", settings.knowledge_base_dir)
        return 0
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
    )
    log.info("Indexed %d KB chunks from %d documents into %s", len(chunks), len({c["source"] for c in chunks}), KB_COLLECTION)
    return len(chunks)


PRIMARY_DOC_BY_CATEGORY = {
    "billing": "billing_payments",
    "delivery": "shipping_policy",
    "technical": "technical_support",
    "returns": "returns_refunds",
    "account": "account_access",
}


def _hits(res) -> list[dict]:
    return [
        {"text": doc, "source": meta["source"], "title": meta["title"], "score": round(1 - dist, 3)}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def retrieve(query: str, k: int = 3, category: str | None = None) -> list[dict]:
    """Top-k policy chunks. If a category is known, at least two chunks come from that
    category's primary policy document; this keeps retrieval on topic for short or
    non-English tickets where pure similarity drifts."""
    if not settings.enable_rag:
        return []
    col = _collection(KB_COLLECTION)
    if col.count() == 0:
        build_index()
    if col.count() == 0:
        return []
    hits: list[dict] = []
    primary = PRIMARY_DOC_BY_CATEGORY.get(category or "")
    if primary:
        hits += _hits(col.query(query_texts=[query], n_results=min(2, col.count()), where={"source": primary}))
    general = _hits(col.query(query_texts=[query], n_results=min(k + 2, col.count())))
    seen = {(h["source"], h["title"]) for h in hits}
    for h in general:
        if (h["source"], h["title"]) not in seen:
            hits.append(h)
            seen.add((h["source"], h["title"]))
        if len(hits) >= k:
            break
    return hits[:k]


# ------------------------------------------------------------------ similar tickets
def index_tickets(items: list[tuple[int, str]]) -> None:
    """items = [(ticket_id, text), ...]. Safe to call repeatedly (upsert)."""
    if not settings.enable_rag or not items:
        return
    col = _collection(TICKET_COLLECTION)
    col.upsert(
        ids=[str(tid) for tid, _ in items],
        documents=[text[:2000] for _, text in items],
        metadatas=[{"ticket_id": tid} for tid, _ in items],
    )


def index_ticket(ticket_id: int, text: str) -> None:
    index_tickets([(ticket_id, text)])


def indexed_ticket_ids() -> set[int]:
    if not settings.enable_rag:
        return set()
    return {int(i) for i in _collection(TICKET_COLLECTION).get(include=[])["ids"]}


def similar_tickets(text: str, k: int = 5, exclude_id: int | None = None) -> list[dict]:
    """[{"ticket_id": int, "score": float}] most similar first; never includes exclude_id."""
    if not settings.enable_rag:
        return []
    col = _collection(TICKET_COLLECTION)
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_texts=[text], n_results=min(k + 1, n))
    hits = [
        {"ticket_id": int(meta["ticket_id"]), "score": round(1 - dist, 3)}
        for meta, dist in zip(res["metadatas"][0], res["distances"][0])
        if exclude_id is None or int(meta["ticket_id"]) != exclude_id
    ]
    return hits[:k]


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
    n = build_index()
    print(f"Indexed {n} chunks into {KB_COLLECTION} using {settings.embedding_model}.")
    for q in ["my order is late and I have no tracking update", "मेरा ऑर्डर अभी तक नहीं आया, ट्रैकिंग अपडेट नहीं हो रही"]:
        print(f"\nQuery: {q}")
        for hit in retrieve(q):
            print(f"- {hit['source']} / {hit['title']} (score {hit['score']})")
