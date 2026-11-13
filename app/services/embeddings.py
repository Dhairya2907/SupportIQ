"""Text embeddings behind one small interface.

Default provider is Gemini's embedding API (`gemini-embedding-001`, multilingual, free tier),
which keeps the API process light enough for 512 MB hosts. The optional `local` provider uses
Sentence Transformers for offline development; it needs `pip install sentence-transformers`.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.config import settings

log = logging.getLogger(__name__)
BATCH = 50


def available() -> bool:
    if settings.embedding_provider == "local":
        return True
    return bool(settings.gemini_api_key)


def _gemini(texts: list[str], task_type: str) -> list[list[float]]:
    from google.genai import types

    from app.services.llm import _client

    client = _client()
    if client is None:
        raise RuntimeError("Gemini not configured")
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        res = client.models.embed_content(
            model=settings.embedding_model,
            contents=chunk,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=settings.embedding_dim),
        )
        out.extend([list(e.values) for e in res.embeddings])
    return out


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.local_embedding_model)


def _local(texts: list[str]) -> list[list[float]]:
    return _local_model().encode(texts, normalize_embeddings=True).tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeddings for things to be searched (policy chunks)."""
    if not texts:
        return []
    return _local(texts) if settings.embedding_provider == "local" else _gemini(texts, "RETRIEVAL_DOCUMENT")


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Embeddings for things doing the searching (tickets). One ticket vector serves both
    policy retrieval and ticket-to-ticket similarity, so each ticket costs one call."""
    if not texts:
        return []
    return _local(texts) if settings.embedding_provider == "local" else _gemini(texts, "RETRIEVAL_QUERY")


def embed_query(text: str) -> list[float]:
    return embed_queries([text])[0]
