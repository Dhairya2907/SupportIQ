"""The triage pipeline: one ticket in, all AI fields out.

    classifier ──► category, confidence ──► routing ──► team
    rag.retrieve ──► policy chunks
    llm.triage (ONE Gemini call) ──► summary, sentiment, draft reply
    priority rules (text + category + sentiment) ──► priority, reasons

One LLM call per ticket keeps us inside the free-tier quota and keeps latency to one round trip.
If Gemini is unavailable or over quota the ticket is still created with classifier, rules,
routing and a truncated-body summary; the draft reply is left empty for the agent.
"""
from __future__ import annotations

import logging
import time

from app.services import classifier, llm, priority, rag, routing

log = logging.getLogger(__name__)


def run(subject: str, body: str) -> dict:
    t0 = time.perf_counter()
    text = f"{subject}\n{body}"

    category, confidence = classifier.predict(text)
    team = routing.route(category)

    # One embedding per ticket: used for policy retrieval now and stored for similar-ticket search.
    vector: list[float] | None = None
    chunks: list[dict] = []
    try:
        vector = rag.embed_for_ticket(text)
        chunks = rag.retrieve(k=3, category=category, query_vector=vector) if vector is not None else []
    except Exception as exc:  # retrieval is optional; never block ticket creation
        log.warning("Embedding/retrieval failed: %s", exc)

    try:
        tri = llm.triage(subject, body, category, chunks)
    except Exception as exc:
        log.warning("LLM triage failed, falling back to rules only: %s", exc)
        tri = {
            "language": llm.guess_language(body),
            "summary": body[:160] + ("..." if len(body) > 160 else ""),
            "sentiment": "neutral",
            "response": None,
        }

    prio, reasons = priority.predict(text, category=category, sentiment=tri["sentiment"])
    suggested = tri["response"]

    return {
        "ai_category": category,
        "ai_category_confidence": confidence,
        "ai_priority": prio,
        "ai_priority_reasons": "; ".join(reasons),
        "ai_sentiment": tri["sentiment"],
        "ai_language": tri.get("language", "en"),
        "ai_team": team,
        "ai_summary": tri["summary"],
        "ai_suggested_response": suggested,
        # Lexical groundedness compares reply words with English policy text, so it is only
        # meaningful for English replies. Hindi/Hinglish get None until an LLM judge exists.
        "ai_groundedness": (
            rag.groundedness(suggested or "", chunks)
            if (chunks and suggested and tri.get("language", "en") == "en")
            else None
        ),
        "ai_sources": ",".join(sorted({c["source"] for c in chunks})) or None,
        "ai_latency_ms": int((time.perf_counter() - t0) * 1000),
        "ticket_vector": vector,  # not a column; the router stores it in the vector table
    }


def warm_up() -> None:
    """Load the classifier and make sure the knowledge base is indexed."""
    t0 = time.perf_counter()
    classifier.predict("warm up")
    rag.ensure_index()
    log.info("Pipeline warm-up done in %.1fs", time.perf_counter() - t0)
