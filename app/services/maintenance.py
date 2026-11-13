"""Startup housekeeping so older tickets gain new features without manual steps:
SLA clocks for tickets created before SLA existed, and the similarity index for all tickets."""
from __future__ import annotations

import logging

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import Ticket
from app.services import llm, rag

log = logging.getLogger(__name__)


def backfill() -> None:
    with SessionLocal() as db:
        tickets = db.scalars(select(Ticket)).all()
        n_sla = 0
        for t in tickets:
            if t.priority and t.response_due_at is None:
                t.set_deadlines()
                n_sla += 1
            if t.ai_language is None and t.ai_category is not None:
                t.ai_language = llm.guess_language(t.body)  # tickets triaged before language detection existed
            if t.status == "resolved" and t.resolved_at is None:
                t.resolved_at = t.updated_at
            if t.approved_at and t.first_response_at is None:
                t.first_response_at = t.approved_at
        db.commit()

        n_idx = 0
        if settings.enable_rag:
            try:
                have = rag.indexed_ticket_ids(db)
                missing = [(t.id, f"{t.subject}\n{t.body}") for t in tickets if t.id not in have]
                rag.index_tickets(missing, db)
                db.commit()
                n_idx = len(missing)
            except Exception as exc:
                log.warning("Similarity backfill failed: %s", exc)
    log.info("Backfill: %d tickets, %d SLA clocks set, %d added to similarity index", len(tickets), n_sla, n_idx)
