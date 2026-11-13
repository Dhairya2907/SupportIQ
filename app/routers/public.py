"""Customer-facing endpoints used by the portal (portal/app.py).

Customers can raise a ticket and check its status with ticket number + email.
Nothing internal (AI fields, priority, team, audit trail) is ever returned here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Ticket
from app.routers.tickets import _new_ticket, _triage_in_background
from app.schemas import PublicTicketCreate, PublicTicketOut
from app.services import sla

log = logging.getLogger(__name__)
router = APIRouter(prefix="/public", tags=["customer portal"])

STATUS_LABEL = {
    "open": "Received. Our team is reviewing it.",
    "pending": "An agent has replied. We are working on it.",
    "resolved": "Resolved.",
}


def _to_public(t: Ticket) -> PublicTicketOut:
    replied = t.approved_at is not None and bool(t.final_response)
    expected = t.response_due_at
    if expected is None:  # not triaged yet: promise the medium-priority window
        expected, _ = sla.deadlines(t.created_at or datetime.now(timezone.utc), "medium")
    def utc(dt):  # always send an explicit UTC marker; SQLite returns naive datetimes
        return sla.aware(dt) if dt else None

    return PublicTicketOut(
        ticket_ref=t.ticket_ref or f"TKT-{1000 + t.id}",
        status=t.status,
        status_label=STATUS_LABEL.get(t.status, t.status),
        subject=t.subject,
        body=t.body,
        created_at=utc(t.created_at),
        expected_reply_by=None if replied else utc(expected),
        agent_reply=t.final_response if replied else None,
        replied_at=utc(t.approved_at) if replied else None,
        resolved_at=utc(t.resolved_at),
    )


@router.post("/tickets", response_model=PublicTicketOut, status_code=201)
def raise_ticket(payload: PublicTicketCreate, background: BackgroundTasks, db: Session = Depends(get_db)):
    """Create a ticket and return the ticket number immediately. AI triage runs in the background."""
    ticket = _new_ticket(db, payload.subject, payload.body, "portal", payload.email.strip().lower())
    ticket.customer_name = payload.name.strip()
    db.commit()
    db.refresh(ticket)
    background.add_task(_triage_in_background, [ticket.id])
    return _to_public(ticket)


@router.get("/tickets/{ticket_ref}", response_model=PublicTicketOut)
def ticket_status(ticket_ref: str, email: str = Query(..., description="email used when raising the ticket"), db: Session = Depends(get_db)):
    """Status check. The email must match; otherwise we answer 404 so ticket numbers cannot be enumerated."""
    ref = ticket_ref.strip().upper()
    if ref.isdigit():
        ref = f"TKT-{ref}"
    t = db.scalar(select(Ticket).where(Ticket.ticket_ref == ref))
    if t is None or (t.customer_email or "").lower() != email.strip().lower():
        raise HTTPException(404, "No ticket found for that number and email.")
    return _to_public(t)


@router.get("/tickets", response_model=list[PublicTicketOut])
def my_tickets(email: str = Query(...), limit: int = Query(20, le=100), db: Session = Depends(get_db)):
    """All tickets raised with this email, newest first."""
    stmt = (
        select(Ticket)
        .where(func.lower(Ticket.customer_email) == email.strip().lower())
        .order_by(Ticket.created_at.desc())
        .limit(limit)
    )
    return [_to_public(t) for t in db.scalars(stmt).all()]
