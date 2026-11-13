from __future__ import annotations

import csv
import io
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import CATEGORIES, PRIORITIES, STATUSES
from app.db import SessionLocal, get_db
from app.models import Feedback, Ticket
from app.schemas import (
    Analytics,
    BulkCreateResult,
    Metrics,
    SimilarTicket,
    TicketCreate,
    TicketOut,
    TicketUpdate,
)
from app.services import pipeline, rag
from app.services.sla import aware

log = logging.getLogger(__name__)
router = APIRouter(prefix="/tickets", tags=["tickets"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------------- helpers
def _apply_ai(ticket: Ticket, ai: dict) -> None:
    """Write pipeline output onto a ticket, set working values, SLA clocks and the similarity index."""
    for k, v in ai.items():
        setattr(ticket, k, v)
    ticket.category, ticket.priority, ticket.team = ai["ai_category"], ai["ai_priority"], ai["ai_team"]
    ticket.set_deadlines()
    try:
        rag.index_ticket(ticket.id, f"{ticket.subject}\n{ticket.body}")
    except Exception as exc:  # similarity is a nice-to-have, never a blocker
        log.warning("Could not index ticket %s for similarity: %s", ticket.id, exc)


def _new_ticket(db: Session, subject: str, body: str, channel: str, email: str | None = None) -> Ticket:
    ticket = Ticket(subject=subject, body=body, channel=channel, customer_email=email, status="open")
    db.add(ticket)
    db.flush()
    ticket.ticket_ref = f"TKT-{1000 + ticket.id}"
    return ticket


def _correction_rate(db: Session, total: int) -> dict[str, float]:
    if not total:
        return {}
    rows = db.execute(select(Feedback.field, func.count(func.distinct(Feedback.ticket_id))).group_by(Feedback.field)).all()
    return {field: round(n / total, 3) for field, n in rows}


def _triage_in_background(ticket_ids: list[int]) -> None:
    """Runs after the upload response is sent. One short transaction per ticket."""
    for tid in ticket_ids:
        with SessionLocal() as db:
            ticket = db.get(Ticket, tid)
            if ticket is None or ticket.ai_category is not None:
                continue
            try:
                ai = pipeline.run(ticket.subject, ticket.body)
            except Exception as exc:
                log.exception("Background triage failed for ticket %s: %s", tid, exc)
                continue
            _apply_ai(ticket, ai)
            db.commit()
            log.info("Triaged %s in background (%s, %s)", ticket.ticket_ref, ai["ai_category"], ai["ai_priority"])


# ----------------------------------------------------------------------------- create
@router.post("", response_model=TicketOut, status_code=201)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    ticket = _new_ticket(db, payload.subject, payload.body, payload.channel, payload.customer_email)
    _apply_ai(ticket, pipeline.run(payload.subject, payload.body))
    db.commit()
    db.refresh(ticket)
    return ticket


@router.post("/upload-csv", response_model=BulkCreateResult)
async def upload_csv(background: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """CSV with columns `subject` and `body` (aliases: title/description/text/message).

    Rows are inserted immediately and the response returns at once. AI triage runs in the
    background one ticket at a time; watch `ai_pending` in GET /tickets/metrics.
    """
    content = (await file.read()).decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(content))
    created, skipped, ids = 0, 0, []
    for row in reader:
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        subject = row.get("subject") or row.get("title") or row.get("ticket subject") or ""
        body = row.get("body") or row.get("description") or row.get("text") or row.get("message") or ""
        if not body:
            skipped += 1
            continue
        ticket = _new_ticket(db, subject[:200] or body[:80], body, "csv_upload")
        ids.append(ticket.id)
        created += 1
    db.commit()
    if ids:
        background.add_task(_triage_in_background, ids)
    return BulkCreateResult(created=created, skipped=skipped, ticket_ids=ids, ai_pending=len(ids))


# ----------------------------------------------------------------------------- read
@router.get("", response_model=list[TicketOut])
def list_tickets(
    status: str | None = Query(None),
    category: str | None = Query(None),
    priority: str | None = Query(None),
    team: str | None = Query(None),
    language: str | None = Query(None),
    sla: str | None = Query(None, description="at_risk | breached: tickets whose response or resolution clock is in that state"),
    q: str | None = Query(None, description="substring match on subject/body"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    stmt = select(Ticket)
    if status:
        stmt = stmt.where(Ticket.status == status)
    if category:
        stmt = stmt.where(Ticket.category == category)
    if priority:
        stmt = stmt.where(Ticket.priority == priority)
    if team:
        stmt = stmt.where(Ticket.team == team)
    if language:
        stmt = stmt.where(Ticket.ai_language == language)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(Ticket.subject.ilike(like) | Ticket.body.ilike(like))
    stmt = stmt.order_by(Ticket.created_at.desc())
    if not sla:
        return db.scalars(stmt.limit(limit)).all()
    # SLA state is derived from "now", so filter in Python.
    wanted = sla.lower()
    out = []
    for t in db.scalars(stmt).all():
        states = {s["status"] for s in (t.response_sla, t.resolution_sla) if s}
        if wanted in states:
            out.append(t)
        if len(out) >= limit:
            break
    return out


@router.get("/metrics", response_model=Metrics)
def metrics(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Ticket.id))) or 0

    def counts(col):
        return {str(k): v for k, v in db.execute(select(col, func.count()).group_by(col)).all() if k is not None}

    at_risk = breached = 0
    for t in db.scalars(select(Ticket).where(Ticket.status != "resolved")).all():
        states = {s["status"] for s in (t.response_sla, t.resolution_sla) if s}
        if "breached" in states:
            breached += 1
        elif "at_risk" in states:
            at_risk += 1

    return Metrics(
        total=total,
        ai_pending=db.scalar(select(func.count(Ticket.id)).where(Ticket.ai_category.is_(None))) or 0,
        by_status=counts(Ticket.status),
        by_category=counts(Ticket.category),
        by_priority=counts(Ticket.priority),
        correction_rate=_correction_rate(db, total),
        avg_latency_ms=db.scalar(select(func.avg(Ticket.ai_latency_ms))),
        avg_groundedness=db.scalar(select(func.avg(Ticket.ai_groundedness))),
        sla_at_risk=at_risk,
        sla_breached=breached,
    )


@router.get("/analytics", response_model=Analytics)
def analytics(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)):
    tickets = db.scalars(select(Ticket)).all()
    now = _utcnow()
    since = now - timedelta(days=days)

    volume = Counter(aware(t.created_at).date().isoformat() for t in tickets if aware(t.created_at) >= since)
    by_cat_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in tickets:
        by_cat_status[t.category or "pending"][t.status] += 1
    by_language = Counter((t.ai_language or "unknown") for t in tickets)

    res_hours: dict[str, list[float]] = defaultdict(list)
    for t in tickets:
        if t.resolved_at:
            res_hours[t.priority or "unknown"].append((aware(t.resolved_at) - aware(t.created_at)).total_seconds() / 3600)
    first_resp = [(aware(t.first_response_at) - aware(t.created_at)).total_seconds() / 3600 for t in tickets if t.first_response_at]

    sla_counts = {"response": Counter(), "resolution": Counter()}
    for t in tickets:
        if t.response_sla:
            sla_counts["response"][t.response_sla["status"]] += 1
        if t.resolution_sla:
            sla_counts["resolution"][t.resolution_sla["status"]] += 1

    top: dict[str, list[dict]] = {}
    docs_by_cat: dict[str, list[str]] = defaultdict(list)
    for t in tickets:
        if t.ai_summary and t.category:
            docs_by_cat[t.category].append(t.ai_summary)
    for cat, docs in docs_by_cat.items():
        if len(docs) < 3:
            continue
        try:
            from sklearn.feature_extraction.text import CountVectorizer

            vec = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=2)
            X = vec.fit_transform(docs)
            freq = X.sum(axis=0).A1
            terms = vec.get_feature_names_out()
            order = freq.argsort()[::-1][:6]
            top[cat] = [{"term": terms[i], "count": int(freq[i])} for i in order]
        except ValueError:  # empty vocabulary
            continue

    conf = [t.ai_category_confidence for t in tickets if t.ai_category_confidence is not None]
    triaged = [t for t in tickets if t.ai_category is not None]
    correction_rate = _correction_rate(db, len(tickets))

    # Live accuracy: how often agents left the AI category alone.
    cat_corrected = db.scalar(
        select(func.count(func.distinct(Feedback.ticket_id))).where(Feedback.field == "category")
    ) or 0
    live_acc = round(1 - cat_corrected / len(triaged), 3) if triaged else None

    # Acceptance: approved replies sent exactly as the AI drafted them.
    approved = [t for t in tickets if t.approved_at is not None]
    unedited = [t for t in approved if t.final_response and t.final_response.strip() == (t.ai_suggested_response or "").strip()]

    # Tickets a human should look at first.
    needs: list[dict] = []
    for t in tickets:
        if t.status == "resolved":
            continue
        reasons = []
        if t.ai_category is None:
            reasons.append("AI triage still pending")
        else:
            if (t.ai_category_confidence or 1) < 0.5:
                reasons.append(f"low category confidence ({t.ai_category_confidence:.2f})")
            if t.ai_suggested_response is None:
                reasons.append("no AI draft (Gemini unavailable)")
            elif t.ai_groundedness is not None and t.ai_groundedness < 0.4:
                reasons.append(f"draft weakly grounded in policy ({t.ai_groundedness:.2f})")
        for clock, label in ((t.response_sla, "first response"), (t.resolution_sla, "resolution")):
            if clock and clock["status"] == "breached":
                reasons.append(f"{label} SLA breached")
            elif clock and clock["status"] == "at_risk":
                reasons.append(f"{label} due in {clock['hours_remaining']:.1f}h")
        if reasons:
            needs.append({"id": t.id, "ticket_ref": t.ticket_ref, "subject": t.subject, "priority": t.priority, "reasons": reasons})
    needs.sort(key=lambda n: (-len(n["reasons"]), n["id"]))

    offline = None
    try:
        import json

        from app.config import BASE_DIR

        report = BASE_DIR / "ml" / "artifacts" / "eval_report.json"
        if report.exists():
            data = json.loads(report.read_text())
            offline = {k: data[k] for k in ("rows", "accuracy", "macro_f1", "per_class_f1", "macro_f1_by_source", "latency_ms_per_ticket") if k in data}
    except Exception as exc:
        log.warning("Could not read eval report: %s", exc)

    return Analytics(
        days=days,
        total=len(tickets),
        triaged=len(triaged),
        volume_by_day=[{"date": d, "count": volume[d]} for d in sorted(volume)],
        by_category_status={k: dict(v) for k, v in by_cat_status.items()},
        by_category=dict(Counter(t.category for t in tickets if t.category)),
        by_priority=dict(Counter(t.priority for t in tickets if t.priority)),
        by_sentiment=dict(Counter(t.ai_sentiment for t in tickets if t.ai_sentiment)),
        by_language=dict(by_language),
        avg_resolution_hours_by_priority={p: round(sum(v) / len(v), 1) for p, v in res_hours.items()},
        avg_first_response_hours=round(sum(first_resp) / len(first_resp), 2) if first_resp else None,
        sla={k: dict(v) for k, v in sla_counts.items()},
        top_complaints=top,
        correction_rate=correction_rate,
        avg_classifier_confidence=round(sum(conf) / len(conf), 3) if conf else None,
        live_category_accuracy=live_acc,
        approved=len(approved),
        approved_unedited=len(unedited),
        acceptance_rate=round(len(unedited) / len(approved), 3) if approved else None,
        offline_eval=offline,
        needs_review=needs[:25],
    )


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return ticket


@router.get("/{ticket_id}/similar", response_model=list[SimilarTicket])
def similar(ticket_id: int, k: int = Query(5, ge=1, le=20), db: Session = Depends(get_db)):
    """Previously seen tickets closest in meaning, with the agent's resolution where one exists."""
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    try:
        hits = rag.similar_tickets(f"{ticket.subject}\n{ticket.body}", k=k, exclude_id=ticket.id)
    except Exception as exc:
        log.warning("Similarity search failed: %s", exc)
        return []
    if not hits:
        return []
    rows = {t.id: t for t in db.scalars(select(Ticket).where(Ticket.id.in_([h["ticket_id"] for h in hits]))).all()}
    out = []
    for h in hits:
        t = rows.get(h["ticket_id"])
        if t is None:
            continue
        out.append(
            SimilarTicket(
                id=t.id, ticket_ref=t.ticket_ref, subject=t.subject, status=t.status, category=t.category,
                priority=t.priority, score=h["score"], resolution=t.final_response, created_at=t.created_at,
            )
        )
    return out


# ----------------------------------------------------------------------------- agent review
@router.patch("/{ticket_id}", response_model=TicketOut)
def update_ticket(ticket_id: int, payload: TicketUpdate, db: Session = Depends(get_db)):
    """Agent review. Every change to an AI field is recorded in the feedback table."""
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")

    def record(field: str, ai_value, human_value):
        if human_value is not None and human_value != ai_value:
            db.add(Feedback(ticket_id=ticket.id, field=field, ai_value=ai_value, human_value=human_value, agent=payload.agent))

    if payload.category is not None:
        if payload.category not in CATEGORIES:
            raise HTTPException(422, f"category must be one of {CATEGORIES}")
        record("category", ticket.ai_category, payload.category)
        ticket.category = payload.category
    if payload.priority is not None:
        if payload.priority not in PRIORITIES:
            raise HTTPException(422, f"priority must be one of {PRIORITIES}")
        record("priority", ticket.ai_priority, payload.priority)
        if payload.priority != ticket.priority:
            ticket.priority = payload.priority
            ticket.set_deadlines()  # SLA clocks follow the corrected priority
    if payload.team is not None:
        record("team", ticket.ai_team, payload.team)
        ticket.team = payload.team
    if payload.final_response is not None:
        record("response", ticket.ai_suggested_response, payload.final_response)
        ticket.final_response = payload.final_response
    if payload.status is not None:
        if payload.status not in STATUSES:
            raise HTTPException(422, f"status must be one of {STATUSES}")
        ticket.status = payload.status
    if payload.approve:
        now = _utcnow()
        ticket.approved_at = now
        if ticket.first_response_at is None:
            ticket.first_response_at = now  # stops the first-response SLA clock
        if ticket.final_response is None:
            ticket.final_response = ticket.ai_suggested_response
        if ticket.status == "open":
            ticket.status = "pending"
    if ticket.status == "resolved" and ticket.resolved_at is None:
        ticket.resolved_at = _utcnow()
    elif ticket.status != "resolved":
        ticket.resolved_at = None

    db.commit()
    db.refresh(ticket)
    return ticket
