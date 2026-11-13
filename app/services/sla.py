"""SLA deadlines and status, derived from data/knowledge_base/escalation_sla.md.

Business hours: 09:00 to 21:00 IST, Monday to Saturday.
First response: high 2h, medium 8h, low 24h, counted in business hours.
Resolution:     high 1, medium 3, low 5 business days (end of business that day).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
BUSINESS_START = 9
BUSINESS_END = 21
BUSINESS_WEEKDAYS = {0, 1, 2, 3, 4, 5}  # Monday=0 .. Saturday=5

FIRST_RESPONSE_HOURS = {"high": 2, "medium": 8, "low": 24}
RESOLUTION_BUSINESS_DAYS = {"high": 1, "medium": 3, "low": 5}
AT_RISK_WITHIN_HOURS = {"response": 1.0, "resolution": 8.0}


def aware(dt: datetime) -> datetime:
    """SQLite hands back naive datetimes; everything we store is UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _is_business_day(t: datetime) -> bool:
    return t.weekday() in BUSINESS_WEEKDAYS


def _next_opening(t: datetime) -> datetime:
    """Earliest instant >= t that falls inside business hours (t in IST)."""
    if _is_business_day(t) and BUSINESS_START <= t.hour < BUSINESS_END:
        return t
    if _is_business_day(t) and t.hour < BUSINESS_START:
        return t.replace(hour=BUSINESS_START, minute=0, second=0, microsecond=0)
    t = (t + timedelta(days=1)).replace(hour=BUSINESS_START, minute=0, second=0, microsecond=0)
    while not _is_business_day(t):
        t += timedelta(days=1)
    return t


def add_business_hours(start: datetime, hours: float) -> datetime:
    t = aware(start).astimezone(IST)
    remaining = timedelta(hours=hours)
    while remaining > timedelta(0):
        t = _next_opening(t)
        close = t.replace(hour=BUSINESS_END, minute=0, second=0, microsecond=0)
        step = min(remaining, close - t)
        t += step
        remaining -= step
    return t.astimezone(timezone.utc)


def add_business_days(start: datetime, days: int) -> datetime:
    t = aware(start).astimezone(IST)
    counted = 0
    while counted < days:
        t += timedelta(days=1)
        if _is_business_day(t):
            counted += 1
    return t.replace(hour=BUSINESS_END, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def deadlines(created_at: datetime, priority: str | None) -> tuple[datetime, datetime]:
    """(first_response_due, resolution_due) for a ticket of this priority."""
    p = priority if priority in FIRST_RESPONSE_HOURS else "medium"
    return (
        add_business_hours(created_at, FIRST_RESPONSE_HOURS[p]),
        add_business_days(created_at, RESOLUTION_BUSINESS_DAYS[p]),
    )


def evaluate(due_at: datetime | None, done_at: datetime | None, kind: str, now: datetime | None = None) -> dict | None:
    """Status of one SLA clock: met | breached | at_risk | on_track."""
    if due_at is None:
        return None
    due = aware(due_at)
    if done_at is not None:
        done = aware(done_at)
        return {
            "status": "met" if done <= due else "breached",
            "due_at": due,
            "hours_remaining": round((due - done).total_seconds() / 3600, 1),
        }
    now = aware(now or datetime.now(timezone.utc))
    remaining_h = (due - now).total_seconds() / 3600
    if remaining_h < 0:
        status = "breached"
    elif remaining_h <= AT_RISK_WITHIN_HOURS[kind]:
        status = "at_risk"
    else:
        status = "on_track"
    return {"status": status, "due_at": due, "hours_remaining": round(remaining_h, 1)}
