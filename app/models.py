from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.services import sla


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_ref: Mapped[str | None] = mapped_column(String(20), unique=True, index=True)
    subject: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(String(30), default="web_form")
    customer_email: Mapped[str | None] = mapped_column(String(200), index=True)
    customer_name: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)

    # AI outputs (what the model said)
    ai_category: Mapped[str | None] = mapped_column(String(30))
    ai_category_confidence: Mapped[float | None] = mapped_column(Float)
    ai_priority: Mapped[str | None] = mapped_column(String(10))
    ai_priority_reasons: Mapped[str | None] = mapped_column(Text)
    ai_sentiment: Mapped[str | None] = mapped_column(String(20))
    ai_language: Mapped[str | None] = mapped_column(String(10))  # en | hi | hinglish
    ai_team: Mapped[str | None] = mapped_column(String(50))
    ai_summary: Mapped[str | None] = mapped_column(Text)
    ai_suggested_response: Mapped[str | None] = mapped_column(Text)
    ai_groundedness: Mapped[float | None] = mapped_column(Float)
    ai_sources: Mapped[str | None] = mapped_column(Text)  # comma-separated KB sources
    ai_latency_ms: Mapped[int | None] = mapped_column(Integer)

    # Current values (AI by default, overwritten when an agent corrects them)
    category: Mapped[str | None] = mapped_column(String(30), index=True)
    priority: Mapped[str | None] = mapped_column(String(10), index=True)
    team: Mapped[str | None] = mapped_column(String(50))
    final_response: Mapped[str | None] = mapped_column(Text)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # SLA clocks (see services/sla.py)
    response_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolve_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_response_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    feedback: Mapped[list["Feedback"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")

    # Derived SLA status, exposed through the API schema.
    @property
    def response_sla(self) -> dict | None:
        return sla.evaluate(self.response_due_at, self.first_response_at, "response")

    @property
    def resolution_sla(self) -> dict | None:
        return sla.evaluate(self.resolve_due_at, self.resolved_at, "resolution")

    def set_deadlines(self) -> None:
        if self.priority:
            self.response_due_at, self.resolve_due_at = sla.deadlines(self.created_at or utcnow(), self.priority)


class Feedback(Base):
    """Audit trail: every time a human changes an AI prediction."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), index=True)
    field: Mapped[str] = mapped_column(String(30))  # category | priority | team | response
    ai_value: Mapped[str | None] = mapped_column(Text)
    human_value: Mapped[str | None] = mapped_column(Text)
    agent: Mapped[str] = mapped_column(String(100), default="agent")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="feedback")
