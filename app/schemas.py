from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TicketCreate(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)
    channel: str = "web_form"
    customer_email: str | None = None


class TicketUpdate(BaseModel):
    """Agent corrections. Any field left as None is untouched."""

    status: str | None = None
    category: str | None = None
    priority: str | None = None
    team: str | None = None
    final_response: str | None = None
    approve: bool = False
    agent: str = "agent"


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field: str
    ai_value: str | None
    human_value: str | None
    agent: str
    created_at: datetime


class SlaStatus(BaseModel):
    status: str  # met | breached | at_risk | on_track
    due_at: datetime
    hours_remaining: float


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticket_ref: str | None
    subject: str
    body: str
    channel: str
    customer_email: str | None
    customer_name: str | None = None
    status: str

    ai_category: str | None
    ai_category_confidence: float | None
    ai_priority: str | None
    ai_priority_reasons: str | None
    ai_sentiment: str | None
    ai_language: str | None = None
    ai_team: str | None
    ai_summary: str | None
    ai_suggested_response: str | None
    ai_groundedness: float | None
    ai_sources: str | None
    ai_latency_ms: int | None

    category: str | None
    priority: str | None
    team: str | None
    final_response: str | None
    approved_at: datetime | None

    response_due_at: datetime | None = None
    resolve_due_at: datetime | None = None
    first_response_at: datetime | None = None
    resolved_at: datetime | None = None
    response_sla: SlaStatus | None = None
    resolution_sla: SlaStatus | None = None

    created_at: datetime
    updated_at: datetime
    feedback: list[FeedbackOut] = []


# ---------------------------------------------------------------- customer portal (public)
class PublicTicketCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=10)


class PublicTicketOut(BaseModel):
    """Only what a customer may see. No AI internals, no priority, no agent notes."""

    ticket_ref: str
    status: str  # open | pending | resolved
    status_label: str  # customer wording
    subject: str
    body: str
    created_at: datetime
    expected_reply_by: datetime | None  # first-response SLA deadline (provisional until triaged)
    agent_reply: str | None  # only once an agent has approved it
    replied_at: datetime | None
    resolved_at: datetime | None


class SimilarTicket(BaseModel):
    id: int
    ticket_ref: str | None
    subject: str
    status: str
    category: str | None
    priority: str | None
    score: float
    resolution: str | None  # the agent's final response, if any
    created_at: datetime


class BulkCreateResult(BaseModel):
    created: int
    skipped: int
    ticket_ids: list[int]
    ai_pending: int = 0  # tickets queued for background triage


class Metrics(BaseModel):
    total: int
    ai_pending: int = 0
    by_status: dict[str, int]
    by_category: dict[str, int]
    by_priority: dict[str, int]
    correction_rate: dict[str, float]  # field -> fraction of tickets corrected
    avg_latency_ms: float | None
    avg_groundedness: float | None
    sla_at_risk: int = 0
    sla_breached: int = 0


class Analytics(BaseModel):
    days: int
    total: int
    triaged: int
    volume_by_day: list[dict]  # [{"date": "2026-11-12", "count": 7}, ...]
    by_category_status: dict[str, dict[str, int]]
    by_category: dict[str, int]
    by_priority: dict[str, int]
    by_sentiment: dict[str, int]
    by_language: dict[str, int]
    avg_resolution_hours_by_priority: dict[str, float]
    avg_first_response_hours: float | None
    sla: dict[str, dict[str, int]]  # {"response": {"met": n, ...}, "resolution": {...}}
    top_complaints: dict[str, list[dict]]  # category -> [{"term": "...", "count": n}]
    correction_rate: dict[str, float]
    avg_classifier_confidence: float | None
    # AI quality in practice
    live_category_accuracy: float | None  # 1 - share of triaged tickets whose category an agent changed
    approved: int
    approved_unedited: int  # approved with the AI draft sent as-is
    acceptance_rate: float | None
    offline_eval: dict | None  # ml/artifacts/eval_report.json, if present
    needs_review: list[dict]  # [{"id", "ticket_ref", "subject", "reasons": [...]}]
