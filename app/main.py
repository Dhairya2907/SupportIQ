import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import init_db
from app.routers import public, tickets
from app.services import maintenance, pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _startup_work() -> None:
    pipeline.warm_up()
    try:
        maintenance.backfill()
    except Exception as exc:
        log.warning("Backfill failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # create tables + add any new columns (forward-only migration)
    # Heavy loads and backfills run in the background so /health answers immediately.
    threading.Thread(target=_startup_work, name="startup", daemon=True).start()
    yield


app = FastAPI(
    title="SupportIQ",
    description=(
        "AI-assisted customer support ticket triage: classification, priority, summary, sentiment, "
        "RAG response drafts, similar-ticket detection, SLA tracking and analytics. English, Hindi and Hinglish."
    ),
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(tickets.router)  # internal: agent dashboard
app.include_router(public.router)  # external: customer portal


@app.get("/health")
def health():
    return {"status": "ok"}
