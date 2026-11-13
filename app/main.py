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


@app.get("/", include_in_schema=False)
def root():
    """Landing page for the hosted API (what you see when you open the Space URL)."""
    from fastapi.responses import HTMLResponse

    return HTMLResponse(
        """
        <html><head><title>SupportIQ API</title>
        <style>body{font-family:system-ui,sans-serif;max-width:640px;margin:48px auto;color:#1F2937;line-height:1.5}
        a{color:#4F6BED}code{background:#F3F4F6;padding:2px 6px;border-radius:4px}</style></head>
        <body><h1>SupportIQ API</h1>
        <p>AI-assisted customer support ticket triage. This is the backend; the user-facing apps are the
        agent dashboard and the customer Help Center.</p>
        <ul><li><a href="/docs">Interactive API docs</a></li><li><a href="/health">Health check</a></li>
        <li><a href="https://github.com/Dhairya2907/SupportIQ">Source on GitHub</a></li></ul>
        <p><small>Free-tier hosting: the first request after idle can take up to a minute while the model loads.</small></p>
        </body></html>
        """
    )
