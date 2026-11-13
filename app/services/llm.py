"""Gemini wrapper. Two jobs: analyse a ticket (summary + sentiment) and draft a grounded reply.

If GEMINI_API_KEY is not set, every call returns a placeholder so the rest of the
pipeline still works offline and in tests.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from app.config import SENTIMENTS, settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client():
    if not settings.gemini_api_key:
        return None
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def available() -> bool:
    return _client() is not None


def _generate(prompt: str, json_mode: bool = False) -> str:
    client = _client()
    if client is None:
        raise RuntimeError("Gemini not configured")
    from google.genai import types

    from google.genai import errors

    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json" if json_mode else "text/plain",
        # Fail fast: 2 quick attempts per model, then our own fallback below takes over.
        http_options=types.HttpOptions(
            timeout=45_000,  # ms
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1.0, max_delay=3.0),
        ),
    )
    models = [settings.gemini_model]
    if settings.gemini_fallback_model and settings.gemini_fallback_model != settings.gemini_model:
        models.append(settings.gemini_fallback_model)
    last_exc: Exception | None = None
    for model in models:
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            return resp.text or ""
        except errors.ServerError as exc:  # 5xx: model overloaded
            log.warning("%s unavailable (%s); trying fallback", model, exc.code)
            last_exc = exc
        except errors.ClientError as exc:
            if exc.code == 429:  # rate limit on the free tier
                log.warning("%s rate-limited; trying fallback", model)
                last_exc = exc
            else:
                raise
    raise last_exc or RuntimeError("No Gemini model responded")


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON: %.200s", raw)
        return {}


ANALYSE_PROMPT = """You are a customer support triage assistant.

Read the customer message and return JSON with exactly these keys:
- "summary": one or two sentences, third person, factual, max 40 words. Include concrete details (durations, counts, amounts).
- "sentiment": one of {sentiments}.

Customer subject: {subject}
Customer message:
\"\"\"{body}\"\"\"
"""


def analyse(subject: str, body: str) -> dict:
    """Return {"summary": str, "sentiment": str}."""
    if not available():
        return {"summary": body[:160] + ("..." if len(body) > 160 else ""), "sentiment": "neutral"}
    raw = _generate(ANALYSE_PROMPT.format(sentiments=SENTIMENTS, subject=subject, body=body), json_mode=True)
    data = _parse_json(raw)
    sentiment = str(data.get("sentiment", "neutral")).lower()
    if sentiment not in SENTIMENTS:
        sentiment = "neutral"
    return {"summary": str(data.get("summary", "")).strip() or body[:160], "sentiment": sentiment}


DRAFT_PROMPT = """You are writing a reply on behalf of the customer support team. An agent will review it before it is sent.

Rules:
- Use ONLY the policy excerpts below for any factual claim about timelines, fees, or procedures. Do not invent policy.
- If the excerpts do not cover the question, say the team will check and follow up; do not guess.
- Acknowledge the customer's situation in one sentence, then be concrete.
- 70 to 130 words. Plain text. No subject line, no placeholders like [Name].
- When you use a policy fact, mention the source in square brackets, e.g. [shipping_policy].

Policy excerpts:
{context}

Ticket category: {category}
Customer subject: {subject}
Customer message:
\"\"\"{body}\"\"\"

Reply:"""


def draft_response(subject: str, body: str, category: str, chunks: list[dict]) -> str:
    if not available():
        return "(Gemini not configured. Set GEMINI_API_KEY to enable AI response drafts.)"
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks) or "(no relevant policy found)"
    return _generate(DRAFT_PROMPT.format(context=context, category=category, subject=subject, body=body)).strip()


TRIAGE_PROMPT = """You are a customer support triage assistant. An agent will review everything you produce.

The customer may write in English, Hindi (Devanagari script), or Hinglish (Hindi and English mixed, Latin script).

Return JSON with exactly these keys:
- "language": "en", "hi", or "hinglish", whichever the customer used.
- "summary": one or two sentences IN ENGLISH, third person, factual, max 40 words, with concrete details (durations, counts, amounts). Describe ONLY what the customer wrote; the policy excerpts are for the reply, never for the summary.
- "sentiment": one of {sentiments}.
- "response": a reply to the customer following the rules below.

Reply rules:
- Write the reply in the same language and script the customer used (Hindi in Devanagari, Hinglish in Latin script, English in English).
- Use ONLY the policy excerpts below for any factual claim about timelines, fees, or procedures. Do not invent policy.
- If the excerpts do not cover the question, say the team will check and follow up; do not guess.
- Acknowledge the customer's situation in one sentence, then be concrete.
- 70 to 130 words. Plain text. No subject line, no placeholders like [Name].
- When you use a policy fact, mention the source in square brackets, e.g. [shipping_policy].

Policy excerpts:
{context}

Ticket category: {category}
Customer subject: {subject}
Customer message:
\"\"\"{body}\"\"\"
"""


def triage(subject: str, body: str, category: str, chunks: list[dict]) -> dict:
    """Summary + sentiment + draft reply in ONE call. Halves quota use and latency versus
    analyse() + draft_response(). Returns {"summary", "sentiment", "response"}."""
    if not available():
        return {
            "language": guess_language(body),
            "summary": body[:160] + ("..." if len(body) > 160 else ""),
            "sentiment": "neutral",
            "response": "(Gemini not configured. Set GEMINI_API_KEY to enable AI response drafts.)",
        }
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks) or "(no relevant policy found)"
    prompt = TRIAGE_PROMPT.format(sentiments=SENTIMENTS, context=context, category=category, subject=subject, body=body)
    data = _parse_json(_generate(prompt, json_mode=True))
    sentiment = str(data.get("sentiment", "neutral")).lower()
    if sentiment not in SENTIMENTS:
        sentiment = "neutral"
    language = str(data.get("language", "")).lower()
    if language not in LANGUAGES:
        language = guess_language(body)
    return {
        "language": language,
        "summary": str(data.get("summary", "")).strip() or body[:160],
        "sentiment": sentiment,
        "response": (str(data.get("response", "")).strip() or None),
    }


LANGUAGES = ("en", "hi", "hinglish")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_HINGLISH_HINTS = re.compile(
    r"\b(nahi|nahin|kyun|kyu|kab|kaise|mera|meri|mujhe|hai|hain|tha|thi|karo|kar|do|abhi|bhi|paisa|paise|wapas|order kiya|aaya|aya)\b",
    re.I,
)


def guess_language(text: str) -> str:
    """Script-based fallback when the LLM is unavailable: Devanagari -> hi, Hindi words in Latin -> hinglish."""
    if _DEVANAGARI.search(text):
        return "hi"
    if len(_HINGLISH_HINTS.findall(text)) >= 2:
        return "hinglish"
    return "en"
