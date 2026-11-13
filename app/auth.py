"""Shared-secret protection for the internal API.

When `API_KEY` is set, every /tickets/* request must carry `X-API-Key: <key>`.
Public portal endpoints (/public/*) and /health are always open.
Locally, leave API_KEY empty and nothing changes.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.config import settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
