"""Test environment: offline (no Gemini), RAG disabled, throwaway SQLite file.

Env vars are set here because conftest is imported before any test module or app code.
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="supportiq_test_"))
os.environ["GEMINI_API_KEY"] = ""
os.environ["ENABLE_RAG"] = "false"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
