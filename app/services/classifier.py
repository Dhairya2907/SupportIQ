"""Ticket category classifier.

Uses the trained TF-IDF + Logistic Regression pipeline from ml/train_classifier.py.
If no artifact exists yet, falls back to a keyword heuristic so the API still runs.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import joblib

from app.config import CATEGORIES, settings

log = logging.getLogger(__name__)

KEYWORDS: dict[str, list[str]] = {
    "billing": ["charged", "charge", "invoice", "payment", "bill", "subscription", "double"],
    "delivery": ["deliver", "shipping", "shipped", "courier", "tracking", "package", "parcel", "arrived"],
    "technical": ["error", "crash", "bug", "not working", "does not work", "app", "website", "install", "update"],
    "returns": ["return", "exchange", "replace", "defective", "damaged", "wrong item", "refund"],
    "account": ["login", "log in", "password", "account", "locked", "otp", "verify", "email address"],
}


@lru_cache(maxsize=1)
def _load_model():
    path = settings.classifier_path
    if path.exists():
        log.info("Loading classifier from %s", path)
        return joblib.load(path)
    log.warning("No classifier artifact at %s. Using keyword fallback. Run ml/train_classifier.py.", path)
    return None


def _keyword_predict(text: str) -> tuple[str, float]:
    t = text.lower()
    scores = {c: sum(t.count(k) for k in kws) for c, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "technical", 0.2
    total = sum(scores.values())
    return best, round(0.4 + 0.4 * scores[best] / total, 2)


def predict(text: str) -> tuple[str, float]:
    """Return (category, confidence in 0..1)."""
    model = _load_model()
    if model is None:
        return _keyword_predict(text)
    proba = model.predict_proba([text])[0]
    idx = int(proba.argmax())
    label = str(model.classes_[idx])
    if label not in CATEGORIES:
        log.warning("Classifier produced unknown label %r", label)
    return label, round(float(proba[idx]), 3)
