"""Train the TF-IDF + Logistic Regression category classifier.

Usage:
    python ml/train_classifier.py
Saves ml/artifacts/classifier.joblib and ml/artifacts/train_metrics.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BASE_DIR, settings  # noqa: E402

DATA = BASE_DIR / "data" / "processed"
ARTIFACTS = BASE_DIR / "ml" / "artifacts"


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train={len(train)} test={len(test)}")

    grid = GridSearchCV(
        build_pipeline(),
        param_grid={"clf__C": [0.5, 1, 3, 10]},
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.perf_counter()
    grid.fit(train["text"], train["category"])
    train_s = time.perf_counter() - t0
    model = grid.best_estimator_
    print(f"best C={grid.best_params_['clf__C']}  cv macro-F1={grid.best_score_:.3f}  ({train_s:.1f}s)")

    pred = model.predict(test["text"])
    report = classification_report(test["category"], pred, digits=3, output_dict=True)
    print(classification_report(test["category"], pred, digits=3))

    t0 = time.perf_counter()
    for _ in range(50):
        model.predict_proba([test["text"].iloc[0]])
    latency_ms = (time.perf_counter() - t0) / 50 * 1000

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.classifier_path)
    metrics = {
        "model": "tfidf(word1-2 + char3-5) + logistic_regression",
        "best_C": grid.best_params_["clf__C"],
        "cv_macro_f1": round(grid.best_score_, 4),
        "test_macro_f1": round(f1_score(test["category"], pred, average="macro"), 4),
        "test_accuracy": round(report["accuracy"], 4),
        "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "train_rows": len(train),
        "test_rows": len(test),
        "predict_latency_ms": round(latency_ms, 2),
    }
    (ARTIFACTS / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {settings.classifier_path.name} and train_metrics.json  (predict ~{latency_ms:.1f} ms)")


if __name__ == "__main__":
    main()
