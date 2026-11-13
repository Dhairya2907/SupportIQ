"""Offline evaluation on the held-out test set.

Reports:
  - classification accuracy, macro F1, per-class F1, confusion matrix
  - routing accuracy (team derived from predicted category vs. from true category)
  - classifier latency

Usage:
    python ml/evaluate.py
Writes ml/artifacts/eval_report.json and confusion_matrix.png.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BASE_DIR, CATEGORIES  # noqa: E402
from app.services import classifier, routing  # noqa: E402

DATA = BASE_DIR / "data" / "processed"
ARTIFACTS = BASE_DIR / "ml" / "artifacts"


def main():
    test = pd.read_csv(DATA / "test.csv")
    t0 = time.perf_counter()
    preds = [classifier.predict(t)[0] for t in test["text"]]
    latency_ms = (time.perf_counter() - t0) / len(test) * 1000

    y = test["category"].tolist()
    cm = confusion_matrix(y, preds, labels=CATEGORIES)
    routing_acc = sum(routing.route(p) == routing.route(t) for p, t in zip(preds, y)) / len(y)

    report = {
        "rows": len(test),
        "accuracy": round(accuracy_score(y, preds), 4),
        "macro_f1": round(f1_score(y, preds, average="macro"), 4),
        "per_class_f1": dict(zip(CATEGORIES, [round(v, 4) for v in f1_score(y, preds, average=None, labels=CATEGORIES)])),
        "routing_accuracy": round(routing_acc, 4),
        "latency_ms_per_ticket": round(latency_ms, 2),
        "confusion_matrix": {"labels": CATEGORIES, "matrix": cm.tolist()},
    }
    by_source = {}
    for src, grp in test.assign(pred=preds).groupby("source"):
        by_source[src] = round(f1_score(grp["category"], grp["pred"], average="macro"), 4)
    report["macro_f1_by_source"] = by_source

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_matrix"}, indent=2))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45, ha="right")
        ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
        for i in range(len(CATEGORIES)):
            for j in range(len(CATEGORIES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=150)
        print(f"\nSaved {ARTIFACTS / 'confusion_matrix.png'}")
    except ImportError:
        print("\n(matplotlib not installed; skipped confusion_matrix.png)")


if __name__ == "__main__":
    main()
