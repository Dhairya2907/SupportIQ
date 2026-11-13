"""Merge the Kaggle dataset with synthetic tickets and split train/test.

Kaggle source: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
Download `customer_support_tickets.csv` into data/raw/.

Its `Ticket Type` values map onto our categories like this:
    Technical issue      -> technical
    Billing inquiry      -> billing
    Refund request       -> returns
    Cancellation request -> account      (closest fit; review a sample before trusting it)
    Product inquiry      -> dropped      (pre-sales, not a support category we model)

Known weakness: the Kaggle descriptions are templated and contain a
`{product_purchased}` placeholder, which we fill from the `Product Purchased` column.
Synthetic data (ml/generate_synthetic.py) supplies `delivery` tickets and adds variety.

Usage:
    python ml/prepare_data.py
Writes data/processed/train.csv and test.csv with columns: text, category, source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BASE_DIR, CATEGORIES  # noqa: E402

RAW = BASE_DIR / "data" / "raw" / "customer_support_tickets.csv"
SYNTH = BASE_DIR / "data" / "processed" / "synthetic.csv"
OUT_DIR = BASE_DIR / "data" / "processed"

KAGGLE_MAP = {
    "Technical issue": "technical",
    "Billing inquiry": "billing",
    "Refund request": "returns",
    "Cancellation request": "account",
}


def load_kaggle(max_per_class: int = 0) -> pd.DataFrame:
    """Default 0 = excluded from training.

    Finding (2026-11-12): a classifier trained on Kaggle + synthetic scored macro-F1 0.98 on
    synthetic test rows and 0.23 on Kaggle test rows (chance = 0.25). The Kaggle descriptions
    are generic templates ("I'm having an issue with the {product}") that carry no signal
    about `Ticket Type`, so its labels are effectively noise. Pass --kaggle-per-class N to
    include it anyway, e.g. to reproduce that result.
    """
    if max_per_class <= 0:
        return pd.DataFrame(columns=["text", "category", "source"])
    if not RAW.exists():
        print(f"Kaggle file not found at {RAW}. Continuing with synthetic data only.")
        return pd.DataFrame(columns=["text", "category", "source"])
    df = pd.read_csv(RAW)
    df["category"] = df["Ticket Type"].map(KAGGLE_MAP)
    df = df.dropna(subset=["category"])
    product = df["Product Purchased"].fillna("the product")
    desc = [d.replace("{product_purchased}", p) for d, p in zip(df["Ticket Description"].fillna(""), product)]
    df["text"] = (df["Ticket Subject"].fillna("") + ". " + pd.Series(desc, index=df.index)).str.strip()
    df["source"] = "kaggle"
    # Cap per class so templated Kaggle text does not swamp the synthetic rows.
    df = df.sample(frac=1, random_state=42).groupby("category").head(max_per_class)
    return df[["text", "category", "source"]]


def load_synthetic() -> pd.DataFrame:
    """All synthetic*.csv files: English (synthetic.csv), Hindi (synthetic_hi.csv), Hinglish (synthetic_hinglish.csv)."""
    files = sorted(SYNTH.parent.glob("synthetic*.csv"))
    if not files:
        print(f"No synthetic data in {SYNTH.parent}. Run ml/generate_synthetic.py first.")
        return pd.DataFrame(columns=["text", "category", "source"])
    frames = [pd.read_csv(f)[["text", "category", "source"]] for f in files]
    print("Synthetic files:", ", ".join(f.name for f in files))
    return pd.concat(frames, ignore_index=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-per-class", type=int, default=0, help="rows per class from the Kaggle set (0 = exclude)")
    args = ap.parse_args()

    df = pd.concat([load_kaggle(args.kaggle_per_class), load_synthetic()], ignore_index=True)
    df = df[df["category"].isin(CATEGORIES)]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10].drop_duplicates("text")
    if df.empty:
        sys.exit("No data. Add the Kaggle CSV and/or generate synthetic tickets.")

    print("Class balance:\n", df.groupby(["category", "source"]).size().unstack(fill_value=0), "\n")
    train, test = train_test_split(df, test_size=0.2, stratify=df["category"], random_state=42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"train: {len(train)} rows, test: {len(test)} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
