"""Generate synthetic labelled tickets with Gemini, in English, Hindi or Hinglish.

Usage:
    python ml/generate_synthetic.py --per-category 120                     # English -> synthetic.csv
    python ml/generate_synthetic.py --per-category 60 --language hi        # Hindi   -> synthetic_hi.csv
    python ml/generate_synthetic.py --per-category 60 --language hinglish  # Hinglish-> synthetic_hinglish.csv
Columns: text, category, source. ml/prepare_data.py picks up every synthetic*.csv.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BASE_DIR, CATEGORIES  # noqa: E402
from app.services.llm import _generate, _parse_json, available  # noqa: E402

PROCESSED = BASE_DIR / "data" / "processed"

SCENARIOS = {
    "billing": [
        "charged twice for one order", "subscription renewed without consent", "promo code not applied",
        "invoice shows wrong GST amount", "refund not credited after 10 days", "EMI option disappeared at checkout",
        "charged in wrong currency", "asking for an invoice copy", "price dropped after purchase, wants difference",
    ],
    "delivery": [
        "order late by several days with no update", "tracking stuck at 'out for delivery'",
        "package marked delivered but not received", "courier asked for extra cash", "wrong address, wants to change",
        "partial order delivered", "delivery attempted while customer at work, needs reschedule",
        "fragile item arrived crushed", "asking for delivery estimate before ordering",
    ],
    "technical": [
        "mobile app crashes on launch after update", "cannot add item to cart on website", "payment page shows blank screen",
        "smart device will not pair with wifi", "firmware update bricked the device", "laptop overheating and shutting down",
        "search results not loading", "push notifications not arriving", "app shows 'error 500' at checkout",
    ],
    "returns": [
        "received wrong size, wants exchange", "product defective out of the box", "wants to return after 20 days",
        "return pickup never happened", "refund amount less than paid", "wants replacement instead of refund",
        "item missing accessories", "opened item, changed mind", "gift return without invoice",
    ],
    "account": [
        "cannot log in, password reset email never arrives", "account locked after failed attempts",
        "OTP not received on phone", "suspicious login notification, worried about hacking",
        "wants to change registered email", "wants to delete account and data", "two-factor device lost",
        "order history disappeared from account", "someone else's orders showing in my account",
    ],
}
TONES = ["polite", "neutral", "frustrated", "angry", "confused", "brief and informal", "formal"]

MEANINGS = {
    "billing": "payments, charges, invoices, subscriptions, refund *amounts* being wrong",
    "delivery": "shipping, courier, tracking, delays, delivery address, package not received",
    "technical": "product or app/website malfunction, errors, device setup, software bugs",
    "returns": "returning, exchanging or replacing an item, return pickup, refund *process* for a return",
    "account": "login, password, OTP, account security, profile/email changes, account deletion",
}

LANGUAGE_RULES = {
    "en": "Write every message in English, the way Indian customers write to an online store.",
    "hi": "Write every message in Hindi using Devanagari script, the way Hindi-speaking customers write to an online store. "
          "Keep brand/product names and order numbers as they are. No English sentences, no transliteration.",
    "hinglish": "Write every message in Hinglish: Hindi words in Latin script freely mixed with English words, the way people "
                "type on WhatsApp (e.g. 'order abhi tak nahi aaya, 5 din ho gaye, please help'). No Devanagari.",
}

PROMPT = """Generate {n} realistic customer support messages for an Indian e-commerce store that sells electronics and home goods.

Every message must belong to the category: "{category}".
Category meaning: {meaning}

Language: {language_rule}

Vary these across the batch:
- scenario (use each at least once): {scenarios}
- tone: {tones}
- length: 1 to 5 sentences
- include concrete details (order numbers like #45812, amounts in rupees, day counts, product names)
- some messages have minor typos or missing punctuation, like real customers write
- do NOT mention the category name explicitly in every message

Return a JSON array of objects: [{{"text": "..."}}, ...]. Nothing else."""


def generate_batch(category: str, n: int, language: str) -> list[str]:
    scen = random.sample(SCENARIOS[category], k=min(5, len(SCENARIOS[category])))
    tones = random.sample(TONES, k=4)
    prompt = PROMPT.format(
        n=n, category=category, meaning=MEANINGS[category], language_rule=LANGUAGE_RULES[language],
        scenarios=scen, tones=tones,
    )
    data = _parse_json(_generate(prompt, json_mode=True))
    if isinstance(data, dict):
        data = data.get("items") or data.get("messages") or []
    return [str(d["text"]).strip() for d in data if isinstance(d, dict) and d.get("text")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=100)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--language", choices=list(LANGUAGE_RULES), default="en")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if not available():
        sys.exit("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")

    out = PROCESSED / ("synthetic.csv" if args.language == "en" else f"synthetic_{args.language}.csv")
    source = "synthetic" if args.language == "en" else f"synthetic_{args.language}"

    rows = []
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
        print(f"Resuming: {len(rows)} existing rows in {out.name}")

    for category in CATEGORIES:
        have = sum(1 for r in rows if r["category"] == category)
        while have < args.per_category:
            n = min(args.batch, args.per_category - have)
            try:
                texts = generate_batch(category, n, args.language)
            except Exception as exc:  # free-tier rate limits are common; back off and retry
                print(f"  [{category}] error: {str(exc)[:120]}. Sleeping 30s.")
                time.sleep(30)
                continue
            rows.extend({"text": t, "category": category, "source": source} for t in texts)
            have += len(texts)
            print(f"  [{args.language}/{category}] {have}/{args.per_category}")
            pd.DataFrame(rows).drop_duplicates("text").to_csv(out, index=False)
            time.sleep(2)

    df = pd.DataFrame(rows).drop_duplicates("text")
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
