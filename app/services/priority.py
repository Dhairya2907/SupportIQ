"""Rule-based priority. Transparent and explainable, which is the point.

Returns (priority, reasons). LLM sentiment is one input among several.
Patterns cover English, Hinglish (Latin script) and Hindi (Devanagari).
"""
from __future__ import annotations

import re

HIGH_SIGNALS = [
    # English
    "urgent", "immediately", "asap", "legal", "lawyer", "consumer court", "fraud",
    "unauthorized", "unauthorised", "charged twice", "double charged", "locked out",
    "cannot access", "can't access", "hacked", "data loss", "lost all", "escalate",
    "no response", "no update", "still not", "third time", "3rd time",
    # Hinglish
    "jaldi", "turant", "abhi tak nahi", "koi jawab nahi", "koi update nahi", "lock ho gaya",
    "do baar", "teen baar", "kai baar", "paise kat gaye", "double kat", "hack",
    # Hindi
    "तुरंत", "जल्दी", "अभी तक नहीं", "कोई जवाब नहीं", "कोई अपडेट नहीं", "लॉक", "धोखाधड़ी",
    "दो बार", "तीन बार", "कई बार", "दोबारा कटे", "हैक",
]
MEDIUM_SIGNALS = [
    # English
    "delayed", "late", "not working", "error", "broken", "damaged", "wrong item",
    "missing", "refund", "cancel", "failed", "stuck",
    # Hinglish
    "late", "nahi chal raha", "kaam nahi kar raha", "kharab", "toota", "galat item", "refund", "cancel", "atka",
    # Hindi
    "देरी", "काम नहीं कर रहा", "खराब", "टूटा", "गलत", "रिफंड", "रद्द", "अटका",
]
SENTIMENT_WEIGHT = {"angry": 3, "frustrated": 2, "dissatisfied": 1, "neutral": 0, "positive": 0}
CATEGORY_WEIGHT = {"account": 1, "billing": 1, "delivery": 0, "technical": 0, "returns": 0}

REPEAT_CONTACT = re.compile(
    r"\b(contacted|called|emailed|reached out|wrote|complained)\b.{0,40}\b(twice|again|\d+\s*times|multiple|several)\b"
    r"|\b(do|teen|char|kai|\d+)\s*baar\b"            # Hinglish: "do baar call kiya"
    r"|(दो|तीन|चार|कई|\d+)\s*बार",                    # Hindi: "दो बार बात की"
    re.I,
)
DURATION = re.compile(
    r"\b(\d+)\s*(day|days|week|weeks|din|dino|hafte|hafta|saptah)\b"   # English + Hinglish
    r"|(\d+)\s*(दिन|दिनों|हफ्ते|हफ्ता|सप्ताह)",                          # Hindi
    re.I,
)
HINDI_NUMERALS = str.maketrans("०१२३४५६७८९", "0123456789")


def predict(text: str, category: str | None = None, sentiment: str | None = None) -> tuple[str, list[str]]:
    t = text.lower().translate(HINDI_NUMERALS)
    score = 0
    reasons: list[str] = []

    hits = [k for k in HIGH_SIGNALS if k in t]
    if hits:
        score += min(2 * len(hits), 4)
        reasons.append(f"high-signal phrases: {', '.join(hits[:3])}")

    hits = [k for k in MEDIUM_SIGNALS if k in t]
    if hits:
        score += min(len(hits), 2)
        reasons.append(f"issue phrases: {', '.join(hits[:3])}")

    if REPEAT_CONTACT.search(t):
        score += 2
        reasons.append("repeated contact with support")

    for m in DURATION.finditer(t):
        num = m.group(1) or m.group(3)
        unit = (m.group(2) or m.group(4) or "").lower()
        days = int(num) * (7 if unit.startswith(("week", "haft", "sapt", "हफ्त", "सप्त")) else 1)
        if days >= 5:
            score += 2 if days >= 10 else 1
            reasons.append(f"waiting {days} days")
            break

    if sentiment and SENTIMENT_WEIGHT.get(sentiment, 0):
        score += SENTIMENT_WEIGHT[sentiment]
        reasons.append(f"sentiment: {sentiment}")

    if category and CATEGORY_WEIGHT.get(category, 0):
        score += CATEGORY_WEIGHT[category]
        reasons.append(f"category baseline: {category}")

    if score >= 5:
        return "high", reasons
    if score >= 2:
        return "medium", reasons
    return "low", reasons or ["no urgency signals found"]
