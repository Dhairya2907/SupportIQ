"""SupportIQ agent dashboard (Streamlit). Talks to the FastAPI backend over HTTP.

Run:  streamlit run dashboard/app.py          (set SUPPORTIQ_API if the API is not on :8000)
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("SUPPORTIQ_API", "http://127.0.0.1:8000")
CATEGORIES = ["billing", "delivery", "technical", "returns", "account"]
PRIORITIES = ["low", "medium", "high"]
STATUSES = ["open", "pending", "resolved"]
TEAMS = ["Billing & Payments", "Logistics Support", "Technical Support", "Returns & Refunds", "Account Security", "General Support"]
LANG_LABEL = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
SLA_ICON = {"on_track": "🟢", "at_risk": "🟠", "breached": "🔴", "met": "✅"}

st.set_page_config(page_title="SupportIQ", page_icon="🎫", layout="wide", initial_sidebar_state="expanded")

# Light, calm styling on top of .streamlit/config.toml. Kept small and readable on purpose.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1300px; }
      h1 { font-weight: 700; letter-spacing: -0.01em; color: #111827; }
      h2 { font-weight: 650; color: #1F2937; margin-top: 1.4rem; }
      h3 { font-weight: 600; color: #374151; }
      /* metric tiles as soft cards */
      [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      }
      [data-testid="stMetricLabel"] { color: #6B7280; font-size: 0.85rem; }
      [data-testid="stMetricValue"] { color: #111827; font-size: 1.6rem; }
      /* tables, inputs, expanders */
      [data-testid="stDataFrame"], [data-testid="stExpander"] {
        border: 1px solid #E5E7EB; border-radius: 12px; background: #FFFFFF;
      }
      [data-testid="stExpander"] summary { font-weight: 500; }
      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background: #FFFFFF;
      }
      /* info/success boxes a little softer */
      .stAlert { border-radius: 10px; }
      /* sidebar */
      section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
      section[data-testid="stSidebar"] .stRadio label { padding: 4px 0; }
      .sidebar-brand { font-size: 1.35rem; font-weight: 700; color: #111827; margin-bottom: 0; }
      .sidebar-sub { color: #6B7280; font-size: 0.85rem; margin-top: 2px; margin-bottom: 14px; }
      .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
              background: #EEF2FF; color: #3730A3; margin-right: 6px; }
      .card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 16px 18px; }
      .muted { color: #6B7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLES = {
    "Late delivery (English)": (
        "Order not delivered",
        "I ordered a product five days ago, but it still hasn't been delivered. I have contacted support twice and received no update. Please resolve this.",
    ),
    "Double charge (English)": (
        "Charged twice",
        "My card was charged Rs 4,999 twice for order #45812 on Monday. Only one order shows in my account. Please refund the duplicate immediately.",
    ),
    "Account locked (Hinglish)": (
        "login nahi ho raha",
        "Bhai mera account lock ho gaya hai, OTP bhi nahi aa raha 4 baar try kiya. Aaj hi order karna hai urgent, please jaldi help karo.",
    ),
    "Wrong size (Hindi)": (
        "गलत साइज़ आया",
        "मुझे मीडियम जैकेट मिली है जबकि मैंने लार्ज ऑर्डर की थी। ऑर्डर #45812, 3 दिन पहले। क्या इसे एक्सचेंज किया जा सकता है?",
    ),
}


def _password() -> str | None:
    """Dashboard password from Streamlit secrets or env. None = no login (local development)."""
    try:
        if "DASHBOARD_PASSWORD" in st.secrets:
            return str(st.secrets["DASHBOARD_PASSWORD"])
    except Exception:
        pass
    return os.getenv("DASHBOARD_PASSWORD") or None


def require_login() -> None:
    """The dashboard is an internal tool. When deployed, gate it behind a shared password."""
    pw = _password()
    if not pw or st.session_state.get("authed"):
        return
    st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div class='sidebar-brand'>🎫 SupportIQ</div><div class='sidebar-sub'>Agent dashboard · staff only</div>", unsafe_allow_html=True)
        with st.form("login"):
            entered = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                if entered == pw:
                    st.session_state["authed"] = True
                    st.rerun()
                st.error("Incorrect password.")
        st.caption("Customers should use the Help Center portal instead.")
    st.stop()


def api(method: str, path: str, retries: int = 3, **kwargs):
    """Call the backend. Retries on connection errors so a sleeping free-tier host can wake up."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.request(method, f"{API}{path}", timeout=180, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError as exc:
            last = exc
            if attempt == 0:
                st.info("Waking up the API server. This can take up to a minute on free hosting.")
            time.sleep(8 * (attempt + 1))
        except requests.RequestException as exc:
            st.error(f"API error: {exc}")
            return None
    st.error(f"Could not reach the API at {API}. ({last})")
    return None


def badge(value: str | None) -> str:
    colors = {"high": "🔴", "medium": "🟠", "low": "🟢", "angry": "🔴", "frustrated": "🟠", "dissatisfied": "🟡", "neutral": "⚪", "positive": "🟢"}
    return f"{colors.get(value or '', '')} {value or '-'}"


def sla_text(s: dict | None) -> str:
    if not s:
        return "-"
    icon = SLA_ICON.get(s["status"], "")
    h = s["hours_remaining"]
    if s["status"] in ("met", "breached") and h is not None and s["status"] == "met":
        return f"{icon} met"
    if s["status"] == "breached":
        return f"{icon} breached by {abs(h):.1f}h"
    return f"{icon} {s['status'].replace('_', ' ')} · {h:.1f}h left"


# ---------------------------------------------------------------- Submit
def page_submit():
    st.header("New ticket")
    st.caption(
        "For tickets that arrive by phone or email: an agent logs them here. Customers use the separate Help Center portal. "
        "Write in English, Hindi or Hinglish. The AI triages it on submit."
    )

    st.markdown("**Try a sample**")
    cols = st.columns(len(SAMPLES))
    for col, (label, (subj, body)) in zip(cols, SAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["draft_subject"], st.session_state["draft_body"] = subj, body

    with st.form("new_ticket"):
        subject = st.text_input("Subject", value=st.session_state.get("draft_subject", ""), placeholder="Order not delivered")
        body = st.text_area("Message", value=st.session_state.get("draft_body", ""), height=150, placeholder="I ordered a product five days ago...")
        email = st.text_input("Email (optional)")
        if st.form_submit_button("Submit ticket", type="primary") and body.strip():
            with st.spinner("Running AI triage (about 10 to 15 seconds)..."):
                t = api("POST", "/tickets", json={"subject": subject or body[:60], "body": body, "customer_email": email or None})
            if t:
                st.success(f"Created {t['ticket_ref']} in {t['ai_latency_ms'] / 1000:.1f} s")
                show_ai_card(t)

    st.divider()
    st.subheader("Bulk upload")
    st.caption("A CSV with columns `subject` and `body`. Tickets appear in the queue immediately and are triaged in the background.")
    up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    if up and st.button("Upload", type="primary"):
        with st.spinner("Uploading..."):
            res = api("POST", "/tickets/upload-csv", files={"file": (up.name, up.getvalue(), "text/csv")})
        if res:
            st.success(f"Created {res['created']} tickets, skipped {res['skipped']}.")
            st.info(
                f"AI triage for {res['ai_pending']} tickets is running in the background, roughly 10 to 15 seconds each. "
                "They appear in the queue immediately and fill in as they finish. Check 'AI pending' on the Analytics page."
            )


def show_ai_card(t: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Category", t["ai_category"], f"{(t['ai_category_confidence'] or 0) * 100:.0f}% conf")
    c2.metric("Priority", t["ai_priority"])
    c3.metric("Sentiment", t["ai_sentiment"])
    c4.metric("Language", LANG_LABEL.get(t.get("ai_language") or "", "-"))
    c5.metric("Team", t["ai_team"])
    st.markdown(f"<div class='card'><div class='muted'>Summary</div>{t['ai_summary']}</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Suggested reply** <span class='muted'>(an agent reviews this before it is sent)</span>", unsafe_allow_html=True)
    st.info(t["ai_suggested_response"] or "No draft: Gemini was unavailable or over quota, so the agent writes the reply.")
    pills = ""
    if t.get("ai_sources"):
        pills += "".join(f"<span class='pill'>📄 {s}</span>" for s in t["ai_sources"].split(","))
    if t.get("ai_groundedness") is not None:
        pills += f"<span class='pill'>grounded {t['ai_groundedness']:.2f}</span>"
    if pills:
        st.markdown(pills, unsafe_allow_html=True)
    st.caption(f"Why this priority: {t['ai_priority_reasons']}")
    st.caption(f"Deadlines · first reply {sla_text(t.get('response_sla'))}  ·  resolution {sla_text(t.get('resolution_sla'))}")


# ---------------------------------------------------------------- Queue
def page_queue():
    st.header("Ticket queue")
    f1, f2, f3, f4, f5 = st.columns(5)
    status = f1.selectbox("Status", ["all"] + STATUSES)
    category = f2.selectbox("Category", ["all"] + CATEGORIES)
    priority = f3.selectbox("Priority", ["all"] + PRIORITIES)
    sla = f4.selectbox("SLA", ["all", "at_risk", "breached"], help="Tickets whose first-response or resolution clock is in this state")
    q = f5.text_input("Search")
    params = {k: v for k, v in {"status": status, "category": category, "priority": priority, "sla": sla, "q": q}.items() if v and v != "all"}
    tickets = api("GET", "/tickets", params=params) or []
    if not tickets:
        st.info("No tickets match.")
        return
    rows = []
    for t in tickets:
        rows.append(
            {
                "id": t["id"], "ref": t["ticket_ref"], "status": t["status"], "priority": t["priority"] or "⏳ pending",
                "category": t["category"] or "⏳ pending", "team": t["team"] or "⏳ pending",
                "sentiment": t["ai_sentiment"] or "⏳", "lang": (t.get("ai_language") or "-"),
                "response SLA": sla_text(t.get("response_sla")), "resolution SLA": sla_text(t.get("resolution_sla")),
                "subject": t["subject"], "created": pd.to_datetime(t["created_at"]).strftime("%Y-%m-%d %H:%M"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    chosen = st.selectbox("Open ticket", df["ref"].tolist())
    if chosen:
        st.session_state["ticket_id"] = int(df.loc[df["ref"] == chosen, "id"].iloc[0])
        page_detail(embedded=True)


# ---------------------------------------------------------------- Detail / review
def page_detail(embedded: bool = False):
    tid = st.session_state.get("ticket_id")
    if not tid:
        tid = st.number_input("Ticket id", min_value=1, step=1)
    t = api("GET", f"/tickets/{tid}")
    if not t:
        return
    (st.subheader if embedded else st.header)(f"{t['ticket_ref']}  ·  {t['subject']}")

    s1, s2, s3 = st.columns(3)
    s1.metric("First-response SLA", sla_text(t.get("response_sla")))
    s2.metric("Resolution SLA", sla_text(t.get("resolution_sla")))
    s3.metric("Language", LANG_LABEL.get(t.get("ai_language") or "", "-"))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Customer message**")
        st.write(t["body"])
        st.markdown("**AI analysis**")
        st.write(
            f"Category: `{t['ai_category']}` ({(t['ai_category_confidence'] or 0) * 100:.0f}%)  ·  "
            f"Priority: {badge(t['ai_priority'])}  ·  Sentiment: {badge(t['ai_sentiment'])}"
        )
        st.write(f"Summary: {t['ai_summary']}")
        st.caption(f"Priority reasons: {t['ai_priority_reasons']}")
        if t.get("ai_sources"):
            st.caption(f"KB sources: {t['ai_sources']}  ·  groundedness {t['ai_groundedness']}  ·  {t['ai_latency_ms']} ms")

        st.markdown("**Similar tickets**")
        sim = api("GET", f"/tickets/{tid}/similar", params={"k": 5}) or []
        if not sim:
            st.caption("None found yet.")
        else:
            resolved = [s for s in sim if s["status"] == "resolved"]
            st.caption(f"{len(sim)} similar, {len(resolved)} already resolved")
            for s in sim:
                label = f"{s['ticket_ref']} · {s['status']} · {s['category'] or '-'} · similarity {s['score']:.2f} · {s['subject'][:60]}"
                with st.expander(label):
                    if s["resolution"]:
                        st.markdown("**How it was resolved**")
                        st.write(s["resolution"])
                    else:
                        st.caption("No agent response recorded yet.")

        if t["feedback"]:
            st.markdown("**Audit trail**")
            st.dataframe(pd.DataFrame(t["feedback"])[["created_at", "field", "ai_value", "human_value", "agent"]], hide_index=True, use_container_width=True)

    with right:
        st.markdown("**Agent review**")
        with st.form(f"review_{tid}"):
            category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(t["category"]) if t["category"] in CATEGORIES else 0)
            priority = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(t["priority"]) if t["priority"] in PRIORITIES else 0)
            team = st.selectbox("Team", TEAMS, index=TEAMS.index(t["team"]) if t["team"] in TEAMS else 0)
            status = st.selectbox("Status", STATUSES, index=STATUSES.index(t["status"]))
            response = st.text_area("Response to customer", value=t["final_response"] or t["ai_suggested_response"] or "", height=220)
            agent = st.text_input("Your name", value="agent")
            c1, c2 = st.columns(2)
            save = c1.form_submit_button("Save corrections")
            approve = c2.form_submit_button("Approve & send", type="primary")
            if save or approve:
                payload = {"category": category, "priority": priority, "team": team, "status": status, "final_response": response, "approve": approve, "agent": agent}
                res = api("PATCH", f"/tickets/{tid}", json=payload)
                if res:
                    st.success("Approved. Response queued for sending." if approve else "Saved.")
                    st.rerun()


# ---------------------------------------------------------------- Analytics
SENTIMENT_ORDER = ["angry", "frustrated", "dissatisfied", "neutral", "positive"]
SLA_LABEL = {"met": "Done on time", "on_track": "On time so far", "at_risk": "Due soon", "breached": "Overdue"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish", "unknown": "unknown"}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def page_analytics():
    st.header("Analytics")
    st.caption("A plain-language view of the support queue and of how well the AI is doing. Each section answers one question.")
    m = api("GET", "/tickets/metrics")
    a = api("GET", "/tickets/analytics", params={"days": 90})
    if not m or not a:
        return

    # 1. At a glance -------------------------------------------------------
    st.subheader("1. Where do things stand right now?")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("All tickets", a["total"], help="Every ticket ever submitted, any status.")
    c2.metric("Waiting for an agent", m["by_status"].get("open", 0), help="Status is open: nobody has approved a reply yet.")
    c3.metric("Overdue", m.get("sla_breached", 0), help="Past the reply or resolution deadline and still not done.")
    c4.metric("Due soon", m.get("sla_at_risk", 0), help="Less than 1 hour left to reply, or less than 8 hours left to resolve.")
    c5.metric("AI still processing", m.get("ai_pending", 0), help="Uploaded tickets the AI has not triaged yet.")

    # 2. AI quality --------------------------------------------------------
    st.subheader("2. Is the AI doing a good job?")
    st.caption(
        "Offline test accuracy is measured on held-out test tickets before deployment. "
        "Accuracy in practice counts a ticket as wrong whenever an agent changed its category. "
        "Drafts sent as-is are approved replies where the agent did not edit the AI text."
    )
    off = a.get("offline_eval") or {}
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Offline test accuracy", _pct(off.get("accuracy")), help=f"From ml/evaluate.py on {off.get('rows', '?')} test tickets.")
    q2.metric("Accuracy in practice", _pct(a["live_category_accuracy"]), help=f"Based on {a['triaged']} triaged tickets and agent corrections.")
    q3.metric(
        "Drafts sent as-is",
        f"{a['approved_unedited']} of {a['approved']}",
        _pct(a["acceptance_rate"]) if a["acceptance_rate"] is not None else None,
        help="Of all approved replies, how many went out exactly as the AI wrote them.",
    )
    q4.metric("Avg confidence", f"{(a['avg_classifier_confidence'] or 0):.2f}", help="The classifier's own certainty from 0 to 1. Below 0.5 deserves a human look.")
    if off.get("macro_f1_by_source"):
        names = {"synthetic": "English", "synthetic_hi": "Hindi", "synthetic_hinglish": "Hinglish"}
        st.caption("Offline score by language (macro F1): " + " · ".join(f"{names.get(k, k)} {v:.2f}" for k, v in off["macro_f1_by_source"].items()))
    if a["correction_rate"]:
        st.markdown("**What agents change most often**")
        st.caption("Share of tickets where an agent corrected each AI output. Lower is better. 'response' means the draft reply was edited.")
        st.bar_chart(pd.Series(a["correction_rate"]).rename("share of tickets corrected"))
    else:
        st.info("No corrections yet. Open a ticket, change its category or edit the reply, press Save, and this section fills in.")

    # 3. What customers write about ---------------------------------------
    st.subheader("3. What are customers contacting us about?")
    x, y, z = st.columns(3)
    x.markdown("**By issue type**")
    x.caption("What the ticket is about.")
    x.bar_chart(pd.Series(a["by_category"]))
    y.markdown("**By urgency**")
    y.caption("Priority set by the rules, as corrected by agents.")
    y.bar_chart(pd.Series(a["by_priority"]).reindex(["high", "medium", "low"]).fillna(0))
    z.markdown("**By mood**")
    z.caption("Sentiment detected in the customer's message.")
    z.bar_chart(pd.Series(a["by_sentiment"]).reindex(SENTIMENT_ORDER).dropna())
    if a["by_language"]:
        st.caption("Languages: " + " · ".join(f"{LANG_NAMES.get(k, k)} {v}" for k, v in a["by_language"].items()))

    # 4. Timeliness --------------------------------------------------------
    st.subheader("4. Are we responding on time?")
    st.caption(
        "Every ticket has two deadlines from the support policy: a first reply within 2 / 8 / 24 business hours "
        "for high / medium / low priority, and a resolution within 1 / 3 / 5 business days. "
        "Business hours are 9 am to 9 pm IST, Monday to Saturday."
    )
    sla_df = pd.DataFrame(a["sla"]).reindex(["met", "on_track", "at_risk", "breached"]).fillna(0).astype(int)
    sla_df.index = [SLA_LABEL.get(i, i) for i in sla_df.index]
    sla_df.columns = [{"response": "First reply", "resolution": "Resolution"}.get(c, c) for c in sla_df.columns]
    t1, t2 = st.columns(2)
    t1.markdown("**Tickets by deadline status**")
    t1.dataframe(sla_df, use_container_width=True)
    with t2:
        if a["avg_first_response_hours"] is not None:
            st.metric("Average time to first reply", f"{a['avg_first_response_hours']:.1f} h", help="From ticket creation to an agent pressing Approve.")
        else:
            st.caption("Average time to first reply appears after the first approved reply.")
        if a["avg_resolution_hours_by_priority"]:
            st.markdown("**Average hours to resolve, by priority**")
            st.bar_chart(pd.Series(a["avg_resolution_hours_by_priority"]).reindex(["high", "medium", "low"]).dropna())
        else:
            st.caption("Average time to resolve appears once tickets are marked resolved.")

    # 5. Needs attention ---------------------------------------------------
    st.subheader("5. Which tickets need a human look first?")
    st.caption("Open tickets that are overdue, due soon, low-confidence, or missing an AI draft. Most reasons first.")
    if a["needs_review"]:
        rows = [
            {"ticket": n["ticket_ref"], "priority": n["priority"] or "-", "subject": n["subject"][:60], "why": "; ".join(n["reasons"])}
            for n in a["needs_review"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.success("Nothing flagged. Every open ticket is on time, confidently classified and has a draft.")

    # 6. Common complaints -------------------------------------------------
    st.subheader("6. What are the most common complaints?")
    st.caption("Frequent phrases in the AI summaries, grouped by issue type. Needs at least 3 triaged tickets per type.")
    if a["top_complaints"]:
        cols = st.columns(min(len(a["top_complaints"]), 5))
        for col, (cat, terms) in zip(cols, a["top_complaints"].items()):
            col.markdown(f"**{cat}**")
            col.dataframe(pd.DataFrame(terms).rename(columns={"term": "phrase", "count": "tickets"}), hide_index=True, use_container_width=True)
    else:
        st.caption("Not enough tickets yet.")

    # 7. Volume over time --------------------------------------------------
    st.subheader("7. How is ticket volume changing?")
    if len(a["volume_by_day"]) > 1:
        st.line_chart(pd.DataFrame(a["volume_by_day"]).set_index("date").rename(columns={"count": "tickets"}))
    else:
        n = a["volume_by_day"][0]["count"] if a["volume_by_day"] else 0
        st.caption(f"All {n} tickets were created on one day, so there is no trend yet. This becomes a line chart from the second day.")


# ---------------------------------------------------------------- Nav
PAGES = {
    "✉️  New ticket": page_submit,
    "📋  Ticket queue": page_queue,
    "🔎  Ticket detail": page_detail,
    "📊  Analytics": page_analytics,
}
require_login()

with st.sidebar:
    st.markdown("<div class='sidebar-brand'>🎫 SupportIQ</div><div class='sidebar-sub'>AI-assisted ticket triage</div>", unsafe_allow_html=True)
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    health = api("GET", "/health")
    st.markdown(
        f"<span class='pill'>{'● API online' if health else '○ API offline'}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='muted' style='margin-top:14px'>How it works<br>"
        "1. Customer raises a ticket in the Help Center portal<br>"
        "2. AI classifies, prioritises, summarises and drafts a reply from policy<br>"
        "3. An agent reviews, corrects and approves<br>"
        "4. Corrections feed the quality metrics</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='muted' style='margin-top:14px'>API: {API}</div>", unsafe_allow_html=True)
PAGES[choice]()
