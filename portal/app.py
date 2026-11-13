"""Customer portal (Streamlit). What a customer sees: raise a ticket, check its status.

Run:  streamlit run portal/app.py --server.port 8502     (set SUPPORTIQ_API if the API is not on :8000)
Talks only to the /public endpoints. Never shows AI internals, priority or agent notes.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("SUPPORTIQ_API", "http://127.0.0.1:8000")
STORE = "Lumen Store"

st.set_page_config(page_title=f"{STORE} Help Center", page_icon="💬", layout="centered")
st.markdown(
    """
    <style>
      .block-container { max-width: 760px; padding-top: 2rem; }
      h1 { font-weight: 700; color: #111827; }
      .hero { color: #6B7280; margin-bottom: 1.2rem; }
      .ticket-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }
      .ref { font-size: 1.4rem; font-weight: 700; color: #4F6BED; }
      .muted { color: #6B7280; font-size: 0.88rem; }
      .status { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
      .s-open { background: #FEF3C7; color: #92400E; }
      .s-pending { background: #DBEAFE; color: #1E40AF; }
      .s-resolved { background: #D1FAE5; color: #065F46; }
      .reply { background: #F5F7FF; border-left: 4px solid #4F6BED; padding: 12px 14px; border-radius: 8px; margin-top: 10px; white-space: pre-wrap; }
      section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api(method: str, path: str, retries: int = 3, **kwargs):
    """Call the backend. Retries on connection errors so a sleeping free-tier host can wake up."""
    for attempt in range(retries):
        try:
            r = requests.request(method, f"{API}{path}", timeout=90, **kwargs)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            if attempt == 0:
                st.info("Connecting to the help center. This can take up to a minute the first time.")
            time.sleep(8 * (attempt + 1))
        except requests.RequestException as exc:
            st.error(f"Sorry, the help center is temporarily unavailable. ({exc})")
            return None
    st.error("Sorry, the help center is not responding right now. Please try again in a minute.")
    return None


def fmt(ts: str | None) -> str:
    if not ts:
        return "-"
    t = pd.to_datetime(ts)
    if t.tzinfo is None:  # the API stores UTC; SQLite drops the zone marker
        t = t.tz_localize("UTC")
    return t.tz_convert("Asia/Kolkata").strftime("%d %b %Y, %I:%M %p IST")


def ticket_card(t: dict):
    cls = {"open": "s-open", "pending": "s-pending", "resolved": "s-resolved"}.get(t["status"], "s-open")
    html = (
        f"<div class='ticket-card'>"
        f"<span class='ref'>{t['ticket_ref']}</span> &nbsp; <span class='status {cls}'>{t['status_label']}</span>"
        f"<div style='margin-top:8px'><b>{t['subject']}</b></div>"
        f"<div class='muted'>Raised {fmt(t['created_at'])}</div>"
    )
    if t["agent_reply"]:
        html += f"<div class='muted' style='margin-top:10px'>Reply from our team · {fmt(t['replied_at'])}</div><div class='reply'>{t['agent_reply']}</div>"
    elif t["expected_reply_by"]:
        html += f"<div class='muted' style='margin-top:10px'>Expected first reply by <b>{fmt(t['expected_reply_by'])}</b></div>"
    if t["resolved_at"]:
        html += f"<div class='muted' style='margin-top:6px'>Resolved {fmt(t['resolved_at'])}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------- pages
def page_raise():
    st.title(f"{STORE} Help Center")
    st.markdown("<div class='hero'>Tell us what went wrong. You will get a ticket number straight away and a reply from our team within the promised time.</div>", unsafe_allow_html=True)
    with st.form("raise"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Your name")
        email = c2.text_input("Email")
        subject = st.text_input("Subject", placeholder="e.g. Order #45812 not delivered")
        body = st.text_area("Describe the issue", height=160, placeholder="You can write in English, Hindi or Hinglish.")
        sent = st.form_submit_button("Submit ticket", type="primary", use_container_width=True)
    if sent:
        if not (name and email and subject and len(body.strip()) >= 10):
            st.warning("Please fill in all fields. The description needs at least 10 characters.")
            return
        r = None
        for attempt in range(3):
            try:
                r = requests.post(f"{API}/public/tickets", json={"name": name, "email": email, "subject": subject, "body": body}, timeout=90)
                break
            except requests.ConnectionError:
                if attempt == 0:
                    st.info("Connecting to the help center. This can take up to a minute the first time.")
                time.sleep(8 * (attempt + 1))
            except requests.RequestException as exc:
                st.error(f"Could not submit right now. ({exc})")
                return
        if r is None:
            st.error("Sorry, the help center is not responding right now. Please try again in a minute.")
            return
        if r.status_code == 422:
            st.warning("Please check the email address and try again.")
            return
        if not r.ok:
            st.error(f"Something went wrong ({r.status_code}).")
            return
        t = r.json()
        st.success("Your ticket has been raised.")
        ticket_card(t)
        st.caption("Keep the ticket number. You can check progress any time under *Check status* using this email.")
        st.session_state["last_email"] = email


def page_status():
    st.title("Check ticket status")
    st.markdown("<div class='hero'>Enter the email you used when raising the ticket. Add a ticket number to look up one ticket, or leave it blank to see all of them.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    email = c1.text_input("Email", value=st.session_state.get("last_email", ""))
    ref = c2.text_input("Ticket number (optional)", placeholder="TKT-1061")
    if st.button("Look up", type="primary") and email:
        if ref.strip():
            t = api("GET", f"/public/tickets/{ref.strip()}", params={"email": email})
            if t is None:
                st.warning("No ticket found for that number and email.")
            else:
                ticket_card(t)
        else:
            items = api("GET", "/public/tickets", params={"email": email}) or []
            if not items:
                st.info("No tickets found for this email.")
            for t in items:
                ticket_card(t)


with st.sidebar:
    st.markdown(f"### 💬 {STORE}")
    page = st.radio("Help Center", ["Raise a ticket", "Check status"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<div class='muted'>Support hours<br>9 am to 9 pm IST, Monday to Saturday<br><br>"
        "Typical first reply<br>Urgent issues: 2 business hours<br>Most issues: 8 business hours</div>",
        unsafe_allow_html=True,
    )

page_raise() if page == "Raise a ticket" else page_status()
