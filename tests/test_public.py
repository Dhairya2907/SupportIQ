"""Customer portal endpoints. Offline environment from conftest.py."""

NEW = {"name": "Priya Sharma", "email": "Priya@Example.com", "subject": "Order #45812 not delivered", "body": "Ordered a week ago, nothing yet and no tracking update."}


def test_raise_ticket_returns_only_customer_fields(client):
    r = client.post("/public/tickets", json=NEW)
    assert r.status_code == 201, r.text
    t = r.json()
    assert t["ticket_ref"].startswith("TKT-")
    assert t["status"] == "open"
    assert t["expected_reply_by"] is not None
    assert t["agent_reply"] is None
    for internal in ("ai_category", "priority", "team", "ai_priority_reasons", "feedback"):
        assert internal not in t


def test_status_requires_matching_email(client):
    ref = client.post("/public/tickets", json=NEW).json()["ticket_ref"]
    ok = client.get(f"/public/tickets/{ref}", params={"email": "priya@example.com"})  # case-insensitive
    assert ok.status_code == 200
    bad = client.get(f"/public/tickets/{ref}", params={"email": "someone@else.com"})
    assert bad.status_code == 404
    # bare number works too
    assert client.get(f"/public/tickets/{ref.split('-')[1]}", params={"email": NEW['email']}).status_code == 200


def test_reply_visible_only_after_agent_approves(client):
    t = client.post("/public/tickets", json=NEW).json()
    tid = int(t["ticket_ref"].split("-")[1]) - 1000
    # Agent drafts but does not approve: customer sees nothing.
    client.patch(f"/tickets/{tid}", json={"final_response": "Draft, not sent yet."})
    assert client.get(f"/public/tickets/{t['ticket_ref']}", params={"email": NEW["email"]}).json()["agent_reply"] is None
    # Agent approves: customer sees the reply, status moves to pending.
    client.patch(f"/tickets/{tid}", json={"final_response": "We have located your parcel; delivery tomorrow.", "approve": True})
    pub = client.get(f"/public/tickets/{t['ticket_ref']}", params={"email": NEW["email"]}).json()
    assert pub["agent_reply"].startswith("We have located")
    assert pub["status"] == "pending"
    assert pub["expected_reply_by"] is None


def test_my_tickets_lists_only_that_email(client):
    client.post("/public/tickets", json=NEW)
    client.post("/public/tickets", json={**NEW, "email": "other@example.com", "subject": "Other person"})
    mine = client.get("/public/tickets", params={"email": NEW["email"]}).json()
    assert len(mine) == 1 and mine[0]["subject"] == NEW["subject"]


def test_portal_ticket_shows_in_agent_queue_with_name(client):
    t = client.post("/public/tickets", json=NEW).json()
    tid = int(t["ticket_ref"].split("-")[1]) - 1000
    agent_view = client.get(f"/tickets/{tid}").json()
    assert agent_view["channel"] == "portal"
    assert agent_view["ai_category"] is not None  # background triage ran (TestClient runs it inline)
