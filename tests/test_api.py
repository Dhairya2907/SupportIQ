"""API tests. See conftest.py for the offline environment."""

SAMPLE = {
    "subject": "Order not delivered",
    "body": "I ordered a product five days ago, but it still hasn't been delivered. I have contacted support twice and received no update. Please resolve this.",
}


def test_create_and_fetch(client):
    r = client.post("/tickets", json=SAMPLE)
    assert r.status_code == 201, r.text
    t = r.json()
    assert t["ticket_ref"].startswith("TKT-")
    assert t["ai_category"] in ["billing", "delivery", "technical", "returns", "account"]
    assert t["ai_priority"] in ["low", "medium", "high"]
    assert t["team"] == t["ai_team"]
    assert t["status"] == "open"

    r = client.get(f"/tickets/{t['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == t["id"]


def test_filters(client):
    client.post("/tickets", json=SAMPLE)
    assert len(client.get("/tickets", params={"status": "open"}).json()) == 1
    assert len(client.get("/tickets", params={"status": "resolved"}).json()) == 0


def test_correction_is_audited(client):
    t = client.post("/tickets", json=SAMPLE).json()
    new_cat = "billing" if t["ai_category"] != "billing" else "returns"
    r = client.patch(
        f"/tickets/{t['id']}",
        json={"category": new_cat, "final_response": "Hello, we are on it.", "approve": True},
    )
    assert r.status_code == 200, r.text
    t2 = r.json()
    assert t2["category"] == new_cat
    assert t2["ai_category"] == t["ai_category"]  # AI value preserved for the audit trail
    assert t2["approved_at"] is not None
    assert t2["status"] == "pending"
    fields = {f["field"] for f in t2["feedback"]}
    assert "category" in fields and "response" in fields

    m = client.get("/tickets/metrics").json()
    assert m["total"] == 1
    assert m["correction_rate"]["category"] == 1.0


def test_invalid_category_rejected(client):
    t = client.post("/tickets", json=SAMPLE).json()
    assert client.patch(f"/tickets/{t['id']}", json={"category": "nonsense"}).status_code == 422


def test_sla_clocks_and_approval(client):
    t = client.post("/tickets", json=SAMPLE).json()
    assert t["response_due_at"] and t["resolve_due_at"]
    assert t["response_sla"]["status"] in ("on_track", "at_risk", "breached")
    assert t["resolution_sla"]["status"] in ("on_track", "at_risk", "breached")

    t2 = client.patch(f"/tickets/{t['id']}", json={"approve": True}).json()
    assert t2["first_response_at"] is not None
    assert t2["response_sla"]["status"] == "met"

    t3 = client.patch(f"/tickets/{t['id']}", json={"status": "resolved"}).json()
    assert t3["resolved_at"] is not None
    assert t3["resolution_sla"]["status"] == "met"

    # Changing priority recomputes the deadlines.
    new_p = "low" if t3["priority"] != "low" else "high"
    t4 = client.patch(f"/tickets/{t['id']}", json={"priority": new_p}).json()
    assert t4["response_due_at"] != t3["response_due_at"]


def test_sla_filter_and_metrics(client):
    client.post("/tickets", json=SAMPLE)
    assert client.get("/tickets", params={"sla": "breached"}).status_code == 200
    m = client.get("/tickets/metrics").json()
    assert "sla_at_risk" in m and "sla_breached" in m


def test_similar_endpoint_without_rag(client):
    t = client.post("/tickets", json=SAMPLE).json()
    r = client.get(f"/tickets/{t['id']}/similar")
    assert r.status_code == 200
    assert r.json() == []  # RAG disabled in tests
    assert client.get("/tickets/9999/similar").status_code == 404


def test_analytics(client):
    client.post("/tickets", json=SAMPLE)
    client.post("/tickets", json={"subject": "OTP", "body": "OTP nahi aa raha hai, login nahi ho pa raha, 3 baar try kiya"})
    r = client.get("/tickets/analytics", params={"days": 7})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["days"] == 7
    assert sum(d["count"] for d in a["volume_by_day"]) == 2
    assert "response" in a["sla"] and "resolution" in a["sla"]
    assert a["by_language"].get("hinglish") == 1  # script-based fallback when Gemini is off
    assert "correction_rate" in a


def test_language_guess_hindi(client):
    t = client.post("/tickets", json={"subject": "ऑर्डर", "body": "मेरा ऑर्डर अभी तक नहीं आया है, कृपया मदद करें"}).json()
    assert t["ai_language"] == "hi"


def test_csv_upload(client):
    csv = (
        "subject,body\n"
        "Late order,My parcel is 8 days late and tracking is stuck.\n"
        "Login,Cannot log in password reset never arrives\n"
        ",\n"
    )
    r = client.post("/tickets/upload-csv", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["created"] == 2
    assert r.json()["skipped"] == 1
