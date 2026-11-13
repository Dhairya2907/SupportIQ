"""API key protection on the internal endpoints (public ones stay open)."""
from app.config import settings

SAMPLE = {"subject": "Late order", "body": "My parcel is 8 days late and tracking is stuck."}
PUBLIC = {"name": "A", "email": "a@example.com", "subject": "Late", "body": "My parcel is 8 days late and tracking is stuck."}


def test_no_key_configured_means_open(client):
    assert client.get("/tickets/metrics").status_code == 200


def test_key_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "s3cret")
    assert client.get("/tickets/metrics").status_code == 401
    assert client.get("/tickets/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/tickets/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.post("/tickets", json=SAMPLE, headers={"X-API-Key": "s3cret"}).status_code == 201
    # customers never need the key
    assert client.post("/public/tickets", json=PUBLIC).status_code == 201
    assert client.get("/health").status_code == 200
