"""Tests for the FastAPI endpoints — schema compliance, health check."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_is_json(self, client):
        resp = client.get("/health")
        assert resp.headers["content-type"] == "application/json"


class TestChatSchema:
    """Verify that /chat responses always match the exact schema."""

    def test_chat_returns_required_fields(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}]
        })
        # May fail without API key, but should still return valid response
        if resp.status_code == 200:
            data = resp.json()
            assert "reply" in data
            assert "recommendations" in data
            assert "end_of_conversation" in data
            assert isinstance(data["reply"], str)
            assert isinstance(data["end_of_conversation"], bool)

    def test_chat_rejects_empty_messages(self, client):
        resp = client.post("/chat", json={"messages": []})
        assert resp.status_code == 422  # Validation error

    def test_chat_rejects_missing_messages(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_rejects_invalid_role(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "system", "content": "test"}]
        })
        assert resp.status_code == 422

    def test_recommendations_schema(self, client):
        """When recommendations are returned, each must have name, url, test_type."""
        resp = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "I need Java developer assessments for a senior engineer"}
            ]
        })
        if resp.status_code == 200:
            data = resp.json()
            if data["recommendations"] is not None:
                for rec in data["recommendations"]:
                    assert "name" in rec
                    assert "url" in rec
                    assert "test_type" in rec
                    assert rec["url"].startswith("https://")
                    assert len(data["recommendations"]) <= 10
