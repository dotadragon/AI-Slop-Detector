from fastapi.testclient import TestClient

from app.main import app


def test_security_headers_present():
    client = TestClient(app)
    response = client.get("/health")
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in response.headers


def test_csrf_required_for_mutating_api_routes():
    client = TestClient(app)
    response = client.post("/api/v1/text/analyze", json={"text": "hello"})
    assert response.status_code == 403


def test_disallowed_method_returns_405():
    client = TestClient(app)
    response = client.put("/api/v1/text/analyze", json={"text": "hello"})
    assert response.status_code == 405
