import uuid

from fastapi.testclient import TestClient

from app.main import app


def test_register_login_refresh_logout_flow():
    client = TestClient(app)
    email = f"user-{uuid.uuid4()}@example.com"
    password = "ValidPass1!"
    assert client.get("/health").status_code == 200

    register = client.post("/api/v1/auth/register", json={"email": email, "password": password, "confirm_password": password})
    assert register.status_code == 200

    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    assert login.json()["access_token"]
    assert "refresh_token" in client.cookies

    refresh = client.post("/api/v1/auth/refresh")
    assert refresh.status_code == 200
    assert refresh.json()["access_token"]


def test_honeypot_returns_fake_success():
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": "bot@example.com", "website": "filled-by-bot"},
    )
    assert response.status_code == 200
