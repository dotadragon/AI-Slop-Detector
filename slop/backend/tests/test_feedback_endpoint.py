"""Tests for the /ops/feedback review-queue endpoints (Tier 6)."""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_session

JOB_ID = str(uuid.uuid4())


@pytest.fixture()
def app_client():
    import app.db.base  # noqa: F401 — register all models on the metadata
    from app.main import app

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
    )
    TestingSession = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    import asyncio

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with TestingSession() as s:
            from app.models.job import DetectionResult, Job

            j = Job(id=uuid.UUID(JOB_ID), job_type="text", status="completed")
            s.add(j)
            s.add(DetectionResult(job_id=j.id, overall_score=0.42, signals={}))
            await s.commit()

    asyncio.run(_setup())

    async def override_get_session():
        async with TestingSession() as s:
            yield s

    app.dependency_overrides[get_session] = override_get_session
    yield app
    app.dependency_overrides.clear()


def _auth(app):
    from app.api.deps import current_user
    from app.models.user import User

    app.dependency_overrides[current_user] = lambda: User(
        id=uuid.uuid4(), email="t@t.co", hashed_password="x", is_active=True
    )


def _csrf_headers(app):
    """Satisfy CsrfMiddleware: set the cookie and matching header."""
    import secrets

    tok = secrets.token_urlsafe(32)
    return {"cookies": {"csrf_token": tok}, "headers": {"X-CSRF-Token": tok}}


def _client(app):
    return TestClient(app, **_csrf_headers(app))


def test_feedback_happy_path(app_client):
    _auth(app_client)
    with _client(app_client) as c:
        r = c.post("/api/v1/ops/feedback", json={
            "job_id": JOB_ID, "verdict_type": "false_positive", "comment": "my own writing"
        })
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "received"


def test_feedback_rejects_invalid_verdict_type(app_client):
    _auth(app_client)
    with _client(app_client) as c:
        r = c.post("/api/v1/ops/feedback", json={"job_id": JOB_ID, "verdict_type": "nonsense"})
    assert r.status_code == 422


def test_feedback_unknown_job_404(app_client):
    _auth(app_client)
    with _client(app_client) as c:
        r = c.post("/api/v1/ops/feedback", json={
            "job_id": str(uuid.uuid4()), "verdict_type": "false_negative"
        })
    assert r.status_code == 404


def test_feedback_review_queue_lists_open_items(app_client):
    _auth(app_client)
    with _client(app_client) as c:
        c.post("/api/v1/ops/feedback", json={"job_id": JOB_ID, "verdict_type": "false_positive"})
        r = c.get("/api/v1/ops/feedback?status=open")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    assert body["items"][0]["verdict_type"] == "false_positive"
    assert body["items"][0]["context"]["modality"] == "text"


def test_calibration_dashboard_endpoint(app_client):
    with _client(app_client) as c:
        r = c.get("/api/v1/ops/calibration-dashboard")
    assert r.status_code == 200
    assert "SLOP Detection" in r.text

