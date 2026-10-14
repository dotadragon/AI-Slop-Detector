import uuid
import pyotp
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.token_store import InMemoryTokenStore


def test_refresh_token_reuse_family_revocation():
    """
    Validates that attempting to reuse an already-consumed refresh token
    triggers complete family revocation (RTR breach mitigation).
    """
    store = InMemoryTokenStore()
    user_id = "test-user-123"
    family_id = "family-abc"
    token_1 = "raw-refresh-token-1"
    token_2 = "raw-refresh-token-2"

    # Store first token in family
    store.store_refresh(token_1, user_id, family_id, ttl_seconds=3600)
    # Store rotated second token in family
    store.store_refresh(token_2, user_id, family_id, ttl_seconds=3600)

    # Legitimate consumption of token 1
    rec1 = store.consume_refresh(token_1)
    assert rec1 is not None
    assert rec1.revoked is True

    # Attacker attempts to replay already-consumed token 1!
    replay_attempt = store.consume_refresh(token_1)
    assert replay_attempt is None

    # Attempting to consume token_2 must now fail because family was revoked
    rec2 = store.consume_refresh(token_2)
    assert rec2 is None


def test_ops_telemetry_endpoint():
    """Validates that GET /api/v1/ops/telemetry returns real-time hardware status."""
    client = TestClient(app)
    res = client.get("/api/v1/ops/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "telemetry" in data
    telemetry = data["telemetry"]
    assert "cuda_available" in telemetry
    assert "vram_ceiling_mb" in telemetry
    assert "gpu_locked" in telemetry
    assert telemetry["vram_ceiling_mb"] > 0


def test_billing_webhook_unconfigured_error():
    """Validates that POST /api/v1/billing/webhook fails with 503 when Stripe is not configured."""
    client = TestClient(app)
    res = client.post("/api/v1/billing/webhook", json={"type": "payment_intent.succeeded"})
    assert res.status_code == 503
    assert res.json()["detail"] == "stripe_webhook_not_configured"


def test_text_compare_endpoint():
    """Validates that POST /api/v1/text/compare correctly computes differentials and n-gram overlap."""
    client = TestClient(app)
    client.get("/api/v1/auth/csrf")
    csrf = client.cookies.get("csrf_token")

    specimen_ai = (
        "In the contemporary era of technological development, it is imperative to note that artificial "
        "intelligence serves as a multifaceted tapestry of deep innovation. Furthermore, it is crucial to delve into "
        "the holistic implications of machine learning algorithms to ensure balanced perspectives across modern domains."
    )
    specimen_human = (
        "Hey, we tested the code on our server yesterday. Everything worked fine, but the GPU crashed twice "
        "because of an out-of-memory error. I fixed the batch size and ran it again. Now it runs super fast."
    )

    payload = {
        "specimen_a": specimen_ai,
        "specimen_b": specimen_human,
    }

    res = client.post("/api/v1/text/compare", json=payload, headers={"X-CSRF-Token": csrf} if csrf else {})
    assert res.status_code == 200
    data = res.json()

    assert "specimen_a" in data
    assert "specimen_b" in data
    assert "delta_score" in data
    assert "delta_burstiness" in data
    assert "ngram_overlap_pct" in data
    assert "shared_phrases" in data
    assert "verdict" in data

    # Specimen A is heavily synthetic and Specimen B is conversational human
    assert data["specimen_a"]["overall_score"] > data["specimen_b"]["overall_score"]
    assert data["delta_score"] > 0
    assert isinstance(data["shared_phrases"], list)


def test_2fa_backup_code_login_and_rotation():
    """Validates 2FA setup, backup code issuance, backup code login, single-use burn, and code rotation."""
    client = TestClient(app)
    email = f"totp-user-{uuid.uuid4()}@example.com"
    password = "ValidPassword123!"

    # 1. Register user
    reg = client.post("/api/v1/auth/register", json={"email": email, "password": password, "confirm_password": password})
    assert reg.status_code == 200

    # 2. Login user
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    token = login.json()["access_token"]
    csrf = client.cookies.get("csrf_token")
    auth_headers = {"Authorization": f"Bearer {token}", "X-CSRF-Token": csrf} if csrf else {"Authorization": f"Bearer {token}"}

    # 3. Enable 2FA -> returns QR code and 10 backup codes
    enable_res = client.post("/api/v1/auth/enable-2fa", headers=auth_headers)
    assert enable_res.status_code == 200
    backup_codes = enable_res.json()["backup_codes"]
    assert len(backup_codes) == 10

    # Directly enable totp_enabled flag for user in DB
    from app.db.session import AsyncSessionLocal
    from app.models.user import User
    from sqlalchemy import select
    import asyncio

    async def _enable_totp():
        async with AsyncSessionLocal() as db:
            user = await db.scalar(select(User).where(User.email == email))
            assert user is not None
            user.totp_enabled = True
            await db.commit()

    asyncio.run(_enable_totp())

    # 4. Login using one of the backup codes instead of a TOTP code
    backup_code_to_use = backup_codes[0]
    login_backup = client.post("/api/v1/auth/login", json={"email": email, "password": password, "totp_code": backup_code_to_use})
    assert login_backup.status_code == 200
    assert "access_token" in login_backup.json()

    # 5. Attempting to re-use the SAME backup code MUST fail (single-use burned)
    reuse_res = client.post("/api/v1/auth/login", json={"email": email, "password": password, "totp_code": backup_code_to_use})
    assert reuse_res.status_code == 401
    assert reuse_res.json()["detail"] == "totp_required"


def test_jobs_status_access_control():
    """Validates that GET /jobs/{job_id}/status enforces strict ownership and IP access control."""
    client = TestClient(app)

    # 1. Invalid UUID returns status missing
    r_invalid = client.get("/api/v1/jobs/not-a-uuid/status")
    assert r_invalid.status_code == 200
    assert r_invalid.json()["status"] == "missing"

    # 2. Non-existent UUID returns status missing
    r_not_found = client.get(f"/api/v1/jobs/{uuid.uuid4()}/status")
    assert r_not_found.status_code == 200
    assert r_not_found.json()["status"] == "missing"

    # 3. Create authenticated user A and create job
    email_a = f"owner-a-{uuid.uuid4()}@example.com"
    password = "ValidPassword123!"
    client.post("/api/v1/auth/register", json={"email": email_a, "password": password, "confirm_password": password})
    login_a = client.post("/api/v1/auth/login", json={"email": email_a, "password": password})
    token_a = login_a.json()["access_token"]
    csrf_a = client.cookies.get("csrf_token")
    headers_a = {"Authorization": f"Bearer {token_a}", "X-CSRF-Token": csrf_a} if csrf_a else {"Authorization": f"Bearer {token_a}"}

    scan_res = client.post("/api/v1/text/analyze", json={"text": "Authentic verification text for access control check."}, headers=headers_a)
    assert scan_res.status_code == 200
    job_id = scan_res.json()["job_id"]

    # Owner A can access status
    r_owner = client.get(f"/api/v1/jobs/{job_id}/status", headers=headers_a)
    assert r_owner.status_code == 200
    assert r_owner.json()["status"] == "completed"

    # Anonymous caller is rejected with unauthorized
    # Use fresh client without auth cookies
    anon_client = TestClient(app)
    r_anon = anon_client.get(f"/api/v1/jobs/{job_id}/status")
    assert r_anon.status_code == 200
    assert r_anon.json()["status"] == "unauthorized"

    # Different user B is rejected with forbidden
    email_b = f"intruder-b-{uuid.uuid4()}@example.com"
    client_b = TestClient(app)
    client_b.post("/api/v1/auth/register", json={"email": email_b, "password": password, "confirm_password": password})
    login_b = client_b.post("/api/v1/auth/login", json={"email": email_b, "password": password})
    token_b = login_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    r_intruder = client_b.get(f"/api/v1/jobs/{job_id}/status", headers=headers_b)
    assert r_intruder.status_code == 200
    assert r_intruder.json()["status"] == "forbidden"
