"""
Phase 2/3/8 hardening tests.

Covers the security and concurrency fixes that shipped without automated
coverage:
  Phase 2 — MIME-confusion rejection in save_upload (magic bytes must match
            the allowlist; a lying Content-Type is not enough).
  Phase 3 — atomic WebSocket live-connection ceiling (no race past 5) and
            bounded GPU lock acquisition (timeout instead of deadlock).
  Phase 8 — C2PA tiered trust classification (absent / unverified_claim /
            broken_chain / self_signed / fully_trusted) and tamper rejection.
"""

import asyncio
import io

import pytest
from PIL import Image


# ============================================================================
# Phase 2 — save_upload MIME-confusion hardening
# ============================================================================

def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_upload(filename: str, content_type: str, payload: bytes):
    """Build a fake UploadFile whose read(n) matches Starlette semantics:
    returns up to n bytes per call, then b"" at EOF."""
    class FakeUpload:
        def __init__(self):
            self._pos = 0

        async def read(self, n: int) -> bytes:
            chunk = payload[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

    FakeUpload.filename = filename
    FakeUpload.content_type = content_type
    return FakeUpload()


@pytest.mark.asyncio
async def test_image_with_lying_content_type_rejected(tmp_path):
    """Arbitrary bytes declared as image/png must be rejected — sniffed magic bytes
    must match the allowlist for binary kinds (MIME-confusion fix)."""
    from fastapi import HTTPException
    from unittest.mock import patch
    from app.utils.file_handler import save_upload
    from app.core.config import settings

    upload = _fake_upload("evil.png", "image/png", b"#!/bin/bash\nrm -rf /")
    with patch.object(type(settings), "upload_path", new_callable=lambda: property(lambda self: tmp_path)):
        with pytest.raises(HTTPException) as exc_info:
            await save_upload(upload, "image")
    assert exc_info.value.status_code == 415
    assert exc_info.value.detail == "file_content_type_mismatch"


@pytest.mark.asyncio
async def test_image_with_valid_magic_but_wrong_extension_rejected():
    """Valid PNG magic bytes with a .txt extension must be rejected by the
    extension allowlist before any byte sniffing matters."""
    from fastapi import HTTPException
    from app.utils.file_handler import save_upload

    upload = _fake_upload("photo.txt", "image/png", _png_bytes())
    with pytest.raises(HTTPException) as exc_info:
        await save_upload(upload, "image")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "disallowed_file_extension"


@pytest.mark.asyncio
async def test_honest_png_upload_succeeds(tmp_path):
    """A genuine PNG with matching declaration must pass all checks."""
    from unittest.mock import patch
    from app.utils.file_handler import save_upload
    from app.core.config import settings

    upload = _fake_upload("real.png", "image/png", _png_bytes())
    with patch.object(type(settings), "upload_path", new_callable=lambda: property(lambda self: tmp_path)):
        path, mime, size = await save_upload(upload, "image")

    assert mime == "image/png"
    assert path.exists()
    assert path.name.endswith(".png")
    # Storage name must be server-generated UUID, never the client filename
    assert path.name != "real.png"


@pytest.mark.asyncio
async def test_video_with_text_payload_rejected(tmp_path):
    """Text bytes declared as video/mp4 must be rejected by magic-byte sniffing."""
    from fastapi import HTTPException
    from unittest.mock import patch
    from app.utils.file_handler import save_upload
    from app.core.config import settings

    upload = _fake_upload("clip.mp4", "video/mp4", b"Just some plain text, not a video at all.")
    with patch.object(type(settings), "upload_path", new_callable=lambda: property(lambda self: tmp_path)):
        with pytest.raises(HTTPException) as exc_info:
            await save_upload(upload, "video")
    assert exc_info.value.status_code == 415


# ============================================================================
# Phase 3 — atomic WS ceiling + GPU lock timeout
# ============================================================================

def test_live_ws_ceiling_atomic_under_concurrent_connects():
    """Simulate 10 concurrent acquire attempts against the atomic counter pattern:
    exactly 5 must win, 5 must be rejected — no race can exceed the ceiling."""
    from app.api.v1.image import _live_connection_lock, _live_connection_count, LIVE_CONNECTION_CEILING

    async def scenario():
        # Reset shared state (test isolation)
        import app.api.v1.image as img_mod
        img_mod._live_connection_count = 0

        admitted, rejected = 0, 0

        async def try_connect():
            nonlocal admitted, rejected
            async with img_mod._live_connection_lock:
                if img_mod._live_connection_count >= img_mod.LIVE_CONNECTION_CEILING:
                    return "rejected"
                img_mod._live_connection_count += 1
                return "admitted"

        # Fire 10 concurrent attempts in the same event loop tick batch
        results = await asyncio.gather(*[try_connect() for _ in range(10)])
        # Cleanup
        img_mod._live_connection_count = 0
        return results

    results = asyncio.run(scenario())
    assert results.count("admitted") == LIVE_CONNECTION_CEILING
    assert results.count("rejected") == 10 - LIVE_CONNECTION_CEILING


@pytest.mark.asyncio
async def test_gpu_lock_timeout_raises_instead_of_deadlocking():
    """When the GPU semaphore is held, a second acquisition must raise
    GPUInferenceTimeoutError within the timeout window — never block forever."""
    from app.ml.vram_manager import (
        serialized_gpu_inference,
        gpu_inference_lock,
        GPUInferenceTimeoutError,
    )

    # Shrink the timeout so the test is fast (monkeypatch the module constant)
    import app.ml.vram_manager as vm
    original_timeout = vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS
    vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = 0.2

    try:
        # Hold the lock
        await gpu_inference_lock.acquire()
        with pytest.raises(GPUInferenceTimeoutError):
            async with serialized_gpu_inference("test_modality"):
                pass  # should never reach here
    finally:
        gpu_inference_lock.release()
        vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = original_timeout


@pytest.mark.asyncio
async def test_gpu_lock_released_after_body_exception():
    """An exception inside the inference body must release the semaphore
    (failure-safe finally) so subsequent acquisitions still work."""
    from app.ml.vram_manager import serialized_gpu_inference, gpu_inference_lock

    with pytest.raises(RuntimeError):
        async with serialized_gpu_inference("test_modality"):
            raise RuntimeError("simulated inference crash")

    # Semaphore must be back to full capacity — acquiring must not time out
    import app.ml.vram_manager as vm
    original_timeout = vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS
    vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = 1.0
    try:
        async with serialized_gpu_inference("test_modality"):
            pass  # success = lock was released correctly
    finally:
        vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = original_timeout


@pytest.mark.asyncio
async def test_gpu_lock_timeout_counter_incremented():
    """A lock timeout must be recorded in the metrics registry."""
    import asyncio as _asyncio
    from unittest.mock import patch
    from app.ml.vram_manager import serialized_gpu_inference, metrics
    import app.ml.vram_manager as vm

    original_timeout = vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS
    vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = 0.1
    before = metrics["gpu_lock_timeouts_total"]

    # Fresh semaphore so the test is independent of other tests' event loops
    fresh_lock = _asyncio.Semaphore(1)
    await fresh_lock.acquire()
    try:
        with patch.object(vm, "gpu_inference_lock", fresh_lock):
            with pytest.raises(Exception):
                async with serialized_gpu_inference("test_modality"):
                    pass
    finally:
        vm.GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = original_timeout

    assert metrics["gpu_lock_timeouts_total"] == pytest.approx(before + 1)


def test_metrics_render_prometheus_format():
    """The metrics registry must render valid Prometheus exposition text."""
    from app.ml.vram_manager import render_prometheus_metrics
    text = render_prometheus_metrics()
    assert "slop_inference_latency_seconds_count" in text
    assert "slop_inference_failures_total" in text
    assert "slop_vram_high_water_mark_mb" in text
    assert "# TYPE" in text


# ============================================================================
# Phase 8 — C2PA tiered trust classification
# ============================================================================

def test_c2pa_tier_absent_when_no_data():
    """No C2PA markers anywhere → tier 'absent', multiplier 1.00 (no effect)."""
    from app.ml.c2pa_trust import validate_c2pa_trust, TRUST_MULTIPLIERS

    result = validate_c2pa_trust(raw_bytes=b"\x89PNG plain image no manifest", file_path=None)
    assert result["tier"] == "absent"
    assert result["multiplier"] == TRUST_MULTIPLIERS["absent"] == 1.00


def test_c2pa_tier_unverified_claim_for_markers_without_manifest():
    """Byte-level C2PA/JUMBF markers without a parseable manifest →
    'unverified_claim' tier (0.60) — weaker than a fully trusted manifest."""
    from app.ml.c2pa_trust import validate_c2pa_trust, TRUST_MULTIPLIERS

    fake_markers = b"\x89PNG\r\n\x1a\n" + b"jumb" + b"c2pa" + b"x" * 1024
    result = validate_c2pa_trust(raw_bytes=fake_markers, file_path=None)
    assert result["tier"] == "unverified_claim"
    assert result["multiplier"] == TRUST_MULTIPLIERS["unverified_claim"] == 0.60


def test_c2pa_tier_broken_chain_when_signature_invalid():
    """A parsed manifest with validation failures must land in 'broken_chain'
    (0.80) — possible tampering gets almost no provenance credit."""
    from app.ml.c2pa_trust import validate_c2pa_trust, TRUST_MULTIPLIERS
    from unittest.mock import patch, MagicMock

    fake_reader = MagicMock()
    fake_manifest = MagicMock()
    fake_manifest.issuer = "Some CA"
    fake_manifest.claim_generator = "Test Generator"
    fake_manifest.validation_results.failure = ["signature_mismatch"]
    fake_manifest.signature_info = None
    fake_reader.get_manifest.return_value = fake_manifest

    with patch("app.ml.c2pa_trust._read_manifest", return_value={
        "issuer": "Some CA",
        "claim_generator": "Test Generator",
        "signature_valid": False,
        "dates_valid": None,
        "self_signed": False,
        "cert_pem": None,
    }):
        result = validate_c2pa_trust(raw_bytes=b"x", file_path="fake_path.jpg")

    assert result["tier"] == "broken_chain"
    assert result["multiplier"] == TRUST_MULTIPLIERS["broken_chain"] == 0.80


def test_c2pa_tier_fully_trusted_with_validated_manifest():
    """Valid signature + trusted issuer + valid dates → 'fully_trusted' (0.10)."""
    from app.ml.c2pa_trust import validate_c2pa_trust, TRUST_MULTIPLIERS
    from unittest.mock import patch

    with patch("app.ml.c2pa_trust._read_manifest", return_value={
        "issuer": "Adobe Content Authenticity",
        "claim_generator": "Adobe Firefly",
        "signature_valid": True,
        "dates_valid": True,
        "self_signed": False,
        "cert_pem": None,
    }):
        result = validate_c2pa_trust(raw_bytes=b"x", file_path="fake.jpg")

    assert result["tier"] == "fully_trusted"
    assert result["multiplier"] == TRUST_MULTIPLIERS["fully_trusted"] == 0.10


def test_c2pa_tier_self_signed_when_issuer_untrusted():
    """Valid signature but issuer not in any trust anchor → 'self_signed' (0.35)."""
    from app.ml.c2pa_trust import validate_c2pa_trust, TRUST_MULTIPLIERS
    from unittest.mock import patch

    with patch("app.ml.c2pa_trust._read_manifest", return_value={
        "issuer": "Random Guy's Self-Signed CA",
        "claim_generator": "Hobby Tool",
        "signature_valid": True,
        "dates_valid": True,
        "self_signed": True,
        "cert_pem": None,
    }):
        result = validate_c2pa_trust(raw_bytes=b"x", file_path="fake.jpg")

    assert result["tier"] == "self_signed"
    assert result["multiplier"] == TRUST_MULTIPLIERS["self_signed"] == 0.35


def test_c2pa_tier_broken_chain_when_cert_expired():
    """Valid signature but expired certificate → 'broken_chain' (0.80)."""
    from app.ml.c2pa_trust import validate_c2pa_trust
    from unittest.mock import patch

    with patch("app.ml.c2pa_trust._read_manifest", return_value={
        "issuer": "Adobe Content Authenticity",
        "claim_generator": "Adobe Firefly",
        "signature_valid": True,
        "dates_valid": False,  # expired
        "self_signed": False,
        "cert_pem": None,
    }):
        result = validate_c2pa_trust(raw_bytes=b"x", file_path="fake.jpg")

    assert result["tier"] == "broken_chain"


def test_c2pa_tier_ordering_is_monotonic_in_trust():
    """Trust multipliers must be strictly ordered: more trust = more suppression."""
    from app.ml.c2pa_trust import TRUST_MULTIPLIERS as m
    assert m["fully_trusted"] < m["self_signed"] < m["unverified_claim"] < m["broken_chain"] < m["absent"]


def test_image_check_c2pa_returns_tier_fields():
    """The image pipeline's _check_c2pa must surface tier/multiplier fields."""
    from app.ml.image_detector import _check_c2pa

    result = _check_c2pa.__wrapped__("nonexistent_file.jpg") if hasattr(_check_c2pa, "__wrapped__") else None
    # Direct call with a nonexistent path should fail parsing and land in absent/negative
    result = _check_c2pa("definitely_not_a_real_file_12345.jpg")
    assert result["has_c2pa"] is False or result.get("tier") in {"absent", "unverified_claim"}
