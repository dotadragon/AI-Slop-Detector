"""Step-1 probe: print flag values + sample disabled-modality API responses.

Read-only with respect to code: uses FastAPI TestClient in-process. Writes a
temp upload into the system tempdir only; never touches project data dirs.
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")
sys.path.insert(0, "slop/backend")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402

print("=== FLAG VALUES (config.py defaults) ===")
for name in (
    "enable_document_forensics",
    "enable_audio_detector",
    "enable_reconstruction_detector",
    "enable_freq_detector",
):
    print(f"  settings.{name} = {getattr(settings, name)}")

print("\n=== video router gate check (grep-equivalent, in code) ===")
import pathlib  # noqa: E402

video_src = pathlib.Path("slop/backend/app/api/v1/video.py").read_text(encoding="utf-8")
print("  video.py contains 'enable_' flag check:", "enable_" in video_src)
print("  video.py contains 'disabled' response:", '"disabled"' in video_src)

print("\n=== registered routes (modality prefixes) ===")
for r in app.routes:
    path = getattr(r, "path", "")
    if any(p in path for p in ("/video", "/audio", "/document", "/text")):
        print("  ", getattr(r, "methods", ""), path)

client = TestClient(app)
# Acquire CSRF cookie + header exactly like the test suite does.
client.get("/api/v1/auth/csrf")
CSRF = client.cookies.get("csrf_token")
CSRF_HEADERS = {"X-CSRF-Token": CSRF} if CSRF else {}

# --- document: disabled-gate fires before save_upload, so bytes don't matter,
# but send a plausible minimal PDF anyway.
print("\n=== SAMPLE RESPONSE: POST /api/v1/document/upload (flag OFF) ===")
pdf_bytes = b"%PDF-1.4\n1 0 obj\nendobj\n%%EOF\n"
resp = client.post(
    "/api/v1/document/upload",
    files={"file": ("probe.pdf", pdf_bytes, "application/pdf")},
    headers=CSRF_HEADERS,
)
print("HTTP", resp.status_code)
print(json.dumps(resp.json(), indent=2)[:600])

# --- audio: same gate-first ordering.
print("\n=== SAMPLE RESPONSE: POST /api/v1/audio/upload (flag OFF) ===")
wav_bytes = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 16
resp = client.post(
    "/api/v1/audio/upload",
    files={"file": ("probe.wav", wav_bytes, "audio/wav")},
    headers=CSRF_HEADERS,
)
print("HTTP", resp.status_code)
print(json.dumps(resp.json(), indent=2)[:600])

# --- audio SSE stream variant.
print("\n=== SAMPLE RESPONSE: POST /api/v1/audio/upload-stream (flag OFF, SSE) ===")
resp = client.post(
    "/api/v1/audio/upload-stream",
    files={"file": ("probe.wav", wav_bytes, "audio/wav")},
    headers=CSRF_HEADERS,
)
print("HTTP", resp.status_code, "| media_type:", resp.headers.get("content-type"))
print(repr(resp.text[:200]))

# --- video: flag OFF -> clean disabled response (gate fires before save_upload).
print("\n=== SAMPLE RESPONSE: POST /api/v1/video/upload (flag OFF) ===")
mp4_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
resp = client.post(
    "/api/v1/video/upload",
    files={"file": ("probe.mp4", mp4_bytes, "video/mp4")},
    headers=CSRF_HEADERS,
)
print("HTTP", resp.status_code)
print(json.dumps(resp.json(), indent=2)[:300])

print("\n=== SAMPLE RESPONSE: POST /api/v1/video/upload-stream (flag OFF, SSE) ===")
resp = client.post(
    "/api/v1/video/upload-stream",
    files={"file": ("probe.mp4", mp4_bytes, "video/mp4")},
    headers=CSRF_HEADERS,
)
print("HTTP", resp.status_code, "| media_type:", resp.headers.get("content-type"))
print(repr(resp.text[:160]))

# --- text analyze: prove banding fields flow through the API response.
print("\n=== SAMPLE RESPONSE: POST /api/v1/text/analyze (banding fields) ===")
resp = client.post(
    "/api/v1/text/analyze",
    json={"text": "The quarterly report outlines the main findings from our regional operations and highlights several areas for improvement next year."},
    headers=CSRF_HEADERS,
)
data = resp.json()
res = data.get("result") or {}
print("HTTP", resp.status_code)
print("  top-level zone_low/zone_high/band:", {k: data.get(k) for k in ("zone_low", "zone_high", "confidence_band")})
print("  result band/range/zones:", {k: res.get(k) for k in ("confidence_band", "confidence_range", "zone_low", "zone_high")})

# --- video: flag ON (probe-only settings override) -> live response.
# Mirrors what a test-settings override would do; production ships the flag OFF.
import os as _os  # noqa: E402

TEST_MP4 = r"C:\Users\karth\Downloads\test.mp4"
if _os.path.exists(TEST_MP4):
    print("\n=== SAMPLE RESPONSE: POST /api/v1/video/upload (flag ON via settings override) ===")
    settings.enable_video_detector = True
    with open(TEST_MP4, "rb") as fh:
        resp = client.post(
            "/api/v1/video/upload",
            files={"file": ("test.mp4", fh.read(), "video/mp4")},
            headers=CSRF_HEADERS,
        )
    settings.enable_video_detector = False
    print("HTTP", resp.status_code)
    data = resp.json()
    print("  keys:", sorted(data.keys()))
    print("  status:", data.get("status"), "| overall_score:", (data.get("result") or {}).get("overall_score"))
else:
    print("\n(test.mp4 not present — flag-ON live response skipped)")

# cleanup tempdir if created (save_upload may not have been reached at all)
print("\n(probe complete — no project files touched)")
