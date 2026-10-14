"""Audio modality tests (Phase D11).

Covers: feature-gate behavior, WAV magic-byte acceptance/rejection in
save_upload, end-to-end detection on a synthetic WAV (deterministic
score contract, not model quality), and fusion-authority compliance.
Model benchmarking lives in benchmark_audio.py with real held-out data.
"""
from __future__ import annotations

import io
import math
import struct
import wave

import pytest
from fastapi import HTTPException
from unittest.mock import patch

from app.utils.file_handler import save_upload, _sniff_mime


def _wav_bytes(seconds: float = 1.0, freq: float = 220.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(int(seconds * sr))
        )
        w.writeframes(frames)
    return buf.getvalue()


class _FakeUpload:
    def __init__(self, data: bytes, filename: str, content_type: str):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self, n: int) -> bytes:
        out, self._data = self._data[:n], self._data[n:]
        return out


def test_wav_magic_bytes_detected():
    assert _sniff_mime(_wav_bytes()[:4096], "application/octet-stream") == "audio/wav"


def test_mp3_magic_bytes_detected():
    assert _sniff_mime(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 100, "application/octet-stream") == "audio/mpeg"


def _patch_upload_path(tmp_path):
    """upload_path is a read-only Settings property; patch like test_hardening_phases.py."""
    from app.core.config import settings

    return patch.object(type(settings), "upload_path", new_callable=lambda: property(lambda self: tmp_path))


@pytest.mark.asyncio
async def test_save_upload_accepts_wav(tmp_path):
    with _patch_upload_path(tmp_path):
        data = _wav_bytes()
        path, mime, size = await save_upload(_FakeUpload(data, "clip.wav", "audio/wav"), "audio")
        assert path.exists() and size == len(data)
        assert mime == "audio/wav"


@pytest.mark.asyncio
async def test_save_upload_rejects_disguised_audio(tmp_path):
    """A .wav extension carrying non-audio bytes must be 415-rejected."""
    with _patch_upload_path(tmp_path):
        with pytest.raises(HTTPException) as exc:
            await save_upload(_FakeUpload(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200, "evil.wav", "audio/wav"), "audio")
        assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_analyze_audio_synthetic_wav(tmp_path):
    """End-to-end detector run on a synthetic tone: contract check only."""
    from app.ml.audio_detector import analyze_audio, evict_audio_model

    p = tmp_path / "tone.wav"
    p.write_bytes(_wav_bytes(seconds=1.0))
    try:
        result = analyze_audio(str(p), "job_test_audio")
        assert 0.0 <= result["overall_score"] <= 1.0
        assert result["verdict"] in {"likely_ai", "likely_human", "uncertain"}
        assert "heuristic_score" in result["signals"]
    finally:
        evict_audio_model()


def test_fuse_audio_score_neural_first():
    """Fusion authority: neural dominates when present, heuristic-only is low-confidence."""
    from app.ml.fusion import fuse_audio_score

    fused = fuse_audio_score({"neural_fake": 0.95, "heuristic_score": 0.4})
    assert fused["verdict"] == "likely_ai"  # 0.75*0.95 + 0.25*0.4 = 0.8125
    assert fused["confidence"] == 0.85

    fused_b = fuse_audio_score({"neural_fake": 0.9, "heuristic_score": 0.2})
    assert fused_b["verdict"] == "uncertain"  # 0.725 < likely_ai threshold

    fused_h = fuse_audio_score({"neural_fake": None, "heuristic_score": 0.9})
    assert fused_h["confidence"] == 0.35  # deliberately low without neural
    assert fused_h["overall_score"] == pytest.approx(0.9)


def test_audio_detector_delegates_to_fusion():
    """Detector must not compute overall itself (no-hard-clamp rule)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "ml" / "audio_detector.py"
    text = src.read_text(encoding="utf-8")
    assert "fuse_audio_score" in text
    assert "0.75 * neural_fake" not in text
