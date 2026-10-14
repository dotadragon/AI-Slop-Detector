"""
SLOP Production Video Forensics Test Suite
Enforces zero-mock verification, VRAM ceiling checking, and real forward passes.
"""

import os
from pathlib import Path
import pytest

from app.ml.video_detector import analyze_video, _video_info
from app.ml.vram_manager import get_vram_stats, release_all_model_singletons


@pytest.fixture(scope="module", autouse=True)
def _clean_vram_baseline():
    """
    Release every cross-modality model singleton before this module's tests run.

    WHY THIS EXISTS: production evicts a modality's weights the instant its
    request completes (`serialized_gpu_inference` -> `evict_gpu_models`), so no
    two modalities share the 6 GB RTX 4050. pytest ordering does not honour that
    discipline. A text test run earlier in the session parks the RoBERTa / Qwen /
    Binoculars weights on the GPU, and the video tests below then measure their
    own footprint PLUS those parked weights — inflating `memory_allocated` past
    the 5 GB ceiling even though the video path alone sits at ~1.6 GB. Replaying
    the governor's eviction contract here makes the VRAM assertion a property of
    the video pipeline instead of a function of which test ran first.

    Module scope on purpose: the eviction happens once, before the module's first
    test, so this module's own reloads are not repeated per test.
    """
    release_all_model_singletons()
    yield


def test_video_info_extraction():
    test_video = r"C:\Users\karth\Downloads\test.mp4"
    if not os.path.exists(test_video):
        pytest.skip("Test video not found at C:\\Users\\karth\\Downloads\\test.mp4")

    info = _video_info(test_video)
    assert info["fps"] > 0
    assert info["total_frames"] > 0
    assert info["duration_seconds"] > 0
    assert info["width"] == 1920
    assert info["height"] == 1080


def test_ai_video_detection_on_test_mp4():
    """
    Validates that test.mp4 (AI-generated video) correctly scores >= 80% AI.
    Previously failed with 13.1% (False Negative) due to face-swap model misapplication.
    """
    test_video = r"C:\Users\karth\Downloads\test.mp4"
    if not os.path.exists(test_video):
        pytest.skip("Test video not found at C:\\Users\\karth\\Downloads\\test.mp4")

    result = analyze_video(test_video, "prod_verify_test")

    print(f"\n[PRODUCTION VERIFICATION] test.mp4 Overall Score: {result['overall_score']}")
    print(f"Signals: {result['signals']}")
    for f in result.get("frame_scores", []):
        print(f"  Frame at {f['timestamp']}s: AI Score={f['score']}, Scene={f['scene_score']}, Face={f['face_deepfake_score']}, SynthID={f['synthid_detected']}")

    assert result["overall_score"] >= 0.80, f"Expected AI score >= 0.80, got {result['overall_score']}"
    assert result["signals"]["total_frames_analyzed"] >= 5

    # Verify VRAM limits
    vram = get_vram_stats()
    print(f"VRAM Stats: {vram}")
    assert vram["allocated_mb"] < 5000, f"VRAM allocation exceeded 5GB ceiling: {vram['allocated_mb']}MB"


def test_screen_recording_detection_on_appeal_mp4():
    """
    Validates that appeal.mp4 (Human Screen Recording) correctly scores <= 0.10 (Authentic Human).
    Previously scored 94% (False Positive) due to Swin-Base camera-model bias and SynthID DCT boundary false alarms.
    """
    appeal_video = r"C:\Users\karth\Downloads\appeal.mp4"
    if not os.path.exists(appeal_video):
        pytest.skip("Appeal video not found at C:\\Users\\karth\\Downloads\\appeal.mp4")

    result = analyze_video(appeal_video, "prod_verify_appeal")

    print(f"\n[PRODUCTION VERIFICATION] appeal.mp4 Overall Score: {result['overall_score']}")
    print(f"Signals: {result['signals']}")
    for f in result.get("frame_scores", []):
        print(f"  Frame at {f['timestamp']}s: AI Score={f['score']}, Scene={f['scene_score']}, Face={f['face_deepfake_score']}, SynthID={f['synthid_detected']}")

    assert result["overall_score"] <= 0.10, f"Expected Human score <= 0.10 for screen recording, got {result['overall_score']}"
    assert result["signals"]["modality"] == "screen_recording"
