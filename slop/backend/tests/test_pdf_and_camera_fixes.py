import pytest
import math
import numpy as np
from PIL import Image
from app.ml.fusion import fuse_image_score
from app.api.v1.text import compute_section_scores_fast
from app.ml.binoculars import compute_binoculars_and_curvature


def test_fuse_image_score_edited_camera_photo():
    # When is_edited_camera_photo is flagged, the score should land in [0.14, 0.24]
    features = {
        "best_neural": 0.88,
        "heuristic_score": 0.15,
        "is_edited_camera_photo": 1.0,
        "is_real_camera": 1.0,
        "cfa_score": 0.15,
        "ela_score": 0.10,
        "fft_score": 0.12,
        "noise_score": 0.10,
        "dct_score": 0.25,
        "metadata_risk": 0.0,
        "is_diagram": 0.0,
        "is_anime_ai": 0.0,
        "signature_detected": 0.0,
        "signature_confidence": 0.0,
        "c2pa_verified": 0.0,
        "clip_probe_score": 0.12,
        "npr_score": 0.133,
    }
    result = fuse_image_score(features)
    assert 0.14 <= result["overall_score"] <= 0.24
    assert result["primary_signal"] == "optical_sensor_edited_photo"
    assert "edited_camera_correction" in result["contributions"]
    assert result["confidence_band"] in ("LIKELY_HUMAN", "DEFINITELY_HUMAN")


def test_compute_section_scores_fast():
    sections = [
        {"id": 0, "title": "Introduction", "text": "This is section one with human style text.", "page": 1},
        {"id": 1, "title": "Methods", "text": "This is section two with detailed experimental steps.", "page": 2},
    ]
    full_doc_result = {
        "overall_score": 0.15,
        "sentences": [
            {"text": "This is section one with human style text.", "ai_probability": 0.12},
            {"text": "This is section two with detailed experimental steps.", "ai_probability": 0.18},
        ],
    }
    scores = compute_section_scores_fast(sections, full_doc_result)
    assert len(scores) == 2
    assert scores[0]["section_id"] == 0
    assert 0.0 <= scores[0]["score"] <= 1.0
    assert scores[1]["section_id"] == 1
    assert 0.0 <= scores[1]["score"] <= 1.0


def test_binoculars_returns_lm_perplexity():
    sample_text = (
        "Artificial intelligence and machine learning have revolutionized computational linguistics. "
        "Modern language models utilize deep transformer architectures to model conditional token probabilities."
    )
    res = compute_binoculars_and_curvature(sample_text)
    assert "lm_perplexity" in res
    assert isinstance(res["lm_perplexity"], float)
    assert res["lm_perplexity"] > 0.0


def test_image_detector_pipeline_signals():
    from app.ml.image_detector import _analyze_pil_image
    arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    arr[:, :, 1] = ((arr[:, :, 0].astype(int) + arr[:, :, 2].astype(int)) // 2).astype(np.uint8)
    import io
    buf = io.BytesIO()
    img = Image.fromarray(arr)
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    test_img = Image.open(buf)
    res = _analyze_pil_image(test_img, file_name="image_17f4df.jpg")
    assert "overall_score" in res
    assert "signals" in res
    assert "is_edited_camera_photo" in res["signals"]
    assert "confidence_band" in res
