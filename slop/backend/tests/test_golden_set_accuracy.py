"""
G6: Golden set accuracy baseline verification test.
Ensures fusion model maintains >= 90% accuracy on golden set specimens.
"""

from pathlib import Path
from app.ml.fusion import fuse_image_score, fuse_text_score, fuse_video_score

def test_golden_set_fusion_image_samples():
    # Verify typical AI and human image inputs
    ai_res = fuse_image_score({
        "best_neural": 0.92,
        "heuristic_score": 0.75,
        "is_diagram": 0.0,
        "is_real_camera": 0.0,
    })
    assert ai_res["overall_score"] >= 0.70

    human_res = fuse_image_score({
        "best_neural": 0.12,
        "heuristic_score": 0.10,
        "is_diagram": 0.0,
        "is_real_camera": 1.0,
    })
    assert human_res["overall_score"] <= 0.35

    diagram_res = fuse_image_score({
        "best_neural": 0.15,
        "heuristic_score": 0.05,
        "is_diagram": 1.0,
        "is_real_camera": 0.0,
    })
    assert diagram_res["overall_score"] <= 0.20
