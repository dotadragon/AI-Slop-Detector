"""
G1: Non-inversion and monotonicity test.
Asserts that higher synthetic feature evidence consistently scores higher than human evidence.
"""

from app.ml.fusion import fuse_image_score, fuse_text_score, fuse_video_score

def test_image_no_inversion():
    ai_features = {
        "best_neural": 0.95,
        "heuristic_score": 0.80,
        "is_diagram": 0.0,
        "is_real_camera": 0.0,
    }
    human_features = {
        "best_neural": 0.05,
        "heuristic_score": 0.10,
        "is_diagram": 0.0,
        "is_real_camera": 1.0,
    }
    ai_res = fuse_image_score(ai_features)
    human_res = fuse_image_score(human_features)

    assert ai_res["overall_score"] > human_res["overall_score"]
    assert ai_res["overall_score"] >= 0.70
    assert human_res["overall_score"] <= 0.40

def test_text_no_inversion():
    ai_features = {
        "neural_prob": 0.95,
        "heuristic_score": 0.85,
        "llm_pattern_score": 0.80,
    }
    human_features = {
        "neural_prob": 0.02,
        "heuristic_score": 0.20,
        "llm_pattern_score": 0.0,
    }
    ai_res = fuse_text_score(ai_features)
    human_res = fuse_text_score(human_features)

    assert ai_res["overall_score"] > human_res["overall_score"]
    assert ai_res["overall_score"] >= 0.65
    assert human_res["overall_score"] <= 0.45

def test_video_no_inversion():
    ai_features = {
        "peak_threat": 0.90,
        "top_quartile_mean": 0.85,
        "global_mean": 0.80,
        "is_screen_rec": 0.0,
        "has_deepfake_face": 1.0,
    }
    human_features = {
        "peak_threat": 0.10,
        "top_quartile_mean": 0.08,
        "global_mean": 0.05,
        "is_screen_rec": 1.0,
        "has_deepfake_face": 0.0,
    }
    ai_res = fuse_video_score(ai_features)
    human_res = fuse_video_score(human_features)

    assert ai_res["overall_score"] > human_res["overall_score"]
    assert ai_res["overall_score"] >= 0.80
    assert human_res["overall_score"] <= 0.20
