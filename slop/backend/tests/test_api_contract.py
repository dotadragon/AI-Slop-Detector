"""
G5: API Contract and Schema Stability Test (ANTIGRAVITY_RULES Rule 6).
Ensures response field names are versioned and stable.
"""

from app.schemas.detection import DetectionResultOut, AnalyzeResponse, SentenceScore

def test_api_detection_contract_fields():
    expected_fields = {
        "overall_score",
        "signals",
        "sentences",
        "frame_scores",
        "heatmap_b64",
        "model_hint",
    }
    actual_fields = set(DetectionResultOut.model_fields.keys())
    assert expected_fields.issubset(actual_fields), f"Missing expected fields: {expected_fields - actual_fields}"

def test_analyze_response_contract():
    expected_fields = {"job_id", "result", "status"}
    actual_fields = set(AnalyzeResponse.model_fields.keys())
    assert expected_fields == actual_fields
