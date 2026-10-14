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


def test_stats_telemetry_alias_contract():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"
    assert "telemetry" in data

