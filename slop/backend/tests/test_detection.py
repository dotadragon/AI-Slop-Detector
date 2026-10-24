import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ml.image_detector import analyze_image, analyze_image_bytes


def test_text_analysis_returns_explainable_signals():
    client = TestClient(app)
    client.get("/api/v1/auth/csrf")
    csrf = client.cookies.get("csrf_token")
    response = client.post(
        "/api/v1/text/analyze",
        json={"text": "Furthermore, it is important to note that this passage presents a balanced overview. In conclusion, the system is robust."},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"]
    assert 0 <= body["result"]["overall_score"] <= 1
    assert body["result"]["sentences"]


def test_text_rejects_null_bytes():
    client = TestClient(app)
    client.get("/api/v1/auth/csrf")
    csrf = client.cookies.get("csrf_token")
    response = client.post("/api/v1/text/analyze", json={"text": "bad\x00input"}, headers={"X-CSRF-Token": csrf})
    assert response.status_code == 422


def test_image_analysis_is_deterministic_and_labeled(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (320, 320), color=(24, 96, 148)).save(image_path)

    first = analyze_image(str(image_path))
    second = analyze_image(str(image_path))

    assert first["overall_score"] == second["overall_score"]
    assert first["signals"]["heuristic_score"] == second["signals"]["heuristic_score"]
    assert "cnn_score" not in first["signals"]
    assert len(first["model_hint"]) > 0 and any(
        kw in first["model_hint"].lower()
        for kw in ("neural", "heuristic", "diagram", "authentic")
    )


def test_image_analysis_accepts_live_frame_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), color=(120, 42, 66)).save(buffer, format="JPEG")

    result = analyze_image_bytes(buffer.getvalue())

    assert 0 <= result["overall_score"] <= 1
    assert result["signals"]["file_name"] == "live-frame"


def test_live_image_websocket_returns_frame_result():
    client = TestClient(app)
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), color=(80, 100, 120)).save(buffer, format="JPEG")

    with client.websocket_connect("/api/v1/image/live") as websocket:
        websocket.send_bytes(buffer.getvalue())
        payload = websocket.receive_json()

    assert payload["status"] == "completed"
    assert payload["frame_index"] == 1
    assert 0 <= payload["result"]["overall_score"] <= 1


def test_diagram_recognized_as_authentic_without_synthid_watermark():
    from pathlib import Path
    diagram_path = Path(r"C:\Users\karth\.gemini\antigravity-ide\brain\ebe44b8f-a3cf-489f-a0c5-840c1ee9d007\.user_uploaded\media_1789756069231.png")
    if not diagram_path.exists():
        import pytest
        pytest.skip("Test diagram artifact not found")

    result = analyze_image(str(diagram_path))
    assert result["overall_score"] <= 0.05, f"Diagram overall score should be <= 0.05, got {result['overall_score']}"
    assert result["digital_signature"]["detected"] is False
    assert result["digital_signature"]["bounding_box"] is None
    assert "diagram" in result["generator_attribution"]["generator"].lower() or "authentic" in result["generator_attribution"]["generator"].lower()


def test_camera_photo_recognized_as_authentic_without_synthid_watermark():
    from pathlib import Path
    photo_path = Path(r"C:\Users\karth\.gemini\antigravity-ide\brain\ebe44b8f-a3cf-489f-a0c5-840c1ee9d007\.user_uploaded\media_1789756044612.jpg")
    if not photo_path.exists():
        import pytest
        pytest.skip("Test camera photo artifact not found")

    result = analyze_image(str(photo_path))
    assert result["overall_score"] <= 0.10, f"Camera photo overall score should be <= 0.10, got {result['overall_score']}"
    assert result["digital_signature"]["detected"] is False
    assert result["digital_signature"]["bounding_box"] is None
    assert "camera" in result["generator_attribution"]["generator"].lower() or "authentic" in result["generator_attribution"]["generator"].lower()
