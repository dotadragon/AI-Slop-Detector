import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ml.image_detector import analyze_image, analyze_image_bytes


def test_text_analysis_returns_explainable_signals():
    client = TestClient(app)
    client.get("/api/v1/auth/csrf")
    csrf_token = client.cookies.get("csrf_token")
    headers: dict[str, str] = {"X-CSRF-Token": str(csrf_token)} if csrf_token else {}
    response = client.post(
        "/api/v1/text/analyze",
        json={"text": "Furthermore, it is important to note that this passage presents a balanced overview. In conclusion, the system is robust."},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"]
    assert 0 <= body["result"]["overall_score"] <= 1
    assert body["result"]["sentences"]


def test_text_rejects_null_bytes():
    client = TestClient(app)
    client.get("/api/v1/auth/csrf")
    csrf_token = client.cookies.get("csrf_token")
    headers: dict[str, str] = {"X-CSRF-Token": str(csrf_token)} if csrf_token else {}
    response = client.post("/api/v1/text/analyze", json={"text": "bad\x00input"}, headers=headers)
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
    """Directive 1 & Rule 6: AI-generated diagrams with elevated neural scores (> 0.60)
    must NOT be short-circuited down to <= 0.05 by diagram heuristics.
    Diagram detection must preserve AI verdicts (score >= 0.35, max downward adjustment -0.15).
    """
    from pathlib import Path
    diagram_path = Path(r"C:\Users\karth\.gemini\antigravity-ide\brain\ebe44b8f-a3cf-489f-a0c5-840c1ee9d007\.user_uploaded\media_1789756069231.png")
    if not diagram_path.exists():
        import pytest
        pytest.skip("Test diagram artifact not found")

    result = analyze_image(str(diagram_path))
    assert result["overall_score"] >= 0.35, f"AI diagram should preserve AI verdict (>= 0.35), got {result['overall_score']}"
    assert result["digital_signature"]["detected"] is False
    assert result["digital_signature"]["bounding_box"] is None


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


def test_code_forensics_human_manual_coded():
    """Directive 6: Python competitive/manual code (human_0252) routed to Code Forensics Track.
    Must evaluate identifier entropy and comment density, asserting score <= 0.08 and
    verdict is 'Manual Coded (Human / Competitive)'."""
    human_0252 = (
        "def find_ind(key, i, n, \n"
        "\t\t\tk, arr):\n"
        "\tind = -1\n"
        "\tstart = i + 1\n"
        "\tend = n - 1;\n"
        "\twhile (start < end):\n"
        "\t\tmid = int(start +\n"
        "\t\t\t\t(end - start) / 2)\n"
        "\t\tif (arr[mid] - key <= k):\n"
        "\t\t\tind = mid\n"
        "\t\t\tstart = mid + 1\n"
        "\t\telse:\n"
        "\t\t\tend = mid\n"
        "\treturn ind\n"
        "def removals(arr, n, k):\n"
        "\tans = n - 1\n"
        "\tarr.sort()\n"
        "\tfor i in range(0, n):\n"
        "\t\tj = find_ind(arr[i], i, \n"
        "\t\t\t\t\tn, k, arr)\n"
        "\t\tif (j != -1):\n"
        "\t\t\tans = min(ans, n -\n"
        "\t\t\t\t\t\t(j - i + 1))\n"
        "\treturn ans\n"
    )
    from app.ml.text_detector import analyze_text
    res = analyze_text(human_0252)
    assert res["signals"]["is_code"] == 1.0
    assert res["overall_score"] <= 0.08, f"Expected score <= 0.08, got {res['overall_score']}"
    verdict = res["signals"].get("code_origin") or res["signals"].get("code_verdict") or ""
    assert "Manual Coded" in verdict, f"Expected 'Manual Coded' in verdict, got '{verdict}'"

    # Also verify via API endpoint
    client = TestClient(app)
    client.get("/api/v1/auth/csrf")
    csrf_token = client.cookies.get("csrf_token")
    headers: dict[str, str] = {"X-CSRF-Token": str(csrf_token)} if csrf_token else {}
    resp = client.post("/api/v1/text/analyze", json={"text": human_0252}, headers=headers)
    assert resp.status_code == 200
    api_res = resp.json()["result"]
    assert api_res["signals"]["is_code"] == 1.0
    assert api_res["overall_score"] <= 0.08
    api_verdict = api_res["signals"].get("code_origin") or api_res["signals"].get("code_verdict") or ""
    assert "Manual Coded" in api_verdict
