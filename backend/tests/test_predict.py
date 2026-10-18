"""API-level tests for POST /predict.

test_predict_blank_image_reports_no_face is the "inference behaviour" test
required beyond a bare HTTP-200 check: it asserts on the *content* of the
prediction (face_detected must be False, no gaze/expression emitted, a
warning is surfaced) for a known input that contains no face, not just that
the call succeeded. See MLOPS_NOTES.md "Inference testing" for the
companion unit tests on the gaze/expression math itself.
"""
import io

from PIL import Image


def _blank_jpeg_bytes(size=(200, 200), color=(120, 120, 120)) -> bytes:
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_blank_image_reports_no_face(client):
    files = {"file": ("frame.jpg", _blank_jpeg_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["face_detected"] is False
    assert body["landmark_count"] == 0
    assert body["gaze"] is None
    assert body["expression"] is None
    assert "no face detected" in body["warnings"][0]
    assert body["inference_time_ms"] >= 0
    model_names = {m["name"] for m in body["models"]}
    assert "mediapipe-face-mesh" in model_names
    assert "dima806/facial_emotions_image_detection" in model_names


def test_predict_rejects_unsupported_content_type(client):
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "unsupported content type" in response.json()["detail"]


def test_predict_rejects_empty_file(client):
    files = {"file": ("frame.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "empty file"


def test_predict_rejects_corrupt_image_bytes(client):
    files = {"file": ("frame.jpg", b"not-actually-a-jpeg", "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"]
