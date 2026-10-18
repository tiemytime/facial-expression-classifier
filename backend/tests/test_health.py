def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_ready_reports_both_models_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True

    models_by_task = {m["task"]: m for m in body["models"]}
    assert models_by_task["landmark_detection"]["name"] == "mediapipe-face-mesh"
    assert models_by_task["landmark_detection"]["version"].startswith("mediapipe-")
    assert models_by_task["image_classification"]["name"] == "dima806/facial_emotions_image_detection"
