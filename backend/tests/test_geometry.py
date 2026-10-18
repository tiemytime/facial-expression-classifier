"""Unit tests for the deterministic gaze/expression heuristics.

These exercise the actual classification math against synthetic landmark
coordinates (no mediapipe, no image decoding involved) - the kind of
inference-behaviour test the assessment asks for beyond "the endpoint
returned 200". Real face-mesh landmarks are normalized (x, y, z) floats in
[0, 1]-ish ranges; the exact magnitudes below are chosen to land cleanly on
each side of the heuristics' thresholds, not to resemble a real face.
"""
import pytest

from app.models.expression import estimate_expression
from app.models.gaze import estimate_gaze


def make_landmarks(overrides: dict, length: int = 478) -> list:
    landmarks = [(0.0, 0.0, 0.0)] * length
    for index, point in overrides.items():
        landmarks[index] = point
    return landmarks


# --- gaze ---------------------------------------------------------------

def _symmetric_eye_landmarks(iris_offset_left: float, iris_offset_right: float) -> dict:
    overrides = {
        33: (0.0, 0.0, 0.0),
        133: (0.2, 0.0, 0.0),
        362: (0.0, 0.0, 0.0),
        263: (0.2, 0.0, 0.0),
    }
    for i in range(468, 473):
        overrides[i] = (iris_offset_left, 0.0, 0.0)
    for i in range(473, 478):
        overrides[i] = (iris_offset_right, 0.0, 0.0)
    return overrides


def test_gaze_centered_iris_reports_center():
    landmarks = make_landmarks(_symmetric_eye_landmarks(0.1, 0.1))
    result = estimate_gaze(landmarks)
    assert result.direction == "center"
    assert result.confidence > 0.9


def test_gaze_iris_near_outer_corner_reports_left():
    landmarks = make_landmarks(_symmetric_eye_landmarks(0.0, 0.0))
    result = estimate_gaze(landmarks)
    assert result.direction == "left"


def test_gaze_iris_near_inner_corner_reports_right():
    landmarks = make_landmarks(_symmetric_eye_landmarks(0.2, 0.2))
    result = estimate_gaze(landmarks)
    assert result.direction == "right"


def test_gaze_requires_iris_landmarks():
    with pytest.raises(ValueError):
        estimate_gaze(make_landmarks({}, length=468))


# --- expression -----------------------------------------------------------

def _base_face_overrides() -> dict:
    return {234: (0.0, 0.0, 0.0), 454: (1.0, 0.0, 0.0)}


def test_expression_neutral_face():
    overrides = _base_face_overrides()
    overrides.update({61: (0.4, 0.0, 0.0), 291: (0.6, 0.0, 0.0)})  # narrow mouth
    overrides.update({13: (0.5, 0.0, 0.0), 14: (0.5, 0.01, 0.0)})  # near-closed mouth
    overrides.update({105: (0.3, 0.0, 0.0), 159: (0.3, 0.01, 0.0)})  # relaxed eyebrow
    result = estimate_expression(make_landmarks(overrides))
    assert result.label == "neutral"


def test_expression_wide_mouth_is_happy():
    overrides = _base_face_overrides()
    overrides.update({61: (0.1, 0.0, 0.0), 291: (0.7, 0.0, 0.0)})  # wide mouth = smile
    overrides.update({13: (0.4, 0.0, 0.0), 14: (0.4, 0.02, 0.0)})
    overrides.update({105: (0.3, 0.0, 0.0), 159: (0.3, 0.01, 0.0)})
    result = estimate_expression(make_landmarks(overrides))
    assert result.label == "happy"


def test_expression_open_mouth_and_raised_eyebrows_is_surprised():
    overrides = _base_face_overrides()
    overrides.update({61: (0.4, 0.0, 0.0), 291: (0.6, 0.0, 0.0)})
    overrides.update({13: (0.5, 0.0, 0.0), 14: (0.5, 0.1, 0.0)})  # wide-open mouth
    overrides.update({105: (0.3, 0.0, 0.0), 159: (0.3, 0.1, 0.0)})  # raised eyebrow
    result = estimate_expression(make_landmarks(overrides))
    assert result.label == "surprised"


def test_expression_requires_full_mesh():
    with pytest.raises(ValueError):
        estimate_expression(make_landmarks({}, length=400))
