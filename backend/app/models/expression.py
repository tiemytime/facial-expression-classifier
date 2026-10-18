"""Deterministic expression heuristic built on top of face-mesh landmarks.

Same design rationale as gaze.py: explainable geometry on pretrained
landmarks rather than a second trained classifier. Produces one of
neutral / happy / surprised with a confidence score.

Limitations: coarse, lighting- and angle-sensitive, not validated against a
labeled expression dataset. Good enough for a demo, not for any
decision-making use case.
"""
from dataclasses import dataclass
from typing import List, Tuple

Landmark = Tuple[float, float, float]

HEURISTIC_VERSION = "expression-heuristic-1.0.0"

MOUTH_CORNERS = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)
LEFT_EYEBROW_TOP = 105
LEFT_EYE_TOP = 159
FACE_WIDTH_REF = (234, 454)

_MOUTH_OPEN_RATIO_THRESHOLD = 0.055
_EYEBROW_RAISE_RATIO_THRESHOLD = 0.065
_SMILE_WIDTH_RATIO_THRESHOLD = 0.42


def _dist(a: Landmark, b: Landmark) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass
class ExpressionEstimate:
    label: str  # "neutral" | "happy" | "surprised"
    confidence: float


def estimate_expression(landmarks: List[Landmark]) -> ExpressionEstimate:
    if len(landmarks) < 468:
        raise ValueError("expression estimation requires the full 468-point face mesh")

    face_width = _dist(landmarks[FACE_WIDTH_REF[0]], landmarks[FACE_WIDTH_REF[1]])
    if face_width == 0:
        return ExpressionEstimate(label="neutral", confidence=0.5)

    mouth_open_ratio = _dist(landmarks[MOUTH_TOP_BOTTOM[0]], landmarks[MOUTH_TOP_BOTTOM[1]]) / face_width
    mouth_width_ratio = _dist(landmarks[MOUTH_CORNERS[0]], landmarks[MOUTH_CORNERS[1]]) / face_width
    eyebrow_raise_ratio = _dist(landmarks[LEFT_EYEBROW_TOP], landmarks[LEFT_EYE_TOP]) / face_width

    if mouth_open_ratio > _MOUTH_OPEN_RATIO_THRESHOLD and eyebrow_raise_ratio > _EYEBROW_RAISE_RATIO_THRESHOLD:
        excess = mouth_open_ratio - _MOUTH_OPEN_RATIO_THRESHOLD
        confidence = round(min(1.0, 0.5 + excess * 10), 3)
        return ExpressionEstimate(label="surprised", confidence=confidence)

    if mouth_width_ratio > _SMILE_WIDTH_RATIO_THRESHOLD:
        excess = mouth_width_ratio - _SMILE_WIDTH_RATIO_THRESHOLD
        confidence = round(min(1.0, 0.5 + excess * 10), 3)
        return ExpressionEstimate(label="happy", confidence=confidence)

    slack = _SMILE_WIDTH_RATIO_THRESHOLD - mouth_width_ratio
    confidence = round(min(1.0, 0.5 + slack * 10), 3)
    return ExpressionEstimate(label="neutral", confidence=confidence)
