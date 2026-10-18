"""Deterministic gaze-direction heuristic built on top of face-mesh landmarks.

This is intentionally NOT a separately trained classifier: it is explainable
post-processing geometry applied to the pretrained landmark detector's
output, which keeps the whole pipeline a single packaged/versioned model
(see MLOPS_NOTES.md). It estimates horizontal gaze by comparing each iris
center's position to its eye corners.

Limitations (also documented in README.md):
* Approximate and not clinically validated - do not use for anything safety
  or medical related.
* Assumes a single, roughly front-facing face; large head yaw will skew the
  ratio even if the eyes themselves are centered.
* Direction is relative to the raw camera frame, not mirror-corrected for
  how the frontend may display the preview to the user.
"""
from dataclasses import dataclass
from typing import List, Tuple

Landmark = Tuple[float, float, float]

HEURISTIC_VERSION = "gaze-heuristic-1.0.0"

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)

# Ratio thresholds along the eye-corner axis: 0.0 = outer corner, 1.0 = inner corner.
_LEFT_BOUND = 0.42
_RIGHT_BOUND = 0.58


@dataclass
class GazeEstimate:
    direction: str  # "left" | "center" | "right"
    confidence: float


def _iris_ratio(landmarks: List[Landmark], iris_indices: range, corner_indices: Tuple[int, int]) -> float:
    iris_xs = [landmarks[i][0] for i in iris_indices]
    iris_x = sum(iris_xs) / len(iris_xs)
    outer_x, inner_x = landmarks[corner_indices[0]][0], landmarks[corner_indices[1]][0]
    span = inner_x - outer_x
    if span == 0:
        return 0.5
    return (iris_x - outer_x) / span


def estimate_gaze(landmarks: List[Landmark]) -> GazeEstimate:
    """landmarks must contain at least 478 points (face mesh + iris refinement)."""
    if len(landmarks) < 478:
        raise ValueError("gaze estimation requires iris landmarks (refine_landmarks=True)")

    left_ratio = _iris_ratio(landmarks, LEFT_IRIS, LEFT_EYE_CORNERS)
    right_ratio = _iris_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS)
    ratio = (left_ratio + right_ratio) / 2

    if ratio < _LEFT_BOUND:
        direction = "left"
        confidence = min(1.0, (_LEFT_BOUND - ratio) / _LEFT_BOUND)
    elif ratio > _RIGHT_BOUND:
        direction = "right"
        confidence = min(1.0, (ratio - _RIGHT_BOUND) / (1 - _RIGHT_BOUND))
    else:
        direction = "center"
        mid = (_LEFT_BOUND + _RIGHT_BOUND) / 2
        half_band = (_RIGHT_BOUND - _LEFT_BOUND) / 2
        confidence = 1.0 - min(1.0, abs(ratio - mid) / half_band)

    # Floor confidence so a borderline reading never reports as 0.0.
    confidence = round(max(confidence, 0.5), 3)
    return GazeEstimate(direction=direction, confidence=confidence)
