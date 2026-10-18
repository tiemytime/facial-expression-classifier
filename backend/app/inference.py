"""Orchestrates one prediction: decode image -> face mesh -> crop -> gaze
heuristic + expression classifier.
"""
import io
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.models.expression_classifier import MODEL_NAME as EXPRESSION_MODEL_NAME
from app.models.expression_classifier import ExpressionClassifier
from app.models.face_mesh_model import MODEL_NAME as FACE_MESH_MODEL_NAME
from app.models.face_mesh_model import FaceMeshModel
from app.models.gaze import estimate_gaze
from app.schemas import ExpressionResult, GazeResult, ModelInfo, PredictionResponse

logger = logging.getLogger(__name__)

# Padding added around the landmark bounding box before handing the crop to
# the expression classifier - the classifier was trained on cropped face
# images, not full frames, and a tight box (zero margin) tends to clip chin
# and forehead.
_FACE_CROP_MARGIN = 0.25


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


def decode_image(raw_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("uploaded file is not a valid image") from exc
    return np.array(image)


def _crop_face(
    image_rgb: np.ndarray,
    landmarks: List[Tuple[float, float, float]],
    margin: float = _FACE_CROP_MARGIN,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    xs = [lm[0] for lm in landmarks]
    ys = [lm[1] for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_w, box_h = x_max - x_min, y_max - y_min

    x_min = max(0.0, x_min - box_w * margin)
    x_max = min(1.0, x_max + box_w * margin)
    y_min = max(0.0, y_min - box_h * margin)
    y_max = min(1.0, y_max + box_h * margin)

    # The max(..., +1) floor guarantees at least a 1px-wide/tall slice even
    # for a degenerate (zero-area) landmark box, so this never returns an
    # empty array.
    px0, px1 = int(x_min * width), max(int(x_max * width), int(x_min * width) + 1)
    py0, py1 = int(y_min * height), max(int(y_max * height), int(y_min * height) + 1)
    return image_rgb[py0:py1, px0:px1]


def run_prediction(
    face_mesh: FaceMeshModel,
    expression_classifier: ExpressionClassifier,
    raw_bytes: bytes,
    request_id: str = "-",
) -> PredictionResponse:
    image_rgb = decode_image(raw_bytes)
    mesh_result = face_mesh.predict(image_rgb)

    models = [
        ModelInfo(name=FACE_MESH_MODEL_NAME, version=face_mesh.version, task="landmark_detection"),
        ModelInfo(name=EXPRESSION_MODEL_NAME, version=expression_classifier.version, task="image_classification"),
    ]

    if not mesh_result.face_detected:
        logger.info(
            "prediction face_detected=false time_ms=%.1f",
            mesh_result.inference_time_ms,
            extra={"request_id": request_id},
        )
        return PredictionResponse(
            face_detected=False,
            landmark_count=0,
            gaze=None,
            expression=None,
            inference_time_ms=round(mesh_result.inference_time_ms, 2),
            models=models,
            warnings=["no face detected in the submitted frame"],
        )

    warnings: list[str] = []
    gaze_result = None
    try:
        gaze = estimate_gaze(mesh_result.landmarks)
        gaze_result = GazeResult(direction=gaze.direction, confidence=gaze.confidence)
    except ValueError as exc:
        warnings.append(f"gaze estimation skipped: {exc}")

    face_crop = _crop_face(image_rgb, mesh_result.landmarks)
    expression_prediction = expression_classifier.predict(face_crop)
    expression_result = ExpressionResult(label=expression_prediction.label, confidence=expression_prediction.confidence)

    total_time_ms = mesh_result.inference_time_ms + expression_prediction.inference_time_ms

    logger.info(
        "prediction face_detected=true time_ms=%.1f gaze=%s expression=%s",
        total_time_ms,
        gaze_result.direction if gaze_result else None,
        expression_result.label,
        extra={"request_id": request_id},
    )

    return PredictionResponse(
        face_detected=True,
        landmark_count=len(mesh_result.landmarks),
        gaze=gaze_result,
        expression=expression_result,
        inference_time_ms=round(total_time_ms, 2),
        models=models,
        warnings=warnings,
    )
