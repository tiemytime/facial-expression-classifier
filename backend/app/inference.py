"""Orchestrates one prediction: decode image -> face mesh -> gaze + expression."""
import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.models.expression import HEURISTIC_VERSION as EXPRESSION_VERSION
from app.models.expression import estimate_expression
from app.models.face_mesh_model import MODEL_NAME, FaceMeshModel
from app.models.gaze import HEURISTIC_VERSION as GAZE_VERSION
from app.models.gaze import estimate_gaze
from app.schemas import ExpressionResult, GazeResult, PredictionResponse

logger = logging.getLogger(__name__)


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


def run_prediction(model: FaceMeshModel, raw_bytes: bytes, request_id: str = "-") -> PredictionResponse:
    image_rgb = decode_image(raw_bytes)
    result = model.predict(image_rgb)

    full_model_version = f"{model.version}+{GAZE_VERSION}+{EXPRESSION_VERSION}"

    if not result.face_detected:
        logger.info(
            "prediction face_detected=false time_ms=%.1f",
            result.inference_time_ms,
            extra={"request_id": request_id},
        )
        return PredictionResponse(
            face_detected=False,
            landmark_count=0,
            gaze=None,
            expression=None,
            inference_time_ms=round(result.inference_time_ms, 2),
            model_name=MODEL_NAME,
            model_version=full_model_version,
            warnings=["no face detected in the submitted frame"],
        )

    warnings: list[str] = []
    gaze_result = None
    expression_result = None

    try:
        gaze = estimate_gaze(result.landmarks)
        gaze_result = GazeResult(direction=gaze.direction, confidence=gaze.confidence)
    except ValueError as exc:
        warnings.append(f"gaze estimation skipped: {exc}")

    try:
        expression = estimate_expression(result.landmarks)
        expression_result = ExpressionResult(label=expression.label, confidence=expression.confidence)
    except ValueError as exc:
        warnings.append(f"expression estimation skipped: {exc}")

    logger.info(
        "prediction face_detected=true time_ms=%.1f gaze=%s expression=%s",
        result.inference_time_ms,
        gaze_result.direction if gaze_result else None,
        expression_result.label if expression_result else None,
        extra={"request_id": request_id},
    )

    return PredictionResponse(
        face_detected=True,
        landmark_count=len(result.landmarks),
        gaze=gaze_result,
        expression=expression_result,
        inference_time_ms=round(result.inference_time_ms, 2),
        model_name=MODEL_NAME,
        model_version=full_model_version,
        warnings=warnings,
    )
