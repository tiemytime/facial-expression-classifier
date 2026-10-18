"""Pydantic request/response contracts for the public API."""
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Liveness: 'ok' if the process is up and serving requests.")
    app_env: str


class ModelInfo(BaseModel):
    name: str
    version: str
    task: str = Field(description="One of: landmark_detection, image_classification.")


class ReadyResponse(BaseModel):
    status: str = Field(description="Readiness: 'ok' only once both models are loaded and able to infer.")
    model_loaded: bool
    models: list[ModelInfo] = Field(default_factory=list)


class GazeResult(BaseModel):
    direction: str = Field(description="One of: left, center, right.")
    confidence: float = Field(ge=0.0, le=1.0)


class ExpressionResult(BaseModel):
    label: str = Field(description="One of: neutral, happy, surprised.")
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    face_detected: bool
    landmark_count: int = Field(description="Number of facial landmarks returned, 0 if no face.")
    gaze: Optional[GazeResult] = None
    expression: Optional[ExpressionResult] = None
    inference_time_ms: float = Field(
        description="Total model time: face-mesh detection plus, if a face was found, expression classification."
    )
    models: list[ModelInfo]
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: str
