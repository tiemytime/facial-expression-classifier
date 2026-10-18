"""Pydantic request/response contracts for the public API."""
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Liveness: 'ok' if the process is up and serving requests.")
    app_env: str


class ReadyResponse(BaseModel):
    status: str = Field(description="Readiness: 'ok' only once the model is loaded and able to infer.")
    model_loaded: bool
    model_name: str
    model_version: str


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
    inference_time_ms: float
    model_name: str
    model_version: str
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: str
