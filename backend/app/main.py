"""FastAPI application entrypoint.

Request flow: frontend -> this API -> MediaPipe Face Mesh (landmark
detection) -> gaze heuristic + pretrained expression classifier -> JSON
prediction. The frontend never loads a model itself; it only ever calls
GET /health, GET /ready and POST /predict over HTTP.
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.inference import InvalidImageError, run_prediction
from app.logging_config import configure_logging
from app.models.expression_classifier import MODEL_NAME as EXPRESSION_MODEL_NAME
from app.models.expression_classifier import ExpressionClassifier
from app.models.face_mesh_model import MODEL_NAME as FACE_MESH_MODEL_NAME
from app.models.face_mesh_model import FaceMeshModel
from app.schemas import ErrorResponse, HealthResponse, ModelInfo, PredictionResponse, ReadyResponse

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

_state: dict = {"face_mesh": None, "expression_classifier": None, "started_at": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application startup app_env=%s", settings.app_env)
    _state["face_mesh"] = FaceMeshModel(
        max_num_faces=settings.max_num_faces,
        min_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
    )
    _state["expression_classifier"] = ExpressionClassifier()
    _state["started_at"] = time.time()
    yield
    logger.info("application shutdown")
    if _state["face_mesh"] is not None:
        _state["face_mesh"].close()


app = FastAPI(title="Lifeverse Facial Feature Tracker", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error path=%s", request.url.path, extra={"request_id": request_id})
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "request method=%s path=%s status=%s time_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        extra={"request_id": request_id},
    )
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe: the process is up. Does not guarantee the model is ready."""
    return HealthResponse(status="ok", app_env=settings.app_env)


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness probe: both models are loaded and able to serve predictions."""
    face_mesh = _state["face_mesh"]
    classifier = _state["expression_classifier"]
    if face_mesh is None or classifier is None:
        return ReadyResponse(status="not_ready", model_loaded=False, models=[])
    models = [
        ModelInfo(name=FACE_MESH_MODEL_NAME, version=face_mesh.version, task="landmark_detection"),
        ModelInfo(name=EXPRESSION_MODEL_NAME, version=classifier.version, task="image_classification"),
    ]
    return ReadyResponse(status="ok", model_loaded=True, models=models)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile) -> PredictionResponse:
    request_id = getattr(request.state, "request_id", "-")
    face_mesh = _state["face_mesh"]
    classifier = _state["expression_classifier"]
    if face_mesh is None or classifier is None:
        logger.warning("predict rejected reason=model_not_ready", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="model is not ready yet")

    if file.content_type not in settings.allowed_content_type_list:
        allowed = settings.allowed_content_type_list
        logger.warning(
            "predict rejected reason=unsupported_content_type content_type=%s",
            file.content_type,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=400,
            detail=f"unsupported content type '{file.content_type}', expected one of {allowed}",
        )

    raw_bytes = await file.read()
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        logger.warning(
            "predict rejected reason=file_too_large size_bytes=%d", len(raw_bytes), extra={"request_id": request_id}
        )
        raise HTTPException(status_code=413, detail=f"file exceeds {settings.max_image_size_mb}MB limit")
    if len(raw_bytes) == 0:
        logger.warning("predict rejected reason=empty_file", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail="empty file")

    logger.info(
        "predict received content_type=%s size_bytes=%d",
        file.content_type,
        len(raw_bytes),
        extra={"request_id": request_id},
    )

    try:
        return run_prediction(face_mesh, classifier, raw_bytes, request_id=request_id)
    except InvalidImageError as exc:
        logger.warning("predict rejected reason=invalid_image", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # inference-time failure, not a client input problem
        logger.exception("inference error", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="inference failed, see server logs") from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.__class__.__name__, detail=str(exc.detail)).model_dump(),
    )


if settings.serve_frontend and os.path.isdir(settings.static_dir):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
    logger.info("serving frontend from %s (combined single-container mode)", settings.static_dir)
