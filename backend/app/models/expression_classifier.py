"""Pretrained image-classification model for facial expression.

Model: dima806/facial_emotions_image_detection - a ViT-base model
fine-tuned for 7-class facial emotion classification (apache-2.0 license),
pinned to an exact, immutable commit revision so the Docker build and every
prediction it produces are fully reproducible (see MLOPS_NOTES.md
"Identification" and "Packaging and loading").

Design decisions:
* Loaded once at process startup into a module-level singleton, mirroring
  FaceMeshModel - never reconstructed per request.
* Weights are pre-downloaded at *Docker build time* (see
  scripts/prefetch_model.py and the Dockerfile) into a fixed HF_HOME cache
  path, and HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set at runtime so the
  running container never makes a network call to load or use this model -
  a cold start never depends on Hugging Face Hub being reachable.
* HF's `pipeline()` object is not documented as safe for concurrent calls
  from multiple threads on one instance, so calls are serialized behind a
  lock - the same reliability trade-off as FaceMeshModel (see its
  docstring, and the "operational risks" section of README.md).
* The pipeline returns independent sigmoid confidences per class - they do
  NOT sum to 1 like a softmax would. Only the top-1 label and its own score
  are surfaced; treat that score as "how confident the model is in this
  specific label", not as a probability share across all classes.
"""
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "dima806/facial_emotions_image_detection"
MODEL_REVISION = "747cf16692eea925b54b7b543cf436848128b68d"


@dataclass
class ExpressionPrediction:
    label: str
    confidence: float
    inference_time_ms: float


class ExpressionClassifier:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline = pipeline(
            "image-classification",
            model=MODEL_NAME,
            revision=MODEL_REVISION,
            top_k=1,
        )
        self.version = MODEL_REVISION[:12]
        logger.info("model loaded name=%s revision=%s", MODEL_NAME, self.version)

    def predict(self, face_crop_rgb: np.ndarray) -> ExpressionPrediction:
        image = Image.fromarray(face_crop_rgb)
        start = time.perf_counter()
        with self._lock:
            result = self._pipeline(image)
        elapsed_ms = (time.perf_counter() - start) * 1000
        top = result[0]
        return ExpressionPrediction(
            label=top["label"],
            confidence=round(float(top["score"]), 3),
            inference_time_ms=elapsed_ms,
        )
