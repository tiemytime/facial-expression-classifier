"""Pretrained facial landmark model wrapper (MediaPipe Face Mesh).

Design decisions (see MLOPS_NOTES.md for the full rationale):

* Loaded exactly once, at process startup, into a module-level singleton.
  It is never re-instantiated per request: constructing `FaceMesh()` loads
  the underlying TFLite graph from disk, which is the exact "reload the
  model on every request" cost the assessment asks us to avoid.
* `static_image_mode=True` is a deliberate reliability choice, not a
  performance default. MediaPipe's tracking mode (`False`) keeps internal
  per-stream state between calls to smooth landmarks across sequential
  video frames from a *single* source. Our API is a stateless HTTP
  endpoint that may interleave frames from multiple concurrent browser
  sessions on the same process; sharing tracker state across unrelated
  callers would corrupt predictions. Treating every request as an
  independent image is slightly more expensive per call but correct.
* The underlying `FaceMesh.process()` call is not documented as
  thread-safe for concurrent invocation on one instance, so every call is
  serialized behind `self._lock`. Under concurrent load this becomes a
  bottleneck - see the "Operational risks" section of README.md for the
  detection/mitigation plan (e.g. a small pool of instances behind a
  semaphore, or multiple worker processes).
* No network access happens during inference: the model graph ships
  inside the `mediapipe` pip package, so builds and predictions are
  reproducible offline.
"""
import logging
import threading
import time
from dataclasses import dataclass

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "mediapipe-face-mesh"

# Landmark indices used by the heuristics in gaze.py / expression.py.
# Indices come from MediaPipe's documented 468-point face mesh topology
# (+10 iris points 468-477 when refine_landmarks=True).
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)
MOUTH_CORNERS = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)
LEFT_EYEBROW_TOP = 105
LEFT_EYE_TOP = 159
FACE_WIDTH_REF = (234, 454)


@dataclass
class FaceMeshPrediction:
    face_detected: bool
    landmarks: list  # list of (x, y, z) normalized coordinates, empty if no face
    inference_time_ms: float


class FaceMeshModel:
    """Thread-safe, load-once wrapper around mp.solutions.face_mesh.FaceMesh."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._lock = threading.Lock()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.version = f"mediapipe-{mp.__version__}"
        logger.info("model loaded name=%s version=%s", MODEL_NAME, self.version)

    def predict(self, image_rgb: np.ndarray) -> FaceMeshPrediction:
        start = time.perf_counter()
        with self._lock:
            results = self._face_mesh.process(image_rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not results.multi_face_landmarks:
            return FaceMeshPrediction(face_detected=False, landmarks=[], inference_time_ms=elapsed_ms)

        landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]
        return FaceMeshPrediction(face_detected=True, landmarks=landmarks, inference_time_ms=elapsed_ms)

    def close(self) -> None:
        self._face_mesh.close()
