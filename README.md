# Lifeverse Facial Feature Tracker

A small ML-backed web application built for the Lifeverse DevOps/MLOps
Engineer technical assessment (`context/Lifeverse_DevOps_MLOps_Technical_Assessment_VF26.pdf`).
A user's live webcam feed is sent to a backend API, which runs it through
two packaged computer-vision models and returns an understandable
prediction: whether a face was found, which way it's looking (gaze), and
its expression.

> **Scope note on the ML task.** The assessment's suggested default is a
> lightweight pretrained image-classification model (e.g. from the PyTorch
> Vision catalogue); it explicitly allows an equivalent alternative task
> "if you agree it with us before you start." This submission's
> **expression** output *is* exactly that default: a genuine pretrained
> image-classification model (`dima806/facial_emotions_image_detection`,
> see MLOPS_NOTES.md), just applied to faces instead of general objects.
> The **facial landmark detection + gaze** layer on top (MediaPipe Face
> Mesh) is additional, and is the part that counts as an "alternative
> task" needing prior agreement - on the assumption that's been given. If
> it hasn't, the landmark/gaze layer can be dropped without touching the
> classification model or any of the Docker/CI/observability work, which
> is the actual priority of this assessment (see `context/HOW_TO_PROCEED.md`).

## Expected flow

```
User (webcam) -> frontend (static HTML/JS) -> backend API (FastAPI)
              -> packaged model (MediaPipe Face Mesh)
              -> gaze + expression heuristics -> JSON prediction -> overlay
```

The frontend never imports or loads the model - it only ever calls the
backend over HTTP (`GET /health`, `GET /ready`, `POST /predict`).

## Repository structure

```
backend/            FastAPI app, model wrappers, tests
  app/
    main.py           routes, middleware, lifespan (load/close both models)
    config.py         env-var driven settings
    inference.py      decode image -> face mesh -> crop -> gaze + classifier
    models/
      face_mesh_model.py        MediaPipe Face Mesh wrapper (loaded once)
      gaze.py                    gaze-direction heuristic (landmark geometry)
      expression_classifier.py  pretrained ViT expression classifier (loaded once)
  scripts/
    prefetch_model.py  bakes classifier weights into the image at build time
  tests/              pytest: API tests + pure-function + model contract tests
  Dockerfile
frontend/           Static HTML/CSS/JS webcam client (no build step)
  Dockerfile          nginx (non-root), templates config.js from env at start
deploy/huggingface/  Combined single-container image for deployment
docker-compose.yml   Local dev: frontend + backend together
.github/workflows/   CI: lint, test, build images, smoke-test
context/
  Lifeverse_DevOps_MLOps_Technical_Assessment_VF26.pdf   the assessment brief
  HOW_TO_PROCEED.md    roadmap for extending this scaffold
MLOPS_NOTES.md       model packaging, versioning, update/rollback plan
INCIDENT_NOTES.md    troubleshooting exercise write-up
```

## Prerequisites

- Docker + Docker Compose (recommended path), **or**
- Python 3.11 for running the backend without Docker (mediapipe wheels are
  published for 3.9-3.12; 3.11 is what the Dockerfile pins)
- A webcam and a browser (camera access requires HTTPS or `localhost`)

## Run locally with Docker Compose (recommended)

```bash
cp .env.example .env   # optional for compose; documents every variable
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:8000 (try http://localhost:8000/health)

Click **Start camera**, grant permission, and watch the status panel update
roughly once per second.

## Run locally without Docker

```bash
# Backend
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-torch.txt   # CPU-only torch, from its own index
pip install -r requirements-dev.txt     # must be a separate command - see
                                         # the comment in requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend, in a second terminal
cd frontend
python3 -m http.server 8080
```

The committed `frontend/config.js` already points at `http://localhost:8000`
for this exact workflow.

## Tests and linting

```bash
cd backend
source .venv/bin/activate
ruff check app        # code-quality check
pytest -q             # 14 tests: API behaviour + pure-function unit tests
```

## Building the Docker images manually

```bash
docker build -f backend/Dockerfile  -t lifeverse-backend  backend
docker build -f frontend/Dockerfile -t lifeverse-frontend frontend

# Combined single-container image (used for deployment),
# built from the repo root:
docker build -f deploy/huggingface/Dockerfile -t lifeverse-combined .
```

## Configuration

All environment-specific values are read from environment variables (see
`.env.example`), never hard-coded.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `APP_ENV` | `development` | backend | Label surfaced in `/health`; informs log verbosity conventions. |
| `LOG_LEVEL` | `INFO` | backend | Python logging level. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | backend | Bind address for uvicorn. |
| `CORS_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` | backend | Comma-separated allow-list for the standalone (two-container) deployment. Ignored in combined mode. |
| `SERVE_FRONTEND` | `false` | backend | `true` mounts `STATIC_DIR` at `/` so one container serves both the API and the frontend (used by the combined deployment Dockerfile at `deploy/huggingface/Dockerfile`). |
| `STATIC_DIR` | `static` | backend | Directory to serve when `SERVE_FRONTEND=true`. |
| `MAX_IMAGE_SIZE_MB` | `8` | backend | Rejects oversized uploads with `413`. |
| `MIN_DETECTION_CONFIDENCE` / `MIN_TRACKING_CONFIDENCE` | `0.5` | backend | MediaPipe Face Mesh thresholds. |
| `MAX_NUM_FACES` | `1` | backend | Face Mesh is configured for a single face; see limitations. |
| `API_BASE_URL` | `http://localhost:8000` | frontend container only | Rendered into `config.js` at container start via `envsubst`. **Resolved by the user's browser, not by the container** - see the callout in `docker-compose.yml` and `INCIDENT_NOTES.md`. |

No secrets exist in this project (no API keys, no database, no third-party
credentials) - there is nothing that needs a secret manager today. If one is
ever introduced, it must be injected via environment variables at deploy
time, never committed, matching the pattern already used for every other
runtime setting.

## API reference

### `GET /health`
Liveness only - "the process is up and serving HTTP." Does **not** imply the
model is ready; see `/ready`.

```json
{ "status": "ok", "app_env": "development" }
```

### `GET /ready`
Readiness - both models are loaded and able to run inference. A load
balancer or orchestrator should gate traffic on this, not `/health`, during
startup.

```json
{
  "status": "ok",
  "model_loaded": true,
  "models": [
    { "name": "mediapipe-face-mesh", "version": "mediapipe-0.10.14", "task": "landmark_detection" },
    { "name": "dima806/facial_emotions_image_detection", "version": "747cf16692ee", "task": "image_classification" }
  ]
}
```

### `POST /predict`
Multipart form upload, field name `file`, content type one of
`image/jpeg`, `image/png`, `image/webp`.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.jpg;type=image/jpeg"
```

**Example successful prediction** - this is real, verified output from a
`TestClient` run against a clearly visible, front-facing, smiling face
(steady-state timing; see "cold start" note below):

```json
{
  "face_detected": true,
  "landmark_count": 478,
  "gaze": { "direction": "center", "confidence": 0.5 },
  "expression": { "label": "happy", "confidence": 0.981 },
  "inference_time_ms": 268.4,
  "models": [
    { "name": "mediapipe-face-mesh", "version": "mediapipe-0.10.14", "task": "landmark_detection" },
    { "name": "dima806/facial_emotions_image_detection", "version": "747cf16692ee", "task": "image_classification" }
  ],
  "warnings": []
}
```

> **Cold start.** The *first* `/predict` call after a fresh process start
> took ~2.5s in testing (torch/ViT warm-up); every call after that on the
> same process was ~250-300ms on a CPU. Don't use the first post-deploy
> request as your latency baseline - see "Observability" below.

**No-face example** (what the blank-image test in `tests/test_predict.py`
asserts on):

```json
{
  "face_detected": false,
  "landmark_count": 0,
  "gaze": null,
  "expression": null,
  "inference_time_ms": 8.1,
  "models": [
    { "name": "mediapipe-face-mesh", "version": "mediapipe-0.10.14", "task": "landmark_detection" },
    { "name": "dima806/facial_emotions_image_detection", "version": "747cf16692ee", "task": "image_classification" }
  ],
  "warnings": ["no face detected in the submitted frame"]
}
```

**Error handling**

| Condition | Status | Body |
|---|---|---|
| Unsupported content type | `400` | `{"error": "...", "detail": "unsupported content type ..."}` |
| Empty file | `400` | `detail: "empty file"` |
| File larger than `MAX_IMAGE_SIZE_MB` | `413` | `detail: "file exceeds ...MB limit"` |
| Bytes that don't decode as an image | `400` | `detail: "uploaded file is not a valid image"` |
| Model not yet loaded | `503` | `detail: "model is not ready yet"` |
| Unexpected inference failure | `500` | `detail: "inference failed, see server logs"` (never leaks internals to the client; full traceback goes to server logs only) |

## Model, expected inputs, and limitations

This pipeline uses **two** pretrained models, each doing one job:

- **MediaPipe Face Mesh** (468 landmarks + iris refinement) - landmark
  detection, shipped inside the `mediapipe` pip package, no separate weight
  download. Its output also drives the **gaze** direction, which is
  deterministic geometry computed from the landmarks (see
  `backend/app/models/gaze.py`) - not a second trained model.
- **`dima806/facial_emotions_image_detection`** (ViT-base, apache-2.0,
  pinned to commit `747cf16692ee`) - a genuine trained image-classification
  model, run on a crop of the detected face, producing the **expression**
  label (`sad`/`disgust`/`angry`/`neutral`/`fear`/`surprise`/`happy`) and
  its confidence. Its weights are pre-downloaded at Docker build time (see
  `backend/scripts/prefetch_model.py`) so no network access happens at
  inference time either. Full rationale and versioning in `MLOPS_NOTES.md`.
- **Expected input**: a single RGB image (JPEG/PNG/WebP) containing at most
  one reasonably front-facing, well-lit human face, under `MAX_IMAGE_SIZE_MB`.
- **Known limitations**:
  - Single face only - `max_num_faces=1`; extra faces are ignored.
  - Sensitive to poor lighting, strong head angles/profile views, and
    occlusion (glasses glare, masks, hair).
  - Gaze direction is a hand-tuned heuristic, not trained or validated
    against a labeled dataset - don't treat it as an accurate attention
    measurement, and it's relative to the raw camera frame, not corrected
    for any mirroring the frontend preview may apply.
  - The expression classifier's confidence scores are independent sigmoid
    outputs per class, not a softmax - they don't sum to 1 across classes;
    only the top-1 label and its own score are surfaced.
  - The expression classifier was trained on a public FER-style dataset,
    not validated against our usage - its label set and accuracy are
    inherited from the upstream model card, not benchmarked here.
  - Webcam capture requires a secure context (HTTPS or `localhost`) per
    browser policy.
  - The frontend polls snapshots (default every 1s) rather than streaming
    video, trading real-time smoothness for a plain static frontend with no
    streaming server of its own.

## Observability

- **Logs**: structured single-line logs to stdout for startup, every
  request (method, path, status, timing, request ID), and inference
  outcomes (face detected, labels, timing). Uploaded image bytes, raw
  filenames, and request bodies are never logged.
- **Health vs. readiness**: `/health` (liveness) is intentionally separate
  from `/ready` (model loaded) - see API reference above.
- **Timing**: every response carries `X-Response-Time-Ms` and
  `X-Request-ID` headers, and `/predict` additionally reports
  `inference_time_ms` in the body (model inference only, excluding decode
  and network time).
- **Investigating elevated latency or repeated failures**: start from
  `X-Request-ID` to correlate a slow/failed browser request with its exact
  server log line; compare `inference_time_ms` (model-only) against the
  total `X-Response-Time-Ms` (includes image decode and the two models'
  lock wait time) to tell "model is slow" apart from "requests are queued
  behind a lock" (see risk below); expect the *first* `/predict` after
  every container start to be ~2.5s (torch/ViT warm-up) and don't mistake
  that for a regression; check `docker stats` / host metrics for CPU
  saturation or memory pressure; check `/ready` to rule out a
  crash-looping model load.

## Operational risks

1. **Two shared-instance inference locks become a throughput bottleneck.**
   Both `FaceMeshModel` and `ExpressionClassifier` are single instances
   each guarded by their own `threading.Lock` (see their docstrings for
   why). Under concurrent load, requests queue up behind whichever is
   slower - the classifier, at ~250ms/call, dominates - inflating
   `X-Response-Time-Ms` even though `inference_time_ms` per call stays flat.
   *Detect*: a growing gap between `X-Response-Time-Ms` and
   `inference_time_ms` under load.
   *Mitigate*: run a small pool of instances behind a semaphore, or scale
   out with multiple worker processes (each gets its own model instances -
   neither mediapipe's nor torch's objects should be shared across
   processes anyway).

2. **Memory and image-size footprint under concurrent image processing.**
   mediapipe/OpenCV hold native buffers per in-flight frame, and the ViT
   classifier (~350MB of weights plus torch's own overhead) is noticeably
   heavier than mediapipe alone; a burst of large uploads near
   `MAX_IMAGE_SIZE_MB` from many clients at once can spike RSS and trigger
   an OOM kill on memory-constrained free-tier hosts.
   *Detect*: container restarts correlated with memory-limit-reached
   events in host/platform logs.
   *Mitigate*: keep `MAX_IMAGE_SIZE_MB` conservative (default 8), cap
   concurrent workers to what the host's memory tier can hold, and alert
   on restart count.

3. **Webcam frames leave the browser over the network on every capture.**
   Frames are sent to the backend for inference by design, but that means
   any plaintext transport or overly permissive `CORS_ORIGINS` would expose
   live camera captures to unintended origins.
   *Detect*: review `CORS_ORIGINS` on every deploy; confirm the deployed
   URL only serves over HTTPS.
   *Mitigate*: scope `CORS_ORIGINS` to the exact deployed frontend origin
   in non-local environments, and never log raw image bytes (already
   enforced - see Observability).

4. **No face, a profile view, or poor lighting silently degrades to "no
   prediction" rather than failing loudly.** This is correct behaviour
   (`face_detected: false` is not an error), but a client that doesn't
   surface it to the user could look like a frozen or broken feature.
   *Detect*: a sustained high ratio of `face_detected: false` responses.
   *Mitigate*: the frontend already renders this state explicitly (red
   border, "Face detected: no") rather than staying silent.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request:
1. Install backend dependencies.
2. Lint with `ruff`.
3. Run the pytest suite (API tests + pure-function unit tests).
4. Build both Docker images and smoke-test the backend container's
   `/health` endpoint.

**What would trigger a deployment.** None of the above pushes an image or
deploys anywhere automatically - deployment is a deliberate, manual step,
gated on CI being green on `main`. In a team setting, the natural next step
is: a merge to `main` that passes CI triggers a job that pushes the image
to a registry and triggers the host's deploy (on Render, a `git push` to
the connected branch **is** the deploy trigger).

**How configuration differs between environments.** The exact same image
runs in every environment; only environment variables change
(`CORS_ORIGINS`, `SERVE_FRONTEND`, confidence thresholds, log level). Dev
uses `.env` / compose environment blocks; the hosted environment (Render)
sets the same variables as Render environment variables. Nothing
environment-specific is hard-coded in source, by design (see the
Configuration table above).

**How to roll back a faulty release.**
- *Render*: Render keeps a history of previous deploys per service and lets
  you roll back to a prior deploy from the dashboard, or `git revert` the
  bad commit and push to trigger a fresh deploy.
- *Generic registry-based deploy*: redeploy the previous image tag rather
  than rebuilding from source, so rollback doesn't depend on the build
  succeeding again.
- In both cases: roll back first to restore service, investigate the root
  cause afterwards (see `INCIDENT_NOTES.md` for the full reasoning).

## Troubleshooting exercise

See `INCIDENT_NOTES.md`.

## MLOps notes

See `MLOPS_NOTES.md` for model packaging, identification, versioning,
update/validation/rollback, and the inference-behaviour test beyond a
plain HTTP-200 check.
