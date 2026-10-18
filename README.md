# Lifeverse Facial Feature Tracker

A small ML-backed web application built for the Lifeverse DevOps/MLOps
Engineer technical assessment (`context/Lifeverse_DevOps_MLOps_Technical_Assessment_VF26.pdf`).
A user's live webcam feed is sent to a backend API, which runs it through a
packaged computer-vision model and returns an understandable prediction:
whether a face was found, which way it's looking (gaze), and its expression.

> **Scope note on the ML task.** The assessment's suggested default is a
> generic lightweight image classifier (e.g. from the PyTorch Vision
> catalogue); it explicitly allows an equivalent alternative task "if you
> agree it with us before you start." This submission implements facial
> landmark tracking (MediaPipe Face Mesh) with gaze/expression heuristics on
> top, on the assumption that's been agreed. If it hasn't, the pretrained
> model can be swapped for a plain torchvision classifier without touching
> the Docker/CI/observability work, which is the actual priority of this
> assessment (see `context/HOW_TO_PROCEED.md`).

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
backend/            FastAPI app, model wrapper, heuristics, tests
  app/
    main.py           routes, middleware, lifespan (model load/close)
    config.py         env-var driven settings
    inference.py      decode image -> model -> heuristics -> response
    models/
      face_mesh_model.py   MediaPipe Face Mesh wrapper (loaded once)
      gaze.py               gaze-direction heuristic
      expression.py         expression heuristic
  tests/              pytest: API tests + pure-function unit tests
  Dockerfile
frontend/           Static HTML/CSS/JS webcam client (no build step)
  Dockerfile          nginx (non-root), templates config.js from env at start
deploy/huggingface/  Combined single-container image + Space metadata
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
pip install -r requirements-dev.txt
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

# Combined single-container image (used for the Hugging Face deployment),
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
| `SERVE_FRONTEND` | `false` | backend | `true` mounts `STATIC_DIR` at `/` so one container serves both the API and the frontend (used by `deploy/huggingface/Dockerfile`). |
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
Readiness - the model is loaded and able to run inference. A load balancer
or orchestrator should gate traffic on this, not `/health`, during startup.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "mediapipe-face-mesh",
  "model_version": "mediapipe-0.10.14"
}
```

### `POST /predict`
Multipart form upload, field name `file`, content type one of
`image/jpeg`, `image/png`, `image/webp`.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.jpg;type=image/jpeg"
```

**Example successful prediction** (a photo with one clearly visible,
front-facing, smiling face):

```json
{
  "face_detected": true,
  "landmark_count": 478,
  "gaze": { "direction": "center", "confidence": 0.91 },
  "expression": { "label": "happy", "confidence": 0.74 },
  "inference_time_ms": 42.3,
  "model_name": "mediapipe-face-mesh",
  "model_version": "mediapipe-0.10.14+gaze-heuristic-1.0.0+expression-heuristic-1.0.0",
  "warnings": []
}
```

**No-face example** (what the blank-image test in `tests/test_predict.py`
asserts on):

```json
{
  "face_detected": false,
  "landmark_count": 0,
  "gaze": null,
  "expression": null,
  "inference_time_ms": 8.1,
  "model_name": "mediapipe-face-mesh",
  "model_version": "mediapipe-0.10.14+gaze-heuristic-1.0.0+expression-heuristic-1.0.0",
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

- **Model**: MediaPipe Face Mesh (468 landmarks + iris refinement), a
  pretrained model shipped inside the `mediapipe` pip package - no
  separate weight download, no network access at inference time.
- **Derived predictions**: gaze direction and expression are **not** a
  second trained model; they're deterministic geometry computed from the
  landmarks (see `backend/app/models/gaze.py` and `expression.py`). This
  keeps the whole pipeline a single packaged, versioned model. Full
  rationale and versioning in `MLOPS_NOTES.md`.
- **Expected input**: a single RGB image (JPEG/PNG/WebP) containing at most
  one reasonably front-facing, well-lit human face, under `MAX_IMAGE_SIZE_MB`.
- **Known limitations**:
  - Single face only - `max_num_faces=1`; extra faces are ignored.
  - Sensitive to poor lighting, strong head angles/profile views, and
    occlusion (glasses glare, masks, hair).
  - Gaze/expression labels are coarse heuristics tuned by hand, not trained
    or validated against a labeled dataset - do not treat them as accurate
    attention or emotion measurements.
  - Gaze direction is relative to the raw camera frame, not corrected for
    any mirroring the frontend preview may apply.
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
  total `X-Response-Time-Ms` (includes image decode and the shared
  inference lock's wait time) to tell "model is slow" apart from "requests
  are queued behind the lock" (see risk below); check `docker stats` /
  host metrics for CPU saturation or memory pressure; check `/ready` to
  rule out a crash-looping model load.

## Operational risks

1. **Shared-instance inference lock becomes a throughput bottleneck.**
   `FaceMeshModel` is a single instance guarded by one `threading.Lock`
   (see the docstring in `face_mesh_model.py` for why). Under concurrent
   load, requests queue up behind it, inflating `X-Response-Time-Ms` even
   though `inference_time_ms` per call stays flat.
   *Detect*: a growing gap between `X-Response-Time-Ms` and
   `inference_time_ms` under load.
   *Mitigate*: run a small pool of `FaceMeshModel` instances behind a
   semaphore, or scale out with multiple worker processes (each gets its
   own model instance - mediapipe objects shouldn't be shared across
   processes anyway).

2. **Memory footprint under concurrent image processing.** mediapipe +
   OpenCV hold native buffers per in-flight frame; a burst of large
   uploads near `MAX_IMAGE_SIZE_MB` from many clients at once can spike
   RSS and trigger an OOM kill on memory-constrained free-tier hosts.
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
deploys anywhere automatically - deployment here is a deliberate, manual
step (see "Deploying to Hugging Face Spaces" below), gated on CI being
green on `main`. In a team setting, the natural next step is: a merge to
`main` that passes CI triggers a job that pushes the image to a registry
and triggers the host's deploy (for Hugging Face Spaces, a `git push` to
the Space's own git remote **is** the deploy trigger - see below).

**How configuration differs between environments.** The exact same image
runs in every environment; only environment variables change
(`CORS_ORIGINS`, `SERVE_FRONTEND`, confidence thresholds, log level). Dev
uses `.env` / compose environment blocks; a hosted environment (e.g. a
Hugging Face Space) sets the same variables as Space secrets/variables.
Nothing environment-specific is hard-coded in source, by design (see the
Configuration table above).

**How to roll back a faulty release.**
- *Hugging Face Spaces*: a Space is a git repository - `git revert` the bad
  commit (or `git reset --hard <previous-good-sha>` and force-push, if the
  team agrees that's acceptable) and push; the Space rebuilds the previous
  state automatically. This is the fastest rollback available on this
  stack because there's no separate registry promotion step.
- *Generic registry-based deploy*: redeploy the previous image tag rather
  than rebuilding from source, so rollback doesn't depend on the build
  succeeding again.
- In both cases: roll back first to restore service, investigate the root
  cause afterwards (see `INCIDENT_NOTES.md` for the full reasoning).

## Deploying to Hugging Face Spaces

A Hugging Face Space is its own git repository, separate from this one, and
it expects its `Dockerfile` and `README.md` (with metadata front-matter) at
its root - which would collide with this repo's own root files. The clean
way to reconcile that without restructuring this repo is a small local
"deploy checkout":

1. Create a new Space -> SDK: **Docker** -> note its git URL
   (`https://huggingface.co/spaces/<you>/<space-name>`).
2. From this repo:
   ```bash
   git clone https://huggingface.co/spaces/<you>/<space-name> /tmp/lifeverse-space
   cp -r backend frontend deploy/huggingface/config.js /tmp/lifeverse-space/
   cp deploy/huggingface/Dockerfile /tmp/lifeverse-space/Dockerfile
   cp deploy/huggingface/README_SPACE.md /tmp/lifeverse-space/README.md
   cd /tmp/lifeverse-space
   git add -A && git commit -m "deploy: sync from main" && git push
   ```
   The `git push` to the Space **is** the deployment trigger - Spaces build
   and redeploy on every push automatically.
3. The Space builds `Dockerfile` (= `deploy/huggingface/Dockerfile`), which
   installs the backend, copies the frontend into `static/`, and serves
   both on port `7860` with `SERVE_FRONTEND=true`.
4. No secrets are required for this app. If any are added later, set them
   as **Space secrets**, never in the Dockerfile or git history.

(A CI job that automates steps 2-3 on every push to `main` is a natural
next step - see `context/HOW_TO_PROCEED.md`.)

A **container registry + staging-to-production promotion flow** (build once,
push to GHCR, deploy the same digest to a staging Space, smoke-test, then
re-point/redeploy the identical digest to production) is described as a
next step in `context/HOW_TO_PROCEED.md` rather than implemented here, to
keep this submission's infrastructure proportionate to a free-tier,
single-environment assessment.

## Troubleshooting exercise

See `INCIDENT_NOTES.md`.

## MLOps notes

See `MLOPS_NOTES.md` for model packaging, identification, versioning,
update/validation/rollback, and the inference-behaviour test beyond a
plain HTTP-200 check.
