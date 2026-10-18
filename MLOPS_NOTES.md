# MLOps notes

## Packaging and loading

The model is **MediaPipe Face Mesh**, pinned via `mediapipe==0.10.14` in
`backend/requirements.txt`. The model graph and weights ship inside the pip
package itself - there is no separate download step at build or run time,
so a `docker build` is reproducible offline and air-gapped once the base
image and pip index are cached.

It is loaded exactly once, in `FastAPI`'s `lifespan` handler
(`backend/app/main.py`), into a module-level singleton
(`backend/app/models/face_mesh_model.py::FaceMeshModel`). It is **not**
reloaded per request - constructing `mp.solutions.face_mesh.FaceMesh()`
parses the TFLite graph from disk, which is exactly the "reload cost" the
assessment asks to avoid unless justified. Because the underlying
`.process()` call is not documented as safe to call concurrently on one
instance, every call is serialized behind a `threading.Lock`; this is a
known throughput limit, tracked as an operational risk in `README.md`.

`static_image_mode=True` is used deliberately (not the video-tracking
default) because this is a stateless multi-client HTTP API, not a
single-stream video pipeline - see the docstring in `face_mesh_model.py`
for the full reasoning.

## Identification

Every `/predict` response and the `/ready` endpoint report the model and
heuristic versions explicitly, e.g.:

```
mediapipe-0.10.14+gaze-heuristic-1.0.0+expression-heuristic-1.0.0
```

This is a composite because the pipeline has two parts with independent
version histories:

- **`mediapipe-<pip version>`** - the pretrained landmark model. Its
  version is read directly from the installed package (`mp.__version__`),
  so it can never drift from what's actually running.
- **`gaze-heuristic-<semver>`** / **`expression-heuristic-<semver>`** - the
  deterministic geometric classifiers built on top of the landmarks
  (`backend/app/models/gaze.py`, `expression.py`). These are *not* a
  second trained model - they are hand-authored, explainable rules
  (mouth/eyebrow/iris distance ratios) over the pretrained model's output,
  versioned as code. This choice keeps the MLOps story to a single
  packaged artifact (the mediapipe wheel) instead of two models with
  independent training/validation lifecycles, which is proportionate to
  this assessment's "small, reliable baseline" goal.

## Lifecycle: updating, validating, deploying, and rolling back

**Updating the landmark model** (bump `mediapipe` in `requirements.txt`):
1. Bump the pin, run the full test suite (`pytest`) - the geometry unit
   tests use synthetic landmark coordinates independent of mediapipe's
   internals, so they catch a breaking change in landmark indexing or
   ordering, not just an import error.
2. Manually validate against a small fixed set of reference webcam
   captures/photos, comparing before/after gaze and expression labels and
   `inference_time_ms` - a new model version should not silently change
   labels on the same input, and shouldn't regress latency.
3. Deploy to a staging Space first (same image, staging env vars), smoke
   test `/health`, `/ready`, and a real `/predict` call.
4. Promote the exact same image/digest to production - never rebuild
   between staging and production.
5. **Rollback**: redeploy the previous pinned `mediapipe` version's image
   (or `git revert` the version bump and let CI rebuild) - safe and fast,
   since there is no persisted state or database migration tied to the
   model.

**Updating a heuristic** (tune a threshold in `gaze.py`/`expression.py`):
same flow, but validation is primarily the unit tests (which encode the
expected behaviour at known threshold boundaries) plus a manual spot-check,
since there's no training/eval dataset involved by design.

**When to roll back vs. forward-fix**: if the new version's labels drift
unexpectedly on the reference set, or `inference_time_ms` regresses
materially, roll back immediately and investigate offline - don't debug a
live production path under load.

## Inference testing beyond "the endpoint returns 200"

Two layers, both in `backend/tests/`:

1. **`test_geometry.py`** - unit tests that call `estimate_gaze()` and
   `estimate_expression()` directly with synthetic landmark coordinates
   constructed to land on each side of a known threshold (e.g. iris
   exactly centered between eye corners -> `direction == "center"` with
   high confidence; iris shifted to the outer corner -> `"left"`). This
   validates the actual classification *behaviour*, independent of
   mediapipe's face detector, and would catch a regression in the geometry
   math that a bare "did it return 200" check never would.
2. **`test_predict.py::test_predict_blank_image_reports_no_face`** - an
   end-to-end API test asserting on response *content* for a known input
   (a face-less blank image): `face_detected` must be `False`, `gaze`/
   `expression` must be `None`, and a `warnings` message must explain why -
   not just that the call succeeded.

A natural next step (see `context/HOW_TO_PROCEED.md`) is adding one or two
committed sample face images with known expected labels for a true
end-to-end "golden output" regression test, once a suitably licensed
sample image is sourced.
