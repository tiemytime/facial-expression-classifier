# MLOps notes

This pipeline uses two pretrained models with independent lifecycles - a
landmark detector and an image-classification model - not one model doing
two jobs. Each is covered separately below.

## Packaging and loading

**MediaPipe Face Mesh**, pinned via `mediapipe==0.10.14` in
`backend/requirements.txt`. The model graph and weights ship inside the pip
package itself - no separate download step at build or run time.

**`dima806/facial_emotions_image_detection`** (ViT-base, apache-2.0),
pinned to the exact, immutable commit `747cf16692eea925b54b7b543cf436848128b68d`
in `backend/app/models/expression_classifier.py`. Unlike the landmark
model, its weights (~350MB) are *not* bundled in a pip package - they live
on the Hugging Face Hub. To keep the build fully reproducible and the
runtime container fully offline:
- `backend/scripts/prefetch_model.py` downloads exactly three files
  (`config.json`, `preprocessor_config.json`, `model.safetensors`) at
  **Docker build time**, pinned to that commit - not the training
  checkpoints also present in the upstream repo, which would add gigabytes
  of unused optimizer state.
- The Dockerfile sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` *after*
  that build step, so the running container cannot make a network call to
  load or use this model even if it wanted to - a cold start never depends
  on Hugging Face Hub being reachable.

Both models are loaded exactly once, in FastAPI's `lifespan` handler
(`backend/app/main.py`), into module-level singletons
(`FaceMeshModel`, `ExpressionClassifier`). Neither is reconstructed per
request - instantiating either one loads/parses a model graph from disk,
which is exactly the "reload cost" the assessment asks to avoid unless
justified. Neither underlying call (`FaceMesh.process()`, the HF
`pipeline()`) is documented as safe to call concurrently on one instance
from multiple threads, so each is serialized behind its own
`threading.Lock` - a known throughput limit, tracked as operational risk #1
in `README.md`.

`static_image_mode=True` on the face mesh is deliberate (not the
video-tracking default) because this is a stateless multi-client HTTP API,
not a single-stream video pipeline - see the docstring in
`face_mesh_model.py` for the full reasoning.

## Identification

`/ready` and every `/predict` response list both models explicitly, by
name, version, and task:

```json
[
  { "name": "mediapipe-face-mesh", "version": "mediapipe-0.10.14", "task": "landmark_detection" },
  { "name": "dima806/facial_emotions_image_detection", "version": "747cf16692ee", "task": "image_classification" }
]
```

- `mediapipe-<pip version>` is read directly from the installed package
  (`mp.__version__`), so it can never drift from what's actually running.
- The classifier's `version` is the first 12 characters of its pinned
  commit SHA - an immutable, auditable reference to the exact weights in
  use, not a mutable tag like `main` that could change under us.
- Gaze direction is explicitly **not** listed as a model: it's deterministic
  geometry over the face-mesh output (`backend/app/models/gaze.py`), not a
  learned classifier, and versioning it as code (not as a model artifact)
  is accurate to what it actually is.

## Lifecycle: updating, validating, deploying, and rolling back

**Updating the landmark model** (bump `mediapipe`):
1. Bump the pin, run `pytest` - the gaze unit tests use synthetic landmark
   coordinates independent of mediapipe's internals, so they'd catch a
   breaking change in landmark indexing, not just an import error.
2. Manually validate against a small fixed set of reference webcam
   captures/photos, comparing before/after gaze labels and
   `inference_time_ms`.
3. Deploy to a staging Space first, smoke test `/health`, `/ready`, and a
   real `/predict` call, then promote the exact same image/digest to
   production - never rebuild between staging and production.
4. **Rollback**: redeploy the previous pinned image (or `git revert` the
   version bump and let CI rebuild) - safe and fast, since there's no
   persisted state or database migration tied to the model.

**Updating the expression classifier** (point `MODEL_REVISION` in
`expression_classifier.py` at a newer commit, or swap to a different HF
model entirely):
1. Change the pin, re-run `scripts/prefetch_model.py` locally to confirm
   the new revision's files resolve and download cleanly.
2. Run `pytest` - `test_expression_classifier.py` confirms the plumbing
   (label is one of the known classes, confidence in `[0, 1]`) still holds
   for the new weights.
3. Manually validate against a small fixed set of reference face photos,
   comparing before/after labels/confidences and `inference_time_ms` - a
   model update should not silently flip labels on the same input, and
   shouldn't regress CPU latency materially (this model is the dominant
   cost in the pipeline - see README "cold start" note).
4. Deploy to staging first, exactly as above, before promoting to production.
5. **Rollback**: revert the `MODEL_REVISION` pin and rebuild/redeploy - the
   old weights are still a valid, resolvable commit on the Hub, so this is
   a one-line change, not a re-training exercise.

**When to roll back vs. forward-fix**: if a new version's labels drift
unexpectedly on the reference set, or latency regresses materially, roll
back immediately and investigate offline - don't debug a live production
path under load.

## Inference testing beyond "the endpoint returns 200"

Three layers, all in `backend/tests/`:

1. **`test_geometry.py`** - unit tests that call `estimate_gaze()` directly
   with synthetic landmark coordinates constructed to land on each side of
   a known threshold (e.g. iris exactly centered between eye corners ->
   `direction == "center"` with high confidence), plus tests of the
   `_crop_face()` helper that feeds the classifier. These validate actual
   *behaviour*, independent of mediapipe's face detector and independent of
   the classifier, and would catch a regression in the geometry math that a
   bare "did it return 200" check never would.
2. **`test_expression_classifier.py`** - a contract test that runs a real
   forward pass through the pinned classifier and asserts the output shape
   (label is one of the seven known classes, confidence in `[0, 1]`). It
   does not assert a specific label, because we don't commit a real,
   clearly-licensed face photo to this repo (see README "Known
   limitations") - there's no fixture to assert a known emotion against.
3. **`test_predict.py::test_predict_blank_image_reports_no_face`** - an
   end-to-end API test asserting on response *content* for a known input
   (a face-less blank image): `face_detected` must be `False`, `gaze`/
   `expression` must be `None`, and a `warnings` message must explain why.
   Because the classifier only ever runs on a crop of a *detected* face,
   this also confirms it's correctly skipped (not called at all) on the
   no-face path.

A natural next step (see `context/HOW_TO_PROCEED.md`) is sourcing one or
two suitably licensed sample face photos with known expected labels, for a
true end-to-end "golden output" regression test that exercises the
classifier's actual predictions, not just its contract.
