"""Pre-fetches the pinned expression-classifier weights at Docker build time.

Run from the backend/ directory as `python -m scripts.prefetch_model` (the
Dockerfile does exactly this). Only the three files inference actually needs
are fetched - not the training checkpoints also present in the upstream
repo - so the image doesn't carry gigabytes of unused optimizer state.

After this runs, HF_HUB_OFFLINE=1 at container runtime (set in the
Dockerfile) guarantees the model loads from this cache with zero network
access - see MLOPS_NOTES.md "Packaging and loading".
"""
from huggingface_hub import snapshot_download

from app.models.expression_classifier import MODEL_NAME, MODEL_REVISION

REQUIRED_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]

if __name__ == "__main__":
    path = snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        allow_patterns=REQUIRED_FILES,
    )
    print(f"prefetched {MODEL_NAME}@{MODEL_REVISION} -> {path}")
