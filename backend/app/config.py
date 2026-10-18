"""Runtime configuration loaded from environment variables.

Every environment-specific value (ports, CORS origins, model thresholds, log
level) lives here and is overridden via env vars or a .env file — never
hard-coded in business logic. See README.md "Configuration" for the full
variable reference.
"""
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    app_env: str = "development"
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS - comma-separated list of allowed origins for the standalone
    # (two-container) deployment mode. Ignored in combined single-origin mode.
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"

    # Combined deployment mode (e.g. Hugging Face Spaces): the backend also
    # serves the static frontend from /static so a single container/port is
    # enough for a free-tier host. Off by default for local dev, where the
    # frontend runs as its own container/process.
    serve_frontend: bool = False
    static_dir: str = "static"

    # Request limits
    max_image_size_mb: float = 8.0
    allowed_content_types: str = "image/jpeg,image/png,image/webp"

    # Model configuration - identifies exactly which pretrained model and
    # heuristic version produced a prediction (see MLOPS_NOTES.md).
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_faces: int = 1

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_content_type_list(self) -> List[str]:
        return [c.strip() for c in self.allowed_content_types.split(",") if c.strip()]

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """Settings are cached (loaded once per process), not re-read per request."""
    return Settings()
