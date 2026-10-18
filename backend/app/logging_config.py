"""Logging setup.

Logs are structured, human-readable lines to stdout (container-friendly).
We deliberately never log: raw image bytes, uploaded filenames from the
client, or any request body content — only metadata (sizes, timings,
outcome, error class) needed to debug without capturing user-submitted
imagery or secrets.
"""
import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid duplicate handlers on reload.
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(_DefaultRequestIdFilter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers that are not actionable at INFO.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class _DefaultRequestIdFilter(logging.Filter):
    """Ensures log records without an explicit request_id still format cleanly."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True
