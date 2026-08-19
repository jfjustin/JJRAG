"""Structured logging with a privacy filter.

Two rules this project cannot break:

1. Document text never appears in a log line.
2. Secrets never appear in a log line.

Rule 2 is enforced mechanically by :class:`RedactingFilter`, which scrubs any
record that slips through. Rule 1 is a convention in the calling code — log
identifiers and counts, never content.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from ..security.pii import PATTERNS

_SENSITIVE = ("secret", "email", "credit_card", "ssn", "iban")


class RedactingFilter(logging.Filter):
    """Last line of defence: scrub PII/secrets out of formatted log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never let logging raise
            return True
        scrubbed = message
        for kind in _SENSITIVE:
            scrubbed = PATTERNS[kind].sub(f"[{kind.upper()}_REDACTED]", scrubbed)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line — for shipping into a log stack."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "doc_id", "stage", "event", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    log_dir: Path | str | None = None,
    json_output: bool = False,
) -> None:
    """Idempotently configure the root logger for the process."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level.upper())

    formatter: logging.Formatter = (
        JsonFormatter()
        if json_output
        else logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    redactor = RedactingFilter()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    stream.addFilter(redactor)
    root.addHandler(stream)

    if log_dir:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            directory / "jjrag.log", maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter() if json_output else formatter)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)

    # Third-party noise we do not need.
    for noisy in ("urllib3", "httpx", "sentence_transformers", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"jjrag.{name}" if not name.startswith("jjrag") else name)
