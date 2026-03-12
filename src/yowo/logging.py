"""Structured logging support for YOWO inference engine.

Provides JsonFormatter for log aggregation pipelines and configure_logging()
for idempotent logger setup.

Usage::

    from yowo.logging import configure_logging
    configure_logging(level="INFO", structured=True)
"""

from __future__ import annotations

import datetime
import json
import logging

__all__ = ["JsonFormatter", "configure_logging"]

# Optional context fields emitted when present on the log record.
_OPTIONAL_FIELDS = (
    "engine_id",
    "model",
    "backend",
    "stream_id",
    "frame_index",
    "latency_ms",
)


class JsonFormatter(logging.Formatter):
    """Logging formatter that emits one JSON object per log record.

    Required fields in every record:
        - ``timestamp``: ISO8601 UTC string ending in ``Z``
        - ``level``: ``record.levelname`` string
        - ``event``: ``record.getMessage()`` string

    Optional fields emitted only when present on the record:
        engine_id, model, backend, stream_id, frame_index, latency_ms
    """

    def format(self, record: logging.LogRecord) -> str:
        """Return the log record as a JSON string."""
        # ISO8601 UTC timestamp
        ts = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        timestamp = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

        payload: dict[str, object] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "event": record.getMessage(),
        }

        for field in _OPTIONAL_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "WARNING", *, structured: bool = False) -> None:
    """Configure the ``yowo`` logger idempotently.

    Clears existing handlers before adding a new one to prevent duplicate
    log entries on repeated calls (e.g. during engine load retry loops).

    Args:
        level: Log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
               ``"ERROR"``, ``"CRITICAL"``).  Case-insensitive.
        structured: When ``True`` attach a :class:`JsonFormatter`;
                    when ``False`` attach a plain ``logging.Formatter``.
    """
    logger = logging.getLogger("yowo")
    # Remove all existing handlers to prevent duplicates on repeated calls.
    logger.handlers.clear()

    handler = logging.StreamHandler()
    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
