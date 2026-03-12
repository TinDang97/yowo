"""Unit tests for JsonFormatter and configure_logging.

Tests structured logging output (JSON lines), duplicate-handler prevention,
and configure_logging idempotency.
"""

from __future__ import annotations

import json
import logging

from yowo.logging import JsonFormatter, configure_logging


class TestJsonFormatter:
    def test_json_formatter_valid_json(self) -> None:
        """format() must return a valid JSON string."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="yowo",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test event",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_formatter_required_fields(self) -> None:
        """Required fields: timestamp, level, event."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="yowo",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="something happened",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "event" in parsed

    def test_json_formatter_level_is_levelname(self) -> None:
        """level field must match record.levelname."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="yowo",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="an error",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "ERROR"

    def test_json_formatter_event_is_message(self) -> None:
        """event field must equal record.getMessage()."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="yowo",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["event"] == "hello world"

    def test_json_formatter_timestamp_iso8601(self) -> None:
        """timestamp must be ISO8601 UTC string ending in Z."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="yowo",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="ts check",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        ts = parsed["timestamp"]
        assert isinstance(ts, str)
        assert ts.endswith("Z"), f"Expected ISO8601 UTC (ends in Z), got: {ts!r}"

    def test_json_formatter_optional_fields_absent_by_default(self) -> None:
        """Optional context fields must NOT appear when not set on record."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="yowo",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="plain",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        optional_fields = {
            "engine_id",
            "model",
            "backend",
            "stream_id",
            "frame_index",
            "latency_ms",
        }
        for field in optional_fields:
            assert field not in parsed, f"Optional field {field!r} present but not set"

    def test_json_formatter_optional_fields_emitted_when_present(self) -> None:
        """Optional fields appear when set on record as extra attributes."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="yowo",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="context event",
            args=(),
            exc_info=None,
        )
        record.engine_id = "eng-1"
        record.model = "yolo26n"
        record.latency_ms = 12.5
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["engine_id"] == "eng-1"
        assert parsed["model"] == "yolo26n"
        assert parsed["latency_ms"] == 12.5


class TestConfigureLogging:
    def test_configure_logging_no_duplicate_handlers(self) -> None:
        """Calling configure_logging twice must not add duplicate handlers."""
        # Reset to known state
        logger = logging.getLogger("yowo")
        logger.handlers.clear()

        configure_logging(level="INFO", structured=False)
        configure_logging(level="INFO", structured=False)

        assert len(logger.handlers) == 1

    def test_configure_logging_sets_level(self) -> None:
        """configure_logging must set the yowo logger level."""
        logger = logging.getLogger("yowo")
        logger.handlers.clear()

        configure_logging(level="DEBUG", structured=False)
        assert logger.level == logging.DEBUG

    def test_configure_logging_structured_uses_json_formatter(self) -> None:
        """structured=True must attach a JsonFormatter handler."""
        logger = logging.getLogger("yowo")
        logger.handlers.clear()

        configure_logging(level="INFO", structured=True)

        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_configure_logging_not_structured_uses_plain_formatter(self) -> None:
        """structured=False must attach a plain (non-JSON) formatter."""
        logger = logging.getLogger("yowo")
        logger.handlers.clear()

        configure_logging(level="WARNING", structured=False)

        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_configure_logging_clears_existing_before_adding(self) -> None:
        """Third+ calls still result in exactly one handler."""
        logger = logging.getLogger("yowo")
        logger.handlers.clear()

        configure_logging(level="INFO", structured=False)
        configure_logging(level="WARNING", structured=True)
        configure_logging(level="ERROR", structured=False)

        assert len(logger.handlers) == 1
