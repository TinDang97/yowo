"""Unit tests for HealthReport dataclass and yowo health CLI.

Tests engine.health_report() fields, HealthReport.as_dict(), and CLI exit codes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.engine import HealthReport, InferenceEngine
from yowo.types import BackendType, HealthStatus

_RESOLVE_PATCH = "yowo.engine.resolve_weights"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)

    def _load(*args: object, **kwargs: object) -> None:
        mock.is_loaded = True

    def _unload(*args: object, **kwargs: object) -> None:
        mock.is_loaded = False

    mock.load.side_effect = _load
    mock.unload.side_effect = _unload
    mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    mock.warmup.return_value = None
    return mock


def _loaded_engine(mock_backend: MagicMock) -> InferenceEngine:
    engine = InferenceEngine(backend_instance=mock_backend)
    with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
        engine.load()
    return engine


# ---------------------------------------------------------------------------
# HealthReport dataclass
# ---------------------------------------------------------------------------


class TestHealthReportDataclass:
    def test_health_report_is_importable(self) -> None:
        """HealthReport must be importable from yowo.engine."""
        from yowo.engine import HealthReport  # noqa: F401

    def test_health_report_is_frozen_dataclass(self) -> None:
        """HealthReport must be a frozen dataclass."""
        import dataclasses

        assert dataclasses.is_dataclass(HealthReport)
        # frozen dataclasses cannot be mutated
        report = HealthReport(
            status=HealthStatus.READY,
            uptime_s=1.0,
            errors_total=0,
            frames_total=0,
            memory_pct=None,
            stream_count=0,
            batch_size_current=1,
            precision_current="fp32",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            report.status = HealthStatus.DEGRADED  # type: ignore[misc]

    def test_health_report_fields(self) -> None:
        """health_report() on loaded engine returns HealthReport with expected values."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            report = engine.health_report()
            assert isinstance(report, HealthReport)
            assert report.status == HealthStatus.READY
            assert report.batch_size_current >= 1
            assert isinstance(report.uptime_s, float)
            assert report.uptime_s >= 0.0
            assert isinstance(report.errors_total, int)
            assert isinstance(report.frames_total, int)
            assert isinstance(report.stream_count, int)
            assert isinstance(report.precision_current, str)
        finally:
            engine.close()

    def test_health_report_memory_pct_is_none_on_cpu(self) -> None:
        """memory_pct must be None when engine runs on CPU (not CUDA)."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            report = engine.health_report()
            # Mock backend uses PYTORCH on CPU (not CUDA device)
            assert report.memory_pct is None
        finally:
            engine.close()

    def test_health_report_as_dict_serialisable(self) -> None:
        """as_dict() must return a JSON-serialisable dict."""
        import json

        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            report = engine.health_report()
            d = report.as_dict()
            assert isinstance(d, dict)
            # Should not raise
            json.dumps(d)
        finally:
            engine.close()

    def test_health_report_as_dict_status_is_string(self) -> None:
        """as_dict() must return status as .value string, not HealthStatus enum."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            report = engine.health_report()
            d = report.as_dict()
            assert isinstance(d["status"], str)
            assert d["status"] == "ready"
        finally:
            engine.close()

    def test_health_report_precision_current_is_string(self) -> None:
        """precision_current must be a str (not a Precision enum)."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            report = engine.health_report()
            assert isinstance(report.precision_current, str)
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestHealthCli:
    def test_health_cli_exit_code_closed(self) -> None:
        """yowo health exits 2 and includes 'closed' in output when no engine running."""
        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])
        assert result.exit_code == 2, f"Expected exit 2, got {result.exit_code}: {result.output}"
        assert "closed" in result.output.lower()

    def test_health_cli_json_default(self) -> None:
        """yowo health default format is JSON."""
        import json

        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])
        # Exit code 2 but output should be valid JSON
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert "status" in parsed

    def test_health_cli_text_format(self) -> None:
        """yowo health --format text prints key: value lines."""
        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--format", "text"])
        assert result.exit_code == 2
        assert "status" in result.output.lower()
