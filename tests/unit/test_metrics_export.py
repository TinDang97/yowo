"""Unit tests for metrics export: Prometheus format and JSON dict.

Tests MetricsCollector.export_prometheus() and engine.export_metrics() /
engine.export_metrics_prometheus().
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from yowo.backends import InferenceBackend
from yowo.engine import InferenceEngine
from yowo.metrics._collector import MetricsCollector
from yowo.types import BackendType

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
# MetricsCollector.export_prometheus()
# ---------------------------------------------------------------------------


class TestPrometheusExport:
    def test_prometheus_format_ends_with_newline(self) -> None:
        """export_prometheus() must always end with a newline."""
        collector = MetricsCollector(enabled=True)
        output = collector.export_prometheus()
        assert output.endswith("\n"), (
            f"Prometheus output must end with newline, got: {output[-10:]!r}"
        )

    def test_prometheus_metric_names(self) -> None:
        """Must include yowo_frames_total and its # TYPE header."""
        collector = MetricsCollector(enabled=True)
        output = collector.export_prometheus()
        assert "yowo_frames_total" in output
        assert "# TYPE yowo_frames_total counter" in output

    def test_prometheus_contains_all_seven_metrics(self) -> None:
        """export_prometheus() must contain all 7 required metric names."""
        collector = MetricsCollector(enabled=True)
        output = collector.export_prometheus()
        required_metrics = [
            "yowo_frames_total",
            "yowo_errors_total",
            "yowo_inference_mean_ms",
            "yowo_inference_p95_ms",
            "yowo_fps",
            "yowo_uptime_seconds",
            "yowo_memory_utilization",
        ]
        for metric in required_metrics:
            assert metric in output, f"Missing metric: {metric!r}"

    def test_prometheus_has_help_lines(self) -> None:
        """Each metric should have a # HELP line."""
        collector = MetricsCollector(enabled=True)
        output = collector.export_prometheus()
        assert "# HELP yowo_frames_total" in output

    def test_prometheus_values_are_numeric(self) -> None:
        """All metric value lines must be parseable as float."""
        collector = MetricsCollector(enabled=True)
        output = collector.export_prometheus()
        for line in output.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rsplit(" ", 1)
            assert len(parts) == 2, f"Expected 'name value', got: {line!r}"
            float(parts[1])  # must not raise

    def test_prometheus_frames_after_inference(self) -> None:
        """After recording 5 frames, yowo_frames_total should reflect that."""
        collector = MetricsCollector(enabled=True)
        collector.record_inference(10.0, batch_size=5)
        output = collector.export_prometheus()
        assert "yowo_frames_total 5" in output

    def test_prometheus_errors_after_record(self) -> None:
        """After recording 3 errors, yowo_errors_total should reflect that."""
        collector = MetricsCollector(enabled=True)
        for _ in range(3):
            collector.record_error()
        output = collector.export_prometheus()
        assert "yowo_errors_total 3" in output


# ---------------------------------------------------------------------------
# engine.export_metrics()
# ---------------------------------------------------------------------------


class TestExportMetrics:
    def test_export_metrics_dict(self) -> None:
        """export_metrics() must return a dict with all EngineMetrics field names."""
        import dataclasses

        from yowo.metrics import EngineMetrics

        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            result = engine.export_metrics()
            assert isinstance(result, dict)
            expected_keys = {f.name for f in dataclasses.fields(EngineMetrics)}
            for key in expected_keys:
                assert key in result, f"Missing key: {key!r}"
        finally:
            engine.close()

    def test_export_metrics_json_serialisable(self) -> None:
        """export_metrics() return value must be JSON-serialisable."""
        import json

        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            result = engine.export_metrics()
            # Should not raise
            json.dumps(result)
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# engine.export_metrics_prometheus()
# ---------------------------------------------------------------------------


class TestExportMetricsPrometheus:
    def test_export_metrics_prometheus_ends_with_newline(self) -> None:
        """engine.export_metrics_prometheus() must delegate and end with newline."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            output = engine.export_metrics_prometheus()
            assert isinstance(output, str)
            assert output.endswith("\n")
        finally:
            engine.close()

    def test_export_metrics_prometheus_contains_frames_total(self) -> None:
        """engine.export_metrics_prometheus() must contain yowo_frames_total."""
        mock_be = _make_mock_backend()
        engine = _loaded_engine(mock_be)
        try:
            output = engine.export_metrics_prometheus()
            assert "yowo_frames_total" in output
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# CLI tests (added in Task 2 but file lives here)
# ---------------------------------------------------------------------------


class TestMetricsCli:
    def test_metrics_cli_json_format(self) -> None:
        """yowo metrics --format json should exit 0 and output valid JSON."""
        import json

        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["metrics", "--format", "json"])
        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_metrics_cli_prometheus_format(self) -> None:
        """yowo metrics --format prometheus should exit 0 and include yowo_frames_total."""
        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["metrics", "--format", "prometheus"])
        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        assert "yowo_frames_total" in result.output
