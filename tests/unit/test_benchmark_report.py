"""Tests for yowo.benchmark._report module."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from yowo.benchmark._report import render_table, results_to_json
from yowo.benchmark._runner import BenchmarkResult


def _make_result(
    fmt: str = "pytorch",
    map_50_95: float | None = 0.35,
    fps: float = 120.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        format=fmt,
        map_50_95=map_50_95,
        map_50=0.55 if map_50_95 is not None else None,
        fps_avg=fps,
        latency_p50_ms=8.3,
        latency_p95_ms=12.1,
        latency_p99_ms=15.4,
        model_size_mb=6.2,
        device="cpu",
        num_images=100,
    )


class TestRenderTable:
    def test_renders_rich_table_with_correct_columns(self) -> None:
        results = [_make_result()]
        # Capture rich console output
        with patch("yowo.benchmark._report.Console") as mock_console_cls:
            mock_console = mock_console_cls.return_value
            render_table(results, model_name="yolo11n")
            mock_console.print.assert_called_once()
            table = mock_console.print.call_args[0][0]
            # Verify table has expected columns
            col_names = [col.header for col in table.columns]
            assert "Format" in col_names
            assert "FPS" in col_names
            assert "Device" in col_names

    def test_ultralytics_column_added_when_present(self) -> None:
        results = [_make_result()]
        ultra_results = {"mAP_50_95": 0.36}
        with patch("yowo.benchmark._report.Console") as mock_console_cls:
            mock_console = mock_console_cls.return_value
            render_table(results, ultralytics_results=ultra_results, model_name="yolo11n")
            mock_console.print.assert_called_once()
            table = mock_console.print.call_args[0][0]
            col_names = [col.header for col in table.columns]
            assert any(
                "Ultralytics" in str(c) or "ultralytics" in str(c).lower() for c in col_names
            )


class TestResultsToJson:
    def test_json_output_has_all_fields(self) -> None:
        results = [_make_result(), _make_result(fmt="onnx", fps=150.0)]
        output = results_to_json(results, model_name="yolo11n", device_info="cpu")
        assert output["model"] == "yolo11n"
        assert output["device"] == "cpu"
        assert "timestamp" in output
        assert len(output["results"]) == 2
        # Check first result has all fields
        r = output["results"][0]
        assert "format" in r
        assert "fps_avg" in r
        assert "latency_p50_ms" in r
        assert "latency_p95_ms" in r
        assert "latency_p99_ms" in r
        assert "model_size_mb" in r

    def test_json_serializable(self) -> None:
        results = [_make_result()]
        output = results_to_json(results, model_name="yolo11n")
        # Should not raise
        serialized = json.dumps(output)
        assert isinstance(serialized, str)

    def test_ultralytics_included_when_present(self) -> None:
        results = [_make_result()]
        ultra = {"mAP_50_95": 0.36, "mAP_50": 0.56}
        output = results_to_json(results, ultralytics_results=ultra, model_name="yolo11n")
        assert output["ultralytics"] == ultra

    def test_ultralytics_delta_computed(self) -> None:
        results = [_make_result(map_50_95=0.35)]
        ultra = {"mAP_50_95": 0.36}
        output = results_to_json(results, ultralytics_results=ultra, model_name="yolo11n")
        r = output["results"][0]
        assert "delta_map" in r
        assert r["delta_map"] == pytest.approx(-0.01, abs=0.001)
