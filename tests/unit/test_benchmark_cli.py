"""Tests for the ``yowo benchmark`` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from yowo.cli._main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def sample_benchmark_json() -> dict:
    """Sample return value from run_benchmark(json_output=True)."""
    return {
        "model": "yolo11n",
        "device": "",
        "timestamp": "2026-03-07T00:00:00+00:00",
        "results": [
            {
                "format": "pytorch",
                "map_50_95": 0.3712,
                "map_50": 0.5234,
                "fps_avg": 42.5,
                "latency_p50_ms": 23.5,
                "latency_p95_ms": 28.1,
                "latency_p99_ms": 32.0,
                "model_size_mb": 12.3,
                "device": "cpu",
                "num_images": 50,
            }
        ],
    }


class TestBenchmarkHelp:
    """Test that --help shows all expected options."""

    def test_benchmark_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--data" in result.output
        assert "--format" in result.output
        assert "--subset" in result.output
        assert "--json" in result.output
        assert "--output" in result.output


class TestBenchmarkMissingData:
    """Test error messages when --data path is invalid."""

    def test_benchmark_missing_data_detection(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["benchmark", "--model", "yolo11n", "--data", "/nonexistent/path"]
        )
        assert result.exit_code != 0
        assert "val2017" in result.output or "cocodataset" in result.output.lower()

    def test_benchmark_missing_data_classification(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["benchmark", "--model", "yolo11n-cls", "--data", "/nonexistent/cls/path"],
        )
        assert result.exit_code != 0
        assert "ImageFolder" in result.output or "image-net" in result.output.lower()

    def test_benchmark_invalid_data_path(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["benchmark", "--model", "yolo11n", "--data", "/no/such/dir"])
        assert result.exit_code != 0
        # Should show expected directory structure, not a stack trace
        assert "Expected" in result.output or "Download" in result.output


class TestBenchmarkJsonFlag:
    """Test --json flag produces valid JSON to stdout."""

    def test_benchmark_json_flag(
        self,
        runner: CliRunner,
        sample_benchmark_json: dict,
        tmp_path: Path,
    ) -> None:
        # Create a fake data dir so path validation passes
        data_dir = tmp_path / "coco"
        data_dir.mkdir()
        (data_dir / "val2017").mkdir()
        (data_dir / "annotations").mkdir()
        (data_dir / "annotations" / "instances_val2017.json").write_text("{}")

        with patch("yowo.cli._main.run_benchmark", return_value=sample_benchmark_json) as mock_rb:
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--model",
                    "yolo11n",
                    "--data",
                    str(data_dir),
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output
            parsed = json.loads(result.output)
            assert "results" in parsed
            mock_rb.assert_called_once()


class TestBenchmarkOutputFile:
    """Test --output flag writes JSON to specified file path."""

    def test_benchmark_output_file(
        self,
        runner: CliRunner,
        sample_benchmark_json: dict,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "coco"
        data_dir.mkdir()
        (data_dir / "val2017").mkdir()
        (data_dir / "annotations").mkdir()
        (data_dir / "annotations" / "instances_val2017.json").write_text("{}")

        out_file = tmp_path / "results.json"

        with patch("yowo.cli._main.run_benchmark", return_value=sample_benchmark_json):
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--model",
                    "yolo11n",
                    "--data",
                    str(data_dir),
                    "--output",
                    str(out_file),
                ],
            )
            assert result.exit_code == 0, result.output
            assert out_file.exists()
            content = json.loads(out_file.read_text())
            assert "results" in content
            assert "Results written to" in result.output


class TestBenchmarkFormatFilter:
    """Test --format passes correct filter to run_benchmark."""

    def test_benchmark_format_filter(
        self,
        runner: CliRunner,
        sample_benchmark_json: dict,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "coco"
        data_dir.mkdir()
        (data_dir / "val2017").mkdir()
        (data_dir / "annotations").mkdir()
        (data_dir / "annotations" / "instances_val2017.json").write_text("{}")

        with patch("yowo.cli._main.run_benchmark", return_value=sample_benchmark_json) as mock_rb:
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--model",
                    "yolo11n",
                    "--data",
                    str(data_dir),
                    "--format",
                    "onnx,trt",
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_rb.call_args
            assert call_kwargs.kwargs.get("formats") == "onnx,trt" or (
                call_kwargs.args and "onnx,trt" in str(call_kwargs)
            )


class TestBenchmarkSubset:
    """Test --subset passes correct limit to run_benchmark."""

    def test_benchmark_subset(
        self,
        runner: CliRunner,
        sample_benchmark_json: dict,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "coco"
        data_dir.mkdir()
        (data_dir / "val2017").mkdir()
        (data_dir / "annotations").mkdir()
        (data_dir / "annotations" / "instances_val2017.json").write_text("{}")

        with patch("yowo.cli._main.run_benchmark", return_value=sample_benchmark_json) as mock_rb:
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--model",
                    "yolo11n",
                    "--data",
                    str(data_dir),
                    "--subset",
                    "10",
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_rb.call_args
            assert call_kwargs.kwargs.get("subset") == 10 or "10" in str(call_kwargs)


class TestBenchmarkInvokesRunBenchmark:
    """Test CLI correctly passes all args to run_benchmark."""

    def test_benchmark_invokes_run_benchmark(
        self,
        runner: CliRunner,
        sample_benchmark_json: dict,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "coco"
        data_dir.mkdir()
        (data_dir / "val2017").mkdir()
        (data_dir / "annotations").mkdir()
        (data_dir / "annotations" / "instances_val2017.json").write_text("{}")

        with patch("yowo.cli._main.run_benchmark", return_value=sample_benchmark_json) as mock_rb:
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--model",
                    "yolo26x",
                    "--data",
                    str(data_dir),
                    "--format",
                    "pytorch",
                    "--subset",
                    "100",
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output
            mock_rb.assert_called_once_with(
                model="yolo26x",
                data=str(data_dir),
                formats="pytorch",
                subset=100,
                json_output=True,
            )
