"""End-to-end CLI tests for yowo using real YOLO26 model inference.

These tests exercise the full pipeline — hardware detection, model loading,
preprocessing, inference, postprocessing, and output serialization — through
the CLI entry point with no mocks.

Markers:
    integration: all tests in this file (require real CLI + library)
    slow: tests that load the model and run real inference (~30-60s total)

Run fast tests only:
    pytest tests/integration/ -m "integration and not slow" -v

Run all integration tests:
    pytest tests/integration/ -m "integration" -v --timeout=300
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from yowo.cli._main import cli
from yowo.postprocess._nms import COCO_CLASSES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(runner: CliRunner, args: list[str]) -> object:
    """Invoke CLI and return result with normalised output."""
    return runner.invoke(cli, args, catch_exceptions=False)


def _load_json(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_box_invariants(box: dict[str, object]) -> None:
    """Assert structural correctness of a single box dict."""
    assert set({"x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"}).issubset(
        box.keys()
    )
    assert box["x1"] < box["x2"], "x1 must be less than x2"
    assert box["y1"] < box["y2"], "y1 must be less than y2"
    assert 0.0 <= float(str(box["confidence"])) <= 1.0
    assert 0 <= int(str(box["class_id"])) <= 79
    assert box["class_name"] in COCO_CLASSES


# ---------------------------------------------------------------------------
# TestCLIHelp — fast, no model needed
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCLIHelp:
    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "detect" in result.output
        assert "export" in result.output
        assert "info" in result.output
        assert "models" in result.output

    def test_version_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_detect_help_shows_weights(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["detect", "--help"])
        assert result.exit_code == 0
        assert "SOURCE" in result.output
        assert "--model" in result.output
        assert "--weights" in result.output
        assert "--backend" in result.output
        assert "--confidence" in result.output
        assert "--output" in result.output


# ---------------------------------------------------------------------------
# TestInfoCommand — fast, no model needed
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInfoCommand:
    def test_info_prints_hardware_sections(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "=== Hardware ===" in result.output
        assert "CPU:" in result.output
        assert "=== Libraries ===" in result.output
        assert "torch:" in result.output

    def test_info_torch_is_installed(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        # Find the torch line and confirm it's not missing
        torch_line = next(
            (line for line in result.output.splitlines() if line.startswith("torch:")),
            None,
        )
        assert torch_line is not None, "torch: line missing from info output"
        assert "not installed" not in torch_line, f"torch should be installed; got: {torch_line}"


# ---------------------------------------------------------------------------
# TestModelsCommand — fast, no model needed
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestModelsCommand:
    _MODEL_PATTERN = re.compile(r"^yolo(11|12|26)(n|s|m|l|x)\s")

    def _model_lines(self, output: str) -> list[str]:
        return [ln for ln in output.splitlines() if self._MODEL_PATTERN.match(ln)]

    def test_models_lists_all_15_variants(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        lines = self._model_lines(result.output)
        assert len(lines) == 15, f"Expected 15 model lines, got {len(lines)}:\n{result.output}"

    def test_models_filter_yolo26_gives_5(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["models", "--family", "yolo26"])
        assert result.exit_code == 0
        lines = self._model_lines(result.output)
        assert len(lines) == 5
        assert all(ln.startswith("yolo26") for ln in lines)

    def test_models_filter_nonexistent_family_returns_empty(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["models", "--family", "yolo99"])
        assert result.exit_code == 0
        assert len(self._model_lines(result.output)) == 0

    def test_models_all_use_640_input_and_80_classes(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        for line in self._model_lines(result.output):
            assert "640" in line, f"Expected 640 input in: {line}"
            assert "80" in line, f"Expected 80 classes in: {line}"


# ---------------------------------------------------------------------------
# TestCLIErrorCases — fast, validate error handling
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCLIErrorCases:
    def test_invalid_model_name_exits_nonzero(
        self, runner: CliRunner, sample_image_path: Path
    ) -> None:
        result = runner.invoke(cli, ["detect", str(sample_image_path), "--model", "yolo99z"])
        assert result.exit_code != 0
        combined = result.output or ""
        assert "Unknown model" in combined or "yolo99z" in combined

    def test_missing_source_file_exits_nonzero(
        self, runner: CliRunner, yolo26_weights: Path
    ) -> None:
        result = runner.invoke(
            cli,
            ["detect", "/nonexistent/path/image.jpg", "--weights", str(yolo26_weights)],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_unsupported_source_extension_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["detect", "file.xyz"])
        assert result.exit_code != 0

    def test_export_missing_format_is_click_error(self, runner: CliRunner) -> None:
        # --format is required; click should return exit_code 2
        result = runner.invoke(cli, ["export", "yolo26n"])
        assert result.exit_code == 2
        combined = result.output or ""
        assert "Missing option" in combined or "--format" in combined

    def test_invalid_backend_choice_is_click_error(
        self, runner: CliRunner, sample_image_path: Path
    ) -> None:
        result = runner.invoke(
            cli, ["detect", str(sample_image_path), "--backend", "badbackend"]
        )
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# TestDetectCommand — slow, requires real model inference
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestDetectCommand:
    def test_detect_single_image_exits_zero(
        self,
        runner: CliRunner,
        sample_image_path: Path,
        yolo26_weights: Path,
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "detect",
                str(sample_image_path),
                "--model", "yolo26n",
                "--weights", str(yolo26_weights),
                "--backend", "pytorch",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"output:\n{result.output}"
        assert "Frame 0:" in result.output
        assert "detections" in result.output

    def test_detect_json_output_schema(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_image_path: Path,
        yolo26_weights: Path,
    ) -> None:
        out_json = tmp_path / "results.json"
        result = runner.invoke(
            cli,
            [
                "detect",
                str(sample_image_path),
                "--model", "yolo26n",
                "--weights", str(yolo26_weights),
                "--backend", "pytorch",
                "--output", str(out_json),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"output:\n{result.output}"
        assert out_json.exists()

        data = _load_json(out_json)
        assert isinstance(data, list)
        assert len(data) == 1  # single image = single frame

        frame = data[0]
        assert frame["frame_index"] == 0
        assert frame["backend"] == "pytorch"
        assert frame["model"] == "yolo26n"
        assert float(str(frame["inference_time_ms"])) > 0
        assert isinstance(frame["boxes"], list)

        for box in frame["boxes"]:
            _assert_box_invariants(box)  # type: ignore[arg-type]

    def test_detect_produces_real_detections_on_bus_image(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_image_path: Path,
        yolo26_weights: Path,
    ) -> None:
        out_json = tmp_path / "det.json"
        result = runner.invoke(
            cli,
            [
                "detect",
                str(sample_image_path),
                "--model", "yolo26n",
                "--weights", str(yolo26_weights),
                "--backend", "pytorch",
                "--confidence", "0.25",
                "--output", str(out_json),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"output:\n{result.output}"

        data = _load_json(out_json)
        boxes = data[0]["boxes"]
        assert isinstance(boxes, list)
        assert len(boxes) >= 1, "bus.jpg should produce at least 1 detection at confidence=0.25"

        class_names = {b["class_name"] for b in boxes}  # type: ignore[index]
        expected_classes = {"person", "bus", "car", "truck", "bicycle", "motorcycle"}
        assert class_names & expected_classes, (
            f"Expected at least one of {expected_classes}, got {class_names}"
        )

    def test_detect_directory_processes_all_frames(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_image_dir: Path,
        yolo26_weights: Path,
    ) -> None:
        out_json = tmp_path / "dir.json"
        result = runner.invoke(
            cli,
            [
                "detect",
                str(sample_image_dir),
                "--model", "yolo26n",
                "--weights", str(yolo26_weights),
                "--backend", "pytorch",
                "--output", str(out_json),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"output:\n{result.output}"

        data = _load_json(out_json)
        assert len(data) == 3, f"Expected 3 frames (one per image), got {len(data)}"
        frame_indices = [d["frame_index"] for d in data]
        assert frame_indices == [0, 1, 2], f"Expected frame indices 0,1,2 got {frame_indices}"

    def test_detect_lower_confidence_yields_more_or_equal_detections(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_image_path: Path,
        yolo26_weights: Path,
    ) -> None:
        common_args = [
            "detect",
            str(sample_image_path),
            "--model", "yolo26n",
            "--weights", str(yolo26_weights),
            "--backend", "pytorch",
        ]

        low_json = tmp_path / "low.json"
        high_json = tmp_path / "high.json"

        r_low = runner.invoke(
            cli, [*common_args, "--confidence", "0.01", "--output", str(low_json)],
            catch_exceptions=False,
        )
        r_high = runner.invoke(
            cli, [*common_args, "--confidence", "0.99", "--output", str(high_json)],
            catch_exceptions=False,
        )

        assert r_low.exit_code == 0
        assert r_high.exit_code == 0

        low_boxes = _load_json(low_json)[0]["boxes"]
        high_boxes = _load_json(high_json)[0]["boxes"]

        assert isinstance(low_boxes, list)
        assert isinstance(high_boxes, list)
        assert len(low_boxes) >= len(high_boxes), (
            f"Lower confidence should yield >= boxes: {len(low_boxes)} vs {len(high_boxes)}"
        )

        # All boxes in high-confidence run must have confidence >= 0.99
        for box in high_boxes:
            assert float(str(box["confidence"])) >= 0.99 - 1e-6  # type: ignore[index]

        # All boxes in low-confidence run must have confidence >= 0.01
        for box in low_boxes:
            assert float(str(box["confidence"])) >= 0.01 - 1e-6  # type: ignore[index]


# ---------------------------------------------------------------------------
# TestExportCommand — slow, requires real model + ultralytics export
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestExportCommand:
    def test_export_onnx_creates_file_and_sidecar(
        self,
        tmp_path: Path,
        yolo26_weights: Path,
    ) -> None:
        """Run export as a real subprocess to avoid CliRunner I/O conflicts with
        ultralytics' logging infrastructure (which writes to sys.stderr directly)."""
        out_dir = tmp_path / "export_out"
        # Resolve the yowo script in the same venv as the test runner.
        yowo_bin = Path(sys.executable).parent / "yowo"
        proc = subprocess.run(
            [
                str(yowo_bin),
                "export", "yolo26n",
                "--weights", str(yolo26_weights),
                "--format", "onnx",
                "--precision", "fp32",
                "--output-dir", str(out_dir),
                "--imgsz", "640",
            ],
            capture_output=True,
            text=True,
        )
        combined = proc.stdout + proc.stderr
        assert proc.returncode == 0, f"Export subprocess failed:\n{combined}"
        assert "Exported:" in combined
        assert "MB" in combined
        assert "Duration:" in combined

        # Locate the exported .onnx file (ultralytics places it next to the .pt)
        onnx_files = list(tmp_path.rglob("*.onnx")) + list(
            yolo26_weights.parent.glob("*.onnx")
        )
        assert len(onnx_files) >= 1, "No .onnx file found after export"
        onnx_path = onnx_files[0]
        assert onnx_path.stat().st_size > 0

        # Verify the .yowo.json sidecar
        sidecar = onnx_path.with_suffix(".yowo.json")
        assert sidecar.exists(), f"Sidecar not found at {sidecar}"
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["model_name"] == "yolo26n"
        assert meta["format"] == "onnx"
        assert meta["precision"] == "fp32"
        assert meta["imgsz"] == 640
        assert int(str(meta["file_size_bytes"])) > 0
        assert float(str(meta["export_duration_sec"])) > 0

    def test_export_int8_without_calibration_data_fails(
        self,
        runner: CliRunner,
    ) -> None:
        result = runner.invoke(
            cli,
            ["export", "yolo26n", "--format", "onnx", "--precision", "int8"],
        )
        assert result.exit_code != 0
        combined = result.output or ""
        assert any(kw in combined.lower() for kw in ("calibration", "int8")), (
            f"Expected calibration error message, got:\n{combined}"
        )
