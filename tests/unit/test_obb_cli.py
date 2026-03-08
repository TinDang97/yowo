"""Unit tests for OBB CLI integration: parse_model_name -obb and detect-obb command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from yowo.cli._main import cli
from yowo.errors import ConfigError
from yowo.types import ModelFamily, ModelSize, OBBBox, OBBDetection

# ---------------------------------------------------------------------------
# parse_model_name -obb suffix tests
# ---------------------------------------------------------------------------


def test_parse_model_name_obb_nano() -> None:
    from yowo._convenience import parse_model_name

    spec = parse_model_name("yolo11n-obb")
    assert spec.family == ModelFamily.YOLO11
    assert spec.size == ModelSize.NANO
    assert spec.task == "obb"


def test_parse_model_name_obb_xlarge() -> None:
    from yowo._convenience import parse_model_name

    spec = parse_model_name("yolo11x-obb")
    assert spec.family == ModelFamily.YOLO11
    assert spec.size == ModelSize.XLARGE
    assert spec.task == "obb"


def test_parse_model_name_obb_all_sizes() -> None:
    from yowo._convenience import parse_model_name

    expected_sizes = {
        "yolo11n-obb": ModelSize.NANO,
        "yolo11s-obb": ModelSize.SMALL,
        "yolo11m-obb": ModelSize.MEDIUM,
        "yolo11l-obb": ModelSize.LARGE,
        "yolo11x-obb": ModelSize.XLARGE,
    }
    for name, expected_size in expected_sizes.items():
        spec = parse_model_name(name)
        assert spec.family == ModelFamily.YOLO11
        assert spec.size == expected_size
        assert spec.task == "obb"


def test_parse_model_name_yolo26_obb_raises() -> None:
    from yowo._convenience import parse_model_name

    with pytest.raises(ConfigError, match="OBB models are only available for YOLO11"):
        parse_model_name("yolo26n-obb")


# ---------------------------------------------------------------------------
# detect-obb CLI tests
# ---------------------------------------------------------------------------


def _make_obb_detection(frame_index: int = 0) -> OBBDetection:
    return OBBDetection(
        frame_index=frame_index,
        source_id="test_src",
        boxes=(),
        inference_time_ms=5.0,
    )


def _make_mock_obb_engine(detections: list[OBBDetection]) -> MagicMock:
    """Build a MagicMock that acts as an OBBEngine context manager."""
    engine = MagicMock()
    engine.__enter__ = MagicMock(return_value=engine)
    engine.__exit__ = MagicMock(return_value=False)
    engine.stream_obb.return_value = iter(detections)
    engine.metrics.frames_total = len(detections)
    engine.metrics.fps = 30.0
    engine.metrics.inference_p50_ms = 5.0
    engine.metrics.errors_total = 0
    return engine


def test_detect_obb_command_exits_zero(tmp_path: Path) -> None:
    """detect-obb exits 0 with a valid image path (mocked engine)."""
    runner = CliRunner()
    det = _make_obb_detection(frame_index=0)

    with patch("yowo.cli._main.OBBEngine") as mock_cls, patch("yowo.io.open_source") as mock_open:
        mock_open.return_value = MagicMock()
        mock_cls.return_value = _make_mock_obb_engine([det])

        result = runner.invoke(cli, ["detect-obb", "fake_image.jpg", "--model", "yolo11n-obb"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"


def test_detect_obb_command_default_model() -> None:
    """detect-obb uses yolo11n-obb as default --model."""
    runner = CliRunner()
    det = _make_obb_detection()

    with patch("yowo.cli._main.OBBEngine") as mock_cls, patch("yowo.io.open_source"):
        mock_cls.return_value = _make_mock_obb_engine([det])
        result = runner.invoke(cli, ["detect-obb", "img.jpg"])

    # Default model is yolo11n-obb — command should not error about model
    assert result.exit_code == 0


def test_detect_obb_command_json_output() -> None:
    """--json flag outputs valid JSON to stdout."""
    runner = CliRunner()
    box = OBBBox(
        cx=100.0,
        cy=200.0,
        w=50.0,
        h=30.0,
        angle=0.5,
        confidence=0.9,
        class_id=0,
        class_name="plane",
    )
    det = OBBDetection(frame_index=0, source_id="src", boxes=(box,), inference_time_ms=4.0)

    with patch("yowo.cli._main.OBBEngine") as mock_cls, patch("yowo.io.open_source"):
        mock_cls.return_value = _make_mock_obb_engine([det])
        result = runner.invoke(cli, ["detect-obb", "img.jpg", "--json"])

    assert result.exit_code == 0, result.output
    # Output should be parseable JSON
    line = result.output.strip().splitlines()[0]
    data = json.loads(line)
    assert data["frame_index"] == 0
    assert len(data["boxes"]) == 1
    assert data["boxes"][0]["class_name"] == "plane"


def test_detect_obb_help_shows_description() -> None:
    """yowo detect-obb --help shows the command description."""
    runner = CliRunner()
    result = runner.invoke(cli, ["detect-obb", "--help"])
    assert result.exit_code == 0
    assert "oriented bounding box" in result.output.lower() or "obb" in result.output.lower()


def test_detect_obb_registered_in_help() -> None:
    """yowo --help lists detect-obb."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "detect-obb" in result.output


def test_obb_engine_imported_at_module_level() -> None:
    """OBBEngine and OBBConfig are importable from cli._main (module-level)."""
    import yowo.cli._main as m

    assert hasattr(m, "OBBEngine"), "OBBEngine must be imported at module level in cli/_main.py"
    assert hasattr(m, "OBBConfig"), "OBBConfig must be imported at module level in cli/_main.py"
