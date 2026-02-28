"""Unit tests for yowo.cli._main — argument parsing and helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yowo.types import ModelFamily, ModelSize

# ---------------------------------------------------------------------------
# _parse_model_spec
# ---------------------------------------------------------------------------


class TestParseModelSpec:
    def test_yolo26n(self) -> None:
        from yowo.cli._main import _parse_model_spec

        spec = _parse_model_spec("yolo26n")
        assert spec.family == ModelFamily.YOLO26
        assert spec.size == ModelSize.NANO

    def test_yolo11x(self) -> None:
        from yowo.cli._main import _parse_model_spec

        spec = _parse_model_spec("yolo11x")
        assert spec.family == ModelFamily.YOLO11
        assert spec.size == ModelSize.XLARGE

    def test_yolo26s(self) -> None:
        from yowo.cli._main import _parse_model_spec

        spec = _parse_model_spec("yolo26s")
        assert spec.family == ModelFamily.YOLO26
        assert spec.size == ModelSize.SMALL

    def test_yolo11m(self) -> None:
        from yowo.cli._main import _parse_model_spec

        spec = _parse_model_spec("yolo11m")
        assert spec.family == ModelFamily.YOLO11
        assert spec.size == ModelSize.MEDIUM

    def test_yolo26l(self) -> None:
        from yowo.cli._main import _parse_model_spec

        spec = _parse_model_spec("yolo26l")
        assert spec.family == ModelFamily.YOLO26
        assert spec.size == ModelSize.LARGE

    def test_unknown_model_raises_bad_parameter(self) -> None:
        import click

        from yowo.cli._main import _parse_model_spec

        with pytest.raises(click.BadParameter, match="Unknown model"):
            _parse_model_spec("yolov8n")

    def test_unknown_size_raises_bad_parameter(self) -> None:
        import click

        from yowo.cli._main import _parse_model_spec

        with pytest.raises(click.BadParameter, match="Unknown model"):
            _parse_model_spec("yolo26z")


# ---------------------------------------------------------------------------
# _load_zones
# ---------------------------------------------------------------------------


class TestLoadZones:
    def test_returns_none_for_none_input(self) -> None:
        from yowo.cli._main import _load_zones

        assert _load_zones(None) is None

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        from yowo.cli._main import _load_zones

        data = [
            {
                "zone_id": "z1",
                "vertices": [[0, 0], [100, 0], [100, 100], [0, 100]],
            }
        ]
        p = tmp_path / "zones.json"
        p.write_text(json.dumps(data))
        zones = _load_zones(str(p))
        assert zones is not None
        assert len(zones) == 1
        assert zones[0].zone_id == "z1"

    def test_loads_multiple_zones(self, tmp_path: Path) -> None:
        from yowo.cli._main import _load_zones

        data = [
            {"zone_id": "a", "vertices": [[0, 0], [1, 0], [1, 1]]},
            {"zone_id": "b", "vertices": [[2, 2], [3, 2], [3, 3]]},
        ]
        p = tmp_path / "zones.json"
        p.write_text(json.dumps(data))
        zones = _load_zones(str(p))
        assert len(zones) == 2

    def test_class_filter_parsed(self, tmp_path: Path) -> None:
        from yowo.cli._main import _load_zones

        data = [
            {
                "zone_id": "z1",
                "vertices": [[0, 0], [1, 0], [1, 1]],
                "class_filter": ["person", "car"],
            }
        ]
        p = tmp_path / "zones.json"
        p.write_text(json.dumps(data))
        zones = _load_zones(str(p))
        assert zones[0].class_filter == frozenset({"person", "car"})


# ---------------------------------------------------------------------------
# _load_lines
# ---------------------------------------------------------------------------


class TestLoadLines:
    def test_returns_none_for_none_input(self) -> None:
        from yowo.cli._main import _load_lines

        assert _load_lines(None) is None

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        from yowo.cli._main import _load_lines

        data = [{"line_id": "l1", "p1": [0, 0], "p2": [100, 100]}]
        p = tmp_path / "lines.json"
        p.write_text(json.dumps(data))
        lines = _load_lines(str(p))
        assert lines is not None
        assert len(lines) == 1
        assert lines[0].line_id == "l1"

    def test_loads_multiple_lines(self, tmp_path: Path) -> None:
        from yowo.cli._main import _load_lines

        data = [
            {"line_id": "a", "p1": [0, 0], "p2": [1, 1]},
            {"line_id": "b", "p1": [2, 2], "p2": [3, 3]},
        ]
        p = tmp_path / "lines.json"
        p.write_text(json.dumps(data))
        lines = _load_lines(str(p))
        assert len(lines) == 2

    def test_class_filter_parsed(self, tmp_path: Path) -> None:
        from yowo.cli._main import _load_lines

        data = [
            {
                "line_id": "l1",
                "p1": [0, 0],
                "p2": [1, 1],
                "class_filter": ["truck"],
            }
        ]
        p = tmp_path / "lines.json"
        p.write_text(json.dumps(data))
        lines = _load_lines(str(p))
        assert lines[0].class_filter == frozenset({"truck"})


# ---------------------------------------------------------------------------
# _write_json
# ---------------------------------------------------------------------------


class TestWriteJsonHelper:
    def test_writes_detection_list_to_file(self, tmp_path: Path) -> None:
        from yowo.cli._main import _write_json
        from yowo.types import BoundingBox, Detection, Frame, ModelSpec

        frame = Frame(
            pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            source_id="test",
            frame_index=0,
            timestamp_ms=0.0,
        )
        det = Detection(
            frame=frame,
            boxes=(
                BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0, class_name="a"),
            ),
            inference_time_ms=1.0,
            backend="pytorch",
            model_spec=ModelSpec(ModelFamily.YOLO26, ModelSize.NANO),
        )
        out = tmp_path / "out.json"
        _write_json([det], out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_non_detection_objects_skipped(self, tmp_path: Path) -> None:
        from yowo.cli._main import _write_json

        out = tmp_path / "skip.json"
        _write_json(["not a detection", 42], out)
        data = json.loads(out.read_text())
        assert data == []

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        from yowo.cli._main import _write_json

        out = tmp_path / "nested" / "result.json"
        _write_json([], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI commands — smoke tests via Click's CliRunner
# ---------------------------------------------------------------------------


class TestInfoCommand:
    def test_info_command_runs(self) -> None:
        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "Hardware" in result.output
        assert "Libraries" in result.output


class TestModelsCommand:
    def test_models_command_runs(self) -> None:
        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "Model" in result.output

    def test_models_command_family_filter(self) -> None:
        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--family", "yolo26"])
        assert result.exit_code == 0
        # Should only show YOLO26 models
        for line in result.output.strip().split("\n")[2:]:  # Skip header lines
            if line.strip():
                assert "yolo26" in line.lower()
