"""Tests for yowo batch CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from yowo.cli._main import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source_dir(tmp_path: Path) -> Path:
    """Create a temporary directory to use as source_dir."""
    src = tmp_path / "images"
    src.mkdir()
    return src


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_batch_command_exists() -> None:
    """yowo batch --help returns exit code 0 and lists expected options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", "--help"])
    assert result.exit_code == 0
    assert "SOURCE_DIR" in result.output or "source_dir" in result.output.lower()
    assert "--model" in result.output
    assert "--output" in result.output
    assert "--no-annotate" in result.output
    assert "--recursive" in result.output
    assert "--no-resume" in result.output
    assert "--format" in result.output
    assert "--workers" in result.output


def test_batch_requires_output(tmp_path: Path) -> None:
    """Invoking without --output raises UsageError (missing required option)."""
    src = _make_source_dir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", str(src), "--model", "yolo11n"])
    # Click's required option validation returns exit code 2
    assert result.exit_code != 0
    assert "output" in result.output.lower() or "output" in (result.stderr or "").lower()


def test_batch_no_resume_flag(tmp_path: Path) -> None:
    """--no-resume passes BatchConfig.no_resume=True to run_batch."""
    src = _make_source_dir(tmp_path)
    out = tmp_path / "out"

    captured: list[object] = []

    def fake_run_batch(cfg: object, engine: object) -> int:
        captured.append(cfg)
        return 0

    runner = CliRunner()
    with (
        patch("yowo.cli._main.run_batch", side_effect=fake_run_batch),
        patch("yowo.cli._main.DetectionEngine") as mock_engine_cls,
    ):
        mock_engine = MagicMock()
        mock_engine.load.return_value = None
        mock_engine.close.return_value = None
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            cli,
            ["batch", str(src), "--model", "yolo11n", "--output", str(out), "--no-resume"],
        )

    # sys.exit(0) raises SystemExit which CliRunner captures as exit_code 0
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.no_resume is True  # type: ignore[union-attr]


def test_batch_recursive_flag(tmp_path: Path) -> None:
    """--recursive passes BatchConfig.recursive=True to run_batch."""
    src = _make_source_dir(tmp_path)
    out = tmp_path / "out"

    captured: list[object] = []

    def fake_run_batch(cfg: object, engine: object) -> int:
        captured.append(cfg)
        return 0

    runner = CliRunner()
    with (
        patch("yowo.cli._main.run_batch", side_effect=fake_run_batch),
        patch("yowo.cli._main.DetectionEngine") as mock_engine_cls,
    ):
        mock_engine = MagicMock()
        mock_engine.load.return_value = None
        mock_engine.close.return_value = None
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            cli,
            ["batch", str(src), "--model", "yolo11n", "--output", str(out), "--recursive"],
        )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.recursive is True  # type: ignore[union-attr]


def test_batch_format_json(tmp_path: Path) -> None:
    """--format json passes BatchConfig.output_format='json' to run_batch."""
    src = _make_source_dir(tmp_path)
    out = tmp_path / "out"

    captured: list[object] = []

    def fake_run_batch(cfg: object, engine: object) -> int:
        captured.append(cfg)
        return 0

    runner = CliRunner()
    with (
        patch("yowo.cli._main.run_batch", side_effect=fake_run_batch),
        patch("yowo.cli._main.DetectionEngine") as mock_engine_cls,
    ):
        mock_engine = MagicMock()
        mock_engine.load.return_value = None
        mock_engine.close.return_value = None
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            cli,
            ["batch", str(src), "--model", "yolo11n", "--output", str(out), "--format", "json"],
        )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.output_format == "json"  # type: ignore[union-attr]


def test_batch_no_annotate_flag(tmp_path: Path) -> None:
    """--no-annotate passes BatchConfig.no_annotate=True to run_batch."""
    src = _make_source_dir(tmp_path)
    out = tmp_path / "out"

    captured: list[object] = []

    def fake_run_batch(cfg: object, engine: object) -> int:
        captured.append(cfg)
        return 0

    runner = CliRunner()
    with (
        patch("yowo.cli._main.run_batch", side_effect=fake_run_batch),
        patch("yowo.cli._main.DetectionEngine") as mock_engine_cls,
    ):
        mock_engine = MagicMock()
        mock_engine.load.return_value = None
        mock_engine.close.return_value = None
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            cli,
            ["batch", str(src), "--model", "yolo11n", "--output", str(out), "--no-annotate"],
        )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.no_annotate is True  # type: ignore[union-attr]


def test_batch_workers_mapping(tmp_path: Path) -> None:
    """--workers N maps to BatchConfig.workers=N."""
    src = _make_source_dir(tmp_path)
    out = tmp_path / "out"

    captured: list[object] = []

    def fake_run_batch(cfg: object, engine: object) -> int:
        captured.append(cfg)
        return 0

    runner = CliRunner()
    with (
        patch("yowo.cli._main.run_batch", side_effect=fake_run_batch),
        patch("yowo.cli._main.DetectionEngine") as mock_engine_cls,
    ):
        mock_engine = MagicMock()
        mock_engine.load.return_value = None
        mock_engine.close.return_value = None
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(
            cli,
            ["batch", str(src), "--model", "yolo11n", "--output", str(out), "--workers", "4"],
        )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.workers == 4  # type: ignore[union-attr]


def test_batch_invalid_workers(tmp_path: Path) -> None:
    """--workers -1 triggers UsageError (negative not allowed)."""
    src = _make_source_dir(tmp_path)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["batch", str(src), "--model", "yolo11n", "--output", str(out), "--workers", "-1"],
    )
    assert result.exit_code != 0
