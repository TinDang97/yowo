"""Tests for yowo tune CLI subcommand."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from yowo.cli._main import cli


@dataclass
class _FakeSweepResult:
    backend: str
    batch_size: int
    precision: str
    fps: float
    skipped: bool = False
    skip_reason: str = ""


def _make_hw():
    """Return a minimal HardwareProfile-like object for mocking."""
    hw = MagicMock()
    hw.has_nvidia_gpu = False
    hw.primary_gpu = None
    return hw


def _make_sweep_result(
    backend: str = "pytorch",
    batch_size: int = 1,
    precision: str = "fp32",
    fps: float = 42.0,
) -> _FakeSweepResult:
    return _FakeSweepResult(
        backend=backend,
        batch_size=batch_size,
        precision=precision,
        fps=fps,
    )


class TestTuneCommandExists:
    def test_tune_command_exists(self) -> None:
        """yowo tune --help exits 0 and lists all expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tune", "--help"])
        assert result.exit_code == 0, result.output
        assert "--model" in result.output
        assert "--weights" in result.output
        assert "--output" in result.output
        assert "--force" in result.output
        assert "--dry-run" in result.output
        assert "--json" in result.output


class TestTuneDryRunFlag:
    def test_tune_dry_run_skips_sweep(self) -> None:
        """--dry-run prints 'Dry run' message and does NOT call run_sweep."""
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep") as mock_sweep,
            patch("yowo.cli._main.load_profile", return_value=None),
            patch("yowo.cli._main._enumerate_sweep_dimensions", return_value=18),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n", "--dry-run"])
        assert result.exit_code == 0, result.output
        mock_sweep.assert_not_called()
        assert "Dry run" in result.output

    def test_tune_dry_run_shows_config_count(self) -> None:
        """--dry-run output contains configuration count."""
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep"),
            patch("yowo.cli._main.load_profile", return_value=None),
            patch("yowo.cli._main._enumerate_sweep_dimensions", return_value=12),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo26n", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "12" in result.output


class TestTuneJsonFlag:
    def test_tune_json_flag_outputs_json_array(self) -> None:
        """--json flag prints a valid JSON array of results to stdout."""
        fake_result = _make_sweep_result()
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep", return_value=[fake_result]),
            patch("yowo.cli._main.load_profile", return_value=None),
            patch("yowo.cli._main.save_profile"),
            patch("yowo.cli._main.compute_fingerprint", return_value="abc12345"),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n", "--json"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["backend"] == "pytorch"
        assert parsed[0]["fps"] == 42.0

    def test_tune_json_flag_no_rich_table(self) -> None:
        """--json flag does not render a rich table (no 'Backend' column header)."""
        fake_result = _make_sweep_result()
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep", return_value=[fake_result]),
            patch("yowo.cli._main.load_profile", return_value=None),
            patch("yowo.cli._main.save_profile"),
            patch("yowo.cli._main.compute_fingerprint", return_value="abc12345"),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n", "--json"])
        # Rich table headers should not appear in JSON-only output
        output_lines = result.output.strip().splitlines()
        # Only the JSON array should be output; no "Best config:" line
        assert not any("Best config:" in line for line in output_lines)


class TestTuneSavesProfile:
    def test_tune_saves_profile_after_sweep(self) -> None:
        """Without --dry-run or --json, save_profile is called with best result."""
        fake_result = _make_sweep_result(fps=99.0)
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep", return_value=[fake_result]),
            patch("yowo.cli._main.load_profile", return_value=None),
            patch("yowo.cli._main.save_profile") as mock_save,
            patch("yowo.cli._main.compute_fingerprint", return_value="abc12345"),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n"])
        assert result.exit_code == 0, result.output
        mock_save.assert_called_once()
        saved_profile = mock_save.call_args[0][0]
        assert saved_profile.fps_achieved == 99.0
        assert saved_profile.backend == "pytorch"
        assert saved_profile.model == "yolo11n"

    def test_tune_prints_profile_saved(self) -> None:
        """Output contains 'Profile saved' confirmation."""
        fake_result = _make_sweep_result()
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep", return_value=[fake_result]),
            patch("yowo.cli._main.load_profile", return_value=None),
            patch("yowo.cli._main.save_profile"),
            patch("yowo.cli._main.compute_fingerprint", return_value="abc12345"),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n"])
        assert result.exit_code == 0, result.output
        assert "Profile saved" in result.output

    def test_tune_prints_best_config(self) -> None:
        """Output contains 'Best config:' summary line."""
        fake_result = _make_sweep_result(fps=55.5)
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep", return_value=[fake_result]),
            patch("yowo.cli._main.load_profile", return_value=None),
            patch("yowo.cli._main.save_profile"),
            patch("yowo.cli._main.compute_fingerprint", return_value="abc12345"),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n"])
        assert result.exit_code == 0, result.output
        assert "Best config:" in result.output
        assert "55.5" in result.output


class TestTuneForceFlag:
    def test_tune_skips_when_profile_exists(self) -> None:
        """Without --force, skip sweep if profile already exists."""
        existing_profile = MagicMock()
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep") as mock_sweep,
            patch("yowo.cli._main.load_profile", return_value=existing_profile),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n"])
        assert result.exit_code == 0, result.output
        mock_sweep.assert_not_called()
        assert "Profile exists" in result.output

    def test_tune_force_re_runs_sweep(self) -> None:
        """--force runs sweep even when profile already exists."""
        existing_profile = MagicMock()
        fake_result = _make_sweep_result()
        runner = CliRunner()
        with (
            patch("yowo.cli._main.get_hardware_profile", return_value=_make_hw()),
            patch("yowo.cli._main.run_sweep", return_value=[fake_result]) as mock_sweep,
            patch("yowo.cli._main.load_profile", return_value=existing_profile),
            patch("yowo.cli._main.save_profile"),
            patch("yowo.cli._main.compute_fingerprint", return_value="abc12345"),
        ):
            result = runner.invoke(cli, ["tune", "--model", "yolo11n", "--force"])
        assert result.exit_code == 0, result.output
        mock_sweep.assert_called_once()
