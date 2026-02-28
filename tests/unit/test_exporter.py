"""Unit tests for yowo.export._exporter — export_model validation paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from yowo.errors import ConfigError, DependencyError
from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)


# ---------------------------------------------------------------------------
# export_model — validation
# ---------------------------------------------------------------------------


class TestExportModelValidation:
    def test_int8_without_calibration_raises_config_error(self, tmp_path: Path) -> None:
        from yowo.export._exporter import export_model

        with pytest.raises(ConfigError, match="INT8"):
            export_model(
                _SPEC,
                ExportFormat.ONNX,
                tmp_path,
                precision=Precision.INT8,
                calibration_data=None,
            )

    def test_int8_with_calibration_passes_validation(self, tmp_path: Path) -> None:
        """INT8 validation should pass when calibration_data is provided.

        We only test that the ConfigError is NOT raised — the actual export
        will fail with DependencyError or similar (torch not available in CI).
        """
        from yowo.export._exporter import export_model

        # This should NOT raise ConfigError, but may raise DependencyError
        # for torch — that's fine, we're testing the INT8 validation gate.
        try:
            export_model(
                _SPEC,
                ExportFormat.ONNX,
                tmp_path,
                precision=Precision.INT8,
                calibration_data="/some/path",
            )
        except ConfigError:
            pytest.fail("ConfigError raised even though calibration_data was provided")
        except (DependencyError, Exception):
            pass  # Expected — torch may not be installed


# ---------------------------------------------------------------------------
# ExportConfig — batch_sizes validation
# ---------------------------------------------------------------------------


class TestExportConfigBatchSizes:
    def test_empty_batch_sizes_raises(self) -> None:
        from yowo.config import ExportConfig

        with pytest.raises(ConfigError, match="batch_sizes must not be empty"):
            ExportConfig(batch_sizes=[])

    def test_negative_batch_size_raises(self) -> None:
        from yowo.config import ExportConfig

        with pytest.raises(ConfigError, match="batch_sizes values must be > 0"):
            ExportConfig(batch_sizes=[-1, 1])

    def test_zero_batch_size_raises(self) -> None:
        from yowo.config import ExportConfig

        with pytest.raises(ConfigError, match="batch_sizes values must be > 0"):
            ExportConfig(batch_sizes=[0, 4])

    def test_valid_batch_sizes_sorted_and_deduped(self) -> None:
        from yowo.config import ExportConfig

        cfg = ExportConfig(batch_sizes=[8, 1, 4, 1])
        assert cfg.batch_sizes == [1, 4, 8]

    def test_none_batch_sizes_is_valid(self) -> None:
        from yowo.config import ExportConfig

        cfg = ExportConfig(batch_sizes=None)
        assert cfg.batch_sizes is None

    def test_imgsz_zero_raises(self) -> None:
        from yowo.config import ExportConfig

        with pytest.raises(ConfigError, match="imgsz"):
            ExportConfig(imgsz=0)

    def test_imgsz_negative_raises(self) -> None:
        from yowo.config import ExportConfig

        with pytest.raises(ConfigError, match="imgsz"):
            ExportConfig(imgsz=-640)
