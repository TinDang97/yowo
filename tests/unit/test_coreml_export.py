"""Tests for CoreML export path.

All tests use mocks — coremltools is NOT required to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yowo.errors import DependencyError, ExportError
from yowo.types import Precision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_coremltools() -> MagicMock:
    """Create a mock coremltools module."""
    mock_ct = MagicMock()
    mock_ct.precision.FLOAT16 = "fp16"
    mock_ct.precision.FLOAT32 = "fp32"
    mock_ct.ComputeUnit.ALL = "all"
    return mock_ct


def _make_mock_torch() -> MagicMock:
    """Create a mock torch module for JIT tracing."""
    mock_torch = MagicMock()
    mock_torch.jit.trace.return_value = MagicMock()
    return mock_torch


# ---------------------------------------------------------------------------
# _convert_coreml tests
# ---------------------------------------------------------------------------


class TestConvertCoreml:
    def test_produces_mlpackage(self, tmp_path: Path) -> None:
        mock_ct = _make_mock_coremltools()
        mock_torch = _make_mock_torch()
        mock_mlmodel = MagicMock()
        mock_ct.convert.return_value = mock_mlmodel

        output = tmp_path / "model.mlpackage"
        dummy = MagicMock()
        dummy.shape = (1, 3, 640, 640)

        with patch.dict(sys.modules, {"coremltools": mock_ct, "torch": mock_torch}):
            from yowo.export._exporter import _convert_coreml

            result = _convert_coreml(MagicMock(), dummy, output, Precision.FP16)

        mock_ct.convert.assert_called_once()
        mock_mlmodel.save.assert_called_once_with(str(output))
        assert result == output

    def test_fp32_precision_uses_float32(self, tmp_path: Path) -> None:
        mock_ct = _make_mock_coremltools()
        mock_torch = _make_mock_torch()
        mock_ct.convert.return_value = MagicMock()

        output = tmp_path / "model.mlpackage"
        dummy = MagicMock()
        dummy.shape = (1, 3, 640, 640)

        with patch.dict(sys.modules, {"coremltools": mock_ct, "torch": mock_torch}):
            from yowo.export._exporter import _convert_coreml

            _convert_coreml(MagicMock(), dummy, output, Precision.FP32)

        call_kwargs = mock_ct.convert.call_args
        assert call_kwargs[1]["compute_precision"] == "fp32"

    def test_conversion_failure_raises_export_error(self, tmp_path: Path) -> None:
        mock_ct = _make_mock_coremltools()
        mock_torch = _make_mock_torch()
        mock_ct.convert.side_effect = RuntimeError("conversion failed")

        output = tmp_path / "model.mlpackage"
        dummy = MagicMock()
        dummy.shape = (1, 3, 640, 640)

        with patch.dict(sys.modules, {"coremltools": mock_ct, "torch": mock_torch}):
            from yowo.export._exporter import _convert_coreml

            with pytest.raises(ExportError, match="CoreML conversion failed"):
                _convert_coreml(MagicMock(), dummy, output, Precision.FP32)

    def test_missing_coremltools_raises_dependency_error(self, tmp_path: Path) -> None:
        output = tmp_path / "model.mlpackage"

        # Setting to None in sys.modules prevents import
        with patch.dict(sys.modules, {"coremltools": None}):
            from yowo.export._exporter import _convert_coreml

            with pytest.raises((DependencyError, ImportError)):
                _convert_coreml(MagicMock(), MagicMock(), output, Precision.FP32)
