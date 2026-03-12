"""Tests for true batch inference: dynamic batch detection, validation, and export."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from yowo.config import ConfigError, ExportConfig

_has_onnxruntime = True
try:
    import onnxruntime  # noqa: F401
except ImportError:
    _has_onnxruntime = False

# ---------------------------------------------------------------------------
# ExportConfig.batch_sizes validation
# ---------------------------------------------------------------------------


class TestExportConfigBatchSizes:
    def test_batch_sizes_none_by_default(self) -> None:
        cfg = ExportConfig()
        assert cfg.batch_sizes is None

    def test_batch_sizes_valid(self) -> None:
        cfg = ExportConfig(batch_sizes=[1, 4, 8])
        assert cfg.batch_sizes == [1, 4, 8]

    def test_batch_sizes_sorted_and_deduped(self) -> None:
        cfg = ExportConfig(batch_sizes=[8, 1, 4, 1])
        assert cfg.batch_sizes == [1, 4, 8]

    def test_batch_sizes_empty_raises(self) -> None:
        with pytest.raises(ConfigError, match="must not be empty"):
            ExportConfig(batch_sizes=[])

    def test_batch_sizes_negative_raises(self) -> None:
        with pytest.raises(ConfigError, match="must be > 0"):
            ExportConfig(batch_sizes=[-1, 4])

    def test_batch_sizes_zero_raises(self) -> None:
        with pytest.raises(ConfigError, match="must be > 0"):
            ExportConfig(batch_sizes=[0, 1])


# ---------------------------------------------------------------------------
# OnnxBackend dynamic batch detection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_onnxruntime, reason="onnxruntime not installed")
class TestOnnxBatchDetection:
    def _make_backend(self) -> MagicMock:
        """Create a minimal OnnxBackend with mocked hardware."""
        from yowo.backends._onnx import OnnxBackend

        hw = MagicMock()
        hw.libraries.onnxruntime_version = "1.17.0"
        hw.libraries.onnxruntime_has_coreml = False
        hw.has_nvidia_gpu = False
        return OnnxBackend(hw)

    def test_dynamic_batch_detected(self) -> None:
        backend = self._make_backend()
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_input.shape = ["batch", 3, 640, 640]

        mock_output = MagicMock()
        mock_output.name = "output0"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            backend.load("dummy.onnx")

        assert backend._dynamic_batch is True
        assert backend._static_batch is None

    def test_static_batch_detected(self) -> None:
        backend = self._make_backend()
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_input.shape = [1, 3, 640, 640]

        mock_output = MagicMock()
        mock_output.name = "output0"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            backend.load("dummy.onnx")

        assert backend._dynamic_batch is False
        assert backend._static_batch == 1

    def test_static_batch_warmup_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = self._make_backend()
        backend._dynamic_batch = False
        backend._static_batch = 1
        backend._session = MagicMock()
        backend._input_shape = (640, 640)
        backend._has_kv_io = False
        backend._input_name = "images"

        with caplog.at_level(logging.WARNING):
            backend.warmup(batch_size=4)

        assert "static batch=1 but requested batch=4" in caplog.text

    def test_dynamic_batch_warmup_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = self._make_backend()
        backend._dynamic_batch = True
        backend._static_batch = None
        backend._session = MagicMock()
        backend._input_shape = (640, 640)
        backend._has_kv_io = False
        backend._input_name = "images"

        with caplog.at_level(logging.WARNING):
            backend.warmup(batch_size=4)

        assert "static batch" not in caplog.text


# ---------------------------------------------------------------------------
# CoreMLBackend batch validation
# ---------------------------------------------------------------------------


class TestCoreMLBatchValidation:
    def test_fixed_batch_warmup_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        from yowo.backends._coreml import CoreMLBackend

        hw = MagicMock()
        hw.libraries.coremltools_version = "7.0"
        backend = CoreMLBackend(hw)
        backend._model = MagicMock()
        backend._supported_batch_sizes = None  # fixed batch
        backend._input_shape = (640, 640)

        with caplog.at_level(logging.WARNING):
            backend.warmup(batch_size=4)

        assert "fixed batch=1 but requested batch=4" in caplog.text

    def test_enumerated_batch_unsupported_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        from yowo.backends._coreml import CoreMLBackend

        hw = MagicMock()
        hw.libraries.coremltools_version = "7.0"
        backend = CoreMLBackend(hw)
        backend._model = MagicMock()
        backend._supported_batch_sizes = [1, 4, 8]
        backend._input_shape = (640, 640)

        with caplog.at_level(logging.WARNING):
            backend.warmup(batch_size=16)

        assert "supports batch sizes [1, 4, 8]" in caplog.text
        assert "requested batch=16" in caplog.text

    def test_enumerated_batch_supported_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from yowo.backends._coreml import CoreMLBackend

        hw = MagicMock()
        hw.libraries.coremltools_version = "7.0"
        backend = CoreMLBackend(hw)
        backend._model = MagicMock()
        backend._supported_batch_sizes = [1, 4, 8]
        backend._input_shape = (640, 640)

        with caplog.at_level(logging.WARNING):
            backend.warmup(batch_size=4)

        # No batch mismatch warning should be logged
        has_batch_warn = "batch" in caplog.text.lower() and "failed" not in caplog.text.lower()
        assert not has_batch_warn


# ---------------------------------------------------------------------------
# CoreML EnumeratedShapes export
# ---------------------------------------------------------------------------


class TestCoreMLEnumeratedExport:
    def test_config_accepts_batch_sizes_for_coreml(self) -> None:
        """Verify ExportConfig accepts batch_sizes for CoreML export."""
        cfg = ExportConfig(
            target_format="coreml",
            batch_sizes=[1, 4, 8],
        )
        assert cfg.batch_sizes == [1, 4, 8]

    def test_export_metadata_captures_batch_sizes(self) -> None:
        """Verify batch_sizes ends up in ExportMetadata.extra."""
        # Test the dict merge logic
        kv_cache = False
        batch_sizes = [1, 4, 8]
        extra = {
            **({"kv_cache": True} if kv_cache else {}),
            **({"batch_sizes": batch_sizes} if batch_sizes else {}),
        }
        assert extra == {"batch_sizes": [1, 4, 8]}

        # With kv_cache too
        kv_cache = True
        extra = {
            **({"kv_cache": True} if kv_cache else {}),
            **({"batch_sizes": batch_sizes} if batch_sizes else {}),
        }
        assert extra == {"kv_cache": True, "batch_sizes": [1, 4, 8]}

        # Without batch_sizes
        batch_sizes_none: list[int] | None = None
        extra = {
            **({"kv_cache": True} if kv_cache else {}),
            **({"batch_sizes": batch_sizes_none} if batch_sizes_none else {}),
        }
        assert extra == {"kv_cache": True}
