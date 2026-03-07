"""Unit tests for warmup validation in engine load().

Covers WarmupValidationError for bad shapes, bad detection ranges,
bad classification softmax, and the happy path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.backends import InferenceBackend
from yowo.classify_engine import ClassificationEngine
from yowo.engine import DetectionEngine
from yowo.errors import BackendError, WarmupValidationError
from yowo.types import BackendType

_RESOLVE_PATCH = "yowo.engine.resolve_weights"


def _make_mock_backend(*, infer_output: np.ndarray | None = None) -> MagicMock:
    """Return a MagicMock satisfying the InferenceBackend protocol."""
    mock = MagicMock(spec=InferenceBackend)
    mock.backend_type = BackendType.PYTORCH
    mock.is_loaded = False
    mock.input_shape = (640, 640)

    def _load(*a: object, **kw: object) -> None:
        mock.is_loaded = True

    def _unload(*a: object, **kw: object) -> None:
        mock.is_loaded = False

    mock.load.side_effect = _load
    mock.unload.side_effect = _unload
    mock.warmup.return_value = None
    if infer_output is not None:
        mock.infer.return_value = infer_output
    else:
        mock.infer.return_value = np.zeros((1, 0, 6), dtype=np.float32)
    return mock


def _make_cls_backend(*, infer_output: np.ndarray | None = None) -> MagicMock:
    """Backend for classification engine tests."""
    mock = _make_mock_backend(infer_output=infer_output)
    if infer_output is None:
        # Default: valid softmax output (uniform over 1000 classes)
        out = np.full((1, 1000), 1.0 / 1000, dtype=np.float32)
        mock.infer.return_value = out
    return mock


# ---------------------------------------------------------------------------
# WarmupValidationError class tests
# ---------------------------------------------------------------------------


class TestWarmupValidationErrorClass:
    def test_class_exists_and_inherits_backend_error(self) -> None:
        assert issubclass(WarmupValidationError, BackendError)

    def test_in_errors_all(self) -> None:
        from yowo import errors

        assert "WarmupValidationError" in errors.__all__

    def test_message_format(self) -> None:
        exc = WarmupValidationError("some detail")
        assert "Warmup validation failed: some detail" in str(exc)
        assert "corrupt or incompatible" in str(exc)


# ---------------------------------------------------------------------------
# Detection warmup validation
# ---------------------------------------------------------------------------


class TestDetectionWarmupValidation:
    def test_bad_shape_1d_raises(self) -> None:
        """Engine rejects model output with ndim < 2."""
        bad_output = np.zeros((10,), dtype=np.float32)
        mock_be = _make_mock_backend(infer_output=bad_output)
        engine = DetectionEngine(backend_instance=mock_be)
        with (
            pytest.raises(WarmupValidationError, match="ndim >= 2"),
            patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")),
        ):
            engine.load()

    def test_bad_confidence_range_above_one(self) -> None:
        """Engine accepts detection output with raw logits > 1 (valid YOLO output)."""
        logit_output = np.zeros((1, 84, 10), dtype=np.float32)
        logit_output[:, 4:, :] = 636.0  # raw logits before sigmoid
        mock_be = _make_mock_backend(infer_output=logit_output)
        engine = DetectionEngine(backend_instance=mock_be)
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
            engine.load()
        assert engine.is_loaded
        engine.close()

    def test_bad_confidence_range_negative(self) -> None:
        """Engine accepts detection output with negative raw logits (valid YOLO output)."""
        logit_output = np.zeros((1, 84, 10), dtype=np.float32)
        logit_output[:, 4:, :] = -15.0  # negative logits map to near-zero after sigmoid
        mock_be = _make_mock_backend(infer_output=logit_output)
        engine = DetectionEngine(backend_instance=mock_be)
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
            engine.load()
        assert engine.is_loaded
        engine.close()

    def test_good_output_passes(self) -> None:
        """Valid output shape and range allows engine to load normally."""
        good_output = np.zeros((1, 84, 10), dtype=np.float32)
        good_output[:, 4:, :] = 0.5  # valid range
        mock_be = _make_mock_backend(infer_output=good_output)
        engine = DetectionEngine(backend_instance=mock_be)
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
            engine.load()
        assert engine.is_loaded
        engine.close()

    def test_empty_detection_output_passes(self) -> None:
        """Zero-detection output (shape (1, 0, 6)) passes validation."""
        empty = np.zeros((1, 0, 6), dtype=np.float32)
        mock_be = _make_mock_backend(infer_output=empty)
        engine = DetectionEngine(backend_instance=mock_be)
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
            engine.load()
        assert engine.is_loaded
        engine.close()

    def test_backend_infer_exception_wrapped(self) -> None:
        """Exception from backend.infer() during warmup is wrapped."""
        mock_be = _make_mock_backend()
        mock_be.infer.side_effect = RuntimeError("GPU OOM")
        engine = DetectionEngine(backend_instance=mock_be)
        with (
            pytest.raises(WarmupValidationError, match="GPU OOM"),
            patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")),
        ):
            engine.load()

    def test_nan_in_output_raises(self) -> None:
        """Engine rejects detection output containing NaN."""
        bad_output = np.zeros((1, 84, 10), dtype=np.float32)
        bad_output[0, 0, 0] = float("nan")
        mock_be = _make_mock_backend(infer_output=bad_output)
        engine = DetectionEngine(backend_instance=mock_be)
        with (
            pytest.raises(WarmupValidationError, match="NaN"),
            patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")),
        ):
            engine.load()

    def test_inf_in_output_raises(self) -> None:
        """Engine rejects detection output containing Inf."""
        bad_output = np.zeros((1, 84, 10), dtype=np.float32)
        bad_output[0, 0, 0] = float("inf")
        mock_be = _make_mock_backend(infer_output=bad_output)
        engine = DetectionEngine(backend_instance=mock_be)
        with (
            pytest.raises(WarmupValidationError, match="Inf"),
            patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")),
        ):
            engine.load()


# ---------------------------------------------------------------------------
# Classification warmup validation
# ---------------------------------------------------------------------------


class TestClassificationWarmupValidation:
    def test_bad_softmax_sum(self) -> None:
        """Classification engine rejects output that doesn't sum to ~1.0."""
        bad_output = np.full((1, 1000), 0.5, dtype=np.float32)  # sum = 500
        mock_be = _make_cls_backend(infer_output=bad_output)
        engine = ClassificationEngine(backend_instance=mock_be)
        with (
            pytest.raises(WarmupValidationError, match="softmax sum"),
            patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")),
        ):
            engine.load()

    def test_good_classification_output_passes(self) -> None:
        """Valid softmax output allows classification engine to load."""
        good_output = np.zeros((1, 1000), dtype=np.float32)
        good_output[0, 0] = 1.0  # one-hot = valid softmax
        mock_be = _make_cls_backend(infer_output=good_output)
        engine = ClassificationEngine(backend_instance=mock_be)
        with patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")):
            engine.load()
        assert engine.is_loaded
        engine.close()

    def test_classification_negative_values_rejected(self) -> None:
        """Classification output with negative values is rejected."""
        bad_output = np.full((1, 10), -0.5, dtype=np.float32)
        mock_be = _make_cls_backend(infer_output=bad_output)
        engine = ClassificationEngine(backend_instance=mock_be)
        with (
            pytest.raises(WarmupValidationError),
            patch(_RESOLVE_PATCH, return_value=Path("/fake/w.pt")),
        ):
            engine.load()
