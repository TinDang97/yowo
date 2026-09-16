"""Tests for yowo.tune._sweep — Plan 02 implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from yowo.errors import BackendError
from yowo.hardware import HardwareProfile
from yowo.types import BackendType

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_hw(
    *,
    has_nvidia_gpu: bool = False,
    tensorrt_available: bool = False,
) -> MagicMock:
    """Build a minimal HardwareProfile mock for sweep tests."""
    hw = MagicMock(spec=HardwareProfile)
    hw.has_nvidia_gpu = has_nvidia_gpu
    hw.primary_gpu = MagicMock() if has_nvidia_gpu else None
    libs = MagicMock()
    libs.tensorrt_version = "8.6" if tensorrt_available else None
    hw.libraries = libs
    return hw


# ---------------------------------------------------------------------------
# Task 1 tests: SweepResult dataclass
# ---------------------------------------------------------------------------


class TestSweepResult:
    def test_sweep_result_fields(self) -> None:
        from yowo.tune._sweep import SweepResult

        r = SweepResult(backend="pytorch", batch_size=2, precision="fp32", fps=42.0)
        assert r.backend == "pytorch"
        assert r.batch_size == 2
        assert r.precision == "fp32"
        assert r.fps == 42.0
        assert r.skipped is False
        assert r.skip_reason == ""

    def test_sweep_result_skipped_defaults(self) -> None:
        from yowo.tune._sweep import SweepResult

        r = SweepResult(
            backend="onnx",
            batch_size=16,
            precision="fp16",
            fps=0.0,
            skipped=True,
            skip_reason="OOM",
        )
        assert r.skipped is True
        assert r.skip_reason == "OOM"


# ---------------------------------------------------------------------------
# Task 1 tests: _enumerate_backends
# ---------------------------------------------------------------------------


class TestEnumerateBackends:
    def test_returns_available_backends(self) -> None:
        from yowo.tune._sweep import _enumerate_backends

        hw = _make_hw()

        def check_avail(backend: BackendType, _hw: HardwareProfile) -> None:
            if backend == BackendType.PYTORCH:
                return  # available
            raise BackendError("not available")

        with patch("yowo.tune._sweep.check_backend_available", side_effect=check_avail):
            result = _enumerate_backends(hw)

        assert result == [BackendType.PYTORCH]

    def test_returns_multiple_available_backends(self) -> None:
        from yowo.tune._sweep import _enumerate_backends

        hw = _make_hw()

        def check_avail(backend: BackendType, _hw: HardwareProfile) -> None:
            if backend in (BackendType.PYTORCH, BackendType.ONNX):
                return
            raise BackendError("not available")

        with patch("yowo.tune._sweep.check_backend_available", side_effect=check_avail):
            result = _enumerate_backends(hw)

        # Order preserved from _SWEEP_BACKENDS: ONNX before PYTORCH
        assert BackendType.PYTORCH in result
        assert BackendType.ONNX in result

    def test_returns_empty_when_none_available(self) -> None:
        from yowo.tune._sweep import _enumerate_backends

        hw = _make_hw()

        with patch(
            "yowo.tune._sweep.check_backend_available",
            side_effect=BackendError("none"),
        ):
            result = _enumerate_backends(hw)

        assert result == []


# ---------------------------------------------------------------------------
# `_precisions_for_backend` and the tests that covered it are GONE.
#
# /tasks/precision-plumbing.md dropped the sweep's precision axis. Nothing
# consumed `precision` when these rows were swept, so every precision row
# measured an IDENTICAL configuration and the stored winner was FPS noise
# picked between indistinguishable runs. Worse, once an explicit request became
# loud, the sweep's own `precision=` argument would have raised on every
# non-PyTorch row on every host — the sweep was a user request as far as the
# engine could tell.
#
# A row's `precision` is now READ BACK from the engine after load: what
# executed, not what was asked for.
# ---------------------------------------------------------------------------


class TestRunSweep:
    def _make_spec(self) -> MagicMock:
        from yowo.types import ModelFamily, ModelSize

        spec = MagicMock()
        spec.family = ModelFamily.YOLO11
        spec.size = ModelSize.NANO
        spec.task = "detect"
        spec.num_classes = None
        return spec

    def test_sweep_returns_sorted_results(self) -> None:
        from yowo.tune._sweep import run_sweep

        hw = _make_hw()
        spec = self._make_spec()

        # Only PYTORCH available; mock _measure_config to return predictable fps
        fps_by_batch: dict[int, float] = {1: 10.0, 2: 20.0, 4: 5.0}

        def fake_measure(_spec, _hw, _backend, batch_size, _warmup, _measure, _frames=None):
            return fps_by_batch[batch_size], "fp32", False

        with (
            patch(
                "yowo.tune._sweep._enumerate_backends",
                return_value=[BackendType.PYTORCH],
            ),
            patch("yowo.tune._sweep._BATCH_SIZES", [1, 2, 4]),
            patch("yowo.tune._sweep._measure_config", side_effect=fake_measure),
        ):
            results = run_sweep(spec, hw, warmup_frames=1, measure_frames=1)

        assert len(results) == 3
        # Sorted by fps descending: 20, 10, 5
        assert results[0].fps == 20.0
        assert results[1].fps == 10.0
        assert results[2].fps == 5.0
        assert all(not r.skipped for r in results)

    def test_sorted_tiebreak_by_batch_size(self) -> None:
        from yowo.tune._sweep import run_sweep

        hw = _make_hw()
        spec = self._make_spec()

        def fake_measure(_spec, _hw, _backend, batch_size, _warmup, _measure, _frames=None):
            return 30.0, "fp32", False  # same fps for all

        with (
            patch(
                "yowo.tune._sweep._enumerate_backends",
                return_value=[BackendType.PYTORCH],
            ),
            patch("yowo.tune._sweep._BATCH_SIZES", [2, 4, 1]),
            patch("yowo.tune._sweep._measure_config", side_effect=fake_measure),
        ):
            results = run_sweep(spec, hw, warmup_frames=1, measure_frames=1)

        # Tiebreak: smaller batch_size wins
        assert results[0].batch_size == 1
        assert results[1].batch_size == 2
        assert results[2].batch_size == 4

    def test_oom_skip_higher_batch_sizes(self) -> None:
        import torch

        from yowo.tune._sweep import run_sweep

        hw = _make_hw()
        spec = self._make_spec()

        call_count = 0

        def fake_measure(_spec, _hw, _backend, batch_size, _warmup, _measure, _frames=None):
            nonlocal call_count
            call_count += 1
            if batch_size >= 4:
                raise torch.cuda.OutOfMemoryError("OOM")
            return 10.0, "fp32", False

        with (
            patch(
                "yowo.tune._sweep._enumerate_backends",
                return_value=[BackendType.PYTORCH],
            ),
            patch("yowo.tune._sweep._BATCH_SIZES", [1, 2, 4, 8, 16, 32]),
            patch("yowo.tune._sweep._measure_config", side_effect=fake_measure),
        ):
            results = run_sweep(spec, hw, warmup_frames=1, measure_frames=1)

        # batch_size 1 and 2 should be in non-skipped results
        non_skipped = [r for r in results if not r.skipped]
        assert len(non_skipped) == 2
        assert {r.batch_size for r in non_skipped} == {1, 2}

    def test_dry_run_returns_empty_list(self) -> None:
        from yowo.tune._sweep import run_sweep

        hw = _make_hw()
        spec = self._make_spec()

        # dry_run=True exits before any backend enumeration or engine creation
        results = run_sweep(spec, hw, dry_run=True)

        assert results == []

    def test_cuda_empty_cache_called_on_oom(self) -> None:
        import torch

        from yowo.tune._sweep import run_sweep

        hw = _make_hw()
        spec = self._make_spec()

        def fake_measure(_spec, _hw, _backend, batch_size, _warmup, _measure, _frames=None):
            raise torch.cuda.OutOfMemoryError("OOM")

        with (
            patch(
                "yowo.tune._sweep._enumerate_backends",
                return_value=[BackendType.PYTORCH],
            ),
            patch("yowo.tune._sweep._BATCH_SIZES", [1]),
            patch("yowo.tune._sweep._measure_config", side_effect=fake_measure),
            patch("yowo.tune._sweep.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.OutOfMemoryError = torch.cuda.OutOfMemoryError
            run_sweep(spec, hw, warmup_frames=1, measure_frames=1)

        mock_torch.cuda.empty_cache.assert_called()

    def test_engine_closed_on_oom(self) -> None:
        import torch

        from yowo.tune._sweep import run_sweep

        hw = _make_hw()
        spec = self._make_spec()

        def fake_measure(_spec, _hw, _backend, batch_size, _warmup, _measure, _frames=None):
            raise torch.cuda.OutOfMemoryError("OOM")

        # We can't directly observe engine.close() from run_sweep since
        # _measure_config handles its own engine lifecycle.
        # Instead, verify the sweep doesn't crash and skipped results are returned
        # (indirectly proves finally block executed).
        with (
            patch(
                "yowo.tune._sweep._enumerate_backends",
                return_value=[BackendType.PYTORCH],
            ),
            patch("yowo.tune._sweep._BATCH_SIZES", [1, 2]),
            patch("yowo.tune._sweep._measure_config", side_effect=fake_measure),
        ):
            results = run_sweep(spec, hw, warmup_frames=1, measure_frames=1)

        # Both batch sizes skipped — no non-skipped results returned
        assert results == []


# ---------------------------------------------------------------------------
# Phase 07 tests: TestMeasureConfigDispatch
# ---------------------------------------------------------------------------


class TestMeasureConfigDispatch:
    """Regression tests verifying _measure_config dispatches the correct engine per task."""

    def _make_spec(self, task: str) -> MagicMock:
        from yowo.types import ModelFamily, ModelSize

        spec = MagicMock()
        spec.family = ModelFamily.YOLO11
        spec.size = ModelSize.NANO
        spec.task = task
        spec.num_classes = None
        spec.weights_path = None
        return spec

    def test_measure_config_dispatches_obb_engine_for_task_obb(self) -> None:
        from yowo.tune._sweep import _measure_config

        hw = _make_hw()
        spec = self._make_spec("obb")
        mock_engine = MagicMock()
        mock_engine.detect_obb.return_value = [MagicMock()]

        # Patch at the source module since OBBEngine is lazily imported inside
        # _measure_config via `from yowo.obb_engine import OBBEngine` — the name
        # is bound in function scope, not in _sweep's module dict.
        with patch("yowo.obb_engine.OBBEngine", return_value=mock_engine):
            mock_engine.load.return_value = None
            _measure_config(spec, hw, BackendType.PYTORCH, 1, 1, 1)

        mock_engine.detect_obb.assert_called()

    def test_measure_config_dispatches_detection_engine_for_task_detect(self) -> None:
        from yowo.tune._sweep import _measure_config

        hw = _make_hw()
        spec = self._make_spec("detect")
        mock_engine = MagicMock()

        # Patch at the source module since DetectionEngine is lazily imported inside
        # _measure_config via `from yowo.engine import DetectionEngine`.
        with patch("yowo.engine.DetectionEngine", return_value=mock_engine):
            mock_engine.load.return_value = None
            _measure_config(spec, hw, BackendType.PYTORCH, 1, 1, 1)

        mock_engine.detect.assert_called()
