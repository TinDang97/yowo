"""Tests for yowo.benchmark._runner module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.benchmark._runner import BenchmarkResult, run_all_backends, run_single_backend
from yowo.types import (
    BackendType,
    BoundingBox,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
)


def _make_detection(inference_time_ms: float = 10.0) -> Detection:
    """Create a mock Detection result."""
    frame = Frame(pixels=np.zeros((480, 640, 3), dtype=np.uint8), source_id="bench")
    spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
    return Detection(
        frame=frame,
        boxes=(BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0, confidence=0.9, class_id=0),),
        inference_time_ms=inference_time_ms,
        backend=BackendType.PYTORCH,
        model_spec=spec,
    )


class TestBenchmarkResult:
    def test_dataclass_fields(self) -> None:
        result = BenchmarkResult(
            format="pytorch",
            map_50_95=0.35,
            map_50=0.55,
            fps_avg=120.5,
            latency_p50_ms=8.3,
            latency_p95_ms=12.1,
            latency_p99_ms=15.4,
            model_size_mb=6.2,
            device="cpu",
            num_images=100,
        )
        assert result.format == "pytorch"
        assert result.map_50_95 == 0.35
        assert result.fps_avg == 120.5
        assert result.latency_p50_ms == 8.3
        assert result.latency_p95_ms == 12.1
        assert result.latency_p99_ms == 15.4


class TestRunnerWarmupExcluded:
    """FPS measurement must exclude warmup passes."""

    @patch("yowo.benchmark._runner.DetectionEngine")
    def test_warmup_passes_not_counted_in_fps(self, mock_engine_cls: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.detect.return_value = [_make_detection()]

        spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
        images = [Path(f"/fake/img{i}.jpg") for i in range(5)]

        result = run_single_backend(
            model_spec=spec,
            backend_type=BackendType.PYTORCH,
            images=images,
            image_ids=None,
            gt_ann_path=None,
            task="detect",
            warmup_passes=3,
        )

        # warmup (3) + actual (5) = 8 total detect calls
        assert mock_engine.detect.call_count == 8
        # FPS should be based on 5 images, not 8
        assert result.num_images == 5


class TestRunnerLatencyPercentiles:
    """Runner must report p50, p95, p99 latency."""

    @patch("yowo.benchmark._runner.DetectionEngine")
    def test_latency_percentiles_computed(self, mock_engine_cls: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.detect.return_value = [_make_detection()]

        spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
        images = [Path(f"/fake/img{i}.jpg") for i in range(20)]

        result = run_single_backend(
            model_spec=spec,
            backend_type=BackendType.PYTORCH,
            images=images,
            image_ids=None,
            gt_ann_path=None,
            task="detect",
            warmup_passes=0,
        )

        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms >= result.latency_p50_ms
        assert result.latency_p99_ms >= result.latency_p95_ms


class TestRunnerSkipsUnavailableBackend:
    """Runner must skip backends that fail to load."""

    @patch("yowo.benchmark._runner._check_backend_available")
    def test_skips_unavailable_backend(self, mock_check: MagicMock) -> None:
        mock_check.return_value = False

        spec = ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)
        images = [Path("/fake/img0.jpg")]

        results = run_all_backends(
            model_spec=spec,
            images=images,
            image_ids=None,
            gt_ann_path=None,
            task="detect",
            formats=["tensorrt"],
        )

        assert results == []


class TestRunnerRecordsModelSize:
    """Runner must record model file size."""

    @patch("yowo.benchmark._runner.DetectionEngine")
    def test_records_model_size(self, mock_engine_cls: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.detect.return_value = [_make_detection()]

        spec = ModelSpec(
            family=ModelFamily.YOLO11,
            size=ModelSize.NANO,
            weights_path=Path("/fake/weights.pt"),
        )
        images = [Path("/fake/img0.jpg")]

        with patch("yowo.benchmark._runner.Path.stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=6_500_000)
            result = run_single_backend(
                model_spec=spec,
                backend_type=BackendType.PYTORCH,
                images=images,
                image_ids=None,
                gt_ann_path=None,
                task="detect",
                warmup_passes=0,
            )

        assert result.model_size_mb == pytest.approx(6.5, rel=0.1)
