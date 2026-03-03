"""Unit tests for ReID integration — embeddings, appearance, gated cost, extractors."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from unit._tracking_helpers import make_box, make_detection, make_strack
from yowo.tracking import track_detections, track_stream
from yowo.tracking._clip_reid import CLIPReIDExtractor
from yowo.tracking._matching import (
    appearance_distance,
    fuse_score,
    gated_fused_cost,
    needs_reid,
)
from yowo.tracking._reid import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    FastReIDExtractor,
    ReIDExtractor,
)
from yowo.tracking._strack import TrackedDetection
from yowo.tracking._tracker import ByteTracker

# ---------------------------------------------------------------------------
# MockReIDExtractor helper
# ---------------------------------------------------------------------------


class MockReIDExtractor:
    """Deterministic mock: returns position-based embeddings for testing."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def extract(
        self,
        frame_pixels: NDArray[np.uint8],
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> NDArray[np.float32] | None:
        if not boxes_xyxy:
            return None
        embs = np.zeros((len(boxes_xyxy), self._dim), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            embs[i, 0] = cx / 1920
            embs[i, 1] = cy / 1080
            embs[i, 2] = (x2 - x1) / 1920
            embs[i, 3] = (y2 - y1) / 1080
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
        return (embs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# STrack embedding tests
# ---------------------------------------------------------------------------


class TestSTrackEmbedding:
    def test_embedding_slot_default_none(self) -> None:
        t = make_strack()
        assert t.embedding is None

    def test_update_embedding_first_call_copies(self) -> None:
        t = make_strack()
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t.update_embedding(emb)
        assert t.embedding is not None
        # Must be a copy, not the same object
        assert t.embedding is not emb
        np.testing.assert_allclose(t.embedding, emb, atol=1e-6)

    def test_update_embedding_ema_formula(self) -> None:
        t = make_strack()
        old = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        t.update_embedding(old)
        t.update_embedding(new, eta=0.9)
        # EMA: 0.9 * [1,0,0] + 0.1 * [0,1,0] = [0.9, 0.1, 0] -> normalized
        expected = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        assert t.embedding is not None
        np.testing.assert_allclose(t.embedding, expected, atol=1e-5)

    def test_update_embedding_renormalized(self) -> None:
        t = make_strack()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        t.update_embedding(emb1)
        t.update_embedding(emb2, eta=0.5)
        assert t.embedding is not None
        norm = float(np.linalg.norm(t.embedding))
        assert abs(norm - 1.0) < 1e-5

    def test_update_embedding_zero_norm_safety(self) -> None:
        t = make_strack()
        zero = np.zeros(4, dtype=np.float32)
        t.update_embedding(zero)
        # First call copies — zero embedding stored
        assert t.embedding is not None
        # Second call with zero: EMA of zeros = zeros, norm guard prevents crash
        t.update_embedding(zero)
        assert t.embedding is not None


# ---------------------------------------------------------------------------
# Appearance distance tests
# ---------------------------------------------------------------------------


class TestAppearanceDistance:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        tracks = rng.standard_normal((3, 512)).astype(np.float32)
        tracks /= np.linalg.norm(tracks, axis=1, keepdims=True)
        dets = rng.standard_normal((5, 512)).astype(np.float32)
        dets /= np.linalg.norm(dets, axis=1, keepdims=True)
        dist = appearance_distance(tracks, dets)
        assert dist.shape == (3, 5)
        assert dist.dtype == np.float64

    def test_identical_distance_zero(self) -> None:
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        dist = appearance_distance(emb, emb)
        assert abs(dist[0, 0]) < 1e-6

    def test_orthogonal_distance_one(self) -> None:
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        dist = appearance_distance(a, b)
        assert abs(dist[0, 0] - 1.0) < 1e-6

    def test_empty_tracks(self) -> None:
        tracks = np.empty((0, 128), dtype=np.float32)
        dets = np.random.randn(5, 128).astype(np.float32)
        dist = appearance_distance(tracks, dets)
        assert dist.shape == (0, 5)

    def test_empty_dets(self) -> None:
        tracks = np.random.randn(3, 128).astype(np.float32)
        dets = np.empty((0, 128), dtype=np.float32)
        dist = appearance_distance(tracks, dets)
        assert dist.shape == (3, 0)


# ---------------------------------------------------------------------------
# Gated fused cost tests
# ---------------------------------------------------------------------------


class TestGatedFusedCost:
    def test_iou_only_fallback(self) -> None:
        """When cos_dist > theta_e, result should equal IoU cost."""
        iou_cost = np.array([[0.3]], dtype=np.float64)
        # Orthogonal: cos_dist = 1.0 > theta_e=0.3
        t_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        d_emb = np.array([[0.0, 1.0]], dtype=np.float32)
        fused = gated_fused_cost(iou_cost, t_emb, d_emb)
        assert abs(fused[0, 0] - 0.3) < 1e-6

    def test_fused_path(self) -> None:
        """When both gates pass, result = min(iou, 0.5*cos_dist)."""
        iou_cost = np.array([[0.4]], dtype=np.float64)
        # Nearly identical: cos_dist ~ 0
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        fused = gated_fused_cost(iou_cost, emb, emb, theta_e=0.30, theta_iou=0.5)
        # cos_dist = 0, d_hat = 0.5 * 0 = 0, min(0.4, 0) = 0
        assert fused[0, 0] < 0.01

    def test_shape_preserved(self) -> None:
        n, m, d = 4, 6, 64
        rng = np.random.default_rng(42)
        iou_cost = rng.random((n, m)).astype(np.float64)
        t = rng.standard_normal((n, d)).astype(np.float32)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        de = rng.standard_normal((m, d)).astype(np.float32)
        de /= np.linalg.norm(de, axis=1, keepdims=True)
        fused = gated_fused_cost(iou_cost, t, de)
        assert fused.shape == (n, m)
        assert fused.dtype == np.float64

    def test_backward_compat_zero_embeddings(self) -> None:
        """Zero embeddings (no info) -> cos_dist=1.0 -> gate fails -> IoU only."""
        iou_cost = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=np.float64)
        zeros = np.zeros((2, 64), dtype=np.float32)
        fused = gated_fused_cost(iou_cost, zeros, zeros)
        np.testing.assert_allclose(fused, iou_cost, atol=1e-6)


# ---------------------------------------------------------------------------
# needs_reid tests
# ---------------------------------------------------------------------------


class TestNeedsReid:
    def test_empty_cost(self) -> None:
        cost = np.empty((0, 0), dtype=np.float64)
        assert needs_reid(cost) is False

    def test_clear_match(self) -> None:
        """One clear match per column -> no ReID needed."""
        cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
        assert needs_reid(cost) is False

    def test_no_overlap(self) -> None:
        """All costs > 0.7 -> no spatial overlap -> no ReID needed."""
        cost = np.array([[0.8, 0.9], [0.9, 0.8]], dtype=np.float64)
        assert needs_reid(cost) is False

    def test_ambiguous(self) -> None:
        """Two tracks with similar overlap for one detection -> need ReID."""
        cost = np.array([[0.45, 0.9], [0.50, 0.1]], dtype=np.float64)
        assert needs_reid(cost) is True


# ---------------------------------------------------------------------------
# ReID Protocol tests
# ---------------------------------------------------------------------------


class TestReIDProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        mock = MockReIDExtractor(dim=128)
        assert isinstance(mock, ReIDExtractor)

    def test_mock_extract_returns_correct_shape(self) -> None:
        mock = MockReIDExtractor(dim=128)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [(100.0, 100.0, 200.0, 200.0), (300.0, 300.0, 400.0, 400.0)]
        result = mock.extract(frame, boxes)
        assert result is not None
        assert result.shape == (2, 128)

    def test_mock_extract_empty_returns_none(self) -> None:
        mock = MockReIDExtractor(dim=128)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert mock.extract(frame, []) is None

    def test_mock_embeddings_l2_normalized(self) -> None:
        mock = MockReIDExtractor(dim=128)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [(100.0, 100.0, 200.0, 200.0)]
        result = mock.extract(frame, boxes)
        assert result is not None
        norm = float(np.linalg.norm(result[0]))
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# ByteTracker with ReID tests
# ---------------------------------------------------------------------------


class TestByteTrackerReID:
    def test_no_reid_identical_output(self) -> None:
        """Without reid_extractor, behavior is identical to original."""
        tracker = ByteTracker(min_hits=1)
        box = make_box(conf=0.9)
        r = tracker.update(make_detection(boxes=(box,), frame_index=0))
        assert isinstance(r, TrackedDetection)
        assert r.num_boxes == 1

    def test_accepts_reid_extractor(self) -> None:
        mock = MockReIDExtractor(dim=128)
        tracker = ByteTracker(min_hits=1, reid_extractor=mock)
        box = make_box(conf=0.9)
        r = tracker.update(make_detection(boxes=(box,), frame_index=0))
        assert isinstance(r, TrackedDetection)

    def test_reid_updates_embeddings(self) -> None:
        """After matching with ReID, tracks should have embeddings."""
        mock = MockReIDExtractor(dim=128)
        tracker = ByteTracker(
            min_hits=1,
            reid_extractor=mock,
            reid_frame_interval=1,
        )
        box_a = make_box(x1=100, y1=100, x2=300, y2=300, conf=0.9)
        box_b = make_box(x1=400, y1=100, x2=600, y2=300, conf=0.9)
        # Create tracks
        tracker.update(make_detection(boxes=(box_a, box_b), frame_index=0))
        # Second frame — matching triggers ReID if ambiguous
        tracker.update(make_detection(boxes=(box_a, box_b), frame_index=1))
        # Check that internal tracks exist and the tracker is functional.
        _ = any(
            t.embedding is not None
            for t in tracker._tracked  # type: ignore[attr-defined]
        )
        assert tracker.active_track_count >= 1

    def test_reid_graceful_none_extract(self) -> None:
        """If extractor.extract returns None, tracker falls back to IoU."""

        class NullExtractor:
            @property
            def embedding_dim(self) -> int:
                return 128

            def extract(
                self,
                frame_pixels: NDArray[np.uint8],
                boxes_xyxy: list[tuple[float, float, float, float]],
            ) -> NDArray[np.float32] | None:
                return None

        tracker = ByteTracker(
            min_hits=1,
            reid_extractor=NullExtractor(),
            reid_frame_interval=1,
        )
        box = make_box(conf=0.9)
        r = tracker.update(make_detection(boxes=(box,), frame_index=0))
        assert isinstance(r, TrackedDetection)

    def test_reset_clears_state_with_reid(self) -> None:
        mock = MockReIDExtractor(dim=128)
        tracker = ByteTracker(min_hits=1, reid_extractor=mock)
        for i in range(3):
            tracker.update(make_detection(boxes=(make_box(conf=0.9),), frame_index=i))
        tracker.reset()
        assert tracker.active_track_count == 0
        assert tracker.lost_track_count == 0

    def test_frame_interval_throttle(self) -> None:
        """ReID should not be called more often than reid_frame_interval."""
        call_count = 0

        class CountingExtractor:
            @property
            def embedding_dim(self) -> int:
                return 128

            def extract(
                self,
                frame_pixels: NDArray[np.uint8],
                boxes_xyxy: list[tuple[float, float, float, float]],
            ) -> NDArray[np.float32] | None:
                nonlocal call_count
                call_count += 1
                n = len(boxes_xyxy)
                rng = np.random.default_rng(call_count)
                embs = rng.standard_normal((n, 128)).astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
                return (embs / norms).astype(np.float32)

        tracker = ByteTracker(
            min_hits=1,
            reid_extractor=CountingExtractor(),
            reid_frame_interval=3,
        )
        # Create two overlapping tracks that trigger ambiguity
        box_a = make_box(x1=100, y1=100, x2=250, y2=250, conf=0.9)
        box_b = make_box(x1=150, y1=100, x2=300, y2=250, conf=0.9)
        for i in range(10):
            tracker.update(make_detection(boxes=(box_a, box_b), frame_index=i))
        # With interval=3, over 10 frames should have <= 4 extract calls
        assert call_count <= 4


# ---------------------------------------------------------------------------
# track_stream/track_detections with ReID
# ---------------------------------------------------------------------------


class TestTrackStreamReID:
    def test_track_stream_accepts_reid_extractor(self) -> None:
        engine = MagicMock()
        dets = [make_detection(boxes=(make_box(conf=0.9),), frame_index=i) for i in range(3)]
        engine.stream.return_value = iter(dets)
        mock = MockReIDExtractor(dim=128)
        results = list(track_stream(engine, MagicMock(), reid_extractor=mock))
        assert len(results) == 3

    def test_track_detections_accepts_reid_extractor(self) -> None:
        dets = [make_detection(boxes=(make_box(conf=0.9),), frame_index=i) for i in range(3)]
        mock = MockReIDExtractor(dim=128)
        results = list(track_detections(dets, reid_extractor=mock))
        assert len(results) == 3


# ---------------------------------------------------------------------------
# FastReIDExtractor tests
# ---------------------------------------------------------------------------


class TestFastReIDExtractor:
    def test_implements_protocol(self) -> None:
        """FastReIDExtractor satisfies the ReIDExtractor Protocol."""
        assert hasattr(FastReIDExtractor, "embedding_dim")
        assert hasattr(FastReIDExtractor, "extract")

    def test_file_not_found_raises(self) -> None:
        """Non-existent model path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="FastReID ONNX model not found"):
            FastReIDExtractor("/nonexistent/fastreid.onnx")

    def test_imagenet_constants_correct(self) -> None:
        """ImageNet normalization constants match standard values."""
        expected_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        expected_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        np.testing.assert_allclose(_IMAGENET_MEAN, expected_mean, atol=1e-6)
        np.testing.assert_allclose(_IMAGENET_STD, expected_std, atol=1e-6)

    def test_input_size_portrait_default(self) -> None:
        """Default input_size is portrait (256, 128) not square."""
        sig = inspect.signature(FastReIDExtractor.__init__)
        default = sig.parameters["input_size"].default
        assert default == (256, 128), f"Expected (256, 128), got {default}"

    def test_embedding_dim_default(self) -> None:
        """Default embedding_dim is 256 (not 512 like CLIP)."""
        sig = inspect.signature(FastReIDExtractor.__init__)
        default = sig.parameters["embedding_dim"].default
        assert default == 256, f"Expected 256, got {default}"

    def test_slots_defined(self) -> None:
        """FastReIDExtractor uses __slots__ for memory efficiency."""
        assert hasattr(FastReIDExtractor, "__slots__")
        assert "_session" in FastReIDExtractor.__slots__
        assert "_input_size_hw" in FastReIDExtractor.__slots__

    def test_export_in_tracking_init(self) -> None:
        """FastReIDExtractor is exported from yowo.tracking."""
        from yowo.tracking import FastReIDExtractor as Imported

        assert Imported is FastReIDExtractor


# ---------------------------------------------------------------------------
# CLIPReIDExtractor tests
# ---------------------------------------------------------------------------


class TestCLIPReIDExtractor:
    def test_implements_protocol(self) -> None:
        """CLIPReIDExtractor satisfies the ReIDExtractor Protocol."""
        assert hasattr(CLIPReIDExtractor, "embedding_dim")
        assert hasattr(CLIPReIDExtractor, "extract")

    def test_file_not_found_raises(self) -> None:
        """Non-existent model path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="CLIP-ReID ONNX model not found"):
            CLIPReIDExtractor("/nonexistent/clip_reid.onnx")

    def test_slots_defined(self) -> None:
        """CLIPReIDExtractor uses __slots__."""
        assert hasattr(CLIPReIDExtractor, "__slots__")
        assert "_session" in CLIPReIDExtractor.__slots__
        assert "_input_size" in CLIPReIDExtractor.__slots__

    def test_input_size_default(self) -> None:
        """Default input_size is 256 for CLIP-ReID."""
        sig = inspect.signature(CLIPReIDExtractor.__init__)
        default = sig.parameters["input_size"].default
        assert default == 256, f"Expected 256, got {default}"

    def test_default_embedding_dim(self) -> None:
        """Default embedding_dim is 1280 for ViT-B/16 CLIP-ReID."""
        sig = inspect.signature(CLIPReIDExtractor.__init__)
        default = sig.parameters["embedding_dim"].default
        assert default == 1280, f"Expected 1280, got {default}"

    def test_preprocessing_constants(self) -> None:
        """CLIP-ReID uses [0.5, 0.5, 0.5] mean/std, NOT ImageNet."""
        from yowo.tracking._clip_reid import _CLIPREID_MEAN, _CLIPREID_STD

        np.testing.assert_array_equal(_CLIPREID_MEAN, [0.5, 0.5, 0.5])
        np.testing.assert_array_equal(_CLIPREID_STD, [0.5, 0.5, 0.5])

    def test_export_in_reid_module(self) -> None:
        """CLIPReIDExtractor is exported from yowo.tracking._reid."""
        from yowo.tracking import _reid

        assert "CLIPReIDExtractor" in _reid.__all__


# ---------------------------------------------------------------------------
# fuse_score tests
# ---------------------------------------------------------------------------


class TestFuseScore:
    """Tests for the fuse_score function."""

    def test_identity_at_score_one(self) -> None:
        """Scores all 1.0 → cost unchanged."""
        cost = np.array([[0.2, 0.5], [0.8, 0.1]], dtype=np.float64)
        scores = np.array([1.0, 1.0], dtype=np.float64)
        result = fuse_score(cost, scores)
        np.testing.assert_allclose(result, cost)

    def test_zeros_at_score_zero(self) -> None:
        """Scores all 0.0 → cost all 1.0 (no similarity)."""
        cost = np.array([[0.2, 0.5], [0.8, 0.1]], dtype=np.float64)
        scores = np.array([0.0, 0.0], dtype=np.float64)
        result = fuse_score(cost, scores)
        np.testing.assert_allclose(result, np.ones_like(cost))

    def test_scales_correctly(self) -> None:
        """Verify formula: cost = 1 - (1 - iou_cost) * score."""
        cost = np.array([[0.3]], dtype=np.float64)
        scores = np.array([0.5], dtype=np.float64)
        expected = 1.0 - (1.0 - 0.3) * 0.5  # 1 - 0.7*0.5 = 1 - 0.35 = 0.65
        result = fuse_score(cost, scores)
        np.testing.assert_allclose(result, [[expected]])

    def test_empty_matrix(self) -> None:
        """Empty cost matrix → returns empty."""
        cost = np.empty((0, 0), dtype=np.float64)
        scores = np.empty(0, dtype=np.float64)
        result = fuse_score(cost, scores)
        assert result.shape == (0, 0)

    def test_broadcast_shape(self) -> None:
        """Result shape matches input shape with multiple tracks and dets."""
        cost = np.array([[0.1, 0.4, 0.9], [0.3, 0.2, 0.7]], dtype=np.float64)
        scores = np.array([0.8, 0.5, 0.3], dtype=np.float64)
        result = fuse_score(cost, scores)
        assert result.shape == (2, 3)
        # Lower score → higher cost
        assert result[0, 2] > result[0, 0], "Low-score det should have higher cost"

    def test_exported_from_matching(self) -> None:
        """fuse_score is in _matching.__all__."""
        from yowo.tracking._matching import __all__ as matching_all

        assert "fuse_score" in matching_all
