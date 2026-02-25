"""Tests for PreprocessBuffer and PostprocessBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from yowo.io._decode import PreprocessBuffer, preprocess, preprocess_into
from yowo.postprocess._nms import PostprocessBuffer, postprocess
from yowo.types import BackendType, Frame, ModelFamily, ModelSize, ModelSpec


def _make_frame(h: int = 480, w: int = 640) -> Frame:
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    pixels[100:200, 100:200] = [128, 64, 200]  # some non-zero content
    return Frame(pixels=pixels, source_id="test", frame_index=0, timestamp_ms=0.0)


def _make_model_spec() -> ModelSpec:
    return ModelSpec(family=ModelFamily.YOLO11, size=ModelSize.NANO)


def _make_frame_for_nms(h: int = 480, w: int = 640) -> Frame:
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=0, timestamp_ms=0.0)


class TestPreprocessBuffer:
    def test_capacity(self) -> None:
        buf = PreprocessBuffer(max_batch=4, target_size=(640, 640))
        assert buf.capacity == 4

    def test_memory_bytes(self) -> None:
        buf = PreprocessBuffer(max_batch=2, target_size=(640, 640))
        assert buf.memory_bytes == 2 * 640 * 640 * 3

    def test_staging_shape(self) -> None:
        buf = PreprocessBuffer(max_batch=3, target_size=(480, 640))
        staging = buf.get_staging(0)
        assert staging.shape == (480, 640, 3)
        assert staging.dtype == np.uint8

    def test_staging_initial_fill(self) -> None:
        buf = PreprocessBuffer(max_batch=1, target_size=(64, 64))
        staging = buf.get_staging(0)
        assert np.all(staging == 114)

    def test_staging_memory_reuse(self) -> None:
        buf = PreprocessBuffer(max_batch=2, target_size=(64, 64))
        first_call_id = id(buf.get_staging(0))
        second_call_id = id(buf.get_staging(0))
        assert first_call_id == second_call_id  # same object

    def test_target_size_property(self) -> None:
        buf = PreprocessBuffer(max_batch=1, target_size=(480, 640))
        assert buf.target_size == (480, 640)


class TestPreprocessInto:
    def test_output_shape_matches_preprocess(self) -> None:
        frame = _make_frame()
        buf = PreprocessBuffer(max_batch=1, target_size=(640, 640))
        result = preprocess_into([frame], (640, 640), buf)
        reference = preprocess([frame], (640, 640))
        assert result.data.shape == reference.data.shape

    def test_numerical_equivalence(self) -> None:
        frame = _make_frame(h=480, w=640)
        buf = PreprocessBuffer(max_batch=1, target_size=(640, 640))
        result = preprocess_into([frame], (640, 640), buf)
        reference = preprocess([frame], (640, 640))
        assert np.allclose(result.data, reference.data, atol=1e-5)

    def test_batch_numerical_equivalence(self) -> None:
        frames = [_make_frame(h=480, w=640), _make_frame(h=320, w=480)]
        buf = PreprocessBuffer(max_batch=2, target_size=(640, 640))
        result = preprocess_into(frames, (640, 640), buf)
        reference = preprocess(frames, (640, 640))
        assert np.allclose(result.data, reference.data, atol=1e-5)

    def test_metadata_equivalence(self) -> None:
        frame = _make_frame(h=480, w=640)
        buf = PreprocessBuffer(max_batch=1, target_size=(640, 640))
        result = preprocess_into([frame], (640, 640), buf)
        reference = preprocess([frame], (640, 640))
        assert result.original_shapes == reference.original_shapes
        assert result.input_shape == reference.input_shape
        assert result.scale_factors == reference.scale_factors
        assert result.pad_offsets == reference.pad_offsets

    def test_empty_frames_raises(self) -> None:
        buf = PreprocessBuffer(max_batch=1, target_size=(640, 640))
        with pytest.raises(ValueError, match="empty"):
            preprocess_into([], (640, 640), buf)

    def test_batch_exceeds_capacity_raises(self) -> None:
        frames = [_make_frame(), _make_frame(), _make_frame()]
        buf = PreprocessBuffer(max_batch=2, target_size=(640, 640))
        with pytest.raises(ValueError, match="capacity"):
            preprocess_into(frames, (640, 640), buf)

    def test_batch_smaller_than_capacity(self) -> None:
        frames = [_make_frame()]
        buf = PreprocessBuffer(max_batch=4, target_size=(640, 640))
        result = preprocess_into(frames, (640, 640), buf)
        assert result.data.shape == (1, 3, 640, 640)

    def test_tall_image_letterbox(self) -> None:
        """Tall image should have horizontal padding."""
        frame = _make_frame(h=800, w=400)
        buf = PreprocessBuffer(max_batch=1, target_size=(640, 640))
        result = preprocess_into([frame], (640, 640), buf)
        reference = preprocess([frame], (640, 640))
        assert np.allclose(result.data, reference.data, atol=1e-5)

    def test_target_size_mismatch_raises(self) -> None:
        """preprocess_into() rejects buffer with mismatched target_size."""
        frame = _make_frame()
        buf = PreprocessBuffer(max_batch=1, target_size=(480, 640))
        with pytest.raises(ValueError, match="target_size"):
            preprocess_into([frame], (640, 640), buf)


class TestPostprocessBuffer:
    def test_memory_bytes(self) -> None:
        buf = PostprocessBuffer(max_detections=8400)
        assert buf.memory_bytes == 8400 * 4 * 4

    def test_get_inverse_out_shape(self) -> None:
        buf = PostprocessBuffer(max_detections=8400)
        out = buf.get_inverse_out(100)
        assert out.shape == (100, 4)
        assert out.dtype == np.float32

    def test_get_inverse_out_is_view(self) -> None:
        buf = PostprocessBuffer(max_detections=8400)
        out1 = buf.get_inverse_out(100)
        out2 = buf.get_inverse_out(100)
        # Both views share the same backing memory
        assert out1.base is out2.base or np.shares_memory(out1, out2)

    def test_postprocess_with_scratch_matches_without(self) -> None:
        """Detection results must be identical with and without scratch buffer."""
        frame = _make_frame_for_nms()
        spec = _make_model_spec()
        tensor = preprocess([frame], (640, 640))

        # Zero output (no detections)
        raw = np.zeros((1, 84, 8400), dtype=np.float32)

        scratch = PostprocessBuffer()
        result_with = postprocess(
            raw,
            tensor,
            [frame],
            model_spec=spec,
            backend=BackendType.PYTORCH,
            scratch=scratch,
        )
        result_without = postprocess(
            raw,
            tensor,
            [frame],
            model_spec=spec,
            backend=BackendType.PYTORCH,
        )

        assert len(result_with) == len(result_without)
        for d_with, d_without in zip(result_with, result_without, strict=True):
            assert len(d_with.boxes) == len(d_without.boxes)

    def test_get_inverse_out_zero_rows(self) -> None:
        """get_inverse_out(0) returns an empty (0, 4) view without error."""
        buf = PostprocessBuffer(max_detections=8400)
        out = buf.get_inverse_out(0)
        assert out.shape == (0, 4)
        assert out.dtype == np.float32

    def test_get_inverse_out_exceeds_capacity_returns_fresh_array(self) -> None:
        """n > max_detections must return a fresh allocation, not a silently truncated view."""
        buf = PostprocessBuffer(max_detections=10)
        out = buf.get_inverse_out(20)
        # Must have the requested size, not be silently clipped to 10
        assert out.shape == (20, 4)
        assert out.dtype == np.float32


class TestPreprocessIntoStagingReset:
    def test_staging_reset_removes_residual_pixels(self) -> None:
        """Second preprocess_into() call must not contain residual pixels from a prior call."""
        buf = PreprocessBuffer(max_batch=1, target_size=(64, 64))

        # First call: a brightly coloured 60x60 frame fills most of the staging area.
        large_pixels = np.full((60, 60, 3), 200, dtype=np.uint8)
        large_frame = Frame(pixels=large_pixels, source_id="", frame_index=0, timestamp_ms=0.0)
        preprocess_into([large_frame], (64, 64), buf)

        # Second call: a tiny 1x1 black frame — only the centre pixel is written,
        # the rest of the staging area must be reset to the letterbox fill (114).
        tiny_pixels = np.zeros((1, 1, 3), dtype=np.uint8)
        tiny_frame = Frame(pixels=tiny_pixels, source_id="", frame_index=1, timestamp_ms=1.0)
        result = preprocess_into([tiny_frame], (64, 64), buf)

        # The output is float32 in [0, 1].  114/255 ≈ 0.447.
        # If residual pixels (200/255 ≈ 0.784) leaked, the mean would exceed 0.5.
        mean_val = float(result.data.mean())
        assert mean_val < 0.5, (
            f"Residual pixels from previous call contaminated staging buffer: mean={mean_val:.3f}"
        )
