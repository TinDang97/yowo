"""Unit tests for yowo.io._decode (preprocess / letterbox pipeline)."""

from __future__ import annotations

import numpy as np
import pytest

from yowo.io._decode import TensorMeta, preprocess
from yowo.types import Frame, PreprocessedTensor


def _make_frame(height: int, width: int, fill: int = 128) -> Frame:
    """Create a dummy BGR uint8 Frame filled with a constant value."""
    pixels = np.full((height, width, 3), fill, dtype=np.uint8)
    return Frame(pixels=pixels, source_id="test", frame_index=0, timestamp_ms=0.0)


class TestPreprocessOutputShape:
    def test_single_frame_shape(self) -> None:
        frame = _make_frame(480, 640)
        tensor = preprocess([frame], (640, 640))

        assert tensor.data.shape == (1, 3, 640, 640)
        assert tensor.data.dtype == np.float32

    def test_batch_of_three_frames_shape(self) -> None:
        frames = [_make_frame(480, 640) for _ in range(3)]
        tensor = preprocess(frames, (640, 640))

        assert tensor.data.shape == (3, 3, 640, 640)
        assert tensor.data.dtype == np.float32

    def test_output_values_in_unit_range(self) -> None:
        frame = _make_frame(100, 100, fill=200)
        tensor = preprocess([frame], (640, 640))

        assert float(tensor.data.min()) >= 0.0
        assert float(tensor.data.max()) <= 1.0

    def test_empty_frame_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            preprocess([], (640, 640))


class TestPreprocessLetterbox:
    def test_tall_image_pads_horizontally(self) -> None:
        # 200x100 (tall) into 640x640: scale = min(640/200, 640/100) = 3.2
        # new_h = 640, new_w = 320. pad_x = (640-320)//2 = 160
        frame = _make_frame(200, 100)
        tensor = preprocess([frame], (640, 640))

        scale_h, scale_w = tensor.scale_factors[0]
        pad_top, pad_left = tensor.pad_offsets[0]

        assert scale_h == pytest.approx(3.2, rel=1e-5)
        assert scale_w == pytest.approx(3.2, rel=1e-5)
        assert pad_top == 0
        assert pad_left == 160

    def test_wide_image_pads_vertically(self) -> None:
        # 100x200 (wide) into 640x640: scale = min(640/100, 640/200) = 3.2
        # new_h = 320, new_w = 640. pad_top = (640-320)//2 = 160
        frame = _make_frame(100, 200)
        tensor = preprocess([frame], (640, 640))

        scale_h, scale_w = tensor.scale_factors[0]
        pad_top, pad_left = tensor.pad_offsets[0]

        assert scale_h == pytest.approx(3.2, rel=1e-5)
        assert scale_w == pytest.approx(3.2, rel=1e-5)
        assert pad_top == 160
        assert pad_left == 0

    def test_square_image_no_padding(self) -> None:
        frame = _make_frame(640, 640)
        tensor = preprocess([frame], (640, 640))

        pad_top, pad_left = tensor.pad_offsets[0]
        assert pad_top == 0
        assert pad_left == 0

    def test_aspect_ratio_preserved_via_scale(self) -> None:
        # Verify scale is uniform (same h and w component).
        frame = _make_frame(480, 640)
        tensor = preprocess([frame], (640, 640))

        scale_h, scale_w = tensor.scale_factors[0]
        assert scale_h == scale_w


class TestPreprocessMetadata:
    def test_original_shapes_recorded(self) -> None:
        f1 = _make_frame(100, 200)
        f2 = _make_frame(300, 150)
        tensor = preprocess([f1, f2], (640, 640))

        assert tensor.original_shapes[0] == (100, 200)
        assert tensor.original_shapes[1] == (300, 150)

    def test_scale_factors_recorded_per_frame(self) -> None:
        f1 = _make_frame(640, 640)  # scale = 1.0
        f2 = _make_frame(320, 320)  # scale = 2.0
        tensor = preprocess([f1, f2], (640, 640))

        assert tensor.scale_factors[0] == pytest.approx((1.0, 1.0), rel=1e-5)
        assert tensor.scale_factors[1] == pytest.approx((2.0, 2.0), rel=1e-5)

    def test_pad_offsets_recorded_per_frame(self) -> None:
        # 200x100 tall: pad_left = 160, pad_top = 0
        f1 = _make_frame(200, 100)
        # 640x640 square: no padding
        f2 = _make_frame(640, 640)
        tensor = preprocess([f1, f2], (640, 640))

        assert tensor.pad_offsets[0] == (0, 160)
        assert tensor.pad_offsets[1] == (0, 0)

    def test_input_shape_stored(self) -> None:
        frame = _make_frame(480, 640)
        tensor = preprocess([frame], (416, 416))

        assert tensor.input_shape == (416, 416)

    def test_batch_size_property(self) -> None:
        frames = [_make_frame(100, 100) for _ in range(4)]
        tensor = preprocess(frames, (640, 640))

        assert tensor.batch_size == 4

    def test_returns_preprocessed_tensor_type(self) -> None:
        frame = _make_frame(100, 100)
        result = preprocess([frame], (640, 640))

        assert isinstance(result, PreprocessedTensor)

    def test_tensor_meta_from_decode(self) -> None:
        from yowo.io._decode import make_tensor_meta

        frame = _make_frame(200, 100)
        tensor = preprocess([frame], (640, 640))
        meta = make_tensor_meta(tensor)

        assert isinstance(meta, TensorMeta)
        assert meta.scale_factors == tensor.scale_factors
        assert meta.pad_offsets == tensor.pad_offsets
        assert meta.original_sizes == tensor.original_shapes


class TestPreprocessNormalization:
    def test_pixel_value_255_normalizes_to_1(self) -> None:
        frame = _make_frame(100, 100, fill=255)
        tensor = preprocess([frame], (100, 100))

        # The non-padded region (entire image since square) should be ~1.0.
        # Padding regions contain 114/255 ≈ 0.447.
        # Using a pixel from the center to avoid padding ambiguity.
        center_val = float(tensor.data[0, :, 50, 50].mean())
        assert center_val == pytest.approx(1.0, abs=1e-3)

    def test_pixel_value_0_normalizes_to_0(self) -> None:
        frame = _make_frame(100, 100, fill=0)
        tensor = preprocess([frame], (100, 100))

        center_val = float(tensor.data[0, :, 50, 50].mean())
        assert center_val == pytest.approx(0.0, abs=1e-3)


class TestPreprocessOptimizations:
    """Regression tests for the blobFromImages + contiguity-guard optimization."""

    def test_non_contiguous_input_does_not_crash(self) -> None:
        # Simulate a non-C-contiguous array (e.g. a transposed view).
        raw = np.zeros((3, 100, 100), dtype=np.uint8)
        non_contig = np.ascontiguousarray(raw.transpose(1, 2, 0))
        # Flip channel dim to make it non-contiguous in memory.
        non_contig_view = non_contig[:, :, ::-1]
        assert not non_contig_view.flags["C_CONTIGUOUS"]
        frame = Frame(
            pixels=non_contig_view,
            source_id="test",
            frame_index=0,
            timestamp_ms=0.0,
        )
        # Must not raise; ascontiguousarray guard handles non-contiguous input.
        result = preprocess([frame], (64, 64))
        assert result.data.shape == (1, 3, 64, 64)

    def test_output_numerical_accuracy(self) -> None:
        # Reference: manual BGR->RGB, transpose, normalize matches blobFromImages.
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        frame = Frame(pixels=pixels, source_id="test", frame_index=0, timestamp_ms=0.0)
        result = preprocess([frame], (120, 160))

        # Reference computation (original manual path).
        import cv2 as _cv2

        rgb = _cv2.cvtColor(pixels, _cv2.COLOR_BGR2RGB)
        ref = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0

        assert np.allclose(result.data[0], ref, atol=1e-5), (
            f"max delta: {np.abs(result.data[0] - ref).max()}"
        )

    def test_output_is_c_contiguous(self) -> None:
        frame = _make_frame(100, 100)
        result = preprocess([frame], (64, 64))
        assert result.data.flags["C_CONTIGUOUS"]

    def test_contiguous_input_fast_path_correct_output(self) -> None:
        # C-contiguous pixels must bypass ascontiguousarray and still produce
        # numerically correct output — verifies the if-branch isn't inverted.
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
        assert pixels.flags["C_CONTIGUOUS"]
        frame = Frame(pixels=pixels, source_id="test", frame_index=0, timestamp_ms=0.0)
        result = preprocess([frame], (60, 80))

        import cv2 as _cv2

        ref = (
            np.transpose(_cv2.cvtColor(pixels, _cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32)
            / 255.0
        )
        assert np.allclose(result.data[0], ref, atol=1e-5)

    def test_non_contiguous_input_correct_pixel_values(self) -> None:
        # Non-contiguous path must produce numerically identical output to
        # the manual reference (not just the right shape).
        rng = np.random.default_rng(13)
        pixels = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
        non_contig = pixels[:, :, ::-1]
        assert not non_contig.flags["C_CONTIGUOUS"]
        frame = Frame(pixels=non_contig, source_id="test", frame_index=0, timestamp_ms=0.0)
        result = preprocess([frame], (60, 80))

        import cv2 as _cv2

        contiguous = np.ascontiguousarray(non_contig)
        ref = (
            np.transpose(_cv2.cvtColor(contiguous, _cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(
                np.float32
            )
            / 255.0
        )
        assert np.allclose(result.data[0], ref, atol=1e-5)

    def test_letterbox_padded_region_fill_value(self) -> None:
        # Padded pixels (114, 114, 114) must normalize to 114/255 in the tensor.
        # 100x200 wide frame into 100x100: scale=0.5, new_h=50, pad_top=25.
        frame = _make_frame(100, 200, fill=0)
        result = preprocess([frame], (100, 100))

        expected_fill = pytest.approx(114.0 / 255.0, abs=2e-3)
        # Top row is entirely in the top padding band.
        assert float(result.data[0, 0, 0, 0]) == expected_fill


class TestPreprocessBufferPool:
    """Tests for the thread-safe PreprocessBufferPool."""

    def test_buffer_pool_acquire_release(self) -> None:
        """Acquire one buffer and release it without hanging."""
        from yowo.io._decode import PreprocessBufferPool

        pool = PreprocessBufferPool(pool_size=2, max_batch=1, target_size=(640, 640))
        buf = pool.acquire()
        pool.release(buf)

    def test_buffer_pool_distinct_buffers(self) -> None:
        """Acquiring two buffers from a size-2 pool yields distinct objects."""
        from yowo.io._decode import PreprocessBufferPool

        pool = PreprocessBufferPool(pool_size=2, max_batch=1, target_size=(640, 640))
        buf1 = pool.acquire()
        buf2 = pool.acquire()

        assert id(buf1) != id(buf2)

        pool.release(buf1)
        pool.release(buf2)

    def test_buffer_pool_exhaustion_blocks(self) -> None:
        """Pool of size 1 blocks a second acquire until the first is released."""
        import threading

        from yowo.io._decode import PreprocessBufferPool

        pool = PreprocessBufferPool(pool_size=1, max_batch=1, target_size=(640, 640))
        buf = pool.acquire()

        acquired = threading.Event()

        def _worker() -> None:
            b = pool.acquire()
            acquired.set()
            pool.release(b)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Worker should be blocked — event not set after a short wait.
        assert not acquired.wait(timeout=0.1)

        # Release from main thread unblocks the worker.
        pool.release(buf)
        assert acquired.wait(timeout=1.0)
        t.join(timeout=2.0)

    def test_buffer_pool_reuse(self) -> None:
        """Released buffer is reused on the next acquire (pool size 1)."""
        from yowo.io._decode import PreprocessBufferPool

        pool = PreprocessBufferPool(pool_size=1, max_batch=1, target_size=(640, 640))
        buf = pool.acquire()
        buf_id = id(buf)
        pool.release(buf)

        buf2 = pool.acquire()
        assert id(buf2) == buf_id
        pool.release(buf2)

    def test_buffer_pool_double_release_raises(self) -> None:
        """Releasing the same buffer twice raises ValueError."""
        from yowo.io._decode import PreprocessBufferPool

        pool = PreprocessBufferPool(pool_size=2, max_batch=1, target_size=(640, 640))
        buf = pool.acquire()
        pool.release(buf)

        with pytest.raises(ValueError, match="not acquired"):
            pool.release(buf)

    def test_buffer_pool_alien_buffer_raises(self) -> None:
        """Releasing a buffer not from this pool raises ValueError."""
        from yowo.io._decode import PreprocessBufferPool

        pool = PreprocessBufferPool(pool_size=1, max_batch=1, target_size=(640, 640))
        alien = PreprocessBufferPool(pool_size=1, max_batch=1, target_size=(640, 640)).acquire()

        with pytest.raises(ValueError, match="not acquired"):
            pool.release(alien)


# ---------------------------------------------------------------------------
# PreprocessBuffer — direct class tests
# ---------------------------------------------------------------------------


class TestPreprocessBuffer:
    """Direct unit tests for PreprocessBuffer properties and methods."""

    def test_capacity_property(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(4, (640, 640))
        assert buf.capacity == 4

    def test_target_size_property(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(2, (320, 320))
        assert buf.target_size == (320, 320)

    def test_memory_bytes_property(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(2, (640, 640))
        assert buf.memory_bytes == 2 * 640 * 640 * 3

    def test_get_staging_returns_ndarray(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(1, (640, 640))
        arr = buf.get_staging(0)
        assert arr.shape == (640, 640, 3)
        assert arr.dtype == np.uint8

    def test_needs_reset_first_use_returns_true(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(1, (640, 640))
        assert buf.needs_reset(0, 480, 640, 80, 0) is True

    def test_needs_reset_same_dims_returns_false(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(1, (640, 640))
        buf.needs_reset(0, 480, 640, 80, 0)
        assert buf.needs_reset(0, 480, 640, 80, 0) is False

    def test_needs_reset_different_dims_returns_true(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(1, (640, 640))
        buf.needs_reset(0, 480, 640, 80, 0)
        assert buf.needs_reset(0, 320, 320, 160, 160) is True

    def test_staging_filled_with_letterbox_value(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(1, (100, 100))
        arr = buf.get_staging(0)
        # Letterbox fill value is 114
        assert arr[0, 0, 0] == 114

    def test_multiple_slots_independent(self) -> None:
        from yowo.io._decode import PreprocessBuffer

        buf = PreprocessBuffer(3, (640, 640))
        assert buf.needs_reset(0, 480, 640, 80, 0) is True
        assert buf.needs_reset(1, 480, 640, 80, 0) is True
        # Slot 0 cached, slot 1 cached
        assert buf.needs_reset(0, 480, 640, 80, 0) is False
        assert buf.needs_reset(1, 480, 640, 80, 0) is False
        # Slot 2 never used
        assert buf.needs_reset(2, 480, 640, 80, 0) is True


# ---------------------------------------------------------------------------
# Auto letterbox tests
# ---------------------------------------------------------------------------


class TestAlignToStride:
    """Tests for _align_to_stride helper."""

    def test_exact_multiple_unchanged(self) -> None:
        from yowo.io._decode import _align_to_stride

        assert _align_to_stride(640) == 640
        assert _align_to_stride(384) == 384

    def test_rounds_up(self) -> None:
        from yowo.io._decode import _align_to_stride

        assert _align_to_stride(385) == 416
        assert _align_to_stride(353) == 384
        assert _align_to_stride(361) == 384

    def test_custom_stride(self) -> None:
        from yowo.io._decode import _align_to_stride

        assert _align_to_stride(97, stride=16) == 112
        assert _align_to_stride(64, stride=16) == 64


class TestAutoLetterbox:
    """Tests for preprocess() with auto_letterbox=True."""

    def test_16_9_input_produces_non_square_tensor(self) -> None:
        """1280x720 (16:9) should produce 384x640, not 640x640."""
        frame = _make_frame(720, 1280)
        tensor = preprocess([frame], (640, 640), auto_letterbox=True)

        # scale = min(640/720, 640/1280) = 0.5
        # new_h = 360, new_w = 640
        # aligned_h = ceil_stride(360, 32) = 384, aligned_w = 640
        assert tensor.data.shape == (1, 3, 384, 640)

    def test_4_3_input_produces_non_square_tensor(self) -> None:
        """640x480 (4:3) should produce 480x640."""
        frame = _make_frame(480, 640)
        tensor = preprocess([frame], (640, 640), auto_letterbox=True)

        # scale = min(640/480, 640/640) = 1.0
        # new_h = 480, new_w = 640
        # aligned_h = 480, aligned_w = 640
        assert tensor.data.shape == (1, 3, 480, 640)

    def test_square_input_stays_square(self) -> None:
        """640x640 input should remain 640x640."""
        frame = _make_frame(640, 640)
        tensor = preprocess([frame], (640, 640), auto_letterbox=True)

        assert tensor.data.shape == (1, 3, 640, 640)

    def test_dimensions_divisible_by_32(self) -> None:
        """All output dimensions must be divisible by 32."""
        test_cases = [
            (720, 1280),  # 16:9
            (1080, 1920),  # 16:9
            (480, 640),  # 4:3
            (480, 854),  # 16:9 ish
            (100, 300),  # 3:1
            (300, 100),  # 1:3
        ]
        for h, w in test_cases:
            frame = _make_frame(h, w)
            tensor = preprocess([frame], (640, 640), auto_letterbox=True)
            _, _, out_h, out_w = tensor.data.shape
            assert out_h % 32 == 0, f"Height {out_h} not divisible by 32 for input ({h}, {w})"
            assert out_w % 32 == 0, f"Width {out_w} not divisible by 32 for input ({h}, {w})"

    def test_scale_factor_uniform(self) -> None:
        """Scale must be uniform (same for h and w)."""
        frame = _make_frame(720, 1280)
        tensor = preprocess([frame], (640, 640), auto_letterbox=True)
        scale_h, scale_w = tensor.scale_factors[0]
        assert scale_h == scale_w

    def test_input_shape_records_model_target(self) -> None:
        """input_shape should still record the model's declared size."""
        frame = _make_frame(720, 1280)
        tensor = preprocess([frame], (640, 640), auto_letterbox=True)
        assert tensor.input_shape == (640, 640)

    def test_default_false_unchanged_behavior(self) -> None:
        """auto_letterbox=False (default) produces same output as before."""
        frame = _make_frame(720, 1280)
        tensor = preprocess([frame], (640, 640), auto_letterbox=False)
        assert tensor.data.shape == (1, 3, 640, 640)

    def test_pixel_values_in_unit_range(self) -> None:
        """Output values must be in [0, 1]."""
        frame = _make_frame(720, 1280, fill=200)
        tensor = preprocess([frame], (640, 640), auto_letterbox=True)
        assert float(tensor.data.min()) >= 0.0
        assert float(tensor.data.max()) <= 1.0

    def test_minimum_size_guard(self) -> None:
        """Very small inputs should produce at least 32x32 output."""
        frame = _make_frame(10, 10)
        tensor = preprocess([frame], (640, 640), auto_letterbox=True)
        _, _, out_h, out_w = tensor.data.shape
        assert out_h >= 32
        assert out_w >= 32

    def test_batch_consistency(self) -> None:
        """All frames in a batch should produce the same tensor shape."""
        f1 = _make_frame(720, 1280)
        f2 = _make_frame(720, 1280)
        tensor = preprocess([f1, f2], (640, 640), auto_letterbox=True)
        assert tensor.data.shape[0] == 2
        assert tensor.data.shape[2] == 384
        assert tensor.data.shape[3] == 640
