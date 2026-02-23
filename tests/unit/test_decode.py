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
