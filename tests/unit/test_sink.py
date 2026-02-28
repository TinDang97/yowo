"""Unit tests for yowo.io._sink — JSON and annotated frame output writers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yowo.types import BoundingBox, Detection, Frame, ModelFamily, ModelSize, ModelSpec


def _make_frame(*, frame_index: int = 0) -> Frame:
    return Frame(
        pixels=np.zeros((480, 640, 3), dtype=np.uint8),
        source_id="test",
        frame_index=frame_index,
        timestamp_ms=0.0,
    )


def _make_detection(*, frame_index: int = 0) -> Detection:
    frame = _make_frame(frame_index=frame_index)
    box = BoundingBox(
        x1=10.0,
        y1=20.0,
        x2=100.0,
        y2=200.0,
        confidence=0.95,
        class_id=0,
        class_name="person",
    )
    return Detection(
        frame=frame,
        boxes=(box,),
        inference_time_ms=5.0,
        backend="pytorch",
        model_spec=ModelSpec(ModelFamily.YOLO26, ModelSize.NANO),
    )


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        from yowo.io._sink import write_json

        det = _make_detection()
        out = tmp_path / "out.json"
        write_json([det], out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_empty_detections_writes_empty_list(self, tmp_path: Path) -> None:
        from yowo.io._sink import write_json

        out = tmp_path / "empty.json"
        write_json([], out)
        assert json.loads(out.read_text()) == []

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        from yowo.io._sink import write_json

        out = tmp_path / "sub" / "dir" / "result.json"
        write_json([_make_detection()], out)
        assert out.exists()

    def test_tmp_file_cleaned_up_on_failure(self, tmp_path: Path) -> None:
        from yowo.io._sink import write_json

        det = MagicMock()
        det.to_dict.side_effect = RuntimeError("serialize fail")
        out = tmp_path / "fail.json"
        with pytest.raises(RuntimeError, match="serialize fail"):
            write_json([det], out)
        assert not out.with_suffix(".json.tmp").exists()

    def test_multiple_detections_roundtrip(self, tmp_path: Path) -> None:
        from yowo.io._sink import write_json

        dets = [_make_detection(frame_index=i) for i in range(3)]
        out = tmp_path / "multi.json"
        write_json(dets, out)
        data = json.loads(out.read_text())
        assert len(data) == 3


# ---------------------------------------------------------------------------
# write_annotated_frames
# ---------------------------------------------------------------------------


class TestWriteAnnotatedFrames:
    @patch("yowo.io._sink.cv2")
    def test_writes_jpeg_files(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        from yowo.io._sink import write_annotated_frames

        det = _make_detection(frame_index=0)
        write_annotated_frames([det], tmp_path)
        mock_cv2.imwrite.assert_called_once()
        # Verify filename pattern is zero-padded
        call_args = mock_cv2.imwrite.call_args[0]
        assert "000000.jpg" in call_args[0]

    @patch("yowo.io._sink.cv2")
    def test_empty_list_creates_no_files(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        from yowo.io._sink import write_annotated_frames

        write_annotated_frames([], tmp_path)
        mock_cv2.imwrite.assert_not_called()

    @patch("yowo.io._sink.cv2")
    def test_multiple_detections(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        from yowo.io._sink import write_annotated_frames

        dets = [_make_detection(frame_index=i) for i in range(3)]
        write_annotated_frames(dets, tmp_path)
        assert mock_cv2.imwrite.call_count == 3


# ---------------------------------------------------------------------------
# write_annotated_frame
# ---------------------------------------------------------------------------


class TestWriteAnnotatedFrame:
    @patch("yowo.io._sink.cv2")
    def test_writes_single_frame(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        from yowo.io._sink import write_annotated_frame

        det = _make_detection()
        out = tmp_path / "result.jpg"
        write_annotated_frame(det, out)
        mock_cv2.imwrite.assert_called_once()

    @patch("yowo.io._sink.cv2")
    def test_jpeg_output_has_quality_params(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        from yowo.io._sink import write_annotated_frame

        det = _make_detection()
        out = tmp_path / "result.jpg"
        write_annotated_frame(det, out)
        call_args = mock_cv2.imwrite.call_args[0]
        params = call_args[2]
        assert mock_cv2.IMWRITE_JPEG_QUALITY in params

    @patch("yowo.io._sink.cv2")
    def test_png_output_no_jpeg_quality_params(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        from yowo.io._sink import write_annotated_frame

        det = _make_detection()
        out = tmp_path / "result.png"
        write_annotated_frame(det, out)
        call_args = mock_cv2.imwrite.call_args[0]
        params = call_args[2]
        assert params == []

    @patch("yowo.io._sink.cv2")
    def test_creates_parent_directories(self, mock_cv2: MagicMock, tmp_path: Path) -> None:
        from yowo.io._sink import write_annotated_frame

        det = _make_detection()
        out = tmp_path / "nested" / "dir" / "result.jpg"
        write_annotated_frame(det, out)
        assert out.parent.exists()
