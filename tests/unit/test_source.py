"""Unit tests for frame_skip wiring and source metadata in frame sources."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from yowo.io._source import (
    ImageDirectorySource,
    ImageFileSource,
    RTSPStreamSource,
    VideoFileSource,
    WebcamSource,
    open_source,
)
from yowo.types import Frame

# ---------------------------------------------------------------------------
# RTSPStreamSource frame_skip
# ---------------------------------------------------------------------------


class TestRTSPFrameSkip:
    def test_init_accepts_frame_skip(self) -> None:
        src = RTSPStreamSource("rtsp://localhost/stream", frame_skip=2)
        assert src._frame_skip == 2

    def test_init_default_frame_skip_is_zero(self) -> None:
        src = RTSPStreamSource("rtsp://localhost/stream")
        assert src._frame_skip == 0

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_iter_skips_frames(self, mock_cap_cls: MagicMock) -> None:
        """With frame_skip=1, only every 2nd frame is yielded."""
        mock_cap = MagicMock()
        fake_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        # Return enough frames for max_frames=2 with frame_skip=1.
        # skip_mod=2: reads 0(yield), 1(skip), 2(yield), 3(skip) -> 2 yielded.
        mock_cap.read.return_value = (True, fake_bgr)
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        src = RTSPStreamSource("rtsp://localhost/stream", frame_skip=1, max_frames=2)
        src._open_cap = lambda: mock_cap  # type: ignore[assignment]

        frames = list(src)
        assert len(frames) == 2
        # 3 reads needed: 0(yield), 1(skip), 2(yield) -> stop at max_frames=2.
        assert mock_cap.read.call_count == 3
        assert all(isinstance(f, Frame) for f in frames)


# ---------------------------------------------------------------------------
# WebcamSource frame_skip
# ---------------------------------------------------------------------------


class TestWebcamFrameSkip:
    @patch("yowo.io._source.cv2.VideoCapture")
    def test_init_accepts_frame_skip(self, mock_cap_cls: MagicMock) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap
        src = WebcamSource(0, frame_skip=3)
        assert src._frame_skip == 3

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_init_default_frame_skip_is_zero(self, mock_cap_cls: MagicMock) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap
        src = WebcamSource(0)
        assert src._frame_skip == 0

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_iter_skips_frames(self, mock_cap_cls: MagicMock) -> None:
        """With frame_skip=2, only every 3rd frame is yielded."""
        mock_cap = MagicMock()
        fake_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        # Return 6 frames then stop.
        mock_cap.read.side_effect = [
            (True, fake_bgr.copy()),
            (True, fake_bgr.copy()),
            (True, fake_bgr.copy()),
            (True, fake_bgr.copy()),
            (True, fake_bgr.copy()),
            (True, fake_bgr.copy()),
            (False, None),
        ]
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        src = WebcamSource.__new__(WebcamSource)
        src._device_index = 0
        src._max_frames = None
        src._frame_skip = 2

        frames = list(src)
        # 6 reads, skip_mod=3 -> frames at index 0 and 3 yielded.
        assert len(frames) == 2
        assert all(isinstance(f, Frame) for f in frames)


# ---------------------------------------------------------------------------
# open_source() wiring
# ---------------------------------------------------------------------------


class TestOpenSourceFrameSkipWiring:
    def test_webcam_receives_frame_skip(self) -> None:
        with patch("yowo.io._source.WebcamSource") as mock_cls:
            mock_cls.return_value = MagicMock()
            open_source("0", frame_skip=5)
            mock_cls.assert_called_once_with(0, max_frames=None, frame_skip=5)

    def test_rtsp_receives_frame_skip(self) -> None:
        with patch("yowo.io._source.RTSPStreamSource") as mock_cls:
            mock_cls.return_value = MagicMock()
            open_source("rtsp://host/stream", frame_skip=3)
            mock_cls.assert_called_once_with(
                "rtsp://host/stream",
                reconnect_timeout_s=30.0,
                max_frames=None,
                frame_skip=3,
            )

    def test_video_file_receives_frame_skip(self, tmp_path: Path) -> None:
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("yowo.io._source.VideoFileSource") as mock_cls:
            mock_cls.return_value = MagicMock()
            open_source(str(video_file), frame_skip=4)
            mock_cls.assert_called_once_with(
                Path(str(video_file)),
                loop=False,
                frame_skip=4,
                max_frames=None,
            )


# ---------------------------------------------------------------------------
# F-1: open_source() accepts int webcam index
# ---------------------------------------------------------------------------


class TestOpenSourceIntIndex:
    """open_source(0) should dispatch to WebcamSource."""

    def test_int_dispatches_to_webcam(self) -> None:
        with patch("yowo.io._source.WebcamSource") as mock_cls:
            mock_cls.return_value = MagicMock()
            open_source(0)
            mock_cls.assert_called_once_with(0, max_frames=None, frame_skip=0)

    def test_int_passes_kwargs(self) -> None:
        with patch("yowo.io._source.WebcamSource") as mock_cls:
            mock_cls.return_value = MagicMock()
            open_source(1, frame_skip=3, max_frames=100)
            mock_cls.assert_called_once_with(1, max_frames=100, frame_skip=3)


# ---------------------------------------------------------------------------
# VideoFileSource metadata (_probe consolidation)
# ---------------------------------------------------------------------------


def _mock_video_cap(
    *,
    frame_count: float = 100,
    width: float = 1920,
    height: float = 1080,
    fps: float = 30.0,
) -> MagicMock:
    """Create a MagicMock cv2.VideoCapture with controlled property returns."""
    mock_cap = MagicMock()
    prop_map: dict[int, float] = {
        cv2.CAP_PROP_FRAME_COUNT: frame_count,
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
        cv2.CAP_PROP_FPS: fps,
    }
    mock_cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)
    return mock_cap


class TestVideoFileMetadata:
    @patch("yowo.io._source.cv2.VideoCapture")
    def test_resolution_from_probe(self, mock_cap_cls: MagicMock) -> None:
        mock_cap_cls.return_value = _mock_video_cap()
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.resolution == (1080, 1920)

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_fps_from_probe(self, mock_cap_cls: MagicMock) -> None:
        mock_cap_cls.return_value = _mock_video_cap(fps=29.97)
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.fps == 29.97

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_probe_called_once(self, mock_cap_cls: MagicMock) -> None:
        """Accessing resolution, fps, total_frames opens VideoCapture only once."""
        mock_cap_cls.return_value = _mock_video_cap()
        src = VideoFileSource(Path("/fake/video.mp4"))
        _ = src.resolution
        _ = src.fps
        _ = src.total_frames
        assert mock_cap_cls.call_count == 1

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_probe_zero_dimensions_returns_none(self, mock_cap_cls: MagicMock) -> None:
        mock_cap_cls.return_value = _mock_video_cap(width=0, height=0)
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.resolution is None

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_probe_zero_fps_returns_none(self, mock_cap_cls: MagicMock) -> None:
        mock_cap_cls.return_value = _mock_video_cap(fps=0.0)
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.fps is None

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_probe_negative_fps_returns_none(self, mock_cap_cls: MagicMock) -> None:
        mock_cap_cls.return_value = _mock_video_cap(fps=-1.0)
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.fps is None

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_total_frames_still_works(self, mock_cap_cls: MagicMock) -> None:
        mock_cap_cls.return_value = _mock_video_cap(frame_count=500)
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.total_frames == 500

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_total_frames_zero_returns_none(self, mock_cap_cls: MagicMock) -> None:
        mock_cap_cls.return_value = _mock_video_cap(frame_count=0)
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.total_frames is None

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_partial_zero_width_returns_none_resolution(self, mock_cap_cls: MagicMock) -> None:
        """Width=0 but height>0 still returns None."""
        mock_cap_cls.return_value = _mock_video_cap(width=0, height=1080)
        src = VideoFileSource(Path("/fake/video.mp4"))
        assert src.resolution is None


# ---------------------------------------------------------------------------
# RTSPStreamSource metadata
# ---------------------------------------------------------------------------


class TestRTSPMetadata:
    def test_resolution_is_none(self) -> None:
        src = RTSPStreamSource("rtsp://localhost/stream")
        assert src.resolution is None

    def test_fps_is_none(self) -> None:
        src = RTSPStreamSource("rtsp://localhost/stream")
        assert src.fps is None


# ---------------------------------------------------------------------------
# WebcamSource metadata
# ---------------------------------------------------------------------------


class TestWebcamMetadata:
    @patch("yowo.io._source.cv2.VideoCapture")
    def test_resolution_is_none(self, mock_cap_cls: MagicMock) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap
        src = WebcamSource(0)
        assert src.resolution is None

    @patch("yowo.io._source.cv2.VideoCapture")
    def test_fps_is_none(self, mock_cap_cls: MagicMock) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap
        src = WebcamSource(0)
        assert src.fps is None


# ---------------------------------------------------------------------------
# ImageFileSource metadata
# ---------------------------------------------------------------------------


class TestImageFileMetadata:
    @patch("yowo.io._source.cv2.imread")
    def test_resolution_from_image(self, mock_imread: MagicMock) -> None:
        mock_imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        src = ImageFileSource(Path("/fake/image.jpg"))
        assert src.resolution == (480, 640)

    @patch("yowo.io._source.cv2.imread")
    def test_resolution_bad_image_returns_none(self, mock_imread: MagicMock) -> None:
        mock_imread.return_value = None
        src = ImageFileSource(Path("/fake/bad.jpg"))
        assert src.resolution is None

    @patch("yowo.io._source.cv2.imread")
    def test_resolution_probed_once(self, mock_imread: MagicMock) -> None:
        """Accessing resolution twice only calls cv2.imread once."""
        mock_imread.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        src = ImageFileSource(Path("/fake/image.jpg"))
        _ = src.resolution
        _ = src.resolution
        assert mock_imread.call_count == 1

    def test_fps_is_none(self) -> None:
        src = ImageFileSource(Path("/fake/image.jpg"))
        assert src.fps is None


# ---------------------------------------------------------------------------
# ImageDirectorySource metadata
# ---------------------------------------------------------------------------


class TestImageDirectoryMetadata:
    def test_resolution_is_none(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").touch()
        src = ImageDirectorySource(tmp_path)
        assert src.resolution is None

    def test_fps_is_none(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").touch()
        src = ImageDirectorySource(tmp_path)
        assert src.fps is None
