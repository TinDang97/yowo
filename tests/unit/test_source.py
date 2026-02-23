"""Unit tests for frame_skip wiring in RTSP and Webcam sources."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from yowo.io._source import RTSPStreamSource, WebcamSource, open_source
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
