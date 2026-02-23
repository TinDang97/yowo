"""Input source abstraction.

FrameSource Protocol + open_source() factory.
Dispatch by file extension and scheme.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from yowo.errors import SourceError, SourceTimeoutError
from yowo.types import Frame

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".ts"})
_RTSP_SCHEMES = ("rtsp://", "rtsps://")


@runtime_checkable
class FrameSource(Protocol):
    """Protocol for any input source that yields Frame objects."""

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None:
        """Release underlying resource. Idempotent."""
        ...

    @property
    def is_live(self) -> bool:
        """True for live sources (RTSP, webcam); disables progress bar."""
        ...

    @property
    def total_frames(self) -> int | None:
        """Total frame count, or None for live / unknown-length sources."""
        ...


# ---------------------------------------------------------------------------
# ImageFileSource
# ---------------------------------------------------------------------------


class ImageFileSource:
    """Reads a single image file and yields exactly one Frame."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._yielded = False

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int | None:
        return 1

    def __iter__(self) -> Iterator[Frame]:
        pixels = cv2.imread(str(self._path))
        if pixels is None:
            raise SourceError(f"cv2.imread failed for: {self._path}")
        yield Frame(
            pixels=pixels.astype(np.uint8),
            source_id=str(self._path),
            frame_index=0,
            timestamp_ms=0.0,
        )

    def close(self) -> None:
        pass

    def __enter__(self) -> ImageFileSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# ImageDirectorySource
# ---------------------------------------------------------------------------


class ImageDirectorySource:
    """Yields frames from all supported image files in a directory, sorted."""

    def __init__(self, directory: Path) -> None:
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not files:
            raise SourceError(f"No supported image files found in directory: {directory}")
        self._files = files

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int | None:
        return len(self._files)

    def __iter__(self) -> Iterator[Frame]:
        for idx, path in enumerate(self._files):
            pixels = cv2.imread(str(path))
            if pixels is None:
                raise SourceError(f"cv2.imread failed for: {path}")
            yield Frame(
                pixels=pixels.astype(np.uint8),
                source_id=str(path),
                frame_index=idx,
                timestamp_ms=0.0,
            )

    def close(self) -> None:
        pass

    def __enter__(self) -> ImageDirectorySource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# VideoFileSource
# ---------------------------------------------------------------------------


class VideoFileSource:
    """Reads a video file via OpenCV, with frame-skip and max-frame support."""

    def __init__(
        self,
        path: Path,
        *,
        loop: bool = False,
        frame_skip: int = 0,
        max_frames: int | None = None,
    ) -> None:
        self._path = path
        self._loop = loop
        self._frame_skip = frame_skip
        self._max_frames = max_frames
        self._cap: cv2.VideoCapture | None = None
        self._total: int | None = None

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int | None:
        if self._total is None:
            cap = cv2.VideoCapture(str(self._path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self._total = count if count > 0 else None
        return self._total

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise SourceError(f"Cannot open video file: {self._path}")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        cap = self._open()
        yielded = 0
        read_index = 0
        try:
            while True:
                if self._max_frames is not None and yielded >= self._max_frames:
                    break

                ok, bgr = cap.read()
                if not ok:
                    if self._loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        read_index = 0
                        ok, bgr = cap.read()
                        if not ok:
                            break
                    else:
                        break

                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                if read_index % (self._frame_skip + 1) == 0:
                    yield Frame(
                        pixels=bgr.astype(np.uint8),
                        source_id=str(self._path),
                        frame_index=yielded,
                        timestamp_ms=ts_ms,
                    )
                    yielded += 1

                read_index += 1
        finally:
            cap.release()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoFileSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# RTSPStreamSource
# ---------------------------------------------------------------------------


class RTSPStreamSource:
    """Reads an RTSP stream with exponential-backoff reconnect."""

    def __init__(
        self,
        url: str,
        *,
        reconnect_timeout_s: float = 30.0,
        max_frames: int | None = None,
    ) -> None:
        self._url = url
        self._reconnect_timeout_s = reconnect_timeout_s
        self._max_frames = max_frames

    @property
    def is_live(self) -> bool:
        return True

    @property
    def total_frames(self) -> int | None:
        return None

    def _open_cap(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise SourceError(f"Cannot open RTSP stream: {self._url}")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        cap = self._open_cap()
        frame_index = 0
        yielded = 0
        retry_count = 0

        try:
            while True:
                if self._max_frames is not None and yielded >= self._max_frames:
                    break

                ok, bgr = cap.read()
                if ok:
                    retry_count = 0
                    yield Frame(
                        pixels=bgr.astype(np.uint8),
                        source_id=self._url,
                        frame_index=frame_index,
                        timestamp_ms=0.0,
                    )
                    frame_index += 1
                    yielded += 1
                else:
                    cap.release()
                    # Reset deadline on each disconnect so the timeout measures
                    # the per-reconnect-attempt window, not the stream lifetime.
                    deadline = time.monotonic() + self._reconnect_timeout_s
                    wait = min(2**retry_count, 10)
                    if time.monotonic() + wait > deadline:
                        raise SourceTimeoutError(
                            f"RTSP stream {self._url} timed out after {self._reconnect_timeout_s}s"
                        )
                    time.sleep(wait)
                    retry_count += 1
                    cap = self._open_cap()
        finally:
            cap.release()

    def close(self) -> None:
        pass

    def __enter__(self) -> RTSPStreamSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# WebcamSource
# ---------------------------------------------------------------------------


class WebcamSource:
    """Reads frames from a local webcam device."""

    def __init__(
        self,
        device_index: int,
        *,
        max_frames: int | None = None,
    ) -> None:
        self._device_index = device_index
        self._max_frames = max_frames
        # Eagerly validate device availability.
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"Cannot open webcam device index: {device_index}")
        cap.release()

    @property
    def is_live(self) -> bool:
        return True

    @property
    def total_frames(self) -> int | None:
        return None

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"Cannot open webcam device index: {self._device_index}")

        yielded = 0
        try:
            while True:
                if self._max_frames is not None and yielded >= self._max_frames:
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                yield Frame(
                    pixels=bgr.astype(np.uint8),
                    source_id=f"webcam:{self._device_index}",
                    frame_index=yielded,
                    timestamp_ms=0.0,
                )
                yielded += 1
        finally:
            cap.release()

    def close(self) -> None:
        pass

    def __enter__(self) -> WebcamSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def open_source(
    source: str | Path,
    *,
    loop: bool = False,
    frame_skip: int = 0,
    max_frames: int | None = None,
    reconnect_timeout_s: float = 30.0,
) -> FrameSource:
    """Factory: inspect *source* and return the appropriate FrameSource.

    Dispatch rules (checked in order):
    - String of digits (``"0"``, ``"1"``, …) → WebcamSource.
    - Starts with ``rtsp://`` or ``rtsps://`` → RTSPStreamSource.
    - Path with image extension → ImageFileSource.
    - Existing directory → ImageDirectorySource.
    - Path with video extension → VideoFileSource.
    - Anything else → raises SourceError.

    Args:
        source: File path, URL string, or webcam index string.
        loop: Repeat video file when exhausted (VideoFileSource only).
        frame_skip: Skip N frames between yields (0 = no skip).
        max_frames: Stop after this many yielded frames; ``None`` is unlimited.
        reconnect_timeout_s: Max seconds for RTSP reconnect attempts.

    Returns:
        Appropriate FrameSource implementation.

    Raises:
        SourceError: If no source type matches or the resource is unavailable.
    """
    source_str = str(source)

    # Webcam: pure digit string.
    if isinstance(source, str) and source.isdigit():
        return WebcamSource(int(source), max_frames=max_frames)

    # RTSP stream.
    if source_str.startswith(_RTSP_SCHEMES):
        return RTSPStreamSource(
            source_str,
            reconnect_timeout_s=reconnect_timeout_s,
            max_frames=max_frames,
        )

    path = Path(source_str)
    suffix = path.suffix.lower()

    if path.is_dir():
        return ImageDirectorySource(path)

    if suffix in _IMAGE_EXTS:
        if not path.exists():
            raise SourceError(f"Image file not found: {path}")
        return ImageFileSource(path)

    if suffix in _VIDEO_EXTS:
        if not path.exists():
            raise SourceError(f"Video file not found: {path}")
        return VideoFileSource(
            path,
            loop=loop,
            frame_skip=frame_skip,
            max_frames=max_frames,
        )

    raise SourceError(
        f"Cannot determine source type for: {source!r}. "
        f"Supported: image files {sorted(_IMAGE_EXTS)}, "
        f"video files {sorted(_VIDEO_EXTS)}, "
        f'RTSP URLs (rtsp://), webcam indices ("0", "1", ...).'
    )


__all__ = [
    "FrameSource",
    "ImageDirectorySource",
    "ImageFileSource",
    "RTSPStreamSource",
    "VideoFileSource",
    "WebcamSource",
    "open_source",
]
