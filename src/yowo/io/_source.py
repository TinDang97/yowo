"""Input source abstraction.

FrameSource Protocol + open_source() factory.
Dispatch by file extension and scheme.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from yowo.errors import SourceError, SourceTimeoutError
from yowo.io._redact import redact_url
from yowo.types import IMAGE_EXTS, RTSP_SCHEMES, VIDEO_EXTS, Frame

logger = logging.getLogger(__name__)


class _ClosedCapture:
    """Stands in between a failed reopen and the next attempt.

    The loop needs *something* to call `read()` and `release()` on while the camera is
    away. A `None` here would mean a branch on every use; this reports "no frame" and
    lets the outage clock -- not an exception -- decide when the stream ends.
    """

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        return None

    def isOpened(self) -> bool:
        return False


_CLOSED_CAP = _ClosedCapture()


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
        self._resolution: tuple[int, int] | None = None
        self._probed_resolution: bool = False

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int | None:
        return 1

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Native image resolution (height, width), or None if unreadable."""
        if not self._probed_resolution:
            img = cv2.imread(str(self._path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self._resolution = (img.shape[0], img.shape[1])
            self._probed_resolution = True
        return self._resolution

    @property
    def fps(self) -> float | None:
        """Always None for single images."""
        return None

    def __iter__(self) -> Iterator[Frame]:
        pixels = cv2.imread(str(self._path))
        if pixels is None:
            raise SourceError(f"cv2.imread failed for: {self._path}")
        yield Frame(
            pixels=np.asarray(pixels, dtype=np.uint8),
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
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise SourceError(f"No supported image files found in directory: {directory}")
        self._files = files

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int | None:
        return len(self._files)

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Always None for image directories (heterogeneous sizes)."""
        return None

    @property
    def fps(self) -> float | None:
        """Always None for image directories."""
        return None

    def __iter__(self) -> Iterator[Frame]:
        for idx, path in enumerate(self._files):
            pixels = cv2.imread(str(path))
            if pixels is None:
                raise SourceError(f"cv2.imread failed for: {path}")
            yield Frame(
                pixels=np.asarray(pixels, dtype=np.uint8),
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
        self._probed: bool = False
        self._resolution: tuple[int, int] | None = None
        self._fps: float | None = None

    @property
    def is_live(self) -> bool:
        return False

    def _probe(self) -> None:
        """Read all video metadata in a single VideoCapture open."""
        if self._probed:
            return
        cap = cv2.VideoCapture(str(self._path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self._total = count if count > 0 else None
        self._resolution = (h, w) if h > 0 and w > 0 else None
        self._fps = fps_val if fps_val and fps_val > 0 else None
        self._probed = True

    @property
    def total_frames(self) -> int | None:
        self._probe()
        return self._total

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Native source resolution (height, width), or None if unavailable."""
        self._probe()
        return self._resolution

    @property
    def fps(self) -> float | None:
        """Source frame rate, or None if unknown."""
        self._probe()
        return self._fps

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
                        pixels=np.asarray(bgr, dtype=np.uint8),
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
        reconnect_timeout_s: float | None = 30.0,
        max_frames: int | None = None,
        frame_skip: int = 0,
        open_timeout_ms: int | None = None,
        read_timeout_ms: int | None = None,
    ) -> None:
        # The connectable URL stays private and is never interpolated into a
        # message or published as an identifier. `safe_url` is what everything
        # else reads, so a surface added later inherits redaction (M2).
        self._url = url
        self._safe_url = redact_url(url)
        self._reconnect_timeout_s = reconnect_timeout_s
        self._max_frames = max_frames
        self._frame_skip = frame_skip
        # `None` means "leave the backend's own default in place", which is what every
        # caller got before these existed -- measured 30.08s against both an unroutable
        # host and a server that accepts and never speaks. It is NOT the same request as
        # `0`, which FFmpeg reads as "no timeout". Conflating them turns "never time out"
        # into "wait thirty seconds", or the reverse.
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._active_cap: cv2.VideoCapture | None = None

    @property
    def safe_url(self) -> str:
        """The stream URL with any credential removed — safe to emit anywhere."""
        return self._safe_url

    @property
    def is_live(self) -> bool:
        return True

    @property
    def total_frames(self) -> int | None:
        return None

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Always None for RTSP streams (unknown until connected)."""
        return None

    @property
    def fps(self) -> float | None:
        """Always None for RTSP streams (unknown until connected)."""
        return None

    def _capture_params(self) -> list[int] | None:
        """The timeout properties to build the capture with, or None for the default.

        These bind at CONSTRUCTION. `cap.set()` afterwards is too late to bound the open,
        which is the wait that matters when a camera is simply gone.
        """
        params: list[int] = []
        if self._open_timeout_ms is not None:
            params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._open_timeout_ms]
        if self._read_timeout_ms is not None:
            params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_timeout_ms]
        return params or None

    def _new_cap(self) -> cv2.VideoCapture:
        """The ONE place a capture for this stream is constructed.

        Every site routes through here so that a site added later is bounded by
        construction rather than by whoever remembers. Unconfigured, the call is
        byte-identical to what it always was.
        """
        params = self._capture_params()
        if params is None:
            return cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self._url, cv2.CAP_FFMPEG, params)

    def _bounds(self) -> str:
        """What the operator needs in the message: which bound was in force."""
        parts = [
            f"open {self._open_timeout_ms}ms"
            if self._open_timeout_ms is not None
            else "open backend default",
            f"read {self._read_timeout_ms}ms"
            if self._read_timeout_ms is not None
            else "read backend default",
        ]
        return ", ".join(parts)

    def _open_cap(self) -> cv2.VideoCapture:
        cap = self._new_cap()
        if not cap.isOpened():
            cap.release()
            # A stall with no output is indistinguishable from a hung process, so an
            # operator kills it and loses the diagnosis. Name the camera and the bound.
            raise SourceError(f"Cannot open RTSP stream: {self._safe_url} [{self._bounds()}]")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        cap = self._open_cap()
        self._active_cap = cap
        frame_index = 0
        yielded = 0
        retry_count = 0
        # None while the stream is healthy; the monotonic time of the first failure
        # of the current outage otherwise.
        outage_started: float | None = None
        skip_mod = self._frame_skip + 1

        try:
            while True:
                if self._max_frames is not None and yielded >= self._max_frames:
                    break

                ok, bgr = cap.read()
                if ok:
                    if outage_started is not None:
                        # A silent recovery is indistinguishable from a stall of the same
                        # length, and an operator has no way to tell which they had.
                        logger.info(
                            "RTSP stream %s recovered after %.1fs",
                            self._safe_url,
                            time.monotonic() - outage_started,
                        )
                        outage_started = None
                    retry_count = 0
                    # Always drain the buffer (cap.read) to avoid RTSP lag,
                    # but only yield every (frame_skip+1)th frame.
                    if frame_index % skip_mod == 0:
                        yield Frame(
                            pixels=np.asarray(bgr, dtype=np.uint8),
                            source_id=self._safe_url,
                            frame_index=frame_index,
                            timestamp_ms=0.0,
                        )
                        yielded += 1
                    frame_index += 1
                else:
                    cap.release()
                    # The clock starts at the FIRST failure of a contiguous outage and is
                    # cleared by a successful read, so `reconnect_timeout_s` measures how
                    # long the camera has been away -- not how long this one attempt took.
                    # Resetting it on every disconnect is what made it unable to
                    # accumulate, so 30 and 60 behaved identically.
                    if outage_started is None:
                        outage_started = time.monotonic()

                    wait = min(2**retry_count, 10)
                    if self._reconnect_timeout_s is not None:
                        elapsed = time.monotonic() - outage_started
                        # The BOUND wins over the backoff: if the next sleep would carry
                        # us past the deadline we stop now rather than sleeping through it
                        # and noticing late. The old guard compared `wait` to the bound,
                        # which reduces to `wait > timeout` -- and `wait` never exceeds 10,
                        # so nothing at or above 10 could ever fire it.
                        if elapsed + wait > self._reconnect_timeout_s:
                            raise SourceTimeoutError(
                                f"RTSP stream {self._safe_url} timed out after "
                                f"{elapsed:.1f}s of a {self._reconnect_timeout_s}s "
                                "reconnect budget"
                            )

                    time.sleep(wait)
                    retry_count += 1
                    try:
                        cap = self._open_cap()
                    except SourceError:
                        # A reopen that fails while the camera reboots is a RETRY, not the
                        # end of the stream. This raise propagating out of the generator is
                        # what killed a stream on a single blip: two frames, one failed
                        # reopen, permanently dead. The outage clock above is what ends it.
                        cap = _CLOSED_CAP
                        continue
                    self._active_cap = cap
        finally:
            cap.release()
            self._active_cap = None

    def reconnect(self) -> None:
        """Release and re-open the active RTSP capture to prevent memory leaks.

        Safe to call from the :class:`ThreadedFrameReader` background thread
        between frame reads. If no active capture exists, this is a no-op.
        """
        cap = self._active_cap
        if cap is None:
            return
        cap.release()
        # Built through `_new_cap` like every other site: a bound applied only to the
        # obvious construction leaves the PERIODIC reconnect unbounded, and that is the
        # one that runs unattended (R:PARTIAL).
        new_cap = self._new_cap()
        if not new_cap.isOpened():
            # Fallback: re-open original — let the iterator handle retry
            new_cap = self._new_cap()
        self._active_cap = new_cap

    def close(self) -> None:
        """Release any active capture."""
        cap = self._active_cap
        if cap is not None:
            cap.release()
            self._active_cap = None

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
        frame_skip: int = 0,
    ) -> None:
        self._device_index = device_index
        self._max_frames = max_frames
        self._frame_skip = frame_skip
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

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Always None for webcam (unknown until streaming)."""
        return None

    @property
    def fps(self) -> float | None:
        """Always None for webcam (unknown until streaming)."""
        return None

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"Cannot open webcam device index: {self._device_index}")

        yielded = 0
        read_index = 0
        skip_mod = self._frame_skip + 1
        try:
            while True:
                if self._max_frames is not None and yielded >= self._max_frames:
                    break
                ok, bgr = cap.read()
                if not ok:
                    break
                if read_index % skip_mod == 0:
                    yield Frame(
                        pixels=np.asarray(bgr, dtype=np.uint8),
                        source_id=f"webcam:{self._device_index}",
                        frame_index=yielded,
                        timestamp_ms=0.0,
                    )
                    yielded += 1
                read_index += 1
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
    source: str | Path | int,
    *,
    loop: bool = False,
    frame_skip: int = 0,
    max_frames: int | None = None,
    reconnect_timeout_s: float = 30.0,
    open_timeout_ms: int | None = None,
    read_timeout_ms: int | None = None,
) -> FrameSource:
    """Factory: inspect *source* and return the appropriate FrameSource.

    Dispatch rules (checked in order):
    - Integer or string of digits (``0``, ``"0"``, ``"1"``, …) → WebcamSource.
    - Starts with ``rtsp://`` or ``rtsps://`` → RTSPStreamSource.
    - Path with image extension → ImageFileSource.
    - Existing directory → ImageDirectorySource.
    - Path with video extension → VideoFileSource.
    - Anything else → raises SourceError.

    Args:
        source: File path, URL string, webcam index string, or integer device index.
        loop: Repeat video file when exhausted (VideoFileSource only).
        frame_skip: Skip N frames between yields (0 = no skip).
        max_frames: Stop after this many yielded frames; ``None`` is unlimited.
        reconnect_timeout_s: Max seconds a contiguous RTSP outage may last before
            `SourceTimeoutError`. The clock starts at the first failure and resets on
            the next successful read. ``0`` makes the first failure terminal; ``None``
            retries without bound.
        open_timeout_ms: Milliseconds to wait for a NETWORK capture to open. ``None``
            leaves the backend's own default in force — measured at ~30s, which is what
            every caller got before this parameter existed. ``0`` is a different request:
            FFmpeg reads it as no timeout at all. Note the unit: ``reconnect_timeout_s``
            beside it is in SECONDS.
        read_timeout_ms: Milliseconds to wait for a frame from a NETWORK capture, same
            conventions. This is the bound that matters when a camera completes the TCP
            handshake and then says nothing — the case a port check calls healthy.

    Returns:
        Appropriate FrameSource implementation.

    Raises:
        SourceError: If no source type matches, the resource is unavailable, a timeout
            is negative, or a capture timeout is given for a non-network source.
    """
    source_str = str(source)
    # One redaction, at the boundary, before any branch can build a message and
    # before Path() can rewrite the value — a message built from the Path form
    # reads `https:/` with one slash, which is the tell that the value was
    # laundered before anyone looked at it. Every `raise` below emits this and
    # only this, so a branch added later inherits the redaction instead of
    # having to remember it. A value with no userinfo is returned byte for byte,
    # so plain paths and webcam indices are unaffected.
    safe_source = redact_url(source_str)

    # Validated here, at the boundary, while the caller can still see their own call. A
    # negative timeout deep inside a capture surfaces as a hang, which is the one symptom
    # this whole node exists to remove.
    for name, value in (("open_timeout_ms", open_timeout_ms), ("read_timeout_ms", read_timeout_ms)):
        if value is not None and value < 0:
            # `safe_source` and only `safe_source`: every raise in this factory reads
            # the redacted form, so a branch added later inherits redaction instead of
            # having to remember it. `test_every_raise_in_open_source_reads_the_redacted_form`
            # enforces that, and caught this raise when it named neither.
            raise SourceError(f"{name} must be >= 0, got {value} (for source {safe_source!r})")

    # These are FFMPEG capture properties. A file, directory, image or webcam source is
    # opened through a different backend that does not honour them, so accepting them
    # there would hand the caller a parameter that silently does nothing -- worse than
    # one that is absent, because they would believe they were protected.
    if (open_timeout_ms is not None or read_timeout_ms is not None) and not source_str.startswith(
        RTSP_SCHEMES
    ):
        raise SourceError(
            f"open_timeout_ms/read_timeout_ms apply to network (rtsp://, rtsps://) "
            f"sources only; {safe_source!r} is not one. Its backend does not honour "
            "these properties, so setting them here would do nothing."
        )

    # Integer webcam index (e.g. open_source(0)).
    if isinstance(source, int):
        return WebcamSource(source, max_frames=max_frames, frame_skip=frame_skip)

    # Webcam: pure digit string.
    if isinstance(source, str) and source.isdigit():
        return WebcamSource(int(source), max_frames=max_frames, frame_skip=frame_skip)

    # RTSP stream.
    if source_str.startswith(RTSP_SCHEMES):
        return RTSPStreamSource(
            source_str,
            reconnect_timeout_s=reconnect_timeout_s,
            max_frames=max_frames,
            frame_skip=frame_skip,
            open_timeout_ms=open_timeout_ms,
            read_timeout_ms=read_timeout_ms,
        )

    path = Path(source_str)
    suffix = path.suffix.lower()

    if path.is_dir():
        return ImageDirectorySource(path)

    if suffix in IMAGE_EXTS:
        if not path.exists():
            raise SourceError(f"Image file not found: {safe_source}")
        return ImageFileSource(path)

    if suffix in VIDEO_EXTS:
        if not path.exists():
            raise SourceError(f"Video file not found: {safe_source}")
        return VideoFileSource(
            path,
            loop=loop,
            frame_skip=frame_skip,
            max_frames=max_frames,
        )

    raise SourceError(
        f"Cannot determine source type for: {safe_source!r}. "
        f"Supported: image files {sorted(IMAGE_EXTS)}, "
        f"video files {sorted(VIDEO_EXTS)}, "
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
