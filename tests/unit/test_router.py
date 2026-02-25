"""Tests for DetectionRouter in the multi-stream pipeline."""

from __future__ import annotations

import numpy as np

from yowo.pipeline._router import DetectionRouter
from yowo.types import (
    BackendType,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
    TaggedFrame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)


def _make_frame(index: int = 0, source_id: str = "test") -> Frame:
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id=source_id, frame_index=index)


def _make_detection(frame: Frame) -> Detection:
    return Detection(
        frame=frame,
        boxes=(),
        inference_time_ms=1.0,
        backend=BackendType.PYTORCH,
        model_spec=_SPEC,
    )


# ---------------------------------------------------------------------------
# TestDetectionRouterBasic
# ---------------------------------------------------------------------------


class TestDetectionRouterBasic:
    """Core dispatch semantics."""

    def test_route_dispatches_to_correct_callback(self) -> None:
        router = DetectionRouter()
        received_1: list[tuple[str, list[Detection]]] = []
        received_2: list[tuple[str, list[Detection]]] = []

        router.register("cam-1", lambda sid, dets: received_1.append((sid, dets)))
        router.register("cam-2", lambda sid, dets: received_2.append((sid, dets)))

        frame_1 = _make_frame(0)
        frame_2 = _make_frame(1)
        tagged_1 = TaggedFrame(stream_id="cam-1", frame=frame_1)
        tagged_2 = TaggedFrame(stream_id="cam-2", frame=frame_2)
        det_1 = _make_detection(frame_1)
        det_2 = _make_detection(frame_2)

        router.route([det_1, det_2], [tagged_1, tagged_2])

        assert len(received_1) == 1
        assert received_1[0][0] == "cam-1"
        assert received_1[0][1] == [det_1]

        assert len(received_2) == 1
        assert received_2[0][0] == "cam-2"
        assert received_2[0][1] == [det_2]

    def test_no_callback_drops_silently(self) -> None:
        router = DetectionRouter()

        frame = _make_frame(0)
        tagged = TaggedFrame(stream_id="unregistered", frame=frame)
        det = _make_detection(frame)

        # Should not raise; detection is silently dropped.
        router.route([det], [tagged])

    def test_register_overwrites_previous(self) -> None:
        router = DetectionRouter()
        old_received: list[tuple[str, list[Detection]]] = []
        new_received: list[tuple[str, list[Detection]]] = []

        router.register("cam-1", lambda sid, dets: old_received.append((sid, dets)))
        router.register("cam-1", lambda sid, dets: new_received.append((sid, dets)))

        frame = _make_frame(0)
        tagged = TaggedFrame(stream_id="cam-1", frame=frame)
        det = _make_detection(frame)

        router.route([det], [tagged])

        assert len(old_received) == 0
        assert len(new_received) == 1


# ---------------------------------------------------------------------------
# TestDetectionRouterRegistration
# ---------------------------------------------------------------------------


class TestDetectionRouterRegistration:
    """Register / unregister lifecycle."""

    def test_unregister_removes_callback(self) -> None:
        router = DetectionRouter()
        received: list[tuple[str, list[Detection]]] = []

        router.register("cam-1", lambda sid, dets: received.append((sid, dets)))
        router.unregister("cam-1")

        frame = _make_frame(0)
        tagged = TaggedFrame(stream_id="cam-1", frame=frame)
        det = _make_detection(frame)

        router.route([det], [tagged])

        assert len(received) == 0

    def test_unregister_nonexistent_noop(self) -> None:
        router = DetectionRouter()
        # Should not raise for a never-registered stream.
        router.unregister("phantom")

    def test_multiple_detections_same_stream(self) -> None:
        router = DetectionRouter()
        received: list[tuple[str, list[Detection]]] = []

        router.register("cam-1", lambda sid, dets: received.append((sid, dets)))

        # Three frames and detections, all from the same stream.
        frames = [_make_frame(i) for i in range(3)]
        tagged_batch = [TaggedFrame(stream_id="cam-1", frame=f) for f in frames]
        detections = [_make_detection(f) for f in frames]

        router.route(detections, tagged_batch)

        # Single callback invocation with all 3 detections grouped.
        assert len(received) == 1
        assert received[0][0] == "cam-1"
        assert len(received[0][1]) == 3


# ---------------------------------------------------------------------------
# TestDetectionRouterEdgeCases
# ---------------------------------------------------------------------------


class TestDetectionRouterEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_empty_detections(self) -> None:
        router = DetectionRouter()
        invoked = False

        def _cb(sid: str, dets: list[Detection]) -> None:
            nonlocal invoked
            invoked = True

        router.register("cam-1", _cb)

        frame = _make_frame(0)
        tagged = TaggedFrame(stream_id="cam-1", frame=frame)

        # No detections at all -- callback should never fire.
        router.route([], [tagged])
        assert not invoked

    def test_empty_batch(self) -> None:
        router = DetectionRouter()
        invoked = False

        def _cb(sid: str, dets: list[Detection]) -> None:
            nonlocal invoked
            invoked = True

        router.register("cam-1", _cb)

        # Detection exists but the batch is empty, so the frame cannot be
        # matched to any stream.  The detection should be dropped.
        orphan_frame = _make_frame(0)
        det = _make_detection(orphan_frame)

        router.route([det], [])
        assert not invoked
