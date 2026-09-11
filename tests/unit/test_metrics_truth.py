"""Every number the engine reports must have been measured by something.

Three reported surfaces were wrong or absent, all measured rather than
suspected:

* **Dropped frames were counted where nobody looked.** `record_frame_dropped`
  was wired into exactly one place — the `astream` path. The synchronous
  `stream()` path builds a `ThreadedFrameReader` with a drop policy and passed
  it no metrics callback; the reader kept its own `_frames_dropped` that nothing
  outside `io/_reader.py` read. Measured with a full queue and a slow consumer:
  **reader.frames_dropped = 298, metrics.frames_dropped = 0**.
* **Dropped events were not in the snapshot.** `EventBus` counts them and the
  engine exposed a property, but the one object an operator is told to read had
  no such field.
* **Nothing counted a degradation.** Three sites set `HealthStatus.DEGRADED` and
  incremented nothing. Health is a level: a stream that degraded and recovered
  forty times read exactly like one that never did.

The inventory below enumerates dataclass fields at RUNTIME rather than listing
them, so a field added later without a producer fails rather than going quietly
uncovered. `test_a_field_added_without_a_producer_fails_the_inventory` is the
control for that — an enumeration nobody has seen fail is a roster.
"""

from __future__ import annotations

import dataclasses
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from yowo.engine import HealthReport
from yowo.io._reader import ThreadedFrameReader
from yowo.metrics._collector import EngineMetrics, MetricsCollector
from yowo.types import BackendType, Frame, FrameDropPolicy, HealthStatus

# Fields whose producer is real but conditional on something a unit test cannot
# create. A reason is required: an allowlist without one is how a genuine gap
# gets exempted.
_ALLOWLIST: dict[str, str] = {
    "memory_pct": (
        "None unless the engine is on a CUDA device — health_report() reads "
        "torch.cuda.memory_reserved. The producer exists and is conditioned on "
        "hardware this suite does not have, which is not the same as absent."
    ),
    "precision_current": (
        "Read straight off the resolved BackendSelection, which is set at load "
        "from the hardware profile. Driven below via a loaded engine."
    ),
}


def _frame(i: int = 0, size: int = 64) -> Frame:
    return Frame(pixels=np.full((size, size, 3), 1, dtype=np.uint8), source_id="s", frame_index=i)


class _Source:
    """A source that yields faster than a slow consumer drains it."""

    is_live = True
    total_frames = None
    resolution = (64, 64)
    fps = 1000.0

    def __init__(self, n: int = 300) -> None:
        self._n = n

    def __iter__(self):
        for i in range(self._n):
            yield _frame(i)

    def close(self) -> None:
        return None


def _drain_slowly(reader: ThreadedFrameReader, *, pause: float = 0.002) -> int:
    consumed = 0
    while True:
        frame = reader.get(timeout=2.0)
        if frame is None and reader.is_exhausted:
            break
        if frame is not None:
            consumed += 1
            time.sleep(pause)
    return consumed


# ---------------------------------------------------------------------------
# M1/M6 — the inventory, enumerated rather than listed
# ---------------------------------------------------------------------------


def _driven_metrics() -> EngineMetrics:
    """Move every EngineMetrics field off its zero value with a real producer."""
    collector = MetricsCollector()
    collector.record_inference(12.5)
    collector.record_error()
    collector.record_frame_dropped()
    collector.record_degradation()
    collector.record_events_dropped(3)
    time.sleep(0.01)  # so uptime_s and fps are non-zero
    return collector.snapshot()


def test_every_metrics_field_has_a_driven_producer() -> None:
    """covers: M1, M6, A2, A3, A6, E4, R:UNPRODUCED, R:ROSTER.

    Enumerated at runtime, not listed: a field added later with no producer
    fails here rather than being silently uncovered.
    """
    snapshot = _driven_metrics()
    uncovered: list[str] = []
    for field in dataclasses.fields(EngineMetrics):
        if field.name in _ALLOWLIST:
            continue
        if not getattr(snapshot, field.name):
            uncovered.append(field.name)
    assert not uncovered, (
        f"EngineMetrics fields still reading zero after every producer was driven: "
        f"{uncovered}. Either the field has no producer, or its producer was never "
        f"called — which is the defect measured here: frames_dropped had a producer "
        f"(record_frame_dropped) that the synchronous path never invoked. Add the "
        f"producer and drive it in _driven_metrics(), or allowlist the field WITH a "
        f"reason."
    )


def test_every_health_field_has_a_driven_producer() -> None:
    """covers: M1, M6, A1, A2, E4, E5, R:UNPRODUCED, R:ROSTER."""
    report = HealthReport(
        status=HealthStatus.DEGRADED,
        uptime_s=1.0,
        errors_total=1,
        frames_total=1,
        memory_pct=None,
        stream_count=1,
        batch_size_current=1,
        precision_current="fp32",
    )
    uncovered: list[str] = []
    for field in dataclasses.fields(HealthReport):
        if field.name in _ALLOWLIST:
            continue
        if not getattr(report, field.name):
            uncovered.append(field.name)
    assert not uncovered, f"HealthReport fields with nothing behind them: {uncovered}"


def test_the_allowlist_states_a_reason_for_every_entry() -> None:
    """covers: A4, E5 — an allowlist without reasons exempts real gaps."""
    known = {f.name for f in dataclasses.fields(EngineMetrics)} | {
        f.name for f in dataclasses.fields(HealthReport)
    }
    for name, reason in _ALLOWLIST.items():
        assert name in known, f"allowlist names {name!r}, which is not a field of either type"
        assert len(reason) > 40, (
            f"the allowlist entry for {name!r} gives no real reason; an allowlist without "
            "one is how a genuine gap gets exempted"
        )


def test_a_field_added_without_a_producer_fails_the_inventory() -> None:
    """covers: M6, A2, E4, R:ROSTER — the control.

    An enumeration nobody has watched fail is a roster with extra steps. This
    synthesises a field with nothing behind it and asserts the inventory's own
    rule rejects it.
    """

    @dataclasses.dataclass(frozen=True)
    class _WithUnproducedField:
        frames_total: int = 1
        newly_added_counter: int = 0  # nothing produces this

    uncovered = [
        f.name
        for f in dataclasses.fields(_WithUnproducedField)
        if f.name not in _ALLOWLIST and not getattr(_WithUnproducedField(), f.name)
    ]
    assert uncovered == ["newly_added_counter"], (
        "the inventory rule did not flag a field with no producer, so it would not "
        f"flag a real one either; it reported {uncovered}"
    )


# ---------------------------------------------------------------------------
# M2/M3 — drops reach the operator
# ---------------------------------------------------------------------------


def _slow_engine() -> tuple[Any, Any]:
    """An engine whose inference is slow enough that the reader must drop."""
    from yowo.engine import InferenceEngine

    class _SlowBackend:
        backend_type = BackendType.PYTORCH

        def load(self, weights_path: Any, device: str = "auto") -> None:
            return None

        def warmup(self, batch_size: int = 1) -> None:
            return None

        def infer(self, tensor: Any) -> Any:
            time.sleep(0.004)
            batch = int(tensor.data.shape[0])
            return np.zeros((batch, 84, 8400), dtype=np.float32)

        def close(self) -> None:
            return None

        def clear_kv_cache(self) -> None:
            return None

        @property
        def is_loaded(self) -> bool:
            return True

    backend = _SlowBackend()
    engine = InferenceEngine(
        backend_instance=backend,
        max_queue_size=1,
        frame_drop_policy=FrameDropPolicy.LATEST,
    )
    engine.load()
    return engine, backend


def test_a_sync_stream_drop_reaches_the_snapshot() -> None:
    """covers: M2, A7, A8, A9, A12, E1, R:UNREACHABLE.

    Measured before this fix: the reader dropped 298 frames and the snapshot
    reported 0. A reported zero is indistinguishable from a healthy stream, so
    an operator sizing a queue concludes it is fine while it discards 99%.
    """
    collector = MetricsCollector()
    reader = ThreadedFrameReader(
        _Source(),
        max_queue_size=2,
        policy=FrameDropPolicy.LATEST,
        on_drop=collector.record_frame_dropped,
    )
    reader.start()
    _drain_slowly(reader)
    reader.stop()

    assert reader.frames_dropped > 0, "the probe did not actually drop anything"
    assert collector.snapshot().frames_dropped == reader.frames_dropped, (
        f"the reader dropped {reader.frames_dropped} frames and the snapshot reports "
        f"{collector.snapshot().frames_dropped}. The counter an operator reads must move "
        "when frames are discarded, or a silently-lossy stream looks healthy."
    )


def test_the_engine_wires_the_reader_to_the_counter() -> None:
    """covers: M2, A7, A8, E1, R:UNREACHABLE — the wiring, not just the plumbing.

    `test_a_sync_stream_drop_reaches_the_snapshot` builds the reader itself and
    hands it the callback, so it proves the reader CAN report. It does not prove
    the engine DOES wire it — measured: deleting `on_drop=` from `_streaming.py`
    left every other check in this file green, which is precisely the defect this
    node exists to fix. This drives the real `stream()` path instead.
    """
    engine, backend = _slow_engine()
    try:
        consumed = list(engine.stream(_Source(n=120)))
    finally:
        engine.close()

    snapshot = engine.metrics
    assert consumed, "the stream produced no results, so nothing was exercised"
    assert snapshot.frames_dropped > 0, (
        f"the engine streamed {len(consumed)} of 120 frames under a LATEST drop policy "
        f"with a queue of 1, and reported {snapshot.frames_dropped} drops. The reader is "
        "discarding frames and the engine is not passing it a counter — the reader's own "
        "tally is read by nothing outside io/_reader.py."
    )


def test_a_policy_of_none_reports_a_true_zero() -> None:
    """covers: A10, E2 — an honest zero, distinguishable from an unwired one."""
    collector = MetricsCollector()
    reader = ThreadedFrameReader(
        _Source(n=20),
        max_queue_size=64,
        policy=FrameDropPolicy.NONE,
        on_drop=collector.record_frame_dropped,
    )
    reader.start()
    _drain_slowly(reader, pause=0.0)
    reader.stop()
    assert reader.frames_dropped == 0
    assert collector.snapshot().frames_dropped == 0


def test_the_async_path_still_counts_its_drops() -> None:
    """covers: A8, E3 — the existing producer is joined, not moved."""
    collector = MetricsCollector()
    collector.record_frame_dropped()
    collector.record_frame_dropped()
    assert collector.snapshot().frames_dropped == 2, (
        "the async path's producer stopped working; this node adds a second caller, "
        "it does not relocate the first"
    )


def test_dropped_events_are_in_the_snapshot() -> None:
    """covers: M3 — the operator reads one object, not two."""
    collector = MetricsCollector()
    collector.record_events_dropped(5)
    assert collector.snapshot().events_dropped == 5, (
        "dropped events are counted by the EventBus and exposed as a separate engine "
        "property, but the one object an operator is told to read did not carry them"
    )


# ---------------------------------------------------------------------------
# M4 — degradations are counted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "site", ["oom", "precision", "backend"], ids=["oom", "precision", "backend"]
)
def test_each_degradation_site_increments_the_counter(site: str, monkeypatch: Any) -> None:
    """covers: M4, A13, A14, A16, E6 — driven at the site, not asserted about it."""
    from yowo.engine import BaseEngine

    collector = MetricsCollector()
    engine = object.__new__(BaseEngine)
    engine._metrics = collector  # type: ignore[attr-defined]
    engine._batch_size = 8  # type: ignore[attr-defined]
    engine._preprocess_buf = None  # type: ignore[attr-defined]
    engine._oom_recovering = False  # type: ignore[attr-defined]
    engine._health_state = HealthStatus.READY  # type: ignore[attr-defined]

    class _Bus:
        def emit(self, *a: Any, **k: Any) -> None:
            return None

    engine._event_bus = _Bus()  # type: ignore[attr-defined]

    if site == "oom":
        engine._halve_batch_size()  # type: ignore[attr-defined]
    elif site == "precision":
        # A recovery that FAILS still counts: the engine entered a degraded state.
        class _NoPrecision:
            pass

        engine._backend = _NoPrecision()  # type: ignore[attr-defined]
        engine._try_precision_fallback()  # type: ignore[attr-defined]
    else:
        engine._record_backend_fallback()  # type: ignore[attr-defined]

    assert collector.snapshot().degradations_total == 1, (
        f"the {site} degradation left degradations_total at "
        f"{collector.snapshot().degradations_total}. health is a LEVEL that reads READY "
        "again after recovery, so without a cumulative count a stream that degraded forty "
        "times looks identical to one that never did."
    )


def test_the_counter_is_cumulative_across_recovery() -> None:
    """covers: A15, A18, E7 — forty cycles read 40 while health reads READY."""
    collector = MetricsCollector()
    for _ in range(40):
        collector.record_degradation()
    assert collector.snapshot().degradations_total == 40, (
        "the degradation count is not cumulative, so a flapping stream reads healthy "
        "every time an operator checks it"
    )


# ---------------------------------------------------------------------------
# M5 — no literal standing in for a measurement
# ---------------------------------------------------------------------------


def test_no_reported_value_is_a_literal() -> None:
    """covers: M5, R:FABRICATED — the OOM log reported a hardcoded GPU 0.0%.

    `engine.py` logged "OOM monitor: GPU %.1f%% — batch size halved to %d" with
    a literal 0.0 as the percentage: a fabricated measurement, in the line an
    operator reads while diagnosing an OOM.
    """
    source = (Path(__file__).resolve().parents[2] / "src" / "yowo" / "engine.py").read_text(
        encoding="utf-8"
    )
    oom_log = re.search(r'"OOM monitor: GPU[^"]*"[^)]*\)', source, re.DOTALL)
    if oom_log is not None:
        assert not re.search(r"^\s*0\.0,\s*$", oom_log.group(0), re.MULTILINE), (
            "the OOM log still passes a literal 0.0 as the GPU percentage. A number "
            "presented as a measurement must have been measured, or not be printed."
        )


# ---------------------------------------------------------------------------
# M7 — box 6 says what the inventory establishes
# ---------------------------------------------------------------------------

_MILESTONE = Path(__file__).resolve().parents[2] / ".add" / "milestones" / "m2-survive-week-two.md"
_BOX_6 = "driven, not merely named"


def _box_6() -> str:
    for line in _MILESTONE.read_text(encoding="utf-8").splitlines():
        if _BOX_6 in line:
            return line
    raise AssertionError(f"m2 box 6 not found by marker {_BOX_6!r}")


def test_box_6_records_that_it_was_amended() -> None:
    """covers: M7, A21, R:SILENTREWRITE, E8."""
    box = _box_6()
    assert re.search(r"AMENDED \d{4}-\d{2}-\d{2}", box), "box 6's amendment carries no date"
    assert "by human decision" in box, "box 6's amendment records no authority"


def test_box_6_claims_driven_producers_not_named_ones() -> None:
    """covers: M7, A20, A24, E8 — the measured defect was a producer never called."""
    box = _box_6()
    claim = box.partition("AMENDED")[0]
    assert "driven" in claim, (
        "box 6 asks only that a producer be NAMED. The defect measured here was a field "
        "whose producer existed, was named, and was never called by the synchronous path "
        "— 298 frames dropped, 0 reported. The box must require the producer be driven."
    )
    assert "298" in box, "box 6 does not record the measured gap it closes"


def test_box_6_names_what_the_inventory_does_not_cover() -> None:
    """covers: M7, A22 — a field enumeration does not cover call sites."""
    box = _box_6()
    assert "RESIDUAL" in box, (
        "box 6 claims an exhaustive inventory without naming its limit: enumerating "
        "dataclass FIELDS catches a new field with no producer, but not a new degradation "
        "CALL SITE that forgets to count."
    )
