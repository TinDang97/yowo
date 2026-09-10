"""Every sink m1 box 6 names is bound by a check that fails when the credential reaches it.

Red-first for ADD task `credential-sinks-are-bound`.

Box 6 names four sinks: a log, an exception message, a result payload, a cache key.
`tests/unit/test_rtsp_redaction.py` claims to bind all four. It does not: on 2026-09-10,
mutating `_source.py:346` from ``source_id=self._safe_url`` to ``source_id=self._url`` --
which puts a raw camera password into every emitted result AND into the feature-cache dict
key -- left ALL TEN of its checks green. Two of its "sink" checks have byte-identical bodies
and both read ``source.safe_url``, an attribute redacted *upstream* of every sink (R:UPSTREAM);
a third asserts a token is not spelled in ``inspect.getsource`` (method M9, R:SOURCESCAN); and
the log sink has no check at all.

Every check here drives a credentialed URL through the REAL code path and asserts on what the
SINK received -- the string a logger emitted, the exception that propagated, the identifier on
an emitted result, the key the cache stored. Never ``safe_url``; never source text; never a
value this module redacted itself (M1, M3, M4, R:UPSTREAM, R:SOURCESCAN, R:SELFANSWER).

That claim is not taken on trust either: ``test_the_mutation_that_leaks_turns_every_sink_check_red``
executes the refutation. It rebuilds each sink's module from source with the leak reinstated and
re-runs every sink assertion against the mutant, requiring each to fail. A check that survives
its mutation is not binding and does not count (M2).

`tests/unit/test_rtsp_redaction.py` is NOT edited -- it belongs to a frozen, gated node, and a
node that fixes weak checks by editing another node's checks is one refactor away from fixing
them by deleting them (M7, A7, E8). Its byte-identity is asserted below.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import numpy as np
import pytest

from yowo.errors import SourceError, SourceTimeoutError
from yowo.io import _source as source_module
from yowo.io._source import RTSPStreamSource
from yowo.pipeline import BatchScheduler, DetectionRouter, FrameCollector, run_pipeline
from yowo.postprocess import postprocess_classify
from yowo.types import (
    BackendType,
    Detection,
    Frame,
    ModelFamily,
    ModelSize,
    ModelSpec,
)

# ---------------------------------------------------------------------------
# The credential. Obviously fake, and never a real host.
# ---------------------------------------------------------------------------

USER = "camop"
PASSWORD = "hunter2"
HOST = "10.0.0.5"
PORT = "554"
PATH = "/Streaming/Channels/101"

URL = f"rtsp://{USER}:{PASSWORD}@{HOST}:{PORT}{PATH}"
URL_USER_ONLY = f"rtsp://{USER}@{HOST}:{PORT}{PATH}"
URL_MALFORMED = f"rtsp://{USER}:{PASSWORD}@[not-a-valid-host:::{PATH}"
URL_HTTP = f"http://{USER}:{PASSWORD}@{HOST}/stream.m3u8"

# Expected sink values are written out as LITERALS. Calling `redact_url` to build the
# expectation would make every assertion below self-answering (R:SELFANSWER): the test
# would be comparing the redactor against itself rather than against a known-good string.
SAFE_URL = f"rtsp://{HOST}:{PORT}{PATH}"
SAFE_URL_USER_ONLY = f"rtsp://{HOST}:{PORT}{PATH}"
SAFE_MALFORMED = "<unparseable url>"
SAFE_HTTP = f"http://{HOST}/stream.m3u8"

_SPEC = ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)

# The one phrase that marks "a credential reached this sink". `_refute` requires it, so a
# mutant that merely crashed cannot be mistaken for a mutant that leaked.
_LEAK_BANNER = "CREDENTIAL LEAK at sink"

# `tests/unit/test_rtsp_redaction.py` as gated by node `rtsp-credential-redaction`
# (commit 3d0594e, the file's only commit).
PARENT_CHECKS = Path(__file__).parent / "test_rtsp_redaction.py"
PARENT_CHECKS_SHA256 = "8a8f4f8f648a093100d29c84449feb7519a20319797aec777f479697048404e7"


# ---------------------------------------------------------------------------
# Assertion helper
#
# A6: a maintainer reading a failure must know WHICH sink leaked and what reached it,
# so the message names the sink and shows the emitted-vs-expected comparison rather
# than asserting `not in` and printing nothing.
# ---------------------------------------------------------------------------


def _assert_sink_clean(sink: str, emitted: object, *, expected: str) -> None:
    """Fail if `emitted` -- a value taken FROM a sink -- carries any of the credential."""
    assert isinstance(emitted, str), (
        f"sink {sink}: expected a string from the sink, got {type(emitted).__name__}"
    )
    leaked = [part for part in (PASSWORD, USER) if part in emitted]
    if leaked:
        raise AssertionError(
            f"{_LEAK_BANNER}: {sink}\n"
            f"  the sink received : {emitted!r}\n"
            f"  leaked component/s: {leaked!r}\n"
            f"  expected at sink  : {expected!r}\n"
            f"  the raw URL       : {URL!r}\n"
            "  Fix at the boundary, where the identifier is first accepted -- not here (M5)."
        )
    assert emitted == expected, (
        f"sink {sink}: no credential leaked, but the identifier is not the redacted form.\n"
        f"  the sink received: {emitted!r}\n"
        f"  expected         : {expected!r}"
    )


def _assert_message_sink_clean(sink: str, message: str, *, expected: str) -> None:
    """Fail if a free-text sink -- an exception message, a log record -- leaks.

    Exact equality is wrong for these: the surrounding prose is not part of the
    guarantee and rewording it must not turn a credential check red. What IS part of
    the guarantee is that no credential appears and that the redacted identifier does,
    because an operator who cannot tell WHICH camera failed cannot act on the message.
    """
    leaked = [part for part in (PASSWORD, USER) if part in message]
    if leaked:
        raise AssertionError(
            f"{_LEAK_BANNER}: {sink}\n"
            f"  the sink received : {message!r}\n"
            f"  leaked component/s: {leaked!r}\n"
            f"  expected to name  : {expected!r}\n"
            f"  the raw URL       : {URL!r}\n"
            "  Fix at the boundary, where the identifier is first accepted -- not here (M5)."
        )
    assert expected in message, (
        f"sink {sink}: nothing leaked, but the message does not name the stream either, "
        f"so an operator cannot tell which camera it refers to (A6).\n"
        f"  the sink received: {message!r}\n"
        f"  expected to contain: {expected!r}"
    )


# ---------------------------------------------------------------------------
# Driving the REAL read path
# ---------------------------------------------------------------------------


@contextmanager
def _patched_capture() -> Iterator[mock.MagicMock]:
    """Patch cv2.VideoCapture so the real RTSPStreamSource read loop can run.

    The mock stands in for the network, nothing else: `_open_cap`, the read loop,
    the Frame construction and the redaction are all the real code.
    """
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap = cap_cls.return_value
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))
        yield cap_cls


def _read_one_frame(url: str, *, source_cls: type = RTSPStreamSource) -> Frame:
    """Return the Frame the REAL read path emitted for `url`.

    Fails -- never skips -- if the read path produced no frame, because a sink that was
    never written to reads exactly like a sink that was written to safely (A4).
    """
    with _patched_capture():
        frames = list(source_cls(url, max_frames=1))
    assert len(frames) == 1, (
        f"the frame sink was never written to: the read path for {url!r} emitted "
        f"{len(frames)} frames, so nothing downstream can be asserted (A4)"
    )
    return frames[0]


# ---------------------------------------------------------------------------
# The sink assertions.
#
# Each takes the source class it should drive, so the mutation harness can re-run the
# exact same assertion against a module rebuilt with the leak reinstated (M2). A sink
# assertion that cannot be pointed at a mutant cannot be refuted, and an unrefuted
# check is the defect this node exists to fix.
# ---------------------------------------------------------------------------


def _sink_emitted_frame_identifier(source_cls: type, url: str, expected: str) -> None:
    """Sink: `Frame.source_id`, io/_source.py:346 -- the id every result inherits."""
    frame = _read_one_frame(url, source_cls=source_cls)
    _assert_sink_clean("Frame.source_id (io/_source.py:346)", frame.source_id, expected=expected)


def _sink_postprocess_result_identifier(source_cls: type, url: str, expected: str) -> None:
    """Sink: the id on an emitted result, and in the dict that becomes result JSON."""
    frame = _read_one_frame(url, source_cls=source_cls)

    # postprocess/_classify.py:67 -- the real decoder, driven with the real Frame.
    results = postprocess_classify(
        np.array([[0.1, 0.9]], dtype=np.float32),
        [frame],
        model_spec=_SPEC,
        backend=BackendType.PYTORCH,
    )
    assert len(results) == 1, "the postprocess sink was never written to (A4)"
    _assert_sink_clean(
        "ClassificationResult.source_id (postprocess/_classify.py:67)",
        results[0].source_id,
        expected=expected,
    )
    _assert_sink_clean(
        "ClassificationResult.to_dict()['source_id'] (result JSON)",
        results[0].to_dict()["source_id"],
        expected=expected,
    )

    # types.py:322 -- the detection result payload reads frame.source_id directly.
    detection = Detection(
        frame=frame,
        boxes=(),
        inference_time_ms=1.0,
        backend=BackendType.PYTORCH,
        model_spec=_SPEC,
    )
    _assert_sink_clean(
        "Detection.to_dict()['source_id'] (result JSON)",
        detection.to_dict()["source_id"],
        expected=expected,
    )
    assert PASSWORD not in detection.to_json(), "the credential reached serialised result JSON"


def _feature_cache_fed_from(source_cls: type, url: str) -> tuple[Any, str]:
    """Drive the real read path into a real FeatureCache, exactly as the engine does.

    engine.py:832 takes ``frames[0].source_id`` and hands it to the backend, which
    uses it verbatim as the feature-cache key. This reproduces that hand-off with the
    real Frame and the real cache.
    """
    from yowo.cache import FeatureCache

    frame = _read_one_frame(url, source_cls=source_cls)
    sid = frame.source_id  # engine.py:832

    cache = FeatureCache()
    tensor = np.zeros((1, 3, 8, 8), dtype=np.float32)
    feats = (
        np.zeros((1, 4, 2, 2), dtype=np.float32),
        np.zeros((1, 4, 2, 2), dtype=np.float32),
        np.zeros((1, 4, 2, 2), dtype=np.float32),
    )
    cache.check_and_load(sid, tensor)
    cache.update(sid, tensor, feats)
    return cache, sid


def _sink_feature_cache_dict_key(source_cls: type, url: str, expected: str) -> None:
    """Sink: `cache/__init__.py:151` -- ``self._last_fingerprints[source_id] = fp``.

    Read back as the DICT KEY the cache actually holds, not as the value passed in.
    A dict key outlives log scrubbing entirely (E4).
    """
    cache, _ = _feature_cache_fed_from(source_cls, url)
    keys = list(cache._last_fingerprints)
    assert len(keys) == 1, (
        f"the fingerprint sink was never written to: {len(keys)} keys present (A4)"
    )
    _assert_sink_clean(
        "FeatureCache._last_fingerprints key (cache/__init__.py:151)", keys[0], expected=expected
    )


def _sink_feature_store_key(source_cls: type, url: str, expected: str) -> None:
    """Sink: `cache/__init__.py:152` -- ``self._store.store(source_id, ...)``.

    The same id is stored a second time, in a second structure. A key in two places
    is two sinks (E4), and this one survives ``_last_fingerprints`` eviction.
    """
    cache, _ = _feature_cache_fed_from(source_cls, url)
    keys = list(cache._store._active_entries())
    assert len(keys) == 1, f"the store sink was never written to: {len(keys)} keys present (A4)"
    _assert_sink_clean("FeatureStore key (cache/__init__.py:152)", keys[0], expected=expected)

    # E4: evicting the fingerprint must not leave the credential behind in the store.
    cache._last_fingerprints.clear()
    survivors = list(cache._store._active_entries())
    assert len(survivors) == 1, "the store key vanished with the fingerprint; E4 is unbound"
    _assert_sink_clean(
        "FeatureStore key after fingerprint eviction (E4)", survivors[0], expected=expected
    )


def _sink_propagated_exception(source_cls: type, url: str, expected: str) -> None:
    """Sink: the `SourceError` that actually propagates out of the real `_open_cap`.

    Bound at the sink -- the exception a caller catches -- rather than trusted to the
    parent node's check (E2, A2).
    """
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap_cls.return_value.isOpened.return_value = False
        with pytest.raises(SourceError) as excinfo:
            source_cls(url)._open_cap()
    message = str(excinfo.value)
    assert message, "the exception sink was never written to (A4)"
    _assert_message_sink_clean(
        "SourceError message (io/_source.py:322)", message, expected=expected
    )
    _assert_message_sink_clean(
        "SourceError repr (what a crash reporter records)", repr(excinfo.value), expected=expected
    )

    # The reconnect-timeout message (io/_source.py:360) is a SECOND exception-message
    # sink. In the parent node its only guard is an `inspect.getsource` assertion that a
    # token is not spelled -- method M9, the shape this node exists to replace. Bind it
    # here on the exception that actually propagates.
    with mock.patch("cv2.VideoCapture") as cap_cls:
        cap = cap_cls.return_value
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with pytest.raises(SourceTimeoutError) as timeout_info:
            list(source_cls(url, reconnect_timeout_s=0.0))
    timeout_message = str(timeout_info.value)
    assert timeout_message, "the reconnect-timeout sink was never written to (A4)"
    _assert_message_sink_clean(
        "SourceTimeoutError message (io/_source.py:360)", timeout_message, expected=expected
    )


# All sinks reachable from `Frame.source_id`, keyed by name for the M2 refutation.
FRAME_DERIVED_SINKS: dict[str, Callable[[type, str, str], None]] = {
    "emitted frame identifier": _sink_emitted_frame_identifier,
    "postprocess result identifier": _sink_postprocess_result_identifier,
    "feature-cache dict key": _sink_feature_cache_dict_key,
    "feature-store key": _sink_feature_store_key,
}


# ---------------------------------------------------------------------------
# Driving the REAL pipeline log sink
# ---------------------------------------------------------------------------


class _ExplodingSource:
    """A real FrameSource whose read fails, so the pipeline records a stream error."""

    is_live = False
    total_frames = None

    def __iter__(self) -> Iterator[Frame]:
        raise RuntimeError("camera unreachable")
        yield  # pragma: no cover - unreachable, marks this a generator

    def close(self) -> None:
        return None


class _WorkingSource:
    """A real FrameSource that delivers frames, so the failure stays PARTIAL.

    Without a healthy stream alongside, `_check_stream_errors` takes the all-failed
    branch and raises instead of logging, and the log sink is never written to.
    """

    is_live = False
    total_frames = 2

    def __init__(self, stream_id: str) -> None:
        self._stream_id = stream_id

    def __iter__(self) -> Iterator[Frame]:
        for i in range(2):
            yield Frame(
                pixels=np.zeros((4, 4, 3), dtype=np.uint8),
                source_id=self._stream_id,
                frame_index=i,
            )

    def close(self) -> None:
        return None


def _make_engine() -> mock.MagicMock:
    engine = mock.MagicMock()
    engine._feature_cache = None
    engine._loaded = True
    engine.detect = mock.MagicMock(
        side_effect=lambda frames: [
            Detection(
                frame=f,
                boxes=(),
                inference_time_ms=1.0,
                backend=BackendType.PYTORCH,
                model_spec=_SPEC,
            )
            for f in frames
        ]
    )
    return engine


def _run_pipeline_with_failing_stream(
    stream_id: str, *, all_failed: bool = False
) -> list[logging.LogRecord]:
    """Run the REAL pipeline with `stream_id` as a failing stream; return what it logged.

    Returns every record the pipeline emitted, which is the log sink in full: the
    `_check_stream_errors` warning at pipeline/__init__.py:196, the collector's
    "Added stream" line, the bridge's consecutive-error warnings, and the router's
    dispatch lines. All of them receive the same identifier.
    """
    handler = _RecordingHandler()
    pipeline_root = logging.getLogger("yowo")
    previous_level = pipeline_root.level
    pipeline_root.addHandler(handler)
    pipeline_root.setLevel(logging.DEBUG)
    try:
        collector = FrameCollector(max_queue_size=4)
        collector.add_stream(stream_id, _ExplodingSource())
        if not all_failed:
            collector.add_stream("cam-healthy", _WorkingSource("cam-healthy"))

        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register(stream_id, lambda sid, dets: None)
        router.register("cam-healthy", lambda sid, dets: None)

        run_pipeline(_make_engine(), collector, scheduler, router, overlap=False)
    finally:
        pipeline_root.removeHandler(handler)
        pipeline_root.setLevel(previous_level)
    return handler.records


def _run_healthy_pipeline() -> list[logging.LogRecord]:
    """Run the REAL pipeline with no failing stream; return what it logged (E3)."""
    handler = _RecordingHandler()
    pipeline_root = logging.getLogger("yowo")
    previous_level = pipeline_root.level
    pipeline_root.addHandler(handler)
    pipeline_root.setLevel(logging.DEBUG)
    try:
        collector = FrameCollector(max_queue_size=4)
        collector.add_stream("cam-healthy", _WorkingSource("cam-healthy"))
        scheduler = BatchScheduler(collector, max_batch_size=1, timeout_ms=500.0)
        router = DetectionRouter()
        router.register("cam-healthy", lambda sid, dets: None)
        run_pipeline(_make_engine(), collector, scheduler, router, overlap=False)
    finally:
        pipeline_root.removeHandler(handler)
        pipeline_root.setLevel(previous_level)
    return handler.records


def _pipeline_failure_warnings(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    """The records pipeline/__init__.py:196 emits -- the log sink box 6 names."""
    return [
        r
        for r in records
        if r.levelno >= logging.WARNING and "failed during pipeline run" in r.getMessage()
    ]


class _RecordingHandler(logging.Handler):
    """Captures records AND forces formatting, so lazy %-args are really rendered."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        record.getMessage()
        self.records.append(record)


def _sink_pipeline_log(stream_id: str, expected: str) -> None:
    """Sink: every log record the real pipeline emitted for `stream_id`."""
    records = _run_pipeline_with_failing_stream(stream_id)
    assert records, "the log sink was never written to: the pipeline emitted no records (A4)"
    for record in records:
        message = record.getMessage()
        if PASSWORD in message or USER in message:
            raise AssertionError(
                f"{_LEAK_BANNER}: log record from {record.name} "
                f"({record.levelname}, {record.funcName})\n"
                f"  the sink received : {message!r}\n"
                f"  expected identifier: {expected!r}\n"
                f"  the raw URL        : {stream_id!r}\n"
                "  Fix at the boundary where the stream id is accepted, not at the logger (M5)."
            )
    # A4: the box-6 sink specifically -- pipeline/__init__.py:196 -- must have fired.
    # "some record somewhere named the stream" is NOT enough: the collector's own
    # "Added stream" and bridge-error lines satisfy that while the sink box 6 names is
    # never written to, which is exactly how a check passes for the wrong reason.
    assert _pipeline_failure_warnings(records), (
        "the log sink named by box 6 was never written to: no 'failed during pipeline run' "
        "warning was emitted (pipeline/__init__.py:196), so there was nothing to inspect for "
        f"a credential (A4). Records seen: {[r.getMessage() for r in records]!r}"
    )
    assert any(expected in record.getMessage() for record in records), (
        "no log record names the stream at all; a check that passes because nothing was "
        "logged about this stream proves nothing (A4)"
    )


# ---------------------------------------------------------------------------
# The mutation harness -- M2 executed rather than claimed.
# ---------------------------------------------------------------------------


@contextmanager
def _module_with(path: Path, old: str, new: str, label: str) -> Iterator[ModuleType]:
    """Import a copy of `path` with `old` replaced by `new`, exactly once.

    This builds a mutant and asserts on its BEHAVIOUR. It is not a source-text
    assertion (M3, R:SOURCESCAN): nothing here concludes anything from the text, and
    if the substitution stops applying the harness fails loudly rather than quietly
    re-running an unmutated copy.
    """
    text = path.read_text(encoding="utf-8")
    occurrences = text.count(old)
    assert occurrences == 1, (
        f"the {label} mutation no longer applies: {old!r} appears {occurrences} times in "
        f"{path.name}. The refutation is stale -- re-derive it before trusting any check here."
    )
    with tempfile.TemporaryDirectory() as tmp:
        mutant_path = Path(tmp) / f"mutant_{label}_{path.name}"
        mutant_path.write_text(text.replace(old, new), encoding="utf-8")
        name = f"_yowo_mutant_{label}"
        spec = importlib.util.spec_from_file_location(name, mutant_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(name, None)


def _refute(label: str, call: Callable[[], None]) -> None:
    """Require `call` -- a sink assertion run against a mutant -- to fail BECAUSE OF THE LEAK.

    Accepting any `AssertionError` would be the node's own defect class reappearing inside
    the check that carries M2: every liveness guard in this module raises `AssertionError`
    too, so a mutant that merely crashed or emitted nothing would score as "the credential
    reached the sink". The refutation only counts when the failure carries the leak banner.
    """
    try:
        call()
    except AssertionError as exc:
        if _LEAK_BANNER not in str(exc):
            raise AssertionError(
                f"INCONCLUSIVE REFUTATION: the '{label}' check failed against its mutant, but "
                f"not because the credential reached the sink:\n  {exc}\n"
                "A mutant that crashes, or whose sink was never written to, does not refute "
                "anything. Fix the harness so the mutant really exercises the sink (M2, A4)."
            ) from exc
        return
    raise AssertionError(
        f"NOT BINDING: the '{label}' check passed against a mutant that leaks the credential "
        "into its sink. A check that survives its own refutation proves nothing and must be "
        "rewritten, not excused (M2)."
    )


# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------


def test_the_log_record_the_pipeline_emitted_carries_no_credential() -> None:
    """covers: M1, M4, A3 -- read `record.getMessage()` from a real run_pipeline failure."""
    _sink_pipeline_log(URL, SAFE_URL)


def test_the_log_sink_was_actually_written_to() -> None:
    """covers: A4, E3 -- assert a record EXISTS before asserting what is in it.

    A pipeline run where nothing failed writes nothing to the log sink, and a credential
    check over an empty list of records passes for the wrong reason.
    """
    records = _run_pipeline_with_failing_stream(URL)
    assert records, "the pipeline emitted no log records at all"
    warnings = _pipeline_failure_warnings(records)
    assert warnings, (
        "pipeline/__init__.py:196 never fired: no 'failed during pipeline run' warning was "
        "emitted, so the log sink named by box 6 was never exercised. Records seen: "
        f"{[r.getMessage() for r in records]!r}"
    )

    # E3: the guard must DISCRIMINATE. A pipeline run where nothing fails never writes to
    # this sink, and a credential assertion over zero records would pass for the wrong
    # reason. Proving the predicate is empty on a healthy run is what makes the assertion
    # above a real guard rather than a tautology.
    healthy = _run_healthy_pipeline()
    assert healthy, "the healthy run emitted no records at all, so nothing was compared"
    assert not _pipeline_failure_warnings(healthy), (
        "a pipeline run with no failures still produced a 'failed during pipeline run' "
        "warning, so the guard cannot tell a written sink from an unwritten one (E3): "
        f"{[r.getMessage() for r in healthy]!r}"
    )


def test_the_all_streams_failed_error_carries_no_credential() -> None:
    """covers: M1, M4 -- pipeline/__init__.py:192 interpolates the id into a RuntimeError."""
    with pytest.raises(RuntimeError) as excinfo:
        _run_pipeline_with_failing_stream(URL, all_failed=True)
    message = str(excinfo.value)
    assert "All pipeline streams failed" in message, (
        f"the all-failed sink was never written to; got {message!r} (A4)"
    )
    _assert_message_sink_clean(
        "RuntimeError('All pipeline streams failed') (pipeline/__init__.py:192)",
        message,
        expected=SAFE_URL,
    )


def test_the_exception_that_propagates_carries_no_credential() -> None:
    """covers: M1, E2, A2 -- the SourceError caught from the real `_open_cap`."""
    _sink_propagated_exception(RTSPStreamSource, URL, SAFE_URL)


def test_the_emitted_frame_identifier_carries_no_credential() -> None:
    """covers: M1, M2, M4 -- `frame.source_id` on a Frame from the real read path."""
    _sink_emitted_frame_identifier(RTSPStreamSource, URL, SAFE_URL)


def test_the_identifier_on_a_postprocess_result_carries_no_credential() -> None:
    """covers: M1, M2 -- the id that survives into a real result object and its JSON."""
    _sink_postprocess_result_identifier(RTSPStreamSource, URL, SAFE_URL)


def test_the_key_the_feature_cache_stored_carries_no_credential() -> None:
    """covers: M1, M2, E4 -- read back out of `_last_fingerprints`, the dict key itself."""
    _sink_feature_cache_dict_key(RTSPStreamSource, URL, SAFE_URL)


def test_the_key_the_feature_store_stored_carries_no_credential() -> None:
    """covers: M1, E4 -- cache/__init__.py:152 stores the same id a second time."""
    _sink_feature_store_key(RTSPStreamSource, URL, SAFE_URL)


def test_the_mutation_that_leaks_turns_every_sink_check_red() -> None:
    """covers: M2 -- the refutation itself, executed.

    Each sink's module is rebuilt from source with its leak reinstated, and every sink
    assertion is re-run against the mutant. Each must fail. This is what makes M2 a
    measurement rather than a claim, and it is the exact measurement the ten checks in
    `test_rtsp_redaction.py` all survive.

    Three sinks are NOT reachable from `_source.py:346` and are refuted by their own
    mutation instead. The log sink and the all-streams-failed error take their identifier
    from `FrameCollector.add_stream`, an operator-supplied argument that never passes
    through `Frame.source_id`; the exception sink is `_source.py:322`. Asserting that
    `:346` reddens them would be asserting something false about the code.
    """
    source_path = Path(source_module.__file__)
    refuted: list[str] = []

    # E1 / M2 -- the node's named mutation, against every Frame-derived sink.
    with _module_with(
        source_path, "source_id=self._safe_url,", "source_id=self._url,", "m2_frame_id"
    ) as mutant:
        for label, sink_check in FRAME_DERIVED_SINKS.items():
            _refute(label, lambda c=sink_check: c(mutant.RTSPStreamSource, URL, SAFE_URL))
            refuted.append(label)
        # E6 and E7 ride the same read path and must be refuted by the same mutation.
        _refute(
            "username with no password at the frame sink",
            lambda: _sink_emitted_frame_identifier(
                mutant.RTSPStreamSource, URL_USER_ONLY, SAFE_URL_USER_ONLY
            ),
        )
        _refute(
            "credentialed http source at the frame sink",
            lambda: _sink_emitted_frame_identifier(mutant.RTSPStreamSource, URL_HTTP, SAFE_HTTP),
        )
        refuted += ["username-only frame sink", "http frame sink"]

    # E2 -- the exception sink binds to :322, not to :346.
    with _module_with(
        source_path,
        'raise SourceError(f"Cannot open RTSP stream: {self._safe_url}")',
        'raise SourceError(f"Cannot open RTSP stream: {self._url}")',
        "e2_exception",
    ) as mutant:
        _refute(
            "propagated exception",
            lambda: _sink_propagated_exception(mutant.RTSPStreamSource, URL, SAFE_URL),
        )
        refuted.append("propagated exception")

    # The log sink and the all-failed error take the operator's stream id. Bypassing the
    # M5 boundary is their refutation: with it neutered, both must go red.
    with (
        mock.patch("yowo.pipeline._collector.safe_stream_id", side_effect=lambda s: s),
        mock.patch("yowo.pipeline._router.safe_stream_id", side_effect=lambda s: s),
    ):
        _refute("pipeline log record", lambda: _sink_pipeline_log(URL, SAFE_URL))
        _refute(
            "all-streams-failed error",
            test_the_all_streams_failed_error_carries_no_credential,
        )
        refuted += ["pipeline log record", "all-streams-failed error"]

    assert len(refuted) == 9, f"expected 9 refuted sink assertions, refuted {refuted!r}"


def test_the_connectable_url_still_carries_the_credential() -> None:
    """covers: M6, E5 -- what `cv2.VideoCapture` ACTUALLY received.

    Guard: must pass throughout. Redacting the URL used to connect would break every
    RTSP stream in the field, so this is asserted on the call itself rather than on the
    private attribute that feeds it.
    """
    with _patched_capture() as cap_cls:
        list(RTSPStreamSource(URL, max_frames=1))
    assert cap_cls.call_args_list, "cv2.VideoCapture was never called (A4)"
    connected_with = [call.args[0] for call in cap_cls.call_args_list]
    assert all(url == URL for url in connected_with), (
        "the connectable URL lost its credential -- every RTSP stream would now fail to "
        f"authenticate. VideoCapture received: {connected_with!r}, expected {URL!r} (M6, E5)"
    )


def test_redaction_happens_once_at_the_boundary() -> None:
    """covers: M5 -- count `redact_url` calls across a full read, measured at the function.

    One boundary, not one per sink. If this count scales with the number of frames or the
    number of sinks, redaction has moved to the sinks and a sink added tomorrow will leak.
    """
    frame_count = 25
    with mock.patch(
        "yowo.io._source.redact_url", side_effect=source_module.redact_url
    ) as redactor, _patched_capture():
        frames = list(RTSPStreamSource(URL, max_frames=frame_count))

    assert len(frames) == frame_count, f"the read path emitted {len(frames)} frames (A4)"
    assert redactor.call_count == 1, (
        f"redact_url ran {redactor.call_count} times across a {frame_count}-frame read. "
        "The boundary must redact exactly once, at construction; a per-frame or per-sink "
        "count means each sink is redacting for itself and the next sink added will not (M5)."
    )
    assert redactor.call_args_list[0].args[0] == URL, (
        "the boundary did not redact the URL it was given"
    )
    assert {f.source_id for f in frames} == {SAFE_URL}, (
        "not every frame in the read carried the single redacted identifier"
    )

    # The pipeline is a SECOND boundary, over a different input: the identifier the
    # operator handed `add_stream`, which never passes through `Frame.source_id`. It must
    # redact once per entry too, or a per-frame call would creep in unnoticed.
    with mock.patch(
        "yowo.pipeline._collector.safe_stream_id",
        side_effect=source_module.redact_url,
    ) as boundary:
        _run_pipeline_with_failing_stream(URL)
    add_and_remove = 2  # add_stream on entry, remove_stream on auto-removal
    assert boundary.call_count <= add_and_remove * 2, (
        f"the pipeline boundary ran {boundary.call_count} times for two streams. It must "
        "normalise once per entry point call, not once per frame or per log line (M5)."
    )

    # M5's other half: a boundary that redacts is worthless if it also mangles. An
    # identifier that cannot carry a credential must survive it byte for byte, because
    # the id is a KEY -- collapsing two of them silently merges two cameras.
    from yowo.pipeline._ids import safe_stream_id

    for benign in ("cam-0", "/videos/a.mp4", "0", "rtsp://10.0.0.5:554/s", "cam@site1"):
        assert safe_stream_id(benign) == benign, (
            f"the boundary rewrote {benign!r}, which carries no credential. Two ids that "
            "collapse to one string become one stream, and DetectionRouter.register "
            "overwrites silently rather than raising (M5)."
        )
    for malformed_but_clean in ("rtsp://host:abc/path", "rtsp://10.0.0.5:99999/s"):
        assert safe_stream_id(malformed_but_clean) == malformed_but_clean, (
            f"{malformed_but_clean!r} carries no credential but was collapsed to a fixed "
            "string. `redact_url` reaches its parse-failure path on an unparseable port, "
            "so the boundary must short-circuit before it when there is no '@' at all."
        )


def test_a_username_with_no_password_is_redacted_at_every_sink() -> None:
    """covers: E6 -- half a credential is a credential, asserted AT the sinks."""
    for sink_check in FRAME_DERIVED_SINKS.values():
        sink_check(RTSPStreamSource, URL_USER_ONLY, SAFE_URL_USER_ONLY)
    _sink_propagated_exception(RTSPStreamSource, URL_USER_ONLY, SAFE_URL_USER_ONLY)
    _sink_pipeline_log(URL_USER_ONLY, SAFE_URL_USER_ONLY)


def test_a_malformed_url_leaks_nothing_at_any_sink() -> None:
    """covers: E6, A4 -- the parse-failure path, at the sinks.

    A URL that cannot be parsed is exactly where a leak hides: the natural fallback is
    to emit the raw string. Every sink must receive the placeholder instead.
    """
    for sink_check in FRAME_DERIVED_SINKS.values():
        sink_check(RTSPStreamSource, URL_MALFORMED, SAFE_MALFORMED)
    _sink_propagated_exception(RTSPStreamSource, URL_MALFORMED, SAFE_MALFORMED)
    _sink_pipeline_log(URL_MALFORMED, SAFE_MALFORMED)


def test_a_credentialed_http_source_is_redacted_at_every_sink() -> None:
    """covers: E7 -- `redact_url` is scheme-agnostic and the sinks are not RTSP-specific."""
    for sink_check in FRAME_DERIVED_SINKS.values():
        sink_check(RTSPStreamSource, URL_HTTP, SAFE_HTTP)
    _sink_propagated_exception(RTSPStreamSource, URL_HTTP, SAFE_HTTP)
    _sink_pipeline_log(URL_HTTP, SAFE_HTTP)


def test_the_parent_redaction_checks_are_untouched() -> None:
    """covers: M7, A7, E8 -- `test_rtsp_redaction.py` is byte-identical to its gated commit.

    Its ten checks are weak but none is wrong, and they belong to a frozen, gated node.
    "The weak check was in the way" is how a weak check becomes no check. If these ten
    should later go, that is a change-request against `rtsp-credential-redaction`,
    carrying this node's mutation evidence -- not an edit from here.
    """
    assert PARENT_CHECKS.is_file(), f"{PARENT_CHECKS} is missing entirely"
    digest = hashlib.sha256(PARENT_CHECKS.read_bytes()).hexdigest()
    assert digest == PARENT_CHECKS_SHA256, (
        f"{PARENT_CHECKS.name} changed.\n"
        f"  expected sha256: {PARENT_CHECKS_SHA256}\n"
        f"  actual sha256  : {digest}\n"
        "It belongs to the frozen, gated node `rtsp-credential-redaction` (M7, A7, E8). "
        "Editing it from this node is out of scope; raise a change-request instead."
    )
    # Byte-identity would also be satisfied by an empty file at a matching digest, so
    # pin that the ten checks are still there and still named.
    body = PARENT_CHECKS.read_text(encoding="utf-8")
    names = re.findall(r"^def (test_\w+)", body, flags=re.MULTILINE)
    assert len(names) == 10, f"expected the parent node's ten checks, found {names!r}"
