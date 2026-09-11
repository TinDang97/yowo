"""When a backend fails after load, every engine must still produce a result.

`BaseEngine._infer_with_retry` retries three times and, on exhaustion, returns
`np.zeros((1, 0, 6))` — described in its own docstring as "an empty result array
so the engine does not crash". Measured against a backend that loads and warms
up cleanly and then fails every inference, it crashed on all three engines:

    classification   ValueError: Expected 2-D output (batch, nc), got ndim=3
    obb              IndexError: max(): Expected reduction dim 1 to have
                                 non-zero size
    detection (N>1)  IndexError: index 1 is out of bounds for axis 0 with size 1

The first two because the sentinel is DETECTION-shaped and the other engines
inherit it unchanged; the third because it is BATCH-1-shaped. Detection looked
healthy only because every probe used a single frame.

The obvious fix for classification would fabricate a prediction: a correctly
shaped `(batch, nc)` zeros sentinel IS accepted by `postprocess_classify`, and
returns `top1_class_id=999, top1_score=0.001` — uniform softmax, argmax taking
the last index. An operator reading a degraded stream would see the model
confidently reporting that nothing is wrong. So a degraded classification is
marked `-1 / 0.0` instead, and that marker is unreachable from a healthy
inference.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from yowo.classify_engine import ClassificationEngine
from yowo.engine import BaseEngine, InferenceEngine
from yowo.io._decode import preprocess as _decode_preprocess
from yowo.obb_engine import OBBEngine
from yowo.postprocess import postprocess_classify
from yowo.types import BackendType, Frame, ModelFamily, ModelSize, ModelSpec

# Raw shapes a HEALTHY backend produces, per engine. Used to get past load and
# warmup validation so the test can reach the failure path at all.
_HEALTHY_DETECTION = (84, 8400)
_HEALTHY_CLASSIFY = (1000,)
_HEALTHY_OBB = (20, 8400)


class FailsAfterLoad:
    """Loads and warms up cleanly, then fails every inference.

    An always-broken backend never reaches the sentinel: `load()` runs warmup
    validation, which calls `infer()` and raises `WarmupValidationError` first.
    The failure therefore has to be armed after `load()` returns — which is also
    the case box 8 is about, a backend that dies mid-stream rather than one that
    was never usable.
    """

    backend_type = BackendType.PYTORCH

    def __init__(self, per_frame_shape: tuple[int, ...]) -> None:
        self._per_frame_shape = per_frame_shape
        self.armed = False
        self.calls = 0

    def load(self, weights_path: Any, device: str = "auto") -> None:
        return None

    def warmup(self, batch_size: int = 1) -> None:
        return None

    def infer(self, tensor: Any) -> NDArray[np.float32]:
        self.calls += 1
        if self.armed:
            raise RuntimeError("backend exploded mid-stream")
        batch = 1
        data = getattr(tensor, "data", None)
        if data is not None and hasattr(data, "shape"):
            batch = int(data.shape[0])
        out = np.zeros((batch, *self._per_frame_shape), dtype=np.float32)
        if out.ndim == 2:
            # A plausible softmax, so classification warmup validation passes.
            out[:, 0] = 1.0
        return out

    def close(self) -> None:
        return None

    @property
    def is_loaded(self) -> bool:
        return True


class RecoversOnRetry(FailsAfterLoad):
    """Fails the first armed inference, then succeeds — the retry must win."""

    def infer(self, tensor: Any) -> NDArray[np.float32]:
        if self.armed:
            self.armed = False
            self.calls += 1
            raise RuntimeError("transient")
        return super().infer(tensor)


_ENGINES = {
    "detection": (InferenceEngine, "detect", _HEALTHY_DETECTION),
    "classification": (ClassificationEngine, "classify", _HEALTHY_CLASSIFY),
    "obb": (OBBEngine, "detect_obb", _HEALTHY_OBB),
}
_ENGINE_IDS = list(_ENGINES)


def _preprocess(frames: list[Frame]) -> Any:
    """The tensor metadata postprocess needs — letterbox scale factors per frame."""
    return _decode_preprocess(frames, (640, 640))


def _frames(n: int) -> list[Frame]:
    return [
        Frame(pixels=np.zeros((640, 640, 3), dtype=np.uint8), source_id="s", frame_index=i)
        for i in range(n)
    ]


# Every engine these tests build owns an event-bus worker thread. Leaving them
# open leaked threads into the rest of the suite and broke a thread-counting
# check in test_events.py — so each one is closed in teardown.
_OPEN: list[Any] = []


@pytest.fixture(autouse=True)
def _close_engines() -> Any:
    yield
    while _OPEN:
        engine = _OPEN.pop()
        # Teardown must not mask the failure the test is reporting.
        with contextlib.suppress(Exception):
            engine.close()


def _track(engine: Any) -> Any:
    _OPEN.append(engine)
    return engine


def _loaded(kind: str, batch_size: int = 1, backend_cls: type = FailsAfterLoad) -> Any:
    engine_cls, method, shape = _ENGINES[kind]
    backend = backend_cls(shape)
    engine = _track(engine_cls(backend_instance=backend, batch_size=batch_size))
    engine.load()
    return engine, backend, method


# ---------------------------------------------------------------------------
# M1/M2 — the sentinel fits the engine and the batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _ENGINE_IDS, ids=_ENGINE_IDS)
def test_each_engine_produces_a_sentinel_its_own_postprocess_accepts(kind: str) -> None:
    """covers: M1, M4, A1, A2, A6 — the sentinel goes through the real postprocess."""
    engine, backend, method = _loaded(kind)
    backend.armed = True
    try:
        results = getattr(engine, method)(_frames(1))
    except Exception as exc:  # pragma: no cover - the defect this check exists for
        pytest.fail(
            f"{kind}: a failed inference raised {type(exc).__name__}: {exc}. "
            "The sentinel is supposed to be an empty result the engine survives; "
            "a traceback here sends an operator into postprocess instead of at the "
            "backend that actually failed."
        )
    assert len(results) == 1, f"{kind}: expected one result for one frame, got {len(results)}"


@pytest.mark.parametrize("kind", _ENGINE_IDS, ids=_ENGINE_IDS)
def test_a_failed_batch_yields_one_result_per_frame(kind: str) -> None:
    """covers: M2, A5, E1, R:BATCHDROP — N frames in, N results out, in frame order."""
    engine, backend, method = _loaded(kind, batch_size=3)
    backend.armed = True
    frames = _frames(3)
    try:
        results = getattr(engine, method)(frames)
    except Exception as exc:  # pragma: no cover - the defect this check exists for
        pytest.fail(
            f"{kind}: a failed batch of 3 raised {type(exc).__name__}: {exc}. "
            "The sentinel is hardcoded to batch 1, so every batch larger than one "
            "frame crashes in postprocess."
        )
    assert len(results) == 3, (
        f"{kind}: 3 frames were inferred and {len(results)} result(s) came back. "
        "A dropped frame is indistinguishable downstream from one never sent."
    )
    indices = [_frame_index_of(r) for r in results]
    assert indices == [0, 1, 2], (
        f"{kind}: results are out of frame order ({indices}); the router keys on position, "
        "so this misattributes results across sources"
    )


def _frame_index_of(result: Any) -> int:
    frame = getattr(result, "frame", None)
    if frame is not None:
        return int(frame.frame_index)
    return int(result.frame_index)


def test_a_failed_single_frame_batch_still_yields_one_result() -> None:
    """covers: E2 — the batch size that masked the bug for detection."""
    engine, backend, method = _loaded("detection", batch_size=1)
    backend.armed = True
    results = getattr(engine, method)(_frames(1))
    assert len(results) == 1
    assert len(results[0].boxes) == 0, "a failed inference must not produce boxes"


def test_the_degraded_path_is_not_a_postprocess_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    """covers: M4, R:BYPASS — the failure path runs the same postprocess the healthy one does."""
    engine, backend, method = _loaded("detection")
    seen: list[int] = []
    original = type(engine)._process_batch

    def spy(self: Any, raw: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(int(raw.shape[0]))
        return original(self, raw, *args, **kwargs)

    monkeypatch.setattr(type(engine), "_process_batch", spy)
    backend.armed = True
    getattr(engine, method)(_frames(1))
    assert seen, (
        "the degraded path did not reach _process_batch, so the failure path and the "
        "healthy path share no code and the failure path is exercised by nothing"
    )


# ---------------------------------------------------------------------------
# M3 — a degraded classification is not a prediction
# ---------------------------------------------------------------------------


def test_a_degraded_classification_is_not_a_prediction() -> None:
    """covers: M3, A8, A12, R:FABRICATE — not the measured 999 @ 0.001."""
    engine, backend, method = _loaded("classification")
    backend.armed = True
    result = getattr(engine, method)(_frames(1))[0]
    assert result.top1_class_id == -1, (
        f"a degraded classification reports class {result.top1_class_id} at "
        f"{result.top1_score}. Measured before this fix: 999 at 0.001 — a uniform softmax "
        "whose argmax takes the last index, indistinguishable from a real low-confidence "
        "prediction, so a threshold filter hides the outage instead of revealing it."
    )
    assert result.top1_score == 0.0, (
        f"a degraded classification reports score {result.top1_score}; 0.0 is unreachable "
        "through softmax, which is what makes the marker unambiguous"
    )


def test_the_degraded_marker_is_unreachable_from_a_healthy_inference() -> None:
    """covers: A8, A12, R:FABRICATE — a real inference can produce neither value."""
    engine, backend, method = _loaded("classification")
    result = getattr(engine, method)(_frames(1))[0]
    assert result.top1_class_id != -1, "a healthy inference produced the degraded marker"
    assert result.top1_score > 0.0, (
        "a healthy inference produced score 0.0, so the degraded marker is ambiguous"
    )


def test_a_degraded_classification_ranks_nothing() -> None:
    """covers: A9, E3 — no fabricated ranking to iterate into."""
    engine, backend, method = _loaded("classification")
    backend.armed = True
    result = getattr(engine, method)(_frames(1))[0]
    assert tuple(result.topk_class_ids) == (), (
        f"a degraded classification offers a ranking: {result.topk_class_ids}. "
        "A consumer looping topk would act on fabricated classes."
    )
    assert tuple(result.topk_scores) == ()
    assert not any(result.all_probs), "a degraded classification carries a non-zero probability"


def test_classification_result_documents_the_degraded_marker() -> None:
    """covers: M3 — a consumer reading the type must learn what -1 means.

    The marker is only useful if it is discoverable from the type a caller
    holds. Documenting it in postprocess helps whoever reads postprocess; the
    person deserialising a result reads `ClassificationResult`.
    """
    from yowo.types import ClassificationResult

    doc = ClassificationResult.__doc__ or ""
    assert "-1" in doc, (
        "ClassificationResult does not document the -1 marker, so a consumer meets it "
        "first as a class index that does not exist and has nowhere to look it up"
    )
    assert "NO PREDICTION" in doc.upper(), (
        "ClassificationResult documents -1 without saying it means no prediction"
    )


def test_postprocess_classify_alone_returns_the_marker() -> None:
    """covers: A10, E7 — the marker is a property of postprocess, not of engine state.

    A result deserialised from a payload carries no engine and no log. If the
    marker depended on engine state it would not survive that trip.
    """
    spec = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
    sentinel = np.zeros((1, 1000), dtype=np.float32)
    results = postprocess_classify(
        sentinel, _frames(1), model_spec=spec, backend=BackendType.PYTORCH
    )
    assert results[0].top1_class_id == -1
    assert results[0].top1_score == 0.0


# ---------------------------------------------------------------------------
# A4/E4, A3/E6, M5 — emptiness, recovery, observability
# ---------------------------------------------------------------------------


def test_the_obb_sentinel_is_empty_structurally_not_by_threshold() -> None:
    """covers: A4, E4 — zero ANCHORS, not one zero-filled anchor.

    A zero-filled anchor also yields no boxes at any ordinary threshold, because
    the class-score filter is a strict ``>`` and the score is 0.0. That makes the
    two sentinels indistinguishable at a threshold of 0.0 — measured, a check
    written that way passed with either one. The difference only shows below
    zero, where a zero-scored box survives the filter: the one-anchor form emits
    a phantom box and the zero-anchor form cannot, because it carries no
    instance to emit.
    """
    import torch

    from yowo.postprocess import postprocess_obb

    engine, _backend, _method = _loaded("obb")
    sentinel = engine._empty_raw_output(1)
    assert sentinel.shape[-1] == 0, (
        f"the OBB sentinel carries {sentinel.shape[-1]} anchor(s). Emptiness that depends "
        "on a score falling under a threshold is not emptiness; drop the threshold and it "
        "emits a phantom box."
    )

    spec = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)
    frames = _frames(1)
    tensor = _preprocess(frames)

    empty = postprocess_obb(
        torch.from_numpy(np.asarray(sentinel)), frames, spec, tensor, conf_threshold=-1.0
    )
    assert len(empty[0].boxes) == 0, (
        "the OBB sentinel produced a box once the threshold stopped hiding it"
    )

    # The control: the sentinel this rejects DOES emit a phantom box here, so the
    # assertion above is discriminating rather than trivially true.
    one_anchor = torch.zeros((1, sentinel.shape[1], 1), dtype=torch.float32)
    phantom = postprocess_obb(one_anchor, frames, spec, tensor, conf_threshold=-1.0)
    assert len(phantom[0].boxes) == 1, (
        "the one-zero-anchor control emitted no box either, so this check cannot tell the "
        "two sentinel shapes apart and proves nothing about the one in use"
    )


def test_a_recovered_retry_returns_the_real_output() -> None:
    """covers: A3, E6 — a transient failure must not become a degraded result."""
    engine, backend, method = _loaded("classification", backend_cls=RecoversOnRetry)
    backend.armed = True
    result = getattr(engine, method)(_frames(1))[0]
    assert result.top1_class_id != -1, (
        "a backend that failed once and then succeeded was reported as degraded; "
        "the retry is supposed to win before the sentinel is reached"
    )


def test_the_failure_is_still_counted_and_emitted() -> None:
    """covers: M5, R:SILENT — a valid empty result must not become a silent one."""
    engine, backend, method = _loaded("detection")
    errors: list[object] = []
    engine.on("error", errors.append)
    backend.armed = True
    getattr(engine, method)(_frames(1))
    # The bus delivers on a worker thread, so the listener has to be waited for.
    # Asserting immediately tests the scheduler, not the emit.
    deadline = time.monotonic() + 2.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert engine.metrics.errors_total >= 1, (
        "the failed inference was not counted; a degraded result that increments nothing "
        "is indistinguishable from a healthy empty one"
    )
    assert errors, "the 'error' event did not fire for a failed inference"


@pytest.mark.parametrize("kind", _ENGINE_IDS, ids=_ENGINE_IDS)
def test_a_healthy_inference_is_unchanged(kind: str) -> None:
    """covers: M7, E5 — pinned so the fix cannot hide a behaviour change."""
    engine, backend, method = _loaded(kind, batch_size=2)
    results = getattr(engine, method)(_frames(2))
    assert len(results) == 2, f"{kind}: a healthy batch of 2 produced {len(results)} result(s)"
    assert backend.calls >= 1, f"{kind}: the backend was never called on the healthy path"


# ---------------------------------------------------------------------------
# M6 — box 8 says what the sentinel establishes
# ---------------------------------------------------------------------------

_MILESTONE = Path(__file__).resolve().parents[2] / ".add" / "milestones" / "m2-survive-week-two.md"
_BOX_8 = "identifiable as degraded"


def _box_8() -> str:
    for line in _MILESTONE.read_text(encoding="utf-8").splitlines():
        if _BOX_8 in line:
            return line
    raise AssertionError(f"m2 box 8 not found by marker {_BOX_8!r}")


def test_box_8_records_that_it_was_amended() -> None:
    """covers: M6, A16, R:SILENTREWRITE, E8 — an undated rewrite reads as the original."""
    import re

    box = _box_8()
    assert re.search(r"AMENDED \d{4}-\d{2}-\d{2}", box), "box 8's amendment carries no date"
    assert "by human decision" in box, "box 8's amendment records no authority"


def test_box_8_claims_identifiable_not_merely_accepted() -> None:
    """covers: M6, A15, A18, E8 — "accepted" alone permits the fabricating sentinel."""
    box = _box_8()
    claim = box.partition("AMENDED")[0]
    assert "identifiable" in claim, (
        "box 8 claims only that postprocess accepts the output. Measured, a (batch, nc) "
        "zeros sentinel IS accepted and returns class 999 at 0.001 — the exact output "
        "R:FABRICATE forbids. The box must require the result be identifiable too."
    )
    for engine in ("detection", "classification", "OBB"):
        assert engine in box, f"box 8 no longer names {engine}"


def test_box_8_claims_nothing_a_probe_refutes() -> None:
    """covers: M6 — every symbol the box names is resolved and driven."""
    box = _box_8()
    assert "`top1_class_id`" in box and "`-1`" in box, (
        "box 8 must name the marker it claims, so the claim is traceable to a value"
    )
    from yowo.types import ClassificationResult

    assert "top1_class_id" in ClassificationResult.__dataclass_fields__
    assert hasattr(BaseEngine, "_empty_raw_output"), (
        "box 8 rests on a per-engine sentinel hook; BaseEngine does not define one"
    )
    assert "`_empty_raw_output`" in box, "box 8 does not name the hook the fix introduces"
