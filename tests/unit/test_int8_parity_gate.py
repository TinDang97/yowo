"""An INT8 artifact either clears its declared floor, or it is never written.

The defect this closes: ``quantize_onnx_static`` wrote whatever ORT produced.
Measured 2026-09-15, a production-path ``yolo11n`` INT8 export shipped a 2.9 MB
artifact that detects NOTHING — every class logit exactly 0.0 — and the export
reported success. An artifact that lies is worse than one that crashes, because
the crash is caught and the lie is deployed.

Three things are proved here: the SCORING (what counts as recovered, what counts
as gated, what a 0/0 ratio means), the REFUSALS (a graph the gate cannot measure
is never quantized), and the WRITE LIFECYCLE (nothing at ``output_path`` until
the floor is cleared, nothing left behind when it is not).

None of it proves an export works — a mocked exporter cannot (Q12). That proof
is ``tests/integration/test_int8_parity.py``, which drives the real export and
the real quantizer end to end.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from yowo.errors import ExportError
from yowo.types import BoundingBox, ModelFamily, ModelSize, ModelSpec

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

_SPEC = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _box(conf: float, cls: int = 0, x: float = 0.0, size: float = 10.0) -> BoundingBox:
    """A square box at ``x`` with a NON-DEFAULT confidence and class (Q4)."""
    return BoundingBox(x1=x, y1=0.0, x2=x + size, y2=size, confidence=conf, class_id=cls)


def _score(fp32: list[BoundingBox], int8: list[BoundingBox], **kw: Any) -> Any:
    from yowo.export._int8 import _score_parity

    params: dict[str, Any] = {
        "floor": 1.0,
        "margin": 2.0,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "images": ("bus.jpg",),
        "output_path": "/out/model_int8.onnx",
        "held_out": False,
        "parity_set_source": "calibration",
        "provider": "CPUExecutionProvider",
    }
    params.update(kw)
    return _score_parity([fp32], [int8], **params)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_a_recovered_strong_detection_passes() -> None:
    """Same class, overlapping box, drifted confidence — still a match."""
    report = _score([_box(0.94)], [_box(0.88, x=0.5)])
    assert report.passed is True
    assert report.gated_recall == 1.0
    assert report.total_recall == 1.0


def test_the_marginal_band_is_reported_not_gated() -> None:
    """FP32's weakest detection (0.3977 measured) must not decide the gate.

    Quantization confidence drift is measured at 0.05-0.15 — the same order as
    that detection's own margin above the 0.25 threshold — so gating on it
    makes the floor a coin flip on scene content. It is REPORTED instead.
    """
    report = _score([_box(0.94), _box(0.3977, x=100.0)], [_box(0.88, x=0.5)])
    assert report.gated_detections == 1
    assert report.gated_recall == 1.0
    assert report.passed is True
    assert report.total_recall == 0.5, "the loss is reported, not defined away"
    assert [(m.class_id, m.confidence) for m in report.missed] == [(0, 0.3977)]


def test_a_lost_strong_detection_fails() -> None:
    report = _score([_box(0.94), _box(0.82, x=100.0)], [_box(0.88, x=0.5)])
    assert report.passed is False
    assert report.gated_recall == 0.5


def test_the_gated_band_boundary_is_inclusive() -> None:
    """A detection exactly at margin * threshold is GATED, not reported."""
    report = _score([_box(0.50)], [])
    assert report.gated_detections == 1
    assert report.passed is False, "0.50 == 2.0 * 0.25 is inside the gate"


def test_a_detection_just_below_the_boundary_is_not_gated() -> None:
    """The other side of the same boundary: 0.4999 is reported, not gated."""
    report = _score([_box(0.94), _box(0.4999, x=100.0)], [_box(0.88, x=0.5)])
    assert report.gated_detections == 1
    assert report.passed is True
    assert report.total_recall == 0.5


def test_one_int8_box_cannot_match_two_fp32_detections() -> None:
    """Greedy matching consumes each INT8 box once, or recall inflates."""
    report = _score([_box(0.94), _box(0.90, x=1.0)], [_box(0.88, x=0.5)])
    assert report.gated_detections == 2
    assert report.gated_matched == 1
    assert report.passed is False


def test_matching_is_deterministic_by_descending_confidence() -> None:
    """The highest-confidence FP32 detection claims the best box first.

    Without a stated order the same inputs give different answers on different
    runs. The 0.94 box overlaps the INT8 box more, so it takes it; the 0.60 one
    is left unmatched regardless of input order.
    """
    strong, weak = _box(0.94, x=0.0), _box(0.60, x=6.0)
    a = _score([strong, weak], [_box(0.88, x=0.5)], floor=0.0)
    b = _score([weak, strong], [_box(0.88, x=0.5)], floor=0.0)
    assert [(m.class_id, m.confidence) for m in a.missed] == [(0, 0.60)]
    assert [(m.class_id, m.confidence) for m in b.missed] == [(0, 0.60)]


def test_a_different_class_is_not_a_match() -> None:
    """Right place, wrong label — an INT8 artifact that relabels is not parity."""
    report = _score([_box(0.94, cls=5)], [_box(0.88, cls=0, x=0.5)])
    assert report.passed is False
    assert report.gated_matched == 0


def test_a_non_overlapping_box_is_not_a_match() -> None:
    report = _score([_box(0.94)], [_box(0.88, x=500.0)])
    assert report.passed is False


def test_no_gated_fp32_detection_raises() -> None:
    """A 0/0 ratio proves nothing and must never be reported as 1.0.

    Without this, a parity set on which FP32 happens to detect nothing — a
    blank frame, a bad threshold — waves every artifact through, including one
    that detects nothing at all.
    """
    with pytest.raises(ExportError, match="no gated"):
        _score([], [], images=("blank.jpg",))


def test_a_waived_export_still_carries_its_numbers() -> None:
    """floor=0.0 lowers enforcement; it never skips the measurement."""
    report = _score([_box(0.94), _box(0.82, x=100.0)], [_box(0.88, x=0.5)], floor=0.0)
    assert report.passed is True
    assert report.enforced is False
    assert report.floor == 0.0
    assert report.gated_recall == 0.5
    assert report.total_recall == 0.5
    assert len(report.missed) == 1


def test_report_names_the_detections_it_lost() -> None:
    """ "4/5" cannot tell you whether the loss was marginal or confident."""
    report = _score([_box(0.94), _box(0.61, cls=7, x=100.0)], [_box(0.88, x=0.5)], floor=0.0)
    lost = [(m.class_id, round(m.confidence, 4), m.image) for m in report.missed]
    assert lost == [(7, 0.61, "bus.jpg")]


def test_the_report_counts_the_int8_boxes_nobody_claimed() -> None:
    """Recall alone cannot see a flood of false positives.

    An INT8 artifact whose activation scales saturate the classification
    sigmoid emits spurious boxes everywhere. Every FP32 detection then finds a
    same-class high-IoU partner in the flood, recall reads 1.0, and the report
    would say "perfect" about a model that hallucinates. The unmatched count is
    REPORTED, never gated — this node's claim is collapse, not precision — but
    it is on the artifact so nobody has to infer it from the recall.
    """
    report = _score(
        [_box(0.94)],
        [_box(0.88, x=0.5), _box(0.31, x=200.0), _box(0.29, cls=7, x=400.0)],
    )
    assert report.gated_recall == 1.0, "recall alone says this artifact is perfect"
    assert report.int8_detections == 3
    assert report.int8_unmatched == 2, "and two of its boxes answer to no FP32 detection"


def test_no_false_positives_means_nothing_unmatched() -> None:
    report = _score([_box(0.94)], [_box(0.88, x=0.5)])
    assert report.int8_unmatched == 0


def test_the_report_stamps_whether_the_parity_set_was_held_out() -> None:
    """The artifact states the limit of its own claim, not an ASSUMPTIONS line."""
    report = _score([_box(0.94)], [_box(0.88, x=0.5)])
    assert report.held_out is False
    assert report.parity_set_source == "calibration"
    assert report.provider == "CPUExecutionProvider"


# ---------------------------------------------------------------------------
# The declared floor
# ---------------------------------------------------------------------------


def test_floor_parameters_are_declared_with_documented_defaults() -> None:
    """The floor is a parameter, not a constant somebody has to read source to find."""
    from yowo.export._int8 import quantize_onnx_static

    params = inspect.signature(quantize_onnx_static).parameters
    assert params["parity_floor"].default == 1.0
    assert params["parity_margin"].default == 2.0
    assert params["parity_threshold"].default == 0.25
    assert params["parity_iou"].default == 0.5
    assert params["parity_images"].default is None
    assert params["parity_sample"].default == 8

    doc = quantize_onnx_static.__doc__ or ""
    for name in (
        "parity_floor",
        "parity_margin",
        "parity_threshold",
        "parity_iou",
        "parity_images",
        "parity_sample",
    ):
        assert name in doc, f"{name} is a declared parameter and must be documented"


# ---------------------------------------------------------------------------
# Fixtures for the write lifecycle
# ---------------------------------------------------------------------------


def _mock_ort(crash: bool = False) -> Any:
    """An onnxruntime.quantization stand-in whose quantize_static really writes."""
    mod = MagicMock()
    mod.QuantFormat.QDQ = "QDQ"
    mod.QuantType.QInt8 = "QInt8"
    mod.CalibrationMethod.Entropy = "Entropy"
    mod.CalibrationDataReader = type("CalibrationDataReader", (), {})

    def _run(src: str, dst: str, *a: Any, **kw: Any) -> None:
        if crash:
            raise RuntimeError("ORT died mid-write")
        Path(dst).write_bytes(b"quantized")

    mod.quantize_static.side_effect = _run
    return mod


def _patched_ort(mod: Any) -> Any:
    ort = types.ModuleType("onnxruntime")
    ort.quantization = mod  # type: ignore[attr-defined]
    return patch.dict(sys.modules, {"onnxruntime": ort, "onnxruntime.quantization": mod})


def _cal_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cal"
    d.mkdir(exist_ok=True)
    for i in range(12):
        cv2.imwrite(str(d / f"c{i:02d}.jpg"), np.full((60, 90, 3), i * 9 % 256, dtype=np.uint8))
    return d


def _save_model(path: Path, nodes: list[Any], inputs: list[str], outputs: list[str]) -> Path:
    g = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(i, TensorProto.FLOAT, None) for i in inputs],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outputs],
    )
    onnx.save(helper.make_model(g), str(path))
    return path


def _source(tmp_path: Path) -> Path:
    """A single-input detection-shaped FP32 source."""
    return _save_model(
        tmp_path / "model.onnx",
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Sigmoid", ["h"], ["output0"], name="sigmoid"),
        ],
        ["images"],
        ["output0"],
    )


def _kv_source(tmp_path: Path) -> Path:
    """A KV-cache export: extra inputs the gate's single-input feed cannot satisfy.

    Measured 2026-09-15: a real KV export's inputs are
    ``['images', 'use_cache', 'past_k_0', 'past_v_0']``.
    """
    return _save_model(
        tmp_path / "model.onnx",
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Sigmoid", ["h"], ["output0"], name="sigmoid"),
            helper.make_node("Mul", ["past_k_0", "use_cache"], ["present_k_0"], name="kv"),
        ],
        ["images", "use_cache", "past_k_0"],
        ["output0", "present_k_0"],
    )


def _leftovers(out: Path) -> list[Path]:
    """Every file in the output dir that is neither the source nor a calibration image."""
    return sorted(
        p for p in out.parent.iterdir() if p.is_file() and p.name not in ("model.onnx", ".DS_Store")
    )


def _passing_report(tmp_path: Path) -> Any:
    from yowo.export._int8 import ParityReport

    return ParityReport(
        passed=True,
        enforced=True,
        output_path=str(tmp_path / "model_int8.onnx"),
        total_recall=0.8,
        gated_recall=1.0,
        fp32_detections=5,
        int8_detections=4,
        gated_detections=4,
        gated_matched=4,
        int8_unmatched=0,
        missed=(),
        floor=1.0,
        margin=2.0,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        held_out=False,
        parity_set_source="calibration",
        provider="CPUExecutionProvider",
        images=("bus.jpg",),
    )


def _failing_report(tmp_path: Path) -> Any:
    from yowo.export._int8 import MissedDetection, ParityReport

    return ParityReport(
        passed=False,
        enforced=True,
        output_path=str(tmp_path / "model_int8.onnx"),
        # DISTINCT on purpose: with both at 0.0 a message printing only gated
        # recall would satisfy every assertion below (Q4 — a fixture built from
        # equal values cannot detect a dropped field).
        total_recall=0.4,
        gated_recall=0.25,
        fp32_detections=5,
        int8_detections=2,
        gated_detections=4,
        gated_matched=1,
        int8_unmatched=0,
        missed=(MissedDetection(class_id=5, confidence=0.9402, image="bus.jpg"),),
        floor=1.0,
        margin=2.0,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        held_out=False,
        parity_set_source="calibration",
        provider="CPUExecutionProvider",
        images=("bus.jpg",),
    )


def _quantize(tmp_path: Path, report: Any, source: Path | None = None, **kw: Any) -> Any:
    from yowo.export import _int8

    out = tmp_path / "model_int8.onnx"
    with (
        _patched_ort(_mock_ort()),
        patch.object(_int8, "measure_int8_parity", return_value=report),
    ):
        return _int8.quantize_onnx_static(
            source or _source(tmp_path),
            out,
            str(_cal_dir(tmp_path)),
            model_spec=_SPEC,
            **kw,
        )


# ---------------------------------------------------------------------------
# The write lifecycle
# ---------------------------------------------------------------------------


def test_above_floor_is_written_and_returns_its_report(tmp_path: Path) -> None:
    report = _quantize(tmp_path, _passing_report(tmp_path))
    out = tmp_path / "model_int8.onnx"
    assert report.output_path == str(out)
    assert out.exists()
    assert report.passed is True
    assert _leftovers(out) == [out], "no partial beside the published artifact"


def test_below_floor_is_not_written_and_raises(tmp_path: Path) -> None:
    """The whole point: a zero-detection artifact does not reach the disk."""
    with pytest.raises(ExportError):
        _quantize(tmp_path, _failing_report(tmp_path))
    assert not (tmp_path / "model_int8.onnx").exists()


def test_below_floor_leaves_no_partial(tmp_path: Path) -> None:
    """Nor does it leave one for someone to find and deploy."""
    with pytest.raises(ExportError):
        _quantize(tmp_path, _failing_report(tmp_path))
    assert _leftovers(tmp_path / "model_int8.onnx") == []


def test_error_message_carries_total_recall(tmp_path: Path) -> None:
    """The loss is visible without reading source."""
    with pytest.raises(ExportError) as exc:
        _quantize(tmp_path, _failing_report(tmp_path))
    msg = str(exc.value)
    assert "0.4" in msg, "TOTAL recall — the number gated recall cannot stand in for"
    assert "0.25" in msg, "gated recall"
    assert "floor" in msg.lower()
    assert "1.0" in msg, "the floor it was judged against"
    assert "margin" in msg.lower()


def test_quantize_crash_leaves_no_artifact(tmp_path: Path) -> None:
    """Quantization is long-running and can die mid-write. Rollback is total."""
    from yowo.export import _int8

    out = tmp_path / "model_int8.onnx"
    with (
        _patched_ort(_mock_ort(crash=True)),
        pytest.raises(ExportError, match="quantization failed"),
    ):
        _int8.quantize_onnx_static(
            _source(tmp_path), out, str(_cal_dir(tmp_path)), model_spec=_SPEC
        )
    assert not out.exists()
    assert _leftovers(out) == []


def test_parity_measurement_failure_is_a_floor_failure(tmp_path: Path) -> None:
    """An artifact we cannot measure is an artifact we do not publish."""
    from yowo.export import _int8

    out = tmp_path / "model_int8.onnx"
    with (
        _patched_ort(_mock_ort()),
        patch.object(_int8, "measure_int8_parity", side_effect=RuntimeError("ORT load failed")),
        pytest.raises(ExportError),
    ):
        _int8.quantize_onnx_static(
            _source(tmp_path), out, str(_cal_dir(tmp_path)), model_spec=_SPEC
        )
    assert not out.exists()
    assert _leftovers(out) == []


def test_a_preexisting_artifact_survives_a_failed_requantization(tmp_path: Path) -> None:
    """A failed re-export must not destroy what is already there, nor refresh it."""
    out = tmp_path / "model_int8.onnx"
    out.write_bytes(b"the artifact from last time")
    with pytest.raises(ExportError):
        _quantize(tmp_path, _failing_report(tmp_path))
    assert out.read_bytes() == b"the artifact from last time"


# ---------------------------------------------------------------------------
# Refusals — a graph we cannot measure is never quantized
# ---------------------------------------------------------------------------


def test_a_kv_cache_graph_is_refused_by_name(tmp_path: Path) -> None:
    """Measured 2026-09-15: a KV export has 4 inputs and 3 outputs.

    Walking back from ``present_k_0`` drags Attention internals into the tail
    (26 nodes instead of 22), and the gate's ``{"images": batch}`` feed cannot
    run the graph at all. ``_exporter.py`` puts the KV branch and the INT8
    branch in sequence with no mutual exclusion, so this combination is
    reachable. Refuse it by name rather than quantize something unmeasurable.
    """
    from yowo.export import _int8

    with (
        _patched_ort(_mock_ort()),
        pytest.raises(ExportError, match="use_cache|past_k_0|single input"),
    ):
        _int8.quantize_onnx_static(
            _kv_source(tmp_path),
            tmp_path / "model_int8.onnx",
            str(_cal_dir(tmp_path)),
            model_spec=_SPEC,
        )
    assert not (tmp_path / "model_int8.onnx").exists()


@pytest.mark.parametrize("task", ["classify", "obb"])
def test_a_non_detection_task_is_refused_by_name(tmp_path: Path, task: str) -> None:
    """The detection decoder cannot decode a classify or OBB output.

    Measured 2026-09-15: a classify tail is 5 nodes and INCLUDES the final
    ``Gemm``; an OBB tail is 44 nodes with Cos/Sin/Slice angle decode. Running
    either through ``postprocess`` produces a number from a decode that never
    happened — a verdict on garbage.
    """
    from yowo.export import _int8

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, task=task)
    with (
        _patched_ort(_mock_ort()),
        pytest.raises(ExportError, match=task),
    ):
        _int8.quantize_onnx_static(
            _source(tmp_path),
            tmp_path / "model_int8.onnx",
            str(_cal_dir(tmp_path)),
            model_spec=spec,
        )
    assert not (tmp_path / "model_int8.onnx").exists()


def test_an_empty_parity_set_is_refused(tmp_path: Path) -> None:
    """parity_sample=0 is the skip-measurement door wearing another hat."""
    with pytest.raises(ExportError, match="parity"):
        _quantize(tmp_path, _passing_report(tmp_path), parity_sample=0)
    with pytest.raises(ExportError, match="parity"):
        _quantize(tmp_path, _passing_report(tmp_path), parity_images=[])


# ---------------------------------------------------------------------------
# The parity set
# ---------------------------------------------------------------------------


def test_the_parity_set_is_capped_by_parity_sample(tmp_path: Path) -> None:
    """The gate runs two ORT sessions per image; the cap is declared, not implied."""
    from yowo.export import _int8

    seen: dict[str, Any] = {}

    def _capture(fp32: Path, int8: Path, images: Any, **kw: Any) -> Any:
        seen["n"] = len(list(images))
        return _passing_report(tmp_path)

    out = tmp_path / "model_int8.onnx"
    with (
        _patched_ort(_mock_ort()),
        patch.object(_int8, "measure_int8_parity", side_effect=_capture),
    ):
        _int8.quantize_onnx_static(
            _source(tmp_path),
            out,
            str(_cal_dir(tmp_path)),
            model_spec=_SPEC,
            parity_sample=3,
        )
    assert seen["n"] == 3


def test_an_explicit_parity_set_overrides_the_calibration_images(tmp_path: Path) -> None:
    """The seam /tasks/accuracy-dataset.md plugs a held-out set into."""
    from yowo.export import _int8

    held = tmp_path / "held_out.jpg"
    cv2.imwrite(str(held), np.full((40, 70, 3), 200, dtype=np.uint8))
    seen: dict[str, Any] = {}

    def _capture(fp32: Path, int8: Path, images: Any, **kw: Any) -> Any:
        seen["images"] = list(images)
        seen["held_out"] = kw.get("held_out")
        seen["source"] = kw.get("parity_set_source")
        return _passing_report(tmp_path)

    out = tmp_path / "model_int8.onnx"
    with (
        _patched_ort(_mock_ort()),
        patch.object(_int8, "measure_int8_parity", side_effect=_capture),
    ):
        _int8.quantize_onnx_static(
            _source(tmp_path),
            out,
            str(_cal_dir(tmp_path)),
            model_spec=_SPEC,
            parity_images=[held],
        )
    assert seen["images"] == [held]
    assert seen["held_out"] is True, "disjoint from the calibration set"
    assert seen["source"] == "declared"


def test_the_default_parity_set_is_not_held_out(tmp_path: Path) -> None:
    """The default judges the artifact on its own calibration data, and says so."""
    from yowo.export import _int8

    seen: dict[str, Any] = {}

    def _capture(fp32: Path, int8: Path, images: Any, **kw: Any) -> Any:
        seen.update(kw)
        return _passing_report(tmp_path)

    with (
        _patched_ort(_mock_ort()),
        patch.object(_int8, "measure_int8_parity", side_effect=_capture),
    ):
        _int8.quantize_onnx_static(
            _source(tmp_path),
            tmp_path / "model_int8.onnx",
            str(_cal_dir(tmp_path)),
            model_spec=_SPEC,
        )
    assert seen["held_out"] is False
    assert seen["parity_set_source"] == "calibration"


# ---------------------------------------------------------------------------
# The report is JSON-native, or the sidecar raises AFTER the artifact is promoted
# ---------------------------------------------------------------------------


def test_the_report_survives_the_sidecar_round_trip(tmp_path: Path) -> None:
    """``ExportMetadata.to_json`` is ``json.dumps(asdict(self))``.

    A ``Path`` anywhere in the report raises ``TypeError`` there — after the
    artifact has already been promoted to ``output_path``. Every field must be
    JSON-native.
    """
    from dataclasses import asdict

    from yowo.export._metadata import ExportMetadata

    report = _passing_report(tmp_path)
    meta = ExportMetadata(
        model_name="yolo11n",
        format="onnx",
        precision="int8",
        imgsz=640,
        batch_size=1,
        dynamic=False,
        input_shape=[1, 3, 640, 640],
        file_path=str(tmp_path / "yolo11n_int8.onnx"),
        file_size_bytes=2_984_512,
        created_at="2026-09-15T00:00:00+00:00",
        export_duration_sec=12.5,
        source_weights=str(tmp_path / "yolo11n.pt"),
        yowo_version="2.4.0",
        extra={"int8_parity": asdict(report)},
    )
    out = meta.save(tmp_path / "yolo11n.yowo.json")
    loaded = ExportMetadata.load(out)
    assert loaded.extra["int8_parity"]["total_recall"] == 0.8
    assert loaded.extra["int8_parity"]["held_out"] is False
    assert loaded.extra["int8_parity"]["provider"] == "CPUExecutionProvider"


def test_the_calibration_batch_is_clamped_to_the_graphs_pinned_dim(tmp_path: Path) -> None:
    """``dynamic_batch=False`` pins the batch dim; a bigger batch dies mid-calibration.

    Measured 2026-09-15: feeding batch 8 to a batch-1 graph raises
    ``INVALID_ARGUMENT: Got invalid dimensions for input: images`` partway
    through the entropy pass. The graph states its own shape, so read it rather
    than require every caller to know which export produced the file.
    """
    from yowo.export import _int8

    g = helper.make_graph(
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Sigmoid", ["h"], ["output0"], name="sigmoid"),
        ],
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, None)],
    )
    pinned = tmp_path / "model.onnx"
    onnx.save(helper.make_model(g), str(pinned))

    seen: dict[str, Any] = {}

    def _capture_batches(images: Any, batch_size: int, input_size: int) -> Any:
        seen["batch_size"] = batch_size
        return iter(())

    with (
        _patched_ort(_mock_ort()),
        patch.object(_int8, "measure_int8_parity", return_value=_passing_report(tmp_path)),
        patch.object(_int8, "calibration_batches", side_effect=_capture_batches),
    ):
        _int8.quantize_onnx_static(
            pinned,
            tmp_path / "model_int8.onnx",
            str(_cal_dir(tmp_path)),
            model_spec=_SPEC,
            input_size=64,
            batch_size=8,
        )
    assert seen["batch_size"] == 1, "clamped to what the graph accepts"


def test_external_data_beside_the_partial_is_refused(tmp_path: Path) -> None:
    """An ONNX graph with external initializers is TWO files.

    Promoting only the graph would publish an artifact that does not load, and
    the cleanup sweep would then delete its companion. Refuse instead.
    """
    from yowo.export import _int8

    out = tmp_path / "model_int8.onnx"

    def _run_with_external_data(src: str, dst: str, *a: Any, **kw: Any) -> None:
        Path(dst).write_bytes(b"graph")
        Path(dst + ".data").write_bytes(b"initializers")

    mod = _mock_ort()
    mod.quantize_static.side_effect = _run_with_external_data

    with (
        _patched_ort(mod),
        patch.object(_int8, "measure_int8_parity", return_value=_passing_report(tmp_path)),
        pytest.raises(ExportError, match="external data"),
    ):
        _int8.quantize_onnx_static(
            _source(tmp_path), out, str(_cal_dir(tmp_path)), model_spec=_SPEC
        )

    assert not out.exists()
    assert _leftovers(out) == [], "and the companion is swept with the partial"
