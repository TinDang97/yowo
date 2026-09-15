"""The real INT8 export path, end to end. A mocked exporter cannot prove this (Q12).

Measured 2026-09-15 at HEAD 714fa79, on a production-path ``yolo11n`` FP32 ONNX
export (325 nodes, 87 Conv) run through the project's own
``yowo.io.preprocess`` -> ``yowo.postprocess.postprocess`` on ``bus.jpg`` at
confidence 0.25:

| variant                         | detections | bytes      |
|---------------------------------|-----------:|-----------:|
| FP32                            |          5 | 10,631,082 |
| INT8, every node quantized      |          0 |  2,917,035 |
| INT8, decode tail excluded      |          5 |  2,984,512 |

Note the middle row's size: the BROKEN artifact is SMALLER than the correct
one, because a graph whose class head is entirely 0.0 compresses
beautifully. A
"smaller than FP32" assertion on its own therefore prefers the bug, and is only
ever asserted here alongside the detection check.

These checks cost a real ONNX export plus a real entropy calibration pass —
tens of seconds each — which is why they live in the integration tier rather
than the unit one.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from yowo.export import export_model
from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

_CONF = 0.25


@pytest.fixture(scope="module")
def calibration_dir(tmp_path_factory: pytest.TempPathFactory, sample_image_path: Path) -> Path:
    """A calibration directory above ``_MIN_CALIBRATION_IMAGES``.

    Deliberately built from copies of one image: these checks measure whether
    the graph COLLAPSES, which is independent of image content, and a fixed
    input keeps the comparison between the two quantizations honest.
    """
    d = tmp_path_factory.mktemp("calibration")
    for i in range(12):
        shutil.copy(sample_image_path, d / f"cal_{i:02d}.jpg")
    return d


def _fp32_export(spec: ModelSpec, out: Path) -> Path:
    meta = export_model(
        spec, ExportFormat.ONNX, out, precision=Precision.FP32, dynamic_batch=False, imgsz=640
    )
    return Path(meta.file_path)


def _raw(model_path: Path) -> np.ndarray:
    """The model's raw output tensor on a fixed synthetic input."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    blank = np.zeros((1, 3, 640, 640), dtype=np.float32)
    return session.run(None, {"images": blank})[0]


def _detect(model_path: Path, image: Path, spec: ModelSpec) -> list[object]:
    """Run one image through the project's real preprocess -> postprocess path."""
    import cv2
    import onnxruntime as ort

    from yowo.io._decode import preprocess
    from yowo.postprocess import postprocess
    from yowo.types import BackendType, Frame

    frame = Frame(pixels=cv2.imread(str(image)), source_id=str(image), frame_index=0)
    tensor = preprocess([frame], (640, 640))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    raw = session.run(None, {"images": tensor.data})[0]
    dets = postprocess(
        raw,
        tensor,
        [frame],
        model_spec=spec,
        backend=BackendType.ONNX,
        confidence_threshold=_CONF,
    )
    return list(dets[0].boxes)


# ---------------------------------------------------------------------------
# The rule, on real graphs
# ---------------------------------------------------------------------------


def test_the_same_rule_covers_both_families(tmp_path: Path) -> None:
    """One topological rule, two families, no branch.

    The m4 box blamed the end2end decode/topk tail. ``end2end`` is YOLO26-only:
    the yolo11n graph contains none of those ops and fails identically. The
    rule is stated over the post-head DECODE SUBGRAPH, so it sweeps up the
    topk ops on YOLO26 without ever naming them, and finds none on YOLO11
    because the graph has none.

    Asserted on op TYPES, never on node counts — a count is a property of
    torch.onnx fusion at opset 17, not of this code.
    """
    onnx = pytest.importorskip("onnx")
    from yowo.export._int8 import decode_tail

    topk_ops = {"TopK", "ArgMax", "GatherElements", "ReduceMax"}
    tails: dict[str, set[str]] = {}
    graph_ops: dict[str, Counter[str]] = {}

    for family in (ModelFamily.YOLO11, ModelFamily.YOLO26):
        out = tmp_path / family.value
        out.mkdir()
        model = onnx.load(str(_fp32_export(ModelSpec(family, ModelSize.NANO), out)))
        by_name = {n.name: n.op_type for n in model.graph.node}
        graph_ops[family.value] = Counter(by_name.values())
        tails[family.value] = {by_name[n] for n in decode_tail(model)}

    # YOLO11: the graph has no topk ops at all, so the tail cannot contain any.
    assert not (topk_ops & set(graph_ops["yolo11"])), "yolo11 graph has no topk ops"
    assert not (topk_ops & tails["yolo11"])
    assert {"Softmax", "Sigmoid"} <= tails["yolo11"], "the DFL softmax and final sigmoid are in"

    # YOLO26: the same walk sweeps them up, with no family branch.
    assert topk_ops <= set(graph_ops["yolo26"]), "yolo26 graph has the topk ops"
    assert topk_ops <= tails["yolo26"], "and the same rule excludes every one of them"


def test_the_tail_never_contains_a_conv(tmp_path: Path) -> None:
    """The Convs are where the size win lives; every one of them still quantizes."""
    onnx = pytest.importorskip("onnx")
    from yowo.export._int8 import decode_tail

    out = tmp_path / "exp"
    out.mkdir()
    model = onnx.load(str(_fp32_export(ModelSpec(ModelFamily.YOLO11, ModelSize.NANO), out)))
    by_name = {n.name: n.op_type for n in model.graph.node}
    assert "Conv" not in {by_name[n] for n in decode_tail(model)}


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


def test_excluding_the_decode_tail_is_what_makes_int8_detect(
    tmp_path: Path, calibration_dir: Path, sample_image_path: Path
) -> None:
    """Quantize the same real graph both ways; only the exclusion detects anything.

    This is the causal proof. Without the second arm, a green first arm could
    come from an ORT version change rather than from this fix.
    """
    ort_quant = pytest.importorskip("onnxruntime.quantization")
    onnx = pytest.importorskip("onnx")

    from yowo.export._calibration import calibration_batches, resolve_calibration_images
    from yowo.export._int8 import decode_tail

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    out = tmp_path / "exp"
    out.mkdir()
    fp32 = _fp32_export(spec, out)
    images = resolve_calibration_images(str(calibration_dir))

    class _Reader(ort_quant.CalibrationDataReader):  # type: ignore[misc]
        def __init__(self) -> None:
            self._it = calibration_batches(images, batch_size=1, input_size=640)

        def get_next(self) -> dict[str, object] | None:  # type: ignore[override]
            try:
                return {"images": next(self._it)}
            except StopIteration:
                return None

    def _quantize(dst: Path, exclude: list[str] | None) -> Path:
        ort_quant.quantize_static(
            str(fp32),
            str(dst),
            _Reader(),
            quant_format=ort_quant.QuantFormat.QDQ,
            activation_type=ort_quant.QuantType.QInt8,
            weight_type=ort_quant.QuantType.QInt8,
            calibrate_method=ort_quant.CalibrationMethod.Entropy,
            nodes_to_exclude=exclude,
        )
        return dst

    fp32_boxes = _detect(fp32, sample_image_path, spec)
    assert len(fp32_boxes) >= 4, "the FP32 baseline must detect something to compare against"

    collapsed = _quantize(out / "all.onnx", None)
    all_nodes = _detect(collapsed, sample_image_path, spec)
    tail = sorted(decode_tail(onnx.load(str(fp32))))
    excluded = _detect(_quantize(out / "excl.onnx", tail), sample_image_path, spec)

    assert len(all_nodes) == 0, "quantizing the decode tail collapses the graph — the defect"
    assert len(excluded) >= 4, "excluding it recovers detection — the fix"

    # WHICH logits die, exactly. Measured here rather than asserted in prose:
    # rows 0:4 are the box geometry and they SURVIVE (they decode from the
    # anchor grid), which is why the output looks well-formed. Rows 4:84 are the
    # class scores and every one of them is exactly 0.0 — the confidence channel
    # is dead regardless of what the image contains.
    #
    # That exactness is what licenses measuring this gate on the calibration
    # set: a dead class head is dead for every input. Were the scores merely
    # small-but-nonzero, sensitivity WOULD be scene-dependent and the
    # non-held-out parity set would be a real weakness rather than a declared
    # limitation.
    collapsed_raw, fp32_raw = _raw(collapsed), _raw(fp32)
    assert np.count_nonzero(collapsed_raw[0, 4:, :]) == 0, (
        "every CLASS logit is exactly 0.0 — not merely below threshold"
    )
    assert np.count_nonzero(collapsed_raw[0, :4, :]) > 0, (
        "box geometry survives, which is why the artifact looks well-formed"
    )
    assert np.count_nonzero(fp32_raw[0, 4:, :]) > 0, "FP32's class head is alive"


def test_real_int8_export_detects_what_fp32_detects(
    tmp_path: Path, calibration_dir: Path, sample_image_path: Path
) -> None:
    """The whole path: ``export_model`` at INT8 writes a gated, measured artifact."""
    pytest.importorskip("onnxruntime.quantization")

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    fp32_dir = tmp_path / "fp32"
    fp32_dir.mkdir()
    fp32 = _fp32_export(spec, fp32_dir)
    fp32_boxes = _detect(fp32, sample_image_path, spec)

    int8_dir = tmp_path / "int8"
    int8_dir.mkdir()
    meta = export_model(
        spec,
        ExportFormat.ONNX,
        int8_dir,
        precision=Precision.INT8,
        dynamic_batch=False,
        imgsz=640,
        calibration_data=str(calibration_dir),
    )
    artifact = Path(meta.file_path)
    assert artifact.exists()

    int8_boxes = _detect(artifact, sample_image_path, spec)
    assert len(int8_boxes) > 0, "a published artifact that detects nothing is the defect"

    # Only ever asserted TOGETHER with the detection check above: the collapsed
    # artifact was 2,917,035 bytes against the correct one's 2,984,512, so a
    # size assertion standing alone prefers the bug.
    assert artifact.stat().st_size < fp32.stat().st_size

    strong = [b for b in fp32_boxes if b.confidence >= 2.0 * _CONF]
    assert len(int8_boxes) >= len(strong), "every confident FP32 detection should survive"


def test_sidecar_records_the_parity_report(
    tmp_path: Path, calibration_dir: Path, sample_image_path: Path
) -> None:
    """The delta is on the artifact, not only in a log line that scrolled away."""
    pytest.importorskip("onnxruntime.quantization")

    from yowo.export._metadata import ExportMetadata

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    out = tmp_path / "int8"
    out.mkdir()
    meta = export_model(
        spec,
        ExportFormat.ONNX,
        out,
        precision=Precision.INT8,
        dynamic_batch=False,
        imgsz=640,
        calibration_data=str(calibration_dir),
    )

    sidecar = Path(meta.file_path).with_suffix("").with_suffix(".yowo.json")
    parity = ExportMetadata.load(sidecar).extra["int8_parity"]

    assert parity["passed"] is True
    assert 0.0 <= parity["total_recall"] <= 1.0
    assert parity["gated_recall"] >= parity["floor"]
    # Self-contained: a reader who never saw this source tree can interpret it.
    for key in ("floor", "margin", "confidence_threshold", "iou_threshold", "provider"):
        assert key in parity
    assert parity["held_out"] is False, "the default parity set IS the calibration set"
    assert parity["parity_set_source"] == "calibration"


def test_an_unreachable_floor_is_refused_by_the_real_path(
    tmp_path: Path, calibration_dir: Path
) -> None:
    """A floor the artifact cannot clear publishes nothing.

    Drives the REAL quantizer, so this proves the gate refuses a REAL artifact
    rather than a stubbed verdict. ``parity_iou=0.999`` demands near-pixel-exact
    boxes, which INT8 measurably does not deliver, so every gated FP32 detection
    goes unmatched and the floor is out of reach.

    Named for what it proves. This exercises the refusal machinery — the score,
    the ExportError, the partial sweep — on a NORMAL artifact judged by an
    unreachable criterion. It is not a collapsed artifact; the collapse itself
    is asserted in
    ``test_excluding_the_decode_tail_is_what_makes_int8_detect``.
    """
    pytest.importorskip("onnxruntime.quantization")

    from yowo.errors import ExportError
    from yowo.export._int8 import quantize_onnx_static

    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)
    out = tmp_path / "exp"
    out.mkdir()
    fp32 = _fp32_export(spec, out)
    artifact = out / "yolo11n_int8.onnx"

    with pytest.raises(ExportError, match="recall"):
        quantize_onnx_static(
            fp32,
            artifact,
            str(calibration_dir),
            model_spec=spec,
            parity_sample=1,
            parity_iou=0.999,
            parity_floor=1.0,
        )

    assert not artifact.exists(), "a refused artifact never reaches the disk"
    leftovers = [p for p in out.iterdir() if p.name.startswith("yolo11n_int8")]
    assert leftovers == [], "and leaves nothing behind for someone to find and deploy"
