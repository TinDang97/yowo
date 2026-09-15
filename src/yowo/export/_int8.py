"""INT8 quantization support for TensorRT and ONNX Runtime.

Provides:
- ``create_tensorrt_calibrator``: Factory for TensorRT INT8 entropy calibrator.
- ``decode_tail``: The post-head decode nodes INT8 must not quantize, computed
  from the graph.
- ``measure_int8_parity`` / ``ParityReport``: What a quantized artifact
  recovered of its FP32 source, and on what terms.
- ``quantize_onnx_static``: Static INT8 quantization for ONNX models via
  ``onnxruntime.quantization``, gated on that measurement.

Both paths use ``calibration_batches()`` from ``_calibration`` as the shared
image preprocessing pipeline -- which is itself the inference preprocessing
pipeline, so INT8 scales are derived from the distribution inference produces.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from yowo.errors import DependencyError, ExportError
from yowo.export._calibration import calibration_batches, resolve_calibration_images

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TensorRT INT8 calibrator
# ---------------------------------------------------------------------------


def create_tensorrt_calibrator(
    image_paths: list[Path],
    batch_size: int = 8,
    input_size: int = 640,
    cache_file: Path | None = None,
) -> object:
    """Create a TensorRT INT8 entropy calibrator.

    Returns an object implementing ``trt.IInt8EntropyCalibrator2``.
    The ``tensorrt`` import is deferred to call time so the module can be
    imported on machines without TensorRT installed.

    Args:
        image_paths: Calibration image file paths (from
            ``resolve_calibration_images``).
        batch_size: Images per calibration batch.
        input_size: Model input spatial size (square).
        cache_file: Optional path to read/write a calibration cache for
            faster repeated builds.

    Returns:
        A calibrator instance compatible with
        ``trt.BuilderConfig.int8_calibrator``.

    Raises:
        DependencyError: ``tensorrt`` or ``pycuda`` is not installed.
    """
    try:
        import tensorrt as trt  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "tensorrt",
            "pip install tensorrt>=10.0 --extra-index-url https://pypi.nvidia.com",
        ) from exc

    try:
        import pycuda.autoinit as _autoinit  # type: ignore[import-untyped]  # noqa: F401
        import pycuda.driver as cuda  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "pycuda",
            "pip install pycuda",
            message="pycuda is required for TensorRT INT8 calibration.",
        ) from exc

    class _Calibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
        """Entropy calibrator feeding preprocessed image batches to TensorRT."""

        def __init__(self) -> None:
            super().__init__()
            self._batch_iter: Iterator[Any] | None = None
            self._device_buffer: Any = None

        def get_batch_size(self) -> int:
            return batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self._batch_iter is None:
                self._batch_iter = calibration_batches(
                    image_paths, batch_size=batch_size, input_size=input_size
                )
            try:
                batch = next(self._batch_iter)
            except StopIteration:
                return None

            if self._device_buffer is None:
                self._device_buffer = cuda.mem_alloc(batch.nbytes)  # type: ignore[attr-defined]
            cuda.memcpy_htod(self._device_buffer, batch)  # type: ignore[attr-defined]
            return [int(self._device_buffer)]

        def read_calibration_cache(self) -> bytes | None:
            if cache_file is not None and cache_file.exists():
                logger.info("Reading INT8 calibration cache: %s", cache_file)
                return cache_file.read_bytes()
            return None

        def write_calibration_cache(self, cache: memoryview) -> None:
            if cache_file is not None:
                cache_file.write_bytes(bytes(cache))
                logger.info("Wrote INT8 calibration cache: %s", cache_file)

    return _Calibrator()


# ---------------------------------------------------------------------------
# ONNX static INT8 quantization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissedDetection:
    """An FP32 detection the quantized artifact did not reproduce.

    Counts alone cannot tell a deployer whether "4/5" lost a marginal
    detection or a confident one, so the loss is named.
    """

    class_id: int
    confidence: float
    image: str


@dataclass(frozen=True)
class ParityReport:
    """What a quantized artifact recovered of its FP32 source, and on what terms.

    Every field is JSON-native on purpose: ``ExportMetadata.to_json`` is
    ``json.dumps(asdict(self))``, and a ``Path`` anywhere in here would raise
    ``TypeError`` *after* the artifact had already been promoted.

    What this measures, and what it does not. Quantizing the decode subgraph
    destroys the class head independently of image content -- measured
    2026-09-15 on yolo11n, every one of the 80 class logits is exactly 0.0
    while the 4 box-geometry rows still decode, which is why the artifact looks
    well-formed -- so this instrument detects COLLAPSE with full sensitivity
    even when measured on the calibration set. It is not an accuracy
    certification: by default the parity set IS the calibration set, and
    ``held_out`` says so on the artifact rather than in a document beside it.

    Attributes:
        passed: Did the artifact clear its floor (always ``True`` when the
            floor is not enforced)?
        enforced: ``False`` when ``floor`` is 0.0 -- measured, not gated.
        output_path: Where the artifact this describes actually landed.
        total_recall: Matched / ALL FP32 detections, marginal ones included.
        gated_recall: Matched / FP32 detections at or above the margin band.
        gated_detections: FP32 detections the floor was actually judged on.
        int8_unmatched: INT8 boxes no FP32 detection claimed. Recall alone
            cannot see these: an artifact whose scales saturate the
            classification sigmoid emits a flood of spurious boxes, every FP32
            detection finds a partner in it, and recall reads 1.0. This is
            reported, never gated -- the node's claim is collapse, not
            precision -- but it is on the artifact so nobody has to infer it.
        missed: Every FP32 detection the artifact did not reproduce.
        floor: The gated recall required to publish.
        margin: Multiplier on the threshold defining a "confident" detection.
        held_out: Was the parity set disjoint from the calibration set?
        parity_set_source: ``"calibration"`` or ``"declared"``.
        provider: The pinned ORT provider that produced these numbers.
    """

    passed: bool
    enforced: bool
    output_path: str
    total_recall: float
    gated_recall: float
    fp32_detections: int
    int8_detections: int
    gated_detections: int
    gated_matched: int
    int8_unmatched: int
    missed: tuple[MissedDetection, ...]
    floor: float
    margin: float
    confidence_threshold: float
    iou_threshold: float
    held_out: bool
    parity_set_source: str
    provider: str
    images: tuple[str, ...]


_PARITY_PROVIDER = "CPUExecutionProvider"
_DETECTION_TASKS = frozenset({"detect"})


def decode_tail(model: Any) -> set[str]:
    """Node names in the post-head decode subgraph, computed from the graph.

    Walk backward from every graph output. A ``Conv`` terminates the walk on
    that path and stays quantizable -- that is where the entire size win lives.
    Everything reached before a Conv is the decode tail: DFL softmax, box
    decode, the final sigmoid, and on an end2end (YOLO26) graph the
    ``ReduceMax -> ArgMax -> TopK -> GatherElements`` selection above them.

    The rule is stated over the SUBGRAPH, never over op types or a model
    family, so it needs no per-family branch. Observed 2026-09-15 on
    torch.onnx at opset 17: 22 nodes on ``yolo11n`` (whose graph contains no
    topk ops at all) and 28 on ``yolo26n``, sweeping up every topk op there
    without naming one. Those counts are a property of that exporter's fusion,
    not of this function, and nothing asserts them.

    Subgraph attributes (``If`` / ``Loop`` bodies) are NOT descended into. No
    current family puts decode ops there; one that did would be silently
    under-excluded, which is the defect class this closes.

    Traversal is keyed on TENSORS, not node names: ONNX does not enforce
    unique node names, so a name-keyed ``seen`` set would skip a second node
    sharing a name with one already visited and silently under-exclude it. (A
    name-keyed set would still HALT on a cycle -- it drains the stack either
    way -- so termination is not the reason.)

    Args:
        model: An ``onnx.ModelProto``.

    Returns:
        The set of node names to exclude from quantization. Empty when every
        graph output is produced directly by a ``Conv``.

    Raises:
        ExportError: A tail node has an empty ``name``. ``nodes_to_exclude``
            matches by name, so such a node cannot be excluded; returning the
            set without it would silently under-exclude and reproduce the
            zero-detection artifact this function exists to prevent.
    """
    graph = model.graph
    producer = {out: node for node in graph.node for out in node.output}

    tail: set[str] = set()
    visited: set[str] = set()
    stack = [out.name for out in graph.output]

    while stack:
        tensor = stack.pop()
        if tensor in visited:
            continue
        visited.add(tensor)

        node = producer.get(tensor)
        if node is None or node.op_type == "Conv":
            continue

        if not node.name:
            raise ExportError(
                f"Cannot quantize: the decode tail contains an unnamed "
                f"{node.op_type} node (outputs {list(node.output)}). "
                "nodes_to_exclude matches by name, so this node cannot be kept "
                "out of quantization, and quantizing it produces an artifact "
                "that returns all zeros."
            )

        tail.add(node.name)
        stack.extend(node.input)

    return tail


# ---------------------------------------------------------------------------
# Parity measurement
# ---------------------------------------------------------------------------


def _iou(a: Any, b: Any) -> float:
    """Intersection over union of two ``BoundingBox`` objects."""
    inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0.0 else 0.0


def _score_parity(
    fp32_per_image: Any,
    int8_per_image: Any,
    *,
    floor: float,
    margin: float,
    confidence_threshold: float,
    iou_threshold: float,
    images: tuple[str, ...],
    output_path: str,
    held_out: bool,
    parity_set_source: str,
    provider: str,
) -> ParityReport:
    """Score INT8 detections against FP32 detections, image by image.

    Matching is greedy and deterministic: FP32 detections are taken in
    descending confidence, and each claims the highest-IoU unmatched INT8 box
    of the same class at or above ``iou_threshold``. Each INT8 box is consumed
    once -- without that, a single INT8 box could satisfy two FP32 detections
    and inflate recall.

    A detection is GATED when its confidence is at or above
    ``margin * confidence_threshold`` (inclusive). Detections inside the band
    below it are reported but do not decide the outcome: measured quantization
    confidence drift is 0.05-0.15, the same order as such a detection's own
    margin above the threshold, so gating on one makes the floor a coin flip on
    scene content.

    Raises:
        ExportError: The FP32 source produced no gated detection at all. That
            is a 0/0 ratio -- a measurement that proves nothing -- and
            reporting it as 1.0 would wave through an artifact that detects
            nothing.
    """
    gate = margin * confidence_threshold

    fp32_total = 0
    int8_total = 0
    gated_total = 0
    gated_matched = 0
    total_matched = 0
    int8_unmatched = 0
    missed: list[MissedDetection] = []

    for index, (fp32_boxes, int8_boxes) in enumerate(
        zip(fp32_per_image, int8_per_image, strict=True)
    ):
        image = images[index] if index < len(images) else str(index)
        unmatched = list(int8_boxes)
        int8_total += len(int8_boxes)

        for fp32_box in sorted(fp32_boxes, key=lambda b: -b.confidence):
            fp32_total += 1
            gated = fp32_box.confidence >= gate
            if gated:
                gated_total += 1

            best = None
            best_iou = -1.0
            for candidate in unmatched:
                if candidate.class_id != fp32_box.class_id:
                    continue
                overlap = _iou(fp32_box, candidate)
                if overlap > best_iou:
                    best, best_iou = candidate, overlap

            if best is not None and best_iou >= iou_threshold:
                unmatched.remove(best)
                total_matched += 1
                if gated:
                    gated_matched += 1
            else:
                missed.append(
                    MissedDetection(
                        class_id=int(fp32_box.class_id),
                        confidence=float(fp32_box.confidence),
                        image=image,
                    )
                )

        int8_unmatched += len(unmatched)

    if gated_total == 0:
        raise ExportError(
            "INT8 parity cannot be measured: the FP32 source produced no gated "
            f"detection over {len(images)} image(s) at confidence "
            f"{confidence_threshold} with margin {margin} (gate {gate}). "
            "A 0/0 recall proves nothing, and reporting it as a pass would "
            "publish an artifact that may detect nothing at all. Use a parity "
            "set the FP32 model actually detects in, via parity_images."
        )

    enforced = floor > 0.0
    gated_recall = gated_matched / gated_total
    total_recall = total_matched / fp32_total if fp32_total else 0.0

    return ParityReport(
        passed=(not enforced) or gated_recall >= floor,
        enforced=enforced,
        output_path=output_path,
        total_recall=total_recall,
        gated_recall=gated_recall,
        fp32_detections=fp32_total,
        int8_detections=int8_total,
        gated_detections=gated_total,
        gated_matched=gated_matched,
        int8_unmatched=int8_unmatched,
        missed=tuple(missed),
        floor=floor,
        margin=margin,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        held_out=held_out,
        parity_set_source=parity_set_source,
        provider=provider,
        images=images,
    )


def measure_int8_parity(
    fp32_path: Path,
    int8_path: Path,
    image_paths: Any,
    *,
    model_spec: Any,
    input_size: int = 640,
    floor: float = 1.0,
    margin: float = 2.0,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    held_out: bool = False,
    parity_set_source: str = "calibration",
) -> ParityReport:
    """Run both artifacts over *image_paths* and score what INT8 recovered.

    Both passes go through the project's own ``preprocess`` ->
    ``postprocess`` path, so the number is post-NMS detections and not a
    tensor-distance proxy for them.

    ONE IMAGE AT A TIME, deliberately. With ``dynamic_batch=False`` the
    exported graph's batch dimension is pinned at 1, so a measurement that
    inherited the calibration batch size would raise a shape error on exactly
    the configuration this gate most needs to run on. It also caps peak memory:
    the FP32 session is closed before the INT8 session opens, so two full
    graphs are never resident at once.

    The provider is pinned to CPU and recorded on the report -- a number whose
    execution provider is unstated does not reproduce.

    Raises:
        ExportError: An image could not be read, or the FP32 source produced
            no gated detection.
    """
    import cv2
    import onnxruntime as ort  # type: ignore[import-untyped]

    from yowo.io._decode import preprocess
    from yowo.postprocess import postprocess
    from yowo.types import BackendType, Frame

    prepared = []
    for path in image_paths:
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise ExportError(f"INT8 parity image could not be read: {path}")
        # cv2 stubs return MatLike; imread of a real file is uint8 HWC BGR.
        frame = Frame(pixels=pixels, source_id=str(path), frame_index=0)  # type: ignore[arg-type]
        prepared.append((frame, preprocess([frame], (input_size, input_size))))

    def _detect(model_path: Path) -> list[list[Any]]:
        session = ort.InferenceSession(str(model_path), providers=[_PARITY_PROVIDER])
        try:
            per_image = []
            for frame, tensor in prepared:
                # ORT stubs widen run() to a union including SparseTensor;
                # a detection graph output is a dense float32 array.
                raw = cast("NDArray[np.float32]", session.run(None, {"images": tensor.data})[0])
                detections = postprocess(
                    raw,
                    tensor,
                    [frame],
                    model_spec=model_spec,
                    backend=BackendType.ONNX,
                    confidence_threshold=confidence_threshold,
                )
                per_image.append(list(detections[0].boxes))
            return per_image
        finally:
            del session

    fp32_detections = _detect(fp32_path)
    int8_detections = _detect(int8_path)

    return _score_parity(
        fp32_detections,
        int8_detections,
        floor=floor,
        margin=margin,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        # Basenames, not absolute paths: the sidecar SHIPS with the artifact,
        # and a machine-local path discloses the builder's home directory while
        # telling the reader nothing -- `calibration_data` already records the
        # directory, and a per-machine string is not a property of the model.
        images=tuple(Path(p).name for p in image_paths),
        output_path=str(int8_path),
        held_out=held_out,
        parity_set_source=parity_set_source,
        provider=_PARITY_PROVIDER,
    )


# ---------------------------------------------------------------------------
# The gated write
# ---------------------------------------------------------------------------


def _resolve_parity_set(
    calibration_images: list[Path],
    parity_images: Any,
    parity_sample: int,
) -> tuple[list[Path], bool, str]:
    """Pick the images the artifact will be judged on, and say where they came from."""
    if parity_sample <= 0:
        raise ExportError(
            f"parity_sample must be at least 1, got {parity_sample}. A quantized "
            "artifact with no measured delta against its source is not a "
            "deliverable; lower parity_floor to 0.0 to measure without gating."
        )

    if parity_images is None:
        selected = list(calibration_images)[:parity_sample]
        return selected, False, "calibration"

    selected = [Path(p) for p in parity_images][:parity_sample]
    if not selected:
        raise ExportError(
            "parity_images resolved to no images. A quantized artifact with no "
            "measured delta against its source is not a deliverable; pass None "
            "to measure on the calibration set, or lower parity_floor to 0.0 to "
            "measure without gating."
        )
    calibration = {str(p) for p in calibration_images}
    held_out = {str(p) for p in selected}.isdisjoint(calibration)
    return selected, held_out, "declared"


def _graph_batch_size(model: Any, requested: int, onnx_path: Path) -> int:
    """Clamp the calibration batch to what the graph will actually accept.

    ``dynamic_batch=False`` pins the exported graph's batch dimension at a
    literal value, and feeding a larger calibration batch raises
    ``INVALID_ARGUMENT: Got invalid dimensions for input: images`` partway
    through the entropy pass. The graph states its own shape, so read it rather
    than make the caller know which export produced this file.
    """
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if not len(dims):
        return requested
    fixed = dims[0].dim_value
    if fixed and fixed < requested:
        logger.info(
            "Calibration batch clamped from %d to %d: %s pins its batch dimension.",
            requested,
            fixed,
            onnx_path.name,
        )
        return int(fixed)
    return requested


def quantize_onnx_static(
    onnx_path: Path,
    output_path: Path,
    calibration_data: str,
    *,
    model_spec: Any,
    input_size: int = 640,
    batch_size: int = 8,
    parity_floor: float = 1.0,
    parity_margin: float = 2.0,
    parity_threshold: float = 0.25,
    parity_iou: float = 0.5,
    parity_images: Any = None,
    parity_sample: int = 8,
) -> ParityReport:
    """Quantize an ONNX model to INT8, and publish it only if it still detects.

    Uses ``onnxruntime.quantization.quantize_static`` with entropy calibration,
    excluding the post-head decode subgraph (see ``decode_tail``) -- quantizing
    that subgraph produces a well-formed output tensor in which every CLASS
    logit is exactly 0.0 while the box geometry still decodes.

    The artifact is quantized to a ``.partial`` sibling, measured against its
    FP32 source, and only then promoted with ``os.replace``. Below the floor it
    is deleted and ``ExportError`` is raised, so a rejected artifact never
    reaches ``output_path`` for someone to find and deploy. A pre-existing file
    at ``output_path`` is left untouched by a failed run -- neither destroyed
    nor refreshed.

    Args:
        onnx_path: Path to the source FP32/FP16 ONNX model.
        output_path: Destination path for the quantized INT8 ONNX model.
        calibration_data: Path to calibration image directory.
        model_spec: The model being quantized. Required: ``postprocess``
            dispatches the YOLO26 NMS-free decode on ``model_spec.family``, so
            a gate without it would report a verdict on a decode that never
            happened.
        input_size: Model input spatial size (square).
        batch_size: Images per CALIBRATION batch. The parity measurement is
            always per-image and never inherits this.
        parity_floor: Gated recall required to publish. ``1.0`` by default.
            ``0.0`` disables ENFORCEMENT -- the measurement still runs, the
            report is still produced, and a warning names what was lost.
        parity_margin: A detection counts toward the floor when its confidence
            is at or above ``parity_margin * parity_threshold``. Detections
            below that are reported, never gated: measured quantization
            confidence drift is 0.05-0.15, the same order as such a
            detection's own margin above the threshold.
        parity_threshold: Confidence threshold for the parity measurement.
        parity_iou: IoU at which an INT8 box counts as the same detection.
        parity_images: Images to judge the artifact on. ``None`` uses the
            calibration images -- in which case the report stamps
            ``held_out: false``, because the artifact is then being judged on
            data it was calibrated on. Pass a held-out set to change that.
        parity_sample: Cap on how many parity images are used.

    Returns:
        The ``ParityReport`` for the published artifact. The report is the
        return value rather than the path so the accuracy evidence cannot be
        discarded with an underscore.

    Raises:
        DependencyError: ``onnxruntime`` or ``onnx`` is not installed.
        ExportError: Quantization failed; the graph cannot be measured (a
            multi-input KV-cache export, or a non-detection task); the parity
            set is empty; or the artifact did not clear ``parity_floor``.
    """
    try:
        from onnxruntime import quantization as ort_quantization  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "onnxruntime",
            "pip install onnxruntime>=1.17",
            message="onnxruntime.quantization is required for ONNX INT8 export.",
        ) from exc

    try:
        import onnx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DependencyError(
            "onnx",
            "pip install onnx>=1.12",
            message="onnx is required to compute the decode tail for INT8 export.",
        ) from exc

    task = getattr(model_spec, "task", "detect")
    if task not in _DETECTION_TASKS:
        raise ExportError(
            f"INT8 ONNX quantization is gated on a measured detection parity, and "
            f"the detection decoder cannot decode a {task!r} output. Refusing to "
            f"produce a {task!r} INT8 artifact this gate cannot verify."
        )

    model = onnx.load(str(onnx_path))
    graph_inputs = [i.name for i in model.graph.input]
    if graph_inputs != ["images"]:
        extra = [name for name in graph_inputs if name != "images"]
        raise ExportError(
            f"INT8 quantization needs a graph with the single input 'images' so the "
            f"parity gate can run it, but {onnx_path.name} also takes {extra}. This "
            "is a KV-cache export; INT8 and kv_cache are not supported together."
        )

    nodes_to_exclude = sorted(decode_tail(model))
    batch_size = _graph_batch_size(model, batch_size, onnx_path)

    calibration_images = resolve_calibration_images(calibration_data)
    parity_set, held_out, parity_set_source = _resolve_parity_set(
        calibration_images, parity_images, parity_sample
    )

    class _CalibrationReader(ort_quantization.CalibrationDataReader):  # type: ignore[misc]
        """Feeds preprocessed batches to ORT quantization."""

        def __init__(self) -> None:
            self._batch_iter = calibration_batches(
                calibration_images, batch_size=batch_size, input_size=input_size
            )

        def get_next(self) -> dict[str, Any] | None:  # type: ignore[override]
            try:
                batch = next(self._batch_iter)
            except StopIteration:
                return None
            return {"images": batch}

    partial = output_path.with_name(output_path.name + ".partial")
    try:
        try:
            ort_quantization.quantize_static(
                str(onnx_path),
                str(partial),
                _CalibrationReader(),
                quant_format=ort_quantization.QuantFormat.QDQ,
                activation_type=ort_quantization.QuantType.QInt8,
                weight_type=ort_quantization.QuantType.QInt8,
                calibrate_method=ort_quantization.CalibrationMethod.Entropy,
                nodes_to_exclude=nodes_to_exclude,
            )
        except Exception as exc:
            raise ExportError(f"ONNX INT8 quantization failed: {exc}") from exc

        try:
            report = measure_int8_parity(
                onnx_path,
                partial,
                parity_set,
                model_spec=model_spec,
                input_size=input_size,
                floor=parity_floor,
                margin=parity_margin,
                confidence_threshold=parity_threshold,
                iou_threshold=parity_iou,
                held_out=held_out,
                parity_set_source=parity_set_source,
            )
        except ExportError:
            raise
        except Exception as exc:
            raise ExportError(
                f"ONNX INT8 parity could not be measured, so the artifact was not published: {exc}"
            ) from exc

        if not report.passed:
            raise ExportError(_floor_failure_message(onnx_path, report))

        # An ONNX graph with external initializers is TWO files, and promoting
        # only the graph would publish something that does not load. ORT emits
        # them past ~2GB; refuse rather than orphan a companion.
        companions = [c for c in output_path.parent.glob(f"{partial.name}*") if c != partial]
        if companions:
            raise ExportError(
                f"INT8 quantization produced external data alongside the graph "
                f"({[c.name for c in companions]}); publishing only the graph would "
                "write an artifact that does not load. Export this model below the "
                "external-data threshold, or add multi-file promotion first."
            )

        os.replace(partial, output_path)
    finally:
        # ORT can leave more than one intermediate beside the target; sweep the
        # whole prefix so a death mid-quantization leaves nothing behind.
        for leftover in output_path.parent.glob(f"{output_path.name}.partial*"):
            leftover.unlink(missing_ok=True)

    report = replace(report, output_path=str(output_path))

    if not report.enforced:
        logger.warning(
            "INT8 parity floor NOT enforced (parity_floor=0.0): gated recall %s, "
            "total recall %s over %d FP32 detection(s). %d detection(s) lost: %s",
            report.gated_recall,
            report.total_recall,
            report.fp32_detections,
            len(report.missed),
            [(m.class_id, round(m.confidence, 4)) for m in report.missed],
        )

    logger.info(
        "ONNX INT8 quantization complete: %s -> %s (gated recall %s, total recall %s, "
        "%d node(s) excluded from quantization, held_out=%s)",
        onnx_path.name,
        output_path.name,
        report.gated_recall,
        report.total_recall,
        len(nodes_to_exclude),
        report.held_out,
    )
    return report


def _floor_failure_message(onnx_path: Path, report: ParityReport) -> str:
    """Say what was lost, in numbers, without making anyone read the source."""
    lost = ", ".join(f"class {m.class_id} at {m.confidence:.4f}" for m in report.missed)
    return (
        f"INT8 quantization of {onnx_path.name} did not clear its declared parity "
        f"floor, so no artifact was written. "
        f"gated recall {report.gated_recall} against floor {report.floor} "
        f"({report.gated_matched}/{report.gated_detections} FP32 detections at or above "
        f"margin {report.margin} x threshold {report.confidence_threshold} = "
        f"{report.margin * report.confidence_threshold}). "
        f"TOTAL recall {report.total_recall} over all {report.fp32_detections} FP32 "
        f"detection(s) on {len(report.images)} image(s); the artifact produced "
        f"{report.int8_detections} box(es), {report.int8_unmatched} of which no FP32 "
        f"detection claimed. "
        f"Lost: {lost or 'none'}. "
        f"Lower parity_floor to publish anyway with the delta recorded."
    )


__all__ = [
    "MissedDetection",
    "ParityReport",
    "create_tensorrt_calibrator",
    "decode_tail",
    "measure_int8_parity",
    "quantize_onnx_static",
]
