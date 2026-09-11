"""Classification postprocessing: raw logits (B, nc) → list[ClassificationResult]."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from yowo.types import BackendType, ClassificationResult, Frame, ModelSpec

# A degraded result is marked with a class id no model can emit and a score
# softmax cannot produce. Both together, so a consumer keying on either one
# alone still reads it correctly.
NO_PREDICTION_CLASS_ID = -1
NO_PREDICTION_SCORE = 0.0


def postprocess_classify(
    raw_output: NDArray[np.float32],
    frames: list[Frame],
    *,
    model_spec: ModelSpec,
    backend: BackendType,
    top_k: int = 5,
    inference_time_ms: float = 0.0,
) -> list[ClassificationResult]:
    """Convert raw model output (B, nc) to a ClassificationResult per frame.

    Applies softmax if the output appears to be raw logits (row sums != 1).
    Uses np.argpartition for O(n + k log k) top-k selection.

    An all-zero row is read as NO PREDICTION and returns
    ``top1_class_id=-1``, ``top1_score=0.0`` and empty top-k, rather than the
    uniform distribution softmax would otherwise produce. This is how a failed
    inference reaches a caller: the engine's sentinel for a dead backend is an
    all-zero ``(B, nc)`` array, and softmaxing it would report the LAST class at
    ``1/nc`` — indistinguishable from a real low-confidence prediction, so a
    threshold filter would hide the outage instead of revealing it. A model that
    genuinely emits an all-zero row is reported the same way, which is the
    honest reading: an all-zero row carries no information.

    Args:
        raw_output: ``(B, nc)`` array of raw logits or softmax probabilities.
            1-D inputs ``(nc,)`` are automatically promoted to ``(1, nc)``.
        frames: Source frames corresponding to the batch. If fewer frames
            are provided than batch items, missing entries use empty source_id
            and positional frame_index.
        model_spec: Spec of the model that produced this output.
        backend: Backend that ran inference.
        top_k: Number of top predictions to return per frame.
        inference_time_ms: Total inference time in milliseconds.

    Returns:
        One ``ClassificationResult`` per frame in the batch.
    """
    raw_output = np.asarray(raw_output, dtype=np.float32)
    if raw_output.ndim == 1:
        raw_output = raw_output[np.newaxis, :]
    if raw_output.ndim != 2:
        raise ValueError(f"Expected 2-D output (batch, nc), got ndim={raw_output.ndim}")

    batch_size = raw_output.shape[0]
    nc = raw_output.shape[1]

    probs = raw_output if _is_softmaxed(raw_output) else _softmax(raw_output)
    effective_k = min(top_k, nc)

    # Read BEFORE softmax: softmax turns an all-zero row into a uniform one, and
    # the information that it carried nothing is gone by then.
    no_prediction: NDArray[np.bool_] = np.logical_not(np.any(raw_output, axis=1))

    results: list[ClassificationResult] = []
    for i in range(batch_size):
        row = probs[i]
        if no_prediction[i]:
            top1_id = NO_PREDICTION_CLASS_ID
            top1_score = NO_PREDICTION_SCORE
            topk_ids: tuple[int, ...] = ()
            topk_vals: tuple[float, ...] = ()
            row = np.zeros(nc, dtype=np.float32)
        else:
            # argpartition: O(n) partial sort, then sort only the k elements
            topk_idx = np.argpartition(row, -effective_k)[-effective_k:]
            topk_idx = topk_idx[np.argsort(row[topk_idx])[::-1]]  # descending
            topk_scores = row[topk_idx]
            top1_id = int(topk_idx[0])
            top1_score = float(topk_scores[0])
            topk_ids = tuple(int(x) for x in topk_idx)
            topk_vals = tuple(float(x) for x in topk_scores)

        frame = frames[i] if i < len(frames) else None
        results.append(
            ClassificationResult(
                top1_class_id=top1_id,
                top1_score=top1_score,
                topk_class_ids=topk_ids,
                topk_scores=topk_vals,
                all_probs=tuple(row.tolist()),
                source_id=frame.source_id if frame is not None else "",
                frame_index=frame.frame_index if frame is not None else i,
                inference_time_ms=inference_time_ms,
                backend=backend,
                model_spec=model_spec,
            )
        )
    return results


def _is_softmaxed(arr: NDArray[np.float32]) -> bool:
    """Heuristic: True if rows sum to ~1.0 (already softmax-applied)."""
    row_sums = arr.sum(axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=1e-3))


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Numerically stable row-wise softmax (single allocation via in-place ops)."""
    out = x - np.max(x, axis=1, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=1, keepdims=True)
    return out


__all__ = ["postprocess_classify"]
