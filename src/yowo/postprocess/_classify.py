"""Classification postprocessing: raw logits (B, nc) → list[ClassificationResult]."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from yowo.types import BackendType, ClassificationResult, Frame, ModelSpec


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

    results: list[ClassificationResult] = []
    for i in range(batch_size):
        row = probs[i]
        # argpartition: O(n) partial sort, then sort only the k elements
        topk_idx = np.argpartition(row, -effective_k)[-effective_k:]
        topk_idx = topk_idx[np.argsort(row[topk_idx])[::-1]]  # descending
        topk_scores = row[topk_idx]

        frame = frames[i] if i < len(frames) else None
        results.append(
            ClassificationResult(
                top1_class_id=int(topk_idx[0]),
                top1_score=float(topk_scores[0]),
                topk_class_ids=tuple(int(x) for x in topk_idx),
                topk_scores=tuple(float(x) for x in topk_scores),
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
