"""Unit tests for postprocess_classify(), _is_softmaxed(), _softmax()."""

from __future__ import annotations

import numpy as np

from yowo.postprocess._classify import _is_softmaxed, _softmax, postprocess_classify
from yowo.types import BackendType, Frame, ModelFamily, ModelSize, ModelSpec


def _make_spec() -> ModelSpec:
    return ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, task="classify")


def _make_frame(source_id: str = "test", frame_index: int = 0) -> Frame:
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    return Frame(pixels=pixels, source_id=source_id, frame_index=frame_index)


class TestPostprocessClassify:
    def test_basic_topk(self) -> None:
        """top1_class_id is index with highest logit; scores are sorted descending."""
        logits = np.array([[0.1, 5.0, 0.3, 0.2, 0.05]], dtype=np.float32)
        results = postprocess_classify(
            logits,
            [_make_frame()],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
            top_k=3,
        )
        assert len(results) == 1
        r = results[0]
        assert r.top1_class_id == 1
        # Scores must be sorted descending
        for i in range(len(r.topk_scores) - 1):
            assert r.topk_scores[i] >= r.topk_scores[i + 1]

    def test_softmax_applied_to_raw_logits(self) -> None:
        """_is_softmaxed returns False for raw logits; after postprocess probs sum to 1."""
        logits = np.array([[2.0, 1.0, 0.5, -1.0, 3.0]], dtype=np.float32)
        assert _is_softmaxed(logits) is False

        results = postprocess_classify(
            logits,
            [_make_frame()],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
            top_k=5,
        )
        total = sum(results[0].all_probs)
        assert abs(total - 1.0) < 1e-3

    def test_already_softmaxed_not_double_applied(self) -> None:
        """Pre-softmaxed input passes through unchanged (sums stay ~1.0)."""
        raw = np.array([[2.0, 1.0, 0.5, -1.0, 3.0]], dtype=np.float32)
        softmaxed = _softmax(raw)
        assert _is_softmaxed(softmaxed) is True

        results = postprocess_classify(
            softmaxed,
            [_make_frame()],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
            top_k=5,
        )
        total = sum(results[0].all_probs)
        assert abs(total - 1.0) < 1e-3

    def test_topk_clamped_to_nc(self) -> None:
        """top_k > nc is clamped to nc."""
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # nc=3
        results = postprocess_classify(
            logits,
            [_make_frame()],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
            top_k=10,
        )
        assert len(results[0].topk_class_ids) == 3

    def test_batch_of_3(self) -> None:
        """Batch of 3 frames returns 3 results."""
        logits = np.random.rand(3, 10).astype(np.float32)
        frames = [_make_frame(frame_index=i) for i in range(3)]
        results = postprocess_classify(
            logits,
            frames,
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
            top_k=5,
        )
        assert len(results) == 3

    def test_frame_metadata_propagated(self) -> None:
        """source_id and frame_index are correctly propagated from Frame."""
        logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        frame = _make_frame(source_id="cam0", frame_index=42)
        results = postprocess_classify(
            logits,
            [frame],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
            top_k=3,
        )
        assert results[0].source_id == "cam0"
        assert results[0].frame_index == 42

    def test_empty_frames_fallback(self) -> None:
        """frames=[] falls back to empty source_id and positional frame_index=0."""
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        results = postprocess_classify(
            logits,
            [],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
            top_k=2,
        )
        assert len(results) == 1
        assert results[0].source_id == ""
        assert results[0].frame_index == 0

    def test_top1_score_in_range(self) -> None:
        """top1_score is in [0, 1]."""
        logits = np.random.randn(1, 100).astype(np.float32)
        results = postprocess_classify(
            logits,
            [_make_frame()],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
        )
        assert 0.0 <= results[0].top1_score <= 1.0

    def test_all_probs_sum_to_1(self) -> None:
        """all_probs sum to ~1.0."""
        logits = np.random.randn(1, 1000).astype(np.float32)
        results = postprocess_classify(
            logits,
            [_make_frame()],
            model_spec=_make_spec(),
            backend=BackendType.PYTORCH,
        )
        total = sum(results[0].all_probs)
        assert abs(total - 1.0) < 1e-3

    def test_is_softmaxed_true(self) -> None:
        """_is_softmaxed returns True for a proper softmax output."""
        raw = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        softmaxed = _softmax(raw)
        assert _is_softmaxed(softmaxed) is True

    def test_is_softmaxed_false(self) -> None:
        """_is_softmaxed returns False for raw logits."""
        raw = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        assert _is_softmaxed(raw) is False
