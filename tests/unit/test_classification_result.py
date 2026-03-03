"""Unit tests for ClassificationResult dataclass."""

from __future__ import annotations

import dataclasses
import json

import pytest

from yowo.types import BackendType, ClassificationResult, ModelFamily, ModelSize, ModelSpec


def _make_result(
    top1_class_id: int = 42,
    top1_score: float = 0.95,
    topk_class_ids: tuple[int, ...] = (42, 7, 13),
    topk_scores: tuple[float, ...] = (0.95, 0.03, 0.01),
    source_id: str = "cam0",
    frame_index: int = 5,
) -> ClassificationResult:
    spec = ModelSpec(ModelFamily.YOLO11, ModelSize.NANO, task="classify")
    all_probs = tuple(0.001 for _ in range(1000))
    # Patch specific indices
    probs_list = list(all_probs)
    probs_list[42] = 0.95
    probs_list[7] = 0.03
    probs_list[13] = 0.01
    return ClassificationResult(
        top1_class_id=top1_class_id,
        top1_score=top1_score,
        topk_class_ids=topk_class_ids,
        topk_scores=topk_scores,
        all_probs=tuple(probs_list),
        source_id=source_id,
        frame_index=frame_index,
        inference_time_ms=1.5,
        backend=BackendType.PYTORCH,
        model_spec=spec,
    )


class TestClassificationResult:
    def test_fields_exist(self) -> None:
        """All 10 fields are present on the dataclass."""
        expected_fields = {
            "top1_class_id",
            "top1_score",
            "topk_class_ids",
            "topk_scores",
            "all_probs",
            "source_id",
            "frame_index",
            "inference_time_ms",
            "backend",
            "model_spec",
        }
        result = _make_result()
        actual_fields = {f.name for f in dataclasses.fields(result)}
        assert expected_fields == actual_fields

    def test_frozen(self) -> None:
        """Assignment raises FrozenInstanceError — dataclass is frozen."""
        result = _make_result()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.top1_class_id = 0  # type: ignore[misc]

    def test_to_dict_keys(self) -> None:
        """to_dict() contains all required serialization keys."""
        result = _make_result()
        d = result.to_dict()
        expected_keys = {
            "source_id",
            "frame_index",
            "inference_time_ms",
            "backend",
            "model",
            "top1_class_id",
            "top1_score",
            "topk",
        }
        assert set(d.keys()) == expected_keys

    def test_to_json_valid(self) -> None:
        """to_json() produces valid JSON."""
        result = _make_result()
        parsed = json.loads(result.to_json())
        assert isinstance(parsed, dict)
        assert parsed["top1_class_id"] == 42

    def test_topk_structure(self) -> None:
        """to_dict()['topk'] is a list of dicts with 'class_id' and 'score' keys."""
        result = _make_result()
        topk = result.to_dict()["topk"]
        assert isinstance(topk, list)
        assert len(topk) == 3
        for entry in topk:
            assert isinstance(entry, dict)
            assert "class_id" in entry
            assert "score" in entry

    def test_model_field_includes_cls_suffix(self) -> None:
        """to_dict()['model'] ends with '-cls'."""
        result = _make_result()
        assert result.to_dict()["model"].endswith("-cls")

    def test_all_probs_excluded_from_to_dict(self) -> None:
        """all_probs (1000 floats) is NOT included in to_dict() output."""
        result = _make_result()
        assert "all_probs" not in result.to_dict()
