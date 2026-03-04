"""Unit tests verifying DetectionEngine rename and InferenceEngine alias."""

from __future__ import annotations


class TestInferenceEngineAlias:
    def test_inference_engine_is_detection_engine(self) -> None:
        """InferenceEngine is exactly DetectionEngine (same object)."""
        from yowo.engine import DetectionEngine, InferenceEngine

        assert InferenceEngine is DetectionEngine

    def test_detection_engine_has_detect_method(self) -> None:
        """DetectionEngine exposes a detect() method."""
        from yowo.engine import DetectionEngine

        assert hasattr(DetectionEngine, "detect")
        assert callable(DetectionEngine.detect)

    def test_inference_engine_has_detect_method(self) -> None:
        """InferenceEngine (alias) also exposes detect() method."""
        from yowo.engine import InferenceEngine

        assert hasattr(InferenceEngine, "detect")
        assert callable(InferenceEngine.detect)

    def test_detection_engine_importable_from_yowo(self) -> None:
        """DetectionEngine can be imported directly from the yowo package."""
        from yowo import DetectionEngine

        assert DetectionEngine is not None

    def test_inference_engine_importable_from_yowo(self) -> None:
        """InferenceEngine can be imported directly from the yowo package."""
        from yowo import InferenceEngine

        assert InferenceEngine is not None

    def test_both_names_refer_to_same_class(self) -> None:
        """Importing both names from yowo yields the same class object."""
        from yowo import DetectionEngine, InferenceEngine

        assert InferenceEngine is DetectionEngine
