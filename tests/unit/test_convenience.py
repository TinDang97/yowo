"""Tests for yowo._convenience: parse_model_name() and detect()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yowo._convenience import detect, parse_model_name
from yowo.errors import ConfigError
from yowo.types import ModelFamily, ModelSize, ModelSpec

# ---------------------------------------------------------------------------
# parse_model_name
# ---------------------------------------------------------------------------


class TestParseModelName:
    """Exhaustive coverage for parse_model_name()."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("yolo11n", ModelSpec(ModelFamily.YOLO11, ModelSize.NANO)),
            ("yolo11s", ModelSpec(ModelFamily.YOLO11, ModelSize.SMALL)),
            ("yolo11m", ModelSpec(ModelFamily.YOLO11, ModelSize.MEDIUM)),
            ("yolo11l", ModelSpec(ModelFamily.YOLO11, ModelSize.LARGE)),
            ("yolo11x", ModelSpec(ModelFamily.YOLO11, ModelSize.XLARGE)),
            ("yolo26n", ModelSpec(ModelFamily.YOLO26, ModelSize.NANO)),
            ("yolo26s", ModelSpec(ModelFamily.YOLO26, ModelSize.SMALL)),
            ("yolo26m", ModelSpec(ModelFamily.YOLO26, ModelSize.MEDIUM)),
            ("yolo26l", ModelSpec(ModelFamily.YOLO26, ModelSize.LARGE)),
            ("yolo26x", ModelSpec(ModelFamily.YOLO26, ModelSize.XLARGE)),
        ],
    )
    def test_valid_variants(self, name: str, expected: ModelSpec) -> None:
        assert parse_model_name(name) == expected

    def test_invalid_family(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model"):
            parse_model_name("yolo99n")

    def test_invalid_size(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model"):
            parse_model_name("yolo26z")

    def test_empty_string(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model"):
            parse_model_name("")


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


class TestDetect:
    """detect() wiring — engine + source mocked."""

    def test_default_wiring(self, tmp_path: pytest.TempPathFactory) -> None:
        img = tmp_path / "photo.jpg"  # type: ignore[operator]
        img.touch()

        mock_det = MagicMock()
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = [mock_det]

        with (
            patch("yowo.engine.InferenceEngine", return_value=mock_engine) as eng_cls,
            patch("yowo.io.open_source") as os_fn,
        ):
            result = detect(str(img))

        eng_cls.assert_called_once_with(
            model_family=ModelFamily.YOLO26,
            model_size=ModelSize.NANO,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device="auto",
        )
        os_fn.assert_called_once_with(str(img))
        assert result == [mock_det]

    def test_custom_model_and_thresholds(self) -> None:
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = []

        with (
            patch("yowo.engine.InferenceEngine", return_value=mock_engine) as eng_cls,
            patch("yowo.io.open_source"),
        ):
            detect("video.mp4", model="yolo11x", confidence=0.5, iou=0.6)

        eng_cls.assert_called_once_with(
            model_family=ModelFamily.YOLO11,
            model_size=ModelSize.XLARGE,
            confidence_threshold=0.5,
            iou_threshold=0.6,
            device="auto",
        )

    def test_int_source_forwarded(self) -> None:
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = []

        with (
            patch("yowo.engine.InferenceEngine", return_value=mock_engine),
            patch("yowo.io.open_source") as os_fn,
        ):
            detect(0)

        os_fn.assert_called_once_with(0)

    def test_extra_kwargs_forwarded(self) -> None:
        mock_engine = MagicMock()
        mock_engine.__enter__ = MagicMock(return_value=mock_engine)
        mock_engine.__exit__ = MagicMock(return_value=False)
        mock_engine.stream.return_value = []

        with (
            patch("yowo.engine.InferenceEngine", return_value=mock_engine) as eng_cls,
            patch("yowo.io.open_source"),
        ):
            detect("img.jpg", batch_size=4, cache=True)

        eng_cls.assert_called_once_with(
            model_family=ModelFamily.YOLO26,
            model_size=ModelSize.NANO,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device="auto",
            batch_size=4,
            cache=True,
        )

    def test_invalid_model_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model"):
            detect("img.jpg", model="badmodel")
