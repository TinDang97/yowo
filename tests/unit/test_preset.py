"""Tests for preset inference auto-tuning."""

from __future__ import annotations

from yowo.types import DeviceCategory, SourceCategory


class TestSourceCategory:
    def test_values(self) -> None:
        assert SourceCategory.IMAGE == "image"
        assert SourceCategory.VIDEO == "video"
        assert SourceCategory.LIVE_STREAM == "live"

    def test_is_str_enum(self) -> None:
        assert isinstance(SourceCategory.IMAGE, str)


class TestDeviceCategory:
    def test_values(self) -> None:
        assert DeviceCategory.CUDA_HIGH == "cuda_high"
        assert DeviceCategory.CUDA_LOW == "cuda_low"
        assert DeviceCategory.JETSON == "jetson"
        assert DeviceCategory.APPLE_SILICON == "apple_silicon"
        assert DeviceCategory.CPU_X86 == "cpu_x86"
        assert DeviceCategory.CPU_ARM == "cpu_arm"

    def test_is_str_enum(self) -> None:
        assert isinstance(DeviceCategory.CUDA_HIGH, str)
