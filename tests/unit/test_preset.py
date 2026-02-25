"""Tests for preset inference auto-tuning."""

from __future__ import annotations

from pathlib import Path

import pytest

from yowo.config import classify_device, classify_source
from yowo.errors import ConfigError
from yowo.hardware import HardwareProfile
from yowo.hardware._capabilities import InstalledLibraries
from yowo.hardware._device import Device
from yowo.types import CPUArch, DeviceCategory, DeviceType, GPUArch, SourceCategory


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


# ---------------------------------------------------------------------------
# classify_source tests
# ---------------------------------------------------------------------------


class TestClassifySource:
    def test_image_jpg(self) -> None:
        assert classify_source("photo.jpg") == SourceCategory.IMAGE

    def test_image_png(self) -> None:
        assert classify_source(Path("photo.png")) == SourceCategory.IMAGE

    def test_image_bmp(self) -> None:
        assert classify_source("scan.bmp") == SourceCategory.IMAGE

    def test_image_webp(self) -> None:
        assert classify_source("img.webp") == SourceCategory.IMAGE

    def test_video_mp4(self) -> None:
        assert classify_source("clip.mp4") == SourceCategory.VIDEO

    def test_video_avi(self) -> None:
        assert classify_source("clip.avi") == SourceCategory.VIDEO

    def test_video_mov(self) -> None:
        assert classify_source(Path("clip.mov")) == SourceCategory.VIDEO

    def test_video_mkv(self) -> None:
        assert classify_source("clip.mkv") == SourceCategory.VIDEO

    def test_rtsp_url(self) -> None:
        assert classify_source("rtsp://192.168.1.1/stream") == SourceCategory.LIVE_STREAM

    def test_rtsps_url(self) -> None:
        assert classify_source("rtsps://cam.example.com/feed") == SourceCategory.LIVE_STREAM

    def test_webcam_zero(self) -> None:
        assert classify_source("0") == SourceCategory.LIVE_STREAM

    def test_webcam_one(self) -> None:
        assert classify_source("1") == SourceCategory.LIVE_STREAM

    def test_directory(self, tmp_path: Path) -> None:
        assert classify_source(tmp_path) == SourceCategory.IMAGE

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ConfigError):
            classify_source("data.xyz")


# ---------------------------------------------------------------------------
# classify_device tests
# ---------------------------------------------------------------------------


def _cpu_device(
    arch: CPUArch = CPUArch.X86_64,
    is_jetson: bool = False,
    memory_mb: int = 16384,
) -> Device:
    return Device(
        type=DeviceType.CPU,
        index=0,
        name="test-cpu",
        arch=None,
        cpu_arch=arch,
        memory_total_mb=memory_mb,
        memory_available_mb=memory_mb // 2,
        is_jetson=is_jetson,
    )


def _gpu_device(
    vram_mb: int = 8192,
    gpu_arch: GPUArch = GPUArch.AMPERE,
    is_jetson: bool = False,
) -> Device:
    return Device(
        type=DeviceType.CUDA,
        index=0,
        name="test-gpu",
        arch=gpu_arch,
        cpu_arch=CPUArch.X86_64,
        memory_total_mb=vram_mb,
        memory_available_mb=vram_mb // 2,
        is_jetson=is_jetson,
    )


def _libs(**kwargs: object) -> InstalledLibraries:
    return InstalledLibraries(**kwargs)  # type: ignore[arg-type]


def _hw(
    gpus: tuple[Device, ...] = (),
    cpu: Device | None = None,
    libs: InstalledLibraries | None = None,
) -> HardwareProfile:
    return HardwareProfile(
        gpus=gpus,
        cpu=cpu or _cpu_device(),
        cpu_features=frozenset(),
        libraries=libs or _libs(),
    )


class TestClassifyDevice:
    def test_jetson(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.AARCH64, is_jetson=True))
        assert classify_device(hw) == DeviceCategory.JETSON

    def test_cuda_high_vram(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=24576),))
        assert classify_device(hw) == DeviceCategory.CUDA_HIGH

    def test_cuda_high_boundary(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=8192),))
        assert classify_device(hw) == DeviceCategory.CUDA_HIGH

    def test_cuda_low_vram(self) -> None:
        hw = _hw(gpus=(_gpu_device(vram_mb=4096),))
        assert classify_device(hw) == DeviceCategory.CUDA_LOW

    def test_apple_silicon(self) -> None:
        hw = _hw(
            cpu=_cpu_device(arch=CPUArch.AARCH64),
            libs=_libs(onnxruntime_version="1.17.0", onnxruntime_has_coreml=True),
        )
        assert classify_device(hw) == DeviceCategory.APPLE_SILICON

    def test_cpu_x86(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.X86_64))
        assert classify_device(hw) == DeviceCategory.CPU_X86

    def test_cpu_arm_no_coreml(self) -> None:
        hw = _hw(cpu=_cpu_device(arch=CPUArch.AARCH64))
        assert classify_device(hw) == DeviceCategory.CPU_ARM

    def test_jetson_priority_over_cuda(self) -> None:
        """Jetson has a GPU but should be classified as JETSON, not CUDA."""
        jetson_gpu = _gpu_device(vram_mb=8192, is_jetson=True)
        hw = _hw(
            gpus=(jetson_gpu,),
            cpu=_cpu_device(arch=CPUArch.AARCH64, is_jetson=True),
        )
        assert classify_device(hw) == DeviceCategory.JETSON
