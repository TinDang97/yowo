"""Comprehensive unit tests for yowo.hardware.

Tests cover:
- detect_cpu_arch()         — machine string normalisation
- detect_system_memory_mb() — /proc/meminfo parsing + graceful fallback
- detect_is_jetson()        — file-system flags
- detect_gpu_arch()         — compute-capability mapping
- detect_gpus()             — pynvml primary, nvidia-smi fallback
- _detect_gpus_smi()        — CSV parsing, incl. Jetson "[N/A]" memory columns
- probe_tensorrt()          — import success / ImportError
- probe_onnxruntime()       — CUDA provider detection
- HardwareProfile           — properties: has_nvidia_gpu, is_jetson, primary_gpu
- get_hardware_profile()    — caching, force_refresh, clear_cache
- Device.supports_fp16()    — arch-gated FP16
- Device.supports_int8()    — arch-gated INT8
- Device.__str__()          — human-readable formatting
"""

from __future__ import annotations

import io
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from yowo.hardware import HardwareProfile, clear_cache, get_hardware_profile
from yowo.hardware._capabilities import (
    InstalledLibraries,
    probe_onnxruntime,
    probe_tensorrt,
    probe_torch,
)
from yowo.hardware._detect import (
    _detect_gpus_smi,
    _parse_compute_cap,
    detect_cpu_arch,
    detect_cpu_features,
    detect_gpu_arch,
    detect_gpus,
    detect_is_jetson,
    detect_system_memory_mb,
)
from yowo.hardware._device import Device
from yowo.types import CPUArch, DeviceType, GPUArch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gpu(
    *,
    index: int = 0,
    name: str = "NVIDIA Test GPU",
    arch: GPUArch = GPUArch.TURING,
    memory_total_mb: int = 8192,
    memory_available_mb: int = 7000,
    is_jetson: bool = False,
    cpu_arch: CPUArch = CPUArch.X86_64,
) -> Device:
    return Device(
        type=DeviceType.CUDA,
        index=index,
        name=name,
        arch=arch,
        cpu_arch=cpu_arch,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        is_jetson=is_jetson,
    )


def _make_cpu(
    *,
    name: str = "Intel Core i7",
    cpu_arch: CPUArch = CPUArch.X86_64,
    memory_total_mb: int = 16384,
    memory_available_mb: int = 10000,
    is_jetson: bool = False,
) -> Device:
    return Device(
        type=DeviceType.CPU,
        index=0,
        name=name,
        arch=None,
        cpu_arch=cpu_arch,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        is_jetson=is_jetson,
    )


def _make_libraries(**kwargs: Any) -> InstalledLibraries:
    return InstalledLibraries(**kwargs)


def _make_profile(
    gpus: tuple[Device, ...] = (),
    cpu: Device | None = None,
    cpu_features: frozenset[str] | None = None,
    libraries: InstalledLibraries | None = None,
) -> HardwareProfile:
    return HardwareProfile(
        gpus=gpus,
        cpu=cpu or _make_cpu(),
        cpu_features=cpu_features if cpu_features is not None else frozenset(),
        libraries=libraries or InstalledLibraries(),
    )


# ---------------------------------------------------------------------------
# detect_cpu_arch
# ---------------------------------------------------------------------------


class TestDetectCpuArch:
    def test_x86_64_returns_x86_64(self) -> None:
        with patch("platform.machine", return_value="x86_64"):
            assert detect_cpu_arch() == CPUArch.X86_64

    def test_aarch64_returns_aarch64(self) -> None:
        with patch("platform.machine", return_value="aarch64"):
            assert detect_cpu_arch() == CPUArch.AARCH64

    def test_arm64_macos_returns_aarch64(self) -> None:
        with patch("platform.machine", return_value="arm64"):
            assert detect_cpu_arch() == CPUArch.AARCH64

    def test_unknown_machine_defaults_to_x86_64(self) -> None:
        with patch("platform.machine", return_value="riscv64"):
            assert detect_cpu_arch() == CPUArch.X86_64

    def test_case_insensitive(self) -> None:
        with patch("platform.machine", return_value="ARM64"):
            assert detect_cpu_arch() == CPUArch.AARCH64


# ---------------------------------------------------------------------------
# detect_system_memory_mb
# ---------------------------------------------------------------------------

_PROC_MEMINFO = """\
MemTotal:       32768000 kB
MemFree:         4096000 kB
MemAvailable:   16384000 kB
Buffers:          512000 kB
Cached:          8192000 kB
"""


class TestDetectSystemMemoryMb:
    def test_linux_parses_proc_meminfo(self) -> None:
        mock_open = MagicMock(return_value=io.StringIO(_PROC_MEMINFO))
        with (
            patch("platform.system", return_value="Linux"),
            patch("yowo.hardware._detect.Path.open", mock_open),
        ):
            total, available = detect_system_memory_mb()

        assert total == 32768000 // 1024  # kB -> MB
        assert available == 16384000 // 1024

    def test_linux_returns_zero_on_oserror(self) -> None:
        mock_open = MagicMock(side_effect=OSError("no file"))
        with (
            patch("platform.system", return_value="Linux"),
            patch("yowo.hardware._detect.Path.open", mock_open),
        ):
            total, available = detect_system_memory_mb()

        assert (total, available) == (0, 0)

    def test_unsupported_platform_returns_zeros(self) -> None:
        with patch("platform.system", return_value="Windows"):
            total, available = detect_system_memory_mb()

        assert (total, available) == (0, 0)

    def test_linux_parses_correct_fields(self) -> None:
        meminfo = "MemTotal:       8192000 kB\nMemAvailable:   4096000 kB\n"
        mock_open = MagicMock(return_value=io.StringIO(meminfo))
        with (
            patch("platform.system", return_value="Linux"),
            patch("yowo.hardware._detect.Path.open", mock_open),
        ):
            total, available = detect_system_memory_mb()

        assert total == 8000  # 8192000 // 1024
        assert available == 4000  # 4096000 // 1024


# ---------------------------------------------------------------------------
# detect_is_jetson
# ---------------------------------------------------------------------------


class TestDetectIsJetson:
    def test_tegra_release_file_present_returns_true(self) -> None:
        with patch("yowo.hardware._detect._JETSON_TEGRA_RELEASE") as mock_path:
            mock_path.exists.return_value = True
            assert detect_is_jetson() is True

    def test_device_tree_model_contains_jetson_returns_true(self) -> None:
        with (
            patch("yowo.hardware._detect._JETSON_TEGRA_RELEASE") as mock_tegra,
            patch("yowo.hardware._detect._JETSON_DEVICE_TREE") as mock_dt,
        ):
            mock_tegra.exists.return_value = False
            mock_dt.exists.return_value = True
            mock_dt.read_text.return_value = "NVIDIA Jetson AGX Orin\x00"
            assert detect_is_jetson() is True

    def test_no_jetson_files_returns_false(self) -> None:
        with (
            patch("yowo.hardware._detect._JETSON_TEGRA_RELEASE") as mock_tegra,
            patch("yowo.hardware._detect._JETSON_DEVICE_TREE") as mock_dt,
        ):
            mock_tegra.exists.return_value = False
            mock_dt.exists.return_value = False
            assert detect_is_jetson() is False

    def test_device_tree_non_jetson_model_returns_false(self) -> None:
        with (
            patch("yowo.hardware._detect._JETSON_TEGRA_RELEASE") as mock_tegra,
            patch("yowo.hardware._detect._JETSON_DEVICE_TREE") as mock_dt,
        ):
            mock_tegra.exists.return_value = False
            mock_dt.exists.return_value = True
            mock_dt.read_text.return_value = "Raspberry Pi 4 Model B"
            assert detect_is_jetson() is False

    def test_oserror_on_read_returns_false(self) -> None:
        with (
            patch("yowo.hardware._detect._JETSON_TEGRA_RELEASE") as mock_tegra,
            patch("yowo.hardware._detect._JETSON_DEVICE_TREE") as mock_dt,
        ):
            mock_tegra.exists.return_value = False
            mock_dt.exists.return_value = True
            mock_dt.read_text.side_effect = OSError("permission denied")
            assert detect_is_jetson() is False


# ---------------------------------------------------------------------------
# detect_gpu_arch
# ---------------------------------------------------------------------------


class TestDetectGpuArch:
    @pytest.mark.parametrize(
        ("major", "minor", "expected"),
        [
            (7, 5, GPUArch.TURING),
            (8, 0, GPUArch.AMPERE),
            (8, 6, GPUArch.AMPERE_GA10X),
            (8, 7, GPUArch.ORIN),
            (8, 9, GPUArch.ADA),
            (9, 0, GPUArch.HOPPER),
            (8, 3, GPUArch.UNKNOWN),  # no mapping
            (6, 1, GPUArch.UNKNOWN),  # Pascal — not in table
            (10, 0, GPUArch.UNKNOWN),  # future
        ],
    )
    def test_compute_capability_mapping(self, major: int, minor: int, expected: GPUArch) -> None:
        assert detect_gpu_arch(major, minor) == expected


# ---------------------------------------------------------------------------
# detect_gpus / _detect_gpus_smi
# ---------------------------------------------------------------------------

# What nvidia-smi actually prints on a Jetson Orin (JetPack 6.2, L4T R36.4.4).
# The memory columns are placeholders because the iGPU shares system RAM.
_SMI_JETSON = "0, Orin (nvgpu), [N/A], [N/A]"
_SMI_DISCRETE = "0, NVIDIA L4, 23034, 22800"


def _smi_result(stdout: str, returncode: int = 0) -> Any:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


class TestDetectGpusSmi:
    """The nvidia-smi fallback, including the Jetson unified-memory case.

    Regression guard: ``int("[N/A]")`` raised ValueError inside the parse loop,
    ``detect_gpus`` swallowed it by contract, and the whole board came back as
    "no GPU" — which made ``has_nvidia_gpu`` False and pushed a perfectly good
    Orin onto the CPU backend with no error anywhere.
    """

    def test_jetson_placeholder_memory_still_yields_the_device(self) -> None:
        with (
            patch(
                "yowo.hardware._detect.subprocess.run",
                return_value=_smi_result(_SMI_JETSON),
            ),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            patch("yowo.hardware._detect._get_compute_capability", return_value=(8, 7)),
        ):
            devices = _detect_gpus_smi()

        assert len(devices) == 1
        gpu = devices[0]
        assert gpu.type == DeviceType.CUDA
        assert gpu.index == 0
        assert gpu.name == "Orin (nvgpu)"
        assert gpu.is_jetson is True
        assert gpu.arch == GPUArch.ORIN

    def test_jetson_memory_falls_back_to_system_ram(self) -> None:
        """Unified memory: system RAM is the GPU's budget, so report it as such."""
        with (
            patch(
                "yowo.hardware._detect.subprocess.run",
                return_value=_smi_result(_SMI_JETSON),
            ),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            patch("yowo.hardware._detect._get_compute_capability", return_value=(8, 7)),
        ):
            gpu = _detect_gpus_smi()[0]

        assert gpu.memory_total_mb == 7620
        assert gpu.memory_available_mb == 4100

    def test_discrete_gpu_memory_is_read_from_smi_not_system_ram(self) -> None:
        with (
            patch(
                "yowo.hardware._detect.subprocess.run",
                return_value=_smi_result(_SMI_DISCRETE),
            ),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=False),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(64000, 32000)),
            patch("yowo.hardware._detect._get_compute_capability", return_value=(8, 9)),
        ):
            gpu = _detect_gpus_smi()[0]

        assert gpu.memory_total_mb == 23034
        assert gpu.memory_available_mb == 22800
        assert gpu.is_jetson is False

    @pytest.mark.parametrize(
        "placeholder",
        ["[N/A]", "N/A", "[Not Supported]", "Unknown", ""],
    )
    def test_every_known_placeholder_is_tolerated(self, placeholder: str) -> None:
        line = f"0, Orin (nvgpu), {placeholder}, {placeholder}"
        with (
            patch("yowo.hardware._detect.subprocess.run", return_value=_smi_result(line)),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            patch("yowo.hardware._detect._get_compute_capability", return_value=None),
        ):
            devices = _detect_gpus_smi()

        assert len(devices) == 1
        assert devices[0].memory_total_mb == 7620

    def test_only_the_memory_column_is_forgiving_a_bad_index_drops_the_row(self) -> None:
        """An unparseable index means we cannot address the device — skip it."""
        line = "GPU, Orin (nvgpu), [N/A], [N/A]"
        with (
            patch("yowo.hardware._detect.subprocess.run", return_value=_smi_result(line)),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
        ):
            assert _detect_gpus_smi() == []

    def test_nonzero_returncode_yields_no_devices(self) -> None:
        with patch(
            "yowo.hardware._detect.subprocess.run",
            return_value=_smi_result("", returncode=9),
        ):
            assert _detect_gpus_smi() == []


# nvidia-smi with the compute_cap column (real output on driver 540+, incl. Jetson).
_SMI_JETSON_CC = "0, Orin (nvgpu), [N/A], [N/A], 8.7"
_SMI_DISCRETE_CC = "0, NVIDIA L4, 23034, 22800, 8.9"


class TestParseComputeCap:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("8.7", (8, 7)),
            (" 8.9 ", (8, 9)),
            ("7.5", (7, 5)),
            ("[N/A]", None),
            ("N/A", None),
            ("", None),
            ("87", None),  # no dot - not a capability
            ("Orin", None),
        ],
    )
    def test_parse(self, value, expected) -> None:
        assert _parse_compute_cap(value) == expected


class TestComputeCapFromSmi:
    """The architecture comes off nvidia-smi, so it is correct WITHOUT torch.

    This is what lets Device.supports_fp16() report True on a Jetson that has
    no torch: the old code left arch UNKNOWN there, so fp16/int8 both read as
    unsupported on a board that supports both.
    """

    def test_jetson_arch_is_orin_without_torch(self) -> None:
        with (
            patch("yowo.hardware._detect.subprocess.run", return_value=_smi_result(_SMI_JETSON_CC)),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            # torch probe returns None (no torch); arch must still be ORIN.
            patch("yowo.hardware._detect._get_compute_capability", return_value=None),
        ):
            gpu = _detect_gpus_smi()[0]

        assert gpu.arch == GPUArch.ORIN
        assert gpu.supports_fp16() is True
        assert gpu.supports_int8() is True

    def test_discrete_arch_from_compute_cap(self) -> None:
        with (
            patch(
                "yowo.hardware._detect.subprocess.run", return_value=_smi_result(_SMI_DISCRETE_CC)
            ),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=False),
            patch("yowo.hardware._detect._get_compute_capability", return_value=None),
        ):
            gpu = _detect_gpus_smi()[0]

        assert gpu.arch == GPUArch.ADA  # 8.9

    def test_smi_compute_cap_wins_over_torch_probe(self) -> None:
        """When nvidia-smi reports it, the torch probe is not even called."""
        with (
            patch("yowo.hardware._detect.subprocess.run", return_value=_smi_result(_SMI_JETSON_CC)),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            patch("yowo.hardware._detect._get_compute_capability") as torch_probe,
        ):
            gpu = _detect_gpus_smi()[0]

        assert gpu.arch == GPUArch.ORIN
        torch_probe.assert_not_called()

    def test_placeholder_compute_cap_falls_back_to_torch(self) -> None:
        """An old driver prints [N/A] for compute_cap; torch fills the gap."""
        line = "0, Orin (nvgpu), [N/A], [N/A], [N/A]"
        with (
            patch("yowo.hardware._detect.subprocess.run", return_value=_smi_result(line)),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            patch("yowo.hardware._detect._get_compute_capability", return_value=(8, 7)),
        ):
            gpu = _detect_gpus_smi()[0]

        assert gpu.arch == GPUArch.ORIN


class TestDetectGpus:
    def test_jetson_is_visible_through_the_public_entry_point(self) -> None:
        """End-to-end for the bug: pynvml absent, nvidia-smi reports [N/A]."""
        with (
            patch(
                "yowo.hardware._detect._detect_gpus_nvml",
                side_effect=ImportError("No module named 'pynvml'"),
            ),
            patch(
                "yowo.hardware._detect.subprocess.run",
                return_value=_smi_result(_SMI_JETSON),
            ),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            patch("yowo.hardware._detect._get_compute_capability", return_value=(8, 7)),
        ):
            devices = detect_gpus()

        assert len(devices) == 1, "a Jetson must not read as a machine with no GPU"

    def test_empty_nvml_result_still_falls_through_to_smi(self) -> None:
        """NVML loads on some Tegra builds but reports zero devices."""
        with (
            patch("yowo.hardware._detect._detect_gpus_nvml", return_value=[]),
            patch(
                "yowo.hardware._detect.subprocess.run",
                return_value=_smi_result(_SMI_JETSON),
            ),
            patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
            patch("yowo.hardware._detect.detect_system_memory_mb", return_value=(7620, 4100)),
            patch("yowo.hardware._detect._get_compute_capability", return_value=(8, 7)),
        ):
            devices = detect_gpus()

        assert len(devices) == 1

    def test_nvml_result_wins_when_it_is_not_empty(self) -> None:
        nvml_gpu = _make_gpu(name="NVIDIA L4")
        with (
            patch("yowo.hardware._detect._detect_gpus_nvml", return_value=[nvml_gpu]),
            patch("yowo.hardware._detect.subprocess.run") as mock_run,
        ):
            devices = detect_gpus()

        assert devices == [nvml_gpu]
        mock_run.assert_not_called()

    def test_both_probes_failing_returns_empty_and_never_raises(self) -> None:
        with (
            patch("yowo.hardware._detect._detect_gpus_nvml", side_effect=ImportError),
            patch(
                "yowo.hardware._detect.subprocess.run",
                side_effect=FileNotFoundError("nvidia-smi"),
            ),
        ):
            assert detect_gpus() == []


class TestHasNvidiaGpuOnJetson:
    def test_a_jetson_profile_reports_a_gpu(self) -> None:
        """The property the whole backend selector hangs off.

        ``has_nvidia_gpu`` False is what made ``TensorRTBackend.__init__``
        refuse with "an NVIDIA GPU is required" on a board that has one.
        """
        jetson_cpu = Device(
            type=DeviceType.CPU,
            index=0,
            name="aarch64 CPU",
            arch=None,
            cpu_arch=CPUArch.AARCH64,
            memory_total_mb=7620,
            memory_available_mb=4100,
            is_jetson=True,
        )
        clear_cache()
        try:
            # `subprocess.run` is patched on the shared module object, so it is
            # visible to anything else the profile builds - pin detect_cpu_device
            # rather than let it read the fake nvidia-smi output as a CPU name.
            with (
                patch(
                    "yowo.hardware._detect._detect_gpus_nvml",
                    side_effect=ImportError,
                ),
                patch(
                    "yowo.hardware._detect.subprocess.run",
                    return_value=_smi_result(_SMI_JETSON),
                ),
                patch("yowo.hardware.detect_cpu_device", return_value=jetson_cpu),
                patch("yowo.hardware._detect.detect_is_jetson", return_value=True),
                patch(
                    "yowo.hardware._detect.detect_system_memory_mb",
                    return_value=(7620, 4100),
                ),
                patch("yowo.hardware._detect._get_compute_capability", return_value=(8, 7)),
            ):
                profile = get_hardware_profile(force_refresh=True)

            assert profile.has_nvidia_gpu is True
            assert profile.is_jetson is True
            assert profile.primary_gpu is not None
            assert profile.primary_gpu.arch == GPUArch.ORIN
        finally:
            clear_cache()


# ---------------------------------------------------------------------------
# probe_tensorrt
# ---------------------------------------------------------------------------


class TestProbeTensorrt:
    def test_returns_version_when_installed(self) -> None:
        fake_trt = ModuleType("tensorrt")
        fake_trt.__version__ = "10.0.1"  # type: ignore[attr-defined]
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(sys.modules, {"tensorrt": fake_trt}),
        ):
            result = probe_tensorrt()
        assert result == "10.0.1"

    def test_returns_none_when_not_installed(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            result = probe_tensorrt()
        assert result is None

    def test_returns_none_on_import_error(self) -> None:
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch("builtins.__import__", side_effect=ImportError("no tensorrt")),
        ):
            result = probe_tensorrt()
        assert result is None


# ---------------------------------------------------------------------------
# probe_onnxruntime
# ---------------------------------------------------------------------------


class TestProbeOnnxruntime:
    def _make_ort_module(
        self, *, has_cuda: bool, has_coreml: bool = False, version: str = "1.17.0"
    ) -> ModuleType:
        fake_ort = ModuleType("onnxruntime")
        fake_ort.__version__ = version  # type: ignore[attr-defined]
        providers = ["CPUExecutionProvider"]
        if has_cuda:
            providers.append("CUDAExecutionProvider")
        if has_coreml:
            providers.append("CoreMLExecutionProvider")
        fake_ort.get_available_providers = lambda: providers  # type: ignore[attr-defined]
        return fake_ort

    def test_returns_version_and_cuda_true_when_cuda_provider_present(self) -> None:
        fake_ort = self._make_ort_module(has_cuda=True, version="1.17.0")
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(sys.modules, {"onnxruntime": fake_ort}),
        ):
            version, has_cuda, _has_coreml = probe_onnxruntime()
        assert version == "1.17.0"
        assert has_cuda is True

    def test_returns_version_and_cuda_false_when_no_cuda_provider(self) -> None:
        fake_ort = self._make_ort_module(has_cuda=False, version="1.17.0")
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(sys.modules, {"onnxruntime": fake_ort}),
        ):
            version, has_cuda, _has_coreml = probe_onnxruntime()
        assert version == "1.17.0"
        assert has_cuda is False

    def test_returns_none_when_not_installed(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            version, has_cuda, has_coreml = probe_onnxruntime()
        assert version is None
        assert has_cuda is False
        assert has_coreml is False

    def test_returns_coreml_true_when_coreml_provider_present(self) -> None:
        fake_ort = self._make_ort_module(has_cuda=False, has_coreml=True)
        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(sys.modules, {"onnxruntime": fake_ort}),
        ):
            version, _has_cuda, has_coreml = probe_onnxruntime()
        assert version == "1.17.0"
        assert has_coreml is True


# ---------------------------------------------------------------------------
# probe_torch
# ---------------------------------------------------------------------------


class TestProbeTorch:
    def test_returns_version_and_cuda_state(self) -> None:
        fake_torch = ModuleType("torch")
        fake_torch.__version__ = "2.3.0"  # type: ignore[attr-defined]
        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = True
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            version, cuda_available = probe_torch()

        assert version == "2.3.0"
        assert cuda_available is True

    def test_returns_none_when_not_installed(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            version, cuda_available = probe_torch()
        assert version is None
        assert cuda_available is False


# ---------------------------------------------------------------------------
# HardwareProfile properties
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    def test_has_nvidia_gpu_true_with_gpus(self) -> None:
        gpu = _make_gpu()
        profile = _make_profile(gpus=(gpu,))
        assert profile.has_nvidia_gpu is True

    def test_has_nvidia_gpu_false_with_no_gpus(self) -> None:
        profile = _make_profile(gpus=())
        assert profile.has_nvidia_gpu is False

    def test_primary_gpu_returns_first_gpu(self) -> None:
        gpu0 = _make_gpu(index=0, name="GPU 0")
        gpu1 = _make_gpu(index=1, name="GPU 1")
        profile = _make_profile(gpus=(gpu0, gpu1))
        assert profile.primary_gpu is gpu0

    def test_primary_gpu_none_when_no_gpus(self) -> None:
        profile = _make_profile(gpus=())
        assert profile.primary_gpu is None

    def test_is_jetson_delegates_to_cpu(self) -> None:
        cpu = _make_cpu(is_jetson=True)
        profile = _make_profile(cpu=cpu)
        assert profile.is_jetson is True

    def test_is_jetson_false_when_cpu_not_jetson(self) -> None:
        cpu = _make_cpu(is_jetson=False)
        profile = _make_profile(cpu=cpu)
        assert profile.is_jetson is False

    def test_profile_is_immutable(self) -> None:
        profile = _make_profile()
        with pytest.raises((AttributeError, TypeError)):
            profile.gpus = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_hardware_profile caching
# ---------------------------------------------------------------------------


class TestGetHardwareProfile:
    def setup_method(self) -> None:
        clear_cache()

    def teardown_method(self) -> None:
        clear_cache()

    def _patch_detectors(self) -> Any:
        """Return a context manager that patches all detection functions."""
        cpu = _make_cpu()
        return (
            patch("yowo.hardware.detect_gpus", return_value=[]),
            patch("yowo.hardware.detect_cpu_device", return_value=cpu),
            patch("yowo.hardware.detect_cpu_features", return_value=frozenset()),
            patch("yowo.hardware.detect_libraries", return_value=InstalledLibraries()),
        )

    def test_returns_hardware_profile_instance(self) -> None:
        patches = self._patch_detectors()
        with patches[0], patches[1], patches[2], patches[3]:
            profile = get_hardware_profile()
        assert isinstance(profile, HardwareProfile)

    def test_second_call_returns_same_object(self) -> None:
        patches = self._patch_detectors()
        with patches[0], patches[1], patches[2], patches[3]:
            first = get_hardware_profile()
            second = get_hardware_profile()
        assert first is second

    def test_force_refresh_re_detects(self) -> None:
        patches = self._patch_detectors()
        with patches[0], patches[1], patches[2], patches[3]:
            first = get_hardware_profile()
            second = get_hardware_profile(force_refresh=True)
        # Both are valid HardwareProfile objects but may or may not be same object;
        # what matters is the second call ran detection again (covered by mock call count)
        assert isinstance(second, HardwareProfile)
        assert first is not second  # new object created on force_refresh

    def test_clear_cache_causes_re_detection(self) -> None:
        patches = self._patch_detectors()
        with patches[0], patches[1], patches[2], patches[3]:
            first = get_hardware_profile()
            clear_cache()
            second = get_hardware_profile()
        assert first is not second


# ---------------------------------------------------------------------------
# Device.supports_fp16 and supports_int8
# ---------------------------------------------------------------------------


class TestDeviceCapabilities:
    @pytest.mark.parametrize(
        "arch",
        [
            GPUArch.TURING,
            GPUArch.AMPERE,
            GPUArch.AMPERE_GA10X,
            GPUArch.ORIN,
            GPUArch.ADA,
            GPUArch.HOPPER,
        ],
    )
    def test_gpu_turing_and_above_supports_fp16(self, arch: GPUArch) -> None:
        device = _make_gpu(arch=arch)
        assert device.supports_fp16() is True

    def test_gpu_unknown_arch_does_not_support_fp16(self) -> None:
        device = _make_gpu(arch=GPUArch.UNKNOWN)
        assert device.supports_fp16() is False

    def test_cpu_device_does_not_support_fp16(self) -> None:
        device = _make_cpu()
        assert device.supports_fp16() is False

    @pytest.mark.parametrize(
        "arch",
        [
            GPUArch.TURING,
            GPUArch.AMPERE,
            GPUArch.AMPERE_GA10X,
            GPUArch.ORIN,
            GPUArch.ADA,
            GPUArch.HOPPER,
        ],
    )
    def test_gpu_turing_and_above_supports_int8(self, arch: GPUArch) -> None:
        device = _make_gpu(arch=arch)
        assert device.supports_int8() is True

    def test_gpu_unknown_arch_does_not_support_int8(self) -> None:
        device = _make_gpu(arch=GPUArch.UNKNOWN)
        assert device.supports_int8() is False

    def test_cpu_device_does_not_support_int8(self) -> None:
        device = _make_cpu()
        assert device.supports_int8() is False

    def test_is_gpu_true_for_cuda_device(self) -> None:
        device = _make_gpu()
        assert device.is_gpu is True

    def test_is_gpu_false_for_cpu_device(self) -> None:
        device = _make_cpu()
        assert device.is_gpu is False


# ---------------------------------------------------------------------------
# Device.__str__
# ---------------------------------------------------------------------------


class TestDeviceStr:
    def test_gpu_str_contains_name_index_arch_memory(self) -> None:
        device = _make_gpu(
            name="NVIDIA L4",
            index=0,
            arch=GPUArch.ADA,
            memory_total_mb=24576,
        )
        result = str(device)
        assert "NVIDIA L4" in result
        assert "cuda:0" in result
        assert "sm_89" in result
        assert "24576MB" in result

    def test_cpu_str_contains_name_and_memory(self) -> None:
        device = _make_cpu(name="aarch64 CPU", memory_total_mb=16384)
        result = str(device)
        assert "aarch64 CPU" in result
        assert "16384MB" in result
        assert "RAM" in result

    def test_gpu_str_does_not_contain_ram_label(self) -> None:
        device = _make_gpu()
        assert "RAM" not in str(device)


# ---------------------------------------------------------------------------
# detect_cpu_features (smoke tests — /proc/cpuinfo not present on all CI)
# ---------------------------------------------------------------------------


class TestDetectCpuFeatures:
    def test_returns_frozenset(self) -> None:
        result = detect_cpu_features()
        assert isinstance(result, frozenset)

    def test_avx2_detected_from_cpuinfo(self) -> None:
        cpuinfo_content = "flags\t\t: fpu avx2 avx512f avx512vnni sse4_1\n"
        mock_open = MagicMock(return_value=io.StringIO(cpuinfo_content))
        with (
            patch("platform.machine", return_value="x86_64"),
            patch("yowo.hardware._detect.Path.exists", return_value=True),
            patch("yowo.hardware._detect.Path.open", mock_open),
        ):
            result = detect_cpu_features()
        assert "AVX2" in result
        assert "AVX512" in result
        assert "VNNI" in result

    def test_returns_empty_frozenset_on_exception(self) -> None:
        with (
            patch("platform.machine", return_value="x86_64"),
            patch(
                "yowo.hardware._detect.detect_cpu_arch",
                side_effect=RuntimeError("probe failure"),
            ),
        ):
            result = detect_cpu_features()
        assert result == frozenset()
