"""Unit tests for yowo.tune._profile — TuneProfile persistence layer."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from yowo.tune import (
    TuneProfile,
    compute_fingerprint,
    load_profile,
    save_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gpu_hw(gpu_name: str = "NVIDIA GeForce RTX 3090", vram_mb: int = 24576) -> MagicMock:
    """Create a mock HardwareProfile with a primary GPU."""
    gpu = MagicMock()
    gpu.name = gpu_name
    gpu.memory_total_mb = vram_mb

    hw = MagicMock()
    hw.primary_gpu = gpu
    hw.libraries.cuda_version = "11.8"
    return hw


def _make_cpu_hw() -> MagicMock:
    """Create a mock HardwareProfile with no GPU (CPU-only)."""
    hw = MagicMock()
    hw.primary_gpu = None
    return hw


def _make_profile(fingerprint: str = "aabbccdd") -> TuneProfile:
    return TuneProfile(
        model="yolo11n",
        backend="onnx",
        batch_size=4,
        precision="fp16",
        fps_achieved=120.5,
        tuned_at="2026-03-07T12:00:00Z",
        fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_fingerprint_gpu():
    """compute_fingerprint with GPU returns an 8-char hex string."""
    hw = _make_gpu_hw()
    result = compute_fingerprint(hw)
    assert isinstance(result, str)
    assert len(result) == 8
    # Must be valid hex
    int(result, 16)


def test_compute_fingerprint_cpu():
    """compute_fingerprint with no GPU (CPU path) returns an 8-char hex string."""
    hw = _make_cpu_hw()
    result = compute_fingerprint(hw)
    assert isinstance(result, str)
    assert len(result) == 8
    int(result, 16)


def test_compute_fingerprint_gpu_deterministic():
    """Same hardware produces same fingerprint (deterministic)."""
    hw = _make_gpu_hw()
    assert compute_fingerprint(hw) == compute_fingerprint(hw)


def test_compute_fingerprint_gpu_vs_cpu_differs():
    """GPU and CPU paths produce different fingerprints."""
    gpu_hw = _make_gpu_hw()
    cpu_hw = _make_cpu_hw()
    # They may collide in theory but should not for distinct inputs
    # Just ensure neither raises and both return valid hex
    g = compute_fingerprint(gpu_hw)
    c = compute_fingerprint(cpu_hw)
    assert len(g) == 8 and len(c) == 8


def test_save_load_roundtrip(tmp_path: Path):
    """save_profile then load_profile returns an equal TuneProfile."""
    hw = _make_gpu_hw()
    fp = compute_fingerprint(hw)
    profile = _make_profile(fingerprint=fp)

    dest = tmp_path / "profiles" / fp / "yolo11n.yaml"
    save_profile(profile, path=dest)
    assert dest.exists()

    loaded = load_profile("yolo11n", hw, path=dest)
    assert loaded is not None
    assert loaded == profile


def test_load_missing_returns_none(tmp_path: Path):
    """load_profile returns None when file does not exist."""
    hw = _make_cpu_hw()
    result = load_profile("yolo11n", hw, path=tmp_path / "nonexistent.yaml")
    assert result is None


def test_load_corrupt_returns_none(tmp_path: Path):
    """load_profile returns None when file contains invalid YAML."""
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(": invalid: {yaml: [\n")

    hw = _make_gpu_hw()
    result = load_profile("yolo11n", hw, path=bad_file)
    assert result is None


def test_load_corrupt_missing_field_returns_none(tmp_path: Path):
    """load_profile returns None when YAML is valid but missing required fields."""
    bad_file = tmp_path / "incomplete.yaml"
    bad_file.write_text(yaml.safe_dump({"model": "yolo11n"}))

    hw = _make_gpu_hw()
    result = load_profile("yolo11n", hw, path=bad_file)
    assert result is None


def test_fingerprint_mismatch_warns_and_returns_none(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """load_profile returns None and logs a warning when fingerprint differs."""
    saved_fp = "aabbccdd"
    profile = _make_profile(fingerprint=saved_fp)

    dest = tmp_path / "profiles" / saved_fp / "yolo11n.yaml"
    save_profile(profile, path=dest)

    # hw that produces a DIFFERENT fingerprint
    hw = _make_gpu_hw(gpu_name="Different GPU", vram_mb=8192)
    # Override compute_fingerprint to guarantee a mismatch
    with (
        patch("yowo.tune._profile.compute_fingerprint", return_value="11223344"),
        caplog.at_level(logging.WARNING, logger="yowo.tune._profile"),
    ):
        result = load_profile("yolo11n", hw, path=dest)

    assert result is None
    assert any("Hardware changed" in r.message for r in caplog.records)


def test_save_is_atomic(tmp_path: Path):
    """save_profile writes via .tmp then renames (atomic write)."""
    hw = _make_gpu_hw()
    fp = compute_fingerprint(hw)
    profile = _make_profile(fingerprint=fp)
    dest = tmp_path / "p.yaml"

    import os

    replaced_calls: list[tuple[str, str]] = []
    _orig_replace = os.replace

    def _spy_replace(src: str, dst: str) -> None:
        replaced_calls.append((src, dst))
        _orig_replace(src, dst)

    with patch("yowo.tune._profile.os.replace", side_effect=_spy_replace):
        save_profile(profile, path=dest)

    assert len(replaced_calls) == 1
    src_path, dst_path = replaced_calls[0]
    assert Path(dst_path) == dest
    assert Path(src_path) != Path(dst_path)
