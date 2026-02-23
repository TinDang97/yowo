"""Session-scoped fixtures for yowo CLI e2e integration tests.

All tests in this directory exercise the real CLI pipeline —
no mocks, real model weights, real inference.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import requests
from click.testing import CliRunner

_LOCAL_WEIGHTS = Path("/Users/tindang/Downloads/Ultralytics YOLO26.pt")
_BUS_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


@pytest.fixture(scope="session")
def runner() -> CliRunner:
    """Stateless Click test runner."""
    return CliRunner()


@pytest.fixture(scope="session")
def yolo26_weights() -> Path:
    """Return path to the local YOLO26 .pt weights file.

    Skips the entire test session if the file is not found.
    """
    if not _LOCAL_WEIGHTS.exists():
        pytest.skip(
            f"Local YOLO26 weights not found at {_LOCAL_WEIGHTS}. "
            "Place the file there or update _LOCAL_WEIGHTS in conftest.py."
        )
    return _LOCAL_WEIGHTS


@pytest.fixture(scope="session")
def sample_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download bus.jpg from ultralytics once per session.

    This is the standard YOLO test image containing persons and a bus —
    guaranteed to produce real detections at confidence >= 0.25.
    Skips if network is unavailable.
    """
    img_dir = tmp_path_factory.mktemp("test_images")
    img_path = img_dir / "bus.jpg"
    try:
        resp = requests.get(_BUS_IMAGE_URL, timeout=30)
        resp.raise_for_status()
        img_path.write_bytes(resp.content)
    except Exception as exc:
        pytest.skip(f"Cannot download bus.jpg: {exc}")
    return img_path


@pytest.fixture()
def sample_image_dir(tmp_path: Path, sample_image_path: Path) -> Path:
    """Directory of 3 copies of bus.jpg for directory-source detection tests."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        shutil.copy(sample_image_path, img_dir / f"frame_{i:03d}.jpg")
    return img_dir
