"""Session-scoped fixtures for yowo integration tests.

All tests in this directory exercise the real pipeline — no mocks, real model
weights, real inference.

Inputs are obtained the way production obtains them. The weight comes through
``resolve_weights()``, which verifies it against the digest pinned in the
registry, so no fixture has to choose between trusting an arbitrary local file
and fetching something unverified.

Nothing here is committed to the repository. The weight is 5.4 MB and AGPL-3.0;
the sample image is served by a third party under no stated licence. Both are
fetched once and cached, and neither belongs in an Apache-2.0 source tree.

A missing input FAILS in CI and skips locally. That asymmetry is deliberate:
these fixtures previously pointed at ``/Users/<someone>/Downloads`` and a
gitignored directory, so every weight-dependent test skipped for everyone but
one person — and a skip is green, so nothing ever reported it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import requests
from click.testing import CliRunner

from yowo.models._weights import resolve_weights
from yowo.types import ModelFamily, ModelSize, ModelSpec

# The smallest pinned weight: yolo26n at 5.29 MB, against yolo11n's 5.35 MB.
# A cold CI download is seconds; pinning the tier to a 109 MB weight would make
# a cache miss punitive enough that someone would eventually disable it. It is
# also the model these tests were written against, so their shape assertions
# hold — feeding them a YOLO11 checkpoint produced 64-vs-16 channel mismatches.
_FIXTURE_FAMILY = ModelFamily.YOLO26
_FIXTURE_SIZE = ModelSize.NANO

_BUS_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
_IMAGE_CACHE = (
    Path(os.environ.get("YOWO_CACHE_DIR", Path.home() / ".cache" / "yowo")) / "test-assets"
)


def _unavailable(what: str, detail: str) -> None:
    """Fail in CI, skip locally.

    In CI an unobtainable input means the run proved nothing, and reporting that
    as green is the failure this tier already suffered. Locally, a contributor
    without network should get a skip rather than a wall.
    """
    message = f"{what} unavailable: {detail}"
    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
        raise RuntimeError(
            f"{message}\nCI must not skip: a green skip is indistinguishable "
            f"from a passing test, which is how this tier went unnoticed."
        )
    pytest.skip(message)


@pytest.fixture(scope="session")
def runner() -> CliRunner:
    """Stateless Click test runner."""
    return CliRunner()


@pytest.fixture(scope="session")
def verified_weight() -> Path:
    """A real weight, digest-verified through the production resolution path."""
    spec = ModelSpec(family=_FIXTURE_FAMILY, size=_FIXTURE_SIZE)
    try:
        return resolve_weights(spec)
    except Exception as exc:
        _unavailable(
            f"weight {_FIXTURE_FAMILY.value}{_FIXTURE_SIZE.value}",
            f"{type(exc).__name__}: {exc}",
        )
        raise  # unreachable; keeps the return type honest


@pytest.fixture(scope="session")
def yolo26_weights(verified_weight: Path) -> Path:
    """Backwards-compatible alias for tests written against the old fixture."""
    return verified_weight


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """The standard YOLO test image, cached across runs rather than committed.

    Contains people and a bus, so a test asserting "found something" at default
    confidence is meaningful rather than vacuously satisfiable.
    """
    _IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    img_path = _IMAGE_CACHE / "bus.jpg"
    if img_path.exists() and img_path.stat().st_size > 0:
        return img_path
    try:
        resp = requests.get(_BUS_IMAGE_URL, timeout=30)
        resp.raise_for_status()
        tmp = img_path.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(img_path)
    except Exception as exc:
        _unavailable("sample image", f"{type(exc).__name__}: {exc}")
    return img_path


@pytest.fixture()
def sample_image_dir(tmp_path: Path, sample_image_path: Path) -> Path:
    """Directory of 3 copies of the sample image for directory-source tests."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        shutil.copy(sample_image_path, img_dir / f"frame_{i:03d}.jpg")
    return img_dir


@pytest.fixture(scope="session")
def torch_available() -> None:
    """Guard the BACKEND import through the same policy as the weight.

    ``pytest.importorskip("torch")`` skips, and a skip is green — the exact
    failure this module's docstring describes, one layer up. A CI job that
    cannot import torch executed no backend, so it proved nothing and must say
    so rather than passing.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        _unavailable("torch", f"{type(exc).__name__}: {exc}")
