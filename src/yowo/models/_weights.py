"""Weight download and cache management.

Resolves ModelSpec to a local .pt file path.
Downloads with retry and progress bar if not cached.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from yowo.errors import ModelNotFoundError
from yowo.models._registry import get
from yowo.types import ModelSpec

_CACHE_DIR = Path.home() / ".cache" / "yowo" / "weights"
_MAX_RETRIES = 3
_BACKOFF_SECONDS = (2, 4, 8)
_CHUNK_SIZE = 8192


def resolve_weights(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """Return local path to weights, downloading if necessary.

    Resolution order:
    1. ``spec.weights_path`` if set — validate file exists and return.
    2. Cache hit at ``cache_dir/family/size/<name>.pt`` — return cached path.
    3. Download from ``ModelMeta.default_weights_url`` with progress bar
       and 3 retries with exponential backoff (2, 4, 8 seconds).

    Args:
        spec: Fully-qualified model identity.
        cache_dir: Override cache root. Defaults to ``~/.cache/yowo/weights``.

    Returns:
        Absolute path to a local ``.pt`` weights file.

    Raises:
        ModelNotFoundError: If ``spec.weights_path`` is set but missing,
            if the family/size is not in the registry, or if download
            fails after all retries.
    """
    if spec.weights_path is not None:
        if not spec.weights_path.exists():
            raise ModelNotFoundError(
                f"weights_path does not exist: {spec.weights_path}"
            )
        return spec.weights_path

    # Resolve from registry (raises ModelNotFoundError if not registered).
    meta = get(spec.family, spec.size)

    root = cache_dir or Path(os.environ.get("YOWO_CACHE_DIR", str(_CACHE_DIR)))
    dest = root / spec.family.value / spec.size.value / f"{meta.ultralytics_name}.pt"

    if dest.exists():
        return dest

    _download(meta.default_weights_url, dest)
    return dest


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar and 3 retries.

    Uses atomic write: downloads to a ``.tmp`` sibling then renames.

    Args:
        url: HTTPS URL of the weights file.
        dest: Final destination path.

    Raises:
        ModelNotFoundError: After all retries are exhausted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(".pt.tmp")

    suppress_progress = os.environ.get("CI", "").lower() in ("1", "true", "yes")

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            _attempt_download(url, tmp_path, suppress_progress=suppress_progress)
            os.replace(tmp_path, dest)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_SECONDS[attempt]
                time.sleep(wait)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    raise ModelNotFoundError(
        f"Failed to download weights from {url} after {_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


def _attempt_download(url: str, tmp_path: Path, *, suppress_progress: bool) -> None:
    """Single download attempt with optional tqdm progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0)) or None

    try:
        from tqdm import tqdm  # type: ignore[import-untyped]

        progress: tqdm[Any] | None = (
            None
            if suppress_progress
            else tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=tmp_path.stem,
            )
        )
    except ImportError:
        progress = None

    try:
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
    finally:
        if progress is not None:
            progress.close()


__all__ = ["resolve_weights"]
