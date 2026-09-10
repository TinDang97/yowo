"""Weight download and cache management.

Resolves ModelSpec to a local .pt file path.
Downloads with retry and progress bar if not cached.
"""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from yowo.errors import ModelNotFoundError
from yowo.io._redact import redact_url
from yowo.models._registry import get_for_task
from yowo.types import ModelSpec

_CACHE_DIR = Path.home() / ".cache" / "yowo" / "weights"
_MAX_RETRIES = 3
_BACKOFF_SECONDS = (2, 4, 8)
_CHUNK_SIZE = 8192

_CREDENTIAL_PLACEHOLDER = "<redacted>"


def _scrub_credential(text: str, url: str) -> str:
    """Strip *url*'s userinfo from arbitrary text, not just from *url* itself.

    `redact_url()` only handles the case where the whole string IS a URL. It is
    not enough here: `requests` embeds the credentialed URL in strings it does
    not let us control -- `PreparedRequest.url`, and the messages of
    `HTTPError`, `InvalidSchema` and `InvalidURL` -- and any of those can end
    up interpolated into a caught exception's ``str()``. Regex-hunting for
    URL-shaped tokens in that prose is fragile; we already know the exact
    credential from *url*, so strip that value wherever it recurs instead.

    `requests` also re-quotes special characters before embedding a URL (a raw
    space becomes ``%20``), so the leaked rendering is not always byte-identical
    to *url*. Both the raw and the percent-encoded form of each credential
    component are scrubbed to cover that.
    """
    try:
        parts = urlsplit(url)
        username, password = parts.username, parts.password
    except ValueError:
        username = password = None

    scrubbed = text
    for credential in (username, password):
        if not credential:
            continue
        variants = {credential, quote(credential, safe="")}
        for variant in variants:
            scrubbed = scrubbed.replace(variant, _CREDENTIAL_PLACEHOLDER)
    return scrubbed


class WeightIntegrityError(ModelNotFoundError):
    """A weight's bytes do not match the digest pinned for it."""


def file_digest(path: Path) -> str:
    """SHA-256 of a file, streamed so a 110 MB weight is not held in memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify_digest(path: Path, expected: str, measured: str | None = None) -> None:
    """Raise unless ``path`` hashes to ``expected``.

    The message carries the model path, the expected digest and the actual one,
    so a reader can tell "corrupt download" from "upstream republished" without
    re-deriving either. There is deliberately no non-raising variant: a
    verification failure must never be downgraded to a warning or a fallback to
    the unverified file (R:SILENT), and offering a soft mode is how that happens.

    ``measured`` exists for the one caller that has ALREADY hashed ``path`` on
    this call for another reason -- `load_verified_state_dict`, which needs the
    file's own digest to authenticate its converted sidecar and must not read
    5-110 MB a second time to also compare it to the pin (R:REHASH). It is a way
    to avoid re-reading the file, never a way to supply the answer: the only
    legal value is a digest of ``path`` taken during this same call. Passing
    anything else -- a caller's parameter, a cached value, a number out of a
    sidecar -- turns this check into the tautology R:SELFKEYED forbids.
    """
    actual = file_digest(path) if measured is None else measured
    if actual != expected:
        raise WeightIntegrityError(
            f"Weight failed integrity check: {path}\n"
            f"  expected sha256: {expected}\n"
            f"  actual   sha256: {actual}\n"
            "The file was not used. If upstream republished this release, the "
            "pinned digest in yowo.models._registry must be updated deliberately."
        )


def resolve_weights(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    """Return local path to weights, downloading if necessary.

    Resolution order:
    1. ``spec.weights_path`` if set — validate file exists and return.
    2. ``spec.task`` selects the registry, and family/size the entry in it.
    3. Cache hit at ``cache_dir/family/size/<name>.pt`` — return cached path.
    4. Download from ``ModelMeta.default_weights_url`` with progress bar
       and 3 retries with exponential backoff (2, 4, 8 seconds).

    Step 2 is the whole point of the task dispatch: a ``classify`` spec must
    reach the ``-cls`` asset and an ``obb`` spec the ``-obb`` one. This used to
    call ``get(spec.family, spec.size)`` unconditionally, so the DETECTION
    registry answered every question and `yowo classify SOURCE --model
    yolo11n-cls` could not work at all. ``spec.task`` is read here, BEFORE the
    lookup, and one lookup supplies both the URL and the pin so the two cannot
    come from different entries (A3).

    ``spec.weights_path`` still wins over all of it, for every task: an explicit
    file is the user's own, no registry entry describes it, and it is returned
    unpinned (A5).

    Args:
        spec: Fully-qualified model identity.
        cache_dir: Override cache root. Defaults to ``~/.cache/yowo/weights``.

    Returns:
        Absolute path to a local ``.pt`` weights file.

    Raises:
        ModelNotFoundError: If ``spec.weights_path`` is set but missing, if
            ``spec.task`` is not a registered task, if the family/size is not
            registered for that task, or if download fails after all retries.
    """
    if spec.weights_path is not None:
        if not spec.weights_path.exists():
            raise ModelNotFoundError(f"weights_path does not exist: {spec.weights_path}")
        return spec.weights_path

    # The task picks the registry; family and size pick the entry. Raises
    # ModelNotFoundError naming the task and the registered ones if the task is
    # not one of them -- never a silent fall back to detection (M6, A4).
    meta = get_for_task(spec.task, spec.family, spec.size)

    root = cache_dir or Path(os.environ.get("YOWO_CACHE_DIR", str(_CACHE_DIR)))
    dest = root / spec.family.value / spec.size.value / f"{meta.weight_stem}.pt"

    if dest.exists():
        if meta.sha256 is None:
            _warn_unpinned(meta.weight_stem)
            return dest
        try:
            # Every load, not once per process: a file swapped mid-run would
            # otherwise stay trusted for the process lifetime (A9).
            verify_digest(dest, meta.sha256)
            return dest
        except WeightIntegrityError:
            # Warn and re-fetch (human decision, 2026-09-08). A weight cached
            # before verification existed is not trusted, but neither is it a
            # hard failure — that would break working and air-gapped installs on
            # upgrade. It is replaced once, and the replacement is verified.
            warnings.warn(
                f"Cached weight for {meta.weight_stem} does not match its pinned "
                "digest. It predates integrity checking or has been modified; "
                "re-downloading once and replacing it.",
                RuntimeWarning,
                stacklevel=2,
            )
            dest.unlink(missing_ok=True)

    if meta.sha256 is None:
        _warn_unpinned(meta.weight_stem)
    _download(meta.default_weights_url, dest, meta.sha256)
    return dest


def _warn_unpinned(stem: str) -> None:
    """A user-registered model has no digest we could know — warn, do not refuse."""
    warnings.warn(
        f"Model {stem!r} has no pinned sha256; its weights cannot be verified.",
        RuntimeWarning,
        stacklevel=3,
    )


def _download(url: str, dest: Path, expected_sha256: str | None = None) -> None:
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
            # Verify before the file becomes reachable. A mismatch leaves
            # nothing behind — not the bad file, not a partial (E1).
            if expected_sha256 is not None:
                verify_digest(tmp_path, expected_sha256)
            os.replace(tmp_path, dest)
            return
        except WeightIntegrityError:
            # Never retried and never swallowed: retrying a digest mismatch just
            # re-downloads the same wrong bytes, and swallowing it is R:SILENT.
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_SECONDS[attempt]
                time.sleep(wait)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    raise ModelNotFoundError(
        f"Failed to download weights from {redact_url(url)} after {_MAX_RETRIES} attempts. "
        f"Last error: {_scrub_credential(str(last_exc), url)}"
    )


_CHUNK_STREAM_TIMEOUT_S = 300  # max wall-clock seconds for the full chunk loop


def _attempt_download(url: str, tmp_path: Path, *, suppress_progress: bool) -> None:
    """Single download attempt with optional tqdm progress bar.

    A wall-clock deadline of ``_CHUNK_STREAM_TIMEOUT_S`` seconds is enforced
    across the entire chunk loop so that a stalled connection does not block
    indefinitely (``timeout=60`` on ``requests.get`` only covers the initial
    connection and the first byte, not individual chunk reads).
    """
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

    chunk_deadline = time.monotonic() + _CHUNK_STREAM_TIMEOUT_S
    try:
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if time.monotonic() > chunk_deadline:
                    raise ModelNotFoundError(
                        f"Download of {redact_url(url)} stalled: no progress within "
                        f"{_CHUNK_STREAM_TIMEOUT_S}s"
                    )
                if chunk:
                    fh.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
    finally:
        if progress is not None:
            progress.close()


__all__ = ["resolve_weights"]
