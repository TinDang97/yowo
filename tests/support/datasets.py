"""Evaluation-dataset acquisition: pinned, bounded, atomic, traversal-refusing.

The project publishes accuracy numbers. Until this module existed it had no
dataset to measure them against, and the one asset the test tiers did fetch
(``bus.jpg``) came down with no integrity check at all — while the weights
beside it were digest-verified through ``resolve_weights``. Same file, same
pattern, different trust. This closes that.

Three properties, in the order they matter:

**Pinned.** Every byte is compared against a SHA-256 recorded in this file
before anything reads or unpacks it, on the download path *and* on the
cache-hit path. A checksum checked only at download time protects nothing,
because the cache-hit path is the one that runs every subsequent time.

**Bounded.** A connect timeout, a read timeout between chunks, and a wall-clock
deadline across the whole stream — ``requests``' ``timeout=`` covers the
connection and the first byte, not the 815 MB after it. Three attempts, two
backoff gaps, and a digest mismatch is never retried: re-downloading the same
wrong bytes is not a recovery strategy.

**Atomic.** Bytes land in a temp file unique to this process and thread, are
verified there, and only then are renamed onto the cache path. A reader never
observes a partial archive, and an interrupted fetch leaves nothing a later run
would accept — ``if dest.exists()`` is not a cache, it is "trust anything at
this path", and it is how a truncated download becomes permanent.

What is kept, and what is verified when: the archives are deleted once they
have been extracted. Their digests are recorded in the completion marker as
*source provenance* — proof of where the tree came from. What is re-verified on
every cache hit is the tree we still have: the annotations file that decides
which images are scored, and the image count. That is not a weaker guarantee
than re-hashing 1.07 GB of zip; it is a different one, and the only one that
can see corruption that happened *after* extraction. It also keeps the per-run
cost low enough that nobody is tempted to switch it off, which is the failure
mode a 2-second-per-session integrity check actually has.

Provenance and licence: ``docs/datasets.md``. The archives are NOT
redistributable as project assets.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

import requests

__all__ = [
    "ANNOTATIONS_JSON_SHA256",
    "BACKOFF_SECONDS",
    "BUS_IMAGE",
    "COCO_VAL2017_ANNOTATIONS",
    "COCO_VAL2017_IMAGES",
    "CONNECT_TIMEOUT_S",
    "MAX_ATTEMPTS",
    "READ_TIMEOUT_S",
    "SUBSET_MANIFEST_PATH",
    "SUBSET_MANIFEST_SHA256",
    "SUBSET_SIZE",
    "VAL2017_IMAGE_COUNT",
    "DatasetError",
    "DatasetIntegrityError",
    "DatasetUnavailableError",
    "RemoteArchive",
    "completion_marker_path",
    "dataset_is_complete",
    "ensure_coco_val2017",
    "extract_zip_safely",
    "fetch_deadline_seconds",
    "fetch_instructions",
    "fetch_verified",
    "file_digest",
    "pinned_subset_entries",
    "pinned_subset_ids",
    "require_coco_val2017",
    "select_subset_ids",
    "write_completion_marker",
]


# ---------------------------------------------------------------------------
# What we consume, and what it must be
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteArchive:
    """A remote file this project consumes, and the exact bytes it must be."""

    name: str
    url: str
    sha256: str
    size_bytes: int


# The canonical COCO host, `images.cocodataset.org`, is an S3 bucket fronted by
# a certificate issued for `s3.amazonaws.com` — so the virtual-host HTTPS URL
# that every recipe copies around FAILS certificate verification, which is why
# most of them quietly downgrade to plain HTTP. The PATH-STYLE url below reaches
# the same objects (identical ETags, verified 2026-09-15) over TLS that actually
# validates. Do not "tidy" this back to https://images.cocodataset.org/... — it
# does not work, and the version that appears to work is unauthenticated HTTP.
_COCO_BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"

COCO_VAL2017_IMAGES = RemoteArchive(
    name="val2017.zip",
    url=f"{_COCO_BUCKET}/zips/val2017.zip",
    sha256="4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    size_bytes=815585330,
)

# COCO publishes no val-only annotations object: `annotations_val2017.zip` and a
# bare `instances_val2017.json` both 404. The train+val archive is the only path
# to the val labels, so 241 MB is fetched in order to use 20 MB of it.
COCO_VAL2017_ANNOTATIONS = RemoteArchive(
    name="annotations_trainval2017.zip",
    url=f"{_COCO_BUCKET}/annotations/annotations_trainval2017.zip",
    sha256="113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    size_bytes=252907541,
)

# Unlike the weights, which come from an immutable versioned release URL, this
# is an unversioned path on a marketing site behind a CDN. A re-encode upstream
# is a re-pin decision, NOT evidence of corruption — see `_verify`, which says
# which is which, so a routine CDN change does not read as a security incident.
BUS_IMAGE = RemoteArchive(
    name="bus.jpg",
    url="https://ultralytics.com/images/bus.jpg",
    sha256="c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
    size_bytes=137419,
)

#: Which 500. `evaluate_coco_map(subset=N)` selects `sorted(getImgIds())[:N]`
#: over GROUND TRUTH; the manifest records the result of that selection so a
#: reader can check WHICH images a published mAP covers without running
#: anything, and so a silent re-selection shows up in review as a diff.
SUBSET_SIZE = 500
SUBSET_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "coco_val2017_subset500.tsv"
)
SUBSET_MANIFEST_SHA256 = "872ac411cf65f03234a725df9711e9ced36d1c8156161ab0b7ae63bd23336aad"

#: The extracted artefacts we keep, and therefore the ones re-verified on every
#: cache hit. The annotations file decides which images exist and what they
#: contain, so it is hashed; the 787 MB of JPEGs are attested by count, which is
#: what detects a partially-deleted tree.
ANNOTATIONS_JSON_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
VAL2017_IMAGE_COUNT = 5000

_MARKER_VERSION = 1
_MARKER_NAME = ".yowo-coco-val2017-complete.json"
_ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
_IMAGES_MEMBER = "val2017"
_REQUIRED_MEMBERS = (_IMAGES_MEMBER, _ANNOTATIONS_MEMBER)

# Bounds. `timeout=` covers the connection and the first byte only; the
# wall-clock deadline is what stops a stream that trickles forever.
#
# 1800 s PER ATTEMPT (a retry restarts from byte 0 — there is no Range resume),
# chosen to sit strictly BELOW the CI job timeout. Above it, the deadline can
# never fire in CI and a slow runner is killed by GitHub with a bare
# cancellation and no message, which would void the whole "the failure names
# the archive and how to fetch it" contract. 1800 s over 1.07 GB is a floor of
# roughly 600 KB/s. Override with YOWO_DATASET_FETCH_DEADLINE_S.
CONNECT_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 60.0
DEFAULT_STREAM_DEADLINE_S = 1800.0
DEADLINE_ENV_VAR = "YOWO_DATASET_FETCH_DEADLINE_S"
#: The job timeout the accuracy CI job MUST declare. The fetch deadline has to
#: sit strictly below its own job timeout, or it can never fire: GitHub cancels
#: the job first and the contributor gets a bare cancellation carrying none of
#: the guidance M7 promises. Existing jobs use 15 minutes, which is too tight
#: for 1.07 GB, so the accuracy job needs its own larger value — this constant
#: is the number handed to whoever writes the workflow, and the check
#: `test_the_deadline_fits_inside_its_declared_job_timeout` binds them together.
CI_JOB_TIMEOUT_MINUTES_REQUIRED = 45
MAX_ATTEMPTS = 3
#: Two gaps, because three attempts have two gaps. A third value would be a
#: sleep nobody ever performs.
BACKOFF_SECONDS = (2, 4)
_CHUNK_BYTES = 1 << 20

#: NOT a zip-bomb guard — every archive reaching `extract_zip_safely` from this
#: module is digest-pinned first, so a "bomb" would have to be COCO's own bytes.
#: It is a DISK-EXHAUSTION guard for the runner: `annotations_trainval2017.zip`
#: also contains `instances_train2017.json` at ~450 MB, so widening or dropping
#: `members_under` is a live regression class, not a hypothetical one.
MAX_EXTRACT_BYTES = 4 << 30


class DatasetError(RuntimeError):
    """Base for every way this module refuses to hand back data."""


class DatasetIntegrityError(DatasetError):
    """The bytes are not the bytes we pinned.

    Deliberately distinct from :class:`DatasetUnavailableError`: "I could not
    get it" is a retryable transport condition and "these are the wrong bytes"
    is not. Collapsing the two is exactly how a digest mismatch gets retried
    into success.
    """


class DatasetUnavailableError(DatasetError):
    """The dataset could not be obtained: transport, deadline, or absence."""


def fetch_deadline_seconds() -> float:
    """The per-attempt wall-clock budget, overridable for slow links."""
    raw = os.environ.get(DEADLINE_ENV_VAR)
    if not raw:
        return DEFAULT_STREAM_DEADLINE_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise DatasetUnavailableError(
            f"{DEADLINE_ENV_VAR}={raw!r} is not a number of seconds."
        ) from exc
    if value <= 0:
        raise DatasetUnavailableError(f"{DEADLINE_ENV_VAR} must be positive, got {value}.")
    return value


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def file_digest(path: Path) -> str:
    """SHA-256 of a file, streamed so an 815 MB archive is not held in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, what: str) -> None:
    actual = file_digest(path)
    if actual != expected:
        raise DatasetIntegrityError(
            f"{what} failed its integrity check: {path}\n"
            f"  expected sha256: {expected}\n"
            f"  actual   sha256: {actual}\n"
            "The bytes were NOT used. Two very different causes look identical "
            "here, so check which one this is before acting:\n"
            "  - upstream republished the object (a CDN re-encode, a new "
            "release). That is a RE-PIN DECISION for a human: update the pin in "
            "tests/support/datasets.py in a commit that says so, and record it "
            "in docs/datasets.md. It is not a security incident.\n"
            "  - the local file was corrupted or substituted. That is."
        )


def fetch_instructions(archive: RemoteArchive) -> str:
    """A copy-pasteable way to obtain *archive* by hand."""
    return (
        f"curl -fL --retry 5 -o {archive.name} {archive.url}  "
        f"# {archive.size_bytes} bytes, sha256 {archive.sha256}"
    )


# In-process record of what we ourselves verified. It is what separates "a file
# of unknown provenance is sitting at this path" (replace it once, the way
# `resolve_weights` treats a pre-integrity cache) from "a file we verified in
# this very process has since changed underneath us" (corruption or
# substitution — refuse, and do not re-fetch over the evidence).
_verified_lock = threading.Lock()
_verified: dict[str, str] = {}


def fetch_verified(
    archive: RemoteArchive,
    cache_dir: Path,
    *,
    session: object | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> Path:
    """Return a local path holding exactly ``archive.sha256``, or raise.

    Never returns a path whose bytes it has not hashed on this call.

    ``session``, ``sleep`` and ``clock`` are part of the contract, not test
    scaffolding that leaked: they are what let the bounded-retry and deadline
    behaviour be asserted deterministically instead of by sleeping through it.
    """
    sleep_fn = sleep if sleep is not None else time.sleep
    clock_fn = clock if clock is not None else time.monotonic
    get = getattr(session, "get", None) or requests.get
    deadline = fetch_deadline_seconds()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / archive.name
    key = str(dest.resolve())

    if dest.exists():
        actual = file_digest(dest)
        if actual == archive.sha256:
            with _verified_lock:
                _verified[key] = actual
            return dest
        with _verified_lock:
            previously = _verified.get(key)
        if previously == archive.sha256:
            # We verified this exact file earlier in this process and it has
            # changed since. That is corruption or substitution, not a stale
            # cache, and it is not something to paper over by re-downloading.
            raise DatasetIntegrityError(
                f"{archive.name} changed on disk after this process verified it: {dest}\n"
                f"  expected sha256: {archive.sha256}\n"
                f"  actual   sha256: {actual}\n"
                "Refusing to use it and refusing to overwrite it. Investigate."
            )
        # Unknown provenance — a file from before verification existed, or a
        # leftover. Replace it once; the replacement is verified before use.
        dest.unlink()

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        # Unique per process AND per thread, so two racing fetchers never share
        # a partial and neither can adopt a stranger's leftover.
        tmp = cache_dir / f"{archive.name}.{os.getpid()}.{threading.get_ident()}.part"
        try:
            _stream_to(get, archive, tmp, clock_fn, deadline)
            # Verified BEFORE it is reachable under its real name, so a reader
            # that finds `dest` finds bytes that already passed.
            _verify(tmp, archive.sha256, archive.name)
            os.replace(tmp, dest)
            with _verified_lock:
                _verified[key] = archive.sha256
            return dest
        except DatasetIntegrityError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:  # transport, truncation, deadline, wrong size
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS - 1:
                sleep_fn(BACKOFF_SECONDS[attempt])

    raise DatasetUnavailableError(
        f"Could not fetch {archive.name} from {archive.url} after "
        f"{MAX_ATTEMPTS} attempts.\n"
        f"  last error: {type(last_error).__name__}: {last_error}\n"
        f"  fetch it by hand with:\n    {fetch_instructions(archive)}"
    )


def _stream_to(
    get: Callable[..., object],
    archive: RemoteArchive,
    tmp: Path,
    clock: Callable[[], float],
    deadline: float,
) -> None:
    """One bounded attempt. Raises on transport failure, wrong size, or deadline."""
    started = clock()
    # verify=True is explicit and asserted by a check. The pinned digest makes
    # tampering detectable either way, but the habit of reaching for
    # verify=False when a cert misbehaves is exactly how an unauthenticated
    # fetch gets normalised — and this project has already met one host whose
    # certificate does not match its name.
    response = get(
        archive.url,
        stream=True,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        verify=True,
        allow_redirects=True,
    )
    try:
        response.raise_for_status()  # type: ignore[attr-defined]

        # A cheap pre-flight, NOT an integrity check: content-length is a
        # progress-bar input that the server is free to get wrong or omit. Its
        # only job here is to abort a plainly-wrong object at byte 1 instead of
        # at byte 815,585,330. The digest remains the sole authority.
        declared = getattr(response, "headers", {}).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) != archive.size_bytes:
            raise DatasetUnavailableError(
                f"{archive.url} offered {int(declared)} bytes but "
                f"{archive.name} is pinned at {archive.size_bytes}. "
                "Aborting before the transfer rather than after it."
            )

        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):  # type: ignore[attr-defined]
                if clock() - started > deadline:
                    raise DatasetUnavailableError(
                        f"Download deadline of {deadline:.0f}s exceeded for "
                        f"{archive.url} — the connection was still delivering "
                        f"bytes, just not fast enough. Raise it with "
                        f"{DEADLINE_ENV_VAR} if the link is simply slow."
                    )
                if chunk:
                    handle.write(chunk)
        if clock() - started > deadline:
            raise DatasetUnavailableError(
                f"Download deadline of {deadline:.0f}s exceeded for {archive.url}. "
                f"Raise it with {DEADLINE_ENV_VAR} if the link is simply slow."
            )
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _reject(member: str, why: str) -> DatasetIntegrityError:
    return DatasetIntegrityError(
        f"Refusing archive member {member!r}: {why}. Nothing was extracted."
    )


def _check_member(info: zipfile.ZipInfo, root: Path) -> None:
    name = info.filename
    normalised = name.replace("\\", "/")

    if normalised.startswith("/") or PurePosixPath(normalised).is_absolute():
        raise _reject(name, "absolute paths are not allowed")
    drive = os.path.splitdrive(normalised)[0]
    if drive:
        raise _reject(name, f"carries a drive letter ({drive!r})")
    if any(part == ".." for part in PurePosixPath(normalised).parts):
        raise _reject(name, "contains a '..' component")

    # Only the FILE-TYPE field is interesting. A zip written by `writestr` or
    # by a DOS-era tool carries permission bits with no type bits at all
    # (external_attr >> 16 == 0o600), so "has any mode" is not the question —
    # "claims to be something other than a file or a directory" is.
    file_type = stat.S_IFMT(info.external_attr >> 16)
    if file_type == stat.S_IFLNK:
        raise _reject(name, "is a symlink, which can point anywhere")
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise _reject(name, f"is neither a regular file nor a directory ({file_type:#o})")

    # Containment is decided by resolution, never by the string. A prefix filter
    # is not containment: "val2017/../../x".startswith("val2017/") is True.
    resolved = (root / normalised).resolve()
    if resolved != root and root not in resolved.parents:
        raise _reject(name, f"resolves outside the destination root ({resolved})")


def extract_zip_safely(
    archive_path: Path,
    dest: Path,
    *,
    members_under: Sequence[str] | None = None,
    max_total_bytes: int = MAX_EXTRACT_BYTES,
) -> Path:
    """Unpack *archive_path* into *dest*, refusing anything that escapes it.

    Every member is validated BEFORE any member is written, so a refused archive
    leaves no partial tree behind.

    On refusing rather than sanitising: CPython's ``ZipFile.extractall`` already
    strips ``..`` and leading separators and writes a symlink member as a plain
    file — measured, not assumed — so traversal through the stdlib is not
    currently exploitable. But it sanitises *silently*, rewriting
    ``../../etc/passwd`` to ``etc/passwd`` and carrying on, and silence is the
    part this project cannot afford. Two concrete reasons to refuse instead:
    ``docs/datasets.md`` documents a manual fetch using ``unzip``, which DOES
    honour symlinks and absolute paths; and this function is a published
    surface other nodes may point at archives that carry no pin at all.
    """
    archive_path = Path(archive_path)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()  # resolved once, so a symlinked root cannot shift it

    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if members_under is not None:
            prefixes = tuple(members_under)
            infos = [i for i in infos if i.filename.startswith(prefixes)]

        total = 0
        for info in infos:
            _check_member(info, root)
            total += info.file_size
            if total > max_total_bytes:
                raise _reject(
                    info.filename,
                    f"the selected members expand past {max_total_bytes} bytes, "
                    "which would fill the runner's disk",
                )

        zf.extractall(dest, members=[i.filename for i in infos])
    return dest


# ---------------------------------------------------------------------------
# The assembled dataset: complete, or absent
# ---------------------------------------------------------------------------


def completion_marker_path(root: Path) -> Path:
    return Path(root) / _MARKER_NAME


def _expected_marker() -> dict[str, object]:
    """Everything that would make an existing tree the wrong tree.

    If you cannot enumerate a cache's invalidation set, the cache does not ship.
    This is that set: the marker format itself, where the bytes came from, which
    images were selected and how many, and what the kept artefacts must hash to.
    """
    return {
        "marker_version": _MARKER_VERSION,
        "images_archive_sha256": COCO_VAL2017_IMAGES.sha256,
        "annotations_archive_sha256": COCO_VAL2017_ANNOTATIONS.sha256,
        "subset_manifest_sha256": SUBSET_MANIFEST_SHA256,
        "subset_size": SUBSET_SIZE,
        "annotations_json_sha256": ANNOTATIONS_JSON_SHA256,
        "image_count": VAL2017_IMAGE_COUNT,
    }


def write_completion_marker(root: Path) -> None:
    """Stamp *root* as fully built. Written LAST and atomically."""
    root = Path(root)
    marker = completion_marker_path(root)
    tmp = marker.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(_expected_marker(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, marker)


def _count_images(root: Path) -> int:
    images = Path(root) / _IMAGES_MEMBER
    if not images.is_dir():
        return 0
    return sum(1 for entry in images.iterdir() if entry.is_file())


def dataset_is_complete(root: Path) -> bool:
    """True only if this tree was finished under THIS pin and is still all there.

    An extraction killed halfway leaves images and no marker, which is treated
    as absent — deliberately. The alternative, trusting a directory because it
    exists, is the defect that makes a truncated download permanent.

    The marker is not believed on its own word: every field is compared, and the
    artefacts it describes are re-derived. The image count is what catches a
    partially-deleted tree, which no digest of an archive we no longer keep
    could ever see.
    """
    root = Path(root)
    marker = completion_marker_path(root)
    if not marker.is_file():
        return False
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if recorded != _expected_marker():
        return False
    if not all(Path(root, member).exists() for member in _REQUIRED_MEMBERS):
        return False
    if _count_images(root) != VAL2017_IMAGE_COUNT:
        return False
    return file_digest(root / _ANNOTATIONS_MEMBER) == ANNOTATIONS_JSON_SHA256


def _default_cache_root() -> Path:
    return Path(os.environ.get("YOWO_CACHE_DIR", str(Path.home() / ".cache" / "yowo")))


def coco_cache_base(cache_dir: Path | None = None) -> Path:
    return Path(cache_dir or _default_cache_root()) / "datasets" / "coco-val2017"


def ensure_coco_val2017(
    cache_dir: Path | None = None,
    *,
    session: object | None = None,
    keep_archives: bool = False,
) -> Path:
    """Return a root holding ``val2017/`` and ``annotations/``, fetching if needed.

    The archives are removed once extracted: their digests survive in the
    completion marker as provenance, and keeping 1.07 GB of zip beside a 787 MB
    tree would push a shared CI cache past its budget and evict the weight store
    that three other jobs depend on.
    """
    base = coco_cache_base(cache_dir)
    root = base / "root"
    if dataset_is_complete(root):
        return root

    archives = []
    for archive, members in (
        (COCO_VAL2017_IMAGES, (f"{_IMAGES_MEMBER}/",)),
        (COCO_VAL2017_ANNOTATIONS, (_ANNOTATIONS_MEMBER,)),
    ):
        # Nothing is unpacked that has not just been verified: fetch_verified
        # hashes on every path it can return from, so the argument handed to
        # extract_zip_safely is never unverified bytes.
        path = fetch_verified(archive, base, session=session)
        extract_zip_safely(path, root, members_under=members)
        archives.append(path)

    missing = [m for m in _REQUIRED_MEMBERS if not Path(root, m).exists()]
    if missing:
        raise DatasetUnavailableError(
            f"Extraction finished but {missing} are missing under {root}."
        )

    if not keep_archives:
        for path in archives:
            path.unlink(missing_ok=True)

    write_completion_marker(root)  # last: the tree is only 'done' once it is
    return root


def _unavailable_message(cache_dir: Path | None, detail: str) -> str:
    base = coco_cache_base(cache_dir)
    lines = [
        f"COCO val2017 is not available: {detail}",
        "",
        "The accuracy tier measures mAP against COCO val2017. It is ~1.07 GB "
        "across two archives and is NOT redistributable as a project asset "
        "(see docs/datasets.md), so it is fetched and cached rather than "
        "committed to the repository.",
        "",
        f"Expected under: {base}",
        "Change that location with YOWO_CACHE_DIR.",
        f"Raise the per-attempt download deadline with {DEADLINE_ENV_VAR} if your link is slow.",
        "",
        "Fetch the archives by hand with:",
    ]
    for archive in (COCO_VAL2017_IMAGES, COCO_VAL2017_ANNOTATIONS):
        lines.append(f"  {archive.name}:")
        lines.append(f"    {fetch_instructions(archive)}")
    lines += [
        "",
        f"Then unpack val2017/ and {_ANNOTATIONS_MEMBER} under {base / 'root'}, "
        "or simply re-run this suite with network access and let it fetch.",
    ]
    return "\n".join(lines)


def require_coco_val2017(
    cache_dir: Path | None = None,
    *,
    offline: bool = False,
    session: object | None = None,
) -> Path:
    """The dataset, or a loud failure under CI and a skip locally.

    The asymmetry is the point, and it is the policy
    ``tests/integration/conftest.py::_unavailable`` already applies: under CI an
    unobtainable input means the run measured nothing, and reporting that as
    green is indistinguishable from a passing test. Locally, a contributor
    without 1.07 GB to spare should get a skip rather than a wall.
    """
    import pytest

    detail = "fetching was disabled by the caller" if offline else ""
    if not offline:
        try:
            return ensure_coco_val2017(cache_dir, session=session)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"

    message = _unavailable_message(cache_dir, detail)
    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
        raise DatasetUnavailableError(message)
    pytest.skip(message)
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# The pinned 500
# ---------------------------------------------------------------------------


def select_subset_ids(image_ids: Iterable[int], count: int = SUBSET_SIZE) -> list[int]:
    """The evaluator's own selection: sorted GROUND-TRUTH ids, first *count*.

    Mirrors ``benchmark/_evaluator.py`` so the fixture and the evaluator cannot
    disagree about which images a number covers. The sort is what makes the
    subset a property of the dataset rather than of whatever order the caller
    happened to supply (R:RESELECT).
    """
    return sorted(image_ids)[:count]


def pinned_subset_entries(
    manifest_path: Path | None = None,
) -> tuple[tuple[int, str], ...]:
    """The committed ``(image_id, file_name)`` rows, digest-verified.

    Absent or unreadable is a hard error, never an empty tuple: an empty subset
    evaluates zero images, and zero images score a vacuous pass.
    """
    path = Path(manifest_path) if manifest_path is not None else SUBSET_MANIFEST_PATH
    if not path.is_file():
        raise DatasetUnavailableError(
            f"The pinned subset manifest is missing: {path}\n"
            "Without it there is no record of which images a published mAP "
            "covers, and an empty subset would score a vacuous pass."
        )
    _verify(path, SUBSET_MANIFEST_SHA256, "COCO val2017 subset manifest")

    rows: list[tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw_id, file_name = line.split("\t")
            rows.append((int(raw_id), file_name))
        except ValueError as exc:
            raise DatasetIntegrityError(
                f"{path}:{lineno} is not '<image_id>\\t<file_name>': {line!r}"
            ) from exc

    if not rows:
        raise DatasetIntegrityError(f"The pinned subset manifest is empty: {path}")
    return tuple(rows)


def pinned_subset_ids() -> tuple[int, ...]:
    """The 500 ground-truth image ids every published mAP is measured over."""
    return tuple(image_id for image_id, _ in pinned_subset_entries())
