"""The COCO val2017 acquisition path: pinned, bounded, atomic, traversal-safe.

Nothing here touches the network or the 1.07 GB archives. Every fetch is served
by a local ``http.server`` on 127.0.0.1, so the transport is real — a genuine
socket, a genuine ``requests`` call, genuine truncation and genuine 503s — while
staying a unit test. Monkeypatching ``requests`` would have proved that the code
calls a mock the way the test taught it to.

The traversal checks assert REFUSAL, not merely the absence of an escape.
CPython's ``zipfile`` already strips ``..`` and leading separators from member
names, so "nothing was written outside the root" is satisfied by the trusting
implementation too and would be a green check over absent behaviour. Silently
rewriting a hostile member's path is not the contract: R:TRAVERSAL says such an
archive is refused and reported.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import stat
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support import datasets
from tests.support.datasets import (
    BUS_IMAGE,
    COCO_VAL2017_ANNOTATIONS,
    COCO_VAL2017_IMAGES,
    DatasetIntegrityError,
    DatasetUnavailableError,
    RemoteArchive,
    extract_zip_safely,
    fetch_verified,
    file_digest,
    select_subset_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# A real local HTTP origin, with controllable misbehaviour.
# --------------------------------------------------------------------------


@dataclass
class OriginConfig:
    body: bytes = b""
    status: int = 200
    #: Send only this many bytes of `body`, then hang up mid-response.
    truncate_after: int | None = None
    #: Write the body in this many separate writes (to drive chunk loops).
    chunks: int = 1
    hits: list[str] = field(default_factory=list)


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        cfg: OriginConfig = self.server.cfg  # type: ignore[attr-defined]
        cfg.hits.append(self.path)

        if cfg.status != 200:
            self.send_response(cfg.status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        body = cfg.body
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        if cfg.truncate_after is not None:
            # Promise the full length, deliver less, then drop the connection.
            self.wfile.write(body[: cfg.truncate_after])
            self.wfile.flush()
            self.close_connection = True
            return

        step = max(1, len(body) // cfg.chunks) if cfg.chunks > 1 else len(body) or 1
        for start in range(0, len(body), step):
            self.wfile.write(body[start : start + step])
            self.wfile.flush()

    def log_message(self, *args: Any) -> None:
        pass


class _QuietServer(http.server.ThreadingHTTPServer):
    """Client aborts are the POINT of several checks, not errors to report.

    When `fetch_verified` abandons a stream at its deadline it hangs up
    mid-response, and the default handler prints a traceback for the resulting
    broken pipe. That is expected behaviour being logged as a failure.
    """

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class Origin:
    """A local HTTP server serving one configurable object."""

    def __init__(self, cfg: OriginConfig) -> None:
        self.cfg = cfg
        self._server = _QuietServer(("127.0.0.1", 0), _Handler)
        self._server.cfg = cfg  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/object.bin"

    @property
    def hits(self) -> int:
        return len(self.cfg.hits)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture()
def origin() -> Any:
    made: list[Origin] = []

    def _make(body: bytes = b"", **kw: Any) -> Origin:
        o = Origin(OriginConfig(body=body, **kw))
        made.append(o)
        return o

    yield _make
    for o in made:
        o.close()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_for(origin_: Origin, body: bytes, *, name: str = "payload.bin") -> RemoteArchive:
    """An archive pinned to exactly the bytes this origin serves."""
    return RemoteArchive(name=name, url=origin_.url, sha256=sha256(body), size_bytes=len(body))


class RecordingSession:
    """A real requests session that records the kwargs of every call."""

    def __init__(self) -> None:
        import requests

        self._session = requests.Session()
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs, url=url))
        return self._session.get(url, **kwargs)


class FakeClock:
    """A monotonic clock the test advances on every read."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def make_zip_with_symlink(path: Path, link_name: str, target: str) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo(link_name)
        # The high 16 bits of external_attr carry the unix mode; S_IFLNK is
        # what `unzip` and `shutil.unpack_archive` honour when recreating it.
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        info.create_system = 3  # unix
        zf.writestr(info, target)
    return path


# --------------------------------------------------------------------------
# M1 / R:UNPINNED — nothing is used before its digest is compared
# --------------------------------------------------------------------------


class TestDigestIsVerifiedBeforeUse:
    def test_digest_mismatch_refuses_the_archive(self, origin: Any, tmp_path: Path) -> None:
        """covers: M1,R:UNPINNED"""
        served = b"the bytes that were actually served" * 100
        o = origin(served)
        archive = RemoteArchive(
            name="payload.bin",
            url=o.url,
            sha256=sha256(b"the bytes we pinned, which are different"),
            size_bytes=len(served),
        )

        with pytest.raises(DatasetIntegrityError) as exc:
            fetch_verified(archive, tmp_path)

        assert archive.sha256 in str(exc.value)
        assert sha256(served) in str(exc.value)
        # The refused bytes are not left anywhere a later run could find them.
        assert list(tmp_path.iterdir()) == []

    def test_a_mismatch_is_never_retried_into_success(self, origin: Any, tmp_path: Path) -> None:
        """covers: M1,R:UNPINNED"""
        served = b"wrong bytes" * 50
        o = origin(served)
        # size_bytes matches what is served, so the SIZE pre-flight passes and
        # the digest is unambiguously what refuses it.
        archive = RemoteArchive(
            name="payload.bin",
            url=o.url,
            sha256=sha256(b"right bytes"),
            size_bytes=len(served),
        )

        with pytest.raises(DatasetIntegrityError):
            fetch_verified(archive, tmp_path)

        # Re-downloading the same wrong bytes is not a recovery strategy.
        assert o.hits == 1

    def test_cache_hit_is_verified_not_trusted(self, origin: Any, tmp_path: Path) -> None:
        """covers: M1,E1"""
        body = b"genuine payload" * 200
        o = origin(body)
        archive = archive_for(o, body)

        first = fetch_verified(archive, tmp_path)
        assert file_digest(first) == archive.sha256
        assert o.hits == 1

        # Something rewrote the cached file after it was verified.
        first.write_bytes(b"substituted payload" * 200)

        with pytest.raises(DatasetIntegrityError):
            fetch_verified(archive, tmp_path)

        # Refused on the strength of the local bytes alone: a cache-hit check
        # that has to ask the network is not a cache-hit check.
        assert o.hits == 1


# --------------------------------------------------------------------------
# M2 — every fetch is bounded
# --------------------------------------------------------------------------


class TestFetchIsBounded:
    def test_fetch_bounds_connect_and_read(self, origin: Any, tmp_path: Path) -> None:
        """covers: M2"""
        body = b"payload" * 100
        o = origin(body)
        session = RecordingSession()

        fetch_verified(archive_for(o, body), tmp_path, session=session)

        assert session.calls, "no request was issued through the session"
        for call in session.calls:
            timeout = call.get("timeout")
            assert isinstance(timeout, tuple), (
                f"timeout must be a (connect, read) pair, got {timeout!r}"
            )
            connect, read = timeout
            assert connect > 0 and read > 0

    def test_a_stalled_stream_is_abandoned_at_the_deadline(
        self, origin: Any, tmp_path: Path
    ) -> None:
        """covers: M2,E2"""
        body = b"x" * (64 * 1024)
        o = origin(body, chunks=8)
        archive = archive_for(o, body)

        # Each clock read advances an hour: the wall-clock deadline is crossed
        # during the chunk loop, with bytes still arriving. No real sleeping.
        clock = FakeClock(step=3600.0)

        with pytest.raises(DatasetUnavailableError) as exc:
            fetch_verified(archive, tmp_path, clock=clock, sleep=lambda _s: None)

        assert "deadline" in str(exc.value).lower()
        assert list(tmp_path.iterdir()) == []

    def test_transport_failure_retries_with_backoff_then_gives_up(
        self, origin: Any, tmp_path: Path
    ) -> None:
        """covers: M2"""
        o = origin(b"", status=503)
        archive = RemoteArchive(
            name="payload.bin", url=o.url, sha256=sha256(b"anything"), size_bytes=0
        )
        slept: list[float] = []

        with pytest.raises(DatasetUnavailableError) as exc:
            fetch_verified(archive, tmp_path, sleep=slept.append)

        assert o.hits == 3, "expected 3 bounded attempts"
        assert slept == [2, 4], "expected exponential backoff between attempts only"
        assert o.url in str(exc.value)


# --------------------------------------------------------------------------
# M3 / R:HALFCACHED — publication is atomic
# --------------------------------------------------------------------------


class TestPublicationIsAtomic:
    def test_interrupted_fetch_leaves_no_artifact_a_later_run_accepts(
        self, origin: Any, tmp_path: Path
    ) -> None:
        """covers: M3,R:HALFCACHED"""
        body = b"a" * (256 * 1024)
        o = origin(body, truncate_after=4096)
        archive = archive_for(o, body)

        with pytest.raises(DatasetUnavailableError):
            fetch_verified(archive, tmp_path, sleep=lambda _s: None)

        leftovers = list(tmp_path.iterdir())
        assert leftovers == [], f"a killed fetch left {leftovers!r} behind"

    def test_reuse_is_decided_by_digest_not_by_path_existence(
        self, origin: Any, tmp_path: Path
    ) -> None:
        """covers: M3,E1

        ``if dest.exists(): return`` is not a cache, it is "trust anything at
        this path" — and it is how a truncated download becomes permanent. The
        discriminator is a file already sitting at the destination that is NOT
        the archive: a path-existence cache hands it back, a digest-decided
        cache replaces it.
        """
        body = b"stable payload" * 500
        o = origin(body)
        archive = archive_for(o, body)

        stale = tmp_path / archive.name
        tmp_path.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"a truncated download from before verification existed")

        resolved = fetch_verified(archive, tmp_path)
        assert file_digest(resolved) == archive.sha256, (
            "bytes already at the destination path were trusted without a digest"
        )
        assert o.hits == 1

        # Now that it IS the archive, the cached copy is reused, not refetched.
        again = fetch_verified(archive, tmp_path)
        assert again == resolved
        assert o.hits == 1, "a cached, verified archive was fetched again"

    def test_concurrent_fetches_never_publish_a_partial(self, origin: Any, tmp_path: Path) -> None:
        """covers: M3,E4

        A watcher polls the destination path while two fetchers race. Writing
        the body straight to `dest` makes a short-sized file observable there;
        writing to a private temp and renaming makes it unobservable. The green
        assertion ("never observed short") cannot flake into a false pass — a
        missed sighting can only ever weaken the RED side.
        """
        body = b"shared payload " * 300_000  # ~4.5 MB, wide enough to observe
        o = origin(body, chunks=64)
        archive = archive_for(o, body)
        dest = tmp_path / archive.name

        # A temp file left by another process, holding bytes that are NOT the
        # archive. It must never be adopted as the published result.
        foreign = tmp_path / f"{archive.name}.1234.part"
        tmp_path.mkdir(parents=True, exist_ok=True)
        foreign.write_bytes(b"not the archive")

        results: list[Path] = []
        errors: list[BaseException] = []
        short_sightings: list[int] = []
        stop = threading.Event()

        def watcher() -> None:
            while not stop.is_set():
                try:
                    size = dest.stat().st_size
                except OSError:
                    continue
                if size != len(body):
                    short_sightings.append(size)

        def worker() -> None:
            try:
                results.append(fetch_verified(archive, tmp_path, sleep=lambda _s: None))
            except BaseException as exc:
                errors.append(exc)

        w = threading.Thread(target=watcher, daemon=True)
        w.start()
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        stop.set()
        w.join(timeout=5)

        assert not errors, f"concurrent fetch raised {errors!r}"
        assert len(results) == 2
        for path in results:
            assert file_digest(path) == archive.sha256

        assert not short_sightings, (
            f"the destination was observable in a partial state at sizes "
            f"{sorted(set(short_sightings))[:5]} — a concurrent reader would "
            f"have consumed an incomplete archive"
        )
        assert foreign.read_bytes() == b"not the archive"
        assert dest.exists()


# --------------------------------------------------------------------------
# M4 / R:TRAVERSAL — extraction refuses, rather than silently rewriting
# --------------------------------------------------------------------------


class TestExtractionRefusesTraversal:
    @pytest.mark.parametrize(
        "member",
        [
            "../escaped.txt",
            "../../escaped.txt",
            "val2017/../../escaped.txt",
            "/absolute/escaped.txt",
        ],
    )
    def test_extraction_refuses_a_member_escaping_the_root(
        self, tmp_path: Path, member: str
    ) -> None:
        """covers: M4,R:TRAVERSAL"""
        archive = make_zip(tmp_path / "hostile.zip", {member: b"owned"})
        dest = tmp_path / "root"
        sentinel = tmp_path / "escaped.txt"

        with pytest.raises(DatasetIntegrityError) as exc:
            extract_zip_safely(archive, dest)

        assert member in str(exc.value)
        assert not sentinel.exists()

    def test_extraction_refuses_a_symlink_member(self, tmp_path: Path) -> None:
        """covers: M4,E3"""
        archive = make_zip_with_symlink(tmp_path / "linky.zip", "val2017/link", "/etc")
        dest = tmp_path / "root"

        with pytest.raises(DatasetIntegrityError) as exc:
            extract_zip_safely(archive, dest)

        assert "symlink" in str(exc.value).lower() or "link" in str(exc.value).lower()

    def test_a_wholesome_archive_still_extracts(self, tmp_path: Path) -> None:
        """Build guidance, not a gated check: true before the refusal exists.

        It is here so that "refuse everything" cannot pass the two checks above.
        """
        archive = make_zip(
            tmp_path / "ok.zip",
            {"val2017/a.jpg": b"aaa", "val2017/b.jpg": b"bbb", "other/c.txt": b"ccc"},
        )
        dest = tmp_path / "root"

        extract_zip_safely(archive, dest, members_under=("val2017/",))

        assert (dest / "val2017" / "a.jpg").read_bytes() == b"aaa"
        assert (dest / "val2017" / "b.jpg").read_bytes() == b"bbb"
        assert not (dest / "other").exists()


# --------------------------------------------------------------------------
# M5 / R:HALFCACHED — the dataset is complete or absent
# --------------------------------------------------------------------------


def build_tree(root: Path, *, images: int, annotations: bytes | None = None) -> Path:
    """A dataset tree with NON-DEFAULT contents.

    The annotations body defaults to the real pinned bytes' digest partner so a
    dropped field cannot compare equal by accident (Q4): an empty `{}` here
    would hash to something the marker never names, which is the point.
    """
    (root / "val2017").mkdir(parents=True, exist_ok=True)
    for i in range(images):
        (root / "val2017" / f"{i:012d}.jpg").write_bytes(b"jpeg-" + str(i).encode())
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "instances_val2017.json").write_bytes(annotations or b"{}")
    return root


class TestDatasetIsCompleteOrAbsent:
    """`complete` means finished under THIS pin and still all there.

    These patch the two digest/count constants to match a small synthetic tree.
    Building a real 787 MB, 5000-image tree to test a predicate would make the
    predicate untestable, which is how it ends up untested.
    """

    @pytest.fixture()
    def small(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        root = build_tree(tmp_path / "root", images=3, annotations=b'{"images": []}')
        monkeypatch.setattr(datasets, "VAL2017_IMAGE_COUNT", 3)
        monkeypatch.setattr(
            datasets,
            "ANNOTATIONS_JSON_SHA256",
            sha256(b'{"images": []}'),
        )
        return root

    def test_half_extracted_tree_is_rebuilt_not_used(self, small: Path) -> None:
        """covers: M5,R:HALFCACHED"""
        # Images and annotations are all present; only the marker is missing,
        # which is exactly what a killed extraction leaves behind.
        assert not datasets.dataset_is_complete(small)

    def test_completion_marker_names_both_pinned_digests(self, small: Path) -> None:
        """covers: M5"""
        datasets.write_completion_marker(small)
        assert datasets.dataset_is_complete(small)

        marker_text = datasets.completion_marker_path(small).read_text(encoding="utf-8")
        assert COCO_VAL2017_IMAGES.sha256 in marker_text
        assert COCO_VAL2017_ANNOTATIONS.sha256 in marker_text

        # A marker written under a different pin does not satisfy this one.
        datasets.completion_marker_path(small).write_text(
            marker_text.replace(COCO_VAL2017_IMAGES.sha256, "0" * 64), encoding="utf-8"
        )
        assert not datasets.dataset_is_complete(small)

    @pytest.mark.parametrize(
        "field",
        [
            "marker_version",
            "images_archive_sha256",
            "annotations_archive_sha256",
            "subset_manifest_sha256",
            "subset_size",
            "annotations_json_sha256",
            "image_count",
        ],
    )
    def test_every_marker_field_invalidates_the_tree(self, small: Path, field: str) -> None:
        """covers: M5

        A cache key that does not name everything which would change the answer
        is a silent-corruption engine. Each field is checked one at a time, so a
        field nobody compares cannot hide behind one that is compared.
        """
        datasets.write_completion_marker(small)
        assert datasets.dataset_is_complete(small)

        marker = datasets.completion_marker_path(small)
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        assert field in recorded, f"{field} is not recorded in the marker at all"
        original = recorded[field]
        recorded[field] = 99 if isinstance(original, int) else "changed"
        marker.write_text(json.dumps(recorded), encoding="utf-8")

        assert not datasets.dataset_is_complete(small), (
            f"changing {field} left the tree looking complete"
        )

    def test_a_complete_tree_missing_its_annotations_is_not_complete(self, small: Path) -> None:
        """covers: M5,R:HALFCACHED"""
        datasets.write_completion_marker(small)
        assert datasets.dataset_is_complete(small)

        (small / "annotations" / "instances_val2017.json").unlink()
        assert not datasets.dataset_is_complete(small)

    def test_a_partially_deleted_image_tree_is_not_complete(self, small: Path) -> None:
        """covers: M5,R:HALFCACHED

        The archives are deleted after extraction, so no archive digest could
        ever notice this. The recorded image count is the only witness.
        """
        datasets.write_completion_marker(small)
        assert datasets.dataset_is_complete(small)

        next(iter((small / "val2017").iterdir())).unlink()
        assert not datasets.dataset_is_complete(small)

    def test_corrupted_annotations_are_caught_on_the_cache_hit_path(self, small: Path) -> None:
        """covers: M1,M5,E1"""
        datasets.write_completion_marker(small)
        assert datasets.dataset_is_complete(small)

        (small / "annotations" / "instances_val2017.json").write_bytes(b'{"images": [1]}')
        assert not datasets.dataset_is_complete(small)


# --------------------------------------------------------------------------
# M6 / R:RESELECT — the pinned 500
# --------------------------------------------------------------------------


class TestPinnedSubset:
    def test_pinned_subset_is_the_sorted_ground_truth_prefix(self) -> None:
        """covers: M6,R:RESELECT"""
        pinned = datasets.pinned_subset_ids()
        assert len(pinned) == datasets.SUBSET_SIZE == 500

        # A ground truth offered in an adversarial order, carrying ids beyond
        # the subset. Selection must sort first and take the prefix — never
        # honour the order it was handed.
        shuffled = [*reversed(pinned), 10**9, 10**9 + 1]
        assert tuple(select_subset_ids(shuffled, 500)) == pinned

        # And the boundary is exactly the one the manifest documents.
        assert pinned[0] == 139
        assert pinned[-1] == 56545
        assert 57027 not in pinned

    def test_pinned_subset_is_identical_across_independent_runs(self) -> None:
        """covers: M6,R:RESELECT

        Two genuinely separate interpreters, each handed the ground-truth ids in
        a DIFFERENT order, must agree on the 500. Reading the manifest twice
        would agree trivially; shuffling the input is what distinguishes a
        sorted selection from one that inherits whatever order it was given.
        """
        import subprocess
        import sys

        script = (
            "import hashlib, random, sys;"
            "from tests.support.datasets import pinned_subset_ids, select_subset_ids;"
            "ids = list(pinned_subset_ids()) + list(range(10**9, 10**9 + 200));"
            "random.Random(int(sys.argv[1])).shuffle(ids);"
            "picked = select_subset_ids(ids, 500);"
            "print(hashlib.sha256(repr(list(picked)).encode()).hexdigest())"
        )
        digests = set()
        for seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO_ROOT))
            out = subprocess.run(
                [sys.executable, "-c", script, seed],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=env,
                check=True,
            )
            digests.add(out.stdout.strip())

        assert len(digests) == 1, (
            "two independent runs, given the same images in different orders, "
            "selected different subsets"
        )

    def test_manifest_matches_its_pinned_digest(self, tmp_path: Path) -> None:
        """covers: M6,R:RESELECT"""
        # The manifest as committed is what the pin says it is.
        assert file_digest(datasets.SUBSET_MANIFEST_PATH) == datasets.SUBSET_MANIFEST_SHA256

        # An edited manifest silently redefines what every published mAP was
        # measured over, so it is refused rather than read.
        tampered = tmp_path / "coco_val2017_subset500.tsv"
        rows = datasets.SUBSET_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        rows[0] = "999999\t000000999999.jpg"
        tampered.write_text("\n".join(rows) + "\n", encoding="utf-8")

        with pytest.raises(DatasetIntegrityError):
            datasets.pinned_subset_entries(manifest_path=tampered)

    def test_manifest_rows_are_ascending_and_unique(self) -> None:
        """Data-shape guard on the committed manifest. Not red-first: it
        asserts the file is what it claims, which was true when generated."""
        entries = datasets.pinned_subset_entries()
        ids = [i for i, _ in entries]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
        for image_id, file_name in entries:
            assert file_name == f"{image_id:012d}.jpg"


# --------------------------------------------------------------------------
# M7 / R:GREENSKIP — an unobtainable dataset fails under CI
# --------------------------------------------------------------------------


class TestUnavailableDatasetFails:
    """A missing dataset must be loud under CI and merciful locally.

    These use an explicit try/except rather than ``pytest.raises``: if the code
    under test calls ``pytest.skip``, that Skipped exception would propagate and
    SKIP this very test — reporting green for a check that never ran, which is
    the exact defect (Q3) the checks exist to prevent. Capturing every
    BaseException and asserting on its type turns that into a failure.
    """

    @staticmethod
    def _outcome(**kwargs: Any) -> BaseException:
        try:
            datasets.require_coco_val2017(**kwargs)
        except BaseException as exc:
            return exc
        pytest.fail("an unobtainable dataset returned a path instead of reporting")

    def test_missing_dataset_fails_under_ci_and_never_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """covers: M7,R:GREENSKIP"""
        monkeypatch.setenv("CI", "true")
        outcome = self._outcome(cache_dir=tmp_path, offline=True)

        assert not isinstance(outcome, pytest.skip.Exception), (
            "a missing dataset was reported as a skip under CI — a skip is "
            "green, so the job would pass having evaluated nothing"
        )
        assert isinstance(outcome, datasets.DatasetUnavailableError), (
            f"expected DatasetUnavailableError, got {type(outcome).__name__}: {outcome}"
        )

    def test_missing_dataset_skips_locally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """covers: M7"""
        monkeypatch.delenv("CI", raising=False)
        outcome = self._outcome(cache_dir=tmp_path, offline=True)

        assert isinstance(outcome, pytest.skip.Exception), (
            "a contributor with no network and no 1.07 GB of disk should get a "
            f"skip, not {type(outcome).__name__}"
        )

    def test_failure_message_names_the_archive_and_how_to_get_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """covers: M7"""
        monkeypatch.setenv("CI", "true")
        message = str(self._outcome(cache_dir=tmp_path, offline=True))

        for archive in (COCO_VAL2017_IMAGES, COCO_VAL2017_ANNOTATIONS):
            assert archive.name in message, f"{archive.name} is not named"
            assert archive.url in message, f"{archive.name}'s URL is not given"
            assert archive.sha256 in message, f"{archive.name}'s pin is not given"
        assert "YOWO_CACHE_DIR" in message, (
            "the message does not say where the dataset is expected to live"
        )


# --------------------------------------------------------------------------
# M8 — bus.jpg is pinned like the weights are
# --------------------------------------------------------------------------


class TestSampleImageIsPinned:
    def test_sample_image_is_digest_verified_on_fetch(self, origin: Any, tmp_path: Path) -> None:
        """covers: M8"""
        served = b"not a bus"
        o = origin(served)
        archive = RemoteArchive(
            name="bus.jpg", url=o.url, sha256=BUS_IMAGE.sha256, size_bytes=len(served)
        )

        with pytest.raises(DatasetIntegrityError):
            fetch_verified(archive, tmp_path)

        assert list(tmp_path.iterdir()) == []

    def test_sample_image_cache_hit_is_verified(self, origin: Any, tmp_path: Path) -> None:
        """covers: M8,R:UNPINNED"""
        body = b"pretend this is a bus"
        o = origin(body)
        archive = archive_for(o, body, name="bus.jpg")
        fetch_verified(archive, tmp_path)

        (tmp_path / "bus.jpg").write_bytes(b"a different bus")
        with pytest.raises(DatasetIntegrityError):
            fetch_verified(archive, tmp_path)

    def test_the_bus_image_pin_is_a_real_sha256(self) -> None:
        """Shape guard on the pin constant. Not red-first."""
        assert len(BUS_IMAGE.sha256) == 64
        assert BUS_IMAGE.size_bytes == 137419

    def test_the_integration_fixture_pins_the_sample_image(self) -> None:
        """covers: M8,R:UNPINNED"""
        source = (REPO_ROOT / "tests" / "integration" / "conftest.py").read_text(encoding="utf-8")
        assert "fetch_verified" in source, (
            "sample_image_path still downloads bus.jpg without a digest comparison"
        )


# --------------------------------------------------------------------------
# M9 — a contributor meets the licence and the fetch path
# --------------------------------------------------------------------------


class TestDocumentation:
    def test_docs_record_provenance_licence_and_fetch_path(self) -> None:
        """covers: M9"""
        doc = REPO_ROOT / "docs" / "datasets.md"
        assert doc.is_file(), "docs/datasets.md does not exist"
        text = doc.read_text(encoding="utf-8")

        for archive in (COCO_VAL2017_IMAGES, COCO_VAL2017_ANNOTATIONS):
            assert archive.url in text, f"{archive.name}'s source URL is undocumented"
            assert archive.sha256 in text, f"{archive.name}'s pinned digest is undocumented"

        lowered = text.lower()
        assert "creative commons" in lowered or "cc by 4.0" in lowered
        assert "not redistribut" in lowered or "no redistribution" in lowered
        assert "flickr" in lowered, "the images' own licence terms are unstated"
        assert "500" in text, "the pinned subset size is undocumented"

    def test_contributing_links_the_dataset_doc(self) -> None:
        """covers: M9"""
        text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "datasets.md" in text, (
            "a contributor reading CONTRIBUTING never meets the dataset licence"
        )


# --------------------------------------------------------------------------
# M10 — ordering, transport hygiene, and the vacuous-pass guard
# --------------------------------------------------------------------------


class TestNothingIsUnpackedBeforeItIsVerified:
    """The control the whole security argument rests on.

    Traversal refusal is defence in depth; verify-BEFORE-extract is the actual
    mitigation. An implementation that verifies inside `fetch_verified` but
    hands `extract_zip_safely` a path obtained some other way would satisfy
    every other check in this file, which is why this one exists.
    """

    def test_extraction_never_runs_on_unverified_bytes(
        self, origin: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """covers: M1,A14,R:UNPINNED"""
        calls: list[Path] = []
        real = datasets.extract_zip_safely

        def spy(archive_path: Path, dest: Path, **kw: Any) -> Path:
            calls.append(Path(archive_path))
            return real(archive_path, dest, **kw)

        monkeypatch.setattr(datasets, "extract_zip_safely", spy)

        # The origin serves bytes that are not the pinned archive.
        served = b"definitely not val2017.zip"
        o = origin(served)
        monkeypatch.setattr(
            datasets,
            "COCO_VAL2017_IMAGES",
            RemoteArchive(
                name="val2017.zip",
                url=o.url,
                sha256=sha256(b"the real archive"),
                size_bytes=len(served),
            ),
        )
        monkeypatch.setattr(datasets, "BACKOFF_SECONDS", (0, 0))

        with pytest.raises(datasets.DatasetError):
            datasets.ensure_coco_val2017(tmp_path)

        assert calls == [], (
            "extract_zip_safely was handed bytes that had not passed their "
            "digest check — verification must gate extraction, not follow it"
        )


class TestTransportHygiene:
    def test_tls_verification_is_never_disabled(self, origin: Any, tmp_path: Path) -> None:
        """covers: M10

        The COCO bucket presents a certificate that does not match its own
        hostname on the virtual-host URL. The reflex when a cert misbehaves is
        `verify=False`, and once that is in the tree nobody removes it. The pin
        makes tampering detectable either way; this is about the habit.
        """
        body = b"payload" * 100
        o = origin(body)
        session = RecordingSession()

        fetch_verified(archive_for(o, body), tmp_path, session=session)

        assert session.calls
        for call in session.calls:
            assert call.get("verify") is not False, (
                "TLS verification was disabled on a dataset fetch"
            )

    def test_every_pinned_url_is_https(self) -> None:
        """covers: M10"""
        for archive in (COCO_VAL2017_IMAGES, COCO_VAL2017_ANNOTATIONS, BUS_IMAGE):
            assert archive.url.startswith("https://"), (
                f"{archive.name} is fetched over an unauthenticated transport"
            )

    def test_a_wrong_sized_object_aborts_before_the_transfer(
        self, origin: Any, tmp_path: Path
    ) -> None:
        """covers: M2

        content-length is a progress-bar input and never an integrity check —
        the digest stays the sole authority. Its only job here is to refuse a
        plainly-wrong object at byte 1 rather than at byte 815,585,330.
        """
        served = b"a much smaller object than we pinned"
        o = origin(served)
        archive = RemoteArchive(
            name="payload.bin",
            url=o.url,
            sha256=sha256(served),  # the DIGEST would have matched
            size_bytes=815585330,  # but the size says this is the wrong object
        )

        with pytest.raises(datasets.DatasetUnavailableError) as exc:
            fetch_verified(archive, tmp_path, sleep=lambda _s: None)

        assert "815585330" in str(exc.value)
        assert list(tmp_path.iterdir()) == []


class TestAnAbsentManifestIsNeverAnEmptySubset:
    """A12: the vacuous-pass guard.

    An empty subset evaluates zero images, and zero images score a vacuous
    pass — the precise failure this milestone exists to stop.
    """

    def test_a_missing_manifest_raises_rather_than_returning_nothing(self, tmp_path: Path) -> None:
        """covers: M6,A12"""
        with pytest.raises(datasets.DatasetError) as exc:
            datasets.pinned_subset_entries(manifest_path=tmp_path / "gone.tsv")
        assert "missing" in str(exc.value).lower()

    def test_an_empty_manifest_raises_rather_than_returning_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """covers: M6,A12"""
        empty = tmp_path / "empty.tsv"
        empty.write_bytes(b"")
        monkeypatch.setattr(datasets, "SUBSET_MANIFEST_SHA256", sha256(b""))

        with pytest.raises(datasets.DatasetError):
            datasets.pinned_subset_entries(manifest_path=empty)


class TestTheFetchDeadlineFitsInsideItsJob:
    def test_the_deadline_fits_inside_its_declared_job_timeout(self) -> None:
        """covers: M2,M7

        A deadline above its job timeout can never fire: GitHub cancels the job
        first and the contributor gets a bare cancellation with none of the
        guidance M7 promises. This binds the two numbers together so raising
        either one alone fails here rather than silently in CI.

        It deliberately checks the module's own declared requirement rather than
        reading `.github/workflows/ci.yml`: that file is shared and is not this
        node's to write, and the existing jobs' `timeout-minutes: 15` is far too
        tight for a 1.07 GB fetch. The constant is the number handed to whoever
        writes the accuracy job.
        """
        assert datasets.DEFAULT_STREAM_DEADLINE_S < (
            datasets.CI_JOB_TIMEOUT_MINUTES_REQUIRED * 60
        ), (
            f"the {datasets.DEFAULT_STREAM_DEADLINE_S:.0f}s per-attempt deadline "
            f"does not fit inside the {datasets.CI_JOB_TIMEOUT_MINUTES_REQUIRED} "
            "minute job timeout this module asks CI to declare"
        )

    def test_the_deadline_is_overridable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """covers: M2"""
        monkeypatch.setenv(datasets.DEADLINE_ENV_VAR, "120")
        assert datasets.fetch_deadline_seconds() == 120.0

        monkeypatch.setenv(datasets.DEADLINE_ENV_VAR, "nonsense")
        with pytest.raises(datasets.DatasetUnavailableError):
            datasets.fetch_deadline_seconds()

    def test_backoff_has_one_gap_fewer_than_it_has_attempts(self) -> None:
        """covers: M2"""
        assert len(datasets.BACKOFF_SECONDS) == datasets.MAX_ATTEMPTS - 1, (
            "three attempts have two gaps; a third backoff value is a sleep that is never performed"
        )
