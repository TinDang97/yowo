"""Three m2 boxes say what their tasks delivered, and every claim is executed.

Red-first for ADD task `m2-box-accounting`.

Three tasks finished and gated -- `capture-timeouts`, `rtsp-reconnect-correctness`,
`reader-shutdown` -- and of their four boxes only ONE could be ticked honestly:

    box 1  "a gate on any bare `cap.read()` without a timeout ARGUMENT"
           cv2.VideoCapture.read is read([, image]); no such argument exists. Its
           `.get()`/`.join()` clauses were already satisfied before the task began.

    box 3  "Thread, FD and capture counts return to baseline"
           threads and captures are asserted. Descriptors are not, anywhere -- and the
           cycle check mocks cv2, so an fd count against it would pass while measuring
           nothing, which is worse than its absence.

    box 4  cites `_source.py:374` for the rebind that motivates it. It is at :504.

Every claim the amended boxes make is EXECUTED here, never restated. That is the box-6
lesson: prose asserting a guarantee drifts silently, and only running it catches that.
"""

from __future__ import annotations

import inspect
import re
import threading
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from yowo.io import _source as source_mod
from yowo.io._reader import ThreadedFrameReader
from yowo.io._source import RTSPStreamSource

_REPO_ROOT = Path(__file__).parent.parent.parent
_MILESTONE = _REPO_ROOT / ".add/milestones/m2-survive-week-two.md"

# Boxes are located by their task citation, not by their prose -- locating a box by a
# phrase from its own wording is what broke both box-6 witnesses the moment the wording
# changed, and this node amends wording on purpose.
_BOX_1 = "← capture-timeouts"
# Boxes 3 AND 4 both cite `← reader-shutdown`, so the citation alone is ambiguous
# here. Each is located by the clause that is unique to it.
_BOX_3 = "return to baseline"
# Located by the clause unique to it. "reconnect() rebind" does not match, because
# the box writes it with backticks -- the kind of near-miss that makes a locator
# match nothing and every check about it pass vacuously.
_BOX_4 = "count check cannot catch this"


def _boxes() -> list[str]:
    return [ln for ln in _MILESTONE.read_text().splitlines() if ln.startswith("- [")]


def _box(marker: str) -> str:
    found = [ln for ln in _boxes() if marker in ln]
    assert len(found) == 1, (
        f"expected exactly one m2 box carrying {marker!r}, found {len(found)}. These "
        "checks assert things ABOUT that box; with it gone they would pass vacuously"
    )
    return found[0]


# ---------------------------------------------------------------------------
# Box 1 — construction, not a read argument
# ---------------------------------------------------------------------------


def test_box_1_demands_construction_not_a_read_argument() -> None:
    """covers: M1, A6, E2 — `read([, image])` has no timeout argument.

    The four bare `cap.read()` calls still in `_source.py` are correct: the CAPTURE is
    bounded, not the call. A box demanding an argument sends a reader to grep, find them,
    and conclude the box is broken.
    """
    box = _box(_BOX_1).lower()
    assert "construct" in box, (
        "m2 box 1 does not say the timeouts bind at CONSTRUCTION. cv2's read signature "
        f"is {cv2.VideoCapture.read.__doc__.strip().splitlines()[0]!r} — there is no "
        "timeout argument for a gate to demand (M1)"
    )
    # The CLAIM half only. The AMENDED clause quotes the old wording verbatim, so the
    # forbidden word legitimately appears there -- scanning the whole line would make
    # recording the amendment impossible, which is the opposite of what M2 wants.
    claim = box.partition("amended")[0]
    assert "argument" not in claim, (
        "m2 box 1 still asks for a timeout ARGUMENT on a call that has no such parameter"
    )
    # The bare calls are still there, and still fine.
    bare = [
        n
        for n, ln in enumerate(
            source_mod.__file__ and Path(source_mod.__file__).read_text().splitlines(), 1
        )
        if re.search(r"\bcap\.read\(\)", ln)
    ]
    assert bare, "no bare cap.read() remains; this check's premise has changed"


def test_box_1_still_demands_the_get_and_join_clauses() -> None:
    """covers: A2, E1 — two true clauses must not leave with the false one."""
    box = _box(_BOX_1)
    for clause in ("queue.get()", "thread.join()"):
        assert clause in box, (
            f"m2 box 1 dropped its {clause} clause. That clause is satisfiable and "
            "satisfied; only the cap.read() one was impossible (A2)"
        )


def test_no_bare_get_or_join_exists_in_the_io_path() -> None:
    """covers: A2, E1 — the clause the box keeps, executed rather than trusted."""
    offenders: list[str] = []
    for rel in ("src/yowo/io", "src/yowo/pipeline", "src/yowo/_streaming.py"):
        target = _REPO_ROOT / rel
        files = target.rglob("*.py") if target.is_dir() else [target]
        for f in files:
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if re.search(r"\.(get|join)\(\s*\)", line) and not line.strip().startswith("#"):
                    offenders.append(f"{f.relative_to(_REPO_ROOT)}:{n}: {line.strip()}")
    assert not offenders, f"bare .get()/.join() in the I/O path: {offenders}"


def test_every_network_capture_is_constructed_with_both_timeouts() -> None:
    """covers: M1, M5 — the guarantee box 1 now states, driven through the real constructor."""
    with mock.patch.object(source_mod, "cv2", mock.MagicMock()) as fake:
        fake.CAP_PROP_OPEN_TIMEOUT_MSEC = cv2.CAP_PROP_OPEN_TIMEOUT_MSEC
        fake.CAP_PROP_READ_TIMEOUT_MSEC = cv2.CAP_PROP_READ_TIMEOUT_MSEC
        fake.VideoCapture.return_value.isOpened.return_value = True
        fake.VideoCapture.return_value.read.return_value = (False, None)
        src = RTSPStreamSource(
            "rtsp://h/s", open_timeout_ms=2000, read_timeout_ms=2000, max_frames=0
        )
        list(src)
        for call in fake.VideoCapture.call_args_list:
            assert len(call.args) >= 3, f"a capture was built bare: {call}"
            assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in call.args[2]
            assert cv2.CAP_PROP_READ_TIMEOUT_MSEC in call.args[2]


# ---------------------------------------------------------------------------
# Box 3 — only what is counted
# ---------------------------------------------------------------------------


def test_box_3_claims_only_threads_and_captures() -> None:
    """covers: M3, A7, E3, R:VACUOUS — "fd" may not survive without a check behind it.

    Nothing anywhere asserts a descriptor count, and the cycle check mocks
    `cv2.VideoCapture`, so no descriptor is ever opened. An fd assertion against that
    harness would pass while measuring nothing (R:VACUOUS).
    """
    box = _box(_BOX_3).lower()
    claim = box.partition("amended")[0]
    assert " fd" not in claim and "descriptor" not in claim, (
        "m2 box 3 still claims file descriptors in its guarantee half. No check counts "
        "them, and the cycle check mocks cv2 so none are opened (M3)"
    )
    for word in ("thread", "capture"):
        assert word in claim, f"m2 box 3 stopped claiming {word}s, which ARE counted"


def test_box_3_says_why_fds_are_absent_and_when_they_return() -> None:
    """covers: A8, A11, E3 — the concern is named, not deleted."""
    box = _box(_BOX_3).lower()
    assert "descriptor" in box or "fd" in box, (
        "m2 box 3 deleted the descriptor concern instead of recording it. An operator "
        "reading this must be able to see that descriptors are NOT covered (A11)"
    )
    assert "real" in box, (
        "m2 box 3 does not name the condition under which descriptors could be counted — "
        "a check that drives a real capture (A8)"
    )


def test_threads_and_captures_return_to_baseline() -> None:
    """covers: M3, M5 — box 3's remaining claim, executed here too."""
    live: list[object] = []

    class Cap:
        def __init__(self, *a: object, **k: object) -> None:
            self.released = False
            live.append(self)

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, object]:
            return True, np.zeros((2, 2, 3), np.uint8)

        def release(self) -> None:
            self.released = True

    with mock.patch.object(source_mod, "cv2", mock.MagicMock(VideoCapture=Cap)):
        baseline = threading.active_count()
        for _ in range(3):
            reader = ThreadedFrameReader(RTSPStreamSource("rtsp://h/s"))
            reader.start()
            time.sleep(0.02)
            reader.stop()
        time.sleep(0.1)
        assert threading.active_count() == baseline
        assert [c for c in live if not c.released] == []


# ---------------------------------------------------------------------------
# Box 4 — a citation that survives an edit
# ---------------------------------------------------------------------------


def test_box_4_cites_a_symbol_that_exists() -> None:
    """covers: M4, A13, E4, E7 — a citation that rots is worse than none.

    The box cited `_source.py:374`; the rebind is at :504. A reviewer who follows a stale
    line lands in unrelated code and concludes the box is confused, rather than that the
    file moved.
    """
    box = _box(_BOX_4)
    assert "reconnect" in box
    src = inspect.getsource(RTSPStreamSource.reconnect)
    assert "self._active_cap = new_cap" in src, (
        "the rebind box 4 describes is no longer in RTSPStreamSource.reconnect. Either "
        "the box or the code moved — re-derive rather than trusting the citation (M4)"
    )
    # Claim half only: the AMENDED clause quotes the stale citation on purpose, as the
    # record of what was corrected. Scanning the whole line would forbid recording it.
    stale = re.findall(r"_source\.py:(\d+)", box.partition("AMENDED")[0])
    for lineno in stale:
        lines = (_REPO_ROOT / "src/yowo/io/_source.py").read_text().splitlines()
        cited = lines[int(lineno) - 1] if int(lineno) <= len(lines) else ""
        assert "_active_cap" in cited, (
            f"m2 box 4 cites _source.py:{lineno}, which reads {cited.strip()!r} — not the "
            "rebind. Cite the symbol, which survives an edit (A13)"
        )


def test_box_4_keeps_the_count_check_warning() -> None:
    """covers: A15 — the box's most useful sentence, and still true."""
    box = _box(_BOX_4).lower()
    assert "count check cannot catch this" in box, (
        "m2 box 4 dropped its warning that a count check cannot catch this defect. The "
        "orphan is REPLACED, so the tally balances — without that sentence a future node "
        "swaps the identity checks for a tally and the defect returns silently (A15)"
    )


def test_box_4_names_the_arm_that_was_taken() -> None:
    """covers: A14 — a disjunction with the chosen arm recorded."""
    box = _box(_BOX_4).lower()
    assert "observ" in box, "box 4 no longer states that the rebind is observed"
    assert "amended" in box, "box 4 does not record which arm of its disjunction was taken"


# ---------------------------------------------------------------------------
# The guards on the amendments themselves
# ---------------------------------------------------------------------------

_AMENDED = [_BOX_1, _BOX_3, _BOX_4]
_AMENDED_IDS = ["box-1", "box-3", "box-4"]


@pytest.mark.parametrize("marker", _AMENDED, ids=_AMENDED_IDS)
def test_every_amended_box_records_that_it_was_amended(marker: str) -> None:
    """covers: M2, A3, E5, R:SILENTREWRITE — an undated rewrite reads as original."""
    box = _box(marker)
    assert "AMENDED 2026-09-11" in box, (
        f"the box carrying {marker!r} was rewritten with no AMENDED clause. A box that "
        "quietly changes shape teaches a reader that boxes are negotiable "
        "(R:SILENTREWRITE)"
    )
    assert "by human decision" in box, f"{marker!r}'s amendment records no authority"


@pytest.mark.parametrize("marker", _AMENDED, ids=_AMENDED_IDS)
def test_no_amended_box_claims_more_than_its_checks_establish(marker: str) -> None:
    """covers: R:WIDEN, E6 — the box-6 guard, applied to three more boxes."""
    claim = _box(marker).lower().partition("amended")[0]
    unearned = {w for w in ("cgroup", "rss", "gpu", "socket") if w in claim}
    assert not unearned, (
        f"the box carrying {marker!r} claims {sorted(unearned)}, which no check here "
        "drives. A box wider than its evidence is the defect this node closes (R:WIDEN)"
    )


def test_no_amended_box_states_a_claim_a_probe_refutes() -> None:
    """covers: M5, R:ROUNDING — every symbol an amended box names is resolved.

    The box-6 check, generalised: prose drifts silently, and reading it a third time does
    not catch that. Running it does.
    """
    import importlib

    checked = 0
    for marker in _AMENDED:
        # `Owner.method` with or WITHOUT parens: box 4 writes `RTSPStreamSource.reconnect`
        # bare, and the paren-only pattern matched nothing at all -- a guard that found
        # zero claims and reported success, which is the vacuous pass this file exists
        # to forbid. The count assertion below is what caught it.
        for module_attr, method in re.findall(r"`([A-Z][\w]*)\.(\w+)(?:\(\))?`", _box(marker)):
            for mod_name in ("yowo.io._source", "yowo.io._reader"):
                mod = importlib.import_module(mod_name)
                owner = getattr(mod, module_attr, None)
                if owner is not None:
                    assert hasattr(owner, method), (
                        f"a box names {module_attr}.{method}(), which does not exist"
                    )
                    checked += 1
    assert checked >= 1, "no symbol claim was found to execute; the guard guards nothing"


def test_the_finished_nodes_checks_are_untouched() -> None:
    """covers: M6 — this node changes what the milestone SAYS, not what the tasks did."""
    for name in (
        "test_capture_timeouts.py",
        "test_rtsp_reconnect.py",
        "test_reader_shutdown.py",
    ):
        assert (_REPO_ROOT / "tests/unit" / name).exists(), f"{name} was removed"
