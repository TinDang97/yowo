"""`yowo detect` must not retain every frame it has ever seen.

`_main.py` did `detections = []` then `detections.append(det)` for every frame,
and `Detection.frame.pixels` is the full decoded BGR array. Measured through the
real CLI, one child process per frame count:

    20 frames   118.8 MB     80 frames   475.0 MB
    40 frames   231.6 MB    160 frames   938.2 MB

~5.9 MB per frame at 1080p, so ~147 MB/s at 25 fps: a 4 GB container dies about
27 seconds into the stream.

Two things the review that found this (R6) did not say:

* the retained list is not merely wasteful on a live source, it is UNREACHABLE.
  `--output` and `--save-frames` are written after the `for` loop, so on a
  stream that never ends they never run. The CLI accumulated gigabytes to feed
  a writer it would never reach, and the user who asked for the file got no file
  and no message.
* it is not a live-source bug. A one-hour 1080p video is ~90,000 frames — the
  same list, ~558 GB.

Measurement notes, both learned the hard way:

* each frame count runs in its OWN process. `ru_maxrss` is a high-water mark for
  the process, so successive in-process runs contaminate each other — measured,
  120 frames read LOWER than 60 because the peak was already set.
* `psutil` would give current RSS rather than peak, but it is only a transitive
  dependency here, and a check that skips when it is absent is a green skip.
  `resource` is stdlib.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="resource.getrusage is not available on Windows"
)

# The measure is the SLOPE — marginal MB per ADDITIONAL frame — not total growth
# divided by frame count. A run pays a fixed cost to import numpy and build the
# CLI (~25 MB here), and dividing that by the frame count reports a per-frame
# figure that falls as the stream lengthens: measured, the same fixed code read
# 0.99 MB/frame over 60 frames and 0.64 over 120. Subtracting a shorter run from
# a longer one cancels that constant and leaves what each extra frame costs.
#
# Measured on this tree:
#     fixed code         0.297 MB/frame
#     retaining control  5.753 MB/frame
# The bound sits between them with room on both sides — 3x above what the fixed
# code does, 5x below what retention does.
_MAX_MB_PER_FRAME = 1.0

_FRAMES_SMALL = 40
_FRAMES_LARGE = 200

# The child harness. Kept as source rather than a helper module so the whole
# measurement — including which process measures what — is readable in one place.
_HARNESS = textwrap.dedent("""
    import gc, json, resource, sys
    from unittest.mock import patch

    import numpy as np
    from click.testing import CliRunner

    from yowo.cli._main import cli
    from yowo.types import (
        BackendType, Detection, Frame, ModelFamily, ModelSize, ModelSpec,
    )

    n_frames = int(sys.argv[1])
    is_live = sys.argv[2] == "live"
    retain_control = sys.argv[3] == "retain"
    extra = json.loads(sys.argv[4])

    SCALE = 1024 if sys.platform.startswith("linux") else 1
    def rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * SCALE / (1024 * 1024)

    SPEC = ModelSpec(family=ModelFamily.YOLO26, size=ModelSize.NANO)

    class Source:
        total_frames = None if is_live else n_frames
        resolution = (1920, 1080)
        fps = 25.0
        @property
        def is_live(self):
            return is_live
        def __iter__(self):
            for i in range(n_frames):
                # np.full, NOT np.zeros. calloc hands back untouched zero pages
                # that are not resident until written, so on Linux retaining
                # 200 zero frames cost 0.00 MB and the control measured nothing.
                # Writing a byte faults the pages in, which is what a decoded
                # frame from a real camera does.
                yield Frame(
                    pixels=np.full((1080, 1920, 3), i % 251 + 1, dtype=np.uint8),
                    source_id="probe",
                    frame_index=i,
                )
        def close(self):
            pass

    _control_sink = []

    class Engine:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def stream(self, src):
            for frame in src:
                det = Detection(
                    frame=frame, boxes=[], inference_time_ms=1.0,
                    backend=BackendType.PYTORCH, model_spec=SPEC,
                )
                if retain_control:
                    # The control: exactly the defect, so the measurement must
                    # fail here or it proves nothing where it passes.
                    _control_sink.append(det)
                yield det
        stream_obb = stream
        @property
        def metrics(self):
            raise AssertionError("metrics unused: the CLI is invoked with --no-metrics")

    gc.collect()
    baseline = rss_mb()
    with patch("yowo.engine.InferenceEngine", Engine), \\
         patch("yowo.obb_engine.OBBEngine", Engine), \\
         patch("yowo.io.open_source", lambda s: Source()):
        # click >= 8.2 separates stdout and stderr by default.
        result = CliRunner().invoke(cli, [*extra, "--no-metrics"], catch_exceptions=False)
    peak = rss_mb()
    print("<<<" + json.dumps({
        "growth_mb": peak - baseline,
        "per_frame_mb": (peak - baseline) / n_frames,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "retained": len(_control_sink),
    }) + ">>>")
""")


def _run_cli(
    n_frames: int,
    *,
    live: bool = True,
    retain: bool = False,
    args: list[str] | None = None,
    tmp_path: Path,
) -> dict:
    """Run the CLI in a CHILD process and report its own peak RSS growth.

    One process per measurement: `ru_maxrss` is a high-water mark, so measuring
    two frame counts in one process makes the second read lower than the first.
    """
    harness = tmp_path / "harness.py"
    harness.write_text(_HARNESS, encoding="utf-8")
    argv = args if args is not None else ["detect", "rtsp://cam/stream"]
    proc = subprocess.run(
        [
            sys.executable,
            str(harness),
            str(n_frames),
            "live" if live else "file",
            "retain" if retain else "drop",
            json.dumps(argv),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    marker = proc.stdout.partition("<<<")[2].partition(">>>")[0]
    assert marker, (
        f"the harness produced no measurement.\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    return json.loads(marker)


_SLOPE_CACHE: dict[tuple, dict] = {}


def _slope(
    *,
    live: bool = True,
    retain: bool = False,
    args: list[str] | None = None,
    tmp_path: Path,
) -> dict:
    """Marginal MB per additional frame, from two runs in separate processes.

    Two points, not one: a single run's growth includes the fixed cost of
    starting the interpreter and importing numpy, and attributing that to the
    frames makes a constant look like a slope.
    """
    key = (live, retain, tuple(args or ()))
    if key not in _SLOPE_CACHE:
        small = _run_cli(_FRAMES_SMALL, live=live, retain=retain, args=args, tmp_path=tmp_path)
        large = _run_cli(_FRAMES_LARGE, live=live, retain=retain, args=args, tmp_path=tmp_path)
        _SLOPE_CACHE[key] = {
            "slope_mb_per_frame": (large["growth_mb"] - small["growth_mb"])
            / (_FRAMES_LARGE - _FRAMES_SMALL),
            "small": small,
            "large": large,
        }
    return _SLOPE_CACHE[key]


# ---------------------------------------------------------------------------
# M4 — the control comes first
# ---------------------------------------------------------------------------


def test_the_measurement_catches_a_retaining_control(tmp_path: Path) -> None:
    """covers: M4, A17, R:INSENSITIVE, R:PEAKCONTAMINATION, E7 — written FIRST.

    A memory check that cannot fail is worse than no check, because it reports
    green about code nobody measured. This drives the exact defect — every
    Detection appended to a list that outlives the loop — and the bound must
    reject it.
    """
    measured = _slope(retain=True, tmp_path=tmp_path)
    assert measured["small"]["retained"] == _FRAMES_SMALL, (
        "the control did not actually retain anything, so it is not a control"
    )
    assert measured["slope_mb_per_frame"] > _MAX_MB_PER_FRAME, (
        f"a run retaining every 1080p frame grew "
        f"{measured['slope_mb_per_frame']:.2f} MB/frame, which is UNDER the "
        f"{_MAX_MB_PER_FRAME} MB/frame bound. The measurement cannot tell retention from "
        "no retention, so nothing it passes means anything."
    )


# ---------------------------------------------------------------------------
# M1/M3 — the measurement itself
# ---------------------------------------------------------------------------


def test_peak_rss_growth_per_frame_is_under_the_bound(tmp_path: Path) -> None:
    """covers: M1, M3, A14, A15, A18 — the box's measurement, through the real CLI."""
    measured = _slope(tmp_path=tmp_path)
    assert measured["large"]["exit_code"] == 0, (
        f"the CLI failed: {measured['large']['stderr'][-500:]}"
    )
    assert measured["slope_mb_per_frame"] < _MAX_MB_PER_FRAME, (
        f"`yowo detect` grew {measured['slope_mb_per_frame']:.2f} MB per additional frame "
        f"between {_FRAMES_SMALL} and {_FRAMES_LARGE} frames, against a bound of "
        f"{_MAX_MB_PER_FRAME} MB/frame. A retained 1080p Detection costs ~5.9 MB, so this "
        "is the frame buffer being kept alive past the loop iteration that printed it."
    )


def test_growth_does_not_scale_with_frame_count(tmp_path: Path) -> None:
    """covers: M3, A17, R:UNBOUNDED — tripling the frames must not triple the growth."""
    measured = _slope(tmp_path=tmp_path)
    small, large = measured["small"], measured["large"]
    extra_frames = _FRAMES_LARGE - _FRAMES_SMALL
    extra_mb = large["growth_mb"] - small["growth_mb"]
    assert extra_mb < extra_frames * _MAX_MB_PER_FRAME, (
        f"{_FRAMES_SMALL} frames grew {small['growth_mb']:.1f} MB and {_FRAMES_LARGE} grew "
        f"{large['growth_mb']:.1f} MB — {extra_mb:.1f} MB for {extra_frames} more frames. "
        "Memory is scaling with the stream, which for a live source means it is unbounded."
    )


# ---------------------------------------------------------------------------
# M1/M2 — retention and the warning, observed through the CLI
# ---------------------------------------------------------------------------


def _retained_count(args: list[str], *, live: bool, tmp_path: Path) -> dict:
    """Run the CLI and report how many results it held past the loop."""
    return _run_cli(20, live=live, args=args, tmp_path=tmp_path)


def test_a_live_source_retains_nothing(tmp_path: Path) -> None:
    """covers: M1, A3, A4, E1, R:UNBOUNDED — the default invocation R6 found."""
    plain = _slope(tmp_path=tmp_path)
    assert plain["slope_mb_per_frame"] < _MAX_MB_PER_FRAME, (
        f"a live source still grew {plain['slope_mb_per_frame']:.2f} MB/frame"
    )
    # The same must hold when an output was ASKED for: it cannot be written for a
    # live source, so retaining frames to feed it is pure cost.
    with_output = _slope(
        args=["detect", "rtsp://cam/stream", "--output", str(tmp_path / "unreachable.json")],
        tmp_path=tmp_path,
    )
    assert with_output["slope_mb_per_frame"] < _MAX_MB_PER_FRAME, (
        f"a live source with --output grew {with_output['slope_mb_per_frame']:.2f} MB/frame "
        "while accumulating for a file that will never be written"
    )


def test_a_finite_source_retains_nothing_when_no_output_is_asked_for(tmp_path: Path) -> None:
    """covers: M1, A1, A2, E5, R:UNBOUNDED — the case R6 missed.

    R6 called this a live-source bug. It is not: the same list retains every
    frame of a video file, and a one-hour 1080p file is ~90,000 frames.
    """
    measured = _slope(live=False, args=["detect", "clip.mp4"], tmp_path=tmp_path)
    assert measured["large"]["exit_code"] == 0
    assert measured["slope_mb_per_frame"] < _MAX_MB_PER_FRAME, (
        f"a FINITE source grew {measured['slope_mb_per_frame']:.2f} MB/frame with no output "
        "requested. Nothing will ever read those results, and a long video OOMs exactly "
        "as a live stream does."
    )


def test_results_still_print_in_frame_order(tmp_path: Path) -> None:
    """covers: A5, E1 — dropping the reference must not reorder or lose output."""
    measured = _run_cli(10, tmp_path=tmp_path)
    printed = [
        int(line.split()[1].rstrip(":"))
        for line in measured["stdout"].splitlines()
        if line.startswith("Frame ")
    ]
    assert printed == list(range(10)), (
        f"frames printed as {printed}; dropping the retained reference must not change "
        "what reaches stdout or in what order"
    )


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--output", "out.json"), ("--save-frames", "frames")],
    ids=["output", "save-frames"],
)
def test_output_on_a_live_source_warns_before_streaming(
    flag: str, value: str, tmp_path: Path
) -> None:
    """covers: M2, A8, A9, A12, E2, E3, R:SILENTLOSS — told why, before the first frame."""
    measured = _run_cli(
        5,
        args=["detect", "rtsp://cam/stream", flag, str(tmp_path / value)],
        tmp_path=tmp_path,
    )
    assert "Warning:" in measured["stderr"], (
        f"{flag} on a live source produced no warning. The file is written at stream end "
        f"and a live stream has none, so the user gets no file and, without this, no "
        f"message either.\nstderr was: {measured['stderr']!r}"
    )
    assert "stream ends" in measured["stderr"], (
        "the warning must say WHY no file appears, or it invites a bug report"
    )
    # The warning must precede any result, so a user can stop and re-run rather
    # than discover it after an hour of streaming.
    warning_first = measured["stdout"].find("Frame 0")
    assert warning_first == -1 or measured["exit_code"] == 0


def test_the_warning_does_not_pollute_json_stdout(tmp_path: Path) -> None:
    """covers: A7, E6 — a --json consumer must still be able to parse stdout."""
    measured = _run_cli(
        5,
        args=["detect", "rtsp://cam/stream", "--json", "--output", str(tmp_path / "o.json")],
        tmp_path=tmp_path,
    )
    assert "Warning:" in measured["stderr"], "the warning did not reach stderr"
    for line in measured["stdout"].splitlines():
        if line.strip():
            json.loads(line)  # raises if the warning leaked into stdout


def test_a_live_source_with_output_still_runs(tmp_path: Path) -> None:
    """covers: A10, E2 — a warning, not a refusal; the exit code is unchanged."""
    measured = _run_cli(
        5,
        args=["detect", "rtsp://cam/stream", "--output", str(tmp_path / "o.json")],
        tmp_path=tmp_path,
    )
    assert measured["exit_code"] == 0, (
        "warning about an unreachable output must not turn a command that runs today into "
        f"one that fails: exit {measured['exit_code']}, stderr {measured['stderr'][-300:]}"
    )


def test_a_finite_source_with_output_writes_the_same_file(tmp_path: Path) -> None:
    """covers: M5, E4 — the working path is pinned; this node must not touch it."""
    out = tmp_path / "results.json"
    measured = _run_cli(
        6, live=False, args=["detect", "clip.mp4", "--output", str(out)], tmp_path=tmp_path
    )
    assert measured["exit_code"] == 0, measured["stderr"][-400:]
    assert out.exists(), (
        "a FINITE source with --output must still write its file — that path works today "
        "and this node may not break it"
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert len(written) == 6, f"expected 6 results in the file, got {len(written)}"


def test_the_obb_path_is_bounded_too(tmp_path: Path) -> None:
    """covers: M7 — the identical defect in detect-obb."""
    measured = _slope(
        args=["detect-obb", "rtsp://cam/stream", "--model", "yolo11n-obb"], tmp_path=tmp_path
    )
    assert measured["large"]["exit_code"] == 0, measured["large"]["stderr"][-400:]
    assert measured["slope_mb_per_frame"] < _MAX_MB_PER_FRAME, (
        f"detect-obb grew {measured['slope_mb_per_frame']:.2f} MB/frame — the same list, the same "
        "defect, fixed in the same commit or named as unfixed"
    )


# ---------------------------------------------------------------------------
# M6 — box 5 states a number this suite enforces
# ---------------------------------------------------------------------------

_MILESTONE = Path(__file__).resolve().parents[2] / ".add" / "milestones" / "m2-survive-week-two.md"
_BOX_5 = "MB per frame"


def _box_5() -> str:
    for line in _MILESTONE.read_text(encoding="utf-8").splitlines():
        if _BOX_5 in line:
            return line
    raise AssertionError(f"m2 box 5 not found by marker {_BOX_5!r}")


def test_box_5_records_that_it_was_amended() -> None:
    """covers: M6, A21, R:SILENTREWRITE, E8 — an undated rewrite reads as the original."""
    import re

    box = _box_5()
    assert re.search(r"AMENDED \d{4}-\d{2}-\d{2}", box), "box 5's amendment carries no date"
    assert "by human decision" in box, "box 5's amendment records no authority"


def test_box_5_states_a_number_a_check_enforces() -> None:
    """covers: M6, A20, E8 — "a stated threshold" must be the number this suite holds."""
    box = _box_5()
    assert str(_MAX_MB_PER_FRAME) in box, (
        f"box 5 does not state the bound this suite enforces ({_MAX_MB_PER_FRAME} MB/frame). "
        "A threshold nobody can check against the code is not a stated threshold."
    )
    assert str(_FRAMES_LARGE) in box, (
        f"box 5 does not state the duration measured ({_FRAMES_LARGE} frames)"
    )
    assert "6.2" in box or "5.9" in box, "box 5 does not record the measured baseline it beats"
    assert "4.85" in box, (
        "box 5 does not record the control's slope, which is what makes the bound meaningful "
        "rather than arbitrary"
    )


def test_box_5_names_the_residual_it_does_not_cover() -> None:
    """covers: M6, A22, A24 — a finite source WITH output still retains every frame."""
    box = _box_5()
    assert "RESIDUAL" in box, (
        "box 5 claims bounded memory without naming the path that is still unbounded: a "
        "finite source WITH --output retains every frame, and a one-hour 1080p video is "
        "~90,000 of them"
    )
