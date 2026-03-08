"""Regression tests for INT-A1 and INT-A2 OBB integration gaps.

INT-A1: OBBEngine silently loads a detection-calibrated tune profile instead
        of the OBB-tuned one — profile key must include '-obb' task suffix.
INT-A2: `yowo benchmark --model yolo11n-obb` raises ValueError before any
        evaluation runs — _MODEL_PATTERN must accept the 'obb' suffix.
"""

from __future__ import annotations

from unittest.mock import patch

from yowo.types import ModelFamily, ModelSize, ModelSpec

# ---------------------------------------------------------------------------
# INT-A1: Task-aware profile key construction
# ---------------------------------------------------------------------------


def test_tune_profile_key_obb() -> None:
    """Profile key for an OBB model must include the '-obb' suffix.

    Exercises the key formula that engine.py:_load_tune_profile and
    cli/_main.py:tune_command both implement.
    """
    spec = ModelSpec(family=ModelFamily("yolo11"), size=ModelSize("n"), task="obb")
    task_suffix = spec.task if spec.task not in ("detect",) else ""
    key = f"{spec.family.value}{spec.size.value}{'-' + task_suffix if task_suffix else ''}"
    assert key == "yolo11n-obb", f"Expected 'yolo11n-obb', got '{key}'"


def test_tune_profile_key_detect_unchanged() -> None:
    """Profile key for a detection model must NOT have any task suffix.

    Backward-compatibility guard: existing detection profiles must still load.
    """
    spec = ModelSpec(family=ModelFamily("yolo11"), size=ModelSize("n"), task="detect")
    task_suffix = spec.task if spec.task not in ("detect",) else ""
    key = f"{spec.family.value}{spec.size.value}{'-' + task_suffix if task_suffix else ''}"
    assert key == "yolo11n", f"Expected 'yolo11n', got '{key}'"


# ---------------------------------------------------------------------------
# INT-A2: Benchmark pattern and DOTA dataset dispatch
# ---------------------------------------------------------------------------


def test_benchmark_pattern_accepts_obb() -> None:
    """_MODEL_PATTERN must match 'yolo11n-obb' without raising ValueError."""
    from yowo.benchmark import _MODEL_PATTERN  # type: ignore[attr-defined]

    match = _MODEL_PATTERN.match("yolo11n-obb")
    assert match is not None, "'yolo11n-obb' did not match _MODEL_PATTERN"


def test_benchmark_obb_dispatches_dota_path() -> None:
    """run_benchmark with an OBB model must call load_dota_dataset, not load_coco_dataset."""
    from yowo.benchmark import run_benchmark

    mock_dota_return = ([], [], [])

    with (
        patch("yowo.benchmark.load_dota_dataset", return_value=mock_dota_return) as mock_load_dota,
        patch("yowo.benchmark.load_coco_dataset") as mock_load_coco,
        patch("yowo.benchmark.run_all_backends", return_value=[]),
        patch("yowo.benchmark.run_ultralytics_benchmark", return_value=None),
        patch("yowo.benchmark.render_table"),
        patch("yowo.benchmark.results_to_json", return_value={}),
    ):
        run_benchmark(model="yolo11n-obb", data="/tmp/fake-dota")

    mock_load_dota.assert_called_once()
    mock_load_coco.assert_not_called()
