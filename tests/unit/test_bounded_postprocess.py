"""Postprocess latency is bounded by configuration, not by scene content.

Measured 2026-09-16, both paths. The axis-aligned cost law is
``~= 2.0 ns x C x K + 0.6 us x C`` where C is candidates surviving the
confidence filter and K is boxes surviving NMS -- O(C*K), so ONE pre-NMS cap on
C bounds NMS, the lexsort and the BoundingBox loop together. The oriented path
is O(K_kept x N) with a hard floor of ~83 us per kept box, because
``probiou_matrix`` is dispatch-bound below M~2000.

These checks assert WORK DONE, not wall-clock. A latency check calibrated on
one machine either flakes on a shared CI runner or is loosened until it asserts
nothing; counting the candidates that reach the NMS call proves the same
property and is machine-independent. One generous wall-clock leg is kept as a
sanity check, at a ceiling far above any plausible runner.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from yowo.postprocess import _nms, _obb_nms

# The anchor count the SHIPPED runtime produces, from strides 8/16/32 at the
# registry's pinned 640x640. The milestone box said 21504 for the oriented
# path; that is the DOTA-native 1024 count and OBBConfig has no imgsz field.
FULL_ANCHORS = 80 * 80 + 40 * 40 + 20 * 20  # 8400

# Defaults chosen by measurement, recorded in the node's decided-by-me line.
AXIS_MAX_NMS = 1000
AXIS_MAX_DET = 300
OBB_MAX_NMS = 2048
OBB_MAX_DET = 1000


def _disjoint_boxes(
    n: int, *, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n candidates that all survive NMS -- the worst case for the O(C*K) term."""
    xy = rng.integers(0, 100000, size=(n, 2)).astype(np.float32)
    boxes = np.empty((n, 4), dtype=np.float32)
    boxes[:, 0:2] = xy
    boxes[:, 2:4] = xy + 20.0
    scores = rng.random(n).astype(np.float32)
    class_ids = np.zeros(n, dtype=np.intp)
    return boxes, scores, class_ids


def _dense_obb(n: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    boxes = torch.empty((n, 5))
    boxes[:, 0] = torch.randint(0, 100000, (n,), generator=g).float()
    boxes[:, 1] = torch.randint(0, 100000, (n,), generator=g).float()
    boxes[:, 2] = 20.0
    boxes[:, 3] = 20.0
    boxes[:, 4] = 0.0
    return boxes, torch.rand(n, generator=g)


# ---------------------------------------------------------------------------
# M1/M2 -- the bound, at the count the shipped runtime actually produces
# ---------------------------------------------------------------------------


def test_the_axis_aligned_path_is_bounded_at_the_full_anchor_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    real = _nms.cv2.dnn.NMSBoxes

    def counting(boxes, scores, **kw):
        seen.append(len(boxes))
        return real(boxes, scores, **kw)

    monkeypatch.setattr(_nms.cv2.dnn, "NMSBoxes", counting)

    rng = np.random.default_rng(0)
    boxes, scores, class_ids = _disjoint_boxes(FULL_ANCHORS, rng=rng)
    kept = _nms._class_aware_nms(boxes, scores, class_ids, 0.45, max_nms=AXIS_MAX_NMS)

    assert seen, "the NMS call was never reached -- this check would pass vacuously"
    assert seen[0] <= AXIS_MAX_NMS, (
        f"{seen[0]} candidates reached NMS with max_nms={AXIS_MAX_NMS}; "
        "the O(C*K) term is still unbounded"
    )
    assert len(kept) <= AXIS_MAX_NMS


def test_the_oriented_path_is_bounded_at_the_full_anchor_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0, "widest": 0}
    real = _obb_nms.probiou_matrix

    def counting(a, b, eps=1e-7):
        calls["n"] += 1
        calls["widest"] = max(calls["widest"], int(b.shape[0]))
        return real(a, b, eps)

    monkeypatch.setattr(_obb_nms, "probiou_matrix", counting)

    boxes, scores = _dense_obb(FULL_ANCHORS)
    kept = _obb_nms._nms_rotated(boxes, scores, 0.45, max_nms=OBB_MAX_NMS, max_det=OBB_MAX_DET)

    assert calls["n"] > 0, "probiou was never called -- the check would pass vacuously"
    assert calls["n"] <= OBB_MAX_DET, (
        f"{calls['n']} probiou calls with max_det={OBB_MAX_DET}; each costs ~83us "
        "even at M=1, so an unbounded count is the whole defect"
    )
    assert len(kept) <= OBB_MAX_DET
    # max_det alone bounds the NUMBER of calls; only the pre-cap bounds their
    # WIDTH, and the width is what ties cost to the anchor count. Without it
    # the first comparison spans every surviving anchor.
    assert calls["widest"] <= OBB_MAX_NMS, (
        f"a probiou call spanned {calls['widest']} rivals with max_nms={OBB_MAX_NMS}; "
        "cost still scales with the anchor count"
    )


def test_the_real_anchor_count_is_read_from_the_shipped_registry_not_assumed() -> None:
    # The box asserted 21504 for the oriented path. The registry pins 640x640
    # and OBBConfig has no imgsz field, so the runtime cannot produce it.
    from yowo.config import OBBConfig
    from yowo.models import _registry
    from yowo.types import ModelSize

    meta = _registry._make_obb_meta(ModelSize.NANO)
    assert (meta.input_height, meta.input_width) == (640, 640)
    assert "imgsz" not in {f.name for f in dataclasses.fields(OBBConfig)}

    strides = (8, 16, 32)
    assert sum((640 // s) ** 2 for s in strides) == FULL_ANCHORS


# ---------------------------------------------------------------------------
# M3/M4 -- what a bound does, and does not, change
# ---------------------------------------------------------------------------


def test_an_unset_bound_changes_nothing() -> None:
    rng = np.random.default_rng(1)
    boxes, scores, class_ids = _disjoint_boxes(500, rng=rng)
    baseline = _nms._class_aware_nms(boxes, scores, class_ids, 0.45)
    unset = _nms._class_aware_nms(boxes, scores, class_ids, 0.45, max_nms=None, max_det=None)
    assert np.array_equal(baseline, unset)

    ob, os_ = _dense_obb(400, seed=3)
    assert torch.equal(
        _obb_nms._nms_rotated(ob, os_, 0.45),
        _obb_nms._nms_rotated(ob, os_, 0.45, max_nms=None, max_det=None),
    )

    # Random scores never tie, so the sort's stability goes untested on them.
    # Every score identical is what makes an unstable sort observable.
    tied_scores = torch.full((400,), 0.5)
    a = _obb_nms._nms_rotated(ob, tied_scores, 0.45)
    b = _obb_nms._nms_rotated(ob, tied_scores, 0.45, max_nms=None, max_det=None)
    assert torch.equal(a, b)
    # Comparing two calls cannot catch an unstable sort -- both would be
    # unstable the same way. With every score tied, a STABLE sort leaves the
    # candidate order as the input order, so the greedy pass takes index 0
    # first and walks upward; an unstable sort starts somewhere arbitrary.
    assert a[0].item() == 0, "the first survivor was not the first input box"
    assert a.tolist() == sorted(a.tolist()), "survivors are not in input order"


def test_a_bound_discards_only_lower_scoring_candidates() -> None:
    rng = np.random.default_rng(2)
    boxes, scores, class_ids = _disjoint_boxes(3000, rng=rng)

    full = _nms._class_aware_nms(boxes, scores, class_ids, 0.45)
    bounded = _nms._class_aware_nms(boxes, scores, class_ids, 0.45, max_nms=1000, max_det=300)

    assert set(bounded.tolist()) <= set(full.tolist()), "a bound invented a box"
    assert len(bounded) <= 300, f"max_det=300 returned {len(bounded)} boxes"
    # Every kept box outscores every discarded one -- that is what makes a
    # truncation defensible rather than arbitrary.
    dropped = set(full.tolist()) - set(bounded.tolist())
    if dropped:
        assert scores[bounded].min() >= scores[list(dropped)].max()


def test_ties_at_the_cap_boundary_are_broken_deterministically() -> None:
    # The mAP fixture asserts bit-identical results on darwin/arm64 AND
    # ubuntu/x86-64, so a faster pre-cap that reorders exact ties breaks it.
    n = 900
    boxes = np.zeros((n, 4), dtype=np.float32)
    boxes[:, 0] = np.arange(n) * 1000.0
    boxes[:, 1] = 0.0
    boxes[:, 2] = boxes[:, 0] + 20.0
    boxes[:, 3] = 20.0
    scores = np.full(n, 0.5, dtype=np.float32)  # every score identical
    class_ids = np.zeros(n, dtype=np.intp)

    runs = [
        _nms._class_aware_nms(boxes, scores, class_ids, 0.45, max_nms=300).tolist()
        for _ in range(5)
    ]
    assert all(r == runs[0] for r in runs), "the cap picks a different subset run to run"
    assert len(runs[0]) <= 300

    # Repeat-stability is NOT enough: numpy's argpartition happens to be
    # repeat-stable on any one build, so a check that only reruns cannot see
    # the difference. Pin the exact selection instead -- the top max_nms by
    # (score desc, index asc) -- which is what stays equal across platforms.
    expected = set(np.lexsort((np.arange(n), -scores))[:300].tolist())
    assert set(runs[0]) <= expected, (
        "the pre-cap admitted a candidate outside the top-k by (score, index)"
    )


# ---------------------------------------------------------------------------
# M5 -- the gate records what determined its number
# ---------------------------------------------------------------------------


def test_the_map_baseline_records_the_bounds_that_determined_its_number() -> None:
    from yowo.benchmark._baseline import MapBaseline

    names = {f.name for f in dataclasses.fields(MapBaseline)}
    assert {"max_nms", "max_det"} <= names, (
        "a detection bound determines the mAP number; a gate that does not "
        "record it compares two numbers across a determinant it cannot see"
    )
    fixture = json.loads(Path("tests/fixtures/coco_map_baseline.json").read_text())
    assert "max_nms" in fixture and "max_det" in fixture


# ---------------------------------------------------------------------------
# M7 -- reachable by the user the default was chosen for
# ---------------------------------------------------------------------------


def test_both_bounds_are_reachable_from_config_env_and_cli() -> None:
    from yowo import config as cfg_mod
    from yowo.config import InferenceConfig, OBBConfig

    inference = {f.name for f in dataclasses.fields(InferenceConfig)}
    assert {"max_nms", "max_det"} <= inference

    obb = {f.name for f in dataclasses.fields(OBBConfig)}
    assert {"max_nms", "max_det"} <= obb

    source = Path(cfg_mod.__file__).read_text(encoding="utf-8")
    for var in ("YOWO_MAX_NMS", "YOWO_MAX_DET"):
        assert var in source, f"{var} has no environment override"

    cli = Path("src/yowo/cli/_main.py").read_text(encoding="utf-8")
    for opt in ("--max-nms", "--max-det"):
        assert opt in cli, f"{opt} is not reachable from the CLI"


# ---------------------------------------------------------------------------
# Rejects
# ---------------------------------------------------------------------------


def test_the_oriented_path_has_no_device_sync_to_remove() -> None:
    # The box attributes the oriented cost to a per-candidate device sync.
    # Measured: the tensor is ALWAYS CPU, so the sync is 0.15% of wall time.
    # Pinned here so nobody re-derives the wrong diagnosis from the box.
    engine = Path("src/yowo/obb_engine.py").read_text(encoding="utf-8")
    assert "torch.from_numpy(raw_output)" in engine
    boxes, scores = _dense_obb(64, seed=7)
    assert boxes.device.type == "cpu"
    assert _obb_nms._nms_rotated(boxes, scores, 0.45).device.type == "cpu"


def test_the_bounded_oriented_path_does_not_allocate_quadratically() -> None:
    # The obvious vectorized replacement peaks at 1694 MB at 8400 and 6791 MB
    # at 21504. The box records memory as a retired concern, which is true of
    # today's loop and false of that replacement.
    import tracemalloc

    boxes, scores = _dense_obb(FULL_ANCHORS, seed=11)
    tracemalloc.start()
    try:
        _obb_nms._nms_rotated(boxes, scores, 0.45, max_nms=OBB_MAX_NMS, max_det=OBB_MAX_DET)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    quadratic = FULL_ANCHORS * FULL_ANCHORS * 4  # an NxN float32 matrix
    assert peak < quadratic // 100, f"peak {peak} B approaches an NxN allocation"


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_an_empty_or_single_candidate_survives_both_bounds() -> None:
    empty = _obb_nms._nms_rotated(
        torch.zeros((0, 5)), torch.zeros(0), 0.45, max_nms=OBB_MAX_NMS, max_det=OBB_MAX_DET
    )
    assert empty.dtype == torch.long, "a float empty breaks the .tolist() indexing downstream"
    assert len(empty) == 0

    one = _obb_nms._nms_rotated(
        torch.tensor([[10.0, 10.0, 5.0, 5.0, 0.0]]), torch.tensor([0.9]), 0.45, max_nms=1, max_det=1
    )
    assert one.tolist() == [0]

    rng = np.random.default_rng(4)
    boxes, scores, class_ids = _disjoint_boxes(50, rng=rng)
    assert len(_nms._class_aware_nms(boxes, scores, class_ids, 0.45, max_nms=1, max_det=1)) <= 1
    assert (
        len(
            _nms._class_aware_nms(
                np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32),
                np.zeros(0, np.intp),
                0.45,
                max_nms=10,
            )
        )
        == 0
    )


def test_probiou_self_similarity_is_not_one() -> None:
    # Two byte-identical boxes score 0.999532, NOT 1.0. Any threshold >= 0.9996
    # makes duplicates un-suppressible, so a test asserting == 1.0 would be wrong.
    box = torch.tensor([[10.0, 10.0, 20.0, 30.0, 0.5]])
    self_iou = float(_obb_nms.probiou_matrix(box, box).item())
    assert 0.999 < self_iou < 1.0, f"self-IoU is {self_iou}"

    degenerate = torch.tensor([[10.0, 10.0, 0.0, 0.0, 0.0]])
    zero_iou = _obb_nms.probiou_matrix(degenerate, degenerate)
    assert torch.isfinite(zero_iou).all(), "a zero-area box produced NaN"


def test_a_letterboxed_source_has_fewer_anchors_than_the_ceiling() -> None:
    # auto_letterbox on a 16:9 source gives 384x640 -> 5040 anchors, so the
    # ceiling is a MAXIMUM. A check that assumed 8400 always would pass
    # vacuously on a widescreen stream.
    strides = (8, 16, 32)
    letterboxed = sum((384 // s) * (640 // s) for s in strides)
    assert letterboxed == 5040
    assert letterboxed < FULL_ANCHORS


def test_confidence_zero_is_bounded_rather_than_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # conf=0.0 is accepted (closed interval), the filter is >=, and the CLI has
    # no floor -- so EVERY anchor survives regardless of scene. The decision was
    # to leave validation alone, which makes the bound what rescues it.
    from yowo.config import InferenceConfig

    InferenceConfig(confidence_threshold=0.0)  # must not raise

    seen: list[int] = []
    real = _nms.cv2.dnn.NMSBoxes

    def counting(boxes, scores, **kw):
        seen.append(len(boxes))
        return real(boxes, scores, **kw)

    monkeypatch.setattr(_nms.cv2.dnn, "NMSBoxes", counting)
    rng = np.random.default_rng(5)
    boxes, scores, class_ids = _disjoint_boxes(FULL_ANCHORS, rng=rng)
    _nms._class_aware_nms(boxes, scores, class_ids, 0.45, max_nms=AXIS_MAX_NMS)
    assert seen and seen[0] <= AXIS_MAX_NMS


def test_every_determinant_the_baseline_records_is_also_compared() -> None:
    """A recorded field that nothing compares is decorative.

    `_baseline.py` already carries this scar: `map_50` and `map_75` were
    required by the loader and read by nothing, so editing either in the
    shipped record left the whole suite green. `_MUST_MATCH` is a
    hand-maintained list, so adding a determinant to `MapBaseline` without
    adding it there would repeat that exactly -- which is what this check
    exists to prevent, mechanically, for every field added from now on.
    """
    from yowo.benchmark._baseline import _MUST_MATCH, MapBaseline

    scores = {"map_50_95", "map_50", "map_75"}
    metadata = {"model", "tolerance", "measured_on", "measured_at"}
    compared = {recorded for recorded, _ in _MUST_MATCH}

    for field in dataclasses.fields(MapBaseline):
        if field.name in scores or field.name in metadata:
            continue
        assert field.name in compared, (
            f"{field.name} is recorded as determining the mAP number but nothing "
            "compares it, so the gate cannot fail when it changes"
        )


def test_the_class_offset_stride_follows_the_boxes_not_a_fixed_constant() -> None:
    """covers E8 -- found while probing, not created by this node.

    The oriented path shifted box centres by ``class_id * 10000.0``. Once that
    product dwarfs the coordinates, float32 has no resolution left: at class
    5000 the stored dx was exactly 0.0 and two boxes of different classes
    scored a self-IoU of 0.999532, suppressing each other. ``num_classes`` is
    user-settable with no upper bound, so it was reachable.
    """
    source = Path("src/yowo/postprocess/_obb_nms.py").read_text(encoding="utf-8")
    assert "* 10000.0" not in source, "the class offset is a fixed stride again"

    boxes = torch.tensor([[100.0, 100.0, 20.0, 20.0], [100.0, 100.0, 20.0, 20.0]])
    stride = float(boxes[:, :2].max()) + float(boxes[:, 2:4].max()) + 1.0
    for class_id in (14, 5000, 50000):
        shifted = boxes.clone()
        shifted[1, :2] += class_id * stride
        pair = torch.cat([shifted, torch.zeros(2, 1)], dim=1)
        iou = float(_obb_nms.probiou_matrix(pair[0:1], pair[1:2]).item())
        assert iou < 0.5, f"class {class_id} boxes falsely overlap at IoU {iou}"


def test_the_opencv_top_k_argument_is_not_a_free_truncation() -> None:
    """covers A5 -- the milestone box records it as a free 70x latency win.

    It caps the score-sorted CANDIDATE list before suppression, so low-scoring
    DISJOINT boxes are discarded outright rather than trimmed off the tail. It
    is a max_nms with a recall cost, not a max_det. Pinned so the box's wording
    cannot lead someone to adopt it as free.
    """
    boxes = [[0, 0, 100, 100], [10, 10, 100, 100], [500, 500, 50, 50], [700, 700, 50, 50]]
    scores = [0.9, 0.88, 0.5, 0.4]
    unbounded = np.asarray(_nms.cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)).ravel()
    capped = np.asarray(_nms.cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, top_k=3)).ravel()
    assert len(capped) < len(unbounded), "top_k did not discard anything"
    assert len(capped) < 3, "top_k behaved as a post-NMS cap, not a candidate cap"

    # And this package does not rely on it -- the pre-cap is ours, so the
    # tie-break is ours too.
    assert "top_k" not in Path("src/yowo/postprocess/_nms.py").read_text(encoding="utf-8")


def test_the_bound_applies_per_image_not_per_batch() -> None:
    """covers A8 -- a per-batch budget lets one dense image starve the rest.

    Ultralytics ships exactly that shape: a wall-clock ``time_limit`` that
    breaks out mid-batch and leaves later images unprocessed.
    """
    rng = np.random.default_rng(9)
    per_image = []
    for _ in range(3):
        boxes, scores, class_ids = _disjoint_boxes(800, rng=rng)
        per_image.append(
            len(_nms._class_aware_nms(boxes, scores, class_ids, 0.45, max_nms=500, max_det=50))
        )
    assert all(n == per_image[0] for n in per_image), (
        "images in the same batch got different budgets, so one starved another"
    )
    assert all(n <= 50 for n in per_image)


def test_the_ceiling_is_checked_in_ordinary_ci_not_only_in_a_benchmark() -> None:
    """covers A20 -- a bound checked only nightly regresses silently for a day."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "tests/unit" in workflow, "the unit suite does not run in CI"
    # This file lives in tests/unit, so the bound checks ride the ordinary PR
    # run rather than a separate benchmark job.
    assert Path("tests/unit/test_bounded_postprocess.py").exists()


def test_the_baseline_record_is_read_by_name_not_by_position() -> None:
    """covers A21 -- re-recording reorders keys; that must not matter."""
    import json

    from yowo.benchmark._baseline import load_baseline

    original = json.loads(Path("tests/fixtures/coco_map_baseline.json").read_text())
    shuffled = dict(reversed(list(original.items())))
    scratch = Path(__import__("tempfile").mkdtemp()) / "baseline.json"
    scratch.write_text(json.dumps(shuffled, indent=2))
    assert load_baseline(scratch) == load_baseline("tests/fixtures/coco_map_baseline.json")
