"""What the feature cache is guaranteed to notice, and what it is not.

Measured 2026-09-16 against the shipped fingerprint (`mean(axis=(2,3))`, three
numbers per RGB frame) at the shipped threshold of 0.01:

    max contrast (0.5)        a 90x90 px object is invisible   1.98% of frame
    mid contrast (0.30)       116x116                          3.29%
    realistic contrast (0.15) 165x165                          6.65%
    640x640 entry vs 320x320 query        CACHE HIT, returns 80x80 for a 40x40 need
    B=1 entry vs B=4 query                CACHE HIT, (1,3)-(4,3) broadcasts
    black-over-white vs uniform grey      CACHE HIT, both average exactly 0.5

Those are DOCUMENTATION of one machine on one day (Q11). What is asserted is
the property: a shape change is refused outright, and a change confined to one
cell is compared against that cell.

The blind spot cannot be removed by any pooled fingerprint -- a difference
spread thinly enough to leave every cell mean intact is invisible to all of
them. It can be bounded, and the bound is what the docs must state.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The grid the shipped fingerprint pools over, per side.
GRID = 8

#: The bound the documents state, and that these checks hold the code to.
#: DECLARED BEFORE THE RUN from the shipped grid and threshold, not fitted to
#: the result -- the same discipline `test_export_parity.py` uses for its
#: tolerances. A 640-pixel frame at grid 8 gives 80x80 cells; an object wholly
#: inside one cell needs `contrast * s^2 / 6400 >= 0.01`, and one straddling a
#: corner puts only a quarter of itself in each of four cells, which doubles
#: the side that stays invisible. That worst case is what is stated.
DOCUMENTED_BLIND_SPOT_PX = {0.5: 24, 0.15: 44}

#: Written as escapes, not literals: ruff's RUF001 flags an ambiguous glyph
#: sitting in source, and these documents use both spellings.
_MULTIPLICATION_SIGN = "\u00d7"
_EN_DASH = "\u2013"


def _ascii(text: str) -> str:
    """Fold the two glyphs the docs mix into their ASCII twins."""
    return text.replace(_MULTIPLICATION_SIGN, "x").replace(_EN_DASH, "-")


#: Every document that describes the cache to a user.
DOC_SURFACES = (
    "README.md",
    "src/yowo/cache/README.md",
    "docs/user-guide.md",
    "src/yowo/config.py",
)


def grey(batch: int = 1, side: int = 640, value: float = 0.5) -> np.ndarray:
    return np.full((batch, 3, side, side), value, dtype=np.float32)


def feats(h: int = 80, w: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((1, 144, h, w), np.float32),
        np.zeros((1, 288, h // 2, w // 2), np.float32),
        np.zeros((1, 576, h // 4, w // 4), np.float32),
    )


def with_square(
    size: int, contrast: float = 0.5, side: int = 640, *, straddle: bool = False
) -> np.ndarray:
    """A square of `size` px, either inside one cell or across four of them.

    `straddle` centres it on a cell corner, so each of the four cells sees only
    a quarter of it. That is the worst placement for a pooled fingerprint and
    therefore the one the documented bound has to cover.
    """
    a = grey(side=side)
    if straddle:
        centre = (side // GRID) * (GRID // 2)
        top = max(0, centre - size // 2)
    else:
        top = 0
    a[:, :, top : top + size, top : top + size] = 0.5 + contrast
    return a


class TestAnEntryOnlyAnswersAQueryItFits:
    def test_a_shape_change_is_refused_before_any_distance_is_computed(self) -> None:
        """M1 · A5 · A8 -- a resize is not a matter of degree.

        Before this, a 640x640 entry answered a 320x320 query and handed back
        P3 of 80x80 for an input needing 40x40. Both frames fingerprint to
        three numbers, so the resize was literally invisible.
        """
        from yowo.cache import FeatureCache

        cache = FeatureCache()
        cache.update("cam", grey(side=640), feats(80, 80))

        assert cache.check_and_load("cam", grey(side=320)) is None

    def test_a_batch_change_is_refused_instead_of_broadcast(self) -> None:
        """M1 · R:BROADCAST · E2 -- the shipped preset's final partial batch.

        `cache=True` appears in exactly two presets and one of them,
        `(CUDA_HIGH, VIDEO)`, also sets `batch_size=4`. Any video whose frame
        count is not a multiple of 4 ends on a partial batch. `np.abs((4,3) -
        (2,3))` does not raise -- numpy broadcasts, and the two compared equal.
        """
        from yowo.cache import FeatureCache

        cache = FeatureCache()
        cache.update("cam", grey(batch=4), feats())

        assert cache.check_and_load("cam", grey(batch=2)) is None

    def test_an_identical_frame_still_hits(self) -> None:
        """E3 -- the cache must remain useful.

        A fingerprint strict enough to never hit would satisfy every other
        check in this file. This is the one that refuses that shortcut.
        """
        from yowo.cache import FeatureCache

        cache = FeatureCache()
        frame = grey()
        cache.update("cam", frame, feats())

        assert cache.check_and_load("cam", frame.copy()) is not None

    def test_a_first_frame_for_a_source_is_a_plain_miss(self) -> None:
        """A6 · E5 -- frame one of every stream, not an anomaly."""
        from yowo.cache import FeatureCache

        assert FeatureCache().check_and_load("never-seen", grey()) is None


class TestALocalChangeIsComparedLocally:
    def test_a_change_confined_to_one_cell_is_not_averaged_away(self) -> None:
        """M2 · R:DILUTE · E4 -- identical global mean, nothing in common.

        Black over white averages exactly 0.5, and so does uniform grey. The
        shipped fingerprint reported them IDENTICAL -- measured, not argued.
        """
        from yowo.cache import FeatureCache

        cache = FeatureCache()
        cache.update("cam", grey(), feats())

        other = grey()
        other[:, :, :320, :] = 0.0
        other[:, :, 320:, :] = 1.0
        assert other.mean() == pytest.approx(0.5), "the premise: the means agree"

        assert cache.check_and_load("cam", other) is None

    def test_the_measured_blind_spot_is_no_larger_than_the_documented_number(self) -> None:
        """M5 · M3 · A11 -- the docs' number is held to the code, by measuring.

        Sweeps object size until the cache stops hitting. The prose cannot
        drift from the behaviour, because this recomputes it.
        """
        from yowo.cache import FeatureCache

        for contrast, documented in DOCUMENTED_BLIND_SPOT_PX.items():
            worst = 0
            for straddle in (False, True):
                largest_invisible = 0
                for size in range(1, 400):
                    cache = FeatureCache()
                    cache.update("cam", grey(), feats())
                    probe = with_square(size, contrast, straddle=straddle)
                    if cache.check_and_load("cam", probe) is None:
                        break
                    largest_invisible = size
                worst = max(worst, largest_invisible)

            assert worst <= documented, (
                f"at contrast {contrast} an object of {worst}x{worst} px is invisible to "
                f"the cache, but the documents state {documented}x{documented}"
            )

    def test_every_pixel_lands_in_exactly_one_cell(self) -> None:
        """E1 · E6 -- 641 is not divisible by 8, and 4 is smaller than 8."""
        from yowo.cache._similarity import spatial_fingerprint

        for side in (641, 640, 4):
            fp = spatial_fingerprint(grey(side=side, value=0.25), GRID)
            assert fp.shape[:2] == (1, 3)
            assert np.isfinite(fp).all(), f"side={side} produced a non-finite cell"
            assert fp == pytest.approx(0.25), (
                f"side={side}: a uniform frame must give uniform cells; a cell that "
                f"covered no pixels or counted some twice would not"
            )


class TestTheDocumentsSayWhatTheCodeDoes:
    def test_every_place_the_feature_is_documented_states_the_bound(self) -> None:
        """M3 · A2 · A10 -- stated where the user meets it, in pixels.

        A threshold value in abstract units is not a statement an operator can
        act on. "An object this big may not be detected" is.
        """
        missing = []
        for rel in DOC_SURFACES:
            text = _ascii((REPO_ROOT / rel).read_text(encoding="utf-8"))
            if not re.search(r"feature.{0,4}cache|FeatureCache", text, re.I):
                continue
            if not re.search(r"\b\d{1,3}\s*x\s*\d{1,3}\s*(?:px|pixel)", text, re.I):
                missing.append(rel)

        assert not missing, (
            f"these describe the feature cache without stating the blind spot in pixels: {missing}"
        )

    def test_no_savings_figure_appears_without_a_dated_experiment(self) -> None:
        """M4 · R:UNBACKED · A7 -- a hedged number is still the number quoted.

        "60-85% compute savings" sat in two READMEs with no experiment behind
        it, unlike every other headline figure in this repo.
        """
        experiments = REPO_ROOT / "docs" / "experiments"
        backing = [
            p
            for p in experiments.glob("*.md")
            if re.search(r"feature.{0,4}cache", p.read_text(encoding="utf-8"), re.I)
        ]

        claims = []
        for rel in DOC_SURFACES:
            text = _ascii((REPO_ROOT / rel).read_text(encoding="utf-8"))
            for line in text.splitlines():
                if re.search(r"\d{1,3}\s*-\s*\d{1,3}\s*%|\d{1,3}\s*%", line) and re.search(
                    r"cache", line, re.I
                ):
                    claims.append(f"{rel}: {line.strip()[:90]}")

        if claims:
            assert backing, (
                f"a cache savings figure is stated with no docs/experiments/ entry "
                f"measuring it: {claims}"
            )

    def test_the_shipped_path_uses_the_guard_this_module_defines(self) -> None:
        """M6 · R:DEADGUARD -- how a shape guard stayed green and unreachable.

        `_similarity.py:26` returned 1.0 on a shape mismatch. `frame_similarity`
        had ZERO callers in `src/` and was tested at `test_cache.py:32-42`, so
        the guard was covered, passing, and never run by the code a user hits.
        """
        import inspect

        import yowo.cache
        import yowo.cache._similarity as similarity

        src = "".join(inspect.getsource(m) for m in (yowo.cache, yowo.cache._store, similarity))
        public = [
            name
            for name, obj in vars(similarity).items()
            if not name.startswith("_") and inspect.isfunction(obj)
        ]
        assert public, "the module defines no comparator at all"

        unreachable = [n for n in public if src.count(n) < 2]
        assert not unreachable, (
            f"{unreachable} is defined in yowo.cache but called nowhere inside it -- "
            f"a guard reachable only from tests"
        )
