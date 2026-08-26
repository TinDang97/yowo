"""`import yowo` must not drag torch in.

``postprocess/__init__.py`` already routes the OBB symbols through a lazy
``__getattr__`` whose docstring says, in as many words, "Lazy import OBB symbols
to avoid top-level ``import torch``". Nothing guarded that intent, and
``obb_engine`` reached straight past the gate into the private ``_obb_nms``
module, so the eager chain ``yowo/__init__ -> obb_engine -> _obb_nms -> torch``
put torch back on the import path of anyone who merely typed ``import yowo``.

That matters most where torch is hardest to get. On a Jetson, torch means
NVIDIA's own multi-gigabyte CUDA wheel — a heavy price for a library whose own
README calls itself an edge inference library, and a pure waste for a detection
pipeline that never touches oriented boxes.

Each check runs in a subprocess: ``sys.modules`` is process-wide, and any
earlier test in the session that imported torch would mask the very thing being
measured.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[2] / "src")


def run_isolated(body: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in a fresh interpreter that can see the package source."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
        check=False,
    )


def test_importing_yowo_does_not_import_torch() -> None:
    """The regression itself, measured where torch IS installed."""
    result = run_isolated(
        """
        import sys
        import yowo  # noqa: F401
        assert "torch" not in sys.modules, sorted(
            m for m in sys.modules if m.split(".")[0] == "torch"
        )
        print("clean")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_yowo_imports_on_a_machine_with_no_torch_at_all() -> None:
    """A Jetson without NVIDIA's CUDA wheel, simulated by blocking the name.

    Before the fix this raised ModuleNotFoundError out of
    ``postprocess/_obb_nms.py`` line 9 and the whole library was unusable.
    """
    result = run_isolated(
        """
        import sys

        class NoTorch:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, NoTorch())
        import yowo

        assert yowo.DetectionEngine is not None
        assert yowo.ByteTracker is not None
        print("imported without torch")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "imported without torch" in result.stdout


def test_the_detection_path_stays_reachable_without_torch() -> None:
    """Detection, tracking and counting are what an edge box actually runs.

    Importing is not the claim - being able to reach the functions is.
    """
    result = run_isolated(
        """
        import sys

        class NoTorch:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, NoTorch())

        from yowo.postprocess import COCO_CLASSES, postprocess
        from yowo.tracking import ByteTracker

        assert callable(postprocess)
        assert len(COCO_CLASSES) == 80
        assert ByteTracker is not None
        print("detection path reachable")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "detection path reachable" in result.stdout


def test_obb_symbols_still_resolve_through_the_lazy_gate() -> None:
    """Deferring the import must not remove the capability.

    ``postprocess.__getattr__`` has to hand back the real ``_obb_nms``
    attributes on first touch, or this "fix" would have quietly deleted OBB
    support instead of deferring it.
    """
    pytest.importorskip("torch", reason="OBB decoding genuinely needs torch")
    result = run_isolated(
        """
        import sys
        import yowo  # noqa: F401
        assert "torch" not in sys.modules

        from yowo.postprocess import DOTA_CLASSES, postprocess_obb, probiou_matrix

        assert callable(postprocess_obb)
        assert callable(probiou_matrix)
        assert len(DOTA_CLASSES) == 15
        assert "torch" in sys.modules, "touching an OBB symbol must load torch"
        print("obb still works, and only now pulled torch")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "obb still works, and only now pulled torch" in result.stdout


def test_the_obb_engine_still_exposes_its_class() -> None:
    """`yowo.OBBEngine` stays a public name; only its decoder is deferred."""
    result = run_isolated(
        """
        import yowo

        assert yowo.OBBEngine is not None
        assert "OBBEngine" in yowo.__all__
        print("OBBEngine exported")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OBBEngine exported" in result.stdout
