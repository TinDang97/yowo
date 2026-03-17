"""Guard tests ensuring yowo stays Python 3.8 compatible.

These tests prevent future contributors from accidentally introducing
runtime Python 3.9+ constructs into the codebase.

Tests:
    1. Every non-__init__.py source file declares `from __future__ import annotations`
    2. No __init__.py file uses bare `X | Y` union syntax at runtime
    3. StrEnumBase backport is usable and str()-correct on Python < 3.11
    4. All documented public package imports succeed without errors
"""

from __future__ import annotations

import ast
import enum
import sys
from pathlib import Path

import pytest

# Root of the yowo source tree
_SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "yowo"


# ---------------------------------------------------------------------------
# Test 1: from __future__ import annotations in all non-__init__.py files
# ---------------------------------------------------------------------------


def _all_non_init_py_files() -> list[Path]:
    """Return all .py files under src/yowo/ excluding __init__.py files."""
    return [p for p in _SRC_ROOT.rglob("*.py") if p.name != "__init__.py"]


@pytest.mark.parametrize("src_file", _all_non_init_py_files(), ids=lambda p: p.name)
def test_future_annotations_present(src_file: Path) -> None:
    """Every non-__init__.py source file must declare from __future__ import annotations."""
    source = src_file.read_text(encoding="utf-8")
    # Verify the file-level future import is present anywhere in the file.
    # PEP 236 requires it to appear before any executable code (after docstrings
    # and comments), but does not restrict it to the first N lines.
    assert "from __future__ import annotations" in source, (
        f"{src_file.relative_to(_SRC_ROOT.parent.parent)} is missing "
        f"`from __future__ import annotations`. All non-__init__.py source files "
        "must declare this to ensure annotation evaluation is deferred on Python 3.8."
    )
    # Additionally verify it appears in a module-level context (not inside a function/class)
    # by checking it comes before the first `def ` or `class ` line.
    lines = source.splitlines()
    future_lineno = next(
        (i for i, ln in enumerate(lines) if "from __future__ import annotations" in ln),
        None,
    )
    first_code_lineno = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.startswith(("def ", "class ", "import ", "from ")) and "from __future__" not in ln
        ),
        None,
    )
    if future_lineno is not None and first_code_lineno is not None:
        assert future_lineno < first_code_lineno, (
            f"{src_file.relative_to(_SRC_ROOT.parent.parent)}: "
            f"`from __future__ import annotations` at line {future_lineno + 1} "
            f"appears after first code at line {first_code_lineno + 1}."
        )


# ---------------------------------------------------------------------------
# Test 2: No __init__.py uses bare X | Y union syntax at runtime
# ---------------------------------------------------------------------------


def _all_init_py_files() -> list[Path]:
    """Return all __init__.py files under src/yowo/."""
    return list(_SRC_ROOT.rglob("__init__.py"))


class _BitOrAnnotationVisitor(ast.NodeVisitor):
    """Detect bare X | Y union expressions in annotation contexts.

    When `from __future__ import annotations` is absent, `X | Y` in function
    signatures and variable annotations is evaluated at runtime. On Python 3.8
    this raises TypeError because `type.__or__` does not exist until 3.10.
    """

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def _check_annotation(self, node: ast.expr | None, lineno: int) -> None:
        if node is None:
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            self.violations.append((lineno, ast.unparse(node)))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annotation(node.annotation, node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self._check_annotation(arg.annotation, node.lineno)
        if node.args.vararg:
            self._check_annotation(node.args.vararg.annotation, node.lineno)
        if node.args.kwarg:
            self._check_annotation(node.args.kwarg.annotation, node.lineno)
        self._check_annotation(node.returns, node.lineno)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


@pytest.mark.parametrize(
    "init_file", _all_init_py_files(), ids=lambda p: str(p.relative_to(_SRC_ROOT))
)
def test_no_runtime_union_syntax_in_init(init_file: Path) -> None:
    """No __init__.py may use bare X | Y union annotations (fails on Python 3.8)."""
    source = init_file.read_text(encoding="utf-8")
    # If the file has from __future__ import annotations, annotations are strings
    # and cannot cause runtime issues — skip the AST check.
    if "from __future__ import annotations" in source:
        pytest.skip(f"{init_file.name} has future annotations — no runtime risk")

    tree = ast.parse(source, filename=str(init_file))
    visitor = _BitOrAnnotationVisitor()
    visitor.visit(tree)

    assert visitor.violations == [], (
        f"{init_file.relative_to(_SRC_ROOT.parent.parent)} uses bare X | Y syntax "
        f"(Python 3.10+ only) in annotation contexts at runtime:\n"
        + "\n".join(f"  line {ln}: {expr}" for ln, expr in visitor.violations)
    )


# ---------------------------------------------------------------------------
# Test 2b: No runtime type aliases using lowercase generics (tuple[...], list[...])
# ---------------------------------------------------------------------------


class _RuntimeGenericVisitor(ast.NodeVisitor):
    """Detect runtime type aliases like `Point = tuple[float, float]`.

    `from __future__ import annotations` only affects annotation contexts (function
    signatures, variable annotations). Plain assignments like `X = tuple[...]` are
    evaluated at runtime and fail on Python 3.8 with TypeError: 'type' object is
    not subscriptable. Use `typing.Tuple` etc. for runtime type aliases.
    """

    _BUILTINS = frozenset({"tuple", "list", "dict", "set", "frozenset", "type"})

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        # Check: X = tuple[...] or X = list[...]
        self._check_subscript(node.value, node.lineno)
        self.generic_visit(node)

    def _check_subscript(self, node: ast.expr, lineno: int) -> None:
        if not isinstance(node, ast.Subscript):
            return
        if isinstance(node.value, ast.Name) and node.value.id in self._BUILTINS:
            self.violations.append((lineno, ast.unparse(node)))


def _all_py_files() -> list[Path]:
    """Return all .py files under src/yowo/."""
    return list(_SRC_ROOT.rglob("*.py"))


@pytest.mark.parametrize("src_file", _all_py_files(), ids=lambda p: str(p.relative_to(_SRC_ROOT)))
def test_no_runtime_lowercase_generic_aliases(src_file: Path) -> None:
    """No source file may use `X = tuple[...]` etc. — fails on Python 3.8.

    Use `typing.Tuple`, `typing.List`, etc. for runtime type aliases instead.
    `from __future__ import annotations` does NOT protect plain assignments.
    """
    source = src_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_file))
    visitor = _RuntimeGenericVisitor()
    visitor.visit(tree)

    assert visitor.violations == [], (
        f"{src_file.relative_to(_SRC_ROOT.parent.parent)} uses runtime lowercase generic "
        f"type aliases (Python 3.9+ only):\n"
        + "\n".join(f"  line {ln}: {expr}" for ln, expr in visitor.violations)
        + "\n  Fix: use typing.Tuple/List/Dict/Set/FrozenSet instead"
    )


# ---------------------------------------------------------------------------
# Test 3: StrEnumBase backport is usable on all Python versions
# ---------------------------------------------------------------------------


def test_str_enum_base_backport_usable() -> None:
    """StrEnumBase must behave like StrEnum: str(member) returns the value."""
    from yowo.types import StrEnumBase

    class _Color(StrEnumBase):  # type: ignore[misc]
        RED = "red"
        GREEN = "green"

    assert str(_Color.RED) == "red"
    assert str(_Color.GREEN) == "green"
    # Must be an enum
    assert isinstance(_Color.RED, enum.Enum)
    # Must be a string subclass (key property for StrEnum compatibility)
    assert isinstance(_Color.RED, str)


def test_str_enum_base_is_correct_type() -> None:
    """StrEnumBase must be enum.StrEnum on 3.11+ or (str, enum.Enum) shim on older."""
    from yowo.types import StrEnumBase

    if sys.version_info >= (3, 11):
        assert StrEnumBase is enum.StrEnum
    else:
        # The backport class must inherit from both str and enum.Enum
        assert issubclass(StrEnumBase, str)
        assert issubclass(StrEnumBase, enum.Enum)


# ---------------------------------------------------------------------------
# Test 4: All public package imports succeed without errors
# ---------------------------------------------------------------------------


def test_public_api_imports() -> None:
    """All documented public names must be importable from the yowo package."""
    import yowo

    # Convenience API
    assert yowo.detect is not None
    assert yowo.classify is not None
    assert yowo.detect_obb is not None

    # Engine classes
    assert yowo.InferenceEngine is not None
    assert yowo.ClassificationEngine is not None
    assert yowo.DetectionEngine is not None
    assert yowo.OBBEngine is not None

    # Config classes
    assert yowo.InferenceConfig is not None
    assert yowo.ClassificationConfig is not None
    assert yowo.OBBConfig is not None

    # Tracking
    assert yowo.ByteTracker is not None
    assert yowo.track_detections is not None

    # Types
    assert yowo.Detection is not None
    assert yowo.Frame is not None
    assert yowo.ModelSpec is not None
    assert yowo.BoundingBox is not None


def test_public_api_no_name_errors() -> None:
    """Importing from yowo must not raise AttributeError for documented names."""
    import yowo

    public_names = [name for name in yowo.__all__ if not name.startswith("_")]
    missing = [name for name in public_names if not hasattr(yowo, name)]
    assert missing == [], f"Missing from yowo namespace: {missing}"
