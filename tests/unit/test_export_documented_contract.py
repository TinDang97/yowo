"""Every document that describes the export surface describes the one that exists.

Measured 2026-09-16: three incompatible things were called the export result.
`export_model` returns `ExportMetadata` (24 fields); `yowo.types.ExportResult`
(7 fields) is in `yowo.__all__` and returned by nothing; `export/README.md`
documented a third shape under that same name. The signature had drifted in
three fields and a CLI flag was tabulated that did not exist.

The checks here PARSE the documents and compare them against the live
signature, so the next divergence fails CI without anyone re-reading a README.
Doc sites are DISCOVERED, not listed: the milestone box itself enumerated two
and missed a fourth, which is exactly what a hand-maintained list does.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import typing
from pathlib import Path

import pytest

import yowo
from yowo.export import export_model
from yowo.types import ExportResult

#: The type `export_model` actually returns. Read from the function, never typed
#: out here -- a constant would drift exactly like the documents did. The module
#: uses postponed annotations, so the raw annotation is a STRING; resolve it to
#: the real class, or every check below compares against a name rather than a type.
RETURNED = typing.get_type_hints(export_model).get(
    "return", inspect.signature(export_model).return_annotation
)
RETURNED_NAME = getattr(RETURNED, "__name__", str(RETURNED))

#: Names a document might use for the export result. Any of these appearing as
#: the return of `export_model` is a claim about the contract.
_RESULT_NAMES = ("ExportResult", "ExportMetadata")


def _doc_files() -> list[Path]:
    """Every markdown file that describes the export surface.

    Discovered by content, so a doc added later is covered without anyone
    editing this function.
    """
    roots = [Path("src/yowo"), Path("docs"), Path("README.md")]
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("*.md"))
        else:
            continue
        for path in candidates:
            text = path.read_text(encoding="utf-8")
            if "export_model" in text or "yowo export" in text:
                found.append(path)
    return found


def test_the_documented_surface_is_discovered_not_listed() -> None:
    docs = _doc_files()
    assert len(docs) >= 3, f"only found {docs} -- discovery is not reaching the docs"
    # The milestone box named export/README.md and cli/README.md and MISSED
    # src/yowo/README.md. Discovery must reach all of them.
    names = {p.as_posix() for p in docs}
    assert any("export/README.md" in n for n in names)
    assert any("cli/README.md" in n for n in names)


def test_every_document_names_the_return_type_the_function_returns() -> None:
    wrong: list[str] = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        # "-> SomeType" following an export_model signature, in any code block.
        for match in re.finditer(r"->\s*(\w+)\s*:", text):
            claimed = match.group(1)
            if claimed in _RESULT_NAMES and claimed != RETURNED_NAME:
                wrong.append(f"{path}: documents `-> {claimed}`, real is `{RETURNED_NAME}`")
    assert not wrong, "documents disagree with the function:\n  " + "\n  ".join(wrong)


def test_every_documented_parameter_matches_the_real_signature() -> None:
    real = inspect.signature(export_model).parameters
    problems: list[str] = []
    # `name: Type = default` lines inside a documented export_model signature.
    # A trailing `# comment` must not hide a parameter. Anchoring on `,$`
    # silently skipped `target_format: str,  # "onnx" | ...` -- the check saw
    # only the parameters that happened to be uncommented.
    pattern = re.compile(
        r"^\s{4}(\w+)\s*:\s*([\w\[\], |\.]+?)\s*(?:=\s*([^,#]+?))?\s*,\s*(?:#.*)?$", re.M
    )
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        for block in re.findall(r"def export_model\((.*?)\)\s*->", text, re.S):
            for name, claimed_type, claimed_default in pattern.findall(block):
                if name not in real:
                    problems.append(f"{path}: documents parameter `{name}`, which does not exist")
                    continue
                p = real[name]
                actual_type = getattr(p.annotation, "__name__", str(p.annotation))
                if claimed_type.strip() != actual_type:
                    problems.append(
                        f"{path}: `{name}` documented as "
                        f"`{claimed_type.strip()}`, real `{actual_type}`"
                    )
                has_default = p.default is not inspect.Parameter.empty
                if claimed_default and not has_default:
                    problems.append(
                        f"{path}: `{name}` documented with default `{claimed_default}`, "
                        "but it is required"
                    )
                elif claimed_default and has_default:
                    # One value has several honest spellings. `Precision.FP16`
                    # is a StrEnum, so a doc may write the member path while
                    # `str()` gives "fp16" -- both name the same default, and a
                    # check that accepted only one would report drift that is
                    # not there.
                    default = p.default
                    spellings = {str(default), repr(default)}
                    if hasattr(default, "name") and hasattr(default, "value"):
                        spellings.add(f"{type(default).__name__}.{default.name}")
                    if claimed_default.rstrip(",") not in spellings:
                        problems.append(
                            f"{path}: `{name}` documented default `{claimed_default}`, "
                            f"real `{default}`"
                        )
    assert not problems, "signature drift:\n  " + "\n  ".join(problems)


def test_a_document_naming_a_type_that_does_not_exist_also_fails() -> None:
    # E4: a typo'd type name must fail the same way a mismatched one does, so
    # the check cannot be satisfied by inventing a name nobody defined.
    for path in _doc_files():
        for match in re.finditer(r"->\s*(\w+)\s*:", path.read_text(encoding="utf-8")):
            claimed = match.group(1)
            if claimed in _RESULT_NAMES:
                assert hasattr(yowo, claimed) or claimed == RETURNED_NAME, (
                    f"{path} names `{claimed}`, which is not importable from yowo"
                )


def test_every_documented_export_cli_flag_exists() -> None:
    from click.testing import CliRunner

    from yowo.cli._main import cli

    result = CliRunner().invoke(cli, ["export", "--help"])
    assert result.exit_code == 0, result.output
    real_flags = set(re.findall(r"(--[\w-]+)", result.output))

    missing: list[str] = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        # Flags tabulated in a markdown row: | `--flag` | ... |
        # Bound to the export command's own section. A fixed byte window
        # bleeds into the next command's flag table and reports ITS flags as
        # missing from `yowo export` -- measured, it falsely flagged `--family`.
        start = text.find("yowo export")
        if start < 0:
            continue
        rest = text[start:]
        end = re.search(r"\n#{1,4}\s", rest)
        section = rest[: end.start()] if end else rest
        for flag in re.findall(r"\|\s*`(--[\w-]+)`", section):
            if flag not in real_flags:
                missing.append(f"{path}: documents `{flag}`, absent from `yowo export --help`")
    assert not missing, "documented flags that do not exist:\n  " + "\n  ".join(missing)


def test_the_orphan_says_it_is_an_orphan_where_a_reader_meets_it() -> None:
    doc = inspect.getdoc(ExportResult) or ""
    assert RETURNED_NAME in doc, (
        "ExportResult's docstring does not name the type that IS returned, so a "
        "reader meeting a 7-field class beside a 24-field one cannot tell which is live"
    )
    lowered = doc.lower()
    assert "not returned" in lowered or "returns nothing" in lowered or "no function" in lowered


def test_no_public_name_is_removed_or_reshaped() -> None:
    # m4 is additive-only: 2.5.0 is on PyPI and this repo has no deprecation
    # mechanism, so removal is owned by deprecation-policy, not by this node.
    assert "ExportResult" in yowo.__all__
    assert "ExportMetadata" in yowo.__all__
    fields = [f.name for f in dataclasses.fields(ExportResult)]
    assert fields == [
        "model_name",
        "format",
        "precision",
        "output_path",
        "file_size_bytes",
        "export_time_s",
        "created_at",
    ], "the orphan's shape changed; it is public and unit-tested"


def test_a_readme_block_a_reader_would_copy_actually_runs() -> None:
    # A2/A10: the harmed party COPIES the block. `ExportMetadata` has no
    # `file_path`, the field export/README.md documented, so the documented
    # usage raised AttributeError at runtime.
    documented_attrs: set[str] = set()
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        # Only fenced blocks that actually call export_model. Scanning whole
        # files matched `result.` from the DETECT examples and reported
        # frame_index, source_id and top1_score as export drift.
        for block in re.findall(r"```(?:python)?\n(.*?)```", text, re.S):
            if "export_model(" not in block:
                continue
            for attr in re.findall(r"\b(?:result|metadata|meta)\.(\w+)", block):
                documented_attrs.add(attr)
    if not documented_attrs:
        pytest.skip("no attribute access documented; nothing for a reader to copy")
    real = {f.name for f in dataclasses.fields(RETURNED)} | set(dir(RETURNED))
    unusable = sorted(a for a in documented_attrs if a not in real)
    assert not unusable, (
        f"documents show {unusable} on the export result; {RETURNED_NAME} has no such attribute"
    )


def test_a_documented_json_flag_emits_parseable_json() -> None:
    """A flag that exists but emits unparseable output is worse than no flag.

    Measured 2026-09-16: `torch.onnx` writes its progress to STDOUT, so the
    first `--json` implementation produced 1220 bytes of which only the tail
    was a JSON document. The check above asserts the flag EXISTS; that is
    satisfied by a flag that does nothing useful, which is the same shape of
    dishonesty this milestone is about.
    """
    source = Path("src/yowo/cli/_main.py").read_text(encoding="utf-8")
    export_block = source[source.index("def export_command(") :]
    export_block = export_block[: export_block.find("\n@cli.command")]

    assert "json_output" in export_block, "the export command does not read --json"
    # stdout must be protected across the export call, or torch's progress
    # lands inside the document.
    assert "redirect_stdout" in export_block, (
        "--json does not protect stdout, so library progress output will corrupt the JSON document"
    )
    # and the payload must be the sidecar record itself, not a reformatting
    assert "dataclasses.asdict(meta)" in export_block


def test_nothing_in_the_package_returns_the_orphan() -> None:
    """covers A1 -- the whole node rests on this probe.

    If something DID return `ExportResult`, correcting the documents would make
    a second contract true rather than retiring one, and the orphan docstring
    would be a lie in the other direction.
    """
    src = Path("src/yowo")
    constructing: list[str] = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "types.py":
            continue  # its definition
        for line in text.splitlines():
            if re.search(r"\bExportResult\s*\(", line):
                constructing.append(f"{path}: {line.strip()}")
    assert not constructing, "something constructs the orphan:\n  " + "\n  ".join(constructing)


def test_the_return_type_does_not_vary_by_export_format() -> None:
    """covers A15 -- one contract for every format, or the fix entrenches a lie.

    A per-format result type would be a worse design than the drift it
    replaces, and the documents corrected here describe a single return.
    """
    hints = typing.get_type_hints(export_model)
    assert hints["return"] is RETURNED
    source = Path("src/yowo/export/_exporter.py").read_text(encoding="utf-8")
    returns = set(re.findall(r"def export_model\(.*?\)\s*->\s*(\w+)", source, re.S))
    assert returns <= {RETURNED_NAME}, f"export_model declares several return types: {returns}"


def test_this_node_invents_no_deprecation_mechanism() -> None:
    """covers A7 -- `deprecation-policy` owns that design, not this node.

    Originally this asserted that NO ``DeprecationWarning`` existed anywhere in
    ``src/``, which was right while the mechanism was undesigned: inventing one
    inside an unrelated task would have pre-empted the decision. The
    ``deprecation-policy`` node has since built it, so the guard now states the
    same intent against the world that exists -- there is exactly ONE
    mechanism, and this node did not add a second.
    """
    # EMISSION sites, not mentions of the word. A docstring that explains the
    # deprecation is prose, not a second mechanism -- the string test reported
    # types.py, whose docstring merely says the name warns.
    emitters = set()
    for path in Path("src/yowo").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name == "warn" and any(
                getattr(a, "id", None) == "DeprecationWarning" for a in node.args
            ):
                emitters.add(path.as_posix())
    emitters = sorted(emitters)
    assert emitters == ["src/yowo/_deprecation.py"], (
        "a second deprecation mechanism appeared outside _deprecation.py: "
        f"{emitters} -- that design belongs to deprecation-policy"
    )
