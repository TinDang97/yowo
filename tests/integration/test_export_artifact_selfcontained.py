"""An export leaves exactly the files its sidecar names, and they load.

Measured 2026-09-16, yolo11n/fp32/dynamic, darwin-arm64, before any of this
existed:

    with onnxslim      .onnx 10,641,583 B self-contained (external 0/194)
                       .onnx.data 10,616,832 B  -- named by nothing
                       sidecar file_size_bytes 10,641,583 vs 21,258,415 on
                       disk (2.00x)               clean-room load PASS
    without onnxslim   .onnx    612,255 B stub (external 130/201)
                       .onnx.data 10,616,832 B  -- required, named by nothing
                       sidecar file_size_bytes    612,255 vs 11,229,087
                       (18.34x)                   clean-room load FAIL
    classify           graph input [batch,3,224,224], sidecar [1,3,640,640]
    both tasks         wrote yolo11n.onnx, model_name yolo11n -- one overwrote
                       the other

Those numbers are DOCUMENTATION of the defect on one machine on one day, not
assertions (Q11). What is asserted is the property: the sidecar agrees with
the directory, and the artifact loads from what the sidecar names.

This module lives in the integration tier because a real export needs the
weight cache, which the `Quality Gate` job does not restore. It is run by
`parity.yml`'s `Export Parity` job, which does -- and
`test_export_sidecar_truth_runs_in_ci` fails if that stops being true.
"""

from __future__ import annotations

import builtins
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

pytestmark = pytest.mark.integration

_FAMILY = ModelFamily.YOLO26
_SIZE = ModelSize.NANO


@pytest.fixture()
def block_onnxslim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented `yowo[pytorch]` install, which no CI job otherwise has.

    `onnxslim` is pinned in the dev group, so every job imports it and the
    environment `_exporter.py:94`'s own DependencyError names is exercised
    nowhere. Blocking the import is the only way to reach it from a suite that
    installs the dev group.
    """
    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxslim" or name.startswith("onnxslim."):
            raise ImportError("blocked: simulating a yowo[pytorch] install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _blocking_onnxslim() -> Callable[..., Any]:
    """An ``__import__`` that refuses onnxslim and passes everything else.

    Defined at module scope rather than inside the loop so the real
    ``__import__`` is captured at definition time -- a closure over a loop
    variable would rebind (ruff B023) and, worse, could recurse into a
    replaced ``__import__`` on the second leg.
    """
    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxslim" or name.startswith("onnxslim."):
            raise ImportError("blocked: simulating a yowo[pytorch] install")
        return real_import(name, *args, **kwargs)

    return guarded


def _export(out: Path, *, task: str = "detect") -> Any:
    from yowo.export import export_model

    return export_model(
        ModelSpec(family=_FAMILY, size=_SIZE, task=task),
        ExportFormat.ONNX,
        out,
        precision=Precision.FP32,
        dynamic_batch=True,
    )


def _on_disk(out: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in out.iterdir() if not p.name.endswith(".yowo.json")}


@pytest.fixture(scope="module")
def exported(verified_weight: Path) -> Iterator[tuple[Any, Path]]:
    """One real detect export, shared -- each one costs seconds."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        yield _export(out), out


class TestTheSidecarAgreesWithTheDirectory:
    def test_the_sidecar_names_every_file_the_export_wrote(
        self, exported: tuple[Any, Path]
    ) -> None:
        """M1 · R:ORPHAN · E6 -- no file on disk that nothing names.

        The orphaned `.onnx.data` is the whole point: 10,616,832 B that a
        deploying operator can neither safely ship nor safely delete, because
        nothing tells them which it is.
        """
        meta, out = exported
        assert set(meta.artifact_files) == set(_on_disk(out))

    def test_the_sidecar_does_not_claim_a_file_it_did_not_write(
        self, verified_weight: Path
    ) -> None:
        """E6 -- `output_dir` is a user directory that may hold anything.

        Supports A6. Not a frozen CHECK on its own; it holds the other edge of
        M1 so that "names every file" cannot be satisfied by claiming all of
        them.
        """
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            stranger = out / "notes.txt"
            stranger.write_text("someone else's file", encoding="utf-8")
            meta = _export(out)
            assert "notes.txt" not in meta.artifact_files
            assert stranger.exists()

    def test_the_recorded_size_accounts_for_every_produced_file(
        self, exported: tuple[Any, Path]
    ) -> None:
        """M4 -- `file_size_bytes` understated the artifact by 2.00x-18.34x."""
        meta, out = exported
        assert meta.total_size_bytes == sum(_on_disk(out).values())


class TestTheSidecarDescribesTheGraph:
    def test_the_recorded_opset_is_the_graphs_opset(self, exported: tuple[Any, Path]) -> None:
        """M2 · R:REQUEST-AS-FACT · A9 -- 17 was asked for, 18 was produced."""
        import onnx

        meta, _ = exported
        graph = onnx.load(meta.file_path, load_external_data=False)
        produced = {o.version for o in graph.opset_import if o.domain in ("", "ai.onnx")}

        assert meta.opset in produced
        if meta.requested_opset is not None:
            assert meta.requested_opset != meta.opset, (
                "requested_opset is recorded only when the request was not honoured"
            )

    def test_the_recorded_input_shape_is_the_graphs_input_shape(
        self, verified_weight: Path
    ) -> None:
        """M3 · R:REQUEST-AS-FACT · E2 -- classify is 224 under a 640 request.

        The detect path hides this defect because request and graph agree at
        640. Classify is where copying the request shows.
        """
        import onnx

        with tempfile.TemporaryDirectory() as td:
            meta = _export(Path(td), task="classify")
            graph = onnx.load(meta.file_path, load_external_data=False)
            dims = [
                d.dim_value if d.HasField("dim_value") else None
                for d in graph.graph.input[0].type.tensor_type.shape.dim
            ]
            assert meta.input_shape[2:] == dims[2:]
            assert meta.input_shape[2:] == [224, 224]
            # `imgsz` describes the graph too. A classify export ignores the
            # argument -- the model uses the registry's 224 -- so a recorded
            # 640 is the same lie in a field a reader skims sooner.
            assert meta.imgsz == 224


class TestTheArtifactLoadsFromWhatTheSidecarNames:
    def test_an_export_loads_from_only_the_files_the_sidecar_names(
        self, verified_weight: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M5 · E1 -- the clean-room load, on BOTH installs, in one check.

        Deliberately not parametrized: the frozen CHECKS line declares this
        name, and a parametrized test reports as `name[slim]`, which is not
        the name that was frozen. Tests move to fit the declaration, not the
        other way round.

        Without onnxslim this failed before this node with
        `ValidationError: Data of TensorProto (backbone.stem.conv.weight)
        should be stored in .../yolo26n.onnx.data`.
        """
        import onnx

        for leg in ("slim", "no-slim"):
            with monkeypatch.context() as m:
                if leg == "no-slim":
                    m.setattr(builtins, "__import__", _blocking_onnxslim())

                with tempfile.TemporaryDirectory() as td:
                    out = Path(td)
                    meta = _export(out)
                    with tempfile.TemporaryDirectory() as clean:
                        for name in meta.artifact_files:
                            shutil.copy2(out / name, Path(clean) / name)
                        onnx.load(
                            str(Path(clean) / Path(meta.file_path).name),
                            load_external_data=True,
                        )

    def test_an_export_without_onnxslim_produces_a_loadable_artifact(
        self, verified_weight: Path, block_onnxslim: None
    ) -> None:
        """A2 · E1 -- `yowo[pytorch]` is a real environment, not a mis-install.

        Distinct from the check above: that one loads whatever the sidecar
        names, which a correct sidecar naming two files would satisfy. This
        one asserts the artifact is SELF-CONTAINED, so the single `.onnx` a
        user copies to a server is the whole model.
        """
        import onnx

        with tempfile.TemporaryDirectory() as td:
            meta = _export(Path(td))
            graph = onnx.load(meta.file_path, load_external_data=False)
            external = [
                i.name
                for i in graph.graph.initializer
                if i.data_location == onnx.TensorProto.EXTERNAL
            ]
            assert external == [], f"{len(external)} initializers still live outside the .onnx"


class TestTwoTasksAreTwoArtifacts:
    def test_a_classify_export_does_not_overwrite_a_detect_export(
        self, verified_weight: Path
    ) -> None:
        """M6 · R:COLLIDE · E3 -- both wrote `yolo11n.onnx` under one name."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            detect = _export(out, task="detect")
            classify = _export(out, task="classify")

            assert detect.file_path != classify.file_path
            assert detect.task == "detect"
            assert classify.task == "classify"
            assert Path(detect.file_path).exists(), "the detect artifact was overwritten"
            assert len([p for p in out.iterdir() if p.suffix == ".json"]) == 2


class TestAFailureThatChangedTheOutputIsRecorded:
    def test_a_swallowed_opset_conversion_failure_is_recorded_not_hidden(
        self, exported: tuple[Any, Path]
    ) -> None:
        """R:SILENT-DEGRADE · E5 · A8 -- the downconversion raises, today.

        `No Adapter To Version $17 for Resize` on detect. The export returns
        success at 18 either way; what must not happen is returning success
        with nothing recorded about the 17 that was asked for and not given.
        """
        meta, _ = exported
        assert meta.opset is not None
        if meta.opset != 17:
            assert meta.requested_opset == 17, (
                "the export produced an opset it was not asked for and recorded no trace of it"
            )
