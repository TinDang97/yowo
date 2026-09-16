"""The sidecar describes the artifact, not the request that produced it.

Measured 2026-09-16 on yolo11n/fp32, which is why these exist: a
``yowo[pytorch]`` export writes a 612,255 B stub plus a 10,616,832 B
``.onnx.data``, records ``file_size_bytes: 612255``, names only the stub, and
fails to load from a directory holding what the sidecar names. With onnxslim
the ``.onnx`` is self-contained and the companion file is merely orphaned --
2.00x the recorded size on disk, named by nothing.

The checks here are the ones that need no weights, so they run in the unit
tier (``Quality Gate`` does not restore the weight cache). The ones that need
a real export live in ``tests/integration/test_export_artifact_selfcontained.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# A sidecar exactly as written before this node: no opset, no task, no file
# list, no total. Captured from the 2026-09-16 probe rather than invented, so
# the backward-compatibility check is against a real artifact of the old code.
PRE_CHANGE_SIDECAR: dict[str, Any] = {
    "model_name": "yolo11n",
    "format": "onnx",
    "precision": "fp32",
    "imgsz": 640,
    "batch_size": 1,
    "dynamic": True,
    "input_shape": [1, 3, 640, 640],
    "file_path": "/somewhere/yolo11n.onnx",
    "file_size_bytes": 612255,
    "created_at": "2026-09-16T02:42:18+00:00",
    "export_duration_sec": 7.31,
    "source_weights": "/somewhere/yolo11n.pt",
    "yowo_version": "2.4.0",
    "python_version": "3.11.15",
    "platform_system": "Darwin",
    "platform_machine": "arm64",
    "gpu_name": None,
    "calibration_data": None,
    "extra": {},
}


def _tiny_onnx(path: Path, *, opset: int, shape: list[int]) -> None:
    """A minimal real ONNX graph -- no weights, no registry, no download."""
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("Relu", ["images"], ["output0"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.save(model, str(path))


class TestSidecarBackwardCompatibility:
    def test_a_sidecar_without_the_new_fields_still_loads(self, tmp_path: Path) -> None:
        """M8 · E4 · A5 -- absent means "predates the field", not an error.

        Every artifact exported before today has a sidecar shaped like this
        one. If the new fields are required, ``load()`` raises on all of them
        and the loader cannot read what the exporter wrote.
        """
        from yowo.export._metadata import ExportMetadata

        p = tmp_path / "yolo11n.yowo.json"
        p.write_text(json.dumps(PRE_CHANGE_SIDECAR), encoding="utf-8")

        meta = ExportMetadata.load(p)

        assert meta.model_name == "yolo11n"
        assert meta.file_size_bytes == 612255
        # The new fields read as "this export predates the field".
        assert meta.opset is None
        assert meta.requested_opset is None
        assert meta.task == "detect"
        assert meta.artifact_files == []
        assert meta.total_size_bytes is None


class TestGraphReadback:
    def test_a_non_onnx_format_records_no_opset_rather_than_a_wrong_one(
        self, tmp_path: Path
    ) -> None:
        """A3 -- an opset is read from a graph or it is None; never guessed.

        tensorrt/openvino/coreml artifacts are produced by converters this
        node does not own, so their opset is unknown. ``None`` says unknown;
        any integer would be a claim nothing measured.
        """
        from yowo.export._readback import read_onnx_graph_facts

        not_a_graph = tmp_path / "model.engine"
        not_a_graph.write_bytes(b"not an onnx graph")

        facts = read_onnx_graph_facts(not_a_graph)

        assert facts.opset is None
        assert facts.input_shape is None

    def test_the_recorded_opset_is_read_from_the_graph_not_the_request(
        self, tmp_path: Path
    ) -> None:
        """Supports M2 at the unit level -- the readback itself is honest.

        Not one of the frozen CHECKS; the frozen M2 check needs a real export
        and lives in the integration module. This pins the primitive it uses.
        """
        from yowo.export._readback import read_onnx_graph_facts

        p = tmp_path / "tiny.onnx"
        _tiny_onnx(p, opset=18, shape=[1, 3, 224, 224])

        facts = read_onnx_graph_facts(p)

        assert facts.opset == 18
        assert facts.input_shape == [1, 3, 224, 224]


class TestCompanionFiles:
    def test_the_large_model_branch_names_its_companion_file(self, tmp_path: Path) -> None:
        """A4 -- above the protobuf ceiling, external data is unavoidable.

        Below it the companion file is internalized and removed. At or above
        it internalization would raise, so the companion file must SURVIVE and
        be NAMED -- the one case where an ``.onnx.data`` on disk is correct.
        Driven by lowering the ceiling rather than by building a 2 GB model.
        """
        from yowo.export._readback import internalize_external_data

        p = tmp_path / "tiny.onnx"
        _tiny_onnx(p, opset=18, shape=[1, 3, 640, 640])
        companion = tmp_path / "tiny.onnx.data"
        companion.write_bytes(b"\x00" * 1024)

        remaining = internalize_external_data(p, size_ceiling_bytes=1)

        assert companion.exists(), "above the ceiling the companion file must survive"
        assert remaining == ["tiny.onnx.data"], (
            "a companion file that survives must be named, or M1 cannot hold"
        )

    def test_internalization_removes_the_companion_below_the_ceiling(self, tmp_path: Path) -> None:
        """Supports M1/M5 at the unit level -- the other side of the branch.

        Not a frozen CHECK; A4's frozen check owns the above-ceiling leg and
        this pins the below-ceiling leg so the branch is not half-tested.
        """
        from yowo.export._readback import internalize_external_data

        p = tmp_path / "tiny.onnx"
        _tiny_onnx(p, opset=18, shape=[1, 3, 640, 640])
        companion = tmp_path / "tiny.onnx.data"
        companion.write_bytes(b"\x00" * 1024)

        remaining = internalize_external_data(p, size_ceiling_bytes=2 << 30)

        assert not companion.exists()
        assert remaining == []


class TestSidecarVerification:
    def test_export_reads_its_own_sidecar_back_before_returning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M7 -- ``ExportMetadata.load()`` gains its first caller in ``src/``.

        It had zero. That is why a sidecar could name one file while the
        directory held two, and nothing noticed: nothing ever read it back.
        A sidecar that does not describe the directory must raise here, at
        export time, not at someone else's deploy time.
        """
        from yowo.errors import ExportError
        from yowo.export._metadata import ExportMetadata
        from yowo.export._readback import verify_sidecar_describes_directory

        meta = ExportMetadata(
            **PRE_CHANGE_SIDECAR
            | {
                "file_path": str(tmp_path / "yolo11n.onnx"),
                "artifact_files": ["yolo11n.onnx"],
                "total_size_bytes": 10,
            }
        )
        (tmp_path / "yolo11n.onnx").write_bytes(b"\x00" * 10)
        (tmp_path / "yolo11n.onnx.data").write_bytes(b"\x00" * 99)
        sidecar = meta.save()

        with pytest.raises(ExportError, match="yolo11n.onnx.data"):
            verify_sidecar_describes_directory(sidecar, tmp_path)


class TestTheseChecksRunInCI:
    def test_export_sidecar_truth_runs_in_ci(self) -> None:
        """A1 -- the no-onnxslim leg must not be invisible the way it is today.

        ``onnxslim`` is pinned in the dev group, so every CI job has it and no
        job exercises the documented ``yowo[pytorch]`` environment. A check
        that proves the artifact loads without onnxslim is worth nothing if no
        CI command ever reaches the module it lives in.
        """
        import yaml

        integration = "tests/integration/test_export_artifact_selfcontained.py"
        assert (REPO_ROOT / integration).exists()

        commands: list[str] = []
        for wf in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
            doc = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
            triggers = doc.get("on", doc.get(True, {})) or {}
            if "pull_request" not in triggers:
                continue
            for job in (doc.get("jobs") or {}).values():
                for step in job.get("steps") or []:
                    if isinstance(step.get("run"), str) and "pytest" in step["run"]:
                        commands.append(step["run"])

        # Deliberately exact. This once read `integration in c or
        # "tests/integration" in c`, and the mutation sweep showed the second
        # clause made it unfailable: pointing the step at a different file in
        # the same directory kept it green. A check that any integration test
        # runs is not a check that THIS one does.
        assert any(integration in c for c in commands), (
            f"no pull_request CI command runs {integration}; "
            f"the no-onnxslim leg would be as invisible as the defect it covers. "
            f"pytest commands found: {commands}"
        )


class TestCompanionNamesCannotEscapeTheDirectory:
    def test_an_external_data_location_outside_the_output_dir_is_ignored(
        self, tmp_path: Path
    ) -> None:
        """Security residue, not a frozen CHECK -- found by the Verify lens.

        `location` is a string carried inside the graph, and
        ``internalize_external_data`` UNLINKS whatever ``external_companions``
        returns. A location of ``../victim`` would resolve outside the output
        directory and delete a file the export never wrote. The only graph
        that reaches this today is one produced seconds earlier by this same
        package, so no reachable exploit exists -- which is exactly when the
        guard is cheap to add.
        """
        import onnx
        from onnx import TensorProto, helper

        from yowo.export._readback import external_companions

        outside = tmp_path / "victim.bin"
        outside.write_bytes(b"someone else's file")
        work = tmp_path / "out"
        work.mkdir()

        init = TensorProto()
        init.name = "w"
        init.data_type = TensorProto.FLOAT
        init.dims.extend([1])
        init.data_location = TensorProto.EXTERNAL
        entry = init.external_data.add()
        entry.key, entry.value = "location", "../victim.bin"

        graph = helper.make_graph(
            [helper.make_node("Relu", ["images"], ["output0"])],
            "escape",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1])],
            initializer=[init],
        )
        p = work / "escape.onnx"
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), str(p))

        assert external_companions(p) == []
        assert outside.exists()


class TestTheFileSetAndTheArtifactItDescribes:
    def test_the_intermediate_onnx_is_named_not_orphaned(self, tmp_path: Path) -> None:
        """A11 -- a tensorrt/openvino run leaves its ONNX intermediate behind.

        `_convert_tensorrt` and `_convert_openvino` both consume an `.onnx`
        that `export_model` wrote into `output_dir` and neither removes it, so
        the directory ends up holding an artifact the sidecar's `file_path`
        does not point at. A11 took the reading that the sidecar NAMES it --
        a reader who deletes it cannot re-run the conversion.

        Bound here at `produced_files`, which is where the decision is made.
        HONEST LIMIT: this does not run a real tensorrt or openvino export.
        Neither package is in the dev group, so no test in this tier can. The
        residual is recorded on the node.
        """
        from yowo.export._readback import produced_files

        before = frozenset({"someone-elses.txt"})
        (tmp_path / "someone-elses.txt").write_text("x", encoding="utf-8")
        entry = tmp_path / "yolo26n.engine"
        entry.write_bytes(b"engine")
        (tmp_path / "yolo26n.onnx").write_bytes(b"the intermediate")

        names = produced_files(tmp_path, before, entry)

        assert names[0] == "yolo26n.engine", "the entry file comes first"
        assert "yolo26n.onnx" in names, "the intermediate the conversion left is named"
        assert "someone-elses.txt" not in names

    def test_the_sidecar_describes_the_last_artifact_in_the_chain(self, tmp_path: Path) -> None:
        """A12 -- for a converted format, the sidecar describes the OUTPUT.

        `export_model` reads its graph facts from `exported_path`, which for
        tensorrt and openvino is the converted artifact rather than the ONNX
        the conversion consumed. Reading the intermediate instead would record
        the shape and opset of a file `file_path` does not point at.

        Asserted on the primitive: handed the converted artifact, the reader
        reports unknowns rather than reaching for a neighbouring `.onnx`.
        """
        from yowo.export._readback import read_onnx_graph_facts

        _tiny_onnx(tmp_path / "yolo26n.onnx", opset=18, shape=[1, 3, 640, 640])
        engine = tmp_path / "yolo26n.engine"
        engine.write_bytes(b"not a graph")

        facts = read_onnx_graph_facts(engine)

        assert facts.opset is None, "the intermediate's opset must not leak onto the engine"
        assert facts.input_shape is None
