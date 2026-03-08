"""Unit tests for OBB export branch in export_model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from yowo.types import ExportFormat, ModelFamily, ModelSize, ModelSpec, Precision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obb_spec(
    family: ModelFamily = ModelFamily.YOLO11,
    size: ModelSize = ModelSize.NANO,
) -> ModelSpec:
    return ModelSpec(family=family, size=size, task="obb")


def _make_mock_model() -> MagicMock:
    model = MagicMock()
    fused = MagicMock()
    evaled = MagicMock()
    model.fuse.return_value = fused
    fused.eval.return_value = evaled
    evaled.half.return_value = evaled
    return model


# ---------------------------------------------------------------------------
# OBB export branch tests
# ---------------------------------------------------------------------------


def test_export_model_obb_calls_build_obb_model(tmp_path: Path) -> None:
    """export_model with task='obb' calls build_obb_model, not build_model."""
    spec = _obb_spec()
    mock_model = _make_mock_model()
    mock_meta = MagicMock()
    mock_meta.num_classes = 15
    mock_meta.input_height = 640
    mock_meta.input_width = 640

    with (
        patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch("yowo.arch.build_obb_model", return_value=mock_model) as mock_build_obb,
        patch("yowo.arch._weights.load_obb_weights") as mock_load_obb,
        patch("yowo.models._registry.get_obb", return_value=mock_meta),
        patch("yowo.export._exporter._export_onnx") as mock_export_onnx,
        patch("yowo.export._exporter.get_hardware_profile"),
        patch("yowo.export._exporter.ExportMetadata") as mock_export_meta_cls,
    ):
        fake_onnx = tmp_path / "yolo11n-obb.onnx"
        fake_onnx.write_bytes(b"fake")
        mock_export_onnx.side_effect = lambda *a, **kw: fake_onnx.write_bytes(b"fake")

        fake_meta = MagicMock()
        fake_meta.file_path = str(fake_onnx)
        fake_meta.file_size_bytes = 4
        fake_meta.save.return_value = None
        mock_export_meta_cls.return_value = fake_meta

        from yowo.export._exporter import export_model

        export_model(spec, ExportFormat.ONNX, tmp_path, precision=Precision.FP32)

        mock_build_obb.assert_called_once()
        mock_load_obb.assert_called_once()


def test_export_model_obb_does_not_call_build_model(tmp_path: Path) -> None:
    """export_model with task='obb' does NOT call build_model (detect path)."""
    spec = _obb_spec()
    mock_model = _make_mock_model()
    mock_meta = MagicMock()
    mock_meta.num_classes = 15
    mock_meta.input_height = 640
    mock_meta.input_width = 640

    with (
        patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch("yowo.arch.build_obb_model", return_value=mock_model),
        patch("yowo.arch._weights.load_obb_weights"),
        patch("yowo.models._registry.get_obb", return_value=mock_meta),
        patch("yowo.arch.build_model") as mock_build_model,
        patch("yowo.arch._weights.load_weights") as mock_load_weights,
        patch("yowo.export._exporter._export_onnx") as mock_export_onnx,
        patch("yowo.export._exporter.get_hardware_profile"),
        patch("yowo.export._exporter.ExportMetadata") as mock_export_meta_cls,
    ):
        fake_onnx = tmp_path / "yolo11n-obb.onnx"
        fake_onnx.write_bytes(b"fake")
        mock_export_onnx.side_effect = lambda *a, **kw: fake_onnx.write_bytes(b"fake")

        fake_meta = MagicMock()
        fake_meta.file_path = str(fake_onnx)
        fake_meta.file_size_bytes = 4
        fake_meta.save.return_value = None
        mock_export_meta_cls.return_value = fake_meta

        from yowo.export._exporter import export_model

        export_model(spec, ExportFormat.ONNX, tmp_path, precision=Precision.FP32)

        mock_build_model.assert_not_called()
        mock_load_weights.assert_not_called()


def test_export_model_obb_stem_includes_obb_suffix(tmp_path: Path) -> None:
    """OBB export uses model_stem with -obb suffix to avoid collision."""
    spec = _obb_spec()
    mock_model = _make_mock_model()
    mock_meta = MagicMock()
    mock_meta.num_classes = 15
    mock_meta.input_height = 640
    mock_meta.input_width = 640

    captured_paths: list[Path] = []

    def _capture_export(model: object, dummy: object, path: Path, **kw: object) -> None:
        captured_paths.append(path)
        path.write_bytes(b"fake")

    with (
        patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch("yowo.arch.build_obb_model", return_value=mock_model),
        patch("yowo.arch._weights.load_obb_weights"),
        patch("yowo.models._registry.get_obb", return_value=mock_meta),
        patch("yowo.export._exporter._export_onnx", side_effect=_capture_export),
        patch("yowo.export._exporter.get_hardware_profile"),
        patch("yowo.export._exporter.ExportMetadata") as mock_export_meta_cls,
    ):
        fake_onnx = tmp_path / "yolo11n-obb.onnx"
        fake_onnx.write_bytes(b"x")
        fake_meta = MagicMock()
        fake_meta.file_path = str(fake_onnx)
        fake_meta.file_size_bytes = 1
        fake_meta.save.return_value = None
        mock_export_meta_cls.return_value = fake_meta

        from yowo.export._exporter import export_model

        export_model(spec, ExportFormat.ONNX, tmp_path, precision=Precision.FP32)

    # The ONNX path should contain "-obb" in the filename
    assert captured_paths, "No ONNX path was captured"
    obb_in_name = "obb" in captured_paths[0].name
    assert obb_in_name, f"Expected '-obb' in stem, got: {captured_paths[0].name}"


# ---------------------------------------------------------------------------
# KV-cache guard tests (INT-P0 regression)
# ---------------------------------------------------------------------------


def _make_export_scaffolding(
    spec: ModelSpec,
    tmp_path: Path,
    task: str,
) -> dict[str, object]:
    """Return a common patch context dict for kv_cache guard tests."""
    mock_model = _make_mock_model()
    mock_meta = MagicMock()
    mock_meta.num_classes = 15
    mock_meta.input_height = 640
    mock_meta.input_width = 640
    return {
        "mock_model": mock_model,
        "mock_meta": mock_meta,
    }


def test_kv_cache_guard_obb_skips_kv_export(tmp_path: Path) -> None:
    """kv_cache=True + task=obb must NOT call _export_onnx_kv (INT-P0 guard)."""
    spec = _obb_spec()
    mock_model = _make_mock_model()
    mock_meta = MagicMock()
    mock_meta.num_classes = 15
    mock_meta.input_height = 640
    mock_meta.input_width = 640

    with (
        patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch("yowo.arch.build_obb_model", return_value=mock_model),
        patch("yowo.arch._weights.load_obb_weights"),
        patch("yowo.models._registry.get_obb", return_value=mock_meta),
        patch("yowo.export._exporter._export_onnx_kv") as mock_kv_export,
        patch("yowo.export._exporter._export_onnx") as mock_export_onnx,
        patch("yowo.export._exporter.get_hardware_profile"),
        patch("yowo.export._exporter.ExportMetadata") as mock_export_meta_cls,
    ):
        fake_onnx = tmp_path / "yolo11n-obb.onnx"
        fake_onnx.write_bytes(b"fake")
        mock_export_onnx.side_effect = lambda *a, **kw: fake_onnx.write_bytes(b"fake")
        fake_meta = MagicMock()
        fake_meta.file_path = str(fake_onnx)
        fake_meta.file_size_bytes = 4
        fake_meta.save.return_value = None
        mock_export_meta_cls.return_value = fake_meta

        from yowo.export._exporter import export_model

        export_model(spec, ExportFormat.ONNX, tmp_path, precision=Precision.FP32, kv_cache=True)

        mock_kv_export.assert_not_called()


def test_kv_cache_guard_classify_skips_kv_export(tmp_path: Path) -> None:
    """kv_cache=True + task=classify must NOT call _export_onnx_kv."""
    spec = ModelSpec(
        family=ModelFamily.YOLO11,
        size=ModelSize.NANO,
        task="classify",
    )
    mock_model = _make_mock_model()
    mock_meta = MagicMock()
    mock_meta.num_classes = 1000
    mock_meta.input_height = 224
    mock_meta.input_width = 224

    with (
        patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch("yowo.arch.build_classify_model", return_value=mock_model),
        patch("yowo.arch._weights.load_classify_weights"),
        patch("yowo.models._registry.get_cls", return_value=mock_meta),
        patch("yowo.export._exporter._export_onnx_kv") as mock_kv_export,
        patch("yowo.export._exporter._export_onnx") as mock_export_onnx,
        patch("yowo.export._exporter.get_hardware_profile"),
        patch("yowo.export._exporter.ExportMetadata") as mock_export_meta_cls,
    ):
        # classify stem is "yolo11n" (no task suffix)
        fake_onnx = tmp_path / "yolo11n.onnx"
        fake_onnx.write_bytes(b"fake")
        mock_export_onnx.side_effect = lambda *a, **kw: fake_onnx.write_bytes(b"fake")
        fake_meta = MagicMock()
        fake_meta.file_path = str(fake_onnx)
        fake_meta.file_size_bytes = 4
        fake_meta.save.return_value = None
        mock_export_meta_cls.return_value = fake_meta

        from yowo.export._exporter import export_model

        export_model(spec, ExportFormat.ONNX, tmp_path, precision=Precision.FP32, kv_cache=True)

        mock_kv_export.assert_not_called()


def test_kv_cache_guard_detect_calls_kv_export(tmp_path: Path) -> None:
    """kv_cache=True + task=detect MUST call _export_onnx_kv (positive case).

    The kv-cache code path branches on task membership: task not in ("classify", "obb").
    For detect tasks, the branch is taken and _export_onnx_kv is called.
    The assert isinstance(model, YOLOModel) check is satisfied by constructing
    a minimal YOLOModel subclass whose fuse/eval chain also returns a YOLOModel
    instance, since export_model reassigns model via model = model.fuse().eval().
    """
    import yowo.arch._yolo as _yolo_module

    spec = ModelSpec(
        family=ModelFamily.YOLO11,
        size=ModelSize.NANO,
        task="detect",
    )

    real_yolo_model_cls = _yolo_module.YOLOModel

    class _FakeYOLOModel(real_yolo_model_cls):  # type: ignore[misc]
        """Minimal subclass that satisfies isinstance through the fuse/eval chain."""

        def __init__(self) -> None:
            pass  # skip super().__init__ — no real model needed

        def fuse(self) -> _FakeYOLOModel:
            return self

        def eval(self) -> _FakeYOLOModel:  # type: ignore[override]
            return self

        def half(self) -> _FakeYOLOModel:
            return self

    fake_model = _FakeYOLOModel()

    with (
        patch("yowo.export._exporter.resolve_weights", return_value=Path("/fake/weights.pt")),
        patch("yowo.arch.build_model", return_value=fake_model),
        patch("yowo.arch._weights.load_weights"),
        patch("yowo.export._exporter._export_onnx_kv") as mock_kv_export,
        patch("yowo.export._exporter.get_hardware_profile"),
        patch("yowo.export._exporter.ExportMetadata") as mock_export_meta_cls,
        # YOLOKVWrapper is a local import inside the kv-branch; patch at source module
        patch("yowo.export._kv_wrapper.YOLOKVWrapper"),
    ):
        fake_onnx = tmp_path / "yolo11n.onnx"
        fake_onnx.write_bytes(b"fake")
        mock_kv_export.side_effect = lambda *a, **kw: fake_onnx.write_bytes(b"fake")
        fake_meta = MagicMock()
        fake_meta.file_path = str(fake_onnx)
        fake_meta.file_size_bytes = 4
        fake_meta.save.return_value = None
        mock_export_meta_cls.return_value = fake_meta

        from yowo.export._exporter import export_model

        export_model(spec, ExportFormat.ONNX, tmp_path, precision=Precision.FP32, kv_cache=True)

        mock_kv_export.assert_called_once()
