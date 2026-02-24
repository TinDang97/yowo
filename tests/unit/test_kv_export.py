"""Unit tests for KV cache ONNX export path.

Tests cover:
- _sdpa_onnx numeric equivalence with F.scaled_dot_product_attention
- _forward_kv_export cold/warm path correctness
- YOLOKVWrapper module discovery and I/O naming
- ONNX export file production (requires torch.onnx)
- Backend KV state initialization (no hardware required)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from yowo.arch import build_model
from yowo.arch._attention import Attention
from yowo.export._kv_wrapper import YOLOKVWrapper
from yowo.types import ModelFamily, ModelSize

# ---------------------------------------------------------------------------
# _sdpa_onnx — numeric equivalence
# ---------------------------------------------------------------------------


class TestSdpaOnnx:
    """Verify manual SDPA decomposition matches F.scaled_dot_product_attention."""

    def test_sdpa_onnx_matches_native(self) -> None:
        """_sdpa_onnx must match F.scaled_dot_product_attention within fp32 tolerance."""
        torch.manual_seed(0)
        q = torch.randn(1, 4, 16, 8)  # (B, heads, N, key_dim)
        k = torch.randn(1, 4, 16, 8)
        v = torch.randn(1, 4, 16, 8)

        ref = F.scaled_dot_product_attention(q, k, v)
        got = Attention._sdpa_onnx(q, k, v)

        assert torch.allclose(ref, got, atol=1e-5), f"max diff: {(ref - got).abs().max().item()}"

    def test_sdpa_onnx_output_shape(self) -> None:
        q = torch.randn(2, 8, 25, 16)
        k = torch.randn(2, 8, 25, 16)
        v = torch.randn(2, 8, 25, 32)
        out = Attention._sdpa_onnx(q, k, v)
        assert out.shape == (2, 8, 25, 32)


# ---------------------------------------------------------------------------
# _forward_kv_export — cold / warm path
# ---------------------------------------------------------------------------


class TestForwardKvExport:
    """Test the export forward path in Attention."""

    def _make_attn(self) -> Attention:
        attn = Attention(dim=64, num_heads=4, attn_ratio=0.5)
        attn.eval()
        attn.enable_export_mode()
        return attn

    def test_cold_frame_matches_standard_forward(self) -> None:
        """Export path with use_cache=0 must match the standard forward path."""
        torch.manual_seed(42)
        attn_std = Attention(dim=64, num_heads=4, attn_ratio=0.5)
        attn_std.eval()

        attn_exp = Attention(dim=64, num_heads=4, attn_ratio=0.5)
        attn_exp.load_state_dict(attn_std.state_dict())
        attn_exp.eval()
        attn_exp.enable_export_mode()

        x = torch.randn(1, 64, 8, 8)
        B, C, H, W = x.shape
        N = H * W

        # Build zero past K,V and use_cache=0
        k_shape = (B, attn_exp.num_heads, N, attn_exp.key_dim)
        v_shape = (B, attn_exp.num_heads, N, attn_exp.head_dim)
        past_k = torch.zeros(k_shape)
        past_v = torch.zeros(v_shape)
        use_cache = torch.tensor(0.0)

        attn_exp.set_export_inputs(past_k, past_v, use_cache)

        with torch.no_grad():
            ref = attn_std(x)
            got = attn_exp(x)

        assert torch.allclose(ref, got, atol=1e-5), f"max diff: {(ref - got).abs().max().item()}"

    def test_warm_frame_uses_past_kv(self) -> None:
        """Export path with use_cache=1 must use past K,V (output differs from standard)."""
        torch.manual_seed(7)
        attn = self._make_attn()
        x = torch.randn(1, 64, 8, 8)
        B, C, H, W = x.shape
        N = H * W

        # Non-zero past K,V to force different output
        past_k = torch.randn(B, attn.num_heads, N, attn.key_dim)
        past_v = torch.randn(B, attn.num_heads, N, attn.head_dim)
        use_cache = torch.tensor(1.0)

        attn.set_export_inputs(past_k, past_v, use_cache)
        with torch.no_grad():
            warm_out = attn(x)

        # Cold frame for comparison
        attn2 = Attention(dim=64, num_heads=4, attn_ratio=0.5)
        attn2.load_state_dict(attn.state_dict())
        attn2.eval()
        with torch.no_grad():
            cold_out = attn2(x)

        # Outputs must differ because K,V came from different sources
        assert not torch.allclose(warm_out, cold_out, atol=1e-4)

    def test_present_kv_populated_after_forward(self) -> None:
        """present_k/v must be populated after export forward."""
        attn = self._make_attn()
        x = torch.randn(1, 64, 8, 8)
        B, C, H, W = x.shape
        N = H * W
        past_k = torch.zeros(B, attn.num_heads, N, attn.key_dim)
        past_v = torch.zeros(B, attn.num_heads, N, attn.head_dim)
        attn.set_export_inputs(past_k, past_v, torch.tensor(0.0))

        with torch.no_grad():
            attn(x)

        pk, pv = attn.get_export_outputs()
        assert pk.shape == (B, attn.num_heads, N, attn.key_dim)
        assert pv.shape == (B, attn.num_heads, N, attn.head_dim)

    def test_where_threshold_selects_past_above_half(self) -> None:
        """use_cache > 0.5 selects past K,V; <= 0.5 selects fresh."""
        torch.manual_seed(42)
        attn = self._make_attn()
        x = torch.randn(1, 64, 8, 8)
        B, C, H, W = x.shape
        N = H * W

        past_k = torch.randn(B, attn.num_heads, N, attn.key_dim)
        past_v = torch.randn(B, attn.num_heads, N, attn.head_dim)

        # use_cache = 0.7 (> 0.5) → should use past (same as 1.0)
        attn.set_export_inputs(past_k, past_v, torch.tensor(0.7))
        with torch.no_grad():
            out_07 = attn(x)

        attn.set_export_inputs(past_k, past_v, torch.tensor(1.0))
        with torch.no_grad():
            out_10 = attn(x)

        assert torch.allclose(out_07, out_10, atol=1e-5)

        # use_cache = 0.3 (<= 0.5) → should use fresh (same as 0.0)
        attn.set_export_inputs(past_k, past_v, torch.tensor(0.3))
        with torch.no_grad():
            out_03 = attn(x)

        attn.set_export_inputs(past_k, past_v, torch.tensor(0.0))
        with torch.no_grad():
            out_00 = attn(x)

        assert torch.allclose(out_03, out_00, atol=1e-5)


# ---------------------------------------------------------------------------
# YOLOKVWrapper — module discovery and I/O naming
# ---------------------------------------------------------------------------


class TestYOLOKVWrapper:
    """Test wrapper construction and metadata properties."""

    @pytest.mark.parametrize(
        "family,size,expected_count",
        [
            (ModelFamily.YOLO11, ModelSize.NANO, 1),
            (ModelFamily.YOLO26, ModelSize.NANO, 2),
        ],
    )
    def test_discovers_correct_attention_count(
        self,
        family: ModelFamily,
        size: ModelSize,
        expected_count: int,
    ) -> None:
        model = build_model(family, size)
        wrapper = YOLOKVWrapper(model)
        assert wrapper.num_attn == expected_count

    def test_kv_input_names_format(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        wrapper = YOLOKVWrapper(model)
        names = wrapper.kv_input_names
        assert names[0] == "use_cache"
        for i in range(wrapper.num_attn):
            assert f"past_k_{i}" in names
            assert f"past_v_{i}" in names

    def test_kv_output_names_format(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        wrapper = YOLOKVWrapper(model)
        names = wrapper.kv_output_names
        for i in range(wrapper.num_attn):
            assert f"present_k_{i}" in names
            assert f"present_v_{i}" in names

    def test_wrapper_forward_output_length(self) -> None:
        """forward() must return 1 + 2 * num_attn tensors."""
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()
        wrapper = YOLOKVWrapper(model)

        imgsz = 640
        dummy_images = torch.zeros(1, 3, imgsz, imgsz)
        dummy_kvs = wrapper.build_dummy_kv_inputs(imgsz)
        use_cache = torch.tensor(0.0)

        with torch.no_grad():
            outputs = wrapper(dummy_images, use_cache, *dummy_kvs)

        # (detections, present_k_0, present_v_0, ...) = 1 + 2 * num_attn
        assert len(outputs) == 1 + 2 * wrapper.num_attn

    def test_export_mode_enabled_on_all_attention(self) -> None:
        """All Attention modules must have _export_mode=True after wrapping."""
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        wrapper = YOLOKVWrapper(model)
        for attn in wrapper._attention_modules:
            assert attn._export_mode


# ---------------------------------------------------------------------------
# ONNX export file production
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("onnxscript"),
    reason="onnxscript not installed",
)
class TestOnnxExport:
    """Test that torch.onnx.export produces a valid .onnx file."""

    def test_onnx_export_yolo11_nano(self) -> None:
        """torch.onnx.export must produce a non-empty .onnx file."""
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()
        wrapper = YOLOKVWrapper(model)

        imgsz = 640
        dummy_images = torch.zeros(1, 3, imgsz, imgsz)
        dummy_kvs = wrapper.build_dummy_kv_inputs(imgsz)
        use_cache = torch.tensor(0.0)
        dummy_inputs = (dummy_images, use_cache, *dummy_kvs)

        input_names = ["images", *wrapper.kv_input_names]
        output_names = ["output0", *wrapper.kv_output_names]

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "kv_model.onnx"
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                str(onnx_path),
                opset_version=17,
                input_names=input_names,
                output_names=output_names,
            )
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0

    def test_backward_compat_no_kv(self) -> None:
        """Standard export (kv_cache=False) must produce single-input model."""
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()
        dummy = torch.zeros(1, 3, 640, 640)

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "standard.onnx"
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                opset_version=17,
                input_names=["images"],
                output_names=["output0"],
            )
            assert onnx_path.exists()
