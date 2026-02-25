"""Unit tests for KV cache and block output cache in attention modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from yowo.arch import build_model
from yowo.arch._attention import C2PSA, Attention, C3k2PSA, PSABlock, _build_q_conv
from yowo.types import ModelFamily, ModelSize


class TestAttentionOutput:
    """Verify attention modules produce correct output shapes."""

    def test_output_shape(self) -> None:
        attn = Attention(dim=64, num_heads=4, attn_ratio=0.5)
        attn.eval()
        x = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (1, 64, 8, 8)

    def test_output_shape_different_heads(self) -> None:
        attn = Attention(dim=128, num_heads=8, attn_ratio=0.5)
        attn.eval()
        x = torch.randn(2, 128, 10, 10)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (2, 128, 10, 10)

    def test_psablock_output_shape(self) -> None:
        block = PSABlock(c=64)
        block.eval()
        x = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (1, 64, 8, 8)


class TestAttentionKVCache:
    """Test KV cache enable/disable/clear and cache hit behavior."""

    def test_kv_cache_disabled_by_default(self) -> None:
        attn = Attention(dim=64, num_heads=4)
        assert not attn._kv_cache_enabled
        assert attn._cached_k is None

    def test_enable_disable_kv_cache(self) -> None:
        attn = Attention(dim=64, num_heads=4)
        attn.enable_kv_cache(True)
        assert attn._kv_cache_enabled
        attn.enable_kv_cache(False)
        assert not attn._kv_cache_enabled

    def test_cache_populated_after_forward(self) -> None:
        attn = Attention(dim=64, num_heads=4)
        attn.eval()
        attn.enable_kv_cache(True)

        x = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            attn(x)

        assert attn._cached_k is not None
        assert attn._cached_v is not None
        assert attn._cached_v_spatial is not None
        assert attn._cache_shape_key == (1, 8, 8)

    def test_cache_hit_produces_valid_output(self) -> None:
        attn = Attention(dim=64, num_heads=4)
        attn.eval()
        attn.enable_kv_cache(True)

        x1 = torch.randn(1, 64, 8, 8)
        x2 = torch.randn(1, 64, 8, 8)

        with torch.no_grad():
            out1 = attn(x1)  # cold — fills cache
            out2 = attn(x2)  # warm — uses cached K,V

        assert out1.shape == out2.shape == (1, 64, 8, 8)
        # Outputs should differ since Q comes from different inputs
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_cache_clear(self) -> None:
        attn = Attention(dim=64, num_heads=4)
        attn.eval()
        attn.enable_kv_cache(True)

        x = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            attn(x)

        assert attn._cached_k is not None
        attn.clear_kv_cache()
        assert attn._cached_k is None
        assert attn._cached_v is None
        assert attn._cache_shape_key is None

    def test_cache_invalidate_on_shape_change(self) -> None:
        attn = Attention(dim=64, num_heads=4)
        attn.eval()
        attn.enable_kv_cache(True)

        x1 = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            attn(x1)

        assert attn._cache_shape_key == (1, 8, 8)

        # Different spatial dims — should recompute KV
        x2 = torch.randn(1, 64, 10, 10)
        with torch.no_grad():
            out = attn(x2)

        assert out.shape == (1, 64, 10, 10)
        assert attn._cache_shape_key == (1, 10, 10)

    def test_cached_tensors_are_detached(self) -> None:
        attn = Attention(dim=64, num_heads=4)
        attn.eval()
        attn.enable_kv_cache(True)

        x = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            attn(x)

        assert attn._cached_k is not None
        assert not attn._cached_k.requires_grad


class TestC2PSABlockCache:
    """Test block-level output caching in C2PSA."""

    def test_block_cache_disabled_by_default(self) -> None:
        block = C2PSA(128, 128)
        assert not block._block_cache_enabled

    def test_block_cache_hit_on_similar_input(self) -> None:
        block = C2PSA(128, 128)
        block.eval()
        block.enable_block_cache(True)

        x = torch.randn(1, 128, 8, 8)
        with torch.no_grad():
            out1 = block(x)
            # Same input = identical fingerprint → cache hit
            out2 = block(x)

        assert torch.equal(out1, out2)

    def test_block_cache_miss_on_different_input(self) -> None:
        block = C2PSA(128, 128)
        block.eval()
        block.enable_block_cache(True)

        x1 = torch.randn(1, 128, 8, 8)
        x2 = torch.randn(1, 128, 8, 8) * 100  # very different → cache miss

        with torch.no_grad():
            out1 = block(x1)
            out2 = block(x2)

        assert not torch.equal(out1, out2)

    def test_block_cache_clear(self) -> None:
        block = C2PSA(128, 128)
        block.eval()
        block.enable_block_cache(True)

        x = torch.randn(1, 128, 8, 8)
        with torch.no_grad():
            block(x)

        assert block._cached_output is not None
        block.clear_block_cache()
        assert block._cached_output is None
        assert block._cached_input_fp is None


class TestC3k2PSABlockCache:
    """Test block-level output caching in C3k2PSA."""

    def test_block_cache_hit(self) -> None:
        block = C3k2PSA(128, 128)
        block.eval()
        block.enable_block_cache(True)

        x = torch.randn(1, 128, 8, 8)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)

        assert torch.equal(out1, out2)

    def test_block_cache_miss(self) -> None:
        block = C3k2PSA(128, 128)
        block.eval()
        block.enable_block_cache(True)

        x1 = torch.randn(1, 128, 8, 8)
        x2 = torch.randn(1, 128, 8, 8) * 100

        with torch.no_grad():
            out1 = block(x1)
            out2 = block(x2)

        assert not torch.equal(out1, out2)


class TestYOLOModelKVCache:
    """Test KV cache integration at the model level."""

    def test_enable_kv_cache_activates_all_modules(self) -> None:
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        model.eval()
        model.enable_kv_cache()

        # All Attention modules should have cache enabled
        attention_count = 0
        for m in model.modules():
            if isinstance(m, Attention):
                assert m._kv_cache_enabled
                attention_count += 1

        assert attention_count > 0

    def test_enable_kv_cache_activates_block_caches(self) -> None:
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        model.eval()
        model.enable_kv_cache()

        c2psa_count = 0
        c3k2psa_count = 0
        for m in model.modules():
            if isinstance(m, C2PSA):
                assert m._block_cache_enabled
                c2psa_count += 1
            elif isinstance(m, C3k2PSA):
                assert m._block_cache_enabled
                c3k2psa_count += 1

        # YOLO26 has C2PSA in backbone and C3k2PSA in neck
        assert c2psa_count >= 1
        assert c3k2psa_count >= 1

    def test_clear_kv_cache(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()
        model.enable_kv_cache()

        # Run a forward pass to populate caches
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            model(x)

        # Verify some caches are populated
        has_cached = False
        for m in model.modules():
            if isinstance(m, Attention) and m._cached_k is not None:
                has_cached = True
                break
        assert has_cached

        # Clear and verify
        model.clear_kv_cache()
        for m in model.modules():
            if isinstance(m, Attention):
                assert m._cached_k is None

    def test_forward_with_kv_cache_produces_valid_shape(self) -> None:
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        model.eval()
        model.enable_kv_cache()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        # YOLO26: (B, max_det, 6)
        assert out.shape[0] == 1
        assert out.shape[2] == 6

    def test_yolo11_enable_kv_cache_no_c3k2psa(self) -> None:
        """YOLO11 has no C3k2PSA — enable_kv_cache should still work."""
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()
        model.enable_kv_cache()

        c3k2psa_count = sum(1 for m in model.modules() if isinstance(m, C3k2PSA))
        assert c3k2psa_count == 0

        # But C2PSA and Attention should be present and enabled
        attn_count = sum(
            1 for m in model.modules() if isinstance(m, Attention) and m._kv_cache_enabled
        )
        assert attn_count > 0


class TestModelForwardWithCache:
    """Test that models with KV/block cache produce valid output."""

    def test_yolo26_builds_and_runs(self) -> None:
        """Verify YOLO26 model runs forward correctly."""
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        model.eval()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1
        assert out.dim() == 3

    def test_yolo11_builds_and_runs(self) -> None:
        model = build_model(ModelFamily.YOLO11, ModelSize.NANO)
        model.eval()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1
        assert out.shape[2] == 80 * 80 + 40 * 40 + 20 * 20  # 8400


class TestQConvSplit:
    """Test QKV weight split into Q-only conv for cache-hit optimization."""

    def _make_fused_attention(self, dim: int = 64, num_heads: int = 4) -> Attention:
        """Create an Attention module and simulate fuse (remove bn)."""
        attn = Attention(dim=dim, num_heads=num_heads, attn_ratio=0.5)
        # Simulate fuse(): fold BN into conv and swap forward method
        from yowo.arch._blocks import Conv, fuse_conv_and_bn

        for m in attn.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                if isinstance(m.act, nn.Identity):
                    m.forward = m.forward_fuse_no_act  # type: ignore[assignment]
                else:
                    m.forward = m.forward_fuse  # type: ignore[assignment]
        attn.eval()
        return attn

    def test_q_conv_output_matches_qkv_slice(self) -> None:
        """q_conv(x) must be numerically identical to Q slice from full QKV."""
        attn = self._make_fused_attention(dim=128, num_heads=4)
        attn.enable_kv_cache(True)
        assert attn.q_conv is not None

        x = torch.randn(2, 128, 8, 8)
        with torch.no_grad():
            # Full QKV path
            full_qkv = attn.qkv.conv(x)
            B, _, H, W = full_qkv.shape
            N = H * W
            qkv_r = full_qkv.view(B, attn.num_heads, -1, N)
            q_expected = qkv_r[:, :, : attn.key_dim, :].reshape(
                B, attn.num_heads * attn.key_dim, H, W
            )

            # Q-only path
            q_direct = attn.q_conv(x)

        assert torch.allclose(q_direct, q_expected, atol=1e-6)

    def test_q_conv_is_none_before_fuse(self) -> None:
        """q_conv must remain None when fuse() hasn't run (bn still present)."""
        attn = Attention(dim=64, num_heads=4)
        attn.enable_kv_cache(True)
        assert attn.q_conv is None

    def test_q_conv_persists_on_disable(self) -> None:
        """q_conv is a weight module, not cache state — survives disable."""
        attn = self._make_fused_attention()
        attn.enable_kv_cache(True)
        assert attn.q_conv is not None
        attn.enable_kv_cache(False)
        assert attn.q_conv is not None

    def test_q_conv_not_rebuilt_on_re_enable(self) -> None:
        """Repeated enable_kv_cache(True) must not rebuild q_conv."""
        attn = self._make_fused_attention()
        attn.enable_kv_cache(True)
        q_conv_id = id(attn.q_conv)
        attn.enable_kv_cache(True)
        assert id(attn.q_conv) == q_conv_id

    def test_q_conv_device_matches(self) -> None:
        """q_conv must live on the same device as the QKV conv."""
        attn = self._make_fused_attention()
        attn.enable_kv_cache(True)
        assert attn.q_conv is not None
        assert attn.q_conv.weight.device == attn.qkv.conv.weight.device

    def test_q_conv_shape(self) -> None:
        """q_conv weight must have correct output channels = num_heads * key_dim."""
        attn = self._make_fused_attention(dim=256, num_heads=4)
        attn.enable_kv_cache(True)
        assert attn.q_conv is not None
        expected_out = attn.num_heads * attn.key_dim
        assert attn.q_conv.weight.shape[0] == expected_out
        assert attn.q_conv.weight.shape[1] == 256  # in_channels

    def test_build_q_conv_index_correctness(self) -> None:
        """Verify per-head interleaved Q index extraction is correct."""
        num_heads, key_dim, head_dim = 4, 32, 64
        stride = 2 * key_dim + head_dim  # 128
        # Create a QKV conv with known weight values
        qkv_conv = nn.Conv2d(256, num_heads * stride, 1, bias=True)
        with torch.no_grad():
            # Set each output channel's weight to its index for verification
            for i in range(num_heads * stride):
                qkv_conv.weight[i].fill_(float(i))
                qkv_conv.bias[i] = float(i)  # type: ignore[index]

        q_conv = _build_q_conv(qkv_conv, num_heads, key_dim, head_dim)

        # Expected Q indices: head 0 [0..31], head 1 [128..159],
        # head 2 [256..287], head 3 [384..415]
        expected_indices = []
        for h in range(num_heads):
            for q in range(key_dim):
                expected_indices.append(h * stride + q)

        for i, expected_idx in enumerate(expected_indices):
            assert q_conv.weight[i, 0, 0, 0].item() == float(expected_idx)
            assert q_conv.bias[i].item() == float(expected_idx)  # type: ignore[index]

    def test_forward_cache_hit_uses_q_conv(self) -> None:
        """Cache-hit path must produce valid output using q_conv."""
        attn = self._make_fused_attention(dim=64, num_heads=4)
        attn.enable_kv_cache(True)
        assert attn.q_conv is not None

        x1 = torch.randn(1, 64, 8, 8)
        x2 = torch.randn(1, 64, 8, 8)

        with torch.no_grad():
            out1 = attn(x1)  # cold — populates cache
            out2 = attn(x2)  # warm — uses q_conv on cache hit

        assert out1.shape == out2.shape == (1, 64, 8, 8)
        assert torch.isfinite(out2).all()

    def test_fused_model_kv_cache_q_split(self) -> None:
        """Full model with fuse() + enable_kv_cache() must build q_conv."""
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        model = model.fuse().eval()
        model.enable_kv_cache()

        q_conv_count = 0
        for m in model.modules():
            if isinstance(m, Attention):
                assert m.q_conv is not None, "q_conv must be built after fuse + enable_kv_cache"
                q_conv_count += 1

        assert q_conv_count >= 2  # YOLO26n has 2 Attention modules

    def test_fused_model_output_with_q_split(self) -> None:
        """Model output must be valid across cold + warm frames with q_conv."""
        model = build_model(ModelFamily.YOLO26, ModelSize.NANO)
        model = model.fuse().eval()
        model.enable_kv_cache()

        x1 = torch.randn(1, 3, 640, 640)
        x2 = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            out1 = model(x1)  # cold
            out2 = model(x2)  # warm — q_conv active on cache hit

        assert out1.shape == out2.shape
        assert out2.shape[0] == 1
        assert out2.shape[2] == 6  # YOLO26: [x1,y1,x2,y2,conf,cls]
        assert torch.isfinite(out2).all()
