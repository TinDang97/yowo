"""Unit tests for KV cache and block output cache in attention modules."""

from __future__ import annotations

import torch

from yowo.arch import build_model
from yowo.arch._attention import C2PSA, Attention, C3k2PSA, PSABlock
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
