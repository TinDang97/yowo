"""Attention modules for YOLO architectures.

Implements PSA (Partial Self-Attention) used in YOLO11, wrapped in the
C2PSA Cross-Stage Partial structure.  Uses ``F.scaled_dot_product_attention``
for automatic FlashAttention / memory-efficient kernel selection.

KV cache support: cache K,V tensors across frames for streaming inference.
On cache-warm frames, Q is computed fresh from the current input while K,V
are reused from the previous frame (cross-attention approximation).
Block-level output caching on C2PSA and C3k2PSA skips entire blocks when
the input is similar across consecutive frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yowo.arch._blocks import Bottleneck, Conv

__all__ = ["C2PSA", "Attention", "C3k2PSA", "PSABlock"]

# Block cache similarity threshold — mean absolute difference of
# spatially-pooled feature fingerprints. 0.01 matches FeatureCache default.
_BLOCK_CACHE_THRESHOLD: float = 0.01


# ---------------------------------------------------------------------------
# Attention — Multi-Head Self-Attention (spatial) with KV cache
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention over spatial feature maps.

    Uses ``F.scaled_dot_product_attention`` for auto-selection of
    FlashAttention / memory-efficient kernels on GPU, with graceful
    fallback on CPU.

    KV cache: when enabled, K and V tensors from the previous forward
    pass are reused. Q is always computed fresh from the current input.
    This converts self-attention into cross-attention between the current
    frame (queries) and the previous frame (keys/values), which is
    effective for video streams with high temporal coherence.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        nh_kd = num_heads * self.key_dim

        self.qkv = Conv(dim, dim + 2 * nh_kd, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # depthwise positional encoding

        # KV cache state (disabled by default)
        self._kv_cache_enabled: bool = False
        self._cached_k: Tensor | None = None
        self._cached_v: Tensor | None = None
        self._cached_v_spatial: Tensor | None = None
        self._cache_shape_key: tuple[int, int, int] | None = None  # (B, H, W)
        self._cached_input_fp: Tensor | None = None  # staleness guard fingerprint

        # ONNX export KV I/O state (set by YOLOKVWrapper before tracing)
        self._export_mode: bool = False
        self._export_past_k: Tensor | None = None
        self._export_past_v: Tensor | None = None
        self._export_use_cache: Tensor | None = None
        self._export_present_k: Tensor | None = None
        self._export_present_v: Tensor | None = None

    # ------------------------------------------------------------------
    # ONNX export helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sdpa_onnx(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Manual scaled dot-product attention for ONNX export.

        ``F.scaled_dot_product_attention`` does not export to ONNX opset 17.
        This decomposition uses only primitive ops (matmul, mul, softmax)
        that are universally supported.
        """
        scale = q.shape[-1] ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        return attn_weights.softmax(dim=-1) @ v

    def _forward_kv_export(self, x: Tensor) -> Tensor:
        """KV-externalized forward for ONNX export.

        Reads past K,V from ``_export_past_k/v`` and ``_export_use_cache``
        (set by ``YOLOKVWrapper`` before tracing). Selects past or fresh
        K,V via ``torch.where`` (single ONNX ``Where`` node per tensor).
        Stores present K,V in ``_export_present_k/v`` for the wrapper to
        collect.

        ``use_cache`` must be binary: ``0.0`` (cold — use fresh K,V) or
        ``1.0`` (warm — use cached K,V). Threshold is ``0.5``.
        """
        assert self._export_past_k is not None
        assert self._export_past_v is not None
        assert self._export_use_cache is not None

        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x)
        qkv = qkv.view(B, self.num_heads, -1, N)

        q, k_new, v_new = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        q = q.transpose(-2, -1)
        k_new = k_new.transpose(-2, -1)
        v_new = v_new.transpose(-2, -1)

        # Select: use_cache > 0.5 → past, otherwise → fresh (1 Where node each)
        cond = self._export_use_cache > 0.5
        k = torch.where(cond, self._export_past_k, k_new)
        v = torch.where(cond, self._export_past_v, v_new)

        # Always output freshly computed K,V for next frame
        self._export_present_k = k_new
        self._export_present_v = v_new

        # Derive v_spatial from blended V (no separate I/O needed)
        v_spatial = v.transpose(-2, -1).contiguous().view(B, C, H, W)

        attn_out = self._sdpa_onnx(q, k, v)
        attn_out = attn_out.transpose(-2, -1).contiguous().view(B, C, H, W)
        return self.proj(attn_out + self.pe(v_spatial))

    # ------------------------------------------------------------------
    # ONNX export interface (called by YOLOKVWrapper)
    # ------------------------------------------------------------------

    def enable_export_mode(self) -> None:
        """Switch this module to KV-externalized export forward path."""
        self._export_mode = True

    def set_export_inputs(self, past_k: Tensor, past_v: Tensor, use_cache: Tensor) -> None:
        """Inject past K,V and use_cache scalar before model forward."""
        self._export_past_k = past_k
        self._export_past_v = past_v
        self._export_use_cache = use_cache

    def get_export_outputs(self) -> tuple[Tensor, Tensor]:
        """Retrieve present K,V after model forward."""
        assert self._export_present_k is not None
        assert self._export_present_v is not None
        return self._export_present_k, self._export_present_v

    # ------------------------------------------------------------------
    # Runtime KV cache control
    # ------------------------------------------------------------------

    def enable_kv_cache(self, enabled: bool = True) -> None:
        """Enable or disable KV caching for streaming inference."""
        self._kv_cache_enabled = enabled
        if not enabled:
            self.clear_kv_cache()

    def clear_kv_cache(self) -> None:
        """Clear cached K, V tensors."""
        self._cached_k = None
        self._cached_v = None
        self._cached_v_spatial = None
        self._cache_shape_key = None
        self._cached_input_fp = None

    def forward(self, x: Tensor) -> Tensor:
        if self._export_mode:
            return self._forward_kv_export(x)

        B, C, H, W = x.shape
        N = H * W

        # Check KV cache validity
        shape_key = (B, H, W)
        cache_hit = (
            self._kv_cache_enabled
            and self._cached_k is not None
            and self._cache_shape_key == shape_key
        )

        # Compute spatial fingerprint once (reused for staleness guard and cache update)
        fp = x.mean(dim=(2, 3)) if self._kv_cache_enabled else None

        # Staleness guard: invalidate KV cache on scene change
        if cache_hit and fp is not None and self._cached_input_fp is not None:
            diff = (fp - self._cached_input_fp).abs().mean().item()
            if diff >= _BLOCK_CACHE_THRESHOLD:
                cache_hit = False

        if cache_hit:
            # Cache hit: compute Q from current input, reuse K,V from cache
            assert self._cached_k is not None
            assert self._cached_v is not None
            assert self._cached_v_spatial is not None

            qkv = self.qkv(x)
            qkv = qkv.view(B, self.num_heads, -1, N)
            # Extract only Q (first key_dim channels per head)
            q = qkv[:, :, : self.key_dim, :].transpose(-2, -1)
            k = self._cached_k
            v = self._cached_v
            v_spatial = self._cached_v_spatial
        else:
            # Full computation path
            qkv = self.qkv(x)
            qkv = qkv.view(B, self.num_heads, -1, N)

            q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

            # (B, heads, dim, N) → (B, heads, N, dim) for SDPA
            q = q.transpose(-2, -1)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            v_spatial = v.transpose(-2, -1).contiguous().view(B, C, H, W)

            # Store K,V for next frame — reuse pre-computed fingerprint
            if self._kv_cache_enabled:
                self._cached_k = k.detach()
                self._cached_v = v.detach()
                self._cached_v_spatial = v_spatial.detach()
                self._cache_shape_key = shape_key
                self._cached_input_fp = (fp if fp is not None else x.mean(dim=(2, 3))).detach()

        # Scaled dot-product attention
        # MPS bug: F.scaled_dot_product_attention returns wrong shape when key_dim != head_dim
        attn_out = self._sdpa_onnx(q, k, v) if q.is_mps else F.scaled_dot_product_attention(q, k, v)

        # Reshape back to spatial — explicit contiguous() for torch.compile visibility
        attn_out = attn_out.transpose(-2, -1).contiguous().view(B, C, H, W)

        # Positional encoding on values (reshaped to spatial)
        return self.proj(attn_out + self.pe(v_spatial))


# ---------------------------------------------------------------------------
# PSABlock — Partial Self-Attention Block with FFN
# ---------------------------------------------------------------------------


class PSABlock(nn.Module):
    """Attention + Feed-Forward Network with residual connections."""

    def __init__(
        self,
        c: int,
        attn_ratio: float = 0.5,
        num_heads: int | None = None,
    ) -> None:
        super().__init__()
        _num_heads = num_heads if num_heads is not None else max(c // 64, 1)
        self.attn = Attention(c, num_heads=_num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# C2PSA — Cross-Stage Partial with Position-Sensitive Attention
# ---------------------------------------------------------------------------


class C2PSA(nn.Module):
    """CSP wrapper for PSABlock: splits channels, applies attention to one
    branch, concatenates and projects back.

    Supports block-level output caching for streaming inference: when the
    input is similar to the previous frame (measured by spatial-mean
    fingerprint), the cached output is returned directly.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *[PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)]
        )

        # Block output cache state
        self._block_cache_enabled: bool = False
        self._cached_output: Tensor | None = None
        self._cached_input_fp: Tensor | None = None

    def enable_block_cache(self, enabled: bool = True) -> None:
        """Enable or disable block-level output caching.

        Raises:
            RuntimeError: If called with ``enabled=True`` while in training mode.
        """
        if enabled and self.training:
            msg = "Block cache must not be enabled during training (breaks gradient flow)"
            raise RuntimeError(msg)
        self._block_cache_enabled = enabled
        if not enabled:
            self.clear_block_cache()

    def clear_block_cache(self) -> None:
        """Clear cached block output and fingerprint."""
        self._cached_output = None
        self._cached_input_fp = None

    def forward(self, x: Tensor) -> Tensor:
        # Block cache: check fingerprint similarity (skip during training)
        fp: Tensor | None = None
        if self._block_cache_enabled and not self.training:
            fp = x.mean(dim=(2, 3))  # (B, C) spatial-mean fingerprint
            cached_fp = self._cached_input_fp
            if self._cached_output is not None and cached_fp is not None:
                diff = (fp - cached_fp).abs().mean().item()
                if diff < _BLOCK_CACHE_THRESHOLD:
                    return self._cached_output

        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        out = self.cv2(torch.cat([a, b], dim=1))

        if self._block_cache_enabled and not self.training:
            self._cached_output = out.detach()
            self._cached_input_fp = (fp if fp is not None else x.mean(dim=(2, 3))).detach()

        return out


# ---------------------------------------------------------------------------
# C3k2PSA — C2f with Sequential(Bottleneck, PSABlock) inner blocks
# ---------------------------------------------------------------------------


class C3k2PSA(nn.Module):
    """C2f-style block with Bottleneck + PSABlock inner modules.

    Used in YOLO26 neck layer 22: same outer structure as C3k2 (C2f) but
    inner modules are ``Sequential(Bottleneck, PSABlock)`` instead of C3k.

    Supports block-level output caching for streaming inference.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut),
                PSABlock(self.c),
            )
            for _ in range(n)
        )

        # Block output cache state
        self._block_cache_enabled: bool = False
        self._cached_output: Tensor | None = None
        self._cached_input_fp: Tensor | None = None

    def enable_block_cache(self, enabled: bool = True) -> None:
        """Enable or disable block-level output caching.

        Raises:
            RuntimeError: If called with ``enabled=True`` while in training mode.
        """
        if enabled and self.training:
            msg = "Block cache must not be enabled during training (breaks gradient flow)"
            raise RuntimeError(msg)
        self._block_cache_enabled = enabled
        if not enabled:
            self.clear_block_cache()

    def clear_block_cache(self) -> None:
        """Clear cached block output and fingerprint."""
        self._cached_output = None
        self._cached_input_fp = None

    def forward(self, x: Tensor) -> Tensor:
        # Block cache: check fingerprint similarity (skip during training)
        fp: Tensor | None = None
        if self._block_cache_enabled and not self.training:
            fp = x.mean(dim=(2, 3))  # (B, C) spatial-mean fingerprint
            cached_fp = self._cached_input_fp
            if self._cached_output is not None and cached_fp is not None:
                diff = (fp - cached_fp).abs().mean().item()
                if diff < _BLOCK_CACHE_THRESHOLD:
                    return self._cached_output

        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        if self._block_cache_enabled and not self.training:
            self._cached_output = out.detach()
            self._cached_input_fp = (fp if fp is not None else x.mean(dim=(2, 3))).detach()

        return out
