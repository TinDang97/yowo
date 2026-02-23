"""Attention modules for YOLO architectures.

Implements PSA (Partial Self-Attention) used in YOLO11, wrapped in the
C2PSA Cross-Stage Partial structure.  Uses ``F.scaled_dot_product_attention``
for automatic FlashAttention / memory-efficient kernel selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from yowo.arch._blocks import Bottleneck, Conv

__all__ = ["C2PSA", "Attention", "C3k2PSA", "PSABlock"]


# ---------------------------------------------------------------------------
# Attention — Multi-Head Self-Attention (spatial)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention over spatial feature maps.

    Uses ``F.scaled_dot_product_attention`` for auto-selection of
    FlashAttention / memory-efficient kernels on GPU, with graceful
    fallback on CPU.
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

        self.qkv = Conv(dim, dim + 2 * nh_kd, 1)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # depthwise positional encoding

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x)
        qkv = qkv.view(B, self.num_heads, -1, N)

        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        # q: (B, heads, key_dim, N)  →  need (B, heads, N, key_dim) for SDPA
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Scaled dot-product attention (auto FlashAttention on GPU)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        # attn_out: (B, heads, N, head_dim)

        # Reshape back to spatial
        attn_out = attn_out.transpose(-2, -1).reshape(B, C, H, W)

        # Positional encoding on values (reshaped to spatial)
        v_spatial = v.transpose(-2, -1).reshape(B, C, H, W)
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

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat([a, b], dim=1))


# ---------------------------------------------------------------------------
# C3k2PSA — C2f with Sequential(Bottleneck, PSABlock) inner blocks
# ---------------------------------------------------------------------------


class C3k2PSA(nn.Module):
    """C2f-style block with Bottleneck + PSABlock inner modules.

    Used in YOLO26 neck layer 22: same outer structure as C3k2 (C2f) but
    inner modules are ``Sequential(Bottleneck, PSABlock)`` instead of C3k.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        shortcut: bool = False,
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

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
