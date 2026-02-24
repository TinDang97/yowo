"""Common YOLO building blocks for inference.

Clean-room implementations based on published YOLO architecture papers.
Every block supports Conv+BN fusion via the ``fuse()`` protocol.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Compute 'same' padding for a given kernel size and dilation."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round *x* up to the nearest value divisible by *divisor*."""
    return math.ceil(x / divisor) * divisor


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold BatchNorm parameters into Conv2d weights.

    Returns a new Conv2d with bias that is mathematically equivalent to
    the sequential ``bn(conv(x))`` call.
    """
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        dilation=conv.dilation,  # type: ignore[arg-type]
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False)

    # Wrap in no_grad: scale/bias are computed from bn.weight (a Parameter),
    # so without no_grad the .copy_() creates a CopyBackwards grad_fn making
    # fused parameters non-leaf, which triggers warnings at load_state_dict.
    with torch.no_grad():
        # Per-channel scale: gamma / sqrt(var + eps)  — O(N) instead of O(N^2) diag
        scale = bn.weight.div(torch.sqrt(bn.running_var + bn.eps))  # type: ignore[arg-type]

        # Fuse weights:  w_fused = w_conv * scale (element-wise broadcast)
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        fused.weight.copy_((w_conv * scale.unsqueeze(1)).view(fused.weight.shape))

        # Fuse bias:  b_fused = scale * b_conv + (beta - gamma * mean / sqrt(var + eps))
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(  # type: ignore[union-attr]
            torch.sqrt(bn.running_var + bn.eps)  # type: ignore[arg-type]
        )
        fused.bias.copy_(scale * b_conv + b_bn)  # type: ignore[union-attr]
    return fused


# ---------------------------------------------------------------------------
# Conv — Conv2d + BatchNorm2d + SiLU (with fuse support)
# ---------------------------------------------------------------------------


class Conv(nn.Module):
    """Standard convolution: Conv2d → BatchNorm2d → SiLU.

    After ``fuse()``, the BN is folded into the conv and ``forward_fuse``
    is used instead.
    """

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act: nn.Module = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: Tensor) -> Tensor:
        """Forward pass after Conv+BN fusion (no BN layer)."""
        return self.act(self.conv(x))


# ---------------------------------------------------------------------------
# DWConv — Depthwise Convolution
# ---------------------------------------------------------------------------


class DWConv(Conv):
    """Depthwise convolution: groups = gcd(c1, c2)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# ---------------------------------------------------------------------------
# Bottleneck — Standard residual block
# ---------------------------------------------------------------------------


class Bottleneck(nn.Module):
    """Two-conv residual block with optional shortcut."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ---------------------------------------------------------------------------
# C2f — Faster CSP Bottleneck with 2 convolutions
# ---------------------------------------------------------------------------


class C2f(nn.Module):
    """Cross-Stage Partial bottleneck with 2 convolutions.

    Splits input into two branches, passes one through *n* bottlenecks,
    then concatenates all branches and projects.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# C3 — CSP Bottleneck with 3 convolutions
# ---------------------------------------------------------------------------


class C3(nn.Module):
    """Cross-Stage Partial bottleneck with 3 convolutions.

    Two branches: ``cv1`` passes through sequential bottleneck blocks,
    ``cv2`` is a skip connection. Both are concatenated and projected
    through ``cv3``.

    This is the base class for ``C3k``.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ---------------------------------------------------------------------------
# C3k — CSP Bottleneck with customisable kernel size
# ---------------------------------------------------------------------------


class C3k(C3):
    """C3 variant where each bottleneck uses k x k kernels.

    Used inside ``C3k2`` when ``c3k=True``.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


# ---------------------------------------------------------------------------
# C3k2 — Core block for YOLO11 / YOLO26
# ---------------------------------------------------------------------------


class C3k2(C2f):
    """C2f with kernel-flexibility: uses ``C3k`` when *c3k=True*.

    This is the primary building block in YOLO11 and YOLO26 architectures.
    For nano/small variants *c3k=False* → standard Bottleneck.
    For medium/large/xlarge *c3k=True* → nested C3k with 3x3 kernels.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


# ---------------------------------------------------------------------------
# SPPF — Spatial Pyramid Pooling Fast
# ---------------------------------------------------------------------------


class SPPF(nn.Module):
    """Fast Spatial Pyramid Pooling with sequential max-pooling.

    When *shortcut=True* (YOLO26), adds a residual connection from input.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        n: int = 3,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        c_ = c1 // 2
        # When shortcut=True (YOLO26), cv1 has no activation (Identity);
        # when shortcut=False (YOLO11), cv1 uses SiLU (default act=True).
        self.cv1 = Conv(c1, c_, 1, 1, act=not shortcut)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(self.m(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        return out + x if self.shortcut else out


# ---------------------------------------------------------------------------
# Concat — Channel-wise concatenation utility
# ---------------------------------------------------------------------------


class Concat(nn.Module):
    """Concatenate tensors along *dim*."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.d = dim

    def forward(self, x: list[Tensor]) -> Tensor:
        return torch.cat(x, self.d)


__all__ = [
    "C3",
    "SPPF",
    "Bottleneck",
    "C2f",
    "C3k",
    "C3k2",
    "Concat",
    "Conv",
    "DWConv",
    "autopad",
    "fuse_conv_and_bn",
    "make_divisible",
]
