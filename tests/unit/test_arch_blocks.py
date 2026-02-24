"""Unit tests for yowo.arch._blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from yowo.arch._blocks import (
    SPPF,
    Bottleneck,
    C2f,
    C3k,
    C3k2,
    Concat,
    Conv,
    DWConv,
    autopad,
    fuse_conv_and_bn,
    make_divisible,
)


class TestAutopad:
    def test_kernel_3(self) -> None:
        assert autopad(3) == 1

    def test_kernel_5(self) -> None:
        assert autopad(5) == 2

    def test_kernel_1(self) -> None:
        assert autopad(1) == 0

    def test_dilation_2(self) -> None:
        assert autopad(3, d=2) == 2

    def test_explicit_padding(self) -> None:
        assert autopad(3, p=0) == 0


class TestMakeDivisible:
    def test_rounds_up(self) -> None:
        assert make_divisible(10, 8) == 16

    def test_already_divisible(self) -> None:
        assert make_divisible(16, 8) == 16

    def test_small_value(self) -> None:
        assert make_divisible(1, 8) == 8


class TestConv:
    def test_output_shape(self) -> None:
        m = Conv(3, 16, 3, 1)
        x = torch.randn(1, 3, 32, 32)
        out = m(x)
        assert out.shape == (1, 16, 32, 32)

    def test_stride_2(self) -> None:
        m = Conv(16, 32, 3, 2)
        x = torch.randn(1, 16, 32, 32)
        out = m(x)
        assert out.shape == (1, 32, 16, 16)

    def test_forward_fuse(self) -> None:
        m = Conv(3, 16, 3, 1)
        # Simulate fused mode: need conv with bias
        fused_conv = fuse_conv_and_bn(m.conv, m.bn)
        m.conv = fused_conv
        delattr(m, "bn")
        x = torch.randn(1, 3, 16, 16)
        out = m.forward_fuse(x)
        assert out.shape == (1, 16, 16, 16)


class TestDWConv:
    def test_output_shape(self) -> None:
        m = DWConv(16, 16, 3, 1)
        x = torch.randn(1, 16, 32, 32)
        out = m(x)
        assert out.shape == (1, 16, 32, 32)

    def test_groups(self) -> None:
        m = DWConv(16, 16, 3, 1)
        assert m.conv.groups == 16


class TestBottleneck:
    def test_output_shape_with_shortcut(self) -> None:
        m = Bottleneck(32, 32, shortcut=True)
        x = torch.randn(1, 32, 16, 16)
        out = m(x)
        assert out.shape == (1, 32, 16, 16)

    def test_output_shape_no_shortcut(self) -> None:
        m = Bottleneck(32, 64, shortcut=False)
        x = torch.randn(1, 32, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)


class TestC2f:
    def test_output_shape(self) -> None:
        m = C2f(64, 64, n=2, shortcut=True)
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)


class TestC3k:
    def test_output_shape(self) -> None:
        m = C3k(64, 64, n=1, shortcut=True)
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)


class TestC3k2:
    def test_output_shape_no_c3k(self) -> None:
        m = C3k2(64, 64, n=1, c3k=False)
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)

    def test_output_shape_with_c3k(self) -> None:
        m = C3k2(64, 64, n=1, c3k=True)
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)


class TestSPPF:
    def test_output_shape_no_shortcut(self) -> None:
        m = SPPF(64, 64, k=5, shortcut=False)
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)

    def test_output_shape_with_shortcut(self) -> None:
        m = SPPF(64, 64, k=5, shortcut=True)
        x = torch.randn(1, 64, 16, 16)
        out = m(x)
        assert out.shape == (1, 64, 16, 16)


class TestConcat:
    def test_concat_channels(self) -> None:
        m = Concat(dim=1)
        a = torch.randn(1, 16, 8, 8)
        b = torch.randn(1, 32, 8, 8)
        out = m([a, b])
        assert out.shape == (1, 48, 8, 8)


class TestConvForwardFuseNoAct:
    def test_identity_act_skipped(self) -> None:
        """forward_fuse_no_act produces identical output to forward_fuse with Identity act."""
        m = Conv(16, 32, 3, 1, act=False)
        m.eval()
        fused_conv = fuse_conv_and_bn(m.conv, m.bn)
        m.conv = fused_conv
        delattr(m, "bn")

        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            expected = m.forward_fuse(x)
            actual = m.forward_fuse_no_act(x)

        torch.testing.assert_close(actual, expected)

    def test_silu_act_not_affected(self) -> None:
        """forward_fuse with SiLU differs from forward_fuse_no_act (sanity)."""
        m = Conv(16, 32, 3, 1, act=True)
        m.eval()
        fused_conv = fuse_conv_and_bn(m.conv, m.bn)
        m.conv = fused_conv
        delattr(m, "bn")

        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            fuse_result = m.forward_fuse(x)
            no_act_result = m.forward_fuse_no_act(x)

        assert not torch.allclose(fuse_result, no_act_result), (
            "SiLU forward_fuse should differ from forward_fuse_no_act"
        )


class TestBottleneckSpecialization:
    def test_forward_shortcut_matches(self) -> None:
        m = Bottleneck(32, 32, shortcut=True)
        m.eval()
        x = torch.randn(1, 32, 16, 16)
        with torch.no_grad():
            expected = m.forward(x)
            actual = m._forward_shortcut(x)
        torch.testing.assert_close(actual, expected)

    def test_forward_no_shortcut_matches(self) -> None:
        m = Bottleneck(32, 64, shortcut=False)
        m.eval()
        x = torch.randn(1, 32, 16, 16)
        with torch.no_grad():
            expected = m.forward(x)
            actual = m._forward_no_shortcut(x)
        torch.testing.assert_close(actual, expected)


class TestFuseConvAndBn:
    def test_numerically_equivalent(self) -> None:
        conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        bn = nn.BatchNorm2d(16)
        bn.eval()

        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            expected = bn(conv(x))
            fused = fuse_conv_and_bn(conv, bn)
            actual = fused(x)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
