"""Unit tests for the Classify head (yowo.arch._heads.Classify)."""

from __future__ import annotations

import pytest
import torch

from yowo.arch._heads import Classify


class TestClassifyHead:
    def test_forward_output_shape(self) -> None:
        """Classify(512, 1000) with (2, 512, 7, 7) input → (2, 1000)."""
        model = Classify(512, 1000).eval()
        x = torch.randn(2, 512, 7, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1000)

    @pytest.mark.parametrize("c1", [256, 512, 1024])
    def test_forward_various_channel_counts(self, c1: int) -> None:
        """Various input channel counts produce (1, 1000) output."""
        model = Classify(c1, 1000).eval()
        x = torch.randn(1, c1, 7, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1000)

    def test_dropout_zero_is_identity(self) -> None:
        """dropout=0.0 in eval mode produces deterministic identical output."""
        model = Classify(512, 1000, dropout=0.0).eval()
        x = torch.randn(1, 512, 7, 7)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_dropout_nonzero_in_train(self) -> None:
        """dropout=0.5 in train mode produces different outputs across passes."""
        torch.manual_seed(0)
        model = Classify(512, 1000, dropout=0.5).train()
        x = torch.randn(1, 512, 7, 7)
        out1 = model(x)
        out2 = model(x)
        # With p=0.5 on 1280 units, probability both are identical is ~2^{-1280}
        assert not torch.allclose(out1, out2)

    def test_c_hidden_is_1280(self) -> None:
        """Internal conv outputs 1280 channels (efficientnet_b0 size)."""
        model = Classify(512, 10)
        assert model.conv.conv.out_channels == 1280

    def test_pool_reduces_spatial(self) -> None:
        """After adaptive avg pool, spatial dims are (1, 1)."""
        model = Classify(512, 1000).eval()
        x = torch.randn(1, 512, 7, 7)
        with torch.no_grad():
            # Run conv then pool manually
            after_conv = model.conv(x)
            after_pool = model.pool(after_conv)
        assert after_pool.shape == (1, 1280, 1, 1)

    def test_linear_output_nc(self) -> None:
        """Linear layer has out_features == nc."""
        for nc in (10, 100, 1000):
            model = Classify(512, nc)
            assert model.linear.out_features == nc

    def test_forward_no_softmax(self) -> None:
        """Output is raw logits — rows do NOT sum to 1.0 in eval mode."""
        model = Classify(512, 100).eval()
        x = torch.randn(2, 512, 7, 7)
        with torch.no_grad():
            out = model(x)
        row_sums = out.sum(dim=1)
        # Raw logits: no row should be close to 1.0 in general
        assert not torch.allclose(row_sums, torch.ones(2), atol=1e-1)
