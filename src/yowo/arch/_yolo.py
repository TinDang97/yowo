"""YOLO model assembler.

Builds a complete YOLOModel (backbone + neck + head) from a ModelConfig.
Supports Conv+BN fusion, channels_last, and torch.compile.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from yowo.arch._attention import C2PSA, Attention, C3k2PSA
from yowo.arch._blocks import SPPF, Bottleneck, C3k2, Conv, fuse_conv_and_bn
from yowo.arch._config import ClassifyConfig, ModelConfig, scale_channels, scale_repeats
from yowo.arch._heads import Classify, Detect, OBBHead
from yowo.arch._neck import FPNPANNeck

logger = logging.getLogger(__name__)


class Backbone(nn.Module):
    """YOLO backbone: layers 0-10.

    Produces three feature maps at strides 8, 16, 32 (P3, P4, P5).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        c1 = scale_channels(64, config)
        c2 = scale_channels(128, config)
        c3 = scale_channels(256, config)
        c4 = scale_channels(512, config)
        c5 = scale_channels(1024, config)

        n2 = scale_repeats(2, config)

        # Layer 0: P1/2 — 3→c1, stride 2
        self.stem = Conv(3, c1, 3, 2)

        # Layer 1: P2/4 — c1→c2, stride 2
        self.conv1 = Conv(c1, c2, 3, 2)

        # Layer 2: c2→c3, C3k2 block
        # YOLO26: c3k=True (all layers use C3k); YOLO11: c3k=False
        self.c3k2_1 = C3k2(c2, c3, n=n2, c3k=config.backbone_c3k, e=0.25, shortcut=True)

        # Layer 3: P3/8 — c3→c3, stride 2
        self.conv2 = Conv(c3, c3, 3, 2)

        # Layer 4: c3→c4, C3k2 block (same c3k flag as layer 2)
        self.c3k2_2 = C3k2(c3, c4, n=n2, c3k=config.backbone_c3k, e=0.25, shortcut=True)

        # Layer 5: P4/16 — c4→c4, stride 2
        self.conv3 = Conv(c4, c4, 3, 2)

        # Layer 6: c4→c4, C3k2 block (c3k=True for m/l/x, False for n/s)
        # Whether c3k is True depends on size, but in YOLO11/26 specs it's always True at layer 6
        self.c3k2_3 = C3k2(c4, c4, n=n2, c3k=True, shortcut=True)

        # Layer 7: P5/32 — c4→c5, stride 2
        self.conv4 = Conv(c4, c5, 3, 2)

        # Layer 8: c5→c5, C3k2 block (c3k=True)
        self.c3k2_4 = C3k2(c5, c5, n=n2, c3k=True, shortcut=True)

        # Layer 9: SPPF — detection only; classification models omit this layer
        if config.has_sppf:
            self.sppf = SPPF(c5, c5, k=5, shortcut=config.sppf_shortcut)

        # Layer 10 (detection) / Layer 9 (classification): C2PSA — attention
        self.c2psa = C2PSA(c5, c5, n=n2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run backbone, return (P3, P4, P5) feature maps.

        Saved outputs correspond to layers 4, 6, and 10 (detection) or 9
        (classification, where SPPF is absent) in the full model.
        """
        x = self.stem(x)  # Layer 0: /2
        x = self.conv1(x)  # Layer 1: /4
        x = self.c3k2_1(x)  # Layer 2

        x = self.conv2(x)  # Layer 3: /8
        p3 = self.c3k2_2(x)  # Layer 4 — save for neck

        x = self.conv3(p3)  # Layer 5: /16
        p4 = self.c3k2_3(x)  # Layer 6 — save for neck

        x = self.conv4(p4)  # Layer 7: /32
        x = self.c3k2_4(x)  # Layer 8
        if hasattr(self, "sppf"):
            x = self.sppf(x)  # Layer 9 (detection only)
        p5 = self.c2psa(x)  # Layer 9 (cls) / Layer 10 (detection)

        return p3, p4, p5


class YOLOModel(nn.Module):
    """Complete YOLO detection model: backbone + neck + head.

    Supports:
    - Conv+BN fusion via ``fuse()`` for inference speedup
    - ``channels_last`` memory format for GPU optimization
    - ``torch.compile`` friendly (no dynamic control flow)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = Backbone(config)
        self.neck = FPNPANNeck(config)

        # Detection head channel widths (P3, P4, P5)
        ch = (
            scale_channels(256, config),
            scale_channels(512, config),
            scale_channels(1024, config),
        )
        self.head = Detect(
            nc=config.num_classes,
            reg_max=config.reg_max,
            end2end=config.end2end,
            ch=ch,
            max_det=config.max_det,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: image tensor → detection output.

        Args:
            x: ``(B, 3, H, W)`` input tensor (RGB, normalized 0-1).

        Returns:
            YOLO11: ``(B, 4+nc, total_anchors)`` — requires NMS.
            YOLO26: ``(B, max_det, 6)`` — NMS-free ``[x1,y1,x2,y2,conf,cls]``.
        """
        features = self.backbone(x)
        enhanced = self.neck(features)
        return self.head(list(enhanced))

    def fuse(self) -> YOLOModel:
        """Fold BatchNorm into Conv2d weights for inference.

        Eliminates BN forward pass entirely (~15-20% fewer operations).
        Also specializes forward methods to eliminate per-frame overhead:
        - Conv layers with Identity activation skip the no-op act() call
        - Bottleneck layers use branch-free forward methods

        Returns self for method chaining.
        """
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                if isinstance(m.act, nn.Identity):
                    m.forward = m.forward_fuse_no_act  # type: ignore[assignment]
                else:
                    m.forward = m.forward_fuse  # type: ignore[assignment]
            elif isinstance(m, Bottleneck):
                if m.add:
                    m.forward = m._forward_shortcut  # type: ignore[assignment]
                else:
                    m.forward = m._forward_no_shortcut  # type: ignore[assignment]
        return self

    def forward_head(self, neck_features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Run detection head only (for cached feature map reuse).

        Args:
            neck_features: ``(P3', P4'', P5'')`` enhanced features from neck.

        Returns:
            Same format as ``forward()``.
        """
        return self.head(list(neck_features))

    def compile_for_inference(self, *, mode: str = "reduce-overhead") -> YOLOModel:
        """Apply ``torch.compile`` for optimized inference.

        Must be called **after** ``fuse()`` and ``eval()``. The compiled model
        traces through the forward graph on the first call (absorbed by
        ``warmup()``), then runs fused kernels on all subsequent calls.

        ``fullgraph=False`` is required because ``C2f.forward`` uses a Python
        ``list.extend`` with a generator, causing a graph break. Partial
        compilation still yields significant speedups from kernel fusion.

        Args:
            mode: Compilation mode passed to ``torch.compile``.
                ``"reduce-overhead"`` (default) fuses kernels and uses CUDA
                graphs for fixed-shape inputs. ``"default"`` is safer but
                slower.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If ``torch.compile`` is unavailable (PyTorch < 2.0).
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"torch.compile requires PyTorch >= 2.0. Current version: {torch.__version__}"
            )
        self.forward = torch.compile(  # type: ignore[assignment]
            self.forward, mode=mode, fullgraph=False
        )
        return self

    def enable_kv_cache(self, enabled: bool = True) -> YOLOModel:
        """Enable or disable KV cache and block output cache for streaming.

        Enables KV caching in all ``Attention`` modules (skips KV projection
        when cache is warm) and block-level output caching in ``C2PSA`` /
        ``C3k2PSA`` modules (skips entire block when input is similar).

        Returns self for method chaining.
        """
        for m in self.modules():
            if isinstance(m, Attention):
                m.enable_kv_cache(enabled)
            elif isinstance(m, C2PSA | C3k2PSA):
                m.enable_block_cache(enabled)
        return self

    def clear_kv_cache(self) -> None:
        """Clear all KV caches and block output caches."""
        for m in self.modules():
            if isinstance(m, Attention):
                m.clear_kv_cache()
            elif isinstance(m, C2PSA | C3k2PSA):
                m.clear_block_cache()


def _classify_to_model_config(config: ClassifyConfig) -> ModelConfig:
    """Convert a ClassifyConfig to a ModelConfig for Backbone construction.

    Fills detection-specific defaults that Backbone needs but ClassifyConfig
    does not carry. The returned config is only used to drive backbone
    channel/repeat scaling — neck and head fields are unused.
    """
    return ModelConfig(
        family=config.family,
        size=config.size,
        depth_mult=config.depth_mult,
        width_mult=config.width_mult,
        max_channels=config.max_channels,
        num_classes=80,
        reg_max=16,
        end2end=False,
        sppf_shortcut=config.sppf_shortcut,
        neck_c3k=False,
        backbone_c3k=config.backbone_c3k,
        has_sppf=config.has_sppf,
        max_det=300,
        input_size=(640, 640),
    )


class ClassifyModel(nn.Module):
    """YOLO classification model: Backbone → Classify head (no neck).

    Uses the same backbone as YOLOModel but discards P3/P4 and feeds P5
    directly into a pooling+linear classification head. No FPN/PAN neck needed.
    """

    def __init__(self, config: ClassifyConfig) -> None:
        super().__init__()
        self.config = config
        det_config = _classify_to_model_config(config)
        self.backbone = Backbone(det_config)
        head_ch = scale_channels(1024, det_config)
        self.head = Classify(head_ch, config.num_classes, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Returns raw logits (B, nc).

        Args:
            x: ``(B, 3, H, W)`` input tensor (RGB, normalized 0-1).

        Returns:
            ``(B, nc)`` raw logits.
        """
        _p3, _p4, p5 = self.backbone(x)
        return self.head(p5)

    def fuse(self) -> ClassifyModel:
        """Fold BatchNorm into Conv2d weights for inference.

        Returns self for method chaining.
        """
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse  # type: ignore[method-assign]
        return self


class OBBModel(nn.Module):
    """YOLO OBB detection model: backbone + neck + OBBHead.

    Reuses the same Backbone and FPNPANNeck as YOLOModel but replaces the
    Detect head with OBBHead, which adds an angle branch (cv4) for oriented
    bounding box prediction on DOTA datasets.

    Default nc=15 matches DOTA v1 (15 aerial object categories).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = Backbone(config)
        self.neck = FPNPANNeck(config)

        # Detection head channel widths (P3, P4, P5) — same as YOLOModel
        ch = (
            scale_channels(256, config),
            scale_channels(512, config),
            scale_channels(1024, config),
        )
        self.head = OBBHead(
            nc=config.num_classes,
            ne=1,
            reg_max=config.reg_max,
            ch=ch,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: image tensor → OBB detection output.

        Args:
            x: ``(B, 3, H, W)`` input tensor (RGB, normalized 0-1).

        Returns:
            ``(B, 4+nc+1, total_anchors)`` — cx, cy, w, h, class_scores..., angle.
        """
        features = self.backbone(x)
        enhanced = self.neck(features)
        return self.head(list(enhanced))

    def fuse(self) -> OBBModel:
        """Fold BatchNorm into Conv2d weights for inference.

        Returns self for method chaining.
        """
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                if isinstance(m.act, nn.Identity):
                    m.forward = m.forward_fuse_no_act  # type: ignore[assignment]
                else:
                    m.forward = m.forward_fuse  # type: ignore[assignment]
            elif isinstance(m, Bottleneck):
                if m.add:
                    m.forward = m._forward_shortcut  # type: ignore[assignment]
                else:
                    m.forward = m._forward_no_shortcut  # type: ignore[assignment]
        return self


__all__ = ["Backbone", "ClassifyModel", "OBBModel", "YOLOModel"]
