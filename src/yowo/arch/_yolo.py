"""YOLO model assembler.

Builds a complete YOLOModel (backbone + neck + head) from a ModelConfig.
Supports Conv+BN fusion, channels_last, and torch.compile.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from yowo.arch._attention import C2PSA
from yowo.arch._blocks import SPPF, C3k2, Conv, fuse_conv_and_bn
from yowo.arch._config import ModelConfig, scale_channels, scale_repeats
from yowo.arch._heads import Detect
from yowo.arch._neck import FPNPANNeck


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

        # Layer 2: c2→c3, C3k2 block (c3k=False for all sizes)
        self.c3k2_1 = C3k2(c2, c3, n=n2, c3k=False, e=0.25, shortcut=True)

        # Layer 3: P3/8 — c3→c3, stride 2
        self.conv2 = Conv(c3, c3, 3, 2)

        # Layer 4: c3→c4, C3k2 block (c3k=False for all sizes)
        self.c3k2_2 = C3k2(c3, c4, n=n2, c3k=False, e=0.25, shortcut=True)

        # Layer 5: P4/16 — c4→c4, stride 2
        self.conv3 = Conv(c4, c4, 3, 2)

        # Layer 6: c4→c4, C3k2 block (c3k=True for m/l/x, False for n/s)
        # Whether c3k is True depends on size, but in YOLO11/26 specs it's always True at layer 6
        self.c3k2_3 = C3k2(c4, c4, n=n2, c3k=True, shortcut=True)

        # Layer 7: P5/32 — c4→c5, stride 2
        self.conv4 = Conv(c4, c5, 3, 2)

        # Layer 8: c5→c5, C3k2 block (c3k=True)
        self.c3k2_4 = C3k2(c5, c5, n=n2, c3k=True, shortcut=True)

        # Layer 9: SPPF
        self.sppf = SPPF(c5, c5, k=5, shortcut=config.sppf_shortcut)

        # Layer 10: C2PSA — attention
        self.c2psa = C2PSA(c5, c5, n=n2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run backbone, return (P3, P4, P5) feature maps.

        Saved outputs correspond to layers 4, 6, 10 in the full model.
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
        x = self.sppf(x)  # Layer 9
        p5 = self.c2psa(x)  # Layer 10 — save for neck

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
        Returns self for method chaining.
        """
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse  # type: ignore[assignment]
        return self


__all__ = ["Backbone", "YOLOModel"]
