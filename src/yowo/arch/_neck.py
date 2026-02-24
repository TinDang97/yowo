"""FPN + PAN Neck for YOLO architectures.

Implements the Feature Pyramid Network (top-down) and Path Aggregation
Network (bottom-up) used identically in YOLO11 and YOLO26, differing only
in the ``c3k`` flag for neck C3k2 blocks.

Input:  [P3/8, P4/16, P5/32] feature maps from the backbone.
Output: [P3/8, P4/16, P5/32] enhanced feature maps for the Detect head.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from yowo.arch._attention import C3k2PSA
from yowo.arch._blocks import C3k2, Concat, Conv
from yowo.arch._config import ModelConfig, scale_channels, scale_repeats


class FPNPANNeck(nn.Module):
    """Feature Pyramid + Path Aggregation Network.

    Architecture (layer numbers reference the full 24-layer model):

    FPN (top-down):
      11: Upsample P5  → concat with P4 (layer 6)  → C3k2 → P4'
      14: Upsample P4' → concat with P3 (layer 4)  → C3k2 → P3'

    PAN (bottom-up):
      17: Conv downsample P3' → concat with P4' (layer 13) → C3k2 → P4''
      20: Conv downsample P4'' → concat with P5 (layer 10) → C3k2 → P5''

    Output: [P3', P4'', P5'']
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Scaled channel widths for each stage
        c3 = scale_channels(256, config)  # P3 channels
        c4 = scale_channels(512, config)  # P4 channels
        c5 = scale_channels(1024, config)  # P5 channels

        neck_c3k = config.neck_c3k
        n2 = scale_repeats(2, config)

        # --- FPN top-down ---
        # Layer 11: Upsample P5
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        # Layer 12: Concat (handled in forward)
        self.concat1 = Concat(dim=1)
        # Layer 13: C3k2 on [upsampled_P5 + P4] → P4'
        self.c3k2_fpn1 = C3k2(c5 + c4, c4, n=n2, c3k=neck_c3k)

        # Layer 14: Upsample P4'
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        # Layer 15: Concat (handled in forward)
        self.concat2 = Concat(dim=1)
        # Layer 16: C3k2 on [upsampled_P4' + P3] → P3'
        # P3 from backbone layer 4 has c4 channels (not c3)
        self.c3k2_fpn2 = C3k2(c4 + c4, c3, n=n2, c3k=neck_c3k)

        # --- PAN bottom-up ---
        # Layer 17: Conv downsample P3'
        self.down1 = Conv(c3, c3, 3, 2)
        # Layer 18: Concat (handled in forward)
        self.concat3 = Concat(dim=1)
        # Layer 19: C3k2 on [downsampled_P3' + P4'] → P4''
        self.c3k2_pan1 = C3k2(c3 + c4, c4, n=n2, c3k=neck_c3k)

        # Layer 20: Conv downsample P4''
        self.down2 = Conv(c4, c4, 3, 2)
        # Layer 21: Concat (handled in forward)
        self.concat4 = Concat(dim=1)
        # Layer 22: C3k2 on [downsampled_P4'' + P5] → P5''
        # YOLO26: n=1, Sequential(Bottleneck, PSABlock); YOLO11: standard C3k2
        n_last = 1 if config.end2end else n2
        if config.end2end:
            self.c3k2_pan2 = C3k2PSA(c4 + c5, c5, n=n_last)
        else:
            self.c3k2_pan2 = C3k2(c4 + c5, c5, n=n_last, c3k=True)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Enhance multi-scale features.

        Args:
            features: (P3, P4, P5) from backbone — layers 4, 6, 10.

        Returns:
            (P3', P4'', P5'') — enhanced features for Detect head.
        """
        p3, p4, p5 = features

        # FPN top-down
        up_p5 = self.up1(p5)
        fpn1 = self.c3k2_fpn1(self.concat1([up_p5, p4]))  # P4'

        up_p4 = self.up2(fpn1)
        fpn2 = self.c3k2_fpn2(self.concat2([up_p4, p3]))  # P3'

        # PAN bottom-up
        down_p3 = self.down1(fpn2)
        pan1 = self.c3k2_pan1(self.concat3([down_p3, fpn1]))  # P4''

        down_p4 = self.down2(pan1)
        pan2 = self.c3k2_pan2(self.concat4([down_p4, p5]))  # P5''

        return fpn2, pan1, pan2


__all__ = ["FPNPANNeck"]
