# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Jump + Flow UNet model for image generation on R^d.
# Based on Generator Matching (Holderrieth et al., ICLR 2025), Appendix F.
#
# The model uses a standard UNet backbone with modified output channels.
# For each pixel dimension, it outputs:
#   - 1 value for flow velocity
#   - num_bins values for jump distribution logits
#   - 1 value for jump intensity (pre-softplus)
# Total out_channels per image channel = num_bins + 2

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from models.unet import UNetModel


@dataclass(eq=False)
class JumpFlowUNetModel(nn.Module):
    """UNet that outputs both flow velocity and jump kernel.

    Output layout per image channel (C=3 for RGB):
        [velocity (1), jump_logits (num_bins), jump_intensity (1)]

    So total UNet out_channels = C * (num_bins + 2).

    The model takes continuous x_t as input (from CondOT path).
    """

    num_bins: int = 256
    in_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    with_fourier_features: bool = False

    def __post_init__(self):
        super().__init__()
        # Total output per image channel: velocity(1) + bins(num_bins) + intensity(1)
        self.out_per_channel = self.num_bins + 2

        self.unet = UNetModel(
            in_channels=self.in_channels,
            model_channels=self.model_channels,
            out_channels=self.in_channels * self.out_per_channel,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            channel_mult=self.channel_mult,
            conv_resample=self.conv_resample,
            dims=self.dims,
            num_classes=self.num_classes,
            use_checkpoint=self.use_checkpoint,
            num_heads=self.num_heads,
            num_head_channels=self.num_head_channels,
            num_heads_upsample=self.num_heads_upsample,
            use_scale_shift_norm=self.use_scale_shift_norm,
            resblock_updown=self.resblock_updown,
            use_new_attention_order=self.use_new_attention_order,
            with_fourier_features=self.with_fourier_features,
            ignore_time=False,
            input_projection=True,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        extra: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning velocity and jump components.

        Args:
            x_t: continuous input state, shape [B, C, H, W], values in [-1, 1].
            t: time, shape [B].
            extra: conditioning dict (e.g., {"label": labels}).

        Returns:
            Dict with keys:
                "velocity": [B, C, H, W]
                "jump_logits": [B, C, num_bins, H, W]
                "jump_intensity": [B, C, H, W]
        """
        B, C, H, W = x_t.shape

        # UNet forward: [B, C * (num_bins + 2), H, W]
        raw = self.unet(x_t, t, extra)

        # Reshape: [B, C, num_bins + 2, H, W]
        raw = raw.reshape(B, C, self.out_per_channel, H, W)

        # Split outputs
        velocity = raw[:, :, 0, :, :]  # [B, C, H, W]
        jump_logits = raw[:, :, 1 : self.num_bins + 1, :, :]  # [B, C, num_bins, H, W]
        jump_intensity = raw[:, :, self.num_bins + 1, :, :]  # [B, C, H, W]

        return {
            "velocity": velocity,
            "jump_logits": jump_logits,
            "jump_intensity": jump_intensity,
        }


@dataclass(eq=False)
class JumpOnlyUNetModel(nn.Module):
    """UNet that outputs only jump kernel (no flow velocity).

    Output layout per image channel:
        [jump_logits (num_bins), jump_intensity (1)]

    Total UNet out_channels = C * (num_bins + 1).
    """

    num_bins: int = 256
    in_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    with_fourier_features: bool = False

    def __post_init__(self):
        super().__init__()
        self.out_per_channel = self.num_bins + 1

        self.unet = UNetModel(
            in_channels=self.in_channels,
            model_channels=self.model_channels,
            out_channels=self.in_channels * self.out_per_channel,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            channel_mult=self.channel_mult,
            conv_resample=self.conv_resample,
            dims=self.dims,
            num_classes=self.num_classes,
            use_checkpoint=self.use_checkpoint,
            num_heads=self.num_heads,
            num_head_channels=self.num_head_channels,
            num_heads_upsample=self.num_heads_upsample,
            use_scale_shift_norm=self.use_scale_shift_norm,
            resblock_updown=self.resblock_updown,
            use_new_attention_order=self.use_new_attention_order,
            with_fourier_features=self.with_fourier_features,
            ignore_time=False,
            input_projection=True,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        extra: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning jump components only.

        Args:
            x_t: continuous input, shape [B, C, H, W].
            t: time, shape [B].
            extra: conditioning dict.

        Returns:
            Dict with keys:
                "jump_logits": [B, C, num_bins, H, W]
                "jump_intensity": [B, C, H, W]
        """
        B, C, H, W = x_t.shape

        raw = self.unet(x_t, t, extra)
        raw = raw.reshape(B, C, self.out_per_channel, H, W)

        jump_logits = raw[:, :, : self.num_bins, :, :]  # [B, C, num_bins, H, W]
        jump_intensity = raw[:, :, self.num_bins, :, :]  # [B, C, H, W]

        return {
            "jump_logits": jump_logits,
            "jump_intensity": jump_intensity,
        }
