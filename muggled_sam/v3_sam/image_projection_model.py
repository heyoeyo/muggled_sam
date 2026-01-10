#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .components.shared import Conv1x1Layer, Conv3x3Layer, UpscaleCT2x2Layer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3ImageProjection(nn.Module):
    """
    Modified implementation of the 'Sam3DualViTDetNeck' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    The original can be found here:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/necks.py#L13

    This model is responsible for 'projecting' encoded image tokens into a lower channel dimension,
    for use in follow-up processing stages. In SAMv1/v2, this projection was part of the image
    encoder itself. For SAMv3, the projection is separated into it's own model, as there are
    two different projection sequences (here referred to as v3 & v2 variants) which are used
    for different tasks. For example, the 'v2' variant is used when doing tasks associated
    with SAMv2 (and SAMv1 image segmentation).
    """

    # .................................................................................................................

    def __init__(self, input_channels: int = 1024, out_channels_v2: int = 256, out_channels_v3: int = 256):

        # Inherit from parent
        super().__init__()

        # Create two projection variants (used for different tasks)
        self.multires_proj_v2 = MultiResProjection(input_channels, out_channels_v2, include_samv2_layers=True)
        self.multires_proj_v3 = MultiResProjection(input_channels, out_channels_v3, include_samv2_layers=False)

    # .................................................................................................................

    def forward(self, tokens_bchw: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        """
        Compute 3 projections at different resolutions, 1x, 2x and 4x
        for both a 'v3' and 'v2' variant. In practice, only one of these may
        be needed, in these cases consider using '.v#_projection#(...)' functions.

        Returns:
            v3_tokens_x1_x2_x4, v2_tokens_x1_x2_x4
        """

        v2_tokens_x1, v2_tokens_x2, v2_tokens_x4 = self.multires_proj_v2(tokens_bchw)
        v3_tokens_x1, v3_tokens_x2, v3_tokens_x4 = self.multires_proj_v3(tokens_bchw)

        return (v3_tokens_x1, v3_tokens_x2, v3_tokens_x4), (v2_tokens_x1, v2_tokens_x2, v2_tokens_x4)

    # .................................................................................................................

    def v3_projection(self, tokens_bchw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Perform only the 'v3' variant of projection. Returns: (tokens_x1, tokens_2, tokens_x4)"""
        return self.multires_proj_v3(tokens_bchw)

    # .................................................................................................................

    def v2_projection(self, tokens_bchw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Perform only the 'v2' variant of projection. Returns: (tokens_x1, tokens_2, tokens_x4)"""
        return self.multires_proj_v2(tokens_bchw)

    # .................................................................................................................


class MultiResProjection(nn.Module):
    """
    Simplified implementation of the functionality of the 'Sam3DualViTDetNeck'.
    This model is responsible for creating three copies of a given set of image
    tokens, with different resolutions and channel counts.
    It supports two variations as per the original code, which produce different
    channel counts among the different resolutions.

    The main reference for this code is here:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/necks.py#L13

    There are additional processing steps not included in the original 'Sam3DualViTDetNeck'
    which are instead scattered across the sam3 code. These steps seem to make sense as
    part of the projection steps directly and including them here seems simpler
    For examples of the original usage see:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_image_processor.py#L63-L72
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L450-L455
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_video_base.py#L382-L386
    """

    # .................................................................................................................

    def __init__(self, input_channels: int = 1024, out_channels: int = 256, include_samv2_layers: bool = False):

        # Inherit from parent
        super().__init__()

        # Set up 3-way upscaling/projection layers
        half_channels = input_channels // 2
        quarter_channels = input_channels // 4
        self.proj_x4 = nn.Sequential(
            UpscaleCT2x2Layer(input_channels),
            nn.GELU(),
            UpscaleCT2x2Layer(half_channels),
            Conv1x1Layer(quarter_channels, out_channels),
            Conv3x3Layer(out_channels),
        )
        self.proj_x2 = nn.Sequential(
            UpscaleCT2x2Layer(input_channels),
            Conv1x1Layer(input_channels // 2, out_channels),
            Conv3x3Layer(out_channels),
        )
        self.proj_x1 = nn.Sequential(
            Conv1x1Layer(input_channels, out_channels),
            Conv3x3Layer(out_channels),
        )

        # Odd detail. SAMv3 has a separate 'v2' projection, used for 'direct prompting'
        # -> The v2 projection includes extra layers that reduce the channel count of the hi-res tokens
        if include_samv2_layers:
            self.proj_x4.append(Conv1x1Layer(out_channels, out_channels // 8))
            self.proj_x2.append(Conv1x1Layer(out_channels, out_channels // 4))

        pass

    # .................................................................................................................

    def forward(self, tokens_bchw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute 3 projections at different resolutions, 1x, 2x and 4x.

        There are two variants of this (depending on the module config)
        In the 'v3' variant, each resolution has the same channel count.
        In the 'v2' variant, the x2 and x4 resolutions have their channel
        counts reduced by a factor of 4 & 8, respectively.

        Returns:
            tokens_x1, tokens_x2, tokens_x4

        The output shapes are as follows (assuming input is shaped: BxCxHxW):
            tokens_x1 -> BxF1xHxW
            tokens_x2 -> BxF2x(2*H)x(2*W)
            tokens_x4 -> BxF3x(4*H)x(4*W)

        Where the features, F1, F2 and F3 are different for the v3/v2 variants.
        With default configs:
            v3 variant: F1 = F2 = F3 = 256
            v2 variant: F1 = 256, F2 = 64, F3 = 32
        """

        tokens_x1 = self.proj_x1(tokens_bchw)
        tokens_x2 = self.proj_x2(tokens_bchw)
        tokens_x4 = self.proj_x4(tokens_bchw)

        return tokens_x1, tokens_x2, tokens_x4

    # .................................................................................................................
