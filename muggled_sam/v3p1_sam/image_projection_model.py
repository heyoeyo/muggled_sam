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


class SAMV3p1ImageProjection(nn.Module):
    """
    Modified implementation of the 'Sam3DualViTDetNeck' (Sam3TriViTDetNeck in v3.1) from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    The original can be found here:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/necks.py#L130

    This model is responsible for 'projecting' encoded image tokens into a lower channel dimension,
    for use in follow-up processing stages. In SAMv1/v2, this projection was part of the image
    encoder itself. For SAMv3.1 the projection is separated into it's own model, as there are
    three different projections (here referred to v1, v2 & v3 variants) which are used for different tasks.
    """

    # .................................................................................................................

    def __init__(
        self,
        input_channels: int = 1024,
        out_channels_v1: int = 256,
        out_channels_v2: int = 256,
        out_channels_v3: int = 256,
    ):

        # Inherit from parent
        super().__init__()

        # Create multiple projection variants (used for different tasks)
        self.multires_proj_image = MultiResProjection(input_channels, out_channels_v1, reduce_hires_channels=True)
        self.multires_proj_video = MultiResProjection(input_channels, out_channels_v2, reduce_hires_channels=True)
        self.multires_proj_detect = MultiResProjection(input_channels, out_channels_v3, reduce_hires_channels=False)

    # .................................................................................................................

    def forward(
        self, tokens_bchw: Tensor
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        """
        Compute 3 projections each with 3 resolutions, 1x, 2x and 4x.
        - The 'v1' projections are used for interactive segmentation
        - The 'v2' projections are used for video segmentation
        - The 'v3' projections are used for object detection

        If only one projection is needed, consider using the '.v#_projection(...)' functions.

        Returns:
            v1_tokens_x1_x2_x4, v2_tokens_x1_x2_x4, v3_tokens_x1_x2_x4
        """

        v1_tokens_x1, v1_tokens_x2, v1_tokens_x4 = self.multires_proj_image(tokens_bchw)
        v1_out = (v1_tokens_x1, v1_tokens_x2, v1_tokens_x4)

        v2_tokens_x1, v2_tokens_x2, v2_tokens_x4 = self.multires_proj_video(tokens_bchw)
        v2_out = (v2_tokens_x1, v2_tokens_x2, v2_tokens_x4)

        v3_tokens_x1, v3_tokens_x2, v3_tokens_x4 = self.multires_proj_detect(tokens_bchw)
        v3_out = (v3_tokens_x1, v3_tokens_x2, v3_tokens_x4)

        return v1_out, v2_out, v3_out

    # .................................................................................................................

    def v1_projection(self, tokens_bchw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.multires_proj_image(tokens_bchw)

    def v2_projection(self, tokens_bchw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.multires_proj_video(tokens_bchw)

    def v3_projection(self, tokens_bchw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.multires_proj_detect(tokens_bchw)

    # .................................................................................................................


class MultiResProjection(nn.Module):
    """
    Simplified implementation of the functionality of the 'Sam3TriViTDetNeck'.
    This model is responsible for creating three copies of a given set of image
    tokens, with different resolutions and channel counts.
    It supports two variations as per the original code, which produce different
    channel counts among the different resolutions.

    The main reference for this code is here:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/necks.py#L231-L260

    There are additional processing steps not included in the original 'Sam3TriViTDetNeck'
    which are instead scattered across the sam3 code for some reason...
    Including them during projection simplfies the code quite a bit.
    For examples of the original (scattered) usage see:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/sam3_image_processor.py#L63-L72
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/sam3_video_base.py#L606-L610
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/sam3_tracker_base.py#L449-L454
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/sam3_multiplex_base.py#L753-L778
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1057-L1077
    """

    # .................................................................................................................

    def __init__(self, input_channels: int = 1024, out_channels: int = 256, reduce_hires_channels: bool = False):

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
            nn.Identity() if not reduce_hires_channels else Conv1x1Layer(out_channels, out_channels // 8),
        )
        self.proj_x2 = nn.Sequential(
            UpscaleCT2x2Layer(input_channels),
            Conv1x1Layer(input_channels // 2, out_channels),
            Conv3x3Layer(out_channels),
            nn.Identity() if not reduce_hires_channels else Conv1x1Layer(out_channels, out_channels // 4),
        )
        self.proj_x1 = nn.Sequential(
            Conv1x1Layer(input_channels, out_channels),
            Conv3x3Layer(out_channels),
        )

    # .................................................................................................................

    def forward(self, tokens_bchw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute 3 projections at different resolutions, 1x, 2x and 4x.

        There are two variants of this (depending on the module config),
        if 'reduce_hires_channels' is set, then the x2 and x4 output
        resolutions have the channel counts reduced by a factor of
        4 and 8 respectively.

        Returns:
            tokens_x1, tokens_x2, tokens_x4

        The output shapes are as follows (assuming input is shaped: BxCxHxW):
            tokens_x1 -> BxF1xHxW
            tokens_x2 -> BxF2x(2*H)x(2*W)
            tokens_x4 -> BxF3x(4*H)x(4*W)

        Where the features, F1, F2 and F3 may be different.
        With default configs:
            reduce_hires_channels=False: F1 = F2 = F3 = 256
            reduce_hires_channels=True:  F1 = 256, F2 = 64, F3 = 32
        """

        tokens_x1 = self.proj_x1(tokens_bchw)
        tokens_x2 = self.proj_x2(tokens_bchw)
        tokens_x4 = self.proj_x4(tokens_bchw)

        return tokens_x1, tokens_x2, tokens_x4

    # .................................................................................................................
