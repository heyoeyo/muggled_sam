#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .shared import LayerNorm2d, Conv1x1Layer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MaskDownsampler(nn.Module):
    """
    The purpose of this model is to prepare raw mask predictions so that they
    can be combined with raw image encoding results to form a 'memory encoding'.

    This implementation combines functionality that is otherwise spread out in
    the original implementation. Most notably, it is made out of the original
    SimpleMaskDownSampler which performs downsampling, see:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/memory.py#L21

    But some pre-processing steps, from a different component in the original
    implementation, is also included here as it seems like a more intuitive
    location for this functionality, see:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1653-L1656
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1676-L1693

    This component also holds the major change introduced by the v3.1 update.
    Input masks are now assumed to always be 'multiplexed' so that instead of
    receiving a single mask, up to 16 (by default) masks can be provided,
    which will be downsampled/projected into a single set of tokens.
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token: int = 256,
        num_layers: int = 4,
        multiplex_channels: int = 16,
        downsample_per_layer: int = 2,
        kernel_size: int = 3,
    ):

        # Inherit from parent
        super().__init__()

        # Pre-compute the number of in/out channels for each downsampling convolution layer
        # -> This is the major change introduced by v3.1 update!!!
        # -> Now expect 'multiplexed' input (2*16 channels by default)
        # -> Add 4x as many channels per layer compared to SAMv2/v3 (presumably to help encode combined mask data)
        channel_upsample_factor = downsample_per_layer**2
        channel_seq = [2 * multiplex_channels]
        channel_seq.extend(16 * channel_upsample_factor**idx for idx in range(num_layers))

        # For clarity
        in_channel_seq = channel_seq[:-1]  # With default settings: [32, 16, 64, 256]
        out_channel_seq = channel_seq[1:]  # With default settings: [16, 64, 256, 1024]

        # Define downsampling module
        layers = []
        for in_channels, out_channels in zip(in_channel_seq, out_channel_seq):
            layers.extend(
                [
                    DownsampleConv2D(in_channels, out_channels, kernel_size, downsample_per_layer),
                    LayerNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
        self.downsample = nn.Sequential(*layers)

        # Add a final channel projection layer
        final_channel_count = channel_seq[-1]
        self.out_proj = Conv1x1Layer(final_channel_count, features_per_image_token)

        # Store run-time variables
        self._downsample_factor = int(downsample_per_layer**num_layers)
        self._in_channel_count = channel_seq[0]

        # Hard-coded scaling/offset values from original implementation, see:
        # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model_builder.py#L1016-L1017
        self.register_buffer("mask_scale", torch.tensor(2.0), persistent=False)
        self.register_buffer("mask_bias", torch.tensor(-1.0), persistent=False)

    # .................................................................................................................

    def forward(
        self,
        mask_prediction_bmhw: Tensor,
        target_hw: tuple[int, int],
        is_prompt_encoding_bm: Tensor,
    ) -> Tensor:
        """
        Encodes mask prediction and resizes it to match the given target height & width.
        The target size is expected to match the image encodings, so that the encoded
        mask data can be 'fused' with image data for forming 'memory encodings'

        Note that the mask predictions should have shape: BxMxHxW, where
        'B' is the multiplex 'buckets' (aka batchs) and 'M' is the number of
        multiplexed masks, H & W are height and width. The multiplex sizing is
        a model config parameter (16 by default), and masks *MUST* be this size
        as inputs (e.g. Bx16, with default model config)

        The 'is_prompt_encoding_bm' input is expected to be a BxM tensor which represents
        whether a mask is a prompt encoding (entry value of 1) or not (value of 0). It
        should have the same BxM as the masks and use the same device/dtype

        Returns:
            encoded_mask_bchw
            -> Has shape: BxCxHxW
            -> B batch size (will be > 1 if more than 16 multiplexed masks are given as input)
            -> C is channels matching image encoding
            -> H & W matching the given target_hw
        """

        # Resize mask so that the result after downsampling will match the target sizing
        upsample_hw = [size * self._downsample_factor for size in target_hw]
        hires_mask = mask_prediction_bmhw
        if mask_prediction_bmhw.shape[-2] != upsample_hw[0] or mask_prediction_bmhw.shape[-1] != upsample_hw[1]:
            hires_mask = nn.functional.interpolate(
                mask_prediction_bmhw,
                size=upsample_hw,
                mode="bilinear",
                align_corners=False,
            )

        # Scale masks into -1 to +1 range & build matching flags to indicate masks that are prompt encoding
        hires_mask = torch.sigmoid(hires_mask) * self.mask_scale + self.mask_bias

        # Merge mask with multiplex flags, following (strange) format introduced in v3.1
        # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1676-L1691
        mask_b, mask_m, hires_h, hires_w = hires_mask.shape
        is_prompt_bmhw = is_prompt_encoding_bm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hires_h, hires_w)
        padded_hires_mask = torch.concat((hires_mask, is_prompt_bmhw), dim=1)

        # Apply downscaling & projection steps
        out_mask_bchw = self.downsample(padded_hires_mask)
        return self.out_proj(out_mask_bchw)

    # .................................................................................................................


class ConvNeXtBlock(nn.Module):
    """
    Slightly modified implementation of the 'CXBlock' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This module itself comes from another paper (see figure 4):
        "A ConvNet for the 2020s"
        By: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
        @ https://arxiv.org/abs/2201.03545

    This implementation is only slightly different from the original, with a minor structural change
    to better indicate the 'inverted bottleneck' component along with renaming layers
    to more closely describe their usage. This implementation also uses 1x1 convolutions
    for the inverted bottleneck instead of linear layers for simplicity.
    -> the original notes that this may be slower, but if it is, it's hard to measure
    -> the two implementations do differ numerically! On the order of 1E-3

    The original implementation can be found here:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/memory.py#L90

    From (very limited) testing, this module doesn't seem to contribute much to the
    overall model performance, that is, it can be skipped without much effect!
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256, kernel_size: int = 7, padding: int = 3):

        # Inherit from parent
        super().__init__()

        # Define light-weight convolutions acting on channels separately
        self.per_channel_conv = Conv2DPerChannel(features_per_token, kernel_size, padding)
        self.norm = LayerNorm2d(features_per_token, eps=1e-6)

        # Define a sequence that up-projects/down-projects channel count
        accordion_dim = 4 * features_per_token
        self.inverted_bottleneck = nn.Sequential(
            Conv1x1Layer(features_per_token, accordion_dim),
            nn.GELU(),
            Conv1x1Layer(accordion_dim, features_per_token),
        )

        # Define per-channel scaling term with broadcastable shape matching: BxCxHxW
        self.per_channel_scale = nn.Parameter(torch.empty(1, features_per_token, 1, 1))

    # .................................................................................................................

    def forward(self, imagelike_bchw: Tensor) -> Tensor:
        """
        Takes in an image-like tensor (shape BxCxHxW) and primarily mixes
        information among the channels.

        Returns:
            encoded_imagelike_bchw (same shape as input)
        """

        residual = self.per_channel_conv(imagelike_bchw)
        residual = self.norm(residual)
        residual = self.inverted_bottleneck(residual) * self.per_channel_scale

        return imagelike_bchw + residual

    # .................................................................................................................


class Conv2DPerChannel(nn.Conv2d):
    """
    Defined to avoid confusion!

    This is referred to as a 'depth-wise' convolution in the
    original implementation (and elsewhere, it seems to be a standard term),
    but I find this to be confusing terminology, since the behavior
    is a convolution that specifically does *not* act along the
    depth (channels) of the input.
    It seems to be meant in the sense of convolution happening
    on each channel independently (e.g separate convolution per-channel).

    Normal 2D convolution performs dot-products between small
    2D windows of the input (based on the 'kernel_size') and
    all values along the entire depth (i.e. all channels) to produce
    a single number. That is, it performs dot-products between
    two tensors of shape: (C, ky, kx), where k is the 2D kernel size
    and C is the 'in_channels' config parameter. This is repeated with 'N'
    different kernels, where 'N' is the 'out_channels' config parameter.

    By contrast, this module performs convolutions on each channel independently.
    This means that each 2D window acts on only a single channel,
    so that values along the depth dimension are not mixed together.
    In other words, it performs dot-products between tensors of
    shape (1, ky, kx). This is repeated C times to produce an output
    that also has C channels (where each channel uses a unique kernel).

    Another way of thinking of this is that 'regular' Conv2D slides
    it's kernel in 2D, but computes results on 3D blocks of values
    (width x height x channels), while this 'per-channel' convolution
    slides in 2D and computes in 2D (width x height, separately per-channel).
    """

    # .................................................................................................................

    def __init__(self, num_channels: int, kernel_size: int, padding: int):
        super().__init__(num_channels, num_channels, kernel_size, padding=padding, groups=num_channels)

    # .................................................................................................................


class DownsampleConv2D(nn.Conv2d):
    """
    Defined for readability. This is simply a strided 2D convolution, which
    has the effect of 'downsampling' it's inputs by some integer factor.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downsample_factor: int = 2):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=downsample_factor, padding=padding)

    # .................................................................................................................
