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
    Simplified implementation of the 'mask downsampler' from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    The purpose of the model is to prepare raw mask predictions so that they
    can be combined with raw image encoding results to form a 'memory encoding'.

    This implementation combines functionality that is otherwise spread out in
    the original implementation. Most notably, it is made out of an original
    module of the same name which performs downsampling, see:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/memory_encoder.py#L17

    But some pre-processing steps, from a different component in the original
    implementation, is also included here as it seems like a more intuitive
    location for this functionality, see:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/sam2_base.py#L684-L695
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token=256,
        num_layers=4,
        downsample_per_layer=2,
        kernel_size=3,
    ):

        # Inherit from parent
        super().__init__()

        # Pre-compute the number of in/out channels for each downsampling convolution layer
        # -> For default settings the channel sequence looks like: [1, 4, 16, 64, 256]
        # -> input sequence: [1, 4, 16, 64], output sequence: [4, 16, 64, 256]
        channel_upsample_factor = downsample_per_layer**2
        channel_seq = [channel_upsample_factor**idx for idx in range(num_layers + 1)]

        # Define downsampling module
        layers = []
        for in_channels, out_channels in zip(channel_seq[:-1], channel_seq[1:]):
            layers.extend(
                [
                    DownsampleConv2D(in_channels, out_channels, kernel_size, downsample_per_layer),
                    LayerNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
        self.downsample = nn.Sequential(*layers)
        self._downsample_factor = int(downsample_per_layer**num_layers)

        # Add a final channel projection layer
        final_channel_count = channel_seq[-1]
        self.out_proj = Conv1x1Layer(final_channel_count, features_per_image_token)

        # Hard-coded scaling/offset values from original implementation, see:
        # https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/sam2_base.py#L692-L695
        self.register_buffer("mask_scale", torch.tensor(20.0), persistent=False)
        self.register_buffer("mask_bias", torch.tensor(-10.0), persistent=False)

    # .................................................................................................................

    def forward(self, mask_prediction: Tensor, target_hw: tuple[int, int], is_prompt_encoding=False) -> Tensor:
        """
        Encodes mask prediction and resizes it to match the given target height & width.
        The target size is expected to match the image encodings, so that the encoded
        mask data can be 'fused' with image data for forming 'memory encodings'

        If 'is_prompt_encoding' is True, the input mask is processed slightly differently,
        in accordance with the original implementation (binarize vs. sigmoid). This is
        somewhat equivalent to treating the mask as 'extremely confident' since it
        produces a result that would have needed to have +infinity values for the
        segmented area if it were to go through the normal sigmoid processing
        (i.e. if 'is_prompt_encoding' were False). See original code here:
        https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/sam2_base.py#L685-L690

        Returns:
            encoded_mask
            -> Has shape: BxCxHxW
            -> B batch size
            -> C is channels matching image encoding
            -> H & W matching the given target_hw
        """

        # Scale mask up, so that the result after downsampling will match the target sizing
        upsample_hw = [size * self._downsample_factor for size in target_hw]
        hires_mask = nn.functional.interpolate(
            mask_prediction,
            size=upsample_hw,
            mode="bilinear",
            align_corners=False,
        )

        # Prepare mask data & downsample for fusion with image encoding
        hires_mask = (hires_mask > 0.0).to(hires_mask.dtype) if is_prompt_encoding else torch.sigmoid(hires_mask)
        hires_mask = hires_mask * self.mask_scale + self.mask_bias
        target_hw_mask = self.downsample(hires_mask)

        return self.out_proj(target_hw_mask)

    # .................................................................................................................


class ConvNeXtBlock(nn.Module):
    """
    Slightly modified implementation of the 'CXBlock' from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

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
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/memory_encoder.py#L62

    From (very limited) testing, this module doesn't seem to contribute much to the
    overall model performance, that is, it can be skipped without much effect!
    """

    # .................................................................................................................

    def __init__(self, features_per_token=256, kernel_size=7, padding=3):

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
    is a convolution that specifically doesn't act along the
    depth (channels) of the input.
    It seems to be meant in the sense of a 'per-channel' convolution.

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
    """

    # .................................................................................................................

    def __init__(self, num_channels, kernel_size, padding):
        super().__init__(num_channels, num_channels, kernel_size, padding=padding, groups=num_channels)

    # .................................................................................................................


class DownsampleConv2D(nn.Conv2d):
    """
    Defined for readability. This is simply a strided 2D convolution, which
    has the effect of 'downsampling' it's inputs by some integer factor.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downsample_factor=2):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=downsample_factor, padding=padding)

    # .................................................................................................................
