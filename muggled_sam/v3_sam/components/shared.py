#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class LayerNorm2d(nn.Module):
    """
    Normalizes 'image-like' inputs along their channel-dimensions.
    Meant for use with image-like tokens from vision transformers

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/common.py#L31
    """

    # .................................................................................................................

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("eps", torch.tensor(eps), persistent=False)

    def forward(self, imagelike_bchw: Tensor) -> Tensor:
        """
        Input is expected to have an 'image-like' shape: BxCxHxW

        Returns:
            W * (input - in_mean) / in_stdev + B
            -> in_mean and in_stdev are calculated along the channels of the input
               (i.e. a unique mean/st.dev for each 'pixel' of the input)
            -> W & B are learned per-channel weights/bias values of this model
            -> Output shape is the same as the input: BxCxHxW
        """

        zeroed_mean = imagelike_bchw - imagelike_bchw.mean(1, keepdim=True)
        channel_stdev = torch.sqrt(zeroed_mean.square().mean(1, keepdim=True) + self.eps)
        return self.bias + self.weight * (zeroed_mean / channel_stdev)

    # .................................................................................................................


class MLPMultiLayer(nn.Sequential):
    """
    Simplified implementation of an MLP used by in SAM3, especially in the 'transformer.decoder':
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/model_misc.py#L160

    This version removes all of the optional toggle flags, leaving a very basic MLP
    with ReLU activations and support for different input/hidden/output channels.
    There is no residual output or layernorm.
    """

    # .................................................................................................................

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, num_layers: int = 2):

        # Sanity check
        assert num_layers > 1, f"MLP must have multiple layers! Got {num_layers=}"

        # Create input layers
        layers = [nn.Linear(input_channels, hidden_channels), nn.ReLU()]

        # Create hidden layers
        num_hidden_layers = max(num_layers - 2, 0)
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())

        # Add final output layer
        layers.append(nn.Linear(hidden_channels, output_channels))

        # Initialize sequential model
        super().__init__(*layers)

    # .................................................................................................................


class MLP2LayersPreNorm(nn.Module):
    """
    Simple 2-layer MLP with pre-norm & residual output.

    This is a specific MLP variant which is used in multiple parts of the code base.
    It doesn't exist as it's own model in the original code, instead the layers are
    executed manually inside of a transformer block.

    Note that the 'pre-norm' and 'residual' aspects are included to help mirror the
    corresponding self-attention & cross-attention blocks that appear in the same part of the codebase.

    Original code for reference:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/encoder.py#L198-L200
    """

    # .................................................................................................................

    def __init__(self, num_features: int, hidden_features_ratio: float = 8.0):

        # Inherit from parent
        super().__init__()

        # Define (pre-norm) mlp layers
        num_hidden_features = int(round(hidden_features_ratio * num_features))
        self.mlp = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_hidden_features),
            nn.ReLU(),
            nn.Linear(num_hidden_features, num_features),
        )

    def forward(self, tokens_channels_last: Tensor) -> Tensor:
        mlp_out = self.mlp(tokens_channels_last)
        return tokens_channels_last + mlp_out

    # .................................................................................................................


class Conv1x1Layer(nn.Conv2d):
    """
    Helper class used to create 1x1 2D convolution layers (i.e. depthwise convolution).
    Configured so that the width & height of the output match the input.
    If an output channel count isn't specified, will default to matching input channel count.
    """

    def __init__(self, in_channels: int, out_channels: int | None = None, bias=True):
        out_channels = in_channels if (out_channels is None) else out_channels
        super().__init__(in_channels, out_channels, bias=bias, kernel_size=1, stride=1, padding=0)

    # .................................................................................................................


class Conv3x3Layer(nn.Conv2d):
    """
    Helper class used to create commonly used 3x3 2D convolution layers.
    Configured so that the width & height of the output match the input.
    If an output channel count isn't specified, will default to matching input channel count.
    """

    def __init__(self, in_channels: int, out_channels: int | None = None, bias=True):
        out_channels = in_channels if (out_channels is None) else out_channels
        super().__init__(in_channels, out_channels, bias=bias, kernel_size=3, padding=1)

    # .................................................................................................................


class UpscaleCT2x2Layer(nn.ConvTranspose2d):
    """
    Helper class used to create 2x2 upscaling layer using 'ConvTranspose2d'.
    These are used to double the width/height of a set of image tokens,
    while (typically) halving the channel count.
    If an output channel count isn't specified, will default to *half* the input channel count.
    """

    def __init__(self, in_channels: int, out_channels: int | None = None, bias=True):
        out_channels = in_channels // 2 if (out_channels is None) else out_channels
        super().__init__(in_channels, out_channels, bias=bias, kernel_size=2, stride=2)

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def imagelike_to_rows_of_tokens(imagelike_bchw: Tensor) -> Tensor:
    """
    Helper used to convert formats. This doesn't look like it should work, but ends up
    giving the correct results, without memory copies due to some pytorch trickery
    in how data is stored. Returns: rows_of_tokens_bnc
    """
    img_b, img_c, img_h, img_w = imagelike_bchw.shape
    img_hw = (img_h, img_w)
    rows_of_tokens_bnc = imagelike_bchw.permute(0, 2, 3, 1).view(img_b, img_h * img_w, img_c)
    return rows_of_tokens_bnc, img_hw


def rows_of_tokens_to_imagelike(rows_of_tokens_bnc: Tensor, image_hw: tuple[int, int]) -> Tensor:
    """Sister function for imagelike_to_rows_of_tokens. Returns: imagelike_tokens_bchw"""
    tok_b, _, tok_c = rows_of_tokens_bnc.shape
    img_h, img_w = image_hw
    imagelike_bchw = rows_of_tokens_bnc.view(tok_b, img_h, img_w, tok_c).permute(0, 3, 1, 2)
    return imagelike_bchw
