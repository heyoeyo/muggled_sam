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
        return self.bias + self.weight * zeroed_mean / channel_stdev

    # .................................................................................................................
