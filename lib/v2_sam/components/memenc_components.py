#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .shared import LayerNorm2d


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MaskDownSampler(nn.Sequential):
    """
    WIP - Intending to simplify/document
    Taken from:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/memory_encoder.py
    """

    # .................................................................................................................

    def __init__(
        self,
        embed_dim=256,
        num_downsample_layers=4,
        kernel_size=3,
        stride=2,
        padding=1,
    ):

        # Pre-compute the number of in/out channels for each downsampling convolution layer
        channel_seq = [(stride**2) ** idx for idx in range(num_downsample_layers + 1)]
        in_channels = channel_seq[:-1]
        out_channels = channel_seq[1:]

        layers = []
        # self.encoder = nn.Sequential()
        for in_ch, out_ch in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
            layers.append(LayerNorm2d(out_ch))
            layers.append(nn.GELU())

        # Add a final convolution layer, without norm/activations
        last_in_ch = channel_seq[-1]
        layers.append(nn.Conv2d(last_in_ch, embed_dim, kernel_size=1))

        # Inherit from parent
        super().__init__(*layers)

    # .................................................................................................................


class CXBlock(nn.Module):
    """
    WIP - Intending to simplify/document
    Taken from:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/memory_encoder.py
    """

    # .................................................................................................................

    def __init__(self, dim, kernel_size=7, padding=3):

        # Inherit from parent
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.empty((dim)))

    # .................................................................................................................

    def forward(self, x):

        res_x = self.dwconv(x)
        res_x = self.norm(res_x)
        res_x = res_x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        res_x = self.pwconv1(res_x)
        res_x = self.act(res_x)

        res_x = self.pwconv2(res_x)
        res_x = self.gamma * res_x
        res_x = res_x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x + res_x
