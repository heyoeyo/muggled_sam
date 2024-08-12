#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PositionEmbeddingSine(nn.Module):
    """
    WIP - Intending to simplify/document
    Taken from:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/position_encoding.py
    """

    def __init__(self, num_pos_feats):
        super().__init__()

        assert num_pos_feats % 2 == 0, "Expecting even model width"

        self.num_pos_feats = num_pos_feats // 2
        self.temperature = 10000
        self.twopi = 2 * math.pi

        self.register_buffer("device_info", torch.empty(1), persistent=False)
        self.cache = {}

    def extra_repr(self):
        return f"features={2*self.num_pos_feats}, temp={self.temperature}"

    @torch.no_grad()
    def forward(self, batch_size: int, height: int, width: int):  # x: torch.Tensor):

        # For convenience
        device, dtype = self.device_info.device, self.device_info.dtype

        # Re-use cached result if available
        cache_key = (height, width)
        if cache_key in self.cache:
            # Note: 'repeat' changes values slightly!
            return self.cache[cache_key].clone()[None].repeat(batch_size, 1, 1, 1)

        y_embed = torch.arange(1, height + 1, dtype=torch.float32, device=device)
        y_embed = y_embed.view(1, -1, 1).repeat(batch_size, 1, width)

        x_embed = torch.arange(1, width + 1, dtype=torch.float32, device=device)
        x_embed = x_embed.view(1, 1, -1).repeat(batch_size, height, 1)

        # Normalize
        eps = 1e-6
        y_embed = (y_embed / (y_embed[:, -1:, :] + eps)) * self.twopi
        x_embed = (x_embed / (x_embed[:, :, -1:] + eps)) * self.twopi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).to(dtype)
        self.cache[cache_key] = pos[0]

        return pos
