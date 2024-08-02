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

        self.cache = {}

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.twopi
        y_embed = y * self.twopi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):

        batch_size, _, h, w = x.shape

        # Re-use cached result if available
        cache_key = (h, w)
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(batch_size, 1, 1, 1)

        y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=x.device)
        y_embed = y_embed.view(1, -1, 1).repeat(batch_size, 1, w)

        x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=x.device)
        x_embed = x_embed.view(1, 1, -1).repeat(batch_size, h, 1)

        # Normalize
        eps = 1e-6
        y_embed = (y_embed / (y_embed[:, -1:, :] + eps)) * self.twopi
        x_embed = (x_embed / (x_embed[:, :, -1:] + eps)) * self.twopi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]

        return pos
