#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn
import numpy as np

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2CoordinateEncoder(nn.Module):
    """
    Modified implementation of the 'prompt positional-encoder' component originally described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    While this version is used in SAMV2, the implementation is indentical to SAMV1.

    The code here is adapted from the original segment-anything v1 & v2 repos:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py#L171
    https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/position_encoding.py#L115

    Performs positional encoding of (x,y) coordinates, used to encode prompts
    both foreground & background points, top-left & bottom-right bounding box coordinates
    as well as grid cell positions (i.e. every image patch token position). More specifically,
    this model converts (x,y) coordinates into larger vectors, which are sized to match the
    image patch tokens from the image encoder. For example, a single (x,y) pair, like (0.5, 0.5)
    will be transformed into a 256-dimensional (by default) vector

    this allows the coordinates
    to be used within a transformer model (i.e. for cross-attention with image tokens).

    The original implementation named this module 'PositionEmbeddingRandom' and included it
    directly within the prompt encoder. However, structurally it seems simpler to be built as
    it's own separate model component, as it is shared by both the prompt encoder and mask decoder.
    """

    # .................................................................................................................

    def __init__(self, output_channels):

        # Inherit from parent
        super().__init__()

        # Calculate how many positional features are needed
        # -> We will calculate sin & cos encodings and concatenate the results
        # -> Therefore, to get target output embedding size, we need to halve pos. features
        num_coords = 2
        num_positional_features = output_channels // num_coords

        # Store image sizing info for generating positional encodings
        self.gaussian_matrix = nn.Parameter(torch.empty(num_coords, num_positional_features))

        # Allocate storage for values that are only calculated once
        self.register_buffer("grid_posenc", torch.empty(0, 0, 0, 0), persistent=False)
        self.register_buffer("twopi", torch.tensor(2.0 * torch.pi), persistent=False)

    # .................................................................................................................

    def forward(self, *xy_norm_coords_tensors: Tensor) -> list[Tensor]:
        """
        Generates positional encodings from normalized (0-to-1) xy coordinates

        When handling single point coordinates, takes in BxNx2 tensors
        and returns BxNxF tensors

        When handling box coordinates (e.g. 2 points, top-left/bottom-right),
        the input will have a shape of BxNx2x2 while the output will have
        a shape of BxNx2xF

        In both cases, N is the number of prompts while F is the number of
        features per encoded token.
        """

        results = []
        for xy_norm_tensor in xy_norm_coords_tensors:

            if xy_norm_tensor is None:
                results.append(None)
                continue

            xy_norm_tensor = 2.0 * xy_norm_tensor - 1.0
            xy_enc = (xy_norm_tensor @ self.gaussian_matrix) * self.twopi
            results.append(torch.cat([torch.sin(xy_enc), torch.cos(xy_enc)], dim=-1))

        return results if len(results) > 1 else results[0]

    # .................................................................................................................

    def get_full_grid_encoding(self, patch_grid_hw: tuple[int, int]) -> Tensor:
        """
        Generates positional encodings for all possible (x,y) coordinates within a
        grid based on the provided sizing. Caches results for the given height & width.

        Returns:
            grid_posenc
            -> Has shape: 1xFxHxW, F features per token, H & W matching grid height & width
            -> This encoding is expected to match the shape of the image encoding
        """

        # Update cached dense encoding if it doesn't match the given size
        h, w = patch_grid_hw
        _, _, curr_h, curr_w = self.grid_posenc.shape
        if curr_h != h or curr_w != w:

            with torch.inference_mode():
                device, dtype = self.grid_posenc.device, self.grid_posenc.dtype
                x_embed = torch.linspace(0.5, w - 0.5, w, device=device, dtype=dtype) / w
                y_embed = torch.linspace(0.5, h - 0.5, h, device=device, dtype=dtype) / h
                xy_embed = torch.stack([x_embed.repeat(h, 1), y_embed.repeat(w, 1).T], dim=-1)
                self.grid_posenc = self.forward(xy_embed).permute(2, 0, 1).unsqueeze(0)

        return self.grid_posenc

    # .................................................................................................................

    def prepare_boxes(self, box_tlbr_norm_list: list[list] | None) -> Tensor:
        """Helper used to convert box inputs into a format usable by the model"""

        # Fill in a blank box entry if none is given
        if box_tlbr_norm_list is None or len(box_tlbr_norm_list) == 0:
            box_tlbr_norm_list = np.empty((0, 2, 2))

        return torch.tensor(box_tlbr_norm_list, device=self.twopi.device, dtype=self.twopi.dtype).unsqueeze(0)

    # .................................................................................................................

    def prepare_points(self, fg_xy_norm_list: list | None, bg_xy_norm_list: list | None) -> tuple[Tensor, Tensor]:
        """Helper used to convert point inputs into a format usable by the model"""

        # Fill in (blank) missing entries
        if fg_xy_norm_list is None or len(fg_xy_norm_list) == 0:
            fg_xy_norm_list = np.empty((0, 2))
        if bg_xy_norm_list is None or len(bg_xy_norm_list) == 0:
            bg_xy_norm_list = np.empty((0, 2))

        device, dtype = self.twopi.device, self.twopi.dtype
        fg_tensor = torch.tensor(fg_xy_norm_list, device=device, dtype=dtype).unsqueeze(0)
        bg_tensor = torch.tensor(bg_xy_norm_list, device=device, dtype=dtype).unsqueeze(0)
        return fg_tensor, bg_tensor

    # .................................................................................................................
