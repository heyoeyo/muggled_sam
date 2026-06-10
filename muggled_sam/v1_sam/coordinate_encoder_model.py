#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

# For type hints
from typing import TypeAlias
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Custom types

# Coordinate formats
XYPoint: TypeAlias = tuple[float, float]
XY1XY2: TypeAlias = tuple[XYPoint, XYPoint]


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV1CoordinateEncoder(nn.Module):
    """
    Modified implementation of the 'prompt positional-encoder' component described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/prompt_encoder.py#L171

    Performs positional encoding of (x,y) coordinates, used to encode prompts
    both foreground & background points, top-left & bottom-right bounding box coordinates
    as well as grid cell positions (i.e. every image patch token position). More specifically,
    this model converts (x,y) coordinates into larger vectors, which are sized to match the
    image patch tokens from the image encoder. For example, a single (x,y) pair, like (0.5, 0.5)
    will be transformed into a 256-dimensional (by default) vector. This allows the coordinates
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
        and returns BxNxF tensors, where N is the number of xy coordinates,
        F is the number of encoded features per coord. (256 by default).

        When handling box coordinates (e.g. 2 points, top-left/bottom-right),
        the input will have a shape of BxNx2x2 while the output will have
        a shape of BxNx2xF. Here N is the number of boxes, the remaining
        '2' in the output shape refers to the top-left/bottom-right entries.

        In both cases, the last '2' in the input shape is assumed to hold the
        (x,y) coordinate pair, which gets converted to a vector of length F.

        Returns:
            encoded_xy_coords
            -> If more than one input is given, the model will output
               a tuple/list in the same order!
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

    def create_grid_xy_coordinates(self, grid_hw: tuple[int, int]) -> Tensor:
        """
        Creates a grid of (x,y) coordinates matching a given grid height/width.
        For example, for a grid of height 2 and width 3, this function gives something like:
            ┌                                          ┐
            │ (0.17, 0.25)  (0.50, 0.25)  (0.83, 0.25) │
            │ (0.17, 0.75)  (0.50, 0.75)  (0.83, 0.75) │
            └                                          ┘
        * Note that the coordinates are inset by a 'half-step',
          so instead of ranging from 0 to 1, they are inset by 0.5/N,
          where N is the number of coordinate points (e.g. h or w).
          (this is done to remain consistent with original implementation)

        Returns:
            xy_grid_coords (shape: HxWx2)
        """

        # For clarity
        h, w = grid_hw
        device, dtype = self.grid_posenc.device, self.grid_posenc.dtype

        # Set up min/max coordinate values (with inset)
        half_step_hw = (0.5 / n for n in grid_hw)
        (y1, y2), (x1, x2) = [(0.0 + step, 1.0 - step) for step in half_step_hw]

        # Compute grid of (x,y) coordinates
        x_vals = torch.linspace(x1, x2, w, device=device, dtype=dtype)
        y_vals = torch.linspace(y1, y2, h, device=device, dtype=dtype)
        xy_coords = torch.stack([x_vals.repeat(h, 1), y_vals.repeat(w, 1).T], dim=-1)

        return xy_coords

    # .................................................................................................................

    def get_grid_position_encoding(self, patch_grid_hw: tuple[int, int]) -> Tensor:
        """
        Generates positional encodings for all possible (x,y) coordinates within a
        grid based on the provided sizing.
        Also caches results for the given height & width.

        Returns:
            grid_posenc
            -> Has shape: 1xFxHxW, F features per token, H & W matching grid height & width
            -> This encoding is expected to match the shape of the image encoding
        """

        # Update cached dense encoding if it doesn't match the given size
        h, w = patch_grid_hw
        _, _, curr_h, curr_w = self.grid_posenc.shape
        if curr_h != h or curr_w != w:
            xy_coords = self.create_grid_xy_coordinates(patch_grid_hw)
            self.grid_posenc = self.forward(xy_coords).permute(2, 0, 1).unsqueeze(0)

        return self.grid_posenc

    # .................................................................................................................

    def prepare_boxes(self, box_xy1xy2_norm_list: XY1XY2 | list[XY1XY2] | Tensor | None) -> Tensor:
        """
        Helper used to convert box inputs into a format usable by the model
        Accepts:
            a single xy1xy2 box - format like: [(x1,y1), (x2,y2)]
            list of xy1xy2 points - format like: [[(x1,y1),(x2,y2)], ...]
            tensor, with shape: Nx2x2 or (for batching) BxNx2x2
            empty list or None, which is considered an N=0 input
        Where N is the number of box prompts and B is the batch size

        Returns:
            box_tensor (shape: BxNx2x2)
        """

        # Fill in a blank box entry if missing
        if box_xy1xy2_norm_list is None or len(box_xy1xy2_norm_list) == 0:
            box_xy1xy2_norm_list = torch.empty((1, 0, 2, 2))

        # Make sure we're dealing with a tensor on device
        if isinstance(box_xy1xy2_norm_list, Tensor):
            box_tensor = box_xy1xy2_norm_list.to(device=self.twopi.device, dtype=self.twopi.dtype)
        else:
            box_tensor = torch.tensor(box_xy1xy2_norm_list, device=self.twopi.device, dtype=self.twopi.dtype)

        # Add batch dimension if needed (want shape: BxNx2x2)
        if box_tensor.ndim == 3:
            box_tensor = box_tensor.unsqueeze(0)
        elif box_tensor.ndim == 2:
            box_tensor = box_tensor.unsqueeze(0).unsqueeze(0)

        return box_tensor

    # .................................................................................................................

    def prepare_points(self, point_xy_norm_list: XYPoint | list[XYPoint] | Tensor | None) -> Tensor:
        """
        Helper used to convert point inputs into a format usable by the model
        Accepts:
            a single xy point - format like: (x,y)
            list of xy points - format like: [(x,y), ...]
            tensor, with shape: Nx2 or (for batching) BxNx2
            empty list or None, which is considered an N=0 input
        Where N is the number of point prompts and B is the batch size

        Returns:
            points_tensor (shape: BxNx2)
        """

        # Fill in black entry
        if point_xy_norm_list is None or len(point_xy_norm_list) == 0:
            point_xy_norm_list = torch.empty((1, 0, 2))

        # Make sure we're dealing with a tensor on device
        if isinstance(point_xy_norm_list, Tensor):
            point_tensor = point_xy_norm_list.to(device=self.twopi.device, dtype=self.twopi.dtype)
        else:
            point_tensor = torch.tensor(point_xy_norm_list, device=self.twopi.device, dtype=self.twopi.dtype)

        # Add batch dimension if needed (want shape: BxNx2)
        if point_tensor.ndim == 2:
            point_tensor = point_tensor.unsqueeze(0)
        elif point_tensor.ndim == 1:
            point_tensor = point_tensor.unsqueeze(0).unsqueeze(0)

        return point_tensor

    # .................................................................................................................
