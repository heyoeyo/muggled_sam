#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .shared import Conv1x1Layer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HalfStepPatchEmbed(nn.Module):
    """
    Patch embedding model used to convert full-sized RGB images into
    a much smaller grid of image 'tokens' for processing by a transformer model.

    In this version (used by SAMV2), patches 'overlap' by having the convolution
    take half-steps between patches. This doubles the number of tokens compared
    to a conventional (whole-step) patch embedding!

    For example, below each 'pixel' is indicated with a vertical bar |,
    and the patch that that pixel belongs to is labeled A, B, C, etc.
    In 'typical' patch embedding, each pixel belongs to a single patch,
    while with half-step embeddings, many pixels will be included in
    multiple patches (e.g. the 3rd pixel down ends up in patch A & B):

      Typical       Half-Step
        A |           A |
        A |           A |
        A |           A | B
        B |             | B
        B |           C | B
        B |           C |
        C |           C | D
        C |             | D
        C |             | D
         etc.          etc.

    """

    # .................................................................................................................

    def __init__(self, features_per_token, patch_size_px=7, num_input_channels=3):

        # Inherit from parent
        super().__init__()

        # Force odd-sized patches
        assert (patch_size_px % 2) == 1, "Must use odd number for patch size"

        # Compute patch stride/padding for half-step patches
        stride = (patch_size_px + 1) // 2
        padding = stride - 1

        # Both patch grouping + linear transformation is handled with a single strided convolution step!
        self.proj = nn.Conv2d(
            num_input_channels,
            features_per_token,
            kernel_size=patch_size_px,
            stride=stride,
            padding=padding,
        )

    # .................................................................................................................

    def forward(self, image_tensor_bchw: Tensor) -> Tensor:
        """
        Reshapes & projects image tensor: BxCxHxW -> BxhxwxF
            -> Where B is batch size
            -> C is image channels (i.e. 3 for RGB image)
            -> F is features per token
            -> H, W are the height & width of the image
            -> h, w are the size of the patch grid
        """

        patch_tokens = self.proj(image_tensor_bchw)
        return patch_tokens.permute(0, 2, 3, 1)

    # .................................................................................................................


class WindowTiledPositionEncoding(nn.Module):
    """
    Simplified implementation of the position encoding components of the image encoder from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    The original implementation also references the following paper:
    https://arxiv.org/abs/2311.05613

    The position encoding is built directly into the image encoding in the original implementation,
    but has been separated for clarity here. Other minor adjustments include support for caching
    the position encodings, as well as potentially more flexibility in the size of encodings.

    Based on the following original code:
    https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/backbones/hieradet.py#L284
    """

    # .................................................................................................................

    def __init__(self, features_per_token, base_patch_grid_hw, window_tile_hw):

        # Inherit from parent
        super().__init__()

        # Storage for fixed-size learned position embedding
        self.base_embedding = nn.Parameter(torch.zeros(1, features_per_token, *base_patch_grid_hw))
        self.base_window_tile = nn.Parameter(torch.zeros(1, features_per_token, *window_tile_hw))

        # Allocate storage for caching positional encoding, so we don't keep re-calculating them
        self.register_buffer("cached_encoding_bhwc", torch.empty((1, 1, 1, features_per_token)), persistent=False)

    # .................................................................................................................

    def extra_repr(self) -> str:
        _, features_per_token, grid_h, grid_w = self.base_embedding.shape
        _, _, win_h, win_w = self.base_window_tile.shape
        features_str = f"features_per_token={features_per_token}"
        base_hw_str = f"base_grid_hw=({grid_h}, {grid_w})"
        win_hw_str = f"window_tile_hw=({win_h}, {win_w})"
        return f"{features_str}, {base_hw_str}, {win_hw_str}"

    # .................................................................................................................

    def forward(self, patch_tokens_bhwc: Tensor) -> Tensor:
        """Adds positional encoding to patch tokens"""
        _, grid_h, grid_w, _ = patch_tokens_bhwc.shape
        return patch_tokens_bhwc + self._scale_to_patch_grid((grid_h, grid_w))

    # .................................................................................................................

    def _scale_to_patch_grid(self, patch_grid_hw: tuple[int, int]) -> Tensor:
        """
        Helper used to make sure the position embeddings are scaled
        to match the input patch sizing, by linear interpolation.
        """

        # If sizing is different from the cache, then re-compute it
        grid_h, grid_w = patch_grid_hw
        _, cache_h, cache_w, _ = self.cached_encoding_bhwc.shape
        if grid_h != cache_h or grid_w != cache_w:

            # Scale base embedding to match patch grid size
            scaled_base = nn.functional.interpolate(self.base_embedding, size=patch_grid_hw, mode="bicubic")

            # Figure out how many x/y copies of the window tile are needed to match patch grid sizing
            _, _, win_h, win_w = self.base_window_tile.shape
            is_int_y_tiles = (grid_h % win_h) == 0
            is_int_x_tiles = (grid_w % win_w) == 0
            num_y_tiles = grid_h // win_h if is_int_y_tiles else 1 + grid_h // win_h
            num_x_tiles = grid_w // win_w if is_int_x_tiles else 1 + grid_w // win_w

            # Repeat/tile window embedding to match the patch grid shape
            # -> Need to truncate to exactly match the grid shape at the end!
            tiled_win_embed = self.base_window_tile.tile(1, 1, num_y_tiles, num_x_tiles)
            if not (is_int_x_tiles and is_int_y_tiles):
                tiled_win_embed = tiled_win_embed[:, :, :grid_h, :grid_w]

            # Add tiled window embedding and convert to channels-last shape: BxCxHxW -> BxHxWxC
            self.cached_encoding_bhwc = (scaled_base + tiled_win_embed).permute(0, 2, 3, 1)

        return self.cached_encoding_bhwc

    # .................................................................................................................


class OutputProjection(nn.Module):
    """
    Simplified implementation of the 'feature-pyramid-network' model from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    This model further processes the multi-resolution image tokens output
    from the Hiera image encoder. Importantly, this model has the effect of
    projecting all image tokens to a shared channel sizing!

    This implementation has been had most of it's flexibility removed. It also
    performs 'scalp' operation (discarding the lowest-res image tokens), which
    was handled by the parent image encoder in the original implementation.

    Code is adapted from:
    https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/backbones/image_encoder.py#L45
    """

    # .................................................................................................................

    def __init__(self, output_channels=256, input_channels_list=(896, 448, 224, 112)):

        # Inherit from parent
        super().__init__()

        in_chs_large_first = sorted(input_channels_list, reverse=True)
        self.multires_projs = nn.ModuleList(Conv1x1Layer(in_ch, output_channels) for in_ch in in_chs_large_first)

    # .................................................................................................................

    def forward(
        self, multires_tokens_largest_first: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        """
        Input is expected to be a list of 4 image tokens at multiple resolutions,
        where each entry has a shape: BxFxHxW
        -> B batch size, F features per token, grid height (H) and width (W)

        The ordering is expected to be largest-to-smallest (in terms of H & W),
        with each entry being progressively halved in size.

        This function applies processing which projects each of these multi-res tokens
        to a single shared channel size, while maintaining the multi-res shapes.
        However, the lowest resolution tokens are discarded!

        Returns:
            image_tokens_smallest_first_list, posembed_list
            -> Output tokens are ordered smallest-to-largest by H & W (this is reversed compared to input!)
        """

        # Project each of the image tokens to a shared channel dimension
        img_tokens_smallest_first = reversed(multires_tokens_largest_first)
        proj_tokens = [proj(tokens) for proj, tokens in zip(self.multires_projs, img_tokens_smallest_first)]

        # Split tokens into lowest-res & outputs
        # -> We only keep the 3 highest resolution tokens for output
        # -> This was done using a 'scalp' setting in the original code:
        # https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/backbones/image_encoder.py#L32
        lowres_features, *tokens_smallest_first_list = proj_tokens

        # Compute 'top-down-features' which are added to only the remaining lowres tokens, see:
        # https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/backbones/image_encoder.py#L115
        initial_dtype = lowres_features.dtype
        top_down_features = nn.functional.interpolate(
            lowres_features.to(dtype=torch.float32),
            size=tokens_smallest_first_list[0].shape[2:],
            mode="nearest",
            align_corners=None,
            antialias=False,
        )
        tokens_smallest_first_list[0] += top_down_features.to(dtype=initial_dtype)

        return tokens_smallest_first_list

    # .................................................................................................................
