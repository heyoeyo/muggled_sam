#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .posenc_sine import PositionEmbeddingSine
from .multiscale_block import MultiScaleBlock
from .shared import Conv1x1Layer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class Hiera(nn.Module):
    """
    Partially simplified (still work to do here) implementation of Hiera image encoder model from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    Code is adapted from:
    https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/backbones/hieradet.py#L171
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token=96,
        num_heads=1,
        blocks_per_stage=(2, 3, 16, 3),
        window_size_per_stage=(8, 4, 14, 7),
        global_attn_spacing=4,
    ):
        # Inherit from parent
        super().__init__()

        # Constants (from original init)
        q_pool = 3
        q_stride = (2, 2)
        dim_mul = 2.0
        head_mul = 2.0

        # Make sure there are matching 'per-stage' settings & pooling setting is ok
        assert len(blocks_per_stage) == len(window_size_per_stage)
        assert 0 <= q_pool <= len(blocks_per_stage) - 1

        # self.q_stride = q_stride
        self.stage_ends = [sum(blocks_per_stage[:i]) - 1 for i in range(1, len(blocks_per_stage) + 1)]
        q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]

        # Figure out which block indices need global attention
        # -> Assumes all global attn indices occur in 3rd stage (which is true for all known model sizes)
        last_global_attn_block_idx = sum(blocks_per_stage[:3]) - 1
        global_attn_idxs = reversed([last_global_attn_block_idx - k * global_attn_spacing for k in range(3)])
        self.global_attn_idxs = tuple(global_attn_idxs)

        cur_stage = 1
        self.blocks = nn.ModuleList()

        total_num_blocks = sum(blocks_per_stage)
        for i in range(total_num_blocks):
            dim_out = features_per_token
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = window_size_per_stage[cur_stage - 1]

            if self.global_attn_idxs is not None:
                window_size = 0 if i in self.global_attn_idxs else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(features_per_token * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=features_per_token,
                dim_out=dim_out,
                num_heads=num_heads,
                q_stride=q_stride if i in q_pool_blocks else None,
                window_size=window_size,
            )

            features_per_token = dim_out
            self.blocks.append(block)

        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]]

    # .................................................................................................................

    def forward(self, x: Tensor) -> list[Tensor]:

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

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

        self.multires_projs = nn.ModuleList(Conv1x1Layer(in_ch, output_channels) for in_ch in input_channels_list)
        self.position_encoding = PositionEmbeddingSine(output_channels)

    # .................................................................................................................

    def forward(self, multires_tokens_largest_first: list[Tensor]):
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
            output_image_tokens_list, posembed_list
            -> Output tokens are ordered smallest-to-largest by H & W (this is reversed compared to input!)
        """

        # Project each of the image tokens to a shared channel dimension
        img_tokens_smallest_first = reversed(multires_tokens_largest_first)
        proj_tokens = [proj(tokens) for proj, tokens in zip(self.multires_projs, img_tokens_smallest_first)]

        # Split tokens into lowest-res & outputs
        # -> We only keep the 3 highest resolution tokens
        # -> This was done using a 'scalp' setting in the original code:
        # https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/backbones/image_encoder.py#L32
        lowres_features, *out_tokens_list = proj_tokens

        # Compute 'top-down-features' which are added to only the remaining lowres tokens, see:
        # https://github.com/facebookresearch/segment-anything-2/blob/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2/modeling/backbones/image_encoder.py#L115
        target_hw = proj_tokens[1].shape[2:]
        initial_dtype = lowres_features.dtype
        top_down_features = nn.functional.interpolate(
            lowres_features.to(dtype=torch.float32),
            size=target_hw,
            mode="nearest",
            align_corners=None,
            antialias=False,
        ).to(dtype=initial_dtype)

        out_tokens_list[0] += top_down_features
        posembed_list = [self.position_encoding(x).to(x.dtype) for x in out_tokens_list]

        return out_tokens_list, posembed_list

    # .................................................................................................................
