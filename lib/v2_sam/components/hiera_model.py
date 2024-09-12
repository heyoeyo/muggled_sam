#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .hiera_blocks import PooledWindowedBlock, WindowedBlock, GlobalBlock

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HieraModel(nn.Module):
    """
    Simplified implementation of Hiera image encoder model from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    This model is a fairly complex, multi-stage, multi-resolution vision-transformer which uses
    windowed attention on most blocks. This implementation (for SAMv2) is expected to have 4 stages.
    At each stage (except the first) inputs are pooled, which results in spatial downsampling of
    processed image tokens. The third stage includes equally spaced non-windowed attention blocks.
    The spacing of the non-windowed attention blocks as well as the window sizes per (windowed)
    block are set by external configs and do not follow an intuitive pattern.

    The output of the model is a list of encoded image tokens output from each of the
    stages of the model. Each set of tokens is progressively halved in width & height,
    while doubled in feature count.

    This implementation hard-codes some of the structural patterns of the original implementation.
    Notably, this version explicitly represents the stages of the model as sub-modules.

    The original implementation can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/57bc94b7391e47e5968004a0698f8bf793a544d1/sam2/modeling/backbones/hieradet.py#L171

    The original model architecture is described in:
        "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles"
        By: Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal,
        Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer
        @ https://arxiv.org/abs/2306.00989
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token_1st_stage=96,
        num_heads_1st_stage=1,
        blocks_per_stage=(2, 3, 16, 3),
        window_size_per_stage=(8, 4, 14, 7),
        global_attention_spacing_per_stage=(None, None, 4, None),
    ):
        # Inherit from parent
        super().__init__()

        # Compute multiplier-based configs
        stage_multiplier = [2**stage_idx for stage_idx, _ in enumerate(blocks_per_stage)]
        features_per_stage = [features_per_token_1st_stage * mult for mult in stage_multiplier]
        heads_per_stage = [num_heads_1st_stage * mult for mult in stage_multiplier]

        # Generate configs that are different on the first stage
        initial_winsize_per_stage = [window_size_per_stage[0], *window_size_per_stage[:-1]]
        is_pooled_per_stage = [stage_idx > 0 for stage_idx, _ in enumerate(blocks_per_stage)]

        # Bundle per-stage config arguments and build stage modules
        stage_iter = zip(
            features_per_stage,
            heads_per_stage,
            blocks_per_stage,
            window_size_per_stage,
            initial_winsize_per_stage,
            global_attention_spacing_per_stage,
            is_pooled_per_stage,
        )
        self.stages = nn.ModuleList(HieraStage(*args) for args in stage_iter)

        # Store feature counts so sizes can be communicate to other models
        self._features_per_stage = tuple(features_per_stage)

    # .................................................................................................................

    def forward(self, patch_tokens_bhwc: Tensor) -> list[Tensor]:

        # Store intermediate results from each stage
        stage_results = []
        for stage in self.stages:
            patch_tokens_bhwc = stage(patch_tokens_bhwc)
            stage_results.append(patch_tokens_bhwc)

        # Return results with shape: BxCxHxW
        return [result.permute(0, 3, 1, 2) for result in stage_results]

    # .................................................................................................................

    def get_features_per_stage(self) -> tuple[int, int, int, int]:
        return self._features_per_stage

    # .................................................................................................................

    def set_window_sizes(self, window_size_per_stage: list[int | None]):
        """
        Updates the window size of each stage of the model. This is
        meant for experimental purposes.

        Window sizes should be provided as a list of integers or None,
        where None indicates that the original window size config should
        be used. For example:
            window_size_per_stage = [2, 4, None, 16]

        Note that the first block of each stage will share it's window
        size with the prior stage, in accordance with the original
        configuration structure of the model.
        """

        # Force window sizing to have as many entries as we have stages
        num_sizes = len(window_size_per_stage)
        num_stages = len(self.stages)
        if num_sizes < num_stages:
            window_size_per_stage = [*window_size_per_stage].extend([None] * (num_stages - num_sizes))

        # Have each stage update it's blocks
        first_layer_sizes = [window_size_per_stage[0], *window_size_per_stage[:-1]]
        for stage, winsize_1st_layer, winsize in zip(self.stages, first_layer_sizes, window_size_per_stage):
            stage.set_window_size(winsize_1st_layer, winsize)

        return self

    # .................................................................................................................


class HieraStage(nn.Sequential):
    """
    Represents a single stage of the hierarchical image encoder (Hiera) from SAMV2.

    Each stage consists of a sequence of (mostly) windowed transformer blocks for
    encoding image patch tokens. Except for the first stage, each stage begins with
    a 2x2 max-pooling, which reduces the spatial size of tokens while doubling the
    features per token. The window sizing varies per stage according to external
    configs, though the first block of each stage can use a different window size
    (usually matched to the stage before it).

    Within the 3rd stage of the model, there are always (at least for SAMv2) 3 blocks
    which use global attention (i.e. not windowed). The final block of stage 3 is
    always a global block, with the remaining two blocks spaced 'N' and '2N' blocks
    earlier in the sequence, where the global block spacing 'N' is given by an
    external config (i.e. the blocks aren't evenly spaced across the stage itself).

    Note: This module is not present in the original implementation. Instead all blocks are
    configured as a single sequence, with per-stage configurations handled on init.
    The equivalent original code can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/backbones/hieradet.py#L232
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token,
        num_heads,
        num_blocks,
        window_size,
        window_size_1st_layer,
        global_attention_spacing,
        requires_first_layer_pooling,
    ):

        # Figure out global attention layer indices
        last_block_idx = num_blocks - 1
        no_global_attn = global_attention_spacing is None
        global_attn_idxs = [] if no_global_attn else [last_block_idx - k * global_attention_spacing for k in range(3)]

        # Figure out the first block of the stage, which may require pooling
        FirstBlockModule = PooledWindowedBlock if requires_first_layer_pooling else WindowedBlock
        first_block = FirstBlockModule(features_per_token, num_heads, window_size_1st_layer)

        # Build remaining blocks
        blocks_list = [first_block]
        for block_idx in range(1, num_blocks):

            # Use windowed or global attention blocks as needed
            is_global_attn_layer = block_idx in global_attn_idxs
            if is_global_attn_layer:
                block = GlobalBlock(features_per_token, num_heads)
            else:
                block = WindowedBlock(features_per_token, num_heads, window_size)
            blocks_list.append(block)

        # Inherit from parent
        super().__init__(*blocks_list)

    # .................................................................................................................

    def set_window_size(self, window_size_1st_layer: int | None = None, window_size: int | None = None):
        """
        Update all blocks to use a new window size. A different
        size can be provided for the first layer, to mirror the
        original structuring of the model, where the first layer
        shares the window sizing of the previous layer.
        Set size to None to reset to initial configuration
        """

        # Tell blocks to update to target window size
        for idx, block in enumerate(self):
            block_winsize = window_size_1st_layer if idx == 0 else window_size
            block.set_window_size(block_winsize)

        return self

    # .................................................................................................................
