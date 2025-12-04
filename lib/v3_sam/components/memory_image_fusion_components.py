#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .memory_image_fusion_attention import RoPESelfAttention, RoPECrossAttention
from .position_encoding import SinusoidalPE2D

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MemoryImageFusionTransformerLayer(nn.Module):
    """
    Simplified implementation of the 'TransformerDecoderLayerv2' model from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model represents a single layer of the memory-image-fusion model
    (called the 'tracker transformer' in sam3, "memory attention' in sam2),
    which is responsible for updating the encoded image tokens
    (from the image encoder) using information from memory tokens encoded
    from prior frames or from initial prompt inputs.

    This implementation removes most of the flexibiity/toggle options of the
    original code and breaks apart some of the functionality into
    standalone modules for easier readability.
    The original code is partly a wrapper around other components, see:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/decoder.py#L886
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L366
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token: int = 256,
        features_per_memory_token: int = 64,
        mlp_ratio: float = 8,
        num_heads: int = 1,
    ):

        # Inherit from parent
        super().__init__()

        # Image encoding layers
        self.image_selfattn = RoPESelfAttention(num_heads, features_per_image_token)
        self.image_crossattn = RoPECrossAttention(num_heads, features_per_image_token, features_per_memory_token)
        self.image_mlp = MLP2Layers(features_per_image_token, mlp_ratio)

    # .................................................................................................................

    def forward(
        self,
        image_patch_hw: tuple[int, int],
        image_tokens_bnc: Tensor,
        memory_tokens_bnc: Tensor,
        memory_posenc_bnc: Tensor,
        num_objpointer_tokens: int = 0,
    ) -> Tensor:
        """
        Encodes image tokens using self + cross attention with memory tokens
        Returns encoded image tokens (same shape as input)
        """

        enc_img_tokens = self.image_selfattn(image_patch_hw, image_tokens_bnc)
        enc_img_tokens = self.image_crossattn(
            image_patch_hw, enc_img_tokens, memory_tokens_bnc, memory_posenc_bnc, num_objpointer_tokens
        )
        enc_img_tokens = self.image_mlp(enc_img_tokens)

        return enc_img_tokens

    # .................................................................................................................


class FusionPositionOffset(nn.Module):
    """
    Helper module used to pre-compute & cache image-like positional encodings meant
    for use with 'memory encoding' tokens, used within the memory fusion steps of the SAMv3 model.

    The positional encodings for 'past memories' include an additive offset/embedding, which
    is a learned value and is different depending on how 'far away' the memory is, relative to
    the frame where it is being used. While these offsets are learned, the underlying 'base'
    positional encoding is fixed for a given image height & width. As a result, it's possible
    to pre-compute the result of adding each of the learned offsets to the fixed base encoding,
    which is what the model does (and caches the result for re-use).

    This module does not exist in the original SAMv2 implementation. Instead computing the base
    positional encoding and adding offsets was handled in separate areas.
    The base positional encodings are generated inside the memory encoder itself:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/memory.py#L199
    While the offsets are added inside the '_prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L676-L678

    In this implementation, these are merged together here, since this is the only place they are used!
    """

    # .................................................................................................................

    def __init__(self, features_per_memory_token: int = 64, max_memory_history: int = 6):

        # Inherit from parent
        super().__init__()

        num_pos_offsets = 1 + max_memory_history
        self.base_memposenc_offsets = nn.Parameter(torch.zeros(num_pos_offsets, 1, 1, features_per_memory_token))
        self.posenc = SinusoidalPE2D(features_per_memory_token)

        # Setup cache for holding pre-computed positional encodings with position offsets already added!
        blank_cache = torch.empty((num_pos_offsets, features_per_memory_token, 1, 1))
        self.register_buffer("pos_offset_cache", blank_cache, persistent=False)

    # .................................................................................................................

    def forward(self, imagelike_shape_bchw: Tensor, position_offset: int) -> Tensor:

        # Generate cached positional encodings + offsets if we get a new image shape
        b, _, h, w = imagelike_shape_bchw
        num_offsets, _, cache_h, cache_w = self.pos_offset_cache.shape
        if h != cache_h or w != cache_w:

            # Create image-like position encoding (shape: 1xFxHxW) and duplicate for each offset
            cached_posencs_bchw = self.posenc(h, w).clone().repeat(num_offsets, 1, 1, 1)

            # Add learned position offsets to each of the entries and store so we don't have to re-compute
            for idx in range(num_offsets):
                cached_posencs_bchw[idx] += self.base_memposenc_offsets[idx].permute(2, 0, 1)
            self.pos_offset_cache = cached_posencs_bchw

        return self.pos_offset_cache[position_offset].repeat(b, 1, 1, 1)

    # .................................................................................................................


class MLP2Layers(nn.Module):
    """
    Simple standalone MLP module, used within the memory function transformer layers.
    This module does not exist in the original implementation, but is used here
    to help clean up the transformer layer code.

    The equivalent original code can be found here:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/decoder.py#L947-L950
    """

    # .................................................................................................................

    def __init__(self, num_features: int, hidden_features_ratio: float = 8):

        # Inherit from parent
        super().__init__()

        # Define (per-norm) mlp layers
        num_hidden_features = int(round(hidden_features_ratio * num_features))
        self.mlp = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_hidden_features),
            nn.ReLU(),
            nn.Linear(num_hidden_features, num_features),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Calculates (pre-normed) MLP with residual output"""
        mlp_out = self.mlp(tokens)
        return tokens + mlp_out

    # .................................................................................................................
