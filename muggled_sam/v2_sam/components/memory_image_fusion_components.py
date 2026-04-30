#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .posenc_sine import SinusoidalPE2D
from .memory_image_fusion_attention import RoPESelfAttention, RoPECrossAttention

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MemoryImageFusionTransformer(nn.Module):
    """
    This model is responsible for the bulk of the memory-to-image 'fusion' computation.
    It's a transformer model made of 'fusion' layers, which make use of both self & cross attention.

    It mostly corresponds to the 'MemoryAttention' module from the original code base:
    https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/memory_attention.py#L102
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token=256,
        features_per_memory_token=64,
        num_layers=4,
        num_heads=1,
        position_encoding_weight=0.1,
    ):
        # Inherit from parent
        super().__init__()

        # Position encoding for image tokens
        self.img_posenc = SinusoidalPE2D(features_per_image_token)
        self._img_posenc_weight = position_encoding_weight

        # Build transformer layers
        layers_list = []
        for _ in range(num_layers):
            layer = MemoryFusionTransformerLayer(
                features_per_image_token, features_per_memory_token, num_heads, mlp_ratio=8
            )
            layers_list.append(layer)
        self.layers = nn.ModuleList(layers_list)
        self.out_norm = nn.LayerNorm(features_per_image_token)

    # .................................................................................................................

    def forward(
        self,
        lowres_image_tokens_bchw: Tensor,
        memory_tokens: Tensor,
        memory_posenc: Tensor,
        num_ptr_tokens: int,
    ) -> Tensor:
        """
        Fuses memory data into image tokens. Also handles batching between memory and image data
        Returns:
            fused_image_tokens_bchw (same shape as low-res input)
        """

        # Get input shape so we can restore it on output & handle batching if needed
        mem_b = memory_tokens.shape[0]
        img_b, img_c, img_h, img_w = lowres_image_tokens_bchw.shape
        patch_hw = (img_h, img_w)
        if mem_b > 1 and img_b == 1:
            lowres_image_tokens_bchw = lowres_image_tokens_bchw.expand(mem_b, -1, -1, -1)
            img_b = mem_b

        # Apply position encoding & flatten to rows-of-tokens format, shape: BxNxC
        # https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/memory_attention.py#L141
        img_posenc_bchw = self.img_posenc(img_h, img_w) * self._img_posenc_weight
        image_with_posenc_tokens = lowres_image_tokens_bchw + img_posenc_bchw
        flat_imgtokens_bnc = image_with_posenc_tokens.flatten(2).permute(0, 2, 1)

        # Run transformer layers to fuse memory results with image tokens
        for layer in self.layers:
            flat_imgtokens_bnc = layer(patch_hw, flat_imgtokens_bnc, memory_tokens, memory_posenc, num_ptr_tokens)

        # Convert back to image-like shape, from: BxNxC -> BxCxHxW
        flat_imgtokens_bnc = self.out_norm(flat_imgtokens_bnc)
        return flat_imgtokens_bnc.permute(0, 2, 1).reshape(img_b, img_c, img_h, img_w)

    # .................................................................................................................


class MemoryFusionTransformerLayer(nn.Module):
    """
    Simplified implementation of the 'MemoryAttentionLayer' model from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    This model represents a single layer of the memory fusion model
    (called 'memory attention' in the original code base), which is
    responsible for updating the encoded image tokens (from the image encoder)
    using information from memory tokens encoded from prior frames or
    from initial prompt inputs.

    This implementation removes most of the flexibiity/toggle options of the
    original code and breaks apart some of the functionality into
    standalone modules for easier readability. The original code can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/memory_attention.py#L17
    """

    # .................................................................................................................

    def __init__(self, features_per_image_token=256, features_per_memory_token=64, num_heads=1, mlp_ratio=8):

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
    for use with 'memory encoding' tokens, used within the memory fusion steps of the SAMv2 model.

    The positional encodings for 'past memories' include an additive offset/embedding, which
    is a learned value and is different depending on how 'far away' the memory is, relative to
    the frame where it is being used. While these offsets are learned, the underlying 'base'
    positional encoding is fixed for a given image height & width. As a result, it's possible
    to pre-compute the result of adding each of the learned offsets to the fixed base encoding,
    which is what the model does (and caches the result for re-use).

    This module does not exist in the original SAMv2 implementation. Instead computing the base
    positional encoding and adding offsets was handled in separate areas.
    The base positional encodings are generated inside the memory encoder itself:
    https://github.com/facebookresearch/segment-anything-2/blob/dce7b5446f260cef9fdf3d3f1bc81519302d386f/sam2/modeling/memory_encoder.py#L179
    While the offsets are added inside the '_prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/segment-anything-2/blob/dce7b5446f260cef9fdf3d3f1bc81519302d386f/sam2/modeling/sam2_base.py#L576-L578

    In this implementation, these are merged together here, since this is the only place they are used!
    """

    # .................................................................................................................

    def __init__(self, features_per_memory_token=64, max_memory_history=6):

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
            cached_posencs = self.posenc(h, w).clone().repeat(num_offsets, 1, 1, 1)

            # Add learned position offsets to each of the entries and store so we don't have to re-compute
            for idx in range(num_offsets):
                cached_posencs[idx] += self.base_memposenc_offsets[idx].permute(2, 0, 1)
            self.pos_offset_cache = cached_posencs

        return self.pos_offset_cache[position_offset].repeat(b, 1, 1, 1)

    # .................................................................................................................


class MLP2Layers(nn.Module):
    """
    Simple standalone MLP module, used within the memory function transformer layers.
    This module does not exist in the original implementation, but is used here
    to help clean up the transformer layer code.
    The equivalent original code can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/memory_attention.py#L96-L98
    """

    # .................................................................................................................

    def __init__(self, num_features, hidden_features_ratio=8):

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
