#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

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
