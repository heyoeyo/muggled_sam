#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .memory_image_fusion_attention import RoPESelfAttention, RoPECrossAttention
from .position_encoding import SinusoidalPE2D

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MemoryImageFusionTransformer(nn.Module):
    """
    This model is responsible for the bulk of the memory-to-image 'fusion' computation.
    It's a transformer model made of 'fusion' layers, which make use of both self & cross attention.

    It mostly corresponds to the 'TransformerEncoderCrossAttention' model from the original code base:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/decoder.py#L614
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L410
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token: int = 256,
        features_per_memory_token: int = 64,
        num_layers: int = 4,
        num_heads: int = 1,
        position_encoding_weight: float = 0.1,
    ):
        # Inherit from parent
        super().__init__()

        # Position encoding for image tokens
        self.img_posenc = SinusoidalPE2D(features_per_image_token)
        self._img_posenc_weight = position_encoding_weight

        # Build transformer layers
        layers_list = []
        for _ in range(num_layers):
            layer = MemoryImageFusionTransformerLayer(
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
        # https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/decoder.py#L683C17-L683C33
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
        num_heads: int = 1,
        mlp_ratio: float = 8,
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
