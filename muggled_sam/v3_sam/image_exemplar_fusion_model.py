#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .components.image_exemplar_fusion_attention import SelfAttentionBlock, CrossAttentionBlock
from .components.position_encoding import SinusoidalPE2D
from .components.shared import MLP2LayersPreNorm, imagelike_to_rows_of_tokens, rows_of_tokens_to_imagelike

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3ImageExemplarFusion(nn.Module):
    """
    Simplified implementation of the 'TransformerEncoderFusion' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model is used to update image tokens with information from encoded exemplar tokens.

    See original implementation:
    https://github.com/facebookresearch/sam3/blob/b26a5f330e05d321afb39d01d3d4881f258f65ff/sam3/model/encoder.py#L462
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256, num_layers: int = 6, num_heads: int = 8, mlp_ratio: float = 8.0):

        # Inherit from parent
        super().__init__()

        self.img_posenc = SinusoidalPE2D(features_per_token)
        self.fusion_layers = nn.ModuleList(
            (ImageFusionLayer(features_per_token, num_heads, mlp_ratio) for _ in range(num_layers))
        )

    def forward(
        self,
        image_tokens_bchw: Tensor,
        exemplar_tokens_bnc: Tensor,
        exemplar_mask_bn: Tensor,
    ) -> Tensor:
        """
        Fuse information from the exemplar tokens into the given image tokens
        Returns:
            fused_image_tokens_bchw
        """

        # For clarity
        img_b, img_c, img_h, img_w = image_tokens_bchw.shape

        # Generate position encoding for image tokens
        image_tokens_bnc, img_hw = imagelike_to_rows_of_tokens(image_tokens_bchw)
        image_posenc_bnc, _ = imagelike_to_rows_of_tokens(self.img_posenc(img_h, img_w))

        # Fuse image tokens with exemplar data
        enc_img_tokens_bnc = image_tokens_bnc
        for layer in self.fusion_layers:
            enc_img_tokens_bnc = layer(enc_img_tokens_bnc, image_posenc_bnc, exemplar_tokens_bnc, exemplar_mask_bn)

        # Return tokens in 'image-like' format for output
        enc_img_tokens_bchw = rows_of_tokens_to_imagelike(enc_img_tokens_bnc, img_hw)
        return enc_img_tokens_bchw

    # .................................................................................................................


class ImageFusionLayer(nn.Module):
    """
    Simplified implementation of the 'TransformerEncoderLayer' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This is a transformer-like block which fuses data from exemplar tokens into image tokens.

    See original implementation:
    https://github.com/facebookresearch/sam3/blob/b26a5f330e05d321afb39d01d3d4881f258f65ff/sam3/model_builder.py#L119
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256, num_heads: int = 8, mlp_ratio: float = 8.0):

        # Inherit from parent
        super().__init__()

        # Set up model components
        self.img_selfattn = SelfAttentionBlock(features_per_token, num_heads)
        self.img_crossattn = CrossAttentionBlock(features_per_token, num_heads)
        self.img_mlp = MLP2LayersPreNorm(features_per_token, mlp_ratio)

    def forward(
        self,
        image_tokens_bnc: Tensor,
        image_posenc_bnc: Tensor,
        exemplar_tokens_bnc: Tensor,
        exemplar_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Fuse information from the exemplar tokens into the given image tokens
        Returns:
            fused_image_tokens_bnc (same shape)
        """

        enc_img_tokens_bnc = self.img_selfattn(image_tokens_bnc, image_posenc_bnc)
        enc_img_tokens_bnc = self.img_crossattn(enc_img_tokens_bnc, exemplar_tokens_bnc, exemplar_padding_mask)
        enc_img_tokens_bnc = self.img_mlp(enc_img_tokens_bnc)

        return enc_img_tokens_bnc

    # .................................................................................................................
