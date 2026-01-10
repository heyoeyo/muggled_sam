#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.shared import (
    MLPMultiLayer,
    Conv1x1Layer,
    Conv3x3Layer,
    imagelike_to_rows_of_tokens,
    rows_of_tokens_to_imagelike,
)

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3ExemplarSegmentation(nn.Module):
    """
    Simplified implementation of the 'UniversalSegmentationHead' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model is responsible for generating the final mask predictions when working from exemplar
    inputs (e.g. text/points/boxes as exemplars).

    This roughly corresponds to the 'run_segmentation_heads' function and
    the 'UniversalSegmentationHead' from the original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/sam3_image.py#L388
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/maskformer_segmentation.py#L222
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 256,
        num_heads: int = 8,
        upscale_group_size: int = 32,
    ):
        # Inherit from parent
        super().__init__()

        # Set up image-exemplar to exemplar token attention
        self.image_cross_attn = ImageExemplarCrossAttentionBlock(features_per_token, num_heads)

        # Set up layers used to upscale image tokens for output
        self.upscale_x2 = ImageUpscaleLayer(features_per_token, upscale_group_size)
        self.upscale_x4 = ImageUpscaleLayer(features_per_token, upscale_group_size)

        # Set up final post-processing/projection components
        self.query_mlp = MLPMultiLayer(features_per_token, features_per_token, features_per_token, num_layers=3)
        self.img_token_proj = Conv1x1Layer(features_per_token, features_per_token)
        self.semantic_proj = Conv1x1Layer(features_per_token, 1)

    # .................................................................................................................

    def forward(
        self,
        detector_tokens_bnc: Tensor,
        image_exemplar_tokens_bchw: Tensor,
        hires_x2_image_tokens_bchw: Tensor,
        hires_x4_image_tokens_bchw: Tensor,
        exemplar_tokens_bnc: Tensor,
        exemplar_mask_bn: Tensor = None,
        interpolation_mode: str = "bilinear",
    ) -> tuple[Tensor, Tensor]:
        """
        Generates a segmentation masks for every detection token provided.
        Detection tokens are expected to come from the outout of the exemplar detector model.

        Note that this also takes in image tokens, but expects that the lowest-resolution
        tokens come from the image-exemplar fusion model, while the higher-resolution
        image tokens come from the image encoder!
        (though in practice the hi-res tokens aren't especially important)

        Returns:
            mask_predictions_bnhw, semantic_seg_bhw
            -> Masks have shape: BxQxHxW, where Q is number of query tokens, H & W will be 4-times
               the size of the lowest-resolution image tokens, by default H=W=288
            -> Semantic result has shape: BxHxW, with H & W matching the mask predictions
        """

        # Mix exemplar token info into image tokens (which have already had exemplar info fused in...)
        # -> Segmentation can still work when this is skipped!
        img_tokens_bnc, img_hw = imagelike_to_rows_of_tokens(image_exemplar_tokens_bchw)
        enc_img_tokens_bnc = self.image_cross_attn(img_tokens_bnc, exemplar_tokens_bnc, exemplar_mask_bn)
        enc_img_tokens_bchw = rows_of_tokens_to_imagelike(enc_img_tokens_bnc, img_hw)

        # Mix & upscale image-exemplar tokens into original hi-res image encoding results
        imode = interpolation_mode
        upscaled_img_tokens_bchw = self.upscale_x2(enc_img_tokens_bchw, hires_x2_image_tokens_bchw, imode)
        upscaled_img_tokens_bchw = self.upscale_x4(upscaled_img_tokens_bchw, hires_x4_image_tokens_bchw, imode)

        # Compute final mask predictions by mixing encoded detection queries with image tokens
        # -> For each image token, do a (channels-to-channels) dot-product with every query token
        # -> This produces 'Q' numbers (200 by default!) for each image token, giving shape: BxQxHxW
        # -> Each number is a mask prediction score, so we end up with Q mask predictions
        # -> Filtering out 'low confidence' query tokens before this can speed up this computation
        #    significantly! But the use of a 'query MLP' means this can introduce errors... (seems ok in practice?)
        det_query_tokens_mlp_bnc = self.query_mlp(detector_tokens_bnc)
        img_proj_bchw = self.img_token_proj(upscaled_img_tokens_bchw)
        mask_preds_bqhw = torch.einsum("bqc,bchw->bqhw", det_query_tokens_mlp_bnc, img_proj_bchw)

        # Compute 'semantic segmentation' result (appears in original code, but is only used for training?)
        semantic_seg_bhw = self.semantic_proj(upscaled_img_tokens_bchw).squeeze(1)

        return mask_preds_bqhw, semantic_seg_bhw

    # .................................................................................................................


class ImageExemplarCrossAttentionBlock(nn.Module):
    """
    Simplified/standalone implementation of the attention block used within the 'UniversalSegmentationHead'.
    This has been separated here for clarity and for consistency with other attention block implementations.

    Original code for reference:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/maskformer_segmentation.py#L282-L289
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_bnc: Tensor, b_tokens_bnc: Tensor, key_padding_mask: Tensor) -> Tensor:
        """
        Multi-headed cross-attention between a_tokens and b_tokens. Uses a pre-norm & residual output
        Returns:
            attention_result (same shape as a_tokens_bnc)
        """

        a_normed = self.norm(a_tokens_bnc)
        attn_result, _ = self.attn(
            a_normed, b_tokens_bnc, b_tokens_bnc, key_padding_mask=key_padding_mask, need_weights=False
        )
        return a_tokens_bnc + attn_result

    # .................................................................................................................


class ImageUpscaleLayer(nn.Module):
    """
    Helper module used to combine low-resolution image tokens with higher resolution
    tokens to create a new set of 'upscaled' image tokens. Includes post-processing steps.

    In the original implementation, this functionality is part of a 'PixelDecoder' model:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/maskformer_segmentation.py#L172
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256, norm_group_size: int = 32):

        # Inherit from parent
        super().__init__()

        # Sanity check
        is_divisible = (features_per_token % norm_group_size) == 0
        assert is_divisible, "Group size ({norm_group_size}) must divide into feature count ({features_per_token})"

        # Build post-processing layers
        num_groups = features_per_token // norm_group_size
        self.postprocess = nn.Sequential(
            Conv3x3Layer(features_per_token),
            nn.GroupNorm(num_groups, features_per_token),
            nn.ReLU(),
        )

    def forward(
        self, lowres_tokens_bchw: Tensor, hires_tokens_bchw: Tensor, interpolation_mode: str = "bilinear"
    ) -> Tensor:
        upsample_hw = hires_tokens_bchw.shape[-2:]
        out = hires_tokens_bchw + nn.functional.interpolate(
            lowres_tokens_bchw, size=upsample_hw, mode=interpolation_mode
        )
        return self.postprocess(out)

    # .................................................................................................................
