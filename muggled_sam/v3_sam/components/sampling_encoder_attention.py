#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block used by the sampling encoder.
    This is a specific configuration of a part of a general model called 'TransformerEncoderLayer'.

    Original code for reference:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L245
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/encoder.py#L179-L184
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_bnc: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Multi-headed attention with key-padding_mask. Uses a pre-norm & residual output
        Returns:
            attention_result (same shape as a_tokens_bnc)
        """
        a_normed = self.norm(a_tokens_bnc)
        attn_result, _ = self.attn(a_normed, a_normed, a_normed, key_padding_mask=key_padding_mask, need_weights=False)
        return a_tokens_bnc + attn_result

    # .................................................................................................................


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block used by the sampling encoder. This is again a specific
    configuration of another part of the 'TransformerEncoderLayer' from the original code base.

    This model is used to further mix data from image tokens into sampling tokens.

    Original code for reference:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L245
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/encoder.py#L188-L197
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_bnc: Tensor, b_tokens_bnc: Tensor, b_posenc_bnc: Tensor) -> Tensor:
        """
        Multi-headed cross-attention between a_tokens and b_tokens. Uses a pre-norm & residual output
        Returns:
            cross_attention_result (same shape as a_tokens_bnc)
        """
        a_normed = self.norm(a_tokens_bnc)
        b_embed = b_tokens_bnc + b_posenc_bnc
        attn_result, _ = self.attn(a_normed, b_embed, b_tokens_bnc, need_weights=False)
        return a_tokens_bnc + attn_result

    # .................................................................................................................
