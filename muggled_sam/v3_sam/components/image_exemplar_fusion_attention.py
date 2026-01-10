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
    Self-attention block used by the image-exemplar fusion model.
    This corresponds to a specific configuration of a part of a component
    originally called 'TransformerEncoderLayer'.

    This is used to further encode image tokens.

    Original code for reference:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L117
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/encoder.py#L179-L184
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_bnc: Tensor, a_posenc_bnc: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Multi-headed attention with pre-norm & residual output, and using query position encodings
        Returns:
            attention_result (same shape as a_tokens_bnc)
        """
        a_normed = self.norm(a_tokens_bnc)
        a_embed = a_normed + a_posenc_bnc
        attn_result, _ = self.attn(a_embed, a_embed, a_normed, key_padding_mask=key_padding_mask, need_weights=False)
        return a_tokens_bnc + attn_result

    # .................................................................................................................


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block used by the image-exemplar fusion model. This is again a specific
    configuration of another part of the 'TransformerEncoderLayer' from the original code base.

    This model is used to mix data from exemplar tokens into image tokens.

    Original code for reference:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L117
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/encoder.py#L188-L197
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_bnc: Tensor, b_tokens_bnc: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Multi-headed cross-attention between a_tokens and b_tokens. Uses a pre-norm & residual output
        Returns:
            cross_attention_result (same shape as a_tokens_bnc)
        """
        a_normed = self.norm(a_tokens_bnc)
        attn_result, _ = self.attn(
            a_normed, b_tokens_bnc, b_tokens_bnc, key_padding_mask=key_padding_mask, need_weights=False
        )
        return a_tokens_bnc + attn_result

    # .................................................................................................................
