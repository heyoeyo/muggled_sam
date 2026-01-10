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
    Self-attention block used by the exemplar detector model.
    This corresponds to a part of what was originally called
    the 'TransformerEncoderLayer', see:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/decoder.py#L113-L145
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_bnc: Tensor, a_posenc_bnc: Tensor) -> Tensor:
        """
        Multi-headed self-attention. Uses a residual output & post-norm
        Returns:
            attention_result (same shape as a_tokens_bnc)
        """
        a_with_posenc = a_tokens_bnc + a_posenc_bnc
        attn_result, _ = self.attn(a_with_posenc, a_with_posenc, a_tokens_bnc, need_weights=False)
        return self.norm(a_tokens_bnc + attn_result)

    # .................................................................................................................


class ExemplarCrossAttentionBlock(nn.Module):
    """
    First cross-attention block used by the exemplar detector model.
    This model mixes data from exemplar tokens into the
    (learned) detection/presence tokens.

    Originally, it corresponds to a part of the 'TransformerEncoderLayer', see:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/decoder.py#L147-L155
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(
        self, a_tokens_bnc: Tensor, a_posenc_bnc: Tensor, b_tokens_bnc: Tensor, key_padding_mask: Tensor
    ) -> Tensor:
        """
        Multi-headed cross-attention between a_tokens and b_tokens. Uses a residual output & post-norm
        Returns:
            attention_result (same shape as a_tokens_bnc)
        """
        a_with_posenc = a_tokens_bnc + a_posenc_bnc
        attn_result, _ = self.attn(
            a_with_posenc, b_tokens_bnc, b_tokens_bnc, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.norm(a_tokens_bnc + attn_result)

    # .................................................................................................................


class ImageCrossAttentionBlock(nn.Module):
    """
    Second cross-attention block used by the exemplar detector model.
    This model mixes data from image tokens into the (learned)
    detection/presence tokens.

    Originally, it corresponds to a part of the 'TransformerEncoderLayer', see:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/decoder.py#L163-L177
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(features_per_token, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(
        self, a_tokens_bnc: Tensor, a_posenc_bnc: Tensor, b_tokens_bnc: Tensor, b_posenc_bnc: Tensor, attn_mask: Tensor
    ) -> Tensor:
        """
        Multi-headed cross-attention between a_tokens and b_tokens. Uses a residual output & post-norm
        Returns:
            attention_result (same shape as a_tokens_bnc)
        """
        a_with_posenc = a_tokens_bnc + a_posenc_bnc
        b_with_posenc = b_tokens_bnc + b_posenc_bnc
        attn_result, _ = self.attn(a_with_posenc, b_with_posenc, b_tokens_bnc, attn_mask=attn_mask, need_weights=False)
        return self.norm(a_tokens_bnc + attn_result)

    # .................................................................................................................


class MLP2LayersPostNorm(nn.Module):
    """
    Simple 2-layer MLP with post-norm & residual output.

    This doesn't exist as-is in the original code, instead it's handled
    through a 'forward_ffn' function, which manually executes the components,
    here it's written as a standalone module for clarity.

    Original code for reference:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/decoder.py#L73
    """

    # .................................................................................................................

    def __init__(self, num_features: int, hidden_features_ratio: float = 8.0):

        # Inherit from parent
        super().__init__()

        # Define mlp layers
        num_hidden_features = int(round(hidden_features_ratio * num_features))
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_hidden_features),
            nn.ReLU(),
            nn.Linear(num_hidden_features, num_features),
        )
        self.norm = nn.LayerNorm(num_features)

    def forward(self, tokens_channels_last: Tensor) -> Tensor:
        """Calculates (post-normed) MLP with residual output"""
        mlp_out = self.mlp(tokens_channels_last)
        return self.norm(tokens_channels_last + mlp_out)

    # .................................................................................................................
