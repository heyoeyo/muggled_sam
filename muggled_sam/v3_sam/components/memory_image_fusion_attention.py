#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .position_encoding import RPEComplex, RPEMatrix

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class RoPEAttentionBNC(nn.Module):
    """
    Slightly modified implementation of the 'RoPEAttention' model from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This model is a base component for self-/cross-attention implementations
    (it's not used directly). It closely resembles the attention used by the
    SAMv3 image encoder, but is slightly different, notably because it
    expects 'rows-of-tokens' format inputs (e.g. shape: BxNxC), instead
    of 'image-like' tokens (shape BxHxWxC).

    This implementation slightly re-works the way the rotary position encodings
    are managed/applied, but is otherwise very similar to the original code, found here:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/sam/transformer.py#L266

    For more information about RoPE, the original paper seem to be the following:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    By: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
    @ https://arxiv.org/abs/2104.09864
    """

    # .................................................................................................................

    def __init__(
        self,
        num_heads: int,
        features_per_token: int,
        features_per_kv_token: int | None = None,
        rope_theta: float = 10000.0,
        use_complex_numbers: bool = True,
    ):
        # Inherit from parent
        super().__init__()

        # Store config for re-use
        self.num_heads = num_heads
        self.features_per_head = features_per_token // num_heads
        features_per_kv_token = features_per_token if features_per_kv_token is None else features_per_kv_token

        # Mappings used to generate QKV vectors for attention calculations
        self.q_proj = nn.Linear(features_per_token, features_per_token)
        self.k_proj = nn.Linear(features_per_kv_token, features_per_token)
        self.v_proj = nn.Linear(features_per_kv_token, features_per_token)

        # Set up rotary position encoder
        PosEncoder = RPEComplex if use_complex_numbers else RPEMatrix
        self.rotposenc = PosEncoder(self.features_per_head, rope_theta)

        # Output layer used to restore input feature count
        self.out_proj = nn.Linear(features_per_token, features_per_token)

    # .................................................................................................................

    def forward(
        self,
        q_tokens_hw: tuple[int, int],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_final_k_to_exclude: int = 0,
    ) -> Tensor:
        """
        Computes 'generic' attention between query/key/value tokens,
        while using rotary positional encodings. Can be used for either
        self or cross attention (self attention is when q, k, v are all the same).
        Inputs are expected to have shapes: BxNxF (i.e. 'rows of tokens' format)
        -> k & v must have the same N, q can be different

        Has support for excluding the final 'X' key tokens when applying
        positional encodings, which is meant to avoid assigning positioning
        info to non-positional tokens (e.g. object pointer tokens).
        Returns:
            encoded_query_tokens (same shape as q input)
        """

        # Compute QKV tokens & split into 'per-head' shape
        # -> Inputs have shape: BxNxC
        # -> Projection changes shape to: BxNxC' (C' matches the C for query tokens)
        # -> Reshape to get features per head: BxNxDxf (D is number of heads, f is features per head)
        # -> Transpose gives final shape: BxDxNxf
        batch_size_q, num_q = q.shape[0:2]
        batch_size_kv, num_k = k.shape[0:2]
        q = self.q_proj(q).reshape(batch_size_q, num_q, self.num_heads, self.features_per_head).transpose(1, 2)
        k = self.k_proj(k).reshape(batch_size_kv, num_k, self.num_heads, self.features_per_head).transpose(1, 2)
        v = self.v_proj(v).reshape(batch_size_kv, num_k, self.num_heads, self.features_per_head).transpose(1, 2)

        # Apply position encoding (shapes: BxDxNxc)
        num_k_keep = num_k - num_final_k_to_exclude
        q, k[:, :, :num_k_keep] = self.rotposenc(q, k[:, :, :num_k_keep], q_tokens_hw)

        # Attention
        # -> Original implementation has extra optimizations using: 'with torch.backends.cuda.sdp_kernel'
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        # Recombine per-head tokens and project back to input feature count
        # -> Tranpose converts shape: BxDxNqxf -> BxNqxDxf
        # -> Flatten merges per-head features to give shape: BxNqxC'
        # -> Output projection maps C' to C features giving output shape: BxNqxC
        enc_q_tokens = attn.transpose(1, 2).flatten(2)
        enc_q_tokens = self.out_proj(enc_q_tokens)

        return enc_q_tokens

    # .................................................................................................................


class RoPESelfAttention(nn.Module):
    """
    This module implements a self-attention model using RoPE attention
    along with a pre-norm layer and residual output

    The original code handles this with configuration changes, see:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L369
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int):
        super().__init__()
        self.attn = RoPEAttentionBNC(num_heads, features_per_token)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens_hw: tuple[int, int], a_tokens: Tensor) -> Tensor:
        a_normed = self.norm(a_tokens)
        attn_result = self.attn(a_tokens_hw, a_normed, a_normed, a_normed)
        return a_tokens + attn_result

    # .................................................................................................................


class RoPECrossAttention(nn.Module):
    """
    This module implements a cross-attention model using RoPE attention
    It includes support for excluding some number of key tokens, which is
    needed by the fusion model to avoid adding position encodings to object pointers.

    This implementation is very specifically tailored to use within the
    memory fusion model, which does not use position encodings for
    the image (query) tokens!

    As with self-attention, the original does this through configuration:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L381
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int, features_per_kv_token: int):
        super().__init__()
        self.attn = RoPEAttentionBNC(num_heads, features_per_token, features_per_kv_token)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(
        self,
        a_tokens_hw: tuple[int, int],
        a_tokens: Tensor,
        b_tokens: Tensor,
        b_posenc: Tensor,
        num_final_k_to_exclude: int = 0,
    ) -> Tensor:
        a_normed = self.norm(a_tokens)
        b_embed = b_tokens + b_posenc
        attn_result = self.attn(a_tokens_hw, a_normed, b_embed, b_tokens, num_final_k_to_exclude)
        return a_tokens + attn_result

    # .................................................................................................................
