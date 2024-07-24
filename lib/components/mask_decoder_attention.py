#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
from torch import nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class GenericAttention(nn.Module):
    """
    Generalized attention module, which can work for both self-attention or cross-attention,
    by altering the data that is provided as the query/key/value inputs. Also supports
    optional internal downscaling of the token features (reduces total computation).

    This is nearly identical to the 'multi-headed attention' model introduced
    in the "Attention Is All You Need" paper:
    https://arxiv.org/abs/1706.03762

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L185
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int, internal_features: int | None = None):

        # Inherit from parent
        super().__init__()

        # Fill in missing internal feature count if needed
        internal_features = internal_features if internal_features is not None else features_per_token
        assert internal_features % num_heads == 0, "num_heads must divide features_per_token"

        # Store sizing for attention calculations
        features_per_head = internal_features // num_heads
        self.num_heads = num_heads
        self.features_per_head = features_per_head

        # Pre-calculate attention scale factor and hold on the same device as the rest of the model
        self.register_buffer("attn_scale", torch.sqrt(torch.tensor(1.0 / features_per_head)), persistent=False)

        # Mappings used to generate QKV vectors for attention calculations
        self.q_proj = nn.Linear(features_per_token, internal_features)
        self.k_proj = nn.Linear(features_per_token, internal_features)
        self.v_proj = nn.Linear(features_per_token, internal_features)

        # Output layer used to restore input feature count
        self.out_proj = nn.Linear(internal_features, features_per_token)

        # Set up softmax as dedicated 'module' so that we can hook into it for debugging/analysis!
        self.softmax = nn.Softmax(dim=-1)

    # .................................................................................................................

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Perform (general) attention between query, key and value tokens
        All tokens are expected to have a shape of: BxNxF
        -> B is batch size, N is number of tokens, F is features per token
        -> keys & values must have the same number of tokens (N), but q can be different!

        Returns:
            encoded_query_tokens
            -> shape: BxNqxF (Nq is number of tokens matching q input)
        """

        # For convenience, get the (shared) batch size for reshaping operations
        b = q.shape[0]

        # Comput QKV tokens & split into 'per-head' shape
        # -> Inputs have shape: BxNxF
        # -> Projection changes shape to: BxNxF' (F' is 'internal' feature count)
        # -> Reshape to get features per head: BxNxHxf (H is number of heads, f is features per head)
        # -> Transpose gives final shape: BxHxNxf
        q = self.q_proj(q).reshape(b, -1, self.num_heads, self.features_per_head).transpose(1, 2)
        k = self.k_proj(k).reshape(b, -1, self.num_heads, self.features_per_head).transpose(1, 2)
        v = self.v_proj(v).reshape(b, -1, self.num_heads, self.features_per_head).transpose(1, 2)

        # Perform query-key 'scaled dot-product' for all heads
        # -> k.transpose converts shape: BxHxNxf -> BxHxfxN
        # -> Gives q/k multiply between shapes: BxHxNqxf @ BxHxfxN, result has shape: BxHxNqxN
        # -> Follow up multiply with v is between: BxHxNqxN @ BxHxNxf, gives final shape: BxHxNqxf
        attn = (q * self.attn_scale) @ k.transpose(-2, -1)
        attn = self.softmax(attn) @ v

        # Recombine per-head tokens and project back to input feature count
        # -> Tranpose converts shape: BxHxNqxf -> BxNqxHxf
        # -> Flatten merges per-head features to give shape: BxNqxF'
        # -> Output projection maps F' to F features giving output shape: BxNqxF
        enc_q_tokens = attn.transpose(1, 2).flatten(2)
        return self.out_proj(enc_q_tokens)

    # .................................................................................................................


class CrossAttentionNormed(nn.Module):
    """
    Helper variant of the attention model, intended for cross-attention betwewn
    two sets of tokens 'a_tokens' and 'b_tokens', along with positional encodings.

    This module is not part of the original SAM implementation as-is, but the
    computation exists as a recurring pattern through the 'TwoWayTransformer'
    As an example of the pattern this module represents, see this code block:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99-L104
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int, internal_features: int | None = None):
        super().__init__()
        self.attn = GenericAttention(num_heads, features_per_token, internal_features)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens, a_posenc, b_tokens, b_posenc):
        a_embed = a_tokens + a_posenc
        b_embed = b_tokens + b_posenc
        attn_result = self.attn(a_embed, b_embed, b_tokens)
        return self.norm(a_tokens + attn_result)

    # .................................................................................................................


class SelfAttentionNormed(nn.Module):
    """
    Self-attention implementation that mimics the cross-attention model, which includes
    a residual + layernorm output step (compared to regular 'attention' model).
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int, internal_features: int | None = None):
        super().__init__()
        self.attn = GenericAttention(num_heads, features_per_token, internal_features)
        self.norm = nn.LayerNorm(features_per_token)

    def forward(self, a_tokens, a_posenc):
        a_embed = a_tokens + a_posenc
        attn_result = self.attn(a_embed, a_embed, a_tokens)
        return self.norm(a_tokens + attn_result)

    # .................................................................................................................


class SelfAttentionNoPosenc(SelfAttentionNormed):
    """
    Variant of the self-attention model but further simplified to not include positional encodings.
    It also uses the layer norm slightly differently, as there is no 'residual connection' between
    the input and attention result!
    Structured to match the position-encoding version, so that it can be used as a drop-in replacement.
    """

    # .................................................................................................................

    def forward(self, a_tokens, a_posenc):
        return self.norm(self.attn(a_tokens, a_tokens, a_tokens))

    # .................................................................................................................
