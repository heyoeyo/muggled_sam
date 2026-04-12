#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .position_encoding import RPEComplex, RPEReal

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
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/decoder.py#L1022

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
        rope_theta: float = 10000.0,
        use_complex_numbers: bool = True,
    ):
        # Inherit from parent
        super().__init__()

        # Store config for re-use
        self.num_heads = num_heads
        self.features_per_head = features_per_token // num_heads

        # Set up rotary position encoder
        PosEncoder = RPEComplex if use_complex_numbers else RPEReal
        self.rotposenc = PosEncoder(self.features_per_head, rope_theta)

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
        # -> Inputs have shape: BxNxC (Nk and Nv are the same, but may be different from Nq!)
        # -> Reshape to get features per head: BxNxDxf (D is number of heads, f is features per head)
        # -> Transpose gives final shape: BxDxNxf
        batch_size_q, num_q = q.shape[0:2]
        batch_size_kv, num_k = k.shape[0:2]
        q = q.reshape(batch_size_q, num_q, self.num_heads, self.features_per_head).transpose(1, 2)
        k = k.reshape(batch_size_kv, num_k, self.num_heads, self.features_per_head).transpose(1, 2)
        v = v.reshape(batch_size_kv, num_k, self.num_heads, self.features_per_head).transpose(1, 2)

        # Apply position encoding (shapes: BxDxNxf)
        # -> 'num_k_keep' is used to only apply position encoding to some (non-object pointers) of the k tokens
        num_k_keep = num_k - num_final_k_to_exclude
        q, k[:, :, :num_k_keep] = self.rotposenc(q, k[:, :, :num_k_keep], q_tokens_hw)

        # Attention
        # -> Original implementation has extra optimizations using: 'with torch.backends.cuda.sdp_kernel'
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        # Recombine per-head tokens and project back to input feature count
        # -> Tranpose converts shape: BxDxNqxf -> BxNqxDxf
        # -> Flatten merges per-head features to give shape: BxNqxC
        enc_q_tokens = attn.transpose(1, 2).flatten(2)

        return enc_q_tokens

    # .................................................................................................................


class RoPESelfAttention(nn.Module):
    """
    This module implements a self-attention model using RoPE attention
    along with a pre-norm layer and residual output

    The original code handles this with configuration changes, see:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model_builder.py#L865
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int):
        super().__init__()
        self.attn = RoPEAttentionBNC(num_heads, features_per_token)
        self.norm = nn.LayerNorm(features_per_token)

        # Mappings used to generate QKV vectors for attention calculations
        self.q_proj = nn.Linear(features_per_token, features_per_token)
        self.k_proj = nn.Linear(features_per_token, features_per_token)
        self.v_proj = nn.Linear(features_per_token, features_per_token)
        self.out_proj = nn.Linear(features_per_token, features_per_token)

    def forward(self, tokens_hw: tuple[int, int], image_tokens_bnc: Tensor) -> Tensor:

        img_tokens_normed = self.norm(image_tokens_bnc)
        q = self.q_proj(img_tokens_normed)
        k = self.k_proj(img_tokens_normed)
        v = self.v_proj(img_tokens_normed)
        attn_result = self.attn(tokens_hw, q, k, v)
        attn_result = self.out_proj(attn_result)

        return image_tokens_bnc + attn_result

    # .................................................................................................................


class RoPECrossAttention(nn.Module):
    """
    This module implements a cross-attention model using RoPE attention
    It includes support for excluding some number of key tokens, which is
    needed by the fusion model to avoid adding position encodings to object pointers.

    This implementation is very specifically tailored to use within the
    memory fusion model, which uses a somewhat complicated collection
    of image & memory token inputs.

    As with self-attention, the original does this through configuration of a more general model:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model_builder.py#L875
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int):
        super().__init__()
        self.attn = RoPEAttentionBNC(num_heads, features_per_token)
        self.norm = nn.LayerNorm(features_per_token)

        # Mappings used to generate QKV vectors for various image tokens
        self.currimg_q_proj = nn.Linear(features_per_token, features_per_token)
        self.previmg_k_proj = nn.Linear(features_per_token, features_per_token)
        self.encimg_q_proj = nn.Linear(features_per_token, features_per_token)

        # Mappings used to generate KV vectors for memory tokens
        self.mem_k_proj = nn.Linear(features_per_token, features_per_token)
        self.mem_v_proj = nn.Linear(features_per_token, features_per_token)

        # Final projection layer
        self.out_proj = nn.Linear(features_per_token, features_per_token)

    def forward(
        self,
        tokens_hw: tuple[int, int],
        encoded_image_tokens_bnc: Tensor,
        curr_image_tokens_bnc: Tensor,
        prev_image_tokens_bnc: Tensor,
        memory_tokens_bnc: Tensor,
        memory_posenc_bnc: Tensor,
        num_final_k_to_exclude: int = 0,
    ) -> Tensor:

        # Fairly elaborate qkv projections for attention calculations, see:
        # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/decoder.py#L1190-L1204
        encimg_tokens_normed = self.norm(encoded_image_tokens_bnc)
        q = self.currimg_q_proj(curr_image_tokens_bnc) + self.encimg_q_proj(encimg_tokens_normed)
        k = self.previmg_k_proj(prev_image_tokens_bnc) + self.mem_k_proj(memory_tokens_bnc) + memory_posenc_bnc
        v = self.mem_v_proj(memory_tokens_bnc)

        attn_result = self.attn(tokens_hw, q, k, v, num_final_k_to_exclude)
        attn_result = self.out_proj(attn_result)
        return encoded_image_tokens_bnc + attn_result

    # .................................................................................................................
