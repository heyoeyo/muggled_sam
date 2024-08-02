#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import math

import torch
import torch.nn as nn

from .mask_decoder_attention import GenericAttention

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MemoryAttentionLayer(nn.Module):
    """
    WIP - Intending to simplify/document
    Taken from:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/memory_attention.py
    """

    # .................................................................................................................

    def __init__(self, d_model=256, d_memory=64, mlp_ratio=8):

        # Inherit from parent
        super().__init__()

        self.self_attn = RoPEAttention(features_per_token=d_model)
        self.cross_attn_image = RoPEAttention(features_per_token=d_model, features_per_kv_token=d_memory)

        # Implementation of Feedforward model
        hidden_dim = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # self.activation_str = activation
        self.relu = nn.ReLU()

        # Where to add pos enc
        self.pos_enc_at_attn = False
        self.pos_enc_at_cross_attn_queries = False
        self.pos_enc_at_cross_attn_keys = True

    # .................................................................................................................

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + tgt2
        return tgt

    # .................................................................................................................

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + tgt2
        return tgt

    # .................................................................................................................

    def forward(
        self,
        tgt,
        memory,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
        num_k_exclude_rope: int = 0,
    ) -> Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.relu(self.linear1(tgt2)))
        tgt = tgt + tgt2
        return tgt

    # .................................................................................................................


class RoPEAttention(GenericAttention):
    """
    WIP - Intending to simplify/document
    Taken from:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/sam/transformer.py
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int,
        features_per_kv_token: int | None = None,
        num_heads=1,
        rope_theta=10000.0,
    ):

        # Inherit from parent
        internal_features = None
        super().__init__(num_heads, features_per_token, internal_features, features_per_kv_token)

        self._axial_cis_dim = features_per_token // self.num_heads
        self._axial_cis_theta = rope_theta
        self.register_buffer("freqs_cis", torch.empty(1, 1), persistent=False)

        self.rope_k_repeat = features_per_kv_token is not None

    # .................................................................................................................

    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        if self.freqs_cis.shape[0] != q.shape[-2]:
            w = h = math.sqrt(q.shape[-2])
            self.freqs_cis = compute_axial_cis(self._axial_cis_dim, w, h, theta=self._axial_cis_theta)
        self.freqs_cis = self.freqs_cis.to(q.device)

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat
        )

        # Attention
        # -> Original implementation has extra optimizations using: 'with torch.backends.cuda.sdp_kernel'
        out = nn.functional.scaled_dot_product_attention(q, k, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

    # .................................................................................................................


# %% Functions


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
