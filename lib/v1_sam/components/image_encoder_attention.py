#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class GlobalAttentionBlock(nn.Module):
    """
    A single transformer 'global' block layer (i.e. no windowing)

    Adapted from segment-anything model (SAM):
    https://github.com/facebookresearch/segment-anything
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    """

    # .................................................................................................................

    def __init__(self, features_per_token, num_heads, base_patch_grid_hw, mlp_ratio=4, norm_eps=1e-6) -> None:

        # Inherit from parent
        super().__init__()

        # Set up nn processing components
        self.norm1 = nn.LayerNorm(features_per_token, eps=norm_eps)
        self.attn = RelPosAttention(features_per_token, num_heads, base_patch_grid_hw)
        self.norm2 = nn.LayerNorm(features_per_token, eps=norm_eps)
        self.mlp = MLP2Layers_GELU(features_per_token, mlp_ratio)

    # .................................................................................................................

    def forward(self, x_in: Tensor) -> Tensor:

        x = x_in + self.attn(self.norm1(x_in))
        x = x + self.mlp(self.norm2(x))

        return x

    # .................................................................................................................

    def set_window_size(self, window_size: int | None):
        """This block does not use windowing, so do nothing. This is included for compatibility with window blocks"""
        return self

    # .................................................................................................................


class WindowedAttentionBlock(GlobalAttentionBlock):
    """
    Transformer block with support for windowed attention.
    This functions similar to regular attention, but operates on smaller HxW 'windows'
    of image tokens. This can greatly reduce computational requirements at the
    cost of limiting the mixing of information across the entire image.

    Adapted from segment-anything model (SAM):
    https://github.com/facebookresearch/segment-anything
    """

    # .................................................................................................................

    def __init__(self, features_per_token, num_heads, base_window_size) -> None:

        # Initialize as parent, with window-sized patch grid
        window_grid_hw = (base_window_size, base_window_size)
        super().__init__(features_per_token, num_heads, window_grid_hw)

        # Store window size (+ backup of initial size, in case we change and need a reset!)
        self._init_window_size = base_window_size
        self._window_size = base_window_size

    # .................................................................................................................

    def forward(self, x_in: Tensor) -> Tensor:

        x = self.norm1(x_in)

        # Window partition
        hw_in = (x.shape[1], x.shape[2])
        x, pad_hw = window_partition(x, self._window_size)

        x = self.attn(x)

        # Reverse window partition
        x = window_unpartition(x, self._window_size, pad_hw, hw_in)

        x = x_in + x
        x = x + self.mlp(self.norm2(x))

        return x

    # .................................................................................................................

    def set_window_size(self, window_size: int | None):
        """
        Modifies the window size used by this transformer block. This
        is meant for experimental use only. If the size is given as
        None, then the block will reset to it's initial config sizing.
        """

        # If given None, use the default/initial config size
        self._window_size = self._init_window_size if window_size is None else max(1, window_size)

        return self

    # .................................................................................................................


class RelPosAttention(nn.Module):
    """
    Multi-head Attention block with relative position embeddings

    Adapted from segment-anything model (SAM):
    https://github.com/facebookresearch/segment-anything
    """

    # .................................................................................................................

    def __init__(self, features_per_token, num_heads, base_patch_grid_hw) -> None:

        # Inherit from parent
        super().__init__()

        features_per_head = features_per_token // num_heads

        self.num_heads = num_heads
        self.scale = features_per_head**-0.5

        self.qkv = nn.Linear(features_per_token, features_per_token * 3, bias=True)
        self.proj = nn.Linear(features_per_token, features_per_token)

        # Initialize relative positional embeddings, if needed
        self.relpos = RelativePositionEncoder(features_per_head, base_patch_grid_hw)
        # has_input_size = (input_w is not None) and (input_h is not None)
        # assert has_input_size, "Input WH must be provided if using relative positional encoding."
        # self.rel_pos_w = nn.Parameter(torch.zeros((2 * input_w) - 1, head_dim))
        # self.rel_pos_h = nn.Parameter(torch.zeros((2 * input_h) - 1, head_dim))

        # Set up softmax as dedicated 'module' so that we can hook into it for debugging/analysis!
        self.softmax = nn.Softmax(dim=-1)

    # .................................................................................................................

    def forward(self, x: Tensor) -> Tensor:

        # For clarity
        B, H, W, _ = x.shape
        num_input_elements = H * W
        qkv_size = 3

        # Linear input mapping to make query, key, value vectors, for each head
        # -> input x has shape: (B, H, W, C)
        # -> linear mapping has shape: (B, H, W, 3*C)
        # -> reshape has shape: (B, num_elems, 3, num_heads, C)  <- duplicates parameters for each head?
        # -> after permutation has shape (3, B, num_heads, num_elems, C)
        qkv = self.qkv(x).reshape(B, num_input_elements, qkv_size, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(qkv_size, B * self.num_heads, num_input_elements, -1).unbind(0)

        # Perform query-key 'scaled dot-product' for all heads
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add position encodings
        in_hw = (H, W)
        attn = self.relpos(attn, q, in_hw)

        # Value weighting + concatenation
        attn = self.softmax(attn)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        # Final linear mapping
        x = self.proj(x)

        return x

    # .................................................................................................................


class RelativePositionEncoder(nn.Module):

    # .................................................................................................................

    def __init__(self, features_per_head, base_patch_grid_hw):

        # Inherit from parent
        super().__init__()

        self.base_h, self.base_w = base_patch_grid_hw
        num_h_idxs = self._get_num_relative_indexes(self.base_h)
        num_w_idxs = self._get_num_relative_indexes(self.base_w)
        self.rel_pos_h = nn.Parameter(torch.zeros(num_w_idxs, features_per_head))
        self.rel_pos_w = nn.Parameter(torch.zeros(num_h_idxs, features_per_head))

    # .................................................................................................................

    def forward(self, attention, query_tokens, patch_grid_hw):

        rel_h = self.rel_pos_h
        rel_w = self.rel_pos_w

        # Check if we need to re-size the relative position encodings
        # (this happens if the patch grid size doesn't match the base embeddings!)
        grid_h, grid_w = patch_grid_hw
        if (grid_h != self.base_h) or (grid_w != self.base_w):
            rel_h = self._scale_to_new_size(self.rel_pos_h, grid_h)
            rel_w = self._scale_to_new_size(self.rel_pos_w, grid_w)

        attn = add_decomposed_rel_pos(attention, query_tokens, rel_h, rel_w, patch_grid_hw, patch_grid_hw)

        return attn

    # .................................................................................................................

    @staticmethod
    def _get_num_relative_indexes(size):
        """
        Helper used to figure out how many relative position indexes would exist
        for a given grid/window size. For example, for a grid with a side length of 4,
        the furthest possible grid points would be +/- 3 indices away. Therefore the relative
        indexing will range from: -3, -2, -1, 0, 1, 2, 3
        For a total of 7 (or 2*4 - 1) possible relative indices
        """

        return (2 * size) - 1

    # .................................................................................................................

    def _scale_to_new_size(self, base_embedding, new_size):
        """
        Helper used to make sure the position embeddings are scaled
        to match the input patch sizing, by linear interpolation.
        """

        # Get original shape/data type, so we can restore this on the output
        _, C = base_embedding.shape
        orig_dtype = base_embedding.dtype
        new_rel_count = (2 * new_size) - 1

        # Force embedding to float32 for computing interpolation
        # -> If we don't do this, we could get bad results/errors on lower precision dtypes
        embed_f32 = base_embedding.float()

        # Convert to shape needed by interpolation function and then convert back
        resized_embedding = (
            nn.functional.interpolate(
                embed_f32.permute(1, 0).unsqueeze(0),
                size=new_rel_count,
                mode="linear",
            )
            .squeeze(0)
            .permute(1, 0)
        )

        return resized_embedding.to(orig_dtype)

    # .................................................................................................................


class MLP2Layers_GELU(nn.Module):
    """
    Modified implementation of the 2-layer MLP model used by the image-encoder attention blocks in SAM:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/image_encoder.py#L162
    """

    # .................................................................................................................

    def __init__(self, num_features, hidden_features_ratio=4):
        super().__init__()

        # Calculate number of hidden features
        num_hidden_features = int(round(hidden_features_ratio * num_features))
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_features),
            nn.GELU(),
            nn.Linear(num_hidden_features, num_features),
        )

    # .................................................................................................................

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, tuple]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


# .....................................................................................................................


def window_unpartition(windows: Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# .....................................................................................................................


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = nn.functional.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    device = rel_pos_resized.device
    q_coords = torch.arange(q_size, device=device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


# .....................................................................................................................


def add_decomposed_rel_pos(
    attn: Tensor,
    q: Tensor,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn


# .....................................................................................................................
