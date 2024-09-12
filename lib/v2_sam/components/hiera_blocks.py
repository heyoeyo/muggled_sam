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


class GlobalBlock(nn.Module):
    """
    Wrapper around self-attention block which adds layernorms & MLP on output.
    This is a standard vision-transformer-encoder with GELU (MLP) activations.

    This implementation corresponds to a specific configuration (window_size=0)
    of what was called a 'MultiScaleBlock' in the original implementation. See:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/backbones/hieradet.py#L82
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int, mlp_ratio=4.0):

        # Inherit from parent
        super().__init__()

        self.norm_preattn = nn.LayerNorm(features_per_token, eps=1e-6)
        self.attn = SelfAttention(num_heads, features_per_token)
        self.norm_premlp = nn.LayerNorm(features_per_token, eps=1e-6)
        self.mlp = MLP2Layers_GELU(features_per_token, mlp_ratio)

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Computes self-attention followed by MLP with pre-layernorms.
        See "An Image is Worth 16x16 Words" @ https://arxiv.org/abs/2010.11929
        Returns:
            encoded_image_tokens_bhwc (same shape as input)
        """

        # Attention (global, no windowing)
        attn = self.norm_preattn(imagelike_bhwc)
        attn = self.attn(attn) + imagelike_bhwc

        # MLP
        output = self.norm_premlp(attn)
        output = self.mlp(output) + attn

        return output

    # .................................................................................................................

    def set_window_size(self, window_size: int | None):
        """This block does not use windowing, so do nothing. This is included for compatibility with window blocks"""
        return self

    # .................................................................................................................


class WindowedBlock(nn.Module):
    """
    A windowed version of the standard transformer block.
    Windowing helps reduce the computational cost of a transformer block
    (by limiting token comparisons) at the cost of some reduction in the
    ability to encode data about long-range pairings of image tokens.

    This implementation corresponds to a specific configuration of what was
    called a 'MultiScaleBlock' in the original implementation. See:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/backbones/hieradet.py#L82
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, num_heads: int, window_size: int, mlp_ratio=4.0):

        # Inherit from parent
        super().__init__()

        # Store config for handling windowing operation
        self._init_window_size = window_size
        self._window_size = window_size

        # Attention components
        self.norm_preattn = nn.LayerNorm(features_per_token, eps=1e-6)
        self.attn = SelfAttention(num_heads, features_per_token)
        self.norm_premlp = nn.LayerNorm(features_per_token, eps=1e-6)
        self.mlp = MLP2Layers_GELU(features_per_token, mlp_ratio)

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Applies windowing to image tokens prior to self-attention calculation
        (then does standard MLP on the results). This helps to reduce
        the computational load comapred to 'normal' (or 'global') attention.
        Returns:
            encoded_image_tokens_bhwc (same shape as input)
        """

        # Windowed attention
        orig_shape = imagelike_bhwc.shape
        attn = self.norm_preattn(imagelike_bhwc)
        attn_windows, num_windows_xy = window_partition(attn, self._window_size)
        attn_windows = self.attn(attn_windows)
        attn = window_unpartition(attn_windows, orig_shape, self._window_size, num_windows_xy)
        attn = attn + imagelike_bhwc

        # MLP
        output = self.norm_premlp(attn)
        output = self.mlp(output) + attn

        return output

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


class PooledWindowedBlock(nn.Module):
    """
    A version of the windowed transformer block that also includes pooled-self-attention,
    which has the effect of downsampling (spatially) the input tokens. This implementation
    also assumes that the token features are doubled after pooling (to balance spatial reduction).

    This was called a 'MultiScaleBlock' in the original SAMV2 implementation:
    https://github.com/facebookresearch/segment-anything-2/blob/57bc94b7391e47e5968004a0698f8bf793a544d1/sam2/modeling/backbones/hieradet.py#L86

    This implementation is a specific hard-coded configuration
    of the original implementation (which is much more flexible).
    """

    # .................................................................................................................

    def __init__(self, output_features_per_token: int, num_heads: int, window_size: int, mlp_ratio=4.0):

        # Inherit from parent
        super().__init__()

        # Store config for handling windowing operation
        self._init_window_size = window_size
        self._window_size = window_size
        self._pool_window_size = window_size // 2

        # Compute number of features we need so that we get target ouput features,
        # assuming we double feature count due to pooling
        input_features_per_token = output_features_per_token // 2

        # Pooling components
        self.pool = MaxPool2x2()
        self.proj = nn.Linear(input_features_per_token, output_features_per_token)

        # Attention components
        self.norm_preattn = nn.LayerNorm(input_features_per_token, eps=1e-6)
        self.attn = PooledSelfAttention(num_heads, input_features_per_token)
        self.norm_premlp = nn.LayerNorm(output_features_per_token, eps=1e-6)
        self.mlp = MLP2Layers_GELU(output_features_per_token, mlp_ratio)

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Applies windowing to image tokens prior to pooled-self-attention calculation
        (then does standard MLP on the results). Pooled attention leads to
        spatially downsampled result, which also has doubled feature count.
        Returns:
            encoded_image_tokens
            -> For input with shape: BxHxWxC,
               output shape is: Bx(H/2)x(W/2)x(C*2)
        """

        # Windowed attention (with pooling)
        # -> The attention calculation includes it's own pooling! So result shape is different from input
        imagelike_bhwc = self.norm_preattn(imagelike_bhwc)
        attn_windows, num_windows_xy = window_partition(imagelike_bhwc, self._window_size)
        attn_windows = self.attn(attn_windows)

        # Create pooled 'residual' for adding to attention result
        pooled_imagelike_bhwc = self.pool(self.proj(imagelike_bhwc))
        pool_shape = pooled_imagelike_bhwc.shape

        # Reverse windowing but accounting for internal pooling on attention result
        attn = window_unpartition(attn_windows, pool_shape, self._pool_window_size, num_windows_xy)
        attn = attn + pooled_imagelike_bhwc

        # MLP
        output = self.norm_premlp(attn)
        output = self.mlp(output) + attn

        return output

    # .................................................................................................................

    def set_window_size(self, window_size: int | None):
        """
        Modifies the window size used by this transformer block. This
        is meant for experimental use only. If the size is given as
        None, then the block will reset to it's initial config sizing.

        Note that windows sizes will be forced to even multiples in order
        to maintain proper support for pooling!
        """

        # If not given a size, fallback to initial config
        if window_size is None:
            self._window_size = self._init_window_size

        else:
            # Make sure we use a window size that divides evenly by two, for compatibility with pooling
            window_size = max(2, window_size)
            safe_window_size = (window_size // 2) * 2
            self._window_size = safe_window_size

        # Pooling always halves windows
        self._pool_window_size = self._window_size // 2

        return self

    # .................................................................................................................


class SelfAttention(nn.Module):
    """
    Simplified implementation of the 'MultiScaleAttention' module originally found in SAMV2:
    https://github.com/facebookresearch/segment-anything-2/blob/57bc94b7391e47e5968004a0698f8bf793a544d1/sam2/modeling/backbones/hieradet.py#L37

    Most of the flexibility of the original implementation has been removed, leaving
    a standard self-attention module which is structured to act on image-like inputs.
    The original implementation had extra capabilities that have been moved to a separate
    'PooledSelfAttention' module.
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, features_per_token: int):

        # Inherit from parent
        super().__init__()

        self.qkv = nn.Linear(features_per_token, features_per_token * 3)
        self.proj = nn.Linear(features_per_token, features_per_token)

        # Store head count for re-shaping operations
        self.num_heads = num_heads
        self.features_per_head = features_per_token // num_heads

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Performs standard self-attention calculation on image-like tokens.
        Expects input shape: BxHxWxC
        -> B batch size, H height and W width, C channels/features per token

        Returns:
            encoded_imagelike_tokens (same shape as input)
        """

        # For convenience
        b, h, w, c = imagelike_bhwc.shape

        # QKV with shape BxHxWxC -to-> BxNx3xhxf -to-> Bxhx3xNxf, for use in attention calculation
        # -> B batch size (but includes windows)
        # -> N number of tokens, 3 for q/k/v, h num heads, C is features per token, f is features per head
        # -> q,k,v each have shape: BxhxNxf
        qkv = self.qkv(imagelike_bhwc)
        qkv = qkv.reshape(b, -1, 3, self.num_heads, self.features_per_head).transpose(1, 3)
        q, k, v = torch.unbind(qkv, 2)

        # Attention then restore original shape (and recombine heads)
        # -> Attention result has shape like: BxhxNxf
        # -> Reshaped result has shape matching input
        output = nn.functional.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(b, h, w, c)

        return self.proj(output)

    # .................................................................................................................


class PooledSelfAttention(nn.Module):
    """
    Helper variant of the attention module that always uses max pooling,
    and supports different feature sizes for input/outputs.

    The original implementation does not have this module, instead
    it uses flags to toggle the pooling functionality on/off at runtime from
    a single (shared) self-attention module called 'MultiScaleAttention', see:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/backbones/hieradet.py#L37
    """

    # .................................................................................................................

    def __init__(self, num_heads: int, input_features_per_token: int, output_features_per_token: int | None = None):

        # Inherit from parent
        super().__init__()

        # Assume feature doubling (to balance pooling) if output feature count isn't specified
        if output_features_per_token is None:
            output_features_per_token = input_features_per_token * 2

        self.q_pool = MaxPool2x2()
        self.qkv = nn.Linear(input_features_per_token, output_features_per_token * 3)
        self.proj = nn.Linear(output_features_per_token, output_features_per_token)

        # Store head count for re-shaping operations
        self.num_heads = num_heads
        self.features_per_head = output_features_per_token // num_heads

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Performs attention on image-like input tokens.
        Uses pooling on query tokens before attention calculation,
        so that the output is spatially downscaled by a factor of 2.
        The output also goes through a linear projection layer which
        can change (e.g. double) the output channel count.

        Returns:
            encoded_imagelike_tokens (shape: Bx(H/2)x(W/2)xC')
            -> Where B is same as input batch size
            -> C' is projected channel count (varies by model config)
            -> H/2 & W/2 are height & width after pooling
        """

        # For convenience
        b, h, w, _ = imagelike_bhwc.shape

        # QKV with shape BxHxWxC -to-> BxNx3xhxf
        # -> B batch size (but may include windows)
        # -> N number of tokens, 3 for q/k/v, h num heads, C is features per token, f is features per head
        # -> q,k,v each have shape: BxNxhxf
        qkv = self.qkv(imagelike_bhwc)
        qkv = qkv.reshape(b, -1, 3, self.num_heads, self.features_per_head)
        q, k, v = torch.unbind(qkv, 2)

        # Pooling on Q result only (reduces the number of q tokens by 4)
        q = self.q_pool(q.reshape(b, h, w, -1))
        pool_h, pool_w = q.shape[1:3]

        # Prepare for attention calculation (requires shape BxhxNxf)
        # -> q has number of tokens (N/4), while k & v have N tokens. B/h/f sizes are matching
        q = q.reshape(b, -1, self.num_heads, self.features_per_head).transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention then restore original shape (and recombine heads)
        # -> Attention result has shape like: Bxhx(N/4)xf
        # -> Reshaped/combined result has shape: Bx(H/2)x(W/2)xF
        output = nn.functional.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(b, pool_h, pool_w, -1)

        return self.proj(output)

    # .................................................................................................................


class MaxPool2x2(nn.Module):
    """
    Helper wrapper used to standardize 2x2 max-pooling layers
    Acts on image-like inputs with shapes: BxHxWxC, pooling the H & W dimensions.
    """

    # .................................................................................................................

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """Apply pooling to 'image like' input (output has height/width reduced by factor of 2)"""
        return self.pool(imagelike_bhwc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    # .................................................................................................................


class MLP2Layers_GELU(nn.Module):
    """Simplified implementation of the 2-layer MLP model used within the image encoder transformer blocks"""

    # .................................................................................................................

    def __init__(self, num_features: int, hidden_features_ratio=4.0):
        super().__init__()

        # Calculate number of hidden features
        num_hidden_features = int(round(hidden_features_ratio * num_features))
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_features),
            nn.GELU(),
            nn.Linear(num_hidden_features, num_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def window_partition(imagelike_bhwc: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """
    Reshape/partition image-like input into window tiles.
    For an input of shape: BxHxWxC, the output will have a shape of:
        (B*Nx*Ny)xSxSxC
        -> Where B is batch size
        -> Nx and Ny are the number of windows in x & y
        -> S is window size
        -> C channels per image token

    The input is padded to fit an integer number of windows in both x & y.

    Returns:
        window_tiles, num_windows_xy
    """

    # For convenience
    b, h, w, c = imagelike_bhwc.shape

    # Figure out how much padding is needed to have an integer number of windows in x & y
    h_pad_amt = (window_size - h % window_size) % window_size
    w_pad_amt = (window_size - w % window_size) % window_size
    if h_pad_amt > 0 or w_pad_amt > 0:
        imagelike_bhwc = nn.functional.pad(imagelike_bhwc, (0, 0, 0, w_pad_amt, 0, h_pad_amt))

    # Reshape input into windowed shape: (B*Nx*Ny)xSxSxC
    # -> B batch size, Nx/Ny number of windows, S window size and C image channels/features
    padded_h, padded_w = h + h_pad_amt, w + w_pad_amt
    num_y_windows = padded_h // window_size
    num_x_windows = padded_w // window_size
    window_tiles = imagelike_bhwc.view(b, num_y_windows, window_size, num_x_windows, window_size, c)
    window_tiles = window_tiles.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)

    return window_tiles, (num_x_windows, num_y_windows)


def window_unpartition(
    window_tiles: Tensor, original_shape: tuple[int, int, int, int], window_size: int, num_windows_xy: tuple[int, int]
) -> Tensor:
    """
    Reverses windowing partitioning step, including removal of padding.
    Expects window-tiles-like input with shape:
        (B*Nx*Ny)xSxSxC
        -> B batch size
        -> Nx/Ny is number of tiles in x/y
        -> S is window size
        -> C channels per image token

    Returns:
        imagelike_tokens_bhwc (shape matching the given 'original_shape')
    """

    # For convenience
    b, h, w, c = original_shape
    num_x_windows, num_y_windows = num_windows_xy

    # Reverse windowing steps
    padded_h = num_y_windows * window_size
    padded_w = num_x_windows * window_size
    window_tiles = window_tiles.view(b, num_y_windows, num_x_windows, window_size, window_size, c)
    imagelike_bhwc = window_tiles.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, padded_h, padded_w, c)

    # Truncate output to match original spatial size
    if padded_h > h or padded_w > w:
        imagelike_bhwc = imagelike_bhwc[:, :h, :w, :].contiguous()

    return imagelike_bhwc
