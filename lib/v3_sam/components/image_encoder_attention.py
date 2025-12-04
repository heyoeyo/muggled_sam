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


class GlobalAttentionBlock(nn.Module):
    """
    A single transformer 'global' block layer (i.e. no windowing)

    The original code does not have a dedicated 'global' attention block, instead
    both global & windowed attention use the same code, with branches to enable/disable
    the windowing. The two blocks have been separated here to better indicate the model structure.

    The original attention block code can be found here:
    https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L518
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int,
        num_heads: int,
        rope_hw: tuple[int, int] = (24, 24),
        mlp_ratio: float = 4.625,
        norm_eps: float = 1e-5,
        parent_stage_index: int = -1,
    ):

        # Inherit from parent
        super().__init__()

        # For use in debugging
        self._parent_stage_idx = parent_stage_index

        # Set up nn processing components
        self.norm_preattn = nn.LayerNorm(features_per_token, eps=norm_eps)
        self.attn = RoPEAttentionBHWC(features_per_token, num_heads, rope_hw)
        self.norm_premlp = nn.LayerNorm(features_per_token, eps=norm_eps)
        self.mlp = MLP2Layers_GELU(features_per_token, mlp_ratio)

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Computes self-attention followed by MLP with pre-layernorms & layer scaling.
        See "An Image is Worth 16x16 Words" @ https://arxiv.org/abs/2010.11929
        Returns:
            encoded_tokens_bhwc (same shape as input)
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


class WindowedAttentionBlock(nn.Module):
    """
    This module is a variant of the global attention block, which
    adds support for windowing of image tokens prior to the attention
    calculation.
    Aside from introducing extra windowing steps, this block is
    identical to the global attention block.

    The original windowing code can be found here:
    https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L601-L608
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int,
        num_heads: int,
        rope_hw: tuple[int, int] = (24, 24),
        window_size: int = 24,
        mlp_ratio: float = 4.625,
        norm_eps: float = 1e-5,
        parent_stage_index: int = -1,
        debug_sequence_index: int = -1,
    ):

        # Inherit from parent
        super().__init__()

        # For use in debugging
        self._parent_stage_idx = parent_stage_index
        self._seq_idx = debug_sequence_index

        # Set up nn processing components
        self.norm_preattn = nn.LayerNorm(features_per_token, eps=norm_eps)
        self.attn = RoPEAttentionBHWC(features_per_token, num_heads, rope_hw)
        self.norm_premlp = nn.LayerNorm(features_per_token, eps=norm_eps)
        self.mlp = MLP2Layers_GELU(features_per_token, mlp_ratio)

        # Store window size (+ backup of initial size, in case we change and need a reset!)
        self._init_window_size = window_size
        self._window_size = window_size

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Computes self-attention, like global attention block, but
        with windowing used before attention step (and reversed after).
        This reduces the amount of computation compared to global attention
        Returns:
            encoded_tokens_bhwc (same shape as input)
        """

        # Attention with windowing (pre-norm & residual are still done 'globally')
        attn = self.norm_preattn(imagelike_bhwc)
        attn_windows, num_windows_xy = self._image_to_windows(attn)
        attn = self.attn(attn_windows)
        attn = self._windows_to_image(attn, imagelike_bhwc.shape, num_windows_xy)
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

    def _image_to_windows(self, imagelike_bhwc: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """
        Reshape/partition image-like input into a stack of window tiles.

        ┌─────┐       ┌─┐┌─┐         ┌─┐
        |     │  ──>  └─┘└─┘  ──>   ┌─┐┘
        │     │       ┌─┐┌─┐       ┌─┐┘
        └─────┘       └─┘└─┘      ┌─┐┘
                                  └─┘
        (graphical example where input breaks into 2x2 windows)

        For an input of shape: BxHxWxC, the output will have a shape of:
            (B*Ny*Nx)xSxSxC
            -> Where B is batch size
            -> Nx and Ny are the number of windows in x & y
            -> S is window size (shared for x & y)
            -> C channels per image token

        The input is padded with zeros to fit an integer number of windows in both x & y.

        This implementation is slightly modified compared to the original, only to make
        the code easier to follow. The changes do not affect the overall behavior.

        The original image-to-windows code can be found here:
        https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L93

        Returns:
            window_tiles, num_windows_xy
        """

        # For convenience
        b, h, w, c = imagelike_bhwc.shape
        winsize = self._window_size

        # Figure out how much padding is needed to have an integer number of windows in x & y
        h_pad_amt = (winsize - (h % winsize)) % winsize
        w_pad_amt = (winsize - (w % winsize)) % winsize
        if h_pad_amt > 0 or w_pad_amt > 0:
            imagelike_bhwc = nn.functional.pad(imagelike_bhwc, (0, 0, 0, w_pad_amt, 0, h_pad_amt))

        # Figure out how many windows we have after padding (if any), for reshaping
        padded_h, padded_w = h + h_pad_amt, w + w_pad_amt
        num_y_windows = padded_h // winsize
        num_x_windows = padded_w // winsize

        # Reshape input into windowed shape from BxHxWxC -to-> (B*Ny*Nx)xSxSxC
        # -> B batch size, Ny/Nx number of windows, S window size and C image channels/features
        # -> This happens in 3 steps:
        #    1. Split H & W dimensions into (Ny, S) & (Nx, S) respectively
        #    2. Re-arrange dimension ordering so that Ny & Nx are 'beside' batch dimension,
        #       and last 3 channels are like HxWxC ordering (except H & W are now window Y & X size!)
        #    3. Merge/flatten the B, Ny & Nx dimensions together so result has an 'image-like' shape
        window_tiles = imagelike_bhwc.view(b, num_y_windows, winsize, num_x_windows, winsize, c)
        window_tiles = window_tiles.permute(0, 1, 3, 2, 4, 5)
        window_tiles = window_tiles.flatten(0, 2)

        return window_tiles, (num_x_windows, num_y_windows)

    # .................................................................................................................

    def _windows_to_image(
        self,
        window_tiles: Tensor,
        original_shape_bhwc: tuple[int, int, int, int],
        num_windows_xy: tuple[int, int],
    ) -> Tensor:
        """
        Reverses windowing partitioning step, including removal of padding.
        Expects window-tiles-like input with shape:
            (B*Ny*Nx)xSxSxC
            -> B batch size
            -> Nx and Ny are the number of windows in x & y
            -> S is window size (shared for x & y)
            -> C channels per image token

        Compared to the original, this implementation is modified slightly to
        account for changes to the window partitioning function and to make the
        code easier to follow.

        The original windows-to-image function can be found here:
        https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L116

        Returns:
            imagelike_tokens_bhwc (shape matching the given 'original_shape')
        """

        # For convenience
        b, h, w, c = original_shape_bhwc
        num_x_windows, num_y_windows = num_windows_xy
        winsize = self._window_size

        # Figure out the padded input size that was used prior to window partitioning
        padded_h = num_y_windows * winsize
        padded_w = num_x_windows * winsize

        # Reverse the 3 window partitioning steps
        # 1. Un-merge B, Ny, Nx dimensions
        # 2. Un-rearrange dimensions from (B, Ny, Nx, S, S, C) -to-> (B, Ny, S, Nx, S, C) ordering
        # 3. Merge (Ny, S) into H and (Nx, S) into W to get final BxHxWxC shape
        window_tiles = window_tiles.unflatten(0, (b, num_y_windows, num_x_windows))
        window_tiles = window_tiles.permute(0, 1, 3, 2, 4, 5).contiguous()
        imagelike_bhwc = window_tiles.view(b, padded_h, padded_w, c)

        # Reverse the padding step, if needed
        if padded_h > h or padded_w > w:
            imagelike_bhwc = imagelike_bhwc[:, :h, :w, :].contiguous()

        return imagelike_bhwc

    # .................................................................................................................


class RoPEAttentionBHWC(nn.Module):
    """
    Modified implementation of the image encoder 'Attention' model/component used in:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    This is a mostly straight-forward implementation of multi-headed attention,
    except it includes (computed, not learned) RoPE position embeddings.
    Note that this implementation differs (very slightly) from the RoPE attention
    used within the memory-image-fusion model!

    The original code can be found here:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/vitdet.py#L339

    For more information about RoPE, the original paper seems to be the following:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    By: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
    @ https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        features_per_token: int = 1024,
        num_heads: int = 16,
        rope_hw: tuple[int, int] = (24, 24),
        rope_theta: float = 10000.0,
        use_complex_numbers: bool = True,
    ):
        # Inherit from parent
        super().__init__()

        # Store config for re-use
        self.num_heads = num_heads
        self.features_per_head = features_per_token // num_heads

        # Set up q/k/v attention mapping
        num_qkv_features = 3 * features_per_token
        self.qkv = nn.Linear(features_per_token, num_qkv_features, bias=True)

        # Set up rotary position encoder (no learned components!)
        PosEncoder = RPEComplex if use_complex_numbers else RPEMatrix
        self.rope_encoder = PosEncoder(self.features_per_head, rope_theta, rope_hw)

        # Output layer used to mix channel information (this seems unnescessary?)
        self.proj = nn.Linear(features_per_token, features_per_token)

    def forward(self, tokens_bhwc: Tensor) -> Tensor:

        # For convenience
        b, h, w, c = tokens_bhwc.shape
        num_tokens = h * w

        # Compute QKV tokens, has shape: BxHxWx3C,
        # then switch to 'rows of tokens' format with per-head features:
        # shape: BxNx3xDxf (N num tokens, D num heads, f features per head)
        qkv = self.qkv(tokens_bhwc)
        qkv = qkv.reshape(b, num_tokens, 3, self.num_heads, self.features_per_head)

        # Rearrange to 3xBxDxNxf & split into 3 q/k/v tensors (each with shape BxDxNxf)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # Apply position encoding & compute attention
        # -> Result has same shape as q/k: BxDxNxf
        q, k = self.rope_encoder(q, k, (h, w))
        attn = nn.functional.scaled_dot_product_attention(q, k, v)

        # Re-arrange output result to shape: BxDxNxf -> BxDxHxWxf -> BxHxWxDxf -> BxHxWxC
        attn = attn.view(b, self.num_heads, h, w, self.features_per_head)
        attn = attn.permute(0, 2, 3, 1, 4).reshape(b, h, w, c)

        # Run final 'projection' (though this doesn't change channel count!) shape is same as input: BxHxWxC
        return self.proj(attn)


class MLP2Layers_GELU(nn.Module):
    """
    Slightly modified implementation of the 2-layer MLP model used by the image-encoder attention blocks in SAM:
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/common.py#L13
    """

    # .................................................................................................................

    def __init__(self, num_features: int, hidden_features_ratio: float = 4.625):
        super().__init__()

        # Calculate number of hidden features
        num_hidden_features = int(round(hidden_features_ratio * num_features))
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_features),
            nn.GELU(),
            nn.Linear(num_hidden_features, num_features),
        )

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        return self.layers(imagelike_bhwc)
