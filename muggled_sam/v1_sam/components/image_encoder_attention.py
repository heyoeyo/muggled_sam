#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch.nn as nn

from .decomposed_relative_position_encoder import DecomposedRelativePositionEncoder

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
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py#L166
    """

    # .................................................................................................................

    def __init__(self, features_per_token, num_heads, base_patch_grid_hw, mlp_ratio=4, norm_eps=1e-6) -> None:

        # Inherit from parent
        super().__init__()

        # Set up nn processing components
        self.norm_preattn = nn.LayerNorm(features_per_token, eps=norm_eps)
        self.attn = RelPosAttention(features_per_token, num_heads, base_patch_grid_hw)
        self.norm_premlp = nn.LayerNorm(features_per_token, eps=norm_eps)
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


class WindowedAttentionBlock(nn.Module):
    """
    This module is a variant of the global attention block, which
    adds support for windowing of image tokens prior to the attention
    calculation. Aside from the extra windowing steps, it is otherwise
    identical to the global attention block.

    The original windowed attention block functionality can be found here:
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py#L166
    """

    # .................................................................................................................

    def __init__(self, features_per_token, num_heads, base_window_size, mlp_ratio=4, norm_eps=1e-6) -> None:

        # Inherit from parent
        super().__init__()

        # Set up 'global' block to act on windowed tokens
        window_grid_hw = (base_window_size, base_window_size)
        self.global_attn = GlobalAttentionBlock(features_per_token, num_heads, window_grid_hw)

        # Store window size (+ backup of initial size, in case we change and need a reset!)
        self._init_window_size = base_window_size
        self._window_size = base_window_size

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """
        Computes self-attention, like global attention block, but
        with windowing used before attention step (and reversed after).
        This reduces the amount of computation compared to global attention
        Returns:
            encoded_image_tokens_bhwc (same shape as input)
        """

        # Break image tokens into separate smaller windows, do attention, then undo windowing!
        attn_windows, num_windows_xy = self._image_to_windows(imagelike_bhwc)
        output = self.global_attn(attn_windows)
        output = self._windows_to_image(output, imagelike_bhwc.shape, num_windows_xy)

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
        │     │  ──>  └─┘└─┘  ──>   ┌─┐┘
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

        The original code can be found here:
        https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py#L243

        Returns:
            window_tiles, num_windows_xy
        """

        # For convenience
        b, h, w, c = imagelike_bhwc.shape
        wsize = self._window_size

        # Figure out how much padding is needed to have an integer number of windows in x & y
        h_pad_amt = (wsize - (h % wsize)) % wsize
        w_pad_amt = (wsize - (w % wsize)) % wsize
        if h_pad_amt > 0 or w_pad_amt > 0:
            imagelike_bhwc = nn.functional.pad(imagelike_bhwc, (0, 0, 0, w_pad_amt, 0, h_pad_amt))

        # Figure out how many windows we have after padding (if any), for reshaping
        padded_h, padded_w = h + h_pad_amt, w + w_pad_amt
        num_y_windows = padded_h // wsize
        num_x_windows = padded_w // wsize

        # Reshape input into windowed shape from BxHxWxC -to-> (B*Ny*Nx)xSxSxC
        # -> B batch size, Ny/Nx number of windows, S window size and C image channels/features
        # -> This happens in 3 steps:
        #    1. Split H & W dimensions into (Ny, S) & (Nx, S) respectively
        #    2. Re-arrange dimension ordering so that Ny & Nx are 'beside' batch dimension,
        #       and last 3 channels are like HxWxC ordering (except H & W are now window Y & X size!)
        #    3. Merge/flatten the B, Ny & Nx dimensions together so result has an 'image-like' shape
        window_tiles = imagelike_bhwc.view(b, num_y_windows, wsize, num_x_windows, wsize, c)
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

        The original unpartition function can be found here:
        https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py#L267

        Returns:
            imagelike_tokens_bhwc (shape matching the given 'original_shape')
        """

        # For convenience
        b, h, w, c = original_shape_bhwc
        num_x_windows, num_y_windows = num_windows_xy
        wsize = self._window_size

        # Figure out the padded input size that was used prior to window partitioning
        padded_h = num_y_windows * wsize
        padded_w = num_x_windows * wsize

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


class RelPosAttention(nn.Module):
    """
    Modified implementation of the image encoder 'Attention' model/component used in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    This module implements a variation of the attention calculation (see "Attention is all you need"),
    which includes additive relative position encodings, following the equation:

        Attention = softmax(s*Q*K + P) * V
        -> Where s, Q, K & V are the scaling factor and query/key/value tensors from 'typical' attention
           and P is the (additive) position encoding term

    This implementation is mostly identical to the original implementation, but
    the code has been broken into more distinct steps, with more detailed comments to
    help explain each of the transformations.
    The position encodings have also been moved to a dedicated module.

    The original implementation can be found here:
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py#L185
    """

    # .................................................................................................................

    def __init__(self, features_per_token, num_heads, base_patch_grid_hw) -> None:

        # Inherit from parent
        super().__init__()

        # Store sizing info, mostly for use in reshaping ops during attention
        self.features_per_head = features_per_token // num_heads
        self.num_heads = num_heads
        self.scale = self.features_per_head**-0.5

        # Set up standard modules for performing attention
        self.qkv = nn.Linear(features_per_token, features_per_token * 3, bias=True)
        self.proj = nn.Linear(features_per_token, features_per_token)

        # Set up special additive relative position encoder, used inside attention calculation
        self.relpos = DecomposedRelativePositionEncoder(self.features_per_head, base_patch_grid_hw)

        # Set up softmax as dedicated 'module' so that we can hook into it for debugging/analysis!
        self.softmax = nn.Softmax(dim=-1)

    # .................................................................................................................

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        """Performs attention calculation on an image-like input. The output has the same shape as the input"""

        # For clarity
        B, H, W, _ = imagelike_bhwc.shape
        num_tokens = H * W
        qkv_size = 3

        # Produce Q, K, V tensors from input, shape goes from: (B,H,W,F) -to-> (B,H,W,3*F)
        qkv = self.qkv(imagelike_bhwc)

        # Break Q,K,V into 'multi-headed' shape, each tensor reshapes from (B,H,W,F) -to-> (B,N,heads,f)
        # -> Where big 'F' is features per token, small 'f' is features per head, 'N' is number of tokens
        # -> Two operations happening here: (1) Merge HxW into 'N' tokens, (2) Split F into (heads, f) shape
        # -> Actual code does all 3 tensors together, so shape is: (B,H,W,3*F) -to-> (B,N,3,heads,f)
        qkv = qkv.reshape(B, num_tokens, qkv_size, self.num_heads, self.features_per_head)

        # Now group 'heads' with batch dimension: (B,N,3,heads,f) -to-> (3,B,heads,N,f) -to-> (3,B*heads,N,f)
        # -> This means attention happens independently across heads, just like batches are handled independently
        qkv = qkv.permute(2, 0, 3, 1, 4)
        qkv = qkv.reshape(qkv_size, B * self.num_heads, num_tokens, self.features_per_head)

        # Break apart Q,K,V components for attention steps. Each tensor has shape: (B*heads, N, f)
        q, k, v = qkv.unbind(0)

        # Perform query-key 'scaled dot-product' for all heads
        # -> Matrix multiply between shapes: Nxf * fxN. With batching gives result shape: (B*heads, N, N)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative-position encodings (this is the only difference compared to 'normal' attention)
        attn = attn + self.relpos(q, (H, W))

        # Finish attention calculation
        # -> Matrix multiply between shapes: NxN * Nxf. With batching gives result shape: (B*heads, N, f)
        attn = self.softmax(attn) @ v

        # Convert back to original image-like shape and apply linear (per token) mapping
        imagelike_bhwc = attn.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        imagelike_bhwc = self.proj(imagelike_bhwc)

        return imagelike_bhwc

    # .................................................................................................................


class MLP2Layers_GELU(nn.Module):
    """
    Slightly modified implementation of the 2-layer MLP model used by the image-encoder attention blocks in SAM:
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/common.py#L13
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

    def forward(self, imagelike_bhwc: Tensor) -> Tensor:
        return self.layers(imagelike_bhwc)
