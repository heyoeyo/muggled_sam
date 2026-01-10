#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

import torch
import torch.nn as nn

from .components.image_encoder_attention import GlobalAttentionBlock, WindowedAttentionBlock

# For type hints
from torch import Tensor
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3ImageEncoder(nn.Module):
    """
    Simplified implementation of the image-encoder model/component described in:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    The code here is adapted from the original sam3 repo:
    https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L616

    Which is itself based on (the same model as SAMv1):
    https://arxiv.org/abs/2203.16527

    The main differences between SAMv1 and SAMv3 are:
        - Different 'size' (e.g. hyperparameters)
        - SAMv3 transformer blocks use (non-learned) RoPE position encodings at every layer
          (SAMv1 used complicated relative-position encodings)
        - SAMv3 has two different 'output projection' models, for use in different tasks,
          these are handled by a separate model rather than being built into the image encoder (like v1)

    """

    # Input image RGB normalization factors (for 0-255 pixel values)
    rgb_offset = [255.0 * v for v in (0.5, 0.5, 0.5)]
    rgb_stdev = [255.0 * v for v in (0.5, 0.5, 0.5)]

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 1024,
        num_stages: int = 4,
        num_blocks: int = 32,
        num_heads: int = 16,
        patch_size_px: int = 14,
        posenc_tile_hw: tuple[int, int] = (24, 24),
        window_size: int = 24,
        mlp_ratio: float = 4.625,
        norm_eps: float = 1e-5,
    ):

        # Inherit from parent
        super().__init__()

        # Create patch embedding to create patch tokens along with positional encoder
        self._patch_size_px = patch_size_px
        self.patch_embed = PatchEmbed(features_per_token, patch_size_px)
        self.posenc = TiledPositionEncoding(features_per_token, posenc_tile_hw)
        self.pre_layernorm = nn.LayerNorm(features_per_token, eps=1e-5)

        # Create transformer stages blocks (bulk of the model)
        num_blocks_per_stage = num_blocks // num_stages
        shared_args = (features_per_token, num_blocks_per_stage, num_heads, window_size, mlp_ratio)
        self.stages = nn.Sequential(
            *(TransformerStage(*shared_args, debug_stage_index=idx) for idx in range(num_stages))
        )

        # Store image scaling values
        self.register_buffer("mean_rgb", torch.tensor(self.rgb_offset).view(-1, 1, 1), persistent=False)
        self.register_buffer("stdev_scale_rgb", 1.0 / torch.tensor(self.rgb_stdev).view(-1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(self, image_tensor_bchw: Tensor) -> Tensor:
        """
        Encodes image tokens for further processing. The model expects the
        input to already be in the correct tensor format, with shape: BCHW
        A helper function, '.prepare_image(...)' exists for handling
        this when loading images through OpenCV.

        Returns:
            encoded_tokens_bchw (same shape as input)
        """

        # Convert to patch tokens with positional encoding
        image_tokens_bchw = self.patch_embed(image_tensor_bchw)
        image_tokens_bchw = self.posenc(image_tokens_bchw)

        # Run transformer layers
        tokens_bhwc = image_tokens_bchw.permute(0, 2, 3, 1)
        tokens_bhwc = self.pre_layernorm(tokens_bhwc)
        tokens_bhwc = self.stages(tokens_bhwc)

        return tokens_bhwc.permute(0, 3, 1, 2)

    # .................................................................................................................

    def prepare_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> Tensor:
        """
        Helper function used to prepare an (OpenCV) image for processing by the image encoder.
        Note that steps are slightly different compared to how the original model handles images.
        This function does the following::
            1. Convert to RGB (assumes OpenCV image, which is uint8 in BGR order)
            2. Convert to model data type (e.g. float32 or float16)
            3. Resize to target HW
            4. Normalize RGB

        By comparison, the original implementation resizes in uint8 format and relies on torchvision.
        This can lead to significant numerical differences compared (to the original model) when processing images!

        Consider using the '.prepare_image_like_original' function if numerical consistency is needed

        Returns:
            image_tensor_bchw
        """

        # Fill in default sizing if not given
        if max_side_length is None:
            max_side_length = 1008

        # Figure out scaling factor to get target side length
        img_h, img_w = image_bgr.shape[0:2]
        largest_side = max(img_h, img_w)
        scale_factor = max_side_length / largest_side

        # Force sizing to multiples of a specific tiling size
        tiling_size = self.get_image_tiling_size_constraint()
        if use_square_sizing:
            num_tiles = max(1, round((largest_side * scale_factor) / tiling_size))
            scaled_side = int(num_tiles * tiling_size)
            scaled_h = scaled_w = scaled_side
        else:
            scaled_h = int(max(1, round((img_h * scale_factor) / tiling_size))) * tiling_size
            scaled_w = int(max(1, round((img_w * scale_factor) / tiling_size))) * tiling_size

        # Scale RGB image to correct size and re-order from HWC to BCHW (with batch of 1)
        device, dtype = self.mean_rgb.device, self.mean_rgb.dtype
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor_bchw = torch.tensor(image_rgb, device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        image_tensor_bchw = torch.nn.functional.interpolate(
            image_tensor_bchw,
            size=(scaled_h, scaled_w),
            align_corners=False,
            antialias=True,
            mode="bilinear",
        )

        # Perform mean/scale normalization
        image_tensor_bchw = (image_tensor_bchw - self.mean_rgb) * self.stdev_scale_rgb
        return image_tensor_bchw

    # .................................................................................................................

    def prepare_image_like_original(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> Tensor:
        """
        This function is included to help with debugging/comparison with the original SAM3 implementation.
        Unlike the normal 'prepare_image' function, this implementation
        exactly matches the behavior of the original model (numerically).

        The steps are as follows:
            1. Convert to uint8
            2. Resize to target HW
            3. Convert to float32
            4. Normalize RGB
            5. Convert to model data type (e.g. bfloat16)
        See: https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/sam3_image_processor.py#L21

        For various reasons, this seems like a 'bad' way to handle inputs however, using alternative
        image preparation can result in dramatic numerical differences in the image encoder outputs
        (on the order of +/-100 errors in feature values).

        Returns:
            image_tensor_bchw
        """
        # Fill in default sizing if not given
        if max_side_length is None:
            max_side_length = 1008

        # Figure out scaling factor to get target side length
        img_h, img_w = image_bgr.shape[0:2]
        largest_side = max(img_h, img_w)
        scale_factor = max_side_length / largest_side

        # Force sizing to multiples of a specific tiling size
        tiling_size = self.get_image_tiling_size_constraint()
        if use_square_sizing:
            num_tiles = max(1, round((largest_side * scale_factor) / tiling_size))
            scaled_side = int(num_tiles * tiling_size)
            scaled_h = scaled_w = scaled_side
        else:
            scaled_h = int(max(1, round((img_h * scale_factor) / tiling_size))) * tiling_size
            scaled_w = int(max(1, round((img_w * scale_factor) / tiling_size))) * tiling_size

        # Convert to RGB format in BCHW ordering and scale to target size
        device, dtype = self.mean_rgb.device, self.mean_rgb.dtype
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor_bchw = torch.tensor(image_rgb, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
        image_tensor_bchw = torch.nn.functional.interpolate(
            image_tensor_bchw,
            size=(scaled_h, scaled_w),
            align_corners=False,
            antialias=True,
            mode="bilinear",
        )

        # Perform mean/scale normalization
        image_tensor_bchw = image_tensor_bchw.to(dtype=torch.float32, device=device) * (1.0 / 255.0)
        image_tensor_bchw = (image_tensor_bchw - 0.5) / 0.5

        return image_tensor_bchw.to(dtype=dtype)

    # .................................................................................................................

    def get_image_tiling_size_constraint(self) -> int:
        """
        This function is mostly used for compatibility with the V2 model, which
        has a more complex requirement on input image sizing. For the V3 model,
        the only constraint is that images must be sized to an integer number
        of the patch sizing.
        """

        return self._patch_size_px

    # .................................................................................................................

    def set_window_sizes(self, window_size_per_stage: list[int | None]):
        """
        Allows for updating per-stage window sizing. This is primarily
        meant for experimental purposes. The window sizing should not
        need to be altered under normal use of the model.

        Window sizes should be provided as a list of integers or None,
        where None indicates that the original window size config should
        be used. For example:
            window_size_per_stage = [2, 4, None, 16]
        """

        # Force window sizing to be at least 4 entries long, to match stages
        num_sizes = len(window_size_per_stage)
        num_stages = len(self.stages)
        if num_sizes < num_stages:
            window_size_per_stage = [*window_size_per_stage].extend([None] * (num_stages - num_sizes))

        # Have each stage update it's blocks
        for stage, winsize in zip(self.stages, window_size_per_stage):
            stage.set_window_size(winsize)

        return self

    # .................................................................................................................


class TransformerStage(nn.Module):
    """
    Helper module which represents a single 'stage' of the SAM image encoder
    Each stage consists of multiple windowed attention blocks, which end
    with a single 'global' (i.e. normal) attention block
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 1024,
        num_blocks: int = 8,
        num_heads: int = 16,
        window_size: int = 24,
        mlp_ratio: float = 4.625,
        norm_eps: float = 1e-5,
        debug_stage_index: int = -1,
    ):

        # Init as parent sequential
        super().__init__()

        # Store stage indexing to help with debugging
        self._stage_idx = debug_stage_index

        # Set up shared config between global/window blocks
        rope_hw = (window_size, window_size)
        shared_kwargs = {
            "features_per_token": features_per_token,
            "num_heads": num_heads,
            "rope_hw": rope_hw,
            "mlp_ratio": mlp_ratio,
            "norm_eps": norm_eps,
            "parent_stage_index": debug_stage_index,
        }

        # Create sequence of windowed attention blocks
        num_win_blocks = max(0, num_blocks - 1)
        self.windowed_attn_blocks = nn.Sequential(
            *(
                WindowedAttentionBlock(**shared_kwargs, window_size=window_size, debug_sequence_index=idx)
                for idx in range(num_win_blocks)
            )
        )

        # Create single global attention as final output 'layer' of a single stage
        self.global_attn_block = GlobalAttentionBlock(**shared_kwargs)

    # .................................................................................................................

    def forward(self, patch_tokens_bhwc: Tensor):
        """Encodes patch tokens using sequence of windowed-attention blocks followed by global attention"""
        patch_tokens_bhwc = self.windowed_attn_blocks(patch_tokens_bhwc)
        return self.global_attn_block(patch_tokens_bhwc)

    # .................................................................................................................

    def set_window_size(self, window_size: int | None = None):
        """
        Update all windowed-attention blocks to use a new window size.
        Set size to None to reset to initial configuration
        """

        for block in self.windowed_attn_blocks:
            block.set_window_size(window_size)

        return self

    # .................................................................................................................


class PatchEmbed(nn.Module):
    """
    Patch embedding model used to convert full-sized RGB images into
    a much smaller grid of image 'tokens' for processing by a transformer model
    Like SAMv1 but unlike SAMv2, the patches do *not* overlap.
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 1024,
        patch_size_px: int = 14,
        num_input_channels: int = 3,
    ):

        # Inherit from parent
        super().__init__()

        # Both grouping + linear transformation is handled with a single strided convolution
        self.proj = nn.Conv2d(
            num_input_channels,
            features_per_token,
            kernel_size=patch_size_px,
            stride=patch_size_px,
            bias=False,
        )

    # .................................................................................................................

    def forward(self, image_tensor_bchw: Tensor) -> Tensor:
        """
        Collapses input image into smaller 'tokens'
        Shape goes from: BxC'xH'xW' -> BxCxHxW
            -> Where B is batch size
            -> C' is image channels (i.e. 3 for RGB image)
            -> H', W' are input image size (both 1008 by default)
            -> H, W are output patch token size (both 72 by default)
            -> C is channels per token (1024 by default)
        Returns:
            patch_tokens_bchw
        """
        return self.proj(image_tensor_bchw)

    # .................................................................................................................


class TiledPositionEncoding(nn.Module):
    """
    Simple model used to add position encoding to image tokens. This
    model does not exist in the original code base, where the position
    encoding is instead handled in-place at runtime. See:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/vitdet.py#L824-L831

    This position encoding is a bit strange! It's tiled from a 24x24
    embedding (with 1024 features), with no other 'base' component (like in SAMv2).
    This means that there is no 'global' positioning information given to
    the patch tokens prior to running through the image transformer!
    (though the transformer blocks add their own position encodings)

    Note that this implementation is structured to match SAMv1 & v2.

    See the original tiling logic:
    https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/vitdet.py#L175
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int = 1024,
        tile_size_hw: tuple[int, int] = (24, 24),
    ):

        # Inherit from parent
        super().__init__()

        # Storage for fixed-size learned position embedding
        tile_h, tile_w = tile_size_hw
        self.tile_embedding_bhwc = nn.Parameter(torch.zeros(1, features_per_token, tile_h, tile_w))

        # Allocate storage for caching positional encoding, so we don't keep re-calculating them
        self.register_buffer("cached_encoding_bchw", torch.empty((1, features_per_token, 1, 1)), persistent=False)

    # .................................................................................................................

    def extra_repr(self) -> str:
        _, features_per_token, emb_h, emb_w = self.cached_encoding_bchw.shape
        return f"features_per_token={features_per_token}, tile_size_hw=({emb_h}, {emb_w})"

    # .................................................................................................................

    def forward(self, tokens_bchw: Tensor) -> Tensor:
        _, _, in_h, in_w = tokens_bchw.shape
        return tokens_bchw + self._scale_to_patch_grid((in_h, in_w))

    # .................................................................................................................

    def _scale_to_patch_grid(self, patch_grid_hw: tuple[int, int]) -> Tensor:
        """
        Helper used to make sure the position embeddings are scaled
        to match the input patch sizing. In SAMv3, this is done purely
        through tiling rather than interpolation.
        Returns:
            scaled_position_embed_bhwc
        """

        # Only compute re-sized embedding if we don't already have it cached
        grid_h, grid_w = patch_grid_hw
        _, _, cache_h, cache_w = self.cached_encoding_bchw.shape
        if grid_h != cache_h or grid_w != cache_w:

            # Figure out how many x/y copies of the tile are needed to match patch grid sizing
            _, _, tile_h, tile_w = self.tile_embedding_bhwc.shape
            is_int_y_tiles = (grid_h % tile_h) == 0
            is_int_x_tiles = (grid_w % tile_w) == 0
            num_y_tiles = grid_h // tile_h if is_int_y_tiles else 1 + grid_h // tile_h
            num_x_tiles = grid_w // tile_w if is_int_x_tiles else 1 + grid_w // tile_w

            # Repeat/tile window embedding to match the patch grid shape
            # -> Need to truncate to exactly match the grid shape at the end!
            resized_embedding_bchw = self.tile_embedding_bhwc.tile(1, 1, num_y_tiles, num_x_tiles)
            if not (is_int_x_tiles and is_int_y_tiles):
                resized_embedding_bchw = resized_embedding_bchw[:, :, :grid_h, :grid_w]

            # Store position encoding for re-use (input size rarely changes after first compute)
            self.cached_encoding_bchw = resized_embedding_bchw

        return self.cached_encoding_bchw

    # .................................................................................................................
