#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

import torch
import torch.nn as nn

from .components.shared import LayerNorm2d
from .components.image_encoder_attention import GlobalAttentionBlock, WindowedAttentionBlock

# For type hints
from torch import Tensor
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV1ImageEncoder(nn.Module):
    """
    Simplified implementation of the 'image-encoder' model/component described in:
        "Segment Anything"
        By: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
            Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
        @ https://arxiv.org/abs/2304.02643

    The code here is adapted from the original segment-anything repo:
    https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/image_encoder.py

    Which itself seems to be adapted from the ViTDet backbone:
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py

    This is a vision transformer model. It is applied to images in order to
    generate embeddings representing 'patches' of a given input image, which
    are then passed along to a mask decoder model to create segmentation masks.

    Structurally, this model consists of 4-stages. Each stage is a sequence
    of windowed-attention blocks, ending with a single 'global attention'
    (i.e. the standard non-windowed attention implementation).

    Additionally, there is a convolution-based projection sequence on the
    output of the model which is used to reduce the embedding dimension
    to a standard channel count, regardless of the model size.
    """

    # Input image RGB normalization factors (for 0-255 pixel values)
    rgb_offset = [255.0 * v for v in (0.485, 0.456, 0.406)]
    rgb_stdev = [255.0 * v for v in (0.229, 0.224, 0.225)]

    # .................................................................................................................

    def __init__(
        self,
        features_per_token=768,
        num_blocks=12,
        num_heads=12,
        window_size=14,
        base_patch_grid_hw=(64, 64),
        output_channels=256,
        patch_size_px=16,
        num_stages=4,
    ):

        # Inherit from parent
        super().__init__()

        # Create patch embedding to create patch tokens along with positional encoder
        self._patch_size_px = patch_size_px
        self.patch_embed = PatchEmbed(features_per_token, patch_size_px)
        self.posenc = PositionEncoding(features_per_token, base_patch_grid_hw)

        # Create transformer stages blocks (bulk of the model)
        num_blocks_per_stage = num_blocks // num_stages
        stages_list = []
        for _ in range(num_stages):
            new_stage = TransformerStage(
                features_per_token, num_blocks_per_stage, num_heads, window_size, base_patch_grid_hw
            )
            stages_list.append(new_stage)
        self.stages = nn.Sequential(*stages_list)

        # Create layers used to project to target number of output channels
        self.output_projection = nn.Sequential(
            nn.Conv2d(features_per_token, output_channels, kernel_size=1, bias=False),
            LayerNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(output_channels),
        )

        # Store image scaling values
        self.register_buffer("mean_rgb", torch.tensor(self.rgb_offset).view(-1, 1, 1), persistent=False)
        self.register_buffer("stdev_scale_rgb", 1.0 / torch.tensor(self.rgb_stdev).view(-1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(self, image_tensor_bchw: Tensor) -> Tensor:
        """
        Outputs a batch of feature tensors of shape: (batch, 256, H, W), where H & W are both 64 (= 1024 / 16)
        - Result of applying repeated transformer blocks on 16x16 patches of input image + final projection layer

        The output is meant to be fed into a mask decoder model (along with prompts) to generate segmentation masks

        Returns:
            encoded_image_patches_tensor
        """

        # Convert to patch tokens with positional encoding
        patch_tokens_bhwc = self.patch_embed(image_tensor_bchw)
        patch_tokens_bhwc = self.posenc(patch_tokens_bhwc)

        # Run transformer layers
        patch_tokens_bhwc = self.stages(patch_tokens_bhwc)

        # Run final projection to standard channel size and convert to bchw shape
        return self.output_projection(patch_tokens_bhwc.permute(0, 3, 1, 2))

    # .................................................................................................................

    def prepare_image(
        self,
        image_bgr: ndarray,
        max_side_length=1024,
        use_square_sizing=True,
        pad_to_square=False,
    ) -> Tensor:

        # Figure out scaling factor to get target side length
        img_h, img_w = image_bgr.shape[0:2]
        largest_side = max(img_h, img_w)
        scale_factor = max_side_length / largest_side

        # Force sizing to multiples of a specific tiling size
        tiling_size = self.get_image_tiling_size_constraint()
        if use_square_sizing:
            scaled_side = int(np.ceil(largest_side * scale_factor / tiling_size)) * tiling_size
            scaled_h = scaled_w = scaled_side
        else:
            scaled_h = int(np.ceil(img_h * scale_factor / tiling_size)) * tiling_size
            scaled_w = int(np.ceil(img_w * scale_factor / tiling_size)) * tiling_size

        # Scale RGB image to correct size and re-order from HWC to BCHW (with batch of 1)
        device, dtype = self.mean_rgb.device, self.mean_rgb.dtype
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor_chw = torch.tensor(np.transpose(image_rgb, (2, 0, 1)), device=device, dtype=dtype)
        image_tensor_bchw = torch.nn.functional.interpolate(
            image_tensor_chw.unsqueeze(0),
            size=(scaled_h, scaled_w),
            align_corners=False,
            antialias=True,
            mode="bilinear",
        )

        # Perform mean/scale normalization
        image_tensor_bchw = (image_tensor_bchw - self.mean_rgb) * self.stdev_scale_rgb

        # The original SAM implementation padded the short side of the image to form a square
        # -> This results in more processing and isn't required in this implementation!
        if pad_to_square:
            pad_left, pad_top = 0, 0
            pad_bottom = max_side_length - scaled_h
            pad_right = max_side_length - scaled_w
            image_tensor_bchw = nn.functional.pad(image_tensor_bchw, (pad_left, pad_right, pad_top, pad_bottom))

        return image_tensor_bchw

    # .................................................................................................................

    def get_image_tiling_size_constraint(self) -> int:
        """
        This function is mostly used for compatibility with the V2 model, which
        has a more complex requirement on input image sizing. For the V1 model,
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
    Each stage consists of multiple windowed attention blocks, which then end
    with a single 'global' (i.e. normal) attention block
    """

    # .................................................................................................................

    def __init__(self, features_per_token, num_blocks, num_heads, base_window_size, base_patch_grid_hw):

        # Init as parent sequential
        super().__init__()

        # Store base window size to use as default at runtime
        self.base_window_size = base_window_size

        # Create sequence of windowed attention blocks
        num_win_blocks = max(0, num_blocks - 1)
        self.windowed_attn_blocks = nn.Sequential(
            *(WindowedAttentionBlock(features_per_token, num_heads, base_window_size) for _ in range(num_win_blocks))
        )

        # Create single global attention as final output 'layer' of a single stage
        self.global_attn_block = GlobalAttentionBlock(features_per_token, num_heads, base_patch_grid_hw)

    # .................................................................................................................

    def forward(self, patch_tokens: Tensor):
        """Encodes patch tokens using sequence of windowed-attention blocks followed by global attention"""
        patch_tokens = self.windowed_attn_blocks(patch_tokens)
        return self.global_attn_block(patch_tokens)

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
    """

    # .................................................................................................................

    def __init__(self, features_per_token, patch_size_px=16, num_input_channels=3):

        # Inherit from parent
        super().__init__()

        # Both grouping + linear transformation is handled with a single strided convolution step!
        self.proj = nn.Conv2d(
            num_input_channels,
            features_per_token,
            kernel_size=patch_size_px,
            stride=patch_size_px,
        )

    # .................................................................................................................

    def forward(self, image_tensor_bchw: Tensor) -> Tensor:
        """
        Reshapes & projects image tensor: BxCxHxW -> BxHxWxF
            -> Where B is batch size
            -> C is image channels (i.e. 3 for RGB image)
            -> F is features per token
            -> H, W are the height & width of the image
        """

        patch_tokens = self.proj(image_tensor_bchw)
        return patch_tokens.permute(0, 2, 3, 1)

    # .................................................................................................................


class PositionEncoding(nn.Module):

    # .................................................................................................................

    def __init__(self, features_per_token, base_patch_grid_hw):

        # Inherit from parent
        super().__init__()

        # Storage for fixed-size learned position embedding
        grid_h, grid_w = base_patch_grid_hw
        self.base_embedding = nn.Parameter(torch.zeros(1, grid_h, grid_w, features_per_token))

    # .................................................................................................................

    def extra_repr(self) -> str:
        _, grid_h, grid_w, features_per_token = self.base_embedding.shape
        return f"features_per_token={features_per_token}, base_grid_hw=({grid_h}, {grid_w})"

    # .................................................................................................................

    def forward(self, patch_tokens_bhwc: Tensor) -> Tensor:

        # Resize embedding if needed
        _, in_h, in_w, _ = patch_tokens_bhwc.shape
        _, embed_h, embed_w, _ = self.base_embedding.shape
        if (in_h != embed_h) or (in_w != embed_w):
            return patch_tokens_bhwc + self._scale_to_patch_grid((in_h, in_w))

        return patch_tokens_bhwc + self.base_embedding

    # .................................................................................................................

    def _scale_to_patch_grid(self, patch_grid_hw: tuple[int, int]) -> Tensor:
        """
        Helper used to make sure the position embeddings are scaled
        to match the input patch sizing, by linear interpolation.
        """

        # Get original shape/data type, so we can restore this on the output
        _, base_h, base_w, C = self.base_embedding.shape
        orig_dtype = self.base_embedding.dtype

        # Force embedding to float32 for computing interpolation
        # -> If we don't do this, we could get bad results/errors on lower precision dtypes
        pos_embed_f32 = self.base_embedding.float()

        # Convert to shape needed by interpolation function and then convert back
        pos_embed_bchw = pos_embed_f32.permute(0, 3, 1, 2)
        pos_embed_bchw = nn.functional.interpolate(
            pos_embed_bchw,
            size=patch_grid_hw,
            mode="bilinear",
            antialias=True,
        )
        resized_embedding_bhwc = pos_embed_bchw.permute(0, 2, 3, 1)

        # Convert back to channels-last format
        return resized_embedding_bhwc.to(orig_dtype)

    # .................................................................................................................
