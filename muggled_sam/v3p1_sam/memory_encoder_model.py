#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.memory_encoder_components import MaskDownsampler, ConvNeXtBlock
from .components.shared import Conv1x1Layer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3p1MemoryEncoder(nn.Module):
    """
    Slightly modified implementation of the 'SimpleMaskEncoder' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    The original code can be found here:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/memory.py#L166

    This has been modified slightly for v3.1. The difference being support for 'multiplex' inputs,
    which is a special kind of batching used to speed up processing by encoding multiple
    masks into a single set of memory tokens.

    This implementation has only minor changes compared to the original model, primarily
    in the form of renamed layers. It also features a slightly different call signature
    due to using a more heavily modified mask downsampler and doesn't compute positional
    encodings for the memory encodings it outputs.
    (this has been moved to another model, where the encoding are used)
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token: int = 256,
        features_per_memory_token: int = 256,
        num_downsample_layers: int = 4,
        num_mixer_layers: int = 2,
        multiplex_channels: int = 16,
    ):

        # Inherit from parent
        super().__init__()

        # Define layers used to pre-process mask & image data prior to fusing
        self.mask_downsampler = MaskDownsampler(features_per_image_token, num_downsample_layers, multiplex_channels)
        self.image_proj = Conv1x1Layer(features_per_image_token)

        # Define layers used to post-process fused mask + image result
        self.channel_mixer = nn.Sequential(*(ConvNeXtBlock(features_per_image_token) for _ in range(num_mixer_layers)))

        # Create special handler for when no object is present
        self.missing_obj_encoder = NoObjectEncoder(features_per_memory_token, multiplex_channels)

    # .................................................................................................................

    def forward(
        self,
        lowres_image_encoding_bchw: Tensor,
        mask_prediction_1mhw: Tensor,
        object_score_m: Tensor | None,
        is_prompt_encoding: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Takes the lowest-resolution image encoding and combines it
        with a mask prediction to form a 'fused' encoding, which can act
        a bit like a prompt to re-segment an object on future frames.
        (without requiring any other prompts)

        The input image encoding is expected to have shape BxCxHxW, the mask
        prediction should have a shape of 1xMx(4H)x(4W), where M
        is the 'multiplex' dimension introduced with SAMv3.1.
        Up to 16 (by default) masks can be provided in the 'M' dimensions
        to be processed in parallel. This behaves similar to batching, but
        is much faster, though special handling of the model predictions is
        needed when more than 1 mask is provided.

        The 'is_prompt_encoding' flag should only be True if the given mask
        prediction was the result of a user-made prompt (e.g. box or points),
        if the mask comes from running on video frames, without user prompts,
        then 'is_prompt_encoding' should be False. This slightly modifies
        how the input mask is processed, but has a surprising impact on
        the quality of segmentation results over long time periods!

        Returns:
            lowres_img_encoding_bchw, fused_image_and_mask_encodings
            -> Both outputs have shape: BxCxHxW
            -> B batch size, H & W are height and width
            -> The 'lowres' output is the same as the input image encodings
            -> Bundling of orig. image encodings is feature specific to v3.1 model!
        """

        # Prepare mask & image data for fusion
        target_hw = lowres_image_encoding_bchw.shape[2:]
        encoded_mask = self.mask_downsampler(mask_prediction_1mhw, target_hw, is_prompt_encoding)
        encoded_image = self.image_proj(lowres_image_encoding_bchw)

        # Add encoded mask & image data together and mix channels to 'fuse' information
        memory_encoding = encoded_mask + encoded_image
        memory_encoding = self.channel_mixer(memory_encoding)

        # Special encoding for missing objects
        if object_score_m is not None:
            memory_encoding = self.missing_obj_encoder(memory_encoding, object_score_m)

        return (lowres_image_encoding_bchw, memory_encoding)

    # .................................................................................................................


class NoObjectEncoder(nn.Module):
    """
    This model comes from the SAMv2.0 -> SAMv2.1 update, which SAMv3.1 re-uses,
    and is responsible for adding a learned embedding vector to all 'pixels'
    of a given memory encoding whenever an object is considered to be missing,
    based on it's object score.

    The original SAMv3.1 version of this can be found here:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1730-L1737
    """

    # .................................................................................................................

    def __init__(self, features_per_memory_token: int, multiplex_channels: int = 16):
        super().__init__()
        self.no_object_embed = nn.Parameter(torch.empty(1, multiplex_channels, features_per_memory_token))

    def forward(self, memory_encoding_bchw: Tensor, object_score_m: Tensor) -> Tensor:
        """
        Adds a learned embedding to memory encodings whenever
        the object score is below 0, otherwise encodings are unchanged.

        Shapes for reference:
          memory_encoding_bchw is expected to have shape: BxCxHxW
          object_score has shape: M (M multiplex entries)
          output has same shape as memory_encoding (BxCxHxW)

        Returns:
            memory_encodings
        """

        # Pad score to match multiplex shape if needed
        num_scores = object_score_m.shape[0]
        num_multiplex = self.no_object_embed.shape[1]
        if num_scores < num_multiplex:
            device, dtype = object_score_m.device, object_score_m.dtype
            pad_score = torch.full([num_multiplex - num_scores], -1, device=device, dtype=dtype)
            object_score_m = torch.concat((object_score_m, pad_score), dim=0)

        # Add embedding to every pixel of memory encoding if no object is present, otherwise 'add' zero
        # -> This is done in a somewhat strange way to account for batching!
        obj_score_1m1 = object_score_m.unsqueeze(-1).unsqueeze(0)
        no_obj_present = (obj_score_1m1 < 0.0).to(dtype=self.no_object_embed.dtype)
        additive_embed_bc = (no_obj_present * self.no_object_embed).sum(dim=1)  # Sum along multiplex channels
        additive_embed_bchw = additive_embed_bc.unsqueeze(-1).unsqueeze(-1)
        return memory_encoding_bchw + additive_embed_bchw

    # .................................................................................................................
