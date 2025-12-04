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


class SAMV3MemoryEncoder(nn.Module):
    """
    Slightly modified implementation of the 'SimpleMaskEncoder' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    The original code can be found here:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/memory.py#L158

    While this version is used in SAMV3, the implementation is identical to SAMV2.
    (This code has been copied verbatim from the existing MuggledSAM implementation of SAMv2)

    This implementation has only minor changes compared to the original model, primarily
    in the form of renamed layers. It also features a slightly different call signature
    due to using a more heavily modified mask downsampler and doesn't compute positional
    encodings for the memory encodings it outputs.
    (this has been moved to another model, where the encoding are used)
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token=256,
        features_per_memory_token=64,
        num_downsample_layers=4,
        num_mixer_layers=2,
    ):

        # Inherit from parent
        super().__init__()

        # Define layers used to pre-process mask & image data prior to fusing
        self.mask_downsampler = MaskDownsampler(features_per_image_token, num_downsample_layers)
        self.image_proj = Conv1x1Layer(features_per_image_token)

        # Define layers used to post-process fused mask + image result
        self.channel_mixer = nn.Sequential(*(ConvNeXtBlock(features_per_image_token) for _ in range(num_mixer_layers)))
        self.out_proj = Conv1x1Layer(features_per_image_token, features_per_memory_token)

        # Create special handler for when no object is present
        self.missing_obj_encoder = NoObjectEncoder(features_per_memory_token)

    # .................................................................................................................

    def forward(
        self, lowres_image_encoding: Tensor, mask_prediction: Tensor, object_score: Tensor, is_prompt_encoding=False
    ) -> Tensor:
        """
        Takes the lowest-resolution image encoding and combines it
        with a mask prediction to form a 'fused' encoding, which can act
        a bit like a prompt to re-segment an object on future frames.
        (without requiring any other prompts)

        The input image encoding is expected to have shape BxCxHxW, the mask
        prediction should have a shape of Bx1x(4H)x(4W).

        The 'is_prompt_encoding' flag should only be True if the given mask
        prediction was the result of a user-made prompt (e.g. box or points),
        if the mask comes from running on video frames, without user prompts,
        then 'is_prompt_encoding' should be False. This slightly modifies
        how the input mask is processed, but has a surprising impact on
        the quality of segmentation results over long time periods!

        Returns:
            fused_image_and_mask_encodings
            -> Has shape: Bx(C/4)xHxW
            -> B is the batch size
            -> Output has 1/4 number of channels, C, as input image encoding
            -> Output height & width match (low-res) image encoding
        """

        # Prepare mask & image data for fusion
        target_hw = lowres_image_encoding.shape[2:]
        encoded_mask = self.mask_downsampler(mask_prediction, target_hw, is_prompt_encoding)
        encoded_image = self.image_proj(lowres_image_encoding)

        # Add encoded mask & image data together and mix channels to 'fuse' information
        memory_encoding = encoded_mask + encoded_image
        memory_encoding = self.channel_mixer(memory_encoding)
        memory_encoding = self.out_proj(memory_encoding)

        # Special encoding for missing objects (specific to version 2.1)
        memory_encoding = self.missing_obj_encoder(memory_encoding, object_score)

        return memory_encoding

    # .................................................................................................................


class NoObjectEncoder(nn.Module):
    """
    This model comes from the SAMv2.0 -> SAMv2.1 update, which SAMv3 re-uses,
    and is responsible for adding a learned embedding vector to all 'pixels'
    of a given memory encoding whenever an object is considered to be missing,
    based on it's object score.

    The original SAMv3 version of this can be found here:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L845-L848
    """

    # .................................................................................................................

    def __init__(self, features_per_memory_token: int):
        super().__init__()
        self.no_object_embed = nn.Parameter(torch.empty(1, features_per_memory_token))

    def forward(self, memory_encoding: Tensor, object_score: Tensor) -> Tensor:
        """
        Adds a learned embedding to memory encodings whenever
        the object score is below 0, otherwise encodings are unchanged.

        Shapes for reference:
          memory_encoding is expected to have shape: BxCxHxW
          object_score has shape: Bx1
          additive component has shape: 1xC
          output has same shape as memory_encoding (BxCxHxW)

        Returns:
            memory_encodings
        """

        # Add embedding to every pixel of memory encoding if no object is present, otherwise 'add' zero
        # -> This is done in a somewhat strange way to account for batching!
        no_object_present = (object_score < 0.0).to(dtype=self.no_object_embed.dtype)
        additive_embed_bchw = (no_object_present * self.no_object_embed).unsqueeze(-1).unsqueeze(-1)
        return memory_encoding + additive_embed_bchw.expand(*memory_encoding.shape)

    # .................................................................................................................
