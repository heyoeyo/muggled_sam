#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.memenc_components import MaskDownsampler, ConvNeXtBlock
from .components.shared import Conv1x1Layer

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2MemoryEncoder(nn.Module):
    """
    Slightly modified implementation of the 'memory encoder' from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    This implementation has only minor changes compared to the original model, primarily
    in the form of renamed layers. It also features a slightly different call signature
    due to using a more heavily modified mask downsampler and doesn't compute positional
    encodings for the memory encodings it outputs.
    (this has been moved to another model, where the encoding are used)

    The original code can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/memory_encoder.py#L138
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token=256,
        features_per_memory_token=64,
        num_downsample_layers=4,
        num_mixer_layers=2,
        is_version_2p1=True,
    ):

        # Inherit from parent
        super().__init__()

        # Define layers used to pre-process mask & image data prior to fusing
        self.mask_downsampler = MaskDownsampler(features_per_image_token, num_downsample_layers)
        self.image_proj = Conv1x1Layer(features_per_image_token)

        # Define layers used to post-process fused mask + image result
        self.channel_mixer = nn.Sequential(*(ConvNeXtBlock(features_per_image_token) for _ in range(num_mixer_layers)))
        self.out_proj = Conv1x1Layer(features_per_image_token, features_per_memory_token)

        # Create extra 'no object' handler specific to version 2.1
        NoObjEncoder = NoObjectEncoder_v2p1 if is_version_2p1 else NoObjectEncoder_v2p0
        self.missing_obj_encoder = NoObjEncoder(features_per_memory_token)

    # .................................................................................................................

    def forward(
        self, lowres_image_encoding: Tensor, mask_prediction: Tensor, object_score: Tensor, is_prompt_encoding=False
    ) -> Tensor:
        """
        Takes the lowest-resolution image encoding of SAMv2 and combines it
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


class NoObjectEncoder_v2p1(nn.Module):
    """
    This model is specific to SAMv2.1 (aka '2p1'), and is responsible
    for adding a learned embedding vector to all 'pixels' of a given
    memory encoding whenever an object is considered to be missing,
    based on it's object score. This was not present in version 2.0.

    The updated code for this (in v2.1) can be found here:
    https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L716
    """

    def __init__(self, features_per_memory_token):
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


class NoObjectEncoder_v2p0(nn.Module):
    """
    This model does nothing! It exists for the sake of
    forward-compatibility with the version 2.1 implementation
    """

    def __init__(self, features_per_memory_token):
        super().__init__()

    def forward(self, memory_encoding: Tensor, object_score: Tensor) -> Tensor:
        """Does nothing! Returns: memory_encoding (unchanged)"""
        return memory_encoding
