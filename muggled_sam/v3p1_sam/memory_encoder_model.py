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
        self._multiplex_channels = multiplex_channels

        # Define layers used to post-process fused mask + image result
        self.channel_mixer = nn.Sequential(*(ConvNeXtBlock(features_per_image_token) for _ in range(num_mixer_layers)))

        # Create special handler for when no object is present
        self.missing_obj_encoder = NoObjectEncoder(features_per_memory_token, multiplex_channels)

    # .................................................................................................................

    def forward(
        self,
        lowres_image_encoding_bchw: Tensor,
        mask_prediction_1mhw: Tensor,
        object_pointers_1mc: Tensor | None,
        object_score_1m: Tensor | None,
        is_prompt_encoding: Tensor | bool,
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
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


        The 'is_prompt_encoding_bm' input should be a tensor where each entry
        is either a 1 (True) corresponding to entries that are meant to be
        prompt encodings, or 0 (False) otherwise.

        The 'is_padded_entry_bm' input is used to flag certain entries of
        the mask input as being 'padded' entries (e.g. to get BxM shape). If
        left as 'None', that all input masks are assumed to be valid masks.

        Note that if the number of multiplexed inputs 'M' is less than the
        maximum (16 by default), the inputs will be padded as needed internally.
        If 'M' is greater than the maximum, the multiplexing will be split
        into batches (called 'buckets' in the original code), so output
        memory encodings may have a batch size > 1!

        Returns:
            lowres_img_encoding_bchw, fused_image_and_mask_encodings
            -> Both outputs have shape: BxCxHxW
            -> B batch size, H & W are height and width
            -> The 'lowres' output is the same as the input image encodings
            -> Bundling of orig. image encodings is feature specific to v3.1 model!
            -> Note the output batch size can be >1 if using many multiplexed inputs
        """

        # Add padding/reshaping to support multiplexing
        masks_bmhw, ptrs_bmc, score_bm, is_prompt_bm, _ = self._make_multiplex_batches(
            mask_prediction_1mhw, object_pointers_1mc, object_score_1m, is_prompt_encoding
        )

        # Prepare mask & image data for fusion
        img_b, img_c, img_h, img_w = lowres_image_encoding_bchw.shape
        encoded_mask = self.mask_downsampler(masks_bmhw, (img_h, img_w), is_prompt_bm)
        encoded_image = self.image_proj(lowres_image_encoding_bchw)

        # Add encoded mask & image data together and mix channels to 'fuse' information
        memory_encoding = encoded_mask + encoded_image
        memory_encoding = self.channel_mixer(memory_encoding)

        # Special encoding for missing objects
        if score_bm is not None:
            memory_encoding = self.missing_obj_encoder(memory_encoding, score_bm)

        # Batch image encodings if needed
        mask_b = masks_bmhw.shape[0]
        if mask_b > 1 and img_b == 1:
            lowres_image_encoding_bchw = lowres_image_encoding_bchw.expand(mask_b, img_c, img_h, img_w)

        return (lowres_image_encoding_bchw, memory_encoding), ptrs_bmc

    # .................................................................................................................

    def _make_multiplex_batches(
        self,
        mask_prediction_1mhw: Tensor,
        object_pointers_1mc: Tensor | None,
        object_score_m: Tensor | None,
        is_prompt_encoding: Tensor | bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Helper function used to separate multiplexed inputs into batches that allow for
        integer multiples of the multiplex channel sizing. This is a v3.1 specific feature!

        For example, the default multiplex channel count is 16, so if given inputs with 23
        multiplex entries, we need to reshape the inputs to a size of 2x16 in order to work
        properly with the 16-size multiplexing.
        However, 2x16 would be 32 entries, and so each input also needs 'padding' to reach
        the target 2x16 sizing. This function does both the padding and reshaping to batches.

        Note that the 'is_prompt_encoding' can be given as a boolean or a tensor. If given as
        a boolean, it will be converted to a tensor matching the number of multiplex mask
        entries. If given as a tensor, it's expected to have 'M' entries matching the other
        inputs where each entry is either 1 (True) for entries which are prompt encodings,
        or 0 for non-prompt encodings.

        Returns:
            mask_prediction_bmhw, object_score_bm, object_pointers_bmc, is_prompt_encoding_bm, is_padded_entry_bm
            -> Notice that each output has a 'b' dimension
            -> Each 'm' dimension will be 16 (with default model config), except
               possibly the pointers (only padded when 'b' is > 1)
            -> The 'is_padded_entry_bm' result can be used to determine which entries
               of the output results are padded (1 if padded, 0 otherwise)
        """

        # For clarity
        mask_b, mask_m, mask_h, mask_w = mask_prediction_1mhw.shape
        ptr_c = object_pointers_1mc.shape[-1] if object_pointers_1mc is not None else 1
        device, dtype = mask_prediction_1mhw.device, mask_prediction_1mhw.dtype
        need_convert_is_prompt_enc = not isinstance(is_prompt_encoding, Tensor)

        # Sanity check. Make sure we're dealing with 1xMxHxW masks
        if mask_b > 1 and mask_m == 1:
            mask_prediction_1mhw = mask_prediction_1mhw.permute(1, 0, 2, 3)
            mask_b, mask_m, mask_h, mask_w = mask_prediction_1mhw.shape
        assert mask_b == 1, f"Mask shape error! Expecting 1xMxHxW, got: {tuple(mask_prediction_1mhw.shape)}"

        # Create missing pointer/score inputs for convenience (avoids messy logic for handling None values)
        is_missing_ptr = object_pointers_1mc is None
        is_missing_score = object_score_m is None
        if is_missing_ptr:
            object_pointers_1mc = torch.empty((1, mask_m, ptr_c), device=device, dtype=dtype)
        if is_missing_score:
            object_score_m = torch.empty([mask_m], device=device, dtype=dtype)

        # Convert 'is prompt' flag to per-multiple-mask entry if we're not given a tensor
        if need_convert_is_prompt_enc:
            is_prompt_enc_bm = torch.full([1, mask_m], is_prompt_encoding, device=device, dtype=dtype)
        else:
            # If user gives a tensor directly, make sure there's one entry for each mask input
            assert is_prompt_encoding.shape[0] == mask_m, "Prompt encoding tensor must match mask multiplex count!"
            is_prompt_enc_bm = is_prompt_encoding.unsqueeze(0).to(device=device, dtype=dtype)

        # Initialize 'batched' outputs
        mask_prediction_bmhw = mask_prediction_1mhw
        object_ptrs_bmc = object_pointers_1mc
        object_score_bm = object_score_m.unsqueeze(0)
        is_padded_entry_bm = torch.zeros((1, mask_m), device=device, dtype=dtype)

        # Generate padding so we have the right number of multiplex entries (possibly with batch size > 1)
        # -> For example, for a multiplex of 16 (default), if we get 40 masks, we
        #    need to pad to 48 entries, so we can form a 3x16 tensor (batch size of 3)
        mplex_b = torch.tensor(mask_m / self._multiplex_channels).ceil().int()
        num_empty_pad = (mplex_b * self._multiplex_channels) - mask_m
        if num_empty_pad > 0:
            pad_masks = torch.zeros((1, num_empty_pad, mask_h, mask_w), device=device, dtype=dtype)
            pad_score = torch.full([1, num_empty_pad], -10.0, device=device, dtype=dtype)
            pad_is_prompt = torch.zeros([1, num_empty_pad], device=device, dtype=dtype)
            pad_is_padded_entry = torch.ones([1, num_empty_pad], device=device, dtype=dtype)

            # Append padding to inputs
            mask_prediction_bmhw = torch.concat((mask_prediction_bmhw, pad_masks), dim=1)
            object_score_bm = torch.concat((object_score_bm, pad_score), dim=1)
            is_prompt_enc_bm = torch.concat((is_prompt_enc_bm, pad_is_prompt), dim=1)
            is_padded_entry_bm = torch.concat((is_padded_entry_bm, pad_is_padded_entry), dim=1)

            # Pointers get special treatment. They don't need padding unless we're batching
            # -> Padding can have a noticable negative effect otherwise!
            num_pad_ptrs = num_empty_pad if mplex_b > 1 else 0
            pad_ptrs = torch.zeros((1, num_pad_ptrs, ptr_c), device=device, dtype=dtype)
            object_ptrs_bmc = torch.concat((object_ptrs_bmc, pad_ptrs), dim=1)

        # Reshape inputs to properly use batch dimension if needed
        if mplex_b > 1:
            mask_prediction_bmhw = mask_prediction_bmhw.view(mplex_b, self._multiplex_channels, mask_h, mask_w)
            object_ptrs_bmc = object_ptrs_bmc.view(mplex_b, self._multiplex_channels, ptr_c)
            object_score_bm = object_score_bm.view(mplex_b, self._multiplex_channels)
            is_prompt_enc_bm = is_prompt_enc_bm.view(mplex_b, self._multiplex_channels)
            is_padded_entry_bm = is_padded_entry_bm.view(mplex_b, self._multiplex_channels)

        # Reset inputs that came in as 'None'
        if is_missing_ptr:
            object_ptrs_bmc = None
        if is_missing_score:
            object_score_bm = None

        return mask_prediction_bmhw, object_ptrs_bmc, object_score_bm, is_prompt_enc_bm, is_padded_entry_bm

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

    def forward(self, memory_encoding_bchw: Tensor, object_score_bm: Tensor) -> Tensor:
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

        # Add embedding to every pixel of memory encoding if no object is present, otherwise 'add' zero
        # -> This is done in a somewhat strange way to account for batching!
        obj_score_1m1 = object_score_bm.unsqueeze(-1)
        no_obj_present = (obj_score_1m1 < 0.0).to(dtype=self.no_object_embed.dtype)
        additive_embed_bc = (no_obj_present * self.no_object_embed).sum(dim=1)  # Sum along multiplex channels
        additive_embed_bchw = additive_embed_bc.unsqueeze(-1).unsqueeze(-1)
        return memory_encoding_bchw + additive_embed_bchw

    # .................................................................................................................
