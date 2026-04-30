#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.memory_image_fusion_components import MemoryImageFusionTransformer, FusionPositionOffset

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3p1MemoryImageFusion(nn.Module):
    """
    Slightly modified implementation of the 'tracker transformer' from:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    The original code builds this model from many separate pieces, see:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py#L366

    While this version is used in SAMV3, the implementation is identical to SAMV2.
    (This code has been copied verbatim from the existing MuggledSAM implementation of SAMv2)

    The purpose of this model is to combine image encodings with information from past
    'memory encoding' tokens (from the memory encoder model) as well as 'object pointers'
    which come from the mask decoder, in order to generate a new set of image tokens.
    These 'memory fused' image tokens are used by the mask decoder to continue to segment
    an object on future frames, without having to provide new prompts.
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token: int = 256,
        features_per_memory_token: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_memory_history: int = 6,
    ):

        # Inherit from parent
        super().__init__()

        # Store sizing info for reshaping operations during inference
        self.features_per_memory_token = features_per_memory_token

        # Embedding added to encoded image features when not using memory encoding
        self.no_mem_embed_bchw = torch.nn.Parameter(torch.empty(1, features_per_image_token, 1, 1))

        # Create model used to help prepare data for transformer
        self.memconcat = MemoryConcatenator(features_per_image_token, features_per_memory_token, max_memory_history)

        # Model used to encode memory data into image tokens
        self.fusion_transformer = MemoryImageFusionTransformer(
            features_per_image_token, features_per_memory_token, num_layers, num_heads
        )

    # .................................................................................................................

    def forward(
        self,
        lowres_image_tokens_bchw: Tensor,
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
        previous_is_recent_first: bool = True,
        is_prompt_frame: bool = False,
    ) -> Tensor:
        """
        Fuses prior memory encodings & object pointers into (low-res) image tokens
        Expects lists of all prior memory encodings & object pointers.

        Prior memory encodings come from the memory encoder model, while object pointers
        come from the mask decoder. Encodings/pointers that come from prompted
        frames are handled separately from non-prompted encodings/pointers
        (which are assumed to be coming from running on previous frames with no prompts).

        If 'previous_is_recent_first' is True, then the first-most (i.e. 0th-index)
        entry of the 'previous' lists are assumed to be the most recent entry
        (this comes from using '.appendleft' on deque data types), otherwise
        assumes the last-most entry is most recent (i.e. using .append on a list).

        Returns:
            memory_fused_image_tokens (same shape as input image tokens)
        """

        # If we're prompting or there is no memory data, do simpler fuse
        if is_prompt_frame or len(prompt_memory_encodings) == 0:
            fused_tokens = lowres_image_tokens_bchw + self.no_mem_embed_bchw
            return fused_tokens

        # Merge all prior memory data into a single set of tokens
        prev_img_tokens, memory_tokens, memory_posenc, num_ptr_tokens = self.memconcat(
            prompt_memory_encodings,
            prompt_object_pointers,
            previous_memory_encodings,
            previous_object_pointers,
            previous_is_recent_first,
        )

        # Fuse memory results into image tokens
        fused_img_tokens = self.fusion_transformer(
            lowres_image_tokens_bchw, prev_img_tokens, memory_tokens, memory_posenc, num_ptr_tokens
        )
        return fused_img_tokens

    # .................................................................................................................


class MemoryConcatenator(nn.Module):
    """
    Model used to prepare prior memory encodings & object pointers for
    use within the memory fusion model. The model primarily just
    concatenates all memory data together, but also handles positional
    encoding of memory data (which follows somewhat complicated rules!).

    This model does not exist in the original implementation. In fact,
    the original functionality exists across many different parts of
    the model and may not be fully represented in this implementation!

    The closest correspondence between this code and the original
    SAMv3 implementation can be found in the 'prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/sam3/blob/2d1cbaeac7b52ca64baf61e58973d0940ae843d0/sam3/model/sam3_tracker_base.py#L560
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token=256,
        features_per_memory_token=256,
        max_memory_history=6,
    ):

        # Inherit from parent
        super().__init__()

        # Store sizing config for re-use
        self.features_per_memory_token = features_per_memory_token
        self._max_mempos_idx = max_memory_history - 1

        # Learned embeddings per 'relative position in time', applied to memory encodings
        self.memposenc = FusionPositionOffset(features_per_memory_token, max_memory_history)

        # Create model responsible for position encodings of object pointers
        self.ptrposenc = ObjectPointerPosEnc(features_per_image_token, features_per_memory_token)

    # .................................................................................................................

    def forward(
        self,
        prompt_memory_encodings: list[tuple[Tensor, Tensor]],
        prompt_object_pointers: list[Tensor],
        previous_frame_memory_encodings: list[tuple[Tensor, Tensor]],
        previous_frame_object_pointers: list[Tensor],
        previous_is_recent_first=True,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Helps to combine all prompt & previous frame memory data into
        a single tensor, along with a corresponding positional encoding
        tensor. The output can be thought of as a 'rows-of-tokens' formatted
        tensor, where the tokens are prior (image-like) memory encodings
        as well as prior (token-like) object pointers, both of which are
        meant to be representations of the object that is being segmented.
        These are used to 'find' the same object in future frames.

        Inputs should be provided as lists of prior memory/pointers. If
        'previous_is_recent_first' is True, then this is meant to imply that
        index-0 of each 'previous_frame' list represents the most recent data.
        If False, then the index-0 entry is interpreted as the oldest data.

        The difference between 'prompt' and 'previous_frame' inputs is that
        the prompt inputs are those used to initialize tracking. They come
        from a user directly prompting the model. The previous_frame inputs
        are meant to come from the model running on it's own (without prompting).

        Returns:
            memory_tokens, memory_posenc, num_object_pointer_tokens
            -> Memory tokens have shape: BxNxC (B batch size, N number of tokens, C features, 256 by default)
            -> Memory position encoding has shape: BxNxC (matching tokens shape)
            -> Number of object pointer tokens is very small compared to the number of memory tokens!
               The object pointer tokens are stored at the end of the memory_tokens tensor.
        """

        # Allocate storage for all 'previous image', memory and positional encodings
        # -> Note that v3.1 stores previous images inside the memory encoding entries!
        imgenc_list, memory_list, posenc_list = [], [], []

        # Build memory encoding input
        for init_imgenc, init_memenc in prompt_memory_encodings:

            # Convert from BxCxHxW to BxNxC
            maskmem_enc = init_memenc.flatten(2).permute(0, 2, 1)
            maskmem_pos = self.memposenc(init_memenc.shape, -1).flatten(2).permute(0, 2, 1)
            previmg_enc = init_imgenc.flatten(2).permute(0, 2, 1)
            memory_list.append(maskmem_enc)
            posenc_list.append(maskmem_pos)
            imgenc_list.append(previmg_enc)

        # Get index representing how 'far away' each previous frame item is from current frame
        # -> This indexing is reversed so that the closest encoding is stored in index 5, oldest is index 0
        buffer_idx_list = [max(0, self._max_mempos_idx - idx) for idx in range(len(previous_frame_memory_encodings))]
        if not previous_is_recent_first:
            buffer_idx_list = list(reversed(buffer_idx_list))  # Gives: [0,1,2,3,4,5] for newest-to-oldest ordering

        # Combine memory encodings from past frames
        for mem_idx, (imgenc, memenc) in zip(buffer_idx_list, previous_frame_memory_encodings):

            # Convert from BxCxHxW to BxNxC
            maskmem_enc = memenc.flatten(2).permute(0, 2, 1)
            maskmem_pos = self.memposenc(memenc.shape, mem_idx).flatten(2).permute(0, 2, 1)
            previmg_enc = imgenc.flatten(2).permute(0, 2, 1)
            memory_list.append(maskmem_enc)
            posenc_list.append(maskmem_pos)
            imgenc_list.append(previmg_enc)

        # Record prompt pointers if present
        num_ptr_tokens = 0
        if len(prompt_object_pointers) > 0:
            prompt_ptr_tokens_bnc, prompt_ptr_posenc_bnc = self.ptrposenc(
                tuple(prompt_object_pointers), previous_is_recent_first, is_prompt_encoding=True
            )
            memory_list.append(prompt_ptr_tokens_bnc)
            posenc_list.append(prompt_ptr_posenc_bnc)
            num_ptr_tokens += prompt_ptr_tokens_bnc.shape[1]

        # Record previous frame pointers if present
        if len(previous_frame_object_pointers) > 0:
            prev_ptr_tokens_bnc, prev_posenc_bnc = self.ptrposenc(
                tuple(previous_frame_object_pointers), previous_is_recent_first, is_prompt_encoding=False
            )
            num_ptr_tokens += prev_ptr_tokens_bnc.shape[1]
            memory_list.append(prev_ptr_tokens_bnc)
            posenc_list.append(prev_posenc_bnc)

        # Stack tokens into large BxNxC tensors
        memory_tokens = torch.cat(memory_list, dim=1)
        memory_posenc = torch.cat(posenc_list, dim=1)

        # Pad previous-image tokens to match memory size (bigger due to pointers)
        # https://github.com/facebookresearch/sam3/blob/967fdd651f71ca14949122fed4c918a778ca9334/sam3/model/decoder.py#L1321-L1332
        mem_b, mem_n, mem_c = memory_tokens.shape
        device, dtype = memory_tokens.device, memory_tokens.dtype
        pad_img_ptrs = torch.zeros((mem_b, num_ptr_tokens, mem_c), device=device, dtype=dtype)
        imgenc_list.append(pad_img_ptrs)
        prev_image_tokens = torch.cat(imgenc_list, dim=1)

        return prev_image_tokens, memory_tokens, memory_posenc, num_ptr_tokens

    # .................................................................................................................


class ObjectPointerPosEnc(nn.Module):
    """
    This model is responsible for creating position encodings for object pointers
    used as part of the 'memory image fusion' steps.

    It works by first computing (non-learned) sinusoidal position encodings
    based on the frame difference of each pointer relative to the current frame
    (i.e. roughly: posenc = [sin(frame_diff), cos(frame_diff)], but with extra scaling terms)
    followed by a learned linear projection step.

    TODO:
    This implementation comes directly from the MuggledSAM SAMv2 implementation.
    The SAMv3 code has a number of changes to the related code that may make
    this implementation even more incorrect (it already deviated from SAMv2 somewhat).
    More work is needed to carefully determine what the SAMv3 version may be doing differently...

    Here are links to the related functionality (at least, based on a comparison with the SAMv2 code):
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L600-L603
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L700-L704
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L718-L732
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L749C32-L749C45
    """

    # .................................................................................................................

    def __init__(self, features_per_image_token: int = 256, features_per_memory_token: int = 256):

        # Inherit from parent
        super().__init__()

        # Set up projection layer which maps image-feature channel count down to memory-feature count
        self.pointer_pos_proj = torch.nn.Linear(features_per_image_token, features_per_memory_token)
        self._num_tokens_per_pointer = features_per_image_token // features_per_memory_token

        # Pre-compute scaling factor used in position encodings
        num_posenc_features = features_per_image_token // 2
        posenc_scale_factor = torch.arange(num_posenc_features, dtype=torch.float32)
        posenc_scale_factor = 2 * (posenc_scale_factor // 2) / num_posenc_features
        posenc_scale_factor = 10000**posenc_scale_factor
        posenc_scale_factor = 1.0 / posenc_scale_factor
        self.register_buffer("posenc_scale_factor", posenc_scale_factor, persistent=False)

        # Create 'parameter' used for recording the model device/dtype information
        self.register_buffer("device_info", torch.empty(0), persistent=False)

    # .................................................................................................................

    def forward(
        self,
        pointer_tokens_bnc_list: list[Tensor],
        is_recent_first: bool,
        is_prompt_encoding: bool,
    ) -> tuple[Tensor, Tensor]:
        """
        Concatenates pointer tokens and creates a corresponding position encoding.

        Returns:
            pointer_tokens_bnc, pointer_position_encoding_bnc
            -> Both shapes are BxNxC, B batch size, N total token count and C features

        Note that the position encoding in this implementation deviates from the original
        as it does not include the frame indexing information used by the original.
        However, pointers (and the position encodings) have very little effect on tracking,
        so this usually shouldn't cause any issues.

        Difference for prompt tokens:
            -> The position index is always set to 0 in this implementation
            -> The original implementation does NOT (generally) do this!!!
            -> Instead the index is based on how many frames have past since the prompt (i.e. steadily increases)
        https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L162

        Difference for previous frame tokens:
            -> The position index is set based on the list indexing of each pointer in this implementation
            -> In this implementation, indexes are always normalized to a max value of 1
            -> In original, pointer indexing is based on number of frames since the pointer was created
            -> In original, indexes are normalized based on a fixed (configurable) value corresponding
               to the maximum pointer count (default 16). The max value will be 1 only when 16 pointers stored
            -> In practice, the two approaches will be similar once 16+ frames have passed
        https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L167-L170
        """

        # For clarity
        total_num_tokens = len(pointer_tokens_bnc_list)
        device, dtype = self.device_info.device, self.device_info.dtype

        # Generate position encoding per-pointer entry
        # -> Have to do this in a loop, because each entry needs a different scaling factor
        #    but not all entries have the same shape (i.e. they can't be stacked/concatenated easily)
        posenc_nc_list = []
        for idx, tokens_bnc in enumerate(pointer_tokens_bnc_list):

            # Compute pointer indexing (depends on prompts vs. non-prompts)
            idx_numer = 1 + idx if is_recent_first else (total_num_tokens - idx)
            posenc_base_value = idx_numer / total_num_tokens if not is_prompt_encoding else 0
            _, num_ptrs, _ = tokens_bnc.shape
            posenc_tensor_nc = torch.full((num_ptrs, 1), posenc_base_value, device=device, dtype=dtype)

            # Compute 1D sinusoidal position embeddings from pointer indices
            # See original here: https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_utils.py#L327
            posenc_tensor_nc = posenc_tensor_nc * self.posenc_scale_factor
            posenc_tensor_nc = torch.cat([posenc_tensor_nc.sin(), posenc_tensor_nc.cos()], dim=-1)
            posenc_nc_list.append(posenc_tensor_nc)

        # Merge all pointers into a single tensor as well as position encodings
        out_ptrs_bnc = torch.concat(pointer_tokens_bnc_list, dim=1)
        out_posenc_bnc = torch.concat(posenc_nc_list, dim=0).unsqueeze(0)
        out_posenc_bnc = self.pointer_pos_proj(out_posenc_bnc)
        if out_ptrs_bnc.shape[0] > 1:
            out_posenc_bnc = out_posenc_bnc.expand(out_ptrs_bnc.shape[0], -1, -1)

        return out_ptrs_bnc, out_posenc_bnc

    # .................................................................................................................
