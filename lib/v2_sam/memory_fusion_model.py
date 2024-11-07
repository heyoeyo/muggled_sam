#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.memfuse_components import MemoryFusionTransformerLayer, FusionPositionOffset
from .components.posenc_sine import PositionEmbeddingSine
from .components.version_2_vs_2p1_variants import ObjectPointerPosEnc_v2p0, ObjectPointerPosEnc_v2p1

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2MemoryFusion(nn.Module):
    """
    Simplified implementation of the 'memory-attention' model from:
        "SAM 2: Segment Anything in Images and Videos"
        By: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma,
        Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan,
        Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
        Piotr Dollár, Christoph Feichtenhofer
        @ https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/

    The purpose of this model is to combine image encodings with information from past
    'memory encoding' tokens (from the memory encoder model) as well as 'object pointers'
    which come from the mask decoder, in order to generate a new set of image tokens.
    These 'memory fused' image tokens are used by the mask decoder to continue to segment
    an object on future frames, without having to provide new prompts.

    The original memory-attention model code can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/memory_attention.py

    This implementation differs significantly from the original. In addition to the original
    memory_attention model, some parts of this implementation are derived from the 'sam2_base' model,
    especially the 'prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/sam2_base.py
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token=256,
        features_per_memory_token=64,
        num_layers=4,
        max_memory_history=6,
        is_version_2p1=True,
    ):

        # Inherit from parent
        super().__init__()

        # Store sizing info for reshaping operations during inference
        self.features_per_memory_token = features_per_memory_token
        self._num_tokens_per_pointer = features_per_image_token // features_per_memory_token

        # Embedding added to encoded image features when not using memory encoding
        self.no_mem_embed = torch.nn.Parameter(torch.empty(1, 1, features_per_image_token))

        # Create models used to help prepare data for transformer layers
        self.memconcat = MemoryConcatenator(
            features_per_image_token, features_per_memory_token, max_memory_history, is_version_2p1
        )
        self.imgposenc = ImageTokenPositionEncoder(features_per_image_token)

        # Build transformer layers
        layers_list = []
        for _ in range(num_layers):
            layer = MemoryFusionTransformerLayer(features_per_image_token, features_per_memory_token, mlp_ratio=8)
            layers_list.append(layer)
        self.layers = nn.ModuleList(layers_list)
        self.out_norm = nn.LayerNorm(features_per_image_token)

    # .................................................................................................................

    def forward(
        self,
        lowres_image_tokens_bchw: Tensor,
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
        previous_is_recent_first=True,
        is_prompt_frame=False,
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
            no_mem_bchw = self.no_mem_embed.squeeze(0).unsqueeze(-1).unsqueeze(-1)
            fused_tokens = lowres_image_tokens_bchw + no_mem_bchw
            return fused_tokens

        # Merge all prior memory data into a single set of tokens
        memory_tokens, memory_posenc, num_ptr_tokens = self.memconcat(
            prompt_memory_encodings,
            prompt_object_pointers,
            previous_memory_encodings,
            previous_object_pointers,
            previous_is_recent_first,
        )

        # Get input shape so we can restore it on output
        b, _, h, w = lowres_image_tokens_bchw.shape
        patch_hw = (h, w)

        # Apply position encoding & flatten to rows-of-tokens format, shape: BxNxC
        image_posenc_tokens = self.imgposenc(lowres_image_tokens_bchw)
        flat_imgtokens_bnc = image_posenc_tokens.flatten(2).permute(0, 2, 1)

        # Run transformer layers to fuse memory results with image tokens
        for layer in self.layers:
            flat_imgtokens_bnc = layer(patch_hw, flat_imgtokens_bnc, memory_tokens, memory_posenc, num_ptr_tokens)

        # Convert back to image-like shape, from: BxNxC -> BxCxHxW
        flat_imgtokens_bnc = self.out_norm(flat_imgtokens_bnc)
        return flat_imgtokens_bnc.permute(0, 2, 1).reshape(b, -1, h, w)

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

    Although this model doesn't entirely match anything from the original
    code base, the closest reference would be the 'prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/sam2_base.py#L493
    """

    # .................................................................................................................

    def __init__(
        self,
        features_per_image_token=256,
        features_per_memory_token=64,
        max_memory_history=6,
        is_version_2p1=True,
    ):

        # Inherit from parent
        super().__init__()

        # Store sizing config for re-use
        self.features_per_memory_token = features_per_memory_token
        self._num_tokens_per_pointer = features_per_image_token // features_per_memory_token
        self._max_mempos_idx = max_memory_history - 1

        # Learned embeddings per 'relative position in time', applied to memory encodings
        self.memposenc = FusionPositionOffset(features_per_memory_token, max_memory_history)

        # Create model responsible for position encodings of object pointers
        ObjectPointerPosenc = ObjectPointerPosEnc_v2p1 if is_version_2p1 else ObjectPointerPosEnc_v2p0
        self.ptrposenc = ObjectPointerPosenc(features_per_image_token, features_per_memory_token)

    # .................................................................................................................

    def forward(
        self,
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_frame_memory_encodings: list[Tensor],
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
            -> Memory tokens have shape: BxNxF (B batch size, N number of tokens, F features, 64 by default)
            -> Memory position encoding has shape: BxNxF (matching tokens shape)
            -> Number of object pointer tokens is very small compared to the number of memory tokens!
               The object pointer tokens are stored at the end of the memory_tokens tensor.
        """

        # Allocate storage for all memory encodings and positional encodings
        memory_list = []
        posenc_list = []

        # Build memory encoding input
        for init_memenc in prompt_memory_encodings:

            # Convert from BxCxHxW to BxNxC
            maskmem_enc = init_memenc.flatten(2).permute(0, 2, 1)
            maskmem_pos = self.memposenc(init_memenc.shape, -1).flatten(2).permute(0, 2, 1)
            memory_list.append(maskmem_enc)
            posenc_list.append(maskmem_pos)

        # Get index representing how 'far away' each previous frame item is from current frame
        # -> Assumes buffer is built with 'appendleft' (i.e. 0th index entry is most recent in time)
        # -> E.g. indexing is: [0, 1, 2, 3, 4, 5]
        buffer_idx_list = list(range(len(previous_frame_memory_encodings)))
        if not previous_is_recent_first:
            # indexing is least-recent first: [5, 4, 3, 2, 1, 0]
            buffer_idx_list = list(reversed(buffer_idx_list))
        buffer_idx_list = [min(idx, self._max_mempos_idx) for idx in buffer_idx_list]

        # Combine memory encodings from past frames
        for mem_idx, memenc in zip(buffer_idx_list, previous_frame_memory_encodings):

            # Convert from BxCxHxW to BxNxC
            maskmem_enc = memenc.flatten(2).permute(0, 2, 1)
            maskmem_pos = self.memposenc(init_memenc.shape, mem_idx).flatten(2).permute(0, 2, 1)
            memory_list.append(maskmem_enc)
            posenc_list.append(maskmem_pos)

        # Build object pointer input if needed
        num_ptr_tokens = 0
        num_prompt_pointers = len(prompt_object_pointers)
        num_prev_pointers = len(previous_frame_object_pointers)
        have_pointers = (num_prompt_pointers + num_prev_pointers) > 0
        if have_pointers:

            # Combine all pointers and figure out token shaping
            # -> Pointers themselves are 'simple' embedding vectors (shape: Bx1xF)
            # -> They have a larger dimension than memory encodings (default sizing is: 256 vs 64)
            # -> Each pointer gets broken into multiple smaller 'tokens' to match memory dimension
            #    so that pointers can be stacked together with memory encodings for attention calculations
            # -> For example, say we have 12 pointers, each with 256 features. Assume memory tokens have
            #    64 features. If we just stack pointers into 'rows-of-tokens' format, we'd get a tensor
            #    with shape: 12x256, but we want to match memory token shape, which is Nx64 (N tokens).
            #    So we break each of the 256-feature pointers into 4 (=256/64) tokens for a total of
            #    48 'pointer tokens' each with 64 features and stack them altogether, giving shape: 48x64
            ptrs = torch.concat(list(prompt_object_pointers) + list(previous_frame_object_pointers), dim=1)
            ptr_batch_size, num_ptrs = ptrs.shape[0:2]
            num_ptr_tokens = num_ptrs * self._num_tokens_per_pointer
            ptr_tokens = ptrs.reshape(ptr_batch_size, num_ptr_tokens, self.features_per_memory_token)

            # Compute position encodings for pointer tokens
            ptrs_posenc = self.ptrposenc(num_prompt_pointers, num_prev_pointers, previous_is_recent_first)
            ptrs_posenc = ptrs_posenc.expand(ptr_batch_size, num_ptr_tokens, self.features_per_memory_token)

            # Add pointer encodings to total memory/position-encoding tokens
            memory_list.append(ptr_tokens)
            posenc_list.append(ptrs_posenc)

        # Stack memory & object pointer tokens (and positional encodings) into large BxNxC tensor
        memory_tokens = torch.cat(memory_list, dim=1)
        memory_posenc = torch.cat(posenc_list, dim=1)

        return memory_tokens, memory_posenc, num_ptr_tokens

    # .................................................................................................................


class ImageTokenPositionEncoder(nn.Module):
    """
    Simple helper used to handle the addition of position encodings
    to image tokens within the memory fusion model. This module does
    not exist in the original implementation, but is separated in
    this repo for the sake of clarity.

    This functionality seems like an odd implementation detail from the
    original code base, and is the only place where image position encodings
    are actually used (with default the configs at least), it can be found here:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/modeling/memory_attention.py#L141

    Removing this behavior (by not running the model or having a weight of 0.0),
    has very little effect on the output!
    """

    # .................................................................................................................

    def __init__(self, features_per_image_token=256, position_encoding_weight=0.1):
        super().__init__()
        self.posenc = PositionEmbeddingSine(features_per_image_token)
        self._posenc_weight = position_encoding_weight

    def forward(self, image_tokens_bchw: Tensor) -> Tensor:
        """
        Applies (additive) position encoding to image tokens
        Returns:
            encoded_image_tokens_bchw (same shape as input)
        """

        # For clarity
        b, _, h, w = image_tokens_bchw.shape

        # Create position encoding matching image shape and add to original tokens
        img_posenc = self.posenc(b, h, w) * self._posenc_weight
        return image_tokens_bchw + img_posenc

    # .................................................................................................................
