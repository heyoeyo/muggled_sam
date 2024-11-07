#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Memory encoder classes


class NoObjectEncoder_v2p1(nn.Module):
    """
    This model is specific to SAMv2.1 (aka '2p1'), and is responsible
    for adding a learned embedding vector to all 'pixels' of a given
    memory encoding whenever an object is considered to be missing,
    based on it's object score. This was not present in version 2.0.

    The updated code for this (in v2.1) can be found here:
    https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L716
    """
    
    # .................................................................................................................

    def __init__(self, features_per_memory_token):
        super().__init__()
        self.no_object_embed = nn.Parameter(torch.empty(1, features_per_memory_token))

    # .................................................................................................................
    
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


# ---------------------------------------------------------------------------------------------------------------------
# %% Memory fusion classes


class ObjectPointerPosEnc_v2p1(nn.Module):
    """
    This model is specific to SAMv2.1 (aka '2p1'), and is responsible
    for creating position encodings for object pointers used as part of
    the 'memory fusion' steps.

    It works by first computing (non-learned) sinusoidal position encodings
    based on the frame difference of each pointer relative to the current frame
    (i.e. roughly: posenc = [sin(frame_diff), cos(frame_diff)], but with extra scaling terms)
    followed by a learned linear projection step.

    This implementation deviates somewhat from the original, particularly through the
    use of pre-computed values (which lead to slight numerical differences) as well
    as significantly different handling of the prompt-object-pointers, which are always
    assumed to have a frame difference of zero (the original keeps track of the actual difference).
    This is done for the sake of simplicity (avoids having to manage per-pointer frame indexing data),
    and seems to have minimal impact on the final result.

    The original code for this (in v2.1) exists within the 'prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L533
    https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L592-L596
    https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L612-L620
    https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L629-L633
    """

    # .................................................................................................................

    def __init__(self, features_per_image_token, features_per_memory_token):

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

    def forward(self, num_prompt_pointers, num_previous_frame_pointers, previous_is_recent_first=True) -> Tensor:
        """
        Creates a position encoding tensor of shape: NxF
        -> N is total number of tokens (equal to 4 times the total number of pointers, by default)
        -> F is features per memory token (64 by default)
        """

        # Set position index of all 'prompt pointers' to 0
        # -> The original implementation does NOT do this!!!
        # -> Instead the index is based on how many frames have past since the prompt (i.e. steadily increases)
        # -> Here using fixed (0) value because it's much simpler and mirrors memory encoding approach
        # See original here: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L628
        total_ptrs = num_prompt_pointers + num_previous_frame_pointers
        pos_norm_tensor = torch.zeros(total_ptrs, dtype=self.device_info.dtype, device=self.device_info.device)

        # Set position index of all 'previous frame pointers' based on their list indexing
        # -> This is also different from the original implementation, though not significantly!
        # -> In original, pointer indexing is based on number of frames since the pointer was created.
        #    In basic usage, this is the same as the approach used here, but in the original
        #    it is possible for pointers to be given with indexing not based on consecutive frame sequencing
        # -> In original, indexes are normalized based on a fixed (configurable) value corresponding
        #    to the maximum pointer count (default 16). Here they're normalized based on the given
        #    pointer count. This approach will match the original once the number of given pointers
        #    reaches a set maximum (until then, this approach over-estimates pointer spacing in time)
        # See original here: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L632
        first_prev_idx = 1.0 / max(total_ptrs - 1, 1)
        start_idx, end_idx = (first_prev_idx, 1.0) if previous_is_recent_first else (1.0, first_prev_idx)
        pos_norm_tensor[num_prompt_pointers:] = torch.linspace(start_idx, end_idx, num_previous_frame_pointers)

        # Compute 1D sinusoidal position embeddings from pointer indices
        # For original code, see:
        # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_utils.py#L64
        pos_embed_base = pos_norm_tensor.unsqueeze(-1) * self.posenc_scale_factor
        pos_embed = torch.cat([pos_embed_base.sin(), pos_embed_base.cos()], dim=-1)

        # Apply projection to reduce image token channel count to memory token channel count
        # -> Result has shape: NxF, where N is total number of pointers, F is features per memory token (64 by default)
        ptrs_posenc = self.pointer_pos_proj(pos_embed)

        # Repeat each position encoding multiple times (default 4) to account for unusual 'split and stack'
        # approach used when constructing object pointer memory, which breaks each 256 dimension pointer
        # into four 64 dimension tokens. So we repeat position encoding of each pointer 4 times to match
        # split pointers. For example, instead of returning just: [a,b,c], we return: [a,a,a,a,b,b,b,b,c,c,c,c]
        return ptrs_posenc.repeat_interleave(self._num_tokens_per_pointer, dim=0)

    # .................................................................................................................


class ObjectPointerPosEnc_v2p0(nn.Module):
    """
    This model is responsible for creating position encodings for object
    pointers for SAM v2.0 models, which are just vectors of all zeros!

    This model mainly exists for the sake of forward-compatibility with
    the version 2.1 implementation. The equivalent original code can be found here:
    https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L636
    """

    def __init__(self, features_per_image_token, features_per_memory_token):
        super().__init__()
        self.register_buffer("zeros_base", torch.zeros((1, features_per_memory_token)), persistent=False)
        self._num_tokens_per_pointer = features_per_image_token // features_per_memory_token

    def forward(self, num_prompt_pointers, num_previous_frame_pointers, previous_is_recent_first=True) -> Tensor:
        """
        Creates a zeros tensor of shape: NxF
        -> N is total number of tokens (equal to 4 times the total number of pointers, by default)
        -> F is features per memory token (64 by default)
        """
        total_num_pointers = num_prompt_pointers + num_previous_frame_pointers
        total_num_tokens = total_num_pointers * self._num_tokens_per_pointer
        return self.zeros_base.repeat(total_num_tokens, 1)
