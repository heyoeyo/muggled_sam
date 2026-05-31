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
          object_score has shape: B
          additive component has shape: 1xC
          output has same shape as memory_encoding (BxCxHxW)

        Returns:
            memory_encodings
        """

        # Add embedding to every pixel of memory encoding if no object is present, otherwise 'add' zero
        # -> This is done in a somewhat strange way to account for batching!
        no_object_present = (object_score[:, None] < 0.0).to(dtype=self.no_object_embed.dtype)
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

    def forward(
        self,
        prompt_pointers_shape_brnc: tuple[int, int, int, int],
        frame_pointers_shape_brnc: tuple[int, int, int, int],
        is_recent_first: bool,
    ) -> tuple[Tensor, Tensor]:
        """
        This function computes the 'position encodings' for object pointers introduced in the
        SAMv2.1 update. There are several deviations from the original implementation here,
        in order to simplify things. Though this leads to numerical differences, the effect
        is negligible in practice (pointers don't do very much).

        The basic idea is to use a sinusoidal position encoding which is scaled by how
        'far away' (in time) each of the memory encodings is, using a normalized value (0 to 1),
        where 0 is 'closest' and 1 is farthest.

        There are 2 main differences here, compared to the original. First, prompt pointers
        are always treated as being 'now' in time, this allows us to avoid storing the
        indexing of prompts. Secondly, the frame pointer timing is normalized based on the
        number of pointers, whereas the original normalized by either the number of frames in
        the video or by a fixed config value (15 by default), whichever was less. Normalizing
        by the count avoids needing to know the video length or having extra configs.

        The 'is_recent_first' input indicates the temporal ordering of the frame pointers.

        Returns:
            prompt_pointer_position_encoding, frame_pointer_position_encoding
            -> Both shaped BxRxNxC, matching the given input shapes

        To see how the position encodings were computed in the original, see:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L628-L634

        To see how prompt vs. frame (cond_frame_outputs vs non_cond_frame_outputs) pointers were handled, see:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L599-L610
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L612-L620
        """

        # For clarity
        pmt_ptr_b, pmt_ptr_r, pmt_ptr_n, _ = prompt_pointers_shape_brnc
        frm_ptr_b, frm_ptr_r, frm_ptr_n, _ = frame_pointers_shape_brnc
        device, dtype = self.device_info.device, self.device_info.dtype

        # Use a temporal position of '0' for all prompt pointers
        # -> The original implementation does NOT do this!!!
        # -> Instead, the original position is given by the absolute frame index divided by min(15, frame_count)
        # -> Here using fixed (0) value because it's much simpler and mirrors memory encoding approach
        pmt_pos_norm_1r11 = torch.zeros((1, pmt_ptr_r, 1, 1), device=device, dtype=dtype)

        # Calculate the temporal position for frame pointers
        # -> In the original, these are just the relative frame difference divided by 15, for example:
        #    [0.0667, 0.1333, 0.2, 0.2667, 0.3333, 0.4, 0.4667, 0.5333, 0.6, 0.6667, 0.7333, 0.8, 0.8667, 0.9333, 1]
        # -> This is the sequence reported at max frames, it's just [1,2,3,...,15] all divided by 15
        # -> Here we do something similar, but always divide by the pointer count instead (avoids hard-coding 15 value)
        # -> For example, with 3 entries we'd get: [1/3, 2/3, 3/3], for 6 entries: [1/6, 2/6, 3/6, 4/6, 5/6, 6/6]
        small_idx = 1 / max(frm_ptr_r, 1)
        start_idx_norm, end_idx_norm = (small_idx, 1.0) if is_recent_first else (1.0, small_idx)
        frm_pos_norm = torch.linspace(start_idx_norm, end_idx_norm, frm_ptr_r, device=device, dtype=dtype)
        frm_pos_norm_1r11 = frm_pos_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        # Compute encodings and expand to handle batching and N > 1 if needed
        pmt_posenc_1r1c = self._encode_normalized_position(pmt_pos_norm_1r11)
        frm_posenc_1r1c = self._encode_normalized_position(frm_pos_norm_1r11)
        pmt_posenc_brnc = pmt_posenc_1r1c.expand(pmt_ptr_b, -1, pmt_ptr_n, -1)
        frm_posenc_brnc = frm_posenc_1r1c.expand(frm_ptr_b, -1, frm_ptr_n, -1)

        return pmt_posenc_brnc, frm_posenc_brnc

    # .................................................................................................................

    def _encode_normalized_position(self, pos_norm_tensor: Tensor) -> Tensor:

        # Compute 1D sinusoidal position embeddings from normalized pointer indices
        # For original code, see:
        # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_utils.py#L64
        pos_embed_base = pos_norm_tensor * self.posenc_scale_factor
        pos_embed = torch.cat([pos_embed_base.sin(), pos_embed_base.cos()], dim=-1)

        # Apply projection to reduce image token channel count to memory token channel count
        # -> Result has shape: 1xNxC, where N is number of pointers, C features per memory token (64 by default)
        return self.pointer_pos_proj(pos_embed)

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
        self.register_buffer("zeros_base", torch.zeros((1, 1, 1, features_per_memory_token)), persistent=False)
        self._num_tokens_per_pointer = features_per_image_token // features_per_memory_token

    def forward(
        self,
        prompt_pointers_shape_brnc: tuple[int, int, int, int],
        frame_pointers_shape_brnc: tuple[int, int, int, int],
        is_recent_first: bool,
    ) -> tuple[Tensor, Tensor]:
        """
        Creates 2 all-zeros tensors with shapes: BxRxNxC, matching the input pointer shapes
        Returns:
            prompt_position_encoding_brnc, frame_position_encoding_brnc
        """
        pmt_posenc_brnc = self.zeros_base.expand(*prompt_pointers_shape_brnc)
        frm_posenc_brnc = self.zeros_base.expand(*frame_pointers_shape_brnc)
        return pmt_posenc_brnc, frm_posenc_brnc
