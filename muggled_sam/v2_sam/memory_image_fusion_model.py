#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.memory_image_fusion_components import MemoryImageFusionTransformer
from .components.posenc_sine import SinusoidalPE2D
from .components.version_2_vs_2p1_variants import ObjectPointerPosEnc_v2p0, ObjectPointerPosEnc_v2p1

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2MemoryImageFusion(nn.Module):
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
        num_heads=1,
        max_memory_history=6,
        is_version_2p1=True,
    ):

        # Inherit from parent
        super().__init__()

        # Store sizing info for reshaping operations during inference
        self._num_mem_features = features_per_memory_token

        # Embedding added to encoded image features when not using memory encoding
        self.no_mem_embed_bchw = torch.nn.Parameter(torch.empty(1, features_per_image_token, 1, 1))

        # Create model used to help prepare data for transformer
        self.memconcat = MemoryConcatenator(
            features_per_image_token, features_per_memory_token, max_memory_history, is_version_2p1
        )

        # Model used to encode memory data into image tokens
        self.fusion_transformer = MemoryImageFusionTransformer(
            features_per_image_token, features_per_memory_token, num_layers, num_heads
        )

    # .................................................................................................................

    def forward(
        self,
        lowres_image_tokens_bchw: Tensor,
        prompt_memory_encodings: list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor],
        frame_memory_encodings: list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor],
        is_recent_first: bool = False,
        is_prompt_frame: bool = False,
    ) -> Tensor:
        """
        Fuses prompt and frame memory into (low-res) image tokens.

        Prompt memory is expected to come from user-specified prompts that
        beging tracking for an object. Frame memory comes from encoding
        the output of model during tracking, without user-input.

        The memory inputs can be provided as either a list entries, like:
            [frame_memory_1, frame_memory_2, ..., etc.]
        Each memory entry is itself expected to be a tuple:
            frame_memory_1 = (mask_memory_1, obj_pointer_1),
        which is output by the memory encoding functions of the model.

        As an alternative, it's also possible to provide inputs directly
        as tensors, in which case a single 2-tuple should be provided:
            (mask_memory_tensor, obj_pointer_tensor)
        These tensors are meant to be stacked versions of the contents of the
        list of memory (tuples) input option. When providing tensors directly,
        ideally both should have shape: BxRxNxC, where B is the batch size,
        R is the number of entries (e.g. length of the list), N is the number
        of tokens (e.g. N=H*W for image-like memory) and C is the channel count.
        The image-like mask memory is naturally shaped like BxCxHxW, so the
        tensor can also be given as BxRxCxHxW (R stacked entries).

        The 'is_recent_first' input is used to indicate the ordering of the
        provided frame_memory_encodings. If set to 'True', then this means
        that the first-most entry (e.g. index 0) of the frame memory is the
        most recent in time, otherwise the last-most entry is assumed to
        be the most recent.

        The 'is_prompt_frame' can be set to True to force a special
        shortcut that skips the normal encoding. This is a feature
        of the original SAM model, but isn't needed in practice.

        Returns:
            memory_fused_image_tokens (same shape as input image tokens)
        """

        # If we're prompting or there is no memory data, do simpler fuse
        if is_prompt_frame or len(prompt_memory_encodings) == 0:
            return lowres_image_tokens_bchw + self.no_mem_embed_bchw

        # Make sure we're dealing with BxRxNxC memory & pointer tensors
        p_mem_brnc, p_ptr_brnc, f_mem_brnc, f_ptr_brnc = self._prepare_memory_tensors(
            prompt_memory_encodings, frame_memory_encodings
        )

        # Handle pointer channel adjustments if needed
        p_ptr_brnc, f_ptr_brnc = self._prepare_reshaped_pointers(p_ptr_brnc, f_ptr_brnc)

        # Merge all memory together and create corresponding position encoding tensor
        token_hw = lowres_image_tokens_bchw.shape[-2:]
        memory_tokens_bnc, memory_posenc_bnc, num_ptr_tokens = self.memconcat(
            token_hw, p_mem_brnc, p_ptr_brnc, f_mem_brnc, f_ptr_brnc, is_recent_first
        )

        # Fuse memory results into image tokens
        fused_img_tokens = self.fusion_transformer(
            lowres_image_tokens_bchw, memory_tokens_bnc, memory_posenc_bnc, num_ptr_tokens
        )

        return fused_img_tokens

    # .................................................................................................................

    def _prepare_memory_tensors(
        self,
        prompt_memory_encodings: list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor],
        frame_memory_encodings: list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Helper used to produce a 2-tuple of tensor data from inputs which may be
        given as a list of many 2-tuples of tensors.
        More specifically, prompt/frame memory encodings may be given as a
        list of:
            [(mask_mem_1, ptr_1), (mask_mem_2, ptr_2), (mask_mem_3, ptr_3), etc...]
        And this function is used to convert each list into 2 tensors:
            all_mask_mems, all_ptrs
        This is done for both the prompt & frame inputs, giving 4 tensors total.
        All outputs will have shapes like: BxRxNxC, where R is the number of entries
        in the list (i.e. separate memory encodings), and is generally different
        for prompt vs. frame memory. Note that the mask memory has a default shape
        of BxCxHxW, but will be still be output as BxRxNxC, where N=H*W.

        It's also possible to directly provide the 2-tuple of tensors for each input,
        in which case nothing will be done.

        Returns:
            prompt_mask_memory, prompt_pointers, frame_mask_memory, frame_pointer
            -> All entries have shape: BxRxNxC
            -> The 'R' shape corresponds to the list length (e.g. how many memory/frames were given)
        """

        # Convert list of prompt memory tuples into 2 (mask memory & object pointer) tensors
        if not isinstance(prompt_memory_encodings[0], Tensor):
            pmt_mem_tensor = torch.stack([data_tuple[0] for data_tuple in prompt_memory_encodings], dim=1)
            pmt_ptr_tensor = torch.stack([data_tuple[1] for data_tuple in prompt_memory_encodings], dim=1)
            prompt_memory_encodings = (pmt_mem_tensor, pmt_ptr_tensor)

        # Create dummy frame encodings if we're not given any, so rest of logic can work without conditional checks
        # -> We do this in a way that can handle mem inputs with shape BxRxCxHxW or BxRxNxC
        if len(frame_memory_encodings) == 0:
            ex_mem, ex_ptr = prompt_memory_encodings
            empty_mem = ex_mem.new_empty(ex_mem.shape[0], 0, *ex_mem.shape[2:])
            empty_ptr = ex_ptr.new_empty(ex_ptr.shape[0], 0, *ex_ptr.shape[2:])
            frame_memory_encodings = (empty_mem, empty_ptr)

        # Here we form tensors from a list of frame memory as we did with prompt memory
        if not isinstance(frame_memory_encodings[0], Tensor):
            frm_mem_tensor = torch.stack([data_tuple[0] for data_tuple in frame_memory_encodings], dim=1)
            frm_ptr_tensor = torch.stack([data_tuple[1] for data_tuple in frame_memory_encodings], dim=1)
            frame_memory_encodings = (frm_mem_tensor, frm_ptr_tensor)

        # Unpack tensors for output
        p_mem_tensor, p_ptr_brnc = prompt_memory_encodings
        f_mem_tensor, f_ptr_brnc = frame_memory_encodings

        # If we get 5 dimensions, assume image-like shape: BxRxCxHxW and convert to: BxRxNxC
        p_mem_brnc = p_mem_tensor.flatten(3).permute(0, 1, 3, 2) if p_mem_tensor.ndim == 5 else p_mem_tensor
        f_mem_brnc = f_mem_tensor.flatten(3).permute(0, 1, 3, 2) if f_mem_tensor.ndim == 5 else f_mem_tensor

        # Sanity checks (mostly to catch errors with direct tensor inputs)
        assert p_mem_brnc.ndim == 4, f"Need prompt memory shape: BxRxNxC, got: {p_mem_brnc.shape}"
        assert f_mem_brnc.ndim == 4, f"Need frame memory shape: BxRxNxC, got: {f_mem_brnc.shape}"
        assert p_ptr_brnc.ndim == 4, f"Need prompt pointers shape: BxRxNxC, got: {p_ptr_brnc.shape}"
        assert f_ptr_brnc.ndim == 4, f"Need frame pointers shape: BxRxNxC, got: {f_ptr_brnc.shape}"

        return p_mem_brnc, p_ptr_brnc, f_mem_brnc, f_ptr_brnc

    # .................................................................................................................

    def _prepare_reshaped_pointers(
        self,
        prompt_pointers_brnc: Tensor,
        frame_pointers_brnc: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Helper used to properly shape object pointer tensors to match
        the image-like mask memory channel count, so they can be concatenated.

        Pointers themselves are simple vectors, but have 256 channels (by default)
        compared to the mask memory which has 64 channels (by default). To get a
        matching channel count, the channels of the pointers are broken into
        4 pieces, each with 64 channels. These are placed in the 'N' dimension,
        so an input pointer with a BxRxNxC shape: 1x6x1x256 becomes 1x6x4x64.

        This function only performs this shape change if needed, so the user
        can technically provide the reshaped pointers directly as input to
        prevent repeatedly doing this every update

        For the original SAM implementation, see:
        https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L639-L645

        Returns:
            prompt_pointers_brnc, frame_pointers_brnc (with 'NxC' shape adjusted if needed)
        """

        # Reshape pointers according to odd re-shaping rules (needed to match memory token channel counts: 64 vs. 256)
        # -> Here we assume pointer feature count is divisible by memory feature count (always true in practice)
        if prompt_pointers_brnc.shape[-1] != self._num_mem_features:
            ptr_b, ptr_r, ptr_n, ptr_c = prompt_pointers_brnc.shape
            c_factor = ptr_c // self._num_mem_features
            new_n = ptr_n * c_factor
            prompt_pointers_brnc = prompt_pointers_brnc.view(ptr_b, ptr_r, new_n, self._num_mem_features)
        if frame_pointers_brnc.shape[-1] != self._num_mem_features:
            ptr_b, ptr_r, ptr_n, ptr_c = frame_pointers_brnc.shape
            c_factor = ptr_c // self._num_mem_features
            new_n = ptr_n * c_factor
            frame_pointers_brnc = frame_pointers_brnc.view(ptr_b, ptr_r, new_n, self._num_mem_features)

        return prompt_pointers_brnc, frame_pointers_brnc

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

        # Learned embeddings per 'relative position in time', applied to memory encodings
        self.memposenc = MemoryTokenPositionEncoder(features_per_memory_token, max_memory_history)

        # Create model responsible for (non-learned) position encodings of object pointers
        ObjectPointerPosenc = ObjectPointerPosEnc_v2p1 if is_version_2p1 else ObjectPointerPosEnc_v2p0
        self.ptrposenc = ObjectPointerPosenc(features_per_image_token, features_per_memory_token)

    # .................................................................................................................

    def forward(
        self,
        token_hw: tuple[int, int],
        prompt_memory_brnc: Tensor,
        prompt_pointers_brnc: Tensor,
        frame_memory_brnc: Tensor,
        frame_pointers_brnc: Tensor,
        is_recent_first: bool,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Combines all prompt & previous frame memory data into a single
        tensor, along with a corresponding positional encoding tensor.
        The output can be thought of as a 'rows-of-tokens' formatted tensor,
        where the tokens are prior (image-like) mask memory encodings as
        well as prior (vector-like) object pointers, both of which are
        meant to be representations of the object that is being segmented.

        Inputs should be given as BxRxNxC shaped tensors, where
        B is the batch size, R is the number of frame entries
        (e.g. 1 for each encoded frame or prompt), N is the number
        of tokens for the given memory (e.g. N=H*W for image-like memory),
        and C is the channel dimension for the memory/pointers.

        Providing is_recent_first=True is used to indicate that the 0-th index of the
        given frame_memory_brnc (and pointers) is the most-recent encoding in time.
        If False, then it's assumed that the last-most entry is most recent.

        Returns:
            memory_tokens, memory_posenc, num_object_pointer_tokens
            -> Memory tokens have shape: BxNxC (B batch size, N number of tokens, C channels, 64 by default)
            -> Memory position encoding has shape: BxNxC (matching tokens shape)
            -> Number of object pointer tokens is very small compared to the number of memory tokens!
               The object pointer tokens are stored at the end of the memory_tokens tensor.
        """

        # Get position encoding for (image-like) memory tokens & pointers
        p_mem_posenc_brnc, f_mem_posenc_brnc = self.memposenc(
            token_hw, prompt_memory_brnc.shape, frame_memory_brnc.shape, is_recent_first
        )
        p_ptr_posenc_brnc, f_ptr_posenc_brnc = self.ptrposenc(
            prompt_pointers_brnc.shape, frame_pointers_brnc.shape, is_recent_first
        )

        # Count total pointers, since these get appended to memory output and need to be accounted for in attention
        # -> Given shapes: BxRxNxC, total comes from adding (R*N) for both inputs
        num_pmt_pointers = p_ptr_posenc_brnc.shape[1] * p_ptr_posenc_brnc.shape[2]
        num_frm_pointers = f_ptr_posenc_brnc.shape[1] * f_ptr_posenc_brnc.shape[2]
        num_pointer_entries = num_pmt_pointers + num_frm_pointers

        # Form final tensors by merging BxRxNxC -> BxN*xC (e.g. flatten RxN dimensions)
        memory_tokens_bnc = torch.concat(
            (
                prompt_memory_brnc.flatten(1, 2),
                frame_memory_brnc.flatten(1, 2),
                prompt_pointers_brnc.flatten(1, 2),
                frame_pointers_brnc.flatten(1, 2),
            ),
            dim=1,
        )
        memory_posenc_bnc = torch.concat(
            (
                p_mem_posenc_brnc.flatten(1, 2),
                f_mem_posenc_brnc.flatten(1, 2),
                p_ptr_posenc_brnc.flatten(1, 2),
                f_ptr_posenc_brnc.flatten(1, 2),
            ),
            dim=1,
        )

        return memory_tokens_bnc, memory_posenc_bnc, num_pointer_entries

    # .................................................................................................................


class MemoryTokenPositionEncoder(nn.Module):
    """
    Module used to compute spatial and temporal position encodings for
    the image-like memory tokens produced during video segmentation.

    The position encodings are made of both a non-learned spatial (e.g. 'per-pixel')
    encoding and an additive global encoding (e.g. applied equally to all 'pixels')
    For prompt memory, the global encoding is a special fixed encoding that's
    used to indicate that the memory comes from a prompt. For frame memory,
    there are several global encodings, each used to encode different relative
    positions in time (i.e. how 'long ago' was the memory encoded).

    This module does not exist in the original SAMv2 implementation. Instead computing the base
    positional encoding and adding offsets was handled in separate areas.
    The base positional encodings are generated inside the memory encoder itself:
    https://github.com/facebookresearch/segment-anything-2/blob/dce7b5446f260cef9fdf3d3f1bc81519302d386f/sam2/modeling/memory_encoder.py#L179
    While the offsets are added inside the '_prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/segment-anything-2/blob/dce7b5446f260cef9fdf3d3f1bc81519302d386f/sam2/modeling/sam2_base.py#L576-L578

    In this implementation, these are merged together here, since this is the only place they are used!
    """

    # .................................................................................................................

    def __init__(self, features_per_memory_token=64, max_memory_history=6):

        # Inherit from parent
        super().__init__()

        num_pos_offsets = 1 + max_memory_history
        self.base_memposenc_offsets = nn.Parameter(torch.zeros(num_pos_offsets, 1, 1, features_per_memory_token))
        self.posenc = SinusoidalPE2D(features_per_memory_token)
        self._max_pos_idx = max_memory_history - 1

        # Setup caches for holding pre-computed positional encodings with position offsets already added
        blank_cache = torch.empty((1, 1, 1, features_per_memory_token))
        self.register_buffer("cache_prompt_enc_11nc", blank_cache.clone(), persistent=False)
        self.register_buffer("cache_frame_enc_11nc", blank_cache.clone(), persistent=False)
        self._cache_hw = (1, 1)

    # .................................................................................................................

    def forward(
        self,
        token_hw: tuple[int, int],
        prompt_memory_shape_brnc: tuple[int, int, int, int],
        frame_memory_shape_brnc: tuple[int, int, int, int],
        is_recent_first: bool,
    ) -> tuple[Tensor, Tensor]:

        # For clarity
        pmt_b, pmt_r = prompt_memory_shape_brnc[0:2]
        frm_b, frm_r = frame_memory_shape_brnc[0:2]

        # Set up frame delta order, either: [0,1,2,3,...] for recent-first or [...,3,2,1,0] for recent-last
        # -> Each index is thought of as how 'far away' in time each memory entry is, 0 being most-recent
        device, idx_dtype = self.cache_frame_enc_11nc.device, torch.int64
        num_frame_entries = frame_memory_shape_brnc[1]
        if is_recent_first:
            frame_deltas = torch.arange(num_frame_entries, device=device, dtype=idx_dtype)
        else:
            oldest_delta = num_frame_entries - 1
            frame_deltas = torch.arange(oldest_delta, -1, -1, device=device, dtype=idx_dtype)
        frame_deltas_clamped = frame_deltas.clamp(0, self._max_pos_idx)

        # Re-generate cached encodings if we get a new image shape
        (in_h, in_w), (cache_h, cache_w) = token_hw, self._cache_hw
        if in_h != cache_h or in_w != cache_w:

            # Generate shared base encoding (shape 1xCxHxW) and convert to BxRxNxC shape (B & R are both 1)
            base_posenc_1chw = self.posenc(*token_hw)
            base_posenc_11nc = base_posenc_1chw.flatten(2).permute(0, 2, 1).unsqueeze(0)

            # Create prompt encoding
            prompt_posenc_111c = self.base_memposenc_offsets[[-1]]
            prompt_posenc_11nc = base_posenc_11nc + prompt_posenc_111c

            # Create base frame encoding (doesn't account for ordering yet)
            frame_posenc_111c = self.base_memposenc_offsets[0:-1, 0].unsqueeze(0)  # 1xN'x1xC (N'=6 by default)
            frame_posenc_111c = base_posenc_11nc + frame_posenc_111c

            # Cache for re-use
            self.cache_prompt_enc_11nc = prompt_posenc_11nc
            self.cache_frame_enc_11nc = frame_posenc_111c
            self._cache_hw = (in_h, in_w)

        # Duplicate base encodings to match batch/prompt counts & account for proper temporal ordering
        prompt_posenc_brnc = self.cache_prompt_enc_11nc.expand(pmt_b, pmt_r, -1, -1)
        frame_posenc_brnc = self.cache_frame_enc_11nc[:, frame_deltas_clamped].expand(frm_b, -1, -1, -1)

        return prompt_posenc_brnc, frame_posenc_brnc
