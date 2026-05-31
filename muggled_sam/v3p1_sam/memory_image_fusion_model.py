#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .components.memory_image_fusion_components import MemoryImageFusionTransformer
from .components.position_encoding import SinusoidalPE2D

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
        prompt_memory_encodings: list[tuple[Tensor, Tensor, Tensor]] | tuple[Tensor, Tensor, Tensor],
        frame_memory_encodings: list[tuple[Tensor, Tensor, Tensor]] | tuple[Tensor, Tensor, Tensor],
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
        Each memory entry is itself expected to be a 3-tuple of:
            frame_memory_1 = (prev_image_features_1, mask_memory_1, obj_pointer_1),
        which is output by the memory encoding functions of the model.

        As an alternative, it's also possible to provide inputs as tensors,
        in which case a single 3-tuple should be provided:
            (prev_image_tensor, mask_memory_tensor, obj_pointer_tensor)
        These tensors are meant to be stacked versions of the contents of the
        list of memory (tuples) input option. When providing tensors directly,
        ideally they should all have shape: BxRxNxC, where B is the batch size,
        R is the number of entries (e.g. length of the list), N is the number
        of tokens (e.g. N=H*W for image-like memory) and C is the channel count.
        Both the 'prev-image' and 'mask memory' inputs have image-like shapes
        by default (e.g. BxCxHxW), so the tensors can also be given as BxRxCxHxW
        (R stacked entries).

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
        p_img_brnc, p_mem_brnc, p_ptr_brnc, f_img_brnc, f_mem_brnc, f_ptr_brnc = self._prepare_memory_tensors(
            prompt_memory_encodings, frame_memory_encodings
        )

        # Merge all memory together and create corresponding position encoding tensor
        token_hw = lowres_image_tokens_bchw.shape[-2:]
        previmg_tokens_bnc, memory_tokens_bnc, memory_posenc_bnc, num_ptr_tokens = self.memconcat(
            token_hw, p_img_brnc, p_mem_brnc, p_ptr_brnc, f_img_brnc, f_mem_brnc, f_ptr_brnc, is_recent_first
        )

        # Fuse memory results into image tokens
        fused_img_tokens = self.fusion_transformer(
            lowres_image_tokens_bchw, previmg_tokens_bnc, memory_tokens_bnc, memory_posenc_bnc, num_ptr_tokens
        )

        return fused_img_tokens

    # .................................................................................................................

    def _prepare_memory_tensors(
        self,
        prompt_memory_encodings: list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor],
        frame_memory_encodings: list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Helper used to produce a 3-tuple of tensor data from inputs which may be
        given as a list of many 3-tuples of tensors.
        More specifically, prompt/frame memory encodings may be given as a
        list of:
            [(prev_img_1, mask_mem_1, ptr_1), (prev_img_2, mask_mem_2, ptr_2), (prev_img_3, mask_mem_3, ptr_3), etc...]
        And this function is used to convert each list into 3 tensors:
            all_prev_imgs, all_mask_mems, all_ptrs
        This is done for both the prompt & frame inputs, giving 6 tensors total.
        All outputs will have shapes like: BxRxNxC, where R is the number of entries
        in the list (i.e. separate memory encodings), and is generally different
        for prompt vs. frame memory. Note that both the prev-image and mask memory
        have default shapes like BxCxHxW, but are output as BxRxNxC, where N=H*W.

        It's also possible to directly provide the 3-tuple of tensors for each input,
        in which case nothing will be done.

        Returns:
            prompt_previmgs, prompt_mask_memory, prompt_pointers, frame_previmgs, frame_mask_memory, frame_pointer
            -> All entries have shape: BxRxNxC
            -> The 'R' shape corresponds to the list length (e.g. how many memory/frames were given)
        """

        # Convert list of prompt memory tuples into 3 tensors (prev-img feats, mask memory & object pointer)
        if not isinstance(prompt_memory_encodings[0], Tensor):
            pmt_img_tensor = torch.stack([data_tuple[0] for data_tuple in prompt_memory_encodings], dim=1)
            pmt_mem_tensor = torch.stack([data_tuple[1] for data_tuple in prompt_memory_encodings], dim=1)
            pmt_ptr_tensor = torch.stack([data_tuple[2] for data_tuple in prompt_memory_encodings], dim=1)
            prompt_memory_encodings = (pmt_img_tensor, pmt_mem_tensor, pmt_ptr_tensor)

        # Create dummy frame encodings if we're not given any, so rest of logic can work without conditional checks
        # -> We do this in a way that can handle mem inputs with shape BxRxCxHxW or BxRxNxC
        if len(frame_memory_encodings) == 0:
            ex_img, ex_mem, ex_ptr = prompt_memory_encodings
            empty_img = ex_mem.new_empty(ex_img.shape[0], 0, *ex_img.shape[2:])
            empty_mem = ex_mem.new_empty(ex_mem.shape[0], 0, *ex_mem.shape[2:])
            empty_ptr = ex_ptr.new_empty(ex_ptr.shape[0], 0, *ex_ptr.shape[2:])
            frame_memory_encodings = (empty_img, empty_mem, empty_ptr)

        # Here we form tensors from a list of frame memory as we did with prompt memory
        if not isinstance(frame_memory_encodings[0], Tensor):
            frm_img_tensor = torch.stack([data_tuple[0] for data_tuple in frame_memory_encodings], dim=1)
            frm_mem_tensor = torch.stack([data_tuple[1] for data_tuple in frame_memory_encodings], dim=1)
            frm_ptr_tensor = torch.stack([data_tuple[2] for data_tuple in frame_memory_encodings], dim=1)
            frame_memory_encodings = (frm_img_tensor, frm_mem_tensor, frm_ptr_tensor)

        # Unpack tensors for output
        p_img_tensor, p_mem_tensor, p_ptr_brnc = prompt_memory_encodings
        f_img_tensor, f_mem_tensor, f_ptr_brnc = frame_memory_encodings

        # If we get 5 dimensions, assume image-like shape: BxRxCxHxW and convert to: BxRxNxC
        p_img_brnc = p_img_tensor.flatten(3).permute(0, 1, 3, 2) if p_img_tensor.ndim == 5 else p_img_tensor
        p_mem_brnc = p_mem_tensor.flatten(3).permute(0, 1, 3, 2) if p_mem_tensor.ndim == 5 else p_mem_tensor
        f_img_brnc = f_img_tensor.flatten(3).permute(0, 1, 3, 2) if f_img_tensor.ndim == 5 else f_img_tensor
        f_mem_brnc = f_mem_tensor.flatten(3).permute(0, 1, 3, 2) if f_mem_tensor.ndim == 5 else f_mem_tensor

        # Sanity checks (mostly to catch errors with direct tensor inputs)
        assert p_img_brnc.ndim == 4, f"Need prompt prev-img shape: BxRxNxC, got: {p_img_brnc.shape}"
        assert p_mem_brnc.ndim == 4, f"Need prompt memory shape: BxRxNxC, got: {p_mem_brnc.shape}"
        assert p_ptr_brnc.ndim == 4, f"Need prompt pointers shape: BxRxNxC, got: {p_ptr_brnc.shape}"
        assert f_img_brnc.ndim == 4, f"Need frame prev-img shape: BxRxNxC, got: {f_img_brnc.shape}"
        assert f_mem_brnc.ndim == 4, f"Need frame memory shape: BxRxNxC, got: {f_mem_brnc.shape}"
        assert f_ptr_brnc.ndim == 4, f"Need frame pointers shape: BxRxNxC, got: {f_ptr_brnc.shape}"

        return p_img_brnc, p_mem_brnc, p_ptr_brnc, f_img_brnc, f_mem_brnc, f_ptr_brnc

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

        # Learned embeddings per 'relative position in time', applied to memory encodings
        self.memposenc = MemoryTokenPositionEncoder(features_per_memory_token, max_memory_history)

        # Create model responsible for position encodings of object pointers
        self.ptrposenc = ObjectPointerPosEnc(features_per_image_token, features_per_memory_token)

    # .................................................................................................................

    def forward(
        self,
        token_hw: tuple[int, int],
        prompt_previmg_brnc: Tensor,
        prompt_memory_brnc: Tensor,
        prompt_pointers_brnc: Tensor,
        frame_previmg_brnc: Tensor,
        frame_memory_brnc: Tensor,
        frame_pointers_brnc: Tensor,
        is_recent_first: bool,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """
        Combines all prompt & previous frame memory data into a single
        tensor, along with a corresponding positional encoding tensor.
        The output can be thought of as a 'rows-of-tokens' stack of the
        given 'previous-image features', along with the (image-like)
        mask memory encodings combined with the (vector-like) object
        pointers, both of which are meant to be representations of the
        object that is being segmented.

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

        # Form BxNxC from previous-image tokens with padding to match memory size (which has pointers)
        # https://github.com/facebookresearch/sam3/blob/967fdd651f71ca14949122fed4c918a778ca9334/sam3/model/decoder.py#L1321-L1332
        mem_b, _, mem_c = memory_tokens_bnc.shape
        device, dtype = memory_tokens_bnc.device, memory_tokens_bnc.dtype
        previmg_padding_bnc = torch.zeros((mem_b, num_pointer_entries, mem_c), device=device, dtype=dtype)
        previmg_tokens_bnc = torch.concat(
            (
                prompt_previmg_brnc.flatten(1, 2),
                frame_previmg_brnc.flatten(1, 2),
                previmg_padding_bnc,
            ),
            dim=1,
        )

        return previmg_tokens_bnc, memory_tokens_bnc, memory_posenc_bnc, num_pointer_entries

    # .................................................................................................................


class MemoryTokenPositionEncoder(nn.Module):
    """
    Helper module used to pre-compute & cache image-like positional encodings meant
    for use with 'memory encoding' tokens, used within the memory fusion steps of the SAMv3 model.

    The positional encodings for 'past memories' include an additive offset/embedding, which
    is a learned value and is different depending on how 'far away' the memory is, relative to
    the frame where it is being used. While these offsets are learned, the underlying 'base'
    positional encoding is fixed for a given image height & width. As a result, it's possible
    to pre-compute the result of adding each of the learned offsets to the fixed base encoding,
    which is what the model does (and caches the result for re-use).

    This module does not exist in the original SAMv3 implementation. Instead computing the base
    positional encoding and adding offsets was handled in separate areas.
    The base positional encodings are generated inside the memory encoder itself:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/memory.py#L207
    While the offsets are added inside the '_prepare_memory_conditioned_features' function:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1434-L1437

    In this implementation, these are merged together here, since this is the only place they are used!
    """

    # .................................................................................................................

    def __init__(self, features_per_memory_token: int = 256, max_memory_history: int = 6):

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

        # Duplicate encodings to match batch/prompt counts & account for proper temporal ordering
        prompt_posenc_brnc = self.cache_prompt_enc_11nc.expand(pmt_b, pmt_r, -1, -1)
        frame_posenc_brnc = self.cache_frame_enc_11nc[:, frame_deltas_clamped].expand(frm_b, -1, -1, -1)

        return prompt_posenc_brnc, frame_posenc_brnc

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
        prompt_pointers_shape_brnc: tuple[int, int, int, int],
        frame_pointers_shape_brnc: tuple[int, int, int, int],
        is_recent_first: bool,
    ) -> tuple[Tensor, Tensor]:
        """
        Function which computes position encodings for object pointers.
        There are several deviations from the original implementation here, in order to
        simplify things. Though this leads to numerical differences, the effect is
        negligible in practice (pointers don't do very much).

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
        # https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_utils.py#L327
        pos_embed_base = pos_norm_tensor * self.posenc_scale_factor
        pos_embed = torch.cat([pos_embed_base.sin(), pos_embed_base.cos()], dim=-1)

        # Apply projection to reduce image token channel count to memory token channel count
        # -> Result has shape: 1xNxC, where N is number of pointers, C features per memory token (64 by default)
        return self.pointer_pos_proj(pos_embed)
