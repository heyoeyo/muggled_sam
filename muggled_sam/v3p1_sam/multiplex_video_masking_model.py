#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn

from .mask_decoder_model import SAMV3p1MaskDecoder
from .coordinate_encoder_model import SAMV3p1CoordinateEncoder

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3p1MultiplexVideoMasking(nn.Module):
    """
    Simplified implementation of the 'multiplex mask decoder' component described in the v3.1 update of:
        "SAM 3: Segment Anything with Concepts"
        By: Nicolas Carion∗, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, et al.
        @ https://arxiv.org/abs/2511.16719

    In v3.1, there are two similar, but not identical implementations of the mask decoder.
    One is the old/original mask decoder, used for interactive segmentation (e.g. SAMv1 task).
    In the original code, this is just referred to as the 'mask decoder':
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/sam/mask_decoder.py#L14

    A second mask decoder was added in the v3.1 update which is used specifically for
    video segmentation and supports multiplexed (e.g. batched) mask generation. This is
    what this class represent and is referred to as the 'multiplex mask decoder' in the original code:
    https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/multiplex_mask_decoder.py#L16

    To simplify usage and help distinguish between the two mask decoders, this multiplex
    implementation is actually a wrapper around a copy of the original mask decoder
    structure and a corresponding coord. encoder. This is made possible by the fact
    that this decoder does not accept prompt inputs.

    In the original implementation, the non-multiplex variant of the model generates
    4 masks but only returns either 1 or 3 masks, based on whether many prompts were
    given (return only 1 mask) or only 1 point or box prompt was given (return 3 masks).

    The multiplex variant generates 3 masks (instead of 4) but additionally generates
    16 copies of these to support multiplexing, for a total of 48 (!) masks. For the
    sake of compatibility between variants (and prior SAM implementations), the
    multiplex variant moves the 16 multiplex copies into the batch dimension on output.
    """

    # .................................................................................................................

    def __init__(
        self,
        input_channels: int = 256,
        downsample_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        num_mask_tokens: int = 3,
        num_output_masks: int = 4,
        multiplex_channels: int = 16,
    ):

        # Inherit from parent
        super().__init__()

        # Store target number of output masks for padding
        self._num_output_masks = num_output_masks
        self._enable_padding = num_mask_tokens < num_output_masks

        # Create a copy of the 'regular' mask decoder components to be used internally
        self.coordinate_encoder = SAMV3p1CoordinateEncoder(input_channels)
        self.mask_decoder = SAMV3p1MaskDecoder(
            input_channels, downsample_dim, num_layers, num_heads, num_mask_tokens, multiplex_channels
        )

        # Special 'empty' tensor used as the prompt encoding on videos
        self.register_buffer("no_prompt_encoding_bnc", torch.zeros((1, 0, input_channels)), persistent=False)

    # .................................................................................................................

    def forward(
        self,
        memory_fused_image_tokens_bchw: Tensor,
        hires_image_tokens_bchw: tuple[Tensor, Tensor],
        num_multiplex_objects: int = 1,
        return_all_multiplex_results: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generates multiple candidate segmentation masks given an image encoding and
        encoded prompts (which specific the part of the image to segment).
        Also returns estimates for the 'quality' of each segmentation mask.

        There are always 16 (by default) copies of each prediction in order to
        support 'multiplexing' (e.g. batched mask predictions). In many cases,
        not all 16 masks are being used (e.g. if fewer than 16 objects are being predicted).
        The 'return_all_multiplex_results' input can be set to True to return
        all 16 predictions (even if not using them), otherwise only the specified
        number of objects will be returned (this value also affects masking results!).

        Returns:
            mask_predictions, iou_predictions, object_pointers, object_score
            -> Mask prediction has shape: Bx4xHxW
            -> IoU has shape: Bx4
            -> Object pointer has shape: Bx4xF (F is features per token, default 256)
            -> Object score has shape Bx1 and indicates (score > 0) if an object is masked
            -> Mask H & W are 4x the size of the image patch encoding size
        """

        # Expand batch sizing of 'no prompt' encoding if needed
        img_b, _, img_h, img_w = memory_fused_image_tokens_bchw.shape
        no_prompt_encoding = self.no_prompt_encoding_bnc
        if img_b > 1:
            no_prompt_encoding = no_prompt_encoding.expand(img_b, -1, -1)

        # Run promptless mask decoder to generate MxN (16x3 by default) mask predictions
        grid_posenc = self.coordinate_encoder.get_grid_position_encoding((img_h, img_w))
        mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc, obj_score_m1 = self.mask_decoder(
            [memory_fused_image_tokens_bchw, *hires_image_tokens_bchw],
            no_prompt_encoding,
            grid_posenc,
            mask_hint=None,
            blank_promptless_output=False,
            num_multiplex_objects=num_multiplex_objects,
        )

        # Remove unused multiplexd entries if needed
        if not return_all_multiplex_results:
            # The decoder generates multiplexed outputs (shapes beginning with 16 entries, by default)
            # -> This is used for batched mask generation (called 'multiplexing'), introduced in SAMv3.1
            # -> We don't want to encode junk multiplex masks, so we remove them before memory encoding
            # -> For simplicity, we assume the first 'N' slots are in use and discard the remaining entries
            # -> This is referred to as 'demuxing' in the original code:
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L819-L822
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/multiplex_utils.py#L391
            mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc, obj_score_m1 = [
                data[:num_multiplex_objects] for data in (mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc, obj_score_m1)
            ]

        # Make sure output sizing matches target shape (e.g. sizing used in non-video segmentation)
        mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc = self._pad_outputs(mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc)

        return mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc, obj_score_m1

    # .................................................................................................................

    def _pad_outputs(
        self,
        mask_preds_mnhw: Tensor,
        iou_preds_mn: Tensor,
        obj_ptrs_mnc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        # Bail if we don't need padding
        if not self._enable_padding:
            return mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc

        # For clarity
        mask_m, mask_n, mask_h, mask_w = mask_preds_mnhw.shape
        device, dtype = mask_preds_mnhw.device, mask_preds_mnhw.dtype
        num_to_pad = max(0, self._num_output_masks - mask_n)

        # Create 'blank' entries to reach padding shape
        pad_masks = torch.full((mask_m, num_to_pad, mask_h, mask_w), -10.0, device=device, dtype=dtype)
        pad_ious = torch.zeros((mask_m, mask_n), device=device, dtype=dtype)
        pad_ptrs = torch.zeros((mask_m, mask_n, obj_ptrs_mnc.shape[-1]), device=device, dtype=dtype)

        # Pad to end so we don't interfere with original data indexing
        padded_masks_mnhw = torch.concat((mask_preds_mnhw, pad_masks), dim=1)
        padded_ious_mn = torch.concat((iou_preds_mn, pad_ious), dim=1)
        padded_ptrs_mnc = torch.concat((obj_ptrs_mnc, pad_ptrs), dim=1)
        return padded_masks_mnhw, padded_ious_mn, padded_ptrs_mnc

    # .................................................................................................................

    @staticmethod
    def get_best_mask_index(iou_predictions: Tensor) -> Tensor:
        """Helper used to select the index of the 'best' output, based on the highest IoU prediction score"""
        return torch.argmax(iou_predictions, dim=-1)

    # .................................................................................................................

    @staticmethod
    def get_best_decoder_results(
        mask_preds_mnhw: Tensor,
        iou_preds_mn: Tensor,
        obj_ptrs_mnc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Helper used to keep only the 'best' result from the mask decoder predictions,
        which always feature multiple 'guesses' to help deal with ambiguity.

        Returns:
            best_index, best_mask_1mhw, best_iou_1m, best_pointer_1mc
            -> Best index is a tensor (!) with shape: M (i.e. multiplex count)
            -> Mask prediction has shape: 1xMxHxW
            -> IoU has shape: 1xM
            -> Object pointer has shape: 1xMxC (C features, 256 by default)
        """

        # Each mask prediction contains multiple (3 by default) options, here we select which to use
        num_multiplex_objects = mask_preds_mnhw.shape[0]
        m_idx = torch.arange(num_multiplex_objects, device=iou_preds_mn.device)
        best_idx_mplex = torch.argmax(iou_preds_mn, dim=-1)

        # Index out best entries, but add back in a batch dimension to keep the same shape
        best_mask_1mhw = mask_preds_mnhw[m_idx, best_idx_mplex].unsqueeze(0)
        best_iou_pred_1m = iou_preds_mn[m_idx, best_idx_mplex].unsqueeze(0)
        best_objptr_1mc = obj_ptrs_mnc[m_idx, best_idx_mplex].unsqueeze(0)

        return best_idx_mplex, best_mask_1mhw, best_iou_pred_1m, best_objptr_1mc
