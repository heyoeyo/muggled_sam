#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp
from time import perf_counter
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# For type hints
from torch import Tensor
from numpy import ndarray
from .image_encoder_model import SAMV3p1ImageEncoder
from .image_projection_model import SAMV3p1ImageProjection
from .coordinate_encoder_model import SAMV3p1CoordinateEncoder
from .prompt_encoder_model import SAMV3p1PromptEncoder
from .mask_decoder_model import SAMV3p1MaskDecoder
from .memory_encoder_model import SAMV3p1MemoryEncoder
from .memory_image_fusion_model import SAMV3p1MemoryImageFusion
from .text_encoder_model import SAMV3p1TextEncoder
from .sampling_encoder import SAMV3p1SamplingEncoder
from .image_exemplar_fusion_model import SAMV3p1ImageExemplarFusion
from .exemplar_detector_model import SAMV3p1ExemplarDetector
from .exemplar_segmentation_model import SAMV3p1ExemplarSegmentation


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3p1Model(nn.Module):
    """
    Wrapper around separated SAMv3.1 model components, so that the model can be used as a singular entity.

    The original SAMv3 (e.g. v3.0) consists of 2 separate models, one for 'interactive'
    segmentation and one for multi-object detection + segmentation.

    In version 3.1, the model is split into 3 distinct components, corresponding to interactive
    segmentation (e.g. SAMv1 task), video segmentation (e.g SAMv2 task) and detection (SAMv3 task).
    All tasks share the same base image encoder, but each uses different projections of the
    encoded image tokens and generally each has fully separate components.
    Internally, the v1/v2 tasks are denoted by 'image' vs. 'video', for example:
        'image_coordinate_encoder' vs. 'video_coordinate_encoder'

    These separate models lead to very different use cases and as such, the functionality of
    this model comes from calling specific methods for different use cases
    (unlike typical models where the .forward(...) method contains all functionality).
    """

    name = "samv3"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV3p1ImageEncoder,
        image_projection_model: SAMV3p1ImageProjection,
        image_coordinate_encoder_model: SAMV3p1CoordinateEncoder,
        video_coordinate_encoder_model: SAMV3p1CoordinateEncoder,
        prompt_encoder_model: SAMV3p1PromptEncoder,
        image_mask_decoder_model: SAMV3p1MaskDecoder,
        video_mask_decoder_model: SAMV3p1MaskDecoder,
        memory_encoder_model: SAMV3p1MemoryEncoder,
        memory_image_fusion_model: SAMV3p1MemoryImageFusion,
        text_encoder_model: SAMV3p1TextEncoder,
        sampling_encoder_model: SAMV3p1SamplingEncoder,
        image_exemplar_fusion_model: SAMV3p1ImageExemplarFusion,
        exemplar_detector_model: SAMV3p1ExemplarDetector,
        exemplar_segmentation_model: SAMV3p1ExemplarSegmentation,
        config_bytes: bytearray,
    ):

        # Inherit from parent
        super().__init__()

        # Store config data
        self.register_buffer("config_muggled_samv3p1", torch.tensor(config_bytes, dtype=torch.uint8))

        # Store base SAM components
        self.image_encoder = image_encoder_model
        self.image_projection = image_projection_model
        self.image_coordinate_encoder = image_coordinate_encoder_model
        self.video_coordinate_encoder = video_coordinate_encoder_model
        self.prompt_encoder = prompt_encoder_model
        self.image_mask_decoder = image_mask_decoder_model
        self.video_mask_decoder = video_mask_decoder_model

        # Video tracking components caried over from SAMv2
        self.memory_encoder = memory_encoder_model
        self.memory_image_fusion = memory_image_fusion_model

        # Components specific to V3
        self.text_encoder = text_encoder_model
        self.sampling_encoder = sampling_encoder_model
        self.image_exemplar_fusion = image_exemplar_fusion_model
        self.exemplar_detector = exemplar_detector_model
        self.exemplar_segmentation = exemplar_segmentation_model

        # Default to eval mode, expecting to use inference only
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        self._infmode = True

    # .................................................................................................................

    def forward(
        self,
        image_rgb_normalized_bchw: Tensor,
        boxes_tensor: Tensor,
        fg_tensor: Tensor,
        bg_tensor: Tensor,
        mask_hint: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        This function implements the full 'direct prompt' mask generating code
        bundled into a single function. Takes an image and set of prompts and
        produces several candidate segmentation masks.

        Note that in practice, it makes more sense to call the component pieces of the model,
        rather than using this function so that image & prompt encoding can happen independently.
        See the 'encode_prompts', 'encode_image' and 'generate_masks' functions for more info

        Returns:
            mask_predictions, iou_predictions, objscore_pred, mask_tokens_out
        """

        # Encode prompts & image inputs
        box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
        encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)
        encoded_image = self.image_encoder(image_rgb_normalized_bchw)

        # Combine encodings to generate mask output
        patch_grid_hw = encoded_image.shape[2:]
        grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
        mask_preds, iou_preds, objscore_pred, mask_tokens_out = self.image_coordinate_encoder(
            encoded_image, encoded_prompts, grid_posenc, mask_hint
        )

        return mask_preds, iou_preds, objscore_pred, mask_tokens_out

    # .................................................................................................................

    def encode_prompts(self, box_tlbr_norm_list: list, fg_xy_norm_list: list, bg_xy_norm_list: list) -> Tensor:
        """
        Function used to encode prompt coordinates. Inputs should be given as lists
        of prompts. The length of each list does not need to match. Enter either
        None or an empty list ([]) to disable any of the prompts.

        Box prompt formatting:
            Each entry should be in top-left/bottom-right form: ((x1, y1), (x2, y2))
            For example:
                [
                    [(0.1, 0.5), (0.3, 0.7)], # Box 1
                    [(0.6, 0.2), (0.8, 0.4)], # Box 2
                    ... etc ...
                ]

        FG/BG prompt formatting:
            Each entry should be a single (x, y) point
            For example:
                [
                    (0.2, 0.6), # Point 1
                    (0.5, 0.4), # Point 2
                    (0.7, 0.7), # Point 3
                    (0.1, 0.9), # Point 4
                    ... etc ..
                ]

        Returns:
            encoded_prompts (shape: 1 x N x F, where N is number of prompt points, F is features per prompt)
        """

        with _inference_mode(self._infmode):
            boxes_tensor = self.image_coordinate_encoder.prepare_boxes(box_tlbr_norm_list)
            fg_tensor, bg_tensor = self.image_coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.image_coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts

    # .................................................................................................................

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[tuple[list[Tensor], list[Tensor], list[Tensor]], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The max_side_length setting is used to set the size at which the image is processed, if
        no value is given, the image will be scaled to the 'preferred' size of the model.
        The use_square_sizing setting determines whether the image is scaled to a square resolution
        or scaled (to the max_side_length) based on it's original aspect ratio.

        Returns:
            encoded_images_list, patch_grid_hw, preencoded_image_hw
            -> Encoded images list contains three separate entries,
               which correspond to image encodings for the SAMv1/v2/v3 tasks.
               Each entry itself contains 3 tensors which are multi-resolution
               image features. The 0th entry is always the lowest resolution
               map (72x72 by default) with the 1st and 2nd index entries
               being 2x and 4x larger
            -> The patch_grid_hw contains the height & width of the low-res
               feature map (72x72 with default 1008x1008 input sizing)
            -> The preencoded_image_hw contains the height & width of the
               input image after pre-processing just before being encoded,
               by default it would be 1008x1008
        """

        with _inference_mode(self._infmode):
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            encoded_img = self.image_encoder(image_rgb_normalized_bchw)
            encoded_image_features_list = self.image_projection(encoded_img)

        # Get patch sizing of lowest-res tokens (as needed by other components) & size of processed image
        patch_grid_hw = encoded_image_features_list[0][0].shape[2:]
        image_preenc_hw = image_rgb_normalized_bchw.shape[2:]

        return encoded_image_features_list, patch_grid_hw, image_preenc_hw

    # .................................................................................................................

    def generate_masks(
        self,
        encoded_image_features_list: list[Tensor],
        encoded_prompts: Tensor,
        mask_hint: Tensor | int | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Function used to generate segmentation masks given an image encoding,
        as well as a prompt encoding and potentially a mask hint/prompt. These
        input encodings are expected to come from other model components.

        The mask hint can either be None (no mask input), an integer or a
        tensor. If an integer is given, this is interpreted to mean that the
        model should run twice, once to get a set of mask predictions and then
        a second time, where the mask_hint (as integer) mask is chosen to be
        used as a mask hint for a second run of the model. The idea being to
        use the models own mask output to refine itself. If a tensor is given,
        it is assumed to be a mask itself. It should be shaped to match the
        model's own output masks for the given input image size, by default
        this would be a shape of: Bx1x288x288

        Returns:
            mask_predictions, iou_predictions
            -> Masks have shape: Bx4xHxW (HxW is 288x288 using default settings)
            -> iou_predictions have shape: Bx4
            -> Batch dimension (B) is also used for 'multiplexing' introduced in v3.1
               (though for non-video use, multiplexing isn't used)
        """

        # Get sizing of the lowest-resolution (v1/interactive) image encoding
        v1_encimgs = encoded_image_features_list[0]
        patch_grid_hw = v1_encimgs[0].shape[2:]

        with _inference_mode(self._infmode):
            grid_posenc = self.image_coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            mask_preds_mnhw, iou_preds_mn, _, _ = self.image_mask_decoder(
                v1_encimgs, encoded_prompts, grid_posenc, mask_hint, blank_promptless_output
            )

        return mask_preds_mnhw, iou_preds_mn

    # .................................................................................................................

    def initialize_video_masking(
        self,
        encoded_image_features_list: list[Tensor],
        box_tlbr_norm_list: list,
        fg_xy_norm_list: list,
        bg_xy_norm_list: list,
        mask_hint: Tensor | int | None = None,
        mask_index_select: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Creates initial 'prompt memory' for segmenting objects through a video.
        Similar to calling the prompt encoder & mask decoder, however, this function
        outputs a single mask prediction along with a memory encoding and
        object pointer, both of which must be passed along to the per-frame video
        masking function.

        If a 'mask_index_select' isn't given, then the 'best' mask will be chosen automatically.

        This function roughly corresponds to `add_new_points` in the SAMv3.1 implementation:
        https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex_demo.py#L1249

        Returns:
            best_mask_prediction, memory_encoding, object_pointer
        """

        # Encode initial prompts
        encoded_prompts = self.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

        with _inference_mode(self._infmode):

            # For convenience
            v1_encimgs, v2_encimgs, _ = encoded_image_features_list
            v1_lowres_imgenc = v1_encimgs[0]
            token_hw = v1_lowres_imgenc.shape[2:]
            device, dtype = v1_lowres_imgenc.device, v1_lowres_imgenc.dtype

            # Generate mask prediction from image/prompt encodings, as usual
            grid_posenc = self.image_coordinate_encoder.get_grid_position_encoding(token_hw)
            mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc, obj_score_m1 = self.image_mask_decoder(
                v1_encimgs,
                encoded_prompts,
                grid_posenc,
                mask_hint=mask_hint,
                blank_promptless_output=False,
            )

            # Fill in missing index with 'best' based on IoU
            if mask_index_select is None:
                mask_index_select = self.video_mask_decoder.get_best_mask_index(iou_preds_mn)

            # Make sure we have a slice/tensor-style index
            if isinstance(mask_index_select, int):
                mask_index_select = torch.tensor([mask_index_select], dtype=torch.int64, device=device)

            # For clarity
            v2_lowres_imgenc = v2_encimgs[0]
            best_mask_pred_mnhw = mask_preds_mnhw[:, mask_index_select, :, :]
            is_prompt_enc = True

            # Encode new memory features
            memory_encoding = self.memory_encoder(v2_lowres_imgenc, best_mask_pred_mnhw, obj_score_m1, is_prompt_enc)

            # Add missing 'multiplex' entries to pointer (image mask decoder doesn't generate these!)
            best_obj_ptr_bmc = obj_ptrs_mnc[:, mask_index_select, :].squeeze(1).unsqueeze(0)
            ptr_b, ptr_m, ptr_c = best_obj_ptr_bmc.shape
            pad_ptr_bmc = torch.zeros((ptr_b, 16 - ptr_m, ptr_c), device=device, dtype=dtype)
            best_obj_ptr_bmc = torch.concat((best_obj_ptr_bmc, pad_ptr_bmc), dim=1)

        # v3.1 uses both the memory encoding & low-res frame tokens, so bundle them together
        bundled_mem_v3p1 = (v2_lowres_imgenc, memory_encoding)
        return best_mask_pred_mnhw, bundled_mem_v3p1, best_obj_ptr_bmc

    # .................................................................................................................

    def initialize_from_mask(
        self,
        encoded_image_features_list: list[Tensor],
        mask_image: ndarray | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Alternate video tracking initialization option. In this case, using a provided mask image as a 'prompt'
        to begin tracking an object.

        The provided mask can either be a numpy array (e.g. image loaded using opencv) or
        a pytorch tensor (e.g. output from another model).

        - If a mask is given with 3 dimensions, but the last dimension has size <= 3, it's assumed
          to be shaped as: HxWxC (e.g. a BGR image from opencv). In this case, the 0-th entry
          of the 3rd dimension will be taken as the (HxW) mask
        - If a mask is given with 3 dimensions, but the last dimension has size > 3, it's assumed
          to be shaped as: BxHxW
        - If a mask is given with 4 dimensions, it will be interpretted as: Bx1xHxW
        - The batch dimension 'B' is used for multiplexing (a v3.1 specific feature)

        Returns:
            memory_encoding, object_pointer
        """

        with _inference_mode(self._infmode):

            # For convenience
            v1_encimgs, v2_encimgs, _ = encoded_image_features_list
            v2_lowres_imgenc = v2_encimgs[0]
            device, dtype = v2_lowres_imgenc.device, v2_lowres_imgenc.dtype
            img_b, img_c, token_h, token_w = v2_lowres_imgenc.shape

            # Force input into a boolean mask & convert to torch tensor
            if isinstance(mask_image, ndarray):
                mask_tensor = torch.tensor(mask_image, device=device, dtype=dtype)
                mask_min, mask_max = mask_tensor.min(), mask_tensor.max()
                mask_tensor = (mask_tensor - mask_min) / (mask_max - mask_min)
                mask_tensor = mask_tensor * 2048 - 1024
            elif isinstance(mask_image, Tensor):
                mask_tensor = mask_image.to(device=device, dtype=dtype)
            assert isinstance(mask_tensor, Tensor), "Unsupported mask type! Must be a numpy array or torch tensor"

            # Make sure we get a mask with shape: BxNxHxW
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  #  HxW -> 1x1xHxW
            elif mask_tensor.ndim == 3 and mask_tensor.shape[-1] <= 3:
                mask_tensor = mask_tensor[:, :, 0].unsqueeze(0).unsqueeze(0)  # HxWxC -> 1x1xHxW
            elif mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(1)  # BxHxW -> Bx1xHxW
            assert mask_tensor.ndim == 4, "Unsupported mask shape, must be: HxW, HxWxC, BxHxW or Bx1xHxW"

            # Make sure we get N = 1
            mask_b, mask_n, mask_h, mask_w = mask_tensor.shape
            if mask_b == 1 and mask_n > 1:
                mask_tensor = mask_tensor.permute(1, 0, 2, 3)
                mask_b, mask_n, mask_h, mask_w = mask_tensor.shape
            assert mask_n == 1, "Mask shape error! Expecting '1' in shape index 1, eg. Bx1xHxW"

            # Encode new memory features
            # Called '_encode_new_memory' in original code
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1616
            obj_score_b1 = torch.full((mask_b, 1), 10.0, device=device, dtype=dtype)
            memory_encoding = self.memory_encoder(v2_lowres_imgenc, mask_tensor, obj_score_b1, is_prompt_encoding=True)
            bundled_mem_v3p1 = (v2_lowres_imgenc, memory_encoding)

            # Build a 'blank' pointer, since we don't get one without running the mask decoder
            mem_b, mem_c, _, _ = memory_encoding.shape
            blank_ptr_bmc = torch.zeros((1, 16, mem_c), device=device, dtype=dtype)

        return bundled_mem_v3p1, blank_ptr_bmc

    # .................................................................................................................

    def step_video_masking(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor], list[Tensor]],
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
        num_multiplex_objects: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Function which makes segmentation predictions for consecutive frames of
        an input video. It takes in the encoded video frame data along with prior
        prompt/previous frame memory data in order to automatically continue
        segmenting some existing object (i.e. without requiring user prompts).

        When multiplexing is being used, the 'num_multiplex_objects' input should
        be provided to indicate how many objects are represented in the given memory encodings.

        This function corresponds to 'track_step' in the SAMv3.1 implementation:
        https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L2528

        Returns:
            object_score, best_mask_index, mask_predictions, memory_encoding, best_object_pointer
            -> object_score score is an indicator of whether an object is present
               values below 0 indicate lost tracking. Has shape: Mx1 (M multiplex outputs)
            -> best_mask_index is the index of the highest iou score. Has shape: M (one index for each multiplex mask)
            -> mask_predictions are the same as with image segmentation, has shape: Mx4xHxW
            -> memory_encoding should be passed back in on future frames, has shape: BxFxH'xW'
            -> best_object_pointer should be passed back in on future frames, has shape: 1xMxF

            The HxW of masks will be 1/4 of the input height and width (256x256 for default 1024 sizing).
            The memory encoding H'xW' is 4 times smaller than the mask sizing (64x64 by default).
            The memory & pointer features F are model configs (256 by default)
        """

        with _inference_mode(self._infmode):

            # Encode image features with previous memory encodings & object pointer data
            # Called '_prepare_memory_conditioned_features' in original code
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1289
            v2_encimgs = encoded_image_features_list[1]
            lowres_imgenc, *hires_imgenc = v2_encimgs
            memfused_encimg = self.memory_image_fusion(
                lowres_imgenc,
                prompt_memory_encodings,
                prompt_object_pointers,
                previous_memory_encodings,
                previous_object_pointers,
            )

            # Run mask decoder on memory-fused features
            # Called '_forward_sam_heads' in original code (specifically the 'Multiplexed propagation path' branch)
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L785-L787
            patch_grid_hw = memfused_encimg.shape[-2:]
            grid_posenc = self.video_coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            mask_preds_mnhw, iou_preds_mn, obj_ptrs_mnc, obj_score_m1 = self.video_mask_decoder(
                [memfused_encimg, *hires_imgenc],
                self.prompt_encoder.create_video_no_prompt_encoding(),
                grid_posenc,
                mask_hint=None,
                blank_promptless_output=False,
                num_multiplex_objects=num_multiplex_objects,
            )

            # The video mask decoder generates a 'multiplexed mask' output (16x3x288x288 masks by default)
            # -> This is used for batched mask generation (called 'multiplexing'), introduced in SAMv3.1
            # -> We don't want to encode junk multiplex masks, so we remove them before memory encoding
            # -> This is referred to as 'demuxing' in the original code:
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L819-L822
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/multiplex_utils.py#L391
            total_multiplex_count = mask_preds_mnhw.shape[0]
            mask_preds_mnhw = mask_preds_mnhw[:num_multiplex_objects]
            iou_preds_mn = iou_preds_mn[:num_multiplex_objects]
            obj_ptrs_mnc = obj_ptrs_mnc[:num_multiplex_objects]
            obj_score_m1 = obj_score_m1[:num_multiplex_objects]

            # Each mask prediction contains multiple (3 by default) options, here we select which to use
            m_idx = torch.arange(num_multiplex_objects, device=iou_preds_mn.device)
            best_idx = self.video_mask_decoder.get_best_mask_index(iou_preds_mn)

            # For clarity
            best_mask_1mhw = mask_preds_mnhw[m_idx, best_idx].unsqueeze(0)
            best_objptr_1mc = obj_ptrs_mnc[m_idx, best_idx].unsqueeze(0)
            is_prompt_enc = False

            # Encode new memory features
            # Called '_encode_new_memory' in original code
            # https://github.com/facebookresearch/sam3/blob/bfbed072a07a6a52c8d5fdc75a7a186251a835b1/sam3/model/video_tracking_multiplex.py#L1616
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_1mhw, obj_score_m1, is_prompt_enc)
            bundled_mem_v3p1 = (lowres_imgenc, memory_encoding)

            # Pad pointer back to fit 'multiplex' shape (needed for memory-image-fusion)
            # -> It may make more sense to do this padding on the memory-image-fusion model,
            #    however, these values are stored and re-used repeatedly. Doing the padding
            #    here means it only needs to be done once (rather than everytime it's used elsewhere)
            device, dtype = best_objptr_1mc.device, best_objptr_1mc.dtype
            ptr_b, ptr_m, ptr_c = best_objptr_1mc.shape
            pad_ptr = torch.zeros((ptr_b, total_multiplex_count - ptr_m, ptr_c), device=device, dtype=dtype)
            best_objptr_1mc = torch.concat((best_objptr_1mc, pad_ptr), dim=1)

            # Pad masks to have a 4th (blank) entry for inter-operability with SAM v2/v3.0 and interactive-mode
            mask_m, _, mask_h, mask_w = mask_preds_mnhw.shape
            pad_masks = torch.full((mask_m, 1, mask_h, mask_w), -10.0, device=device, dtype=dtype)
            mask_preds_mnhw = torch.concat((mask_preds_mnhw, pad_masks), dim=1)

        return obj_score_m1, best_idx, mask_preds_mnhw, bundled_mem_v3p1, best_objptr_1mc

    # .................................................................................................................

    def get_best_mask_index(self, iou_predictions: Tensor) -> int:
        """Returns the index of the highest IoU prediction score"""
        return self.image_coordinate_encoder.get_best_mask_index(iou_predictions)

    # .................................................................................................................

    def check_have_prompts(self, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list) -> bool:
        """Helper used to check if there are any prompts"""
        return self.prompt_encoder.check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

    # .................................................................................................................

    def make_detector_model(self, bpe_vocab_path: str | None = None) -> nn.Module:
        """Creates 'detection' model, used for generating bounding box & segmentation masks from exemplars"""
        detector_model = SAMV3DetectorModel(
            self.image_encoder,
            self.image_projection,
            self.text_encoder,
            self.sampling_encoder,
            self.image_exemplar_fusion,
            self.exemplar_detector,
            self.exemplar_segmentation,
            bpe_vocab_path,
        )
        detector_model.toggle_inference_mode(self._infmode)
        return detector_model

    # .................................................................................................................

    def toggle_inference_mode(self, enable_inference_mode: bool | None = None) -> bool:
        """
        Helper used to toggle internal 'with torch.inference_mode' on/off.
        When training the model, inference mode can become problematic, so disabling it can be helpful.
        If 'enable_inference_mode' is None, then the current state will be toggled.
        Otherwise, the state can be explicitly set by providing a True or False argument.
        Returns: is_inference_mode_enabled
        """
        self._infmode = not self._infmode if enable_inference_mode is None else enable_inference_mode
        return self._infmode

    # .................................................................................................................


class SAMV3DetectorModel(nn.Module):
    """
    Wrapper around SAMv3 detection components. These components are almost entirely
    separate from the components used for the image/video segmentation found in SAMv1/SAMv2
    (with the exception of the image encoder). This model exists to help avoid overlap with
    similar looking functionality from the v1/v2 components.

    Initializing this model also loads a BPE vocabulary from the file system, needed by the text encoder.

    The functions of this model (roughly) correspond with the original code base as follows:
        .encode_detection_image -> Sam3Processor.set_image
        .encode_exemplars       -> Sam3Processor.set_text_prompt + Sam3Image._encode_prompt + Sam3Image._run_encoder
        .generate_detections    -> Sam3Image._run_decoder + Sam3Image._run_segmentation_heads
    For more context, see the 'forward_grounding' function from the original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/sam3_image.py#L442
    """

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV3p1ImageEncoder,
        image_projection_model: SAMV3p1ImageProjection,
        text_encoder_model: SAMV3p1TextEncoder,
        sampling_encoder_model: SAMV3p1SamplingEncoder,
        image_exemplar_fusion_model: SAMV3p1ImageExemplarFusion,
        exemplar_detector_model: SAMV3p1ExemplarDetector,
        exemplar_segmentation_model: SAMV3p1ExemplarSegmentation,
        bpe_vocab_path: str | None = None,
    ):
        # Inherit from parent
        super().__init__()

        # Copy over detector components
        self.image_encoder = image_encoder_model
        self.image_projection = image_projection_model
        self.text_encoder = text_encoder_model
        self.sampling_encoder = sampling_encoder_model
        self.image_exemplar_fusion = image_exemplar_fusion_model
        self.exemplar_detector = exemplar_detector_model
        self.exemplar_segmentation = exemplar_segmentation_model

        # Fill in default vocab path
        if bpe_vocab_path is None:
            parent_folder_path = osp.dirname(__file__)
            bpe_vocab_path = osp.join(parent_folder_path, "resources", "samv3_bpe_vocab_table.txt.lzma")

        # Try to load vocab file on init, so user doesn't have to remember to do it later...
        if osp.exists(bpe_vocab_path):
            self.text_encoder.load_bpe_vocab(bpe_vocab_path)
        else:
            warning_txt_list = [
                "Unable to load BPE vocabulary table!",
                f"Couldn't find file: {bpe_vocab_path}",
                "The vocab table can be loaded manually using:",
                ".text_encoder.load_bpe_vocab('/path/to/bpe.txt.gz')",
            ]
            raise Warning("\n".join(warning_txt_list))

        # Default to eval mode, expecting to use inference only
        self.eval()
        self._infmode = True

    # .................................................................................................................

    def encode_detection_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The max_side_length setting is used to set the size at which the image is processed, if
        no value is given, the image will be scaled to the 'preferred' size of the model.
        The use_square_sizing setting determines whether the image is scaled to a square resolution
        or scaled (to the max_side_length) based on it's original aspect ratio.

        IMPORTANT:
            This function produces *different* results from the .encode_image(...)
            function that is part of the base model, though the outputs have the same structure.

        Returns:
            encoded_detection_images_list, patch_grid_hw, preencoded_image_hw
            -> Encoded detection images list contains 3 multi-resolution feature maps
               they have shapes: Bx256x72x72, Bx256x144x144, Bx256x288x288 (using default settings)
            -> The patch_grid_hw contains the height & width of the low-res
               feature map (72x72 with default 1008x1008 input sizing)
            -> The preencoded_image_hw contains the height & width of the
               input image after pre-processing just before being encoded,
               by default it would be 1008x1008
        """

        with _inference_mode(self._infmode):
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            encoded_img = self.image_encoder(image_rgb_normalized_bchw)
            encoded_image_features_list = self.image_projection(encoded_img)

        # Get patch sizing of lowest-res tokens (as needed by other components) & size of processed image
        patch_grid_hw = encoded_image_features_list[0][0].shape[2:]
        image_preenc_hw = image_rgb_normalized_bchw.shape[2:]

        return encoded_image_features_list, patch_grid_hw, image_preenc_hw

    # .................................................................................................................

    def encode_exemplars(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor], list[Tensor]],
        text: str | None = None,
        box_xy1xy2_norm_list: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        point_xy_norm_list: list[tuple[float, float]] | None = None,
        negative_boxes_list: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        negative_points_list: list[tuple[float, float]] | None = None,
        include_coordinate_encodings: bool = True,
    ) -> Tensor:
        """
        Function used to encode text & coordinate inputs into a single
        set of 'exemplar tokens'. These act like prompts for the detection model.
        The output of this model is not useful by itself, but is needed as
        an input into the .generate_detection(...) function

        - Box inputs should be lists of normalized [(x1,y1), (x2,y2)] pairs, eg:
            boxes = [[(0.1,0.5), (0.4, 0.7)], ...]
        - Point inputs should be lists of normalized (x1,y1) pairs, eg:
            points = [(0.5,0.75), ...]
        - Any inputs left as 'None' will not be used in the encoding

        Setting the 'include_coordinate_encodings' flag to false will
        force coordinates encodings to only make use of the associated
        image data at the given point/box, and not the coordinates
        themselves. This will generally degrade performance, but may
        be useful in cases where the positioning is not desirable
        for segmentation results, for example, when using exemplars
        from one image to segment a different image.

        Returns:
            exemplar_tokens_bnc
            -> Shape is BxNxC, B batches, N number of tokens, C channels
            -> The number of tokens will vary based on the provided inputs, generally
               1 token per point, 2 tokens per box and 3+ tokens for text
        """

        # For clarity
        v3_encimgs = encoded_image_features_list[-1]
        lowres_imgenc_bchw = v3_encimgs[0]
        img_b, img_c, _, _ = lowres_imgenc_bchw.shape

        with _inference_mode(self._infmode):

            # For convenience, set up fallback tensors for missing inputs
            device, dtype = lowres_imgenc_bchw.device, lowres_imgenc_bchw.dtype
            missing_input_tensor_bnc = torch.empty((img_b, 0, img_c), device=device, dtype=dtype)
            encoded_text_bnc = missing_input_tensor_bnc
            encoded_sampling_bnc = missing_input_tensor_bnc

            # Handle text inputs
            if isinstance(text, str) and len(text) > 0:
                encoded_text_bnc = self.text_encoder(text)

            # Handle sampling coordinates
            in_coords = (box_xy1xy2_norm_list, point_xy_norm_list, negative_boxes_list, negative_points_list)
            if any(coords is not None for coords in in_coords):
                encoded_sampling_bnc = self.sampling_encoder(
                    lowres_imgenc_bchw,
                    boxes_bn22=self.sampling_encoder.prepare_box_input(box_xy1xy2_norm_list),
                    points_bn2=self.sampling_encoder.prepare_point_input(point_xy_norm_list),
                    negative_boxes_bn22=self.sampling_encoder.prepare_box_input(negative_boxes_list),
                    negative_points_bn2=self.sampling_encoder.prepare_point_input(negative_points_list),
                    include_coordinate_encodings=include_coordinate_encodings,
                )

            # Join sampling and text tokens for output
            encoded_exemplar_tokens_bnc = torch.cat((encoded_sampling_bnc, encoded_text_bnc), dim=1)

        return encoded_exemplar_tokens_bnc

    # .................................................................................................................

    def generate_detections(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor], list[Tensor]],
        encoded_exemplars_bnc: Tensor,
        detection_filter_threshold: float = 0.0,
        exemplar_padding_mask_bn: Tensor | None = None,
        blank_no_exemplar_outputs: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Takes in encoded image features and exemplar tokens, along with an optional padding mask,
        and produces segmentation masks and bounding box predictions.

        An optional filtering threshold can be provided which will filter out results
        (e.g. mask predictions) with detection scores below the given threshold value.
        Importantly, this also skips producing masks for low scoring detections,
        which may provide a small speed improvement.
        Note that this can result in outputs where the number of masks is 0!

        The padding mask is optional and is only meant for batched exemplar inputs. It should be a
        boolean tensor with shape BxN (matching exemplar BxN), with 'True' values in locations where
        the corresponding exemplar tokens are being padded.
        This is meant to allow for different sized exemplar tokens to be batched together into
        a single larger tensor (via padding). As a simplified diagram:
            exemplar_1 = [1,2]        -pad-> [1,2,0,0,0], then padding mask = [0,0,1,1,1]
            exemplar_2 = [3,4,5,6,7]  -----> [3,4,5,6,7]                      [0,0,0,0,0]
        The padding mask prevents padded entries from affecting attention calculations.

        Returns:
            mask_predictions, box_xy1xy2_predictions, detection_scores, presence_score
            -> Masks have shape: BxNxHxW (B batches, N masks, H & W are 4x the image tokens, 288 by default)
            -> Boxes have shape: BxNx2x2 (2x2 is for top-left/bottom-right coords: [(x1,y1), (x2,y2)])
            -> Detection scores have shape: BxN
            -> Presence score is a B-length vector (e.g. one score per batch)

        Notes:
            - Masks are in floating point format. To get a binary mask use: bin_mask = (mask > 0)
            - Box coordinates are normalized 0-to-1, where (0,0) is the top-left of the image
            - Detection scores are normalized 0-to-1, indicating the 'confidence' of each detection entry
            - Presence score is normalized 0-to-1, indicating if at least 1 exemplar is present
            - Detection scores are scaled by presence score, so are upper-bounded by presence score
        """

        # For clarity
        v3_encimgs = encoded_image_features_list[-1]
        lowres_imgenc_bchw, hiresx2_imgenc_bchw, hiresx4_imgenc_bchw = v3_encimgs

        with _inference_mode(self._infmode):

            # Return 'blank' results if we don't get any exemplars
            # -> Not required (model still works with no exemplars), but blanked results make more sense
            no_exemplars = encoded_exemplars_bnc.shape[1] == 0
            if no_exemplars and blank_no_exemplar_outputs:
                blk_tok, blk_box, blk_score, blk_pres = self.exemplar_detector.create_blank_output(lowres_imgenc_bchw)
                blk_masks, _ = self.exemplar_segmentation.create_blank_output(blk_tok, lowres_imgenc_bchw)
                return blk_masks, blk_box, blk_score, blk_pres

            # Mix exemplar data into image tokens
            fused_imgexm_tokens_bchw = self.image_exemplar_fusion(
                lowres_imgenc_bchw,
                encoded_exemplars_bnc,
                exemplar_padding_mask_bn,
            )

            # Compute detections
            enc_det_tokens_bnc, boxes_xy1xy2_bn22, det_scores_bn, pres_scores = self.exemplar_detector(
                fused_imgexm_tokens_bchw, encoded_exemplars_bnc, exemplar_padding_mask_bn
            )

            # Special optimization: Filter out 'bad' detections before performing segmentation
            # -> For example, if there are only 3 'valid' detections based on filter threshold
            #    then we only need to generate 3 masks, instead of all 200 of them
            if detection_filter_threshold > 1e-3:
                assert det_scores_bn.shape[0] == 1, "Cannot pre-filter detections when using batched inputs!"
                ok_filter = det_scores_bn[0] > detection_filter_threshold
                enc_det_tokens_bnc = enc_det_tokens_bnc[:, ok_filter]
                boxes_xy1xy2_bn22 = boxes_xy1xy2_bn22[:, ok_filter]
                det_scores_bn = det_scores_bn[:, ok_filter]

            # Generate masks
            mask_preds_bnhw, _ = self.exemplar_segmentation(
                enc_det_tokens_bnc,
                fused_imgexm_tokens_bchw,
                hiresx2_imgenc_bchw,
                hiresx4_imgenc_bchw,
                encoded_exemplars_bnc,
                exemplar_padding_mask_bn,
            )

        return mask_preds_bnhw, boxes_xy1xy2_bn22, det_scores_bn, pres_scores

    # .................................................................................................................

    def filter_results(
        self,
        mask_predictions: Tensor | None,
        box_predictions: Tensor | None,
        detection_scores: Tensor,
        presence_score: Tensor | None = None,
        score_threshold: float = 0.5,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor | None]:
        """
        Helper used to filter out 'bad' detection results, based on detection scores
        Note that the arguments to this function are set up to match the output of:
            outputs = model.generate_detections(...)
        So it can be called using:
            results = model.filter_results(*outputs, score_threshold=0.75)

        If needed, masks and/or box inputs can be set to 'None' to skip filtering
        if they are not being used. Though no matter what inputs are provided,
        the output length is the same.

        Returns:
            filtered_masks, filtered_boxes, filtered_scores, presence_score (as-is)
        """

        # Sanity check. We don't handle batched inputs because the output cannot (generally) be a tensor
        is_batched = detection_scores.ndim == 2 and detection_scores.shape[0] > 1
        assert not is_batched, "Cannot directly filter batched inputs, call this function in a loop instead"

        is_valid_detection = detection_scores > score_threshold
        filtered_masks = mask_predictions[is_valid_detection] if mask_predictions is not None else None
        filtered_boxes = box_predictions[is_valid_detection] if box_predictions is not None else None
        filtered_scores = detection_scores[is_valid_detection]

        return filtered_masks, filtered_boxes, filtered_scores, presence_score

    # .................................................................................................................

    def encode_tracking_and_detection_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        ***Special case function***

        This function is used to compute both the image encodings needed for tracking
        AND the features for detections, from a bgr formatted input image (e.g. loaded from opencv).
        It is purely for optimization purposes when combining SAMv3 detections with video tracking.

        This is equivalent to running .encode_image(...) and .encode_detection_image(...)
        using the SAMv3 model and detection variant (respectively). However, this approach
        is much more efficient, as the heaviest part of this encoding can be shared between
        both outputs, so only needs to be computed once.

        Note that the return format is slightly different from the original encoding functions!

        Returns:
            encoded_tracking_features_list, encoded_detection_features_list

        -> Both outputs contain 3 multi-resolution feature maps
        -> The tracking features have shapes: Bx256xHxW, Bx64x(2H)x(2W) & Bx32x(4H)x(4W)
           The detection features have shapes: Bx256xHxW, Bx256x(2H)x(2W) & Bx256x(4H)x(4W)
        -> Where H & W are the 'low-res' feature map size (72x72 by default)

        The tracking features are the ones used in video tracking (e.g. inside the 'step_video_masking' function)
        while the detection features are used in the exemplar encoding and generate detections functions.
        """

        with _inference_mode(self._infmode):
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            encoded_img = self.image_encoder(image_rgb_normalized_bchw)
            tracking_features_list = self.image_projection.v2_projection(encoded_img)
            detection_features_list = self.image_projection.v3_projection(encoded_img)

        return tracking_features_list, detection_features_list

    # .................................................................................................................

    def enable_compilation(
        self,
        example_image_bgr: ndarray | None = None,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
        compile_image_encoding: bool = True,
        compile_exemplar_encoding: bool = True,
        compile_detection_segmentation: bool = True,
        custom_config: dict | None = None,
    ) -> int:
        """
        Enable (experimental) compilation of model components.
        Anecdotally, this seems to give around a 10% reduction in
        inference time of large components and 30-40% reduction for
        smaller components, though this is likely dependent on hardware!

        Note that any adjustments to model usage (e.g. different input parameters)
        after calling this function may result in re-compilation.

        The 'custom_config' input can be given to provide custom compilation
        settings. This is used as:
            torch.compile(<model_code>, **custom_config)
        If not provided, some (fairly aggressive) default settings will be used.

        Compilation for certain components can be toggled on/off using the
        'compile_image_encoding' and similar args. This can be useful to disable
        heavy compilation on components that are only called once for example.
        This can also be used to provide different custom compilation configs
        to different model components, if needed.

        Returns:
            total_time_taken_ms
        """

        # Special optimization to speed up float32 usage
        if next(self.parameters()).dtype == torch.float32:
            torch.set_float32_matmul_precision("high")

        # Fill in default compilation settings
        comp_kwargs = custom_config
        dyncomp_kwargs = custom_config
        if custom_config is None:
            compile_options = {
                "shape_padding": True,
                "epilogue_fusion": True,
                "max-autotune": True,
            }
            comp_kwargs = {"mode": None, "fullgraph": True, "options": compile_options}
            dyncomp_kwargs = {**comp_kwargs, "dynamic": True}

        # Create dummy arguments to run model
        exm_args_1 = ("visual", [[(0.2, 0.3), (0.5, 0.6)]], [(0.5, 0.5)], [[(0.1, 0.1), (0.2, 0.2)]], [(0.9, 0.1)])
        exm_args_2 = ("visual", None, None, None, None)
        exm_args_3 = (None, *exm_args_1[1:])
        if example_image_bgr is None:
            example_image_bgr = np.zeros((1, 1, 3), dtype=np.uint8)

        # Switch away from using complex numbers (not well supported by compiler)
        if compile_image_encoding:
            self.image_encoder.toggle_use_complex_numbers(False)

        # Run model to fill caches
        encoded_imgs, _, _ = self.encode_detection_image(example_image_bgr, max_side_length, use_square_sizing)
        encoded_exemplars = self.encode_exemplars(encoded_imgs, *exm_args_1)
        self.generate_detections(encoded_imgs, encoded_exemplars)

        # Start timing
        t1 = perf_counter()

        if compile_image_encoding:
            self.image_encoder.forward = torch.compile(self.image_encoder.forward, **comp_kwargs)
            self.image_projection.forward = torch.compile(self.image_projection.forward, **comp_kwargs)

        if compile_exemplar_encoding:
            self.text_encoder.transformer.forward = torch.compile(
                self.text_encoder.transformer.forward, **dyncomp_kwargs
            )
            self.sampling_encoder.fusion_transformer.forward = torch.compile(
                self.sampling_encoder.fusion_transformer.forward, **dyncomp_kwargs
            )
            self.image_exemplar_fusion.forward = torch.compile(self.image_exemplar_fusion.forward, **dyncomp_kwargs)

        if compile_detection_segmentation:
            self.exemplar_detector.forward = torch.compile(self.exemplar_detector.forward, **dyncomp_kwargs)
            self.exemplar_segmentation.forward = torch.compile(self.exemplar_segmentation.forward, **dyncomp_kwargs)

        # Compilation doesn't occur until we actually run the model!
        for exm_args in [exm_args_3, exm_args_2, exm_args_1]:
            encoded_imgs, _, _ = self.encode_detection_image(example_image_bgr, max_side_length, use_square_sizing)
            encoded_exemplars = self.encode_exemplars(encoded_imgs, *exm_args)
            self.generate_detections(encoded_imgs, encoded_exemplars)

        # Finish timing
        t2 = perf_counter()
        time_taken_ms = round(1000 * (t2 - t1))

        return time_taken_ms

    # .................................................................................................................

    def forward(self, *args, **kwargs) -> None:
        """
        Placeholder to prevent users from trying to call this model using the forward function.
        This model is meant to be called in stages, roughly as follows:

            # Example procedure for generating detections
            enc_image, _, _ = msam_det.encode_detection_image(image_bgr)
            enc_exemplars = msam_det.encode_exemplars(enc_image, text="person")
            mask_preds, box_preds, scores, presence = msam_det.generate_detections(enc_image, enc_exemplars)

        Please see individual functions for more information
        """

        name = self.__class__.__name__
        print(
            "",
            f"The .forward(...) function of this model ({name}) isn't meant to be called directly!",
            "Instead, use functions (in order):",
            "  model.encode_detection_image(...)",
            "  model.encode_exemplars(...)",
            "  model.generate_detections(...)",
            "",
            "In order to generate mask & bounding-box predictions",
            sep="\n",
        )

        raise NotImplementedError("This model isn't meant to be used with .forward(...)")

    # .................................................................................................................

    @staticmethod
    def make_exemplar_batch(*encoded_exemplars_bnc: Tensor) -> tuple[Tensor, Tensor]:
        """
        Function which takes in multiple encoded exemplars and produces
        a single 'batched' set of exemplar tokens.
        The encoded tokens are expected to come from calling:
            model.encode_exemplars(...)

        Note that exemplar tokens will generally be different sizes
        and so will require a padding mask, which is also returned
        by this function.

        Returns:
            batched_exemplar_tokens_bnc, exemplar_padding_mask_bn
        """

        # Get info about all tokens for batching
        device_dtype_list = []
        batch_size_list = []
        num_tokens_list = []
        channel_counts_list = []
        for tokens_bnc in encoded_exemplars_bnc:
            assert tokens_bnc.ndim == 3, f"Expecting exmplars to have shape: BxNxC, got: {tokens_bnc.shape}"
            b, n, c = tokens_bnc.shape
            batch_size_list.append(b)
            num_tokens_list.append(n)
            channel_counts_list.append(c)
            device_dtype_list.append((tokens_bnc.device, tokens_bnc.dtype))

        # Figure out data sizing/config
        total_batch_size = sum(batch_size_list)
        max_num_tokens = max(num_tokens_list)
        num_channels = channel_counts_list[0]
        device, dtype = device_dtype_list[0]

        # Sanity checks
        all_same_channel_count = all(c == num_channels for c in channel_counts_list)
        all_same_device_and_dtype = all(d == device and t == dtype for d, t in device_dtype_list)
        assert all_same_channel_count, f"Error, got different exemplar channel counts ({channel_counts_list})"
        assert all_same_device_and_dtype, f"Error, got different exemplar device/dtypes ({device_dtype_list})"

        # Create single batched tensor & corresponding padding mask
        next_batch_idx = 0
        batched_tokens_bnc = torch.zeros((total_batch_size, max_num_tokens, num_channels), device=device, dtype=dtype)
        padding_mask_bn = torch.ones((total_batch_size, max_num_tokens), device=device, dtype=torch.bool)
        for tokens_bnc in encoded_exemplars_bnc:
            # Figure out batch indexing (doing it this way allows us to take in already-batched tokens!)
            b, n, _ = tokens_bnc.shape
            b_slice = slice(next_batch_idx, next_batch_idx + b)
            next_batch_idx += b

            # Fill in batched data & mask
            batched_tokens_bnc[b_slice, :n, :] = tokens_bnc
            padding_mask_bn[b_slice, :n] = False

        return batched_tokens_bnc, padding_mask_bn

    # .................................................................................................................

    def toggle_inference_mode(self, enable_inference_mode: bool | None = None) -> bool:
        """
        Helper used to toggle internal 'with torch.inference_mode' on/off.
        When training the model, inference mode can become problematic, so disabling it can be helpful.
        If 'enable_inference_mode' is None, then the current state will be toggled.
        Otherwise, the state can be explicitly set by providing a True or False argument.
        Returns: is_inference_mode_enabled
        """
        self._infmode = not self._infmode if enable_inference_mode is None else enable_inference_mode
        return self._infmode


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


@contextmanager
def _inference_mode(enable: bool = True):
    """
    Custom wrapper around 'torch.inference_mode' that can fully disable it, useful for training
    Though the torch function has it's own 'mode=False', it isn't the same as disabling.
    The follow example is a case where the built-in behavior is counter-intuitive:
        with torch.no_grad():
            with torch.inference_mode(False):
                output_data = model(input_data)
                # ^^^ output will have gradients tracked, even though it's inside a no_grad block
    """
    if enable:
        with torch.inference_mode():
            yield
    else:
        yield
    return
