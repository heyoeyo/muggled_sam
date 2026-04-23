#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# For type hints
from torch import Tensor
from numpy import ndarray
from .image_encoder_model import SAMV2ImageEncoder
from .coordinate_encoder_model import SAMV2CoordinateEncoder
from .prompt_encoder_model import SAMV2PromptEncoder
from .mask_decoder_model import SAMV2MaskDecoder
from .memory_encoder_model import SAMV2MemoryEncoder
from .memory_image_fusion_model import SAMV2MemoryImageFusion


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2Model(nn.Module):
    """
    Wrapper around separated SAMV2 model components, so that the model can be used as a singular entity
    """

    name = "samv2"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV2ImageEncoder,
        coordinate_encoder: SAMV2CoordinateEncoder,
        prompt_encoder_model: SAMV2PromptEncoder,
        mask_decoder_model: SAMV2MaskDecoder,
        memory_encoder_model: SAMV2MemoryEncoder,
        memory_image_fusion_model: SAMV2MemoryImageFusion,
        config_bytes: bytearray,
    ):

        # Inherit from parent
        super().__init__()

        # Store config data
        self.register_buffer("config_muggled_samv2", torch.tensor(config_bytes, dtype=torch.uint8))

        # Store SAM model components
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model
        self.memory_encoder = memory_encoder_model
        self.memory_image_fusion = memory_image_fusion_model

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
        All image/mask generating code of SAMV2 model, bundled into a single function.
        Takes an image and set of prompts and produces several candidate segmentation masks.

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
        mask_preds, iou_preds, objscore_pred, mask_tokens_out = self.mask_decoder(
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
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_tlbr_norm_list)
            fg_tensor, bg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts

    # .................................................................................................................

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length=1024,
        use_square_sizing=True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The max_side_length setting is used to set the size at which the image is processed,
        while the use_square_sizing determines whether the image is scaled to a square resolution
        or scaled (to the max_side_length) based on it's original aspect ratio.

        Returns:
            encoded_images_list, patch_grid_hw, preencoded_image_hw
            -> Encoded images list contains 3 multi-resolution feature maps
               they have shapes: Bx256x64x64, Bx64x128x128, Bx32x256x256
               (using default settings). The first-most feature map is
               the 'low-res' map needed by several other parts of the model
            -> The patch_grid_hw contains the height & width of the low-res
               feature map (64x64 with default 1024x1024 input sizing)
            -> The preencoded_image_hw contains the height & width of the
               input image after pre-processing, just before being encoded
               by default it would be 1024x1024
        """

        with _inference_mode(self._infmode):
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            image_preenc_hw = image_rgb_normalized_bchw.shape[2:]
            encoded_image_features_list = self.image_encoder(image_rgb_normalized_bchw)

        # Get patch sizing of lowest-res tokens (as needed by other components)
        patch_grid_hw = encoded_image_features_list[0].shape[2:]

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
        this would be a shape of: Bx1x256x256

        Returns:
            mask_predictions, iou_predictions
            -> Masks have shape: Bx4xHxW (HxW is 256x256 using default settings)
            -> iou_predictions have shape: Bx4
        """

        # Get sizing of the lowest-resolution image encoding
        patch_grid_hw = encoded_image_features_list[0].shape[2:]

        with _inference_mode(self._infmode):
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                encoded_image_features_list, encoded_prompts, grid_posenc, mask_hint, blank_promptless_output
            )

        return mask_preds, iou_preds

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

        This function roughly corresponds to `add_new_points_or_box` in the original SAMv2 implementation:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/sam2_video_predictor.py#L173

        Returns:
            best_mask_prediction, memory_encoding, object_pointer
        """

        # Encode initial prompts
        encoded_prompts = self.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

        with _inference_mode(self._infmode):

            # For convenience
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            token_hw = lowres_imgenc.shape[2:]

            # Generate mask prediction from image/prompt encodings, as usual
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(token_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                encoded_image_features_list,
                encoded_prompts,
                grid_posenc,
                mask_hint=mask_hint,
                blank_promptless_output=False,
            )

            # Select mask to use for initial encodings (auto-select if not given an index)
            if mask_index_select is None:
                mask_index_select = self.mask_decoder.get_best_mask_index(iou_preds)
            best_mask_pred = mask_preds[:, [mask_index_select], :, :]
            best_obj_ptr = obj_ptrs[:, [mask_index_select]]

            # Encode initial memory
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_pred, obj_score, is_prompt_encoding=True)

        return best_mask_pred, memory_encoding, best_obj_ptr

    # .................................................................................................................

    def initialize_from_mask(self, encoded_image_features_list: list[Tensor], mask_image: ndarray | Tensor) -> Tensor:
        """
        Alternate video tracking initialization option. In this case, using a provided mask image as a 'prompt'
        to begin tracking an object.

        The provided mask can either be a numpy array (e.g. image loaded using opencv) or
        a pytorch tensor (e.g. output from another model). In the simplest cases, a boolean
        mask should be provided with shape: HxW. If a non-boolean input is given, it will
        be converted to a boolean mask used a simple 'greater than 0' check
        (e.g. bool_mask = mask_image > 0).

        - If a mask is given with 3 dimensions, but the last dimension has size <= 3, it's assumed
          to be shaped as: HxWxC (e.g. a BGR image from opencv). In this case, the 0-th entry
          of the 3rd dimension will be taken as the (HxW) mask
        - If a mask is given with 3 dimensions, but the last dimension has size > 3, it's assumed
          to be shaped as: BxHxW, which is acceptable though batch sizes > 1 may not
          be directly supported when performing video segmentation steps!
        - If a mask is given with 4 dimensions, it will be interpretted as: Bx1xHxW, where it
          must have size '1' in the second-most dimension.

        Note that with this form of initializtion, there is no object pointer! The pointer normally
        comes from the mask prediction, so without a prediction, there is no pointer. The video
        masking should therefore be initialized with only the memory encoding and an empty pointer list.
        This doesn't have a substantial impact on the tracking

        Returns:
            memory_encoding
        """

        with _inference_mode(self._infmode):

            # For convenience
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            device, dtype = lowres_imgenc.device, lowres_imgenc.dtype
            token_hw = lowres_imgenc.shape[2:]

            # Force input into a boolean mask & convert to torch tensor
            if isinstance(mask_image, ndarray):
                mask_tensor = torch.tensor(mask_image, device=device, dtype=dtype)
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

            # Try to force Bx1xHxW shape
            mask_b, mask_n, mask_h, mask_w = mask_tensor.shape
            if mask_b == 1 and mask_n > 1:
                mask_tensor = mask_tensor.permute(1, 0, 2, 3)
                mask_b, mask_n, mask_h, mask_w = mask_tensor.shape
            assert mask_tensor.shape[1] == 1, "Mask shape error! Expecting '1' in shape index 1, eg. Bx1xHxW"

            # Make special-case pointer from mask, since we don't normally get one without prompting
            # https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L442
            pad_prompt_enc = self.prompt_encoder.create_padding_point_encoding()
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(token_hw)
            ptrs_b1c = self.mask_decoder.make_pointer_from_mask(
                encoded_image_features_list, pad_prompt_enc, grid_posenc, mask_tensor
            )

            # Generate memory encoding from mask
            memory_encoding = self.memory_encoder(lowres_imgenc, mask_tensor, None, is_prompt_encoding=True)

        return memory_encoding, ptrs_b1c

    # .................................................................................................................

    def step_video_masking(
        self,
        encoded_image_features_list: list[Tensor],
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Function which makes segmentation predictions for consecutive frames of
        an input video. It takes in the encoded video frame data along with prior
        prompt/previous frame memory data in order to automatically continue
        segmenting some existing object (i.e. without requiring user prompts).

        This function corresponds to 'track_step' in the original SAMv2 implementation:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L812

        Returns:
            object_score, best_mask_index, mask_predictions, memory_encoding, best_object_pointer
            -> object_score score is an indicator of whether an object is present
               values below 0 indicate lost tracking. Has shape: Bx1
            -> best_mask_index is the index of the highest iou score. Has shape: B (one index for each batch)
            -> mask_predictions are the same as with image segmentation, has shape: Bx4xHxW
            -> memory_encoding should be passed back in on future frames, has shape: BxFxH'xW'
            -> best_object_pointer should be passed back in on future frames, has shape: Bx1xF'

            The HxW of masks will be 1/4 of the input height and width (256x256 for default 1024 sizing).
            The memory encoding H'xW' is 4 times smaller than the mask sizing (64x64 by default).
            The memory & pointer features F & F' are model configs (64, 256 respectively, by default)
        """

        with _inference_mode(self._infmode):

            # Encode image features with previous memory encodings & object pointer data
            # Called '_prepare_memory_conditioned_features' in original code
            # See: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L759
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            memfused_encimg = self.memory_image_fusion(
                lowres_imgenc,
                prompt_memory_encodings,
                prompt_object_pointers,
                previous_memory_encodings,
                previous_object_pointers,
            )

            # Run mask decoder on memory-fused features
            # Called '_forward_sam_heads' in original code
            # See: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L777
            patch_grid_hw = memfused_encimg.shape[2:]
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                [memfused_encimg, *hires_imgenc],
                self.prompt_encoder.create_video_no_prompt_encoding(),
                grid_posenc,
                mask_hint=None,
                blank_promptless_output=False,
            )

            # Keep only the 'best' results
            # Part of the '_forward_sam_heads' function in original code
            # See: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L383
            best_mask_idx, best_mask_pred, _, best_obj_ptr = self.mask_decoder.get_best_decoder_results(
                mask_preds,
                iou_preds,
                obj_ptrs,
                exclude_0th_index=True,
            )

            # Encode new memory features
            # Called '_encode_memory_in_output' in original code
            # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L867
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_pred, obj_score)

        return obj_score, best_mask_idx, mask_preds, memory_encoding, best_obj_ptr

    # .................................................................................................................

    def get_best_mask_index(self, iou_predictions: Tensor) -> int:
        """Returns the index of the highest IoU prediction score"""
        return self.mask_decoder.get_best_mask_index(iou_predictions)

    # .................................................................................................................

    def check_have_prompts(self, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list) -> bool:
        """Helper used to check if there are any prompts"""
        return self.prompt_encoder.check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

    # .................................................................................................................

    def prepare_image_batch(
        self,
        images_bgr_list: list[ndarray],
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> Tensor:
        """
        Helper used to convert BGR image (from opencv) into the tensor format needed for image encoding
        Expects a list of numpy arrays (BGR images) as input. Returns a single tensor of shape: BxCxHxW
        """

        # Prepare each image individually
        img_tensors_list = []
        for image_bgr in images_bgr_list:
            img_tensor_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            img_tensors_list.append(img_tensor_bchw)

        # Check that all images have the same size if square sizing isn't used
        if not use_square_sizing:
            all_h_list = [img.shape[2] for img in img_tensors_list]
            all_w_list = [img.shape[3] for img in img_tensors_list]
            assert all(all_h_list[0] == h for h in all_h_list), "Mismatched image heights (different aspect ratios)"
            assert all(all_w_list[0] == w for w in all_w_list), "Mismatched image widths (different aspect ratios)"

        return torch.concat(img_tensors_list, dim=0)

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
