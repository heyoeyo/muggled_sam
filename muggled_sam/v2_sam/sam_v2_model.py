#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from contextlib import contextmanager

import torch
import torch.nn as nn

from .compilation import enable_compilation as _enable_compilation

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


class SAMV2Core(nn.Module):
    """
    Holds core SAMv2 model components.

    This module is not meant to be used directly, instead, it's used to create
    separate 'contexts' to perform specific tasks supported by the model.
    For example:
        interact_model = sam2_model.get_interactive_context()    # SAMv1 task
        track_model = sam2_model.get_tracking_context()          # SAMv2 task

    Since the core holds all components, it can be used to move the model to a device/dtype,
    and this will affect all contexts. For example, using:
        sam2_model.to("cuda")
        interact_model = sam2_model.get_interactive_context()
        track_model = sam2_model.get_tracking_context()
    In this case, both the interactive and tracking contexts will be on the 'cuda' device.

    Note that contexts do not instantiate new data (e.g doesn't increase VRAM use),
    they're just used to group related functionality together. However, the core can be
    deleted after creating contexts in order to recover memory from unused core components.

    This core module also holds legacy implementations of segmentation/tracking functionality,
    though this will be removed in future updates!
    """

    name = "samv2"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV2ImageEncoder,
        coordinate_encoder_model: SAMV2CoordinateEncoder,
        prompt_encoder_model: SAMV2PromptEncoder,
        mask_decoder_model: SAMV2MaskDecoder,
        memory_encoder_model: SAMV2MemoryEncoder,
        memory_image_fusion_model: SAMV2MemoryImageFusion,
        config_bytes: bytearray,
        enable_inference_mode: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Store config data
        self.register_buffer("config_muggled_samv2", torch.tensor(config_bytes, dtype=torch.uint8))

        # Store SAM model components
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder_model
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model
        self.memory_encoder = memory_encoder_model
        self.memory_image_fusion = memory_image_fusion_model

        # Default to eval mode, expecting to use inference only
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        self._infmode = enable_inference_mode

    # .................................................................................................................

    def encode_prompts(self, box_xy1xy2_norm_list: list, fg_xy_norm_list: list, bg_xy_norm_list: list) -> Tensor:
        """Temporary placeholder for backwards compatibility"""
        return SAMV2InteractiveModel.encode_prompts(self, box_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list)

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """Temporary placeholder for backwards compatibility"""
        return SAMV2InteractiveModel.encode_image(self, image_bgr, max_side_length, use_square_sizing)

    def generate_masks(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        encoded_prompts: Tensor,
        mask_hint: Tensor | int | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        return SAMV2InteractiveModel.generate_masks(
            self, encoded_image_features_list, encoded_prompts, mask_hint, blank_promptless_output
        )

    def initialize_video_masking(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        box_xy1xy2_norm_list: list,
        fg_xy_norm_list: list,
        bg_xy_norm_list: list,
        mask_hint: Tensor | None = None,
        mask_index_select: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        return self.get_tracking_context().encode_prompt_memory(
            encoded_image_features_list,
            box_xy1xy2_norm_list,
            fg_xy_norm_list,
            bg_xy_norm_list,
            mask_hint,
            mask_index_select,
        )

    def initialize_from_mask(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        mask_image: ndarray | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        return self.get_tracking_context().encode_prompt_memory_from_mask(encoded_image_features_list, mask_image)

    def step_video_masking(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
        is_recent_first: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        # Run mask prediction
        track_model = self.get_tracking_context()
        masks_bnhw, ious_bn, ptrs_bnc, obj_score_b = track_model.step_video_masking(
            encoded_image_features_list,
            prompt_memory_encodings,
            prompt_object_pointers,
            frame_memory_encodings=previous_memory_encodings,
            frame_object_pointers=previous_object_pointers,
            return_best_only=False,
            is_recent_first=is_recent_first,
        )

        # Find 'best' result (can't use 'return_best_only' because old API always returned all masks)
        best_mask_idx, best_mask_pred, _, best_obj_ptr = self.mask_decoder.get_best_decoder_results(
            masks_bnhw,
            ious_bn,
            ptrs_bnc,
            exclude_0th_index=True,
        )

        # Run memory encoding
        memory_encoding, best_obj_ptr = track_model.encode_frame_memory(
            encoded_image_features_list, best_mask_pred, best_obj_ptr, obj_score_b
        )

        # Old API returned a combination of predictions & memory encoding!
        return obj_score_b, best_mask_idx, masks_bnhw, memory_encoding, best_obj_ptr

    def prepare_image_batch(
        self,
        images_bgr_list: list[ndarray],
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> Tensor:
        """Temporary placeholder for backwards compatibility"""
        return SAMV2InteractiveModel.prepare_image_batch(self, images_bgr_list, max_side_length, use_square_sizing)

    # .................................................................................................................

    def get_interactive_context(self) -> nn.Module:
        """Creates a tracking model, used for video segmentation"""
        return SAMV2InteractiveModel(
            self.image_encoder, self.coordinate_encoder, self.prompt_encoder, self.mask_decoder, self._infmode
        )

    def get_tracking_context(self) -> nn.Module:
        """Creates a tracking model, used for video segmentation"""
        return SAMV2TrackingModel(
            self.image_encoder,
            self.coordinate_encoder,
            self.prompt_encoder,
            self.mask_decoder,
            self.memory_encoder,
            self.memory_image_fusion,
            self._infmode,
        )

    def get_detector_context(self, *args, **kwargs) -> None:
        """Warning for unsupported feature"""
        raise AttributeError("SAMv2 does not support object detection (requires SAMv3)")

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

    def forward(self, *args, **kwargs) -> None:
        _model_usage_error(self)

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Task-Specific classes


class SAMV2InteractiveModel(nn.Module):
    """
    Wrapper around the SAMV2 interactive components, used for image segmentation  (e.g. SAMv1 task).

    The basic usage of this model is to call the 3 main functions (in order) to generate mask predictions:
        model.encode_image(...)
        model.encode_prompts(...)
        model.generate_masks(...)
    See the image segmentation example for more details:
    https://github.com/heyoeyo/muggled_sam/blob/main/simple_examples/image_segmentation.py
    """

    name = "samv2"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV2ImageEncoder,
        coordinate_encoder_model: SAMV2CoordinateEncoder,
        prompt_encoder_model: SAMV2PromptEncoder,
        mask_decoder_model: SAMV2MaskDecoder,
        enable_inference_mode: bool = True,
    ):
        # Re-store all components
        super().__init__()
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder_model
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model
        self._infmode = enable_inference_mode

    # .................................................................................................................

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = 1024,
        use_square_sizing: bool = True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The max_side_length setting is used to set the size at which the image is processed,
        while the use_square_sizing determines whether the image is scaled to a square resolution
        or scaled (to the max_side_length) based on it's original aspect ratio.

        Returns:
            encoded_image, patch_grid_hw, preencoded_image_hw
            -> Encoded image is a list containing 3 multi-resolution feature maps
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

    def encode_prompts(self, box_xy1xy2_norm_list: list, fg_xy_norm_list: list, bg_xy_norm_list: list) -> Tensor:
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
            encoded_prompts_bnc
            -> shape: BxNxC, B batch size, N number of prompt points, C is channels/feature count
        """

        with _inference_mode(self._infmode):
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_xy1xy2_norm_list)
            fg_tensor, bg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts_bnc = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts_bnc

    # .................................................................................................................

    def generate_masks(
        self,
        encoded_image: list[Tensor],
        encoded_prompts: Tensor,
        mask_hint: Tensor | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Function used to generate segmentation masks given an image encoding,
        as well as a prompt encoding and potentially a mask hint/prompt. These
        input encodings are expected to come from other model components.

        The mask hint can either be None (no mask input) or a tensor. If a
        tensor is given, it's assumed to be a mask itself. It should be shaped
        to match the model's own output masks for the given input image size,
        by default this would be a shape of: Bx1x256x256

        Returns:
            mask_predictions, iou_predictions
            -> Masks have shape: Bx4xHxW (HxW is 256x256 using default settings)
            -> iou_predictions have shape: Bx4
        """

        # Get sizing of the lowest-resolution image encoding
        patch_grid_hw = encoded_image[0].shape[2:]

        with _inference_mode(self._infmode):
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            masks_bnhw, ious_bn, _, _ = self.mask_decoder(
                encoded_image, encoded_prompts, grid_posenc, mask_hint, blank_promptless_output
            )

        return masks_bnhw, ious_bn

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

    def enable_compilation(
        self,
        example_image_bgr: ndarray | None = None,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
        compile_image_encoding: bool = True,
        compile_mask_generation: bool = True,
        custom_config: dict | None = None,
    ) -> None:
        """Enable (experimental) compilation of model components"""

        # Run model to fill in cache
        if example_image_bgr is not None:
            encoded_imgs, _, _ = self.encode_image(example_image_bgr, max_side_length, use_square_sizing)
            encoded_prompts = self.encode_prompts([], [(0.5, 0.5)], [])
            self.generate_masks(encoded_imgs, encoded_prompts)

        return _enable_compilation(
            self,
            compile_image_encoder=compile_image_encoding,
            compile_coordinate_encoder=compile_mask_generation,
            compile_prompt_encoder=compile_mask_generation,
            compile_mask_decoder=compile_mask_generation,
            custom_config=custom_config,
        )

    def toggle_inference_mode(self, enable_inference_mode: bool | None = None) -> bool:
        self._infmode = not self._infmode if enable_inference_mode is None else enable_inference_mode
        return self._infmode

    def forward(self, *args, **kwargs) -> None:
        _model_usage_error(self)

    # .................................................................................................................


class SAMV2TrackingModel(nn.Module):
    """
    Wrapper around SAMV2 tracking components, used for video segmentation

    The basic usage of this model involves two phases. The first is to encode a prompt or mask
    'memory' which determines the object to be tracked. The second phase is to repeatedly make
    mask predictions for new incoming frames of a video using prior memory encodings.

    The basic usage for encoding an initial object is:
        model.encode_image(...)
        model.encode_prompt_memory(...) OR .encode_prompt_memory_from_mask(...)

    The usage for repeatedly encoding new frames is:
        model.encode_image(...)
        model.step_video_masking(...)
        model.encode_frame_memory(...)

    See the video segmentation example for more details:
    https://github.com/heyoeyo/muggled_sam/blob/main/simple_examples/video_segmentation.py
    """

    name = "samv2"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV2ImageEncoder,
        coordinate_encoder_model: SAMV2CoordinateEncoder,
        prompt_encoder_model: SAMV2PromptEncoder,
        mask_decoder_model: SAMV2MaskDecoder,
        memory_encoder_model: SAMV2MemoryEncoder,
        memory_image_fusion_model: SAMV2MemoryImageFusion,
        enable_inference_mode: bool = True,
    ):
        # Inherit from parent
        super().__init__()

        # Store both interactive & tracking components
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder_model
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model
        self.memory_encoder = memory_encoder_model
        self.memory_image_fusion = memory_image_fusion_model
        self._infmode = enable_inference_mode

    # .................................................................................................................

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The 'max_side_length' and 'use_square_sizing' inputs control the resolution and aspect ratio
        of the image before encoding.
        Returns:
            encoded_image, patch_grid_hw, preencoded_image_hw
        """
        # Re-use interactive model implementation
        return SAMV2InteractiveModel.encode_image(self, image_bgr, max_side_length, use_square_sizing)

    # .................................................................................................................

    def encode_prompt_memory(
        self,
        encoded_image: list[Tensor],
        box_xy1xy2_norm_list: list,
        fg_xy_norm_list: list,
        bg_xy_norm_list: list,
        mask_hint: Tensor | None = None,
        mask_index: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        This function acts similarly to encoding prompts on the interactive
        usage of the model. However, this instead creates a 'memory encoding',
        which acts like a prompt encodings, but for video segmentation. This
        function also returns the mask associated with the provided prompt.
        The mask will be chosen according to the 'mask_index' input, or if
        an index isn't provided, a 'best' mask will be chosen automatically.

        In addition to 'prompt memory' there are 'frame memory' encodings. These
        both serve the same role, but prompt memory is expected to come from user
        input, and has a stronger influnce on tracking.

        This function roughly corresponds to `add_new_points_or_box` in the original SAMv2 implementation:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/sam2_video_predictor.py#L173
        -> The original implementation uses the term 'cond_frame_outputs' instead of 'prompt memory'

        Returns:
            mask_prediction, memory_encoding, object_pointer
        """

        # Encode initial prompts (re-use interactive implementation as it's the same process)
        # -> This lets us 'hide' the prompt encoding function, which isn't meant to be directly used on tracking model
        encoded_prompts = SAMV2InteractiveModel.encode_prompts(
            self, box_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list
        )

        with _inference_mode(self._infmode):

            # For convenience
            lowres_imgenc = encoded_image[0]
            token_hw = lowres_imgenc.shape[2:]

            # Generate mask prediction from image/prompt encodings, as usual
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(token_hw)
            masks_bnhw, ious_bn, ptrs_bnc, obj_score_b = self.mask_decoder(
                encoded_image,
                encoded_prompts,
                grid_posenc,
                mask_hint=mask_hint,
                blank_promptless_output=False,
            )

            # Auto-select mask if not given an index
            if mask_index is None:
                mask_index = ious_bn.argmax(-1)
            elif isinstance(mask_index, int):
                mask_index = torch.tensor([mask_index], dtype=torch.int64, device=masks_bnhw.device)

            # Encode memory
            best_mask_pred = masks_bnhw[:, mask_index, :, :]
            best_obj_ptr = ptrs_bnc[:, mask_index]
            memory_encoding, obj_ptr = self.encode_frame_memory(
                encoded_image, best_mask_pred, best_obj_ptr, obj_score_b, is_prompt_encoding=True
            )

        return best_mask_pred, memory_encoding, best_obj_ptr

    # .................................................................................................................

    def encode_prompt_memory_from_mask(
        self,
        encoded_image: tuple[list[Tensor], list[Tensor]],
        mask_image: ndarray | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Alternate option for creating a prompt memory encoding, in the case
        where a segmentation mask is directly available. This is in contrast
        to providing a point/box prompt (see 'encode_prompt_memory(...)')

        The provided mask can either be a numpy array (e.g. image loaded using opencv) or
        a pytorch tensor (e.g. output from another model). In the simplest cases, a boolean
        mask should be provided with shape: HxW. A non-boolean mask, for example a mask
        prediction from SAM itself, can also be given.

        - If a mask is given with 3 dimensions, but the last dimension has size <= 3, it's assumed
          to be shaped as: HxWxC (e.g. a BGR image from opencv). In this case, the 0-th entry
          of the 3rd dimension will be taken as the (HxW) mask
        - If a mask is given with 3 dimensions, but the last dimension has size > 3, it's assumed
          to be shaped as: BxHxW, which is acceptable though batch sizes > 1 may not
          be well supported when performing video segmentation steps!
        - If a mask is given with 4 dimensions, it should have shape: Bx1xHxW

        Returns:
            memory_encoding, object_pointer
        """

        with _inference_mode(self._infmode):

            # For convenience
            lowres_imgenc = encoded_image[0]
            device, dtype = lowres_imgenc.device, lowres_imgenc.dtype

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

        # Generate memory encoding directly from mask
        obj_ptr, obj_score, mask_index = None, None, 0
        memenc_bchw, objptrs_b1c = self.encode_frame_memory(
            encoded_image, mask_tensor, obj_ptr, obj_score, mask_index, is_prompt_encoding=True
        )

        return memenc_bchw, objptrs_b1c

    # .................................................................................................................

    def encode_frame_memory(
        self,
        encoded_image: list[Tensor],
        mask_predictions: Tensor,
        object_pointers: Tensor | None,
        object_score: Tensor | None,
        mask_index: Tensor | int | None = None,
        is_prompt_encoding: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Function which computes 'frame memory' encodings. These act like a
        prompt encoding (from interactive model usage), but without requiring
        input from the user. This memory encoding is used in conjunction with
        'prompt memory' to 'tell the model' what the object looks like.

        Since the prompt memory is made from direct user prompts, it's a
        strong (direct) signal of what the object looks like for tracking.
        By comparison, frame memory is computed from the model's own output
        and helps the model (automatically) keep track of objects as they
        change appearance over time.

        The inputs to this function are expected to come from the output
        of the 'step_video_masking' function. If multiple non-batched masks
        are given (e.g. masks with shape BxNxHxW with N > 1), then the
        'mask_index' input must be provided to indicate which masks to encode!

        This function roughly corresponds to `_encode_new_memory` in the original SAMv2 implementation:
        https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L678
        -> The original implementation uses the term 'non_cond_frame_outputs' instead of 'frame memory'

        Returns:
            memory_encoding, object_pointer
        """

        # If we don't get a mask index, then user must provide only 1 mask prediction
        if mask_index is None:
            assert (
                mask_predictions.shape[1] == 1
            ), f"Cannot determine which mask to encode! Mask must be shape: Bx1xHxW if no index is provided. Got: {tuple(mask_predictions.shape)}"
            mask_index = 0

        # Make sure we have a slice/tensor-style index
        if isinstance(mask_index, int):
            mask_index = torch.tensor([mask_index], dtype=torch.int64, device=mask_predictions.device)

        # Compute memory encoding (and optionally object pointers, if not provided)
        with _inference_mode(self._infmode):

            # Index out best entries, while accounting for batch dimension
            b_idx = torch.arange(mask_predictions.shape[0], device=mask_predictions.device)
            best_mask_b1hw = mask_predictions[b_idx, mask_index].unsqueeze(1)
            best_objptr_b1c = object_pointers[b_idx, mask_index].unsqueeze(1) if object_pointers is not None else None

            # Encode new memory features
            # Called '_encode_memory_in_output' in original code
            # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L867
            lowres_imgenc = encoded_image[0]
            memenc_bchw = self.memory_encoder(lowres_imgenc, best_mask_b1hw, object_score, is_prompt_encoding)

            # Special case. If we're not given a pointer, make it from the mask
            # -> Mostly required for encoding from initial mask provided direct from user (without a prompt)
            # https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L442
            if best_objptr_b1c is None:
                token_hw = lowres_imgenc.shape[2:]
                pad_prompt_enc = self.prompt_encoder.create_padding_point_encoding()
                grid_posenc = self.coordinate_encoder.get_grid_position_encoding(token_hw)
                best_objptr_b1c = self.mask_decoder.make_pointer_from_mask(
                    encoded_image, pad_prompt_enc, grid_posenc, best_mask_b1hw
                )

        return memenc_bchw, best_objptr_b1c

    # .................................................................................................................

    def step_video_masking(
        self,
        encoded_image: list[Tensor],
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        frame_memory_encodings: list[Tensor],
        frame_object_pointers: list[Tensor],
        return_best_only: bool = True,
        is_recent_first: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Function which makes segmentation predictions for consecutive frames of
        an input video. It takes in the encoded video frame data along with prior
        prompt/frame memory encodings in order to continue segmenting an existing
        object (i.e. without requiring user prompts).

        If 'return_best_only' is True, only a single result will be returned,
        corresponding to the prediction with the highest IoU.

        This function corresponds to 'track_step' in the original SAMv2 implementation:
        https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L812

        The mask (and object pointer) returned from this function are meant to be
        provided to the 'encode_frame_memory' function to produce future memory
        encodings used to continue video masking into the future.

        Returns:
            mask_predictions, iou_predictions, object_pointers, object_score
            -> mask_predictions have shape: BxNxHxW
            -> iou_predictions have shape: BxN
            -> object_pointers are meant to be passed in as future input and have shape BxNxC
            -> object_score is an indicator of whether an object is present,
               values below 0 indicate lost tracking. Each batch entry will have 1 value
            -> The HxW of masks will be 1/4 of the input height and width (256x256 for default 1024 sizing)
        """

        with _inference_mode(self._infmode):

            # Encode image features with previous memory encodings & object pointer data
            # Called '_prepare_memory_conditioned_features' in original code
            # See: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L759
            lowres_imgenc, *hires_imgenc = encoded_image
            memfused_encimg = self.memory_image_fusion(
                lowres_imgenc,
                prompt_memory_encodings,
                prompt_object_pointers,
                frame_memory_encodings,
                frame_object_pointers,
                is_recent_first,
            )

            # Run mask decoder on memory-fused features
            # Called '_forward_sam_heads' in original code
            # See: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L777
            patch_grid_hw = memfused_encimg.shape[2:]
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            masks_bnhw, ious_bn, ptrs_bnc, obj_score_b = self.mask_decoder(
                [memfused_encimg, *hires_imgenc],
                self.prompt_encoder.create_video_no_prompt_encoding(),
                grid_posenc,
                mask_hint=None,
                blank_promptless_output=False,
            )

            # Keep only the 'best' results
            # Part of the '_forward_sam_heads' function in original code
            # See: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/modeling/sam2_base.py#L383
            if return_best_only:
                _, masks_bnhw, ious_bn, ptrs_bnc = self.mask_decoder.get_best_decoder_results(
                    masks_bnhw,
                    ious_bn,
                    ptrs_bnc,
                    exclude_0th_index=False,
                )

        return masks_bnhw, ious_bn, ptrs_bnc, obj_score_b

    # .................................................................................................................

    def enable_compilation(
        self,
        example_image_bgr: ndarray | None = None,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
        compile_image_encoding: bool = True,
        compile_mask_generation: bool = True,
        compile_memory_encoding: bool = True,
        custom_config: dict | None = None,
    ) -> None:
        """Enable (experimental) compilation of model components"""

        # Run model to fill in cache
        if example_image_bgr is not None:
            encoded_imgs, _, _ = self.encode_image(example_image_bgr, max_side_length, use_square_sizing)
            _, init_mem, init_ptr = self.encode_prompt_memory(encoded_imgs, [], [(0.5, 0.5)], [])
            self.step_video_masking(encoded_imgs, [init_mem], [init_ptr], [], [])

        return _enable_compilation(
            self,
            compile_image_encoder=compile_image_encoding,
            compile_coordinate_encoder=compile_mask_generation,
            compile_prompt_encoder=compile_mask_generation,
            compile_mask_decoder=compile_mask_generation,
            compile_memory_encoder=compile_memory_encoding,
            compile_memory_image_fusion=compile_memory_encoding,
            custom_config=custom_config,
        )

    def toggle_inference_mode(self, enable_inference_mode: bool | None = None) -> bool:
        self._infmode = not self._infmode if enable_inference_mode is None else enable_inference_mode
        return self._infmode

    def forward(self, *args, **kwargs) -> None:
        _model_usage_error(self)

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


def _model_usage_error(model: nn.Module) -> None:
    """Helper used to provide an error when using the model in unintended ways. Prints out the model docstring"""
    print("*" * 36, "Model usage error! See docstring:", "*" * 36, model.__doc__, sep="\n")
    raise NotImplementedError("Unintended model usage! See explanation above")
