#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp
from contextlib import contextmanager

import torch
import torch.nn as nn

from .compilation import enable_compilation as _enable_compilation

# For type hints
from torch import Tensor
from numpy import ndarray
from .image_encoder_model import SAMV3ImageEncoder
from .image_projection_model import SAMV3ImageProjection
from .coordinate_encoder_model import SAMV3CoordinateEncoder
from .prompt_encoder_model import SAMV3PromptEncoder
from .mask_decoder_model import SAMV3MaskDecoder
from .memory_encoder_model import SAMV3MemoryEncoder
from .memory_image_fusion_model import SAMV3MemoryImageFusion
from .text_encoder_model import SAMV3TextEncoder
from .sampling_encoder import SAMV3SamplingEncoder
from .image_exemplar_fusion_model import SAMV3ImageExemplarFusion
from .exemplar_detector_model import SAMV3ExemplarDetector
from .exemplar_segmentation_model import SAMV3ExemplarSegmentation


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV3Model(nn.Module):
    """
    Holds all core SAMv3 model components.

    This module is not meant to be used directly, instead, it's used to create
    separate 'contexts' to perform specific tasks supported by the model.
    For example:
        interact_model = sam3_model.get_interactive_context()    # SAMv1 task
        track_model = sam3_model.get_tracking_context()          # SAMv2 task
        detect_model = sam3_model.get_detector_context()         # SAMv3 task

    Since the core holds all components, it can be used to move the model to a device/dtype,
    and this will affect all contexts. For example, using:
        sam3_model.to("cuda")
        interact_model = sam3_model.get_interactive_context()
        track_model = sam3_model.get_tracking_context()
    In this case, both the interactive and tracking contexts will be on the 'cuda' device.

    Note that contexts do not instantiate new data (e.g doesn't increase VRAM use),
    they're just used to group related functionality together. However, the core can be
    deleted after creating contexts in order to recover memory from unused core components.

    This core module also holds legacy implementations of segmentation/tracking functionality,
    though this will be removed in future updates!
    """

    name = "samv3"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV3ImageEncoder,
        image_projection_model: SAMV3ImageProjection,
        coordinate_encoder_model: SAMV3CoordinateEncoder,
        prompt_encoder_model: SAMV3PromptEncoder,
        mask_decoder_model: SAMV3MaskDecoder,
        memory_encoder_model: SAMV3MemoryEncoder,
        memory_image_fusion_model: SAMV3MemoryImageFusion,
        text_encoder_model: SAMV3TextEncoder,
        sampling_encoder_model: SAMV3SamplingEncoder,
        image_exemplar_fusion_model: SAMV3ImageExemplarFusion,
        exemplar_detector_model: SAMV3ExemplarDetector,
        exemplar_segmentation_model: SAMV3ExemplarSegmentation,
        config_bytes: bytearray,
        enable_inference_mode: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Store config data
        self.register_buffer("config_muggled_samv3", torch.tensor(config_bytes, dtype=torch.uint8))

        # Store base SAM components
        self.image_encoder = image_encoder_model
        self.image_projection = image_projection_model
        self.coordinate_encoder = coordinate_encoder_model
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model

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
        self._infmode = enable_inference_mode

    # .................................................................................................................

    def encode_prompts(self, box_tlbr_norm_list: list, fg_xy_norm_list: list, bg_xy_norm_list: list) -> Tensor:
        """Temporary placeholder for backwards compatibility"""
        return SAMV3InteractiveModel.encode_prompts(self, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[tuple[list[Tensor], list[Tensor]], tuple[int, int], tuple[int, int]]:
        """Temporary placeholder for backwards compatibility"""
        return SAMV3InteractiveModel.encode_image(self, image_bgr, max_side_length, use_square_sizing)

    def generate_masks(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        encoded_prompts: Tensor,
        mask_hint: Tensor | int | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        return SAMV3InteractiveModel.generate_masks(
            self, encoded_image_features_list, encoded_prompts, mask_hint, blank_promptless_output
        )

    def initialize_video_masking(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        box_tlbr_norm_list: list,
        fg_xy_norm_list: list,
        bg_xy_norm_list: list,
        mask_hint: Tensor | int | None = None,
        mask_index_select: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        return SAMV3TrackingModel.initialize_video_masking(
            self,
            encoded_image_features_list,
            box_tlbr_norm_list,
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
        return SAMV3TrackingModel.initialize_from_mask(self, encoded_image_features_list, mask_image)

    def step_video_masking(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        prompt_memory_encodings: list[Tensor],
        prompt_object_pointers: list[Tensor],
        previous_memory_encodings: list[Tensor],
        previous_object_pointers: list[Tensor],
        num_multiplex_objects: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        return SAMV3TrackingModel.step_video_masking(
            self,
            encoded_image_features_list,
            prompt_memory_encodings,
            prompt_object_pointers,
            previous_memory_encodings,
            previous_object_pointers,
            num_multiplex_objects,
        )

    def prepare_image_batch(
        self,
        images_bgr_list: list[ndarray],
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> Tensor:
        """Temporary placeholder for backwards compatibility"""
        return SAMV3InteractiveModel.prepare_image_batch(self, images_bgr_list, max_side_length, use_square_sizing)

    def make_detector_model(self, bpe_vocab_path: str | None = None) -> nn.Module:
        """Temporary placeholder for backwards compatibility"""
        return self.get_detector_context(bpe_vocab_path)

    # .................................................................................................................

    def get_interactive_context(self) -> nn.Module:
        """Creates an interactive model, used for user-prompted image segmentation"""
        return SAMV3InteractiveModel(
            self.image_encoder,
            self.image_projection,
            self.coordinate_encoder,
            self.prompt_encoder,
            self.mask_decoder,
            self._infmode,
        )

    def get_tracking_context(self) -> nn.Module:
        """Creates a tracking model, used for video segmentation"""
        return SAMV3TrackingModel(
            self.image_encoder,
            self.image_projection,
            self.coordinate_encoder,
            self.prompt_encoder,
            self.mask_decoder,
            self.memory_encoder,
            self.memory_image_fusion,
            self._infmode,
        )

    def get_detector_context(self, bpe_vocab_path: str | None = None) -> nn.Module:
        """Creates 'detection' model, used for generating bounding box & segmentation masks from exemplars"""
        return SAMV3DetectorModel(
            self.image_encoder,
            self.image_projection,
            self.text_encoder,
            self.sampling_encoder,
            self.image_exemplar_fusion,
            self.exemplar_detector,
            self.exemplar_segmentation,
            self._infmode,
            bpe_vocab_path,
        )

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


class SAMV3InteractiveModel(nn.Module):
    """
    Wrapper around SAMV3 interactive components, used for image segmentation (e.g. SAMv1 task).

    The basic usage of this model is to call the 3 main functions (in order) to generate mask predictions:
        model.encode_image(...)
        model.encode_prompts(...)
        model.generate_masks(...)
    See the image segmentation example for more details:
    https://github.com/heyoeyo/muggled_sam/blob/main/simple_examples/image_segmentation.py
    """

    name = "samv3"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV3ImageEncoder,
        image_projection_model: SAMV3ImageProjection,
        coordinate_encoder_model: SAMV3CoordinateEncoder,
        prompt_encoder_model: SAMV3PromptEncoder,
        mask_decoder_model: SAMV3MaskDecoder,
        enable_inference_mode: bool = True,
    ):
        # Store modules for interactive use
        super().__init__()
        self.image_encoder = image_encoder_model
        self.image_projection = image_projection_model
        self.coordinate_encoder = coordinate_encoder_model
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model
        self._infmode = enable_inference_mode

    # .................................................................................................................

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = 1008,
        use_square_sizing: bool = True,
    ) -> tuple[tuple[list[Tensor], list[Tensor]], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The max_side_length setting is used to set the size at which the image is processed, if
        no value is given, the image will be scaled to the 'preferred' size of the model.
        The use_square_sizing setting determines whether the image is scaled to a square resolution
        or scaled (to the max_side_length) based on it's original aspect ratio.

        Returns:
            encoded_images_list, patch_grid_hw, preencoded_image_hw
            -> Encoded images list contains 2 sets of multi-resolution
               feature maps, which correspond to SAMv2 or SAMv3 tasks.
               Each entry is itself contains 3 tensors where are
               multiresolution image features. The 0th entry is always
               the lowest resolution map with the 1st and 2nd index
               entries being 2x and 4x larger.
               Note, the v2 & v3 entries have different channel counts!
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
            encoded_prompts_bnc
            -> shape: BxNxC, B batch size, N number of prompt points, C is channels/feature count
        """

        with _inference_mode(self._infmode):
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_tlbr_norm_list)
            fg_tensor, bg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts_bnc = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts_bnc

    # .................................................................................................................

    def generate_masks(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
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
        """

        # Get sizing of the lowest-resolution image encoding
        v12_encimgs = encoded_image_features_list[0]
        patch_grid_hw = v12_encimgs[0].shape[2:]

        with _inference_mode(self._infmode):
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                v12_encimgs, encoded_prompts, grid_posenc, mask_hint, blank_promptless_output
            )

        return mask_preds, iou_preds

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
            compile_image_projection=compile_image_encoding,
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


class SAMV3TrackingModel(nn.Module):
    """
    Wrapper around SAMv3 tracking components, used for video segmentation (e.g. SAMv2 task).

    The basic usage of this model involves two phases. The first is to encode a prompt or mask
    'memory' which determines the object to be tracked. The second phase is to repeatedly make
    mask predictions for new incoming frames of a video using prior memory encodings.

    The basic usage for encoding an initial object is:
        model.encode_image(...)
        model.initialize_video_masking(...) OR .initialize_from_mask(...)

    The usage for repeatedly encoding new frames is:
        model.encode_image(...)
        model.step_video_masking(...)

    See the video segmentation example for more details:
    https://github.com/heyoeyo/muggled_sam/blob/main/simple_examples/video_segmentation.py
    """

    name = "samv3"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV3ImageEncoder,
        image_projection_model: SAMV3ImageProjection,
        coordinate_encoder_model: SAMV3CoordinateEncoder,
        prompt_encoder_model: SAMV3PromptEncoder,
        mask_decoder_model: SAMV3MaskDecoder,
        memory_encoder_model: SAMV3MemoryEncoder,
        memory_image_fusion_model: SAMV3MemoryImageFusion,
        enable_inference_mode: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Store components
        self.image_encoder = image_encoder_model
        self.image_projection = image_projection_model
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
        max_side_length: int | None = 1008,
        use_square_sizing: bool = True,
    ) -> tuple[tuple[list[Tensor], list[Tensor]], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The 'max_side_length' and 'use_square_sizing' inputs control the resolution and aspect ratio
        of the image before encoding.
        Returns:
            encoded_images_list, patch_grid_hw, preencoded_image_hw
        """

        return SAMV3InteractiveModel.encode_image(self, image_bgr, max_side_length, use_square_sizing)

    # .................................................................................................................

    def initialize_video_masking(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
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

        This function roughly corresponds to `add_new_points_or_box` in the SAMv3 implementation:
        https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracking_predictor.py#L179

        Returns:
            best_mask_prediction, memory_encoding, object_pointer
        """

        # Encode initial prompts (re-use interactive implementation as it's the same process)
        # -> This lets us 'hide' the prompt encoding function, which isn't meant to be used on tracking model
        encoded_prompts = SAMV3InteractiveModel.encode_prompts(
            self, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list
        )

        with _inference_mode(self._infmode):

            # For convenience
            v12_encimgs = encoded_image_features_list[0]
            lowres_imgenc, *hires_imgenc = v12_encimgs
            token_hw = lowres_imgenc.shape[2:]

            # Generate mask prediction from image/prompt encodings, as usual
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(token_hw)
            mask_preds, iou_preds, obj_ptrs, obj_score = self.mask_decoder(
                v12_encimgs,
                encoded_prompts,
                grid_posenc,
                mask_hint=mask_hint,
                blank_promptless_output=False,
            )

            # Select mask to use for initial encodings (auto-select if not given an index)
            if mask_index_select is None:
                mask_index_select = iou_preds[0].argmax()
            best_mask_pred = mask_preds[:, [mask_index_select], :, :]
            best_obj_ptr = obj_ptrs[:, [mask_index_select]]

            # Encode initial memory
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_pred, obj_score, is_prompt_encoding=True)

        return best_mask_pred, memory_encoding, best_obj_ptr

    # .................................................................................................................

    def initialize_from_mask(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        mask_image: ndarray | Tensor,
    ) -> tuple[Tensor, Tensor]:
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

        Returns:
            memory_encoding, object_pointer
        """

        with _inference_mode(self._infmode):

            # For convenience
            v12_encimgs = encoded_image_features_list[0]
            lowres_imgenc, *hires_imgenc = v12_encimgs
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

            # Try to force into Bx1xHxW shape
            mask_b, mask_n, mask_h, mask_w = mask_tensor.shape
            if mask_b == 1 and mask_n > 1:
                mask_tensor = mask_tensor.permute(1, 0, 2, 3)
                mask_b, mask_n, mask_h, mask_w = mask_tensor.shape
            assert mask_tensor.shape[1] == 1, "Mask shape error! Expecting '1' in shape index 1, eg. Bx1xHxW"

            # Make special-case pointer from mask, since we don't normally get one without prompting
            # https://github.com/facebookresearch/sam3/blob/86ed77094094e5cabb16b0414ec60c5ba9ce0a0f/sam3/model/sam3_tracker_base.py#L412
            pad_prompt_enc = self.prompt_encoder.create_padding_point_encoding()
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(token_hw)
            ptrs_b1c = self.mask_decoder.make_pointer_from_mask(v12_encimgs, pad_prompt_enc, grid_posenc, mask_tensor)

            # Generate memory encoding from mask
            memory_encoding = self.memory_encoder(lowres_imgenc, mask_tensor, None, is_prompt_encoding=True)

        return memory_encoding, ptrs_b1c

    # .................................................................................................................

    def step_video_masking(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
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

        Note:
        The 'num_multiplex_objects' is a dummy input included for compatibility
        with the v3.1 update. It doesn't do anything!

        This function corresponds to 'track_step' in the SAMv3 implementation:
        https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L930

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
            # See: https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L971
            v12_encimgs = encoded_image_features_list[0]
            lowres_imgenc, *hires_imgenc = v12_encimgs
            memfused_encimg = self.memory_image_fusion(
                lowres_imgenc,
                prompt_memory_encodings,
                prompt_object_pointers,
                previous_memory_encodings,
                previous_object_pointers,
            )

            # Run mask decoder on memory-fused features
            # Called '_forward_sam_heads' in original code
            # See: https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L992
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
            # See: https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L363
            best_mask_idx, best_mask_pred, _, best_obj_ptr = self.mask_decoder.get_best_decoder_results(
                mask_preds,
                iou_preds,
                obj_ptrs,
                exclude_0th_index=True,
            )

            # Encode new memory features
            # Called '_encode_new_memory' in original code
            # https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L1030
            memory_encoding = self.memory_encoder(lowres_imgenc, best_mask_pred, obj_score)

        return obj_score, best_mask_idx, mask_preds, memory_encoding, best_obj_ptr

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
            _, init_mem, init_ptr = self.initialize_video_masking(encoded_imgs, [], [(0.5, 0.5)], [])
            self.step_video_masking(encoded_imgs, [init_mem], [init_ptr], [], [])

        return _enable_compilation(
            self,
            compile_image_encoder=compile_image_encoding,
            compile_image_projection=compile_image_encoding,
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


class SAMV3DetectorModel(nn.Module):
    """
    Wrapper around SAMV3 detection components.
    Initializing this model also loads a BPE vocabulary from the file system, needed by the text encoder.

    The basic usage of this model is to call the 3 main functions (in order) to generate mask detections:
        model.encode_image(...)
        model.encode_exemplars(...)
        model.generate_detections(...)
    See the object detection example for more details:
    https://github.com/heyoeyo/muggled_sam/blob/main/simple_examples/object_detection.py

    The functions of this model (roughly) correspond with the original code base as follows:
        .encode_detection_image -> Sam3Processor.set_image
        .encode_exemplars       -> Sam3Processor.set_text_prompt + Sam3Image._encode_prompt + Sam3Image._run_encoder
        .generate_detections    -> Sam3Image._run_decoder + Sam3Image._run_segmentation_heads
    For more context, see the 'forward_grounding' function from the original implementation:
    https://github.com/facebookresearch/sam3/blob/5eb25fb54b70ec0cb16f949289053be091e16705/sam3/model/sam3_image.py#L442
    """

    name = "samv3"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV3ImageEncoder,
        image_projection_model: SAMV3ImageProjection,
        text_encoder_model: SAMV3TextEncoder,
        sampling_encoder_model: SAMV3SamplingEncoder,
        image_exemplar_fusion_model: SAMV3ImageExemplarFusion,
        exemplar_detector_model: SAMV3ExemplarDetector,
        exemplar_segmentation_model: SAMV3ExemplarSegmentation,
        enable_inference_mode: bool = True,
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
        self._infmode = enable_inference_mode

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

    # .................................................................................................................

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = 1008,
        use_square_sizing: bool = True,
    ) -> tuple[tuple[list[Tensor], list[Tensor]], tuple[int, int], tuple[int, int]]:
        """
        Function used to compute image encodings from a bgr formatted image (e.g. loaded from opencv)
        The 'max_side_length' and 'use_square_sizing' inputs control the resolution and aspect ratio
        of the image before encoding.
        Returns:
            encoded_images_list, patch_grid_hw, preencoded_image_hw
        """

        return SAMV3InteractiveModel.encode_image(self, image_bgr, max_side_length, use_square_sizing)

    # .................................................................................................................

    def encode_exemplars(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
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
        v3_encimgs = encoded_image_features_list[1]
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
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
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
        v3_encimgs = encoded_image_features_list[1]
        lowres_imgenc_bchw, hiresx2_imgenc_bchw, hiresx4_imgenc_bchw = v3_encimgs

        with _inference_mode(self._infmode):

            # Return 'blank' results if we don't get any exemplars
            # -> Not required (model still works with no exemplars), but blanked results make more sense
            exm_b, exm_n, _ = encoded_exemplars_bnc.shape
            no_exemplars = exm_n == 0
            if no_exemplars and blank_no_exemplar_outputs:
                blk_tok, blk_box, blk_score, blk_pres = self.exemplar_detector.create_blank_output(lowres_imgenc_bchw)
                blk_masks, _ = self.exemplar_segmentation.create_blank_output(blk_tok, lowres_imgenc_bchw)
                return blk_masks, blk_box, blk_score, blk_pres

            # Batch image encodings if exemplars are batched
            if exm_b > 1 and lowres_imgenc_bchw.shape[0] == 1:
                lowres_imgenc_bchw = lowres_imgenc_bchw.expand(exm_b, -1, -1, -1)
                hiresx2_imgenc_bchw = hiresx2_imgenc_bchw.expand(exm_b, -1, -1, -1)
                hiresx4_imgenc_bchw = hiresx4_imgenc_bchw.expand(exm_b, -1, -1, -1)

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

    def filter_detections(
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
            results = model.filter_detections(*outputs, score_threshold=0.75)

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

    def enable_compilation(
        self,
        example_image_bgr: ndarray | None = None,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
        compile_image_encoding: bool = True,
        compile_exemplar_encoding: bool = True,
        compile_detection_segmentation: bool = True,
        custom_config: dict | None = None,
    ) -> None:
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
        If not provided, some (fairly aggressive) default settings will be used
        """

        # Run model to fill in cache
        if example_image_bgr is not None:
            encoded_imgs, _, _ = self.encode_image(example_image_bgr, max_side_length, use_square_sizing)
            encoded_exemplars = self.encode_exemplars(
                encoded_imgs,
                text="visual",
                box_xy1xy2_norm_list=[[(0.2, 0.3), (0.5, 0.6)]],
                point_xy_norm_list=[(0.5, 0.5)],
                negative_boxes_list=[[(0.1, 0.1), (0.2, 0.2)]],
                negative_points_list=[(0.9, 0.1)],
            )
            self.generate_detections(encoded_imgs, encoded_exemplars)

        return _enable_compilation(
            self,
            compile_image_encoder=compile_image_encoding,
            compile_image_projection=compile_image_encoding,
            compile_text_encoder=compile_exemplar_encoding,
            compile_sampling_encoder=compile_exemplar_encoding,
            compile_image_exemplar_fusion=compile_exemplar_encoding,
            compile_exemplar_detector=compile_detection_segmentation,
            compile_exemplar_segmentation=compile_detection_segmentation,
            custom_config=custom_config,
        )

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

    def encode_detection_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[tuple[list[Tensor], list[Tensor]], tuple[int, int], tuple[int, int]]:
        """Temporary function for backwards compatibility. Will be removed in the future"""
        return self.encode_image(image_bgr, max_side_length, use_square_sizing)

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
