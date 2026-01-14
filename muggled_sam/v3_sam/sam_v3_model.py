#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os.path as osp

import numpy as np
import torch
import torch.nn as nn

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
    Wrapper around separated SAMv3 model components, so that the model can be used as a singular entity.
    Note that SAMv3 consists of 2 (mostly) separate models, one for user-directed prompting
    (e.g. points and bounding boxes) and one for automated detections (more like a YOLO model).

    These separate models lead to very different use cases and as such, the functionality of
    this model comes from calling specific methods for different use cases
    (unlike typical models where the .forward(...) method contains all functionality).
    """

    name = "samv3"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV3ImageEncoder,
        image_projection_model: SAMV3ImageProjection,
        coordinate_encoder: SAMV3CoordinateEncoder,
        prompt_encoder_model: SAMV3PromptEncoder,
        mask_decoder_model: SAMV3MaskDecoder,
        memory_encoder_model: SAMV3MemoryEncoder,
        memory_image_fusion_model: SAMV3MemoryImageFusion,
        text_encoder_model: SAMV3TextEncoder,
        sampling_encoder_model: SAMV3SamplingEncoder,
        image_exemplar_fusion_model: SAMV3ImageExemplarFusion,
        exemplar_detector_model: SAMV3ExemplarDetector,
        exemplar_segmentation_model: SAMV3ExemplarSegmentation,
    ):

        # Inherit from parent
        super().__init__()

        # Store base SAM components
        self.image_encoder = image_encoder_model
        self.image_projection = image_projection_model
        self.coordinate_encoder = coordinate_encoder
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
        self.eval()

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

        with torch.inference_mode():
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_tlbr_norm_list)
            fg_tensor, bg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list, bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts

    # .................................................................................................................

    def encode_image(
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

        Returns:
            encoded_images_list, patch_grid_hw, preencoded_image_hw
            -> Encoded images list contains 3 multi-resolution feature maps
               they have shapes: Bx256x72x72, Bx64x144x144, Bx32x288x288
               (using default settings). The first-most feature map is
               the 'low-res' map needed by several other parts of the model
            -> The patch_grid_hw contains the height & width of the low-res
               feature map (72x72 with default 1008x1008 input sizing)
            -> The preencoded_image_hw contains the height & width of the
               input image after pre-processing just before being encoded,
               by default it would be 1008x1008
        """

        with torch.inference_mode():
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            encoded_img = self.image_encoder(image_rgb_normalized_bchw)
            encoded_image_features_list = self.image_projection.v2_projection(encoded_img)

        # Get patch sizing of lowest-res tokens (as needed by other components) & size of processed image
        patch_grid_hw = encoded_image_features_list[0].shape[2:]
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
        """

        # Get sizing of the lowest-resolution image encoding
        patch_grid_hw = encoded_image_features_list[0].shape[2:]

        with torch.inference_mode():
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

        This function roughly corresponds to `add_new_points_or_box` in the SAMv3 implementation:
        https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracking_predictor.py#L179

        Returns:
            best_mask_prediction, memory_encoding, object_pointer
        """

        # Encode initial prompts
        encoded_prompts = self.encode_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

        with torch.inference_mode():

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
        Alternate video tracking initialization option. In this case, using a provided mask image as a 'prompt'.
        The provided image is assumed to be loaded using opencv, so that it has shape: HxW or HxWxC
        If the image has channels (e.g. RGB), only the 0th channel (e.g. red) will be used.

        Note that with this form of initializtion, there is no object pointer! The pointer normally
        comes from the mask prediction, so without a prediction, there is not pointer. The video
        masking should therefore be initialized with only the memory encoding and an empty pointer list.
        This doesn't have a substantial impact on the tracking

        Returns:
            memory_encoding
        """

        with torch.inference_mode():

            # For convenience
            lowres_imgenc, *hires_imgenc = encoded_image_features_list
            device, dtype = lowres_imgenc.device, lowres_imgenc.dtype
            token_h, token_w = lowres_imgenc.shape[2:]

            # Hard-code the object score as being 'high/confident', since we assume the given mask is accurate
            obj_score = torch.tensor(100.0, device=device, dtype=dtype)

            # Force input into a boolean mask & convert to torch tensor
            if isinstance(mask_image, ndarray):
                if mask_image.dtype != np.bool:
                    mask_image = mask_image > 0
                mask_tensor = torch.tensor(mask_image, device=device, dtype=dtype)
            elif isinstance(mask_image, Tensor):
                if mask_image.dtype != torch.bool:
                    mask_image = mask_image > 0
                mask_tensor = mask_image.to(device=device, dtype=dtype)
            assert isinstance(mask_tensor, Tensor), "Unsupported mask type! Must be a numpy array or torch tensor"

            # Make sure we get a mask with shape: BxNxHxW
            if mask_tensor.ndim == 2:
                # Convert HxW -> 1x1xHxW
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.ndim == 3 and mask_tensor.shape[-1] <= 3:
                # Convert (opencv image format) HxWxC -> 1x1xHxW
                mask_tensor = mask_tensor[:, :, 0].unsqueeze(0).unsqueeze(0)
            elif mask_tensor.ndim == 3:
                # Convert BxHxW -> Bx1xHxW
                mask_tensor = mask_tensor.unsqueeze(1)
            assert (
                mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1
            ), "Unsupported mask shape, must be: HxW, HxWxC, BxHxW or Bx1xHxW"

            # Scale input to correct size before encoding
            mask_tensor = nn.functional.interpolate(mask_tensor, size=(4 * token_h, 4 * token_w))
            memory_encoding = self.memory_encoder(lowres_imgenc, mask_tensor, obj_score, is_prompt_encoding=True)

        return memory_encoding

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

        with torch.inference_mode():

            # Encode image features with previous memory encodings & object pointer data
            # Called '_prepare_memory_conditioned_features' in original code
            # See: https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/sam3_tracker_base.py#L971
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

    def get_best_mask_index(self, iou_predictions: Tensor) -> int:
        """Returns the index of the highest IoU prediction score"""
        return self.mask_decoder.get_best_mask_index(iou_predictions)

    # .................................................................................................................

    def check_have_prompts(self, box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list) -> bool:
        """Helper used to check if there are any prompts"""
        return self.prompt_encoder.check_have_prompts(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list)

    # .................................................................................................................

    def make_detector_model(self, bpe_vocab_path: str | None = None):
        """Creates 'detection' model, used for generating bounding box & segmentation masks from exemplars"""
        return SAMV3DetectorModel(
            self.image_encoder,
            self.image_projection,
            self.text_encoder,
            self.sampling_encoder,
            self.image_exemplar_fusion,
            self.exemplar_detector,
            self.exemplar_segmentation,
            bpe_vocab_path,
        )

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
        image_encoder_model: SAMV3ImageEncoder,
        image_projection_model: SAMV3ImageProjection,
        text_encoder_model: SAMV3TextEncoder,
        sampling_encoder_model: SAMV3SamplingEncoder,
        image_exemplar_fusion_model: SAMV3ImageExemplarFusion,
        exemplar_detector_model: SAMV3ExemplarDetector,
        exemplar_segmentation_model: SAMV3ExemplarSegmentation,
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

    # .................................................................................................................

    def encode_detection_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
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

        with torch.inference_mode():
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            encoded_img = self.image_encoder(image_rgb_normalized_bchw)
            encoded_image_features_list = self.image_projection.v3_projection(encoded_img)

        # Get patch sizing of lowest-res tokens (as needed by other components) & size of processed image
        patch_grid_hw = encoded_image_features_list[0].shape[2:]
        image_preenc_hw = image_rgb_normalized_bchw.shape[2:]

        return encoded_image_features_list, patch_grid_hw, image_preenc_hw

    # .................................................................................................................

    def encode_exemplars(
        self,
        encoded_image_features_list: list[Tensor],
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
        lowres_imgenc_bchw = encoded_image_features_list[0]
        img_b, img_c, _, _ = lowres_imgenc_bchw.shape

        with torch.inference_mode():

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
        encoded_image_features_list: list[Tensor],
        encoded_exemplars_bnc: Tensor,
        detection_filter_threshold: float = 0.0,
        exemplar_padding_mask_bn: Tensor | None = None,
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
        lowres_imgenc_bchw, hiresx2_imgenc_bchw, hiresx4_imgenc_bchw = encoded_image_features_list

        with torch.inference_mode():

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
