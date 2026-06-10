#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from json import loads as load_json_str
from contextlib import contextmanager

# For legacy warnings
from inspect import currentframe
from time import sleep
import sys

import torch
import torch.nn as nn

from .compilation import enable_compilation as _enable_compilation

# For type hints
from typing import TypeAlias
from torch import Tensor
from numpy import ndarray
from .image_encoder_model import SAMV1ImageEncoder
from .coordinate_encoder_model import SAMV1CoordinateEncoder
from .prompt_encoder_model import SAMV1PromptEncoder
from .mask_decoder_model import SAMV1MaskDecoder


# ---------------------------------------------------------------------------------------------------------------------
# %% Custom data types

SAMv1ImageEncoding: TypeAlias = list[Tensor]
XYPoint: TypeAlias = tuple[float, float]
XY1XY2: TypeAlias = tuple[XYPoint, XYPoint]


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV1Core(nn.Module):
    """
    Holds core SAMv1 model components.

    This module is not meant to be used directly, instead, it's used to create
    separate 'contexts' to perform specific tasks supported by the model.
    For SAMv1, there is only a single context:
        interact_model = sam1_model.get_interactive_context()

    This makes the usage a bit over-complicated, but is done to enable inter-operability
    with future SAM models (v2 & v3) that have additional capabilities.

    This module also holds legacy implementations of segmentation functionality,
    though this will be removed in future updates!
    """

    name = "samv1"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV1ImageEncoder,
        coordinate_encoder_model: SAMV1CoordinateEncoder,
        prompt_encoder_model: SAMV1PromptEncoder,
        mask_decoder_model: SAMV1MaskDecoder,
        config_bytes: bytearray,
        enable_inference_mode: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Store config data
        self.register_buffer("config_muggled_samv1", torch.tensor(config_bytes, dtype=torch.uint8))

        # Store SAM model components
        self.image_encoder = image_encoder_model
        self.coordinate_encoder = coordinate_encoder_model
        self.prompt_encoder = prompt_encoder_model
        self.mask_decoder = mask_decoder_model

        # Default to eval mode, expecting to use inference only
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        self._infmode = enable_inference_mode

    # .................................................................................................................

    def encode_prompts(self, box_xy1xy2_norm_list: list, fg_xy_norm_list: list, bg_xy_norm_list: list) -> Tensor:
        """Temporary placeholder for backwards compatibility"""
        _legacy_warning(currentframe())
        return SAMV1InteractiveModel.encode_prompts(self, box_xy1xy2_norm_list, fg_xy_norm_list, bg_xy_norm_list)

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> tuple[list[Tensor], tuple[int, int], tuple[int, int]]:
        """Temporary placeholder for backwards compatibility"""
        _legacy_warning(currentframe())
        encoded_img = SAMV1InteractiveModel.encode_image(self, image_bgr, max_side_length, use_square_sizing)

        # Figure out additional outputs as needed by old API
        img_tensor = self.prepare_image_batch([image_bgr], max_side_length, use_square_sizing)
        preencode_hw = tuple(img_tensor.shape[-2:])
        token_hw = tuple(encoded_img[0].shape[-2:])
        return encoded_img, token_hw, preencode_hw

    def generate_masks(
        self,
        encoded_image_features_list: tuple[list[Tensor], list[Tensor]],
        encoded_prompts: Tensor,
        mask_hint: Tensor | int | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Temporary placeholder for backwards compatibility"""
        _legacy_warning(currentframe())
        return SAMV1InteractiveModel.generate_masks(
            self, encoded_image_features_list, encoded_prompts, mask_hint, blank_promptless_output
        )

    def prepare_image_batch(
        self,
        images_bgr_list: list[ndarray],
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
    ) -> Tensor:
        """Temporary placeholder for backwards compatibility"""
        _legacy_warning(currentframe())
        return SAMV1InteractiveModel.prepare_image_batch(self, images_bgr_list, max_side_length, use_square_sizing)

    # .................................................................................................................

    def get_config(self) -> dict:
        """Returns model config dictionary"""
        # Convert config from tensor->bytes->string/json->dictionary
        config_as_bytes = bytearray(self.config_muggled_samv1.cpu().tolist())
        config_as_str = config_as_bytes.decode()
        return load_json_str(config_as_str)

    # .................................................................................................................

    def get_interactive_context(self) -> nn.Module:
        """Creates an interactive model, used for user-prompted image segmentation"""
        return SAMV1InteractiveModel(
            self.image_encoder,
            self.coordinate_encoder,
            self.prompt_encoder,
            self.mask_decoder,
            self._infmode,
        )

    def get_tracking_context(self, *args, **kwargs) -> None:
        """Warning for unsupported feature"""
        raise AttributeError("SAMv1 does not support video tracking (requires SAMv2/v3)")

    def get_detector_context(self, *args, **kwargs) -> None:
        """Warning for unsupported feature"""
        raise AttributeError("SAMv1 does not support object detection (requires SAMv3)")

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


class SAMV1InteractiveModel(nn.Module):
    """
    Wrapper around the SAMV1 interactive components, used for image segmentation.
    This isn't really neccessary, as SAMv1 only supports interactive use, but is done
    in order to maintain compatibility with SAMv2 & v3 implementations.

    The basic usage of this model is to call the 3 main functions (in order) to generate mask predictions:
        model.encode_image(...)
        model.encode_prompts(...)
        model.generate_masks(...)
    See the image segmentation example for more details:
    https://github.com/heyoeyo/muggled_sam/blob/main/simple_examples/image_segmentation.py
    """

    name = "samv1"

    # .................................................................................................................

    def __init__(
        self,
        image_encoder_model: SAMV1ImageEncoder,
        coordinate_encoder_model: SAMV1CoordinateEncoder,
        prompt_encoder_model: SAMV1PromptEncoder,
        mask_decoder_model: SAMV1MaskDecoder,
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
    ) -> SAMv1ImageEncoding:
        """
        Encode image data, this is one of the inputs needed to generate masks

        Returns:
            encoded_image
            -> The encoded image is a list containing a single feature map with
               shape: Bx256x64x64 (using default settings). This is
               wrapped in a list for compatibility with SAMv2/v3
        """

        with _inference_mode(self._infmode):
            image_rgb_normalized_bchw = self.image_encoder.prepare_image(image_bgr, max_side_length, use_square_sizing)
            encoded_image_tensor = self.image_encoder(image_rgb_normalized_bchw)

        # Create list version of image encoding, purely for compatibility with SAMv2/v3
        encoded_image = [encoded_image_tensor]
        return encoded_image

    # .................................................................................................................

    def encode_prompts(
        self,
        box_xy1xy2_norm_list: list[XY1XY2] | Tensor | None,
        fg_xy_norm_list: list[XYPoint] | Tensor | None,
        bg_xy_norm_list: list[XYPoint] | Tensor | None,
    ) -> Tensor:
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

        Alternatively, boxes or points can be given directly as tensors.
        In this case, boxes must have shape: Nx2x2 or (for batching) BxNx2x2,
        while points must have shape: Nx2 or BxNx2,
        where B is the batch size and N is the number of prompts.
        If using a batch size > 1, all non-empty prompts must have the same batch size!

        Returns:
            encoded_prompts_bnc
            -> shape: BxNxC, B batch size, N number of prompt points, C is channels/feature count
        """

        with _inference_mode(self._infmode):
            boxes_tensor = self.coordinate_encoder.prepare_boxes(box_xy1xy2_norm_list)
            fg_tensor = self.coordinate_encoder.prepare_points(fg_xy_norm_list)
            bg_tensor = self.coordinate_encoder.prepare_points(bg_xy_norm_list)
            box_posenc, fg_posenc, bg_posenc = self.coordinate_encoder(boxes_tensor, fg_tensor, bg_tensor)
            encoded_prompts_bnc = self.prompt_encoder(box_posenc, fg_posenc, bg_posenc)

        return encoded_prompts_bnc

    # .................................................................................................................

    def generate_masks(
        self,
        encoded_image: SAMv1ImageEncoding,
        encoded_prompts: Tensor,
        mask_hint: Tensor | None = None,
        blank_promptless_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Function used to generate segmentation masks given an image encoding,
        as well as a prompt encoding and potentially a mask hint/prompt. These
        input encodings are expected to come from other model components.

        Returns:
            mask_predictions_bnhw, iou_predictions_bn
            -> Masks have shape: Bx4xHxW (HxW is 256x256 using default settings)
            -> iou_predictions have shape: Bx4
        """

        with _inference_mode(self._infmode):
            encoded_img_tensor = encoded_image[0]
            patch_grid_hw = encoded_img_tensor.shape[2:]
            grid_posenc = self.coordinate_encoder.get_grid_position_encoding(patch_grid_hw)
            masks_bnhw, ious_bn, _ = self.mask_decoder(
                encoded_img_tensor, encoded_prompts, grid_posenc, mask_hint, blank_promptless_output
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
            encoded_imgs = self.encode_image(example_image_bgr, max_side_length, use_square_sizing)
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


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers

# Global used to store names of legacy function calls (so we don't repeat warning/delay)
_LEGACY_WARN_SET = set()


def _legacy_warning(inspect_stack_frame, sleep_delay_sec: float = 1.0):
    """Helper used to warn about removal of old functions"""
    caller_func_name = inspect_stack_frame.f_code.co_name
    if caller_func_name not in _LEGACY_WARN_SET:
        _LEGACY_WARN_SET.add(caller_func_name)
        print(
            f"{'*' * 12} WARNING {'*' * 12}",
            f"Function ({caller_func_name}) will be removed by June 2026",
            "Please update to use new API. See CHANGELOG:",
            "https://github.com/heyoeyo/muggled_sam/blob/main/CHANGELOG.md",
            "",
            sep="\n",
            flush=True,
            file=sys.stderr,
        )
        sleep(sleep_delay_sec)
    return


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
