#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from typing import Protocol, TypeAlias

from torch import Tensor
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Model-specific data types

# Various image encoding formats
SAMv1ImageEncoding: TypeAlias = list[Tensor]
SAMv2ImageEncoding: TypeAlias = list[Tensor]
SAMv3ImageEncoding: TypeAlias = tuple[list[Tensor], list[Tensor]]
SAMv3p1ImageEncoding: TypeAlias = tuple[list[Tensor], list[Tensor], list[Tensor]]

# Various memory encoding formats
SAMv2MemoryEncoding: TypeAlias = tuple[Tensor, Tensor]
SAMv3MemoryEncoding: TypeAlias = tuple[Tensor, Tensor]
SAMv3p1MemoryEncoding: TypeAlias = tuple[Tensor, Tensor, Tensor]
SAMMemoryEncoding: TypeAlias = SAMv2MemoryEncoding | SAMv3MemoryEncoding | SAMv3p1MemoryEncoding

# Coordinate formats
XYPoint: TypeAlias = tuple[float, float]
XY1XY2: TypeAlias = tuple[XYPoint, XYPoint]


# ---------------------------------------------------------------------------------------------------------------------
# %% Model interfaces


class SAMInteractive(Protocol):

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None,
        use_square_sizing: bool,
    ) -> SAMv1ImageEncoding | SAMv2ImageEncoding | SAMv3ImageEncoding | SAMv3p1ImageEncoding: ...

    def encode_prompts(
        self,
        box_xy1xy2_norm_list: list[XY1XY2] | Tensor | None,
        fg_xy_norm_list: list[XYPoint] | Tensor | None,
        bg_xy_norm_list: list[XYPoint] | Tensor | None,
    ) -> Tensor: ...

    def generate_masks(
        self,
        encoded_image: SAMv1ImageEncoding | SAMv2ImageEncoding | SAMv3ImageEncoding | SAMv3p1ImageEncoding,
        encoded_prompts: Tensor,
        mask_hint: Tensor | None,
        blank_promptless_output: bool,
    ) -> tuple[Tensor, Tensor]: ...

    def prepare_image_batch(
        self,
        images_bgr_list: list[ndarray],
        max_side_length: int | None,
        use_square_sizing: bool,
    ) -> Tensor: ...

    def enable_compilation(
        self,
        example_image_bgr: ndarray | None,
        max_side_length: int | None,
        use_square_sizing: bool,
        compile_image_encoding: bool,
        compile_mask_generation: bool,
        custom_config: dict | None,
    ) -> None: ...

    def toggle_inference_mode(self, enable_inference_mode: bool | None) -> bool: ...


# .....................................................................................................................


class SAMTracking(Protocol):

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None,
        use_square_sizing: bool,
    ) -> SAMv2ImageEncoding | SAMv3ImageEncoding | SAMv3p1ImageEncoding: ...

    def encode_prompt_memory(
        self,
        encoded_image: SAMv2ImageEncoding | SAMv3ImageEncoding | SAMv3p1ImageEncoding,
        box_xy1xy2_norm_list: list[XY1XY2] | Tensor | None,
        fg_xy_norm_list: list[XYPoint] | Tensor | None,
        bg_xy_norm_list: list[XYPoint] | Tensor | None,
        mask_hint: Tensor | None,
        mask_index: int | None,
    ) -> tuple[Tensor, SAMMemoryEncoding]: ...

    def encode_prompt_memory_from_mask(
        self,
        encoded_image: SAMv2ImageEncoding | SAMv3ImageEncoding | SAMv3p1ImageEncoding,
        mask_image: ndarray | Tensor,
    ) -> SAMMemoryEncoding: ...

    def encode_frame_memory(
        self,
        encoded_image: SAMv2ImageEncoding | SAMv3ImageEncoding | SAMv3p1ImageEncoding,
        mask_predictions: Tensor,
        object_pointers: Tensor | None,
        object_score: Tensor | None,
        mask_index: Tensor | int | None,
        is_prompt_encoding: bool,
    ) -> SAMMemoryEncoding: ...

    def step_video_masking(
        self,
        encoded_image: SAMv2ImageEncoding | SAMv3ImageEncoding | SAMv3p1ImageEncoding,
        prompt_memory_encodings: list[SAMMemoryEncoding] | SAMMemoryEncoding,
        frame_memory_encodings: list[SAMMemoryEncoding] | SAMMemoryEncoding,
        return_best_only: bool,
        is_recent_first: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def enable_compilation(
        self,
        example_image_bgr: ndarray | None,
        max_side_length: int | None,
        use_square_sizing: bool,
        compile_image_encoding: bool,
        compile_mask_generation: bool,
        compile_memory_encoding: bool,
        custom_config: dict | None,
    ) -> None: ...

    def toggle_inference_mode(self, enable_inference_mode: bool | None) -> bool: ...


# .....................................................................................................................


class SAMDetector(Protocol):

    def encode_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None,
        use_square_sizing: bool,
    ) -> SAMv3ImageEncoding | SAMv3p1ImageEncoding: ...

    def encode_exemplars(
        self,
        encoded_image: SAMv3ImageEncoding | SAMv3p1ImageEncoding,
        text: str | None,
        box_xy1xy2_norm_list: list[XY1XY2] | None,
        point_xy_norm_list: list[XYPoint] | None,
        negative_boxes_list: list[XY1XY2] | None,
        negative_points_list: list[XYPoint] | None,
        include_coordinate_encodings: bool,
    ) -> Tensor: ...

    def generate_detections(
        self,
        encoded_image: SAMv3ImageEncoding | SAMv3p1ImageEncoding,
        encoded_exemplars_bnc: Tensor,
        detection_filter_threshold: float,
        exemplar_padding_mask_bn: Tensor | None,
        blank_no_exemplar_outputs: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def filter_detections(
        self,
        mask_predictions: Tensor | None,
        box_predictions: Tensor | None,
        detection_scores: Tensor,
        presence_score: Tensor | None,
        score_threshold: float,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor | None]: ...

    def make_exemplar_batch(*encoded_exemplars_bnc: Tensor) -> tuple[Tensor, Tensor]: ...

    def enable_compilation(
        self,
        example_image_bgr: ndarray | None,
        max_side_length: int | None,
        use_square_sizing: bool,
        compile_image_encoding: bool,
        compile_exemplar_encoding: bool,
        compile_detection_segmentation: bool,
        custom_config: dict | None,
    ) -> None: ...

    def toggle_inference_mode(self, enable_inference_mode: bool | None) -> bool: ...


# .....................................................................................................................


class SAMCore(Protocol):

    name: str

    def get_config(self) -> dict: ...

    def get_interactive_context(self) -> SAMInteractive: ...

    def get_tracking_context(self) -> SAMTracking: ...

    def get_detector_context(self, bpe_vocab_path: str | None) -> SAMDetector: ...

    def toggle_inference_mode(self, enable_inference_mode: bool | None) -> bool: ...
