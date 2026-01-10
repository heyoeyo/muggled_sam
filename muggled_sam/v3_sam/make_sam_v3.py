#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch

from .sam_v3_model import SAMV3Model

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

from .state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict
from .state_dict_conversion.convert_original_state_dict_keys import SAM3StageType, convert_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions

# .....................................................................................................................


def make_samv3_from_original_state_dict(
    original_state_dict: dict | str, strict_load=True, weights_only=True
) -> [dict, SAMV3Model]:
    """
    Function used to initialize a SAMV3 model from a state dictionary (i.e. model weights) file.
    This function will automatically figure out the model sizing parameters from the state dict,
    assuming it comes from the original SAMV3 repo (or matches the implementation).

    The state dict can be provided directly (e.g. from state_dict = torch.load(...)) or
    a string can be given, in which case it will be assumed to be a path to load the state dict

    Returns:
        model_config_dict, sam_v3_model
    """

    # If we're given a string, assume it's a path to the state dict
    need_to_load = isinstance(original_state_dict, str)
    if need_to_load:
        path_to_state_dict = original_state_dict
        # Load model weights with fail check in case weights are in cuda format and user doesn't have cuda
        try:
            original_state_dict = torch.load(path_to_state_dict, weights_only=weights_only)
        except RuntimeError:
            original_state_dict = torch.load(path_to_state_dict, map_location="cpu", weights_only=weights_only)

    # Feedback on using non-strict loading
    if not strict_load:
        print(
            "",
            "WARNING:",
            "  Loading model weights without 'strict' mode enabled!",
            "  Some weights may be missing or unused!",
            sep="\n",
            flush=True,
        )

    # Feedback for not using weights_only
    if not weights_only:
        print(
            "",
            "*" * 16,
            "*" * 16,
            "*" * 16,
            "WARNING (dangerous):",
            "  Loading model weights without 'weights_only' enabled!",
            "  This can allow for arbitrary executable code to be loaded.",
            "  Be sure that the loaded weights are from a trusted source!",
            "",
            "  !!! This should not normally be disabled !!!",
            "*" * 16,
            "*" * 16,
            "*" * 16,
            sep="\n",
            flush=True,
        )

    # Get model config from weights (i.e. sam large vs sam base) & convert to new keys/state dict
    model_config_dict = get_model_config_from_state_dict(original_state_dict)
    new_state_dict = convert_state_dict_keys(model_config_dict, original_state_dict)

    # Load model & set model weights
    sam_model = make_sam_v3(**model_config_dict)

    # Image encoding & masking
    sam_model.image_encoder.load_state_dict(new_state_dict[SAM3StageType.image_encoder], strict_load)
    sam_model.image_projection.load_state_dict(new_state_dict[SAM3StageType.image_projection], strict_load)
    sam_model.coordinate_encoder.load_state_dict(new_state_dict[SAM3StageType.coordinate_encoder], strict_load)
    sam_model.prompt_encoder.load_state_dict(new_state_dict[SAM3StageType.prompt_encoder], strict_load)
    sam_model.mask_decoder.load_state_dict(new_state_dict[SAM3StageType.mask_decoder], strict_load)

    # Video components
    sam_model.memory_encoder.load_state_dict(new_state_dict[SAM3StageType.memory_encoder], strict_load)
    sam_model.memory_image_fusion.load_state_dict(new_state_dict[SAM3StageType.memory_image_fusion], strict_load)

    # Detector components
    sam_model.text_encoder.load_state_dict(new_state_dict[SAM3StageType.text_encoder], strict_load)
    sam_model.sampling_encoder.load_state_dict(new_state_dict[SAM3StageType.sampling_encoder], strict_load)
    sam_model.image_exemplar_fusion.load_state_dict(new_state_dict[SAM3StageType.image_exemplar_fusion], strict_load)
    sam_model.exemplar_detector.load_state_dict(new_state_dict[SAM3StageType.exemplar_detector], strict_load)
    sam_model.exemplar_segmentation.load_state_dict(new_state_dict[SAM3StageType.exemplar_segmentation], strict_load)

    return model_config_dict, sam_model


# .....................................................................................................................


def make_sam_v3(
    features_per_prompt_token=256,
    features_per_decoder_token=128,
    features_per_memory_token=64,
    features_per_detection_token=256,
    imgencoder_features=1024,
    imgencoder_num_stages=4,
    imgencoder_num_blocks=32,
    imgencoder_num_heads=16,
    imgencoder_patch_size_px=14,
    imgencoder_posenc_tile_hw=(24, 24),
    imgencoder_window_size=24,
    maskdecoder_num_blocks=2,
    maskdecoder_num_heads=8,
    maskdecoder_num_mask_tokens=4,
    memencoder_num_downsample_layers=4,
    memencoder_num_mixer_layers=2,
    memimgfusion_num_fusion_layers=4,
    txtencoder_features=1024,
    txtencoder_num_blocks=24,
    txtencoder_num_heads=16,
    txtencoder_vocab_size=49408,
    samplingenc_num_blocks=3,
    samplingenc_num_heads=8,
    imgexmfuse_num_blocks=6,
    imgexmfuse_num_heads=8,
    exmdetector_num_detections=200,
    exmdetector_num_blocks=6,
    exmdetector_num_heads=8,
    exmsegment_num_heads=8,
) -> SAMV3Model:
    """
    Helper used to build all SAMV3 model components. The arguments for this function are
    expected to come from the 'make_samv3_from_original_state_dict' function, which
    will fill in the function arguments based on a loaded state dictionary.

    See the original 'model builder' code for more information:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model_builder.py
    """

    # Construct model components
    imgenc_model = SAMV3ImageEncoder(
        imgencoder_features,
        imgencoder_num_stages,
        imgencoder_num_blocks,
        imgencoder_num_heads,
        imgencoder_patch_size_px,
        imgencoder_posenc_tile_hw,
        imgencoder_window_size,
    )
    imgproj_model = SAMV3ImageProjection(imgencoder_features, features_per_prompt_token, features_per_detection_token)
    coordenc_model = SAMV3CoordinateEncoder(features_per_prompt_token)
    promptenc_model = SAMV3PromptEncoder(features_per_prompt_token)
    maskdec_model = SAMV3MaskDecoder(
        features_per_prompt_token,
        features_per_decoder_token,
        maskdecoder_num_blocks,
        maskdecoder_num_heads,
        maskdecoder_num_mask_tokens,
    )

    # Video components
    memenc_model = SAMV3MemoryEncoder(
        features_per_prompt_token,
        features_per_memory_token,
        num_downsample_layers=memencoder_num_downsample_layers,
        num_mixer_layers=memencoder_num_mixer_layers,
    )
    memfuse_model = SAMV3MemoryImageFusion(
        features_per_prompt_token,
        features_per_memory_token,
        num_layers=memimgfusion_num_fusion_layers,
    )

    # Detection components
    txtenc_model = SAMV3TextEncoder(
        txtencoder_features,
        features_per_detection_token,
        txtencoder_num_blocks,
        txtencoder_num_heads,
        txtencoder_vocab_size,
    )
    sample_model = SAMV3SamplingEncoder(features_per_detection_token, samplingenc_num_blocks, samplingenc_num_heads)
    imgexm_model = SAMV3ImageExemplarFusion(features_per_detection_token, imgexmfuse_num_blocks, imgexmfuse_num_heads)
    exmdet_model = SAMV3ExemplarDetector(
        features_per_detection_token,
        exmdetector_num_detections,
        exmdetector_num_blocks,
        exmdetector_num_heads,
    )
    exmseg_model = SAMV3ExemplarSegmentation(features_per_detection_token, exmsegment_num_heads)

    # Bundle components into final model
    return SAMV3Model(
        imgenc_model,
        imgproj_model,
        coordenc_model,
        promptenc_model,
        maskdec_model,
        memenc_model,
        memfuse_model,
        txtenc_model,
        sample_model,
        imgexm_model,
        exmdet_model,
        exmseg_model,
    )
