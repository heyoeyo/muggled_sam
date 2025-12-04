#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch

from .sam_v1_model import SAMV1Model

from .image_encoder_model import SAMV1ImageEncoder
from .coordinate_encoder_model import SAMV1CoordinateEncoder
from .prompt_encoder_model import SAMV1PromptEncoder
from .mask_decoder_model import SAMV1MaskDecoder

from .state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict
from .state_dict_conversion.convert_original_state_dict_keys import convert_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions

# .....................................................................................................................


def make_samv1_from_original_state_dict(
    original_state_dict: dict | str, strict_load=True, weights_only=True
) -> [dict, SAMV1Model]:
    """
    Function used to initialize a SAM (v1) model from a state dictionary (i.e. model weights) file.
    This function will automatically figure out the model sizing parameters from the state dict,
    assuming it comes from the original SAM repo (or matches the implementation).

    The state dict can be provided directly (e.g. from state_dict = torch.load()) or
    a string can be given, in which case it will be assumed to be a path to load the state dict

    Returns:
        model_config_dict, sam_model
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

    # Get model config from weights (i.e. beit_large_512 vs beit_base_384) & convert to new keys/state dict
    model_config_dict = get_model_config_from_state_dict(original_state_dict)
    new_state_dict = convert_state_dict_keys(model_config_dict, original_state_dict)

    # Load model & set model weights
    sam_model = make_sam_v1(**model_config_dict)
    sam_model.image_encoder.load_state_dict(new_state_dict["imgencoder"], strict_load)
    sam_model.coordinate_encoder.load_state_dict(new_state_dict["coordencoder"], strict_load)
    sam_model.prompt_encoder.load_state_dict(new_state_dict["promptencoder"], strict_load)
    sam_model.mask_decoder.load_state_dict(new_state_dict["maskdecoder"], strict_load)

    return model_config_dict, sam_model


# .....................................................................................................................


def make_sam_v1(
    features_per_image_token=768,
    num_encoder_blocks=12,
    num_encoder_heads=12,
    num_encoder_stages=4,
    base_patch_grid_hw=(64, 64),
    base_window_size=14,
    patch_size_px=16,
    features_per_prompt_token=256,
    features_per_decoder_token=128,
    num_decoder_blocks=2,
    num_decoder_heads=8,
    num_output_mask_tokens=4,
) -> SAMV1Model:
    """
    Helper used to build all SAM model components. The arguments for this function are
    expected to come from the 'make_samv1_from_original_state_dict' function, which
    will fill in the function arguments based on a loaded state dictionary.

    However, if you want to make a model without pretrained weights
    here are the following standard configs (from original SAM):
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/build_sam.py

    # vit-huge:
        features_per_image_token = 1280
        num_encoder_blocks = 32
        num_encoder_heads = 16
        num_encoder_stages = 4
        base_patch_grid_hw = (64, 64)
        base_window_size = 14
        patch_size_px = 16
        features_per_prompt_token = 256
        features_per_decoder_token = 128
        num_output_mask_tokens = 4
        num_decoder_heads = 8
        num_decoder_blocks = 2

    # vit-large
        features_per_image_token = 1024
        num_encoder_blocks = 24
        num_encoder_heads = 16
        num_encoder_stages = 4
        base_patch_grid_hw = (64, 64)
        base_window_size = 14
        patch_size_px = 16
        features_per_prompt_token = 256
        features_per_decoder_token = 128
        num_output_mask_tokens = 4
        num_decoder_heads = 8
        num_decoder_blocks = 2

    # vit-base
        features_per_image_token = 768
        num_encoder_blocks = 12
        num_encoder_heads = 12
        num_encoder_stages = 4
        base_patch_grid_hw = (64, 64)
        base_window_size = 14
        patch_size_px = 16
        features_per_prompt_token = 256
        features_per_decoder_token = 128
        num_output_mask_tokens = 4
        num_decoder_heads = 8
        num_decoder_blocks = 2
    """

    # Construct model components
    imgenc_model = SAMV1ImageEncoder(
        features_per_image_token,
        num_encoder_blocks,
        num_encoder_heads,
        base_window_size,
        base_patch_grid_hw,
        features_per_prompt_token,
        patch_size_px,
        num_encoder_stages,
    )

    coordenc_model = SAMV1CoordinateEncoder(features_per_prompt_token)

    promptenc_model = SAMV1PromptEncoder(features_per_prompt_token)

    maskdec_model = SAMV1MaskDecoder(
        features_per_prompt_token,
        features_per_decoder_token,
        num_decoder_blocks,
        num_decoder_heads,
        num_output_mask_tokens,
    )

    # Bundle components into complete SAM model!
    return SAMV1Model(imgenc_model, coordenc_model, promptenc_model, maskdec_model)
