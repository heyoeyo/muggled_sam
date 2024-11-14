#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch

from .sam_v2_model import SAMV2Model

from .image_encoder_model import SAMV2ImageEncoder
from .coordinate_encoder_model import SAMV2CoordinateEncoder
from .prompt_encoder_model import SAMV2PromptEncoder
from .mask_decoder_model import SAMV2MaskDecoder
from .memory_fusion_model import SAMV2MemoryFusion
from .memory_encoder_model import SAMV2MemoryEncoder

from .state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict
from .state_dict_conversion.convert_original_state_dict_keys import convert_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions

# .....................................................................................................................


def make_samv2_from_original_state_dict(
    original_state_dict: dict | str, strict_load=True, weights_only=True
) -> [dict, SAMV2Model]:
    """
    Function used to initialize a SAMV2 model from a state dictionary (i.e. model weights) file.
    This function will automatically figure out the model sizing parameters from the state dict,
    assuming it comes from the original SAMV2 repo (or matches the implementation).

    The state dict can be provided directly (e.g. from state_dict = torch.load(...)) or
    a string can be given, in which case it will be assumed to be a path to load the state dict

    Returns:
        model_config_dict, sam_v2_model
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

    # Remove first layer 'model' key, if present
    if "model" in original_state_dict.keys():
        original_state_dict = original_state_dict["model"]

    # Get model config from weights (i.e. sam large vs sam base) & convert to new keys/state dict
    model_config_dict = get_model_config_from_state_dict(original_state_dict)
    new_state_dict = convert_state_dict_keys(model_config_dict, original_state_dict)

    # Load model & set model weights
    sam_model = make_sam_v2(**model_config_dict)
    sam_model.image_encoder.load_state_dict(new_state_dict["imgencoder"], strict_load)
    sam_model.coordinate_encoder.load_state_dict(new_state_dict["coordencoder"], strict_load)
    sam_model.prompt_encoder.load_state_dict(new_state_dict["promptencoder"], strict_load)
    sam_model.mask_decoder.load_state_dict(new_state_dict["maskdecoder"], strict_load)
    sam_model.memory_encoder.load_state_dict(new_state_dict["memoryencoder"], strict_load)
    sam_model.memory_fusion.load_state_dict(new_state_dict["memoryfusion"], strict_load)

    return model_config_dict, sam_model


# .....................................................................................................................


def make_sam_v2(
    features_per_image_token=112,
    features_per_prompt_token=256,
    features_per_decoder_token=128,
    features_per_memory_token=64,
    patch_size_px=7,
    imgencoder_heads=2,
    imgencoder_blocks_per_stage=(2, 3, 16, 3),
    imgencoder_global_attn_spacing_per_stage=(None, None, 4, None),
    imgencoder_window_size_per_stage=(8, 4, 14, 17),
    imgencoder_window_tile_posenc_hw=(8, 8),
    base_patch_grid_hw=(14, 14),
    num_decoder_blocks=2,
    num_decoder_heads=8,
    num_output_mask_tokens=4,
    num_memory_downsample_layers=4,
    num_memory_encoder_mixer_layers=2,
    num_memory_fusion_layers=4,
    is_version_2p1=True,
) -> SAMV2Model:
    """
    Helper used to build all SAMV2 model components. The arguments for this function are
    expected to come from the 'make_samv2_from_original_state_dict' function, which
    will fill in the function arguments based on a loaded state dictionary.

    However, if you want to make a model without pretrained weights
    here are the following standard configs (based on the original SAMV2 configs):
    https://github.com/facebookresearch/segment-anything-2/tree/main/sam2_configs

    # sam-large:
        ???

    # sam-base+
        ???

    # sam-small
        ???

    # sam-tiny
        ???
    """

    # Construct model components
    imgenc_model = SAMV2ImageEncoder(
        features_per_image_token,
        features_per_prompt_token,
        imgencoder_heads,
        imgencoder_blocks_per_stage,
        imgencoder_global_attn_spacing_per_stage,
        imgencoder_window_size_per_stage,
        imgencoder_window_tile_posenc_hw,
        base_patch_grid_hw,
        patch_size_px,
    )
    coordenc_model = SAMV2CoordinateEncoder(features_per_prompt_token)
    promptenc_model = SAMV2PromptEncoder(features_per_prompt_token)
    maskdec_model = SAMV2MaskDecoder(
        features_per_prompt_token,
        features_per_decoder_token,
        num_decoder_blocks,
        num_decoder_heads,
        num_output_mask_tokens,
    )

    memenc_model = SAMV2MemoryEncoder(
        features_per_prompt_token,
        features_per_memory_token,
        num_downsample_layers=num_memory_downsample_layers,
        num_mixer_layers=num_memory_encoder_mixer_layers,
        is_version_2p1=is_version_2p1,
    )
    memfuse_model = SAMV2MemoryFusion(
        features_per_prompt_token,
        features_per_memory_token,
        num_layers=num_memory_fusion_layers,
        is_version_2p1=is_version_2p1,
    )

    # Bundle components into complete SAM model!
    return SAMV2Model(imgenc_model, coordenc_model, promptenc_model, maskdec_model, memenc_model, memfuse_model)
