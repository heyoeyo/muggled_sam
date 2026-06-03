#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import json
from time import sleep

import torch

from .sam_v1_model import SAMV1Core

from .image_encoder_model import SAMV1ImageEncoder
from .coordinate_encoder_model import SAMV1CoordinateEncoder
from .prompt_encoder_model import SAMV1PromptEncoder
from .mask_decoder_model import SAMV1MaskDecoder

from .state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict
from .state_dict_conversion.convert_original_state_dict_keys import SAM1ModuleType, convert_state_dict_keys

# For type hints
from pathlib import Path
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def make_samv1_from_state_dict(
    state_dict_or_path: dict[str, Tensor] | str | Path,
    strict_load: bool = True,
    weights_only: bool = True,
) -> SAMV1Core:
    """
    Helper used to load SAMv1 using either an 'original state dict'
    (e.g. original SAM model weights) or 'muggled state dict'
    (e.g. weights saved from a MuggledSAM instance).
    Returns:
        sam_v1_core
    """

    # Load state dict data & figure out how to handle model instantiation
    loaded_state_dict = _load_state_dict(state_dict_or_path, strict_load, weights_only)
    is_mugsam_sd = "config_muggled_samv1" in loaded_state_dict.keys()
    if is_mugsam_sd:
        sam_core = make_samv1_from_muggled_state_dict(loaded_state_dict, strict_load, weights_only)
    else:
        sam_core = make_samv1_from_original_state_dict(loaded_state_dict, strict_load, weights_only)

    return sam_core


# .....................................................................................................................


def make_samv1_from_original_state_dict(
    original_state_dict: dict | str | Path, strict_load: bool = True, weights_only: bool = True
) -> SAMV1Core:
    """
    Function used to initialize a SAM (v1) model from a state dictionary (i.e. model weights) file.
    This function will automatically figure out the model sizing parameters from the state dict,
    assuming it comes from the original SAM repo (or matches the implementation).

    The state dict can be provided directly (e.g. from state_dict = torch.load()) or
    a string can be given, in which case it will be assumed to be a path to load the state dict

    Returns:
        sam_v1_core
    """

    # Load state dict data
    original_state_dict = _load_state_dict(original_state_dict, strict_load, weights_only)

    # Get model config from weights (i.e. beit_large_512 vs beit_base_384) & convert to new keys/state dict
    model_config_dict = get_model_config_from_state_dict(original_state_dict)
    new_state_dict, _ = convert_state_dict_keys(model_config_dict, original_state_dict)

    # Load model & set model weights
    sam_core = make_sam_v1(**model_config_dict)
    sam_core.image_encoder.load_state_dict(new_state_dict[SAM1ModuleType.image_encoder], strict_load)
    sam_core.coordinate_encoder.load_state_dict(new_state_dict[SAM1ModuleType.coordinate_encoder], strict_load)
    sam_core.prompt_encoder.load_state_dict(new_state_dict[SAM1ModuleType.prompt_encoder], strict_load)
    sam_core.mask_decoder.load_state_dict(new_state_dict[SAM1ModuleType.mask_decoder], strict_load)

    return sam_core


# .....................................................................................................................


def make_samv1_from_muggled_state_dict(
    muggled_state_dict: dict | str | Path,
    strict_load: bool = True,
    weights_only: bool = True,
) -> SAMV1Core:
    """
    Similar to the '...from_original_state_dict' function, this function instantiates a
    SAMV1 model from a state dictionary file (e.g. model weights) and automatically
    handles setting up the model configuration/sizing parameters.

    This variant of the function is meant for loading from weights that were directly
    saved from a muggled-SAMV1 instance, rather than the original model weights.

    The state dict can be provided directly (e.g. from state_dict = torch.load(...)) or
    a string can be given, in which case it will be assumed to be a path to load the state dict

    Returns:
        sam_v1_core
    """

    # Load state dict data
    muggled_state_dict = _load_state_dict(muggled_state_dict, strict_load, weights_only)

    # Try to get config from state dict
    config_key = "config_muggled_samv1"
    config_as_tensor = muggled_state_dict.get(config_key, None)
    if config_as_tensor is None:
        raise KeyError(f"Cannot load model! State dict is missing configuration ({config_key})")

    # Convert config from tensor->bytes->string/json->dictionary
    config_as_bytes = bytearray(config_as_tensor.cpu().tolist())
    config_as_str = config_as_bytes.decode()
    config_dict = json.loads(config_as_str)

    # Load model & set model weights
    sam_core = make_sam_v1(**config_dict)
    sam_core.load_state_dict(muggled_state_dict, strict_load)

    return sam_core


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
) -> SAMV1Core:
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

    # Convert config to byte-data so it can be stored with the model
    config_dict = locals()
    config_as_str = json.dumps(config_dict, separators=(",", ":"), indent=None)
    config_bytes = bytearray(config_as_str.encode())

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
    return SAMV1Core(imgenc_model, coordenc_model, promptenc_model, maskdec_model, config_bytes)


# .....................................................................................................................


def _load_state_dict(
    state_dict_or_path: dict[str, Tensor] | str | Path,
    strict_load: bool = True,
    weights_only: bool = True,
) -> dict[str, Tensor]:
    """
    Helper used to handling load of model weights, which may
    be given directly (as a dictionary) or as a file path
    Returns: loaded_state_dict
    """

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
        sleep(5)

    # If we're given a string, assume it's a path to the state dict
    out_state_dict = state_dict_or_path
    need_to_load = isinstance(state_dict_or_path, (str, Path))
    if need_to_load:
        # Load model weights with fail check in case weights are in cuda format and user doesn't have cuda
        try:
            out_state_dict = torch.load(state_dict_or_path, weights_only=weights_only)
        except RuntimeError:
            out_state_dict = torch.load(state_dict_or_path, map_location="cpu", weights_only=weights_only)

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

    return out_state_dict
