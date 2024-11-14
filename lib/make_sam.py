#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch

# Note: There are additional dynamic imports inside the function: `import_model_functions`
#       This is done to avoid importing code associated with a model that isn't being loaded


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def make_sam_from_state_dict(path_to_state_dict: str, strict_load=True, weights_only=True, model_type=None):

    # Load model weights with fail check in case weights are in cuda format and user doesn't have cuda
    try:
        state_dict = torch.load(path_to_state_dict, weights_only=weights_only)
    except RuntimeError:
        state_dict = torch.load(path_to_state_dict, map_location="cpu", weights_only=weights_only)

    # Try to figure out which type of model we're creating from state dict keys (e.g. samv1 vs v2)
    if model_type is None:
        model_type = determine_model_type_from_state_dict(path_to_state_dict, state_dict)

    # Error out if we don't understand the model type
    known_model_types = ["sam_v2", "sam_v1"]
    if model_type not in known_model_types:
        print("Accepted model types:", *known_model_types, sep="\n")
        raise NotImplementedError(f"Bad model type: {model_type}, no support for this yet!")

    # Build the model & supporting data
    make_sam_func = import_model_functions(model_type)
    config_dict, sam_model = make_sam_func(state_dict, strict_load, weights_only)

    return config_dict, sam_model


# .....................................................................................................................


def determine_model_type_from_state_dict(model_path, state_dict):
    """
    Helper used to figure out which model type (e.g. v1 vs. v2) we're working with,
    given a state dict (e.g. model weights). This works by looking for (hard-coded) keys
    that are expected to be unique among different model's state dicts
    """

    sd_keys = state_dict.keys()

    samv2_target_key = "model"
    if samv2_target_key in sd_keys:
        return "sam_v2"

    samv1_target_key = "image_encoder.pos_embed"
    if samv1_target_key in sd_keys:
        return "sam_v1"

    return "unknown"


# .....................................................................................................................


def import_model_functions(model_type):
    """
    Function used to import the 'make sam' functions for
    all known model types. This is a hacky-ish thing to do, but helps avoid
    importing all model code even though we're only loading one model
    """

    if model_type == "sam_v2":
        from .v2_sam.make_sam_v2 import make_samv2_from_original_state_dict as make_sam_func

    elif model_type == "sam_v1":
        from .v1_sam.make_sam_v1 import make_samv1_from_original_state_dict as make_sam_func

    else:
        raise TypeError(f"Cannot import model functions, Unknown model type: {model_type}")

    return make_sam_func
