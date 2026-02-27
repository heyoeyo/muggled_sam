#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch

# For type hints
from typing import Callable
from torch import Tensor

# Note: There are additional dynamic imports inside the function: `import_model_functions`
#       This is done to avoid importing code associated with a model that isn't being loaded


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def make_sam_from_state_dict(
    path_to_state_dict: str | dict[str, Tensor],
    strict_load: bool = True,
    weights_only: bool = True,
) -> tuple[dict, torch.nn.Module]:
    """
    Function used to load a SAM v1, v2 or v3 model from a state dict (e.g. model weights).
    The model version and sizing is automatically inferred from the weights.

    Either a path to the state dict file (e.g. '/path/to/sam_model.pt')
    or otherwise the state dict can be provided directly

    Returns:
        model_config_dict, sam_model
    """

    # Load weights if needed (otherwise assume we were given weights directly)
    need_to_load = isinstance(path_to_state_dict, str)
    if need_to_load:
        # Load model weights with fail check in case weights are in cuda format and user doesn't have cuda
        try:
            state_dict = torch.load(path_to_state_dict, weights_only=weights_only)
        except RuntimeError:
            state_dict = torch.load(path_to_state_dict, map_location="cpu", weights_only=weights_only)
    else:
        state_dict = path_to_state_dict

    # Try to figure out which model loading function we should use (for different versions of SAM)
    make_sam_func = import_model_functions(state_dict)
    if make_sam_func is None:
        raise NotImplementedError("Cannot load model! Unrecognized weights...")

    # Build the model & supporting data
    config_dict, sam_model = make_sam_func(state_dict, strict_load, weights_only)
    return config_dict, sam_model


# .....................................................................................................................


def import_model_functions(state_dict: dict[str, Tensor]) -> Callable | None:
    """
    Helper used to figure out which model type (e.g. v1/v2/v3) we're working with,
    given a state dict (e.g. model weights). This works by looking for (hard-coded) keys
    that are expected to be unique among different model state dicts

    Returns:
        make_sam_function
        -> Returns None if the model type (e.g. SAM v1/v2/v3) cannot be determined
    """

    # Strip outer model key if needed
    sd_keys = state_dict.keys()
    if "model" in sd_keys:
        sd_keys = state_dict["model"].keys()

    # Search for original SAM model weights
    samv3_target_key = "detector.backbone.vision_backbone.trunk.pos_embed"
    samv2_target_key = "image_encoder.trunk.pos_embed_window"
    samv1_target_key = "image_encoder.pos_embed"
    make_sam_func = None
    if samv3_target_key in sd_keys:
        from .v3_sam.make_sam_v3 import make_samv3_from_original_state_dict as make_sam_func
    elif samv2_target_key in sd_keys:
        from .v2_sam.make_sam_v2 import make_samv2_from_original_state_dict as make_sam_func
    elif samv1_target_key in sd_keys:
        from .v1_sam.make_sam_v1 import make_samv1_from_original_state_dict as make_sam_func

    # If we haven't found a match, check if we're loading from muggled-sam weights directly
    if make_sam_func is None:
        mugsamv3_target_key = "config_muggled_samv3"
        mugsamv2_target_key = "config_muggled_samv2"
        mugsamv1_target_key = "config_muggled_samv1"
        if mugsamv3_target_key in sd_keys:
            from .v3_sam.make_sam_v3 import make_samv3_from_muggled_state_dict as make_sam_func
        elif mugsamv2_target_key in sd_keys:
            from .v2_sam.make_sam_v2 import make_samv2_from_muggled_state_dict as make_sam_func
        elif mugsamv1_target_key in sd_keys:
            from .v1_sam.make_sam_v1 import make_samv1_from_muggled_state_dict as make_sam_func

    return make_sam_func
