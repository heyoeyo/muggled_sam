#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def enable_compilation(
    model: torch.nn.Module,
    compile_image_encoder: bool = False,
    compile_coordinate_encoder: bool = False,
    compile_prompt_encoder: bool = False,
    compile_mask_decoder: bool = False,
    custom_config: dict | None = None,
) -> None:
    """
    Helper used to enable compilation of various components of SAMv1
    Note that compilation doesn't occur until the model actually runs.

    Also note that the model input parameters can affect compilation
    and changes to the input may trigger re-compilation!

    This is an experimental feature and may not be well supported on
    all hardware and/or pytorch versions.
    """

    # Special optimization to speed up float32 usage
    if next(model.parameters()).dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    # Fill in default compilation settings
    comp_kwargs = custom_config
    dyncomp_kwargs = custom_config
    if custom_config is None:
        compile_options = {
            "shape_padding": True,
            "epilogue_fusion": True,
            "max-autotune": True,
        }
        comp_kwargs = {"mode": None, "fullgraph": True, "options": compile_options}
        dyncomp_kwargs = {**comp_kwargs, "dynamic": True}

    # Handle compilation of individual modules
    if compile_image_encoder:
        model.image_encoder.forward = torch.compile(model.image_encoder.forward, **comp_kwargs)
    if compile_coordinate_encoder:
        model.coordinate_encoder.forward = torch.compile(model.coordinate_encoder.forward, **comp_kwargs)
    if compile_prompt_encoder:
        model.prompt_encoder.forward = torch.compile(model.prompt_encoder.forward, **dyncomp_kwargs)
    if compile_mask_decoder:
        model.mask_decoder.forward = torch.compile(model.mask_decoder.forward, **dyncomp_kwargs)

    return
