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
    compile_image_projection: bool = False,
    compile_coordinate_encoder: bool = False,
    compile_prompt_encoder: bool = False,
    compile_mask_decoder: bool = False,
    compile_multiplex_masking: bool = False,
    compile_memory_encoder: bool = False,
    compile_memory_image_fusion: bool = False,
    compile_text_encoder: bool = False,
    compile_sampling_encoder: bool = False,
    compile_image_exemplar_fusion: bool = False,
    compile_exemplar_detector: bool = False,
    compile_exemplar_segmentation: bool = False,
    custom_config: dict | None = None,
) -> None:
    """
    Helper used to enable compilation of various components of SAMv3.1
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

    # Switch away from using complex numbers (not well supported by compiler)
    if compile_image_encoder:
        model.image_encoder.toggle_use_complex_numbers(False)
        model.image_encoder.forward = torch.compile(model.image_encoder.forward, **comp_kwargs)
    if compile_image_projection:
        model.image_projection.forward = torch.compile(model.image_projection.forward, **comp_kwargs)

    # Handle compilation of V1-specific modules
    if compile_coordinate_encoder:
        model.coordinate_encoder.forward = torch.compile(model.coordinate_encoder.forward, **comp_kwargs)
    if compile_prompt_encoder:
        model.prompt_encoder.forward = torch.compile(model.prompt_encoder.forward, **dyncomp_kwargs)
    if compile_mask_decoder:
        model.mask_decoder.forward = torch.compile(model.mask_decoder.forward, **dyncomp_kwargs)

    # V2-Specific modules
    if compile_multiplex_masking:
        model.multiplex_video_masking.forward = torch.compile(model.multiplex_video_masking.forward, **dyncomp_kwargs)
    if compile_memory_encoder:
        model.memory_encoder.forward = torch.compile(model.memory_encoder.forward, **dyncomp_kwargs)
    if compile_memory_image_fusion:
        model.memory_image_fusion.fusion_transformer.forward = torch.compile(
            model.memory_image_fusion.fusion_transformer.forward, **dyncomp_kwargs
        )

    # V3-Specific modules
    if compile_text_encoder:
        model.text_encoder.transformer.forward = torch.compile(model.text_encoder.transformer.forward, **dyncomp_kwargs)
    if compile_sampling_encoder:
        model.sampling_encoder.fusion_transformer.forward = torch.compile(
            model.sampling_encoder.fusion_transformer.forward, **dyncomp_kwargs
        )
    if compile_image_exemplar_fusion:
        model.image_exemplar_fusion.forward = torch.compile(model.image_exemplar_fusion.forward, **dyncomp_kwargs)
    if compile_exemplar_detector:
        model.exemplar_detector.forward = torch.compile(model.exemplar_detector.forward, **dyncomp_kwargs)
    if compile_exemplar_segmentation:
        model.exemplar_segmentation.forward = torch.compile(model.exemplar_segmentation.forward, **dyncomp_kwargs)

    return
