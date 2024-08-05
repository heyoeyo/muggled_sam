#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_default_device_string():
    """Helper used to select a default device depending on available hardware"""

    # Figure out which device we can use
    default_device = "cpu"
    if torch.backends.mps.is_available():
        default_device = "mps"
    if torch.cuda.is_available():
        default_device = "cuda"

    return default_device


def make_device_config(device_str, use_float32, use_channels_last=True, prefer_bfloat16=True):
    """Helper used to construct a dict for device usage. Meant to be used with 'model.to(**config)'"""

    f16_type = torch.bfloat16 if prefer_bfloat16 else torch.float16
    fallback_dtype = torch.float32 if (device_str == "cpu") else f16_type
    dtype = torch.float32 if use_float32 else fallback_dtype
    memory_format = torch.channels_last if use_channels_last else None

    return {"device": device_str, "dtype": dtype, "memory_format": memory_format}


def find_best_display_arrangement(image_shape, mask_shape, target_ar=2.0, num_masks=4):
    """
    Helper function used to decide how to arrange a display made up of a
    color display image and corresponding mask predictions, stacked either
    above or to the right of the display.

    This is needed to account for displaying images of varying aspect ratio,
    where stacking multiple mask images may make the display too tall
    or too wide if not aranged properly.

    Returns:
        best_side_str, best_order_str
        -> side string is one of: "vertical", "horizontal" or "grid"
        -> order string is one of: "right" or "top"
    """

    # For convenience
    img_h, img_w = image_shape[0:2]
    mask_h, mask_w = mask_shape[0:2]

    # Set shared values for figuring out stacking
    right_side_str, top_side_str = "right", "top"
    vert_str, horz_str, grid_str = "vertical", "horizontal", "grid"
    tallmask_w = mask_w * (img_h / mask_h)
    widemask_h = mask_h * (img_w / mask_w)

    # Set up sizing configuration for each right/top + vert/horz/grid stacking arrangment
    configurations_list = [
        (right_side_str, vert_str, 0, (tallmask_w // num_masks)),
        (right_side_str, horz_str, 0, img_w + (tallmask_w * num_masks)),
        (right_side_str, grid_str, 0, tallmask_w),
        (top_side_str, vert_str, widemask_h * num_masks, 0),
        (top_side_str, horz_str, (widemask_h // num_masks), 0),
        (top_side_str, grid_str, widemask_h, 0),
    ]

    # Figure out which arrangement gives the best aspect ratio for display
    ardelta_side_order_list = []
    for side_str, order_str, add_h, add_w in configurations_list:
        ar_delta = abs(target_ar - (img_w + add_w) / (img_h + add_h))
        ardelta_side_order_list.append((ar_delta, side_str, order_str))
    _, best_side, best_order = min(ardelta_side_order_list)

    return best_side, best_order
