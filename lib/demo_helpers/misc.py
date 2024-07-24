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
