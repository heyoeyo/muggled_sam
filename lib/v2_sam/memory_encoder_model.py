#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports


import torch
import torch.nn as nn

from .components.memenc_components import MaskDownSampler, CXBlock

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2MemoryEncoder(nn.Module):
    """
    WIP - Intending to simplify/document
    Taken from:
    https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/memory_encoder.py
    """

    # .................................................................................................................

    def __init__(self, in_dim=256, out_dim=64, num_downsample_layers=4, num_fuse_layers=2):

        # Inherit from parent
        super().__init__()

        self.mask_downsampler = MaskDownSampler(in_dim, num_downsample_layers)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = nn.Sequential(*(CXBlock(in_dim) for _ in range(num_fuse_layers)))
        self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        # Store upsampling config for restoring proper mask shape during forward pass
        self._upsample_factor = int(2**num_downsample_layers)

    # .................................................................................................................

    def forward(self, lowres_image_encoding: Tensor, lowres_mask: Tensor, is_prompt_encoding=False) -> Tensor:

        # Scale mask up, so that downsample result matches the lowres image encoding
        img_hw = lowres_image_encoding.shape[2:]
        hires_hw = [size * self._upsample_factor for size in img_hw]
        hires_mask = nn.functional.interpolate(lowres_mask, size=hires_hw, mode="bilinear", align_corners=False)

        # Prepare mask
        mask_for_mem = (hires_mask > 0.0).to(hires_mask.dtype) if is_prompt_encoding else torch.sigmoid(hires_mask)
        mask_for_mem = mask_for_mem * 20.0 - 10.0
        mask_for_mem = self.mask_downsampler(mask_for_mem)

        ## Fuse pix_feats and downsampled masks
        x = self.pix_feat_proj(lowres_image_encoding)
        x = x + mask_for_mem
        x = self.fuser(x)
        x = self.out_proj(x)

        return x
