#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports


import torch
import torch.nn as nn

from .components.memenc_components import MaskDownSampler, CXBlock
from .components.posenc_sine import PositionEmbeddingSine

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SAMV2MemoryEncoder(nn.Module):

    # .................................................................................................................

    def __init__(self, in_dim=256, out_dim=64, num_downsample_layers=4, num_fuse_layers=2):

        # Inherit from parent
        super().__init__()

        self.mask_downsampler = MaskDownSampler(in_dim, num_downsample_layers)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = nn.Sequential(*(CXBlock(in_dim) for _ in range(num_fuse_layers)))

        self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=out_dim)

    # .................................................................................................................

    def forward(self, pix_feat: Tensor, masks: Tensor, skip_mask_sigmoid: bool = False) -> tuple[Tensor, Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = nn.functional.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        ## Fuse pix_feats and downsampled masks
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}
