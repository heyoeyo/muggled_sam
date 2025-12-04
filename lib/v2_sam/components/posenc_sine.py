#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SinusoidalPE2D(nn.Module):
    """
    Implements position encodings similar to 'sine/cosine' approach originally
    used in paper: "Attention Is All You Need" (see page 5), but adapted
    to work with 'image-like' tokens.

    See original code:
    https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/position_encoding.py#L16
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int, temperature: float = 10000.0):

        # Inherit from parent
        super().__init__()

        # Sanity check. Need even number so we can make (x,y) pairings
        assert features_per_token % 2 == 0, "Need an even number of features for sinusoidal position encoding!"

        # Pre-compute periods with 'geometric scaling' (e.g. temperature ^ scaling)
        # -> These are associated with frequencies: 2πf where 'f' is 1/p, p being the calculate frequency,
        #    the values are calculated this way to maintain numerical consistency with original implementation
        # -> There are C/2 'scale_factors' and they look like repeated-normalized values,
        #    for example, with features_per_token = 16, result is: [0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75]
        half_features = features_per_token // 2
        scale_factors = (torch.arange(half_features, dtype=torch.float32) // 2) * (2 / half_features)
        per_channel_period = temperature**scale_factors
        self.register_buffer("per_channel_period", per_channel_period, persistent=False)
        self.register_buffer("twopi", torch.tensor(2.0 * torch.pi), persistent=False)

        # Also allocate storage for re-using encodings
        self.register_buffer("cached_posenc_bchw", torch.empty((1, 1, 1, 1)), persistent=False)

    # .................................................................................................................

    def forward(self, height: int, width: int) -> torch.Tensor:
        """Computes (non-learned) 2D sinusoidal position encodings, with shape: BxCxHxW"""

        # Re-generate cached result if needed
        cache_h, cache_w = self.cached_posenc_bchw.shape[-2:]
        if cache_h != height or cache_w != width in self.cache:

            # For convenience
            device, dtype = self.cached_posenc_bchw.device, self.cached_posenc_bchw.dtype

            # Make index sequence (e.g. sequences like: 1, 2, 3, 4, ..., height or width) that is then normalized
            # -> For a height of 5, result would be: [0.2, 0.4, 0.6, 0.8, 1.0]
            eps = 1e-6
            y_idx_norm = torch.arange(1, 1 + height, dtype=torch.float32, device=device).unsqueeze(-1) / (height + eps)
            x_idx_norm = torch.arange(1, 1 + width, dtype=torch.float32, device=device).unsqueeze(-1) / (width + eps)

            # Compute 'xy frequency' terms, these are of the form: α*2πf
            # where α is a normalized x-/y-index and f is pre-compute frequency value
            # -> Note the shapes here are: Hx(C/2) and Wx(C/2), where C is expected features-per-token
            y_frequencies = y_idx_norm * self.twopi / self.per_channel_period
            x_frequencies = x_idx_norm * self.twopi / self.per_channel_period

            # Calculate the sine of even index angles, cosine of odd index angles & stack together
            # -> Gives intermediate shapes: Hx(C/4)x2 & Wx(C/4)x2
            # -> Then flatten last 2 dimensions back to: Hx(C/2), Wx(C/2)
            y_sincos = torch.stack((y_frequencies[:, 0::2].sin(), y_frequencies[:, 1::2].cos()), dim=-1).flatten(-2)
            x_sincos = torch.stack((x_frequencies[:, 0::2].sin(), x_frequencies[:, 1::2].cos()), dim=-1).flatten(-2)

            # Repeat x/y components along h/w dimensions and stack to form a single BxCxHxW tensor
            x_sincos_hwc = x_sincos.unsqueeze(0).repeat(height, 1, 1)
            y_sincos_hwc = y_sincos.unsqueeze(1).repeat(1, width, 1)
            xy_stacked_bchw = torch.cat((y_sincos_hwc, x_sincos_hwc), dim=-1).permute(2, 0, 1).unsqueeze(0)
            self.cached_posenc_bchw = xy_stacked_bchw.to(dtype)

        return self.cached_posenc_bchw
