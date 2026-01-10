#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os

import torch
import torch.nn as nn

from .shared import imagelike_to_rows_of_tokens
from .position_encoding import SinusoidalPE2D

# Optional torchvision import (we'll fallback to custom implementation otherwise)
torchvision_roi_align = None
USE_TORCHVISION = os.environ.get("USE_TORCHVISION", False)
if USE_TORCHVISION:
    try:
        from torchvision.ops import roi_align as torchvision_roi_align

    except ImportError:
        USE_TORCHVISION = False

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PointSampleEncoder(nn.Module):
    """
    Module used to handle point encodings for the sampling encoder.
    Point coordinates are used to sample from image tokens,
    as well as (optionally) encoding xy coordinates into
    a set of 'sampling' tokens.

    This functionality is handled by a function in the original implementation,
    but has been separated here for readability. See:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/geometry_encoders.py#L600
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256):

        # Inherit from parent
        super().__init__()

        # Create position encoder (no learned components)
        self.posenc = SinusoidalPE2D(features_per_token)

        # Model components
        self.xy_proj = nn.Linear(2, features_per_token)
        self.posenc_proj = nn.Linear(features_per_token, features_per_token)
        self.img_sample_proj = nn.Linear(features_per_token, features_per_token)

    # .................................................................................................................

    def forward(
        self, image_tokens_bchw: Tensor, points_bn2: Tensor, include_coordinate_encodings: bool = True
    ) -> Tensor:

        # Bail on missing inputs
        pt_b, num_points, _ = points_bn2.shape
        if num_points == 0:
            return points_bn2

        # Sample image tokens at provided points
        img_sample_encoding_bnc = self._sample_image_tokens(image_tokens_bchw, points_bn2)
        if not include_coordinate_encodings:
            return img_sample_encoding_bnc

        # -> Flatten B & N together to get (BN)x2 and unbind into x/y
        # -> Encode x & y (each has shape: (BN)x(C/2), where C is channels/features)
        # -> Combine into single (BN)xC tokens, then reshape as: BxNxC, then apply projection
        pt_x, pt_y = points_bn2.flatten(0, 1).unbind(-1)
        x_enc, y_enc = self.posenc.encode_xy(pt_x, pt_y)
        posenc_tokens = torch.cat((x_enc, y_enc), dim=-1)
        posenc_tokens = posenc_tokens.view(pt_b, num_points, -1)
        pos_encoding_bnc = self.posenc_proj(posenc_tokens)

        # Return additive encoding
        base_embedding = self.xy_proj(points_bn2)
        return base_embedding + pos_encoding_bnc + img_sample_encoding_bnc

    # .................................................................................................................

    def _sample_image_tokens(self, image_tokens_bchw: Tensor, points_bn2: Tensor) -> Tensor:
        """Returns: image_samples_bnc (N matching the number of input points)"""

        # For convenience
        img_b, img_c, img_h, img_w = image_tokens_bchw.shape

        # Set up 'grid' of sampling points (do this so we can use grid_sample function)
        # -> The grid needs to be shaped: BxHxWx2
        # -> We're not working in 2D, we have only 'N' xy points, so we form: BxNx1x2
        # -> Also need convert to -1 to +1 normalization (req. by grid sample), from 0-to-1
        grid_bn12_renorm = (points_bn2.unsqueeze(2) - 0.5) * 2.0
        img_samples_bcn1 = nn.functional.grid_sample(image_tokens_bchw, grid_bn12_renorm, align_corners=False)

        # Convert 'image-like' samples (really Nx1, not HxW) back to rows-of-tokens format for output
        img_samples_bnc, _ = imagelike_to_rows_of_tokens(img_samples_bcn1)
        return self.img_sample_proj(img_samples_bnc)

    # .................................................................................................................


class BoxSampleEncoder(nn.Module):
    """
    Module used to handle box encodings for the sampling encoder.
    Box coordinates are used to sample a grid of points from image tokens
    which are then combined into a single 'sampling token' (one per box).
    The box coordinates themselves can (optionally) be included in the
    output token.

    This functionality is handled by a function in the original implementation,
    but has been separated here for readability. See:
    https://github.com/facebookresearch/sam3/blob/757bbb0206a0b68bee81b17d7eb4877177025b2f/sam3/model/geometry_encoders.py#L643
    """

    # .................................................................................................................

    def __init__(self, features_per_token: int = 256, roi_samples: int = 7):

        # Inherit from parent
        super().__init__()

        # Create position encoder (no learned components)
        self.posenc = SinusoidalPE2D(features_per_token)

        # Model components
        self.cxcywh_proj = nn.Linear(4, features_per_token)
        self.posenc_proj = nn.Linear(features_per_token + 2, features_per_token)  # +2 for encoding box W & H
        self.img_sample_conv = nn.Conv2d(features_per_token, features_per_token, roi_samples)

        # Store sizing info for reshaping operations
        self._roi_samples = roi_samples
        self._features_per_token = features_per_token

    # .................................................................................................................

    def forward(
        self, image_tokens_bchw: Tensor, boxes_bn22: Tensor, include_coordinate_encodings: bool = True
    ) -> Tensor:
        """
        Expects top-left/bottom-right boxes with normalized (0-to-1) coordinates:
            boxes_bn22 = [(x1, y1), (x2, y2)] = [(0.1, 0.2), (0.5, 0.4)]

        Returns encoded tokens, shape: BxNxC (N matches number of boxes, C matches image channels by default)

        See original code:
        https://github.com/facebookresearch/sam3/blob/b26a5f330e05d321afb39d01d3d4881f258f65ff/sam3/model/geometry_encoders.py#L643
        """

        # Bail on missing input
        box_b, num_boxes = boxes_bn22.shape[0:2]
        if num_boxes == 0:
            return boxes_bn22

        # Sample image tokens based on provided bounding boxes
        img_sample_encoding_bnc = self._sample_image_tokens(image_tokens_bchw, boxes_bn22)
        if not include_coordinate_encodings:
            return img_sample_encoding_bnc

        # Get x-/y-center, width & height format
        box_xy_cen = boxes_bn22.mean(dim=2, keepdim=False)
        box_wh = boxes_bn22.diff(dim=2).squeeze(2)
        boxes_cxcywh_bn4 = torch.cat((box_xy_cen, box_wh), dim=-1)

        # Create position encoding tokens out of box center coordinates and width/height (strange implementation!)
        # -> We flatten B & N together and split the 4 components (cx,cy,w,h) to run through encoder
        # -> Encoded cx/cy values are stacked with box width & height (which are not encoded!!) in order: y,x,w,h
        # -> Finally, we need to 'unflatten' the B & N dimensions back to original shape
        # See original code: https://github.com/facebookresearch/sam3/blob/b26a5f330e05d321afb39d01d3d4881f258f65ff/sam3/model/geometry_encoders.py#L679-L683
        box_cx, box_cy, box_w, box_h = boxes_cxcywh_bn4.flatten(0, 1).unbind(-1)
        x_enc, y_enc = self.posenc.encode_xy(box_cx, box_cy)
        posenc_tokens = torch.cat((y_enc, x_enc, box_h.unsqueeze(-1), box_w.unsqueeze(-1)), dim=-1)  # Weird!
        posenc_tokens = posenc_tokens.view(box_b, num_boxes, -1)
        pos_encoding_bnc = self.posenc_proj(posenc_tokens)

        # Return additive encoding
        base_encoding_bnc = self.cxcywh_proj(boxes_cxcywh_bn4)
        return base_encoding_bnc + pos_encoding_bnc + img_sample_encoding_bnc

    # .................................................................................................................

    def _sample_image_tokens(self, image_tokens_bchw: Tensor, boxes_bn22: Tensor) -> Tensor:
        """Returns: image_samples_bnc (N matching the number of input boxes)"""

        # For convenience
        img_b, img_c, img_h, img_w = image_tokens_bchw.shape
        box_b, num_boxes = boxes_bn22.shape[0:2]
        device, dtype = image_tokens_bchw.device, image_tokens_bchw.dtype

        if USE_TORCHVISION:

            # Convert to pixelized xyxy format needed by torchvision roi align
            norm_to_px_scale = torch.tensor((img_w, img_h), device=device, dtype=dtype)
            boxes_bn4_px = (boxes_bn22 * norm_to_px_scale).view(num_boxes, box_b, 4)

            # Switch to float32 if needed (torchvision roi doesn't work on bfloat16)
            in_dtype = image_tokens_bchw.dtype
            need_typecast = in_dtype == torch.bfloat16
            if in_dtype == torch.bfloat16:
                image_tokens_bchw = image_tokens_bchw.float()
                boxes_bn4_px = boxes_bn4_px.float()

            # Get samples from image based on boxes
            boxes_batch_list = boxes_bn4_px.unbind(0)
            img_samples_tcrr = torchvision_roi_align(image_tokens_bchw, boxes_batch_list, self._roi_samples)
            if need_typecast:
                img_samples_tcrr = img_samples_tcrr.to(dtype=in_dtype)

            # Result from roi_align has shape: TxCxRxR, where T is (B*N), C is channels, R is roi size
            # -> We first convert this to TxCx1x1 using a convolution (like a patch embedding)
            #    then split T into BxN and merge 1x1 dimensions
            patch_samples_tc11 = self.img_sample_conv(img_samples_tcrr)
            img_samples_bnc = patch_samples_tc11.view(box_b, num_boxes, self._features_per_token)

        else:
            # Run roi align using custom implementation (gives results with ~1E-5 differences numerically)
            boxes_batch_list = boxes_bn22.unbind(0)
            img_samples_bnc = torch.empty((box_b, num_boxes, self._features_per_token)).to(boxes_bn22)
            for batch_idx, boxes_xy1xy2_n22 in enumerate(boxes_batch_list):
                img_samples_ncrr = custom_roi_align(image_tokens_bchw, boxes_xy1xy2_n22, self._roi_samples)
                img_samples_bnc[batch_idx] = self.img_sample_conv(img_samples_ncrr).flatten(1)

        return img_samples_bnc

    # .................................................................................................................


def custom_roi_align(
    imagelike_bchw: torch.Tensor, box_x1y1x2y2_norm_list: list[torch.Tensor], roi_hw: int | tuple[int, int]
) -> torch.Tensor:
    """
    Custom implementation of 'roi_align' from torchvision.ops
    Doesn't support batched box prompts, call in a loop to handle batches if needed.

    Input boxes are expected to be in [x1,y1,x2,y2] format, with 0-to-1 normalized coordinates
    """

    # For convenience
    img_b, img_c, img_h, img_w = imagelike_bchw.shape
    roi_h, roi_w = (roi_hw, roi_hw) if isinstance(roi_hw, int) else roi_hw
    device, dtype = imagelike_bchw.device, imagelike_bchw.dtype

    # Set up odd normalization factor used by original implementation
    # -> This is a result of 'inclusive' indexing of roi_align
    # -> We need this to properly interpret box width/height as well as interpolation coords
    norm_scale = torch.tensor(((img_w / (img_w - 1), (img_h / (img_h - 1)))), device=device, dtype=dtype)

    # Iterate over a single batch of N boxes
    num_boxes = len(box_x1y1x2y2_norm_list)
    output = torch.empty((num_boxes, img_c, roi_h, roi_w), device=device, dtype=dtype)
    for box_idx, box_x1y1x2y2 in enumerate(box_x1y1x2y2_norm_list):

        # Handle (xy1, xy2) formatting (e.g. convert [(x1, y1), (x2, y2)] -to-> [x1,y1,x2,y2])
        if box_x1y1x2y2.shape[-1] == 2:
            box_x1y1x2y2 = box_x1y1x2y2.flatten(-2)

        # Remove extra dimensions (at this point, we should only have a tensor of 4 values)
        if box_x1y1x2y2.ndim > 1:
            box_x1y1x2y2 = box_x1y1x2y2.squeeze()
            assert box_x1y1x2y2.ndim == 1, "Boxes are expected to have 4 values: [x1,y1,x2,y2]"

        # Assume boxes are batched in format: (x1,y1), (x2,y2)
        (x1_norm, y1_norm), (x2_norm, y2_norm) = box_x1y1x2y2.view(2, 2) * norm_scale

        # Figure out how many samples to take per output 'bin'
        box_width = torch.clip((x2_norm - x1_norm) * (img_w - 1), min=1)
        box_height = torch.clip((y2_norm - y1_norm) * (img_h - 1), min=1)
        num_sample_points_x = torch.ceil(box_width / roi_w).to(dtype=torch.int32)
        num_sample_points_y = torch.ceil(box_height / roi_h).to(dtype=torch.int32)

        # Compute all sample points in x (note: we need -1 to +1 normalization)
        num_x_interp = (roi_w * num_sample_points_x).to(dtype=torch.int32)
        x_halfstep = (x2_norm - x1_norm) / (2 * num_x_interp).to(dtype=dtype)
        ix1, ix2 = iy1, iy2 = [(2.0 * x) - 1.0 for x in [(x1_norm + x_halfstep), (x2_norm - x_halfstep)]]
        x_interp = torch.linspace(ix1, ix2, num_x_interp, device=device, dtype=dtype)

        # Compute all sample points in y (note: we need -1 to +1 normalization)
        num_y_interp = (roi_h * num_sample_points_y).to(dtype=torch.int32)
        y_halfstep = (y2_norm - y1_norm) / (2 * num_y_interp).to(dtype=dtype)
        iy1, iy2 = [(2.0 * y) - 1.0 for y in [(y1_norm + y_halfstep), (y2_norm - y_halfstep)]]
        y_interp = torch.linspace(iy1, iy2, num_y_interp, device=device, dtype=dtype)

        # Build 2D grid of sample points
        xy_pts = torch.stack(torch.meshgrid(x_interp, y_interp, indexing="xy"), dim=-1).unsqueeze(0)
        image_samples = torch.nn.functional.grid_sample(
            imagelike_bchw, xy_pts, align_corners=True, padding_mode="border"
        )

        # Average samples to form final bins
        output[box_idx] = torch.nn.functional.avg_pool2d(image_samples, (num_sample_points_y, num_sample_points_x))

    return output
