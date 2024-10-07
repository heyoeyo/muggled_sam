#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np
import torch

from lib.demo_helpers.contours import (
    get_contours_from_mask,
    get_largest_contour,
    get_contours_containing_xy,
    simplify_contour_px,
    normalize_contours,
    pixelize_contours,
)

# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MaskPostProcessor:

    # .................................................................................................................

    def __init__(self):
        self._use_largest_contour = False
        self._mask_simplify = 0
        self._mask_rounding = 0
        self._mask_padding = 0
        self._invert_mask = False

        # Storage for morphological filtering used in rounding contour edges
        self._round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        self._round_type = cv2.MORPH_OPEN

        # Storage for morphological filtering used in padding
        self._pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        self._pad_type = cv2.MORPH_DILATE

    # .................................................................................................................

    def __call__(self, mask_uint8, mask_contours_norm, point_hint_xy_norm=None):

        # Don't do any processing if there are no masks!
        if len(mask_contours_norm) == 0:
            return mask_contours_norm, mask_uint8

        # For convenience, get pixel sizing for operations that need it
        mask_shape = mask_uint8.shape

        # Keep only the largest contour if needed
        if self._use_largest_contour:
            mask_contours_norm = self.get_largest_contour(mask_contours_norm, mask_shape, point_hint_xy_norm)

        # Simplify contour shape if needed
        if self._mask_simplify > 0.001:
            mask_contours_norm = self.get_simplfied_contours(mask_contours_norm, mask_shape)

        # Build final mask
        final_mask_uint8 = self.draw_binary_mask(mask_contours_norm, mask_shape)

        # Apply rounding to get better contour edges if needed
        if self._mask_rounding != 0:
            mask_contours_norm, final_mask_uint8 = self.get_rounded_contours(mask_contours_norm, final_mask_uint8)

        # Apply mask padding if needed
        if self._mask_padding != 0:
            mask_contours_norm, final_mask_uint8 = self.get_padded_mask(mask_contours_norm, final_mask_uint8)

        if self._invert_mask:
            final_mask_uint8 = np.bitwise_not(final_mask_uint8)

        return mask_contours_norm, final_mask_uint8

    # .................................................................................................................

    def update(
        self,
        use_largest_contour: bool,
        mask_simplify: float,
        mask_rounding: int,
        mask_padding: int,
        invert_mask: bool,
    ):

        # Update rounding kernel if needed
        round_changed = self._mask_rounding != mask_rounding
        if round_changed:
            round_kernel_size = [max(1, abs(mask_rounding))] * 2
            self._round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, round_kernel_size)
            self._round_type = cv2.MORPH_CLOSE if mask_rounding > 0 else cv2.MORPH_OPEN

        # Update padding kernel if needed
        pad_changed = self._mask_padding != mask_padding
        if pad_changed:
            pad_kernel_size = [max(1, abs(mask_padding))] * 2
            self._pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, pad_kernel_size)
            self._pad_type = cv2.MORPH_DILATE if mask_padding > 0 else cv2.MORPH_ERODE

        # Store updated settings
        self._use_largest_contour = use_largest_contour
        self._mask_rounding = mask_rounding
        self._mask_padding = mask_padding
        self._mask_simplify = mask_simplify
        self._invert_mask = invert_mask

        return self

    # .................................................................................................................

    def get_largest_contour(self, mask_contours_norm, mask_shape_px, point_hint_xy_norm=None):

        # Special case, if we're given a 'point hint' when using the largest contour only, we
        # should prefer to pick from contours that include the given point!
        # -> This leads to more intuitive behavior when using interactive system
        if point_hint_xy_norm is not None:
            have_result, hinted_contours_norm = get_contours_containing_xy(mask_contours_norm, point_hint_xy_norm)
            if have_result:
                mask_contours_norm = hinted_contours_norm

        # Keep only the largest contour
        _, largest_contour_norm = get_largest_contour(mask_contours_norm, reference_shape=mask_shape_px)
        mask_contours_norm = [largest_contour_norm]

        return mask_contours_norm

    # .................................................................................................................

    def get_simplfied_contours(self, mask_contours_norm, mask_shape):

        # Perform simplification in pixel units (required by opencv) and convert back
        mask_contours_px = pixelize_contours(mask_contours_norm, mask_shape)
        mask_contours_px = [simplify_contour_px(contour, self._mask_simplify) for contour in mask_contours_px]
        mask_contours_norm = normalize_contours(mask_contours_px, mask_shape)

        return mask_contours_norm

    # .................................................................................................................

    def draw_binary_mask(self, mask_contours_norm, mask_shape):

        # Draw mask from contours
        mask_contours_px = pixelize_contours(mask_contours_norm, mask_shape)
        final_mask_uint8 = np.zeros(mask_shape, dtype=np.uint8)
        final_mask_uint8 = cv2.fillPoly(final_mask_uint8, mask_contours_px, 255, cv2.LINE_AA)

        return final_mask_uint8

    # .................................................................................................................

    def get_rounded_contours(self, mask_contours_norm, final_mask_uint8):

        final_mask_uint8 = cv2.morphologyEx(final_mask_uint8, self._round_type, self._round_kernel)
        ok_pad_contour, rounded_contour_norm = get_contours_from_mask(final_mask_uint8, normalize=True)
        if ok_pad_contour:
            mask_contours_norm = rounded_contour_norm

        return mask_contours_norm, final_mask_uint8

    # .................................................................................................................

    def get_padded_mask(self, mask_contours_norm, final_mask_uint8):

        final_mask_uint8 = cv2.morphologyEx(final_mask_uint8, self._pad_type, self._pad_kernel)
        ok_pad_contour, padded_contour_norm = get_contours_from_mask(final_mask_uint8, normalize=True)
        if ok_pad_contour:
            mask_contours_norm = padded_contour_norm

        return mask_contours_norm, final_mask_uint8

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def calculate_mask_stability_score(mask_predictions, stability_offset=2.0, mask_threshold=0.0, sum_dtype=torch.int32):
    """
    Stability score measures the ratio of masked pixels when using a
    lower than normal vs. higher than normal threshold (called easy vs. hard here).
    The closer the hard result is to the easy result, the more 'stable'
    the result considered to be.

    See:
    https://github.com/facebookresearch/segment-anything-2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/sam2/utils/amg.py#L158

    Returns:
        stability_score (shape: N, for N masks input)
    """

    # For clarity. Compute sum along H/W dimensions only
    # -> We'll get a unique sum for each 'channel' (i.e. separate masks) and batch index
    # -> For example, given preds. with shape: 1x4x32x32, we'll get sum result of shape: 1x4
    # -> Given preds. with shape: 2x64x64, we'll get sum result of shape: 2
    sum_dim = (-2, -1)

    # Compute number of masked pixels with a more difficult (higher) threshold
    hard_thresh = mask_threshold + stability_offset
    hard_count = (mask_predictions > hard_thresh).sum(sum_dim, dtype=sum_dtype)

    # Compute number of masked pixels with an easier (lower) threhsold
    easy_thresh = mask_threshold - stability_offset
    easy_count = (mask_predictions > easy_thresh).sum(sum_dim, dtype=sum_dtype)

    # Stability is a measure of how similar hard vs. easy results are
    return hard_count / torch.max(easy_count, torch.full_like(easy_count, 1))
