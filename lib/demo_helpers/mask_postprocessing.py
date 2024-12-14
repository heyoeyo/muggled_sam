#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import torch

from lib.demo_helpers.contours import MaskContourData

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class MaskPostProcessor:

    # .................................................................................................................

    def __init__(self):
        self._mask_holes_thresh = 0
        self._mask_islands_thresh = 0
        self._mask_bridging = 0
        self._mask_padding = 0

        # Storage for morphological filtering used in briding contour edges
        self._bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        self._bridge_type = cv2.MORPH_OPEN

        # Storage for morphological filtering used in padding
        self._pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        self._pad_type = cv2.MORPH_DILATE

    # .................................................................................................................

    def __call__(self, mask_uint8: ndarray, external_masks_only=False) -> tuple[ndarray, MaskContourData]:
        """
        Main post-processing function
        Applies bridging/padding etc. to the given mask image
        If external_masks_only is True, then the masking will not contain holes

        Returns:
            new_mask_uint8, mask_contour_data
        """

        # Apply bridging to open/close gaps between contours
        output_mask_uint8 = mask_uint8
        if self._mask_bridging != 0:
            output_mask_uint8 = self.get_bridged_contours(output_mask_uint8)

        # Get contours and bail early if there aren't any (i.e. no mask segmentation)
        mask_contour_data = MaskContourData(output_mask_uint8, external_masks_only)
        if len(mask_contour_data) == 0:
            return output_mask_uint8, mask_contour_data
        need_new_contours = False

        # Filter out small contours
        if self._mask_holes_thresh != 0 or self._mask_islands_thresh != 0:
            contour_filter_array = mask_contour_data.filter_by_size_thresholds(
                self._mask_holes_thresh, self._mask_islands_thresh
            )
            output_mask_uint8 = mask_contour_data.draw_mask(mask_uint8.shape, contour_filter_array)
            need_new_contours = True  # Inefficient but easier...

        # Apply mask padding if needed
        if self._mask_padding != 0:
            output_mask_uint8 = self.get_padded_mask(output_mask_uint8)
            need_new_contours = True

        # Re-generate contours if mask image was altered
        if need_new_contours:
            mask_contour_data = MaskContourData(output_mask_uint8)

        return output_mask_uint8, mask_contour_data

    # .................................................................................................................

    def update(self, mask_holes_threshold: int, mask_islands_threshold: int, mask_bridging: int, mask_padding: int):
        """
        Updates mask post-processing configuration.
        Includes some 'caching' so repeat values don't lead to repeat computation
        Returns self
        """

        # Update bridging kernel if needed
        bridge_changed = self._mask_bridging != mask_bridging
        if bridge_changed:
            bridge_kernel_size = [max(1, abs(mask_bridging))] * 2
            self._bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, bridge_kernel_size)
            self._bridge_type = cv2.MORPH_CLOSE if mask_bridging > 0 else cv2.MORPH_OPEN

        # Update padding kernel if needed
        pad_changed = self._mask_padding != mask_padding
        if pad_changed:
            pad_kernel_size = [max(1, abs(mask_padding))] * 2
            self._pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, pad_kernel_size)
            self._pad_type = cv2.MORPH_DILATE if mask_padding > 0 else cv2.MORPH_ERODE

        # Store updated settings
        self._mask_holes_thresh = mask_holes_threshold
        self._mask_islands_thresh = mask_islands_threshold
        self._mask_bridging = mask_bridging
        self._mask_padding = mask_padding

        return self

    # .................................................................................................................

    def get_bridged_contours(self, mask_uint8: ndarray) -> ndarray:
        """Helper used to apply morphological open/close operation to mask image"""
        return cv2.morphologyEx(mask_uint8, self._bridge_type, self._bridge_kernel)

    # .................................................................................................................

    def get_padded_mask(self, mask_uint8: ndarray) -> ndarray:
        """Helper used to apply morphological dilation to mask image"""
        return cv2.morphologyEx(mask_uint8, self._pad_type, self._pad_kernel)

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
