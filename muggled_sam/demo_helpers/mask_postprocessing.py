#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import cv2
import numpy as np

from .contours import MaskContourData

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
        self._mask_simplify_eps = 0
        self._mask_simplify_by_perimeter = False

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

        # Re-draw mask according to contour data when using external only,
        # since it is generally very different than the original mask!
        mask_contour_data = MaskContourData(mask_uint8, external_masks_only)
        output_mask_uint8 = mask_uint8
        if external_masks_only:
            output_mask_uint8 = mask_contour_data.draw_mask(mask_uint8.shape)

        # Get contours and bail early if there aren't any (i.e. no mask segmentation)
        if len(mask_contour_data) == 0:
            return output_mask_uint8, mask_contour_data

        # Apply bridging to open/close gaps between contours
        need_bridging = self._mask_bridging != 0
        if need_bridging:
            output_mask_uint8 = self.get_bridged_contours(output_mask_uint8)
            mask_contour_data = MaskContourData(output_mask_uint8, external_masks_only)

        # Filter out small contours
        need_size_filtering = self._mask_holes_thresh != 0 or self._mask_islands_thresh != 0
        if need_size_filtering:
            contour_filter_array = mask_contour_data.filter_by_size_thresholds(
                self._mask_holes_thresh, self._mask_islands_thresh
            )
            output_mask_uint8 = mask_contour_data.draw_mask(mask_uint8.shape, contour_filter_array)

        # Apply mask padding if needed
        need_padding = self._mask_padding != 0
        if need_padding:
            output_mask_uint8 = self.get_padded_mask(output_mask_uint8)

        # Re-generate contours if mask image was altered
        if need_size_filtering or need_padding:
            mask_contour_data = MaskContourData(output_mask_uint8, external_masks_only)

        # Reduce contour complexity if needed
        need_simplify = self._mask_simplify_eps > 0
        if need_simplify:
            mask_contour_data.simplify_inplace(self._mask_simplify_eps, self._mask_simplify_by_perimeter)
            output_mask_uint8 = mask_contour_data.draw_mask(mask_uint8.shape)

        return output_mask_uint8, mask_contour_data

    # .................................................................................................................

    def update(
        self,
        mask_holes_threshold: int,
        mask_islands_threshold: int,
        mask_bridging: int,
        mask_padding: int,
        mask_simplify_eps: float,
        mask_simplify_by_perimeter: bool,
    ):
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
        self._mask_simplify_eps = mask_simplify_eps
        self._mask_simplify_by_perimeter = mask_simplify_by_perimeter

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


def get_box_xy1xy2_norm_from_mask(
    mask: ndarray,
    normalize: bool = True,
) -> tuple[tuple[int | float], tuple[int, float]]:
    """
    Function used to get the bounding box coordinates from a binary mask.
    Expects a numpy array as input (either with 0/non-zero values or False/True)
    Returns:
        ((x1, y1), (x2, y2))

    - xy1/xy2 will be in pixel units if normalize=False
      otherwise, values will be in 0-to-1 range
    """

    # First find row/columns that have at least 1 masked pixel (True) or not (False)
    rows_containing_masked_pixels = np.any(mask, axis=1)
    columns_containing_masked_pixels = np.any(mask, axis=0)

    # Find first/last row/column indexes that contain masked pixels to get bounds
    y1_px, y2_px = np.where(rows_containing_masked_pixels)[0][[0, -1]]
    x1_px, x2_px = np.where(columns_containing_masked_pixels)[0][[0, -1]]

    # Build final ((x1, y1), (x2, y2)) output
    if normalize:
        mask_h, mask_w = mask.shape[0:2]
        x1_norm, x2_norm = [float(x / (mask_w - 1)) for x in (x1_px, x2_px)]
        y1_norm, y2_norm = [float(y / (mask_h - 1)) for y in (y1_px, y2_px)]
        return ((x1_norm, y1_norm), (x2_norm, y2_norm))
    return ((x1_px, y1_px), (x2_px, y2_px))


def get_box_nms_indexing(
    box_xy1xy2_list: list[tuple] | ndarray,
    frame_shape: tuple | None = None,
    iou_threshold: float = 0.5,
) -> list[int]:
    """
    Function which performs non-max suppression on given box ((x1, y1), (x2, y2)) coordinates.
    - Expects box_xy1xy2 in separate tuple format: [(x1,y1), (x2,y2)]
    - Assumes xy coords. are normalized. Frame shape is used to produce correct area calculations
    - If frame_shape is None, then box xy values are assumed to be in pixel units,
      otherwise, assumes coordinates are normalized 0-to-1 and frame_shape
      (i.e. coming from frame.shape or tensor.shape) will be used to convert to pixel units

    Returns a list of indexes which are meant to indicate which of the input boxes should be kept
    """

    # Bail if we don't get any boxes!
    if len(box_xy1xy2_list) == 0:
        return []

    # Convert box coords to pixels, so area calculations are correct (normalized coords get areas wrong!)
    box_xy1xy2_px = np.float32(box_xy1xy2_list)
    if box_xy1xy2_px.shape[-1] == 4:
        box_xy1xy2_px = box_xy1xy2_px.reshape(-1, 2, 2)
    if frame_shape is not None:
        img_h, img_w = frame_shape[0:2]
        norm_to_px_scale = np.float32((img_w, img_h)) - 1
        box_xy1xy2_px = box_xy1xy2_px * norm_to_px_scale

    # For convenience, get area/bounding coords for iou calcs
    areas = np.prod(np.diff(box_xy1xy2_px, axis=-2), axis=-1)
    xy1_px = box_xy1xy2_px[:, 0, :]
    xy2_px = box_xy1xy2_px[:, 1, :]

    # Compute IoU between every pair of boxes and discard boxes with high overlap
    # - Approach is to take 1st box and compare to all others, discarding any close matches
    # - Then take the next (non-discarded) box and repeat, until there are no more 'next' boxes
    num_boxes = box_xy1xy2_px.shape[0]
    box_idx_left_to_check = np.arange(num_boxes)
    box_idx_to_keep = []
    while len(box_idx_left_to_check) > 0:

        # For clarity, get index of box we're checking vs. all other boxes
        idx_to_check, other_idxs = box_idx_left_to_check[0], box_idx_left_to_check[1:]
        box_idx_to_keep.append(int(idx_to_check))

        # Index out box attributes we need for iou calculation
        box_area, other_areas = areas[idx_to_check], areas[other_idxs]
        box_xy1, other_xy1s = xy1_px[idx_to_check], xy1_px[other_idxs]
        box_xy2, other_xy2s = xy2_px[idx_to_check], xy2_px[other_idxs]

        # Find union bounding box between the box we're checking & all other boxes
        overlap_xy1 = np.maximum(box_xy1, other_xy1s)
        overlap_xy2 = np.minimum(box_xy2, other_xy2s)
        overlap_wh = np.maximum((0, 0), overlap_xy2 - overlap_xy1 + 1)

        # Find intersection-over-union of box we're checking vs. all other boxes
        intersection_area = np.prod(overlap_wh, axis=1, keepdims=True)
        union_area = box_area + other_areas - intersection_area
        iou = intersection_area / union_area

        # Any box that doesn't overlap with current box needs to be checked on next loop iteration
        idx_of_non_overlaps = np.where(iou <= iou_threshold)[0]
        box_idx_left_to_check = other_idxs[idx_of_non_overlaps]

    return box_idx_to_keep


def sample_points_from_mask(mask_image: ndarray, num_points_approx: int = 25) -> list[tuple[float, float]]:
    """
    Helper used to sample points from a mask.
    Uses a fibonnaci spiral sampling pattern to give somewhat
    jittered sampling pattern.

    The input mask image is expected to be a numpy array with a
    shape of either HxW or HxWxC (i.e. an image loaded using opencv).

    The 'num_points_approx' setting controls how densely the mask
    will be sampled, but actual sample counts can be much lower,
    depending on the mask shape. Also note that each 'island' of the
    input mask will be sampled separately!

    Returns:
        sample_xy_norm_list
        -> This is a list with 'N' entries, each of the form: (x,y),
           where both x & y are normalized 0-to-1
    """

    # Compute fib. spiral polar values
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    num_fib_pts = golden_ratio * num_points_approx
    pt_idx = np.arange(0, num_fib_pts, dtype=np.float32)
    r = (1) * np.sqrt(pt_idx / num_fib_pts) / np.sqrt(2, dtype=np.float32)
    theta = 2.0 * np.pi * (pt_idx / golden_ratio)

    # Convert from polar to xy coords
    fib_sample_x_norm = 0.5 + r * np.cos(theta)
    fib_sample_y_norm = 0.5 + r * np.sin(theta)

    # Remove out-of-bounds points (we generate lots of these due to circular perimeter vs. square image!)
    ok_x_pts = np.bitwise_and(fib_sample_x_norm > 0.0, fib_sample_x_norm < 1.0)
    ok_y_pts = np.bitwise_and(fib_sample_y_norm > 0.0, fib_sample_y_norm < 1.0)
    ok_pts = np.bitwise_and(ok_x_pts, ok_y_pts)
    fib_sample_x_norm, fib_sample_y_norm = fib_sample_x_norm[ok_pts], fib_sample_y_norm[ok_pts]

    # Force mask to be grayscale
    ref_h, ref_w = mask_image.shape[0:2]
    if mask_image.ndim > 2:
        assert mask_image.ndim == 3, "Must provide a mask with shape HxW or HxWxC"
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) if mask_image.shape[2] == 3 else mask_image[:, :, 0]

    # Get each 'island' of the mask for sampling
    mask_bin = mask_image > 0
    contours_px_list, _ = cv2.findContours(np.uint8(mask_bin), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Overlay fib. sampling pattern with each mask island and remove out-of-mask points
    final_sample_xy_px_list = []
    for contour_pts_list in contours_px_list:

        # Skip bad contours
        if len(contour_pts_list) < 3:
            continue

        # Skip over zero-dimension masks (can happen due to how opencv handles contours)
        x1_px, y1_px, w_px, h_px = cv2.boundingRect(contour_pts_list)
        if w_px < 1 or h_px < 1:
            continue

        # Map sample points into pixel units
        sample_x_px = np.round(x1_px + fib_sample_x_norm * (w_px - 1)).astype(np.int32)
        sample_y_px = np.round(y1_px + fib_sample_y_norm * (h_px - 1)).astype(np.int32)

        # Only store points that land on valid mask points
        is_in_mask = mask_bin[sample_y_px, sample_x_px]
        final_sample_xy_px = np.column_stack((sample_x_px[is_in_mask], sample_y_px[is_in_mask]))
        final_sample_xy_px_list.append(final_sample_xy_px)

    # Normalize sample points for output
    out_xy_norm = np.concatenate(final_sample_xy_px_list) / np.float32((ref_w - 1, ref_h - 1))
    return out_xy_norm.tolist()
