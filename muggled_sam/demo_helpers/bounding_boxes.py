#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import torch
import numpy as np

# For type hints
from torch import Tensor
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_one_mask_bounding_box(mask_prediction_hw: Tensor) -> tuple[bool, tuple[ndarray, ndarray]]:
    """
    Helper used to find the bounding box of a single mask prediction.
    Mask (and box) may be invalid if the given prediction is empty!
    Returns:
        is_valid_mask, ((x1, y1), (x2, y2))
    """

    assert mask_prediction_hw.squeeze().ndim == 2, "Must give mask with only height and width! No batching"

    binary_mask = (mask_prediction_hw > 0.0).to(dtype=torch.int32).squeeze()
    row_axis, col_axis = 0, 1
    num_rows, num_cols = binary_mask.shape

    # Bail if we get an empty mask, as it will break our attempt to find edges
    is_valid_mask = binary_mask.nonzero().numel() > 0
    if not is_valid_mask:
        return is_valid_mask, (np.zeros(2), np.zeros(2))

    # Pad mask with a single 0. Without this, we can get off-by-one errors when mask is on frame boundary
    pad_amt = 1
    pad_mask = torch.nn.functional.pad(binary_mask, [pad_amt] * 4, value=0)

    # From each edge direction, find the index of the first non-zero element
    # -> This assumes the mask is binary and not empty!
    # -> Each result holds the first index per row/column (depending on search direction)
    # -> x2/y2 results are considered offsets, since they are measure from far edges (not the direct index)
    x1_idx_col_per_row = torch.argmax(pad_mask, axis=col_axis)
    y1_idx_row_per_col = torch.argmax(pad_mask, axis=row_axis)
    x2_off_col_per_row = torch.argmax(torch.flip(pad_mask, dims=[col_axis]), axis=col_axis)
    y2_off_row_per_col = torch.argmax(torch.flip(pad_mask, dims=[row_axis]), axis=row_axis)

    # For each search direction, pick the smallest non-zero index found as being the 'bounding' index
    # -> If we find ONLY zero for all indexes, we have to assume boundary index is 0
    argmax_iter = (x1_idx_col_per_row, y1_idx_row_per_col, x2_off_col_per_row, y2_off_row_per_col)
    far_edge_indices = []
    for argmax_result in argmax_iter:
        nonzero_indices = argmax_result.nonzero()
        edge_idx = argmax_result[nonzero_indices].min() if nonzero_indices.numel() > 0 else torch.tensor(0)
        far_edge_indices.append(edge_idx)

    # Account for padding and far-edge offsets
    x1, y1, x2_offset, y2_offet = far_edge_indices
    x1 = x1 - pad_amt
    y1 = y1 - pad_amt
    x2 = num_cols - x2_offset + pad_amt
    y2 = num_rows - y2_offet + pad_amt

    # Bundle return in (xy1, xy2) format
    xy1 = np.float32((x1.item(), y1.item()))
    xy2 = np.float32((x2.item(), y2.item()))
    return is_valid_mask, (xy1, xy2)


# .....................................................................................................................


def box_xywh_to_xy1xy2(x, y, w, h) -> tuple[ndarray, ndarray]:
    """
    Helper used to convert [x-center, y-center, width, height] box format
    to [(x1, y1), (x2, y2)] format.
    """

    half_w, half_h = w * 0.5, h * 0.5

    x1 = x - half_w
    x2 = x + half_w
    y1 = y - half_h
    y2 = y + half_h

    return (np.float32((x1, y1)), np.float32((x2, y2)))


# .....................................................................................................................


def box_xy1xy2_to_xywh(xy1, xy2) -> tuple[float, float, float, float]:
    """
    Helper used to convert [(x1, y1), (x2, y2)] box format to
    [x-center, y-center, width, height] format.
    """

    x1, y1 = xy1
    x2, y2 = xy2

    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    return x, y, w, h


# .....................................................................................................................


def get_2box_iou(box_a_xy1xy2: tuple, box_b_xy1xy2: tuple, min_union_threshold=0.001) -> float:
    """
    Helper used to compute the intersection-over-union between 2 boxes,
    also called the 'Jaccard index'.
    Assumes both boxes are given in [(x1,y1), (x2,y2)] format.

    The min_union_threshold is used to zero out results with overly small
    union areas, which helps avoid divide-by-zero errors

    Returns:
        intersection_over_union
    """

    # For convenience
    (ax1, ay1), (ax2, ay2) = box_a_xy1xy2
    (bx1, by1), (bx2, by2) = box_b_xy1xy2

    # Compute intersection area between boxes
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    inter_area = max(0, inter_w) * max(0, inter_h)

    # Compute union area
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union_area = a_area + b_area - inter_area

    return inter_area / union_area if union_area > min_union_threshold else 0
