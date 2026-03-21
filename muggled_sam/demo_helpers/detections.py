#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# For type hints
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_filtered_detections(
    mask_predictions: Tensor | None,
    box_predictions: Tensor | None,
    detection_scores: Tensor,
    detection_threshold: float = 0.5,
    top_n_if_missing: int = 5,
) -> tuple[bool, Tensor, Tensor, Tensor]:
    """
    Helper used to get filtered detection predictions (e.g. from SAMv3).
    Includes support for reporting a 'top-N' predictions in case
    there are no predictions above the given detection threshold.

    Returns:
        has_valid_detections, filtered_masks, filtered_boxes, filtered_scores
    """

    filter_idx = detection_scores > detection_threshold
    num_valid = filter_idx.count_nonzero()
    has_valid_detections = num_valid > 0
    if not has_valid_detections and top_n_if_missing > 0:
        sorted_scores, sorted_idx = detection_scores.float().cpu().sort(descending=True)
        top_scores = sorted_scores.squeeze()[(top_n_if_missing - 1) : (top_n_if_missing + 1)]
        new_thresh = top_scores.mean()
        filter_idx = detection_scores > new_thresh

    filtered_masks = mask_predictions[filter_idx] if mask_predictions is not None else None
    filtered_boxes = box_predictions[filter_idx] if box_predictions is not None else None
    filtered_scores = detection_scores[filter_idx]
    return has_valid_detections, filtered_masks, filtered_boxes, filtered_scores
