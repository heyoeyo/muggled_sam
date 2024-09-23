#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def get_largest_contour_from_mask(
    mask_binary_uint8,
    minimum_contour_area_norm=None,
    normalize=True,
    simplification_eps=None,
) -> [bool, np.ndarray]:
    """
    Helper used to get only the largest contour (by area) from a a given binary mask image.

    Inputs:
        mask_uint8 - A uint8 numpy array where bright values indicate areas to be masked
        minimum_contour_area_norm - (None or number 0-to-1) Any contour with area making up less
                                    than this percentage of the mask will be excluded from the output
        normalize - If true, contour xy coords. will be in range (0.0 to 1.0), otherwise they're in pixel coords
        simplification_eps - Value indicating how much to simplify the resulting contour. Larger values lead
                             to greater simplification (value is roughly a 'pixel' unit). Set to None to disable

    Returns:
        ok_contour (boolean), largest_contour
    """

    # Initialize outputs
    ok_contour = False
    largest_contour = None

    # Get all contours, bail if we don't get any
    ok_contour, contours_list = get_contours_from_mask(mask_binary_uint8, normalize=False)
    if not ok_contour:
        return ok_contour, largest_contour

    # Grab largest contour by area
    contour_areas = [cv2.contourArea(each_contour) for each_contour in contours_list]
    idx_of_largest_contour = np.argmax(contour_areas)
    largest_contour = contours_list[idx_of_largest_contour]
    if minimum_contour_area_norm is not None:
        mask_h, mask_w = mask_binary_uint8.shape[0:2]
        max_area = mask_h * mask_w
        min_area_px = int(max_area * minimum_contour_area_norm)
        largest_area = contour_areas[idx_of_largest_contour]
        ok_contour = largest_area >= min_area_px
        if not ok_contour:
            largest_contour = None
            return ok_contour, largest_contour

    # Simplify if needed
    need_to_simplify = simplification_eps is not None
    if need_to_simplify:
        largest_contour = simplify_contour_px(largest_contour, simplification_eps)

    # Apply normalization if needed
    # (couldn't apply earlier, since we need to use pixel coords for area calculations!)
    if normalize:
        mask_h, mask_w = mask_binary_uint8.shape[0:2]
        norm_scale_factor = 1.0 / np.float32((mask_w - 1, mask_h - 1))
        largest_contour = largest_contour * norm_scale_factor

    return ok_contour, largest_contour.squeeze(1)


# .....................................................................................................................


def get_contours_from_mask(
    mask_binary_uint8,
    minimum_contour_area_norm=0,
    normalize=True,
) -> [bool, tuple]:
    """
    Function which takes in a binary black & white mask and returns contours around each independent 'blob'
    within the mask. Note that only the external-most contours are returned, without holes!

    Inputs:
        mask_binary_uint8 - A uint8 numpy array where bright values indicate areas to be masked
        minimum_contour_area_norm - (None or number 0-to-1) Any contour with area making up less
                                    than this percentage of the mask will be excluded from the output
        normalize - If true, contour xy coords. will be in range (0.0 to 1.0), otherwise they're in pixel coords

    Returns:
        have_contours (boolean), mask_contours_as_tuple
    """

    # Initialize outputs
    have_contours = False
    mask_contours_list = []

    # Generate outlines from the segmentation mask
    mask_contours_list, _ = cv2.findContours(mask_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bail if we have no contours from the mask
    have_contours = len(mask_contours_list) > 0
    if not have_contours:
        return have_contours, tuple(mask_contours_list)

    # Filter out small contours, if needed
    if minimum_contour_area_norm > 0:
        mask_h, mask_w = mask_binary_uint8.shape[0:2]
        max_area = mask_h * mask_w
        min_area_px = int(max_area * minimum_contour_area_norm)
        mask_contours_list = [cont for cont in mask_contours_list if cv2.contourArea(cont) > min_area_px]
        have_contours = len(mask_contours_list) > 0

    # Normalize xy coords if needed
    if normalize:
        mask_contours_list = normalize_contours(mask_contours_list, mask_binary_uint8.shape)

    # Remove any 1 or 2 point 'contours' since these aren't valid shapes
    mask_contours_list = [c for c in mask_contours_list if len(c) > 2]

    return have_contours, tuple(mask_contours_list)


# .....................................................................................................................


def get_contours_containing_xy(contours_list, xy) -> [bool, list]:
    """Helper used to filter out contours that do not contain the given xy coordinate"""
    filtered_list = [contour for contour in contours_list if cv2.pointPolygonTest(contour, xy, False) > 0]
    have_results = len(filtered_list) > 0
    return have_results, filtered_list


# .....................................................................................................................


def get_largest_contour(contours_list, reference_shape=None) -> [bool, np.ndarray]:
    """
    Helper used to filter out only the largest contour from a list of contours

    If the given contours use normalized coordinates, then the 'largest' calculation can be
    incorrect, due to uneven width/height scaling. In these cases, a reference frame shape
    can be given, which will be used to scale the normalized values appropriately
    before determining which is the largest.

    Returns:
        index of the largest contour, largest_contour
    """

    # Use aspect-ratio adjusted area calculation, if possible
    area_calc = lambda contour: cv2.contourArea(contour)
    if reference_shape is not None:
        frame_h, frame_w = reference_shape[0:2]
        scale_factor = np.float32((frame_w - 1, frame_h - 1))
        area_calc = lambda contour: cv2.contourArea(contour * scale_factor)

    # Grab largest contour by area
    contour_areas = [area_calc(contour) for contour in contours_list]
    idx_of_largest_contour = np.argmax(contour_areas)
    largest_contour = contours_list[idx_of_largest_contour]

    return idx_of_largest_contour, largest_contour


# .....................................................................................................................


def simplify_contour_px(contour_px, simplification_eps=1.0, scale_to_perimeter=False) -> np.ndarray:
    """
    Function used to simplify a contour, without completely altering the overall shape
    (as compared to finding the convex hull, for example). Uses the Ramer–Douglas–Peucker algorithm

    Inputs:
        contour_px - A single contour to be simplified (from opencv findContours() function), must be in px units!
        simplification_eps - Value that determines how 'simple' the result should be. Larger values
                             result in more heavily approximated contours
        scale_to_perimeter - If True, the eps value is scaled by the contour perimeter before performing
                             the simplification. Otherwise, the eps value is used as-is

    Returns:
        simplified_contour
    """

    # Decide whether to use perimeter scaling for approximation value
    epsilon = simplification_eps
    if scale_to_perimeter:
        epsilon = cv2.arcLength(contour_px, closed=True) * simplification_eps

    return cv2.approxPolyDP(contour_px, epsilon, closed=True)


# .....................................................................................................................


def normalize_contours(contours_px_list, frame_shape):
    """Helper used to normalize contour data, according to a given frame shape (i.e. [height, width]"""

    frame_h, frame_w = frame_shape[0:2]
    norm_scale_factor = 1.0 / np.float32((frame_w - 1, frame_h - 1))

    return [np.float32(contour) * norm_scale_factor for contour in contours_px_list]


# .....................................................................................................................


def pixelize_contours(contours_norm_list, frame_shape):
    """Helper used to convert normalized contours to pixel coordinates"""

    frame_h, frame_w = frame_shape[0:2]
    scale_factor = np.float32((frame_w - 1, frame_h - 1))

    return [np.int32(np.round(contour * scale_factor)) for contour in contours_norm_list]
