#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
import json
import tarfile
from io import BytesIO

import cv2
import numpy as np

from .contours import pixelize_contours

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Main saving functions


def save_image_segmentation(
    save_folder_path: str,
    save_index: str,
    original_image_bgr: ndarray,
    display_image: ndarray,
    raw_result_uint8: ndarray,
    mask_contours_norm: list,
    all_prompts_dict: dict[str, list],
    is_inverted=False,
    yx_crop_slices: tuple[slice, slice] | None = None,
    base_save_folder: str | None = None,
) -> None:
    """Helper used to handle saving of image segmentation results"""

    # Make sure we're only using valid contours!
    cleaned_contours_norm = remove_invalid_contours(mask_contours_norm)

    # Crop (full) input image if needed
    is_precropped = yx_crop_slices is not None
    precrop_img_bgr = original_image_bgr[yx_crop_slices] if is_precropped else original_image_bgr

    # Generate all base image results for saving
    precrop_mask_1ch = make_mask_1ch(precrop_img_bgr, cleaned_contours_norm, is_inverted)
    precrop_seg_bgra = make_alpha_masked_image(precrop_img_bgr, precrop_mask_1ch)
    postcrop_seg_bgra, postcrop_img_bgr = make_cropped_images(precrop_seg_bgra, precrop_img_bgr, cleaned_contours_norm)

    # Bundle results for saving
    name_to_image_lut = {
        "raw_result_mask": raw_result_uint8,
        "full_mask": precrop_mask_1ch,
        "full_segmentation": precrop_seg_bgra,
        "postcrop_segmentation": postcrop_seg_bgra,
        "postcrop": postcrop_img_bgr,
        "display": display_image,
        "precrop_mask": None,
        "precrop_segmentation": None,
    }

    name_to_dict_lut = {
        "prompts": all_prompts_dict,
        "precrop_coords": None,
        "uncropped_prompts": None,
    }

    # Include additional results if the user pre-cropped the input image
    if is_precropped:

        # Make full sized saving results
        full_mask_1ch = make_mask_1ch(original_image_bgr, [], is_inverted)
        full_mask_1ch[yx_crop_slices] = precrop_mask_1ch
        full_seg_bgra = make_alpha_masked_image(original_image_bgr, full_mask_1ch)

        # Update image saving table (prior 'full' results are actually pre-cropped, and now store full results!)
        name_to_image_lut["precrop_mask"] = precrop_mask_1ch
        name_to_image_lut["precrop_segmentation"] = precrop_seg_bgra
        name_to_image_lut["full_mask"] = full_mask_1ch
        name_to_image_lut["full_segmentation"] = full_seg_bgra

        # Save cropping coords for reference
        crop_coord_dict = make_crop_coord_save_data(yx_crop_slices)
        full_shape, crop_shape = original_image_bgr.shape, precrop_img_bgr.shape
        uncropped_prompts_dict = make_uncropped_prompts(full_shape, crop_shape, yx_crop_slices, all_prompts_dict)

        # Update json saving table
        name_to_dict_lut["precrop_coords"] = crop_coord_dict
        name_to_dict_lut["uncropped_prompts"] = uncropped_prompts_dict

    # Save all image results
    for name, image_data in name_to_image_lut.items():
        if image_data is not None:
            save_file_path = osp.join(save_folder_path, f"{save_index}_{name}.png")
            cv2.imwrite(save_file_path, image_data)

    # Save all dictionary/json results
    for name, data_dict in name_to_dict_lut.items():
        if data_dict is not None:
            save_json_data(save_folder_path, save_index, name, data_dict)

    return


# .....................................................................................................................


def save_video_frames(
    save_folder_path: str,
    save_index: str,
    object_index: int,
    save_frames_dict: dict,
    base_save_folder: str | None = None,
) -> str:
    """Helper used to handle saving of video segmentation results. Returns save file pathing"""

    # Get frame index range for file name
    all_frame_idxs = list(save_frames_dict.keys())
    min_frame_idx, max_frame_idx = min(all_frame_idxs), max(all_frame_idxs)

    # Save tarfile containing all frames
    file_name = f"{save_index}_obj{1+object_index}_{min_frame_idx}_to_{max_frame_idx}_frames.tar"
    save_file_path = os.path.join(save_folder_path, file_name)
    with tarfile.open(save_file_path, "w") as tar:
        for frame_idx, png_encoding in save_frames_dict.items():
            tarinfo = tarfile.TarInfo(name=f"{frame_idx:0>8}.png")
            tarinfo.size = len(png_encoding)
            tar.addfile(tarinfo, BytesIO(png_encoding.tobytes()))

    return save_file_path


# ---------------------------------------------------------------------------------------------------------------------
# %% Helper functions


def remove_invalid_contours(mask_contours_norm: list):
    """Helper which removes contours with fewer than 3 points (opencv can generate 1 & 2 points contours!)"""
    return [c for c in mask_contours_norm if len(c) > 2]


# .....................................................................................................................


def get_save_name(
    input_file_path: str,
    sub_folder_name: str,
    base_save_folder: str | None = None,
    create_save_folder=True,
) -> tuple[str, str]:
    """
    Helper used to build the pathing to a save folder for saving segmentation images.
    Will produce a save path of the form:
        {base_save_folder} / saved_images / {sub_folder_name} / {input_file_name}

    If no base_save_folder is given, then the pathing will be saved relative to
    the script calling this function.

    The sub_folder_name is used to separate save results for different use cases.

    This function also checks the folder for existing saved results and will produce
    a 'save index', which can be used to prefix saved results to make them unique.
    For example, the first time a result is saved, it will be given index: 000,
    follow up results will be given indexes: 001, 002, 003 etc.

    Returns:
        save_folder_path, save_index_as_str
    """

    # Use file name (without ext) as base for saving
    file_name_no_ext, _ = osp.splitext(osp.basename(input_file_path))

    # Build saving path
    save_folder = osp.join("saved_images", sub_folder_name, file_name_no_ext)
    if base_save_folder is not None:
        save_folder = osp.join(base_save_folder, save_folder)

    # Create the folder path if needed
    if create_save_folder:
        os.makedirs(save_folder, exist_ok=True)

    # Figure out the file indexing (used to group together all results)
    save_idx = 0
    existing_files_list = os.listdir(save_folder)
    all_prefixes = [str(name).split("_")[0] for name in existing_files_list if len(str(name).split("_")) > 0]
    all_idxs = [int(prefix) for prefix in all_prefixes if prefix.isnumeric()]
    save_idx = 1 + max(all_idxs) if len(all_idxs) > 0 else 0

    # Save all results
    save_idx_str = str(save_idx).zfill(3)

    return save_folder, save_idx_str


# .....................................................................................................................


def make_mask_1ch(original_image_bgr: ndarray, mask_contours_norm: list, is_inverted=False):
    """Helper used to make a mask matching the original image resolution, in 1 channel. Returns: mask_1ch"""

    # Set appropriate masking values
    mask_bg_value = 255 if is_inverted else 0
    mask_fill_value = 255 - mask_bg_value

    # Make a mask matching the original image resolution
    img_hw = original_image_bgr.shape[0:2]
    full_mask_1ch = np.full(img_hw, mask_bg_value, dtype=np.uint8)
    mask_contours_px = pixelize_contours(mask_contours_norm, original_image_bgr.shape)
    full_mask_1ch = cv2.fillPoly(full_mask_1ch, mask_contours_px, mask_fill_value, cv2.LINE_AA)

    return full_mask_1ch


# .....................................................................................................................


def make_alpha_masked_image(original_image_bgr: ndarray, mask_1ch: ndarray):
    """Helper used to make a version of the input image with the segmentation mask as an alpha channel"""

    # Make full sized image, with mask transparency
    mask_3ch = cv2.cvtColor(mask_1ch, cv2.COLOR_GRAY2BGR)
    alpha_img_bgra = cv2.bitwise_and(original_image_bgr, mask_3ch)
    alpha_img_bgra = cv2.cvtColor(alpha_img_bgra, cv2.COLOR_BGR2BGRA)
    alpha_img_bgra[:, :, -1] = mask_1ch

    return alpha_img_bgra


# .....................................................................................................................


def make_cropped_images(image_bgra: ndarray, image_bgr: ndarray, mask_contours_norm: list) -> tuple[ndarray, ndarray]:
    """
    Helper used to make a tightly cropped image from a larger (given) image, based on a segmentation mask
    The tightly cropped image uses the segmentation mask as an alpha channel as well.
    Also creates a slightly padded cropped image, without an alpha channel

    Returns:
        cropped_bgra, padded_cropped_bgr
    """

    # Find bounding coordinates of mask contours
    tl_xy_norm = np.min([np.min(contour.squeeze(), axis=0) for contour in mask_contours_norm], 0)
    br_xy_norm = np.max([np.max(contour.squeeze(), axis=0) for contour in mask_contours_norm], 0)

    # Create tightly cropped color image with the segmentation mask as an alpha channel
    img_h, img_w = image_bgra.shape[0:2]
    max_xy = (img_w - 1, img_h - 1)
    norm2px_scale = np.float32(max_xy)
    tl_xy_px = np.clip(np.int32(np.floor(tl_xy_norm * norm2px_scale)), 0, max_xy)
    br_xy_px = np.clip(np.int32(np.ceil(br_xy_norm * norm2px_scale)), 0, max_xy)
    (x1, y1), (x2, y2) = tl_xy_px, br_xy_px
    cropped_bgra = image_bgra[y1:y2, x1:x2, :]

    # Create (padded) crop of full color image
    pad_amount = np.int32((br_xy_px - tl_xy_px) / 12)
    padded_tl_xy_px = np.clip(tl_xy_px - pad_amount, 0, max_xy)
    padded_br_xy_px = np.clip(br_xy_px + pad_amount, 0, max_xy)
    (x1, y1), (x2, y2) = padded_tl_xy_px, padded_br_xy_px
    padded_cropped_bgr = image_bgr[y1:y2, x1:x2, 0:3]

    return cropped_bgra, padded_cropped_bgr


# .....................................................................................................................


def make_prompt_save_data(box_tlbr_norm_list, fg_xy_norm_list, bg_xy_norm_list) -> dict[str, list]:
    """Helper used to standardize save formatting of prompt data"""
    return {
        "boxes": box_tlbr_norm_list,
        "fg_points": fg_xy_norm_list,
        "bg_points": bg_xy_norm_list,
    }


# .....................................................................................................................


def make_uncropped_prompts(
    original_image_shape: tuple,
    cropped_image_shape: tuple,
    yx_crop_slices: tuple[slice, slice],
    all_prompts_dict: dict,
) -> dict[str, list]:
    """
    Helper used to take coords that are in 'crop-space' and map them back into the original coordinate system
    Returns:
        uncropped_prompts_dict
    """

    # Unpack prompt data
    box_tlbr_norm_list = all_prompts_dict.get("boxes", tuple())
    fg_xy_norm_list = all_prompts_dict.get("fg_points", tuple())
    bg_xy_norm_list = all_prompts_dict.get("bg_points", tuple())

    # Set up helper function used to map cropped xy coords to un-cropped coords
    crop_y1, crop_x1 = yx_crop_slices[0].start, yx_crop_slices[1].start
    full_h, full_w = original_image_shape[0:2]
    crop_h, crop_w = cropped_image_shape[0:2]
    uncrop_x = lambda x: (x * (crop_w - 1) + crop_x1) / (full_w - 1)
    uncrop_y = lambda y: (y * (crop_h - 1) + crop_y1) / (full_h - 1)
    uncrop_xy = lambda xy: (uncrop_x(xy[0]), uncrop_y(xy[1]))

    # Apply un-crop function to every xy coord in prompts
    uncropped_box_list = []
    for tlbr_xy in box_tlbr_norm_list:
        uncropped_box_list.append([uncrop_xy(xy_norm) for xy_norm in tlbr_xy])
    uncropped_fg_xy_list = [uncrop_xy(xy_norm) for xy_norm in fg_xy_norm_list]
    uncropped_bg_xy_list = [uncrop_xy(xy_norm) for xy_norm in bg_xy_norm_list]

    return make_prompt_save_data(uncropped_box_list, uncropped_fg_xy_list, uncropped_bg_xy_list)


# .....................................................................................................................


def make_crop_coord_save_data(yx_crop_slices: tuple[slice, slice]) -> dict[str, tuple]:
    """Helper used to make json-saveable record of the crop coordinates that were used"""

    y_crop_slice, x_crop_slice = yx_crop_slices
    crop_x1x2 = (x_crop_slice.start, x_crop_slice.stop)
    crop_y1y2 = (y_crop_slice.start, y_crop_slice.stop)
    crop_data_dict = {"crop_x": crop_x1x2, "crop_y": crop_y1y2}

    return crop_data_dict


# .....................................................................................................................


def save_json_data(save_folder_path: str, save_index: str, plain_file_name: str, save_data_dict: dict) -> str:
    """Helper used to save json-friendly data (i.e. dictionaries with coords)"""

    save_file_name = f"{save_index}_{plain_file_name}.json"
    save_file_path = osp.join(save_folder_path, save_file_name)
    with open(save_file_path, "w") as outfile:
        json.dump(save_data_dict, outfile, indent=2)

    return save_file_path
