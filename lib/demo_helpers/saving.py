#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
import json

import cv2
import numpy as np

from .contours import pixelize_contours


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def save_segmentation_results(
    image_path, display_image, mask_contours_norm, raw_result_uint8, all_prompts_dict, base_save_folder=None
):
    """Helper used to handle saving of segmentation results"""

    # Load copy of original image for saving results
    image_bgr = cv2.imread(image_path)
    image_name, _ = osp.splitext(osp.basename(image_path))

    # Make a mask matching the original image resolution
    img_h, img_w = image_bgr.shape[0:2]
    full_mask_1ch = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_contours_px = pixelize_contours(mask_contours_norm, image_bgr.shape)
    full_mask_1ch = cv2.fillPoly(full_mask_1ch, mask_contours_px, 255, cv2.LINE_AA)

    # Make full sized image, with mask transparency
    full_transparent = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    full_transparent[:, :, -1] = full_mask_1ch

    # Find bounding coordinates of mask contours
    tl_xy_norm = np.min([np.min(contour.squeeze(), axis=0) for contour in mask_contours_norm], 0)
    br_xy_norm = np.max([np.max(contour.squeeze(), axis=0) for contour in mask_contours_norm], 0)

    # Crop full image down to bounding region
    max_xy = (img_w - 1, img_h - 1)
    norm2px_scale = np.float32(max_xy)
    tl_xy_px = np.clip(np.int32(np.floor(tl_xy_norm * norm2px_scale)), 0, max_xy)
    br_xy_px = np.clip(np.int32(np.ceil(br_xy_norm * norm2px_scale)), 0, max_xy)
    (x1, y1), (x2, y2) = tl_xy_px, br_xy_px
    cropped_transparent = full_transparent[y1:y2, x1:x2, :]

    # Create (padded) crop of full color image, which can be re-segmented for higher resolution result
    pad_amount = np.int32((br_xy_px - tl_xy_px) / 12)
    padded_tl_xy_px = np.clip(tl_xy_px - pad_amount, 0, max_xy)
    padded_br_xy_px = np.clip(br_xy_px + pad_amount, 0, max_xy)
    (x1, y1), (x2, y2) = padded_tl_xy_px, padded_br_xy_px
    cropped_color = full_transparent[y1:y2, x1:x2, 0:3]

    # Bundle results for saving
    name_to_image_lut = {
        "raw_result_mask": raw_result_uint8,
        "full_mask": full_mask_1ch,
        "full_segmentation": full_transparent,
        "cropped_segmentation": cropped_transparent,
        "cropped": cropped_color,
        "display": display_image,
    }

    # Build saving path
    save_folder = osp.join("saved_images", "manual", image_name)
    if base_save_folder is not None:
        save_folder = osp.join(base_save_folder, save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # Figure out the file indexing (used to group together all results)
    save_idx = 0
    existing_files_list = os.listdir(save_folder)
    all_prefixes = [str(name).split("_")[0] for name in existing_files_list if len(str(name).split("_")) > 0]
    all_idxs = [int(prefix) for prefix in all_prefixes if prefix.isnumeric()]
    save_idx = 1 + max(all_idxs) if len(all_idxs) > 0 else 0

    # Save all results
    idx_str = str(save_idx).zfill(3)
    make_save_name = lambda name: f"{idx_str}_{name}.png"
    for name, image_data in name_to_image_lut.items():
        save_name = make_save_name(name)
        save_path = osp.join(save_folder, save_name)
        cv2.imwrite(save_path, image_data)

    # Save prompt data
    prompt_save_name = f"{idx_str}_prompts.json"
    prompt_save_path = osp.join(save_folder, prompt_save_name)
    with open(prompt_save_path, "w") as outfile:
        json.dump(all_prompts_dict, outfile, indent=2)

    return save_folder, save_idx
