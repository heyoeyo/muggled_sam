#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

from time import perf_counter
import torch
import cv2
import numpy as np
from lib.make_sam import make_sam_from_state_dict
from lib.demo_helpers.mask_postprocessing import (
    calculate_mask_stability_score,
    get_box_nms_indexing,
    get_box_xy1xy2_norm_from_mask,
)

# Define pathing
image_path = "/path/to/image.jpg"
model_path = "/path/to/model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Auto-mask generation settings
visualize_results = True
points_per_side = 8
stability_offset = 2.5
stability_threshold = 0.5
iou_threshold = 0.5
mask_threshold = 0.0
min_area, max_area = 0.02, 0.75
box_nms_threshold = 0.75
use_mask_0, use_mask_1, use_mask_2, use_mask_3 = False, True, True, True

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Pre-compute values needed for auto-segmentation
img_hw = img_bgr.shape[0:2]
img_pixel_count = img_hw[0] * img_hw[1]
num_total_prompts = points_per_side * points_per_side
pts_1d = np.linspace(0, 1, max(1, points_per_side) + 2, dtype=np.float32)[1:-1]
pts_2d = np.dstack(np.meshgrid(pts_1d, pts_1d)).reshape(num_total_prompts, 2)
visualize_wait_ms = max(1, min(100, round(5000 / num_total_prompts)))
use_mask_idx = [use_mask_0, use_mask_1, use_mask_2, use_mask_3]
assert any(use_mask_idx), "Must use at least one mask prediction from model!"

# Set up model
print("Loading model & encoding image data...")
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)
encoded_img, _, _ = sammodel.encode_image(img_bgr, max_side_length=1024, use_square_sizing=True)

# Generate grid of point prompts & generate mask for each
print(f"Generating masks ({num_total_prompts} total prompts)...")
t1 = perf_counter()
try:
    raw_results_list = []
    for fg_xy_norm in pts_2d:
        encoded_prompt = sammodel.encode_prompts(None, np.expand_dims(fg_xy_norm, 0), None)
        mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompt)

        # Allow some mask predictions to be ignored
        # -> Certain indexes favor large/middle/small-scale segmentations (indexing varies by model)
        # -> Can choose to ignore indexes associated segmentation level, if you know you don't need/want it
        # -> This will also speed up auto-segmentation
        ok_masks = mask_preds[:, use_mask_idx, :, :]
        ok_iou = iou_preds[:, use_mask_idx]

        # Skip bad iou results
        is_ok_iou = ok_iou >= iou_threshold
        if not is_ok_iou.any():
            continue
        ok_masks = ok_masks[is_ok_iou]
        ok_iou = ok_iou[is_ok_iou]

        # Skip bad stability results
        stability_scores = calculate_mask_stability_score(ok_masks, stability_offset, mask_threshold)
        is_ok_stability = stability_scores >= stability_threshold
        if not is_ok_stability.any():
            continue
        ok_masks = ok_masks[is_ok_stability]
        ok_iou = ok_iou[is_ok_stability]
        ok_stability = stability_scores[is_ok_stability]

        # Binarize & filter by size
        bin_masks = torch.nn.functional.interpolate(ok_masks.unsqueeze(0), size=img_hw).squeeze(0) > mask_threshold
        areas_norm = bin_masks.sum((1, 2)) / img_pixel_count
        is_ok_area = torch.bitwise_and(areas_norm > min_area, areas_norm < max_area)
        if not is_ok_area.any():
            continue
        ok_masks = bin_masks[is_ok_area]
        ok_iou = ok_iou[is_ok_area]
        ok_stability = ok_stability[is_ok_area]
        ok_area = areas_norm[is_ok_area]

        # Store 'ok' mask results
        for bin_mask, iou, stability, area in zip(ok_masks, ok_iou, ok_stability, ok_area):
            mask_uint8 = (bin_mask.byte() * 255).cpu().numpy()
            box_xy1xy2_norm = get_box_xy1xy2_norm_from_mask(mask_uint8)
            raw_results_list.append(
                {
                    "mask": mask_uint8,
                    "prompt_xy_norm": fg_xy_norm,
                    "stability_score": stability.item(),
                    "iou_score": iou.item(),
                    "area_norm": area.item(),
                    "box_xy1xy2_norm": box_xy1xy2_norm,
                }
            )

            # Display each mask result
            if visualize_results:
                iou_pct, area_pct, stable_pct = [round(100 * val.item()) for val in [iou, area, stability]]
                txt_color, txt_scale = (255, 0, 255), 0.5
                disp_scale = min(1, 800 / max(img_hw))
                disp_mask = cv2.resize(mask_uint8, dsize=None, fx=disp_scale, fy=disp_scale)
                disp_mask = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2BGR)
                norm_to_px_scale = np.float32((disp_mask.shape[1] - 1, disp_mask.shape[0] - 1))
                pt_xy_px = np.round(fg_xy_norm * norm_to_px_scale).astype(np.int32)
                pt1, pt2 = np.int32(box_xy1xy2_norm * norm_to_px_scale)
                cv2.circle(disp_mask, pt_xy_px, 8, (255, 0, 255), -1)
                cv2.rectangle(disp_mask, pt1, pt2, (0, 255, 255), 1)
                cv2.putText(disp_mask, f"IoU: {iou_pct}", (5, 16), 0, txt_scale, txt_color, 1)
                cv2.putText(disp_mask, f"Area: {area_pct}", (5, 36), 0, txt_scale, txt_color, 1)
                cv2.putText(disp_mask, f"Stablility: {stable_pct}", (5, 56), 0, txt_scale, txt_color, 1)
                cv2.imshow("Valid Mask", disp_mask)
                cv2.waitKey(visualize_wait_ms)
finally:
    cv2.destroyAllWindows()
t2 = perf_counter()
time_taken_ms = round(1000 * (t2 - t1))
print(f"Took {time_taken_ms} ms{' (with rendering)' if visualize_results else ''}")

# Perform nms to get rid of overlapping masks (and order by largest area)
raw_box_xy1xy2_list = [item["box_xy1xy2_norm"] for item in raw_results_list]
nms_idx_list = get_box_nms_indexing(raw_box_xy1xy2_list, img_bgr.shape, box_nms_threshold)
final_results_list = [raw_results_list[idx] for idx in nms_idx_list]
final_results_list = sorted(final_results_list, key=lambda item: item["area_norm"], reverse=True)
print(f"Found {len(final_results_list)} unique masks (from {len(raw_results_list)} valid masks)")

# Display final results, showing bounding boxes & binary mask
if visualize_results and len(final_results_list) > 0:
    disp_scale = min(1, 800 / max(img_hw))
    disp_img = cv2.resize(img_bgr, dsize=None, fx=disp_scale, fy=disp_scale)
    disp_h, disp_w = disp_img.shape[0:2]
    winname = "Final - any key to close"
    combined_mask = np.zeros_like(final_results_list[0]["mask"])
    for res_idx, item in enumerate(final_results_list):
        xy1_norm, xy2_norm = np.float32(item["box_xy1xy2_norm"])
        pt1, pt2 = np.int32(item["box_xy1xy2_norm"] * np.float32((disp_w - 1, disp_h - 1)))
        color = np.random.randint(150, 255, 3, dtype=np.uint8).tolist()
        cv2.rectangle(disp_img, pt1, pt2, (0, 0, 0), 3)
        cv2.rectangle(disp_img, pt1, pt2, color, 2)
        combined_mask = np.bitwise_or(combined_mask, item["mask"])
    combined_mask = cv2.resize(combined_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    sidebyside = np.hstack((disp_img, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)))
    cv2.imshow(winname, sidebyside)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
