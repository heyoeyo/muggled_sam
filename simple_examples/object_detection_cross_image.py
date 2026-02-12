#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import muggled_sam  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_sam folder!")
import cv2
import numpy as np
import torch
from muggled_sam.make_sam import make_sam_from_state_dict

# Define pathing
reference_image_path = "/path/to/reference_image.jpg"
target_image_path = "/path/to/target_image.jpg"
model_path = "/path/to/sam3.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# All coordinates are normalized between 0 and 1. Top left of image is (0,0), bottom-right is (1,1)
pos_box_xy1xy2_norm_list = []  # Format is: [[(x1, y1), (x2, y2)]]
neg_box_xy1xy2_norm_list = []
pos_point_xy_norm_list = [(0.5, 0.5)]
neg_point_xy_norm_list = []
text_prompt = "visual"  # This is a default from original implementation. Can be set to None to disable
imgenc_config_dict = {"max_side_length": 1008, "use_square_sizing": True}
detection_score_threshold = 0.5

# Load images
ref_img_bgr = cv2.imread(reference_image_path)
target_img_bgr = cv2.imread(target_image_path)
if ref_img_bgr is None:
    raise FileNotFoundError(f"Error loading reference image: {reference_image_path}")
if target_img_bgr is None:
    raise FileNotFoundError(f"Error loading target image: {target_image_path}")

# Load and set up detector model
model_config_dict, full_model = make_sam_from_state_dict(model_path)
full_model.to(device=device, dtype=dtype)
detmodel = full_model.make_detector_model()

# Encode exemplars from *reference* image
enc_ref_img, _, _ = detmodel.encode_detection_image(ref_img_bgr, **imgenc_config_dict)
enc_ref_exemplars = detmodel.encode_exemplars(
    enc_ref_img,
    text_prompt,
    pos_box_xy1xy2_norm_list,
    pos_point_xy_norm_list,
    neg_box_xy1xy2_norm_list,
    neg_point_xy_norm_list,
)

# Detect exemplars on *target* image
enc_targ_img, token_hw, preencode_hw = detmodel.encode_detection_image(target_img_bgr, **imgenc_config_dict)
mask_preds, box_preds, detection_scores, presence_score = detmodel.generate_detections(enc_targ_img, enc_ref_exemplars)
filtered_masks, filtered_boxes, filtered_scores, presence_score = detmodel.filter_results(
    mask_preds, box_preds, detection_scores, presence_score, detection_score_threshold
)

# Scale images for display
disp_scale_factor = 0.5
scaled_targ_img = cv2.resize(target_img_bgr, dsize=None, fx=disp_scale_factor, fy=disp_scale_factor)
ref_scaled_w = round(ref_img_bgr.shape[1] * scaled_targ_img.shape[0] / target_img_bgr.shape[0])
ref_scaled_h = scaled_targ_img.shape[0]
scaled_ref_img = cv2.resize(ref_img_bgr, dsize=(ref_scaled_w, ref_scaled_h))

# Draw prompts onto reference image
ref_norm_to_px_scale = np.float32((ref_scaled_w - 1, ref_scaled_h - 1))
for pt in pos_point_xy_norm_list:
    pt_xy_px = np.round((np.float32(pt) * ref_norm_to_px_scale)).astype(np.int32).tolist()
    cv2.circle(scaled_ref_img, pt_xy_px, 3, (0, 255, 0), -1)
for pt in neg_box_xy1xy2_norm_list:
    pt_xy_px = np.round((np.float32(pt) * ref_norm_to_px_scale)).astype(np.int32).tolist()
    cv2.circle(scaled_ref_img, pt_xy_px, 3, (0, 0, 255), -1)
for xy1xy2 in pos_box_xy1xy2_norm_list:
    xy1_px, xy2_px = np.round(np.float32(xy1xy2) * ref_norm_to_px_scale).astype(np.int32).tolist()
    cv2.rectangle(scaled_ref_img, xy1_px, xy2_px, (0, 255, 0), 2)
for xy1xy2 in neg_point_xy_norm_list:
    xy1_px, xy2_px = np.round(np.float32(xy1xy2) * ref_norm_to_px_scale).astype(np.int32).tolist()
    cv2.rectangle(scaled_ref_img, xy1_px, xy2_px, (0, 0, 255), 2)

# Draw masked target image (with black-out if we didn't get any masks)
num_filtered_detections = filtered_masks.shape[0]
if num_filtered_detections == 0:
    filtered_masks = torch.full((1, filtered_masks.shape[-2], filtered_masks.shape[-1]), -10.0).to(filtered_masks)
    print("", "No masks detected!", sep="\n")
scaled_masks = torch.nn.functional.interpolate(filtered_masks[None], scaled_targ_img.shape[0:2]).squeeze(0)
combined_mask, _ = (scaled_masks > 0).max(dim=0)
scaled_targ_img = cv2.copyTo(scaled_targ_img, combined_mask.byte().cpu().numpy())

# Final display
cv2.imshow("Cross-image result", np.hstack((scaled_ref_img, scaled_targ_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()
