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
import torch
from muggled_sam.make_sam import make_sam_from_state_dict

# Define pathing
image_path = "/path/to/image.jpg"
model_path = "/path/to/samv3_model.pth"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# All coordinates are normalized between 0 and 1. Top left of image is (0,0), bottom-right is (1,1)
pos_box_xy1xy2_norm_list = [[(0.25, 0.25), (0.75, 0.75)]]  # Format is: [[(x1, y1), (x2, y2)]]
neg_box_xy1xy2_norm_list = []
pos_point_xy_norm_list = [(0.5, 0.5)]
neg_point_xy_norm_list = []
text_prompt = "visual"  # This is a default from original implementation. Can be set to None to disable
detection_score_threshold = 0.5
max_side_length = 1008
use_square_sizing = True

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Load and set up detector model
model_config_dict, base_model = make_sam_from_state_dict(model_path)
base_model.to(device=device, dtype=dtype)
detmodel = base_model.make_detector_model()

# Run detection
encoded_imgs, token_hw, preencode_hw = detmodel.encode_detection_image(img_bgr, max_side_length, use_square_sizing)
encoded_exemplars = detmodel.encode_exemplars(
    encoded_imgs,
    text_prompt,
    pos_box_xy1xy2_norm_list,
    pos_point_xy_norm_list,
    neg_box_xy1xy2_norm_list,
    neg_point_xy_norm_list,
)
mask_preds, box_preds, detection_scores, presence_score = detmodel.generate_detections(encoded_imgs, encoded_exemplars)

# (Optional) Typical post-processing to filter out low-scoring results
filtered_masks, filtered_boxes, filtered_scores, presence_score = detmodel.filter_results(
    mask_preds, box_preds, detection_scores, presence_score, detection_score_threshold
)
num_filtered_detections = filtered_masks.shape[0]
# To get binary masks, use: binary_masks = filtered_masks > 0

# Feedback
print("")
print("***** Results *****")
print("Input image shape:", img_bgr.shape)
print("Pre-encoded image height & width:", tuple(preencode_hw))
print("Image tokens height & width:", tuple(token_hw))
print("Raw masks shape:", tuple(mask_preds.shape))
print("Raw boxes shape:", tuple(box_preds.shape))
print("Raw scores shape:", tuple(detection_scores.shape))
print("Presence score:", *presence_score.tolist())
print("Num filtered detections:", num_filtered_detections)
print("Filtered masks shape:", tuple(filtered_masks.shape))
